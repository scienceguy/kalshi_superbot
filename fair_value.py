#!/usr/bin/env python3
"""
fair_value.py — T15-21: Fair-value engine (maker vs taker).

Computes:
1. Ensemble probability p_hat with uncertainty band (from multiple models)
2. Executable price per fill model (maker/taker)
3. Fee-adjusted EV for maker and taker entry
4. Minimum edge thresholds
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
import fees
import gbm
import kl_divergence

log = logging.getLogger(__name__)


@dataclass
class ModelProb:
    """Individual model probability with metadata."""
    name: str
    prob_yes: float
    weight: float = 1.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FairValueResult:
    """Complete fair-value computation result."""
    # Ensemble probability
    p_hat: float
    p_hat_low: float  # Lower bound of uncertainty band
    p_hat_high: float  # Upper bound of uncertainty band
    ensemble_kl: float  # Disagreement among models
    
    # Individual model probabilities
    model_probs: List[ModelProb]
    
    # Market prices (cents)
    market_yes_bid: int
    market_yes_ask: int
    market_no_bid: int
    market_no_ask: int
    
    # Executable prices (dollars)
    maker_yes_price: float  # Our limit price if buying YES as maker
    maker_no_price: float   # Our limit price if buying NO as maker
    taker_yes_price: float  # Effective price if taking YES
    taker_no_price: float   # Effective price if taking NO
    
    # Fee-adjusted EV (dollars per contract)
    ev_maker_yes: float
    ev_maker_no: float
    ev_taker_yes: float
    ev_taker_no: float
    
    # Edge in cents (p_hat - price)
    edge_maker_yes_cents: float
    edge_maker_no_cents: float
    edge_taker_yes_cents: float
    edge_taker_no_cents: float
    
    # Recommendation
    recommend_side: str  # "yes", "no", "none"
    recommend_venue: str  # "maker", "taker", "none"
    reason: str


# ─── Ensemble probability computation ──────────────────────────────────────────

def compute_ensemble_probability(
    model_probs: List[ModelProb],
    method: str = "log_loss_weighted",
    calibration_monitor: Optional["RollingCalibrationMonitor"] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute ensemble probability with uncertainty band.
    
    Args:
        model_probs: List of ModelProb objects
        method: "equal", "log_loss_weighted", "inverse_kl_weighted"
        calibration_monitor: Optional RollingCalibrationMonitor for online calibration
    
    Returns:
        (p_hat, p_hat_low, p_hat_high, ensemble_kl)
    """
    if not model_probs:
        return 0.5, 0.5, 0.5, 0.0
    
    # Extract probabilities and weights
    probs = [m.prob_yes for m in model_probs]
    weights = [m.weight for m in model_probs]
    
    # Compute ensemble KL (disagreement)
    ensemble_kl = kl_divergence.ensemble_kl(probs)
    
    if method == "equal":
        p_hat = sum(probs) / len(probs)
    elif method == "log_loss_weighted":
        # Weights should be pre-computed from historical log-loss
        total_w = sum(weights)
        if total_w > 0:
            p_hat = sum(p * w for p, w in zip(probs, weights)) / total_w
        else:
            p_hat = sum(probs) / len(probs)
    elif method == "inverse_kl_weighted":
        # Weight by inverse KL from median
        median_p = sorted(probs)[len(probs) // 2]
        kls = [kl_divergence.bernoulli_kl(p, median_p) for p in probs]
        inv_kls = [1.0 / (kl + 1e-6) for kl in kls]
        total_inv = sum(inv_kls)
        p_hat = sum(p * w for p, w in zip(probs, inv_kls)) / total_inv
    else:
        p_hat = sum(probs) / len(probs)
    
    # Apply calibration if available
    if calibration_monitor is not None:
        p_hat = calibration_monitor.apply(p_hat)
    
    # Uncertainty band: use ensemble KL as measure of disagreement
    # Higher KL -> wider band
    kl_factor = min(ensemble_kl * 2.0, 0.5)  # Cap at 0.5
    uncertainty = max(0.02, kl_factor)  # Minimum 2% uncertainty
    
    p_hat_low = max(0.001, p_hat - uncertainty)
    p_hat_high = min(0.999, p_hat + uncertainty)
    
    return p_hat, p_hat_low, p_hat_high, ensemble_kl


# ─── Executable prices ─────────────────────────────────────────────────────────

def maker_executable_price(
    market_bid: int,
    market_ask: int,
    our_price_cents: int,
    side: str,
) -> float:
    """
    Compute executable price for a maker limit order.
    
    As maker, we POST at our_price_cents. If filled, we pay exactly that price.
    No fee for maker orders on Kalshi.
    
    Args:
        market_bid: Current market bid (cents)
        market_ask: Current market ask (cents)
        our_price_cents: Our limit price (cents)
        side: "yes" or "no"
    
    Returns:
        Executable price in dollars (our limit price if fillable, else None)
    """
    if side == "yes":
        # We're bidding to buy YES
        # Can only fill if our bid >= market ask (cross spread) or market maker hits us
        if our_price_cents >= market_ask:
            return our_price_cents / 100.0
        # If our bid is between bid and ask, we sit in the book
        if our_price_cents > market_bid:
            return our_price_cents / 100.0  # Posted in book
        return None  # Below bid, won't fill
    else:
        # We're bidding to buy NO
        # NO ask = 100 - YES bid, NO bid = 100 - YES ask
        no_ask = 100 - market_bid
        no_bid = 100 - market_ask
        if our_price_cents >= no_ask:
            return our_price_cents / 100.0
        if our_price_cents > no_bid:
            return our_price_cents / 100.0
        return None


def taker_executable_price(
    market_bid: int,
    market_ask: int,
    side: str,
) -> float:
    """
    Compute executable price for a taker order.
    
    As taker, we CROSS the spread. We pay market ask (for YES) or market bid (for NO).
    Includes taker fee.
    
    Args:
        market_bid: Current market bid (cents)
        market_ask: Current market ask (cents)
        side: "yes" or "no"
    
    Returns:
        Effective price in dollars including fee
    """
    if side == "yes":
        # Taking YES: pay ask
        if market_ask <= 0:
            return None
        base_price = market_ask / 100.0
        # Taker fee: 0.07 * p * (1-p) per contract
        fee = fees.taker_fee_dollars(1.0, base_price)
        return base_price + fee
    else:
        # Taking NO: pay (100 - bid) cents = NO ask
        no_ask = 100 - market_bid
        if no_ask <= 0:
            return None
        base_price = no_ask / 100.0
        fee = fees.taker_fee_dollars(1.0, base_price)
        return base_price + fee


# ─── EV computation ────────────────────────────────────────────────────────────

def compute_ev(
    p_hat: float,
    executable_price: float,
    side: str,
) -> float:
    """
    Compute expected value per contract.
    
    For YES: EV = p_hat * (1 - price) - (1 - p_hat) * price
             = p_hat - price
    For NO:  EV = (1 - p_hat) * (1 - price) - p_hat * price
             = 1 - p_hat - price
    
    Args:
        p_hat: Our estimated probability of YES winning
        executable_price: Price we pay per contract (dollars)
        side: "yes" or "no"
    
    Returns:
        EV in dollars per contract
    """
    if executable_price is None:
        return float("-inf")
    
    if side == "yes":
        return p_hat - executable_price
    else:
        return (1.0 - p_hat) - executable_price


# ─── Main fair-value computation ───────────────────────────────────────────────

def compute_fair_value(
    model_probs: List[ModelProb],
    market_yes_bid: int,
    market_yes_ask: int,
    market_no_bid: int,
    market_no_ask: int,
    our_yes_price_cents: int = None,
    our_no_price_cents: int = None,
    min_edge_cents: float = 1.0,
    calibration_monitor: Optional["RollingCalibrationMonitor"] = None,
) -> FairValueResult:
    """
    Main entry point: compute complete fair-value analysis.
    
    Args:
        model_probs: List of model probabilities with weights
        market_yes_bid: Market YES bid (cents)
        market_yes_ask: Market YES ask (cents)
        market_no_bid: Market NO bid (cents)
        market_no_ask: Market NO ask (cents)
        our_yes_price_cents: Our limit price for YES (cents), defaults to bid+1
        our_no_price_cents: Our limit price for NO (cents), defaults to bid+1
        min_edge_cents: Minimum edge in cents to recommend trade
        calibration_monitor: Optional calibration monitor
    
    Returns:
        FairValueResult with all computations
    """
    # Compute ensemble probability
    p_hat, p_hat_low, p_hat_high, ensemble_kl = compute_ensemble_probability(
        model_probs, calibration_monitor=calibration_monitor
    )
    
    # Default our prices: post at bid + 1 cent (favourite maker)
    if our_yes_price_cents is None:
        our_yes_price_cents = market_yes_bid + 1
    if our_no_price_cents is None:
        our_no_price_cents = market_no_bid + 1
    
    # Executable prices
    maker_yes_price = maker_executable_price(
        market_yes_bid, market_yes_ask, our_yes_price_cents, "yes"
    )
    maker_no_price = maker_executable_price(
        market_no_bid, market_no_ask, our_no_price_cents, "no"
    )
    taker_yes_price = taker_executable_price(market_yes_bid, market_yes_ask, "yes")
    taker_no_price = taker_executable_price(market_no_bid, market_no_ask, "no")
    
    # EV calculations
    ev_maker_yes = compute_ev(p_hat, maker_yes_price, "yes") if maker_yes_price else float("-inf")
    ev_maker_no = compute_ev(p_hat, maker_no_price, "no") if maker_no_price else float("-inf")
    ev_taker_yes = compute_ev(p_hat, taker_yes_price, "yes") if taker_yes_price else float("-inf")
    ev_taker_no = compute_ev(p_hat, taker_no_price, "no") if taker_no_price else float("-inf")
    
    # Edge in cents
    edge_maker_yes_cents = (p_hat - maker_yes_price) * 100 if maker_yes_price else float("-inf")
    edge_maker_no_cents = ((1 - p_hat) - maker_no_price) * 100 if maker_no_price else float("-inf")
    edge_taker_yes_cents = (p_hat - taker_yes_price) * 100 if taker_yes_price else float("-inf")
    edge_taker_no_cents = ((1 - p_hat) - taker_no_price) * 100 if taker_no_price else float("-inf")
    
    # Recommendation
    edges = {
        ("yes", "maker"): edge_maker_yes_cents,
        ("no", "maker"): edge_maker_no_cents,
        ("yes", "taker"): edge_taker_yes_cents,
        ("no", "taker"): edge_taker_no_cents,
    }
    
    # Filter by minimum edge
    valid_edges = {k: v for k, v in edges.items() if v >= min_edge_cents}
    
    if valid_edges:
        best = max(valid_edges, key=valid_edges.get)
        recommend_side, recommend_venue = best
        reason = f"Edge {valid_edges[best]:.1f}c >= min {min_edge_cents}c"
    else:
        recommend_side = "none"
        recommend_venue = "none"
        best_edge = max(edges.values()) if edges else 0
        reason = f"Best edge {best_edge:.1f}c < min {min_edge_cents}c"
    
    return FairValueResult(
        p_hat=p_hat,
        p_hat_low=p_hat_low,
        p_hat_high=p_hat_high,
        ensemble_kl=ensemble_kl,
        model_probs=model_probs,
        market_yes_bid=market_yes_bid,
        market_yes_ask=market_yes_ask,
        market_no_bid=market_no_bid,
        market_no_ask=market_no_ask,
        maker_yes_price=maker_yes_price or 0,
        maker_no_price=maker_no_price or 0,
        taker_yes_price=taker_yes_price or 0,
        taker_no_price=taker_no_price or 0,
        ev_maker_yes=ev_maker_yes,
        ev_maker_no=ev_maker_no,
        ev_taker_yes=ev_taker_yes,
        ev_taker_no=ev_taker_no,
        edge_maker_yes_cents=edge_maker_yes_cents,
        edge_maker_no_cents=edge_maker_no_cents,
        edge_taker_yes_cents=edge_taker_yes_cents,
        edge_taker_no_cents=edge_taker_no_cents,
        recommend_side=recommend_side,
        recommend_venue=recommend_venue,
        reason=reason,
    )


# ─── Integration helper for signals/gates ──────────────────────────────────────

def fair_value_from_gates_context(
    spot: float,
    strike: float,
    minutes_left: float,
    sigma_frac_per_min: float,
    market_yes_bid: int,
    market_yes_ask: int,
    market_no_bid: int,
    market_no_ask: int,
    our_yes_price_cents: int = None,
    our_no_price_cents: int = None,
    min_edge_cents: float = 1.0,
) -> FairValueResult:
    """
    Convenience function that builds model_probs from GBM + other models
    and computes fair value.
    
    This is the integration point for gates.py and signals.py.
    """
    # Build model probabilities
    model_probs = []
    
    # GBM model (point-in-time)
    p_gbm = gbm.gbm_probability_point(spot, strike, sigma_frac_per_min, minutes_left)
    model_probs.append(ModelProb("gbm", p_gbm, weight=1.0))
    
    # Arithmetic Brownian (gates cushion)
    dollar_sigma = gbm.dollar_sigma_from_fractional(spot, sigma_frac_per_min)
    p_arith = gbm.arithmetic_probability(spot, strike, dollar_sigma, minutes_left, "yes")
    model_probs.append(ModelProb("arithmetic", p_arith, weight=0.8))
    
    # BRTI average window model (if in final minutes)
    if minutes_left < 2.0:
        p_avg = gbm.gbm_probability_average_window(spot, strike, sigma_frac_per_min, minutes_left)
        model_probs.append(ModelProb("average_window", p_avg, weight=1.2))
    
    # Locked-share model (if in final minute)
    if minutes_left < 1.0:
        seconds_into = int((1.0 - minutes_left) * 60)
        p_locked = gbm.locked_share_gbm(spot, strike, sigma_frac_per_min, seconds_into)
        model_probs.append(ModelProb("locked_share", p_locked, weight=1.5))
    
    return compute_fair_value(
        model_probs=model_probs,
        market_yes_bid=market_yes_bid,
        market_yes_ask=market_yes_ask,
        market_no_bid=market_no_bid,
        market_no_ask=market_no_ask,
        our_yes_price_cents=our_yes_price_cents,
        our_no_price_cents=our_no_price_cents,
        min_edge_cents=min_edge_cents,
    )


# ─── RollingCalibrationMonitor (re-export from gbm for convenience) ────────────

from gbm import RollingCalibrationMonitor


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test
    spot = 95000.0
    strike = 95000.0
    minutes_left = 10.0
    sigma_annual = 0.80
    sigma_per_min = gbm.annualized_to_per_minute(sigma_annual)
    
    # Market prices (cents)
    market_yes_bid = 48
    market_yes_ask = 52
    market_no_bid = 48
    market_no_ask = 52
    
    result = fair_value_from_gates_context(
        spot=spot,
        strike=strike,
        minutes_left=minutes_left,
        sigma_frac_per_min=sigma_per_min,
        market_yes_bid=market_yes_bid,
        market_yes_ask=market_yes_ask,
        market_no_bid=market_no_bid,
        market_no_ask=market_no_ask,
        min_edge_cents=1.0,
    )
    
    print(f"Ensemble p_hat: {result.p_hat:.4f} [{result.p_hat_low:.4f}, {result.p_hat_high:.4f}]")
    print(f"Ensemble KL: {result.ensemble_kl:.4f}")
    print(f"Maker YES price: {result.maker_yes_price:.4f}, EV: {result.ev_maker_yes:.4f}, Edge: {result.edge_maker_yes_cents:.1f}c")
    print(f"Maker NO price: {result.maker_no_price:.4f}, EV: {result.ev_maker_no:.4f}, Edge: {result.edge_maker_no_cents:.1f}c")
    print(f"Taker YES price: {result.taker_yes_price:.4f}, EV: {result.ev_taker_yes:.4f}, Edge: {result.edge_taker_yes_cents:.1f}c")
    print(f"Taker NO price: {result.taker_no_price:.4f}, EV: {result.ev_taker_no:.4f}, Edge: {result.edge_taker_no_cents:.1f}c")
    print(f"Recommendation: {result.recommend_side} {result.recommend_venue} - {result.reason}")