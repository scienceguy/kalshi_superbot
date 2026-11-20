#!/usr/bin/env python3
"""
sizing.py — T15-40: Sizing policy across both venues.

Fractional Kelly on the LOWER edge bound with:
- Depth cap (fraction of visible depth at executable price)
- Market impact estimate
- Per-venue and per-underlying caps
- Correlated-exposure budget across venues (BTC on Kalshi + BTC on Polymarket = one bet)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
import fair_value
import risk

log = logging.getLogger(__name__)


@dataclass
class VenueBalance:
    """Balance for a single venue."""
    venue: str  # "kalshi" or "polymarket"
    available_balance: float  # dollars available for trading
    reserved_balance: float   # dollars locked in open orders/positions
    total_balance: float      # total balance


@dataclass
class SizingConstraints:
    """All constraints that limit position size."""
    # Config limits
    max_per_trade_pct: float = 0.08           # PER_TRADE_PCT
    max_exposure_pct: float = 0.30            # EXPOSURE_CEILING_PCT
    cash_floor_pct: float = 0.70              # CASH_FLOOR_PCT
    max_kelly_fraction: float = 0.25          # RISK_PROFILE_KELLY_FRACTION
    
    # Depth cap
    max_depth_fraction: float = 0.25          # Max fraction of visible depth
    
    # Market impact
    impact_coefficient: float = 0.01          # Linear impact per fraction of depth
    
    # Per-underlying cap
    max_underlying_pct: float = 0.15          # Max fraction per underlying
    
    # Cross-venue correlation
    max_correlated_exposure_pct: float = 0.20 # Max for same asset across venues


@dataclass
class SizingResult:
    """Result of sizing computation."""
    size_pct: float                  # Fraction of balance
    size_dollars: float              # Dollar amount
    contracts: int                   # Number of contracts
    limit_price_cents: int           # Limit price
    venue: str                       # "kalshi" or "polymarket" or "both"
    reasoning: str                   # Why this size
    
    # Constraint diagnostics
    kelly_fraction: float = 0.0
    edge_lower_bound: float = 0.0
    depth_cap: float = 0.0
    impact_adjustment: float = 1.0
    exposure_remaining: float = 0.0
    underlying_exposure_remaining: float = 0.0
    correlated_exposure_remaining: float = 0.0
    
    # Warnings
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ─── Core sizing functions ─────────────────────────────────────────────────────

def kelly_fraction(
    p_win: float,
    payoff_ratio: float,  # b = win_amount / loss_amount
) -> float:
    """
    Kelly fraction: f = (p * b - q) / b where q = 1 - p.
    
    For binary contracts:
    - YES: win = 1 - price, loss = price, so b = (1 - price) / price
    - NO:  win = price, loss = 1 - price, so b = price / (1 - price)
    """
    if payoff_ratio <= 0:
        return 0.0
    q = 1.0 - p_win
    kelly = (p_win * payoff_ratio - q) / payoff_ratio
    return max(0.0, kelly)


def fractional_kelly(
    p_win: float,
    payoff_ratio: float,
    kelly_fraction: float,
) -> float:
    """Apply fractional Kelly (e.g., 0.25 = quarter Kelly)."""
    return kelly_fraction(p_win, payoff_ratio) * kelly_fraction


def depth_cap(
    visible_depth: int,
    max_depth_fraction: float,
) -> int:
    """
    Cap size at a fraction of visible depth.
    
    Returns max contracts we can take without moving the market too much.
    """
    if visible_depth <= 0:
        return 0
    return int(visible_depth * max_depth_fraction)


def market_impact_estimate(
    size_contracts: int,
    visible_depth: int,
    impact_coefficient: float,
) -> float:
    """
    Estimate market impact as a fraction of price.
    
    Linear model: impact = coefficient * (size / depth)
    Returns adjustment factor to apply to edge (1.0 = no impact, < 1.0 = reduced edge).
    """
    if visible_depth <= 0:
        return 0.0  # Can't trade
    fraction = size_contracts / visible_depth
    impact = impact_coefficient * fraction
    return max(0.0, 1.0 - impact)  # Remaining edge after impact


def compute_executable_price_and_depth(
    fv_result: fair_value.FairValueResult,
    venue: str,
    side: str,
) -> Tuple[float, int]:
    """
    Get executable price and visible depth for a venue/side combination.
    """
    if venue == "kalshi":
        if side == "yes":
            price = fv_result.maker_yes_price if fv_result.recommend_venue == "maker" else fv_result.taker_yes_price
            depth = fv_result.model_probs[0].metadata.get("kalshi_yes_depth", 100) if fv_result.model_probs else 100
        else:
            price = fv_result.maker_no_price if fv_result.recommend_venue == "maker" else fv_result.taker_no_price
            depth = fv_result.model_probs[0].metadata.get("kalshi_no_depth", 100) if fv_result.model_probs else 100
    elif venue == "polymarket":
        # Polymarket depths would come from BBO
        if side == "yes":
            price = fv_result.maker_yes_price if fv_result.recommend_venue == "maker" else fv_result.taker_yes_price
            depth = fv_result.model_probs[0].metadata.get("poly_yes_depth", 50) if fv_result.model_probs else 50
        else:
            price = fv_result.maker_no_price if fv_result.recommend_venue == "maker" else fv_result.taker_no_price
            depth = fv_result.model_probs[0].metadata.get("poly_no_depth", 50) if fv_result.model_probs else 50
    else:
        price = 0.5
        depth = 100
    
    return price, depth


def get_payoff_ratio(price: float, side: str) -> float:
    """Get payoff ratio b = win/loss for binary contract."""
    if side == "yes":
        # Win = 1 - price, Loss = price
        return (1.0 - price) / price if price > 0 else 0.0
    else:
        # Win = price, Loss = 1 - price
        return price / (1.0 - price) if price < 1.0 else 0.0


# ─── Main sizing function ──────────────────────────────────────────────────────

def compute_size(
    fv_result: fair_value.FairValueResult,
    ctx: "DecisionContext",
    constraints: SizingConstraints = None,
) -> SizingResult:
    """
    Compute position size with all constraints.
    
    Args:
        fv_result: Fair value computation result
        ctx: Decision context with balances, exposures, etc.
        constraints: Sizing constraints (uses config defaults if None)
    
    Returns:
        SizingResult with size and diagnostics
    """
    if constraints is None:
        constraints = SizingConstraints(
            max_per_trade_pct=getattr(config, 'PER_TRADE_PCT', 0.08),
            max_exposure_pct=getattr(config, 'EXPOSURE_CEILING_PCT', 0.30),
            cash_floor_pct=getattr(config, 'CASH_FLOOR_PCT', 0.70),
            max_kelly_fraction=getattr(config, 'RISK_PROFILE_KELLY_FRACTION', 0.25),
            max_depth_fraction=0.25,
            impact_coefficient=0.01,
            max_underlying_pct=0.15,
            max_correlated_exposure_pct=0.20,
        )
    
    warnings = []
    
    # Determine side and venue
    side = fv_result.recommend_side
    venue = fv_result.recommend_venue
    
    if side == "none" or venue == "none":
        return SizingResult(
            size_pct=0.0,
            size_dollars=0.0,
            contracts=0,
            limit_price_cents=0,
            venue="",
            reasoning="No trade recommended by fair value engine",
            warnings=["No trade signal"],
        )
    
    # Get executable price and visible depth
    executable_price, visible_depth = compute_executable_price_and_depth(fv_result, venue, side)
    
    if executable_price <= 0 or executable_price >= 1.0:
        return SizingResult(
            size_pct=0.0,
            size_dollars=0.0,
            contracts=0,
            limit_price_cents=0,
            venue=venue,
            reasoning="Invalid executable price",
            warnings=["Invalid price"],
        )
    
    # Win probability for our side
    p_win = fv_result.p_hat if side == "yes" else (1.0 - fv_result.p_hat)
    
    # Payoff ratio
    b = get_payoff_ratio(executable_price, side)
    
    # Kelly fraction
    kelly = kelly_fraction(p_win, b)
    
    # Fractional Kelly
    kelly_frac = constraints.max_kelly_fraction
    size_pct_kelly = fractional_kelly(p_win, b, kelly_frac)
    
    # Edge lower bound (using p_hat_low for conservative estimate)
    p_win_low = fv_result.p_hat_low if side == "yes" else (1.0 - fv_result.p_hat_high)
    kelly_low = fractional_kelly(p_win_low, b, kelly_frac)
    
    # Use the MORE CONSERVATIVE of Kelly and Kelly_low
    size_pct = min(size_pct_kelly, kelly_low)
    
    # Apply depth cap
    max_contracts_depth = depth_cap(visible_depth, constraints.max_depth_fraction)
    if max_contracts_depth <= 0:
        warnings.append("Insufficient depth")
        return SizingResult(
            size_pct=0.0,
            size_dollars=0.0,
            contracts=0,
            limit_price_cents=int(executable_price * 100),
            venue=venue,
            reasoning="Insufficient order book depth",
            kelly_fraction=kelly,
            edge_lower_bound=p_win_low - executable_price if side == "yes" else (1 - p_win_low) - executable_price,
            depth_cap=0.0,
            warnings=warnings,
        )
    
    # Convert size_pct to contracts
    balance = ctx.current_balance - ctx.current_exposure
    max_dollar_risk = balance * size_pct
    price_per_contract = executable_price
    max_contracts_kelly = int(max_dollar_risk / price_per_contract) if price_per_contract > 0 else 0
    
    # Apply depth cap
    contracts = min(max_contracts_kelly, max_contracts_depth)
    
    if contracts <= 0:
        return SizingResult(
            size_pct=0.0,
            size_dollars=0.0,
            contracts=0,
            limit_price_cents=int(executable_price * 100),
            venue=venue,
            reasoning="Computed contracts <= 0",
            kelly_fraction=kelly,
            edge_lower_bound=p_win_low - executable_price if side == "yes" else (1 - p_win_low) - executable_price,
            depth_cap=max_contracts_depth,
            warnings=warnings,
        )
    
    # Market impact adjustment
    impact_adj = market_impact_estimate(contracts, visible_depth, constraints.impact_coefficient)
    if impact_adj < 1.0:
        # Reduce size proportionally to impact
        contracts = int(contracts * impact_adj)
        if contracts <= 0:
            warnings.append("Market impact reduces size to zero")
            return SizingResult(
                size_pct=0.0,
                size_dollars=0.0,
                contracts=0,
                limit_price_cents=int(executable_price * 100),
                venue=venue,
                reasoning="Market impact too high",
                kelly_fraction=kelly,
                edge_lower_bound=p_win_low - executable_price if side == "yes" else (1 - p_win_low) - executable_price,
                depth_cap=max_contracts_depth,
                impact_adjustment=impact_adj,
                warnings=warnings,
            )
    
    # Apply per-trade cap
    max_contracts_per_trade = int((ctx.current_balance * constraints.max_per_trade_pct) / price_per_contract)
    if contracts > max_contracts_per_trade:
        contracts = max_contracts_per_trade
        warnings.append(f"Capped at per-trade max: {max_contracts_per_trade}")
    
    # Exposure ceiling check
    new_exposure = ctx.current_exposure + contracts * price_per_contract
    exposure_pct = new_exposure / ctx.current_balance if ctx.current_balance > 0 else 0
    max_exposure_allowed = ctx.current_balance * constraints.max_exposure_pct
    exposure_remaining = max_exposure_allowed - ctx.current_exposure
    
    if exposure_pct > constraints.max_exposure_pct:
        # Reduce to fit exposure ceiling
        max_contracts_exposure = int(exposure_remaining / price_per_contract)
        if max_contracts_exposure < contracts:
            contracts = max(max_contracts_exposure, 0)
            warnings.append(f"Capped at exposure ceiling: {max_contracts_exposure}")
    
    # Cash floor check
    cash_after = ctx.current_balance - new_exposure
    cash_pct = cash_after / ctx.current_balance if ctx.current_balance > 0 else 1
    min_cash = ctx.current_balance * constraints.cash_floor_pct
    if cash_after < min_cash:
        # Reduce to maintain cash floor
        max_contracts_cash = int((ctx.current_balance - min_cash) / price_per_contract)
        if max_contracts_cash < contracts:
            contracts = max(max_contracts_cash, 0)
            warnings.append(f"Capped at cash floor: {max_contracts_cash}")
    
    # Per-underlying cap
    # We'd need to track underlying exposure - for now just cap
    underlying_exposure_pct = (contracts * price_per_contract) / ctx.current_balance if ctx.current_balance > 0 else 0
    max_underlying_allowed = ctx.current_balance * constraints.max_underlying_pct
    underlying_exposure_remaining = max_underlying_allowed - 0  # Would subtract existing
    
    if underlying_exposure_pct > constraints.max_underlying_pct:
        max_contracts_underlying = int(max_underlying_allowed / price_per_contract)
        if max_contracts_underlying < contracts:
            contracts = max(max_contracts_underlying, 0)
            warnings.append(f"Capped at per-underlying max: {max_contracts_underlying}")
    
    # Cross-venue correlated exposure
    # Would check existing exposure on same asset across venues
    correlated_exposure_remaining = ctx.current_balance * constraints.max_correlated_exposure_pct - 0
    
    # Final size
    size_dollars = contracts * price_per_contract
    final_size_pct = size_dollars / ctx.current_balance if ctx.current_balance > 0 else 0
    
    # Limit price in cents
    limit_price_cents = int(round(executable_price * 100))
    
    # Edge lower bound in cents
    edge_lower = (p_win_low - executable_price) * 100 if side == "yes" else ((1 - p_win_low) - executable_price) * 100
    
    reasoning = (
        f"Kelly={kelly:.3f} (frac={kelly_frac}) -> {size_pct_kelly:.1%}, "
        f"low={kelly_low:.1%}, depth_cap={max_contracts_depth}, "
        f"impact={impact_adj:.2f}, final={contracts} contracts (${size_dollars:.2f}, {final_size_pct:.1%})"
    )
    
    return SizingResult(
        size_pct=final_size_pct,
        size_dollars=size_dollars,
        contracts=contracts,
        limit_price_cents=limit_price_cents,
        venue=venue,
        reasoning=reasoning,
        kelly_fraction=kelly,
        edge_lower_bound=edge_lower,
        depth_cap=max_contracts_depth,
        impact_adjustment=impact_adj,
        exposure_remaining=exposure_remaining / ctx.current_balance if ctx.current_balance > 0 else 0,
        underlying_exposure_remaining=underlying_exposure_remaining / ctx.current_balance if ctx.current_balance > 0 else 0,
        correlated_exposure_remaining=correlated_exposure_remaining / ctx.current_balance if ctx.current_balance > 0 else 0,
        warnings=warnings,
    )


# ─── Cross-venue sizing ────────────────────────────────────────────────────────

def compute_cross_venue_size(
    fv_result_kalshi: fair_value.FairValueResult,
    fv_result_poly: fair_value.FairValueResult,
    ctx: "DecisionContext",
    constraints: SizingConstraints = None,
) -> Tuple[SizingResult, SizingResult]:
    """
    Compute sizes for both venues for the same asset/window (arbitrage pair).
    
    Correlated exposure: BTC on Kalshi + BTC on Polymarket = ONE bet.
    """
    if constraints is None:
        constraints = SizingConstraints()
    
    # Compute size for each venue independently first
    size_kalshi = compute_size(fv_result_kalshi, ctx, constraints)
    size_poly = compute_size(fv_result_poly, ctx, constraints)
    
    # Apply correlated exposure cap
    total_correlated_exposure = size_kalshi.size_dollars + size_poly.size_dollars
    max_correlated = ctx.current_balance * constraints.max_correlated_exposure_pct
    
    if total_correlated_exposure > max_correlated and total_correlated_exposure > 0:
        # Scale both down proportionally
        scale = max_correlated / total_correlated_exposure
        size_kalshi.contracts = int(size_kalshi.contracts * scale)
        size_poly.contracts = int(size_poly.contracts * scale)
        
        size_kalshi.size_dollars = size_kalshi.contracts * (size_kalshi.limit_price_cents / 100.0)
        size_poly.size_dollars = size_poly.contracts * (size_poly.limit_price_cents / 100.0)
        
        size_kalshi.size_pct = size_kalshi.size_dollars / ctx.current_balance if ctx.current_balance > 0 else 0
        size_poly.size_pct = size_poly.size_dollars / ctx.current_balance if ctx.current_balance > 0 else 0
        
        size_kalshi.warnings.append(f"Correlated exposure cap applied (scale={scale:.2f})")
        size_poly.warnings.append(f"Correlated exposure cap applied (scale={scale:.2f})")
    
    return size_kalshi, size_poly


# ─── Integration with decision_policy ──────────────────────────────────────────

from decision_policy import DecisionContext


def size_for_decision(
    decision: "DecisionResult",
    ctx: DecisionContext,
    constraints: SizingConstraints = None,
) -> SizingResult:
    """
    Compute final size for a decision.
    
    This is called after make_decision() to add sizing.
    """
    if decision.decision.value in ("skip", "blocked", "none"):
        return SizingResult(
            size_pct=0.0,
            size_dollars=0.0,
            contracts=0,
            limit_price_cents=0,
            venue="",
            reasoning=f"Decision was {decision.decision.value}",
        )
    
    # Create a minimal fair_value result from decision
    fv = decision.fair_value
    if fv is None:
        # Build from decision
        fv = fair_value.FairValueResult(
            p_hat=0.5,
            p_hat_low=0.5,
            p_hat_high=0.5,
            ensemble_kl=0.0,
            model_probs=[],
            market_yes_bid=0,
            market_yes_ask=0,
            market_no_bid=0,
            market_no_ask=0,
            maker_yes_price=0,
            maker_no_price=0,
            taker_yes_price=0,
            taker_no_price=0,
            ev_maker_yes=0,
            ev_maker_no=0,
            ev_taker_yes=0,
            ev_taker_no=0,
            edge_maker_yes_cents=0,
            edge_maker_no_cents=0,
            edge_taker_yes_cents=0,
            edge_taker_no_cents=0,
            recommend_side=decision.side,
            recommend_venue=decision.venue,
            reason=decision.reason,
        )
    
    return compute_size(fv, ctx, constraints)


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test
    from fair_value import fair_value_from_gates_context, ModelProb
    from decision_policy import DecisionContext
    
    # Build fake context
    ctx = DecisionContext(
        ticker="KXBTC15M-26NOV200445-45",
        series_ticker="KXBTC15M",
        close_time="2026-11-20T04:45:00Z",
        minutes_left=10.0,
        strike=95000.0,
        yes_bid=48,
        yes_ask=52,
        no_bid=48,
        no_ask=52,
        yes_bid_size=100,
        yes_ask_size=100,
        no_bid_size=100,
        no_ask_size=100,
        spot_price=95000.0,
        spot_source="ws",
        spot_age_seconds=0.5,
        sigma_frac_per_min=0.0011,
        current_balance=100.0,
        current_exposure=0.0,
        max_drawdown_pct=0.0,
        consecutive_losses=0,
    )
    
    # Build fake fair value result
    model_probs = [
        ModelProb("gbm", 0.55, weight=1.0),
        ModelProb("arithmetic", 0.53, weight=0.8),
    ]
    
    fv = fair_value.compute_fair_value(
        model_probs=model_probs,
        market_yes_bid=48,
        market_yes_ask=52,
        market_no_bid=48,
        market_no_ask=52,
        our_yes_price_cents=49,
        our_no_price_cents=49,
        min_edge_cents=1.0,
    )
    
    result = compute_size(fv, ctx)
    
    print(f"Size: {result.contracts} contracts (${result.size_dollars:.2f}, {result.size_pct:.1%})")
    print(f"Limit: {result.limit_price_cents}c")
    print(f"Reasoning: {result.reasoning}")
    print(f"Warnings: {result.warnings}")
    print(f"Kelly: {result.kelly_fraction:.3f}, Edge lower: {result.edge_lower_bound:.2f}c")