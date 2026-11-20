#!/usr/bin/env python3
"""
decision_policy.py — T15-25: Decision policy.

Combines gates, fair-value, KL health, and calibration breaker
into one documented decision function for entry/exit/sizing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import config
import fair_value
import gates
import gbm
import kl_divergence
import risk

log = logging.getLogger(__name__)


class Decision(Enum):
    """Trading decision."""
    ENTER_YES = "enter_yes"
    ENTER_NO = "enter_no"
    SKIP = "skip"
    BLOCKED = "blocked"
    REDUCE = "reduce"
    EXIT = "exit"


class BlockReason(Enum):
    """Reasons for blocking a trade."""
    GATE_ATR = "gate_atr"
    GATE_EV = "gate_ev"
    GATE_CUSHION = "gate_cushion"
    GATE_HIGH_GAMMA = "gate_high_gamma"
    GATE_TIME = "gate_time"
    GATE_SPREAD = "gate_spread"
    FAIR_VALUE_EDGE = "fair_value_edge"
    KL_DISAGREEMENT = "kl_disagreement"
    CALIBRATION_DRIFT = "calibration_drift"
    RISK_DRAWDOWN = "risk_drawdown"
    RISK_EXPOSURE = "risk_exposure"
    RISK_CASH_FLOOR = "risk_cash_floor"
    RISK_CONSECUTIVE_LOSS = "risk_consecutive_loss"
    BOOK_STALE = "book_stale"
    NO_LIQUIDITY = "no_liquidity"


@dataclass
class DecisionContext:
    """All inputs needed for a decision."""
    # Market data
    ticker: str
    series_ticker: str
    close_time: str
    minutes_left: float
    strike: float
    
    # Orderbook
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    yes_bid_size: int
    yes_ask_size: int
    no_bid_size: int
    no_ask_size: int
    
    # Spot / reference
    spot_price: float
    spot_source: str
    spot_age_seconds: float
    brti_60s_avg: Optional[float] = None
    brti_age_seconds: Optional[float] = None
    
    # Models
    sigma_frac_per_min: float = 0.0011
    model_probs: Optional[List[fair_value.ModelProb]] = None
    
    # Risk state
    current_balance: float = 0.0
    current_exposure: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    
    # KL calibration monitor
    calibration_monitor: Optional[gbm.RollingCalibrationMonitor] = None
    
    # Venue books (for book-age gate)
    kalshi_book_age: Optional[float] = None
    polymarket_book_age: Optional[float] = None
    
    # Cross-venue arb data
    arb_opportunity: bool = False
    arb_edge_cents: float = 0.0


@dataclass
class DecisionResult:
    """Complete decision output."""
    decision: Decision
    side: str  # "yes", "no", ""
    venue: str  # "maker", "taker", ""
    size_pct: float  # Fraction of balance/kelly
    limit_price_cents: int
    reason: str
    blocked_reasons: List[BlockReason]
    gate_results: Dict
    fair_value: Optional[fair_value.FairValueResult]
    kl_health: Dict
    calibration_health: Dict
    risk_health: Dict


# ─── Gate evaluation ──────────────────────────────────────────────────────────

def evaluate_gates(ctx: DecisionContext) -> Tuple[bool, List[BlockReason], Dict]:
    """
    Evaluate all pre-entry gates using gates.evaluate_entry.
    
    Returns (all_pass, block_reasons, gate_metrics)
    """
    block_reasons = []
    gate_metrics = {}
    
    from gates import evaluate_entry, GateResult
    from types import SimpleNamespace
    
    # Build market object for gates (needs attributes, not dict)
    market = SimpleNamespace(
        ticker=ctx.ticker,
        close_time=ctx.close_time,
        minutes_left=ctx.minutes_left,
        strike=ctx.strike,
        mid_yes=(ctx.yes_bid + ctx.yes_ask) / 2 if ctx.yes_bid and ctx.yes_ask else 0,
        yes_ask=ctx.yes_ask,
        no_ask=ctx.no_ask,
    )
    
    # Build spot_state
    spot_state = {
        "spot": ctx.spot_price,
        "sigma_per_min": ctx.sigma_frac_per_min,  # fractional per minute
        "atr_1m": None,  # Will be fetched by gates if needed
    }
    
    orderbook = SimpleNamespace(
        yes_bid=ctx.yes_bid,
        yes_ask=ctx.yes_ask,
        no_bid=ctx.no_bid,
        no_ask=ctx.no_ask,
    )
    
    # Evaluate for both sides (we'll check both YES and NO)
    # Use side=None to let gates auto-detect from mid
    gate_result = evaluate_entry(market, spot_state=spot_state, orderbook=orderbook, side=None)
    
    gate_metrics = gate_result.metrics
    
    if not gate_result.allowed:
        for reason in gate_result.reasons:
            if "atr" in reason.lower():
                block_reasons.append(BlockReason.GATE_ATR)
            elif "ev" in reason.lower():
                block_reasons.append(BlockReason.GATE_EV)
            elif "cushion" in reason.lower():
                block_reasons.append(BlockReason.GATE_CUSHION)
            elif "high-gamma" in reason.lower():
                block_reasons.append(BlockReason.GATE_HIGH_GAMMA)
            elif "time" in reason.lower() or "window" in reason.lower():
                block_reasons.append(BlockReason.GATE_TIME)
            elif "spread" in reason.lower():
                block_reasons.append(BlockReason.GATE_SPREAD)
    
    return gate_result.allowed, block_reasons, gate_metrics


# ─── Fair-value evaluation ────────────────────────────────────────────────────

def evaluate_fair_value(ctx: DecisionContext) -> Tuple[fair_value.FairValueResult, List[BlockReason]]:
    """
    Evaluate fair-value engine.
    
    Returns (fair_value_result, block_reasons)
    """
    block_reasons = []
    
    # Build model probs if not provided
    if ctx.model_probs is None:
        # Use default models from spot
        model_probs = []
        
        p_gbm = gbm.gbm_probability_point(
            ctx.spot_price, ctx.strike, ctx.sigma_frac_per_min, ctx.minutes_left
        )
        model_probs.append(fair_value.ModelProb("gbm", p_gbm, weight=1.0))
        
        dollar_sigma = gbm.dollar_sigma_from_fractional(ctx.spot_price, ctx.sigma_frac_per_min)
        p_arith = gbm.arithmetic_probability(
            ctx.spot_price, ctx.strike, dollar_sigma, ctx.minutes_left, "yes"
        )
        model_probs.append(fair_value.ModelProb("arithmetic", p_arith, weight=0.8))
        
        if ctx.minutes_left < 2.0:
            p_avg = gbm.gbm_probability_average_window(
                ctx.spot_price, ctx.strike, ctx.sigma_frac_per_min, ctx.minutes_left
            )
            model_probs.append(fair_value.ModelProb("average_window", p_avg, weight=1.2))
        
        if ctx.minutes_left < 1.0:
            seconds_into = int((1.0 - ctx.minutes_left) * 60)
            p_locked = gbm.locked_share_gbm(
                ctx.spot_price, ctx.strike, ctx.sigma_frac_per_min, seconds_into
            )
            model_probs.append(fair_value.ModelProb("locked_share", p_locked, weight=1.5))
    else:
        model_probs = ctx.model_probs
    
    fv_result = fair_value.compute_fair_value(
        model_probs=model_probs,
        market_yes_bid=ctx.yes_bid,
        market_yes_ask=ctx.yes_ask,
        market_no_bid=ctx.no_bid,
        market_no_ask=ctx.no_ask,
        our_yes_price_cents=ctx.yes_bid + 1,
        our_no_price_cents=ctx.no_bid + 1,
        min_edge_cents=config.FAIR_VALUE_MIN_EDGE_CENTS if hasattr(config, 'FAIR_VALUE_MIN_EDGE_CENTS') else 1.0,
        calibration_monitor=ctx.calibration_monitor,
    )
    
    # Check edge threshold
    if fv_result.recommend_side == "none":
        block_reasons.append(BlockReason.FAIR_VALUE_EDGE)
    
    return fv_result, block_reasons


# ─── KL health evaluation ─────────────────────────────────────────────────────

def evaluate_kl_health(ctx: DecisionContext) -> Tuple[Dict, List[BlockReason]]:
    """
    Evaluate KL divergence health (ensemble disagreement, cross-venue).
    
    Returns (kl_metrics, block_reasons)
    """
    block_reasons = []
    kl_metrics = {}
    
    if ctx.model_probs and len(ctx.model_probs) >= 2:
        probs = [m.prob_yes for m in ctx.model_probs]
        kl_metrics["ensemble_kl"] = kl_divergence.ensemble_kl(probs)
        kl_metrics["ensemble_disagreement"] = kl_metrics["ensemble_kl"]
        
        # Check disagreement gate
        threshold = config.KL_DISAGREEMENT_THRESHOLD if hasattr(config, 'KL_DISAGREEMENT_THRESHOLD') else 0.1
        if kl_metrics["ensemble_kl"] > threshold:
            block_reasons.append(BlockReason.KL_DISAGREEMENT)
            kl_metrics["disagreement_blocked"] = True
        else:
            kl_metrics["disagreement_blocked"] = False
    else:
        kl_metrics["ensemble_kl"] = 0.0
        kl_metrics["disagreement_blocked"] = False
    
    # Cross-venue symmetric KL if arb data available
    if ctx.arb_opportunity and ctx.arb_edge_cents > 0:
        # We have cross-venue data - could compute symmetric KL here
        kl_metrics["cross_venue_arb"] = True
        kl_metrics["arb_edge_cents"] = ctx.arb_edge_cents
    else:
        kl_metrics["cross_venue_arb"] = False
    
    return kl_metrics, block_reasons


# ─── Calibration health evaluation ────────────────────────────────────────────

def evaluate_calibration_health(ctx: DecisionContext) -> Tuple[Dict, List[BlockReason]]:
    """
    Evaluate calibration monitor health (drift detection).
    
    Returns (cal_metrics, block_reasons)
    """
    block_reasons = []
    cal_metrics = {}
    
    if ctx.calibration_monitor is not None:
        cal_metrics = ctx.calibration_monitor.get_metrics()
        
        if cal_metrics.get("status") == "calibrated":
            # Check PSI if available (would need expected distribution)
            # For now, check Brier improvement
            brier_improvement = cal_metrics.get("brier_improvement", 0)
            if brier_improvement < -0.01:  # Negative improvement = worse
                block_reasons.append(BlockReason.CALIBRATION_DRIFT)
                cal_metrics["drift_detected"] = True
            else:
                cal_metrics["drift_detected"] = False
        else:
            cal_metrics["drift_detected"] = False  # Not enough data yet
    else:
        cal_metrics["status"] = "no_monitor"
        cal_metrics["drift_detected"] = False
    
    return cal_metrics, block_reasons


# ─── Risk health evaluation ───────────────────────────────────────────────────

def evaluate_risk_health(ctx: DecisionContext) -> Tuple[Dict, List[BlockReason]]:
    """
    Evaluate risk limits.
    
    Returns (risk_metrics, block_reasons)
    """
    block_reasons = []
    risk_metrics = {}
    
    # Drawdown halt
    max_dd = config.MAX_DRAWDOWN_PCT if hasattr(config, 'MAX_DRAWDOWN_PCT') else 0.10
    if ctx.max_drawdown_pct > max_dd:
        block_reasons.append(BlockReason.RISK_DRAWDOWN)
        risk_metrics["drawdown_halt"] = True
    else:
        risk_metrics["drawdown_halt"] = False
    risk_metrics["max_drawdown_pct"] = ctx.max_drawdown_pct
    
    # Exposure ceiling
    exposure_ceiling = config.EXPOSURE_CEILING_PCT if hasattr(config, 'EXPOSURE_CEILING_PCT') else 0.30
    exposure_pct = ctx.current_exposure / ctx.current_balance if ctx.current_balance > 0 else 0
    if exposure_pct > exposure_ceiling:
        block_reasons.append(BlockReason.RISK_EXPOSURE)
        risk_metrics["exposure_halt"] = True
    else:
        risk_metrics["exposure_halt"] = False
    risk_metrics["exposure_pct"] = exposure_pct
    
    # Cash floor
    cash_floor = config.CASH_FLOOR_PCT if hasattr(config, 'CASH_FLOOR_PCT') else 0.70
    cash_pct = (ctx.current_balance - ctx.current_exposure) / ctx.current_balance if ctx.current_balance > 0 else 1
    if cash_pct < cash_floor:
        block_reasons.append(BlockReason.RISK_CASH_FLOOR)
        risk_metrics["cash_floor_halt"] = True
    else:
        risk_metrics["cash_floor_halt"] = False
    risk_metrics["cash_pct"] = cash_pct
    
    # Consecutive loss cooldown
    max_consecutive = config.MAX_CONSECUTIVE_LOSSES if hasattr(config, 'MAX_CONSECUTIVE_LOSSES') else 3
    if ctx.consecutive_losses >= max_consecutive:
        block_reasons.append(BlockReason.RISK_CONSECUTIVE_LOSS)
        risk_metrics["consecutive_loss_halt"] = True
    else:
        risk_metrics["consecutive_loss_halt"] = False
    risk_metrics["consecutive_losses"] = ctx.consecutive_losses
    
    return risk_metrics, block_reasons


# ─── Book-age gate ────────────────────────────────────────────────────────────

def evaluate_book_age(ctx: DecisionContext) -> Tuple[bool, List[BlockReason]]:
    """
    Evaluate book-age gate (from venue_books.py).
    
    Returns (is_tradable, block_reasons)
    """
    block_reasons = []
    
    max_age = config.BOOK_MAX_AGE_SECONDS if hasattr(config, 'BOOK_MAX_AGE_SECONDS') else 2.0
    
    if ctx.kalshi_book_age is not None and ctx.kalshi_book_age > max_age:
        block_reasons.append(BlockReason.BOOK_STALE)
        return False, block_reasons
    
    if ctx.polymarket_book_age is not None and ctx.polymarket_book_age > max_age:
        block_reasons.append(BlockReason.BOOK_STALE)
        return False, block_reasons
    
    return True, block_reasons


# ─── Liquidity check ──────────────────────────────────────────────────────────

def evaluate_liquidity(ctx: DecisionContext) -> Tuple[bool, List[BlockReason]]:
    """
    Check minimum liquidity for fill probability.
    
    Returns (has_liquidity, block_reasons)
    """
    block_reasons = []
    
    # Need minimum size on both sides for maker orders
    min_size = config.MIN_BOOK_SIZE if hasattr(config, 'MIN_BOOK_SIZE') else 10
    
    if ctx.yes_bid_size < min_size and ctx.yes_ask_size < min_size:
        block_reasons.append(BlockReason.NO_LIQUIDITY)
        return False, block_reasons
    
    if ctx.no_bid_size < min_size and ctx.no_ask_size < min_size:
        block_reasons.append(BlockReason.NO_LIQUIDITY)
        return False, block_reasons
    
    return True, block_reasons


# ─── Sizing computation ───────────────────────────────────────────────────────

def compute_size(
    fv_result: fair_value.FairValueResult,
    ctx: DecisionContext,
    gate_metrics: Dict,
) -> float:
    """
    Compute position size as fraction of balance.
    
    Uses fractional Kelly with risk adjustments.
    """
    # Base Kelly fraction from config
    kelly_fraction = config.RISK_PROFILE_KELLY_FRACTION if hasattr(config, 'RISK_PROFILE_KELLY_FRACTION') else 0.25
    
    # Edge in probability terms
    if fv_result.recommend_side == "yes":
        edge = fv_result.edge_maker_yes_cents / 100.0 if fv_result.recommend_venue == "maker" else fv_result.edge_taker_yes_cents / 100.0
    else:
        edge = fv_result.edge_maker_no_cents / 100.0 if fv_result.recommend_venue == "maker" else fv_result.edge_taker_no_cents / 100.0
    
    if edge <= 0:
        return 0.0
    
    # Win probability
    p_win = fv_result.p_hat if fv_result.recommend_side == "yes" else (1.0 - fv_result.p_hat)
    
    # Kelly: f = (p * b - q) / b where b = payoff ratio
    # For binary: payoff = (1 - price) / price for YES, price / (1 - price) for NO
    if fv_result.recommend_side == "yes":
        price = fv_result.maker_yes_price if fv_result.recommend_venue == "maker" else fv_result.taker_yes_price
    else:
        price = fv_result.maker_no_price if fv_result.recommend_venue == "maker" else fv_result.taker_no_price
    
    if price <= 0 or price >= 1:
        return 0.0
    
    if fv_result.recommend_side == "yes":
        b = (1.0 - price) / price
    else:
        b = price / (1.0 - price)
    
    kelly = (p_win * b - (1 - p_win)) / b if b > 0 else 0
    kelly = max(0, kelly)  # No negative Kelly
    
    # Apply fractional Kelly
    size = kelly * kelly_fraction
    
    # Cap at per-trade max
    max_per_trade = config.PER_TRADE_PCT if hasattr(config, 'PER_TRADE_PCT') else 0.08
    size = min(size, max_per_trade)
    
    # Reduce if high ensemble KL
    ensemble_kl = gate_metrics.get("ensemble_kl", 0)
    if ensemble_kl > 0.05:
        size *= max(0.5, 1.0 - ensemble_kl * 5)
    
    # Reduce if calibration drift
    if gate_metrics.get("drift_detected", False):
        size *= 0.5
    
    return size


# ─── Main decision function ───────────────────────────────────────────────────

def make_decision(ctx: DecisionContext) -> DecisionResult:
    """
    Main decision function combining all components.
    
    This is the single entry point for trading decisions.
    """
    all_block_reasons = []
    
    # 1. Evaluate gates
    gates_pass, gate_blocks, gate_metrics = evaluate_gates(ctx)
    all_block_reasons.extend(gate_blocks)
    
    # 2. Evaluate fair value
    fv_result, fv_blocks = evaluate_fair_value(ctx)
    all_block_reasons.extend(fv_blocks)
    
    # 3. Evaluate KL health
    kl_metrics, kl_blocks = evaluate_kl_health(ctx)
    all_block_reasons.extend(kl_blocks)
    
    # 4. Evaluate calibration health
    cal_metrics, cal_blocks = evaluate_calibration_health(ctx)
    all_block_reasons.extend(cal_blocks)
    
    # 5. Evaluate risk health
    risk_metrics, risk_blocks = evaluate_risk_health(ctx)
    all_block_reasons.extend(risk_blocks)
    
    # 6. Evaluate book age
    book_tradable, book_blocks = evaluate_book_age(ctx)
    if not book_tradable:
        all_block_reasons.extend(book_blocks)
    
    # 7. Evaluate liquidity
    liquid, liq_blocks = evaluate_liquidity(ctx)
    if not liquid:
        all_block_reasons.extend(liq_blocks)
    
    # Combine all metrics
    combined_metrics = {**gate_metrics, **kl_metrics, **cal_metrics, **risk_metrics}
    
    # Determine decision
    if all_block_reasons:
        # Blocked by one or more reasons
        primary_block = all_block_reasons[0]
        reason = f"BLOCKED: {primary_block.value}"
        if len(all_block_reasons) > 1:
            reason += f" (+{len(all_block_reasons)-1} more)"
        
        return DecisionResult(
            decision=Decision.BLOCKED,
            side="",
            venue="",
            size_pct=0.0,
            limit_price_cents=0,
            reason=reason,
            blocked_reasons=all_block_reasons,
            gate_results=combined_metrics,
            fair_value=fv_result,
            kl_health=kl_metrics,
            calibration_health=cal_metrics,
            risk_health=risk_metrics,
        )
    
    # All gates pass - check fair value recommendation
    if fv_result.recommend_side == "none":
        return DecisionResult(
            decision=Decision.SKIP,
            side="",
            venue="",
            size_pct=0.0,
            limit_price_cents=0,
            reason="No edge above threshold",
            blocked_reasons=[],
            gate_results=combined_metrics,
            fair_value=fv_result,
            kl_health=kl_metrics,
            calibration_health=cal_metrics,
            risk_health=risk_metrics,
        )
    
    # Compute size
    size_pct = compute_size(fv_result, ctx, combined_metrics)
    
    if size_pct <= 0:
        return DecisionResult(
            decision=Decision.SKIP,
            side="",
            venue="",
            size_pct=0.0,
            limit_price_cents=0,
            reason="Computed size <= 0",
            blocked_reasons=[],
            gate_results=combined_metrics,
            fair_value=fv_result,
            kl_health=kl_metrics,
            calibration_health=cal_metrics,
            risk_health=risk_metrics,
        )
    
    # Determine limit price
    if fv_result.recommend_venue == "maker":
        limit_price = fv_result.maker_yes_price * 100 if fv_result.recommend_side == "yes" else fv_result.maker_no_price * 100
        limit_price = int(round(limit_price))
    else:
        limit_price = fv_result.taker_yes_price * 100 if fv_result.recommend_side == "yes" else fv_result.taker_no_price * 100
        limit_price = int(round(limit_price))
    
    return DecisionResult(
        decision=Decision.ENTER_YES if fv_result.recommend_side == "yes" else Decision.ENTER_NO,
        side=fv_result.recommend_side,
        venue=fv_result.recommend_venue,
        size_pct=size_pct,
        limit_price_cents=limit_price,
        reason=f"ENTER {fv_result.recommend_side.upper()} {fv_result.recommend_venue} @ {limit_price}c size={size_pct:.1%} - {fv_result.reason}",
        blocked_reasons=[],
        gate_results=combined_metrics,
        fair_value=fv_result,
        kl_health=kl_metrics,
        calibration_health=cal_metrics,
        risk_health=risk_metrics,
    )


# ─── Integration helper for trader.py ─────────────────────────────────────────

def decide(
    ticker: str,
    series_ticker: str,
    close_time: str,
    minutes_left: float,
    strike: float,
    yes_bid: int,
    yes_ask: int,
    no_bid: int,
    no_ask: int,
    yes_bid_size: int,
    yes_ask_size: int,
    no_bid_size: int,
    no_ask_size: int,
    spot_price: float,
    spot_source: str,
    spot_age_seconds: float,
    sigma_frac_per_min: float,
    current_balance: float,
    current_exposure: float,
    max_drawdown_pct: float,
    consecutive_losses: int,
    brti_60s_avg: float = None,
    brti_age_seconds: float = None,
    calibration_monitor: gbm.RollingCalibrationMonitor = None,
    kalshi_book_age: float = None,
    polymarket_book_age: float = None,
    arb_opportunity: bool = False,
    arb_edge_cents: float = 0.0,
    model_probs: List[fair_value.ModelProb] = None,
) -> DecisionResult:
    """
    Convenience function for trader.py integration.
    
    Call this in the trading loop with all market data.
    """
    ctx = DecisionContext(
        ticker=ticker,
        series_ticker=series_ticker,
        close_time=close_time,
        minutes_left=minutes_left,
        strike=strike,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_bid_size=yes_bid_size,
        yes_ask_size=yes_ask_size,
        no_bid_size=no_bid_size,
        no_ask_size=no_ask_size,
        spot_price=spot_price,
        spot_source=spot_source,
        spot_age_seconds=spot_age_seconds,
        brti_60s_avg=brti_60s_avg,
        brti_age_seconds=brti_age_seconds,
        sigma_frac_per_min=sigma_frac_per_min,
        current_balance=current_balance,
        current_exposure=current_exposure,
        max_drawdown_pct=max_drawdown_pct,
        consecutive_losses=consecutive_losses,
        calibration_monitor=calibration_monitor,
        kalshi_book_age=kalshi_book_age,
        polymarket_book_age=polymarket_book_age,
        arb_opportunity=arb_opportunity,
        arb_edge_cents=arb_edge_cents,
        model_probs=model_probs,
    )
    
    return make_decision(ctx)


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test
    result = decide(
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
    
    print(f"Decision: {result.decision.value}")
    print(f"Side: {result.side}")
    print(f"Venue: {result.venue}")
    print(f"Size: {result.size_pct:.1%}")
    print(f"Limit: {result.limit_price_cents}c")
    print(f"Reason: {result.reason}")
    print(f"Blocked: {[r.value for r in result.blocked_reasons]}")