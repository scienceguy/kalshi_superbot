"""
sizing.py — Kelly fraction sizing with all caps (T13-33).

Full Kelly for a binary contract at price p with lower-bound win prob q_lcb:
  f_full = (q_lcb - p_eff) / (1 - p_eff),  where p_eff = p + fee_per_contract

Then: f = fraction * f_full, where fraction comes from RISK_PROFILE
      (e.g. 0.10 conservative, 0.25 moderate, 0.50 aggressive).

Finally cap by:
  - T13-03 equity-scaled limits (exposure ceiling, cash floor, per-trade cap)
  - Touch depth (available size at our price)
  - Per-market venue position limits
  - Max contracts per trade (config.MAX_CONTRACTS_CAP)
  - Max dollar risk per trade (config.MAX_TRADE_DOLLARS)
  - Minimum 0.01 contract lot size

Sizing uses the venue's current balance so both balances compound separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import config
import fees

log = logging.getLogger("sizing")


@dataclass
class SizingResult:
    """Result of sizing computation."""
    contracts: float           # floored to 0.01, 0 if no trade
    dollar_risk: float         # contracts * entry_price
    f_full: float              # full Kelly fraction
    f_applied: float           # fraction * f_full
    cap_hit: Optional[str]     # which cap limited the size (for debugging)
    q_lcb: float               # lower confidence bound used
    p_eff: float               # effective price including fee


def compute_full_kelly(q_lcb: float, entry_price_cents: int, fee_rate: float = None) -> float:
    """
    Compute full Kelly fraction for a binary contract.

    q_lcb: lower confidence bound on P(YES wins) [0,1]
    entry_price_cents: contract price in cents (1-99)
    fee_rate: taker fee rate (default from config)
    """
    if fee_rate is None:
        fee_rate = config.KALSHI_FEE_RATE

    p = entry_price_cents / 100.0
    fee_per_contract = fees.taker_fee_per_contract(entry_price_cents)
    p_eff = p + fee_per_contract / 100.0  # fee is in cents, convert to dollars

    if p_eff >= 1.0:
        return 0.0

    if q_lcb <= p_eff:
        return 0.0

    f_full = (q_lcb - p_eff) / (1.0 - p_eff)
    return max(0.0, f_full)


def get_risk_profile_fraction() -> float:
    """
    Get Kelly fraction from RISK_PROFILE env var.
    Values: 'conservative'=0.10, 'moderate'=0.25, 'aggressive'=0.50, or explicit float.
    """
    profile = config._env_str("RISK_PROFILE", "moderate").lower()
    if profile in ("conservative", "low", "0.1"):
        return 0.10
    if profile in ("moderate", "med", "0.25"):
        return 0.25
    if profile in ("aggressive", "high", "0.5"):
        return 0.50
    # Try to parse as float
    try:
        val = float(profile)
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.25  # default moderate


def apply_all_caps(
    contracts: float,
    entry_price_cents: int,
    cash: float,
    exposure: float,
    resting_orders_value: float = 0.0,
    paper: bool = False,
    mark_to_market_exposure: float = None,
    available_depth: Optional[float] = None,
    max_position_fraction: float = None,
    max_contracts: int = None,
) -> tuple[float, Optional[str]]:
    """
    Apply all sizing caps in priority order.

    Returns (capped_contracts, cap_hit_name).
    """
    if contracts <= 0:
        return 0.0, "zero_or_negative"

    entry_price = entry_price_cents / 100.0
    dollar_risk = contracts * entry_price

    # 1. Minimum lot size (Kalshi min 0.01)
    if contracts < 0.01:
        return 0.0, "min_lot"

    # 2. Max contracts per trade
    max_contracts = max_contracts or config.MAX_CONTRACTS_CAP
    if contracts > max_contracts:
        return float(max_contracts), "max_contracts"

    # 3. Max dollar risk per trade (legacy cap)
    if dollar_risk > config.MAX_TRADE_DOLLARS:
        return config.MAX_TRADE_DOLLARS / entry_price, "max_trade_dollars"

    # 4. T13-03 equity-scaled limits via RiskManager
    from risk import RiskManager
    rm = RiskManager()
    rm.state.start_balance = cash + exposure  # approximate if not initialized
    max_trade = rm.max_trade_dollars(
        cash=cash,
        exposure=exposure,
        resting_orders_value=resting_orders_value,
        paper=paper,
        mark_to_market_exposure=mark_to_market_exposure,
    )
    if dollar_risk > max_trade:
        return max_trade / entry_price, "risk_manager"

    # 5. Available depth (touch depth)
    if available_depth is not None and contracts > available_depth:
        return available_depth, "depth"

    # 6. Per-market venue position limit (fraction of total open interest)
    if max_position_fraction is not None and max_position_fraction > 0:
        # This would need access to venue's total open interest for this market
        # For now we just pass through - the check happens at order placement
        pass

    return contracts, None


def size_for_entry(
    q_lcb: float,
    entry_price_cents: int,
    cash: float,
    exposure: float,
    resting_orders_value: float = 0.0,
    paper: bool = False,
    mark_to_market_exposure: float = None,
    available_depth: Optional[float] = None,
    max_position_fraction: float = None,
    max_contracts: int = None,
    fee_rate: float = None,
) -> SizingResult:
    """
    Main entry point: compute sized contracts for one entry.

    Returns SizingResult with all details for logging/debugging.
    """
    # Full Kelly
    f_full = compute_full_kelly(q_lcb, entry_price_cents, fee_rate)

    # Apply risk profile fraction
    fraction = get_risk_profile_fraction()
    f_applied = f_full * fraction

    # Convert to contracts
    if f_applied <= 0:
        return SizingResult(
            contracts=0.0, dollar_risk=0.0, f_full=f_full, f_applied=f_applied,
            cap_hit="zero_kelly", q_lcb=q_lcb, p_eff=entry_price_cents/100.0
        )

    portfolio_value = cash + exposure
    if portfolio_value <= 0:
        return SizingResult(
            contracts=0.0, dollar_risk=0.0, f_full=f_full, f_applied=f_applied,
            cap_hit="zero_portfolio", q_lcb=q_lcb, p_eff=entry_price_cents/100.0
        )

    target_dollar_risk = portfolio_value * f_applied
    entry_price = entry_price_cents / 100.0
    contracts = target_dollar_risk / entry_price

    # Floor to 0.01 lot
    from signals import floor_contracts
    contracts = floor_contracts(contracts)

    # Apply all caps
    capped_contracts, cap_hit = apply_all_caps(
        contracts=contracts,
        entry_price_cents=entry_price_cents,
        cash=cash,
        exposure=exposure,
        resting_orders_value=resting_orders_value,
        paper=paper,
        mark_to_market_exposure=mark_to_market_exposure,
        available_depth=available_depth,
        max_position_fraction=max_position_fraction,
        max_contracts=max_contracts,
    )

    dollar_risk = capped_contracts * entry_price

    return SizingResult(
        contracts=capped_contracts,
        dollar_risk=dollar_risk,
        f_full=f_full,
        f_applied=f_applied,
        cap_hit=cap_hit,
        q_lcb=q_lcb,
        p_eff=entry_price + (fees.taker_fee_per_contract(entry_price_cents) / 100.0),
    )


# ─── Convenience for backtest/forecast using same function ────────────────────

def size_for_forecast(
    q_lcb: float,
    entry_price_cents: int,
    balance: float,
    paper: bool = True,
    **kwargs,
) -> SizingResult:
    """Wrapper for backtest/forecast: cash=balance, exposure=0."""
    return size_for_entry(
        q_lcb=q_lcb,
        entry_price_cents=entry_price_cents,
        cash=balance,
        exposure=0.0,
        paper=paper,
        **kwargs,
    )


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Demo
    print("Kelly sizing demo:")
    for q in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for p_cents in [15, 35, 55, 75, 85]:
            res = size_for_entry(q_lcb=q, entry_price_cents=p_cents, cash=1000, exposure=0, paper=True)
            if res.contracts > 0:
                print(f"  q={q:.2f} p={p_cents}c -> {res.contracts:.2f} contracts (${res.dollar_risk:.2f}) f_full={res.f_full:.3f} f={res.f_applied:.3f} cap={res.cap_hit}")