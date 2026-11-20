"""
position_sizing.py — Kelly-based position sizing optimizer (T13-33).

Implements optimal position sizing using Kelly criterion and variants:
- Full Kelly: f* = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-p
- Half-Kelly: f*/2 for reduced variance
- Quarter-Kelly: f*/4 for conservative sizing
- Risk-adjusted: caps at max drawdown tolerance

Integrates with backtest results for data-driven sizing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("position_sizing")


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    full_kelly: float      # Full Kelly fraction (0-1)
    half_kelly: float      # Half Kelly
    quarter_kelly: float   # Quarter Kelly
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float    # avg_win / avg_loss
    edge: float            # Expected value per unit risk
    kelly_fraction: str    # Which fraction was used


@dataclass
class PositionSizingParams:
    """Parameters for position sizing."""
    # Kelly fraction to use: "full", "half", "quarter", or custom float
    kelly_fraction: float = 0.5  # Half-Kelly by default
    
    # Hard caps (fraction of balance)
    max_position_pct: float = 0.20      # Max 20% per position
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure
    
    # Minimum trade size
    min_trade_dollars: float = 1.0
    max_trade_dollars: float = 50.0
    
    # Risk limits
    max_drawdown_pct: float = 0.20
    kelly_lookback_trades: int = 50  # Number of trades to estimate Kelly from
    
    # Confidence adjustment
    min_win_rate_for_kelly: float = 0.45  # Below this, use fixed fraction
    fixed_fraction_fallback: float = 0.05  # 5% fixed if Kelly not reliable


def calculate_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5,
) -> KellyResult:
    """
    Calculate Kelly criterion position size.
    
    Kelly formula: f* = (b*p - q) / b
    where:
        b = avg_win / avg_loss (payoff ratio)
        p = win_rate
        q = 1 - p
    
    Args:
        win_rate: Fraction of winning trades (0-1)
        avg_win: Average profit per winning trade (dollars)
        avg_loss: Average loss per losing trade (dollars, positive)
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
    
    Returns:
        KellyResult with full/half/quarter Kelly and metadata
    """
    if win_rate <= 0 or win_rate >= 1:
        return KellyResult(
            full_kelly=0.0, half_kelly=0.0, quarter_kelly=0.0,
            win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
            payoff_ratio=0.0, edge=0.0, kelly_fraction="none"
        )
    
    if avg_loss <= 0:
        return KellyResult(
            full_kelly=0.0, half_kelly=0.0, quarter_kelly=0.0,
            win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
            payoff_ratio=float('inf'), edge=float('inf'), kelly_fraction="none"
        )
    
    payoff_ratio = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    # Full Kelly: f* = (b*p - q) / b = p - q/b
    full_kelly = (payoff_ratio * p - q) / payoff_ratio
    
    # Edge per unit risk
    edge = p * avg_win - q * avg_loss
    
    # Clamp to [0, 1]
    full_kelly = max(0.0, min(1.0, full_kelly))
    
    half_kelly = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25
    custom_kelly = full_kelly * kelly_fraction
    
    return KellyResult(
        full_kelly=full_kelly,
        half_kelly=half_kelly,
        quarter_kelly=quarter_kelly,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        edge=edge,
        kelly_fraction=f"{kelly_fraction:.0%}-Kelly",
    )


def estimate_kelly_from_trades(
    trades: List[Dict],
    kelly_fraction: float = 0.5,
    lookback: int = 50,
) -> KellyResult:
    """
    Estimate Kelly parameters from a list of trades.
    
    Args:
        trades: List of trade dicts with 'pnl' and 'won' keys
        kelly_fraction: Fraction of Kelly to use
        lookback: Number of recent trades to use
    
    Returns:
        KellyResult
    """
    if not trades:
        return KellyResult(
            full_kelly=0.0, half_kelly=0.0, quarter_kelly=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            payoff_ratio=0.0, edge=0.0, kelly_fraction="none"
        )
    
    # Use most recent trades
    recent = trades[-lookback:] if len(trades) > lookback else trades
    
    wins = [t for t in recent if t.get('won', False)]
    losses = [t for t in recent if not t.get('won', False)]
    
    win_rate = len(wins) / len(recent) if recent else 0.0
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 0.0
    
    return calculate_kelly(win_rate, avg_win, avg_loss, kelly_fraction)


def calculate_position_size(
    balance: float,
    kelly_result: KellyResult,
    entry_price: float,  # cents
    params: PositionSizingParams,
    current_exposure: float = 0.0,
    open_positions: int = 0,
    max_open_positions: int = 4,
) -> Tuple[float, Dict]:
    """
    Calculate position size for a new trade.
    
    Args:
        balance: Current account balance
        kelly_result: KellyResult from calculate_kelly or estimate_kelly_from_trades
        entry_price: Entry price in cents (1-99)
        params: PositionSizingParams
        current_exposure: Current total exposure in dollars
        open_positions: Number of currently open positions
        max_open_positions: Maximum allowed open positions
    
    Returns:
        (contracts, metadata_dict)
    """
    metadata = {
        "kelly_fraction_used": kelly_result.kelly_fraction,
        "full_kelly": kelly_result.full_kelly,
        "win_rate": kelly_result.win_rate,
        "payoff_ratio": kelly_result.payoff_ratio,
        "edge": kelly_result.edge,
    }
    
    # Check position limit
    if open_positions >= max_open_positions:
        return 0.0, {**metadata, "reason": "max_open_positions_reached"}
    
    # Use Kelly fraction
    kelly_pct = kelly_result.full_kelly * params.kelly_fraction
    
    # Fallback to fixed fraction if Kelly not reliable
    if kelly_result.win_rate < params.min_win_rate_for_kelly or kelly_result.full_kelly <= 0:
        kelly_pct = params.fixed_fraction_fallback
        metadata["reason"] = "kelly_unreliable_using_fixed"
    else:
        metadata["reason"] = "kelly_based"
    
    # Apply caps
    kelly_pct = min(kelly_pct, params.max_position_pct)
    
    # Check total exposure limit
    max_new_exposure = balance * params.max_total_exposure_pct - current_exposure
    if max_new_exposure <= 0:
        return 0.0, {**metadata, "reason": "exposure_limit_reached"}
    
    # Calculate dollar amount
    dollar_amount = balance * kelly_pct
    dollar_amount = min(dollar_amount, max_new_exposure)
    dollar_amount = max(dollar_amount, params.min_trade_dollars)
    dollar_amount = min(dollar_amount, params.max_trade_dollars)
    
    # Convert to contracts
    cost_per_contract = entry_price / 100.0
    if cost_per_contract <= 0:
        return 0.0, {**metadata, "reason": "invalid_entry_price"}
    
    contracts = dollar_amount / cost_per_contract
    
    # Floor to reasonable precision (0.01 contracts)
    contracts = math.floor(contracts * 100) / 100
    
    if contracts < 0.01:
        return 0.0, {**metadata, "reason": "position_too_small"}
    
    metadata.update({
        "kelly_pct": kelly_pct,
        "dollar_amount": dollar_amount,
        "contracts": contracts,
        "entry_price_cents": entry_price,
    })
    
    return contracts, metadata


def calculate_position_size_from_backtest(
    balance: float,
    backtest_results: Dict,
    entry_price: float,
    params: PositionSizingParams,
    current_exposure: float = 0.0,
    open_positions: int = 0,
    max_open_positions: int = 4,
) -> Tuple[float, Dict]:
    """
    Calculate position size using backtest results.
    
    Args:
        balance: Current account balance
        backtest_results: Dict with 'trades' list from backtest
        entry_price: Entry price in cents
        params: PositionSizingParams
        current_exposure: Current total exposure
        open_positions: Number of open positions
        max_open_positions: Max open positions allowed
    
    Returns:
        (contracts, metadata_dict)
    """
    trades = backtest_results.get('trades', [])
    if not trades:
        # No backtest data, use conservative fixed fraction
        kelly_result = KellyResult(
            full_kelly=0.0, half_kelly=0.0, quarter_kelly=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            payoff_ratio=0.0, edge=0.0, kelly_fraction="fixed"
        )
    else:
        kelly_result = estimate_kelly_from_trades(
            trades,
            kelly_fraction=params.kelly_fraction,
            lookback=params.kelly_lookback_trades,
        )
    
    return calculate_position_size(
        balance=balance,
        kelly_result=kelly_result,
        entry_price=entry_price,
        params=params,
        current_exposure=current_exposure,
        open_positions=open_positions,
        max_open_positions=max_open_positions,
    )


def simulate_kelly_sizing(
    initial_balance: float,
    trades: List[Dict],
    params: PositionSizingParams,
    entry_price_fn=None,
) -> Dict:
    """
    Simulate Kelly-based position sizing over a series of trades.
    
    Args:
        initial_balance: Starting balance
        trades: List of trade dicts with 'pnl', 'won', 'entry_price' keys
        params: PositionSizingParams
        entry_price_fn: Optional function(trade_idx) -> entry_price_cents
    
    Returns:
        Dict with balance trajectory and metrics
    """
    balance = initial_balance
    balances = [balance]
    position_sizes = []
    kelly_fractions = []
    
    for i, trade in enumerate(trades):
        # Estimate Kelly from trades up to this point
        prior_trades = trades[:i]
        kelly_result = estimate_kelly_from_trades(
            prior_trades,
            kelly_fraction=params.kelly_fraction,
            lookback=params.kelly_lookback_trades,
        )
        
        # Get entry price for this trade
        entry_price = trade.get('entry_price', 50)
        if entry_price_fn:
            entry_price = entry_price_fn(i)
        
        # Calculate position size
        contracts, meta = calculate_position_size(
            balance=balance,
            kelly_result=kelly_result,
            entry_price=entry_price,
            params=params,
            current_exposure=0.0,  # Simplified: assume sequential trades
            open_positions=0,
            max_open_positions=1,
        )
        
        position_sizes.append(contracts)
        kelly_fractions.append(kelly_result.full_kelly * params.kelly_fraction)
        
        # Apply trade result
        if contracts > 0:
            cost_per_contract = entry_price / 100.0
            dollar_risk = contracts * cost_per_contract
            pnl_pct = trade['pnl'] / dollar_risk if dollar_risk > 0 else 0
            balance *= (1 + pnl_pct * kelly_result.full_kelly * params.kelly_fraction)
        
        balances.append(balance)
    
    total_return = (balance - initial_balance) / initial_balance
    max_dd = max((max(balances[:i+1]) - b) / max(balances[:i+1]) for i, b in enumerate(balances)) if balances else 0
    
    return {
        "final_balance": balance,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "balances": balances,
        "position_sizes": position_sizes,
        "kelly_fractions": kelly_fractions,
        "n_trades": len(trades),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Position sizing optimizer (T13-33)")
    parser.add_argument("--win-rate", type=float, default=0.55, help="Win rate (0-1)")
    parser.add_argument("--avg-win", type=float, default=10.0, help="Avg win ($)")
    parser.add_argument("--avg-loss", type=float, default=8.0, help="Avg loss ($)")
    parser.add_argument("--kelly-frac", type=float, default=0.5, help="Kelly fraction (0.5=half)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Account balance")
    parser.add_argument("--entry-price", type=int, default=50, help="Entry price (cents)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Calculate Kelly
    kelly = calculate_kelly(args.win_rate, args.avg_win, args.avg_loss, args.kelly_frac)
    print(f"Win rate: {kelly.win_rate:.1%}")
    print(f"Avg win: ${kelly.avg_win:.2f}")
    print(f"Avg loss: ${kelly.avg_loss:.2f}")
    print(f"Payoff ratio: {kelly.payoff_ratio:.2f}")
    print(f"Edge per trade: ${kelly.edge:.2f}")
    print(f"Full Kelly: {kelly.full_kelly:.1%}")
    print(f"Half Kelly: {kelly.half_kelly:.1%}")
    print(f"Quarter Kelly: {kelly.quarter_kelly:.1%}")
    print(f"Using: {kelly.kelly_fraction}")
    
    # Calculate position size
    params = PositionSizingParams(kelly_fraction=args.kelly_frac)
    contracts, meta = calculate_position_size(
        balance=args.balance,
        kelly_result=kelly,
        entry_price=args.entry_price,
        params=params,
    )
    print(f"\nPosition size: {contracts:.2f} contracts")
    print(f"Dollar amount: ${meta.get('dollar_amount', 0):.2f}")
    print(f"Kelly % used: {meta.get('kelly_pct', 0):.1%}")
    print(f"Reason: {meta.get('reason', 'unknown')}")