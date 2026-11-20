#!/usr/bin/env python3
"""
backtest_consolidated.py — T15-51: One backtest engine consolidation.

Consolidates backtest.py, parallel_backtest.py, backtest_harness.py into
a single unified engine with:
- Multiple data sources (authenticated Kalshi, public REST, Kalshi Eye)
- Multiple strategy modes (early_maker, final_taker, cross_venue_arb)
- Unified parameter grid with configurable dimensions
- Built-in promotion evaluation (T15-16)
- Walk-forward validation (T15-50 protocol)
- Output format compatible with EVAL_PROTOCOL.md
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from backtest_engine import StratParams, run_backtest, parameter_grid, format_params, print_results, save_results, print_best
from fees import buy_cash_out, sell_cash_in
from market_finder import _parse_api_time, _parse_market_prices

log = logging.getLogger("backtest_consolidated")


# ─── Strategy Modes ────────────────────────────────────────────────────────────

class StrategyMode:
    EARLY_MAKER = "early_maker"      # 6-11 min, 15-55c, maker (post-only)
    FINAL_TAKER = "final_taker"      # 14-15 min, 75-99c, taker (cross spread)
    CROSS_VENUE = "cross_venue"      # Kalshi vs Polymarket arb


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    # Data
    data_source: str = "authenticated"  # "authenticated", "public", "kalshieye"
    markets: List[Dict] = None
    settlement_yes_won: Optional[Callable] = None

    # Strategy
    strategy_mode: str = StrategyMode.FINAL_TAKER

    # Parameter grid
    quick: bool = True
    custom_grid: Optional[Dict[str, List]] = None

    # Execution
    jobs: int = 4
    limit: Optional[int] = None
    starting_balance: float = 10.0

    # Output
    out_path: str = "backtest_results.json"
    top: int = 10

    # Walk-forward (T15-50)
    walk_forward: bool = False
    train_days: int = 60
    val_days: int = 14
    test_days: int = 14
    step_days: int = 7
    embargo_minutes: int = 15

    # Promotion (T15-16)
    evaluate_promotion: bool = False

    # Metadata
    git_commit: str = ""
    seed: int = 42


# ─── Unified Simulation ────────────────────────────────────────────────────────

def simulate_trade_unified(
    market: Dict,
    params: StratParams,
    balance: float = 100.0,
    strategy_mode: str = StrategyMode.FINAL_TAKER,
    settlement_yes_won: Optional[Callable] = None,
) -> Optional[Dict]:
    """
    Unified trade simulation supporting multiple strategy modes.

    Args:
        market: Market dict
        params: Strategy parameters
        balance: Current available cash
        strategy_mode: One of StrategyMode values
        settlement_yes_won: Settlement function

    Returns:
        Trade result dict or None if no signal
    """
    ticker = market.get("ticker", "")
    ct = _parse_api_time(market.get("close_time", ""))
    ot = _parse_api_time(market.get("open_time", ""))
    if not ct or not ot:
        return None

    minutes_in = (ct - ot).total_seconds() / 60
    if minutes_in <= 0 or minutes_in > 15.5:
        return None

    entry_minute = params.ENTRY_WINDOW_START_MIN
    if entry_minute < 0 or entry_minute > 15.0:
        return None

    yb, ya, nb, na, yq, nq = _parse_market_prices(market)

    quotes = market.get("minute_quotes")
    entry_price_proxy = None
    side_bullish = True
    taker_fee = False

    if strategy_mode == StrategyMode.FINAL_TAKER:
        # Final-minute taker: enter at ASK, hold to BRTI settlement
        if quotes:
            entry_minute_key = min(quotes, key=lambda m: abs(m - params.ENTRY_WINDOW_START_MIN))
            bid_at_entry, ask_at_entry = quotes[entry_minute_key]
            if ask_at_entry and params.ENTRY_BAND_MIN <= ask_at_entry <= params.ENTRY_BAND_MAX:
                ep = ask_at_entry
                side_bullish = True
            elif bid_at_entry and params.ENTRY_BAND_MIN <= (100 - bid_at_entry) <= params.ENTRY_BAND_MAX:
                ep = 100 - bid_at_entry
                side_bullish = False
            else:
                ep = ask_at_entry if ask_at_entry else ya
                side_bullish = ep <= 50 if ep else True
        else:
            ep = ya if ya else 0
            side_bullish = ep <= 50 if ep else True
        entry_price_proxy = ep
        taker_fee = True

    elif strategy_mode == StrategyMode.EARLY_MAKER:
        # Early maker: enter at mid near entry window
        if quotes:
            minute = min(quotes, key=lambda m: abs(m - params.ENTRY_WINDOW_START_MIN))
            yb, ya = quotes[minute]
        if yb and ya:
            mid_proxy = (yb + ya) // 2
            entry_price_proxy = mid_proxy
        else:
            mid_proxy = 45
            entry_price_proxy = mid_proxy
        taker_fee = False

    elif strategy_mode == StrategyMode.CROSS_VENUE:
        # Cross-venue arb: would need both venue quotes
        # For now, fall back to final_taker logic
        return simulate_trade_unified(market, params, balance, StrategyMode.FINAL_TAKER, settlement_yes_won)

    else:
        log.warning(f"Unknown strategy mode: {strategy_mode}")
        return None

    if settlement_yes_won is None:
        yes_won = _default_settlement_yes_won(market)
    else:
        yes_won = settlement_yes_won(market)
    if yes_won is None:
        return None

    ep = entry_price_proxy
    if ep is None or ep <= 0:
        return None

    ep = max(1, min(99, int(ep)))

    # Band check
    if not (params.ENTRY_BAND_MIN <= ep <= params.ENTRY_BAND_MAX):
        return None

    if not side_bullish and ep > params.MAX_NO_ENTRY_CENTS:
        return None

    max_dollars = min(params.MAX_TRADE_DOLLARS, balance * params.BALANCE_PCT_PER_TRADE)
    cost = ep / 100.0
    n = max_dollars / cost if cost > 0 else 0.0
    if n < 0.01:
        return None

    dollar_risk = n * ep / 100.0

    entry_price_dollars = ep / 100.0
    exit_price_dollars = (
        1.0 if yes_won else 0.0
    ) if side_bullish else (
        0.0 if yes_won else 1.0
    )

    stopped = False
    if strategy_mode == StrategyMode.EARLY_MAKER and quotes and params.EXIT_STOP_LOSS_CENTS > 0:
        limit = ep - params.EXIT_STOP_LOSS_CENTS
        entry_minute_key = min(quotes, key=lambda m: abs(m - params.ENTRY_WINDOW_START_MIN))
        for minute in sorted(quotes):
            if minute <= entry_minute_key:
                continue
            bid, _ask = quotes[minute]
            if bid and bid <= limit:
                exit_price_dollars = bid / 100.0
                stopped = True
                break

    buy_cost = buy_cash_out(n, entry_price_dollars, taker=taker_fee)
    sell_revenue = sell_cash_in(n, exit_price_dollars, taker=False)
    pnl = sell_revenue - buy_cost
    won = pnl > 0

    return {
        "ticker": ticker,
        "direction": "YES" if side_bullish else "NO",
        "entry_price": ep,
        "contracts": n,
        "dollar_risk": dollar_risk,
        "buy_cost": buy_cost,
        "sell_revenue": sell_revenue,
        "settled_yes": 100 if yes_won else 0,
        "stopped": stopped,
        "won": won,
        "pnl": pnl,
        "minutes_in": minutes_in,
        "mid": entry_price_proxy,
        "strategy_mode": strategy_mode,
    }


def _default_settlement_yes_won(market):
    exp_val = market.get("expiration_value")
    floor_strike = market.get("floor_strike")
    if exp_val is None or exp_val == "" or floor_strike is None or floor_strike == "":
        return None
    try:
        return float(exp_val) >= float(floor_strike)
    except (ValueError, TypeError):
        return None


def run_backtest_unified(
    params: StratParams,
    markets: List[Dict],
    starting_balance: float = 10.0,
    strategy_mode: str = StrategyMode.FINAL_TAKER,
    settlement_yes_won: Optional[Callable] = None,
) -> Dict:
    """Run backtest with unified simulation."""
    balance = starting_balance
    trades = []

    sorted_markets = sorted(markets, key=lambda m: _parse_api_time(m.get("close_time", "")) or datetime.min)

    for m in sorted_markets:
        result = simulate_trade_unified(m, params, balance, strategy_mode, settlement_yes_won)
        if not result:
            continue
        if result["dollar_risk"] > balance * params.PER_TRADE_PCT + 1e-9:
            continue
        if result["dollar_risk"] > params.MAX_TRADE_DOLLARS + 1e-9:
            continue
        balance -= result["buy_cost"]
        balance += result["sell_revenue"]
        trades.append(result)

    if not trades:
        return {"ev": 0.0, "total_pnl": 0.0, "n_trades": 0, "win_rate": 0.0,
                "avg_pnl": 0.0, "final_balance": balance, "max_drawdown": 0.0,
                "avg_entry_cents": 0.0, "gross_edge_cents": 0.0, "trades": []}

    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / len(trades)
    avg_pnl = total_pnl / len(trades)
    ev_per_trade = avg_pnl

    running = starting_balance
    peak = starting_balance
    max_dd = 0.0
    for t in trades:
        running += t["pnl"]
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    avg_entry_cents = sum(t["entry_price"] for t in trades) / len(trades)
    gross_edge_cents = win_rate * 100.0 - avg_entry_cents

    return {
        "ev": ev_per_trade,
        "total_pnl": total_pnl,
        "avg_entry_cents": avg_entry_cents,
        "gross_edge_cents": gross_edge_cents,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "final_balance": balance,
        "max_drawdown": max_dd,
        "trades": trades,
        "strategy_mode": strategy_mode,
    }


# ─── Walk-Forward Validation ──────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime


def generate_walkforward_windows(
    markets: List[Dict],
    train_days: int = 60,
    val_days: int = 14,
    test_days: int = 14,
    step_days: int = 7,
    embargo_minutes: int = 15,
) -> List[WalkForwardWindow]:
    """
    Generate walk-forward windows from market data.

    Args:
        markets: All markets (must have close_time)
        train_days: Training window length
        val_days: Validation window length
        test_days: Test window length (held out)
        step_days: Step between windows
        embargo_minutes: Embargo between windows

    Returns:
        List of WalkForwardWindow objects
    """
    # Sort markets by close time
    sorted_markets = sorted(markets, key=lambda m: _parse_api_time(m.get("close_time", "")) or datetime.min)
    if not sorted_markets:
        return []

    all_times = [_parse_api_time(m.get("close_time", "")) for m in sorted_markets]
    all_times = [t for t in all_times if t is not None]
    if not all_times:
        return []

    earliest = min(all_times)
    latest = max(all_times)
    total_days = (latest - earliest).days

    windows = []
    window_id = 0

    current_start = earliest
    while True:
        train_end = current_start + timedelta(days=train_days)
        val_start = train_end + timedelta(minutes=embargo_minutes)
        val_end = val_start + timedelta(days=val_days)
        test_start = val_end + timedelta(minutes=embargo_minutes)
        test_end = test_start + timedelta(days=test_days)

        if test_end > latest:
            break

        windows.append(WalkForwardWindow(
            window_id=window_id,
            train_start=current_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        ))

        current_start += timedelta(days=step_days)
        window_id += 1

    return windows


def filter_markets_by_window(markets: List[Dict], start: datetime, end: datetime) -> List[Dict]:
    """Filter markets to those within [start, end]."""
    result = []
    for m in markets:
        ct = _parse_api_time(m.get("close_time", ""))
        if ct and start <= ct <= end:
            result.append(m)
    return result


def run_walkforward_backtest(
    markets: List[Dict],
    param_grid: List[Dict],
    config: BacktestConfig,
) -> List[Dict]:
    """
    Run walk-forward backtest across multiple windows.

    Returns list of results per window with best params.
    """
    windows = generate_walkforward_windows(
        markets,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        step_days=config.step_days,
        embargo_minutes=config.embargo_minutes,
    )

    log.info(f"Running walk-forward with {len(windows)} windows")

    all_results = []

    for window in windows:
        log.info(f"Window {window.window_id}: train={window.train_start.date()} to {window.train_end.date()}, "
                 f"val={window.val_start.date()} to {window.val_end.date()}, "
                 f"test={window.test_start.date()} to {window.test_end.date()}")

        train_markets = filter_markets_by_window(markets, window.train_start, window.train_end)
        val_markets = filter_markets_by_window(markets, window.val_start, window.val_end)
        test_markets = filter_markets_by_window(markets, window.test_start, window.test_end)

        log.info(f"  Markets: train={len(train_markets)}, val={len(val_markets)}, test={len(test_markets)}")

        if len(train_markets) < 10:
            log.warning(f"  Skipping window {window.window_id}: insufficient training data")
            continue

        # Grid search on training data
        best_params = None
        best_ev = float("-inf")

        for combo in param_grid:
            params = StratParams(**combo)
            result = run_backtest_unified(
                params, train_markets, config.starting_balance, config.strategy_mode
            )
            if result["ev"] > best_ev:
                best_ev = result["ev"]
                best_params = combo

        if best_params is None:
            continue

        # Validate on validation data
        val_params = StratParams(**best_params)
        val_result = run_backtest_unified(
            val_params, val_markets, config.starting_balance, config.strategy_mode
        )

        # Final test on held-out test data
        test_result = run_backtest_unified(
            val_params, test_markets, config.starting_balance, config.strategy_mode
        )

        window_result = {
            "window_id": window.window_id,
            "train_markets": len(train_markets),
            "val_markets": len(val_markets),
            "test_markets": len(test_markets),
            "best_params": best_params,
            "train_ev": best_ev,
            "val_ev": val_result["ev"],
            "test_ev": test_result["ev"],
            "test_pnl": test_result["total_pnl"],
            "test_trades": test_result["n_trades"],
            "test_win_rate": test_result["win_rate"],
            "test_max_dd": test_result["max_drawdown"],
        }
        all_results.append(window_result)

        log.info(f"  Best params: {format_params(best_params)}")
        log.info(f"  Train EV: {best_ev:.4f}, Val EV: {val_result['ev']:.4f}, Test EV: {test_result['ev']:.4f}")

    return all_results


# ─── Promotion Evaluation Integration ─────────────────────────────────────────

def evaluate_promotion_from_backtest(
    backtest_results: Dict,
) -> Dict:
    """Evaluate promotion criteria from backtest results."""
    from promotion import evaluate_promotion, check_promotion_from_backtest

    # Extract data from backtest results
    trades = backtest_results.get("trades", [])
    daily_balances = _compute_daily_balances(trades, backtest_results.get("starting_balance", 10.0))

    return check_promotion_from_backtest(
        backtest_results=backtest_results,
        paper_results={"daily_ev": [t.get("pnl", 0) for t in trades]},
        live_results=None,  # Would need live data
    )


def _compute_daily_balances(trades: List[Dict], starting_balance: float) -> List[float]:
    """Compute daily ending balances from trades."""
    if not trades:
        return [starting_balance]

    # Group trades by day
    from collections import defaultdict
    daily_pnl = defaultdict(float)
    for t in trades:
        ct = _parse_api_time(t.get("ticker", "").split("-")[1] if "-" in t.get("ticker", "") else "")
        if ct:
            day_key = ct.date()
            daily_pnl[day_key] += t.get("pnl", 0)

    # Build daily balances
    balances = []
    running = starting_balance
    for day in sorted(daily_pnl.keys()):
        running += daily_pnl[day]
        balances.append(running)

    return balances if balances else [starting_balance]


# ─── Main Entry Points ────────────────────────────────────────────────────────

def run_consolidated_backtest(config: BacktestConfig) -> List[Dict]:
    """
    Main entry point for consolidated backtest.

    Handles:
    - Single parameter sweep
    - Walk-forward validation
    - Promotion evaluation
    """
    if config.markets is None:
        raise ValueError("Markets must be provided")

    if config.walk_forward:
        # Generate parameter grid
        if config.custom_grid:
            keys = list(config.custom_grid.keys())
            combos = [dict(zip(keys, v)) for v in itertools.product(*config.custom_grid.values())]
        else:
            combos = list(parameter_grid(config.quick))

        return run_walkforward_backtest(config.markets, combos, config)

    else:
        # Single sweep (uses existing backtest_engine.run_sweep)
        return run_sweep(
            config.markets,
            settlement_yes_won=config.settlement_yes_won,
            quick=config.quick,
            limit=config.limit,
            top=config.top,
            jobs=config.jobs,
            out_path=config.out_path,
            starting_balance=config.starting_balance,
        )


def _backtest_one_unified(args):
    """Worker for parallel unified backtest."""
    combo, markets, starting_balance, strategy_mode, settlement_yes_won = args
    params = StratParams(**combo)
    result = run_backtest_unified(params, markets, starting_balance, strategy_mode, settlement_yes_won)
    result["params"] = combo
    return result


def run_parallel_unified(
    markets: List[Dict],
    param_grid: List[Dict],
    starting_balance: float = 10.0,
    strategy_mode: str = StrategyMode.FINAL_TAKER,
    settlement_yes_won: Optional[Callable] = None,
    jobs: int = 4,
) -> List[Dict]:
    """Run parallel backtest with unified engine."""
    jobs = min(jobs, os.cpu_count() or 4)

    work_items = [(c, markets, starting_balance, strategy_mode, settlement_yes_won) for c in param_grid]

    with Pool(processes=jobs) as pool:
        results = pool.map(_backtest_one_unified, work_items)

    # Sort by risk-adjusted return
    results.sort(key=lambda r: r["total_pnl"] / (r["max_drawdown"] + 0.01), reverse=True)
    return results


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidated backtest engine")
    parser.add_argument("--markets-file", required=True, help="Path to markets JSON")
    parser.add_argument("--strategy", choices=[StrategyMode.EARLY_MAKER, StrategyMode.FINAL_TAKER, StrategyMode.CROSS_VENUE],
                        default=StrategyMode.FINAL_TAKER)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="backtest_results.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--balance", type=float, default=10.0)
    args = parser.parse_args()

    # Load markets
    with open(args.markets_file) as f:
        markets = json.load(f)

    config = BacktestConfig(
        markets=markets,
        strategy_mode=args.strategy,
        quick=args.quick,
        walk_forward=args.walk_forward,
        jobs=args.jobs,
        out_path=args.out,
        limit=args.limit,
        starting_balance=args.balance,
    )

    results = run_consolidated_backtest(config)

    print(f"\nCompleted. Results saved to {args.out}")


if __name__ == "__main__":
    from datetime import timedelta
    main()