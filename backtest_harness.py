"""
backtest_harness.py — Unified backtest engine (T13-30).

Consolidates all backtest variants into a single engine with:
- Replay from tick_store (1s resolution)
- True fee models (maker/taker, Kalshi fee schedule)
- Maker/taker fill models (queue position, toxicity, adverse selection)
- Walk-forward splits with purge/embargo (no lookahead)
- Parameter freezing (config hash for reproducibility)
- Multiple strategy modes: early_maker, final_taker, latency_arb
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from fees import buy_cash_out, sell_cash_in, FEE_RATE as KALSHI_FEE_RATE
from tick_store import get_tick_store
from position_sizing import (
    KellyResult, PositionSizingParams, calculate_kelly, 
    estimate_kelly_from_trades, calculate_position_size
)

log = logging.getLogger("backtest_harness")


# =====================================================================
# Strategy Parameters (frozen via config hash)
# =====================================================================

@dataclass
class BacktestParams:
    """Parameter set for a backtest run. Frozen via config hash."""
    
    # Entry timing
    ENTRY_WINDOW_START_MIN: float = 6.0
    ENTRY_WINDOW_END_MIN: float = 11.0
    MIN_MINUTES_TO_HOLD: float = 1.0
    MAX_MINUTES_TO_HOLD: float = 15.5
    
    # Entry band (cents)
    ENTRY_BAND_MIN: int = 15
    ENTRY_BAND_MAX: int = 55
    MAX_NO_ENTRY_CENTS: int = 55
    
    # Strategy mode
    STRATEGY_MODE: str = "early_maker"  # "early_maker", "final_taker", "latency_arb"
    
    # Sizing
    PER_TRADE_PCT: float = 0.08
    BALANCE_PCT_PER_TRADE: float = 0.08
    MAX_TRADE_DOLLARS: float = 10.0
    MAX_OPEN_POSITIONS: int = 4
    STARTING_BALANCE: float = 100.0
    
    # Risk
    CASH_FLOOR_PCT: float = 0.70
    EXPOSURE_CEILING_PCT: float = 0.30
    MAX_DAILY_LOSS_PCT: float = 0.10
    CONSECUTIVE_LOSS_LIMIT: int = 3
    
    # Exit
    EXIT_STOP_LOSS_CENTS: int = 0
    TAKE_PROFIT_CENTS: int = 0
    AGGRESSIVE_FILL: bool = False
    
    # Fees
    FEE_RATE: float = KALSHI_FEE_RATE
    MAKER_REBATE: float = 0.0
    
    # Fill model
    FILL_PROBABILITY_BASE: float = 0.8
    FILL_TOXICITY_PENALTY: float = 0.3
    FILL_QUEUE_POSITION_DECAY: float = 0.95
    
    # Feature toggles
    USE_ML_FILTER: bool = False
    USE_KL_FILTER: bool = False
    USE_GATES: bool = True
    GATES_SHADOW_MODE: bool = True
    
    # Walk-forward
    TRAIN_WINDOW_DAYS: int = 30
    TEST_WINDOW_DAYS: int = 7
    PURGE_DAYS: int = 1
    EMBARGO_DAYS: int = 1
    N_SPLITS: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "BacktestParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def config_hash(self) -> str:
        """Hash of all parameters for reproducibility."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =====================================================================
# Fill Model (Maker/Taker)
# =====================================================================

@dataclass
class FillModel:
    """Models fill probability for maker/taker orders."""
    
    base_fill_prob: float = 0.8
    toxicity_penalty: float = 0.3
    queue_decay: float = 0.95
    min_fill_prob: float = 0.05
    
    def maker_fill_prob(
        self,
        price_cents: int,
        minutes_left: float,
        spread_bps: float,
        book_imbalance: float,
        queue_position: float = 0.5
    ) -> float:
        """
        Probability a maker limit order gets filled.
        
        Args:
            price_cents: Entry price in cents (1-99)
            minutes_left: Time to expiry
            spread_bps: Bid-ask spread in basis points
            book_imbalance: (bid_size - ask_size) / (bid_size + ask_size)
            queue_position: 0=front, 1=back of queue
        """
        # Base probability
        p = self.base_fill_prob
        
        # Toxicity: 5-45c and 55-95c with 7+ min left (adverse selection risk)
        in_toxic = (5 <= price_cents <= 45) or (55 <= price_cents <= 95)
        if in_toxic and minutes_left >= 7:
            p *= (1 - self.toxicity_penalty)
        
        # Spread: tighter spread = harder to get filled as maker (less edge)
        # Reasonable scaling: at 100 bps (1%) spread, factor ~0.8; at 1000 bps (10%), factor ~0.3
        if spread_bps > 0:
            spread_factor = max(0.2, min(1.0, 200 / spread_bps))
            p *= spread_factor
        
        # Book imbalance: favorable imbalance helps our side
        if book_imbalance > 0:  # More bids than asks
            p *= (1 + 0.1 * book_imbalance)
        else:
            p *= (1 + 0.1 * book_imbalance)  # Negative hurts
        
        # Queue position decay
        p *= (self.queue_decay ** (queue_position * 10))
        
        return max(self.min_fill_prob, min(1.0, p))
    
    def taker_fill_prob(
        self,
        price_cents: int,
        size_contracts: float,
        available_liquidity: float
    ) -> float:
        """Probability a taker order gets filled (usually 1 if liquidity exists)."""
        if available_liquidity >= size_contracts:
            return 1.0
        return min(1.0, available_liquidity / size_contracts) if size_contracts > 0 else 0.0


# =====================================================================
# Market Replay from Tick Store
# =====================================================================

@dataclass
class MarketSnapshot:
    """Reconstructed market state at a point in time."""
    ticker: str
    timestamp: float
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    yes_size: float
    no_size: float
    last_price: float
    volume: float
    expiration_value: Optional[float] = None
    floor_strike: Optional[float] = None


class MarketReplayer:
    """Replays market data from tick_store for backtesting."""
    
    def __init__(self, tick_store=None):
        self.tick_store = tick_store or get_tick_store()
    
    def get_market_series(
        self,
        series: str,  # e.g., "KXBTC15M"
        start_ts: float,
        end_ts: float
    ) -> Dict[str, List[MarketSnapshot]]:
        """
        Get reconstructed market snapshots for all markets in a series.
        
        Returns dict: ticker -> list of MarketSnapshot (chronological)
        """
        # Query ticks for this series
        symbol = series.replace("KX", "").replace("15M", "USDT")  # KXBTC15M -> BTCUSDT
        
        all_ticks = self.tick_store.query_ticks_all_venues(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=100000
        )
        
        # Group by market ticker (need to infer from strike/time)
        # For now, use a simplified approach
        markets = self._reconstruct_markets(all_ticks, series, start_ts, end_ts)
        
        return markets
    
    def _reconstruct_markets(
        self,
        ticks_by_venue: Dict[str, List[Dict]],
        series: str,
        start_ts: float,
        end_ts: float
    ) -> Dict[str, List[MarketSnapshot]]:
        """Reconstruct per-market snapshots from raw ticks."""
        # This is a simplified reconstruction
        # In production, would use Kalshi market metadata to map ticks to specific markets
        
        # Group ticks into 1-second buckets
        all_ticks = []
        for venue, ticks in ticks_by_venue.items():
            for t in ticks:
                t["venue"] = venue
                all_ticks.append(t)
        
        # Sort by timestamp
        all_ticks.sort(key=lambda x: x["ts"])
        
        # Bin into 1-second intervals
        snapshots_by_second = {}
        for t in all_ticks:
            sec = int(t["recv_ts"])
            if sec not in snapshots_by_second:
                snapshots_by_second[sec] = {"binance": None, "coinbase": None}
            snapshots_by_second[sec][t["venue"]] = t
        
        # Create synthetic market snapshots (one market per 15-min window)
        markets = {}
        current_window_start = None
        current_market_ticker = None
        
        for sec in sorted(snapshots_by_second.keys()):
            minute_of_window = (sec - int(start_ts)) // 60
            window_idx = minute_of_window // 15
            
            market_ticker = f"{series}-W{window_idx:04d}"
            
            if market_ticker != current_market_ticker:
                current_market_ticker = market_ticker
                if market_ticker not in markets:
                    markets[market_ticker] = []
            
            # Get best bid/ask across venues
            snaps = snapshots_by_second[sec]
            bid = None
            ask = None
            bid_size = 0
            ask_size = 0
            
            for venue, tick in snaps.items():
                if tick and tick.get("bid") is not None and tick.get("ask") is not None:
                    if bid is None or tick["bid"] > bid:
                        bid = tick["bid"]
                    if ask is None or tick["ask"] < ask:
                        ask = tick["ask"]
                    bid_size += 1
                    ask_size += 1
            
            if bid is not None and ask is not None:
                # Convert to cents (assuming BTC ~$100k -> 1 tick = 1 cent)
                yes_bid = int((bid / 1000) * 100)  # Rough conversion
                yes_ask = int((ask / 1000) * 100)
                yes_bid = max(1, min(99, yes_bid))
                yes_ask = max(1, min(99, yes_ask))
                
                ms = MarketSnapshot(
                    ticker=market_ticker,
                    timestamp=sec,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=100 - yes_ask,
                    no_ask=100 - yes_bid,
                    yes_size=bid_size,
                    no_size=ask_size,
                    last_price=(yes_bid + yes_ask) / 2,
                    volume=bid_size + ask_size
                )
                markets[market_ticker].append(ms)
        
        return markets
    
    def get_market_metadata(self, ticker: str) -> Optional[Dict]:
        """Get settlement metadata for a market ticker."""
        # Would query Kalshi API or local DB for expiration_value, floor_strike
        return None


# =====================================================================
# Backtest Engine
# =====================================================================

@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    ticker: str
    timestamp: float
    direction: str  # "YES" or "NO"
    entry_price: int  # cents
    contracts: float
    dollar_risk: float
    buy_cost: float
    sell_revenue: float
    pnl: float
    won: bool
    stopped: bool
    fill_prob: float
    fees_paid: float
    hold_time_sec: float


class BacktestEngine:
    """Main backtest engine with walk-forward validation."""
    
    def __init__(self, params: BacktestParams):
        self.params = params
        self.fill_model = FillModel(
            base_fill_prob=params.FILL_PROBABILITY_BASE,
            toxicity_penalty=params.FILL_TOXICITY_PENALTY,
            queue_decay=params.FILL_QUEUE_POSITION_DECAY
        )
        self.replayer = MarketReplayer()
        self.trades: List[TradeResult] = []
        self.balance = params.STARTING_BALANCE
        self.peak_balance = params.STARTING_BALANCE
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.open_positions = {}
        
        # Kelly position sizing
        self.kelly_params = PositionSizingParams(
            kelly_fraction=0.5,  # Half-Kelly by default
            max_position_pct=params.PER_TRADE_PCT,
            max_total_exposure_pct=params.EXPOSURE_CEILING_PCT,
            min_trade_dollars=1.0,
            max_trade_dollars=params.MAX_TRADE_DOLLARS,
            max_drawdown_pct=params.MAX_DAILY_LOSS_PCT,
            kelly_lookback_trades=params.CONSECUTIVE_LOSS_LIMIT * 10,  # Use more trades
            min_win_rate_for_kelly=0.45,
            fixed_fraction_fallback=params.BALANCE_PCT_PER_TRADE,
        )
        self.completed_trades_for_kelly: List[Dict] = []  # Track trades for Kelly estimation
        
    def run_backtest(
        self,
        markets: Dict[str, List[MarketSnapshot]],
        settlement_fn: Callable[[str], Optional[bool]]
    ) -> Dict[str, Any]:
        """
        Run backtest on a set of markets.
        
        Args:
            markets: ticker -> list of MarketSnapshot
            settlement_fn: callable(ticker) -> bool (True if YES won)
        
        Returns:
            Dict with results and metrics
        """
        self.trades = []
        self.balance = self.params.STARTING_BALANCE
        self.peak_balance = self.params.STARTING_BALANCE
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.open_positions = {}
        
        # Sort markets by first timestamp
        sorted_tickers = sorted(
            markets.keys(),
            key=lambda t: markets[t][0].timestamp if markets[t] else float('inf')
        )
        
        for ticker in sorted_tickers:
            snapshots = markets[ticker]
            if not snapshots:
                continue
            
            # Check daily loss limit
            day_key = datetime.fromtimestamp(snapshots[0].timestamp, tz=timezone.utc).date().isoformat()
            if day_key not in self.daily_pnl:
                self.daily_pnl[day_key] = 0.0
            
            if self.daily_pnl[day_key] <= -self.params.STARTING_BALANCE * self.params.MAX_DAILY_LOSS_PCT:
                log.debug(f"Daily loss limit hit for {day_key}, skipping")
                continue
            
            # Check consecutive losses
            if self.consecutive_losses >= self.params.CONSECUTIVE_LOSS_LIMIT:
                log.debug(f"Consecutive loss limit hit, skipping")
                continue
            
            # Check max open positions
            if len(self.open_positions) >= self.params.MAX_OPEN_POSITIONS:
                continue
            
            # Simulate this market
            result = self._simulate_market(ticker, snapshots, settlement_fn)
            if result:
                self.trades.append(result)
                self._update_state(result)
        
        return self._compute_metrics()
    
    def _simulate_market(
        self,
        ticker: str,
        snapshots: List[MarketSnapshot],
        settlement_fn: Callable[[str], Optional[bool]]
    ) -> Optional[TradeResult]:
        """Simulate trading on a single market."""
        
        # Get settlement outcome
        yes_won = settlement_fn(ticker)
        if yes_won is None:
            return None
        
        # Find entry window
        start_ts = snapshots[0].timestamp
        entry_start = start_ts + self.params.ENTRY_WINDOW_START_MIN * 60
        entry_end = start_ts + self.params.ENTRY_WINDOW_END_MIN * 60
        
        # Find first snapshot in entry window
        entry_snap = None
        for snap in snapshots:
            if entry_start <= snap.timestamp <= entry_end:
                entry_snap = snap
                break
        
        if not entry_snap:
            return None
        
        # Determine side and entry price
        if self.params.STRATEGY_MODE == "final_taker":
            return self._simulate_final_taker(ticker, entry_snap, snapshots, yes_won, start_ts)
        elif self.params.STRATEGY_MODE == "latency_arb":
            return self._simulate_latency_arb(ticker, entry_snap, snapshots, yes_won, start_ts)
        else:
            return self._simulate_early_maker(ticker, entry_snap, snapshots, yes_won, start_ts)
    
    def _simulate_early_maker(
        self,
        ticker: str,
        entry_snap: MarketSnapshot,
        all_snapshots: List[MarketSnapshot],
        yes_won: bool,
        start_ts: float
    ) -> Optional[TradeResult]:
        """Simulate early maker strategy (post at bid/ask, wait for fill)."""
        
        # Determine side (favorite = higher probability side)
        # YES favorite if mid > 50, NO favorite if mid < 50
        mid = (entry_snap.yes_bid + entry_snap.yes_ask) / 2
        side = "YES" if mid >= 50 else "NO"
        entry_price = int(entry_snap.yes_bid) if side == "YES" else int(entry_snap.no_bid)
        
        # Check entry band on the actual entry price for the chosen side
        if not (self.params.ENTRY_BAND_MIN <= entry_price <= self.params.ENTRY_BAND_MAX):
            return None
        
        # Additional NO price limit
        if side == "NO" and entry_price > self.params.MAX_NO_ENTRY_CENTS:
            return None
        
        # Fill probability
        spread = entry_snap.yes_ask - entry_snap.yes_bid
        spread_bps = (spread / mid) * 10000 if mid > 0 else 100
        imbalance = (entry_snap.yes_size - entry_snap.no_size) / (entry_snap.yes_size + entry_snap.no_size + 1)
        
        fill_prob = self.fill_model.maker_fill_prob(
            price_cents=entry_price,
            minutes_left=(all_snapshots[-1].timestamp - entry_snap.timestamp) / 60,
            spread_bps=spread_bps,
            book_imbalance=imbalance,
            queue_position=0.5
        )
        
        if np.random.random() > fill_prob:
            return None  # Not filled
        
        # Kelly-based position sizing
        kelly_result = estimate_kelly_from_trades(
            self.completed_trades_for_kelly,
            kelly_fraction=self.kelly_params.kelly_fraction,
            lookback=self.kelly_params.kelly_lookback_trades,
        )
        
        # Calculate current exposure (simplified - sum of dollar risk of open positions)
        current_exposure = sum(p.dollar_risk for p in self.open_positions.values())
        
        contracts, meta = calculate_position_size(
            balance=self.balance,
            kelly_result=kelly_result,
            entry_price=entry_price,
            params=self.kelly_params,
            current_exposure=current_exposure,
            open_positions=len(self.open_positions),
            max_open_positions=self.params.MAX_OPEN_POSITIONS,
        )
        
        if contracts < 0.01:
            return None
        
        log.debug(f"Kelly sizing: {meta}")
        
        # Fees: maker pays no fee (or gets rebate)
        taker_fee = False
        buy_cost = buy_cash_out(contracts, entry_price / 100.0, taker=taker_fee)
        
        # Exit at settlement (or stop loss)
        exit_price = 1.0 if (side == "YES" and yes_won) or (side == "NO" and not yes_won) else 0.0
        
        # Check stop loss
        stopped = False
        if self.params.EXIT_STOP_LOSS_CENTS > 0:
            for snap in all_snapshots:
                if snap.timestamp <= entry_snap.timestamp:
                    continue
                current_mid = (snap.yes_bid + snap.yes_ask) / 2 if snap.yes_bid and snap.yes_ask else mid
                if side == "YES" and current_mid <= entry_price - self.params.EXIT_STOP_LOSS_CENTS:
                    exit_price = (entry_price - self.params.EXIT_STOP_LOSS_CENTS) / 100.0
                    stopped = True
                    break
                elif side == "NO" and current_mid >= entry_price + self.params.EXIT_STOP_LOSS_CENTS:
                    exit_price = (100 - entry_price - self.params.EXIT_STOP_LOSS_CENTS) / 100.0
                    stopped = True
                    break
        
        sell_revenue = sell_cash_in(contracts, exit_price, taker=False)
        pnl = sell_revenue - buy_cost
        won = pnl > 0
        
        return TradeResult(
            ticker=ticker,
            timestamp=entry_snap.timestamp,
            direction=side,
            entry_price=entry_price,
            contracts=contracts,
            dollar_risk=contracts * entry_price / 100.0,
            buy_cost=buy_cost,
            sell_revenue=sell_revenue,
            pnl=pnl,
            won=won,
            stopped=stopped,
            fill_prob=fill_prob,
            fees_paid=0.0,  # Maker fee = 0
            hold_time_sec=all_snapshots[-1].timestamp - entry_snap.timestamp
        )
    
    def _simulate_final_taker(
        self,
        ticker: str,
        entry_snap: MarketSnapshot,
        all_snapshots: List[MarketSnapshot],
        yes_won: bool,
        start_ts: float
    ) -> Optional[TradeResult]:
        """Simulate final taker strategy (pay ask, hold to settlement)."""
        
        # Taker pays ask
        ask = entry_snap.yes_ask
        if not (self.params.ENTRY_BAND_MIN <= ask <= self.params.ENTRY_BAND_MAX):
            return None
        
        side = "YES"
        entry_price = int(ask)
        
        # Taker fill is immediate if liquidity exists
        fill_prob = self.fill_model.taker_fill_prob(
            price_cents=entry_price,
            size_contracts=1.0,
            available_liquidity=entry_snap.no_size  # NO size = YES liquidity
        )
        
        if np.random.random() > fill_prob:
            return None
        
        # Kelly-based position sizing
        kelly_result = estimate_kelly_from_trades(
            self.completed_trades_for_kelly,
            kelly_fraction=self.kelly_params.kelly_fraction,
            lookback=self.kelly_params.kelly_lookback_trades,
        )
        
        current_exposure = sum(p.dollar_risk for p in self.open_positions.values())
        
        contracts, meta = calculate_position_size(
            balance=self.balance,
            kelly_result=kelly_result,
            entry_price=entry_price,
            params=self.kelly_params,
            current_exposure=current_exposure,
            open_positions=len(self.open_positions),
            max_open_positions=self.params.MAX_OPEN_POSITIONS,
        )
        
        if contracts < 0.01:
            return None
        
        log.debug(f"Kelly sizing: {meta}")
        
        # Taker fee on entry
        taker_fee = True
        buy_cost = buy_cash_out(contracts, entry_price / 100.0, taker=taker_fee)
        fees_paid = buy_cost - contracts * entry_price / 100.0
        
        # Hold to settlement
        exit_price = 1.0 if yes_won else 0.0
        sell_revenue = sell_cash_in(contracts, exit_price, taker=False)
        pnl = sell_revenue - buy_cost
        won = pnl > 0
        
        return TradeResult(
            ticker=ticker,
            timestamp=entry_snap.timestamp,
            direction=side,
            entry_price=entry_price,
            contracts=contracts,
            dollar_risk=contracts * entry_price / 100.0,
            buy_cost=buy_cost,
            sell_revenue=sell_revenue,
            pnl=pnl,
            won=won,
            stopped=False,
            fill_prob=fill_prob,
            fees_paid=fees_paid,
            hold_time_sec=all_snapshots[-1].timestamp - entry_snap.timestamp
        )
    
    def _simulate_latency_arb(
        self,
        ticker: str,
        entry_snap: MarketSnapshot,
        all_snapshots: List[MarketSnapshot],
        yes_won: bool,
        start_ts: float
    ) -> Optional[TradeResult]:
        """Simulate latency arbitrage (last-minute taker edge)."""
        # Simplified: same as final_taker but with tighter timing
        return self._simulate_final_taker(ticker, entry_snap, all_snapshots, yes_won, start_ts)
    
    def _update_state(self, result: TradeResult):
        """Update running balance and risk state."""
        self.balance -= result.buy_cost
        self.balance += result.sell_revenue
        
        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, dd)
        
        # Track consecutive losses
        if result.won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # Daily PnL
        day_key = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).date().isoformat()
        if day_key not in self.daily_pnl:
            self.daily_pnl[day_key] = 0.0
        self.daily_pnl[day_key] += result.pnl
        
        # Track completed trade for Kelly estimation
        self.completed_trades_for_kelly.append({
            'pnl': result.pnl,
            'won': result.won,
            'entry_price': result.entry_price,
            'dollar_risk': result.dollar_risk,
        })
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute final backtest metrics."""
        if not self.trades:
            return {
                "ev_per_trade": 0.0,
                "total_pnl": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "final_balance": self.balance,
                "max_drawdown": self.max_drawdown,
                "avg_entry_cents": 0.0,
                "gross_edge_cents": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "profit_factor": 0.0,
                "total_fees": 0.0,
                "avg_fill_prob": 0.0,
                "params_hash": self.params.config_hash(),
                "trades": []
            }
        
        total_pnl = sum(t.pnl for t in self.trades)
        wins = sum(1 for t in self.trades if t.won)
        win_rate = wins / len(self.trades)
        avg_pnl = total_pnl / len(self.trades)
        
        # Entry price stats
        avg_entry = sum(t.entry_price for t in self.trades) / len(self.trades)
        gross_edge = win_rate * 100 - avg_entry
        
        # Returns series for Sharpe/Sortino
        returns = [t.pnl / self.params.STARTING_BALANCE for t in self.trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sortino (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        sortino = np.mean(returns) / (np.std(downside_returns) + 1e-9) * np.sqrt(252) if downside_returns else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_fees = sum(t.fees_paid for t in self.trades)
        avg_fill = np.mean([t.fill_prob for t in self.trades])
        
        return {
            "ev_per_trade": avg_pnl,
            "total_pnl": total_pnl,
            "n_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "final_balance": self.balance,
            "max_drawdown": self.max_drawdown,
            "avg_entry_cents": avg_entry,
            "gross_edge_cents": gross_edge,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "total_fees": total_fees,
            "avg_fill_prob": avg_fill,
            "params_hash": self.params.config_hash(),
            "consecutive_losses_max": self.consecutive_losses,
            "trades": [asdict(t) for t in self.trades]
        }


# =====================================================================
# Walk-Forward Validation
# =====================================================================

@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    split_idx: int
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    train_metrics: Dict
    test_metrics: Dict
    params: BacktestParams
    params_hash: str


class WalkForwardValidator:
    """Walk-forward validation with purge/embargo."""
    
    def __init__(self, params: BacktestParams):
        self.params = params
        self.results: List[WalkForwardResult] = []
    
    def validate(
        self,
        markets_by_time: List[Tuple[float, Dict[str, List[MarketSnapshot]]]],
        settlement_fn: Callable[[str], Optional[bool]]
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward validation.
        
        Args:
            markets_by_time: List of (timestamp, markets_dict) sorted by time
            settlement_fn: Settlement outcome function
        
        Returns:
            List of WalkForwardResult per split
        """
        if len(markets_by_time) < self.params.N_SPLITS * 2:
            log.warning("Insufficient data for walk-forward, using single split")
            return self._single_split(markets_by_time, settlement_fn)
        
        n = len(markets_by_time)
        split_size = n // (self.params.N_SPLITS + 1)
        
        for i in range(self.params.N_SPLITS):
            # Train window
            train_start_idx = max(0, i * split_size)
            train_end_idx = (i + 1) * split_size
            
            # Purge
            purge_idx = int(self.params.PURGE_DAYS * 96)  # 96 periods/day
            train_end_idx = max(train_start_idx, train_end_idx - purge_idx)
            
            # Test window
            test_start_idx = train_end_idx + int(self.params.EMBARGO_DAYS * 96)
            test_end_idx = min(n, test_start_idx + split_size)
            
            if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx:
                continue
            
            train_markets = self._merge_markets(markets_by_time[train_start_idx:train_end_idx])
            test_markets = self._merge_markets(markets_by_time[test_start_idx:test_end_idx])
            
            if not train_markets or not test_markets:
                continue
            
            # Train (fit any model parameters, though our engine is param-driven)
            engine_train = BacktestEngine(self.params)
            train_metrics = engine_train.run_backtest(train_markets, settlement_fn)
            
            # Test
            engine_test = BacktestEngine(self.params)
            test_metrics = engine_test.run_backtest(test_markets, settlement_fn)
            
            result = WalkForwardResult(
                split_idx=i,
                train_start=markets_by_time[train_start_idx][0],
                train_end=markets_by_time[train_end_idx-1][0],
                test_start=markets_by_time[test_start_idx][0],
                test_end=markets_by_time[test_end_idx-1][0],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                params=self.params,
                params_hash=self.params.config_hash()
            )
            self.results.append(result)
        
        return self.results
    
    def _single_split(
        self,
        markets_by_time: List[Tuple[float, Dict[str, List[MarketSnapshot]]]],
        settlement_fn: Callable[[str], Optional[bool]]
    ) -> List[WalkForwardResult]:
        """Fallback: single train/test split."""
        n = len(markets_by_time)
        split = int(n * 0.7)
        
        train_markets = self._merge_markets(markets_by_time[:split])
        test_markets = self._merge_markets(markets_by_time[split:])
        
        engine_train = BacktestEngine(self.params)
        train_metrics = engine_train.run_backtest(train_markets, settlement_fn)
        
        engine_test = BacktestEngine(self.params)
        test_metrics = engine_test.run_backtest(test_markets, settlement_fn)
        
        return [WalkForwardResult(
            split_idx=0,
            train_start=markets_by_time[0][0],
            train_end=markets_by_time[split-1][0],
            test_start=markets_by_time[split][0],
            test_end=markets_by_time[-1][0],
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            params=self.params,
            params_hash=self.params.config_hash()
        )]
    
    def _merge_markets(
        self,
        markets_list: List[Tuple[float, Dict[str, List[MarketSnapshot]]]]
    ) -> Dict[str, List[MarketSnapshot]]:
        """Merge multiple market dicts into one."""
        merged = {}
        for _, markets in markets_list:
            for ticker, snaps in markets.items():
                if ticker not in merged:
                    merged[ticker] = []
                merged[ticker].extend(snaps)
        # Sort each market's snapshots
        for ticker in merged:
            merged[ticker].sort(key=lambda s: s.timestamp)
        return merged
    
    def get_summary(self) -> Dict:
        """Get aggregate walk-forward summary."""
        if not self.results:
            return {}
        
        train_evs = [r.train_metrics["ev_per_trade"] for r in self.results]
        test_evs = [r.test_metrics["ev_per_trade"] for r in self.results]
        train_dds = [r.train_metrics["max_drawdown"] for r in self.results]
        test_dds = [r.test_metrics["max_drawdown"] for r in self.results]
        
        return {
            "n_splits": len(self.results),
            "train_ev_mean": np.mean(train_evs),
            "train_ev_std": np.std(train_evs),
            "test_ev_mean": np.mean(test_evs),
            "test_ev_std": np.std(test_evs),
            "train_dd_mean": np.mean(train_dds),
            "test_dd_mean": np.mean(test_dds),
            "consistency": np.mean([1 if t > 0 else 0 for t in test_evs]),
            "params_hash": self.params.config_hash()
        }


# =====================================================================
# Parameter Grid Search
# =====================================================================

def parameter_grid(quick: bool = False) -> List[BacktestParams]:
    """Generate parameter grid for sweep."""
    if quick:
        grid = {
            "STRATEGY_MODE": ["early_maker", "final_taker"],
            "ENTRY_WINDOW_START_MIN": [6.0],
            "ENTRY_WINDOW_END_MIN": [11.0],
            "ENTRY_BAND_MIN": [15, 50],
            "ENTRY_BAND_MAX": [55, 85],
            "PER_TRADE_PCT": [0.08],
            "BALANCE_PCT_PER_TRADE": [0.08],
            "EXIT_STOP_LOSS_CENTS": [0, 10],
            "AGGRESSIVE_FILL": [False],
        }
    else:
        grid = {
            "STRATEGY_MODE": ["early_maker", "final_taker", "latency_arb"],
            "ENTRY_WINDOW_START_MIN": [4.0, 6.0, 8.0, 14.0],
            "ENTRY_WINDOW_END_MIN": [9.0, 11.0, 15.0],
            "ENTRY_BAND_MIN": [10, 15, 30, 50, 75, 85],
            "ENTRY_BAND_MAX": [55, 70, 85, 95, 99],
            "PER_TRADE_PCT": [0.04, 0.08, 0.12, 0.15, 0.20],
            "BALANCE_PCT_PER_TRADE": [0.05, 0.08, 0.15, 0.25],
            "MAX_TRADE_DOLLARS": [5.0, 10.0, 20.0],
            "EXIT_STOP_LOSS_CENTS": [0, 5, 10, 15],
            "TAKE_PROFIT_CENTS": [0, 20, 30],
            "AGGRESSIVE_FILL": [False, True],
            "FILL_PROBABILITY_BASE": [0.6, 0.8],
            "USE_ML_FILTER": [False, True],
            "USE_GATES": [True],
            "GATES_SHADOW_MODE": [True],
        }
    
    keys = list(grid.keys())
    # Use islice to limit without generating all combos
    import itertools
    max_combos = 5000 if not quick else 1000
    for i, combo in enumerate(itertools.product(*[grid[k] for k in keys])):
        if i >= max_combos:
            break
        params_dict = dict(zip(keys, combo))
        # Validate combo
        if params_dict["ENTRY_WINDOW_START_MIN"] >= params_dict["ENTRY_WINDOW_END_MIN"]:
            continue
        if params_dict["ENTRY_BAND_MIN"] >= params_dict["ENTRY_BAND_MAX"]:
            continue
        yield BacktestParams(**params_dict)


# =====================================================================
# Main Sweep Runner
# =====================================================================

def run_sweep(
    markets: Dict[str, List[MarketSnapshot]],
    settlement_fn: Callable[[str], Optional[bool]],
    params_list: List[BacktestParams] = None,
    quick: bool = False,
    top: int = 10,
    jobs: int = 4,
    out_path: str = "backtest_harness_results.json",
    walkforward: bool = True,
    markets_by_time: List[Tuple[float, Dict]] = None
) -> List[Dict]:
    """
    Run parameter sweep with optional walk-forward validation.
    
    Args:
        markets: Merged markets dict (ticker -> snapshots)
        settlement_fn: Settlement outcome function
        params_list: Pre-generated parameter list (or generate from grid)
        quick: Use quick grid
        top: Number of top results to display
        jobs: Parallel workers
        out_path: Output JSON path
        walkforward: Run walk-forward validation
        markets_by_time: Time-sorted markets for walk-forward
    
    Returns:
        Sorted results list
    """
    if params_list is None:
        params_list = list(parameter_grid(quick))
    
    jobs = min(jobs, cpu_count())
    log.info(f"Testing {len(params_list)} parameter combinations with {jobs} workers...")
    
    # Run backtests in parallel
    work_items = [(p, markets, settlement_fn, walkforward, markets_by_time) for p in params_list]
    
    with Pool(processes=jobs) as pool:
        results = pool.map(_backtest_one_param, work_items)
    
    # Sort by risk-adjusted return (PnL / (MaxDD + epsilon))
    results.sort(key=lambda r: r["total_pnl"] / (r["max_drawdown"] + 0.01), reverse=True)
    
    # Print results
    _print_results(results, len(markets), len(params_list), top)
    
    # Save
    _save_results(results, len(markets), len(params_list), out_path)
    
    return results


def _backtest_one_param(args) -> Dict:
    """Worker function for parallel backtest."""
    params, markets, settlement_fn, walkforward, markets_by_time = args
    
    if walkforward and markets_by_time:
        validator = WalkForwardValidator(params)
        wf_results = validator.validate(markets_by_time, settlement_fn)
        wf_summary = validator.get_summary()
        
        # Use test metrics as primary
        if wf_results:
            test_metrics = wf_results[-1].test_metrics  # Last split test
            test_metrics["walkforward"] = wf_summary
            test_metrics["params"] = params.to_dict()
            return test_metrics
    
    # Simple backtest
    engine = BacktestEngine(params)
    metrics = engine.run_backtest(markets, settlement_fn)
    metrics["params"] = params.to_dict()
    return metrics


def _print_results(results: List[Dict], n_markets: int, n_combos: int, top: int):
    """Print ranked results."""
    print("\n" + "=" * 100)
    print(f"BACKTEST HARNESS RESULTS — {n_markets} markets, {n_combos} parameter sets")
    print("=" * 100)
    print(f"\n{'Rank':<5} {'EV/trade':<10} {'Total P&L':<11} {'Trades':<7} "
          f"{'Win%':<7} {'Entry':<7} {'Edge':<8} {'MaxDD':<8} {'Sharpe':<8} {'PF':<6} {'Hash'}")
    print("-" * 100)
    for i, r in enumerate(results[:top]):
        p = r.get("params", {})
        wf = r.get("walkforward", {})
        wf_tag = " WF" if wf else ""
        print(f"{i+1:<5} {r['ev_per_trade']:<10.4f} {r['total_pnl']:<11.2f} {r['n_trades']:<7} "
              f"{r['win_rate']*100:<6.1f}% {r['avg_entry_cents']:<7.1f} "
              f"{r['gross_edge_cents']:<+8.1f} {r['max_drawdown']*100:<7.1f}% "
              f"{r['sharpe_ratio']:<8.2f} {r['profit_factor']:<6.2f} "
              f"{r['params_hash']}{wf_tag}")


def _save_results(results: List[Dict], n_markets: int, n_combos: int, out_path: str):
    """Save results to JSON."""
    out = {
        "n_markets": n_markets,
        "n_combos": n_combos,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in results[:50]
        ],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"Full results saved to {out_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Backtest Harness (T13-30)")
    parser.add_argument("--quick", action="store_true", help="Quick grid")
    parser.add_argument("--top", type=int, default=10, help="Top N results")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel workers")
    parser.add_argument("--no-walkforward", action="store_true", help="Disable walk-forward")
    parser.add_argument("--out", default="backtest_harness_results.json", help="Output path")
    parser.add_argument("--series", nargs="+", default=["KXBTC15M", "KXETH15M"], help="Series to test")
    parser.add_argument("--days", type=int, default=30, help="Days of history")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--max-markets", type=int, default=None, help="Max markets per series")
    parser.add_argument("--use-historical", action="store_true", default=True, help="Use historical settled markets (default: True)")
    parser.add_argument("--use-tick-store", action="store_true", help="Use tick_store for spot replay instead")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    
    if args.use_tick_store:
        # Original tick_store-based approach (for spot price replay)
        tick_store = get_tick_store()
        end_ts = time.time()
        start_ts = end_ts - args.days * 86400
        
        all_markets = {}
        markets_by_time = []
        
        for series in args.series:
            markets = tick_store.query_ticks_all_venues(
                symbol=series.replace("KX", "").replace("15M", "USDT"),
                start_ts=start_ts,
                end_ts=end_ts
            )
            # Reconstruct (simplified)
            replayer = MarketReplayer(tick_store)
            series_markets = replayer.get_market_series(series, start_ts, end_ts)
            
            # Add to combined
            for ticker, snaps in series_markets.items():
                all_markets[ticker] = snaps
            
            # For walk-forward, create time-sorted list
            # Group by day
            daily = {}
            for ticker, snaps in series_markets.items():
                for snap in snaps:
                    day = datetime.fromtimestamp(snap.timestamp, tz=timezone.utc).date().isoformat()
                    if day not in daily:
                        daily[day] = {}
                    if ticker not in daily[day]:
                        daily[day][ticker] = []
                    daily[day][ticker].append(snap)
            
            for day in sorted(daily.keys()):
                day_ts = datetime.fromisoformat(day).timestamp()
                markets_by_time.append((day_ts, daily[day]))
        
        # Settlement function (would query Kalshi or local DB)
        def _cli_settlement_fn(ticker):
            # Placeholder: would look up actual settlement
            return np.random.random() > 0.5
        
        settlement_fn = _cli_settlement_fn
    else:
        # New: Use historical settled markets with real outcomes
        from historical_loader import load_backtest_data, get_markets_by_time
        
        # Calculate date range
        if args.start_date:
            start_date = args.start_date
        else:
            start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=args.days)).date().isoformat()
        
        end_date = args.end_date
        
        log.info(f"Loading historical data for {args.series} from {start_date} to {end_date or 'now'}")
        
        all_markets, settlement_fn, all_metadata = load_backtest_data(
            series=args.series,
            start_date=start_date,
            end_date=end_date,
            max_markets_per_series=args.max_markets,
            snapshots_per_market=10,
        )
        
        # Create walk-forward grouping
        markets_by_time = get_markets_by_time(all_markets, all_metadata)
        
        log.info(f"Loaded {len(all_markets)} markets, {len(markets_by_time)} walk-forward days")
    
    # Run sweep
    results = run_sweep(
        markets=all_markets,
        settlement_fn=settlement_fn,
        quick=args.quick,
        top=args.top,
        jobs=args.jobs,
        out_path=args.out,
        walkforward=not args.no_walkforward,
        markets_by_time=markets_by_time
    )
    
    # Print best detail
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("BEST PARAMETERS:")
        print("=" * 80)
        for k, v in best.get("params", {}).items():
            print(f"  {k}: {v}")
        print(f"\n  EV/trade: ${best['ev_per_trade']:.4f}")
        print(f"  Total P&L: ${best['total_pnl']:.2f}")
        print(f"  Trades: {best['n_trades']}")
        print(f"  Win rate: {best['win_rate']*100:.1f}%")
        print(f"  Max drawdown: {best['max_drawdown']*100:.1f}%")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Config Hash: {best['params_hash']}")


if __name__ == "__main__":
    main()