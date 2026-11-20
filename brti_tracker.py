#!/usr/bin/env python3
"""
brti_tracker.py — Kalshi reference index (CF Benchmarks BRTI) tracker (T15-11).

CF Benchmarks BRTI is a 60-second volume-weighted average of BTC prices across
major exchanges. Kalshi 15m contracts settle on this index.

Since real-time BRTI is not available under our license, we build a proxy:
- Median of Binance and Coinbase spot prices (public WS feeds)
- Compute running 60-second average to match BRTI methodology
- Track proxy-vs-settlement error for calibration

This module provides:
- BRTITracker: maintains running 60s average from WS ticks
- get_brti_proxy(): synchronous fallback using REST composite_price
- Error tracking for basis measurement
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import binance_feed

log = logging.getLogger(__name__)


@dataclass
class TickSample:
    """A single price sample with timestamp."""
    price: float
    timestamp: float  # Unix timestamp (seconds)
    venue: str


@dataclass
class BRTIState:
    """Current BRTI proxy state for a symbol."""
    symbol: str
    running_average: Optional[float] = None
    samples_in_window: int = 0
    window_start: Optional[float] = None
    last_update: Optional[float] = None
    # Error tracking
    last_settlement_price: Optional[float] = None
    last_basis_bps: Optional[float] = None


class BRTITracker:
    """
    Tracks CF Benchmarks BRTI proxy using WS tick data.
    
    Maintains a running 60-second volume-weighted (approximated as time-weighted
    with median venue price) average per symbol.
    """
    
    def __init__(self, symbols: List[str] = None, window_seconds: float = 60.0):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        self.window_seconds = window_seconds
        # Per-symbol tick buffers
        self._tick_buffers: Dict[str, deque] = {
            sym: deque(maxlen=1000) for sym in self.symbols
        }
        # Per-symbol state
        self._states: Dict[str, BRTIState] = {
            sym: BRTIState(symbol=sym) for sym in self.symbols
        }
        self._running = False
        self._lock = asyncio.Lock()
    
    async def add_tick(self, symbol: str, price: float, venue: str, timestamp: float = None):
        """Add a tick sample to the buffer."""
        if symbol not in self._tick_buffers:
            return
        if timestamp is None:
            timestamp = time.time()
        
        async with self._lock:
            self._tick_buffers[symbol].append(TickSample(
                price=price, timestamp=timestamp, venue=venue
            ))
            await self._recompute_average(symbol, timestamp)
    
    async def _recompute_average(self, symbol: str, now: float):
        """Recompute the 60-second running average."""
        buffer = self._tick_buffers[symbol]
        state = self._states[symbol]
        
        # Remove ticks older than window
        cutoff = now - self.window_seconds
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()
        
        if not buffer:
            state.running_average = None
            state.samples_in_window = 0
            state.window_start = None
            return
        
        # Group by venue and compute median per venue, then median across venues
        venue_prices: Dict[str, List[float]] = {}
        for tick in buffer:
            venue_prices.setdefault(tick.venue, []).append(tick.price)
        
        venue_medians = []
        for venue, prices in venue_prices.items():
            sorted_prices = sorted(prices)
            n = len(sorted_prices)
            if n % 2:
                venue_medians.append(sorted_prices[n // 2])
            else:
                venue_medians.append((sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2.0)
        
        if venue_medians:
            sorted_medians = sorted(venue_medians)
            n = len(sorted_medians)
            if n % 2:
                state.running_average = sorted_medians[n // 2]
            else:
                state.running_average = (sorted_medians[n // 2 - 1] + sorted_medians[n // 2]) / 2.0
        
        state.samples_in_window = len(buffer)
        state.window_start = buffer[0].timestamp if buffer else None
        state.last_update = now
    
    def get_state(self, symbol: str) -> Optional[BRTIState]:
        """Get current BRTI state for a symbol (sync, thread-safe read)."""
        return self._states.get(symbol)
    
    def get_running_average(self, symbol: str) -> Optional[float]:
        """Get the current 60-second running average (sync)."""
        state = self._states.get(symbol)
        return state.running_average if state else None
    
    def get_age_seconds(self, symbol: str) -> Optional[float]:
        """Get age of the running average in seconds (sync)."""
        state = self._states.get(symbol)
        if state is None or state.last_update is None:
            return None
        return time.time() - state.last_update
    
    def record_settlement(self, symbol: str, settlement_price: float):
        """Record a settlement price for basis error tracking."""
        state = self._states.get(symbol)
        if state is None or state.running_average is None:
            return
        state.last_settlement_price = settlement_price
        # Basis in bps: (settlement - proxy) / proxy * 10000
        state.last_basis_bps = (settlement_price - state.running_average) / state.running_average * 10000
        log.info("BRTI basis for %s: proxy=%.2f, settlement=%.2f, basis=%.1f bps",
                 symbol, state.running_average, settlement_price, state.last_basis_bps)
    
    def get_basis_bps(self, symbol: str) -> Optional[float]:
        """Get last recorded basis error in bps."""
        state = self._states.get(symbol)
        return state.last_basis_bps if state else None


# ─── Global singleton ──────────────────────────────────────────────────────────

_brti_tracker: Optional[BRTITracker] = None


async def get_brti_tracker(symbols: List[str] = None) -> BRTITracker:
    """Get or create the module-level BRTITracker singleton."""
    global _brti_tracker
    if _brti_tracker is None:
        _brti_tracker = BRTITracker(symbols)
    return _brti_tracker


async def close_brti_tracker():
    """Close the module-level BRTITracker singleton."""
    global _brti_tracker
    if _brti_tracker:
        _brti_tracker = None


# ─── Synchronous fallback (REST) ───────────────────────────────────────────────

def get_brti_proxy(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Synchronous fallback using REST composite_price.
    Returns a single-point proxy (not a 60s average).
    """
    return binance_feed.composite_price(symbol)


def get_brti_proxy_with_age(symbol: str = "BTCUSDT") -> tuple[Optional[float], Optional[float]]:
    """
    Get BRTI proxy with age estimate.
    For REST fallback, age is unknown (returns None).
    """
    price = get_brti_proxy(symbol)
    return price, None


# ─── Basis measurement utilities ───────────────────────────────────────────────

async def measure_basis_error(tracker: BRTITracker, symbol: str,
                               settlement_price: float) -> Optional[float]:
    """Measure and record basis error against a settlement price."""
    tracker.record_settlement(symbol, settlement_price)
    return tracker.get_basis_bps(symbol)


async def compute_hourly_basis_stats(tracker: BRTITracker, symbol: str,
                                      hours: int = 24) -> Dict:
    """
    Compute basis statistics over the last N hours.
    Requires settlement data to be recorded via record_settlement().
    This is a placeholder - full implementation needs persistent storage.
    """
    state = tracker.get_state(symbol)
    if not state or state.last_basis_bps is None:
        return {"error": "No basis data available"}
    
    return {
        "symbol": symbol,
        "last_basis_bps": state.last_basis_bps,
        "last_proxy": state.running_average,
        "last_settlement": state.last_settlement_price,
        "samples_in_window": state.samples_in_window,
        "window_age_seconds": tracker.get_age_seconds(symbol),
    }


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test():
        tracker = BRTITracker(["BTCUSDT"])
        
        # Simulate ticks
        now = time.time()
        for i in range(10):
            await tracker.add_tick("BTCUSDT", 95000 + i * 10, "binance", now - 5 + i)
            await asyncio.sleep(0.01)
        
        print(f"Running average: {tracker.get_running_average('BTCUSDT')}")
        print(f"Samples in window: {tracker.get_state('BTCUSDT').samples_in_window}")
        print(f"Age: {tracker.get_age_seconds('BTCUSDT'):.2f}s")
        
        # Test basis measurement
        tracker.record_settlement("BTCUSDT", 95050)
        print(f"Basis: {tracker.get_basis_bps('BTCUSDT'):.1f} bps")
    
    asyncio.run(test())