#!/usr/bin/env python3
"""
brti_tracker.py — CF Benchmarks BRTI 60-second simple average replication.

Implements T15-11: BRTI tracker for Kalshi/Polymarket settlement replication.

BRTI methodology (from CF Benchmarks):
- 60-second simple average of constituent exchange prices
- Constituents: Coinbase, Kraken, Bitstamp, LMAX Digital (as of 2026)
- Outlier filter: prices > 3 standard deviations from median are excluded
- Published every minute at :00 seconds
- Used as settlement index for Kalshi KX*15M and Polymarket US cpc-*-updown-15m-*

This module replicates the BRTI calculation in real-time using WebSocket feeds
from constituent exchanges, enabling pre-settlement price prediction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncIterator

import websockets

log = logging.getLogger("brti_tracker")

# Constituent exchanges for BRTI (per CF Benchmarks methodology)
# Note: Binance is NOT a BRTI constituent — we use Coinbase, Kraken, Bitstamp, LMAX
CONSTITUENTS = {
    "coinbase": {
        "ws_url": "wss://ws-feed.exchange.coinbase.com",
        "subscribe": {"type": "subscribe", "product_ids": ["BTC-USD", "ETH-USD"], "channels": ["ticker"]},
        "symbol_map": {"BTC-USD": "BTC", "ETH-USD": "ETH"},
    },
    "kraken": {
        "ws_url": "wss://ws.kraken.com/v2",
        "subscribe": {"method": "subscribe", "params": {"channel": "ticker", "symbol": ["BTC/USD", "ETH/USD"]}},
        "symbol_map": {"BTC/USD": "BTC", "ETH/USD": "ETH"},
    },
    "bitstamp": {
        "ws_url": "wss://ws.bitstamp.net",
        "subscribe": {"event": "bts:subscribe", "data": {"channel": "live_trades_btcusd"}},
        "symbol_map": {"btcusd": "BTC", "ethusd": "ETH"},
    },
    "lmax": {
        "ws_url": "wss://web-socket.lmax.com/v1/market-data",
        "subscribe": {"type": "SUBSCRIBE", "channels": [{"name": "ORDER_BOOK_1", "instruments": ["BTC/USD", "ETH/USD"]}]},
        "symbol_map": {"BTC/USD": "BTC", "ETH/USD": "ETH"},
    },
}

# Assets tracked
TRACKED_ASSETS = {"BTC", "ETH"}

# BRTI calculation parameters
BRTI_WINDOW_SECONDS = 60
OUTLIER_STD_THRESHOLD = 3.0
MIN_CONSTITUENTS = 2  # Minimum venues needed for valid BRTI


@dataclass
class VenuePrice:
    """Single price observation from a venue."""
    venue: str
    asset: str
    price: float
    timestamp: float  # Exchange timestamp (seconds)
    recv_timestamp: float  # Local receive timestamp

    @property
    def age(self) -> float:
        """Age of price in seconds (local time - exchange time)."""
        return self.recv_timestamp - self.timestamp


@dataclass
class BRTIValue:
    """Computed BRTI value for one asset at one minute."""
    asset: str
    value: float
    window_start: float
    window_end: float
    constituent_count: int
    constituent_prices: Dict[str, float]
    excluded_venues: List[str]
    computation_time: float


class BRTITracker:
    """Real-time BRTI replication using constituent exchange WebSocket feeds."""

    def __init__(self, assets: List[str] = None):
        self.assets = assets or ["BTC", "ETH"]
        self.venue_queues: Dict[str, asyncio.Queue] = {}
        self.latest_prices: Dict[str, Dict[str, VenuePrice]] = {asset: {} for asset in self.assets}
        self.brti_history: Dict[str, List[BRTIValue]] = {asset: [] for asset in self.assets}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._ready_event = asyncio.Event()

    async def start(self):
        """Start WebSocket connections to all constituent exchanges."""
        if self._running:
            return
        self._running = True

        # Start WebSocket tasks for each constituent
        for venue_name, config in CONSTITUENTS.items():
            self.venue_queues[venue_name] = asyncio.Queue(maxsize=1000)
            self._tasks.append(asyncio.create_task(self._run_venue(venue_name, config)))

        # Start BRTI computation loop
        self._tasks.append(asyncio.create_task(self._compute_brti_loop()))

        # Wait for at least one venue to connect
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
            log.info("BRTITracker started for assets: %s", self.assets)
        except asyncio.TimeoutError:
            log.warning("BRTITracker: no venue connected within 30s, continuing anyway")

    async def stop(self):
        """Stop all WebSocket connections and computation."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("BRTITracker stopped")

    def get_latest_brti(self, asset: str) -> Optional[BRTIValue]:
        """Get the most recent computed BRTI value for an asset."""
        history = self.brti_history.get(asset, [])
        return history[-1] if history else None

    def get_brti_history(self, asset: str, minutes: int = 60) -> List[BRTIValue]:
        """Get BRTI history for the last N minutes."""
        history = self.brti_history.get(asset, [])
        return history[-minutes:] if history else []

    def get_constituent_prices(self, asset: str) -> Dict[str, VenuePrice]:
        """Get latest prices from all constituent venues for an asset."""
        return self.latest_prices.get(asset, {}).copy()

    # ═══════════════════════════════════════════════════════════════
    # Venue WebSocket handlers
    # ═══════════════════════════════════════════════════════════════

    async def _run_venue(self, venue_name: str, config: dict):
        """Run WebSocket connection for a single venue with reconnection."""
        url = config["ws_url"]
        subscribe_msg = config["subscribe"]
        symbol_map = config["symbol_map"]
        backoff = 1
        max_backoff = 60

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    open_timeout=10,
                    close_timeout=10,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self._ready_event.set()
                    backoff = 1
                    log.info("BRTI venue %s connected", venue_name)

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._handle_venue_msg(venue_name, msg, symbol_map)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            log.debug("BRTI venue %s parse error: %s", venue_name, e)
                            continue

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("BRTI venue %s disconnected: %s — reconnecting in %ds", venue_name, e, backoff)
                await asyncio.sleep(backoff + 0.1 * (hash(venue_name) % 10))
                backoff = min(backoff * 2, max_backoff)

    def _handle_venue_msg(self, venue_name: str, msg: dict, symbol_map: dict):
        """Parse venue message and extract price."""
        recv_ts = time.time()

        # Coinbase ticker format
        if venue_name == "coinbase" and msg.get("type") == "ticker":
            product_id = msg.get("product_id", "")
            asset = symbol_map.get(product_id)
            if asset and asset in self.assets:
                try:
                    price = float(msg["price"])
                    # Coinbase provides ISO timestamp
                    time_str = msg.get("time")
                    exch_ts = self._parse_iso_timestamp(time_str) if time_str else recv_ts
                    self.latest_prices[asset][venue_name] = VenuePrice(
                        venue=venue_name,
                        asset=asset,
                        price=price,
                        timestamp=exch_ts,
                        recv_timestamp=recv_ts,
                    )
                except (ValueError, TypeError):
                    pass

        # Kraken v2 ticker format
        elif venue_name == "kraken" and msg.get("channel") == "ticker":
            for item in msg.get("data", []):
                symbol = item.get("symbol", "")
                asset = symbol_map.get(symbol)
                if asset and asset in self.assets:
                    try:
                        price = float(item.get("last", 0))
                        if price > 0:
                            exch_ts = time.time()  # Kraken doesn't always provide exch timestamp
                            self.latest_prices[asset][venue_name] = VenuePrice(
                                venue=venue_name,
                                asset=asset,
                                price=price,
                                timestamp=exch_ts,
                                recv_timestamp=recv_ts,
                            )
                    except (ValueError, TypeError):
                        pass

        # Bitstamp trade format (using trades as price proxy)
        elif venue_name == "bitstamp" and "data" in msg:
            data = msg.get("data", {})
            channel = msg.get("channel", "")
            if "btcusd" in channel.lower():
                asset = "BTC"
            elif "ethusd" in channel.lower():
                asset = "ETH"
            else:
                return
            if asset in self.assets:
                try:
                    price = float(data.get("price", 0))
                    if price > 0:
                        exch_ts = recv_ts  # Bitstamp trades don't always have timestamp
                        self.latest_prices[asset][venue_name] = VenuePrice(
                            venue=venue_name,
                            asset=asset,
                            price=price,
                            timestamp=exch_ts,
                            recv_timestamp=recv_ts,
                        )
                except (ValueError, TypeError):
                    pass

        # LMAX order book format
        elif venue_name == "lmax" and msg.get("type") == "ORDER_BOOK_1":
            for update in msg.get("updates", []):
                instrument = update.get("instrumentId", "")
                asset = symbol_map.get(instrument)
                if asset and asset in self.assets:
                    try:
                        bids = update.get("bids", [])
                        asks = update.get("asks", [])
                        if bids and asks:
                            bid = float(bids[0][0])
                            ask = float(asks[0][0])
                            price = (bid + ask) / 2.0
                            exch_ts = recv_ts
                            self.latest_prices[asset][venue_name] = VenuePrice(
                                venue=venue_name,
                                asset=asset,
                                price=price,
                                timestamp=exch_ts,
                                recv_timestamp=recv_ts,
                            )
                    except (ValueError, TypeError, IndexError):
                        pass

    @staticmethod
    def _parse_iso_timestamp(time_str: str) -> float:
        """Parse ISO timestamp string to Unix timestamp."""
        try:
            from datetime import datetime, timezone
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            return datetime.fromisoformat(time_str).timestamp()
        except Exception:
            return time.time()

    # ═══════════════════════════════════════════════════════════════
    # BRTI computation loop
    # ═══════════════════════════════════════════════════════════════

    async def _compute_brti_loop(self):
        """Compute BRTI every minute at the :00 second mark."""
        while self._running:
            try:
                now = time.time()
                # Sleep until next minute boundary
                next_minute = math.ceil(now / 60) * 60
                await asyncio.sleep(max(0.1, next_minute - now))

                window_start = next_minute - BRTI_WINDOW_SECONDS
                window_end = next_minute

                for asset in self.assets:
                    brti = self._compute_brti_for_window(asset, window_start, window_end, next_minute)
                    if brti:
                        self.brti_history[asset].append(brti)
                        # Keep last 24 hours (1440 minutes)
                        if len(self.brti_history[asset]) > 1440:
                            self.brti_history[asset] = self.brti_history[asset][-1440:]
                        log.debug("BRTI %s: %.2f (%d constituents)", asset, brti.value, brti.constituent_count)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("BRTI computation error: %s", e)
                await asyncio.sleep(1)

    def _compute_brti_for_window(
        self,
        asset: str,
        window_start: float,
        window_end: float,
        computation_time: float
    ) -> Optional[BRTIValue]:
        """Compute BRTI for a single asset over a 60-second window."""
        prices = self.latest_prices.get(asset, {})
        if not prices:
            return None

        # Filter prices within the window (allow some latency)
        valid_prices = {}
        excluded = []

        for venue, vp in prices.items():
            # Use exchange timestamp if available, otherwise recv timestamp
            ts = vp.timestamp if vp.timestamp > 0 else vp.recv_timestamp
            if window_start <= ts <= window_end:
                valid_prices[venue] = vp.price
            else:
                excluded.append(f"{venue}(ts={ts:.0f})")

        if len(valid_prices) < MIN_CONSTITUENTS:
            log.debug("BRTI %s: only %d valid constituents in window", asset, len(valid_prices))
            return None

        # Outlier detection using median absolute deviation (more robust than std)
        price_values = list(valid_prices.values())
        median_price = statistics.median(price_values)
        deviations = [abs(p - median_price) for p in price_values]
        mad = statistics.median(deviations) if deviations else 0

        # Convert MAD to std equivalent (for normal distribution, MAD * 1.4826 ≈ std)
        robust_std = mad * 1.4826 if mad > 0 else 0

        # Filter outliers
        final_prices = {}
        for venue, price in valid_prices.items():
            if robust_std > 0 and abs(price - median_price) > OUTLIER_STD_THRESHOLD * robust_std:
                excluded.append(f"{venue}(outlier)")
                log.debug("BRTI %s: excluded outlier %s price=%.2f", asset, venue, price)
            else:
                final_prices[venue] = price

        if len(final_prices) < MIN_CONSTITUENTS:
            log.debug("BRTI %s: only %d constituents after outlier filter", asset, len(final_prices))
            return None

        # Simple average of remaining prices
        brti_value = sum(final_prices.values()) / len(final_prices)

        return BRTIValue(
            asset=asset,
            value=brti_value,
            window_start=window_start,
            window_end=window_end,
            constituent_count=len(final_prices),
            constituent_prices=final_prices,
            excluded_venues=excluded,
            computation_time=computation_time,
        )


# ════════════════════════════════════════════════════════════════
# Convenience functions for sync callers
# ═══════════════════════════════════════════════════════════════

_brti_tracker: Optional[BRTITracker] = None


async def get_brti_tracker(assets: List[str] = None) -> BRTITracker:
    """Get or create the module-level BRTITracker singleton."""
    global _brti_tracker
    if _brti_tracker is None:
        _brti_tracker = BRTITracker(assets)
        await _brti_tracker.start()
    return _brti_tracker


async def close_brti_tracker():
    """Close the module-level BRTITracker singleton."""
    global _brti_tracker
    if _brti_tracker:
        await _brti_tracker.stop()
        _brti_tracker = None


def get_latest_brti(asset: str) -> Optional[float]:
    """Synchronous getter for latest BRTI value (returns None if tracker not running)."""
    if _brti_tracker:
        brti = _brti_tracker.get_latest_brti(asset)
        return brti.value if brti else None
    return None


# ═══════════════════════════════════════════════════════════════
# Demo / CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    async def main():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s"
        )

        tracker = BRTITracker(["BTC", "ETH"])
        await tracker.start()

        try:
            for _ in range(120):  # Run for 2 minutes
                await asyncio.sleep(1)
                for asset in ["BTC", "ETH"]:
                    brti = tracker.get_latest_brti(asset)
                    if brti:
                        constituents = ", ".join(f"{v}:{p:.2f}" for v, p in brti.constituent_prices.items())
                        print(f"{asset} BRTI: {brti.value:.2f} ({brti.constituent_count} venues) [{constituents}]")
                    else:
                        print(f"{asset} BRTI: computing...")
        finally:
            await tracker.stop()

    asyncio.run(main())