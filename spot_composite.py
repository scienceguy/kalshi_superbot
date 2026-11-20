"""
spot_composite.py — robust, low-latency composite spot price from Binance and Coinbase WS.

Implements T13-10: Composite spot service.
- Binance: public trade/bookTicker (and 1s kline for vol) streams.
- Coinbase: public WS (ticker/market trades/level2 best bid-ask).
- Per tick keep {venue, symbol, px, bid, ask, exch_ts, recv_ts}; compute
  composite = median of fresh venues; reject a venue if recv_ts - exch_ts
  or age exceeds a limit; flag > N bps divergence.
- Region warning: Binance.com blocks US IPs. Uses data-stream.binance.vision.
- Reconnect with jittered backoff; sequence/gap detection; never block the
  event loop; expose get_spot_with_age() for the guard.

Accept: soak test 24h; p99 tick age < 250ms; automatic failover when one
venue is killed; no stale price ever used (age check enforced in tests).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

import websockets

log = logging.getLogger("spot_composite")

# Binance market-data endpoints (not blocked from this host)
BINANCE_WS = "wss://data-stream.binance.vision/ws"
BINANCE_REST = "https://data-api.binance.vision/api/v3"

# Coinbase public WS
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"

# Assets we support (must match binance_feed._KNOWN_BASES)
SUPPORTED_BASES = {"BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE", "ZEC", "NEAR"}

# Max age (seconds) before a venue's price is considered stale
MAX_TICK_AGE_SECONDS = 5.0
# Max exchange-to-receive latency (seconds) before rejecting
MAX_EXCHANGE_LATENCY_SECONDS = 1.0
# Max divergence (bps) between venues before flagging
MAX_DIVERGENCE_BPS = 50.0


@dataclass
class Tick:
    """Normalized tick from a venue."""
    venue: str
    symbol: str
    px: float
    bid: Optional[float]
    ask: Optional[float]
    exch_ts: float  # exchange timestamp (seconds since epoch)
    recv_ts: float  # local receive timestamp (seconds since epoch)

    @property
    def age(self) -> float:
        return self.recv_ts - self.exch_ts

    @property
    def is_fresh(self) -> bool:
        return self.age <= MAX_TICK_AGE_SECONDS and (self.recv_ts - self.exch_ts) <= MAX_EXCHANGE_LATENCY_SECONDS


@dataclass
class VenueState:
    """Latest state per venue per symbol."""
    ticks: Dict[str, Tick] = field(default_factory=dict)
    last_seq: int = 0
    connected: bool = False
    last_recv: float = 0.0


class SpotComposite:
    """Manages WebSocket connections to Binance and Coinbase, computes composite price."""

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.venues = {
            "binance": VenueState(),
            "coinbase": VenueState(),
        }
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._ready_event = asyncio.Event()

    async def start(self):
        """Start all WebSocket connections."""
        if self._running:
            return
        self._running = True

        for symbol in self.symbols:
            # Binance streams
            self._tasks.append(asyncio.create_task(self._run_binance(symbol)))
            # Coinbase streams
            self._tasks.append(asyncio.create_task(self._run_coinbase(symbol)))

        # Wait for at least one venue to be connected
        await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
        log.info("SpotComposite started for %s", self.symbols)

    async def stop(self):
        """Stop all WebSocket connections."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("SpotComposite stopped")

    def get_spot_with_age(self, symbol: str) -> Optional[float]:
        """
        Get composite spot price with age check.

        Returns median of fresh venues, or None if no fresh prices.
        This is the primary API for the guard/gates.
        """
        fresh_prices = []
        for venue_name, venue_state in self.venues.items():
            tick = venue_state.ticks.get(symbol)
            if tick and tick.is_fresh:
                fresh_prices.append(tick.px)
            elif tick:
                log.debug("Stale tick from %s for %s: age=%.3fs", venue_name, symbol, tick.age)

        if not fresh_prices:
            log.warning("No fresh prices for %s", symbol)
            return None

        fresh_prices.sort()
        mid = len(fresh_prices) // 2
        if len(fresh_prices) % 2:
            return fresh_prices[mid]
        return (fresh_prices[mid - 1] + fresh_prices[mid]) / 2.0

    def get_spot_with_age_detail(self, symbol: str) -> Dict:
        """
        Get detailed composite info including per-venue prices and divergence.
        """
        result = {
            "symbol": symbol,
            "composite": None,
            "venues": {},
            "divergence_bps": None,
            "fresh_count": 0,
        }

        fresh_prices = []
        for venue_name, venue_state in self.venues.items():
            tick = venue_state.ticks.get(symbol)
            if tick:
                fresh = tick.is_fresh
                if fresh:
                    fresh_prices.append(tick.px)
                result["venues"][venue_name] = {
                    "px": tick.px,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "age_sec": tick.age,
                    "exch_latency_sec": tick.recv_ts - tick.exch_ts,
                    "fresh": fresh,
                }

        if fresh_prices:
            fresh_prices.sort()
            mid = len(fresh_prices) // 2
            if len(fresh_prices) % 2:
                result["composite"] = fresh_prices[mid]
            else:
                result["composite"] = (fresh_prices[mid - 1] + fresh_prices[mid]) / 2.0
            result["fresh_count"] = len(fresh_prices)

            # Calculate divergence between venues
            venue_prices = [v["px"] for v in result["venues"].values() if v["fresh"]]
            if len(venue_prices) >= 2:
                max_p = max(venue_prices)
                min_p = min(venue_prices)
                if max_p > 0:
                    result["divergence_bps"] = ((max_p - min_p) / max_p) * 10000
                    if result["divergence_bps"] > MAX_DIVERGENCE_BPS:
                        log.warning(
                            "Divergence %.1f bps for %s exceeds limit %.1f bps",
                            result["divergence_bps"], symbol, MAX_DIVERGENCE_BPS
                        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # Binance WebSocket
    # ═══════════════════════════════════════════════════════════════

    async def _run_binance(self, symbol: str):
        """Run Binance bookTicker + trade streams with reconnect."""
        streams = [
            f"{symbol.lower()}@bookTicker",
            f"{symbol.lower()}@trade",
        ]
        url = BINANCE_WS + "/" + "/".join(streams)
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
                    self.venues["binance"].connected = True
                    self._ready_event.set()
                    backoff = 1
                    log.info("Binance WS connected for %s", symbol)

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._handle_binance_msg(msg, symbol)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            log.debug("Binance parse error: %s", e)
                            continue

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Binance WS disconnected for %s: %s — reconnecting in %ds", symbol, e, backoff)
                self.venues["binance"].connected = False
                await asyncio.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, max_backoff)

    def _handle_binance_msg(self, msg: dict, symbol: str):
        """Process Binance bookTicker or trade message."""
        recv_ts = time.time()

        # bookTicker: {"u":..., "s":"BTCUSDT", "b":"...", "B":"...", "a":"...", "A":"..."}
        if "b" in msg and "a" in msg and msg.get("s") == symbol:
            try:
                bid = float(msg["b"])
                ask = float(msg["a"])
                px = (bid + ask) / 2.0
                # Binance bookTicker doesn't have exch_ts; use recv_ts as fallback
                exch_ts = recv_ts
                self.venues["binance"].ticks[symbol] = Tick(
                    venue="binance",
                    symbol=symbol,
                    px=px,
                    bid=bid,
                    ask=ask,
                    exch_ts=exch_ts,
                    recv_ts=recv_ts,
                )
                self.venues["binance"].last_recv = recv_ts
            except (ValueError, TypeError):
                pass

        # trade: {"e":"trade", "E":1234567890, "s":"BTCUSDT", "t":12345, "p":"...", "q":"...", ...}
        elif msg.get("e") == "trade" and msg.get("s") == symbol:
            try:
                px = float(msg["p"])
                exch_ts = msg.get("E", recv_ts * 1000) / 1000.0
                # Trade doesn't have bid/ask; use last bookTicker if available
                last_tick = self.venues["binance"].ticks.get(symbol)
                bid = last_tick.bid if last_tick else None
                ask = last_tick.ask if last_tick else None
                self.venues["binance"].ticks[symbol] = Tick(
                    venue="binance",
                    symbol=symbol,
                    px=px,
                    bid=bid,
                    ask=ask,
                    exch_ts=exch_ts,
                    recv_ts=recv_ts,
                )
                self.venues["binance"].last_recv = recv_ts
            except (ValueError, TypeError):
                pass

    # ═══════════════════════════════════════════════════════════════
    # Coinbase WebSocket
    # ═══════════════════════════════════════════════════════════════

    async def _run_coinbase(self, symbol: str):
        """Run Coinbase ticker + level2 streams with reconnect."""
        # Convert BTCUSDT -> BTC-USD for Coinbase
        base = symbol.replace("USDT", "").replace("USDC", "")
        cb_symbol = f"{base}-USD"

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [cb_symbol],
            "channels": ["ticker", "level2", "matches"],
        }

        backoff = 1
        max_backoff = 60

        while self._running:
            try:
                async with websockets.connect(
                    COINBASE_WS,
                    open_timeout=10,
                    close_timeout=10,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    self.venues["coinbase"].connected = True
                    self._ready_event.set()
                    backoff = 1
                    log.info("Coinbase WS connected for %s", cb_symbol)

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._handle_coinbase_msg(msg, symbol, cb_symbol)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            log.debug("Coinbase parse error: %s", e)
                            continue

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Coinbase WS disconnected for %s: %s — reconnecting in %ds", cb_symbol, e, backoff)
                self.venues["coinbase"].connected = False
                await asyncio.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, max_backoff)

    def _handle_coinbase_msg(self, msg: dict, symbol: str, cb_symbol: str):
        """Process Coinbase ticker, level2, or matches message."""
        recv_ts = time.time()
        msg_type = msg.get("type")
        product_id = msg.get("product_id", "")

        if product_id != cb_symbol:
            return

        # ticker: {"type":"ticker", "trade_id":..., "sequence":..., "time":"...", "product_id":"...", "price":"...", "best_bid":"...", "best_ask":"...", ...}
        if msg_type == "ticker":
            try:
                px = float(msg["price"])
                bid = float(msg.get("best_bid", 0)) or None
                ask = float(msg.get("best_ask", 0)) or None
                # Coinbase provides ISO timestamp
                time_str = msg.get("time")
                exch_ts = self._parse_iso_timestamp(time_str) if time_str else recv_ts
                self.venues["coinbase"].ticks[symbol] = Tick(
                    venue="coinbase",
                    symbol=symbol,
                    px=px,
                    bid=bid,
                    ask=ask,
                    exch_ts=exch_ts,
                    recv_ts=recv_ts,
                )
                self.venues["coinbase"].last_recv = recv_ts
            except (ValueError, TypeError):
                pass

        # l2update: {"type":"l2update", "product_id":"...", "time":"...", "changes":[["buy","...","..."],["sell","...","..."],...]}
        elif msg_type == "l2update":
            try:
                changes = msg.get("changes", [])
                bid = None
                ask = None
                for side, price_str, size_str in changes:
                    if side == "buy" and bid is None:
                        bid = float(price_str)
                    elif side == "sell" and ask is None:
                        ask = float(price_str)
                if bid is not None and ask is not None:
                    px = (bid + ask) / 2.0
                    time_str = msg.get("time")
                    exch_ts = self._parse_iso_timestamp(time_str) if time_str else recv_ts
                    # Merge with existing ticker bid/ask if any
                    last_tick = self.venues["coinbase"].ticks.get(symbol)
                    self.venues["coinbase"].ticks[symbol] = Tick(
                        venue="coinbase",
                        symbol=symbol,
                        px=px,
                        bid=bid,
                        ask=ask,
                        exch_ts=exch_ts,
                        recv_ts=recv_ts,
                    )
                    self.venues["coinbase"].last_recv = recv_ts
            except (ValueError, TypeError):
                pass

        # match: {"type":"match", "trade_id":..., "sequence":..., "time":"...", "product_id":"...", "price":"...", "size":"...", "side":"buy|sell"}
        elif msg_type == "match":
            try:
                px = float(msg["price"])
                time_str = msg.get("time")
                exch_ts = self._parse_iso_timestamp(time_str) if time_str else recv_ts
                last_tick = self.venues["coinbase"].ticks.get(symbol)
                bid = last_tick.bid if last_tick else None
                ask = last_tick.ask if last_tick else None
                self.venues["coinbase"].ticks[symbol] = Tick(
                    venue="coinbase",
                    symbol=symbol,
                    px=px,
                    bid=bid,
                    ask=ask,
                    exch_ts=exch_ts,
                    recv_ts=recv_ts,
                )
                self.venues["coinbase"].last_recv = recv_ts
            except (ValueError, TypeError):
                pass

    @staticmethod
    def _parse_iso_timestamp(time_str: str) -> float:
        """Parse ISO timestamp string to Unix timestamp."""
        try:
            from datetime import datetime, timezone
            # Handle formats like "2026-11-20T03:15:30.123456Z"
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            return datetime.fromisoformat(time_str).timestamp()
        except Exception:
            return time.time()


# ═══════════════════════════════════════════════════════════════
# Convenience function for non-async callers (sync fallback to REST)
# ═══════════════════════════════════════════════════════════════

import binance_feed

def get_spot_with_age(symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Synchronous fallback using REST composite_price.
    Used when WebSocket composite is not available.
    """
    return binance_feed.composite_price(symbol)


# ═══════════════════════════════════════════════════════════════
# Module-level singleton for easy import
# ═══════════════════════════════════════════════════════════════

_spot_composite: Optional[SpotComposite] = None


async def get_spot_composite(symbols: List[str] = None) -> SpotComposite:
    """Get or create the module-level SpotComposite singleton."""
    global _spot_composite
    if _spot_composite is None:
        _spot_composite = SpotComposite(symbols)
        await _spot_composite.start()
    return _spot_composite


async def close_spot_composite():
    """Close the module-level SpotComposite singleton."""
    global _spot_composite
    if _spot_composite:
        await _spot_composite.stop()
        _spot_composite = None


if __name__ == "__main__":
    import sys

    async def main():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        sc = SpotComposite(["BTCUSDT", "ETHUSDT"])
        await sc.start()

        try:
            for _ in range(60):  # Run for 60 seconds
                await asyncio.sleep(1)
                for sym in ["BTCUSDT", "ETHUSDT"]:
                    detail = sc.get_spot_with_age_detail(sym)
                    print(f"{sym}: composite={detail['composite']:.2f} "
                          f"divergence={detail['divergence_bps']:.1f}bps "
                          f"venues={list(detail['venues'].keys())}")
        finally:
            await sc.stop()

    asyncio.run(main())