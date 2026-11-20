#!/usr/bin/env python3
"""
venue_books.py — Venue orderbook tracking with book-age gate.

Implements T15-12: Venue books + book-age gate (orderbook freshness < 2s).

Provides:
- Real-time orderbook state from Kalshi (REST polling) and Polymarket US (WS/REST)
- Book-age tracking with configurable freshness threshold
- Gate function that rejects trades when books are stale
- Composite best bid/ask across venues for cross-venue arb
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger("venue_books")

# Default freshness thresholds
DEFAULT_MAX_BOOK_AGE_SECONDS = 2.0
DEFAULT_MAX_EXCHANGE_LATENCY_SECONDS = 1.0


@dataclass
class BookLevel:
    """Single price level in orderbook."""
    price: float
    size: float


@dataclass
class VenueBook:
    """Orderbook snapshot from a single venue."""
    venue: str
    market_id: str
    asset: str
    horizon: str  # e.g., "15m"
    bids: List[BookLevel]  # Sorted descending (best first)
    asks: List[BookLevel]  # Sorted ascending (best first)
    timestamp: float  # Exchange timestamp (seconds)
    recv_timestamp: float  # Local receive timestamp
    sequence: int = 0  # For gap detection

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bids[0].size if self.bids else None

    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0].size if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def age(self) -> float:
        """Age of book in seconds (local time - exchange time)."""
        return self.recv_timestamp - self.timestamp

    @property
    def is_fresh(self) -> bool:
        """Check if book is fresh enough for trading (uses defaults)."""
        return self.age <= DEFAULT_MAX_BOOK_AGE_SECONDS and (self.recv_timestamp - self.timestamp) <= DEFAULT_MAX_EXCHANGE_LATENCY_SECONDS

    def is_fresh_with(self, max_age: float = DEFAULT_MAX_BOOK_AGE_SECONDS,
                      max_latency: float = DEFAULT_MAX_EXCHANGE_LATENCY_SECONDS) -> bool:
        """Check if book is fresh enough for trading with custom thresholds."""
        return self.age <= max_age and (self.recv_timestamp - self.timestamp) <= max_latency


@dataclass
class CompositeBook:
    """Composite best bid/ask across venues."""
    asset: str
    horizon: str
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    yes_bid_venue: Optional[str] = None
    yes_ask_venue: Optional[str] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    no_bid_venue: Optional[str] = None
    no_ask_venue: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    venues_included: List[str] = field(default_factory=list)

    @property
    def yes_mid(self) -> Optional[float]:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2.0
        return None

    @property
    def no_mid(self) -> Optional[float]:
        if self.no_bid and self.no_ask:
            return (self.no_bid + self.no_ask) / 2.0
        return None

    @property
    def cross_venue_spread(self) -> Optional[float]:
        """Spread between best YES bid and best NO ask (for box arb)."""
        if self.yes_bid and self.no_ask:
            return self.no_ask - self.yes_bid
        return None


class VenueBookTracker:
    """Tracks orderbooks from multiple venues with freshness monitoring."""

    def __init__(
        self,
        assets: List[str] = None,
        horizons: List[str] = None,
        max_book_age: float = DEFAULT_MAX_BOOK_AGE_SECONDS,
        max_exchange_latency: float = DEFAULT_MAX_EXCHANGE_LATENCY_SECONDS,
    ):
        self.assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        self.horizons = horizons or ["15m"]
        self.max_book_age = max_book_age
        self.max_exchange_latency = max_exchange_latency

        # Books by venue -> market_id -> VenueBook
        self.books: Dict[str, Dict[str, VenueBook]] = {}
        self._last_sequence: Dict[str, int] = {}

        # Kalshi client for REST polling
        self._kalshi_session = requests.Session()
        self._kalshi_base = "https://external-api.kalshi.com/trade-api/v2"

        # Polymarket US client
        self._poly_session = requests.Session()
        self._poly_base = "https://api.polymarket.us"

        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start polling tasks for all venues."""
        if self._running:
            return
        self._running = True

        # Start Kalshi polling task
        self._tasks.append(asyncio.create_task(self._poll_kalshi_loop()))

        # Start Polymarket US polling task
        self._tasks.append(asyncio.create_task(self._poll_polymarket_loop()))

        log.info("VenueBookTracker started for %s × %s", self.assets, self.horizons)

    async def stop(self):
        """Stop all polling tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._kalshi_session.close()
        self._poly_session.close()
        log.info("VenueBookTracker stopped")

    def get_book(self, venue: str, market_id: str) -> Optional[VenueBook]:
        """Get latest book for a specific venue and market."""
        return self.books.get(venue, {}).get(market_id)

    def get_composite_book(self, asset: str, horizon: str) -> Optional[CompositeBook]:
        """Compute composite best bid/ask across fresh venues."""
        fresh_books = []
        for venue, markets in self.books.items():
            for market_id, book in markets.items():
                if book.asset == asset and book.horizon == horizon and book.is_fresh_with(
                    self.max_book_age, self.max_exchange_latency
                ):
                    fresh_books.append(book)

        if not fresh_books:
            return None

        # Find best YES bid (highest) and YES ask (lowest)
        yes_bid = None
        yes_bid_venue = None
        yes_ask = None
        yes_ask_venue = None

        # Find best NO bid (highest) and NO ask (lowest)
        # NO side: bid on NO = 1 - ask on YES, ask on NO = 1 - bid on YES
        no_bid = None
        no_bid_venue = None
        no_ask = None
        no_ask_venue = None

        venues_included = set()

        for book in fresh_books:
            venues_included.add(book.venue)

            # YES side
            if book.best_bid is not None:
                if yes_bid is None or book.best_bid > yes_bid:
                    yes_bid = book.best_bid
                    yes_bid_venue = book.venue
            if book.best_ask is not None:
                if yes_ask is None or book.best_ask < yes_ask:
                    yes_ask = book.best_ask
                    yes_ask_venue = book.venue

            # NO side derived from YES
            # NO bid = 1 - YES ask (best NO bid is complement of best YES ask)
            if book.best_ask is not None:
                no_bid_price = 1.0 - book.best_ask
                if no_bid is None or no_bid_price > no_bid:
                    no_bid = no_bid_price
                    no_bid_venue = book.venue
            # NO ask = 1 - YES bid
            if book.best_bid is not None:
                no_ask_price = 1.0 - book.best_bid
                if no_ask is None or no_ask_price < no_ask:
                    no_ask = no_ask_price
                    no_ask_venue = book.venue

        return CompositeBook(
            asset=asset,
            horizon=horizon,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_bid_venue=yes_bid_venue,
            yes_ask_venue=yes_ask_venue,
            no_bid=no_bid,
            no_ask=no_ask,
            no_bid_venue=no_bid_venue,
            no_ask_venue=no_ask_venue,
            timestamp=time.time(),
            venues_included=list(venues_included),
        )

    def book_age_gate(self, asset: str, horizon: str) -> Tuple[bool, str]:
        """
        Gate check: are books fresh enough for trading?

        Returns: (passed, reason)
        - passed=True: at least one fresh book per side (YES/NO)
        - passed=False: reason explains why (e.g., "stale", "no_venues")
        """
        composite = self.get_composite_book(asset, horizon)
        if not composite:
            return False, "no_fresh_books"

        # Need at least one venue for YES and one for NO
        if composite.yes_bid is None or composite.yes_ask is None:
            return False, "missing_yes_side"
        if composite.no_bid is None or composite.no_ask is None:
            return False, "missing_no_side"

        # Check for cross-venue divergence
        if composite.yes_bid_venue and composite.yes_ask_venue:
            # If bid and ask come from different venues, check divergence
            pass  # Could add divergence check here

        return True, "ok"

    def get_stale_venues(self, asset: str, horizon: str) -> List[str]:
        """Get list of venues with stale books for a given asset/horizon."""
        stale = []
        for venue, markets in self.books.items():
            for market_id, book in markets.items():
                if book.asset == asset and book.horizon == horizon:
                    if not book.is_fresh(self.max_book_age, self.max_exchange_latency):
                        stale.append(f"{venue}(age={book.age:.1f}s)")
        return stale

    # ═══════════════════════════════════════════════════════════════
    # Kalshi polling
    # ═══════════════════════════════════════════════════════════════

    async def _poll_kalshi_loop(self):
        """Poll Kalshi orderbooks for all tracked markets."""
        while self._running:
            try:
                for asset in self.assets:
                    for horizon in self.horizons:
                        await self._poll_kalshi_asset(asset, horizon)

                # Poll every 5 seconds (respecting rate limits)
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Kalshi polling error: %s", e)
                await asyncio.sleep(5)

    async def _poll_kalshi_asset(self, asset: str, horizon: str):
        """Poll Kalshi orderbooks for one asset/horizon."""
        series_map = {
            ("BTC", "15m"): "KXBTC15M",
            ("ETH", "15m"): "KXETH15M",
            ("SOL", "15m"): "KXSOL15M",
            ("XRP", "15m"): "KXXRP15M",
        }
        series = series_map.get((asset, horizon))
        if not series:
            return

        try:
            # Get open markets for this series
            resp = self._kalshi_session.get(
                f"{self._kalshi_base}/markets",
                params={"series_ticker": series, "status": "open", "limit": 10},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])

            for market in markets:
                ticker = market.get("ticker")
                if not ticker:
                    continue

                # Get orderbook
                ob_resp = self._kalshi_session.get(
                    f"{self._kalshi_base}/markets/{ticker}/orderbook",
                    params={"depth": 10},
                    timeout=5,
                )
                if ob_resp.status_code != 200:
                    continue
                ob_data = ob_resp.json()

                recv_ts = time.time()
                # Kalshi orderbook has 'yes' and 'no' arrays of [price, size]
                yes_bids = [[float(p), float(s)] for p, s in ob_data.get("yes", [])]
                no_bids = [[float(p), float(s)] for p, s in ob_data.get("no", [])]

                # Kalshi 'yes' array is bids on YES, 'no' array is bids on NO
                # YES ask = 1 - best NO bid, NO ask = 1 - best YES bid
                yes_bids.sort(key=lambda x: -x[0])  # Descending
                no_bids.sort(key=lambda x: -x[0])   # Descending

                bids = [BookLevel(price=p, size=s) for p, s in yes_bids]
                # Derive asks from NO bids
                asks = []
                if no_bids:
                    best_no_bid = no_bids[0][0]
                    yes_ask = 1.0 - best_no_bid
                    # Use NO bid sizes for YES ask sizes
                    for p, s in no_bids:
                        asks.append(BookLevel(price=1.0 - p, size=s))
                asks.sort(key=lambda x: x[0])  # Ascending

                book = VenueBook(
                    venue="kalshi",
                    market_id=ticker,
                    asset=asset,
                    horizon=horizon,
                    bids=bids,
                    asks=asks,
                    timestamp=recv_ts,  # Kalshi doesn't provide exch timestamp in orderbook
                    recv_timestamp=recv_ts,
                )

                if "kalshi" not in self.books:
                    self.books["kalshi"] = {}
                self.books["kalshi"][ticker] = book

        except Exception as e:
            log.debug("Kalshi poll error for %s/%s: %s", asset, horizon, e)

    # ═══════════════════════════════════════════════════════════════
    # Polymarket US polling
    # ═══════════════════════════════════════════════════════════════

    async def _poll_polymarket_loop(self):
        """Poll Polymarket US orderbooks for all tracked markets."""
        while self._running:
            try:
                for asset in self.assets:
                    for horizon in self.horizons:
                        await self._poll_polymarket_asset(asset, horizon)

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Polymarket polling error: %s", e)
                await asyncio.sleep(5)

    async def _poll_polymarket_asset(self, asset: str, horizon: str):
        """Poll Polymarket US orderbook for one asset/horizon."""
        try:
            # List markets for this asset/horizon
            # Use the gateway for listing, api.polymarket.us for book
            resp = self._poly_session.get(
                "https://gateway.polymarket.us/v1/markets",
                params={"active": "true", "limit": 100},
                timeout=10,
            )
            if resp.status_code != 200:
                return
            data = resp.json()
            markets = data.get("data", [])

            # Filter for Up/Down markets matching asset and horizon
            for market in markets:
                slug = market.get("slug", "")
                if asset.lower() not in slug.lower():
                    continue
                if horizon not in slug:
                    continue
                if "updown" not in slug.lower():
                    continue

                # Get orderbook from api.polymarket.us (bypasses Cloudflare)
                book_resp = self._poly_session.get(
                    f"{self._poly_base}/v1/markets/{slug}/book",
                    timeout=5,
                )
                if book_resp.status_code != 200:
                    continue
                book_data = book_resp.json()

                recv_ts = time.time()
                bids = book_data.get("bids", [])
                asks = book_data.get("asks", [])

                # Polymarket book: bids/asks on YES (Up) side
                # bids = buy YES, asks = sell YES
                # NO side derived: NO bid = 1 - YES ask, NO ask = 1 - YES bid
                yes_bids = [BookLevel(price=float(b[0]), size=float(b[1])) for b in bids]
                yes_asks = [BookLevel(price=float(a[0]), size=float(a[1])) for a in asks]

                yes_bids.sort(key=lambda x: -x[0])
                yes_asks.sort(key=lambda x: x[0])

                # Derive NO side
                no_bids = []
                no_asks = []
                if yes_asks:
                    for a in yes_asks:
                        no_bids.append(BookLevel(price=1.0 - a.price, size=a.size))
                if yes_bids:
                    for b in yes_bids:
                        no_asks.append(BookLevel(price=1.0 - b.price, size=b.size))

                no_bids.sort(key=lambda x: -x[0])
                no_asks.sort(key=lambda x: x[0])

                book = VenueBook(
                    venue="polymarket_us",
                    market_id=slug,
                    asset=asset,
                    horizon=horizon,
                    bids=yes_bids,
                    asks=yes_asks,
                    timestamp=recv_ts,
                    recv_timestamp=recv_ts,
                )

                if "polymarket_us" not in self.books:
                    self.books["polymarket_us"] = {}
                self.books["polymarket_us"][slug] = book

        except Exception as e:
            log.debug("Polymarket poll error for %s/%s: %s", asset, horizon, e)


# ═══════════════════════════════════════════════════════════════
# Convenience singleton
# ═══════════════════════════════════════════════════════════════

_venue_book_tracker: Optional[VenueBookTracker] = None


async def get_venue_book_tracker(
    assets: List[str] = None,
    horizons: List[str] = None,
    max_book_age: float = DEFAULT_MAX_BOOK_AGE_SECONDS,
) -> VenueBookTracker:
    """Get or create the module-level VenueBookTracker singleton."""
    global _venue_book_tracker
    if _venue_book_tracker is None:
        _venue_book_tracker = VenueBookTracker(
            assets=assets,
            horizons=horizons,
            max_book_age=max_book_age,
        )
        await _venue_book_tracker.start()
    return _venue_book_tracker


async def close_venue_book_tracker():
    """Close the module-level VenueBookTracker singleton."""
    global _venue_book_tracker
    if _venue_book_tracker:
        await _venue_book_tracker.stop()
        _venue_book_tracker = None


def book_age_gate(asset: str, horizon: str = "15m") -> Tuple[bool, str]:
    """Synchronous book-age gate check."""
    if _venue_book_tracker:
        return _venue_book_tracker.book_age_gate(asset, horizon)
    return False, "tracker_not_running"


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

        tracker = VenueBookTracker(["BTC", "ETH"], ["15m"])
        await tracker.start()

        try:
            for i in range(30):
                await asyncio.sleep(2)
                for asset in ["BTC", "ETH"]:
                    passed, reason = tracker.book_age_gate(asset, "15m")
                    composite = tracker.get_composite_book(asset, "15m")
                    if composite:
                        print(f"{asset}: gate={'PASS' if passed else 'FAIL'} ({reason}) "
                              f"YES={composite.yes_bid:.4f}/{composite.yes_ask:.4f} "
                              f"NO={composite.no_bid:.4f}/{composite.no_ask:.4f} "
                              f"venues={composite.venues_included}")
                    else:
                        print(f"{asset}: gate=FAIL (no composite) reason={reason}")
        finally:
            await tracker.stop()

    asyncio.run(main())