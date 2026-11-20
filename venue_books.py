#!/usr/bin/env python3
"""
venue_books.py — Venue books and book-age gate (T15-12).

Provides real-time orderbook tracking for:
- Kalshi: WS orderbook from event_hotpath (kalshi WS)
- Polymarket US: WS BBO from polymarket_us_client (or REST fallback)

Implements book-age gate: trading refuses on stale or one-sided books.
Exports book-age metrics for monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config

# Polymarket US client
import sys
sys.path.insert(0, '/home/ubuntu/kalshi_superbot/archive')
from polymarket_client import PolymarketUSClient

log = logging.getLogger(__name__)


@dataclass
class BookLevel:
    """Single level in an orderbook."""
    price: float  # in dollars
    size: int     # number of contracts


@dataclass
class VenueBook:
    """Orderbook snapshot for a venue."""
    venue: str  # "kalshi" or "polymarket"
    asset: str  # "BTC", "ETH", "SOL", "XRP"
    window_start: datetime
    yes_bid: Optional[float] = None  # dollars
    yes_ask: Optional[float] = None
    yes_bid_size: int = 0
    yes_ask_size: int = 0
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    no_bid_size: int = 0
    no_ask_size: int = 0
    timestamp: float = field(default_factory=time.time)  # Unix timestamp
    exchange_timestamp: Optional[float] = None  # From exchange if available
    
    def age_seconds(self) -> float:
        """Age of this book snapshot in seconds."""
        return time.time() - self.timestamp
    
    def exchange_latency_seconds(self) -> Optional[float]:
        """Exchange-to-receive latency if exchange_timestamp available."""
        if self.exchange_timestamp is not None:
            return self.timestamp - self.exchange_timestamp
        return None
    
    def is_stale(self, max_age_seconds: float = None) -> bool:
        """Check if book is stale."""
        if max_age_seconds is None:
            max_age_seconds = config.BOOK_MAX_AGE_SECONDS if hasattr(config, 'BOOK_MAX_AGE_SECONDS') else 2.0
        return self.age_seconds() > max_age_seconds
    
    def is_one_sided(self) -> bool:
        """Check if book is one-sided (missing bids or asks on either side)."""
        return (self.yes_bid is None or self.yes_ask is None or
                self.no_bid is None or self.no_ask is None)
    
    def is_tradable(self, max_age_seconds: float = None) -> bool:
        """Check if book is usable for trading (not stale, not one-sided)."""
        return not self.is_stale(max_age_seconds) and not self.is_one_sided()
    
    def mid_price(self, side: str = "yes") -> Optional[float]:
        """Get mid price for a side in cents."""
        if side == "yes":
            if self.yes_bid is not None and self.yes_ask is not None:
                return (self.yes_bid + self.yes_ask) / 2.0 * 100
        else:
            if self.no_bid is not None and self.no_ask is not None:
                return (self.no_bid + self.no_ask) / 2.0 * 100
        return None
    
    def spread_cents(self, side: str = "yes") -> Optional[float]:
        """Get spread in cents."""
        if side == "yes":
            if self.yes_bid is not None and self.yes_ask is not None:
                return (self.yes_ask - self.yes_bid) * 100
        else:
            if self.no_bid is not None and self.no_ask is not None:
                return (self.no_ask - self.no_bid) * 100
        return None


class KalshiBookTracker:
    """Tracks Kalshi orderbooks via WS (from event_hotpath)."""
    
    def __init__(self):
        self._books: Dict[str, VenueBook] = {}  # ticker -> VenueBook
        self._lock = asyncio.Lock()
    
    def update_from_ws(self, ticker: str, data: dict):
        """Update book from Kalshi WS message."""
        # Kalshi WS returns orderbook with yes/no bids/asks in cents
        # Format: {"yes": [[price_cents, size], ...], "no": [[price_cents, size], ...]}
        try:
            yes_bids = data.get("yes", [])
            no_bids = data.get("no", [])
            
            yes_bid = yes_bids[0][0] / 100.0 if yes_bids else None
            yes_bid_size = yes_bids[0][1] if yes_bids else 0
            yes_ask = yes_bids[0][0] / 100.0 if yes_bids else None  # Simplified - Kalshi WS may have separate asks
            yes_ask_size = 0
            no_bid = no_bids[0][0] / 100.0 if no_bids else None
            no_bid_size = no_bids[0][1] if no_bids else 0
            no_ask = no_bids[0][0] / 100.0 if no_bids else None
            no_ask_size = 0
            
            # Extract asset and window from ticker
            # Format: KXBTC15M-26NOV200445-45
            parts = ticker.split("-")
            asset = parts[0][2:-3] if parts[0].startswith("KX") and parts[0].endswith("15M") else "UNKNOWN"
            
            # Parse window start from ticker (approximate)
            # This is a simplification - in production, use the market data
            window_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            
            book = VenueBook(
                venue="kalshi",
                asset=asset,
                window_start=window_start,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                yes_bid_size=yes_bid_size,
                yes_ask_size=yes_ask_size,
                no_bid=no_bid,
                no_ask=no_ask,
                no_bid_size=no_bid_size,
                no_ask_size=no_ask_size,
                timestamp=time.time(),
            )
            
            self._books[ticker] = book
            
        except Exception as e:
            log.warning("Kalshi WS book update error for %s: %s", ticker, e)
    
    def get_book(self, ticker: str) -> Optional[VenueBook]:
        """Get current book for a ticker."""
        return self._books.get(ticker)
    
    def get_all_books(self) -> Dict[str, VenueBook]:
        """Get all current books."""
        return self._books.copy()


class PolymarketUSBookTracker:
    """Tracks Polymarket US BBO via REST (WS not available for US venue yet)."""
    
    def __init__(self):
        self._client = PolymarketUSClient()
        self._books: Dict[str, VenueBook] = {}  # slug -> VenueBook
        self._lock = asyncio.Lock()
        self._last_fetch: Dict[str, float] = {}
        self._fetch_interval = 1.0  # seconds
    
    async def update_book(self, asset: str, window_start: datetime) -> Optional[VenueBook]:
        """Fetch and update book for an asset/window."""
        now = time.time()
        
        # First get the market to find the correct slug
        market = self._client.get_updown_15m_market(asset, window_start)
        if not market:
            return None
        
        slug = market.get("slug")
        if not slug:
            return None
        
        # Rate limit
        if slug in self._last_fetch and now - self._last_fetch[slug] < self._fetch_interval:
            return self._books.get(slug)
        
        try:
            bbo = self._client.get_bbo(slug)
            if not bbo:
                return None
            
            # Parse BBO
            best_bid = bbo.get("bestBid")
            best_ask = bbo.get("bestAsk")
            
            yes_bid = float(best_bid.get("value", 0)) if best_bid else None
            yes_ask = float(best_ask.get("value", 0)) if best_ask else None
            yes_bid_size = int(best_bid.get("size", 0)) if best_bid else 0
            yes_ask_size = int(best_ask.get("size", 0)) if best_ask else 0
            
            # No side is complement
            no_bid = 1.0 - yes_ask if yes_ask is not None else None
            no_ask = 1.0 - yes_bid if yes_bid is not None else None
            no_bid_size = yes_ask_size
            no_ask_size = yes_bid_size
            
            book = VenueBook(
                venue="polymarket",
                asset=asset,
                window_start=window_start,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                yes_bid_size=yes_bid_size,
                yes_ask_size=yes_ask_size,
                no_bid=no_bid,
                no_ask=no_ask,
                no_bid_size=no_bid_size,
                no_ask_size=no_ask_size,
                timestamp=time.time(),
            )
            
            self._books[slug] = book
            self._last_fetch[slug] = now
            return book
            
        except Exception as e:
            log.warning("Polymarket US book update error for %s: %s", slug, e)
            return None
    
    def get_book(self, slug: str) -> Optional[VenueBook]:
        """Get current book for a slug."""
        return self._books.get(slug)
    
    def get_all_books(self) -> Dict[str, VenueBook]:
        """Get all current books."""
        return self._books.copy()


class VenueBookManager:
    """
    Unified manager for venue books with book-age gate.
    
    Provides:
    - Kalshi book tracking (WS)
    - Polymarket US book tracking (REST fallback)
    - Book-age gate: is_tradable() checks
    - Metrics export
    """
    
    def __init__(self):
        self.kalshi = KalshiBookTracker()
        self.polymarket = PolymarketUSBookTracker()
        self._metrics: Dict[str, Dict] = {}
    
    def update_kalshi_ws(self, ticker: str, data: dict):
        """Update Kalshi book from WS message."""
        self.kalshi.update_from_ws(ticker, data)
    
    async def update_polymarket(self, asset: str, window_start: datetime) -> Optional[VenueBook]:
        """Update Polymarket US book (REST)."""
        return await self.polymarket.update_book(asset, window_start)
    
    def get_kalshi_book(self, ticker: str) -> Optional[VenueBook]:
        """Get Kalshi book for a ticker."""
        return self.kalshi.get_book(ticker)
    
    def get_polymarket_book(self, slug: str) -> Optional[VenueBook]:
        """Get Polymarket US book for a slug."""
        return self.polymarket.get_book(slug)
    
    def get_book_for_asset(self, asset: str, window_start: datetime, venue: str = "kalshi") -> Optional[VenueBook]:
        """Get book for an asset/window from specified venue."""
        if venue == "kalshi":
            # Need to find ticker for asset/window
            # This is a simplification - in production, use market finder
            for ticker, book in self.kalshi.get_all_books().items():
                if book.asset == asset and book.window_start == window_start:
                    return book
            return None
        else:
            slug = PolymarketUSClient.window_event_slug(asset, window_start)
            return self.polymarket.get_book(slug)
    
    def check_tradable(self, book: Optional[VenueBook], max_age_seconds: float = None) -> bool:
        """Book-age gate: check if book is tradable."""
        if book is None:
            return False
        return book.is_tradable(max_age_seconds)
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Export book-age metrics for monitoring."""
        metrics = {}
        now = time.time()
        
        # Kalshi metrics
        for ticker, book in self.kalshi.get_all_books().items():
            metrics[f"kalshi_{ticker}"] = {
                "venue": "kalshi",
                "asset": book.asset,
                "age_seconds": book.age_seconds(),
                "exchange_latency": book.exchange_latency_seconds(),
                "is_stale": book.is_stale(),
                "is_one_sided": book.is_one_sided(),
                "is_tradable": book.is_tradable(),
                "yes_spread_cents": book.spread_cents("yes"),
                "no_spread_cents": book.spread_cents("no"),
            }
        
        # Polymarket metrics
        for slug, book in self.polymarket.get_all_books().items():
            metrics[f"polymarket_{slug}"] = {
                "venue": "polymarket",
                "asset": book.asset,
                "age_seconds": book.age_seconds(),
                "is_stale": book.is_stale(),
                "is_one_sided": book.is_one_sided(),
                "is_tradable": book.is_tradable(),
                "yes_spread_cents": book.spread_cents("yes"),
                "no_spread_cents": book.spread_cents("no"),
            }
        
        self._metrics = metrics
        return metrics
    
    def log_metrics(self):
        """Log current book metrics."""
        metrics = self.get_metrics()
        for key, m in metrics.items():
            log.info("Book metrics %s: age=%.2fs tradable=%s stale=%s onesided=%s spread_yes=%.2f spread_no=%.2f",
                     key, m["age_seconds"], m["is_tradable"], m["is_stale"], m["is_one_sided"],
                     m.get("yes_spread_cents", 0) or 0, m.get("no_spread_cents", 0) or 0)


# ─── Global singleton ──────────────────────────────────────────────────────────

_venue_book_manager: Optional[VenueBookManager] = None


def get_venue_book_manager() -> VenueBookManager:
    """Get or create the module-level VenueBookManager singleton."""
    global _venue_book_manager
    if _venue_book_manager is None:
        _venue_book_manager = VenueBookManager()
    return _venue_book_manager


# ─── Config defaults (add to config.py) ────────────────────────────────────────

# BOOK_MAX_AGE_SECONDS = 2.0  # Max age for entry decisions
# BOOK_MAX_EXCHANGE_LATENCY = 1.0  # Max exchange-to-receive latency


if __name__ == "__main__":
    # Quick test
    import asyncio
    
    async def test():
        manager = get_venue_book_manager()
        
        # Test Kalshi book update
        manager.update_kalshi_ws("KXBTC15M-26NOV200445-45", {
            "yes": [[7500, 100]],
            "no": [[2500, 100]]
        })
        
        book = manager.get_kalshi_book("KXBTC15M-26NOV200445-45")
        if book:
            print(f"Kalshi book: YES {book.yes_bid}/{book.yes_ask} NO {book.no_bid}/{book.no_ask}")
            print(f"Age: {book.age_seconds():.2f}s, Tradable: {book.is_tradable()}")
        
        # Test Polymarket book update
        now = datetime.now(timezone.utc)
        window = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        poly_book = await manager.update_polymarket("BTC", window)
        if poly_book:
            print(f"Polymarket book: YES {poly_book.yes_bid}/{poly_book.yes_ask} NO {poly_book.no_bid}/{poly_book.no_ask}")
            print(f"Age: {poly_book.age_seconds():.2f}s, Tradable: {poly_book.is_tradable()}")
        
        # Log metrics
        manager.log_metrics()
    
    asyncio.run(test())