"""Tests for venue_books.py — Venue orderbook tracking and book-age gate."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from venue_books import (
    BookLevel,
    VenueBook,
    CompositeBook,
    VenueBookTracker,
    DEFAULT_MAX_BOOK_AGE_SECONDS,
    DEFAULT_MAX_EXCHANGE_LATENCY_SECONDS,
)

# Remove async tests since pytest-asyncio not available


class TestBookLevel:
    """BookLevel dataclass."""

    def test_creation(self):
        level = BookLevel(price=100.0, size=10.0)
        assert level.price == 100.0
        assert level.size == 10.0


class TestVenueBook:
    """VenueBook dataclass and properties."""

    def test_best_bid_ask(self):
        book = VenueBook(
            venue="test",
            market_id="mkt1",
            asset="BTC",
            horizon="15m",
            bids=[BookLevel(99.0, 10), BookLevel(98.0, 5)],
            asks=[BookLevel(101.0, 10), BookLevel(102.0, 5)],
            timestamp=time.time(),
            recv_timestamp=time.time(),
        )
        assert book.best_bid == 99.0
        assert book.best_ask == 101.0
        assert book.best_bid_size == 10
        assert book.best_ask_size == 10

    def test_mid_price(self):
        book = VenueBook(
            venue="test",
            market_id="mkt1",
            asset="BTC",
            horizon="15m",
            bids=[BookLevel(99.0, 10)],
            asks=[BookLevel(101.0, 10)],
            timestamp=time.time(),
            recv_timestamp=time.time(),
        )
        assert book.mid_price == 100.0

    def test_spread(self):
        book = VenueBook(
            venue="test",
            market_id="mkt1",
            asset="BTC",
            horizon="15m",
            bids=[BookLevel(99.0, 10)],
            asks=[BookLevel(101.0, 10)],
            timestamp=time.time(),
            recv_timestamp=time.time(),
        )
        assert book.spread == 2.0

    def test_age_and_freshness(self):
        now = time.time()
        book = VenueBook(
            venue="test",
            market_id="mkt1",
            asset="BTC",
            horizon="15m",
            bids=[BookLevel(99.0, 10)],
            asks=[BookLevel(101.0, 10)],
            timestamp=now - 1.0,  # 1 second ago
            recv_timestamp=now,
        )
        assert book.age == 1.0
        assert book.is_fresh_with(max_age=2.0, max_latency=1.0) is True
        assert book.is_fresh_with(max_age=0.5, max_latency=1.0) is False

    def test_stale_book(self):
        now = time.time()
        book = VenueBook(
            venue="test",
            market_id="mkt1",
            asset="BTC",
            horizon="15m",
            bids=[BookLevel(99.0, 10)],
            asks=[BookLevel(101.0, 10)],
            timestamp=now - 5.0,  # 5 seconds ago
            recv_timestamp=now,
        )
        assert book.age == 5.0
        assert book.is_fresh_with(max_age=2.0) is False


class TestCompositeBook:
    """CompositeBook dataclass."""

    def test_yes_no_mid(self):
        comp = CompositeBook(
            asset="BTC",
            horizon="15m",
            yes_bid=0.95,
            yes_ask=0.97,
            no_bid=0.03,
            no_ask=0.05,
        )
        assert comp.yes_mid == 0.96
        assert comp.no_mid == 0.04

    def test_cross_venue_spread(self):
        # cross_venue_spread = no_ask - yes_bid
        comp = CompositeBook(
            asset="BTC",
            horizon="15m",
            yes_bid=0.95,
            no_ask=0.06,
        )
        assert abs(comp.cross_venue_spread - (-0.89)) < 0.001

        # For a valid box: yes_bid + no_ask < 1
        comp2 = CompositeBook(
            asset="BTC",
            horizon="15m",
            yes_bid=0.45,
            no_ask=0.52,
        )
        assert abs(comp2.cross_venue_spread - 0.07) < 0.001


class TestVenueBookTracker:
    """VenueBookTracker functionality (mocked)."""

    def test_init(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        assert tracker.assets == ["BTC"]
        assert tracker.horizons == ["15m"]
        assert tracker.max_book_age == DEFAULT_MAX_BOOK_AGE_SECONDS
        assert tracker._running is False

    def test_get_book_empty(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        assert tracker.get_book("kalshi", "nonexistent") is None

    def test_book_age_gate_no_books(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        passed, reason = tracker.book_age_gate("BTC", "15m")
        assert passed is False
        assert reason == "no_fresh_books"

    def test_get_composite_book_empty(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        assert tracker.get_composite_book("BTC", "15m") is None

    def test_get_stale_venues_empty(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        assert tracker.get_stale_venues("BTC", "15m") == []


class TestBookAgeGateLogic:
    """Test book-age gate logic with simulated books."""

    def test_gate_passes_with_fresh_books(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        now = time.time()

        # Add fresh Kalshi book
        tracker.books["kalshi"] = {
            "KXBTC15M-123": VenueBook(
                venue="kalshi",
                market_id="KXBTC15M-123",
                asset="BTC",
                horizon="15m",
                bids=[BookLevel(0.95, 100)],
                asks=[BookLevel(0.97, 100)],
                timestamp=now,
                recv_timestamp=now,
            )
        }

        # Add fresh Polymarket book
        tracker.books["polymarket_us"] = {
            "cpc-btc-updown-15m-123": VenueBook(
                venue="polymarket_us",
                market_id="cpc-btc-updown-15m-123",
                asset="BTC",
                horizon="15m",
                bids=[BookLevel(0.94, 100)],
                asks=[BookLevel(0.96, 100)],
                timestamp=now,
                recv_timestamp=now,
            )
        }

        passed, reason = tracker.book_age_gate("BTC", "15m")
        assert passed is True
        assert reason == "ok"

    def test_gate_fails_with_stale_books(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        now = time.time()

        # Add stale Kalshi book (5 seconds old)
        tracker.books["kalshi"] = {
            "KXBTC15M-123": VenueBook(
                venue="kalshi",
                market_id="KXBTC15M-123",
                asset="BTC",
                horizon="15m",
                bids=[BookLevel(0.95, 100)],
                asks=[BookLevel(0.97, 100)],
                timestamp=now - 5.0,
                recv_timestamp=now,
            )
        }

        passed, reason = tracker.book_age_gate("BTC", "15m")
        assert passed is False
        assert reason == "no_fresh_books"

    def test_gate_fails_missing_yes_side(self):
        tracker = VenueBookTracker(["BTC"], ["15m"])
        now = time.time()

        # Only NO side (no YES bids/asks)
        tracker.books["kalshi"] = {
            "KXBTC15M-123": VenueBook(
                venue="kalshi",
                market_id="KXBTC15M-123",
                asset="BTC",
                horizon="15m",
                bids=[],  # Empty YES bids
                asks=[BookLevel(0.97, 100)],
                timestamp=now,
                recv_timestamp=now,
            )
        }

        passed, reason = tracker.book_age_gate("BTC", "15m")
        assert passed is False
        assert reason == "missing_yes_side"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])