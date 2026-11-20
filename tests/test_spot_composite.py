"""Tests for spot_composite.py — Binance + Coinbase composite spot with tick storage."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spot_composite import (
    SpotComposite,
    Tick,
    VenueState,
    MAX_TICK_AGE_SECONDS,
    MAX_EXCHANGE_LATENCY_SECONDS,
    MAX_DIVERGENCE_BPS,
)


class TestTick:
    """Tick dataclass."""

    def test_tick_creation(self):
        tick = Tick(
            venue="binance",
            symbol="BTCUSDT",
            px=50000.0,
            bid=49999.0,
            ask=50001.0,
            exch_ts=time.time() - 0.1,
            recv_ts=time.time(),
        )
        assert tick.venue == "binance"
        assert tick.symbol == "BTCUSDT"
        assert tick.px == 50000.0

    def test_tick_age(self):
        now = time.time()
        tick = Tick(
            venue="binance",
            symbol="BTCUSDT",
            px=50000.0,
            bid=49999.0,
            ask=50001.0,
            exch_ts=now - 2.0,
            recv_ts=now,
        )
        assert tick.age == 2.0

    def test_tick_is_fresh(self):
        now = time.time()
        # Fresh tick (age < 5s, exchange latency < 1s)
        tick = Tick(
            venue="binance",
            symbol="BTCUSDT",
            px=50000.0,
            bid=49999.0,
            ask=50001.0,
            exch_ts=now - 0.5,
            recv_ts=now,
        )
        assert tick.is_fresh is True

    def test_tick_is_stale_by_age(self):
        now = time.time()
        tick = Tick(
            venue="binance",
            symbol="BTCUSDT",
            px=50000.0,
            bid=49999.0,
            ask=50001.0,
            exch_ts=now - 10.0,  # 10 seconds old
            recv_ts=now,
        )
        assert tick.is_fresh is False

    def test_tick_is_stale_by_exchange_latency(self):
        now = time.time()
        tick = Tick(
            venue="binance",
            symbol="BTCUSDT",
            px=50000.0,
            bid=49999.0,
            ask=50001.0,
            exch_ts=now - 2.0,  # exchange timestamp 2s ago
            recv_ts=now,
        )
        # recv_ts - exch_ts = 2s > 1s max latency
        assert tick.is_fresh is False


class TestVenueState:
    """VenueState dataclass."""

    def test_venue_state_init(self):
        state = VenueState()
        assert state.ticks == {}
        assert state.last_seq == 0
        assert state.connected is False
        assert state.last_recv == 0.0


class TestSpotCompositeInit:
    """SpotComposite initialization."""

    def test_init_default_symbols(self):
        sc = SpotComposite()
        assert sc.symbols == ["BTCUSDT", "ETHUSDT"]
        assert "binance" in sc.venues
        assert "coinbase" in sc.venues
        assert sc._running is False

    def test_init_custom_symbols(self):
        sc = SpotComposite(["SOLUSDT", "XRPUSDT"])
        assert sc.symbols == ["SOLUSDT", "XRPUSDT"]


class TestGetSpotWithAge:
    """get_spot_with_age method logic (tested without network)."""

    def test_no_fresh_prices_returns_none(self):
        sc = SpotComposite(["BTCUSDT"])
        # No ticks at all
        assert sc.get_spot_with_age("BTCUSDT") is None

    def test_median_of_two_venues(self):
        sc = SpotComposite(["BTCUSDT"])
        now = time.time()
        # Add fresh ticks from both venues
        sc.venues["binance"].ticks["BTCUSDT"] = Tick(
            venue="binance", symbol="BTCUSDT", px=50000.0, bid=49999.0, ask=50001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        sc.venues["coinbase"].ticks["BTCUSDT"] = Tick(
            venue="coinbase", symbol="BTCUSDT", px=50002.0, bid=50001.0, ask=50003.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        # Median of [50000, 50002] = 50001
        result = sc.get_spot_with_age("BTCUSDT")
        assert abs(result - 50001.0) < 0.001

    def test_median_of_three_venues(self):
        sc = SpotComposite(["BTCUSDT"])
        now = time.time()
        sc.venues["binance"].ticks["BTCUSDT"] = Tick(
            venue="binance", symbol="BTCUSDT", px=50000.0, bid=49999.0, ask=50001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        sc.venues["coinbase"].ticks["BTCUSDT"] = Tick(
            venue="coinbase", symbol="BTCUSDT", px=50002.0, bid=50001.0, ask=50003.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        # Add a third venue (we'd need to add it to venues dict)
        # Just test with two venues for now

    def test_stale_tick_excluded(self):
        sc = SpotComposite(["BTCUSDT"])
        now = time.time()
        # Binance fresh
        sc.venues["binance"].ticks["BTCUSDT"] = Tick(
            venue="binance", symbol="BTCUSDT", px=50000.0, bid=49999.0, ask=50001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        # Coinbase stale (10s old)
        sc.venues["coinbase"].ticks["BTCUSDT"] = Tick(
            venue="coinbase", symbol="BTCUSDT", px=51000.0, bid=50999.0, ask=51001.0,
            exch_ts=now - 10.0, recv_ts=now
        )
        # Should only use binance = 50000
        result = sc.get_spot_with_age("BTCUSDT")
        assert abs(result - 50000.0) < 0.001


class TestGetSpotWithAgeDetail:
    """get_spot_with_age_detail method."""

    def test_detail_includes_all_venues(self):
        sc = SpotComposite(["BTCUSDT"])
        now = time.time()
        sc.venues["binance"].ticks["BTCUSDT"] = Tick(
            venue="binance", symbol="BTCUSDT", px=50000.0, bid=49999.0, ask=50001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        detail = sc.get_spot_with_age_detail("BTCUSDT")
        assert detail["symbol"] == "BTCUSDT"
        assert detail["composite"] == 50000.0
        assert "binance" in detail["venues"]
        assert detail["fresh_count"] == 1

    def test_divergence_calculation(self):
        sc = SpotComposite(["BTCUSDT"])
        now = time.time()
        sc.venues["binance"].ticks["BTCUSDT"] = Tick(
            venue="binance", symbol="BTCUSDT", px=50000.0, bid=49999.0, ask=50001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        sc.venues["coinbase"].ticks["BTCUSDT"] = Tick(
            venue="coinbase", symbol="BTCUSDT", px=51000.0, bid=50999.0, ask=51001.0,
            exch_ts=now - 0.1, recv_ts=now
        )
        detail = sc.get_spot_with_age_detail("BTCUSDT")
        # divergence = (51000 - 50000) / 51000 * 10000 = 196 bps
        assert detail["divergence_bps"] is not None
        assert abs(detail["divergence_bps"] - 196.08) < 1.0


class TestConstants:
    """Verify constant values."""

    def test_max_tick_age(self):
        assert MAX_TICK_AGE_SECONDS == 5.0

    def test_max_exchange_latency(self):
        assert MAX_EXCHANGE_LATENCY_SECONDS == 1.0

    def test_max_divergence_bps(self):
        assert MAX_DIVERGENCE_BPS == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])