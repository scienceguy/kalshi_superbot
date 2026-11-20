"""Tests for polymarket_us_client.py fee model, quote parsing, and market listing."""

import pytest

from polymarket_us_client import (
    taker_theta,
    taker_fee_dollars,
    maker_fee_dollars,
    UpDownMarket,
    Book,
    PolymarketUSClient,
)


class TestFeeModel:
    """Polymarket US fee schedule: taker = Theta * C * P * (1-P), maker = 0 (Tier 1)."""

    def test_taker_theta_tiers(self):
        assert taker_theta(0) == 0.095
        assert taker_theta(9999) == 0.095
        assert taker_theta(10000) == 0.085
        assert taker_theta(50000) == 0.075
        assert taker_theta(250000) == 0.0695
        assert taker_theta(1_000_000) == 0.0695

    def test_taker_fee_quadratic(self):
        # 100 contracts at 50c with Tier 1 Theta=0.095
        fee = taker_fee_dollars(100, 0.5, 0.095)
        # ceil(0.095 * 100 * 0.5 * 0.5 * 100) = ceil(237.5) = 238 cents = $2.38
        assert fee == 2.38

    def test_taker_fee_scales_with_price(self):
        # Same contracts, different prices
        fee_50 = taker_fee_dollars(100, 0.5, 0.095)
        fee_10 = taker_fee_dollars(100, 0.1, 0.095)
        fee_90 = taker_fee_dollars(100, 0.9, 0.095)
        # P*(1-P) is symmetric and max at 0.5
        assert fee_10 == fee_90
        assert fee_50 > fee_10

    def test_maker_fee_is_zero(self):
        assert maker_fee_dollars(100, 0.5) == 0.0
        assert maker_fee_dollars(1000, 0.1) == 0.0


class TestBookQuotes:
    """Book/BBO -> executable YES-side quotes with NO derived as complement."""

    def test_book_to_quotes(self):
        b = Book(
            slug="test",
            yes_bid=0.45,
            yes_ask=0.55,
            yes_bid_size=100.0,
            yes_ask_size=200.0,
        )
        # Buy YES at ask, buy NO at 1 - bid
        assert b.yes_ask == 0.55
        assert abs(b.no_ask - 0.55) < 1e-10  # 1 - 0.45
        assert abs(b.no_bid - 0.45) < 1e-10  # 1 - 0.55
        assert b.yes_ask_size == 200.0
        assert b.no_ask_size == 100.0  # bid queue is NO ask depth

    def test_partial_book(self):
        b = Book(slug="test", yes_bid=0.40, yes_ask=None, yes_bid_size=50.0)
        assert b.no_ask == 0.60  # 1 - 0.40
        assert b.no_bid is None  # no ask -> no bid

    def test_sides_sum_to_one(self):
        """YES ask + NO ask = YES ask + (1 - YES bid) >= 1 - spread."""
        b = Book(slug="test", yes_bid=0.48, yes_ask=0.52)
        assert b.yes_ask + b.no_ask == 1.04  # cross-spread adds up to > 1


class TestUpDownMarket:
    """Market listing fields from the live API."""

    def test_asset_symbol(self):
        m = UpDownMarket(slug="t", asset="btc", horizon="15m", index_symbol="BRTI",
                         window_start="s", window_end="e")
        assert m.asset_symbol == "BTC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])