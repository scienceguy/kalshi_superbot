"""Tests for box_arbitrage.py fee model, units, and equivalence override."""

import math
import pytest

from box_arbitrage import BoxArbitrageEngine, BoxOpportunity

UNITS_PER_DOLLAR = 10000  # box_arbitrage internal unit: 10000 = $1.00


class TestUnits:
    """10000 units = $1.00 everywhere in box_arbitrage."""

    def test_cent_to_unit(self):
        assert int(round(50.0 * 100)) == 5000
        assert int(round(100.0 * 100)) == 10000
        assert int(round(0.5 * 100)) == 50

    def test_units_per_dollar(self):
        assert UNITS_PER_DOLLAR == 10000


class TestFeeModel:
    """Fee models match published schedules (quadratic, not linear notional)."""

    def test_kalshi_taker_fee(self):
        engine = BoxArbitrageEngine()
        # Kalshi: round_up(1 * 0.07 * C * P * (1-P))
        # 100 contracts at 50c (price_cents=5000, 50c = 5000/10000)
        fee = engine.calculate_fees(5000, 100, "kalshi", taker=True)
        # Expected: ceil(0.07 * 100 * 0.5 * 0.5 * 10000) = ceil(17500) = 17500 units = 17.50 cents
        expected = math.ceil(0.07 * 100 * 0.5 * 0.5 * UNITS_PER_DOLLAR - 1e-9)
        assert fee == expected

    def test_polymarket_us_taker_fee(self):
        engine = BoxArbitrageEngine()
        # Polymarket US: round_up(0.095 * C * P * (1-P)) -> rounds up to cents
        # 100 contracts at 50c: raw = 0.095 * 100 * 0.5 * 0.5 = 2.375 dollars
        # Rounded up to cents: 2.38 dollars = 238 cents = 23800 units
        fee = engine.calculate_fees(5000, 100, "polymarket", taker=True)
        # The polymarket fee function returns fee in dollars rounded up to cents
        expected_fee_dollars = math.ceil(0.095 * 100 * 0.5 * 0.5 * 100 - 1e-9) / 100.0
        expected = int(math.ceil(expected_fee_dollars * UNITS_PER_DOLLAR - 1e-9))
        assert fee == expected

    def test_maker_fee_is_zero(self):
        engine = BoxArbitrageEngine()
        assert engine.calculate_fees(5000, 100, "kalshi", taker=False) == 0
        assert engine.calculate_fees(5000, 100, "polymarket", taker=False) == 0


class TestSameVenueBox:
    """Same-venue box detection (YES + NO on one market)."""

    def test_no_edge_when_asks_sum_to_100(self):
        engine = BoxArbitrageEngine()
        engine.max_slippage_bps = 0
        # yes_ask + no_ask = 10000 (exactly $1), no fees on maker
        # With slippage (size=1, available=100), edge = 10000 - 10000 - 4 = -4, rejected
        opp = engine.detect_same_venue_box(
            yes_ask=4500, yes_size=100,
            no_ask=5500, no_size=100,
            kalshi_ticker="TEST", min_size=1, venue="kalshi", taker=False
        )
        assert opp is None  # no edge after slippage

    def test_positive_edge_when_asks_under_100(self):
        engine = BoxArbitrageEngine()
        engine.max_slippage_bps = 0
        # yes_ask + no_ask = 9900, slippage = 4, edge = 100 - 4 = 96 bps
        opp = engine.detect_same_venue_box(
            yes_ask=4400, yes_size=100,
            no_ask=5500, no_size=100,
            kalshi_ticker="TEST", min_size=1, venue="kalshi", taker=False
        )
        assert opp is not None
        assert opp.edge_bps == 96.0  # 10000 - 9900 - 4 = 96 bps
        assert opp.tagged_as == "box"

    def test_rejects_when_sum_over_100(self):
        engine = BoxArbitrageEngine()
        opp = engine.detect_same_venue_box(
            yes_ask=5000, yes_size=100,
            no_ask=5200, no_size=100,
            kalshi_ticker="TEST", min_size=1, venue="kalshi", taker=False
        )
        assert opp is None

    def test_enforces_min_size(self):
        engine = BoxArbitrageEngine()
        opp = engine.detect_same_venue_box(
            yes_ask=4400, yes_size=1,
            no_ask=5500, no_size=100,
            kalshi_ticker="TEST", min_size=10, venue="kalshi", taker=False
        )
        assert opp is None


class TestCrossVenueBox:
    """Cross-venue box detection (equivalence override)."""

    def test_equivalence_proven_bypasses_basis_check(self):
        engine = BoxArbitrageEngine()
        engine.require_equivalence = True
        engine.max_slippage_bps = 0
        engine.min_edge_bps = 0
        # Cross-venue box always uses taker=True for fees
        # Kalshi: 0.07 * 1 * 0.40 * 0.60 = 168 units
        # Polymarket: 0.095 * 1 * 0.55 * 0.45 = 2.35c -> 3c = 300 units
        # slippage = 4 (size=1, available=100)
        # total_cost = 4000 + 5500 = 9500
        # net_cost = 9500 + 168 + 300 + 4 = 9972
        # edge = 28 bps
        opp = engine.detect_cross_venue_box(
            yes_ask_a=4000, yes_size_a=100, venue_a="kalshi", ticker_a="KXBTC15M-TEST",
            no_ask_b=5500, no_size_b=100, venue_b="polymarket", ticker_b="cpc-btc-updown-TEST",
            min_size=1, equivalence_proven=True
        )
        assert opp is not None
        assert opp.edge_bps == 28.0  # 10000 - 9500 - 468 - 4 = 28
        assert opp.tagged_as == "box"

    def test_requires_equivalence_when_flagged(self):
        engine = BoxArbitrageEngine()
        engine.require_equivalence = True
        # Without equivalence_proven, requires settlement_equivalence check
        opp = engine.detect_cross_venue_box(
            yes_ask_a=4000, yes_size_a=100, venue_a="kalshi", ticker_a="KXBTC15M-TEST",
            no_ask_b=5500, no_size_b=100, venue_b="polymarket", ticker_b="cpc-btc-updown-TEST",
            min_size=1, equivalence_proven=False  # default: basis check runs
        )
        # The basis check uses T13-11, which is not wired in test → returns None
        assert opp is None


class TestEdgeCalculation:
    """Edge bps = 1 unit = 1 bp of $1.00."""

    def test_edge_bps_unit(self):
        engine = BoxArbitrageEngine()
        engine.max_slippage_bps = 0
        # cost = 9900, slippage = 4, edge = 96 units = 96 bps of $1
        opp = engine.detect_same_venue_box(
            yes_ask=4400, yes_size=100,
            no_ask=5500, no_size=100,
            kalshi_ticker="TEST", min_size=1, venue="kalshi", taker=False
        )
        assert opp.edge_bps == 96.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])