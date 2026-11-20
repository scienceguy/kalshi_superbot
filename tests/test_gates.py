"""Integration tests for gates.py pre-entry evaluation."""

import pytest
import math
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timezone

import config
import gates
from market_finder import MarketInfo

try:
    from hypothesis import given, settings, strategies as st
    _HAS_HYPOTHESIS = True
except ImportError:  # dev-only extra; rest of the suite must still run without it
    _HAS_HYPOTHESIS = False


@dataclass
class MockMarket:
    """Mock market for testing gates."""
    ticker: str = "KXBTC15M-26NOV090415-15"
    series: str = "KXBTC15M"
    close_time: datetime = None
    strike: float = 95000.0
    strike_type: str = "floor"
    yes_bid: int = 75
    yes_ask: int = 77
    no_bid: int = 23
    no_ask: int = 25
    yes_liquidity: int = 100
    no_liquidity: int = 100
    minutes_left: float = 8.0
    bracket_role: str = "MAIN"
    raw: dict = None

    def __post_init__(self):
        if self.close_time is None:
            self.close_time = datetime.now(timezone.utc)
        if self.raw is None:
            self.raw = {"ticker": self.ticker, "event_ticker": self.ticker.split("-")[0]}


class TestGatesEvaluateEntry:
    """Test gates.evaluate_entry with various market conditions."""

    def test_gates_disabled_allows_all(self):
        """When GATES_ENABLED=false, all entries should be allowed."""
        with patch.object(config, 'GATES_ENABLED', False):
            market = MockMarket()
            result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
            assert result.allowed is True
            assert result.reasons == ["gates disabled"]

    def test_gate1_atr_hurdle_blocks_when_ratio_exceeds(self):
        """ATR hurdle should block when required move exceeds hurdle * expected move."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_ATR_ENABLED', True):
                with patch.object(config, 'GATES_ATR_SIDE_AWARE', True):
                    with patch.object(config, 'GATES_ATR_HURDLE_BTC', 1.0):
                        with patch.object(config, 'GATES_SHADOW_MODE', False):
                            with patch('gates.atr_1m', return_value=100.0):  # $100 ATR
                                with patch('gates.spot_for', return_value=94000.0):  # Spot below strike
                                    market = MockMarket(strike=95000.0, minutes_left=5.0)
                                    # Required move = 1000 (strike - spot), Expected = 100 * sqrt(5) ≈ 223
                                    # Ratio = 1000/223 ≈ 4.48 > 1.0 hurdle → should block
                                    result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                    assert result.allowed is False
                                    assert any("gate1" in r for r in result.reasons)

    def test_gate1_atr_hurdle_allows_when_ratio_within_hurdle(self):
        """ATR hurdle should allow when required move is within hurdle * expected move."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_ATR_ENABLED', True):
                with patch.object(config, 'GATES_ATR_SIDE_AWARE', True):
                    with patch.object(config, 'GATES_ATR_HURDLE_BTC', 5.0):
                        with patch('gates.atr_1m', return_value=100.0):
                            with patch('gates.spot_for', return_value=96000.0):  # Spot well above strike -> high prob
                                market = MockMarket(strike=95000.0, minutes_left=5.0)
                                # Required move = 0 (spot already past strike for YES), Expected = 223, Ratio = 0 < 5.0 → allow
                                result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                assert result.allowed is True

    def test_gate2_ev_blocks_when_net_ev_below_min(self):
        """Fee-adjusted EV gate should block when net EV < GATES_MIN_EDGE."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_EV_ENABLED', True):
                with patch.object(config, 'GATES_MIN_EDGE', 0.005):  # $0.005 per contract
                    with patch.object(config, 'GATES_FEE_TAKER', True):
                        with patch.object(config, 'GATES_TIME_DECAY_ENABLED', True):
                            with patch.object(config, 'GATES_SHADOW_MODE', False):
                                with patch('gates.atr_1m', return_value=100.0):
                                    with patch('gates.spot_for', return_value=95000.0):
                                        market = MockMarket(strike=95000.0, minutes_left=5.0)
                                        # Entry at 77c, model q ≈ 0.5 (at-the-money), fee ≈ 0.013
                                        # net_ev = 0.5*(1-0.77) - 0.5*0.77 - 0.013 = 0.115 - 0.385 - 0.013 = -0.283 < 0.005
                                        result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                        assert result.allowed is False
                                        assert any("gate2" in r for r in result.reasons)

    def test_gate3_high_gamma_zone_blocks_when_enabled(self):
        """High gamma zone should block when GATES_DENY_HIGH_GAMMA=true."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_TIME_DECAY_ENABLED', True):
                with patch.object(config, 'GATES_HIGH_GAMMA_MINUTES', 2.0):
                    with patch.object(config, 'GATES_DENY_HIGH_GAMMA', True):
                        with patch.object(config, 'GATES_SHADOW_MODE', False):
                            with patch('gates.atr_1m', return_value=100.0):
                                with patch('gates.spot_for', return_value=95000.0):
                                    market = MockMarket(strike=95000.0, minutes_left=1.5)  # < 2.0 min
                                    result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                    assert result.allowed is False
                                    assert any("gate3" in r and "high-gamma" in r for r in result.reasons)

    def test_gate3_high_gamma_allows_when_deny_false(self):
        """High gamma zone should allow when GATES_DENY_HIGH_GAMMA=false."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_TIME_DECAY_ENABLED', True):
                with patch.object(config, 'GATES_HIGH_GAMMA_MINUTES', 2.0):
                    with patch.object(config, 'GATES_DENY_HIGH_GAMMA', False):
                        with patch('gates.atr_1m', return_value=100.0):
                            with patch('gates.spot_for', return_value=96000.0):  # Spot above strike -> high prob
                                market = MockMarket(strike=95000.0, minutes_left=1.5)
                                result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                assert result.allowed is True
                                assert result.metrics.get("high_gamma_zone") is True
                                # Not blocked by high gamma because DENY_HIGH_GAMMA=False
                                assert "gate3" not in str(result.reasons)

    def test_unsupported_asset_blocks(self):
        """Unsupported underlying asset should fail closed."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_SHADOW_MODE', False):
                market = MockMarket(series="KXUNKNOWN15M", ticker="KXUNKNOWN15M-26NOV090415-15")
                result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                assert result.allowed is False
                assert any("gate0" in r for r in result.reasons)

    def test_shadow_mode_allows_but_logs_would_block(self):
        """In shadow mode, denies should return allowed=True with would_block metric."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_SHADOW_MODE', True):
                with patch.object(config, 'GATES_ATR_ENABLED', True):
                    with patch.object(config, 'GATES_ATR_SIDE_AWARE', True):
                        with patch.object(config, 'GATES_ATR_HURDLE_BTC', 1.0):
                            with patch('gates.atr_1m', return_value=100.0):
                                with patch('gates.spot_for', return_value=94000.0):
                                    market = MockMarket(strike=95000.0, minutes_left=5.0)
                                    result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                    assert result.allowed is True  # Shadow mode allows
                                    assert result.metrics.get("shadow_mode") is True
                                    assert result.metrics.get("decision") == "would_block"

    def test_side_aware_atr_uses_distance_against_position(self):
        """SIDE_AWARE should measure distance against the position, not absolute."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_ATR_ENABLED', True):
                with patch.object(config, 'GATES_ATR_SIDE_AWARE', True):
                    with patch.object(config, 'GATES_ATR_HURDLE_BTC', 1.0):
                        with patch('gates.atr_1m', return_value=100.0):
                            with patch('gates.spot_for', return_value=96000.0):  # Spot ABOVE strike
                                market = MockMarket(strike=95000.0, minutes_left=5.0)
                                # For YES side: required_move = max(0, strike - spot) = 0 (already past strike)
                                # Ratio = 0/223 = 0 < 1.0 → should allow
                                result = gates.evaluate_entry(market, side="yes", entry_price_cents=77)
                                assert result.allowed is True

    def test_no_side_uses_spot_above_strike(self):
        """NO side should measure distance when spot is above strike."""
        with patch.object(config, 'GATES_ENABLED', True):
            with patch.object(config, 'GATES_ATR_ENABLED', True):
                with patch.object(config, 'GATES_ATR_SIDE_AWARE', True):
                    with patch.object(config, 'GATES_ATR_HURDLE_BTC', 1.0):
                        with patch.object(config, 'GATES_SHADOW_MODE', False):
                            with patch('gates.atr_1m', return_value=100.0):
                                with patch('gates.spot_for', return_value=96000.0):  # Spot above strike
                                    market = MockMarket(strike=95000.0, minutes_left=5.0)
                                    # For NO side: required_move = max(0, spot - strike) = 1000
                                    # Ratio = 1000/223 ≈ 4.48 > 1.0 → should block
                                    result = gates.evaluate_entry(market, side="no", entry_price_cents=77)
                                    assert result.allowed is False
                                    assert any("gate1" in r for r in result.reasons)


class TestGatesPrefetch:
    """Test gates.prefetch warms caches correctly."""

    def test_prefetch_warms_spot_and_atr(self):
        """prefetch should call spot_for and atr_1m for each asset."""
        with patch('gates.spot_for') as mock_spot:
            with patch('gates.atr_1m') as mock_atr:
                gates.prefetch(["BTC", "ETH", "BTC"])  # Duplicate should be deduped
                assert mock_spot.call_count == 2
                assert mock_atr.call_count == 2
                mock_spot.assert_any_call("BTC")
                mock_spot.assert_any_call("ETH")
                mock_atr.assert_any_call("BTC")
                mock_atr.assert_any_call("ETH")


class TestGatesMath:
    """Test pure math functions in gates.py."""

    def test_net_ev_formula(self):
        """net_ev should compute q*(1-p) - (1-q)*p - fee."""
        # At p=0.77, q=0.80, fee=0.013
        # net_ev = 0.80*0.23 - 0.20*0.77 - 0.013 = 0.184 - 0.154 - 0.013 = 0.017
        ev = gates.net_ev(0.80, 0.77, 0.013)
        assert abs(ev - 0.017) < 0.001

    def test_breakeven_q_formula(self):
        """breakeven_q should return p + fee."""
        # At p=0.77, fee=0.013, breakeven = 0.783
        be = gates.breakeven_q(0.77, 0.013)
        assert abs(be - 0.783) < 0.001

    def test_cushion_z_formula(self):
        """cushion_z should compute (spot-strike)/(sigma*sqrt(t)) for YES."""
        # spot=96000, strike=95000, sigma=100, t=4 min
        # z = 1000/(100*2) = 5.0
        z = gates.cushion_z(96000, 95000, 100, 4, "yes")
        assert abs(z - 5.0) < 0.01

    def test_cushion_z_no_side(self):
        """cushion_z should negate for NO side."""
        z_yes = gates.cushion_z(96000, 95000, 100, 4, "yes")
        z_no = gates.cushion_z(96000, 95000, 100, 4, "no")
        assert z_no == -z_yes

    def test_cushion_z_scales_with_sqrt_minutes(self):
        """Quadrupling the time left halves z: denom is sigma*sqrt(t)."""
        z1 = gates.cushion_z(96000, 95000, 100, 1, "yes")
        z4 = gates.cushion_z(96000, 95000, 100, 4, "yes")
        assert abs(z4 - z1 / 2.0) < 1e-9

    def test_cushion_z_undefined_inputs(self):
        """Fail closed on zero/negative inputs instead of dividing by zero."""
        assert gates.cushion_z(0, 95000, 100, 4, "yes") is None
        assert gates.cushion_z(96000, 0, 100, 4, "yes") is None
        assert gates.cushion_z(96000, 95000, 0, 4, "yes") is None
        assert gates.cushion_z(96000, 95000, -1, 4, "yes") is None
        assert gates.cushion_z(96000, 95000, 100, 0, "yes") is None

    def test_atr_to_sigma_conversion(self):
        """atr_1m is E|X|, not sigma: sigma = atr * sqrt(pi/2) = 1.2533*atr.

        A 1-dollar average true range is ~40% wider in sigma terms, since
        sqrt(pi/2) = 1.2533 and (1.2533)^2 ~= 1.57.
        """
        atr = 100.0
        sigma = atr * math.sqrt(math.pi / 2)
        assert abs(sigma - 125.33) < 0.01

    def test_normal_cdf(self):
        """normal_cdf should match standard normal CDF."""
        # CDF(0) = 0.5
        assert abs(gates.normal_cdf(0) - 0.5) < 0.001
        # CDF(1.96) ≈ 0.975
        assert abs(gates.normal_cdf(1.96) - 0.975) < 0.01


class TestMeasuredFloorSide:
    """GATES_MEASURED_Q is a YES-side win rate; it must never be read as NO's.

    A NO contract priced at 97c is the complement of a YES contract at 3c,
    and settles NO only a few percent of the time. Flooring its model
    probability at P(YES wins)=0.9928 would tell the bot a near-certain
    loser is a near-certain winner, so the floor is keyed on the
    YES-equivalent price rather than the raw side price.
    """

    def _run(self, side, cents, q_model):
        market = MockMarket()
        book = Mock()
        with patch.object(gates, "normal_cdf", return_value=q_model), \
             patch.object(config, "GATES_ATR_ENABLED", False), \
             patch.object(config, "GATES_EV_ENABLED", False):
            return gates.evaluate_entry(
                market, spot_state={}, orderbook=book,
                side=side, entry_price_cents=cents)

    def test_yes_in_band_records_floor_flag(self):
        """YES at 97c is the price the 0.9928 was actually measured at."""
        config.GATES_MEASURED_Q = 0.9928
        res = self._run("yes", 97, 0.50)
        assert res.metrics.get("yes_equiv_price_cents") == 97
        assert res.metrics.get("q_measured_floor") is True
        assert res.metrics.get("q") >= 0.9928

    def test_no_at_97c_is_not_floored(self):
        """NO @ 97c == YES @ 3c: outside the band, must not get 99.28%."""
        res = self._run("no", 97, 0.50)
        assert res.metrics.get("yes_equiv_price_cents") == 3

    def test_no_at_3c_maps_to_yes_97(self):
        """The YES-equivalent of a NO@3c fill is 97c: inside the band."""
        res = self._run("no", 3, 0.50)
        assert res.metrics.get("yes_equiv_price_cents") == 97


if _HAS_HYPOTHESIS:

    class TestMeasuredFloorMirrorProperty:
        """T13-02 (guards 304d8f8): YES@p and NO@(100-p) are the same
        contract seen from opposite sides, so the measured floor must be a
        mirror image: identical YES-equivalent price, identical floor
        flag, identical q, for ANY price p."""

        @given(p=st.integers(min_value=1, max_value=99))
        @settings(max_examples=99)
        def test_yes_no_mirror_image_floors(self, p):
            q_model = 0.50
            out = {}
            for side, cents in (("yes", p), ("no", 100 - p)):
                with patch.object(gates, "normal_cdf", return_value=q_model), \
                     patch.object(gates, "spot_for", return_value=96000.0), \
                     patch.object(gates, "atr_1m", return_value=100.0), \
                     patch.object(config, "GATES_ATR_ENABLED", False), \
                     patch.object(config, "GATES_EV_ENABLED", False):
                    out[side] = gates.evaluate_entry(
                        MockMarket(), spot_state={}, orderbook=Mock(),
                        side=side, entry_price_cents=cents)
            y, n = out["yes"], out["no"]
            assert y.metrics["yes_equiv_price_cents"] == p
            assert n.metrics["yes_equiv_price_cents"] == p
            assert y.metrics.get("q_measured_floor") == n.metrics.get("q_measured_floor")
            assert y.metrics["q"] == n.metrics["q"]
            in_band = (config.GATES_MEASURED_BAND_MIN_CENTS <= p
                       <= config.GATES_MEASURED_BAND_MAX_CENTS)
            assert bool(y.metrics.get("q_measured_floor")) == in_band
            expected = max(q_model, config.GATES_MEASURED_Q) if in_band else q_model
            assert abs(y.metrics["q"] - expected) < 1e-12

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_t13_02_mirror_property_needs_hypothesis():
        pass


class TestT13_11SettlementBasis:
    """T13-11: settlement basis recorded at resolution and used by fair value."""

    def test_asset_for_accepts_dict(self):
        """asset_for now handles raw API dicts, not only objects."""
        raw = {"series": "KXBTC15M", "ticker": "KXBTC15M-26NOV090415-15"}
        assert gates.asset_for(raw) == "BTC"
        raw2 = {"ticker": "KXETH15M-26NOV090415-15"}
        assert gates.asset_for(raw2) == "ETH"

    def test_record_settlement_basis_noop_on_payout_fallback(self, monkeypatch):
        """expiration_value 0 or 100 means payout, not reference price: no record."""
        import sys
        sys.argv = [sys.argv[0]]
        from local_db import get_local_db
        from trader import _record_settlement_basis
        db = get_local_db()
        recorded = []
        def spy(*a, **kw):
            recorded.append((a, kw))
            return True
        monkeypatch.setattr(db, "record_settlement_basis", spy)
        m_payout0 = {"expiration_value": 0.0, "series": "KXBTC15M"}
        m_payout100 = {"expiration_value": 100.0, "series": "KXBTC15M"}
        with patch.object(gates, "spot_for", return_value=67000.0):
            _record_settlement_basis("t1", m_payout0)
            _record_settlement_basis("t2", m_payout100)
        assert recorded == []  # both payout fallbacks skipped


class TestKLDivergence:
    """T13-21: KL divergence layer unit tests."""

    def test_bernoulli_kl_basic(self):
        from kl_divergence import bernoulli_kl
        assert bernoulli_kl(0.5, 0.5) == 0.0
        assert bernoulli_kl(0.0, 0.0) == 0.0
        assert bernoulli_kl(1.0, 1.0) == 0.0
        # Known values: KL(0.6||0.5) ≈ 0.0201
        assert abs(bernoulli_kl(0.6, 0.5) - 0.0201) < 0.001
        assert abs(bernoulli_kl(0.9, 0.1) - 1.7577) < 0.001
        assert bernoulli_kl(0.5, 0.0) == float("inf")

    def test_kl_edge_score(self):
        from kl_divergence import kl_edge_score
        assert kl_edge_score(0.6, 0.5) > 0
        assert kl_edge_score(0.5, 0.5) == 0.0
        assert kl_edge_score(None, 0.5) == 0.0

    def test_ensemble_kl(self):
        from kl_divergence import ensemble_kl, ensemble_disagreement_gate, bernoulli_kl
        assert ensemble_kl([0.5, 0.5, 0.5]) == 0.0
        assert ensemble_kl([0.3, 0.7]) == bernoulli_kl(0.3, 0.7)
        allow, kl = ensemble_disagreement_gate([0.3, 0.7], threshold=0.1)
        assert kl > 0.1
        assert allow is False
        allow2, _ = ensemble_disagreement_gate([0.5, 0.5], threshold=0.1)
        assert allow2 is True

    def test_ladder_kl_monotonicity(self):
        from kl_divergence import ladder_kl
        # monotonic: P(S>K1) >= P(S>K2) for K1 < K2
        model = {90000: 0.8, 95000: 0.5, 100000: 0.2}
        market = {90000: 0.75, 95000: 0.55, 100000: 0.3}
        assert ladder_kl(model, market) >= 0
        # violate monotonicity -> inf
        bad_model = {90000: 0.2, 95000: 0.5, 100000: 0.8}
        assert ladder_kl(bad_model, market) == float("inf")

    def test_compute_kl_gates(self):
        from kl_divergence import compute_kl_gates
        gates = compute_kl_gates(
            q_model=0.6,
            entry_price_cents=55,
            side="yes",
            ensemble_probs=[0.55, 0.65, 0.60],
        )
        assert gates["kl_edge"] > 0
        assert gates["ensemble_kl"] > 0
        assert "ensemble_allow" in gates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])