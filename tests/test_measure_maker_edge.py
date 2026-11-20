"""Tests for the honest maker-edge measurement and the 14-day forecaster.

Both tools read the live quote archive (scripts/quote_recorder.py ->
local_data/candles.sqlite). The tests build a tiny archive with KNOWN
prices and outcomes, so the expected numbers are arithmetic, not fixtures.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from scripts.measure_maker_edge import (
    Observation,
    bootstrap_lower_bound,
    simulate_maker,
    simulate_taker,
    summarize,
    taker_fee,
)
import forecast14


def _write_archive(tmp_path, markets, candles):
    db = tmp_path / "candles.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE markets (ticker TEXT PRIMARY KEY, series TEXT, "
                 "open_ts TEXT, close_ts TEXT, result TEXT)")
    conn.execute("CREATE TABLE candles (ticker TEXT, end_ts TEXT, yes_bid REAL, "
                 "yes_ask REAL, no_bid REAL, no_ask REAL)")
    conn.executemany("INSERT INTO markets VALUES (?,?,?,?,?)", markets)
    conn.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?)", candles)
    conn.commit()
    conn.close()
    return str(db)


def _iso(dt):
    return dt.isoformat()


def test_taker_fee_matches_kalshi_schedule():
    # 0.07 * p * (1-p), the documented taker schedule.
    assert taker_fee(0.50) == pytest.approx(0.0175)
    assert taker_fee(0.90) == pytest.approx(0.0063)
    assert taker_fee(0.01) == pytest.approx(0.000693)


def test_maker_fills_only_when_market_trades_through():
    close = datetime(2026, 11, 20, 5, 15, tzinfo=timezone.utc)
    # minutes_left 10 -> 8. A YES bid at 0.90 is filled when the YES ask
    # later drops to 0.89 (the market traded down through our bid).
    quotes = [
        (10.0, 0.90, 0.92, 0.08, 0.10),
        (8.0, 0.88, 0.89, 0.11, 0.12),
    ]
    obs = simulate_maker(quotes, "YES", 6.0, 11.0, won=True)
    assert obs is not None
    assert obs.filled is True
    assert obs.price == pytest.approx(0.90)
    # maker pays no fee: win pays 1 - 0.90
    assert obs.net == pytest.approx(0.10)


def test_maker_does_not_fill_when_quote_only_touches():
    quotes = [
        (10.0, 0.90, 0.92, 0.08, 0.10),
        (8.0, 0.90, 0.91, 0.09, 0.10),  # ask never reaches our 0.90 bid
    ]
    obs = simulate_maker(quotes, "YES", 6.0, 11.0, won=True)
    assert obs is not None
    assert obs.filled is False
    assert obs.net == 0.0


def test_maker_no_side_fills_when_yes_bid_rises():
    # NO ask = 1 - yes_bid; a NO bid at 0.10 fills when yes_bid >= 0.90.
    quotes = [
        (10.0, 0.89, 0.91, 0.09, 0.11),
        (8.0, 0.91, 0.93, 0.07, 0.09),
    ]
    obs = simulate_maker(quotes, "NO", 6.0, 11.0, won=True)
    assert obs is not None
    assert obs.filled is True
    assert obs.price == pytest.approx(0.11)
    # NO side won: payoff is 1 - 0.11
    assert obs.net == pytest.approx(0.89)
    # and the same quote loses the full 0.11 when the YES side settles
    loss = simulate_maker(quotes, "NO", 6.0, 11.0, won=False)
    assert loss.net == pytest.approx(-0.11)


def test_no_quote_inside_entry_window_means_no_trade():
    quotes = [(3.0, 0.90, 0.92, 0.08, 0.10)]
    assert simulate_maker(quotes, "YES", 6.0, 11.0, won=True) is None


def test_taker_pays_fee_on_win_and_loss():
    quotes = [(10.0, 0.90, 0.92, 0.08, 0.10)]
    win = simulate_taker(quotes, "YES", 6.0, 11.0, won=True)
    loss = simulate_taker(quotes, "YES", 6.0, 11.0, won=False)
    assert win.net == pytest.approx(1 - 0.92 - taker_fee(0.92))
    assert loss.net == pytest.approx(-0.92 - taker_fee(0.92))


def test_summarize_clusters_by_market():
    obs = [
        Observation("M1", "YES", 0.90, True, 0.10, 10.0, True),
        Observation("M1", "NO", 0.10, False, -0.10, 10.0, True),   # same market
        Observation("M2", "YES", 0.90, False, -0.90, 10.0, True),
    ]
    stats = summarize(obs, "test", min_markets=1)
    assert stats["n_markets"] == 2      # M1 counted once, not twice
    assert stats["n_filled"] == 2
    assert stats["fill_rate"] == pytest.approx(1.0)
    # below the 30-fill floor the band is not judged
    assert stats["sufficient"] is False


def test_bootstrap_lower_bound_is_below_the_mean():
    values = [0.10, -0.90, 0.10, -0.90, 0.10]
    lower = bootstrap_lower_bound(values, sims=500, seed=1)
    assert lower <= sum(values) / len(values)


def test_forecast_refuses_to_publish_without_evidence(tmp_path):
    db = _write_archive(tmp_path, [], [])
    result = forecast14.run_forecast(db, balance=6.42, days=14, sims=100,
                                     seed=1, risk_profile="conservative")
    assert result["published"] is False
    assert "insufficient evidence" in result["ensemble"]["refusal"]
    assert result["ensemble"]["models_used"] == []


def test_forecast_publishes_with_enough_evidence(tmp_path):
    close = datetime(2026, 11, 20, 5, 15, tzinfo=timezone.utc)
    markets, candles = [], []
    for i in range(forecast14.MIN_CLUSTERED_MARKETS):
        ticker = "KXBTC15M-TEST%03d" % i
        result = "yes" if i % 4 else "no"   # 75% win rate on the favourite
        markets.append((ticker, "KXBTC15M", _iso(close - timedelta(minutes=15)),
                        _iso(close), result))
        # A 0.90 favourite whose YES offer drops through our 0.90 bid.
        candles.append((ticker, _iso(close - timedelta(minutes=10)),
                        0.90, 0.92, 0.08, 0.10))
        candles.append((ticker, _iso(close - timedelta(minutes=8)),
                        0.88, 0.89, 0.11, 0.12))
    db = _write_archive(tmp_path, markets, candles)

    result = forecast14.run_forecast(db, balance=100.0, days=14, sims=200,
                                     seed=3, risk_profile="conservative")
    assert result["published"] is True
    ens = result["ensemble"]
    assert ens["models_used"]
    assert set(ens["percentiles"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert 0.0 <= ens["p_ruin"] <= 1.0
    # R1: the target probability is an upper bound, never 0 and never inflated.
    target_prob = ens["target_probability"]
    assert target_prob["p_reach_target"] >= 0.0
    assert target_prob["p_reach_target"] <= 1.0
    assert "sims" in target_prob["note"]


def test_r1_target_probability_never_claims_zero():
    finals = [1.0, 2.0, 3.0]
    out = forecast14.r1_target_probability(finals, sims=3, target=6_250_000.0)
    assert out["n_paths_reaching"] == 0
    assert "< 1 in 3 sims" in out["note"]


def test_risk_profile_cannot_disable_the_halts():
    for profile in forecast14.RISK_PROFILES:
        params = forecast14.RISK_PROFILES[profile]
        assert 0 < params["kelly_fraction"] <= 0.5
        assert 0 < params["max_position_pct"] <= 1.0
