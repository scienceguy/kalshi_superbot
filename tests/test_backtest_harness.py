"""Tests for backtest_harness.py (T13-30)."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

import backtest_harness as bh


def _make_market_snapshots(n=100):
    """Create synthetic market snapshots for testing."""
    np.random.seed(42)
    snapshots = []
    base_ts = 1000000000.0
    
    for i in range(n):
        ts = base_ts + i * 60  # 1-minute intervals
        mid = np.random.uniform(20, 80)
        spread = np.random.uniform(1, 10)
        
        snap = bh.MarketSnapshot(
            ticker=f"KXBTC15M-TEST-{i:04d}",
            timestamp=ts,
            yes_bid=int(mid - spread/2),
            yes_ask=int(mid + spread/2),
            no_bid=int(100 - mid - spread/2),
            no_ask=int(100 - mid + spread/2),
            yes_size=np.random.uniform(10, 100),
            no_size=np.random.uniform(10, 100),
            last_price=mid,
            volume=np.random.uniform(100, 1000),
            expiration_value=100 if np.random.random() > 0.5 else 0,
            floor_strike=100000
        )
        snapshots.append(snap)
    
    return snapshots


def _make_markets_dict(n_markets=5, snaps_per_market=20):
    """Create markets dict for testing."""
    markets = {}
    for m in range(n_markets):
        snaps = _make_market_snapshots(snaps_per_market)
        # Shift timestamps for each market
        for s in snaps:
            s.timestamp += m * 3600  # 1 hour apart
            s.ticker = f"KXBTC15M-M{m:02d}-SNAP"
        markets[f"KXBTC15M-M{m:02d}"] = snaps
    return markets


def test_backtest_params_creation():
    """Test BacktestParams can be created with defaults."""
    params = bh.BacktestParams()
    assert params.STRATEGY_MODE == "early_maker"
    assert params.ENTRY_WINDOW_START_MIN == 6.0
    assert params.ENTRY_WINDOW_END_MIN == 11.0
    assert params.PER_TRADE_PCT == 0.08


def test_backtest_params_hash():
    """Test config hash is deterministic."""
    params1 = bh.BacktestParams(ENTRY_BAND_MIN=20, ENTRY_BAND_MAX=60)
    params2 = bh.BacktestParams(ENTRY_BAND_MIN=20, ENTRY_BAND_MAX=60)
    params3 = bh.BacktestParams(ENTRY_BAND_MIN=25, ENTRY_BAND_MAX=60)
    
    assert params1.config_hash() == params2.config_hash()
    assert params1.config_hash() != params3.config_hash()


def test_backtest_params_to_from_dict():
    """Test serialization roundtrip."""
    params = bh.BacktestParams(
        ENTRY_WINDOW_START_MIN=8.0,
        ENTRY_BAND_MIN=30,
        STRATEGY_MODE="final_taker"
    )
    d = params.to_dict()
    params2 = bh.BacktestParams.from_dict(d)
    assert params.config_hash() == params2.config_hash()


def test_fill_model_maker():
    """Test maker fill probability model."""
    fm = bh.FillModel(base_fill_prob=0.8, toxicity_penalty=0.3, queue_decay=0.95)
    
    # Normal case
    p = fm.maker_fill_prob(price_cents=50, minutes_left=5.0, 
                           spread_bps=10, book_imbalance=0.0, queue_position=0.5)
    assert 0.05 <= p <= 1.0
    
    # Toxic band with 7+ min
    p_toxic = fm.maker_fill_prob(price_cents=30, minutes_left=8.0,
                                 spread_bps=10, book_imbalance=0.0, queue_position=0.5)
    # Should be penalized
    assert p_toxic < p


def test_fill_model_taker():
    """Test taker fill probability."""
    fm = bh.FillModel()
    
    # Full liquidity
    p = fm.taker_fill_prob(price_cents=75, size_contracts=1.0, available_liquidity=10.0)
    assert p == 1.0
    
    # Partial liquidity
    p = fm.taker_fill_prob(price_cents=75, size_contracts=5.0, available_liquidity=3.0)
    assert 0 < p < 1.0
    
    # No liquidity
    p = fm.taker_fill_prob(price_cents=75, size_contracts=5.0, available_liquidity=0.0)
    assert p == 0.0


def test_backtest_engine_early_maker():
    """Test early maker strategy simulation."""
    params = bh.BacktestParams(
        STRATEGY_MODE="early_maker",
        ENTRY_WINDOW_START_MIN=0.0,
        ENTRY_WINDOW_END_MIN=15.0,
        ENTRY_BAND_MIN=10,
        ENTRY_BAND_MAX=90,
        STARTING_BALANCE=100.0
    )
    engine = bh.BacktestEngine(params)
    
    # Create market with favorable conditions
    markets = _make_markets_dict(n_markets=3, snaps_per_market=15)
    
    # Settlement function - deterministic
    def settlement_fn(ticker):
        # YES wins for even-indexed markets
        idx = int(ticker.split("-M")[-1].split("-")[0]) if "-M" in ticker else 0
        return idx % 2 == 0
    
    results = engine.run_backtest(markets, settlement_fn)
    
    assert "ev_per_trade" in results
    assert "total_pnl" in results
    assert "n_trades" in results
    assert "win_rate" in results
    assert "max_drawdown" in results
    assert "params_hash" in results


def test_backtest_engine_final_taker():
    """Test final taker strategy simulation."""
    params = bh.BacktestParams(
        STRATEGY_MODE="final_taker",
        ENTRY_WINDOW_START_MIN=14.0,
        ENTRY_WINDOW_END_MIN=15.0,
        ENTRY_BAND_MIN=70,
        ENTRY_BAND_MAX=99,
        STARTING_BALANCE=100.0
    )
    engine = bh.BacktestEngine(params)
    
    markets = _make_markets_dict(n_markets=3, snaps_per_market=15)
    
    def settlement_fn(ticker):
        idx = int(ticker.split("-M")[-1].split("-")[0]) if "-M" in ticker else 0
        return idx % 2 == 0
    
    results = engine.run_backtest(markets, settlement_fn)
    
    assert results["n_trades"] >= 0


def test_parameter_grid():
    """Test parameter grid generation."""
    grid_quick = list(bh.parameter_grid(quick=True))
    # Test first 10 of full grid
    grid_full = list(bh.parameter_grid(quick=False))[:10]
    
    assert len(grid_quick) > 0
    assert len(grid_full) == 10
    
    # Validate each param set
    for params in grid_quick:
        assert params.ENTRY_WINDOW_START_MIN < params.ENTRY_WINDOW_END_MIN
        assert params.ENTRY_BAND_MIN < params.ENTRY_BAND_MAX
    for params in grid_full:
        assert params.ENTRY_WINDOW_START_MIN < params.ENTRY_WINDOW_END_MIN
        assert params.ENTRY_BAND_MIN < params.ENTRY_BAND_MAX


def test_walk_forward_validator():
    """Test walk-forward validation."""
    params = bh.BacktestParams(N_SPLITS=3, PURGE_DAYS=0, EMBARGO_DAYS=0)
    validator = bh.WalkForwardValidator(params)
    
    # Create time-sorted markets
    markets_by_time = []
    base_ts = 1000000000.0
    for day in range(10):
        markets = _make_markets_dict(n_markets=2, snaps_per_market=5)
        for m in markets:
            for snap in markets[m]:
                snap.timestamp = base_ts + day * 86400
        markets_by_time.append((base_ts + day * 86400, markets))
    
    def settlement_fn(ticker):
        return True
    
    results = validator.validate(markets_by_time, settlement_fn)
    
    assert len(results) > 0
    for r in results:
        assert r.split_idx >= 0
        assert "train_metrics" in r.__dict__
        assert "test_metrics" in r.__dict__
    
    summary = validator.get_summary()
    assert "n_splits" in summary
    assert "test_ev_mean" in summary


def test_market_snapshot_creation():
    """Test MarketSnapshot dataclass."""
    snap = bh.MarketSnapshot(
        ticker="KXBTC15M-TEST",
        timestamp=1000000000.0,
        yes_bid=50,
        yes_ask=52,
        no_bid=48,
        no_ask=50,
        yes_size=100.0,
        no_size=100.0,
        last_price=51.0,
        volume=500.0
    )
    
    assert snap.ticker == "KXBTC15M-TEST"
    assert snap.yes_bid == 50
    assert snap.yes_ask == 52


# Global settlement function for pickling
def _test_settlement_fn(ticker):
    return "M00" in ticker


def test_run_sweep_integration():
    """Test full sweep (small scale)."""
    markets = _make_markets_dict(n_markets=2, snaps_per_market=10)
    
    params_list = [
        bh.BacktestParams(STRATEGY_MODE="early_maker", ENTRY_BAND_MIN=10, ENTRY_BAND_MAX=90),
        bh.BacktestParams(STRATEGY_MODE="final_taker", ENTRY_BAND_MIN=70, ENTRY_BAND_MAX=99),
    ]
    
    results = bh.run_sweep(
        markets=markets,
        settlement_fn=_test_settlement_fn,
        params_list=params_list,
        jobs=1,  # Single process for test
        top=2
    )
    
    assert len(results) == 2
    assert "ev_per_trade" in results[0]
    assert "params" in results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])