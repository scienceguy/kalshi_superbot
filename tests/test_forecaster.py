"""Tests for forecaster.py (T13-31)."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

import forecaster as fc


def _make_historical_markets(n=200):
    """Create synthetic historical market data."""
    np.random.seed(42)
    data = pd.DataFrame({
        "ticker": [f"KXBTC15M-{i:04d}" for i in range(n)],
        "outcome": np.random.binomial(1, 0.52, n),
        "gbm_q": np.random.beta(5, 5, n),
        "kl_edge": np.random.normal(0, 0.1, n),
        "kl_ladder": np.random.normal(0, 0.05, n),
        "kl_ensemble": np.random.normal(0, 0.05, n),
        "basis_bps_kalshi": np.random.normal(0, 5, n),
        "basis_bps_poly": np.random.normal(0, 5, n),
        "vol_regime": np.random.choice([0, 1, 2], n),
        "minutes_left": np.random.uniform(1, 14, n),
        "book_imbalance": np.random.uniform(-1, 1, n),
        "spread_bps": np.random.exponential(10, n),
        "fill_rate_recent": np.random.beta(5, 2, n),
        "atr_hurdle": np.random.exponential(5, n),
        "fee_adjusted_ev": np.random.normal(0, 2, n),
    })
    return data


def _make_signals_df(n=10):
    """Create synthetic signals DataFrame."""
    np.random.seed(42)
    data = pd.DataFrame({
        "ticker": [f"KXBTC15M-TEST-{i:04d}" for i in range(n)],
        "gbm_q": np.random.beta(5, 5, n),
        "kl_edge": np.random.normal(0, 0.1, n),
        "kl_ladder": np.random.normal(0, 0.05, n),
        "kl_ensemble": np.random.normal(0, 0.05, n),
        "basis_bps_kalshi": np.random.normal(0, 5, n),
        "basis_bps_poly": np.random.normal(0, 5, n),
        "vol_regime": np.random.choice([0, 1, 2], n),
        "minutes_left": np.random.uniform(1, 14, n),
        "book_imbalance": np.random.uniform(-1, 1, n),
        "spread_bps": np.random.exponential(10, n),
        "fill_rate_recent": np.random.beta(5, 2, n),
        "atr_hurdle": np.random.exponential(5, n),
        "fee_adjusted_ev": np.random.normal(0, 2, n),
    })
    return data.set_index("ticker")


def test_clustered_bootstrap_forecaster():
    """Test M1: Clustered Bootstrap forecaster."""
    hist_data = _make_historical_markets(100)
    forecaster = fc.ClusteredBootstrapForecaster(n_clusters=3, n_bootstrap=100)
    forecaster.fit(hist_data)
    
    assert len(forecaster.clusters) > 0
    assert len(forecaster.clusters) <= 3
    
    # Test prediction
    signals = _make_signals_df(10)
    result = forecaster.predict(signals)
    
    assert isinstance(result, fc.ForecastResult)
    assert result.model_name == "clustered_bootstrap"
    assert 0 <= result.p_yes <= 1
    assert 0 <= result.ci_lower <= result.ci_upper <= 1
    assert 0 <= result.confidence <= 1


def test_gbm_monte_carlo_forecaster():
    """Test M2: GBM Monte Carlo forecaster."""
    forecaster = fc.GBMMonteCarloForecaster(n_paths=1000)
    
    result = forecaster.predict(
        spot=100000,
        strike=100500,
        sigma_per_minute=0.0007,
        minutes_left=7.5,
        basis_bps=0,
        skew=0,
        kurtosis=3
    )
    
    assert isinstance(result, fc.ForecastResult)
    assert result.model_name == "gbm_monte_carlo"
    assert 0 <= result.p_yes <= 1
    assert 0 <= result.ci_lower <= result.ci_upper <= 1
    assert "n_paths" in result.metadata
    assert "spot" in result.metadata


def test_bayesian_beta_binomial_forecaster():
    """Test M3: Bayesian Beta-Binomial forecaster."""
    forecaster = fc.BayesianBetaBinomialForecaster()
    
    # Fit with outcomes
    outcomes = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0])
    forecaster.fit(outcomes)
    
    assert forecaster.posterior_alpha > forecaster.prior_alpha
    assert forecaster.posterior_beta > forecaster.prior_beta
    assert forecaster.n_observations == 10
    
    # Predict
    result = forecaster.predict(n_future=1)
    
    assert isinstance(result, fc.ForecastResult)
    assert result.model_name == "bayesian_beta_binomial"
    assert 0 <= result.p_yes <= 1
    assert 0 <= result.ci_lower <= result.ci_upper <= 1
    assert "posterior_alpha" in result.metadata


def test_ml_ensemble_forecaster():
    """Test M4: ML Ensemble forecaster."""
    forecaster = fc.MLEnsembleForecaster(model_type="xgboost", use_calibration=False)
    
    # Need training data - settled_df needs 'ticker' column, not index
    signals = _make_signals_df(100)
    settled = pd.DataFrame({
        "ticker": signals.index.tolist(),
        "outcome": np.random.binomial(1, 0.5, 100)
    })
    
    # Train
    metrics = forecaster.train(signals, settled)
    
    # Should have trained (if enough data)
    if metrics:
        assert forecaster.is_trained
        assert forecaster.model_manager.current_model is not None
    
    # Predict (even if not trained, should return default)
    test_signals = _make_signals_df(10)
    result = forecaster.predict(test_signals)
    
    assert isinstance(result, fc.ForecastResult)
    assert result.model_name == "ml_ensemble"


def test_ensemble_forecaster_integration():
    """Test full EnsembleForecaster with all models."""
    ensemble = fc.EnsembleForecaster(
        n_bootstrap=100,
        n_mc_paths=1000,
        ml_model_type="xgboost"
    )
    
    # Create historical data
    hist_data = _make_historical_markets(100)
    signals = _make_signals_df(50)
    settled = pd.DataFrame({
        "ticker": signals.index.tolist(),
        "outcome": np.random.binomial(1, 0.5, 50)
    })
    
    # Fit all models
    ensemble.fit(hist_data, signals, settled)
    
    # Forecast
    market_signals = _make_signals_df(10)
    result = ensemble.forecast(
        spot=100000,
        strike=100500,
        sigma_per_minute=0.0007,
        minutes_left=7.5,
        basis_bps=0,
        market_signals=market_signals
    )
    
    assert isinstance(result, fc.EnsembleForecast)
    assert 0 <= result.p_yes_ensemble <= 1
    assert 0 <= result.ci_lower <= result.ci_upper <= 1
    assert len(result.model_results) == 4
    assert result.cluster_lower <= result.cluster_upper


def test_ensemble_forecast_without_ml():
    """Test ensemble works when ML model not trained."""
    ensemble = fc.EnsembleForecaster(
        n_bootstrap=50,
        n_mc_paths=500
    )
    
    # Only fit bootstrap and bayesian
    hist_data = _make_historical_markets(50)
    ensemble.fit(hist_data)
    
    # Forecast without market_signals for ML
    result = ensemble.forecast(
        spot=100000,
        strike=100500,
        sigma_per_minute=0.0007,
        minutes_left=7.5
    )
    
    assert isinstance(result, fc.EnsembleForecast)
    assert len(result.model_results) == 4
    # ML should be skipped
    ml_result = [r for r in result.model_results if r.model_name == "ml_ensemble"][0]
    assert ml_result.confidence < 0.2  # Low confidence when skipped


def test_ensemble_weight_updates():
    """Test model weight updates from performance."""
    ensemble = fc.EnsembleForecaster()
    
    # Simulate performance history
    performance = [
        {"model": "clustered_bootstrap", "brier": 0.25},
        {"model": "gbm_monte_carlo", "brier": 0.22},
        {"model": "bayesian_beta_binomial", "brier": 0.24},
        {"model": "ml_ensemble", "brier": 0.20},
    ]
    
    ensemble.update_weights(performance)
    
    # Weights should be updated and normalized
    total = sum(ensemble.weights.values())
    assert abs(total - 1.0) < 0.01
    
    # Lower Brier = higher weight
    assert ensemble.weights["ml_ensemble"] > ensemble.weights["clustered_bootstrap"]


def test_forecaster_factory():
    """Test create_forecaster factory function."""
    forecaster = fc.create_forecaster({
        "n_bootstrap": 50,
        "n_mc_paths": 500,
        "ml_model_type": "lightgbm"
    })
    
    assert isinstance(forecaster, fc.EnsembleForecaster)
    assert forecaster.m1_bootstrap.n_bootstrap == 50
    assert forecaster.m2_gbm.n_paths == 500
    assert forecaster.m4_ml.model_type == "lightgbm"


def test_forecast_market_convenience():
    """Test forecast_market convenience function."""
    forecaster = fc.create_forecaster({"quick": True})
    hist_data = _make_historical_markets(50)
    signals = _make_signals_df(20)
    settled = pd.DataFrame({
        "ticker": signals.index.tolist(),
        "outcome": np.random.binomial(1, 0.5, 20)
    })
    forecaster.fit(hist_data, signals, settled)
    
    market_info = {
        "spot": 100000,
        "strike": 100500,
        "sigma_per_minute": 0.0007,
        "minutes_left": 7.5,
        "basis_bps": 10
    }
    
    result = fc.forecast_market(forecaster, market_info, signals[:5])
    
    assert isinstance(result, fc.EnsembleForecast)
    assert result.horizon_days == 0  # 7.5 min < 1 day


def test_forecast_store():
    """Test ForecastStore persistence."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        store = fc.ForecastStore(db_path)
        
        # Create a forecast
        ensemble = fc.EnsembleForecaster()
        hist_data = _make_historical_markets(20)
        ensemble.fit(hist_data)
        
        result = ensemble.forecast(
            spot=100000, strike=100500, sigma_per_minute=0.0007,
            minutes_left=7.5, market_signals=_make_signals_df(5)
        )
        
        # Save
        store.save_forecast("KXBTC15M-TEST", result)
        
        # Record outcome
        store.record_outcome("KXBTC15M-TEST", 1)
        
        # Get evaluation data
        eval_data = store.get_evaluation_data(days=30)
        assert len(eval_data) == 1
        assert eval_data.iloc[0]["outcome"] == 1
        assert eval_data.iloc[0]["ticker"] == "KXBTC15M-TEST"
    
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])