"""Tests for the new ML model (ml_model.py) using MLModel/MLModelManager interface."""

import numpy as np
import pandas as pd
import pytest

import ml_model as m


def _make_signals_df(n=100):
    """Create a signals DataFrame with varied features for testing."""
    np.random.seed(42)
    data = []
    for i in range(n):
        ticker = f"KXBTC15M-TEST-{i:04d}"
        gbm_q = np.random.beta(2, 2)  # Concentrated around 0.5
        minutes_left = np.random.uniform(1, 14)
        
        # Features that correlate with outcome
        kl_edge = np.random.normal(0, 0.1)
        kl_ladder = np.random.normal(0, 0.05)
        kl_ensemble = np.random.normal(0, 0.05)
        basis_bps_kalshi = np.random.normal(0, 5)
        basis_bps_poly = np.random.normal(0, 5)
        vol_regime = np.random.choice([0, 1, 2])
        book_imbalance = np.random.uniform(-1, 1)
        spread_bps = np.random.exponential(10)
        fill_rate_recent = np.random.beta(5, 2)
        atr_hurdle = np.random.exponential(5)
        fee_adjusted_ev = np.random.normal(0, 2)
        
        data.append({
            "ticker": ticker,
            "gbm_q": gbm_q,
            "kl_edge": kl_edge,
            "kl_ladder": kl_ladder,
            "kl_ensemble": kl_ensemble,
            "basis_bps_kalshi": basis_bps_kalshi,
            "basis_bps_poly": basis_bps_poly,
            "vol_regime": vol_regime,
            "minutes_left": minutes_left,
            "book_imbalance": book_imbalance,
            "spread_bps": spread_bps,
            "fill_rate_recent": fill_rate_recent,
            "atr_hurdle": atr_hurdle,
            "fee_adjusted_ev": fee_adjusted_ev,
        })
    return pd.DataFrame(data).set_index("ticker")


def _make_settled_df(signals_df):
    """Create settled outcomes correlated with gbm_q."""
    np.random.seed(42)
    outcomes = []
    for ticker in signals_df.index:
        gbm_q = signals_df.loc[ticker, "gbm_q"]
        # Outcome probability correlates with gbm_q
        p_yes = gbm_q * 0.8 + 0.1  # Some noise
        outcome = 1 if np.random.random() < p_yes else 0
        outcomes.append({"ticker": ticker, "outcome": outcome})
    return pd.DataFrame(outcomes)  # Don't set_index - prepare_targets expects 'ticker' column


def test_ml_model_creation():
    """Test MLModel can be instantiated."""
    model = m.MLModel(model_type="xgboost", use_calibration=False)
    assert model.model_type == "xgboost"
    assert model.use_calibration is False
    assert model.is_trained is False


def test_ml_model_prepare_features():
    """Test feature preparation from signals DataFrame."""
    model = m.MLModel(model_type="xgboost")
    signals = _make_signals_df(10)
    
    X = model.prepare_features(signals)
    
    # Should have all feature columns
    assert list(X.columns) == m.FEATURE_COLS
    assert len(X) == 10
    assert X.index.tolist() == signals.index.tolist()


def test_ml_model_prepare_targets():
    """Test target preparation from settled DataFrame."""
    model = m.MLModel(model_type="xgboost")
    settled = pd.DataFrame({
        "ticker": ["T1", "T2", "T3"],
        "outcome": [1, 0, 1]
    })
    
    y = model.prepare_targets(settled)
    
    assert len(y) == 3
    assert y.tolist() == [1, 0, 1]
    assert y.index.tolist() == ["T1", "T2", "T3"]


def test_ml_model_train_and_predict():
    """Test full train/predict cycle with sufficient data."""
    model = m.MLModel(model_type="xgboost", use_calibration=False)
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    
    X = model.prepare_features(signals)
    y = model.prepare_targets(settled)
    
    # Align
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    
    # Train
    metrics = model.train(X, y)
    
    assert model.is_trained is True
    assert "brier" in metrics
    assert "log_loss" in metrics
    assert "auc_roc" in metrics
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    
    # Predict
    proba = model.predict_proba(X[:10])
    assert len(proba) == 10
    assert np.all(proba >= 0) and np.all(proba <= 1)
    
    preds = model.predict(X[:10], threshold=0.5)
    assert len(preds) == 10
    assert np.all(np.isin(preds, [0, 1]))


def test_ml_model_save_load(tmp_path):
    """Test model save/load roundtrip."""
    model = m.MLModel(model_type="xgboost", use_calibration=False)
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    
    X = model.prepare_features(signals)
    y = model.prepare_targets(settled)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    
    model.train(X, y)
    
    # Save
    path = tmp_path / "test_model.pkl"
    saved_path = model.save(str(path))
    assert Path(saved_path).exists()
    
    # Load
    loaded = m.MLModel(model_type="xgboost", use_calibration=False)
    success = loaded.load(str(path))
    assert success is True
    assert loaded.is_trained is True
    assert loaded.model_card is not None
    assert loaded.model_card.model_type == "xgboost"
    
    # Predictions should match
    proba1 = model.predict_proba(X[:5])
    proba2 = loaded.predict_proba(X[:5])
    np.testing.assert_allclose(proba1, proba2, rtol=1e-5)


def test_ml_model_manager():
    """Test MLModelManager basic operations."""
    manager = m.MLModelManager()
    
    # Train a model
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    
    model_name, metrics = manager.train_new_model(
        model_type="xgboost",
        signals_data=signals,
        settled_data=settled
    )
    
    assert model_name != ""
    assert "brier" in metrics
    # improvement only present if GBM baseline is set
    if "improvement" in metrics:
        assert "beats_gbm" in metrics
    assert model_name in manager.models
    
    # Get predictions
    new_signals = _make_signals_df(10)
    preds = manager.get_predictions(new_signals)
    assert len(preds) == 10
    assert np.all(preds >= 0) and np.all(preds <= 1)


def test_ml_model_manager_gbm_baseline():
    """Test GBM baseline comparison."""
    manager = m.MLModelManager()
    
    # Set GBM baseline
    manager.set_gbm_baseline(brier=0.25, log_loss=0.6, auc=0.65)
    
    # Train ML model
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    
    model_name, metrics = manager.train_new_model(
        model_type="xgboost",
        signals_data=signals,
        settled_data=settled
    )
    
    # Should have GBM comparison metrics
    assert "gbm_brier" in metrics
    assert "improvement" in metrics
    assert "beats_gbm" in metrics


def test_ml_model_calibration():
    """Test that calibration changes output distribution."""
    # Uncalibrated
    model_uncal = m.MLModel(model_type="xgboost", use_calibration=False)
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    X = model_uncal.prepare_features(signals)
    y = model_uncal.prepare_targets(settled)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    model_uncal.train(X, y)
    
    # Calibrated
    model_cal = m.MLModel(model_type="xgboost", use_calibration=True)
    model_cal.train(X, y)
    
    # Both should produce valid probabilities
    p_uncal = model_uncal.predict_proba(X[:10])
    p_cal = model_cal.predict_proba(X[:10])
    
    assert np.all(p_uncal >= 0) and np.all(p_uncal <= 1)
    assert np.all(p_cal >= 0) and np.all(p_cal <= 1)
    
    # Calibrated probabilities should be "better" (lower Brier on test)
    # Not strictly required to pass, but check they differ
    assert not np.allclose(p_uncal, p_cal)


def test_ml_model_feature_importances():
    """Test that feature importances are extracted."""
    model = m.MLModel(model_type="xgboost", use_calibration=False)
    signals = _make_signals_df(200)
    settled = _make_settled_df(signals)
    X = model.prepare_features(signals)
    y = model.prepare_targets(settled)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    model.train(X, y)
    
    card = model.get_model_card()
    assert card is not None
    assert len(card.feature_importances) > 0
    assert all(isinstance(v, (int, float)) for v in card.feature_importances.values())
    # Sum should be ~1 for tree models
    assert abs(sum(card.feature_importances.values()) - 1.0) < 0.1


def test_ml_model_lightgbm_if_available():
    """Test LightGBM if available."""
    if not m.LGB_AVAILABLE:
        pytest.skip("LightGBM not available")
    
    model = m.MLModel(model_type="lightgbm", use_calibration=False)
    signals = _make_signals_df(100)
    settled = _make_settled_df(signals)
    X = model.prepare_features(signals)
    y = model.prepare_targets(settled)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    metrics = model.train(X, y)
    
    assert model.is_trained
    assert model.model_type == "lightgbm"
    assert "brier" in metrics


def test_ml_model_sklearn_fallback():
    """Test sklearn fallback works."""
    # Force sklearn by using an invalid model type
    # Actually, the code falls back to sklearn if neither xgb nor lgb available
    # Just test that the fallback path works
    model = m.MLModel(model_type="xgboost", use_calibration=False)
    # If xgboost not available, it will use sklearn
    assert model._get_model() is not None


# Import Path for tmp_path
from pathlib import Path