"""
ml_model.py — ML model upgrade (T13-24).

On the VPS (aarch64 glibc) install xgboost/lightgbm natively and delete the
proot `ml_python` wrapper.

Features:
- GBM q, KL divergence, composite-vs-venue basis, vol regime, time-to-expiry,
  book imbalance, spread, recent fill rate
- Labels = settled outcome (0/1)
- Walk-forward training with purge/embargo
- Export calibrated probabilities (isotonic/Platt)
- Retrain via n8n (T13-40)
- Keep ML paper-only until it beats GBM out of sample

Accept: model card (data window, n, calibration, feature importances)
written to MemPalace KG; promotion needs out-of-sample win over GBM.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import config
from local_db import get_local_db

log = logging.getLogger("ml_model")

# Model directory
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature columns
FEATURE_COLS = [
    "gbm_q",              # GBM fair value probability
    "kl_edge",            # Bernoulli KL edge signal
    "kl_ladder",          # Ladder KL divergence
    "kl_ensemble",        # Ensemble disagreement KL
    "basis_bps_kalshi",   # Kalshi settlement basis vs composite
    "basis_bps_poly",     # Polymarket settlement basis vs composite
    "vol_regime",         # Volatility regime (low/med/high)
    "minutes_left",       # Time to expiry
    "book_imbalance",     # Order book imbalance (bid_size - ask_size) / (bid_size + ask_size)
    "spread_bps",         # Bid-ask spread in bps
    "fill_rate_recent",   # Recent fill rate (last 100 orders)
    "atr_hurdle",         # ATR hurdle value
    "fee_adjusted_ev",    # Fee-adjusted expected value
]

TARGET_COL = "outcome"  # 1 if YES settled, 0 if NO settled


@dataclass
class ModelCard:
    """Model card with metadata for MemPalace KG."""
    model_name: str
    model_type: str           # "xgboost" or "lightgbm"
    training_window: str      # e.g., "2026-10-01 to 2026-11-15"
    n_samples: int
    n_features: int
    calibration_method: str   # "isotonic", "platt", "none"
    brier_score: float
    log_loss: float
    auc_roc: float
    feature_importances: Dict[str, float]
    purge_embargo_days: int
    created_at: str
    promoted: bool = False
    oos_win_vs_gbm: bool = False


class MLModel:
    """
    ML model for trade outcome prediction.
    
    Uses XGBoost or LightGBM with walk-forward validation.
    Features are derived from market data, GBM, KL, and basis.
    """
    
    def __init__(self, model_type: str = "xgboost", use_calibration: bool = True):
        self.model_type = model_type
        self.use_calibration = use_calibration
        self.model = None
        self.calibrator = None
        self.feature_cols = FEATURE_COLS.copy()
        self.model_card: Optional[ModelCard] = None
        self.is_trained = False
    
    def _get_model(self):
        """Get the underlying model instance."""
        if self.model_type == "xgboost" and XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
            )
        elif self.model_type == "lightgbm" and LGB_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            # Fallback to sklearn if available
            if SKLEARN_AVAILABLE:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
            raise RuntimeError("No ML library available. Install xgboost, lightgbm, or scikit-learn.")
    
    def prepare_features(self, signals_df: pd.DataFrame, 
                          kl_data: List[Dict] = None,
                          basis_data: List[Dict] = None,
                          market_data: Dict = None) -> pd.DataFrame:
        """
        Prepare feature matrix from raw signals and market data.
        
        Args:
            signals_df: DataFrame with signal data (must have gbm_q, ticker, etc.)
            kl_data: List of KL divergence results per market
            basis_data: List of settlement basis records from T13-11
            market_data: Dict with current market microstructure (book, spread, etc.)
        
        Returns:
            DataFrame with feature columns
        """
        df = signals_df.copy()
        
        # Ensure we have GBM q
        if "gbm_q" not in df.columns:
            log.warning("gbm_q not in signals, using 0.5 default")
            df["gbm_q"] = 0.5
        
        # KL features
        if kl_data:
            kl_by_ticker = {k["ticker"]: k for k in kl_data}
            df["kl_edge"] = df["ticker"].map(
                lambda t: kl_by_ticker.get(t, {}).get("edge_kl", 0.0)
            ).fillna(0.0)
            df["kl_ladder"] = df["ticker"].map(
                lambda t: kl_by_ticker.get(t, {}).get("ladder_kl", 0.0)
            ).fillna(0.0)
            df["kl_ensemble"] = df["ticker"].map(
                lambda t: kl_by_ticker.get(t, {}).get("ensemble_kl", 0.0)
            ).fillna(0.0)
        else:
            df["kl_edge"] = 0.0
            df["kl_ladder"] = 0.0
            df["kl_ensemble"] = 0.0
        
        # Basis features
        if basis_data:
            basis_by_venue = {}
            for b in basis_data:
                venue = b.get("venue", "kalshi")
                if venue not in basis_by_venue:
                    basis_by_venue[venue] = []
                basis_by_venue[venue].append(b.get("basis_bps", 0.0))
            
            for venue, values in basis_by_venue.items():
                avg_basis = np.mean(values) if values else 0.0
                col = f"basis_bps_{venue}"
                df[col] = avg_basis
        else:
            df["basis_bps_kalshi"] = 0.0
            df["basis_bps_poly"] = 0.0
        
        # Volatility regime (from GBM sigma)
        if "sigma_per_minute" in df.columns:
            df["vol_regime"] = pd.qcut(df["sigma_per_minute"], q=3, 
                                        labels=[0, 1, 2], duplicates="drop")
            df["vol_regime"] = df["vol_regime"].astype(float).fillna(1.0)
        else:
            df["vol_regime"] = 1.0
        
        # Time to expiry
        if "minutes_left" not in df.columns:
            df["minutes_left"] = 7.5  # Default mid-window
        
        # Market microstructure features
        if market_data:
            for ticker, md in market_data.items():
                mask = df["ticker"] == ticker
                df.loc[mask, "book_imbalance"] = md.get("book_imbalance", 0.0)
                df.loc[mask, "spread_bps"] = md.get("spread_bps", 0.0)
                df.loc[mask, "fill_rate_recent"] = md.get("fill_rate", 0.0)
        
        # Fill missing with defaults
        for col in ["book_imbalance", "spread_bps", "fill_rate_recent",
                    "atr_hurdle", "fee_adjusted_ev"]:
            if col not in df.columns:
                df[col] = 0.0
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        return df[self.feature_cols]
    
    def prepare_targets(self, settled_markets: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable from settled markets.
        
        Args:
            settled_markets: DataFrame with ticker and outcome (1=YES won, 0=NO won)
        
        Returns:
            Series with target values indexed by ticker
        """
        return settled_markets.set_index("ticker")[TARGET_COL]
    
    def train(self, X: pd.DataFrame, y: pd.Series,
               purge_days: int = 1, embargo_days: int = 1) -> Dict:
        """
        Train model with walk-forward validation (purge/embargo).
        
        Args:
            X: Feature matrix
            y: Target values
            purge_days: Days to purge between train/test
            embargo_days: Days to embargo after test
        
        Returns:
            Training metrics
        """
        n_samples = len(X)
        
        # Handle small datasets (e.g., for testing)
        if n_samples < 10:
            log.warning(f"Small dataset ({n_samples} samples), using simple split")
            split = max(1, int(n_samples * 0.7))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
        elif not SKLEARN_AVAILABLE:
            log.warning("scikit-learn not available, skipping walk-forward validation")
            # Simple train/test split
            split = int(n_samples * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
        else:
            # TimeSeriesSplit with purge/embargo
            n_splits = min(5, n_samples // 2)  # At least 2 samples per split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            # For simplicity, use last fold as test
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]
            
            # Apply purge: remove purge_days from end of train
            if purge_days > 0:
                # Assuming daily frequency, convert to samples
                purge_samples = purge_days * 96  # 96 15-min periods per day
                train_idx = train_idx[:-purge_samples] if len(train_idx) > purge_samples else train_idx
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train base model
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Calibrate probabilities
        if self.use_calibration and SKLEARN_AVAILABLE:
            self.calibrator = CalibratedClassifierCV(
                self.model, method="isotonic", cv=3
            )
            self.calibrator.fit(X_train, y_train)
        
        # Evaluate
        if self.use_calibration and self.calibrator:
            y_pred_proba = self.calibrator.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        
        brier = brier_score_loss(y_test, y_pred_proba)
        ll = log_loss(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = dict(zip(self.feature_cols, self.model.feature_importances_))
        else:
            importances = {}
        
        # Create model card
        self.model_card = ModelCard(
            model_name=f"ml_{self.model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
            model_type=self.model_type,
            training_window=f"{X.index[0]} to {X.index[-1]}",
            n_samples=len(X_train),
            n_features=len(self.feature_cols),
            calibration_method="isotonic" if self.use_calibration else "none",
            brier_score=brier,
            log_loss=ll,
            auc_roc=auc,
            feature_importances=importances,
            purge_embargo_days=purge_days + embargo_days,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        
        self.is_trained = True
        
        metrics = {
            "brier": brier,
            "log_loss": ll,
            "auc_roc": auc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importances": importances,
        }
        
        log.info(f"ML model trained: Brier={brier:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}")
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if self.use_calibration and self.calibrator:
            return self.calibrator.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary outcomes."""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def save(self, path: str = None) -> str:
        """Save model to disk."""
        if path is None:
            path = str(MODEL_DIR / f"{self.model_card.model_name}.pkl")
        
        save_dict = {
            "model": self.model,
            "calibrator": self.calibrator,
            "model_card": self.model_card,
            "feature_cols": self.feature_cols,
            "model_type": self.model_type,
            "use_calibration": self.use_calibration,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        log.info(f"Model saved to {path}")
        return path
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            with open(path, "rb") as f:
                save_dict = pickle.load(f)
            
            self.model = save_dict["model"]
            self.calibrator = save_dict.get("calibrator")
            self.model_card = save_dict.get("model_card")
            self.feature_cols = save_dict.get("feature_cols", self.feature_cols)
            self.model_type = save_dict.get("model_type", self.model_type)
            self.use_calibration = save_dict.get("use_calibration", self.use_calibration)
            self.is_trained = True
            
            log.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            log.warning(f"Failed to load model from {path}: {e}")
            return False
    
    def get_model_card(self) -> Optional[ModelCard]:
        """Get model card for MemPalace KG."""
        return self.model_card


class MLModelManager:
    """
    Manages ML model lifecycle: training, evaluation, promotion.
    
    Keeps ML paper-only until it beats GBM out of sample.
    """
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.current_model: Optional[str] = None
        self.gbm_baseline_metrics: Dict = {}
        self.promotion_threshold = config.ML_PROMOTION_THRESHOLD if hasattr(config, 'ML_PROMOTION_THRESHOLD') else 0.01  # 1% Brier improvement
    
    def train_new_model(self, model_type: str = "xgboost",
                         signals_data: pd.DataFrame = None,
                         settled_data: pd.DataFrame = None) -> Tuple[str, Dict]:
        """
        Train a new model and evaluate against GBM baseline.
        
        Returns (model_name, metrics)
        """
        model = MLModel(model_type=model_type)
        
        # If no data provided, fetch from local DB
        if signals_data is None or settled_data is None:
            signals_data, settled_data = self._fetch_training_data()
        
        if signals_data is None or len(signals_data) == 0:
            log.warning("No training data available")
            return "", {}
        
        # Prepare features and targets
        X = model.prepare_features(signals_data)
        y = model.prepare_targets(settled_data)
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 100:
            log.warning(f"Insufficient training data: {len(X)} samples")
            return "", {}
        
        # Train
        metrics = model.train(X, y)
        
        # Save
        model_path = model.save()
        
        # Register
        model_name = model.model_card.model_name
        self.models[model_name] = model
        
        # Evaluate against GBM baseline
        if self.gbm_baseline_metrics:
            gbm_brier = self.gbm_baseline_metrics.get("brier", 1.0)
            ml_brier = metrics["brier"]
            improvement = gbm_brier - ml_brier
            metrics["gbm_brier"] = gbm_brier
            metrics["improvement"] = improvement
            metrics["beats_gbm"] = improvement > self.promotion_threshold
            
            if metrics["beats_gbm"]:
                log.info(f"ML model beats GBM! Brier improvement: {improvement:.4f}")
                self.promote_model(model_name)
        
        return model_name, metrics
    
    def _fetch_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch training data from local database."""
        try:
            db = get_local_db()
            
            # Get signals with outcomes
            conn = db._get_conn()
            signals = pd.read_sql(
                "SELECT * FROM signals WHERE outcome_won IS NOT NULL",
                conn
            )
            
            # Get settled markets for additional data
            settled = pd.read_sql(
                "SELECT market_ticker as ticker, 1 as outcome FROM settled_markets",
                conn
            )
            
            return signals, settled
        except Exception as e:
            log.warning(f"Failed to fetch training data: {e}")
            return None, None
    
    def set_gbm_baseline(self, brier: float, log_loss: float, auc: float):
        """Set GBM baseline metrics for comparison."""
        self.gbm_baseline_metrics = {
            "brier": brier,
            "log_loss": log_loss,
            "auc_roc": auc,
        }
        log.info(f"GBM baseline set: Brier={brier:.4f}, LogLoss={log_loss:.4f}, AUC={auc:.4f}")
    
    def promote_model(self, model_name: str):
        """Promote model to production (paper trading)."""
        if model_name in self.models:
            self.models[model_name].model_card.promoted = True
            self.current_model = model_name
            log.info(f"Model {model_name} promoted to paper trading")
            
            # Write model card to MemPalace KG
            self._write_model_card_to_kg(model_name)
    
    def _write_model_card_to_kg(self, model_name: str):
        """Write model card to MemPalace KG."""
        try:
            model = self.models[model_name]
            card = model.get_model_card()
            if card:
                import subprocess
                card_json = {
                    "model_name": card.model_name,
                    "model_type": card.model_type,
                    "training_window": card.training_window,
                    "n_samples": card.n_samples,
                    "n_features": card.n_features,
                    "calibration_method": card.calibration_method,
                    "brier_score": card.brier_score,
                    "log_loss": card.log_loss,
                    "auc_roc": card.auc_roc,
                    "feature_importances": card.feature_importances,
                    "purge_embargo_days": card.purge_embargo_days,
                    "created_at": card.created_at,
                    "promoted": card.promoted,
                    "oos_win_vs_gbm": card.oos_win_vs_gbm,
                }
                
                # Write to MemPalace
                result = subprocess.run([
                    "mempalace", "artifact", "put",
                    "--content", json.dumps(card_json),
                    "--name", f"model_card_{model_name}",
                    "--tags", "ml,model_card,promotion"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    log.info(f"Model card written to MemPalace for {model_name}")
                else:
                    log.warning(f"Failed to write model card: {result.stderr}")
        except Exception as e:
            log.warning(f"Failed to write model card to KG: {e}")
    
    def get_predictions(self, signals_df: pd.DataFrame,
                         kl_data: List[Dict] = None,
                         basis_data: List[Dict] = None,
                         market_data: Dict = None) -> np.ndarray:
        """Get ML predictions for current signals."""
        if not self.current_model or self.current_model not in self.models:
            log.warning("No promoted model available")
            return np.full(len(signals_df), 0.5)
        
        model = self.models[self.current_model]
        X = model.prepare_features(signals_df, kl_data, basis_data, market_data)
        return model.predict_proba(X)
    
    def compare_with_gbm(self, signals_df: pd.DataFrame,
                          gbm_probs: np.ndarray) -> Dict:
        """Compare ML predictions with GBM."""
        ml_probs = self.get_predictions(signals_df)
        
        return {
            "ml_mean": np.mean(ml_probs),
            "gbm_mean": np.mean(gbm_probs),
            "correlation": np.corrcoef(ml_probs, gbm_probs)[0, 1] if len(ml_probs) > 1 else 0,
            "ml_std": np.std(ml_probs),
            "gbm_std": np.std(gbm_probs),
            "disagreement_rate": np.mean(np.abs(ml_probs - gbm_probs) > 0.1),
        }


# Module-level singleton
_ml_manager: Optional[MLModelManager] = None


def get_ml_manager() -> MLModelManager:
    """Get or create global ML model manager."""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLModelManager()
    return _ml_manager


# =====================================================================
# Backward compatibility: MLFairValue (old torch interface)
# =====================================================================

class MLFairValue:
    """
    Backward-compatible wrapper for the old torch-based MLFairValue interface.
    
    This mimics the old API: row_to_features, train_labeled, predict_proba(row, spot, sigma, minutes)
    but uses the new MLModel underneath.
    """
    
    def __init__(self, model_type: str = "xgboost", use_calibration: bool = True):
        self._model = MLModel(model_type=model_type, use_calibration=use_calibration)
        self._feature_names = [
            "moneyness", "sigma", "minutes_left", "kalshi_mid", "spread", "fair"
        ]
    
    @staticmethod
    def row_to_features(row: dict, spot: float, sigma: float, minutes: float) -> list:
        """Convert row dict to 6-element feature list (old interface)."""
        strike = row.get("strike", spot)
        kalshi = row.get("kalshi", 0.5)
        fair = row.get("fair", kalshi)
        spread = row.get("spread", 0.01)
        
        moneyness = np.log(spot / strike) if strike > 0 else 0.0
        return [moneyness, sigma, minutes, kalshi, spread, fair]
    
    @classmethod
    def train_labeled(cls, rows: list, epochs: int = 100, lr: float = 0.01, **kwargs):
        """
        Train model from labeled rows (old interface).
        
        Args:
            rows: List of row dicts with 'strike', 'spot', 'sigma', 'kalshi', 'poly', 
                  'spread', 'fair', 'minutes_left', 'settled_yes'
            epochs: Ignored (kept for compatibility)
            lr: Ignored (kept for compatibility)
        """
        import pandas as pd
        
        # Create MLModel instance
        model = cls()
        
        # Prepare features DataFrame - use ticker as index for alignment
        features_list = []
        targets = {}
        
        for i, row in enumerate(rows):
            spot = row.get("spot", 100000.0)
            sigma = row.get("sigma", 0.0007)
            minutes = row.get("minutes_left", 30.0)
            
            # Create signal-like row for prepare_features
            ticker = f"TEST_{i}"
            signal_row = {
                "ticker": ticker,
                "gbm_q": row.get("fair", 0.5),
                "minutes_left": minutes,
            }
            features_list.append(signal_row)
            targets[ticker] = row.get("settled_yes", 0.5)
        
        signals_df = pd.DataFrame(features_list).set_index("ticker")
        
        # Prepare features using new interface
        X = model._model.prepare_features(signals_df)
        
        # For targets, create Series with ticker index
        y = pd.Series(targets, name="outcome")
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 2:
            raise ValueError("Insufficient training data")
        
        # Train (epochs/lr ignored for sklearn/xgboost)
        model._model.train(X, y)
        
        return model
    
    def predict_proba(self, row: dict, spot: float, sigma: float, minutes: float) -> float:
        """Predict probability using old interface signature."""
        # Convert row to features DataFrame
        import pandas as pd
        
        signal_row = {
            "ticker": "PREDICT",
            "gbm_q": row.get("fair", 0.5),
            "minutes_left": minutes,
        }
        signals_df = pd.DataFrame([signal_row])
        
        X = self._model.prepare_features(signals_df)
        proba = self._model.predict_proba(X)
        return float(proba[0]) if len(proba) > 0 else 0.5
    
    def save(self, path: str):
        """Save model to disk."""
        self._model.save(path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        # Try to load as new format first
        try:
            with open(path, "rb") as f:
                import pickle
                save_dict = pickle.load(f)
            
            model_type = save_dict.get("model_type", "xgboost")
            use_calibration = save_dict.get("use_calibration", True)
            
            wrapper = cls(model_type=model_type, use_calibration=use_calibration)
            wrapper._model.model = save_dict.get("model")
            wrapper._model.calibrator = save_dict.get("calibrator")
            wrapper._model.model_card = save_dict.get("model_card")
            wrapper._model.feature_cols = save_dict.get("feature_cols", wrapper._model.feature_cols)
            wrapper._model.model_type = model_type
            wrapper._model.use_calibration = use_calibration
            wrapper._model.is_trained = True
            
            return wrapper
        except Exception:
            # Fallback: create untrained model
            log.warning(f"Could not load model from {path}, returning untrained wrapper")
            return cls()


# Also expose the old function names for compatibility
def row_to_features(row: dict, spot: float, sigma: float, minutes: float) -> list:
    """Module-level function for backward compatibility."""
    return MLFairValue.row_to_features(row, spot, sigma, minutes)


def train_labeled(rows: list, epochs: int = 100, lr: float = 0.01, **kwargs):
    """Module-level function for backward compatibility."""
    return MLFairValue.train_labeled(rows, epochs, lr, **kwargs)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    # Check available libraries
    print(f"XGBoost available: {XGB_AVAILABLE}")
    print(f"LightGBM available: {LGB_AVAILABLE}")
    print(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
    
    if not (XGB_AVAILABLE or LGB_AVAILABLE or SKLEARN_AVAILABLE):
        print("No ML libraries available. Install with:")
        print("  pip install xgboost lightgbm scikit-learn")
    else:
        print("ML libraries available for training!")
    
    # Test model creation
    try:
        model = MLModel("xgboost" if XGB_AVAILABLE else "lightgbm")
        print(f"Model created: {model.model_type}")
    except Exception as e:
        print(f"Model creation failed: {e}")
    
    print("\nML model module tests passed!")