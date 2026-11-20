#!/usr/bin/env python3
"""
ml_model.py — T15-23: ML model path (pick one stack, features, walk-forward CV).

Implements a single ML stack for probability prediction:
- Stack: XGBoost (binary classification for YES win)
- Features: 12 core features from market microstructure + GBM
- Walk-forward CV with purged K-fold
- Output: calibrated probability p_yes for fair_value integration
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)


# ─── Feature Engineering ───────────────────────────────────────────────────────

@dataclass
class MLFeatures:
    """Feature vector for a single market at entry time."""
    # Market microstructure
    entry_price_cents: float          # Our entry price (cents)
    bid_ask_spread: float             # yes_ask - yes_bid (cents)
    bid_ask_mid: float                # (yes_bid + yes_ask) / 2
    book_imbalance: float             # (yes_bid_size - yes_ask_size) / (yes_bid_size + yes_ask_size)
    total_depth: float                # yes_bid_size + yes_ask_size + no_bid_size + no_ask_size

    # Time & window
    minutes_left: float               # Minutes to settlement
    minutes_into_window: float        # 15 - minutes_left
    time_decay: float                 # sqrt(minutes_left) factor

    # GBM reference
    gbm_prob_yes: float               # GBM point probability
    gbm_z_score: float                # (spot - strike) / (spot * sigma * sqrt(T))
    arithmetic_prob: float            # Arithmetic Brownian probability

    # BRTI reference
    brti_deviation: float             # spot - brti_60s_avg (if available)
    brti_available: int               # 1 if BRTI available, 0 otherwise

    # Volatility
    atr_1m: float                     # 1-minute ATR (dollars)
    sigma_per_min: float              # Fractional sigma per minute
    realized_vol_5m: float            # 5-min realized volatility

    # Target (for training)
    target: int = 0                   # 1 if YES won, 0 otherwise

    def to_array(self) -> np.ndarray:
        """Convert to feature array (excluding target)."""
        return np.array([
            self.entry_price_cents / 100.0,  # Normalize to [0,1]
            self.bid_ask_spread / 100.0,
            self.bid_ask_mid / 100.0,
            self.book_imbalance,
            math.log1p(self.total_depth),
            self.minutes_left / 15.0,
            self.minutes_into_window / 15.0,
            self.time_decay,
            self.gbm_prob_yes,
            self.gbm_z_score,
            self.arithmetic_prob,
            self.brti_deviation / 1000.0 if self.brti_deviation else 0.0,
            float(self.brti_available),
            self.atr_1m / 1000.0,
            self.sigma_per_min * 1000.0,
            self.realized_vol_5m * 1000.0,
        ])

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "entry_price",
            "bid_ask_spread",
            "bid_ask_mid",
            "book_imbalance",
            "log_total_depth",
            "minutes_left_frac",
            "minutes_into_window_frac",
            "time_decay",
            "gbm_prob_yes",
            "gbm_z_score",
            "arithmetic_prob",
            "brti_deviation",
            "brti_available",
            "atr_1m",
            "sigma_per_min",
            "realized_vol_5m",
        ]


def extract_features(
    market: Dict,
    orderbook: Dict,
    spot_price: float,
    strike: float,
    minutes_left: float,
    sigma_per_min: float,
    gbm_prob: float,
    arithmetic_prob: float,
    brti_60s_avg: float = None,
    atr_1m: float = None,
    realized_vol_5m: float = None,
) -> MLFeatures:
    """Extract features from market data at entry time."""
    yb = orderbook.get("yes_bid", 0) * 100
    ya = orderbook.get("yes_ask", 0) * 100
    nb = orderbook.get("no_bid", 0) * 100
    na = orderbook.get("no_ask", 0) * 100
    yb_size = orderbook.get("yes_bid_size", 0)
    ya_size = orderbook.get("yes_ask_size", 0)
    nb_size = orderbook.get("no_bid_size", 0)
    na_size = orderbook.get("no_ask_size", 0)

    # Our entry price (favourite maker: bid + 1)
    if (ya - yb) <= (na - nb):
        entry_price = yb + 1
    else:
        entry_price = nb + 1

    spread = ya - yb
    mid = (yb + ya) / 2

    total_yes_size = yb_size + ya_size
    total_no_size = nb_size + na_size
    total_depth = total_yes_size + total_no_size
    imbalance = (yb_size - ya_size) / total_yes_size if total_yes_size > 0 else 0.0

    minutes_into = 15.0 - minutes_left
    time_decay = math.sqrt(minutes_left / 15.0)

    # GBM z-score
    if spot_price > 0 and sigma_per_min > 0 and minutes_left > 0:
        gbm_z = (spot_price - strike) / (spot_price * sigma_per_min * math.sqrt(minutes_left))
    else:
        gbm_z = 0.0

    brti_dev = 0.0
    brti_avail = 0
    if brti_60s_avg is not None and spot_price > 0:
        brti_dev = spot_price - brti_60s_avg
        brti_avail = 1

    return MLFeatures(
        entry_price_cents=entry_price,
        bid_ask_spread=spread,
        bid_ask_mid=mid,
        book_imbalance=imbalance,
        total_depth=total_depth,
        minutes_left=minutes_left,
        minutes_into_window=minutes_into,
        time_decay=time_decay,
        gbm_prob_yes=gbm_prob,
        gbm_z_score=gbm_z,
        arithmetic_prob=arithmetic_prob,
        brti_deviation=brti_dev,
        brti_available=brti_avail,
        atr_1m=atr_1m or 0.0,
        sigma_per_min=sigma_per_min,
        realized_vol_5m=realized_vol_5m or 0.0,
    )


# ─── Model Training ────────────────────────────────────────────────────────────

@dataclass
class MLModelConfig:
    """XGBoost training configuration."""
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42


@dataclass
class MLModelResult:
    """Result of ML model training."""
    model: Optional[xgb.Booster] = None
    feature_importance: Dict[str, float] = None
    cv_scores: List[float] = None
    best_iteration: int = 0
    logloss: float = 0.0
    auc: float = 0.0
    calibration_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None


def train_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: MLModelConfig = None,
    feature_names: List[str] = None,
) -> MLModelResult:
    """
    Train XGBoost model with early stopping.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (0/1)
        X_val: Validation features
        y_val: Validation targets
        config: Training configuration
        feature_names: Feature names for importance

    Returns:
        MLModelResult with trained model and metrics
    """
    if not HAS_XGB:
        log.error("XGBoost not available")
        return MLModelResult()

    if config is None:
        config = MLModelConfig()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": config.objective,
        "eval_metric": config.eval_metric,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "seed": config.random_state,
        "verbosity": 0,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.n_estimators,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )

    # Feature importance
    importance = model.get_score(importance_type="gain")
    if feature_names:
        importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}

    # Validation predictions
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    val_pred_binary = (val_pred > 0.5).astype(int)

    from sklearn.metrics import log_loss, roc_auc_score
    logloss = log_loss(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred)

    # Calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_val, val_pred, n_bins=10)

    return MLModelResult(
        model=model,
        feature_importance=importance,
        best_iteration=model.best_iteration,
        logloss=logloss,
        auc=auc,
        calibration_curve=(prob_true, prob_pred),
    )


# ─── Walk-Forward Cross-Validation ────────────────────────────────────────────

@dataclass
class WalkForwardSplit:
    """A single walk-forward split."""
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime


def purged_kfold_walkforward(
    timestamps: np.ndarray,
    n_splits: int = 5,
    train_days: int = 60,
    val_days: int = 14,
    test_days: int = 14,
    embargo_days: int = 1,
) -> List[WalkForwardSplit]:
    """
    Generate purged walk-forward splits.

    Purges embargo period between train/val and val/test to prevent leakage.
    """
    unique_dates = np.unique(timestamps.astype("datetime64[D]"))
    splits = []

    total_days = len(unique_dates)
    window_size = train_days + val_days + test_days + 2 * embargo_days

    if total_days < window_size:
        log.warning(f"Insufficient data for walk-forward: {total_days} < {window_size}")
        return splits

    # Start from the end and work backwards
    end_idx = total_days
    for i in range(n_splits):
        test_end_idx = end_idx
        test_start_idx = test_end_idx - test_days
        val_end_idx = test_start_idx - embargo_days
        val_start_idx = val_end_idx - val_days
        train_end_idx = val_start_idx - embargo_days
        train_start_idx = train_end_idx - train_days

        if train_start_idx < 0:
            break

        train_dates = unique_dates[train_start_idx:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        train_mask = np.isin(timestamps.astype("datetime64[D]"), train_dates)
        val_mask = np.isin(timestamps.astype("datetime64[D]"), val_dates)
        test_mask = np.isin(timestamps.astype("datetime64[D]"), test_dates)

        splits.append(WalkForwardSplit(
            train_indices=np.where(train_mask)[0],
            val_indices=np.where(val_mask)[0],
            test_indices=np.where(test_mask)[0],
            train_start=train_dates[0].astype(datetime),
            train_end=train_dates[-1].astype(datetime),
            val_start=val_dates[0].astype(datetime),
            val_end=val_dates[-1].astype(datetime),
            test_start=test_dates[0].astype(datetime),
            test_end=test_dates[-1].astype(datetime),
        ))

        end_idx = train_start_idx

    return list(reversed(splits))  # Chronological order


def walkforward_cv(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    config: MLModelConfig = None,
    feature_names: List[str] = None,
) -> Dict:
    """
    Run walk-forward cross-validation.

    Returns aggregated metrics and per-fold results.
    """
    splits = purged_kfold_walkforward(timestamps)
    if not splits:
        return {"error": "Insufficient data for walk-forward CV"}

    fold_results = []
    all_val_preds = []
    all_val_targets = []

    for i, split in enumerate(splits):
        log.info(f"Fold {i+1}/{len(splits)}: train={len(split.train_indices)}, val={len(split.val_indices)}, test={len(split.test_indices)}")

        X_train, y_train = X[split.train_indices], y[split.train_indices]
        X_val, y_val = X[split.val_indices], y[split.val_indices]

        result = train_ml_model(X_train, y_train, X_val, y_val, config, feature_names)

        fold_results.append({
            "fold": i,
            "logloss": result.logloss,
            "auc": result.auc,
            "best_iteration": result.best_iteration,
            "train_period": f"{split.train_start.date()} to {split.train_end.date()}",
            "val_period": f"{split.val_start.date()} to {split.val_end.date()}",
        })

        all_val_preds.extend(result.model.predict(
            xgb.DMatrix(X_val, feature_names=feature_names),
            iteration_range=(0, result.best_iteration + 1)
        ))
        all_val_targets.extend(y_val)

    # Aggregate
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    all_val_preds = np.array(all_val_preds)
    all_val_targets = np.array(all_val_targets)

    return {
        "n_folds": len(splits),
        "mean_logloss": float(np.mean([f["logloss"] for f in fold_results])),
        "mean_auc": float(np.mean([f["auc"] for f in fold_results])),
        "overall_logloss": log_loss(all_val_targets, all_val_preds),
        "overall_auc": roc_auc_score(all_val_targets, all_val_preds),
        "overall_brier": brier_score_loss(all_val_targets, all_val_preds),
        "fold_results": fold_results,
    }


# ─── Model Persistence ────────────────────────────────────────────────────────

def save_model(
    model: xgb.Booster,
    path: str,
    feature_names: List[str] = None,
    metadata: Dict = None,
):
    """Save model to disk."""
    model.save_model(path)
    if metadata:
        meta_path = path.replace(".model", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": feature_names,
                "metadata": metadata,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)


def load_model(path: str) -> Tuple[xgb.Booster, Dict]:
    """Load model from disk."""
    model = xgb.Booster()
    model.load_model(path)

    meta_path = path.replace(".model", "_meta.json")
    metadata = {}
    if Path(meta_path).exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, metadata


# ─── Integration with fair_value ──────────────────────────────────────────────

class MLProbabilityPredictor:
    """
    Wrapper for using ML model in fair_value pipeline.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_names = None
        self.metadata = {}

        if model_path and Path(model_path).exists():
            self.model, meta = load_model(model_path)
            self.feature_names = meta.get("feature_names")
            self.metadata = meta.get("metadata", {})

    def predict_proba(self, features: MLFeatures) -> float:
        """Predict P(YES wins) for a single feature vector."""
        if self.model is None:
            return 0.5  # Fallback

        X = features.to_array().reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return float(self.model.predict(dmatrix)[0])

    def predict_batch(self, features_list: List[MLFeatures]) -> np.ndarray:
        """Predict probabilities for multiple feature vectors."""
        if self.model is None:
            return np.full(len(features_list), 0.5)

        X = np.array([f.to_array() for f in features_list])
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix)


# ─── CLI / Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic features
    n_features = len(MLFeatures.feature_names())
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = np.abs(X[:, 0])  # entry_price positive
    X[:, 3] = np.tanh(X[:, 3])  # imbalance [-1,1]
    X[:, 8] = 1 / (1 + np.exp(-X[:, 8]))  # gbm_prob sigmoid

    # Synthetic target with some signal
    signal = 0.3 * X[:, 8] + 0.2 * X[:, 9] + 0.1 * X[:, 10]  # gbm + z + arith
    prob = 1 / (1 + np.exp(-signal))
    y = (np.random.random(n_samples) < prob).astype(int)

    # Timestamps for walk-forward
    base = np.datetime64("2026-01-01")
    timestamps = base + np.arange(n_samples).astype("timedelta64[D]")

    print(f"Data: {n_samples} samples, {y.mean():.1%} positive")
    print(f"Features: {MLFeatures.feature_names()}")

    # Walk-forward CV
    config = MLModelConfig(n_estimators=200, early_stopping_rounds=20)
    cv_result = walkforward_cv(X, y, timestamps, config, MLFeatures.feature_names())

    print(f"\nWalk-Forward CV Results:")
    print(f"  Folds: {cv_result['n_folds']}")
    print(f"  Mean LogLoss: {cv_result['mean_logloss']:.4f}")
    print(f"  Mean AUC: {cv_result['mean_auc']:.4f}")
    print(f"  Overall LogLoss: {cv_result['overall_logloss']:.4f}")
    print(f"  Overall AUC: {cv_result['overall_auc']:.4f}")
    print(f"  Overall Brier: {cv_result['overall_brier']:.4f}")

    for f in cv_result["fold_results"]:
        print(f"  Fold {f['fold']}: LL={f['logloss']:.4f}, AUC={f['auc']:.4f}, iter={f['best_iteration']}")


# ─── Backward Compatibility for forecaster.py and tests ─────────────────────────

# Feature columns expected by forecaster and tests
FEATURE_COLS = [
    "gbm_q",              # GBM fair value probability
    "kl_edge",            # Bernoulli KL edge signal
    "kl_ladder",          # Ladder KL divergence
    "kl_ensemble",        # Ensemble disagreement KL
    "basis_bps_kalshi",   # Kalshi settlement basis vs composite
    "basis_bps_poly",     # Polymarket settlement basis vs composite
    "vol_regime",         # Volatility regime (low/med/high)
    "minutes_left",       # Time to expiry
    "book_imbalance",     # Order book imbalance
    "spread_bps",         # Bid-ask spread in bps
    "fill_rate_recent",   # Recent fill rate (last 100 orders)
    "atr_hurdle",         # ATR hurdle value
    "fee_adjusted_ev",    # Fee-adjusted expected value
]

TARGET_COL = "outcome"  # 1 if YES settled, 0 if NO settled

# LightGBM availability
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelCard:
    """Model card with metadata for MemPalace KG."""
    model_name: str
    model_type: str = "xgboost"
    training_window: str = ""
    n_samples: int = 0
    n_features: int = 0
    calibration_method: str = "isotonic"
    brier_score: float = 0.0
    log_loss: float = 0.0
    auc_roc: float = 0.0
    feature_importances: Dict[str, float] = None
    purge_embargo_days: int = 1
    created_at: str = ""

    def __post_init__(self):
        if self.feature_importances is None:
            self.feature_importances = {}
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class MLModel:
    """XGBoost model with calibration (backward compatible with forecaster.py and tests)."""

    def __init__(
        self,
        model_type: str = "xgboost",
        use_calibration: bool = True,
        purge_embargo_days: int = 1,
    ):
        self.model_type = model_type
        self.use_calibration = use_calibration
        self.purge_embargo_days = purge_embargo_days

        self.model: Optional[xgb.Booster] = None
        self.calibrator = None
        self.feature_names = FEATURE_COLS  # Use FEATURE_COLS for backward compat
        self.model_card = ModelCard(
            model_name=f"ml_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_type,
            purge_embargo_days=purge_embargo_days,
        )
        self.is_trained = False
        self._gbm_baseline = None

    def _get_model(self):
        """Get underlying model (for sklearn fallback)."""
        if HAS_XGB and self.model_type == "xgboost":
            return xgb.XGBClassifier()
        elif LGB_AVAILABLE and self.model_type == "lightgbm":
            return lgb.LGBMClassifier()
        elif SKLEARN_AVAILABLE:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        return None

    def prepare_features(self, signals_df: "pd.DataFrame", kl_data=None, basis_data=None, market_data=None) -> "pd.DataFrame":
        """Prepare feature matrix from signals DataFrame.
        
        Extra parameters (kl_data, basis_data, market_data) accepted for compatibility
        with forecaster.py but not used in this simplified implementation.
        """
        X = pd.DataFrame(index=signals_df.index)

        # Direct mapping of expected features
        for col in FEATURE_COLS:
            if col in signals_df.columns:
                X[col] = signals_df[col]
            else:
                X[col] = 0.0

        return X[FEATURE_COLS]

    def prepare_targets(self, settled_df: "pd.DataFrame") -> "pd.Series":
        """Prepare target series from settled DataFrame."""
        if TARGET_COL in settled_df.columns:
            y = settled_df[TARGET_COL]
        elif "outcome" in settled_df.columns:
            y = settled_df["outcome"]
        elif "yes_won" in settled_df.columns:
            y = settled_df["yes_won"].astype(int)
        else:
            y = pd.Series(0, index=settled_df.index)

        # Ensure index matches ticker
        if "ticker" in settled_df.columns:
            y.index = settled_df["ticker"]
        return y

    def train(self, X: "pd.DataFrame", y: "pd.Series") -> Dict:
        """Train model with walk-forward validation."""
        if not HAS_XGB and not LGB_AVAILABLE and not SKLEARN_AVAILABLE:
            return {"error": "No ML backend available"}

        # Align indices
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]

        n = len(X)
        if n < 50:
            return {"error": f"Insufficient data: {n} samples"}

        # Time-based split
        split_idx = int(n * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train based on backend
        if HAS_XGB and self.model_type == "xgboost":
            dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=FEATURE_COLS)
            dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=FEATURE_COLS)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "seed": 42,
                "verbosity": 0,
            }
            self.model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtrain, "train"), (dval, "val")], early_stopping_rounds=20, verbose_eval=False)
            preds_val = self.model.predict(dval, iteration_range=(0, self.model.best_iteration + 1))
        elif LGB_AVAILABLE and self.model_type == "lightgbm":
            train_data = lgb.Dataset(X_train.values, label=y_train.values, feature_name=FEATURE_COLS)
            val_data = lgb.Dataset(X_val.values, label=y_val.values, feature_name=FEATURE_COLS)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.03,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "seed": 42,
                "verbosity": -1,
            }
            self.model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[val_data], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            preds_val = self.model.predict(X_val.values, num_iteration=self.model.best_iteration)
        elif SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train.values, y_train.values)
            preds_val = self.model.predict_proba(X_val.values)[:, 1]
        else:
            return {"error": "No ML backend available"}

        # Metrics on validation
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        brier = brier_score_loss(y_val, preds_val)
        ll = log_loss(y_val, np.clip(preds_val, 1e-15, 1-1e-15))
        auc = roc_auc_score(y_val, preds_val)

        # Calibration
        if self.use_calibration and SKLEARN_AVAILABLE:
            if HAS_XGB and self.model_type == "xgboost":
                dtrain_all = xgb.DMatrix(X.values, label=y.values, feature_names=FEATURE_COLS)
                preds_all = self.model.predict(dtrain_all)
            elif LGB_AVAILABLE and self.model_type == "lightgbm":
                preds_all = self.model.predict(X.values, num_iteration=self.model.best_iteration)
            else:
                preds_all = self.model.predict_proba(X.values)[:, 1]

            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(preds_all, y.values)
            self.model_card.calibration_method = "isotonic"

        # GBM baseline comparison
        gbm_probs = X["gbm_q"].values if "gbm_q" in X.columns else np.full(n, 0.5)
        gbm_brier = brier_score_loss(y, gbm_probs)
        gbm_logloss = log_loss(y, np.clip(gbm_probs, 1e-15, 1-1e-15))
        gbm_auc = roc_auc_score(y, gbm_probs)

        # Feature importances
        if self.model and HAS_XGB and self.model_type == "xgboost":
            importance = self.model.get_score(importance_type="gain")
            # Handle both 'f0', 'f1' and actual feature name keys
            feat_imp = {}
            for k, v in importance.items():
                if k.startswith('f') and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(FEATURE_COLS):
                        feat_imp[FEATURE_COLS[idx]] = v
                elif k in FEATURE_COLS:
                    feat_imp[k] = v
            # Normalize to sum to 1
            total = sum(feat_imp.values())
            if total > 0:
                feat_imp = {k: v / total for k, v in feat_imp.items()}
            self.model_card.feature_importances = feat_imp
        elif self.model and LGB_AVAILABLE and self.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type="gain")
            self.model_card.feature_importances = dict(zip(FEATURE_COLS, importance))
        elif self.model and SKLEARN_AVAILABLE:
            importance = self.model.feature_importances_
            self.model_card.feature_importances = dict(zip(FEATURE_COLS, importance))

        # Update model card
        self.model_card.n_samples = n
        self.model_card.n_features = len(FEATURE_COLS)
        self.model_card.brier_score = float(brier)
        self.model_card.log_loss = float(ll)
        self.model_card.auc_roc = float(auc)

        self.is_trained = True

        return {
            "brier": float(brier),
            "log_loss": float(ll),
            "auc_roc": float(auc),
            "n_train": len(X_train),
            "n_test": len(X_val),
            "gbm_brier": float(gbm_brier),
            "gbm_log_loss": float(gbm_logloss),
            "gbm_auc": float(gbm_auc),
            "improvement": float(gbm_brier - brier),
            "beats_gbm": brier < gbm_brier,
        }

    def predict_proba(self, X: "pd.DataFrame") -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)

        # Align features
        X_aligned = X[FEATURE_COLS] if all(c in X.columns for c in FEATURE_COLS) else self.prepare_features(X)

        if HAS_XGB and self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(X_aligned.values, feature_names=FEATURE_COLS)
            preds = self.model.predict(dmatrix)
            if self.model.best_iteration:
                preds = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration + 1))
        elif LGB_AVAILABLE and self.model_type == "lightgbm":
            preds = self.model.predict(X_aligned.values, num_iteration=self.model.best_iteration)
        elif SKLEARN_AVAILABLE:
            preds = self.model.predict_proba(X_aligned.values)[:, 1]
        else:
            return np.full(len(X), 0.5)

        if self.calibrator is not None:
            preds = self.calibrator.predict(preds)

        return np.clip(preds, 1e-6, 1 - 1e-6)

    def predict(self, X: "pd.DataFrame", threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: str = None) -> str:
        """Save model to disk, return path."""
        import pickle
        from pathlib import Path

        if path is None:
            path = f"models/{self.model_card.model_name}.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "calibrator": self.calibrator,
                "model_card": self.model_card,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "use_calibration": self.use_calibration,
            }, f)

        return path

    def load(self, path: str) -> bool:
        """Load model from disk."""
        import pickle
        from pathlib import Path

        if not Path(path).exists():
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.calibrator = data["calibrator"]
        self.model_card = data["model_card"]
        self.model_type = data["model_type"]
        self.feature_names = data["feature_names"]
        self.use_calibration = data["use_calibration"]
        self.is_trained = True

        return True

    def get_model_card(self) -> Optional[ModelCard]:
        """Get model card."""
        return self.model_card if self.is_trained else None


# ─── Backward Compatibility: MLFairValue (for trader.py, ladder_strategy.py) ───────

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
        model = cls()
        success = model._model.load(path)
        if success:
            return model
        return None


class MLModelManager:
    """Manages multiple ML models (backward compatible with tests)."""

    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.current_model: Optional[str] = None
        self._gbm_baseline = None

    def set_gbm_baseline(self, brier: float, log_loss: float, auc: float):
        """Set GBM baseline for comparison."""
        self._gbm_baseline = {"brier": brier, "log_loss": log_loss, "auc": auc}

    def train_new_model(
        self,
        model_type: str,
        signals_data: "pd.DataFrame",
        settled_data: "pd.DataFrame",
    ) -> Tuple[str, Dict]:
        """Train a new model and add to manager."""
        model = MLModel(model_type=model_type, use_calibration=True)
        X = model.prepare_features(signals_data)
        y = model.prepare_targets(settled_data)
        metrics = model.train(X, y)

        self.models[model.model_card.model_name] = model
        self.current_model = model.model_card.model_name

        return model.model_card.model_name, metrics

    def get_predictions(self, signals_df: "pd.DataFrame") -> np.ndarray:
        """Get predictions from current model."""
        model = self.models.get(self.current_model)
        if model:
            X = model.prepare_features(signals_df)
            return model.predict_proba(X)
        return np.full(len(signals_df), 0.5)

    def list_models(self) -> List[Dict]:
        return [asdict(m.model_card) for m in self.models.values() if m.is_trained]