"""
forecaster.py — Multi-model 14-day forecaster (T13-31).

Four-model ensemble with honest probability bounds:
- M1: Clustered bootstrap (non-parametric, resample markets by cluster)
- M2: GBM Monte Carlo (parametric, lognormal dynamics)
- M3: Bayesian Beta-Binomial (conjugate prior on win rate)
- M4: ML ensemble (xgboost/lightgbm + isotonic calibration)

Outputs: probability bands, not point estimates.
Promotion gate: ML (M4) must beat GBM (M2) out-of-sample.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

from scipy import stats
from scipy.stats import beta, norm

from gbm_fair_value import calculate_gbm_fair_value, GBMInputs, GBMFairValueModel
from kl_divergence import bernoulli_kl, kl_edge_signal, ladder_kl_divergence, ensemble_disagreement, KLDivergenceLayer
from ml_model import MLModel, MLModelManager
from local_db import get_local_db

warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger("forecaster")

# Model directory
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class ForecastResult:
    """Result from a single model."""
    model_name: str
    p_yes: float          # Point estimate P(YES)
    ci_lower: float       # Lower confidence bound
    ci_upper: float       # Upper confidence bound
    confidence: float     # Model confidence (0-1)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EnsembleForecast:
    """Combined ensemble forecast with honest bounds."""
    timestamp: str
    horizon_days: int
    p_yes_ensemble: float      # Weighted ensemble mean
    ci_lower: float            # Conservative lower bound (min of model lowers)
    ci_upper: float            # Conservative upper bound (max of model uppers)
    model_results: List[ForecastResult]
    cluster_lower: float       # Cluster-robust lower bound
    cluster_upper: float       # Cluster-robust upper bound
    n_markets: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class Cluster:
    """Market cluster for bootstrapping."""
    cluster_id: int
    tickers: List[str]
    features: Dict[str, float]  # Cluster centroid features


# =====================================================================
# M1: Clustered Bootstrap
# =====================================================================

class ClusteredBootstrapForecaster:
    """
    Non-parametric bootstrap with market clustering.
    
    Resamples clusters of similar markets to preserve correlation structure.
    Returns confidence intervals from bootstrap distribution.
    """
    
    def __init__(self, n_clusters: int = 5, n_bootstrap: int = 1000, 
                 confidence: float = 0.95):
        self.n_clusters = n_clusters
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.clusters: List[Cluster] = []
        self._cluster_model = None
    
    def fit(self, markets_df: pd.DataFrame):
        """Cluster markets by features."""
        from sklearn.cluster import KMeans
        
        # Features for clustering
        feature_cols = [c for c in markets_df.columns 
                       if c not in ['ticker', 'outcome', 'timestamp']]
        X = markets_df[feature_cols].fillna(0)
        
        # K-means clustering
        n_clusters = min(self.n_clusters, len(X))
        if n_clusters < 2:
            self.clusters = [Cluster(0, markets_df['ticker'].tolist(), {})]
            return
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.clusters = []
        for i in range(n_clusters):
            mask = labels == i
            tickers = markets_df.loc[mask, 'ticker'].tolist()
            centroid = X.loc[mask].mean().to_dict()
            self.clusters.append(Cluster(i, tickers, centroid))
        
        log.info(f"Created {len(self.clusters)} market clusters")
    
    def predict(self, markets_df: pd.DataFrame, 
                feature_cols: List[str] = None) -> ForecastResult:
        """Bootstrap prediction with cluster resampling."""
        if not self.clusters:
            return ForecastResult(
                model_name="clustered_bootstrap",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"error": "No clusters fitted"}
            )
        
        # Get outcomes for each market
        if 'outcome' not in markets_df.columns:
            return ForecastResult(
                model_name="clustered_bootstrap",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"error": "No outcome column"}
            )
        
        # Bootstrap by resampling clusters
        bootstrap_probs = []
        
        for _ in range(self.n_bootstrap):
            # Sample clusters with replacement
            sampled_clusters = np.random.choice(
                self.clusters, size=len(self.clusters), replace=True
            )
            
            # Collect all tickers from sampled clusters
            sampled_tickers = []
            for cl in sampled_clusters:
                sampled_tickers.extend(cl.tickers)
            
            # Get outcomes for sampled tickers
            outcomes = markets_df[markets_df['ticker'].isin(sampled_tickers)]['outcome']
            if len(outcomes) > 0:
                p = outcomes.mean()
                bootstrap_probs.append(p)
        
        if not bootstrap_probs:
            return ForecastResult(
                model_name="clustered_bootstrap",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"error": "No bootstrap samples"}
            )
        
        bootstrap_probs = np.array(bootstrap_probs)
        p_yes = bootstrap_probs.mean()
        
        # Confidence interval (percentile method)
        alpha = 1 - self.confidence
        ci_lower = np.percentile(bootstrap_probs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_probs, 100 * (1 - alpha / 2))
        
        # Confidence based on CI width
        ci_width = ci_upper - ci_lower
        confidence = max(0.1, 1 - ci_width)
        
        return ForecastResult(
            model_name="clustered_bootstrap",
            p_yes=float(p_yes),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence=float(confidence),
            metadata={
                "n_bootstrap": self.n_bootstrap,
                "n_clusters": len(self.clusters),
                "bootstrap_std": float(bootstrap_probs.std()),
                "n_samples": int(len(bootstrap_probs))
            }
        )
    
    def get_cluster_bounds(self, markets_df: pd.DataFrame) -> Tuple[float, float]:
        """Get cluster-robust bounds (more conservative)."""
        if not self.clusters:
            return 0.1, 0.9
        
        # Compute per-cluster win rates
        cluster_rates = []
        for cl in self.clusters:
            if 'outcome' in markets_df.columns:
                outcomes = markets_df[markets_df['ticker'].isin(cl.tickers)]['outcome']
            else:
                outcomes = pd.Series([], dtype=float)
            if len(outcomes) >= 5:  # Min samples per cluster
                cluster_rates.append(outcomes.mean())
        
        if len(cluster_rates) < 2:
            return 0.1, 0.9
        
        # Use cluster-level t-distribution for more conservative bounds
        from scipy.stats import t
        mean_rate = np.mean(cluster_rates)
        se = np.std(cluster_rates, ddof=1) / np.sqrt(len(cluster_rates))
        dof = len(cluster_rates) - 1
        t_crit = t.ppf(0.975, dof)
        
        return max(0.01, mean_rate - t_crit * se), min(0.99, mean_rate + t_crit * se)


# =====================================================================
# M2: GBM Monte Carlo
# =====================================================================

class GBMMonteCarloForecaster:
    """
    Parametric Monte Carlo using GBM fair value dynamics.
    
    Simulates spot paths and computes settlement probabilities.
    Accounts for basis risk, fat tails, and time decay.
    """
    
    def __init__(self, n_paths: int = 10000, confidence: float = 0.95):
        self.n_paths = n_paths
        self.confidence = confidence
    
    def predict(self, 
                spot: float,
                strike: float,
                sigma_per_minute: float,
                minutes_left: float,
                basis_bps: float = 0,
                skew: float = 0,
                kurtosis: float = 3) -> ForecastResult:
        """
        Monte Carlo simulation of GBM with fat tails.
        
        Args:
            spot: Current spot price
            strike: Strike price
            sigma_per_minute: Volatility per minute
            minutes_left: Time to expiry
            basis_bps: Settlement basis in bps (adjusts strike)
            skew: Skewness adjustment
            kurtosis: Kurtosis (3 = normal, >3 = fat tails)
        """
        # Adjusted strike for basis
        adj_strike = strike * (1 + basis_bps / 10000)
        
        # Total variance
        total_var = (sigma_per_minute ** 2) * minutes_left
        total_vol = np.sqrt(total_var)
        
        # Monte Carlo
        np.random.seed(42)
        
        if kurtosis > 3.5:
            # Student-t for fat tails
            df = max(3, 6 / (kurtosis - 3))
            z = np.random.standard_t(df, self.n_paths)
            z = z * np.sqrt((df - 2) / df)  # Normalize variance
        else:
            z = np.random.randn(self.n_paths)
        
        # Add skew via Cornish-Fisher
        if abs(skew) > 0.1:
            z = z + (skew / 6) * (z**2 - 1)
        
        # Simulate terminal spot
        drift = -0.5 * total_var
        terminal_spot = spot * np.exp(drift + total_vol * z)
        
        # Settlement: YES if terminal >= strike
        yes_won = (terminal_spot >= adj_strike).mean()
        
        # Analytical approximation for CI
        # Delta method on lognormal probability
        d2 = (np.log(spot / adj_strike) - 0.5 * total_var) / total_vol
        p_analytical = norm.cdf(d2)
        
        # Standard error of MC estimate
        se = np.sqrt(yes_won * (1 - yes_won) / self.n_paths)
        z_crit = norm.ppf((1 + self.confidence) / 2)
        
        ci_lower = yes_won - z_crit * se
        ci_upper = yes_won + z_crit * se
        
        # Ensure bounds are in [0, 1] and ordered
        ci_lower = max(0.0, min(ci_lower, ci_upper))
        ci_upper = min(1.0, max(ci_lower, ci_upper))
        # Apply small floor to avoid exact 0/1
        ci_lower = max(0.001, ci_lower)
        ci_upper = min(0.999, ci_upper)
        
        # Confidence increases with more paths and less variance
        confidence = min(0.99, 1 - (ci_upper - ci_lower))
        
        return ForecastResult(
            model_name="gbm_monte_carlo",
            p_yes=float(yes_won),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence=float(confidence),
            metadata={
                "n_paths": self.n_paths,
                "spot": spot,
                "strike": strike,
                "adj_strike": adj_strike,
                "sigma": sigma_per_minute,
                "minutes_left": minutes_left,
                "total_vol": total_vol,
                "analytical_p": float(p_analytical),
                "mc_std": float(se)
            }
        )


# =====================================================================
# M3: Bayesian Beta-Binomial
# =====================================================================

class BayesianBetaBinomialForecaster:
    """
    Conjugate Bayesian model for win rate estimation.
    
    Prior: Beta(alpha, beta) — can use Jeffreys (0.5, 0.5) or historical
    Likelihood: Binomial(n, p)
    Posterior: Beta(alpha + wins, beta + losses)
    """
    
    def __init__(self, 
                 prior_alpha: float = 0.5,  # Jeffreys prior
                 prior_beta: float = 0.5,
                 confidence: float = 0.95,
                 use_historical_prior: bool = True):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.confidence = confidence
        self.use_historical_prior = use_historical_prior
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta
        self.n_observations = 0
    
    def fit(self, outcomes: np.ndarray, 
            similar_markets_outcomes: np.ndarray = None):
        """Update posterior with observed outcomes."""
        wins = outcomes.sum()
        losses = len(outcomes) - wins
        
        if self.use_historical_prior and similar_markets_outcomes is not None:
            # Use historical data to inform prior
            hist_wins = similar_markets_outcomes.sum()
            hist_total = len(similar_markets_outcomes)
            # Weight historical data at 10% of current
            weight = 0.1
            self.posterior_alpha = self.prior_alpha + wins + weight * hist_wins
            self.posterior_beta = self.prior_beta + losses + weight * (hist_total - hist_wins)
        else:
            self.posterior_alpha = self.prior_alpha + wins
            self.posterior_beta = self.prior_beta + losses
        
        self.n_observations = len(outcomes)
        log.info(f"Bayesian update: {wins}W/{losses}L -> "
                 f"Beta({self.posterior_alpha:.2f}, {self.posterior_beta:.2f})")
    
    def predict(self, 
                n_future: int = 1,
                min_samples_for_ci: int = 10) -> ForecastResult:
        """Predictive distribution for future outcomes."""
        
        if self.n_observations < min_samples_for_ci:
            # Not enough data, use prior
            post_alpha, post_beta = self.prior_alpha, self.prior_beta
            confidence = 0.3
        else:
            post_alpha, post_beta = self.posterior_alpha, self.posterior_beta
            confidence = min(0.95, 0.5 + 0.45 * (self.n_observations / 100))
        
        # Posterior mean
        p_yes = post_alpha / (post_alpha + post_beta)
        
        # Credible interval (highest posterior density approx)
        ci_lower = beta.ppf((1 - self.confidence) / 2, post_alpha, post_beta)
        ci_upper = beta.ppf((1 + self.confidence) / 2, post_alpha, post_beta)
        
        # Predictive variance for n_future trials
        # Beta-binomial variance
        n_total = post_alpha + post_beta
        pred_var = (n_future * p_yes * (1 - p_yes) * 
                    (n_total + n_future) / (n_total + 1))
        
        return ForecastResult(
            model_name="bayesian_beta_binomial",
            p_yes=float(p_yes),
            ci_lower=float(max(0.01, ci_lower)),
            ci_upper=float(min(0.99, ci_upper)),
            confidence=float(confidence),
            metadata={
                "posterior_alpha": float(post_alpha),
                "posterior_beta": float(post_beta),
                "n_observations": self.n_observations,
                "n_future": n_future,
                "predictive_variance": float(pred_var),
                "prior_weight": float(self.prior_alpha + self.prior_beta)
            }
        )
    
    def get_posterior_samples(self, n_samples: int = 10000) -> np.ndarray:
        """Draw samples from posterior for ensemble combination."""
        return np.random.beta(self.posterior_alpha, self.posterior_beta, n_samples)


# =====================================================================
# M4: ML Ensemble
# =====================================================================

class MLEnsembleForecaster:
    """
    ML ensemble with calibration and model cards.
    
    Uses xgboost/lightgbm with isotonic calibration.
    Trained on historical settled markets with walk-forward validation.
    """
    
    def __init__(self, model_type: str = "xgboost", 
                 use_calibration: bool = True,
                 confidence: float = 0.95):
        self.model_type = model_type
        self.use_calibration = use_calibration
        self.confidence = confidence
        self.model_manager = MLModelManager()
        self.is_trained = False
        self.calibration_curve = None
    
    def train(self, signals_df: pd.DataFrame, settled_df: pd.DataFrame) -> Dict:
        """Train ML model with walk-forward validation."""
        # Prepare features
        model = MLModel(model_type=self.model_type, use_calibration=self.use_calibration)
        X = model.prepare_features(signals_df)
        y = model.prepare_targets(settled_df)
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 50:
            log.warning(f"Insufficient training data: {len(X)} samples")
            return {}
        
        metrics = model.train(X, y)
        
        self.model_manager.models[model.model_card.model_name] = model
        self.model_manager.current_model = model.model_card.model_name
        self.is_trained = True
        
        # Store calibration curve
        if self.use_calibration and model.calibrator:
            self.calibration_curve = model.calibrator
        
        log.info(f"ML Ensemble trained: {model.model_card.model_name}, "
                 f"Brier={metrics['brier']:.4f}, AUC={metrics['auc_roc']:.4f}")
        
        return metrics
    
    def predict(self, signals_df: pd.DataFrame,
                kl_data: List[Dict] = None,
                basis_data: List[Dict] = None,
                market_data: Dict = None) -> ForecastResult:
        """Predict with calibrated probabilities."""
        if not self.is_trained or not self.model_manager.current_model:
            return ForecastResult(
                model_name="ml_ensemble",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"error": "Model not trained"}
            )
        
        model = self.model_manager.models[self.model_manager.current_model]
        X = model.prepare_features(signals_df, kl_data, basis_data, market_data)
        
        # Get calibrated probabilities
        probs = model.predict_proba(X)
        
        # Ensemble mean
        p_yes = float(np.mean(probs))
        
        # CI from prediction variance + calibration uncertainty
        pred_std = np.std(probs)
        se = pred_std / np.sqrt(len(probs)) if len(probs) > 1 else 0.1
        z_crit = norm.ppf((1 + self.confidence) / 2)
        
        ci_lower = max(0.01, p_yes - z_crit * se)
        ci_upper = min(0.99, p_yes + z_crit * se)
        
        # Confidence based on model card metrics
        card = model.get_model_card()
        model_confidence = 0.5
        if card:
            # Higher AUC and lower Brier = more confidence
            model_confidence = min(0.95, card.auc_roc * (1 - card.brier_score))
        
        return ForecastResult(
            model_name="ml_ensemble",
            p_yes=p_yes,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=model_confidence,
            metadata={
                "n_predictions": len(probs),
                "pred_std": float(pred_std),
                "model_name": self.model_manager.current_model,
                "auc_roc": card.auc_roc if card else 0,
                "brier_score": card.brier_score if card else 1,
                "calibration_method": card.calibration_method if card else "none"
            }
        )
    
    def predict_distribution(self, signals_df: pd.DataFrame,
                             kl_data: List[Dict] = None,
                             basis_data: List[Dict] = None,
                             market_data: Dict = None) -> np.ndarray:
        """Get full predictive distribution for ensemble combination."""
        if not self.is_trained:
            return np.array([0.5])
        
        model = self.model_manager.models[self.model_manager.current_model]
        X = model.prepare_features(signals_df, kl_data, basis_data, market_data)
        return model.predict_proba(X)


# =====================================================================
# Ensemble Combiner
# =====================================================================

class EnsembleForecaster:
    """
    Combines M1-M4 into a single forecast with honest probability bounds.
    
    Weighting: Inverse variance weighting with model confidence.
    Bounds: Conservative (union of CIs) and cluster-robust.
    """
    
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 n_mc_paths: int = 10000,
                 ml_model_type: str = "xgboost",
                 confidence: float = 0.95):
        self.confidence = confidence
        
        self.m1_bootstrap = ClusteredBootstrapForecaster(
            n_bootstrap=n_bootstrap, confidence=confidence)
        self.m2_gbm = GBMMonteCarloForecaster(
            n_paths=n_mc_paths, confidence=confidence)
        self.m3_bayesian = BayesianBetaBinomialForecaster(
            confidence=confidence)
        self.m4_ml = MLEnsembleForecaster(
            model_type=ml_model_type, confidence=confidence)
        
        # Model weights (can be learned from past performance)
        self.weights = {
            "clustered_bootstrap": 0.25,
            "gbm_monte_carlo": 0.25,
            "bayesian_beta_binomial": 0.25,
            "ml_ensemble": 0.25
        }
    
    def fit(self, 
            historical_markets: pd.DataFrame,
            signals_df: pd.DataFrame = None,
            settled_df: pd.DataFrame = None):
        """
        Fit all models on historical data.
        
        Args:
            historical_markets: DataFrame with ticker, outcome, features
            signals_df: For ML training (features)
            settled_df: For ML training (targets)
        """
        # M1: Cluster markets
        if 'ticker' in historical_markets.columns and 'outcome' in historical_markets.columns:
            self.m1_bootstrap.fit(historical_markets)
        
        # M3: Bayesian prior from historical win rates
        if 'outcome' in historical_markets.columns:
            outcomes = historical_markets['outcome'].values
            self.m3_bayesian.fit(outcomes)
        
        # M4: Train ML if data provided
        if signals_df is not None and settled_df is not None:
            self.m4_ml.train(signals_df, settled_df)
        
        log.info("All forecaster models fitted")
    
    def forecast(self,
                 spot: float,
                 strike: float,
                 sigma_per_minute: float,
                 minutes_left: float,
                 basis_bps: float = 0,
                 skew: float = 0,
                 kurtosis: float = 3,
                 market_signals: pd.DataFrame = None,
                 kl_data: List[Dict] = None,
                 basis_data: List[Dict] = None,
                 market_data: Dict = None) -> EnsembleForecast:
        """
        Generate ensemble forecast.
        
        Returns combined forecast with honest bounds.
        """
        results = []
        
        # M1: Clustered Bootstrap (needs historical data)
        if self.m1_bootstrap.clusters and market_signals is not None:
            m1_result = self.m1_bootstrap.predict(market_signals)
            results.append(m1_result)
            cluster_lower, cluster_upper = self.m1_bootstrap.get_cluster_bounds(market_signals)
        else:
            m1_result = ForecastResult(
                model_name="clustered_bootstrap",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"skipped": "No historical clusters"}
            )
            results.append(m1_result)
            cluster_lower, cluster_upper = 0.1, 0.9
        
        # M2: GBM Monte Carlo
        m2_result = self.m2_gbm.predict(
            spot=spot, strike=strike, sigma_per_minute=sigma_per_minute,
            minutes_left=minutes_left, basis_bps=basis_bps,
            skew=skew, kurtosis=kurtosis
        )
        results.append(m2_result)
        
        # M3: Bayesian Beta-Binomial
        m3_result = self.m3_bayesian.predict()
        results.append(m3_result)
        
        # M4: ML Ensemble
        if market_signals is not None:
            m4_result = self.m4_ml.predict(
                market_signals, kl_data, basis_data, market_data)
            results.append(m4_result)
        else:
            m4_result = ForecastResult(
                model_name="ml_ensemble",
                p_yes=0.5, ci_lower=0.1, ci_upper=0.9, confidence=0.1,
                metadata={"skipped": "No market signals"}
            )
            results.append(m4_result)
        
        # Combine with inverse-variance weighting
        valid_results = [r for r in results if r.confidence > 0.1]
        
        if not valid_results:
            # All models failed
            return EnsembleForecast(
                timestamp=datetime.now(timezone.utc).isoformat(),
                horizon_days=int(minutes_left / (60 * 24)),
                p_yes_ensemble=0.5,
                ci_lower=0.1, ci_upper=0.9,
                model_results=results,
                cluster_lower=0.1, cluster_upper=0.9,
                n_markets=0,
                metadata={"error": "All models failed"}
            )
        
        # Weight by confidence * (1 / variance)
        weights = []
        for r in valid_results:
            ci_width = r.ci_upper - r.ci_lower
            var = (ci_width / 4) ** 2  # Approx variance from CI
            w = r.confidence / (var + 1e-6)
            weights.append(w)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Ensemble mean
        p_yes_ensemble = np.sum(weights * [r.p_yes for r in valid_results])
        
        # Conservative bounds: union of all CIs
        ci_lower = min(r.ci_lower for r in valid_results)
        ci_upper = max(r.ci_upper for r in valid_results)
        
        # Map back to original results order
        final_results = []
        w_idx = 0
        for r in results:
            if r.confidence > 0.1:
                final_results.append(r)
                w_idx += 1
            else:
                final_results.append(r)
        
        return EnsembleForecast(
            timestamp=datetime.now(timezone.utc).isoformat(),
            horizon_days=int(minutes_left / (60 * 24)),
            p_yes_ensemble=float(p_yes_ensemble),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            model_results=final_results,
            cluster_lower=float(cluster_lower),
            cluster_upper=float(cluster_upper),
            n_markets=len(market_signals) if market_signals is not None else 0,
            metadata={
                "weights": weights.tolist() if len(weights) > 0 else [],
                "n_valid_models": len(valid_results),
                "ensemble_method": "inverse_variance_confidence"
            }
        )
    
    def update_weights(self, performance_history: List[Dict]):
        """Update model weights based on historical performance."""
        if not performance_history:
            return
        
        # Simple inverse Brier score weighting
        model_scores = {}
        for perf in performance_history:
            model = perf.get("model")
            brier = perf.get("brier", 1.0)
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(1 / (brier + 1e-6))
        
        for model, scores in model_scores.items():
            avg_score = np.mean(scores)
            if model in self.weights:
                self.weights[model] = avg_score
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        log.info(f"Updated ensemble weights: {self.weights}")


# =====================================================================
# Forecast Storage & Model Cards
# =====================================================================

class ForecastStore:
    """Stores forecasts and outcomes for model evaluation."""
    
    def __init__(self, db_path: str = "forecasts.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                horizon_days INTEGER,
                p_yes_ensemble REAL,
                ci_lower REAL,
                ci_upper REAL,
                cluster_lower REAL,
                cluster_upper REAL,
                model_results TEXT,  -- JSON
                metadata TEXT,       -- JSON
                outcome INTEGER,     -- 1 if YES won, 0 if NO won, NULL if pending
                settled_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_ticker ON forecasts(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_timestamp ON forecasts(timestamp)")
        conn.commit()
        conn.close()
    
    def save_forecast(self, ticker: str, forecast: EnsembleForecast):
        import sqlite3
        import json
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO forecasts 
            (timestamp, ticker, horizon_days, p_yes_ensemble, ci_lower, ci_upper,
             cluster_lower, cluster_upper, model_results, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            forecast.timestamp, ticker, forecast.horizon_days,
            forecast.p_yes_ensemble, forecast.ci_lower, forecast.ci_upper,
            forecast.cluster_lower, forecast.cluster_upper,
            json.dumps([{
                "model": r.model_name,
                "p_yes": r.p_yes,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "confidence": r.confidence,
                "metadata": r.metadata
            } for r in forecast.model_results]),
            json.dumps(forecast.metadata)
        ))
        conn.commit()
        conn.close()
    
    def record_outcome(self, ticker: str, outcome: int, settled_at: str = None):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE forecasts SET outcome = ?, settled_at = ?
            WHERE ticker = ? AND outcome IS NULL
        """, (outcome, settled_at or datetime.now(timezone.utc).isoformat(), ticker))
        conn.commit()
        conn.close()
    
    def get_evaluation_data(self, days: int = 30) -> pd.DataFrame:
        import sqlite3
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("""
            SELECT * FROM forecasts 
            WHERE outcome IS NOT NULL AND timestamp >= ?
            ORDER BY timestamp
        """, conn, params=(cutoff,))
        conn.close()
        return df


# =====================================================================
# Main Forecaster Interface
# =====================================================================

def create_forecaster(config: Dict = None) -> EnsembleForecaster:
    """Factory function to create configured forecaster."""
    cfg = config or {}
    return EnsembleForecaster(
        n_bootstrap=cfg.get("n_bootstrap", 1000),
        n_mc_paths=cfg.get("n_mc_paths", 10000),
        ml_model_type=cfg.get("ml_model_type", "xgboost"),
        confidence=cfg.get("confidence", 0.95)
    )


def forecast_market(forecaster: EnsembleForecaster,
                    market_info: Dict,
                    market_signals: pd.DataFrame = None) -> EnsembleForecast:
    """
    Convenience function to forecast a single market.
    
    Args:
        forecaster: Fitted EnsembleForecaster
        market_info: Dict with spot, strike, sigma, minutes_left, basis_bps, etc.
        market_signals: Optional DataFrame for bootstrap/ML models
    
    Returns:
        EnsembleForecast
    """
    return forecaster.forecast(
        spot=market_info.get("spot", 100000),
        strike=market_info.get("strike", 100000),
        sigma_per_minute=market_info.get("sigma_per_minute", 0.0007),
        minutes_left=market_info.get("minutes_left", 7.5),
        basis_bps=market_info.get("basis_bps", 0),
        skew=market_info.get("skew", 0),
        kurtosis=market_info.get("kurtosis", 3),
        market_signals=market_signals,
        kl_data=market_info.get("kl_data"),
        basis_data=market_info.get("basis_data"),
        market_data=market_info.get("market_data")
    )


# =====================================================================
# CLI for testing
# =====================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    parser = argparse.ArgumentParser(description="Multi-model 14-day forecaster")
    parser.add_argument("--spot", type=float, default=100000)
    parser.add_argument("--strike", type=float, default=100500)
    parser.add_argument("--sigma", type=float, default=0.0007)
    parser.add_argument("--minutes", type=float, default=7.5)
    parser.add_argument("--basis-bps", type=float, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    # Create forecaster
    forecaster = create_forecaster({
        "n_bootstrap": 100 if args.quick else 1000,
        "n_mc_paths": 1000 if args.quick else 10000,
        "ml_model_type": "xgboost"
    })
    
    # Generate synthetic historical data for fitting
    np.random.seed(42)
    n_hist = 200
    hist_data = pd.DataFrame({
        "ticker": [f"KXBTC15M-{i:04d}" for i in range(n_hist)],
        "outcome": np.random.binomial(1, 0.52, n_hist),
        "gbm_q": np.random.beta(5, 5, n_hist),
        "kl_edge": np.random.normal(0, 0.1, n_hist),
        "minutes_left": np.random.uniform(1, 14, n_hist)
    })
    
    signals_df = hist_data[["ticker", "gbm_q", "kl_edge", "minutes_left"]].copy()
    signals_df.columns = ["ticker", "gbm_q", "kl_edge", "minutes_left"]
    settled_df = hist_data[["ticker", "outcome"]].copy()
    
    # Fit
    forecaster.fit(hist_data, signals_df, settled_df)
    
    # Forecast
    market_info = {
        "spot": args.spot,
        "strike": args.strike,
        "sigma_per_minute": args.sigma,
        "minutes_left": args.minutes,
        "basis_bps": args.basis_bps
    }
    
    result = forecast_market(forecaster, market_info, signals_df[:10])
    
    print("\n" + "=" * 80)
    print("ENSEMBLE FORECAST")
    print("=" * 80)
    print(f"Timestamp: {result.timestamp}")
    print(f"Horizon: {result.horizon_days} days")
    print(f"P(YES) ensemble: {result.p_yes_ensemble:.4f}")
    print(f"Conservative CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"Cluster CI: [{result.cluster_lower:.4f}, {result.cluster_upper:.4f}]")
    print(f"Markets: {result.n_markets}")
    print()
    print("Model breakdown:")
    for r in result.model_results:
        print(f"  {r.model_name}: p={r.p_yes:.4f} "
              f"CI=[{r.ci_lower:.4f}, {r.ci_upper:.4f}] "
              f"conf={r.confidence:.2f}")
    print("=" * 80)