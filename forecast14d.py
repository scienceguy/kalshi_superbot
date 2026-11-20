#!/usr/bin/env python3
"""
forecast14d.py — T15-52: 14-day bankroll forecast with 7 model families.

Implements the 7 model families from EVAL_PROTOCOL.md:
1. trade_bootstrap — resample per-trade P&L with replacement
2. daily_block_bootstrap — resample daily P&L blocks (preserves autocorrelation)
3. gbm_lognormal — Geometric Brownian Motion with lognormal returns
4. regime_switching — 2-state HMM (low/high vol) with GBM in each state
5. fat_tail — Student-t returns with fitted DoF
6. ml_quantile — XGBoost quantile regression on features (vol, skew, trend, time)
7. stress_* — Fee +25%, Fill -25%, Edge -50%

Outputs per EVAL_PROTOCOL.md: median, p5, p25, p75, p95, p_target($3.25M),
p_drawdown_30, p_ruin, daily_growth_required.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ForecastConfig:
    """Configuration for 14-day forecast."""
    # Current state
    current_balance: float = 10.0
    target_balance: float = 3_250_000.0

    # Simulation
    n_simulations: int = 10_000
    horizon_days: int = 14
    seed: int = 42

    # Model weights (for ensemble)
    model_weights: Dict[str, float] = None

    # Data
    historical_trades: List[Dict] = None
    historical_daily_pnl: List[float] = None
    historical_daily_balances: List[float] = None

    # Fee/fill/edge assumptions
    base_fee_rate: float = 0.07
    base_fill_prob: float = 0.30
    base_edge_cents: float = 5.0

    # Stress scenarios
    stress_fee_mult: float = 1.25
    stress_fill_mult: float = 0.75
    stress_edge_mult: float = 0.50

    # Output
    out_dir: str = "forecasts"
    save_details: bool = True


# ─── Model Family 1: Trade Bootstrap ──────────────────────────────────────────

def model_trade_bootstrap(
    trades: List[Dict],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
) -> np.ndarray:
    """
    Resample per-trade P&L with replacement.
    Projects forward by sampling trades per day.
    """
    np.random.seed(seed)

    if not trades:
        return np.full(n_sims, current_balance)

    # Extract per-trade P&L
    trade_pnls = np.array([t.get("pnl", 0) for t in trades])
    if len(trade_pnls) == 0:
        return np.full(n_sims, current_balance)

    # Estimate trades per day
    trades_per_day = max(1, len(trades) / max(1, horizon_days))

    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        for day in range(horizon_days):
            n_trades_today = max(1, int(np.random.poisson(trades_per_day)))
            sampled_pnls = np.random.choice(trade_pnls, size=n_trades_today, replace=True)
            balance += sampled_pnls.sum()
            if balance <= 0:
                balance = 0
                break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 2: Daily Block Bootstrap ────────────────────────────────────

def model_daily_block_bootstrap(
    daily_pnl: List[float],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
    block_size: int = 3,
) -> np.ndarray:
    """
    Resample daily P&L blocks to preserve autocorrelation.
    """
    np.random.seed(seed + 1)

    daily_pnl_arr = np.array(daily_pnl)
    n_days_hist = len(daily_pnl_arr)

    if n_days_hist < block_size:
        # Fall back to single-day bootstrap
        block_size = max(1, n_days_hist)

    n_blocks = max(1, (n_days_hist + block_size - 1) // block_size)

    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        days_simulated = 0
        while days_simulated < horizon_days and balance > 0:
            # Sample a block
            start_idx = np.random.randint(0, n_days_hist - block_size + 1)
            block = daily_pnl_arr[start_idx:start_idx + block_size]
            for pnl in block:
                balance += pnl
                days_simulated += 1
                if balance <= 0:
                    balance = 0
                    break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 3: GBM Lognormal ────────────────────────────────────────────

def model_gbm_lognormal(
    daily_returns: List[float],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
) -> np.ndarray:
    """
    Geometric Brownian Motion with lognormal returns.
    Fits mu and sigma from historical daily returns.
    """
    np.random.seed(seed + 2)

    if not daily_returns or len(daily_returns) < 2:
        # Default params
        mu = 0.001
        sigma = 0.02
    else:
        returns_arr = np.array(daily_returns)
        # Remove outliers
        q1, q99 = np.percentile(returns_arr, [1, 99])
        clean_returns = returns_arr[(returns_arr >= q1) & (returns_arr <= q99)]
        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        if sigma == 0:
            sigma = 0.02

    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        for day in range(horizon_days):
            # Log return ~ N(mu, sigma)
            log_ret = np.random.normal(mu, sigma)
            balance *= math.exp(log_ret)
            if balance <= 0:
                balance = 0
                break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 4: Regime Switching (2-state HMM) ───────────────────────────

def model_regime_switching(
    daily_returns: List[float],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
) -> np.ndarray:
    """
    2-state regime switching model (low vol / high vol).
    Uses simple threshold-based regime detection.
    """
    np.random.seed(seed + 3)

    if not daily_returns or len(daily_returns) < 10:
        return model_gbm_lognormal(daily_returns, n_sims, horizon_days, current_balance, seed + 3)

    returns_arr = np.array(daily_returns)

    # Simple regime detection: split by volatility
    vol_threshold = np.percentile(np.abs(returns_arr - np.mean(returns_arr)), 66)
    regime = np.where(np.abs(returns_arr - np.mean(returns_arr)) > vol_threshold, 1, 0)  # 0=low, 1=high

    # Fit params per regime
    low_returns = returns_arr[regime == 0]
    high_returns = returns_arr[regime == 1]

    if len(low_returns) < 2:
        low_returns = returns_arr
    if len(high_returns) < 2:
        high_returns = returns_arr * 2  # Higher vol

    mu_low, sigma_low = np.mean(low_returns), np.std(low_returns)
    mu_high, sigma_high = np.mean(high_returns), np.std(high_returns)

    if sigma_low == 0: sigma_low = 0.01
    if sigma_high == 0: sigma_high = 0.03

    # Regime transition matrix (estimated from data)
    n_trans = len(regime) - 1
    if n_trans > 0:
        p_00 = np.sum((regime[:-1] == 0) & (regime[1:] == 0)) / np.sum(regime[:-1] == 0) if np.sum(regime[:-1] == 0) > 0 else 0.9
        p_11 = np.sum((regime[:-1] == 1) & (regime[1:] == 1)) / np.sum(regime[:-1] == 1) if np.sum(regime[:-1] == 1) > 0 else 0.8
    else:
        p_00, p_11 = 0.9, 0.8

    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        current_regime = 0  # Start in low vol
        for day in range(horizon_days):
            # Transition
            if current_regime == 0:
                if np.random.random() > p_00:
                    current_regime = 1
            else:
                if np.random.random() > p_11:
                    current_regime = 0

            # Generate return
            if current_regime == 0:
                log_ret = np.random.normal(mu_low, sigma_low)
            else:
                log_ret = np.random.normal(mu_high, sigma_high)

            balance *= math.exp(log_ret)
            if balance <= 0:
                balance = 0
                break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 5: Fat Tail (Student-t) ─────────────────────────────────────

def model_fat_tail(
    daily_returns: List[float],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
) -> np.ndarray:
    """
    Student-t returns with fitted degrees of freedom.
    Captures fat tails better than Gaussian.
    """
    np.random.seed(seed + 4)

    if not daily_returns or len(daily_returns) < 5:
        return model_gbm_lognormal(daily_returns, n_sims, horizon_days, current_balance, seed + 4)

    returns_arr = np.array(daily_returns)
    mu = np.mean(returns_arr)
    sigma = np.std(returns_arr)

    # Fit DoF using method of moments (kurtosis)
    # Kurtosis of t-distribution = 6/(df-4) for df > 4
    excess_kurt = np.mean(((returns_arr - mu) / sigma) ** 4) - 3
    if excess_kurt > 0:
        df = 4 + 6 / excess_kurt
        df = max(4.1, min(df, 30))  # Clamp
    else:
        df = 10  # Default moderately fat tail

    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        for day in range(horizon_days):
            # Student-t return
            t_sample = np.random.standard_t(df)
            log_ret = mu + sigma * t_sample * math.sqrt((df - 2) / df)  # Scale to match variance
            balance *= math.exp(log_ret)
            if balance <= 0:
                balance = 0
                break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 6: ML Quantile Regression ───────────────────────────────────

def model_ml_quantile(
    trades: List[Dict],
    n_sims: int,
    horizon_days: int,
    current_balance: float,
    seed: int,
) -> np.ndarray:
    """
    XGBoost quantile regression on features.
    Features: vol, skew, trend, time-to-settlement, entry_band, strategy_mode.
    """
    np.random.seed(seed + 5)

    if not trades or len(trades) < 20:
        # Fallback to trade bootstrap
        return model_trade_bootstrap(trades, n_sims, horizon_days, current_balance, seed + 5)

    try:
        import xgboost as xgb
    except ImportError:
        log.warning("XGBoost not available, falling back to trade bootstrap")
        return model_trade_bootstrap(trades, n_sims, horizon_days, current_balance, seed + 5)

    # Build feature matrix
    X = []
    y = []
    for t in trades:
        features = [
            t.get("entry_price", 50) / 100.0,  # Entry price as fraction
            t.get("minutes_in", 10) / 15.0,    # Time in window
            1.0 if t.get("direction") == "YES" else 0.0,
            t.get("contracts", 1.0),
        ]
        X.append(features)
        y.append(t.get("pnl", 0))

    X = np.array(X)
    y = np.array(y)

    # Train quantile models for p5, p50, p95
    quantiles = [0.05, 0.50, 0.95]
    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective=f"reg:quantileerror",
            quantile_alpha=q,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X, y)
        models[q] = model

    # For simulation, sample from the conditional distribution
    # Use p50 as mean, (p95-p5)/2 as scale
    final_balances = np.zeros(n_sims)
    for i in range(n_sims):
        balance = current_balance
        for day in range(horizon_days):
            # Sample a random trade's features
            idx = np.random.randint(len(trades))
            feat = X[idx:idx+1]
            p5 = models[0.05].predict(feat)[0]
            p50 = models[0.50].predict(feat)[0]
            p95 = models[0.95].predict(feat)[0]

            # Sample from asymmetric distribution
            u = np.random.random()
            if u < 0.05:
                pnl = p5 + np.random.normal(0, abs(p50 - p5) * 0.1)
            elif u < 0.50:
                pnl = p5 + (u - 0.05) / 0.45 * (p50 - p5)
            elif u < 0.95:
                pnl = p50 + (u - 0.50) / 0.45 * (p95 - p50)
            else:
                pnl = p95 + np.random.normal(0, abs(p95 - p50) * 0.1)

            balance += pnl
            if balance <= 0:
                balance = 0
                break
        final_balances[i] = balance

    return final_balances


# ─── Model Family 7: Stress Scenarios ─────────────────────────────────────────

def model_stress(
    base_balances: np.ndarray,
    stress_type: str,
    config: ForecastConfig,
) -> np.ndarray:
    """
    Apply stress scenarios to base simulation results.

    Types:
    - "fee_plus_25": Increase fees by 25%
    - "fill_minus_25": Reduce fill probability by 25%
    - "edge_minus_50": Reduce edge by 50%
    """
    if stress_type == "fee_plus_25":
        # Fees reduce PnL: scale down by (1 + 0.25 * fee_rate) approx
        fee_impact = 1.0 / (1.0 + 0.25 * config.base_fee_rate)
        return base_balances * fee_impact

    elif stress_type == "fill_minus_25":
        # Fewer fills = fewer trades = less PnL
        fill_impact = 1.0 - 0.25 * config.base_fill_prob
        return config.current_balance + (base_balances - config.current_balance) * fill_impact

    elif stress_type == "edge_minus_50":
        # Half the edge
        edge_impact = 0.5
        return config.current_balance + (base_balances - config.current_balance) * edge_impact

    else:
        return base_balances


# ─── Ensemble & Metrics ───────────────────────────────────────────────────────

@dataclass
class ForecastMetrics:
    """Metrics per EVAL_PROTOCOL.md."""
    median_balance: float
    p5_balance: float
    p25_balance: float
    p75_balance: float
    p95_balance: float
    p_target: float              # P(balance >= target)
    p_drawdown_30: float         # P(max DD > 30%)
    p_ruin: float                # P(balance <= 0)
    daily_growth_required: float # Daily growth rate needed

    # Per-model breakdown
    model_balances: Dict[str, np.ndarray] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.model_balances:
            d["model_balances"] = {k: v.tolist() for k, v in self.model_balances.items()}
        return d


def compute_metrics(
    balances: np.ndarray,
    current_balance: float,
    target_balance: float,
    daily_balances_sims: Optional[np.ndarray] = None,
) -> ForecastMetrics:
    """Compute all required metrics from simulation balances."""
    if len(balances) == 0:
        return ForecastMetrics(
            median_balance=current_balance,
            p5_balance=current_balance,
            p25_balance=current_balance,
            p75_balance=current_balance,
            p95_balance=current_balance,
            p_target=0.0,
            p_drawdown_30=0.0,
            p_ruin=1.0 if current_balance <= 0 else 0.0,
            daily_growth_required=0.0,
        )

    median_bal = float(np.median(balances))
    p5_bal = float(np.percentile(balances, 5))
    p25_bal = float(np.percentile(balances, 25))
    p75_bal = float(np.percentile(balances, 75))
    p95_bal = float(np.percentile(balances, 95))

    p_target = float(np.mean(balances >= target_balance))
    p_ruin = float(np.mean(balances <= 0))

    # Daily growth required
    if current_balance > 0 and target_balance > current_balance:
        daily_growth = (target_balance / current_balance) ** (1.0 / 14) - 1
    else:
        daily_growth = 0.0

    # Max drawdown > 30% - need daily paths
    if daily_balances_sims is not None and daily_balances_sims.ndim == 2:
        # daily_balances_sims: (n_sims, n_days+1) including starting balance
        max_dd_pct = []
        for sim in daily_balances_sims:
            peak = np.maximum.accumulate(sim)
            dd = (peak - sim) / peak
            max_dd = np.max(dd)
            max_dd_pct.append(max_dd)
        p_drawdown_30 = float(np.mean(np.array(max_dd_pct) > 0.30))
    else:
        # Approximate from final balance distribution
        # If final balance < 70% of start, likely had >30% DD
        p_drawdown_30 = float(np.mean(balances < 0.70 * current_balance))

    return ForecastMetrics(
        median_balance=median_bal,
        p5_balance=p5_bal,
        p25_balance=p25_bal,
        p75_balance=p75_bal,
        p95_balance=p95_bal,
        p_target=p_target,
        p_drawdown_30=p_drawdown_30,
        p_ruin=p_ruin,
        daily_growth_required=daily_growth,
    )


def run_forecast(config: ForecastConfig) -> Dict[str, ForecastMetrics]:
    """
    Run all 7 model families and return metrics.

    Returns dict mapping model name -> ForecastMetrics.
    """
    np.random.seed(config.seed)

    # Prepare data
    trades = config.historical_trades or []
    daily_pnl = config.historical_daily_pnl or []
    daily_balances = config.historical_daily_balances or []

    # Compute daily returns from balances
    daily_returns = []
    if len(daily_balances) > 1:
        for i in range(1, len(daily_balances)):
            if daily_balances[i-1] > 0:
                daily_returns.append((daily_balances[i] - daily_balances[i-1]) / daily_balances[i-1])

    n_sims = config.n_simulations
    horizon = config.horizon_days
    start_bal = config.current_balance

    # Run each model
    model_results = {}

    log.info("Running trade_bootstrap...")
    model_results["trade_bootstrap"] = model_trade_bootstrap(trades, n_sims, horizon, start_bal, config.seed)

    log.info("Running daily_block_bootstrap...")
    model_results["daily_block_bootstrap"] = model_daily_block_bootstrap(daily_pnl, n_sims, horizon, start_bal, config.seed)

    log.info("Running gbm_lognormal...")
    model_results["gbm_lognormal"] = model_gbm_lognormal(daily_returns, n_sims, horizon, start_bal, config.seed)

    log.info("Running regime_switching...")
    model_results["regime_switching"] = model_regime_switching(daily_returns, n_sims, horizon, start_bal, config.seed)

    log.info("Running fat_tail...")
    model_results["fat_tail"] = model_fat_tail(daily_returns, n_sims, horizon, start_bal, config.seed)

    log.info("Running ml_quantile...")
    model_results["ml_quantile"] = model_ml_quantile(trades, n_sims, horizon, start_bal, config.seed)

    # Stress scenarios (applied to trade_bootstrap as base)
    base = model_results["trade_bootstrap"]
    model_results["stress_fees_plus_25"] = model_stress(base, "fee_plus_25", config)
    model_results["stress_fill_minus_25"] = model_stress(base, "fill_minus_25", config)
    model_results["stress_edge_minus_50"] = model_stress(base, "edge_minus_50", config)

    # Compute metrics for each
    metrics = {}
    for name, balances in model_results.items():
        metrics[name] = compute_metrics(balances, start_bal, config.target_balance)
        metrics[name].model_balances = {name: balances}
        log.info(f"  {name}: median=${metrics[name].median_balance:.2f}, p_target={metrics[name].p_target:.1%}")

    # Ensemble (weighted average)
    if config.model_weights:
        ensemble_balances = np.zeros(n_sims)
        total_w = 0
        for name, weight in config.model_weights.items():
            if name in model_results:
                ensemble_balances += model_results[name] * weight
                total_w += weight
        if total_w > 0:
            ensemble_balances /= total_w
            metrics["ensemble"] = compute_metrics(ensemble_balances, start_bal, config.target_balance)
            metrics["ensemble"].model_balances = {"ensemble": ensemble_balances}
            log.info(f"  ensemble: median=${metrics['ensemble'].median_balance:.2f}, p_target={metrics['ensemble'].p_target:.1%}")

    return metrics


# ─── Output Formatting (EVAL_PROTOCOL.md) ─────────────────────────────────────

def format_forecast_output(
    metrics: Dict[str, ForecastMetrics],
    config: ForecastConfig,
    git_commit: str = "",
) -> Dict:
    """Format output per EVAL_PROTOCOL.md JSON schema."""
    return {
        "protocol_version": "1.0",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or "unknown",
        "seed": config.seed,
        "data_version": "historical_trades",
        "venues": {
            "kalshi": {name: m.to_dict() for name, m in metrics.items()},
        },
        "warnings": _check_warnings(config),
    }


def _check_warnings(config: ForecastConfig) -> List[str]:
    warnings_list = []
    if config.historical_trades and len(config.historical_trades) < 500:
        warnings_list.append(f"Sample size: {len(config.historical_trades)} trades < 500 minimum")
    if config.historical_daily_balances and len(config.historical_daily_balances) < 60:
        warnings_list.append(f"Sample size: {len(config.historical_daily_balances)} days < 60 day minimum")
    return warnings_list


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_forecast_pipeline(
    historical_data_path: str,
    current_balance: float = 10.0,
    target_balance: float = 3_250_000.0,
    n_simulations: int = 10_000,
    out_dir: str = "forecasts",
    git_commit: str = "",
) -> Dict:
    """
    Full pipeline: load data, run forecast, save results.
    """
    # Load historical data
    with open(historical_data_path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    daily_balances = data.get("daily_balances", [])

    # Compute daily PnL from balances
    daily_pnl = []
    if len(daily_balances) > 1:
        for i in range(1, len(daily_balances)):
            daily_pnl.append(daily_balances[i] - daily_balances[i-1])

    config = ForecastConfig(
        current_balance=current_balance,
        target_balance=target_balance,
        n_simulations=n_simulations,
        historical_trades=trades,
        historical_daily_pnl=daily_pnl,
        historical_daily_balances=daily_balances,
        out_dir=out_dir,
    )

    # Run forecast
    metrics = run_forecast(config)

    # Format output
    output = format_forecast_output(metrics, config, git_commit)

    # Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"Forecast saved to {out_file}")

    return output


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="14-day bankroll forecast")
    parser.add_argument("--data", required=True, help="Path to historical data JSON")
    parser.add_argument("--balance", type=float, default=10.0, help="Current balance")
    parser.add_argument("--target", type=float, default=3250000.0, help="Target balance")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--out", default="forecasts", help="Output directory")
    parser.add_argument("--commit", default="", help="Git commit hash")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

    result = run_forecast_pipeline(
        historical_data_path=args.data,
        current_balance=args.balance,
        target_balance=args.target,
        n_simulations=args.sims,
        out_dir=args.out,
        git_commit=args.commit,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("14-DAY FORECAST SUMMARY")
    print("=" * 80)
    for name, m in result.get("venues", {}).get("kalshi", {}).items():
        if name == "ensemble":
            print(f"\n{name.upper()} (ensemble):")
            print(f"  Median: ${m['median_balance']:,.2f}")
            print(f"  P5/P25/P75/P95: ${m['p5_balance']:,.2f} / ${m['p25_balance']:,.2f} / ${m['p75_balance']:,.2f} / ${m['p95_balance']:,.2f}")
            print(f"  P(target): {m['p_target']:.2%}")
            print(f"  P(DD>30%): {m['p_drawdown_30']:.2%}")
            print(f"  P(ruin): {m['p_ruin']:.2%}")
            print(f"  Daily growth needed: {m['daily_growth_required']:.2%}")


if __name__ == "__main__":
    main()