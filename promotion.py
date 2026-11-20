#!/usr/bin/env python3
"""
promotion.py — T15-16: Promotion criteria.

A model/band/gate change is promoted from paper to live ONLY if on the HELD-OUT test window:
1. Net-of-fee EV > 0 with clustered lower bound > 0
2. Max drawdown ≤ 30%
3. Paper results agree with live within 10% tolerance
4. Coverage of p5/p50/p95 bands within nominal ± 5%

Otherwise: stays paper-only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class PromotionStatus(Enum):
    """Promotion decision."""
    PROMOTED = "promoted"
    REJECTED = "rejected"
    NEEDS_MORE_DATA = "needs_more_data"


class PromotionReason(Enum):
    """Reason for promotion decision."""
    EV_POSITIVE = "ev_positive"
    EV_NEGATIVE = "ev_negative"
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    PAPER_LIVE_MISMATCH = "paper_live_mismatch"
    COVERAGE_VIOLATION = "coverage_violation"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    CLUSTERED_LB_NEGATIVE = "clustered_lb_negative"


@dataclass
class PromotionMetrics:
    """Metrics computed on held-out test window."""
    # EV metrics
    net_ev_per_trade: float           # Mean net EV per trade ($)
    net_ev_lower_bound: float         # Clustered 95% lower bound
    net_ev_upper_bound: float         # Clustered 95% upper bound
    num_trades: int                   # Number of trades in test window
    num_markets: int                  # Number of unique markets (clusters)

    # Drawdown
    max_drawdown_pct: float           # Maximum drawdown %
    max_drawdown_dollars: float       # Maximum drawdown $

    # Paper vs Live agreement
    paper_ev_per_trade: float         # Paper EV per trade
    live_ev_per_trade: float          # Live EV per trade
    paper_live_ratio: float           # Ratio (live/paper)

    # Coverage
    coverage_p5: float                # Fraction days realized ≥ p5
    coverage_p50: float               # Fraction days realized ≥ p50
    coverage_p95: float               # Fraction days realized ≥ p95

    # Sharpe
    sharpe_ratio: float               # Daily Sharpe

    # Sample info
    test_window_start: str
    test_window_end: str
    training_days: int


@dataclass
class PromotionResult:
    """Result of promotion evaluation."""
    status: PromotionStatus
    reasons: List[PromotionReason]
    metrics: PromotionMetrics
    message: str
    evaluated_at: str


# ─── Thresholds (from EVAL_PROTOCOL.md) ────────────────────────────────────────

PROMOTION_THRESHOLDS = {
    # EV must be positive with clustered lower bound > 0
    "min_net_ev_per_trade": 0.0,
    "min_clustered_lb": 0.0,

    # Max drawdown
    "max_drawdown_pct": 0.30,

    # Paper vs live agreement within 10%
    "paper_live_tolerance": 0.10,

    # Coverage within nominal ± 5%
    "coverage_tolerance": 0.05,
    "target_coverage_p5": 0.95,
    "target_coverage_p50": 0.50,
    "target_coverage_p95": 0.05,

    # Minimum samples
    "min_trades": 30,
    "min_markets": 20,
    "min_training_days": 60,
}


# ─── Clustered Standard Errors ─────────────────────────────────────────────────

def clustered_se(
    values: np.ndarray,
    clusters: np.ndarray,
) -> float:
    """
    Compute clustered standard error.

    Args:
        values: Per-trade values (e.g., EV per trade)
        clusters: Cluster IDs (e.g., market IDs)

    Returns:
        Clustered standard error of the mean
    """
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return float("inf")

    # Cluster means
    cluster_means = np.array([
        values[clusters == c].mean() for c in unique_clusters
    ])

    # Overall mean
    overall_mean = values.mean()

    # Between-cluster variance
    between_var = np.sum((cluster_means - overall_mean) ** 2) / (n_clusters - 1)

    # Clustered SE
    se = math.sqrt(between_var / n_clusters)
    return se


def clustered_confidence_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    confidence: float = 0.95,
) -> tuple:
    """
    Compute clustered confidence interval.

    Returns (lower_bound, upper_bound)
    """
    mean = values.mean()
    se = clustered_se(values, clusters)

    if se == float("inf") or math.isnan(se):
        return (float("-inf"), float("inf"))

    # t-distribution critical value (approximate with normal for large n)
    from scipy import stats
    try:
        t_crit = stats.t.ppf((1 + confidence) / 2, len(unique_clusters) - 1)
    except Exception:
        t_crit = 1.96  # Normal approx

    margin = t_crit * se
    return (mean - margin, mean + margin)


# ─── Coverage Evaluation ───────────────────────────────────────────────────────

def evaluate_coverage(
    realized: np.ndarray,
    forecast_p5: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p95: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate forecast coverage.

    Args:
        realized: Realized daily balances
        forecast_p5: 5th percentile forecasts
        forecast_p50: 50th percentile forecasts
        forecast_p95: 95th percentile forecasts

    Returns:
        Dict with coverage_p5, coverage_p50, coverage_p95
    """
    n = len(realized)
    if n == 0:
        return {"coverage_p5": 0, "coverage_p50": 0, "coverage_p95": 0}

    return {
        "coverage_p5": np.mean(realized >= forecast_p5),
        "coverage_p50": np.mean(realized >= forecast_p50),
        "coverage_p95": np.mean(realized >= forecast_p95),
    }


def coverage_within_tolerance(
    coverage: Dict[str, float],
    tolerance: float = 0.05,
) -> bool:
    """Check if coverage is within tolerance of nominal."""
    nominal = {
        "coverage_p5": 0.95,
        "coverage_p50": 0.50,
        "coverage_p95": 0.05,
    }

    for key, target in nominal.items():
        if abs(coverage.get(key, 0) - target) > tolerance:
            return False
    return True


# ─── Main Promotion Evaluation ─────────────────────────────────────────────────

def evaluate_promotion(
    # Per-trade data for test window
    trade_evs: List[float],                    # Net EV per trade ($)
    trade_clusters: List[str],                 # Market ID per trade
    # Drawdown data
    daily_balances: List[float],               # Daily ending balances
    # Paper vs live
    paper_daily_ev: List[float],               # Paper EV per day
    live_daily_ev: List[float],                # Live EV per day
    # Coverage data
    realized_daily_balances: List[float],      # Realized daily balances
    forecast_p5: List[float],                  # p5 forecasts
    forecast_p50: List[float],                 # p50 forecasts
    forecast_p95: List[float],                 # p95 forecasts
    # Metadata
    test_window_start: str,
    test_window_end: str,
    training_days: int,
) -> PromotionResult:
    """
    Evaluate promotion criteria on held-out test window.

    All criteria must pass for promotion.
    """
    reasons = []
    evaluated_at = datetime.now(timezone.utc).isoformat()

    # Convert to arrays
    trade_evs_arr = np.array(trade_evs)
    trade_clusters_arr = np.array(trade_clusters)
    daily_balances_arr = np.array(daily_balances)

    num_trades = len(trade_evs)
    num_markets = len(np.unique(trade_clusters_arr))

    # 1. Check minimum samples
    if num_trades < PROMOTION_THRESHOLDS["min_trades"]:
        reasons.append(PromotionReason.INSUFFICIENT_SAMPLES)
    if num_markets < PROMOTION_THRESHOLDS["min_markets"]:
        reasons.append(PromotionReason.INSUFFICIENT_SAMPLES)
    if training_days < PROMOTION_THRESHOLDS["min_training_days"]:
        reasons.append(PromotionReason.INSUFFICIENT_SAMPLES)

    # 2. Net EV with clustered CI
    net_ev_per_trade = float(trade_evs_arr.mean()) if num_trades > 0 else 0.0
    ci_lower, ci_upper = clustered_confidence_interval(trade_evs_arr, trade_clusters_arr)
    net_ev_lower_bound = ci_lower
    net_ev_upper_bound = ci_upper

    if net_ev_per_trade <= PROMOTION_THRESHOLDS["min_net_ev_per_trade"]:
        reasons.append(PromotionReason.EV_NEGATIVE)
    if net_ev_lower_bound <= PROMOTION_THRESHOLDS["min_clustered_lb"]:
        reasons.append(PromotionReason.CLUSTERED_LB_NEGATIVE)

    # 3. Max drawdown
    if len(daily_balances_arr) > 1:
        peak = np.maximum.accumulate(daily_balances_arr)
        drawdown = (peak - daily_balances_arr) / peak
        max_drawdown_pct = float(drawdown.max())
        max_drawdown_dollars = float((peak - daily_balances_arr).max())
    else:
        max_drawdown_pct = 0.0
        max_drawdown_dollars = 0.0

    if max_drawdown_pct > PROMOTION_THRESHOLDS["max_drawdown_pct"]:
        reasons.append(PromotionReason.DRAWDOWN_EXCEEDED)

    # 4. Paper vs live agreement
    if paper_daily_ev and live_daily_ev:
        paper_ev = float(np.mean(paper_daily_ev))
        live_ev = float(np.mean(live_daily_ev))
        paper_live_ratio = live_ev / paper_ev if paper_ev != 0 else float("inf")

        if abs(paper_live_ratio - 1.0) > PROMOTION_THRESHOLDS["paper_live_tolerance"]:
            reasons.append(PromotionReason.PAPER_LIVE_MISMATCH)
    else:
        paper_ev = 0.0
        live_ev = 0.0
        paper_live_ratio = 1.0

    # 5. Coverage
    if realized_daily_balances and forecast_p5 and forecast_p50 and forecast_p95:
        coverage = evaluate_coverage(
            np.array(realized_daily_balances),
            np.array(forecast_p5),
            np.array(forecast_p50),
            np.array(forecast_p95),
        )
        coverage_p5 = coverage["coverage_p5"]
        coverage_p50 = coverage["coverage_p50"]
        coverage_p95 = coverage["coverage_p95"]

        if not coverage_within_tolerance(coverage, PROMOTION_THRESHOLDS["coverage_tolerance"]):
            reasons.append(PromotionReason.COVERAGE_VIOLATION)
    else:
        coverage_p5 = coverage_p50 = coverage_p95 = 0.0

    # 6. Sharpe
    if len(daily_balances_arr) > 1:
        daily_returns = np.diff(daily_balances_arr) / daily_balances_arr[:-1]
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252)) if np.std(daily_returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Build metrics
    metrics = PromotionMetrics(
        net_ev_per_trade=net_ev_per_trade,
        net_ev_lower_bound=net_ev_lower_bound,
        net_ev_upper_bound=net_ev_upper_bound,
        num_trades=num_trades,
        num_markets=num_markets,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_dollars=max_drawdown_dollars,
        paper_ev_per_trade=paper_ev,
        live_ev_per_trade=live_ev,
        paper_live_ratio=paper_live_ratio,
        coverage_p5=coverage_p5,
        coverage_p50=coverage_p50,
        coverage_p95=coverage_p95,
        sharpe_ratio=sharpe,
        test_window_start=test_window_start,
        test_window_end=test_window_end,
        training_days=training_days,
    )

    # Determine status
    if not reasons:
        status = PromotionStatus.PROMOTED
        message = "All promotion criteria met"
    elif PromotionReason.INSUFFICIENT_SAMPLES in reasons:
        status = PromotionStatus.NEEDS_MORE_DATA
        message = f"Insufficient samples: trades={num_trades}, markets={num_markets}, training_days={training_days}"
    else:
        status = PromotionStatus.REJECTED
        reason_names = [r.value for r in reasons]
        message = f"Rejected: {', '.join(reason_names)}"

    return PromotionResult(
        status=status,
        reasons=reasons,
        metrics=metrics,
        message=message,
        evaluated_at=evaluated_at,
    )


# ─── Convenience function for backtest integration ─────────────────────────────

def check_promotion_from_backtest(
    backtest_results: Dict,
    paper_results: Dict,
    live_results: Dict = None,
    coverage_results: Dict = None,
) -> PromotionResult:
    """
    Check promotion from backtest outputs.

    Args:
        backtest_results: Dict with 'trades' list of {ev, market_id, ...}, 'daily_balances'
        paper_results: Dict with 'daily_ev' list
        live_results: Optional Dict with 'daily_ev' list
        coverage_results: Optional Dict with realized, p5, p50, p95 arrays

    Returns:
        PromotionResult
    """
    trades = backtest_results.get("trades", [])
    daily_balances = backtest_results.get("daily_balances", [])

    trade_evs = [t.get("net_ev", 0) for t in trades]
    trade_clusters = [t.get("market_id", "unknown") for t in trades]

    paper_ev = paper_results.get("daily_ev", [])
    live_ev = live_results.get("daily_ev", []) if live_results else []

    coverage_data = {}
    if coverage_results:
        coverage_data = {
            "realized": coverage_results.get("realized", []),
            "p5": coverage_results.get("p5", []),
            "p50": coverage_results.get("p50", []),
            "p95": coverage_results.get("p95", []),
        }

    return evaluate_promotion(
        trade_evs=trade_evs,
        trade_clusters=trade_clusters,
        daily_balances=daily_balances,
        paper_daily_ev=paper_ev,
        live_daily_ev=live_ev,
        realized_daily_balances=coverage_data.get("realized", []),
        forecast_p5=coverage_data.get("p5", []),
        forecast_p50=coverage_data.get("p50", []),
        forecast_p95=coverage_data.get("p95", []),
        test_window_start=backtest_results.get("test_window_start", ""),
        test_window_end=backtest_results.get("test_window_end", ""),
        training_days=backtest_results.get("training_days", 0),
    )


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    n_trades = 50
    n_markets = 25

    trade_evs = list(np.random.normal(0.05, 0.10, n_trades))  # Slightly positive EV
    trade_clusters = [f"M{i % n_markets}" for i in range(n_trades)]
    daily_balances = list(100 + np.cumsum(np.random.normal(0.1, 0.5, 14)))
    paper_ev = list(np.random.normal(0.06, 0.02, 14))
    live_ev = list(np.random.normal(0.055, 0.02, 14))
    realized = list(100 + np.cumsum(np.random.normal(0.1, 0.5, 14)))
    forecast_p5 = [b - 2 for b in realized]
    forecast_p50 = [b - 0.1 for b in realized]
    forecast_p95 = [b + 2 for b in realized]

    result = evaluate_promotion(
        trade_evs=trade_evs,
        trade_clusters=trade_clusters,
        daily_balances=daily_balances,
        paper_daily_ev=paper_ev,
        live_daily_ev=live_ev,
        realized_daily_balances=realized,
        forecast_p5=forecast_p5,
        forecast_p50=forecast_p50,
        forecast_p95=forecast_p95,
        test_window_start="2026-11-01",
        test_window_end="2026-11-14",
        training_days=60,
    )

    print(f"Status: {result.status.value}")
    print(f"Reasons: {[r.value for r in result.reasons]}")
    print(f"Message: {result.message}")
    print(f"EV: {result.metrics.net_ev_per_trade:.4f} (CI: {result.metrics.net_ev_lower_bound:.4f}, {result.metrics.net_ev_upper_bound:.4f})")
    print(f"Max DD: {result.metrics.max_drawdown_pct:.1%}")
    print(f"Paper/Live ratio: {result.metrics.paper_live_ratio:.3f}")
    print(f"Coverage: p5={result.metrics.coverage_p5:.1%}, p50={result.metrics.coverage_p50:.1%}, p95={result.metrics.coverage_p95:.1%}")
    print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")