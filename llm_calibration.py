#!/usr/bin/env python3
"""
llm_calibration.py — T15-24: LLM evaluation calibration.

Outcome-linked LLM evaluation reporting:
- Stores LLM evals with actual settlement outcomes
- Computes calibration curves per model/provider
- Tracks Brier score, log-loss, reliability diagrams
- Feeds back into ensemble weights for fair_value
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class LLMEvalRecord:
    """Single LLM evaluation linked to outcome."""
    eval_id: str
    timestamp: str
    market_ticker: str
    window_start: str
    strike: float
    spot_at_eval: float
    minutes_left: float

    # LLM inputs
    prompt_version: str
    model_provider: str  # "openrouter", "groq", "deepseek", "ollama", etc.
    model_name: str
    prompt_template: str

    # LLM output
    llm_prob_yes: float          # LLM's predicted P(YES wins)
    llm_confidence: float        # LLM's self-reported confidence
    llm_reasoning: str
    llm_tokens: int

    # Market context at eval time
    entry_price_cents: int
    bid_ask_spread: int
    gbm_prob_yes: float
    volume: int

    # Actual outcome (filled after settlement)
    settled: bool = False
    yes_won: Optional[bool] = None
    settled_at: str = ""
    settlement_price: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMEvalRecord":
        return cls(**d)


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a model/provider."""
    model_key: str  # "provider:model"
    n_samples: int
    n_settled: int
    brier_score: float
    log_loss: float
    accuracy: float
    calibration_slope: float  # 1.0 = perfectly calibrated
    calibration_intercept: float  # 0.0 = perfectly calibrated
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_curve: List[Tuple[float, float]]  # (mean_pred, frac_pos)
    last_updated: str


# ─── Database ──────────────────────────────────────────────────────────────────

DB_PATH = Path("llm_evals.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_evals (
    eval_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    window_start TEXT NOT NULL,
    strike REAL NOT NULL,
    spot_at_eval REAL NOT NULL,
    minutes_left REAL NOT NULL,
    prompt_version TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_template TEXT NOT NULL,
    llm_prob_yes REAL NOT NULL,
    llm_confidence REAL NOT NULL,
    llm_reasoning TEXT,
    llm_tokens INTEGER,
    entry_price_cents INTEGER,
    bid_ask_spread INTEGER,
    gbm_prob_yes REAL,
    volume INTEGER,
    settled INTEGER DEFAULT 0,
    yes_won INTEGER,
    settled_at TEXT,
    settlement_price REAL
);

CREATE INDEX IF NOT EXISTS idx_model_provider ON llm_evals(model_provider);
CREATE INDEX IF NOT EXISTS idx_model_name ON llm_evals(model_name);
CREATE INDEX IF NOT EXISTS idx_settled ON llm_evals(settled);
CREATE INDEX IF NOT EXISTS idx_market ON llm_evals(market_ticker);
CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_evals(timestamp);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Get database connection (creates if needed)."""
    return init_db()


# ─── Storage ──────────────────────────────────────────────────────────────────

def store_llm_eval(record: LLMEvalRecord, conn: sqlite3.Connection = None) -> bool:
    """Store an LLM evaluation record."""
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False

    try:
        conn.execute("""
            INSERT OR REPLACE INTO llm_evals (
                eval_id, timestamp, market_ticker, window_start, strike,
                spot_at_eval, minutes_left, prompt_version, model_provider,
                model_name, prompt_template, llm_prob_yes, llm_confidence,
                llm_reasoning, llm_tokens, entry_price_cents, bid_ask_spread,
                gbm_prob_yes, volume, settled, yes_won, settled_at, settlement_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.eval_id, record.timestamp, record.market_ticker,
            record.window_start, record.strike, record.spot_at_eval,
            record.minutes_left, record.prompt_version, record.model_provider,
            record.model_name, record.prompt_template, record.llm_prob_yes,
            record.llm_confidence, record.llm_reasoning, record.llm_tokens,
            record.entry_price_cents, record.bid_ask_spread,
            record.gbm_prob_yes, record.volume,
            int(record.settled), int(record.yes_won) if record.yes_won is not None else None,
            record.settled_at, record.settlement_price
        ))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Failed to store LLM eval: {e}")
        return False
    finally:
        if close_after:
            conn.close()


def update_outcome(
    eval_id: str,
    yes_won: bool,
    settlement_price: float,
    conn: sqlite3.Connection = None,
) -> bool:
    """Update an eval with settlement outcome."""
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False

    try:
        conn.execute("""
            UPDATE llm_evals
            SET settled = 1, yes_won = ?, settled_at = ?, settlement_price = ?
            WHERE eval_id = ?
        """, (int(yes_won), datetime.now(timezone.utc).isoformat(), settlement_price, eval_id))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Failed to update outcome: {e}")
        return False
    finally:
        if close_after:
            conn.close()


def get_settled_evals(
    model_provider: str = None,
    model_name: str = None,
    days_back: int = 30,
    conn: sqlite3.Connection = None,
) -> List[LLMEvalRecord]:
    """Get settled evaluations for calibration."""
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False

    try:
        query = "SELECT * FROM llm_evals WHERE settled = 1"
        params = []

        if model_provider:
            query += " AND model_provider = ?"
            params.append(model_provider)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if days_back > 0:
            cutoff = datetime.now(timezone.utc).timestamp() - days_back * 86400
            query += " AND timestamp > ?"
            params.append(datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat())

        query += " ORDER BY timestamp DESC"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        records = []
        for row in rows:
            d = dict(zip(cols, row))
            d["yes_won"] = bool(d["yes_won"]) if d["yes_won"] is not None else None
            d["settled"] = bool(d["settled"])
            records.append(LLMEvalRecord.from_dict(d))

        return records
    finally:
        if close_after:
            conn.close()


# ─── Calibration Computation ──────────────────────────────────────────────────

def compute_calibration_metrics(
    records: List[LLMEvalRecord],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute calibration metrics for a set of settled evaluations.

    Metrics:
    - Brier score: mean squared error of probabilities
    - Log-loss: cross-entropy
    - Reliability curve: mean predicted vs actual frequency per bin
    - ECE: Expected Calibration Error
    - MCE: Maximum Calibration Error
    - Calibration slope/intercept from logistic regression
    """
    if not records:
        return CalibrationMetrics(
            model_key="none",
            n_samples=0,
            n_settled=0,
            brier_score=0.0,
            log_loss=0.0,
            accuracy=0.0,
            calibration_slope=1.0,
            calibration_intercept=0.0,
            ece=0.0,
            mce=0.0,
            reliability_curve=[],
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    probs = np.array([r.llm_prob_yes for r in records])
    outcomes = np.array([1.0 if r.yes_won else 0.0 for r in records])

    # Brier score
    brier = np.mean((probs - outcomes) ** 2)

    # Log-loss (clip to avoid log(0))
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    logloss = -np.mean(outcomes * np.log(probs_clipped) + (1 - outcomes) * np.log(1 - probs_clipped))

    # Accuracy at 0.5 threshold
    preds = (probs > 0.5).astype(float)
    accuracy = np.mean(preds == outcomes)

    # Reliability curve (binning)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    reliability = []
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            mean_pred = np.mean(probs[mask])
            frac_pos = np.mean(outcomes[mask])
            reliability.append((float(mean_pred), float(frac_pos)))

    # ECE and MCE
    ece = 0.0
    mce = 0.0
    for mean_pred, frac_pos in reliability:
        weight = np.sum(bin_indices == b) / len(probs) if 'b' in dir() else 0
        # Find the bin index for this reliability point
        bin_idx = min(int(mean_pred * n_bins), n_bins - 1)
        bin_mask = bin_indices == bin_idx
        if np.sum(bin_mask) > 0:
            weight = np.sum(bin_mask) / len(probs)
            ece += weight * abs(mean_pred - frac_pos)
            mce = max(mce, abs(mean_pred - frac_pos))

    # Calibration slope/intercept (Platt scaling)
    # Fit logistic regression: logit(p) = a + b * logit(prob)
    logit_probs = np.log(probs_clipped / (1 - probs_clipped))
    from sklearn.linear_model import LogisticRegression
    try:
        cal_model = LogisticRegression(solver="lbfgs", max_iter=1000)
        cal_model.fit(logit_probs.reshape(-1, 1), outcomes)
        slope = float(cal_model.coef_[0][0])
        intercept = float(cal_model.intercept_[0])
    except Exception:
        slope = 1.0
        intercept = 0.0

    model_key = f"{records[0].model_provider}:{records[0].model_name}" if records else "unknown"

    return CalibrationMetrics(
        model_key=model_key,
        n_samples=len(records),
        n_settled=len(records),
        brier_score=float(brier),
        log_loss=float(logloss),
        accuracy=float(accuracy),
        calibration_slope=slope,
        calibration_intercept=intercept,
        ece=float(ece),
        mce=float(mce),
        reliability_curve=reliability,
        last_updated=datetime.now(timezone.utc).isoformat(),
    )


def compute_all_calibrations(days_back: int = 30) -> Dict[str, CalibrationMetrics]:
    """Compute calibration for all model/providers."""
    conn = get_db()
    try:
        # Get unique model combinations
        cursor = conn.execute("""
            SELECT DISTINCT model_provider, model_name
            FROM llm_evals
            WHERE settled = 1
        """)
        models = cursor.fetchall()

        results = {}
        for provider, name in models:
            records = get_settled_evals(provider, name, days_back, conn)
            if len(records) >= 10:  # Minimum samples
                metrics = compute_calibration_metrics(records)
                results[f"{provider}:{name}"] = metrics
                log.info(f"Calibrated {provider}:{name}: n={len(records)}, Brier={metrics.brier_score:.4f}, ECE={metrics.ece:.4f}")

        return results
    finally:
        conn.close()


# ─── Ensemble Weight Computation ──────────────────────────────────────────────

def compute_ensemble_weights(
    calibration_metrics: Dict[str, CalibrationMetrics],
    method: str = "inverse_brier",
) -> Dict[str, float]:
    """
    Compute ensemble weights from calibration metrics.

    Methods:
    - inverse_brier: weight = 1 / brier_score
    - inverse_logloss: weight = 1 / log_loss
    - inverse_ece: weight = 1 / ece
    - accuracy: weight = accuracy
    - softmax: softmax of negative logloss
    """
    if not calibration_metrics:
        return {}

    weights = {}
    for key, metrics in calibration_metrics.items():
        if method == "inverse_brier":
            w = 1.0 / (metrics.brier_score + 1e-6)
        elif method == "inverse_logloss":
            w = 1.0 / (metrics.log_loss + 1e-6)
        elif method == "inverse_ece":
            w = 1.0 / (metrics.ece + 1e-6)
        elif method == "accuracy":
            w = metrics.accuracy
        elif method == "softmax":
            w = math.exp(-metrics.log_loss)
        else:
            w = 1.0

        weights[key] = max(w, 0.0)

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ─── Integration with fair_value ──────────────────────────────────────────────

def get_llm_ensemble_weights(
    days_back: int = 30,
    method: str = "inverse_brier",
) -> Dict[str, float]:
    """
    Get LLM ensemble weights for fair_value integration.

    Returns dict mapping "provider:model" -> weight.
    """
    calibrations = compute_all_calibrations(days_back)
    return compute_ensemble_weights(calibrations, method)


# ─── Reporting ────────────────────────────────────────────────────────────────

def generate_calibration_report(
    days_back: int = 30,
    out_path: str = "llm_calibration_report.json",
) -> Dict:
    """Generate full calibration report."""
    calibrations = compute_all_calibrations(days_back)
    weights = compute_ensemble_weights(calibrations)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": days_back,
        "models": {},
        "ensemble_weights": weights,
    }

    for key, metrics in calibrations.items():
        report["models"][key] = asdict(metrics)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"Calibration report saved to {out_path}")
    return report


# ─── CLI / Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    # Demo: create some test records
    conn = init_db()

    for i in range(100):
        # Simulate LLM evals with varying calibration quality
        provider = np.random.choice(["openrouter", "groq", "ollama"])
        model = np.random.choice(["model-a", "model-b"])

        # Model-a is well-calibrated, model-b is overconfident
        if model == "model-a":
            true_prob = np.random.random()
            llm_prob = true_prob + np.random.normal(0, 0.05)
        else:
            true_prob = np.random.random()
            llm_prob = 0.5 + 1.5 * (true_prob - 0.5)  # Overconfident

        llm_prob = np.clip(llm_prob, 0.01, 0.99)
        yes_won = np.random.random() < true_prob

        record = LLMEvalRecord(
            eval_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_ticker=f"KXBTC15M-26NOV{i:04d}-45000",
            window_start="2026-11-20T04:00:00Z",
            strike=95000.0,
            spot_at_eval=95000.0 + np.random.normal(0, 100),
            minutes_left=np.random.uniform(5, 14),
            prompt_version="v1",
            model_provider=provider,
            model_name=model,
            prompt_template="standard",
            llm_prob_yes=llm_prob,
            llm_confidence=np.random.uniform(0.5, 0.9),
            llm_reasoning="Test reasoning",
            llm_tokens=np.random.randint(100, 500),
            entry_price_cents=np.random.randint(15, 95),
            bid_ask_spread=np.random.randint(1, 10),
            gbm_prob_yes=0.5 + np.random.normal(0, 0.1),
            volume=np.random.randint(100, 1000),
            settled=True,
            yes_won=yes_won,
            settled_at=datetime.now(timezone.utc).isoformat(),
            settlement_price=100.0 if yes_won else 0.0,
        )
        store_llm_eval(record, conn)

    print("Stored 100 test records")

    # Compute calibrations
    report = generate_calibration_report(days_back=30)
    print(f"\nCalibration Report:")
    for key, metrics in report["models"].items():
        print(f"  {key}: n={metrics['n_settled']}, Brier={metrics['brier_score']:.4f}, "
              f"LogLoss={metrics['log_loss']:.4f}, ECE={metrics['ece']:.4f}, "
              f"Slope={metrics['calibration_slope']:.2f}")
    print(f"\nEnsemble Weights: {report['ensemble_weights']}")