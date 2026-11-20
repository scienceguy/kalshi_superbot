#!/usr/bin/env python3
"""
forecast_honesty.py — T15-53: Forecast honesty tracking.

Tracks forecast calibration over time to detect overconfidence, bias, or drift.
- Stores forecast vs actual outcomes
- Computes rolling Brier score, log-loss, calibration curves
- Alerts when metrics degrade beyond thresholds
- Integrates with MemPalace for persistence across sessions
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

DB_PATH = Path("forecast_honesty.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    window_start TEXT NOT NULL,
    strike REAL NOT NULL,
    forecast_p_yes REAL NOT NULL,
    forecast_ci_lower REAL,
    forecast_ci_upper REAL,
    forecast_confidence REAL,
    horizon_minutes INTEGER,
    features_json TEXT,
    -- Actual outcome (filled after settlement)
    settled INTEGER DEFAULT 0,
    yes_won INTEGER,
    settled_at TEXT,
    settlement_price REAL,
    -- Derived metrics
    brier_score REAL,
    log_loss REAL,
    calibration_error REAL
);

CREATE INDEX IF NOT EXISTS idx_model_name ON forecasts(model_name);
CREATE INDEX IF NOT EXISTS idx_settled ON forecasts(settled);
CREATE INDEX IF NOT EXISTS idx_timestamp ON forecasts(timestamp);
CREATE INDEX IF NOT EXISTS idx_market ON forecasts(market_ticker);
"""

@dataclass
class ForecastRecord:
    """Single forecast linked to outcome."""
    forecast_id: str
    timestamp: str
    model_name: str
    market_ticker: str
    window_start: str
    strike: float
    forecast_p_yes: float
    forecast_ci_lower: float = 0.0
    forecast_ci_upper: float = 0.0
    forecast_confidence: float = 0.5
    horizon_minutes: int = 15
    features_json: str = "{}"
    settled: bool = False
    yes_won: Optional[bool] = None
    settled_at: str = ""
    settlement_price: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_error: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class HonestyMetrics:
    """Honesty metrics for a model over a time window."""
    model_name: str
    n_forecasts: int
    n_settled: int
    mean_brier: float
    mean_log_loss: float
    mean_calibration_error: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_curve: List[Tuple[float, float]]  # (mean_pred, frac_pos)
    bias: float  # Mean(forecast - outcome)
    overconfidence: float  # Fraction of forecasts where |pred-0.5| > |outcome-0.5|
    sharpness: float  # Mean(|pred - 0.5|)
    window_start: str
    window_end: str

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

def get_db() -> sqlite3.Connection:
    return init_db()

def store_forecast(record: ForecastRecord, conn: sqlite3.Connection = None) -> bool:
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False
    try:
        conn.execute("""
            INSERT OR REPLACE INTO forecasts (
                forecast_id, timestamp, model_name, market_ticker, window_start,
                strike, forecast_p_yes, forecast_ci_lower, forecast_ci_upper,
                forecast_confidence, horizon_minutes, features_json,
                settled, yes_won, settled_at, settlement_price,
                brier_score, log_loss, calibration_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.forecast_id, record.timestamp, record.model_name,
            record.market_ticker, record.window_start, record.strike,
            record.forecast_p_yes, record.forecast_ci_lower, record.forecast_ci_upper,
            record.forecast_confidence, record.horizon_minutes, record.features_json,
            int(record.settled), int(record.yes_won) if record.yes_won is not None else None,
            record.settled_at, record.settlement_price,
            record.brier_score, record.log_loss, record.calibration_error
        ))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Failed to store forecast: {e}")
        return False
    finally:
        if close_after:
            conn.close()

def update_outcome(
    forecast_id: str,
    yes_won: bool,
    settlement_price: float,
    conn: sqlite3.Connection = None,
) -> bool:
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False
    try:
        # Get the forecast to compute metrics
        cursor = conn.execute(
            "SELECT forecast_p_yes FROM forecasts WHERE forecast_id = ?",
            (forecast_id,)
        )
        row = cursor.fetchone()
        if not row:
            return False
        p_yes = row[0]
        
        outcome = 1.0 if yes_won else 0.0
        eps = 1e-15
        p_clipped = np.clip(p_yes, eps, 1 - eps)
        
        brier = (p_yes - outcome) ** 2
        log_loss = - (outcome * np.log(p_clipped) + (1 - outcome) * np.log(1 - p_clipped))
        cal_error = abs(p_yes - outcome)
        
        conn.execute("""
            UPDATE forecasts
            SET settled = 1, yes_won = ?, settled_at = ?, settlement_price = ?,
                brier_score = ?, log_loss = ?, calibration_error = ?
            WHERE forecast_id = ?
        """, (
            int(yes_won), datetime.now(timezone.utc).isoformat(), settlement_price,
            brier, log_loss, cal_error, forecast_id
        ))
        conn.commit()
        return True
    except Exception as e:
        log.error(f"Failed to update outcome: {e}")
        return False
    finally:
        if close_after:
            conn.close()

def get_settled_forecasts(
    model_name: str = None,
    days_back: int = 30,
    conn: sqlite3.Connection = None,
) -> List[ForecastRecord]:
    if conn is None:
        conn = get_db()
        close_after = True
    else:
        close_after = False
    try:
        query = "SELECT * FROM forecasts WHERE settled = 1"
        params = []
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
            records.append(ForecastRecord(**d))
        return records
    finally:
        if close_after:
            conn.close()

def compute_honesty_metrics(
    records: List[ForecastRecord],
    n_bins: int = 10,
) -> HonestyMetrics:
    if not records:
        return HonestyMetrics(
            model_name="none", n_forecasts=0, n_settled=0,
            mean_brier=0.0, mean_log_loss=0.0, mean_calibration_error=0.0,
            ece=0.0, mce=0.0, reliability_curve=[],
            bias=0.0, overconfidence=0.0, sharpness=0.0,
            window_start="", window_end=""
        )
    
    probs = np.array([r.forecast_p_yes for r in records])
    outcomes = np.array([1.0 if r.yes_won else 0.0 for r in records])
    
    # Brier score
    mean_brier = float(np.mean((probs - outcomes) ** 2))
    
    # Log-loss
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    mean_log_loss = float(-np.mean(outcomes * np.log(probs_clipped) + (1 - outcomes) * np.log(1 - probs_clipped)))
    
    # Calibration error (absolute)
    mean_cal_error = float(np.mean(np.abs(probs - outcomes)))
    
    # Bias: mean(forecast - outcome)
    bias = float(np.mean(probs - outcomes))
    
    # Sharpness: mean distance from 0.5
    sharpness = float(np.mean(np.abs(probs - 0.5)))
    
    # Overconfidence: fraction where |pred-0.5| > |outcome-0.5|
    overconf = np.mean(np.abs(probs - 0.5) > np.abs(outcomes - 0.5))
    overconfidence = float(overconf)
    
    # Reliability curve (binning)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    reliability = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            mean_pred = np.mean(probs[mask])
            frac_pos = np.mean(outcomes[mask])
            weight = np.sum(mask) / len(probs)
            reliability.append((float(mean_pred), float(frac_pos)))
            ece += weight * abs(mean_pred - frac_pos)
            mce = max(mce, abs(mean_pred - frac_pos))
    
    timestamps = [r.timestamp for r in records]
    
    return HonestyMetrics(
        model_name=records[0].model_name if records else "unknown",
        n_forecasts=len(records),
        n_settled=len(records),
        mean_brier=mean_brier,
        mean_log_loss=mean_log_loss,
        mean_calibration_error=mean_cal_error,
        ece=ece,
        mce=mce,
        reliability_curve=reliability,
        bias=bias,
        overconfidence=overconfidence,
        sharpness=sharpness,
        window_start=min(timestamps),
        window_end=max(timestamps),
    )

def compute_all_honesty(days_back: int = 30) -> Dict[str, HonestyMetrics]:
    conn = get_db()
    try:
        cursor = conn.execute("""
            SELECT DISTINCT model_name FROM forecasts WHERE settled = 1
        """)
        models = [row[0] for row in cursor.fetchall()]
        
        results = {}
        for model in models:
            records = get_settled_forecasts(model, days_back, conn)
            if len(records) >= 10:
                metrics = compute_honesty_metrics(records)
                results[model] = metrics
                log.info(f"Honesty {model}: n={len(records)}, Brier={metrics.mean_brier:.4f}, ECE={metrics.ece:.4f}, Bias={metrics.bias:.4f}")
        return results
    finally:
        conn.close()

def check_honesty_alerts(
    metrics: Dict[str, HonestyMetrics],
    brier_threshold: float = 0.25,
    ece_threshold: float = 0.10,
    bias_threshold: float = 0.10,
) -> List[Dict]:
    """Check if any model's honesty metrics exceed thresholds."""
    alerts = []
    for model, m in metrics.items():
        if m.mean_brier > brier_threshold:
            alerts.append({
                "model": model,
                "type": "high_brier",
                "value": m.mean_brier,
                "threshold": brier_threshold,
                "severity": "warning"
            })
        if m.ece > ece_threshold:
            alerts.append({
                "model": model,
                "type": "high_ece",
                "value": m.ece,
                "threshold": ece_threshold,
                "severity": "warning"
            })
        if abs(m.bias) > bias_threshold:
            alerts.append({
                "model": model,
                "type": "bias",
                "value": m.bias,
                "threshold": bias_threshold,
                "severity": "warning"
            })
    return alerts

def generate_honesty_report(days_back: int = 30, out_path: str = "forecast_honesty_report.json") -> Dict:
    metrics = compute_all_honesty(days_back)
    alerts = check_honesty_alerts(metrics)
    
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_back": days_back,
        "models": {},
        "alerts": alerts,
    }
    
    for model, m in metrics.items():
        report["models"][model] = asdict(m)
        # Convert reliability curve to list of dicts for JSON
        report["models"][model]["reliability_curve"] = [
            {"mean_pred": x, "frac_pos": y} for x, y in m.reliability_curve
        ]
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    log.info(f"Honesty report saved to {out_path}")
    if alerts:
        log.warning(f"Honesty alerts: {len(alerts)} issues found")
        for a in alerts:
            log.warning(f"  {a['model']}: {a['type']} = {a['value']:.4f} > {a['threshold']:.4f}")
    
    return report

# Integration with MemPalace
def store_honesty_in_palace(metrics: Dict[str, HonestyMetrics]):
    """Store honesty metrics in MemPalace for cross-session persistence."""
    try:
        import mempalace
        for model, m in metrics.items():
            key = f"forecast_honesty/{model}"
            mempalace.kg_supersede(
                subject=key,
                predicate="metrics",
                object=json.dumps(asdict(m), default=str),
                source="forecast_honesty.py"
            )
    except Exception as e:
        log.debug(f"MemPalace store failed: {e}")

if __name__ == "__main__":
    import uuid
    
    # Demo
    conn = init_db()
    
    np.random.seed(42)
    for i in range(100):
        true_prob = np.random.random()
        forecast_prob = true_prob + np.random.normal(0, 0.1)
        forecast_prob = np.clip(forecast_prob, 0.01, 0.99)
        yes_won = np.random.random() < true_prob
        
        record = ForecastRecord(
            forecast_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name="test_model",
            market_ticker=f"KXBTC15M-TEST-{i:04d}",
            window_start="2026-11-20T04:00:00Z",
            strike=95000.0,
            forecast_p_yes=forecast_prob,
            settled=True,
            yes_won=yes_won,
            settled_at=datetime.now(timezone.utc).isoformat(),
            settlement_price=100.0 if yes_won else 0.0,
        )
        store_forecast(record, conn)
    
    report = generate_honesty_report(days_back=30)
    print(f"Report generated with {len(report['models'])} models")
    for model, m in report["models"].items():
        print(f"  {model}: Brier={m['mean_brier']:.4f}, ECE={m['ece']:.4f}, Bias={m['bias']:.4f}")
    print(f"Alerts: {len(report['alerts'])}")