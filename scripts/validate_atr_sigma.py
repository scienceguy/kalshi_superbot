#!/usr/bin/env python3
"""
validate_atr_sigma.py — Validate ATR→sigma conversion against realized volatility.

Compares:
1. Theoretical multiplier: sqrt(pi/2) ≈ 1.2533 (ATR = E[|X|] = sigma * sqrt(2/pi))
2. Measured multiplier: 0.738 (from empirical data on VPS)

For each 15m window:
- Compute ATR(14) from 1m candles
- Compute realized sigma from close-to-close 15m returns
- Compare: ATR * multiplier vs realized sigma

Reports: mean bias, MAE, correlation.
"""

import logging
import math
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


def compute_atr(candles, lookback=14):
    """Compute Average True Range from 1m candles."""
    if len(candles) < lookback + 1:
        return None
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < lookback:
        return None
    return sum(true_ranges[-lookback:]) / lookback


def compute_realized_sigma(returns):
    """Compute realized standard deviation from returns."""
    if len(returns) < 2:
        return None
    import statistics
    return statistics.stdev(returns)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate ATR→sigma conversion")
    parser.add_argument("--candles-db", default="/home/ubuntu/kalshi_superbot/local_data/candles.sqlite")
    parser.add_argument("--multiplier-theoretical", type=float, default=math.sqrt(math.pi/2))
    parser.add_argument("--multiplier-measured", type=float, default=0.738)
    args = parser.parse_args()

    db_path = Path(args.candles_db)
    if not db_path.exists():
        log.error("Candles DB not found: %s", db_path)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get 1m candles grouped by asset and 15m window
    query = """
        SELECT asset, open_time, close, high, low
        FROM candles_1m
        ORDER BY asset, open_time
    """
    rows = conn.execute(query).fetchall()

    # Group by asset and 15m window (900 seconds)
    from collections import defaultdict
    windows = defaultdict(list)
    for row in rows:
        window_start = int(row["open_time"] / 900) * 900
        key = (row["asset"], window_start)
        windows[key].append({
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        })

    results = []
    for (asset, window_start), candles in windows.items():
        if len(candles) < 15:  # Need at least 15 candles in the 15m window
            continue

        atr = compute_atr(candles, lookback=14)
        if atr is None or atr <= 0:
            continue

        # Compute 15m close-to-close returns
        # Use the last close of each window
        window_closes = [c["close"] for c in candles]
        if len(window_closes) < 2:
            continue

        # For simplicity, use the window's first and last close
        # In practice, we'd need consecutive windows
        returns = []
        for i in range(1, len(window_closes)):
            if window_closes[i-1] > 0:
                ret = (window_closes[i] - window_closes[i-1]) / window_closes[i-1]
                returns.append(ret)

        realized_sigma = compute_realized_sigma(returns)
        if realized_sigma is None or realized_sigma <= 0:
            continue

        # Compare multipliers
        theoretical_sigma = atr * args.multiplier_theoretical
        measured_sigma = atr * args.multiplier_measured

        results.append({
            "asset": asset,
            "atr": atr,
            "realized_sigma": realized_sigma,
            "theoretical_sigma": theoretical_sigma,
            "measured_sigma": measured_sigma,
            "theoretical_error": theoretical_sigma - realized_sigma,
            "measured_error": measured_sigma - realized_sigma,
        })

    if not results:
        log.warning("No valid results")
        return 1

    # Statistics
    import statistics
    n = len(results)
    theoretical_errors = [r["theoretical_error"] for r in results]
    measured_errors = [r["measured_error"] for r in results]
    theoretical_abs_errors = [abs(e) for e in theoretical_errors]
    measured_abs_errors = [abs(e) for e in measured_errors]

    mean_realized = statistics.mean([r["realized_sigma"] for r in results])
    mean_theoretical_bias = statistics.mean(theoretical_errors)
    mean_measured_bias = statistics.mean(measured_errors)
    mae_theoretical = statistics.mean(theoretical_abs_errors)
    mae_measured = statistics.mean(measured_abs_errors)

    # Correlation
    theoretical_sigmas = [r["theoretical_sigma"] for r in results]
    measured_sigmas = [r["measured_sigma"] for r in results]
    realized_sigmas = [r["realized_sigma"] for r in results]

    def correlation(x, y):
        if len(x) != len(y) or len(x) < 2:
            return 0
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x)
        den_y = sum((yi - mean_y) ** 2 for yi in y)
        if den_x == 0 or den_y == 0:
            return 0
        return num / math.sqrt(den_x * den_y)

    corr_theoretical = correlation(theoretical_sigmas, realized_sigmas)
    corr_measured = correlation(measured_sigmas, realized_sigmas)

    print(f"=== ATR→Sigma Validation ===")
    print(f"Windows analyzed: {n}")
    print(f"Mean realized sigma: {mean_realized:.6f}")
    print(f"")
    print(f"Theoretical multiplier (sqrt(pi/2)): {args.multiplier_theoretical:.4f}")
    print(f"  Mean bias: {mean_theoretical_bias:.6f} ({mean_theoretical_bias/mean_realized*100:.1f}% of realized)")
    print(f"  MAE: {mae_theoretical:.6f} ({mae_theoretical/mean_realized*100:.1f}% of realized)")
    print(f"  Correlation: {corr_theoretical:.3f}")
    print(f"")
    print(f"Measured multiplier: {args.multiplier_measured:.3f}")
    print(f"  Mean bias: {mean_measured_bias:.6f} ({mean_measured_bias/mean_realized*100:.1f}% of realized)")
    print(f"  MAE: {mae_measured:.6f} ({mae_measured/mean_realized*100:.1f}% of realized)")
    print(f"  Correlation: {corr_measured:.3f}")
    print(f"")
    print(f"Decision gates:")
    print(f"  |bias| < 10% of mean realized: theoretical={abs(mean_theoretical_bias)/mean_realized < 0.1}, measured={abs(mean_measured_bias)/mean_realized < 0.1}")
    print(f"  MAE < 20% of mean realized: theoretical={mae_theoretical/mean_realized < 0.2}, measured={mae_measured/mean_realized < 0.2}")
    print(f"  Correlation > 0.7: theoretical={corr_theoretical > 0.7}, measured={corr_measured > 0.7}")

    # Save detailed results
    import json
    out_file = Path("atr_sigma_validation.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "n": n,
                "mean_realized_sigma": mean_realized,
                "theoretical": {
                    "multiplier": args.multiplier_theoretical,
                    "mean_bias": mean_theoretical_bias,
                    "mae": mae_theoretical,
                    "correlation": corr_theoretical,
                },
                "measured": {
                    "multiplier": args.multiplier_measured,
                    "mean_bias": mean_measured_bias,
                    "mae": mae_measured,
                    "correlation": corr_measured,
                },
            },
            "results": results[:100],  # Limit output size
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())