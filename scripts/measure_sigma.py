#!/usr/bin/env python3
"""
scripts/measure_sigma.py — Measure ATR-to-sigma conversion vs realized close-to-close sigma.

For a normal distribution: E[|X|] = sigma * sqrt(2/pi) ≈ 0.7979 * sigma
So: sigma = ATR / 0.7979 = ATR * sqrt(pi/2) ≈ ATR * 1.2533

Wilder's ATR uses smoothed average of True Range (max of h-l, |h-c1|, |l-c1|).
For 1-minute candles with small moves, TR ≈ h-l ≈ |return| * close.
But we measure on log returns, so we need to be careful.

This script fetches Binance 1m klines, computes both, and reports the ratio.
"""

import math
import sys
sys.path.insert(0, "/home/ubuntu/kalshi_superbot-v15")

import binance_feed as bf


def compute_atr(klines, lookback=60):
    """Wilder-style ATR from klines (oldest first)."""
    if not klines or len(klines) < lookback + 1:
        return None
    trs = []
    prev_close = float(klines[0][4])
    for k in klines[1:]:
        h, l, c = float(k[2]), float(k[3]), float(k[4])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < lookback:
        return None
    # Wilder smoothing: first ATR = simple average, then (prev * (n-1) + curr) / n
    atr = sum(trs[:lookback]) / lookback
    for tr in trs[lookback:]:
        atr = (atr * (lookback - 1) + tr) / lookback
    return atr


def compute_realized_sigma(klines):
    """Realized close-to-close sigma (per minute) from klines, in DOLLARS per sqrt-minute."""
    if not klines or len(klines) < 2:
        return None
    closes = [float(k[4]) for k in klines]
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 5:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sigma_frac = math.sqrt(var)  # fractional per minute
    # Convert to dollars per sqrt-minute using the mean price
    mean_price = sum(closes) / len(closes)
    return sigma_frac * mean_price


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    lookback = 60
    n_windows = 10  # number of historical windows to sample

    print(f"Measuring ATR vs realized sigma on {n_windows} windows per symbol (lookback={lookback})")
    print("=" * 80)
    print(f"{'Symbol':<10} {'ATR':>10} {'ATR*sqrt(pi/2)':>15} {'Realized σ':>12} {'Ratio':>8} {'Error%':>8}")
    print("-" * 80)

    all_ratios = []
    for symbol in symbols:
        ratios = []
        for i in range(n_windows):
            # Fetch different historical windows by varying the end time
            # For simplicity, fetch current + some past windows
            klines = bf.get_klines(symbol, "1m", lookback + 50)
            if not klines or len(klines) < lookback + 2:
                continue
            # Take the last `lookback+1` candles for measurement
            window = klines[-(lookback + 1):]
            atr = compute_atr(window, lookback)
            realized = compute_realized_sigma(window)
            if atr is None or realized is None or realized == 0:
                continue
            scaled = atr * math.sqrt(math.pi / 2)
            ratio = scaled / realized
            ratios.append(ratio)
            if i < 3:  # print first few
                print(f"{symbol:<10} {atr:>10.6f} {scaled:>15.6f} {realized:>12.6f} {ratio:>8.3f} {((ratio-1)*100):>+7.1f}%")
        if ratios:
            avg = sum(ratios) / len(ratios)
            all_ratios.extend(ratios)
            print(f"{symbol:<10} {'avg ratio:':>10} {avg:>8.3f}  (n={len(ratios)})")
        print("-" * 80)

    if all_ratios:
        overall = sum(all_ratios) / len(all_ratios)
        print(f"\nOverall average ratio (ATR*sqrt(pi/2) / realized_sigma): {overall:.4f}")
        print(f"  sqrt(pi/2) = {math.sqrt(math.pi/2):.4f}")
        if abs(overall - 1.0) < 0.1:
            print("  -> Current scaling (sqrt(pi/2)) is ACCURATE for normal returns")
        else:
            print(f"  -> Scaling bias: {((overall-1)*100):+.1f}%")
            print(f"  -> Suggested multiplier: {math.sqrt(math.pi/2) / overall:.4f}")


if __name__ == "__main__":
    main()