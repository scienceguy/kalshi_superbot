#!/usr/bin/env python3
"""
scripts/measure_high_gamma.py — Live calibration of high-gamma zone threshold.

For 15-minute binary options, the "high-gamma zone" is the final minutes before
expiry where gamma explodes and the option price becomes extremely sensitive to
the underlying price. In this zone:
- Time decay (theta) accelerates exponentially
- Small underlying moves cause large price swings
- The Brownian motion model used in gates.py becomes unreliable

This script measures the win rate of favourite-maker entries at different
minutes-to-expiry to find the optimal cutoff where the strategy stops working.

Methodology:
- Fetch settled 15M markets from Kalshi
- For each market, check what the entry price would have been at different minutes
- Measure win rate at each minute bucket
- Find the minute where win rate drops below acceptable threshold
"""

import math
import sys
import os
sys.path.insert(0, "/home/ubuntu/kalshi_superbot-v15")

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import requests
import config
import fees

log = logging.getLogger("measure_high_gamma")

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

def fetch_settled_markets(series: str, limit: int = 1000) -> list:
    """Fetch settled markets for a series."""
    try:
        resp = requests.get(
            f"{KALSHI_BASE}/markets",
            params={"series_ticker": series, "status": "settled", "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])
    except Exception as e:
        log.warning("Failed to fetch settled markets for %s: %s", series, e)
        return []


def fetch_market_settlement(ticker: str) -> dict:
    """Fetch settlement result for a market."""
    try:
        resp = requests.get(
            f"{KALSHI_BASE}/markets/{ticker}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("market", {})
    except Exception as e:
        log.debug("Failed to fetch settlement for %s: %s", ticker, e)
        return {}


def analyze_high_gamma_zone():
    """
    Analyze win rates by minutes-to-expiry for favourite-maker entries.
    
    For each settled market, we simulate what would have happened if we entered
    at different minutes remaining, using the favourite-maker band (95-99c).
    """
    series_list = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"]
    
    # Bucket by minutes remaining
    bucket_wins = defaultdict(int)
    bucket_total = defaultdict(int)
    
    for series in series_list:
        print(f"\nAnalyzing {series}...")
        markets = fetch_settled_markets(series, limit=500)
        print(f"  Fetched {len(markets)} settled markets")
        
        for market in markets:
            ticker = market.get("ticker")
            if not ticker:
                continue
            
            settlement = fetch_market_settlement(ticker)
            if not settlement:
                continue
            
            # Check if market settled
            yes_won = settlement.get("yes_settlement_price", 0) == 100
            close_time = settlement.get("close_time")
            if not close_time:
                continue
            
            # Parse close time
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            except Exception:
                continue
            
            # Get the entry price range for this market
            # We need the orderbook at different minutes before close
            # For now, use the strike and estimate entry price
            strike = market.get("floor_strike")
            if strike is None:
                continue
            
            # Get tick history if available (would need orderbook snapshots)
            # For live calibration, we'd need historical orderbook data
            # This is a placeholder - in production, use tick_store replay
            
            # Simulate: at each minute, the favourite-maker entry would be at 95-99c
            # The model probability q would be computed from sigma and time
            # We measure actual win rate at each minute bucket
            
            # For now, use a simplified model: if we entered at X minutes,
            # what's the probability the favourite wins?
            # This requires historical orderbook data which we may not have
            
            pass
    
    # Print results
    print("\n=== High-Gamma Zone Analysis ===")
    print(f"{'Minutes':>8} {'Wins':>6} {'Total':>6} {'Win Rate':>10}")
    print("-" * 35)
    
    for mins in sorted(bucket_total.keys()):
        wins = bucket_wins[mins]
        total = bucket_total[mins]
        if total > 0:
            rate = wins / total
            print(f"{mins:>8} {wins:>6} {total:>6} {rate:>10.3f}")
    
    # Find optimal cutoff
    print("\n=== Recommended GATES_HIGH_GAMMA_MINUTES ===")
    print("Based on win rate dropping below ~50% or negative EV")
    return bucket_wins, bucket_total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analyze_high_gamma_zone()