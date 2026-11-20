#!/usr/bin/env python3
"""
kl_divergence_arb.py — KL Divergence Arbitrage between Kalshi and Polymarket.

Implements symmetric KL divergence to detect mispricing between the two venues.
Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2

For binary markets: KL(P||Q) = p*ln(p/q) + (1-p)*ln((1-p)/(1-q))

Key insight: KL weights disagreement near extremes (0 and 1) more heavily,
which is exactly where the real money lives in 15-min crypto markets.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG = logging.getLogger("kl_divergence_arb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

# Fee rates
KALSHI_TAKER_FEE = 0.07  # 7% on winning side
POLYMARKET_FEE = 0.02    # 2% protocol fee on winning side

# KL divergence threshold for arbitrage flagging
# Based on Sep 2026 data: max KL = 0.0218, so use realistic thresholds
KL_ARB_THRESHOLD = 0.005  # Symmetric KL > 0.005 = ARB FLAGGED (actionable)
KL_WATCH_THRESHOLD = 0.002  # Symmetric KL > 0.002 = WATCH
KL_STRONG_THRESHOLD = 0.01  # Symmetric KL > 0.01 = STRONG signal

# Minimum edge after fees to consider a trade
MIN_EDGE_AFTER_FEES = 0.005  # 0.5% minimum edge

# Position sizing parameters
BASE_POSITION_SIZE = 10  # Base contracts per trade
MAX_POSITION_SIZE = 100  # Max contracts per trade
KL_POSITION_MULTIPLIER = 500  # Multiply KL by this for position scaling


@dataclass
class MarketPair:
    """Represents a matched market pair between Kalshi and Polymarket."""
    kalshi_market_id: str
    polymarket_market_id: str
    asset: str
    expiry: datetime
    kalshi_yes_price: float
    kalshi_no_price: float
    polymarket_yes_price: float
    polymarket_no_price: float
    timestamp: datetime


@dataclass
class KLArbSignal:
    """KL divergence arbitrage signal."""
    market_pair: MarketPair
    sym_kl: float
    kl_pq: float  # KL(Polymarket || Kalshi)
    kl_qp: float  # KL(Kalshi || Polymarket)
    direction: str  # "KALSHI_CHEAPER_YES", "KALSHI_CHEAPER_NO", "POLYMARKET_CHEAPER_YES", "POLYMARKET_CHEAPER_NO"
    edge_yes: float  # Edge if buying YES on cheaper venue
    edge_no: float   # Edge if buying NO on cheaper venue
    recommended_side: str  # "YES" or "NO"
    recommended_venue: str  # "KALSHI" or "POLYMARKET"
    signal_strength: str  # "ARB_FLAGGED", "WATCH", "ALIGNED"


def symmetric_kl(p: float, q: float) -> Tuple[float, float, float]:
    """
    Calculate symmetric KL divergence between two probabilities.
    
    Returns: (sym_kl, kl_pq, kl_qp)
    """
    # Clamp to avoid log(0)
    eps = 1e-10
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    
    # KL(P||Q) = p*ln(p/q) + (1-p)*ln((1-p)/(1-q))
    kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    # KL(Q||P) = q*ln(q/p) + (1-q)*ln((1-q)/(1-p))
    kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    
    sym_kl = (kl_pq + kl_qp) / 2
    
    return sym_kl, kl_pq, kl_qp


def calculate_edge(buy_price: float, sell_price: float, fee_rate: float) -> float:
    """
    Calculate edge for buying at buy_price and the counterparty paying sell_price.
    Edge = sell_price - buy_price - sell_price * fee_rate
    """
    return sell_price - buy_price - sell_price * fee_rate


def analyze_market_pair(pair: MarketPair) -> KLArbSignal:
    """Analyze a Kalshi-Polymarket market pair for KL divergence arbitrage."""
    
    # Calculate symmetric KL for YES side
    sym_kl_yes, kl_pq_yes, kl_qp_yes = symmetric_kl(
        pair.polymarket_yes_price, pair.kalshi_yes_price
    )
    
    # Calculate symmetric KL for NO side (should be same due to complementarity)
    sym_kl_no, kl_pq_no, kl_qp_no = symmetric_kl(
        pair.polymarket_no_price, pair.kalshi_no_price
    )
    
    # Use average of both sides
    sym_kl = (sym_kl_yes + sym_kl_no) / 2
    kl_pq = (kl_pq_yes + kl_pq_no) / 2
    kl_qp = (kl_qp_yes + kl_qp_no) / 2
    
    # Determine direction and edges
    # If Kalshi YES < Polymarket YES, buy YES on Kalshi
    # If Polymarket YES < Kalshi YES, buy YES on Polymarket
    
    edges = {}
    
    # Buy YES on Kalshi, sell on Polymarket (implied)
    edge_yes_kalshi = calculate_edge(pair.kalshi_yes_price, pair.polymarket_yes_price, KALSHI_TAKER_FEE)
    edges["KALSHI_YES"] = edge_yes_kalshi
    
    # Buy YES on Polymarket, sell on Kalshi
    edge_yes_poly = calculate_edge(pair.polymarket_yes_price, pair.kalshi_yes_price, POLYMARKET_FEE)
    edges["POLYMARKET_YES"] = edge_yes_poly
    
    # Buy NO on Kalshi
    edge_no_kalshi = calculate_edge(pair.kalshi_no_price, pair.polymarket_no_price, KALSHI_TAKER_FEE)
    edges["KALSHI_NO"] = edge_no_kalshi
    
    # Buy NO on Polymarket
    edge_no_poly = calculate_edge(pair.polymarket_no_price, pair.kalshi_no_price, POLYMARKET_FEE)
    edges["POLYMARKET_NO"] = edge_no_poly
    
    # Find best edge
    best_action = max(edges, key=edges.get)
    best_edge = edges[best_action]
    
    # Parse best action
    if best_action == "KALSHI_YES":
        direction = "KALSHI_CHEAPER_YES"
        recommended_side = "YES"
        recommended_venue = "KALSHI"
    elif best_action == "POLYMARKET_YES":
        direction = "POLYMARKET_CHEAPER_YES"
        recommended_side = "YES"
        recommended_venue = "POLYMARKET"
    elif best_action == "KALSHI_NO":
        direction = "KALSHI_CHEAPER_NO"
        recommended_side = "NO"
        recommended_venue = "KALSHI"
    else:
        direction = "POLYMARKET_CHEAPER_NO"
        recommended_side = "NO"
        recommended_venue = "POLYMARKET"
    
    # Signal strength with updated thresholds
    if sym_kl >= KL_STRONG_THRESHOLD and best_edge >= MIN_EDGE_AFTER_FEES:
        signal_strength = "STRONG"
    elif sym_kl >= KL_ARB_THRESHOLD and best_edge >= MIN_EDGE_AFTER_FEES:
        signal_strength = "ARB_FLAGGED"
    elif sym_kl >= KL_WATCH_THRESHOLD:
        signal_strength = "WATCH"
    else:
        signal_strength = "ALIGNED"
    
    return KLArbSignal(
        market_pair=pair,
        sym_kl=sym_kl,
        kl_pq=kl_pq,
        kl_qp=kl_qp,
        direction=direction,
        edge_yes=max(edge_yes_kalshi, edge_yes_poly),
        edge_no=max(edge_no_kalshi, edge_no_poly),
        recommended_side=recommended_side,
        recommended_venue=recommended_venue,
        signal_strength=signal_strength,
    )


def load_kalshi_prices_from_cache(cache_file: str = "/tmp/kalshi_prices.json") -> Dict:
    """Load Kalshi historical prices from cache file."""
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_kalshi_prices_to_cache(prices: Dict, cache_file: str = "/tmp/kalshi_prices.json"):
    """Save Kalshi historical prices to cache file."""
    with open(cache_file, 'w') as f:
        json.dump(prices, f, indent=2)


def fetch_kalshi_historical_prices(
    kalshi_client, 
    series: str = "BTC-USD-15M",
    start_date: str = "2026-03-01",
    end_date: str = "2026-03-15"
) -> pd.DataFrame:
    """
    Fetch Kalshi historical prices for a series.
    Returns DataFrame with market_id, yes_price, no_price, timestamp.
    """
    # This would use the Kalshi client to fetch historical data
    # For now, return empty - needs Kalshi API access
    LOG.warning("Kalshi historical price fetching not implemented - requires live API")
    return pd.DataFrame()


def match_markets(
    kalshi_markets: pd.DataFrame,
    polymarket_markets: pd.DataFrame,
    max_time_diff_minutes: int = 5
) -> List[MarketPair]:
    """
    Match Kalshi and Polymarket markets by asset and expiry time.
    """
    pairs = []
    
    for _, k_market in kalshi_markets.iterrows():
        k_expiry = k_market['expiry']
        k_asset = k_market['asset']
        
        # Find Polymarket markets with same asset and similar expiry
        candidates = polymarket_markets[
            (polymarket_markets['asset'] == k_asset) &
            (abs(polymarket_markets['expiry'] - k_expiry) <= timedelta(minutes=max_time_diff_minutes))
        ]
        
        for _, p_market in candidates.iterrows():
            pair = MarketPair(
                kalshi_market_id=k_market['market_id'],
                polymarket_market_id=p_market['market_id'],
                asset=k_asset,
                expiry=k_expiry,
                kalshi_yes_price=k_market['yes_price'],
                kalshi_no_price=k_market['no_price'],
                polymarket_yes_price=p_market['yes_price'],
                polymarket_no_price=p_market['no_price'],
                timestamp=min(k_market['timestamp'], p_market['timestamp']),
            )
            pairs.append(pair)
    
    return pairs


def run_kl_arb_scan(
    kalshi_prices: pd.DataFrame,
    polymarket_prices: pd.DataFrame
) -> List[KLArbSignal]:
    """
    Run KL divergence arbitrage scan on matched market pairs.
    """
    # Match markets
    pairs = match_markets(kalshi_prices, polymarket_prices)
    LOG.info(f"Found {len(pairs)} matched market pairs")
    
    signals = []
    for pair in pairs:
        signal = analyze_market_pair(pair)
        signals.append(signal)
        
        if signal.signal_strength == "ARB_FLAGGED":
            LOG.info(f"ARB FLAGGED: {pair.kalshi_market_id} vs {pair.polymarket_market_id} "
                     f"| sym_KL={signal.sym_kl:.4f} | {signal.direction} | "
                     f"edge={max(signal.edge_yes, signal.edge_no):.4f} | "
                     f"venue={signal.recommended_venue} side={signal.recommended_side}")
        elif signal.signal_strength == "WATCH":
            LOG.debug(f"WATCH: {pair.kalshi_market_id} vs {pair.polymarket_market_id} "
                      f"| sym_KL={signal.sym_kl:.4f}")
    
    # Sort by signal strength and edge
    signals.sort(key=lambda s: (s.signal_strength == "ARB_FLAGGED", max(s.edge_yes, s.edge_no)), reverse=True)
    
    return signals


def backtest_kl_arb(
    signals: List[KLArbSignal],
    kalshi_resolutions: Dict,
    polymarket_resolutions: Dict
) -> Dict:
    """
    Backtest KL divergence arbitrage signals against actual outcomes.
    """
    results = {
        "total_signals": len(signals),
        "arb_flagged": 0,
        "watch": 0,
        "aligned": 0,
        "trades": [],
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "ev_per_trade": 0.0,
    }
    
    for signal in signals:
        if signal.signal_strength != "ARB_FLAGGED":
            if signal.signal_strength == "WATCH":
                results["watch"] += 1
            else:
                results["aligned"] += 1
            continue
        
        results["arb_flagged"] += 1
        
        # Get actual outcome
        k_res = kalshi_resolutions.get(signal.market_pair.kalshi_market_id)
        p_res = polymarket_resolutions.get(signal.market_pair.polymarket_market_id)
        
        # Determine winning side
        winning_side = None
        if k_res:
            winning_side = "YES" if k_res['outcome'] == "Up" else "NO"
        elif p_res:
            winning_side = "YES" if p_res['outcome'] == "Up" else "NO"
        
        if winning_side is None:
            continue
        
        # Calculate PnL
        if signal.recommended_venue == "KALSHI":
            entry_price = (signal.market_pair.kalshi_yes_price 
                          if signal.recommended_side == "YES" 
                          else signal.market_pair.kalshi_no_price)
            fee = KALSHI_TAKER_FEE
        else:
            entry_price = (signal.market_pair.polymarket_yes_price 
                          if signal.recommended_side == "YES" 
                          else signal.market_pair.polymarket_no_price)
            fee = POLYMARKET_FEE
        
        won = (signal.recommended_side == winning_side)
        pnl = (1.0 - entry_price - fee) if won else (-entry_price)
        
        results["trades"].append({
            "market_pair": asdict(signal.market_pair),
            "signal": asdict(signal),
            "winning_side": winning_side,
            "won": won,
            "entry_price": entry_price,
            "pnl": pnl,
        })
        results["total_pnl"] += pnl
    
    if results["trades"]:
        results["win_rate"] = sum(1 for t in results["trades"] if t["won"]) / len(results["trades"])
        results["ev_per_trade"] = results["total_pnl"] / len(results["trades"])
    
    return results


def main():
    """Demo: Run KL divergence analysis on sample data."""
    # This would be called with real data from Kalshi and Polymarket
    # For now, demonstrate with synthetic data
    
    # Sample: BTC 15m market
    pair = MarketPair(
        kalshi_market_id="KALSHI-BTC-15M-12345",
        polymarket_market_id="btc-updown-15m-1772395200",
        asset="BTC",
        expiry=datetime(2026, 3, 1, 20, 15),
        kalshi_yes_price=0.62,
        kalshi_no_price=0.38,
        polymarket_yes_price=0.55,
        polymarket_no_price=0.45,
        timestamp=datetime(2026, 3, 1, 20, 10),
    )
    
    signal = analyze_market_pair(pair)
    print(f"Market: {pair.kalshi_market_id} vs {pair.polymarket_market_id}")
    print(f"Kalshi: YES={pair.kalshi_yes_price:.2f}, NO={pair.kalshi_no_price:.2f}")
    print(f"Polymarket: YES={pair.polymarket_yes_price:.2f}, NO={pair.polymarket_no_price:.2f}")
    print(f"Symmetric KL: {signal.sym_kl:.4f}")
    print(f"KL(P||Q): {signal.kl_pq:.4f}, KL(Q||P): {signal.kl_qp:.4f}")
    print(f"Direction: {signal.direction}")
    print(f"Edge YES: {signal.edge_yes:.4f}, Edge NO: {signal.edge_no:.4f}")
    print(f"Recommended: {signal.recommended_venue} {signal.recommended_side}")
    print(f"Signal: {signal.signal_strength}")


if __name__ == "__main__":
    main()