#!/usr/bin/env python3
"""
cross_venue_arb.py — Cross-Venue Box Arbitrage between Kalshi and Polymarket.

Implements the "buy both" arbitrage: Buy YES on one venue + Buy NO on the other
when the combined cost < $1.00 after fees.

Arbitrage condition: P_poly_YES + P_kalshi_NO < 1 - total_fees
Or: P_kalshi_YES + P_poly_NO < 1 - total_fees

This is a true risk-free arbitrage if both legs fill and contracts match exactly.
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

LOG = logging.getLogger("cross_venue_arb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")

# Fee models (from research)
# Polymarket: ceil_4dp(shares * price * 0.25 * (price * (1 - price))^2) — quartic, max ~1.56% at 0.50
# Kalshi: ceil_cents(0.07 * contracts * price * (1 - price)) — parabolic, max 2 cents at 0.50
# Simplified for backtesting:
KALSHI_FEE_RATE = 0.07  # 7% on winning side (parabolic, but use flat for simplicity)
POLYMARKET_FEE_RATE = 0.02  # 2% protocol fee on winning side
SLIPPAGE_BUFFER = 0.005  # 0.5 cents per leg slippage

# Minimum edge after all costs
MIN_EDGE_AFTER_COSTS = 0.01  # 1 cent minimum profit


@dataclass
class MarketSnapshot:
    """Single market snapshot from one venue."""
    venue: str  # "KALSHI" or "POLYMARKET"
    market_id: str
    asset: str
    expiry: datetime
    yes_bid: float
    yes_ask: float
    yes_mid: float
    no_bid: float
    no_ask: float
    no_mid: float
    timestamp: datetime
    volume: float = 0.0


@dataclass
class MatchedPair:
    """Matched Kalshi-Polymarket market pair."""
    kalshi: MarketSnapshot
    polymarket: MarketSnapshot
    time_diff_seconds: float


@dataclass
class ArbOpportunity:
    """Cross-venue arbitrage opportunity."""
    pair: MatchedPair
    arb_type: str  # "POLY_YES_KALSHI_NO" or "KALSHI_YES_POLY_NO"
    poly_leg_price: float
    kalshi_leg_price: float
    total_cost: float
    gross_profit: float  # 1 - total_cost
    poly_fee: float
    kalshi_fee: float
    slippage_cost: float
    net_profit: float
    edge_after_costs: float
    recommended_size: float  # Max contracts based on liquidity
    signal_strength: str  # "STRONG", "MODERATE", "WEAK"
    timestamp: datetime


def calculate_polymarket_fee(price: float, size: float = 1.0) -> float:
    """Calculate Polymarket protocol fee for a trade.
    
    Formula: ceil_4dp(shares * price * 0.25 * (price * (1 - price))^2)
    Max ~1.56% at price 0.50
    """
    # Simplified: fee is paid on winning side only
    # For a $1 payout contract, fee = price * 0.25 * (price * (1 - price))^2
    fee_rate = 0.25 * (price * (1 - price)) ** 2
    return size * price * fee_rate


def calculate_kalshi_fee(price: float, size: float = 1.0) -> float:
    """Calculate Kalshi taker fee for a trade.
    
    Formula: ceil_cents(0.07 * contracts * price * (1 - price))
    Max 2 cents per contract at price 0.50
    """
    # Fee in cents per contract
    fee_cents = 0.07 * price * (1 - price)
    # Ceil to nearest cent
    fee_cents = math.ceil(fee_cents * 100) / 100
    return size * fee_cents / 100  # Convert to dollars


def calculate_total_cost(
    poly_price: float,
    kalshi_price: float,
    size: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Calculate total cost including fees and slippage.
    
    Returns: (total_cost, poly_fee, kalshi_fee, slippage_cost)
    """
    # Base cost
    base_cost = poly_price + kalshi_price
    
    # Fees (only paid on winning side, but we don't know which wins)
    # Conservative: assume we pay both fees
    poly_fee = calculate_polymarket_fee(poly_price, size)
    kalshi_fee = calculate_kalshi_fee(kalshi_price, size)
    
    # Slippage buffer
    slippage = SLIPPAGE_BUFFER * 2 * size  # Both legs
    
    total = base_cost + poly_fee + kalshi_fee + slippage
    
    return total, poly_fee, kalshi_fee, slippage


def detect_arbitrage(pair: MatchedPair) -> List[ArbOpportunity]:
    """Detect cross-venue arbitrage opportunities in a matched pair."""
    opportunities = []
    
    # Type 1: Buy YES on Polymarket + Buy NO on Kalshi
    # Cost = poly_yes_ask + kalshi_no_ask
    poly_yes_ask = pair.polymarket.yes_ask
    kalshi_no_ask = pair.kalshi.no_ask
    
    if poly_yes_ask > 0 and kalshi_no_ask > 0:
        total_cost, poly_fee, kalshi_fee, slippage = calculate_total_cost(
            poly_yes_ask, kalshi_no_ask
        )
        gross_profit = 1.0 - total_cost
        edge = gross_profit
        
        if edge >= MIN_EDGE_AFTER_COSTS:
            # Determine signal strength
            if edge >= 0.03:
                strength = "STRONG"
            elif edge >= 0.02:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            # Estimate max size based on liquidity (simplified)
            max_size = min(
                pair.polymarket.volume * 0.1 if pair.polymarket.volume > 0 else 10,
                pair.kalshi.volume * 0.1 if pair.kalshi.volume > 0 else 10,
                100  # Cap at 100 contracts
            )
            
            opp = ArbOpportunity(
                pair=pair,
                arb_type="POLY_YES_KALSHI_NO",
                poly_leg_price=poly_yes_ask,
                kalshi_leg_price=kalshi_no_ask,
                total_cost=total_cost,
                gross_profit=gross_profit,
                poly_fee=poly_fee,
                kalshi_fee=kalshi_fee,
                slippage_cost=slippage,
                net_profit=gross_profit,
                edge_after_costs=edge,
                recommended_size=max(1, max_size),
                signal_strength=strength,
                timestamp=datetime.now(),
            )
            opportunities.append(opp)
    
    # Type 2: Buy YES on Kalshi + Buy NO on Polymarket
    # Cost = kalshi_yes_ask + poly_no_ask
    kalshi_yes_ask = pair.kalshi.yes_ask
    poly_no_ask = pair.polymarket.no_ask
    
    if kalshi_yes_ask > 0 and poly_no_ask > 0:
        total_cost, poly_fee, kalshi_fee, slippage = calculate_total_cost(
            kalshi_yes_ask, poly_no_ask
        )
        gross_profit = 1.0 - total_cost
        edge = gross_profit
        
        if edge >= MIN_EDGE_AFTER_COSTS:
            if edge >= 0.03:
                strength = "STRONG"
            elif edge >= 0.02:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            max_size = min(
                pair.polymarket.volume * 0.1 if pair.polymarket.volume > 0 else 10,
                pair.kalshi.volume * 0.1 if pair.kalshi.volume > 0 else 10,
                100
            )
            
            opp = ArbOpportunity(
                pair=pair,
                arb_type="KALSHI_YES_POLY_NO",
                poly_leg_price=poly_no_ask,
                kalshi_leg_price=kalshi_yes_ask,
                total_cost=total_cost,
                gross_profit=gross_profit,
                poly_fee=poly_fee,
                kalshi_fee=kalshi_fee,
                slippage_cost=slippage,
                net_profit=gross_profit,
                edge_after_costs=edge,
                recommended_size=max(1, max_size),
                signal_strength=strength,
                timestamp=datetime.now(),
            )
            opportunities.append(opp)
    
    return opportunities


def match_markets(
    kalshi_snapshots: List[MarketSnapshot],
    polymarket_snapshots: List[MarketSnapshot],
    max_time_diff_seconds: int = 30
) -> List[MatchedPair]:
    """Match Kalshi and Polymarket snapshots by asset and expiry."""
    pairs = []
    
    for k_snap in kalshi_snapshots:
        for p_snap in polymarket_snapshots:
            # Match by asset
            if k_snap.asset != p_snap.asset:
                continue
            
            # Match by expiry (within tolerance)
            time_diff = abs((k_snap.expiry - p_snap.expiry).total_seconds())
            if time_diff > max_time_diff_seconds:
                continue
            
            pair = MatchedPair(
                kalshi=k_snap,
                polymarket=p_snap,
                time_diff_seconds=time_diff,
            )
            pairs.append(pair)
    
    return pairs


def load_polymarket_snapshots_from_hf(date_str: str) -> List[MarketSnapshot]:
    """Load Polymarket snapshots from HuggingFace dataset for a specific date."""
    from huggingface_hub import hf_hub_download
    
    try:
        path = hf_hub_download(
            "BrockMisner/polymarket-crypto-5m-15m",
            f"orderbooks/{date_str}.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        snapshots = []
        for market_id, market_df in df.groupby('market_id'):
            # Get asset from market_id
            if 'btc' in market_id.lower():
                asset = 'BTC'
            elif 'eth' in market_id.lower():
                asset = 'ETH'
            elif 'sol' in market_id.lower():
                asset = 'SOL'
            else:
                asset = 'UNKNOWN'
            
            # Parse expiry from market_id or use end_time from markets
            # For now, estimate from timestamp
            expiry = market_df['timestamp'].max() + timedelta(minutes=1)
            
            # Identify tokens by price (higher mid = YES/Up)
            token_ids = market_df['token_id'].unique()
            if len(token_ids) != 2:
                continue
            
            token_prices = {}
            for tid in token_ids:
                tdf = market_df[market_df['token_id'] == tid]
                valid_mid = tdf['mid_price'].dropna()
                if len(valid_mid) > 0:
                    token_prices[tid] = valid_mid.mean()
                else:
                    token_prices[tid] = 0.5
            
            sorted_tokens = sorted(token_prices.items(), key=lambda x: x[1], reverse=True)
            yes_token = sorted_tokens[0][0]
            no_token = sorted_tokens[1][0]
            
            # Get latest snapshot for each token
            yes_df = market_df[market_df['token_id'] == yes_token].sort_values('timestamp')
            no_df = market_df[market_df['token_id'] == no_token].sort_values('timestamp')
            
            if yes_df.empty or no_df.empty:
                continue
            
            yes_latest = yes_df.iloc[-1]
            no_latest = no_df.iloc[-1]
            
            snap = MarketSnapshot(
                venue="POLYMARKET",
                market_id=market_id,
                asset=asset,
                expiry=expiry.to_pydatetime(),
                yes_bid=float(yes_latest['best_bid']),
                yes_ask=float(yes_latest['best_ask']),
                yes_mid=float(yes_latest['mid_price']),
                no_bid=float(no_latest['best_bid']),
                no_ask=float(no_latest['best_ask']),
                no_mid=float(no_latest['mid_price']),
                timestamp=yes_latest['timestamp'].to_pydatetime(),
            )
            snapshots.append(snap)
        
        return snapshots
    except Exception as e:
        LOG.error(f"Failed to load Polymarket snapshots for {date_str}: {e}")
        return []


def load_kalshi_snapshots_from_cache(date_str: str) -> List[MarketSnapshot]:
    """Load Kalshi snapshots from local cache/database."""
    # This would load from the Kalshi price data cache
    # For now, return empty - needs implementation
    LOG.warning("Kalshi snapshot loading not implemented - needs cache/database")
    return []


def run_arb_scan(date_str: str) -> List[ArbOpportunity]:
    """Run arbitrage scan for a specific date."""
    LOG.info(f"Loading Polymarket data for {date_str}...")
    poly_snapshots = load_polymarket_snapshots_from_hf(date_str)
    LOG.info(f"Loaded {len(poly_snapshots)} Polymarket snapshots")
    
    LOG.info(f"Loading Kalshi data for {date_str}...")
    kalshi_snapshots = load_kalshi_snapshots_from_cache(date_str)
    LOG.info(f"Loaded {len(kalshi_snapshots)} Kalshi snapshots")
    
    if not kalshi_snapshots:
        LOG.warning("No Kalshi data available - using synthetic for demo")
        # Generate synthetic Kalshi data for demo
        kalshi_snapshots = generate_synthetic_kalshi(poly_snapshots)
    
    pairs = match_markets(kalshi_snapshots, poly_snapshots)
    LOG.info(f"Matched {len(pairs)} market pairs")
    
    all_opportunities = []
    for pair in pairs:
        opportunities = detect_arbitrage(pair)
        all_opportunities.extend(opportunities)
    
    # Sort by edge descending
    all_opportunities.sort(key=lambda o: o.edge_after_costs, reverse=True)
    
    return all_opportunities


def generate_synthetic_kalshi(poly_snapshots: List[MarketSnapshot]) -> List[MarketSnapshot]:
    """Generate synthetic Kalshi snapshots for testing (adds small random noise)."""
    import random
    random.seed(42)
    np.random.seed(42)
    
    synthetic = []
    for p in poly_snapshots:
        # Add small noise to simulate Kalshi prices
        noise = np.random.normal(0, 0.01)  # ~1 cent noise
        k = MarketSnapshot(
            venue="KALSHI",
            market_id=p.market_id.replace("btc-updown", "KXBTC15M"),
            asset=p.asset,
            expiry=p.expiry,
            yes_bid=max(0.01, min(0.99, p.yes_bid + noise)),
            yes_ask=max(0.01, min(0.99, p.yes_ask + noise)),
            yes_mid=max(0.01, min(0.99, p.yes_mid + noise)),
            no_bid=max(0.01, min(0.99, p.no_bid - noise)),
            no_ask=max(0.01, min(0.99, p.no_ask - noise)),
            no_mid=max(0.01, min(0.99, p.no_mid - noise)),
            timestamp=p.timestamp,
        )
        synthetic.append(k)
    
    return synthetic


def backtest_arb(
    opportunities: List[ArbOpportunity],
    polymarket_resolutions: Dict,
    kalshi_resolutions: Dict
) -> Dict:
    """Backtest arbitrage opportunities against actual outcomes."""
    results = {
        "total_opportunities": len(opportunities),
        "executed": 0,
        "won": 0,
        "lost": 0,
        "total_pnl": 0.0,
        "total_profit": 0.0,
        "total_loss": 0.0,
        "avg_edge": 0.0,
        "win_rate": 0.0,
        "trades": [],
    }
    
    for opp in opportunities:
        # Get actual outcome
        p_res = polymarket_resolutions.get(opp.pair.polymarket.market_id)
        k_res = kalshi_resolutions.get(opp.pair.kalshi.market_id)
        
        # Determine winning side
        winning_side = None
        if p_res:
            winning_side = "YES" if p_res.get('outcome') == "Up" else "NO"
        elif k_res:
            winning_side = "YES" if k_res.get('outcome') == "Up" else "NO"
        
        if winning_side is None:
            continue
        
        # In box arbitrage, one leg ALWAYS wins (by construction)
        # But we need to verify both contracts resolve the same way
        # For backtest, assume perfect match
        results["executed"] += 1
        results["won"] += 1
        results["total_profit"] += opp.net_profit * opp.recommended_size
        results["total_pnl"] += opp.net_profit * opp.recommended_size
        
        results["trades"].append({
            "opp": asdict(opp),
            "winning_side": winning_side,
            "pnl": opp.net_profit * opp.recommended_size,
        })
    
    if results["executed"] > 0:
        results["avg_edge"] = results["total_pnl"] / results["executed"]
        results["win_rate"] = results["won"] / results["executed"]
    
    return results


def main():
    """Demo: Run arbitrage scan on sample date."""
    # Test with synthetic data
    LOG.info("Running demo with synthetic data...")
    
    # Create a sample Polymarket snapshot
    poly = MarketSnapshot(
        venue="POLYMARKET",
        market_id="btc-updown-15m-1772395200",
        asset="BTC",
        expiry=datetime(2026, 3, 1, 20, 15),
        yes_bid=0.44,
        yes_ask=0.46,
        yes_mid=0.45,
        no_bid=0.54,
        no_ask=0.56,
        no_mid=0.55,
        timestamp=datetime(2026, 3, 1, 20, 10),
        volume=1000,
    )
    
    # Create synthetic Kalshi with slightly different prices
    kalshi = MarketSnapshot(
        venue="KALSHI",
        market_id="KXBTC15M-26MAR011515",
        asset="BTC",
        expiry=datetime(2026, 3, 1, 20, 15),
        yes_bid=0.53,
        yes_ask=0.55,
        yes_mid=0.54,
        no_bid=0.44,
        no_ask=0.46,
        no_mid=0.45,
        timestamp=datetime(2026, 3, 1, 20, 10),
        volume=500,
    )
    
    pair = MatchedPair(kalshi=kalshi, polymarket=poly, time_diff_seconds=0)
    opportunities = detect_arbitrage(pair)
    
    print(f"\n=== Cross-Venue Arbitrage Demo ===")
    print(f"Polymarket: YES={poly.yes_ask:.4f} (bid={poly.yes_bid:.4f}), NO={poly.no_ask:.4f} (bid={poly.no_bid:.4f})")
    print(f"Kalshi:     YES={kalshi.yes_ask:.4f} (bid={kalshi.yes_bid:.4f}), NO={kalshi.no_ask:.4f} (bid={kalshi.no_bid:.4f})")
    print()
    
    if opportunities:
        for opp in opportunities:
            print(f"ARBITRAGE FOUND: {opp.arb_type}")
            print(f"  Polymarket leg: ${opp.poly_leg_price:.4f}")
            print(f"  Kalshi leg:     ${opp.kalshi_leg_price:.4f}")
            print(f"  Total cost:     ${opp.total_cost:.4f}")
            print(f"  Gross profit:   ${opp.gross_profit:.4f}")
            print(f"  Polymarket fee: ${opp.poly_fee:.4f}")
            print(f"  Kalshi fee:     ${opp.kalshi_fee:.4f}")
            print(f"  Slippage:       ${opp.slippage_cost:.4f}")
            print(f"  Net profit:     ${opp.net_profit:.4f} per contract")
            print(f"  Edge after costs: {opp.edge_after_costs:.4%}")
            print(f"  Strength: {opp.signal_strength}")
            print(f"  Recommended size: {opp.recommended_size:.0f} contracts")
            print(f"  Total expected profit: ${opp.net_profit * opp.recommended_size:.2f}")
    else:
        print("No arbitrage opportunities found.")
    
    # Test fee calculations
    print("\n=== Fee Calculations ===")
    print(f"Polymarket fee at 0.50: ${calculate_polymarket_fee(0.50):.4f}")
    print(f"Polymarket fee at 0.80: ${calculate_polymarket_fee(0.80):.4f}")
    print(f"Kalshi fee at 0.50:     ${calculate_kalshi_fee(0.50):.4f}")
    print(f"Kalshi fee at 0.80:     ${calculate_kalshi_fee(0.80):.4f}")


if __name__ == "__main__":
    main()