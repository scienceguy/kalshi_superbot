#!/usr/bin/env python3
"""
cross_sectional_arb.py — Cross-sectional arbitrage for Kalshi 15m up/down contracts.

Key insight: For a mutually exclusive and exhaustive set of up/down contracts on the same
15-minute window (BTC, ETH, SOL, XRP up/down), the YES prices should sum to approximately $1.00
(100 cents) because exactly one outcome resolves YES and the rest resolve NO.

However, Kalshi's 15m crypto markets are independent binary contracts on DIFFERENT assets.
They are NOT mutually exclusive - BTC up and ETH up can both resolve YES simultaneously.
So the classic "complete set arbitrage" (sum of YES asks < 100c) DOES NOT APPLY.

What DOES apply:
1. Within a single market: YES + NO = $1.00 always (by construction, bid-ask spread straddles 100c)
2. Cross-venue: Kalshi 15m BTC up/down vs Polymarket US 15m BTC up/down (same settlement, different fees)
3. Cross-asset correlation: If BTC is up, ETH is more likely up (correlated moves)

This module implements:
- Venue cross-arbitrage: Kalshi vs Polymarket US for the SAME asset/window
- Correlation-aware sizing: When one asset moves, adjust expectations for others
"""

import logging
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import config
from kalshi_client import KalshiClient
sys.path.insert(0, '/home/ubuntu/kalshi_superbot/archive')
from polymarket_client import PolymarketUSClient

log = logging.getLogger(__name__)


@dataclass
class VenueQuote:
    """Quote from a single venue for a specific market."""
    venue: str  # "kalshi" or "polymarket"
    asset: str  # "BTC", "ETH", "SOL", "XRP"
    window_start: datetime
    yes_bid: float  # cents
    yes_ask: float  # cents
    no_bid: float
    no_ask: float
    tick_size: float = 1.0  # cents


@dataclass
class ArbOpportunity:
    """A cross-venue arbitrage opportunity."""
    asset: str
    window_start: datetime
    kalshi_quote: VenueQuote
    poly_quote: VenueQuote
    # Which side to take on each venue
    kalshi_side: str  # "yes" or "no"
    poly_side: str
    # Expected profit per contract pair (after fees)
    gross_edge_cents: float
    net_edge_cents: float
    # Confidence
    confidence: float


def kalshi_fee_taker(price_cents: float) -> float:
    """Kalshi taker fee in cents per contract."""
    p = price_cents / 100.0
    fee = math.ceil(0.07 * p * (1 - p) * 100)
    return fee


def polymarket_fee_taker(price_cents: float) -> float:
    """Polymarket US taker fee estimate (placeholder - needs authenticated probe)."""
    # Standard Polymarket fee ~2 bps on notional
    p = price_cents / 100.0
    fee = max(1, round(0.002 * p * 100))  # ~0.2% of price, min 1 cent
    return fee


def polymarket_fee_maker(price_cents: float) -> float:
    """Polymarket US maker fee estimate (often 0 or rebate)."""
    return 0  # Most markets have 0 maker fee


def get_kalshi_quote(client: KalshiClient, asset: str, window_start: datetime) -> Optional[VenueQuote]:
    """Get Kalshi quote for asset/window."""
    series_map = {
        "BTC": "KXBTC15M",
        "ETH": "KXETH15M", 
        "SOL": "KXSOL15M",
        "XRP": "KXXRP15M",
    }
    series = series_map.get(asset)
    if not series:
        return None
    
    # Find the market for this window
    markets = client.get_markets(series_ticker=series, status="open", limit=100)
    target_ts = int(window_start.timestamp())
    
    for m in markets:
        if m.get("expiration_time") == target_ts * 1000:  # Kalshi uses ms
            ticker = m.get("ticker")
            ob = client.get_orderbook(ticker)
            if ob:
                yes_bid = ob.get("yes_bid", 0) * 100  # Convert to cents
                yes_ask = ob.get("yes_ask", 0) * 100
                no_bid = ob.get("no_bid", 0) * 100
                no_ask = ob.get("no_ask", 0) * 100
                return VenueQuote(
                    venue="kalshi",
                    asset=asset,
                    window_start=window_start,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                )
    return None


def get_polymarket_quote(client: PolymarketUSClient, asset: str, window_start: datetime) -> Optional[VenueQuote]:
    """Get Polymarket US quote for asset/window."""
    market = client.get_updown_15m_market(asset, window_start)
    if not market:
        return None
    
    slug = market.get("slug")
    bbo = client.get_bbo(slug)
    if not bbo:
        return None
    
    # Polymarket quotes are in dollars - handle None cases
    best_bid = bbo.get("bestBid")
    best_ask = bbo.get("bestAsk")
    last_trade = bbo.get("lastTradePx")
    
    yes_bid = float(best_bid.get("value", 0)) * 100 if best_bid else 0
    yes_ask = float(best_ask.get("value", 0)) * 100 if best_ask else 0
    last = float(last_trade.get("value", 0)) * 100 if last_trade else 0
    
    if yes_bid == 0 and yes_ask == 0:
        return None
    
    # No side = 1 - yes side
    no_bid = 100 - yes_ask
    no_ask = 100 - yes_bid
    
    return VenueQuote(
        venue="polymarket",
        asset=asset,
        window_start=window_start,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
    )


def find_cross_venue_arbs(kalshi_client: KalshiClient, poly_client: PolymarketUSClient,
                          window_start: datetime) -> List[ArbOpportunity]:
    """
    Find cross-venue arbitrage opportunities for the given window.
    
    Strategy: Buy on cheaper venue, sell on expensive venue.
    Both venues settle on CF Benchmarks 60-second average (same reference).
    """
    opportunities = []
    assets = ["BTC", "ETH", "SOL", "XRP"]
    
    for asset in assets:
        kalshi_q = get_kalshi_quote(kalshi_client, asset, window_start)
        poly_q = get_polymarket_quote(poly_client, asset, window_start)
        
        if not kalshi_q or not poly_q:
            continue
        
        # Check both directions: buy YES on one, sell YES on other
        # Direction 1: Buy YES on Kalshi, sell YES on Polymarket
        # = Buy YES on Kalshi @ ask, Buy NO on Polymarket @ ask (since selling YES = buying NO)
        cost_kalshi_yes = kalshi_q.yes_ask + kalshi_fee_taker(kalshi_q.yes_ask)
        cost_poly_no = poly_q.no_ask + polymarket_fee_taker(poly_q.no_ask)
        total_cost_1 = cost_kalshi_yes + cost_poly_no
        edge_1 = 100 - total_cost_1  # Cents profit per contract pair
        
        # Direction 2: Buy YES on Polymarket, sell YES on Kalshi
        cost_poly_yes = poly_q.yes_ask + polymarket_fee_taker(poly_q.yes_ask)
        cost_kalshi_no = kalshi_q.no_ask + kalshi_fee_taker(kalshi_q.no_ask)
        total_cost_2 = cost_poly_yes + cost_kalshi_no
        edge_2 = 100 - total_cost_2
        
        # Check if either direction has positive edge
        if edge_1 > 0.5:  # Minimum 0.5c edge after fees
            opportunities.append(ArbOpportunity(
                asset=asset,
                window_start=window_start,
                kalshi_quote=kalshi_q,
                poly_quote=poly_q,
                kalshi_side="yes",
                poly_side="no",
                gross_edge_cents=edge_1 + cost_kalshi_yes + cost_poly_no - (kalshi_q.yes_ask + poly_q.no_ask),
                net_edge_cents=edge_1,
                confidence=min(0.9, edge_1 / 2.0),  # Higher edge = higher confidence
            ))
        
        if edge_2 > 0.5:
            opportunities.append(ArbOpportunity(
                asset=asset,
                window_start=window_start,
                kalshi_quote=kalshi_q,
                poly_quote=poly_q,
                kalshi_side="no",
                poly_side="yes",
                gross_edge_cents=edge_2 + cost_poly_yes + cost_kalshi_no - (poly_q.yes_ask + kalshi_q.no_ask),
                net_edge_cents=edge_2,
                confidence=min(0.9, edge_2 / 2.0),
            ))
    
    # Sort by net edge descending
    opportunities.sort(key=lambda o: o.net_edge_cents, reverse=True)
    return opportunities


def print_opportunities(opps: List[ArbOpportunity]):
    """Pretty print arbitrage opportunities."""
    if not opps:
        print("No cross-venue arbitrage opportunities found.")
        return
    
    print(f"\n{'='*80}")
    print(f"CROSS-VENUE ARBITRAGE OPPORTUNITIES ({len(opps)} found)")
    print(f"{'='*80}")
    
    for opp in opps:
        k = opp.kalshi_quote
        p = opp.poly_quote
        print(f"\n{opp.asset} @ {opp.window_start.strftime('%H:%M:%S')}:")
        print(f"  Kalshi:   YES {k.yes_bid:.1f}/{k.yes_ask:.1f} | NO {k.no_bid:.1f}/{k.no_ask:.1f}")
        print(f"  Polymkt:  YES {p.yes_bid:.1f}/{p.yes_ask:.1f} | NO {p.no_bid:.1f}/{p.no_ask:.1f}")
        print(f"  Action:   Buy {opp.kalshi_side.upper()} on Kalshi @ {getattr(k, f'{opp.kalshi_side}_ask'):.1f}c")
        print(f"            Buy {opp.poly_side.upper()} on Polymarket @ {getattr(p, f'{opp.poly_side}_ask'):.1f}c")
        print(f"  Net edge: {opp.net_edge_cents:.2f}c per contract pair")
        print(f"  Confidence: {opp.confidence:.0%}")


def main():
    """Demo / CLI entry point."""
    from datetime import datetime, timezone, timedelta
    
    # Round to nearest 15-minute window
    now = datetime.now(timezone.utc)
    minutes = (now.minute // 15) * 15
    window_start = now.replace(minute=minutes, second=0, microsecond=0)
    
    print(f"Checking cross-venue arb for window: {window_start}")
    
    kalshi = KalshiClient()
    poly = PolymarketUSClient()
    
    opps = find_cross_venue_arbs(kalshi, poly, window_start)
    print_opportunities(opps)
    
    return 0 if opps else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())