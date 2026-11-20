#!/usr/bin/env python3
"""
fee_probe.py — Probe Polymarket US fee tier.

Queries live markets to determine the fee structure.
"""

import sys
sys.path.insert(0, '/home/ubuntu/kalshi_superbot/archive')

import requests
import json
from datetime import datetime, timezone, timedelta
from polymarket_client import PolymarketUSClient


def probe_fee_tier():
    """Probe the fee tier for Polymarket US markets."""
    client = PolymarketUSClient()
    now = datetime.now(timezone.utc)
    
    # Round down to nearest 15 min
    minutes = (now.minute // 15) * 15
    window_start = now.replace(minute=minutes, second=0, microsecond=0)
    
    results = {}
    
    for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
        market = client.get_updown_15m_market(asset, window_start)
        if not market:
            # Try previous window
            prev = window_start - timedelta(minutes=15)
            market = client.get_updown_15m_market(asset, prev)
        
        if market:
            slug = market.get('slug')
            # Get market details from event
            resp = requests.get(f'https://gateway.polymarket.us/v1/events/slug/{client.window_event_slug(asset, window_start if market else prev)}', timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ev = data.get('event', data) if isinstance(data, dict) else {}
                mkts = ev.get('markets') or []
                for m in mkts:
                    if m.get('slug') == slug:
                        # Extract fee info
                        fee_info = {
                            'slug': slug,
                            'asset': asset,
                            'windowStart': m.get('assetPriceTerms', {}).get('windowStart'),
                            'tickSize': m.get('tickSize'),
                            'minSize': m.get('minimumOrderSize'),
                            'feeTier': m.get('feeTier'),
                            'resolutionSource': m.get('resolutionSource'),
                            'orderTypes': m.get('orderTypes'),
                            'clobTokenIds': m.get('clob_token_ids'),
                        }
                        results[asset] = fee_info
                        break
    
    return results


def main():
    print("=" * 60)
    print("Polymarket US Fee Tier Probe")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    results = probe_fee_tier()
    
    if results:
        for asset, info in results.items():
            print(f"\n{asset} ({info['slug']}):")
            for key, value in info.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo markets found or unable to fetch details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())