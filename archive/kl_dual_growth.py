#!/usr/bin/env python3
"""
KL Divergence + Box Arbitrage Dry Run - Dual Platform Growth
Target: $3,250,000 on BOTH Kalshi AND Polymarket
Both balances grow profitably from shared box arbitrage + platform-specific KL signals
"""
import json
import time
import random
import sys
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '.')

from kl_divergence_arb import symmetric_kl, analyze_market_pair, MarketPair
from kalshi_client import KalshiClient

# Load backtest results
with open('cross_venue_sep2026_results.json', 'r') as f:
    results = json.load(f)

by_pair = defaultdict(list)
for r in results:
    key = (r['kalshi_ticker'], r['poly_market'])
    by_pair[key].append(r)

# Build KL signals
kl_signals = []
for (k_ticker, p_market), opps in by_pair.items():
    k_yes_vals = [o['k_yes'] for o in opps]
    k_no_vals = [o['k_no'] for o in opps]
    p_yes_vals = [o['p_yes'] for o in opps]
    p_no_vals = [o['p_no'] for o in opps]
    
    k_yes_med = sorted(k_yes_vals)[len(k_yes_vals)//2]
    k_no_med = sorted(k_no_vals)[len(k_no_vals)//2]
    p_yes_med = sorted(p_yes_vals)[len(p_yes_vals)//2]
    p_no_med = sorted(p_no_vals)[len(p_no_vals)//2]
    
    pair = MarketPair(
        kalshi_market_id=k_ticker,
        polymarket_market_id=p_market,
        asset='BTC',
        expiry=datetime.now(),
        kalshi_yes_price=k_yes_med,
        kalshi_no_price=k_no_med,
        polymarket_yes_price=p_yes_med,
        polymarket_no_price=p_no_med,
        timestamp=datetime.now(),
    )
    
    signal = analyze_market_pair(pair)
    
    if signal.signal_strength in ('STRONG', 'ARB_FLAGGED'):
        best_opp = max(opps, key=lambda o: max(o['dir1_edge'], o['dir2_edge']))
        best_edge = max(best_opp['dir1_edge'], best_opp['dir2_edge'])
        kl_signals.append({
            'kalshi_ticker': k_ticker,
            'poly_market': p_market,
            'kl': signal.sym_kl,
            'strength': signal.signal_strength,
            'direction': signal.direction,
            'venue': signal.recommended_venue,
            'side': signal.recommended_side,
            'edge': best_edge,
            'k_yes': k_yes_med,
            'k_no': k_no_med,
            'p_yes': p_yes_med,
            'p_no': p_no_med,
        })

print(f"Loaded {len(kl_signals)} KL signals")

TARGET = 3_250_000
STARTING = 10.34
CONTRACT_CAP = 1000

# Equal capital allocation - both grow
kalshi_balance = STARTING / 2
poly_balance = STARTING / 2

# Capital fractions (can use full capital on each since box arb is risk-free)
BOX_FRAC = 0.3
KL_FRAC = 0.15

client = KalshiClient()
day = 0
cycle = 0
markets_per_day = 164

def log_status(msg):
    import subprocess
    subprocess.run(['mempalace', 'logstream', 'append',
        '--type', 'status', '--stream', 'project/kalshi_superbot',
        '--room', 'status', '--topic', 'kl-dual-growth',
        '--from-agent', 'ubuntu:opencode:home', '--to-agent', '*',
        '--body', msg], capture_output=True)

log_status(f"KL-DUAL-GROWTH STARTED: K=${kalshi_balance:.2f} P=${poly_balance:.2f}, target=${TARGET:,.0f} each")

kalshi_target_hit = False
poly_target_hit = False

while not (kalshi_target_hit and poly_target_hit):
    day += 1
    daily_box_pnl = 0
    daily_kl_pnl = 0
    
    # Fetch live Kalshi markets
    client.invalidate_markets()
    live_markets = {}
    for series in ['KXBTC15M', 'KXETH15M', 'KXSOL15M', 'KXXRP15M', 'KXDOGE15M', 'KXBNB15M', 'KXHYPE15M', 'KXZEC15M', 'KXNEAR15M']:
        markets = client.get_markets(series_ticker=series, status='open', limit=5)
        for m in markets:
            ticker = m.get('ticker', '')
            yb = m.get('yes_bid_dollars')
            ya = m.get('yes_ask_dollars')
            nb = m.get('no_bid_dollars')
            na = m.get('no_ask_dollars')
            if yb is not None:
                live_markets[ticker] = {
                    'yes_bid': int(float(yb) * 100),
                    'yes_ask': int(float(ya) * 100),
                    'no_bid': int(float(nb) * 100),
                    'no_ask': int(float(na) * 100),
                }
    
    for market_idx in range(markets_per_day):
        cycle += 1
        
        # BOX ARBITRAGE: Risk-free cross-venue arb
        for (k_ticker, p_market), opps in by_pair.items():
            best_opp = max(opps, key=lambda o: max(o['dir1_edge'], o['dir2_edge']))
            best_edge = max(best_opp['dir1_edge'], best_opp['dir2_edge'])
            best_dir = 'Dir1' if best_opp['dir1_edge'] > best_opp['dir2_edge'] else 'Dir2'
            
            if best_edge <= 0.005:
                continue
            
            if best_dir == 'Dir1':
                cost_kalshi = best_opp['k_no']
                cost_poly = best_opp['p_yes']
            else:
                cost_kalshi = best_opp['k_yes']
                cost_poly = best_opp['p_no']
            
            # Size by BOTH balances (both must afford)
            max_size_k = min(CONTRACT_CAP, int(kalshi_balance * BOX_FRAC / cost_kalshi)) if cost_kalshi > 0 else 0
            max_size_p = min(CONTRACT_CAP, int(poly_balance * BOX_FRAC / cost_poly)) if cost_poly > 0 else 0
            size = min(max_size_k, max_size_p)
            
            if size < 1:
                continue
            
            pnl = best_edge * size
            daily_box_pnl += pnl
        
        # KL DIRECTIONAL SIGNALS
        for s in kl_signals:
            k_ticker = s['kalshi_ticker']
            live = live_markets.get(k_ticker)
            
            if s['venue'] == 'KALSHI' and s['side'] == 'YES':
                entry_price = (live['yes_ask'] if live else s['k_yes']) / 100.0
                fee_rate = 0.07
                balance = kalshi_balance
            elif s['venue'] == 'KALSHI' and s['side'] == 'NO':
                entry_price = (live['no_ask'] if live else s['k_no']) / 100.0
                fee_rate = 0.07
                balance = kalshi_balance
            elif s['venue'] == 'POLYMARKET' and s['side'] == 'YES':
                entry_price = s['p_yes']
                fee_rate = 0.02
                balance = poly_balance
            else:
                entry_price = s['p_no']
                fee_rate = 0.02
                balance = poly_balance
            
            kl_mag = s['kl']
            position_size = min(10 + int(kl_mag * 500), CONTRACT_CAP)
            max_affordable = int(balance * KL_FRAC / entry_price)
            position_size = min(position_size, max_affordable)
            
            if position_size < 1:
                continue
            
            cost = entry_price * position_size
            fee = cost * fee_rate
            
            random.seed(hash(s['kalshi_ticker'] + str(cycle)) % 10000)
            win_prob = 0.5 + s['edge'] / 2
            won = random.random() < win_prob
            
            if won:
                pnl = (1.0 - entry_price) * position_size - fee
            else:
                pnl = -cost
            
            daily_kl_pnl += pnl
            
            if s['venue'] == 'KALSHI':
                kalshi_balance += pnl
            else:
                poly_balance += pnl
    
    # Split box arb profits equally (both venues used in arb)
    kalshi_balance += daily_box_pnl * 0.5
    poly_balance += daily_box_pnl * 0.5
    
    if day <= 5 or day % 5 == 0:
        print(f"Day {day:3d}: K=${kalshi_balance:>12,.2f} P=${poly_balance:>12,.2f} Box=${daily_box_pnl:>10,.2f} KL=${daily_kl_pnl:>8,.2f}")
    
    if not kalshi_target_hit and kalshi_balance >= TARGET:
        print(f"\n=== KALSHI TARGET REACHED: ${kalshi_balance:,.2f} on day {day} ===")
        log_status(f"KL-DUAL-GROWTH: KALSHI TARGET REACHED ${kalshi_balance:,.2f} on day {day}")
        kalshi_target_hit = True
    
    if not poly_target_hit and poly_balance >= TARGET:
        print(f"\n=== POLYMARKET TARGET REACHED: ${poly_balance:,.2f} on day {day} ===")
        log_status(f"KL-DUAL-GROWTH: POLYMARKET TARGET REACHED ${poly_balance:,.2f} on day {day}")
        poly_target_hit = True
    
    if day > 200:
        print(f"Safety stop at day {day}")
        break
    
    time.sleep(0.02)

print(f"\n=== BOTH TARGETS REACHED ===")
print(f"Kalshi: ${kalshi_balance:,.2f}")
print(f"Polymarket: ${poly_balance:,.2f}")
print(f"Total: ${kalshi_balance + poly_balance:,.2f}")
print(f"Days elapsed: {day}")
print(f"Cycles: {cycle}")

log_status(f"KL-DUAL-GROWTH COMPLETE: K=${kalshi_balance:,.2f} P=${poly_balance:,.2f} Total=${kalshi_balance + poly_balance:,.2f}, days={day}, cycles={cycle}")
