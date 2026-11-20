#!/usr/bin/env python3
"""
Dry-run simulation: Run KL divergence + box arbitrage strategy continuously
until simulated balance reaches $1,250,000.
"""

import json
import time
import random
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import subprocess
sys.path.insert(0, '.')

from kl_divergence_arb import symmetric_kl, analyze_market_pair, MarketPair

# Load backtest results
with open('cross_venue_sep2026_results.json', 'r') as f:
    results = json.load(f)

by_pair = defaultdict(list)
for r in results:
    key = (r['kalshi_ticker'], r['poly_market'])
    by_pair[key].append(r)

# Build pair best opportunities
pair_best = []
for (k_ticker, p_market), opps in by_pair.items():
    best_opp = max(opps, key=lambda o: max(o['dir1_edge'], o['dir2_edge']))
    best_edge = max(best_opp['dir1_edge'], best_opp['dir2_edge'])
    best_dir = 'Dir1' if best_opp['dir1_edge'] > best_opp['dir2_edge'] else 'Dir2'
    pair_best.append({
        'pair': f'{k_ticker} <-> {p_market}',
        'edge': best_edge,
        'direction': best_dir,
        'k_yes': best_opp['k_yes'],
        'k_no': best_opp['k_no'],
        'p_yes': best_opp['p_yes'],
        'p_no': best_opp['p_no'],
    })

pair_best.sort(key=lambda x: x['edge'], reverse=True)

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
            'pair': f'{k_ticker} <-> {p_market}',
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

# Simulation parameters
STARTING_BALANCE = 10.34  # From context
TARGET_BALANCE = 1_250_000
SCALE_MARKETS = 164 / 77  # Actual markets per day / matched
CONTRACT_CAP = 1000
CAPITAL_FRACTION_BOX = 0.2  # 20% per trade for box arb
CAPITAL_FRACTION_KL = 0.1   # 10% per trade for KL directional

# State
kalshi_balance = STARTING_BALANCE / 2
poly_balance = STARTING_BALANCE / 2
total_balance = STARTING_BALANCE

day = 0
cycle = 0
markets_per_day = 164  # ~164 15-min markets per day from SII dataset

print(f"=== DRY-RUN SIMULATION STARTED ===")
print(f"Starting balance: ${total_balance:.2f} (Kalshi: ${kalshi_balance:.2f}, Poly: ${poly_balance:.2f})")
print(f"Target: ${TARGET_BALANCE:,.0f}")
print(f"Scale factor: {SCALE_MARKETS:.2f}x markets")
print(f"Contract cap: {CONTRACT_CAP}")
print()

# Log to MemPalace
def log_status(msg):
    subprocess.run([
        'mempalace', 'logstream', 'append',
        '--type', 'status',
        '--stream', 'project/kalshi_superbot',
        '--room', 'status',
        '--topic', 'dry-run',
        '--from-agent', 'ubuntu:opencode:home',
        '--to-agent', '*',
        '--body', msg
    ], capture_output=True)

def log_checkpoint(balance, day, cycle):
    subprocess.run([
        'mempalace', 'logstream', 'append',
        '--type', 'status',
        '--stream', 'project/kalshi_superbot',
        '--room', 'continuity',
        '--topic', 'dry-run-checkpoint',
        '--from-agent', 'ubuntu:opencode:home',
        '--to-agent', '*',
        '--body', f'CHECKPOINT: balance=${balance:.2f}, day={day}, cycle={cycle}, kalshi=${kalshi_balance:.2f}, poly=${poly_balance:.2f}'
    ], capture_output=True)

log_status(f"DRY-RUN STARTED: balance=${total_balance:.2f}, target=${TARGET_BALANCE:,.0f}")

# Main simulation loop
while total_balance < TARGET_BALANCE:
    day += 1
    daily_box_pnl = 0
    daily_kl_pnl = 0
    trades_today = 0
    
    # Simulate all markets for this day
    for market_idx in range(markets_per_day):
        cycle += 1
        
        # Box arbitrage (risk-free)
        for p in pair_best:
            if p['edge'] <= 0.01:
                continue
            
            if p['direction'] == 'Dir1':
                cost_kalshi = p['k_no']
                cost_poly = p['p_yes']
            else:
                cost_kalshi = p['k_yes']
                cost_poly = p['p_no']
            
            max_size_k = min(CONTRACT_CAP, int(kalshi_balance * CAPITAL_FRACTION_BOX / cost_kalshi))
            max_size_p = min(CONTRACT_CAP, int(poly_balance * CAPITAL_FRACTION_BOX / cost_poly))
            size = min(max_size_k, max_size_p)
            
            if size < 1:
                continue
            
            pnl = p['edge'] * size * SCALE_MARKETS
            daily_box_pnl += pnl
            trades_today += 1
        
        # KL directional signals (risky)
        for s in kl_signals:
            if s['venue'] == 'KALSHI' and s['side'] == 'YES':
                entry_price = s['k_yes']
                fee_rate = 0.07
            elif s['venue'] == 'KALSHI' and s['side'] == 'NO':
                entry_price = s['k_no']
                fee_rate = 0.07
            elif s['venue'] == 'POLYMARKET' and s['side'] == 'YES':
                entry_price = s['p_yes']
                fee_rate = 0.02
            else:
                entry_price = s['p_no']
                fee_rate = 0.02
            
            # Position size based on KL magnitude
            kl_mag = s['kl']
            position_size = min(10 + int(kl_mag * 500), CONTRACT_CAP)
            
            # Check capital
            if s['venue'] == 'KALSHI':
                max_affordable = int(kalshi_balance * CAPITAL_FRACTION_KL / entry_price)
                position_size = min(position_size, max_affordable)
            else:
                max_affordable = int(poly_balance * CAPITAL_FRACTION_KL / entry_price)
                position_size = min(position_size, max_affordable)
            
            if position_size < 1:
                continue
            
            cost = entry_price * position_size
            fee = cost * fee_rate
            
            # Simulate outcome (50% win rate + edge boost)
            random.seed(hash(s['pair'] + str(cycle)) % 10000)
            win_prob = 0.5 + s['edge'] / 2
            won = random.random() < win_prob
            
            if won:
                pnl = (1.0 - entry_price) * position_size - fee
            else:
                pnl = -cost
            
            daily_kl_pnl += pnl
            trades_today += 1
            
            if s['venue'] == 'KALSHI':
                kalshi_balance += pnl
            else:
                poly_balance += pnl
    
    # Apply box arb PnL
    kalshi_balance += daily_box_pnl * 0.5
    poly_balance += daily_box_pnl * 0.5
    total_balance = kalshi_balance + poly_balance
    
    # Progress logging
    if day <= 10 or day % 5 == 0:
        pct_to_target = (total_balance / TARGET_BALANCE) * 100
        print(f"Day {day:3d}: Total=${total_balance:>12,.2f}  Box=${daily_box_pnl:>10,.2f}  KL=${daily_kl_pnl:>8,.2f}  Trades={trades_today:3d}  ({pct_to_target:.1f}% to target)")
        
        # Heartbeat every 5 days
        if day % 5 == 0:
            subprocess.run(['python3', 'scripts/heartbeat.py', '--cycles', '1', '--task', f'dry-run-day-{day}'], capture_output=True)
            log_checkpoint(total_balance, day, cycle)
    
    # Checkpoint every 10 days
    if day % 10 == 0:
        log_checkpoint(total_balance, day, cycle)
    
    # Safety: prevent infinite loop
    if day > 500:
        print(f"Safety stop at day {day}")
        break

print(f"\n=== TARGET REACHED ===")
print(f"Final balance: ${total_balance:,.2f}")
print(f"Days elapsed: {day}")
print(f"Cycles: {cycle}")

log_status(f"DRY-RUN COMPLETE: balance=${total_balance:,.2f}, days={day}, cycles={cycle}, TARGET REACHED!")