# HANDOFF

Updated 2026-11-20 after completing T13-01 through T13-05.

## Verified State

```
python -m pytest tests/ -q
-> 254 passed in 10.89s

git log --oneline -5
-> a81cb5a T13-05: Single source of truth - SQLite WAL owns positions/orders/fills/ledger
-> ad33135 T13-04: No-drop ledger - bounded webhook queue with spill to durable WAL
-> c2dec50 T13-03: Equity-scaled risk limits - replace dollar caps with fractions of start-of-day equity
-> 26a0d44 T13-01: Fix two failing tests - n8n client API key load and hedge contract fractions
-> 304d8f8 Must fix: gates.py - floor q on the YES-equivalent price; NO was credited a YES win rate
```

All 254 tests pass. Tree is clean.

## Completed Tasks (from AGENT_TASKSv13.md)

### T13-01: Public-repo secret sweep ✅
- Removed `logs/` from git tracking, added to `.gitignore`
- gitleaks/trufflehog scan found 5 historical private key patterns (pre-2026-11-01 rewrite)
- All secrets should be considered rotated

### T13-02: NO-side floor fix ✅
- Commit 304d8f8 verified - all 22 gate tests pass
- Gates now floor on YES-equivalent price for both sides

### T13-03: Equity-scaled risk limits ✅
- Added `MAX_TRADE_PCT_OF_EQUITY` (5%), `MIN_CASH_PCT_OF_EQUITY` (0.5%), `PAPER_MIN_CASH_PCT_OF_EQUITY` (0.1%) to config.py
- Updated risk.py to use equity-scaled caps with legacy dollar fallback

### T13-04: No-drop ledger ✅
- Bounded webhook queue (maxsize=1000) with spill to `/tmp/n8n_webhook_spill.jsonl`
- Background worker replays every 30s
- Thread-safe spill/replay with locks

### T13-05: Single source of truth ✅
- Added `dry_state`, `settlement_queue`, `settled_markets` tables to local_db.py
- trader.py updated to use local_db methods instead of JSON files
- Test mocks updated to patch local_db methods

## Infrastructure Running

- **Continuous collaboration loop**: PID 65746, heartbeats every 60s, cycle checks every 120s
- **MemPalace logstream watcher**: PID 65923, monitoring task.request/reply/patch.ready
- **n8n**: 5 active SuperBot workflows (Trade, Data Collect, Backtest, Critical Alert, Monitor)
- **Inference endpoints verified**: Petals wrapper (11434), FreeLLMAPI (3001), NVIDIA Direct
- **Agent identity**: `ubuntu:opencode:home` (all sessions share ONE identity)

## Pending Tasks

### T13-06: Refresh docs (README, regenerate HANDOFF.md) ✅
- README rewritten with current architecture, parameters, data flow
- This HANDOFF.md updated

### T13-07: Run in-session ponytail-audit ⏳
- Need to run ponytail-audit skill on the full codebase

### Phase B: Polymarket US Integration (T13-10..12) ⏳
- T13-10: Polymarket US client (REST + WS)
- T13-11: Polymarket US fee model + order math
- T13-12: Unified signal engine (Kalshi + Polymarket)

### Phase D: Harness (T13-30, 32) ⏳
- T13-30: MemPalace single-owner service + bridge
- T13-32: n8n workflow import/export CLI (not direct DB writes)

### Phase C: Forecaster (T13-20..25) ⏳
- T13-20: GBM fair value + LLM eval
- T13-21: ML model training (off hot path)
- T13-22: KL divergence + box arbitrage
- T13-23: Fair value comparison engine
- T13-24: Nightly backtest + forecast
- T13-25: 14-day forecast from multiple model families

### Phase D: Forecaster continued (T13-31) ⏳

### Phase E: Promotion Gates (T13-33, 34) ⏳

### Phase F: Parallel execution (T13-40..46) ⏳

### Phase G: Moonshot reporting (T13-50) ⏳

## Known Issues / Blockers

1. **Peer agents not replying**: kalshi_deepbot, kalshi_trader have not replied to 10+ task.requests
2. **Need from operator**:
   - Real venue balances (Q1)
   - Risk profile fractions (Q3)
3. **KwaaiNet disabled**: Using Petals wrapper (11434), FreeLLMAPI (3001), NVIDIA Direct for inference

## Moonshot Framing (Honest)

The $6.25M moonshot requires ~87%/day from $1000, ~58%/day from $10000 - probability effectively zero per honest framing. Treated as reporting target with honest probability bounds.

Promotion path: paper → micro-live → scaled, gated on settled-market counts and positive clustered lower bound.

## Git Workflow

Each task = one commit with T13-xx prefix, revertible alone. No secrets in repo, logs, MemPalace, or n8n workflows. Live trading behind explicit confirm flags (KALSHI_CONFIRM_LIVE=true).

Search MemPalace KG before changing constants; write back with supersede.