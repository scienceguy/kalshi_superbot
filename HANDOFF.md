# HANDOFF

Updated 2026-11-20 after completing v15 Phase 0 (T15-00 through T15-05) + cross-sectional arbitrage stack.

## Verified State

```
python -m pytest tests/ -q
-> 384 passed in 9.95s

git log --oneline -15
-> 2078324 T15-10/T15-13: spot_composite tests + tick_store verified
-> 56fa8c1 T15-11/T15-12: BRTI tracker + venue books with book-age gate
-> 0bd7eda T15-03/T15-04: Sigma calibration + config consistency
-> 393bd85 T15-05: Docs refresh - README, AGENTS.md, HANDOFF.md, PHASE0_STATUS.md
-> 96b8bcd T15-24/T15-25: xsec_arb_scan.py + tests
-> 2e3d12e T15-23: fees.py fix fudge factor causing 1-unit fee error at exact boundaries
-> 5d911c1 T15-22: box_arbitrage.py fix fee model, units, equivalence override
-> 7b6003d T15-21: polymarket_us_client.py new module for Polymarket US public API
-> dfb0652 T15-20: kalshi_client.py public read-only mode
-> 1baeaca T15-00/T15-01/T15-02: Phase 0 re-verification on HEAD f47919f
-> f47919f (origin/main) peer pushed during session
```

All 384 tests pass. Tree is clean on branch `v15-task-list`.

## Completed v15 Tasks

### T15-00/T15-01/T15-02: Phase 0 Re-verification on HEAD f47919f ✅
- `python3 -m pytest tests/ -q` → **314 passed** at start, **340 passed** now (26 new tests)
- Inventory: `docs/FILES_f47919f.txt` (146 files)
- Committed `docs/PHASE0_STATUS.md` with F-01..F-16 and C-a..C-g verdicts + command output
- New findings F-17..F-21 documented
- `gitleaks detect` → 5 `private-key` hits, all documentation false positives
- `git log -S VERTEX_PROJECT_ID` → empty

### T15-03: Sigma Calibration ✅
- `scripts/measure_sigma.py` measured ATR*sqrt(pi/2) overestimates realized close-to-close sigma by 61-78% (mean ratio 1.70)
- Correct multiplier ≈ 0.738 (sigma ≈ ATR * 0.738)
- Implemented in `gates.py:367-372` — replaced `ATR * math.sqrt(math.pi/2)` with `ATR * 0.738`
- Verified: `measure_sigma.py` with 0.738 gives ratio ~0.98 (near 1.0)

### T15-04: Config Consistency ✅
- Created `tests/test_config_consistency.py` (6 tests)
- Stale 15-55c / 6-11min comments in `config.py` updated to reflect live defaults:
  - Entry window: 4-14 min (was 6-11)
  - Entry band: 95-99c (was 15-55c)
  - MIN_YES_PRICE_CENTS=75
  - GATES_HIGH_GAMMA_MINUTES=2.0
  - MAX_CONTRACTS_CAP=3

### T15-05: Docs Refresh ✅
- `README.md` rewritten: current architecture, cross-sectional arb, live defaults
- `AGENTS.md` updated: v15 paths, 340 tests, ubuntu:opencode:home identity, live defaults table
- `HANDOFF.md` updated (this file)

### Phase 1 Data Plane (T15-10..13) ✅

| Commit | File | Change |
|--------|------|--------|
| `56fa8c1` | `brti_tracker.py` + tests | CF Benchmarks BRTI 60s avg replication: 4 venues (Coinbase, Kraken, Bitstamp, LMAX), outlier filter (MAD), WS reconnection |
| `56fa8c1` | `venue_books.py` + tests | Venue orderbook tracking: Kalshi REST + Polymarket US REST, composite book, book-age gate (<2s) |
| `2078324` | `tests/test_spot_composite.py` | 17 tests for Binance + Coinbase composite spot with tick storage |
| (existing) | `spot_composite.py` | Binance bookTicker/trade + Coinbase ticker/level2/matches WS, median composite, tick_store integration |
| (existing) | `tick_store.py` | Date-partitioned SQLite WAL, append/replay, 1s resolution |

### Cross-Sectional Arbitrage Stack (T15-20..25) ✅

| Commit | File | Change |
|--------|------|--------|
| `dfb0652` | `kalshi_client.py` | `require_key=False` public read mode; `_PUBLIC_PREFIXES` guard |
| `7b6003d` | `polymarket_us_client.py` | New module: dual-host (gateway + `api.polymarket.us`), quadratic fee model (Θ=0.095), executable book quotes |
| `5d911c1` | `box_arbitrage.py` | Real quadratic fees (Kalshi 0.07, PolyUS 0.095); units=10000=$1; `equivalence_proven` override |
| `2e3d12e` | `fees.py` | Fixed 1e-12 fudge factor that pushed exact boundaries up by 1 unit |
| `96b8bcd` | `scripts/xsec_arb_scan.py` + tests | Live scanner matching on `close_time==window_end`; UTC timestamps; 20 new tests |

**Verified live:**
- Settlement equivalence: Kalshi KX*15M and Polymarket US cpc-*-updown-15m-* both settle on CF Benchmarks BRTI 60s simple average with `>=` tie rule
- Fee math: Kalshi round_up(1×0.07×C×P×(1-P)); PolyUS round_up(0.095×C×P×(1-P)) to cent; maker=0 both venues
- Scanner: reads executable quotes with real depth from `api.polymarket.us/v1/markets/{slug}/book` (bypasses gateway Cloudflare 1015); logs to `logs/xsec_arb.jsonl`
- Edge: 1-2 bps maker/maker after fees (matches published 4.13c drag at 50c)

## Infrastructure Running

- **n8n**: `http://127.0.0.1:5678` — 4 workflows managed via `n8n/workflows/*.json` + `scripts/import_n8n_workflows.py`
- **MemPalace**: palace at `/home/ubuntu/.mempalace`, wing `ubuntu` — agent identity `ubuntu:opencode:home`
- **Secrets**: `~/.secrets/` has `kalshi_private.pem`, `n8n_api_key`, `supabase_service_key`, `superbot_webhook_secret`, `bw/`
- **Inference**: `openrouter-free` provider (primary), `dsh-our-free-model` gateway at `http://127.0.0.1:18900`

## Pending v15 Tasks

### Phase 1 Data Plane (T15-10..13) ✅ DONE
- T15-10: Event-driven spot snapshot (Binance WS + Coinbase WS) — `spot_composite.py` + tests
- T15-11: BRTI tracker (CF Benchmarks BRTI 60s avg replication) — `brti_tracker.py` + tests
- T15-12: Venue books + book-age gate (orderbook freshness < 2s) — `venue_books.py` + tests
- T15-13: Backtest data capture (tick-store for cross-venue replay) — `tick_store.py` (existing)

### Phase 2 Strategy (T15-20..25) ✅ DONE
- T15-20: Kalshi public mode (keyless market data)
- T15-21: Polymarket US public client
- T15-22: Box engine fix (fees, units, equivalence)
- T15-23: Fees fudge fix
- T15-24: Cross-sectional arb scanner
- T15-25: Tests for arb stack

### Phase 3 Gates & Risk (T15-30..35) ⏳
- T15-30: Time-decay cushion with corrected sigma (DONE in T15-03)
- T15-31: High-gamma zone with live calibration
- T15-32: Measured floor re-verification (YES-equivalent only)
- T15-33: Risk limits audit (equity-scaled, per-market loss)
- T15-34: Latency regression gate (P95 < 100ms)
- T15-35: KL-divergence + box arb integration

## Known Issues / Blockers

1. **Live Kalshi credentials**: no `KALSHI_KEY_ID` configured; `KALSHI_CONFIRM_LIVE=true` not supplied (operator-only)
2. **Polymarket US authenticated API**: no credentials on this host (`~/.secrets/` has only bw/, kalshi_private.pem, n8n_api_key, supabase_service_key, superbot_webhook_secret) → account-level fee tier and authenticated order path unverified
3. **Polymarket US gateway rate-limits bursts** (Cloudflare 1015 / HTTP 429) → depth fetches need backoff/low request rate (using `api.polymarket.us` as workaround)
4. **Peer agents**: `ubuntu:service:kalshi_trader` has not replied to money-path review request (evt_20261120T105934)

## Moonshot Framing (Honest)

The $3.25M moonshot is a reporting/forecast target, never a sizing/risk input. Probability effectively zero per honest framing. Treated as upper bound only.

Promotion path: paper → micro-live → scaled, gated on settled-market counts and positive clustered lower bound.

## Git Workflow

Each task = one commit with T15-xx prefix, revertible alone. No secrets in repo, logs, MemPalace, or n8n workflows. Live trading behind explicit confirm flags (`KALSHI_CONFIRM_LIVE=true`).

Branch `v15-task-list` pushed to origin. Search MemPalace KG before changing constants; write back with supersede.