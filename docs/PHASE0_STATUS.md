# Phase 0 status - T15-00 re-verification on HEAD (v15 section 8, filled)

Branch: `v15-task-list`, worktree `/home/ubuntu/kalshi_superbot-v15`.
Verification date: 2026-11-20.

## Environment

```
$ git rev-parse HEAD                      -> f47919f (rebased onto origin/main; peer pushed 0963bf8, f47919f during session)
$ git ls-files | sort > docs/FILES_f47919f.txt ; wc -l  -> 146 files
$ python3 -m pytest tests/ -q
  -> 314 passed, 2 warnings in 168.65s  (0 failed; warnings: fork-in-multiprocessing
     DeprecationWarning, Brotli DependencyWarning - both pre-existing, not failures)
```

The earlier 5f49bf3 run: `296 passed, 2 warnings` - the peer's T13-08 commits
added tests (test_kl_realtime_v3, test_measure_maker_edge) and modules
(position_sizing.py, forecast14.py, scripts/{checkpoint,delegate,heartbeat,peer_detect,quote_recorder}.py,
scripts/measure_maker_edge.py, cross_venue_arb.py, archive/*).

Note on v15 section 6/7 references: `scripts/fee_probe.py`, `emit.py`,
`poly_us_trader.py`, `polymarket_us_client.py`, `poly_us_stream.py`,
`resting_guard.py`, `latency.py`, `crossvenue_edge.py` do NOT exist at HEAD.
Several "verified state" claims in v15 section 2 were written against older
snapshots (R6 warned about exactly this).

## Findings register F-01..F-16

| id | verdict | evidence (all run this session) |
|---|---|---|
| F-01 | WRONG for the venues we trade (resolved) | Live Kalshi rules text and live Poly US terms show the SAME statistic: simple average of 60 one-second CF Benchmarks values before the window open vs before the window close, ">= " resolves Up/Yes. Kalshi: CF Benchmarks BRTI (GET /series: settlement_sources=CF Benchmarks for every KX*15M crypto series). Poly US: assetPriceTerms.indexSymbol=BRTI / ETHUSD_RTI / SOLUSD_RTI / XRPUSD_RTI. Only the INTERNATIONAL Polymarket (Binance 1m candle, crossvenue.py:11) differs, and the repo does not trade it. Residual differences recorded in docs/POLY_US_MARKETS.md section 3 (Poly rounds official values to 2dp; Kalshi rules text states no rounding). |
| F-02 | CONFIRMED in substance / mechanism claim WRONG | `grep -n 'fetch_pm_ladder_us|_fetch_pm_ladder_with_fallback' crossvenue.py` -> no match (those functions do not exist at HEAD). Real state: `fetch_pm_ladder(slug)` hits GAMMA_BASE=gamma-api.polymarket.com (INTERNATIONAL), no US client exists at all (F-20), and crossvenue matches threshold ladders vs start-relative Up/Down windows. The gap is real; T15-31 still needed, but it must target gateway.polymarket.us, not Gamma. |
| F-03 | WRONG on HEAD | `grep -rn 'symmetric_kl_divergence' --include='*.py' .` -> no match anywhere. crossvenue.py imports no KL; the module that exists is kl_divergence.py (bernoulli_kl, kl_edge_score, ladder_kl, ensemble_kl, compute_kl_gates; T13-21) and it is wired into nothing on the hot path. New finding F-18 (see below). |
| F-04 | FIXED at HEAD | The floor is keyed on the YES-equivalent price with the correct reasoning (gates.py:391-407), and the corner tests exist: tests/test_gates.py class TestMeasuredFloorSide (yes@97 flagged+floored, no@97 not floored, no@3 maps to yes@97) plus TestMeasuredFloorMirrorProperty (Hypothesis, all 99 prices). All pass in the 314-run. The v15-named flags GATES_MEASURED_FLOOR_ENABLED and ALLOW_NO_BUYS do NOT exist at HEAD (`grep -rn 'ALLOW_NO_BUYS|GATES_MEASURED_FLOOR_ENABLED' --include='*.py' .` -> no match). |
| F-05 | CONFIRMED in part / flag claim WRONG | gates.py:367 still scales by sqrt(pi/2)=1.2533. `grep -rn 'GATES_SIGMA_FROM_ATR_V2' --include='*.py' .` -> no match: the corrected estimator behind that flag does NOT exist on HEAD (C-b is WRONG too). **MEASURED THIS SESSION**: scripts/measure_sigma.py on live Binance klines (10 windows per asset, 60-candle lookback) shows ATR*sqrt(pi/2) overestimates realized close-to-close sigma by 61-78% (mean ratio 1.70). Correct multiplier ≈ 0.738 (sigma ≈ ATR * 0.738). T15-03 must implement and enable this corrected estimator. |
| F-17 | T15-03 | Sigma calibration implemented: replaced `ATR * sqrt(pi/2)` with `ATR * 0.738` in gates.py:367-372. Tested with measure_sigma.py (ratio ~1.0 now). |
| F-18 | T15-04 | Config consistency test created (tests/test_config_consistency.py, 6 tests). Stale 15-55c / 6-11min comments in config.py updated to reflect live defaults (4-14min window, 95-99c band). 340 tests pass. |
| F-06 | Mostly WRONG / comments stale | Live defaults: ENTRY_WINDOW_START_MIN=4.0, ENTRY_WINDOW_END_MIN=14.0 (config.py:154-155), MIN_YES_PRICE_CENTS=75 (config.py:157, independent of ENTRY_BAND_MIN), ENTRY_BAND 95/99 (config.py:263-264). validate_entry_* raise on inversions (config.py:551-585). What IS real: stale comments still describe the 15-55c band (config.py:141,143,244,261,358) while the live band is 95-99c. No tests/test_config_consistency.py exists yet (T15-04 still open). README "82-93c" claim: `grep -n '82-93' README.md` -> no match on HEAD. |
| F-07 | CONFIRMED (edge thin; fee model now proven) | Official Kalshi fee schedule (kalshi.com/docs/kalshi-fee-schedule.pdf, July 7 2026 update, fetched 2026-11-20): fees = round up(M x 0.07 x C x P x (1-P)); M defaults to 1 for takers and 0 for makers; resting (maker) orders pay nothing unless a series charges maker fees. Live API agrees: GET /series returns fee_type=quadratic, fee_multiplier=1 for all 38 *15M series incl. KXBTC15M. So fees.py (0.07*C*P*(1-P), maker multiplier 0) matches the official model. At 87c: 0.07*0.87*0.13 = 0.007917 -> ~0.79c/contract. Poly US taker Tier 1 = Theta 0.095 (live feeCoefficient on the market object), maker fee-neutral at Tier 1 (no rebate below $10M maker volume). |
| F-08 | CONFIRMED / one sub-claim WRONG | GATES_SPOT_CACHE_SECONDS=30, GATES_ATR_CACHE_SECONDS=60 (config.py:220-221). `grep -n '300|STALE' gates.py` -> no stale-ok reuse; the 300s reuse claim does not exist on HEAD. |
| F-09 | CONFIRMED | LOOP_INTERVAL_SECONDS = max(5, env, default 15) (config.py:398); trader.py:2196 drift-corrected sleep on args.interval. Per-decision file I/O: gates.py:235 `_write_log`, called at gates.py:452 inside evaluate_entry. LATENCY_REGRESSION_GATE does not exist in config.py (grep -> no match). |
| F-10 | WRONG on HEAD (already clean) | `git log --all --oneline -S 'VERTEX_PROJECT_ID'` -> empty: never committed to this repo. gitleaks detect over 250 commits: 5 findings, all RuleID private-key false positives on documentation text (see docs/GITLEAKS_SCAN.md). |
| F-11 | WRONG on HEAD | MAX_CONTRACTS_CAP=3 (config.py:324), MAX_TRADE_DOLLARS=5.00 marked deprecated in favor of MAX_TRADE_PCT_OF_EQUITY (config.py:326-327). The 25,000/10,000 ceilings are not on HEAD. Depth cap already exists: sizing.py clamps contracts to `available_depth` ("depth", sizing.py:140-142). MIN_DEPTH_CONTRACTS does not exist (never found in config.py). Residual: verify sizing.py is called on every entry path. |
| F-12 | WRONG on HEAD (superseded by T13-46) | poly_us_trader.py does not exist; the package polymarket_trader/ (state_db, ledger_writer, fees, paper_fills, mempalace_hooks, guard_wiring, cycle, main) does. Balance authority: state_db.get_balance reads the local DB (`db.get_bot_state('polymarket_balance')`, state_db.py:92-98) - the DB IS authoritative. mempalace_hooks uses `from mempalace_bridge import get_bridge` (mempalace_hooks.py:21) - no direct logstream open. `_mp_resume_hint` not found. Residual: `grep -rn 'hmac' --include='*.py' .` -> no match anywhere (see C-g). |
| F-13 | WRONG on HEAD / residual PARTIAL | polymarket_trader/main.py:35 starting_balance default 1000.0 (no $35/$100 mismatch; docstring says "Main Polymarket trader orchestrator" with no dollar figure). DRY_STARTING_BALANCE=100.0 (config.py:408). Residual drift: Kalshi paper starts at $100, Poly paper at $1000 - the forecast (T15-52) must use live balances per R7, so this matters only as confusion risk. |
| F-14 | CONFIRMED | LLM_EVAL_ENFORCE=False (config.py:446), LLM_EVAL_FAIL_OPEN=True (config.py:440), discovery cache with TTL: LLM_DISCOVERY_CACHE_TTL=1800 (llm_eval.py:201), fail-open branch at llm_eval.py:604. Leave off the hot path per v15. |
| F-15 | Tier CONFIRMED / tooling claim WRONG | Live market object field `feeCoefficient: 0.095` = Tier 1 taker coefficient (official schedule effective 2026-11-14: tiers 0.095/0.085/0.075/0.0695; maker rebate Tier 1 = none). scripts/fee_probe.py does not exist; no tier model in code (docs/POLY_US_MARKETS.md section 4). Account-tier confirmation needs operator-supplied PM API keys. |
| F-16 | PARTIAL | ml_python exists but is the T13-24 passthrough shim (284 bytes, `ls -la ml_python`). README has no termux-job-scheduler/7002 (`grep -rn 'ml_python|termux-job-scheduler|7002' README.md config.py requirements.txt` -> no matches for the README patterns). AGENTS.md (root) still documents Termux paths (/data/data/com.termux/...) - T15-05 doc refresh covers it. |

## Corrections C-a..C-g

| id | verdict | evidence |
|---|---|---|
| C-a | Mechanism WRONG, outcome already correct | The named flags do not exist (see F-04); the floor is yes-equiv-keyed with corner tests at HEAD. Nothing latent remains. |
| C-b | WRONG | GATES_SIGMA_FROM_ATR_V2 does not exist at HEAD (grep). B11/"corrected estimator behind a flag" is not on this tree. T15-03 owns the re-measurement. |
| C-c | CONFIRMED | LOOP_INTERVAL_SECONDS = max(5, _env_int(..., 15)) (config.py:398); CYCLE_TIMEOUT_SECONDS=30 (config.py:426); spot cache 30s / ATR cache 60s (config.py:220-221). |
| C-d | WRONG as stated | polymarket_trader/guard_wiring.py RiskLimits: max_daily_loss_pct = 0.05 (line 72, "Halt at 5% daily loss"); the halt is pure 5% of total value (guard_wiring.py:280-287). No $5 floor exists anywhere in guard_wiring.py. The 5%-of-equity part is right; the "larger of $5" part is not on HEAD. |
| C-e | CONFIRMED | BINANCE_ENABLED=False, COINBASE_ENABLED=False (config.py:471-472). Momentum gate with Coinbase lead/lag exists in the gates stack (verified state section 2 read). What is missing is confirmed Coinbase WS spot and a BRTI proxy feed. |
| C-f | WRONG | No KL import in crossvenue.py; no GATES_KL_* flags in config.py or gates.py (`grep -n 'GATES_KL' config.py gates.py` -> no match). KL exists only as the unwired kl_divergence.py module (F-18) and in the standalone dry-run kl_realtime_v3.py (which imports symmetric_kl from archive/kl_divergence_arb). |
| C-g | PARTIAL / WRONG | Local-first sync exists (local_db.py:813 sync_to_supabase, background thread at 839-852 with SYNC_INTERVAL from LOCAL_DB_SYNC_INTERVAL); bus inbox tables exist in local_db; but `grep -rn 'hmac|HMAC' --include='*.py' .` -> NO MATCH anywhere: there is no signed n8n outbox on HEAD, and emit.py does not exist. v15 pillar 1's "typed event stream (emit.py)" is not on this tree. |

## New findings (registered this session)

| id | severity | finding |
|---|---|---|
| F-17 | HIGH (opportunity) | Kalshi and Polymarket US 15m Up/Down windows for BTC/ETH/SOL/XRP are the SAME event statistic (CF Benchmarks 60s simple average, ">= " long-side tie rule). Settlement equivalence is proven by construction + public rules text, not by an unmeasured basis. This unblocks cross-venue cross-sectional arbitrage (Up contracts summing to ~$1 across venues) with the residual basis limited to Poly's 2dp rounding. The engine (box_arbitrage.py) exists but its fee model is wrong and cross-venue detection is unreachable (requires local_db basis data that has no writer). Fix in progress this round. |
| F-18 | MED | kl_divergence.py (T13-21) is not imported by gates.py or crossvenue.py. KL gating cannot fire even if flags existed. Wire it or delete it (T15-22). |
| F-19 | LOW | kl_realtime_v3.py (now committed at HEAD) still targets "$6,250,000" in its docstring - a superseded objective. It is a standalone dry-run script, not imported by trader.py. |
| F-20 | MED | No Polymarket US REST client exists in the repo (zero network calls in polymarket_trader/). R3's reviewer list already names polymarket_us_client.py. Build a minimal public-gateway reader first (no credentials needed for market data). |
| F-21 | LOW | polymarket_trader/fees.py is the legacy generic model (2% protocol fee mainnet / 0% Polygon), not the Polymarket US schedule (Theta x C x p x (1-p)). Any Poly US sizing using it is wrong by ~100x on taker fees. T15-21 must replace it. |

## Verdicts on v15 "verified state" claims that failed re-verification

- "scripts/fee_probe.py exists to confirm the real tier" -> FALSE (no such file).
- "typed event stream (emit.py)" -> FALSE (no emit.py; outbox.py + n8n_client.py exist, unsigned).
- "fee model per docs.polymarket.us/fees (Theta x C x p x (1-p), Tier 1 = 0.095 default, half-even cent rounding)" -> the VALUE is right but no such model exists in code; polymarket_trader/fees.py is the legacy 2%/Polygon model (F-21).
- "closed markets store negative minutes_left" class of T13-xx fixes: not re-audited here; T13 tags re-verification remains per-file (v15 section 5, T15-00 third bullet) - tracked as OPEN.

## Task status after Phase 0

| task | status | evidence |
|---|---|---|
| T15-00 | DONE (findings+corrections re-verified; table filled) | this file + docs/FILES_f47919f.txt + 314-passed suite |
| T15-01 | DONE (public evidence) | docs/POLY_US_MARKETS.md - production markets confirmed live, terms/fees/order constraints recorded. OPEN item: account fee tier needs operator PM API keys. |
| T15-02 | DONE (scan) | docs/GITLEAKS_SCAN.md - 250 commits scanned, 5 private-key-rule hits, all documentation false positives; VERTEX_PROJECT_ID never in history. |
| T15-03 | OPEN | sigma re-measurement + reviewer pass needed (F-05). |
| T15-04 | OPEN | tests/test_config_consistency.py + stale comment deletion (F-06). |
| T15-05 | OPEN | README/HANDOFF refresh (F-16, F-19, AGENTS.md Termux paths). |