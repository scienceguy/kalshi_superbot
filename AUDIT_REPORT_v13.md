# AUDIT_REPORT_v13.md — ponytail-audit findings for kalshi_superbot

Generated 2026-11-20 by in-session audit following AGENT_TASKSv13.md T13-07.

## What this repo does

Single-operator live-trading bot for Kalshi 15-minute binary crypto markets. Places favourite-maker limit orders (YES/NO at 15-55c entry, 75-99c target), evaluates signals via LLM with 7-layer fallback, writes to Supabase, routes settlement P&L through n8n webhooks. Load assumed: single bot process, 15-min cycles, ~100 markets/cycle, local SQLite WAL source of truth.

---

## MUST FIX

### 1. Task acceptance and patch application NOT IMPLEMENTED (scripts/inbox_sweep.py:82,92)
- **What this is:** Inbox sweep handles incoming MemPalace events for peer coordination.
- **Problem:** Two TODOs — task acceptance logic and patch application are stubs with `pass`.
- **Fix:** Implement or remove if handled elsewhere. If this is the coordination entry point, it must work.
- **If we skip it:** Peer agent delegation via MemPalace logstream will not complete; tasks sent to this agent will never be claimed.

---

## SHOULD FIX

### 2. legacy/ directory contains 11 unused Python files (~70KB)
- **What this is:** Old Polymarket, Coinbase, Redstone clients, HF backtests, etc.
- **Problem:** Not imported anywhere in the active codebase, adds confusion, increases repo size.
- **Fix:** Move to `archive/` or delete. Keep only if actively being ported to Phase B (T13-10..12).
- **If we skip it:** Future developers waste time reading dead code.

### 3. Multiple backtest variants — 8+ files, unclear which is canonical
- **Files:** `backtest.py`, `parallel_backtest.py`, `backtest_engine.py`, `backtest_real.py`, `polymarket_backtest_simple.py`, `polymarket_backtest_fast.py`, `polymarket_backtest_hf.py`, `run_sep2026_backtest.py`, plus KL variants.
- **Problem:** Hard to know which backtest to run or trust; near-duplicate logic.
- **Fix:** Keep 1-2 (`backtest.py` + `backtest_engine.py` as shared core), move rest to `archive/` or delete.
- **If we skip it:** Maintenance burden, inconsistent results.

### 4. Multiple KL divergence variants — 10+ files, unclear which is production-ready
- **Files:** `kl_divergence_arb.py`, `kl_divergence_dryrun.py`, `kl_divergence_live.py`, `kl_dual_growth.py`, `kl_dual_growth_final.py`, `kl_live_dry_run.py`, `kl_realtime_v2.py`, `kl_realtime_v3.py`, `kl_realtime_dryrun.py`, `kl_box_live_dry.py`, `kl_single_target.py`, `crossvenue.py`, `cross_venue_arb.py`, `crossvenue_collect.py`, `crossvenue_history.py`.
- **Problem:** 15+ files with overlapping names, unclear which implements the Phase C spec.
- **Fix:** Consolidate to 1-2, document which is active in README.
- **If we skip it:** Impossible to verify KL layer implementation (T13-21).

### 5. Equity-scaled risk limits (T13-03) need test coverage
- **What this is:** New config constants `MAX_TRADE_PCT_OF_EQUITY`, `MIN_CASH_PCT_OF_EQUITY`, `PAPER_MIN_CASH_PCT_OF_EQUITY` added to config.py and used in risk.py.
- **Problem:** Only 6 existing tests in `test_risk.py`; no tests for the new equity-scaled parameters.
- **Fix:** Add `test_min_cash_pct_of_equity`, `test_max_trade_pct_of_equity`, `test_paper_min_cash_pct_of_equity` at $35, $1k, $50k bankrolls.
- **If we skip it:** Regression risk on the core safety feature.

### 6. Spill file path hardcoded to `/tmp/n8n_webhook_spill.jsonl` (n8n_client.py:45)
- **Problem:** Not configurable, `/tmp` may be cleaned on reboot, loses spilled webhooks.
- **Fix:** Make configurable via `config.py` (`N8N_SPILL_FILE_PATH` with default).
- **If we skip it:** Durability gap on host restart.

---

## NICE TO HAVE

### 7. trader.py is 2258 lines — consider splitting by responsibility
- Cycle logic, settlement, position management, risk integration could be separate modules.
- Would improve testability and readability.

### 8. config.py has 600+ lines — consider splitting by domain
- Credentials, market targeting, risk limits, gates, paper mode, etc. in one file.

### 9. Many config constants have no validation (range checks)
- Only `FEE_RATE` and boolean parsing have validation. Others accept any value.

### 10. Consider type hints on more functions for IDE support
- Partial coverage; full hints would catch interface mismatches earlier.

---

## VERDICT

**Healthy** — all 254 tests pass, core logic solid, recent fixes (T13-01..05) verified.

**Top priorities:**
1. Implement `inbox_sweep.py` TODOs (breaks peer coordination)
2. Clean up `legacy/` and consolidate duplicate backtest/KL files
3. Add tests for equity-scaled risk parameters

**Lean:** ~15 files removable (~100KB), consolidate 8+ backtest and 15+ KL/crossvenue variants.

**Not checked:** Full manual review of all 53 Python files, n8n workflow JSON, Supabase schema, MemPalace KG facts.