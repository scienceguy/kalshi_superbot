# Kalshi SuperBot

Single-operator live-trading bot for Kalshi 15-minute binary crypto markets.

## Overview

Places favourite-maker limit orders (two-sided YES/NO at 95–99c entry band, 75c+ target), evaluates signals via an LLM with 7-layer fallback, writes signals/orders/bot-state to Supabase, and routes settlement P&L through n8n webhooks.

Cross-sectional arbitrage: Kalshi 15M Up/Down vs Polymarket US 15M Up/Down (both settle on CF Benchmarks BRTI 60s avg, >= tie rule).

## Quick Start

```bash
cd kalshi_superbot-v15
cp .env.example .env  # Fill in your credentials
python trader.py --once --dry  # Test run
python trader.py --once --live --dry  # One live cycle (needs KALSHI_CONFIRM_LIVE=true)
python trader.py --live  # Live loop
```

## Environment Variables

See `.env.example` for all required variables. Never commit real credentials.

Key variables:
- `KALSHI_KEY_ID` / `KALSHI_KEY_FILE` — Kalshi API credentials
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — Supabase project
- `N8N_API_KEY` / `N8N_BASE_URL` — n8n instance
- `DRY_RUN=true` — Dry-run mode (default)
- `KALSHI_CONFIRM_LIVE=true` — Required for live trading
- `MAX_TRADE_PCT_OF_EQUITY=0.05` — Max trade as fraction of start-of-day equity
- `MIN_CASH_PCT_OF_EQUITY=0.005` — Min cash floor as fraction of equity
- `ENTRY_WINDOW_START_MIN=4.0` / `ENTRY_WINDOW_END_MIN=14.0` — Entry window
- `ENTRY_BAND_MIN=95` / `ENTRY_BAND_MAX=99` — Entry band (cents)
- `MIN_YES_PRICE_CENTS=75` — Minimum YES price for entry

## Architecture

- `trader.py` — Main trading loop (entry point)
- `signals.py` — Signal engine (two-sided favourite maker)
- `market_finder.py` — Market discovery (parallel series fetch, orderbook lookup)
- `gates.py` — Pre-entry gates (ATR hurdle, fee-adjusted EV, time-decay cushion)
- `kalshi_client.py` — Signed Kalshi REST client (RSA-PSS, `/trade-api/v2`) + keyless public mode
- `polymarket_us_client.py` — Polymarket US public API (gateway + api.polymarket.us dual-host)
- `fees.py` — Kalshi fee math (configurable `KALSHI_FEE_RATE`, default 0.07) + fudge fix
- `risk.py` — Risk management (equity-scaled limits, drawdown halt, consecutive-loss cooldown)
- `resilience.py` — Cycle supervisor with auto client rebuild
- `n8n_client.py` — n8n webhook/API client (bounded queue + WAL spill)
- `supabase_client.py` — Supabase REST client (circuit breaker + batched writes)
- `llm_eval.py` — LLM trade evaluation (7-layer fallback: OpenRouter/Groq/DeepSeek/OpenAI/Gemini/DSH/Ollama)
- `latency_arb.py` — Latency arbitrage strategy (off hot path)
- `backtest_engine.py` / `backtest_harness.py` — **canonical** backtesters (shared entry/P&L core + unified harness)
- `position_sizing.py` — Kelly-fraction sizing with equity/depth/per-market caps
- `kl_divergence.py` — **canonical** KL-divergence layer (Bernoulli KL edge score, ladder KL, ensemble disagreement gate)
- `kl_realtime_v3.py` — real-time KL + box-arbitrage dry-run loop
- `gbm_fair_value.py` / `fair_value_edge.py` / `box_arbitrage.py` — GBM fair value, edge = `q_lcb - ask - fee - slippage`, cross-venue box engine
- `forecaster.py` / `forecast14.py` — multi-model 14-day forecast; moonshot probability is an upper bound only
- `spot_composite.py` / `binance_feed.py` — Binance + Coinbase composite spot with failover
- `dual_venue.py` — independent Kalshi/Polymarket orchestration
- `outbox.py` — transactional outbox feeding the Supabase/n8n replicas
- `mempalace_bridge.py` / `agent_protocol.py` — single-owner MemPalace access + inter-agent envelope/veto protocol
- `event_hotpath.py` — event-driven hot path (WS-tick-driven, off the 15 s poll)
- `scripts/measure_edge.py` — realized win rate per price band from settled markets
- `scripts/measure_sigma.py` — ATR-to-sigma calibration (measured 0.738, was sqrt(pi/2)=1.253)
- `scripts/measure_maker_edge.py` / `scripts/quote_recorder.py` — maker-vs-taker net edge + order-book recorder
- `scripts/xsec_arb_scan.py` — cross-sectional arb scanner (Kalshi + Polymarket US, logs JSONL)
- `data_collector.py` — Historical market data collector
- `kalshieye_downloader.py` — Kalshi Eye orderbook downloader
- `inject_n8n_workflows.py` / `scripts/import_n8n_workflows.py` — n8n workflow injector
- `scripts/n8n_creds_sync.py` — n8n credential sync from `~/.secrets/bw/`
- `local_db.py` — Local SQLite WAL (source of truth for decisions, fills, ledger, settlements)
- `archive/` — retired variants; **not on any live or paper path**
- `tests/` — Pytest test suite (340 tests)

## Data Flow (Local-First)

1. **Local SQLite WAL** (`local_db.py`) — single source of truth for:
   - Signals, orders, bot state
   - Dry-run cash balance (`dry_state`)
   - Settlement queue (`settlement_queue`) + replay
   - Settled markets tracking (`settled_markets`)
   - P&L ledger (`pnl_ledger`) with idempotency keys

2. **Supabase** — downstream replica (batched sync, circuit breaker)

3. **n8n** — control plane (webhooks for settlement processing, monitoring, backtest, data collect)

4. **MemPalace** — long-term memory (KG facts, checkpoints, coordination)

## n8n Workflows

Four workflows managed via `n8n/workflows/*.json` + `scripts/import_n8n_workflows.py`:
- `superbot-backtest` — Parameter sweep backtest
- `superbot-collect` — Market data collection
- `superbot-trade` — Settlement processor (webhook → Supabase PATCH)
- `superbot-monitor` — Position monitoring + alerting

## Risk Management (Equity-Scaled)

- `MAX_TRADE_PCT_OF_EQUITY=0.05` — 5% of start-of-day equity per trade
- `CASH_FLOOR_PCT=0.70` / `EXPOSURE_CEILING_PCT=0.30` — Portfolio shape
- `MIN_CASH_PCT_OF_EQUITY=0.005` — 0.5% equity-scaled cash floor (live)
- `PAPER_MIN_CASH_PCT_OF_EQUITY=0.001` — 0.1% equity-scaled cash floor (paper)
- `DRAWDOWN_HALT_PCT=0.08` — 8% drawdown halt
- `MAX_CONTRACTS_CAP=3` / `MAX_OPEN_POSITIONS=2` — Position limits

## Testing

```bash
python -m pytest tests/ -q  # All 340 tests should pass
```

## License

Private