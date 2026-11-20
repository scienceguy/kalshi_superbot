# Kalshi SuperBot

Single-operator live-trading bot for Kalshi 15-minute binary crypto markets.

## Overview

Places favourite-maker limit orders (two-sided YES/NO at 15–55c entry band, 75–99c target), evaluates signals via an LLM with 7-layer fallback, writes signals/orders/bot-state to Supabase, and routes settlement P&L through n8n webhooks.

## Quick Start

```bash
cd kalshi_superbot
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
- `ENTRY_WINDOW_START_MIN=6.0` / `ENTRY_WINDOW_END_MIN=11.0` — Entry window

## Architecture

- `trader.py` — Main trading loop (entry point)
- `signals.py` — Signal engine (two-sided favourite maker)
- `market_finder.py` — Market discovery (parallel series fetch, orderbook lookup)
- `gates.py` — Pre-entry gates (ATR hurdle, fee-adjusted EV, time-decay cushion)
- `kalshi_client.py` — Signed Kalshi REST client (RSA-PSS, `/trade-api/v2`)
- `fees.py` — Kalshi fee math (configurable `KALSHI_FEE_RATE`, default 0.07)
- `risk.py` — Risk management (equity-scaled limits, drawdown halt, consecutive-loss cooldown)
- `resilience.py` — Cycle supervisor with auto client rebuild
- `n8n_client.py` — n8n webhook/API client (bounded queue + WAL spill)
- `supabase_client.py` — Supabase REST client (circuit breaker + batched writes)
- `llm_eval.py` — LLM trade evaluation (7-layer fallback: OpenRouter/Groq/DeepSeek/OpenAI/Gemini/DSH/Ollama)
- `latency_arb.py` — Latency arbitrage strategy (off hot path)
- `backtest.py` / `parallel_backtest.py` — Parameter sweep backtester
- `data_collector.py` — Historical market data collector
- `kalshieye_downloader.py` — Kalshi Eye orderbook downloader
- `inject_n8n_workflows.py` — n8n workflow injector (idempotent)
- `add_n8n_credentials.py` — n8n credential injector (deprecated, use `scripts/n8n_creds_sync.py`)
- `local_db.py` — Local SQLite WAL (source of truth for decisions, fills, ledger, settlements)
- `tests/` — Pytest test suite (254 tests)

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

Four workflows injected into n8n via `inject_n8n_workflows.py`:
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
python -m pytest tests/ -q  # All 254 tests should pass
```

## License

Private