# AGENT_TASKSv13.md - kalshi_deepbot patch round 13

Audience: the OpenCode agent running on the OCI VPS.
Status of everything below: SUGGESTIONS. Verify each against the live tree
(`git log --oneline -20`, `grep`, `pytest`) before acting. Where this file and
the code disagree, the code wins; record the disagreement in MemPalace.

Written 2026-11-19 from a read of the public repo page, README.md, HANDOFF.md
(HEAD dedc68a, 230 commits) and the first ~1000 lines of poly_us_trader.py.
The repo is now at 424 commits, so HANDOFF.md is stale. Items I could not
verify are marked [VERIFY].

---------------------------------------------------------------------------
## 0. SUMMARY (read this if nothing else)

Goal: trade Polymarket US and Kalshi from the CURRENT REAL balances on each
venue, using GBM fair value, LLM evaluation, ML, KL divergence, box
arbitrage and fair-value comparison, with both balances growing, tracked
against a $6,250,000 moonshot over 14 days. Backtests must drive honest
14-day forecasts from several model families.

The five pillars every task below serves. They are the point of this round:

1. **n8n integration and workflow management.** n8n is the control plane:
   scheduling, retraining, nightly backtest+forecast, settlement P&L,
   reconciliation, alerting, agent dispatch. Workflows live in the repo as
   JSON, are imported by CLI/API (not by writing n8n's database), and are
   NEVER on the order hot path. If n8n is down, trading continues and events
   queue in the outbox.
2. **Local-first database synchronization.** Local SQLite (WAL) is the
   source of truth for decisions, fills, ledger, settlements. Supabase and
   n8n are downstream replicas fed by a durable outbox. Sync lag is a
   measured SLO. Nothing is ever dropped from a full in-memory queue.
   (The venue is the source of truth for MONEY; local DB for INTENT.)
3. **Full MemPalace utilization.** One owner process for the palace; all
   agents go through the bridge. Use checkpoints, drawers, KG facts (with
   supersede/invalidate), status heartbeats AND event topics, and search it
   BEFORE changing any constant. It is the long-term memory of measured
   facts, not a log dump.
4. **Live inter-agent communication and collaboration.** Defined roles,
   topics, message schema, acks, heartbeats, and a binding risk veto.
   In-process asyncio for sub-second paths, MemPalace logstream as the
   durable mirror.
5. **Subsecond latency.** Budget: Binance/Coinbase WS tick -> decision
   < 250 ms p99; decision -> order on the wire < 300 ms p99; tick -> order
   < 1 s p99. LLM and ML training are OFF the hot path (precomputed/cached).

Honest framing of the moonshot is in section 2. Short version: it is a
sizing-and-reporting target, not a plan the evidence supports.

---------------------------------------------------------------------------
## 1. REPO ASSESSMENT (evidence-tagged)

What is good (keep it):
- Typed event stream to n8n (`emit.py`), local DB with a sync-health breaker
  (`local_db.py`, `PolyLocalDB`, `emit_sync_lag`).
- MemPalace bridge already wired into poly_us_trader (checkpoint, drawer,
  kg_add/supersede/invalidate, status, resume hint).
- Resting-order guard on a fast tick with spot-move and feed-age checks
  (`resting_guard.py`), WS BBO cache with REST fallback (`poly_us_stream`).
- Polymarket US fee model matches the published schedule: taker
  fee = Theta x C x p x (1-p), banker's rounding to the cent, makers free.
- Honest self-audit culture: HANDOFF.md corrects its own stale balance claim
  and reports edge with clustered intervals.

Findings, highest priority first:

F1 [P0, VERIFY] REPO IS NOW PUBLIC. The page loads logged-out and shows
   public metadata, but HANDOFF.md says "private". It has 424 commits and
   tracks `logs/agent-autopilot/opencode`, `paper_state_precutover_*.json`,
   `legacy/`, and `supabase/`. Agent logs are the classic place for leaked
   keys. Action: confirm visibility, then scan full history (see T13-01).
   Rotate anything that ever appeared, even if history is rewritten.

F2 [P0] The measured edge cannot carry a moonshot. Per HANDOFF.md: 95-99c
   favourites, n=246 markets (557 quotes), 4 losses, net EV +1.75c per
   contract, clustered lower bound clears breakeven by only ~0.4c, one
   7-day slice, BTC/ETH/SOL only. The 15-55c band is significantly NEGATIVE.
   A loss at 97c costs ~32 wins. That is a steady-pennies strategy.
   See section 2.

F3 [P1] Ledger writer DROPS records when its bounded queue is full
   (`put_nowait` -> `queue.Full` -> counter + error log). Dropping a fill or
   settlement is unacceptable for a compounding bankroll. Spill to a durable
   WAL instead (T13-04).

F4 [P1] Two sources of truth: `poly_us_state.json` is rewritten atomically
   every save and described as authoritative for cash, while the local DB is
   described as source of truth for settlements (D1) and only "best-effort
   mirrored" (D2). Consolidate to SQLite WAL with JSON as an export (T13-05).

F5 [P1] Risk limits are fixed dollars, not fractions of equity
   (`POLY_DAILY_LOSS_LIMIT_DOLLARS` default 5.0, `initial_cash` default 35,
   `max_open_positions` 25). Fine at $35; meaningless at $5,000, reckless
   relative to a high-growth plan. Express limits as % of start-of-day equity.

F6 [P1] MemPalace lock. HANDOFF.md recorded the palace held by one
   `mempalace-mcp` PID with no read/write possible. `_mp_resume_hint()`
   opens the Logstream SQLite file directly, bypassing whatever owns the
   lock [VERIFY]. Single owner service + bridge-only access (T13-20).

F7 [P1] 304d8f8 (gates.py NO-side floor: used a YES win rate for NO fills,
   off by ~98 points) is committed but was still awaiting human review.
   Re-read it, add a symmetric property test (T13-02).

F8 [P2] README.md is stale: titled "Kalshi SuperBot", lists `latency_arb.py`,
   `data_collector.py`, `kalshieye_downloader.py`, `add_n8n_credentials.py`
   (not visible in the root listing, which was truncated) and Termux
   job-scheduler / proot instructions that die with the move to the VPS.

F9 [P2] `poly_us_trader.py` is 3,478 lines / 165 KB in one file mixing
   state, ledger, MemPalace, fees, paper fills, guard wiring. Hard to test
   and slow to reason about under latency work. Split by responsibility.

F10 [P2] Polling cadence: `--interval 15`. Entry decisions are
   poll-driven; only the resting guard has a fast tick. Subsecond needs an
   event-driven loop.

F11 [P2] `inject_n8n_workflows.py` injects into n8n's database directly.
   Fragile across n8n upgrades; use the CLI/API.

F12 [P2] ponytail-audit/review produced ZERO findings (subagents died on a
   signup gate). There is no clean audit. Run it in-session on a model that
   needs no gate.

F13 [P2] Cross-venue "box" legs may settle on DIFFERENT references/windows
   (Polymarket US `assetPriceTerms.windowEnd` vs Kalshi's 15-minute index).
   Without proven equivalence it is not an arbitrage, it is two bets.

---------------------------------------------------------------------------
## 2. THE MOONSHOT, STATED HONESTLY

Required constant daily growth G = (6,250,000 / B0)^(1/14):

| Start balance B0 | Multiple needed | Required per day |
|------------------|-----------------|------------------|
| $100             | 62,500x         | about +120%/day  |
| $1,000           | 6,250x          | about +87%/day   |
| $10,000          | 625x            | about +58%/day   |

Against a measured edge of ~+1.75c per contract with a thin lower bound, a
risk of ruin that rises with every Kelly multiple, thin 15-minute books, and
per-market position limits, the probability of reaching the target is
effectively zero, and aggressive sizing toward it raises the probability of
ruin sharply. No backtest can create evidence that is not in the data.

How the agent should handle this (rules, not suggestions):

R1. Treat $6,250,000 as a REPORTING TARGET. Every forecast prints the
    probability of reaching it as an upper bound ("< 1 in N_sims"), never 0
    and never inflated.
R2. A `RISK_PROFILE` knob (`conservative | aggressive | moonshot`) may change
    only the Kelly fraction and size caps. It must NOT disable the daily
    loss halt, ruin halt, ledger-reconcile halt, or venue-balance reconcile.
R3. Never fabricate or "carry forward" a balance. Day-0 balance comes from
    the venue APIs and is recorded once (T13-30).
R4. Never tune a backtest until the forecast reaches the target. Parameter
    choices are frozen BEFORE the out-of-sample window is evaluated.
R5. Promotion path: paper -> micro-live at minimum size -> scaled sizing,
    each step gated on settled-market counts and a positive CLUSTERED lower
    bound (see T13-31). `KALSHI_CONFIRM_LIVE` style flags stay.

---------------------------------------------------------------------------
## 3. STANDING RULES FOR THE AGENT

- One commit per task, message prefixed `T13-xx`, revertable alone.
- After ANY write: `LC_ALL=C grep -n '[^ -~]' FILE` and
  `python scripts/verify_source.py --all`, then `pytest -q`. py_compile
  passing proves nothing (HANDOFF "standing trap"). Keep the `.githooks`
  gate on.
- No secrets in the repo, logs, MemPalace drawers, or n8n workflow JSON.
- State a claim, run the check, report the check's output. Do not report
  from memory.
- Live trading stays behind explicit confirm flags. This round ships paper
  and micro-live paths; it does not flip a venue to full size.
- Before changing a constant: search MemPalace KG for prior measurements
  and write the new one back (supersede, do not duplicate).

---------------------------------------------------------------------------
## 4. TASKS

Format: ID | why | suggested change | acceptance test.

### PHASE A - Safety and hygiene (do first)

**T13-01 Public-repo secret sweep.**
Why: F1. Suggest: confirm visibility with `gh repo view`; run a history
scanner (gitleaks/trufflehog) over all 424+ commits including
`logs/agent-autopilot/`; add `logs/` to `.gitignore` and `git rm --cached`;
rotate any key that ever appeared. Consider making the repo private again.
Accept: scanner exits clean; written finding in MemPalace; list of rotated
credentials handed to Gentry (not committed).

**T13-02 Review and harden 304d8f8.**
Why: F7. Suggest: re-read the NO-side floor in `gates.py`; add a hypothesis
property test: for any price p, YES@p and NO@(100-p) get mirror-image floors.
Accept: test passes; `git revert 304d8f8` remains clean.

**T13-03 Equity-scaled risk limits.**
Why: F5. Suggest: replace dollar limits with fractions of start-of-day equity
(`DAILY_LOSS_FRAC`, `MAX_POSITION_FRAC`, `MAX_ASSET_EXPOSURE_FRAC`,
`MAX_EXPIRY_EXPOSURE_FRAC`), reuse the existing `exposure_by_asset/direction/
expiry` helpers, and apply the same on the Kalshi side. Gentry to confirm the
numbers (open question Q3).
Accept: tests at $35, $1k, $50k bankrolls give proportional halts.

**T13-04 No-drop ledger.**
Why: F3. Suggest: on `queue.Full`, append to a fsync'd spill file
(`ledger.spill.jsonl`) and have the writer drain spill first on next
iteration. Same for the n8n/Supabase outbox. Dropping stays only for
explicitly lossy telemetry.
Accept: test floods the queue with 10x capacity; zero records lost; order
preserved.

**T13-05 Single source of truth.**
Why: F4. Suggest: SQLite (WAL) owns positions, orders, fills, ledger,
settlements, config_hash history. `poly_us_state.json` becomes a periodic
export for operators. State mutations are transactions.
Accept: kill -9 mid-cycle at 100 random points; restart state equals the
last committed transaction; ledger gap == 0.

**T13-06 Refresh docs.**
Why: F8. Suggest: rewrite README (name, real module list, VPS run
instructions, systemd units), regenerate HANDOFF.md from commands, delete
Termux-only instructions or move them to `legacy/`.
Accept: a fresh clone follows README to a green `pytest` and `--once --dry`.

**T13-07 Run the in-session audit.**
Why: F12. Suggest: run ponytail-audit on a model with no signup gate, work
the MUST FIX list, commit findings as `AUDIT_REPORT_v13.md`.
Accept: non-empty report with evidence per finding.

### PHASE B - Data plane: Binance and Coinbase WebSocket spot

**T13-10 Composite spot service.**
Why: the model needs a robust, low-latency reference price. `binance_feed.py`
and `crossvenue_edge.get_spot_with_age` exist; extend to a single
`spot_composite` module. Suggest:
- Binance: public trade/bookTicker (and 1s kline for vol) streams.
- Coinbase: public WS (ticker/market trades/level2 best bid-ask); verify the
  current endpoint and channel names in Coinbase docs [VERIFY].
- Per tick keep `{venue, symbol, px, bid, ask, exch_ts, recv_ts}`; compute
  composite = median (or size-weighted) of fresh venues; reject a venue if
  `recv_ts - exch_ts` or age exceeds a limit; flag > N bps divergence.
- Region warning: Binance.com blocks US IPs. Pick the OCI region/endpoint
  (or Binance.US) deliberately and test connectivity first [VERIFY].
- Reconnect with jittered backoff; sequence/gap detection; never block the
  event loop; expose `get_spot_with_age()` unchanged for the guard.
Accept: soak test 24 h; p99 tick age < 250 ms; automatic failover when one
venue is killed; no stale price ever used (age check enforced in tests).

**T13-11 Settlement-reference basis tracker.**
Why: F13. Suggest: for each venue, record the settlement reference's price
(from market terms / settled outcome) versus the composite at window end;
store basis stats in the local DB and as a MemPalace KG fact. Model fair
value should add basis uncertainty to sigma.
Accept: basis histogram per asset/venue published nightly.

**T13-12 Tick store.**
Why: backtests need the same ticks the live bot saw. Suggest: append compact
ticks (1 s resolution is enough for 15-min markets; keep BBO deltas for
latency studies) to Parquet/SQLite partitions; rotate and compress.
Accept: backtest can replay any past market window from stored ticks only.

### PHASE C - Models and signals

All models emit a calibrated probability `q` for "YES settles" plus an
uncertainty (interval or ensemble std). Entry requires edge after fees,
slippage and uncertainty, not raw edge.

**T13-20 GBM fair value (baseline model).**
Formula (r ~ 0, drift ~ 0 over 15 minutes):
`q = Phi(d2)`, `d2 = (ln(S/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))`, with S
= composite spot, K = strike (or window-open reference for up/down markets),
tau in years. Suggest: sigma from realized 1-s returns over several windows
blended with `AdaptiveVolatilityScaler` (already present), plus a fat-tail
option (Merton jump or Student-t) and basis uncertainty from T13-11.
Accept: Brier and calibration curve on held-out settled markets beat the
market mid by a margin that survives the clustered bootstrap.

**T13-21 KL divergence layer.**
Why: use KL as a signal and as a gate, not decoration. Suggest:
1. Bernoulli KL(q_model || p_market) per market as an edge-strength score.
2. Across a strike ladder, KL between model-implied and market-implied
   discrete distributions to spot mispriced regions and ladder arbitrage
   (monotonicity: P(S>K1) >= P(S>K2) for K1<K2).
3. KL between ensemble members as a disagreement gate (high KL -> skip or
   cut size).
Accept: unit tests with known distributions; gate activity logged; backtest
shows whether each use adds value (keep only those that do).

**T13-22 Box arbitrage engine.**
Same venue: buy YES and NO when `yes_ask + no_ask + fees < 1.00`.
Cross venue: buy YES on A and NO on B when
`yes_ask_A + no_ask_B + fee_A + fee_B + slippage < 1.00` AND settlement
equivalence is proven (T13-11, F13). Suggest: build on `crossvenue.py` and
`crossvenue_edge.py`; use `fees.py` and `_taker_fee` (Poly fee is rounded per
order, so order size changes the fee); require both legs' depth; leg risk
handling (place the thinner leg first as maker, cancel on timeout via the
resting guard); treat near-zero-edge boxes as latency races, not free money.
Accept: paper replay over stored books shows realized box capture net of
fees; unproven-equivalence pairs are tagged `directional`, not `box`.

**T13-23 Fair-value comparison and edge definition.**
`edge = q_lcb - ask - fee(ask) - slippage`, where `q_lcb` is a lower
confidence bound from the ensemble. Trade only if `edge > min_edge` and
KL/disagreement gates pass. Per-venue comparison picks the better venue for
each side. Reuse `gates.py` (ATR hurdle, fee-adjusted EV, time-decay
cushion); do not re-enable the negative 15-55c band.
Accept: every order row stores q, q_lcb, ask, fee, edge, gate verdicts.

**T13-24 ML model upgrade.**
Why: torch `ml_model.py` is opt-in and retrains on a Termux timer. Suggest:
on the VPS (aarch64 glibc) install xgboost/lightgbm natively and delete the
proot `ml_python` wrapper; features = GBM q, KL, composite-vs-venue basis,
vol regime, time-to-expiry, book imbalance, spread, recent fill rate; labels
= settled outcome; walk-forward training with purge/embargo; export
calibrated probabilities (isotonic/Platt); retrain via n8n (T13-40). Keep
ML paper-only until it beats GBM out of sample.
Accept: model card (data window, n, calibration, feature importances)
written to MemPalace KG; promotion needs out-of-sample win over GBM.

**T13-25 LLM evaluation (cached, advisory, vetoing).**
Why: `llm_eval.py` has a 7-layer fallback, but LLM latency is not subsecond.
Suggest: run the LLM when a market is DISCOVERED (minutes before the window),
not at decision time; structured JSON output {direction_bias, event_risk,
confidence, reasons[]}; cache per market with TTL; hot path reads the cache.
Role: veto/size-down on event risk (news, outage, exchange anomalies), not
originate trades. Log every verdict and later score it against outcomes; if
it has no demonstrated lift, weight it to zero.
Accept: p99 hot-path cost of the LLM gate < 1 ms; verdict-vs-outcome table
exists after one week.

### PHASE D - Backtests and 14-day forecasts

**T13-30 Backtest harness upgrade.**
Why: `backtest.py`, `backtest_engine.py`, `backtest_real.py`,
`parallel_backtest.py` overlap. Suggest: one engine with: replay from stored
ticks and books (T13-12), the true fee models, taker-at-ask-plus-slippage and
maker fill models (keep the `PM_PAPER_FILL_MODEL` separation), market-level
clustering for every interval, walk-forward splits with purge/embargo,
parameter freezing before evaluation (R4). Minimum evidence threshold:
report "insufficient evidence" below a configurable number of clustered
markets (suggest 200) instead of a forecast.
Accept: reproduces `measure_edge.py --cluster market` numbers on the same
data; a seeded run is bit-for-bit repeatable.

**T13-31 Multi-model 14-day forecaster.**
Produce, per venue and combined, an equity-path distribution from the
CURRENT real balance for these model families, all scored against each other
and against a naive constant-daily-return baseline:
- M1 clustered bootstrap of per-market net returns, resampled to the
  expected number of trades per day.
- M2 GBM equity Monte Carlo with drift and vol fit to daily P&L.
- M3 Bayesian beta-binomial win rate per price band (credible lower bound)
  compounded through the real payoff asymmetry (small win vs large loss).
- M4 regime-switching (2-3 state, vol-driven) simulation.
- M5 gradient-boosted per-market win probability -> simulated paths.
- M6 existing torch model -> simulated paths.
- M7 stacked ensemble weighted by walk-forward pinball/CRPS score.
Include: fees, slippage, position and liquidity caps (size <= a fraction of
touch depth), halts firing mid-path, and ruin counted. Output: percentiles
(5/25/50/75/95) of day-14 equity, P(drawdown > X), P(ruin), and the
upper-bound probability of reaching $6,250,000 per R1.
Accept: forecasts for past days are stored and scored against what happened
(forecast calibration KG fact); the forecaster refuses to publish when
evidence is below threshold.

**T13-32 Day-0 real balance snapshot.**
Suggest: at start, read balances from both venue APIs, store `{venue,
balance, ts, source}` once in the local DB, as a MemPalace drawer, and as KG
facts. All forecasts and growth reporting reference this snapshot.
Accept: no code path reads a balance from prose or from `start_balance`
defaults for live reporting.

### PHASE E - Sizing and execution

**T13-33 Sizing.**
For a binary bought at price p with lower-bound win probability q_lcb:
`f_full = (q_lcb - p_eff) / (1 - p_eff)`, `p_eff = p + fee_per_contract`.
Use `fraction * f_full` with fraction from `RISK_PROFILE` (e.g. 0.10 / 0.25 /
0.50), then cap by T13-03 limits, touch depth, and per-market venue position
limits [VERIFY current limits on both venues]. Size from the venue's current
balance, so both balances compound separately.
Accept: simulation shows size never exceeds any cap; the 14-day forecast
uses the same sizing function as live.

**T13-34 Dual-venue orchestration.**
Suggest: Kalshi and Polymarket US run as independent traders sharing the
composite spot, model service, and risk officer, with a combined dashboard
(each venue's equity, growth vs required G, forecast fan). Neither venue's
failure halts the other unless the risk officer says so.
Accept: killing one venue's process leaves the other trading.

### PHASE F - Platform pillars

#### n8n (pillar 1)

**T13-40 Workflows as code.**
Suggest: export workflows to `n8n/workflows/*.json`, import via the n8n CLI
or public API (retire `inject_n8n_workflows.py` DB injection); credentials
only in n8n's credential store; HMAC-signed webhooks; idempotency key on
every event (`emit_event(..., key=...)` already exists); retries with
backoff; dead-letter workflow.
Workflows:
- `wf-nightly-backtest-forecast` (cron): T13-30/31, posts forecast to
  MemPalace and the operator channel.
- `wf-retrain-ml` (every 6 h; replaces Termux job 7002).
- `wf-settlement-pnl` (webhook): settlement events -> ledger confirmation.
- `wf-reconcile` (every 1-5 min): venue balances/positions vs local DB;
  mismatch beyond tolerance -> halt event.
- `wf-sync-health`: alert when sync lag SLO breached.
- `wf-agent-dispatch`: routes incidents/proposals to the right agent.
- `wf-daily-report`, `wf-breaker-alert`.
Hard rule: no n8n call sits between a signal and an order.
Accept: workflows import cleanly on a fresh n8n; killing n8n for an hour
loses zero events (they replay from the outbox on return).

#### Local-first sync (pillar 2)

**T13-41 Outbox and replicas.**
Suggest: SQLite WAL + transactional outbox table; sync worker pushes to
Supabase and n8n with idempotent upserts keyed `(venue, event_id)`, monotonic
sequence numbers, exponential backoff, and a circuit breaker (the existing
sync-health breaker). Conflict policy: local wins for decisions/fills/
settlements; the venue wins for balances/positions after reconcile. Add
Litestream (or equivalent) replication to OCI Object Storage as backup.
SLOs: sync lag p95 < 5 s, outbox depth alarmed, zero loss.
Accept: network partition test (10 min offline, then heal) ends with
replicas identical; lag metric visible in the dashboard.

#### MemPalace (pillar 3)

**T13-42 Single-owner service and bridge-only access.**
Why: F6. Suggest: one systemd service owns the palace; every agent uses
`mempalace_bridge`; replace the direct Logstream open in `_mp_resume_hint()`
with a bridge call; document recovery if the owner dies; do not kill a live
holder.
Accept: two agents read/write concurrently with no lock errors.

**T13-43 Use all of it, deliberately.**
- Checkpoints: keep operational-only payloads; add forecast id and model
  version.
- Drawers (durable narrative): halts, daily calibration (exists), each
  nightly forecast report, model cards, T13-01 findings, parameter-change
  rationale, incident postmortems.
- KG facts with supersede/invalidate: measured win rate per band and asset,
  composite-vs-settlement basis, latency p99 by stage, model weights, sync
  lag, forecast calibration scores.
- Status heartbeats per agent; event topics for inter-agent traffic (T13-44).
- Search before change: any constant edit starts with a MemPalace query and
  ends with a KG write.
- File the three pending findings from HANDOFF (NO-side floor bug, 4-loss/246
  market limit, stale-balance error).
Accept: a fresh agent session can reconstruct current model versions,
measured edges and open incidents from MemPalace alone.

#### Inter-agent collaboration (pillar 4)

**T13-44 Roles, topics, protocol.**
Roles: `data_feed`, `quant_forecaster`, `poly_trader`, `kalshi_trader`,
`risk_officer` (binding veto), `ops_n8n`, `opencode_dev` (code changes via
commits only).
Topics (existing `agent-checkpoint` plus new): `signal.proposal`,
`risk.veto`, `forecast.published`, `param.change.request`, `incident`,
`sync.health`.
Envelope: `{id, correlation_id, from, to, topic, ts, ttl_ms, payload,
ack_required}`; hot-path messages ack within 2 s; heartbeat every 5 s; a
missed heartbeat is an incident.
Rules: nobody edits live config at runtime. A parameter change is
proposal -> risk_officer approval -> commit -> new `config_hash` (already
computed) -> restart or hot-reload. Sub-second coordination stays in-process
(asyncio tasks/queues); logstream is the durable mirror and the cross-process
path.
Accept: a vetoed signal never reaches an order; a killed agent is detected
within 15 s and reported.

#### Subsecond latency (pillar 5)

**T13-45 Event-driven hot path.**
Suggest: replace the 15 s poll for entries with an asyncio loop driven by WS
ticks and BBO updates (keep slow polling only for discovery/reconcile);
`uvloop` and `orjson`; persistent HTTP/2 keep-alive pool to each venue;
pre-built signed-request templates/keys loaded once (Kalshi signing);
precomputed per-market state (strike, tau, cached LLM verdict, ML q) so the
decision is arithmetic on fresh ticks; ledger/DB writes stay on the writer
thread (L6). Measure with the existing latency recorder per stage.
Budget (p99): tick->decision 250 ms; decision->wire 300 ms; end-to-end
1 s. Measure RTT to each venue from candidate OCI regions and choose the
closest [VERIFY]. Enforce clock sync (chrony) since Kalshi requests are
timestamp-signed.
Accept: 24 h soak with stage histograms published; breach of budget raises an
incident; regression test fails if hot path imports anything that blocks.

**T13-46 Split poly_us_trader.py.**
Why: F9. Suggest: modules for state/db, ledger writer, fees, paper fills,
MemPalace hooks, guard wiring, cycle. Behavior-preserving; tests first.
Accept: identical ledger output on a recorded replay before and after.

### PHASE G - OCI VPS deployment

**T13-50 Run it as services.**
Suggest: systemd units per role with `Restart=always` and `WatchdogSec`
(`resilience.systemd_notify` already exists), `EnvironmentFile` or OCI Vault
for secrets, separate unprivileged user, firewall closed except what n8n
needs, chrony, log rotation, disk alerts, `Litestream` unit. Remove
Termux-only pieces (`termux-job-scheduler`, `ml_python` proot wrapper).
Accept: reboot test brings every service back; `--once --dry` passes.

---------------------------------------------------------------------------
## 5. CARRY-FORWARD LEDGER (unapplied items from earlier rounds)

I could not read AGENT_TASKSv10-v12 (not visible in the root listing; the
docs/ directory page was blocked to my fetcher). So the agent must do this
FIRST:

1. Diff each task ID in AGENT_TASKSv10-v12 against `git log` and tests. Any
   task with no commit and no passing acceptance check is unapplied: copy it
   into a new section "Carried forward" at the bottom of THIS file with its
   original ID, and schedule it before Phase C.
2. Known-open items from HANDOFF.md that remain relevant:
   - human review of 304d8f8 (-> T13-02)
   - MemPalace lock (-> T13-42)
   - three MemPalace findings to file (-> T13-43)
   - ponytail-audit never actually ran (-> T13-07)
   - re-run `scripts/measure_edge.py --cluster market` on fresh data before
     touching any band constant; if the clustered lower bound stops clearing
     breakeven, widen the band back (that is the honest outcome)
3. From the v12 goal: Binance + Coinbase WS (-> T13-10), both venues with
   growth tracking (-> T13-34), multi-model 14-day forecasts (-> T13-31).

---------------------------------------------------------------------------
## 6. SUGGESTED ORDER AND DEFINITION OF DONE

Order: A (T13-01..07) -> B (10..12) -> D harness (30, 32) -> C (20..25) ->
D forecaster (31) -> E (33, 34) -> F (40..46) in parallel where independent
-> G (50).

Done for this round means: repo clean of secrets; no dropped money records;
one source of truth; composite Binance+Coinbase spot live with failover;
GBM, KL, box, fair-value and gated ML/LLM paths paper-trading on both
venues; nightly multi-model forecast scored against reality; n8n workflows
imported from repo; sync lag measured; MemPalace single-owner with KG facts
flowing; agents exchanging vetoed/acked messages; p99 tick-to-order under
1 s with a published histogram. Reaching $6.25M is NOT a done criterion.

---------------------------------------------------------------------------
## 7. OPEN QUESTIONS FOR GENTRY

Q1. Current real balance on each venue (the agent will read them, but you
    know which account is which)?
Q2. Is the repo meant to be public? (F1)
Q3. Acceptable drawdown and daily-loss fractions for each RISK_PROFILE?
Q4. Which OCI region/shape is the VPS, and is a non-US region acceptable
    for Binance.com data, or should Binance.US be used?
Q5. Which operator channel should n8n alerts go to?
