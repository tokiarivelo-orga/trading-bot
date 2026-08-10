# Observability & Trading-Performance Plan

Goal: make the bot **debuggable** (know exactly why every signal did or didn't
become a trade), **measurable** (execution quality, not just win rate), and
**improvable** (regime-aware, cost-aware strategy research).

Derived from a read of the current code on branch `feat/bot-session-replay`
(2026-08-05). Findings that motivated it:

- `shared/logging/setup.py` is plain `logging.basicConfig` — printf messages,
  no structured logging, no metrics, no tracing.
- `activity/application/bot_signals.py` **regex-parses free-text log lines**
  back into `BotSignal`s. The decision trail is therefore coupled to log
  message wording; it has already broken twice (space-containing symbols,
  `\S+` skill parsing).
- `_OUTCOME_PREFIXES` collapses 5 distinct block reasons into `risk_rejected`,
  so "what did the volatility guard cost me" is unanswerable.
- Grep of `backend/src` for `mfe|mae|excursion|slippage|latency|retcode`
  returns **zero** hits — no execution-quality telemetry exists anywhere.
- `journal/domain/analytics.py` computes only outcome metrics (win rate, PF,
  expectancy, max DD).
- Backtests do not simulate broker `stops_level`, so M1 scalp backtests are
  fictional (see memory `broker-stops-level-not-enforced`).

## Rules for every phase

- Each phase runs in a **fresh subagent context**.
- Before-done gate, from `backend/`: `uv run ruff check src tests` and
  `uv run pytest`. Frontend phases: `make lint-frontend` and
  `make build-frontend`.
- CLAUDE.md is binding: hexagonal layering, explicit `response_model` +
  documented `Field`/`Query` on every route, no pydantic in `domain/`,
  never modify `configs/risk.yaml` limits.
- After each phase the orchestrator does a **repo-wide git diff audit**
  (including `git log --all` and `git worktree list`) — subagent
  self-reports of "I only touched X" are not trusted.
- This file is updated with the outcome before the next phase starts.

---

## Phase 1 — Typed signal decisions (replace log scraping)

**Status:** ✅ done (commit `a246d11`, 2026-08-05)

Landed: `SignalDecision` in `activity/domain/models.py`; `signal_decisions`
table (migration `b7c1d2e3f4a5`) + `activity/adapters/signal_decision_repository.py`;
`activity/ports/signal_decisions.py` Protocol wired in `container.py`;
`activity/application/signal_decision_service.py`; recording in
`engine/application/trade_loop.py` and `broker/application/order_service.py`;
`activity/api/routes.py` serves the trail from the table and falls back to
the legacy log-scrape for older rows; `bot_signals.py` marked
LEGACY / BACKFILL ONLY. Tests: `test_signal_decision_repository.py`,
`test_order_service_signal_decisions.py`, `test_signal_decisions.py`.

Audit notes: the agent also deleted four in-use strategy/skill files
(`scalp_bollinger_reversion_v1_v2.py`, `scalp_ema_cross_v1_v1.py` and their
YAMLs) — out of scope, reverted. Two `E501` violations it introduced in
`trade_loop.py` were fixed.

Introduce a first-class decision record so the live decision trail stops being
parsed out of log text.

- New `SignalDecision` domain dataclass in `activity/domain/models.py`:
  `signal_id` (uuid), `account_id`, `bot` (skill), `strategy`, `symbol`,
  `timeframe`, `direction`, `price`, `created_at`, `outcome` (enum),
  `reason`, `confidence`.
- Persisted via a new `signal_decisions` table + repository in
  `activity/adapters/`, written by the engine through a port so
  `engine/application/` stays adapter-free.
- `TradeEngine._try_enter` / `_enter_for_bot` and
  `OrderService.open_position` record the decision and its outcome instead of
  relying on downstream regex.
- `extract_bot_signals` keeps working for **historical** rows (backfill path)
  but new reads come from the table.
- Human-readable log lines stay — they just stop being the source of truth.

**Done when:** the chart's signal trail is served from `signal_decisions`,
`bot_signals.py` is documented as legacy-only, tests cover each outcome.

## Phase 2 — Veto funnel (structured checks)

**Status:** ✅ done (uncommitted working tree, 2026-08-05)

Landed: `DecisionCheck` + `SIGNAL_OUTCOMES` and `SignalDecision.checks` in
`activity/domain/models.py`; `signal_decisions.checks` JSON column (migration
`c8d2e3f4a5b6`) with append semantics in the repository; `RiskDecision.code`
and `SpreadVeto.kind/value/threshold` so the gates are identified structurally
rather than by parsing their prose; per-gate outcomes stamped in
`trade_loop.py` (`htf_veto`, `volatility_guard`, `max_positions`,
`risk_sizing`, `daily_loss_breaker`) and `order_service.py` (`spread_veto` vs
`rr_gate`, `broker_rejected`, `opened`); pure aggregation in
`activity/domain/funnel.py`; `GET /activity/signals/funnel`; frontend
`SIGNAL_OUTCOME_META` extended with all five new values in the same change
(it is indexed unguarded), plus `features/analytics/SignalFunnelPanel.tsx` +
`useSignalFunnel.ts`.

Notes: the funnel's stage order follows the engine's **real** gate order
(HTF → volatility/cap/sizing → spread/RR → fill), not the prose order below,
so the counts stay monotonic. The funnel has no legacy log-scrape fallback on
purpose — the old vocabulary collapsed every risk block into one bucket, which
is the ambiguity it exists to remove. `bot_signals.py` still maps historical
log lines onto the old vocabulary, and `risk_rejected` stays in both the
backend vocabulary and `SIGNAL_OUTCOME_META` so those rows keep rendering.

- Add `checks: tuple[DecisionCheck, ...]` to `SignalDecision` —
  `(name, value, threshold, comparison, passed)`, mirroring the shape already
  used by `TradeRecord.indicators`.
- Split the collapsed `risk_rejected` bucket into distinct outcomes:
  `htf_veto`, `spread_veto`, `rr_gate`, `volatility_guard`, `max_positions`,
  `risk_sizing`, `broker_rejected`, `daily_loss_breaker`, `opened`.
  Note: the chart indexes `SIGNAL_OUTCOME_META[outcome]` unguarded — it must
  be extended in the same change or it will crash.
- New endpoint returning the funnel per bot per period: signals fired →
  passed HTF → passed spread → sized OK → filled, with counts and drop
  reasons.
- Frontend: funnel panel in `features/analytics/`.

**Done when:** a bot's page answers "of 120 signals, why did only 14 trade?"

## Phase 3 — Execution telemetry

**Status:** ✅ done (uncommitted working tree, 2026-08-05)

Landed: `TradeRecord.requested_price/slippage/execution_latency_ms/
broker_retcode/mfe/mae` + migration `d9e0f1a2b3c4` + ORM + repository
mapping; `broker.domain.trading.execution_slippage` (pure, one sign
convention: positive always means the fill cost the trader);
`ExecutionResult.retcode` and `OrderRejected.retcode`; measurement in
`OrderService.open_position` (injected `clock` for deterministic latency
tests), carried to the journal on `PositionOpened`; `trade_loop.py` passes
`signal_emitted_at=now`, the same instant the signal's `SignalDecision` was
recorded with. `journal/domain/excursion.py` holds the pure MFE/MAE
arithmetic, accumulated by `TradeJournalService.on_candle_closed`
(subscribed to `CandleClosed`, M5 only) and finalized against the exit price
in `on_position_closed`; new `MarketContextPort.latest_candle` and
`JournalRepository.get_open_excursions`/`update_excursion` keep that path
cheap. `BotAnalytics` gains `avg_slippage`, `measured_slippage_count`,
`avg_execution_latency_ms`, `retcode_histogram`, `avg_mfe`, `avg_mae`,
`mfe_mae_ratio`, `avg_mfe_on_losers`, `avg_mae_on_winners`, all on
`GET /journal/analytics/bots`; frontend surfaces them as four new columns on
the existing `BotPerformanceTable`.

**The gateway did drop the retcode** — it only ever appeared inside the
free-text `Mt5Error` message, and never at all on a successful fill. The
gateway response contract was extended: `OrderResultOut.retcode`, and 502
refusals now carry `detail={"message", "retcode"}`. The backend adapter
accepts a plain-string `detail` too, so a gateway/backend version skew
degrades to "no retcode recorded" rather than crashing the order path.

A *rejected* order produces no `TradeRecord`, so its retcode is recorded on
the signal's decision trail instead, as a `DecisionCheck(name=
"broker_retcode", threshold=10009, passed=False)`.

Add to `TradeRecord` (+ migration, + journal write path):

- `requested_price` and resulting **slippage** (fill vs requested).
- **execution latency** — signal emit → broker ack, milliseconds.
- **broker retcode** on both fills and rejects (e.g. MT5 `10016` invalid
  stops, which silently killed a whole VIX75 fleet before).
- **MFE / MAE** per trade — max favorable / adverse excursion, computed from
  candles while the position is open and finalized on close.

Surface them in `journal/domain/analytics.py`: avg slippage, avg latency,
retcode histogram, and MFE/MAE ratios (are TPs too far? SLs too tight?).

**Done when:** analytics shows cost-per-trade and excursion stats per bot.

## Phase 4 — Backtest realism & live/backtest divergence

**Status:** ✅ done (`5660aca`)

- Simulate broker constraints in the backtest engine: `stops_level`, min lot,
  lot step, spread widening, and a slippage distribution sampled from the
  real data collected in Phase 3.
- Reject backtest entries that a live broker would reject, and count them.
- A divergence report comparing live vs backtest fills for the same signal
  conditions — systematic gaps mean the simulator is lying or the edge decayed.

**Done when:** an M1 scalp backtest no longer reports trades the broker would
have refused.

**What landed:**

- `broker/domain/broker_constraints.py` — pure `stops_level` / volume-grid
  rules carrying the real MT5 retcodes (10016 invalid stops, 10014 invalid
  volume). The VIX75 case from the project notes (minimum stop distance
  107.70 price units vs M1 scalp stops of 30–70) has a direct regression test.
- `broker/domain/slippage.py` — slippage calibrated from the real
  `TradeRecord.slippage` values Phase 3 began recording, with a documented
  pessimistic fallback below `MIN_CALIBRATION_SAMPLES` (20) and an explicitly
  seeded RNG, so identical inputs still produce identical trades.
- `broker/ports/execution_simulator.py` — the behaviour is *injected* into
  `PaperBroker`, not baked into it; `execution_simulator=None` keeps the old
  frictionless fills for live paper trading and plumbing tests.
- `backtest/adapters/constraint_simulator.py` — counts every rejection by
  reason, reported as `broker_realism` on the report and rendered by
  `BrokerRealismPanel.tsx`.
- `backtest/domain/divergence.py` + `GET /backtest/reports/{id}/divergence` —
  separates "the simulator is lying" from "the edge decayed" by *which*
  metrics diverge, rather than averaging both into one score.
- `backtest/adapters/decision_sink.py` — backtests now record signals through
  `SignalDecisionSinkPort` in the Phase 2 split vocabulary instead of
  regex-scraping their own logs, so live and backtest funnels are finally
  comparable. `extract_signals()` remains only as a fallback for old reports.

**Note:** `simulate_broker_constraints` defaults to **True**, so every
existing backtest changes. Reports predating this carry `enabled: false`,
which correctly describes them — they filled every order at the bar's closing
quote and may list entries a live broker would have refused.

**Bug found and fixed while verifying this phase:** `run_backtest` passed the
simulated clock to the bookkeeper, trade engine and activity capture but
*not* to `OrderService`. Phase 3's execution-latency measurement therefore
subtracted a historical candle timestamp from wall-clock time, recording a
~1.6-year latency on every backtest trade (`latency=50291226875ms` observed)
and poisoning the Phase 3 latency analytics. Now reads `0ms`.

`test_phase5_backtest_flow` pins the *nominal* R multiples `breakout_v1`'s
`TP_RR` defines, so it runs frictionless; slippage's effect on realized R is
covered by the Phase 4 suites instead.

**Not measured:** the before/after profit-factor impact on a real M1 scalp.
The A/B harness ran ~10 min per side and was cut short; the deliverable
itself is test-verified, but the size of the PF drop on the live fleet is
still unquantified. Worth doing before trusting any existing PF number.

## Phase 5 — Metrics, correlation IDs, log hygiene

**Status:** done (2026-08-07, `2d3c92b`)

- Prometheus metrics endpoint: engine loop duration, gateway RTT, signals/min,
  veto counts by reason, open positions, WS client count.
- Correlation IDs — extend the existing `ContextVar` pattern in
  `shared/logging/account_context.py` to also carry `signal_id`, so every line
  from signal → sizing → order → fill → journal joins up.
- Structured (JSON) log output alongside the human format.
- Log-level hygiene: routine per-bar chatter moves to DEBUG; INFO becomes
  decisions only.
- Silence alerting: warn when a bot emits no signal for N× its median
  interval (a dead bot currently looks like a quiet market).

**Done when:** `/metrics` scrapes clean and INFO is readable end to end.

**Outcome:** `GET /metrics` (unauthenticated, like `/health`) exposes all
six planned families via `shared/metrics/registry.py`, wired at the event
bus (`PositionOpened`/`PositionClosed` → `tradingbot_open_positions`) and via
an `httpx` event-hook pair scoped per account for gateway RTT. `signal_id` is
minted in `trade_loop._try_enter` before the `SIGNAL:` log line, carried
through `ContextFilter` across the `QueueHandler` thread boundary (the
existing ContextVars don't survive that hop), stored on `activity_logs`, and
filterable via `GET .../activity/history?signal_id=`. `configs/alerting.yaml`
gained a `silence:` block (poll interval, lookback, multiplier, min-signals
floor) driving `SilenceMonitor`, which publishes `BotWentSilent` per account,
alerted through the existing `AlertService`. Verified: full suite (excluding
two pre-existing, unrelated collection failures in
`tests/unit/strategies/test_scalp_bollinger_reversion_v1.py` /
`test_scalp_ema_cross_v2.py` that import strategy modules absent from
`src/strategies/generated/` at HEAD) — 1501 passed, 1 failed
(`test_risk_config_has_user_owned_caps`, the known pre-existing
`risk.yaml`/`max_trades_per_day_enabled` gap, unrelated to this phase);
`ruff check` clean on all 40 touched files; `/metrics` scrape and OpenAPI
schema spot-checked directly.

## Phase 6 — Regime tagging & walk-forward research

**Status:** done — Pass A (`831136a`) + Pass B (`0f32b86`)

- Tag every trade and decision with the market regime at entry: ATR
  percentile bucket, trading session, trend/range classification. **[Pass A]**
- Regime-split analytics — PF and expectancy per regime, per bot. **[Pass A]**
- Walk-forward backtesting harness (rolling in-sample/out-of-sample) to
  replace single-window PF numbers, which are very likely overfit
  (e.g. `pob_snd_zones_xauusd` PF 3.70 on one window). **[Pass B]**
- Cost-as-%-of-gross-edge metric per bot — expected to show M1 scalps
  spending their entire edge on spread + slippage. **[Pass A]**

**Done when:** each bot's edge is reported per regime, out-of-sample.

**Pass A outcome (2026-08-07, `831136a`):** New `engine/domain/regime.py`
classifies trend/range via ADX against a fixed threshold (not
percentile-ranked like ATR — ADX is already a normalized 0-100 score) and
trading session via UTC hour; reuses the existing ATR-percentile volatility
classifier as-is. `trade_loop.py` computes one regime snapshot per signal,
always on, independent of the existing gated volatility guard, so every
decision is tagged, not only the ones that reach that gate. `order_service.py`
computes `transaction_cost` per fill. `compute_regime_analytics()` slices
win-rate/PF/expectancy by one regime dimension at a time (volatility, trend,
session) rather than the full sparse cross-product; `BotAnalytics` gains
`cost_pct_of_gross_edge`. New `GET .../journal/analytics/regimes` route;
frontend regime panel (tabbed by dimension) plus a "Cost % of edge" column on
the bot performance table. Migration adds nullable `regime_*`/
`transaction_cost` columns to `trades` and `signal_decisions` — `None`
throughout for pre-Phase-6 rows or signals whose entry timeframe had no
candles, never fabricated.

Gates: ruff clean on every Pass A file, verified in an **isolated git
worktree** (not just the shared working tree) since several touched files
(`journal/domain/models.py`, `repository.py`, `trade_journal.py`,
`journal/api/{schemas,routes}.py`, `client.ts`, `test_analytics.py`,
`test_repository.py`) had unrelated concurrent-session work interleaved line-
by-line — staged by reconstructed content, not by path, to exclude a
substring-search/`total_profit` search feature, an MFE/MAE-timestamp
excursion feature, and `client.ts`'s CandleGap additions, none of which are
this phase's. Full suite in that isolated worktree: only the one known
pre-existing failure (`test_risk_config_has_user_owned_caps`); `make
lint-frontend` clean (frontend build itself couldn't be verified in the
isolated worktree — Turbopack rejects a symlinked `node_modules` outside the
repo root — so the three new frontend files were verified by manual
line-by-line review against the live `useAnalytics.ts`/`BotPerformanceTable.tsx`
patterns instead).

Two things the audit caught beyond the diff itself: (1) a subagent pass had
added one unrequested line to `trade_loop.py` changing empty-`strategy.spec.
symbols`-list routing semantics (unrelated to regime tagging, in a
money-touching path) — reverted before staging; (2) `backend/src/backtest/
reports/writer.py` (and `__init__.py`) turned out to have never been
committed at all — `.gitignore`'s blanket `reports/*` rule (meant to keep
generated report JSON out of git) had no carve-out for source files sharing
that directory, so HEAD has been fresh-checkout-broken since Phase 4,
invisible only because every session shares one long-lived working tree.
Fixed separately in `28f4402`, ahead of the Pass A commit.

**Pass B outcome (2026-08-10, `0f32b86`):** This repo's generated strategies
have fixed parameters and no parameter-sweep infrastructure exists, so
"walk-forward" here means independent out-of-sample folds via the existing,
unmodified `run_backtest()` — not classic in-sample-refit optimization,
since there's nothing to refit; documented explicitly in `walk_forward.py`'s
own docstring so it's never mistaken for the textbook definition. New
`backtest/application/period.py::split_into_folds()` divides a period into
consecutive `fold_months`-sized windows; `walk_forward.py::run_walk_forward()`
runs each fold independently, skips a fold on missing history rather than
aborting the whole run, and aggregates mean/stddev/worst-fold PF and
expectancy — a zero-trade fold reports `profit_factor=None` (undefined),
never `0.0` (which would read as a loss). New `reports/walk_forward_writer.py`
writes one aggregate JSON per run to its own `reports/walk_forward/`
directory, so the existing single-backtest report listing never has to
handle a differently-shaped file; `reports/writer.py`'s JSON-serialization
helper is now public (`to_jsonable`) so both writers share it. New
`POST/GET/DELETE .../backtest/walk-forward/{run,reports}` routes in a
separate job store from the existing single-run machinery; new
`walk_forward_cli.py` mirrors the existing single-backtest CLI; new `make
walk-forward-backtest` target.

Gates: ruff clean on every touched file; `tests/unit/backtest/` — 162
passed, confirmed mocked throughout (`run_backtest`/`run_walk_forward`
never actually invoked, so the suite never runs a real ~7-minute backtest);
OpenAPI spot-check confirms every new route/field is documented per
CLAUDE.md. A third concurrent session (a deep-learning "SMC" strategy
effort) is actively editing `run_backtest.py` — left untouched and verified
byte-identical throughout, treated strictly as a black box. That same
session had also added its own Makefile targets (`train-dl`,
`export-dataset`) landing adjacent to this phase's new
`walk-forward-backtest` target in one diff hunk — excluded from this commit
by reconstructed content, same treatment as Pass A's shared-file collisions.

---

## Progress log

| Date | Phase | Result |
|---|---|---|
| 2026-08-05 | — | Plan created from code audit. |
| 2026-08-05 | 1 | Done, committed `a246d11`. Gates: ruff clean on touched paths; pytest 1133 passed / 1 failed; `make lint-frontend` + `make build-frontend` pass. |
| 2026-08-05 | 3 | Done, left uncommitted for review. Gates: ruff clean on every touched path (repo-wide `src tests` still reports the same ~298 pre-existing errors, all in `strategies/generated/xauusd_snd_qm_structure_*` + `test_position_manager.py`/`test_ws.py`); pytest passes apart from the three known pre-existing failures; `make lint-frontend` + `make build-frontend` pass. Note: `alembic upgrade head` had to be run on the dev DB — Phase 1/2's migrations had never been applied either, which is what `tests/unit/test_health.py` was failing on. |
| 2026-08-05 | 2 | Done, left uncommitted for review. Gates: ruff clean on touched paths (repo-wide ruff has ~300 pre-existing errors, all in `strategies/generated/` + two unrelated test files); pytest passes apart from the three known pre-existing failures below; `make lint-frontend` + `make build-frontend` pass. |
| 2026-08-06 | 4 | Done, committed `5660aca`. The phase's subagent was killed by a session limit, so the work was audited, finished and gated by the orchestrator instead. Gates: ruff clean on every Phase 4 path (repo-wide `src tests` still reports the same 295 pre-existing errors in `strategies/generated/xauusd_snd_qm_structure_*`, unmodified vs HEAD); pytest 1333 passed, 1 failed — the known `risk.yaml` assertion, down from 2 because this phase fixed the `r_multiple` regression it had introduced; `make lint-frontend` + `make build-frontend` pass. Audit caught the recurring scope creep for the third time: `scalp_bollinger_reversion_v1.yaml` and `scalp_ema_cross_v1.yaml` deleted again, both restored. Committed by explicit path — a concurrent session's in-flight market-data/chart work shares `shared/api/client.ts`, so only this phase's three hunks were staged from it. |
| 2026-08-07 | 5 | Done, committed `2d3c92b`. Metrics endpoint, signal_id correlation threaded signal→sizing→order→fill→journal, structured JSON log option, log-hygiene pass, silence alerting. Gates: ruff clean on all 40 touched files; pytest 1501 passed, 1 failed (known `risk.yaml` assertion); `/metrics` scrape and OpenAPI schema spot-checked directly. |
| 2026-08-07 | 6 (Pass A) | Done, committed `831136a` (+ `28f4402`, `ebef05a`). See outcome notes above. |
| 2026-08-10 | 6 (Pass B) | Done, committed `0f32b86`. See outcome notes above. Phase 6 complete. |

## Known pre-existing breakage (NOT caused by this plan's work)

Present at HEAD `8bfc925`, before any Phase work — recorded so later phases
aren't blamed for it:

1. `tests/unit/shared/test_config.py::test_risk_config_has_user_owned_caps`
   fails — it asserts `max_trades_per_day_enabled` is in `configs/risk.yaml`,
   but that key has never existed in the file. `risk.yaml` is user-owned
   (CLAUDE.md), so the **test** is what should change, not the config —
   needs a user decision on whether that flag is wanted at all.
2. `tests/unit/strategies/test_scalp_bollinger_reversion_v1.py` and
   `test_scalp_ema_cross_v2.py` fail to import: they reference
   `scalp_bollinger_reversion_v1_v1` and `scalp_ema_cross_v1_v2`, modules
   that have never existed in git. Orphan tests; likely safe to delete.
