"""FastAPI entrypoint.

Run with: uv run uvicorn src.main:socket_app --reload --port 8000

`socket_app` wraps `app` with the Socket.IO ASGI layer (see
`src.market_data.api.ws`) — Socket.IO handles its own path prefix
(`/socket.io/`) and forwards everything else to the FastAPI app underneath.

Interactive API docs (generated from the `response_model`/`summary`/
`description` on every route — see each module's `api/routes.py` and
`api/schemas.py`):
  - Swagger UI: http://localhost:8000/docs
  - ReDoc:      http://localhost:8000/redoc
  - Raw schema: http://localhost:8000/openapi.json  (or `make openapi`)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import socketio
from fastapi import Depends, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.activity.api.routes import router as activity_router
from src.ai.api.routes import router as ai_router
from src.ai.api.routes_refinement import router as ai_refinement_router
from src.ai.api.routes_regeneration import router as ai_regeneration_router
from src.ai.api.routes_settings import router as ai_settings_router
from src.backtest.api.routes import router as backtest_router
from src.broker.api.accounts import router as accounts_router
from src.broker.api.routes import router as account_router
from src.broker.api.trading_routes import router as trading_router
from src.broker.api.trading_routes import spread_router as broker_spread_router
from src.broker.domain.account import BrokerUnavailable
from src.container import build_container
from src.engine.api.routes import router as engine_router
from src.indicators.api.routes import router as indicators_router
from src.journal.api.routes import router as journal_router
from src.market_data.api.routes import router as market_data_router
from src.market_data.api.ws import bind_auth, bind_candle_stream, bind_live_candle, sio
from src.market_data.application.candle_stream import poll_lookback_for
from src.market_data.domain.models import Timeframe
from src.news.api.routes import router as news_router
from src.order_book.api.routes import router as order_book_router
from src.shared.auth.api.routes import router as auth_router
from src.shared.auth.dependencies import require_session
from src.shared.config.settings import Settings, load_yaml_config
from src.shared.logging.setup import configure_logging
from src.shared.metrics.registry import REGISTRY, set_open_positions, set_ws_client_source
from src.skills.api.routes import router as skills_router
from src.strategies.api.routes import router as strategies_router
from src.strategies.api.routes import sandbox_router as strategies_sandbox_router
from src.strategies.api.routes_training import router as model_training_router

logger = logging.getLogger(__name__)


API_DESCRIPTION = """
REST + WebSocket API for the AI trading bot backend.

The backend never talks to MetaTrader5 directly — every broker/market-data
call is proxied through the MT5 Gateway HTTP service (see
`gateway/src/gateway/main.py`). Money-touching endpoints (`/broker/orders`,
`/broker/positions/*`) go through the spread/RR gate in
`broker/application/spread_gate.py` before ever reaching the broker adapter.

Live candle streaming is Socket.IO, not REST — see the `market-data` tag
description and `src/market_data/api/ws.py` for the `subscribe` /
`unsubscribe` / `candle_closed` / `candle_update` event contract.
"""

OPENAPI_TAGS = [
    {
        "name": "meta",
        "description": "Liveness check — no auth, no gateway dependency. Runtime configuration "
        "(`/config/app`) requires a session like every other route below.",
    },
    {
        "name": "auth",
        "description": "Login/logout for the shared app password (§11) — the bot can place "
        "live trades, so every route except this one and `/health` requires the session "
        "token `POST /auth/login` issues. When `TB_APP_PASSWORD` is unset, no session is "
        "required anywhere (bare local dev).",
    },
    {
        "name": "accounts",
        "description": "`GET /accounts` — every account wired up from `configs/accounts.yaml`, "
        "for the frontend's account switcher. Every other tagged route below (except `auth`, "
        "`ai-settings`, `skills`, `news`, `indicators`, and `backtest`, which are process-wide) "
        "is scoped under `/accounts/{account_id}/...` — that `account_id` comes from here "
        "(MULTI_ACCOUNT_PLAN.md Phase 6).",
    },
    {
        "name": "account",
        "description": "MT5 login/logout and connection status for one account "
        "(`/accounts/{account_id}/account/...`). Passwords transit request bodies only — "
        "never query strings, logs, or responses.",
    },
    {
        "name": "broker",
        "description": "Manual order placement and open-position management for one account "
        "(`/accounts/{account_id}/broker/...`). Every order passes the per-symbol spread/RR "
        "gate before reaching the broker adapter (paper or live, per `configs/app.yaml: mode`). "
        "The per-symbol spread/RR config itself (`/broker/symbols/{symbol}/...`) is process-wide "
        "and stays unprefixed — `SpreadGate` is shared across every account.",
    },
    {
        "name": "market-data",
        "description": "Historical candles and live symbol specs over REST, per account "
        "(`/accounts/{account_id}/market-data/...` — different brokers can quote different "
        "spreads/specs for a nominally identical symbol). Live candle streaming is Socket.IO "
        "(rooms per `symbol:timeframe`), documented in `src/market_data/api/ws.py` — Next.js "
        "rewrites don't proxy WS, so the frontend connects to this backend directly for it.",
    },
    {
        "name": "journal",
        "description": "Read-only trade history and chart markers for one account "
        "(`/accounts/{account_id}/journal/...`), written automatically by the broker's "
        "`PositionOpened`/`PositionClosed` events.",
    },
    {
        "name": "activity",
        "description": "Read-only, persisted activity log for one account "
        "(`/accounts/{account_id}/activity/...`) — every backend module's INFO+ log line "
        "(signal generated, HTF veto, risk gate block, spread veto, order filled, circuit "
        "breaker), searchable after the fact. This is the durable record behind 'what is the "
        "bot doing right now and why', independent of stdout/journal (which only covers "
        "trades that actually filled).",
    },
    {
        "name": "engine",
        "description": "Automated trade-loop status and the manual kill switch for one account "
        "(`/accounts/{account_id}/engine/...`). AI refinement logic and dev tooling must never "
        "call `.../engine/kill` or `.../engine/resume` — those are user-triggered controls only.",
    },
    {
        "name": "backtest",
        "description": "Read-only backtest reports written by `python -m src.backtest.cli` "
        "(or `make backtest`). There is no run-a-backtest endpoint — a backtest reads "
        "the same DB the live app does and can take a while, so it stays a CLI-only "
        "operation for now; this API only lists/reads the report files it produces.",
    },
    {
        "name": "ai",
        "description": "PDF -> StrategySpec -> code pipeline (F4), and the 10-trade "
        "self-refinement loop (F5), per account (`/accounts/{account_id}/ai/...` — each "
        "account has its own draft/proposal/strategy-version history). Every step is "
        "human-gated: upload/review only produces a draft or proposal, and code generation/"
        "refinement only ever produces a 'validated' strategy version — never an active, "
        "tradeable one, unless the refinement policy is 'auto' and its backtest threshold is "
        "met. See the `strategies` tag for activation.",
    },
    {
        "name": "ai-settings",
        "description": "Per-task AI provider selection (Claude Code, Hermes Agent via Ollama, "
        "Ollama, OpenClaw) for document analysis, strategy generation, and trade-review/"
        "refinement. Changes apply without a backend restart — see the `ai` tag for the "
        "tasks themselves.",
    },
    {
        "name": "model-training",
        "description": "Trigger and inspect deep-learning model training (`/model-training/...`). "
        "Process-wide and unprefixed by account, like `backtest` and `indicators`: there is one "
        "set of weights in `data/ml_models/` shared by every account. `POST /model-training/runs` "
        "launches `scripts/train_smc_dl_v2.py` as a background subprocess — it overwrites that "
        "symbol's weights/scaler/metadata and nothing else, never activating a strategy, touching "
        "a position, or reading a risk cap. The read endpoints expose each model's walk-forward "
        "**out-of-sample** numbers, not just its existence, because a model with a large "
        "in-sample/out-of-sample gap must not be switched on.",
    },
    {
        "name": "strategies",
        "description": "Strategy version history and activation, per account "
        "(`/accounts/{account_id}/strategies/...` — each account has its own registry, so a "
        "refinement promoted on one never changes what another trades). Activation registers a "
        "version live in that account's `StrategyRegistry`; it never changes "
        "`configs/app.yaml`'s paper/live mode or any risk cap. `POST /strategies/"
        "evaluate-custom` is a process-wide sandbox scratchpad and stays unprefixed.",
    },
    {
        "name": "indicators",
        "description": "Custom Python indicators computed server-side in a sandbox "
        "(math/statistics/numpy/pandas only, no I/O or network) — independent of the chart's "
        "built-in client-side indicators (EMA/SMA/RSI/...). Create/edit/duplicate/delete here "
        "or from the chart's indicator picker; edits apply in place, since indicators never "
        "trade and carry no live-trading risk to roll back from.",
    },
    {
        "name": "skills",
        "description": "Symbol -> strategy routing (§6.6): which strategy family trades each "
        "symbol live, read by TradeEngine._try_enter via SkillSelector. Reassigning here "
        "rewrites skills/normal/<symbol>.yaml and takes effect immediately (no restart); for a "
        "symbol not yet in the automated-trading universe it also persists the symbol into "
        "configs/app.yaml and hot-activates candle streaming/the spread gate for it — this is "
        "the one deliberate action that turns on live trading for a new symbol. It never "
        "activates or changes a StrategyVersion itself — see the `strategies` tag for that.",
    },
    {
        "name": "news",
        "description": "Economic calendar and active news-window status (F8) — read-only. "
        "The engine reacts to news windows internally: `NewsSkillSelector` blocks/overrides "
        "entries and the trade engine flattens positions on `NewsWindowEntered` when the "
        "matched news skill's `pre_event.close_all` requests it "
        "(`backend/src/skills/news/*.yaml`); this API only reports that state for the UI.",
    },
    {
        "name": "order-book",
        "description": "Per-signal market-depth snapshots captured at the moment a trading "
        "signal fires, for AI-training data export. Gracefully degrades to storing nothing "
        "when the broker/symbol reports no depth (common for OTC CFD/synthetic symbols) — "
        "absence of a row is expected, not an error.",
    },
    {
        "name": "observability",
        "description": "`GET /metrics` — Prometheus exposition of engine loop duration, gateway "
        "RTT, signals/min, veto counts by reason, open positions, and WS client count "
        "(OBSERVABILITY_PLAN.md Phase 5). Process-wide like `/health`, not per-account, and "
        "unauthenticated for the same reason `/health` is: a scraper has no session token.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    log_listener = configure_logging(
        database_url=settings.database_url, log_format=settings.log_format
    )
    container = build_container(settings)
    app.state.container = container
    # One candle-stream/live-candle binding per enabled account (Phase 8 of
    # MULTI_ACCOUNT_PLAN.md) — `ws.py`'s subscribe/unsubscribe handlers key
    # off the client-supplied `account_id` to route watch/unwatch calls to
    # the right account's services.
    for account_id, runtime in container.accounts.items():
        bind_candle_stream(account_id, runtime.candle_stream)
        bind_live_candle(account_id, runtime.live_candle)
    bind_auth(container.session_issuer, lambda: container.settings.app_password)
    # `tradingbot_ws_clients_connected` (OBSERVABILITY_PLAN.md Phase 5):
    # sampled at scrape time from the underlying engine.io transport socket
    # count, read-only — see src.shared.metrics.registry.set_ws_client_source.
    set_ws_client_source(lambda: len(sio.eio.sockets))
    for runtime in container.accounts.values():
        # Reconnect with stored credentials if the gateway is already up,
        # then start the candle streams — they idle harmlessly until login
        # succeeds.
        if await runtime.account.reconnect_from_stored():
            # Catch any broker-side close (SL/TP fill) that happened while
            # the backend was down — see broker/application/reconciliation.py.
            try:
                await runtime.reconciliation.reconcile_all()
            except BrokerUnavailable as exc:
                logger.warning(
                    "reconciliation failed at startup for account=%s (broker unavailable): %s",
                    runtime.id,
                    exc,
                )
        else:
            logger.info(
                "account=%s: reconnect from stored credentials skipped or failed; "
                "skipping startup reconciliation",
                runtime.id,
            )
        # Fill any hole left by a downtime longer than `poll_once` re-fetches
        # on its own (OPTIMIZATION_CHECKLIST.md §1) before streaming resumes
        # and starts trusting the DB as caught-up.
        reconciled = await runtime.candle_history.reconcile_gaps(
            runtime.symbols, list(Timeframe), poll_lookback_for
        )
        if reconciled:
            logger.info(
                "account=%s: startup gap reconciliation backfilled: %s", runtime.id, reconciled
            )
        # Seeds `tradingbot_open_positions` from the ground truth right after
        # startup reconciliation above, so a backend restart doesn't report
        # zero open positions until the next fill/close event — best-effort,
        # since the gateway may still be down for this account.
        try:
            open_positions = await runtime.order_service.get_positions()
            set_open_positions(account_id=runtime.id, count=len(open_positions))
        except Exception:
            logger.warning("account=%s: could not seed open-positions gauge at startup", runtime.id)
        runtime.candle_stream.start()
        runtime.live_candle.start()
        runtime.health_monitor.start()
        runtime.silence_monitor.start()
        runtime.reconciliation_poller.start()
    container.news_window_service.start()
    container.activity_log_retention_service.start()
    container.wal_checkpoint_service.start()
    # One shared training service for the API button and the bi-weekly
    # scheduler — `routes_training` reads it off app.state.
    app.state.model_training = container.model_training
    container.training_scheduler.start()
    yield
    await container.training_scheduler.stop()
    await container.model_training.stop()
    await container.aclose()
    if log_listener is not None:
        log_listener.stop()


app = FastAPI(
    title="AI Trading Bot",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)
_SESSION_REQUIRED = [Depends(require_session)]

app.include_router(auth_router)
app.include_router(accounts_router, dependencies=_SESSION_REQUIRED)
app.include_router(account_router, dependencies=_SESSION_REQUIRED)
app.include_router(market_data_router, dependencies=_SESSION_REQUIRED)
app.include_router(trading_router, dependencies=_SESSION_REQUIRED)
app.include_router(broker_spread_router, dependencies=_SESSION_REQUIRED)
app.include_router(journal_router, dependencies=_SESSION_REQUIRED)
app.include_router(activity_router, dependencies=_SESSION_REQUIRED)
app.include_router(engine_router, dependencies=_SESSION_REQUIRED)
app.include_router(backtest_router, dependencies=_SESSION_REQUIRED)
app.include_router(ai_router, dependencies=_SESSION_REQUIRED)
app.include_router(ai_refinement_router, dependencies=_SESSION_REQUIRED)
app.include_router(ai_regeneration_router, dependencies=_SESSION_REQUIRED)
app.include_router(ai_settings_router, dependencies=_SESSION_REQUIRED)
app.include_router(strategies_router, dependencies=_SESSION_REQUIRED)
app.include_router(strategies_sandbox_router, dependencies=_SESSION_REQUIRED)
app.include_router(model_training_router, dependencies=_SESSION_REQUIRED)
app.include_router(indicators_router, dependencies=_SESSION_REQUIRED)
app.include_router(skills_router, dependencies=_SESSION_REQUIRED)
app.include_router(news_router, dependencies=_SESSION_REQUIRED)
app.include_router(order_book_router, dependencies=_SESSION_REQUIRED)


class HealthOut(BaseModel):
    status: str = Field(description="Always 'ok' when the process is serving requests.")


class EngineConfigOut(BaseModel):
    enabled: bool = Field(description="Whether the automated trade loop runs at all.")
    entry_timeframe: str = Field(description="Timeframe the engine looks for entries on.")


class AppConfigOut(BaseModel):
    """Runtime mode/symbols from `configs/app.yaml` — the UI shows this
    prominently so it's always obvious whether the bot can place real trades."""

    mode: str = Field(description="'paper' (simulated fills) or 'live' (real broker orders).")
    timezone: str = Field(description="IANA timezone used for session windows and daily resets.")
    symbols: list[str] = Field(description="Symbols the engine and skills are configured for.")
    engine: EngineConfigOut


@app.get(
    "/health",
    response_model=HealthOut,
    tags=["meta"],
    summary="Liveness check",
    description="Unauthenticated liveness probe — returns 200 as soon as the ASGI app is serving, "
    "independent of gateway/DB connectivity. Use `/account/status` to check gateway health.",
)
async def health() -> HealthOut:
    return HealthOut(status="ok")


@app.get(
    "/metrics",
    tags=["observability"],
    summary="Prometheus metrics",
    description=(
        "Prometheus text-exposition-format scrape endpoint (OBSERVABILITY_PLAN.md Phase 5): "
        "`tradingbot_engine_loop_duration_seconds`, `tradingbot_gateway_request_duration_seconds`, "
        "`tradingbot_signals_total` (rate this for signals/min), "
        "`tradingbot_signal_outcomes_total` (labeled `outcome`, the same closed vocabulary as `GET "
        "/accounts/{account_id}/activity/signals/funnel`), `tradingbot_open_positions`, and "
        "`tradingbot_ws_clients_connected`. Unauthenticated, like `/health` — a Prometheus "
        "scraper has no session token to send, and this endpoint carries no secrets or "
        "trading data, only counts/durations."
    ),
    # `response_model=None`: Prometheus's text exposition format
    # (`prometheus_client.CONTENT_TYPE_LATEST`, currently `text/plain;
    # version=1.0.0`) is a line-oriented metrics format, not JSON — a
    # Pydantic model can't describe it, so this returns a plain `Response`
    # with the exposition media type and documents the shape in prose above
    # instead. `response_class=Response` matters here too, separately from
    # `response_model`: FastAPI's OpenAPI generator derives its *default*
    # 200 content-type from the route's `response_class` (defaulting to
    # `JSONResponse` if left unset), regardless of the function's return
    # annotation — leaving it unset would silently merge a spurious
    # empty-schema `application/json` entry alongside the real one below.
    response_model=None,
    response_class=Response,
    responses={
        200: {
            "description": "Prometheus text exposition format.",
            "content": {CONTENT_TYPE_LATEST: {"schema": {"type": "string"}}},
        }
    },
)
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/config/app",
    response_model=AppConfigOut,
    tags=["meta"],
    summary="Get runtime app configuration",
    description="Current runtime mode/symbols/engine config from `configs/app.yaml` — the UI "
    "shows this prominently so paper vs. live mode is never ambiguous.",
    dependencies=_SESSION_REQUIRED,
    responses={401: {"description": "Missing or invalid session (see the `auth` tag)."}},
)
async def app_config() -> AppConfigOut:
    return AppConfigOut(**load_yaml_config("app"))


socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
