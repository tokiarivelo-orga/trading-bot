# AGENTS.md

## Commands (Makefile is canonical — `make help`)
```bash
make setup                          # uv sync backend+gateway, pnpm install frontend, .env with random secret
make dev                            # backend :8000 + frontend :3000 + default-account gateway (Wine)
make dev-backend                    # uv run uvicorn src.main:socket_app --reload --port 8000
make dev-frontend                   # pnpm dev --port 3000 (reads TB_FRONTEND_PORT from .env)
make dev-gateway ACCOUNT=ftmo-1     # one account's gateway under Wine (resolves port/host from configs/accounts.yaml)
make dev-gateway-all                # every enabled account's gateway
make check                          # GATE before done: lint + test + build-frontend
make lint / make test / make build-frontend  # individual gates
make doctor [ACCOUNT=...]           # gateway :8787/health + backend :8000/account/status
make openapi                        # curl localhost:8000/openapi.json (backend must be running)
make backtest strategy=X symbol=XAUUSD period=2025-01:2025-06
make train-dl                       # SMC deep-learning training loop
make db-upgrade / db-downgrade / db-revision m="msg"
```

## Monorepo boundaries
- `backend/` — FastAPI + Socket.IO (`src/main.py:socket_app`), Python 3.12, `uv` only (`uv sync`/`uv run`). `uv run ruff check src tests` + `uv run pytest`. `pythonpath=[.]` ; alembic at `backend/migrations`.
- `frontend/` — Next.js App Router + Tailwind + TypeScript, `pnpm@11.11.0` pinned (`packageManager` in package.json). Never npm/yarn. `pnpm lint` is `oxlint`, `pnpm build` includes type-check.
- `gateway/` — only code allowed to `import MetaTrader5` is `gateway/src/gateway/mt5_client.py`. Runs on Windows/Wine only; backend talks to it via HTTP (`TB_GATEWAY_URL`). `uv` managed, `pythonpath=["src"]`. Gateway never containerized.

## Architecture (hexagonal, enforced)
- Per-module layout: `domain/` + `application/` (pure business logic) | `ports/` (Protocols) | `adapters/` (implementations) | `api/` (FastAPI routes). Domain never imports adapters, FastAPI, SQLAlchemy, or pydantic.
- Cross-module communication only via event bus (`backend/src/shared/events/bus.py`) or explicit calls wired in `backend/src/container.py` (composition root). Never import another module's internals.
- Multi-account: `configs/accounts.yaml` — one `AccountRuntime` per enabled account (own gateway, bus, engine, journal, registry). Routes under `/accounts/{account_id}/...`; process-wide routes (`/health`, `/metrics`, `/config/app`, `/indicators`, `/skills`, `/news`, backtest) are unprefixed. `container.py` exposes primary-account delegates for backward compat.

## Backend API conventions (binding — see CLAUDE.md)
- Every route needs `response_model` from `that-module/api/schemas.py` (Pydantic `BaseModel`, never `dict`), `summary` + `description` (include event-bus side effects), `responses=` for non-2xx, `Field(description=...)` on every field, `Query(description=...)` on every param.
- New tag → add to `OPENAPI_TAGS` in `backend/src/main.py` first.
- Verify typed schema: `make openapi` or `/docs` — no `Body_xxx` placeholders or untyped `object` responses.

## Frontend conventions
- `next.config.ts` proxies ` /api/:path* → BACKEND_URL/:path*`; WS bypasses proxy and connects directly via `src/shared/api/ws.ts` (rooms `symbol:timeframe`).
- Data fetching via TanStack Query (`shared/api/queryKeys.ts`), not hand-rolled polling. Charting via `lightweight-charts` only.
- Indicator panes: `features/chart/paneTargets.ts` + `useIndicators.ts` (`paneKeyOf()`/`getPaneForManualIndicator()`/`getOscillatorPane()`) is the only pane-target extension point. `ManualIndicator.paneTarget` decides main vs shared vs new split pane — never hardcode `scaleMargins` or `chart.addPane()` unconditionally.
- Tokens in `src/app/globals.css` `@theme`, no raw hex in components. `output: "standalone"` for `frontend/Dockerfile`.

## Safety (non-negotiable)
- Strategies `backend/src/strategies/generated/` — sandbox imports only (`math`, `statistics`, `numpy`, `pandas`), no I/O/network/broker/dunder. Same sandbox for `indicators/`.
- `configs/risk.yaml` is user-owned; generated code/AI/skills must never modify it or bypass limits. Engine circuit breakers in `backend/src/engine/` are off-limits to AI refinements.
- MT5 credentials never in `installer/`, `.env` committed, or Docker images — only via UI → OS keyring → gateway at connect time.

## Verification
- `make check` = `lint` (ruff `E,F,I,UP,B,SIM` line-length 100 + oxlint) + `test` (pytest `asyncio_mode=auto`) + `build-frontend`. CI (`ci.yml`) runs same plus `ruff format --check` and `docker-build` / `installer-windows` (Task Scheduler) jobs.
- Focused: `make test-backend` / `make test-gateway`; `cd backend && uv run pytest tests/unit` or `tests/integration` or `tests/path/test_file.py -k name`.
- Secrets: `.env` is gitignored, generated from `.env.example` (`TB_GATEWAY_SHARED_SECRET` randomized). Per-account secrets like `TB_GATEWAY_SHARED_SECRET_<ACCOUNT_ID>` — one per gateway process.

## Gotchas
- Adding a new `.env` var, `configs/accounts.yaml` field, or service dependency → update `installer/wizard.py` + `installer/services/` (linux + windows) + `docker-compose.prod.yml`/Dockerfiles in the same change. `.env.example` alone is insufficient.
- `make dev-gateway` resolves host/port/secret per account via `scripts/print_account_gateway_env.py`; `WINEPREFIX` defaults to `TB_WINEPREFIX` from `.env` or `~/.mt5`. Gateway writes `gateway/run/<account_id>.pid`; `make kill` kills by port (backend/frontend) and PID file (gateway).
- Config YAML in `configs/` loaded via `shared/config`; gateway terminal path per account (`terminal_path`/`terminal_subpath` in accounts.yaml).

## Detailed rules
- `CLAUDE.md` — full architecture, OpenAPI, strategy safety, frontend, installer rules (binding).
- `IMPLEMENTATION_PLAN.md` — design, module specs, trading logic, AI layer.
- `gateway/README.md` — Wine/VPS setup, terminal configuration.
- `AI_PROVIDERS_CONFIGURATION.md` — per-task provider setup.
