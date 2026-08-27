# Project rules — AI Trading Bot

`IMPLEMENTATION_PLAN.md` holds the full design, organized by phase — consult
the phase/section relevant to the task at hand rather than reading the whole
671-line file on every task. These rules are binding.

## Architecture
- Hexagonal per module: business logic lives in `domain/` and `application/` ONLY.
  `ports/` hold interfaces (Protocols), `adapters/` hold implementations, `api/`
  holds FastAPI routes. Domain code never imports adapters, FastAPI, or SQLAlchemy.
- Nothing outside `gateway/src/gateway/mt5_client.py` may import `MetaTrader5`.
  The backend talks to MT5 only through the gateway HTTP API via `ports` adapters.
- Modules communicate through the event bus (`backend/src/shared/events/`) or
  explicit application-service calls wired in `backend/src/container.py` — never
  by importing another module's internals.

## API documentation (OpenAPI) — binding for every backend route
- Every FastAPI route needs an explicit `response_model` — a Pydantic
  `BaseModel` in that module's `api/schemas.py`, never a bare `dict`/`list[dict]`.
  Request bodies are Pydantic models too. Schemas mirror the module's
  `domain/` dataclasses; the domain itself stays framework-free (no pydantic
  imports in `domain/`).
- Every `Field` gets a `description`; every path/query `Param` gets a
  `description` via `Query(...)`. This is what renders in `/docs` — a field
  with no description is treated as undocumented and should be fixed.
- Every route needs `summary` (short) and `description` (what it does, when
  to use it, and any non-obvious side effect — e.g. "publishes `PositionOpened`
  on the event bus"). Every non-2xx status a route can raise gets a `responses=`
  entry explaining when it happens.
- Add new tags to `OPENAPI_TAGS` in `backend/src/main.py` with a one-line
  description before using a new tag on a router.
- Verify: run the backend (`make dev-backend`) and check `/docs`, or
  `make openapi` to dump the raw schema — every path should show a summary,
  description, and typed request/response schema, not `Body_xxx` placeholders
  or untyped `object`/`dict` responses.

## Strategies & AI safety (non-negotiable)
- Strategy files in `backend/src/strategies/generated/` implement the `Strategy`
  protocol and must be sandbox-safe: imports limited to `math`, `statistics`,
  `numpy`, `pandas`; no I/O, no network, no broker access, no dunder tricks.
- Risk caps (`configs/risk.yaml`: risk %, daily loss limit, max positions) are
  user-owned. Generated code, AI refinement logic, and dev skills must NEVER
  modify this file or route around its limits.
- Engine circuit breakers (consecutive-loss pause, kill switch) are engine-level
  code; AI refinements must not touch `backend/src/engine/`.

## Quality bar
- Every broker-affecting change requires unit tests plus a paper-mode
  integration test.
- Money-touching code paths: explicit over clever. Log every decision (signal,
  veto reason, spread, lot calculation) at INFO.
- Before declaring any task done, run from `backend/`:
  `uv run ruff check src tests` and `uv run pytest`.

## Frontend
- Stack: **Next.js (App Router) + Tailwind CSS + TypeScript**. No Vite, no CRA,
  no CSS-in-JS libraries. Routes/layout live in `frontend/src/app/`.
- Styling via Tailwind utilities only; design tokens live in the `@theme` block
  of `frontend/src/app/globals.css` — no raw hex in components, no separate
  CSS files.
- Backend REST is proxied under `/api` (rewrites in `next.config.ts`).
  Live streaming uses Socket.IO (`src/shared/api/ws.ts`, rooms per
  `symbol:timeframe`) and connects to the backend directly, because Next
  rewrites don't proxy WS.
- Feature folders under `frontend/src/features/` mirror backend modules.
- Polled/cached data fetching goes through TanStack Query (`@tanstack/react-query`,
  query keys documented in `shared/api/queryKeys.ts`) rather than a hand-rolled
  `useState`/`setInterval` poll — adopted in `features/trading/`, the pattern
  for migrating other feature folders.
- All charting via `lightweight-charts` only.
- API types come from the backend OpenAPI schema — don't hand-write duplicates.
  See "API documentation (OpenAPI)" above: every backend route is fully typed
  and documented, so the schema at `/openapi.json` (`make openapi`) is always
  a complete, accurate source for generated types.
- Package manager is **pnpm** (version pinned via `packageManager` in
  `frontend/package.json`) — never npm or yarn; never commit a
  `package-lock.json` / `yarn.lock`.
- Before declaring any frontend task done, run `make lint-frontend` and
  `make build-frontend` (or from `frontend/`: `pnpm lint` and `pnpm build`).
- Keep components and hooks single-concern: when a component's state grows
  to mix multiple unrelated concerns (e.g. data-fetching + UI toggles + a
  third-party engine's lifecycle), extract each concern into its own flat
  `use*.ts` hook in the feature folder (see `features/trading/useAllPositions.ts`
  for the pattern: no props required, owns its own state/effects internally,
  exports a `ReturnType<typeof useX>` type alias for consumers). Extract
  substantial presentational JSX into its own memoized component with a
  narrow typed `Props` interface rather than growing one file. Shared types
  used by more than one file in a feature live in that feature's `types.ts`,
  not re-exported from a component file.

## Conventions
- Python 3.12, `uv` for dependency management, `ruff` for lint+format.
- Frontend dependency management via `pnpm` only.
- The root `Makefile` is the canonical entry point for every dev command
  (setup, dev servers, lint, tests, build, migrations, docker). Add a target
  there when introducing a new recurring command; `make check` is the
  before-done gate for the whole repo.
- **Always use the latest stable package versions** when adding or updating
  dependencies (`pnpm view <pkg> version` for the frontend; `uv add` resolves
  latest). Never stay on an older major out of habit; pin below latest only
  for a real incompatibility, and record why in the commit (e.g. TypeScript is
  pinned to 6.x until Next.js supports TS 7). `make outdated` shows drift.
- Config is YAML in `configs/`, loaded through `shared/config`; secrets only via
  `.env` / OS keyring — never hardcoded, never logged.
