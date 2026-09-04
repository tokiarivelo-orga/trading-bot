> 🇫🇷 Version française : [README.fr.md](README.fr.md)

# AI Trading Bot — XAUUSD / XAGUSD / BTCUSD

An MT5-connected, AI-assisted trading bot. Entries on M5 with higher-timeframe
confirmation, TradingView-style chart, strategies generated from PDF documents
by an AI (choice of provider per task — Claude API, Claude Code, Ollama/Hermes
Agent, or OpenClaw), and automatic self-refinement every 10 trades.

**Full design & roadmap:** see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
**AI provider setup (per provider, step by step):** see
[`AI_PROVIDERS_CONFIGURATION.md`](AI_PROVIDERS_CONFIGURATION.md).

## Repository layout

| Path | What |
|------|------|
| `backend/` | FastAPI backend — engine, strategies, AI layer, journal (hexagonal modules) |
| `frontend/` | Next.js + Tailwind CSS + TypeScript UI — chart, bot controls, PDF upload, reports |
| `gateway/` | MT5 Gateway service — the **only** code touching MetaTrader5 (runs on Windows/Wine/VPS, see `gateway/README.md`) |
| `configs/` | Runtime configuration (symbols, risk caps, AI providers, news) |
| `Makefile` | Canonical dev commands — setup, dev servers, checks, DB, docker (`make help`) |
| `.claude/` | Claude Code dev skills and settings |

## Quick start (development)

Everything goes through the root `Makefile` — run `make help` for the full list.

```bash
make setup             # backend (uv sync) + frontend (pnpm install) + gateway (uv sync) + .env
make dev               # backend :8000 + frontend :3000 + gateway :8787 — Ctrl-C stops all

# or individually:
make dev-backend       # FastAPI with auto-reload — http://localhost:8000
make dev-frontend      # Next.js dev server — http://localhost:3000
make dev-gateway       # MT5 gateway under Wine — http://localhost:8787
```

The gateway requires a running MT5 terminal under Wine (development) or on a
Windows VPS (live trading). See [`gateway/README.md`](gateway/README.md) for
full setup instructions.

## Installing on another machine

**Full installation guide:** see [`INSTALL.md`](INSTALL.md) for the
complete, standalone walkthrough (requirements, every installer prompt,
Docker, uninstalling, troubleshooting, FAQ). The summary below is the
short version. **Configuring secrets/env vars:** see [`SECRETS.md`](SECRETS.md)
for every variable this project uses, where to get a real value, and how to
set it on each platform.

Two install paths, depending on where the MT5 gateway runs:

- **CLI installer** — full stack including the gateway, any OS. Sets up a
  Windows/Wine (or Windows VPS) MT5 terminal alongside backend + frontend.
  See `installer/install.py` (built in a separate track).
- **Docker (backend + frontend only)** — for when the MT5 gateway already
  runs elsewhere, e.g. a separate Windows VPS reached over the network via
  `TB_GATEWAY_URL` in `.env`. Pulls prebuilt images from Docker Hub instead
  of building from source:
  ```bash
  cp .env.example .env   # then set TB_GATEWAY_URL to your gateway
  docker compose -f docker-compose.prod.yml up -d
  # or: make docker-up-prod
  ```
  <!-- TODO: images publish under the placeholder namespace "tradingbot" —
  swap for the real Docker Hub username/org once confirmed. --> Published by
  [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml)
  on a version tag push. The gateway itself is deliberately not
  containerized — see `gateway/README.md`.

**Backend API docs** (once `make dev-backend` is running): interactive Swagger
UI at <http://localhost:8000/docs>, ReDoc at <http://localhost:8000/redoc>, raw
schema at <http://localhost:8000/openapi.json> (or `make openapi`). Every
route is fully typed and documented — see `backend/src/*/api/schemas.py`.

Under the hood: backend is Python 3.12 via `uv`, frontend is Next.js via
`pnpm` (version pinned in `frontend/package.json`), gateway runs Windows
Python 3.12 under Wine.

## Checks

```bash
make check             # lint (ruff + oxlint) + backend tests + frontend build
```

Individual gates: `make lint`, `make test`, `make build-frontend` — see `make help`.

## Artificial Intelligence & Deep Learning (SMC)

The project includes an advanced multi-layer neural network that autonomously learns Smart Money Concepts (Order Blocks, FVGs). Complete architectural documentation is available in [`docs/SMC_DEEP_LEARNING.md`](docs/SMC_DEEP_LEARNING.md).

**To trigger the model's training:**
```bash
make train-dl          # Extracts data from MT5, trains the neural net, and runs a validation backtest
```

**To visualize the model's decisions:**
Open the *Model Dashboard* in the Web App (`make dev`, then use the sidebar menu) for an interactive visualization of the neural activations for every trade!

## Safety model (do not weaken)

- Everything starts in **paper mode** (`configs/app.yaml: mode: paper`) —
  orders are simulated in-memory and never reach MT5. Switching to
  `mode: live` sends real orders through the gateway to your real account;
  before doing that, the MT5 terminal's **AutoTrading** button (toolbar, or
  Tools → Options → Expert Advisors → "Allow algorithmic trading") must be
  enabled, or every order is rejected with retcode `10027` — see
  [`gateway/README.md`](gateway/README.md#terminal-configuration-both-options).
- Risk caps live in `configs/risk.yaml` and are user-owned — AI/generated code
  never writes them.
- AI-generated strategies run sandboxed: no I/O, no network, no broker access.
- Engine-level circuit breakers: daily loss limit, consecutive-loss pause, kill switch.
