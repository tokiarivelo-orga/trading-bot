# ══════════════════════════════════════════════════════════════════════════
#  AI Trading Bot — developer Makefile
#
#  Run `make` or `make help` to list every target.
#
#  Tooling (see CLAUDE.md — binding):
#    backend/gateway : Python 3.12 managed with uv   (https://docs.astral.sh/uv)
#    frontend        : Next.js managed with pnpm     (pinned in package.json)
#
#  The MT5 gateway needs a Windows environment (Wine or a Windows VPS) with a
#  running MT5 terminal.  `make dev` starts it alongside backend + frontend
#  under Wine.  See gateway/README.md for full setup & VPS deployment.
# ══════════════════════════════════════════════════════════════════════════

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Locations & tools
BACKEND_DIR  := backend
FRONTEND_DIR := frontend
GATEWAY_DIR  := gateway
UV           := uv
PNPM         := pnpm

# Ports (override like: make dev-backend BACKEND_PORT=8001)
# Gateway ports are no longer a single override here — each account in
# configs/accounts.yaml carries its own gateway_url/port, resolved per
# invocation by scripts/print_account_gateway_env.py (see dev-gateway below).
BACKEND_PORT  ?= 8000

# Frontend port defaults to TB_FRONTEND_PORT from .env (falls back to 3000
# if unset/no .env yet) — edit .env once instead of retyping the override
# every time 3000 is already taken. Still overridable per-invocation:
# make dev-frontend FRONTEND_PORT=3001
FRONTEND_PORT := $(shell [ -f .env ] && grep -E '^TB_FRONTEND_PORT=' .env | cut -d= -f2-)
FRONTEND_PORT := $(if $(FRONTEND_PORT),$(FRONTEND_PORT),3000)

# Wine prefix dedicated to the MT5 terminal + Windows Python (see gateway/README.md)
WINEPREFIX ?= $(HOME)/.mt5

# Silences Wine's "fixme:" stub-implementation noise (harddisk_ioctl,
# cryptasn CryptDecodeObjectEx, etc.) that floods the console once several
# bots start driving the MT5 terminal — cosmetic, not an actual error; err/
# warn channels stay on. Override per-invocation for full Wine debug output:
# make dev-gateway WINEDEBUG=
WINEDEBUG ?= fixme-all


# ════════════════════════════════ HELP ══════════════════════════════════════

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "AI Trading Bot — make targets"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^# ───/ {gsub(/^# ─+ ?| ?─+$$/, ""); printf "\n\033[1m%s\033[0m\n", $$0} \
		/^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
	@echo ""

# ─── Setup ───────────────────────────────────────────────────────────────────

.PHONY: setup
setup: setup-backend setup-frontend setup-gateway env ## Full dev setup: backend + frontend + gateway deps + .env

.PHONY: setup-backend
setup-backend: ## Install backend deps into backend/.venv (uv sync)
	cd $(BACKEND_DIR) && $(UV) sync

.PHONY: setup-frontend
setup-frontend: ## Install frontend deps (pnpm install)
	cd $(FRONTEND_DIR) && $(PNPM) install

.PHONY: setup-gateway
setup-gateway: ## Install gateway deps (real runtime is Windows/Wine — see gateway/README.md)
	cd $(GATEWAY_DIR) && $(UV) sync

.PHONY: env
env: ## Create .env from .env.example if it doesn't exist yet (auto-generates the gateway secret)
	@if [ -f .env ]; then \
		echo ".env already exists — leaving it untouched"; \
	else \
		cp .env.example .env; \
		SECRET=$$(openssl rand -hex 32); \
		sed -i.bak "s#^TB_GATEWAY_SHARED_SECRET=.*#TB_GATEWAY_SHARED_SECRET=$$SECRET#" .env && rm -f .env.bak; \
		echo "created .env from .env.example with a random TB_GATEWAY_SHARED_SECRET"; \
		echo "MT5 login/password/server are NOT put in .env — enter them via the UI's"; \
		echo "MT5 Account panel, or 'make mt5-login LOGIN=... PASSWORD=... SERVER=...'"; \
	fi

.PHONY: setup-wine
setup-wine: ## One-time host setup for the Wine-hosted MT5 gateway (Ubuntu/Debian; Linux dev only)
	@echo "Installing Wine + winetricks (sudo required)..."
	sudo dpkg --add-architecture i386
	sudo apt update && sudo apt install --install-recommends -y wine64 wine32 winetricks
	@echo "Creating Wine prefix at $(WINEPREFIX)..."
	WINEPREFIX=$(WINEPREFIX) winetricks -q corefonts
	@echo ""
	@echo "Host setup done. Remaining manual steps (see gateway/README.md):"
	@echo "  1. WINEPREFIX=$(WINEPREFIX) wine mt5setup.exe                 # install the MT5 terminal"
	@echo "  2. WINEPREFIX=$(WINEPREFIX) wine python-3.12.x-amd64.exe /quiet InstallAllUsers=0 PrependPath=1"
	@echo "  3. WINEPREFIX=$(WINEPREFIX) wine python -m pip install MetaTrader5 fastapi uvicorn pydantic"
	@echo "  4. Start the terminal, log in to your MT5 demo account, enable Algo Trading,"
	@echo "     and add XAUUSD/XAGUSD/BTCUSD to Market Watch — see 'Terminal configuration'"
	@echo "     in gateway/README.md."

# ─── Dev servers ─────────────────────────────────────────────────────────────

.PHONY: dev
dev: ## Run backend + frontend + the default account's gateway together (Ctrl-C stops all)
	@trap 'kill 0' EXIT; \
	( cd $(BACKEND_DIR) && $(UV) run uvicorn src.main:socket_app --reload --port $(BACKEND_PORT) ) & \
	( cd $(FRONTEND_DIR) && $(PNPM) dev --port $(FRONTEND_PORT) ) & \
	$(MAKE) dev-gateway & \
	wait

.PHONY: dev-backend
dev-backend: ## Run the FastAPI backend with auto-reload (default http://localhost:8000)
	cd $(BACKEND_DIR) && $(UV) run uvicorn src.main:socket_app --reload --port $(BACKEND_PORT)

.PHONY: dev-frontend
dev-frontend: ## Run the Next.js dev server (default http://localhost:3000)
	cd $(FRONTEND_DIR) && $(PNPM) dev --port $(FRONTEND_PORT)

# Wine Python used by the gateway (override: make dev-gateway WINE_PYTHON=/path/to/python.exe)
# The Windows Python installer run with InstallAllUsers=0 (see gateway/README.md)
# lands under the per-user AppData path, not C:\Python312.
WINE_PYTHON ?= $(WINEPREFIX)/drive_c/users/$(shell whoami)/AppData/Local/Programs/Python/Python312/python.exe

.PHONY: dev-gateway
dev-gateway: ## Run one account's MT5 gateway under Wine: make dev-gateway ACCOUNT=ftmo-1 (defaults to the first enabled account in configs/accounts.yaml); auto-starts that account's MT5 terminal first
	@mkdir -p $(GATEWAY_DIR)/run
	@eval "$$(cd $(BACKEND_DIR) && $(UV) run python -m scripts.print_account_gateway_env $(ACCOUNT))" && \
	SECRET=$$(grep -E "^$${TB_RESOLVED_GATEWAY_SECRET_ENV}=" .env 2>/dev/null | cut -d= -f2-) && \
	if [ -n "$$TB_RESOLVED_TERMINAL_SUBPATH" ]; then \
		TERM_PATH="$(WINEPREFIX)/drive_c/$$TB_RESOLVED_TERMINAL_SUBPATH"; \
		PGREP_ARGS="-f $$TERM_PATH"; \
	else \
		TERM_PATH="$(WINEPREFIX)/drive_c/Program Files/MetaTrader 5/terminal64.exe"; \
		PGREP_ARGS="-x terminal64"; \
	fi; \
	if ! pgrep $$PGREP_ARGS > /dev/null; then \
		echo "MT5 terminal not running for account '$$TB_RESOLVED_ACCOUNT_ID' — launching it now ($$TERM_PATH, give it ~10 s to connect)..."; \
		WINEPREFIX=$(WINEPREFIX) WINEDEBUG=$(WINEDEBUG) wine "$$TERM_PATH" & \
		sleep 10; \
	fi; \
	echo "starting gateway for account '$$TB_RESOLVED_ACCOUNT_ID' -> http://$$TB_RESOLVED_GATEWAY_HOST:$$TB_RESOLVED_GATEWAY_PORT (pid file: $(GATEWAY_DIR)/run/$$TB_RESOLVED_ACCOUNT_ID.pid)"; \
	( cd $(GATEWAY_DIR) && WINEPREFIX=$(WINEPREFIX) WINEDEBUG=$(WINEDEBUG) GATEWAY_SHARED_SECRET=$$SECRET GATEWAY_HOST=$$TB_RESOLVED_GATEWAY_HOST GATEWAY_PORT=$$TB_RESOLVED_GATEWAY_PORT MT5_TERMINAL_SUBPATH=$$TB_RESOLVED_TERMINAL_SUBPATH wine $(WINE_PYTHON) run_gateway.py & \
	  echo $$! > run/$$TB_RESOLVED_ACCOUNT_ID.pid; \
	  wait $$! )

.PHONY: dev-gateway-all
dev-gateway-all: ## Run every enabled account's gateway concurrently (Ctrl-C stops all); see configs/accounts.yaml
	@mkdir -p $(GATEWAY_DIR)/run
	@ids=$$(cd $(BACKEND_DIR) && $(UV) run python -m scripts.print_account_gateway_env --list-ids) && \
	if [ -z "$$ids" ]; then echo "no enabled accounts in configs/accounts.yaml"; exit 1; fi; \
	trap 'kill 0' EXIT; \
	for id in $$ids; do \
		$(MAKE) dev-gateway ACCOUNT=$$id & \
	done; \
	wait

.PHONY: mt5-terminal
mt5-terminal: ## Launch an account's MT5 terminal in the dedicated Wine prefix (leave it running): make mt5-terminal ACCOUNT=ftmo-1 (defaults to the first enabled account's terminal)
	@eval "$$(cd $(BACKEND_DIR) && $(UV) run python -m scripts.print_account_gateway_env $(ACCOUNT))" && \
	if [ -n "$$TB_RESOLVED_TERMINAL_SUBPATH" ]; then \
		TERM_PATH="$(WINEPREFIX)/drive_c/$$TB_RESOLVED_TERMINAL_SUBPATH"; \
	else \
		TERM_PATH="$(WINEPREFIX)/drive_c/Program Files/MetaTrader 5/terminal64.exe"; \
	fi; \
	WINEPREFIX=$(WINEPREFIX) WINEDEBUG=$(WINEDEBUG) wine "$$TERM_PATH" &

.PHONY: mt5-login
mt5-login: ## Log in to an account's gateway from the CLI: make mt5-login ACCOUNT=ftmo-1 LOGIN=123 PASSWORD=x SERVER=Broker-Demo (ACCOUNT defaults to the first enabled account)
	@test -n "$(LOGIN)" && test -n "$(PASSWORD)" && test -n "$(SERVER)" || \
		{ echo 'usage: make mt5-login ACCOUNT=ftmo-1 LOGIN=12345678 PASSWORD=*** SERVER=MetaQuotes-Demo'; exit 1; }
	@eval "$$(cd $(BACKEND_DIR) && $(UV) run python -m scripts.print_account_gateway_env $(ACCOUNT))" && \
	SECRET=$$(grep -E "^$${TB_RESOLVED_GATEWAY_SECRET_ENV}=" .env 2>/dev/null | cut -d= -f2-) && \
	curl -sf -X POST http://$$TB_RESOLVED_GATEWAY_HOST:$$TB_RESOLVED_GATEWAY_PORT/login \
		-H "X-Gateway-Secret: $$SECRET" \
		-H "Content-Type: application/json" \
		-d '{"login": $(LOGIN), "password": "$(PASSWORD)", "server": "$(SERVER)"}' \
		| python3 -m json.tool

.PHONY: backtest
backtest: ## Run a strategy backtest: make backtest strategy=breakout_v1 symbol=XAUUSD period=2025-01:2025-06
	@test -n "$(strategy)" && test -n "$(symbol)" && test -n "$(period)" || \
		{ echo 'usage: make backtest strategy=breakout_v1 symbol=XAUUSD period=2025-01:2025-06'; exit 1; }
	cd backend && uv run python -m src.backtest.cli "$(strategy)" "$(symbol)" "$(period)"

.PHONY: seed-indicators
seed-indicators: ## Seed the 15 PoB pattern/confirmation indicators into the indicator DB (safe to re-run)
	cd backend && uv run python -m scripts.seed_pob_indicators

.PHONY: seed-strategies
seed-strategies: ## Seed the baseline strategies (breakout_v1, trend_structure_v1/v2) into the StrategyVersion DB and backfill missing spec snapshots (safe to re-run)
	cd backend && uv run python -m scripts.seed_baseline_strategies

# ─── Quality gates (run `make check` before declaring any task done) ─────────

.PHONY: check
check: lint test build-frontend ## EVERYTHING: lint + tests + frontend production build

.PHONY: lint
lint: lint-backend lint-gateway lint-frontend ## Lint backend + gateway (ruff) + frontend (oxlint)

.PHONY: lint-backend
lint-backend: ## ruff check backend src + tests
	cd $(BACKEND_DIR) && $(UV) run ruff check src tests

.PHONY: lint-gateway
lint-gateway: ## ruff check gateway src + tests
	cd $(GATEWAY_DIR) && $(UV) run ruff check src tests

.PHONY: lint-frontend
lint-frontend: ## oxlint the frontend
	cd $(FRONTEND_DIR) && $(PNPM) lint

.PHONY: format
format: format-backend format-gateway ## Auto-format backend + gateway (ruff format + safe fixes)

.PHONY: format-backend
format-backend: ## Auto-format backend (ruff format + safe fixes)
	cd $(BACKEND_DIR) && $(UV) run ruff format src tests && $(UV) run ruff check --fix src tests

.PHONY: format-gateway
format-gateway: ## Auto-format gateway (ruff format + safe fixes)
	cd $(GATEWAY_DIR) && $(UV) run ruff format src tests && $(UV) run ruff check --fix src tests

.PHONY: test
test: test-backend test-gateway ## All tests (backend + gateway; frontend has no test suite yet)

.PHONY: test-backend
test-backend: ## Full backend pytest suite
	cd $(BACKEND_DIR) && $(UV) run pytest

.PHONY: test-gateway
test-gateway: ## Gateway contract tests (stubbed MetaTrader5 — runs on Linux)
	cd $(GATEWAY_DIR) && $(UV) run pytest

.PHONY: test-unit
test-unit: ## Backend unit tests only
	cd $(BACKEND_DIR) && $(UV) run pytest tests/unit

.PHONY: test-integration
test-integration: ## Backend integration tests only (paper-mode broker paths)
	cd $(BACKEND_DIR) && $(UV) run pytest tests/integration

.PHONY: build-frontend
build-frontend: ## Next.js production build (includes TypeScript type-checking)
	cd $(FRONTEND_DIR) && $(PNPM) build

# ─── Database (SQLite via alembic) ───────────────────────────────────────────

.PHONY: db-upgrade
db-upgrade: ## Apply all pending migrations (alembic upgrade head)
	mkdir -p $(BACKEND_DIR)/data
	cd $(BACKEND_DIR) && $(UV) run alembic upgrade head

.PHONY: db-downgrade
db-downgrade: ## Roll back the last migration (alembic downgrade -1)
	cd $(BACKEND_DIR) && $(UV) run alembic downgrade -1

.PHONY: db-revision
db-revision: ## New autogenerated migration: make db-revision m="add journal table"
	@test -n "$(m)" || { echo 'usage: make db-revision m="describe the change"'; exit 1; }
	cd $(BACKEND_DIR) && $(UV) run alembic revision --autogenerate -m "$(m)"

.PHONY: db-history
db-history: ## Show migration history and current revision
	cd $(BACKEND_DIR) && $(UV) run alembic history && $(UV) run alembic current

# ─── Docker (Linux services only — backend + frontend; gateway excluded) ─────

.PHONY: docker-up
docker-up: ## Start the dev stack in the background (docker compose up -d)
	docker compose up -d --build

.PHONY: docker-down
docker-down: ## Stop the dev stack
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail logs from all services
	docker compose logs -f --tail=100

.PHONY: docker-ps
docker-ps: ## Show service status
	docker compose ps

# ─── Dependency maintenance (rule: always latest stable — see CLAUDE.md) ─────

.PHONY: outdated
outdated: ## Show outdated deps in backend (uv), gateway (uv) and frontend (pnpm)
	cd $(BACKEND_DIR) && $(UV) tree --outdated --depth 1 || true
	cd $(GATEWAY_DIR) && $(UV) tree --outdated --depth 1 || true
	cd $(FRONTEND_DIR) && $(PNPM) outdated || true

.PHONY: update-deps
update-deps: ## Upgrade everything to latest stable (then run `make check`!)
	cd $(BACKEND_DIR) && $(UV) lock --upgrade && $(UV) sync
	cd $(GATEWAY_DIR) && $(UV) lock --upgrade && $(UV) sync
	cd $(FRONTEND_DIR) && $(PNPM) update --latest
	@echo "──> now run 'make check' and review lockfile diffs before committing"

# ─── Utilities ────────────────────────────────────────────────────────────────

.PHONY: openapi
openapi: ## Dump the backend OpenAPI schema (backend must be running)
	curl -sf http://localhost:$(BACKEND_PORT)/openapi.json | python3 -m json.tool

.PHONY: doctor
doctor: ## Diagnose the full stack: gateway up? terminal connected? backend can reach it? (make doctor ACCOUNT=ftmo-1 to check a non-default account's gateway)
	@eval "$$(cd $(BACKEND_DIR) && $(UV) run python -m scripts.print_account_gateway_env $(ACCOUNT))"; \
	echo "── gateway '$$TB_RESOLVED_ACCOUNT_ID' :$$TB_RESOLVED_GATEWAY_PORT/health ──"; \
	if curl -sf http://$$TB_RESOLVED_GATEWAY_HOST:$$TB_RESOLVED_GATEWAY_PORT/health; then echo; else \
		echo; echo "gateway is DOWN — run 'make dev-gateway ACCOUNT=$$TB_RESOLVED_ACCOUNT_ID' (Wine + terminal must be up first)"; exit 0; \
	fi; \
	echo; echo "── backend :$(BACKEND_PORT)/account/status ──"; \
	if curl -sf http://127.0.0.1:$(BACKEND_PORT)/account/status; then echo; else \
		echo; echo "backend is DOWN — run 'make dev-backend'"; exit 0; \
	fi; \
	echo; echo "If terminal_connected is false: MT5 isn't logged in yet — that's why"; \
	echo "/symbol_info etc. return 502/503. Fix with the UI's MT5 Account panel, or:"; \
	echo "  make mt5-login ACCOUNT=$$TB_RESOLVED_ACCOUNT_ID LOGIN=12345678 PASSWORD=*** SERVER=MetaQuotes-Demo"

.PHONY: kill stop
kill stop: ## Kill dev processes (backend, frontend, gateway(s), MT5 terminal); make kill ACCOUNT=ftmo-1 to stop just that one account's gateway
	@echo "Stopping dev processes…"
	@# Kill whatever is actually bound to each port first — this is the part
	@# that matters. `pnpm dev` execs `next dev`, which execs a `next-server`
	@# child; none of those descendants have "pnpm" in their argv, so the old
	@# `pkill -f "pnpm.*dev"` only ever killed the top pnpm wrapper and left
	@# next-server running (and still serving/proxying to a dead backend).
	@# Similarly uvicorn's --reload can leave its actual worker in a state
	@# where the parent is alive but nothing is listening, or vice versa —
	@# killing by port sidesteps all of that regardless of process shape.
	@# Gateway processes aren't killed by port any more: with N accounts each
	@# on its own port, that would mean re-deriving every account's port here.
	@# `make dev-gateway` writes gateway/run/<account_id>.pid on start instead,
	@# so gateways are looked up and killed by PID file below — precise per
	@# account, and it doesn't matter if the port moved or the process wedged
	@# without ever binding it.
	@if [ -n "$(ACCOUNT)" ]; then \
		pidfile=$(GATEWAY_DIR)/run/$(ACCOUNT).pid; \
		if [ -f "$$pidfile" ] && kill -0 $$(cat "$$pidfile") 2>/dev/null; then \
			pid=$$(cat "$$pidfile"); \
			kill -TERM $$pid 2>/dev/null; sleep 1; kill -KILL $$pid 2>/dev/null; \
			rm -f "$$pidfile"; \
			echo "  ✓ gateway '$(ACCOUNT)' (pid $$pid) stopped"; \
		else \
			echo "  – no running gateway tracked for account '$(ACCOUNT)' ($$pidfile)"; \
		fi; \
		echo "Done."; \
		exit 0; \
	fi; \
	for port in $(BACKEND_PORT) $(FRONTEND_PORT); do \
		pids=$$(fuser $$port/tcp 2>/dev/null); \
		if [ -n "$$pids" ]; then \
			echo "  killing pid(s) $$pids listening on :$$port"; \
			kill -TERM $$pids 2>/dev/null; \
			sleep 1; \
			kill -KILL $$pids 2>/dev/null; \
		fi; \
	done; \
	pkill -f "uvicorn src.main:socket_app"       2>/dev/null && echo "  ✓ backend process(es) stopped"  || echo "  – no leftover backend process"; \
	pkill -f "next-server|next dev|pnpm.*dev"    2>/dev/null && echo "  ✓ frontend process(es) stopped" || echo "  – no leftover frontend process"; \
	if ls $(GATEWAY_DIR)/run/*.pid >/dev/null 2>&1; then \
		for pidfile in $(GATEWAY_DIR)/run/*.pid; do \
			acct=$$(basename "$$pidfile" .pid); \
			pid=$$(cat "$$pidfile" 2>/dev/null); \
			if [ -n "$$pid" ] && kill -0 $$pid 2>/dev/null; then \
				kill -TERM $$pid 2>/dev/null; sleep 1; kill -KILL $$pid 2>/dev/null; \
				echo "  ✓ gateway '$$acct' (pid $$pid) stopped"; \
			fi; \
			rm -f "$$pidfile"; \
		done; \
	else \
		echo "  – no gateway/run/*.pid files (nothing started via make dev-gateway)"; \
	fi; \
	pkill -f "run_gateway.py" 2>/dev/null && echo "  ✓ swept a leftover gateway process not tracked by any PID file" || true; \
	pkill -x terminal64 2>/dev/null && echo "  ✓ MT5 terminal stopped" || echo "  – MT5 terminal not running"; \
	echo "Done."

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps .venv and node_modules)
	rm -rf $(FRONTEND_DIR)/.next $(FRONTEND_DIR)/out
	find $(BACKEND_DIR) $(GATEWAY_DIR) -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## clean + remove .venv and node_modules (full reinstall needed after)
	rm -rf $(BACKEND_DIR)/.venv $(GATEWAY_DIR)/.venv $(FRONTEND_DIR)/node_modules
