> 🇫🇷 Version française : [SECRETS.fr.md](SECRETS.fr.md)

# Secrets & Configuration Variables — AI Trading Bot

This is the full reference for every secret and configuration variable this
project uses — what each one is for, whether it's required, where to get a
real value, and how to actually set it on each platform this repo supports
(Linux native, Windows native, Docker, GitHub Actions).

This file assumes you're setting up an install, not necessarily contributing
code. For the step-by-step install flow itself (running the installer,
answering its prompts, Docker vs. native, uninstalling) see
[`INSTALL.md`](INSTALL.md) — this doc doesn't repeat that content, it only
covers secrets/config values in depth.

## The one rule that matters most

**Your MT5 broker login, password, and server name never go in `.env`, in
any file under `configs/`, or into any installer prompt.** They are entered
exactly once, through the running app's own **MT5 Account** UI panel, which
sends them to `POST /accounts/{account_id}/broker/connect`
(`backend/src/broker/api/routes.py`). From there they're encrypted with a
key held in your OS's keyring and forwarded to the gateway only at connect
time — they're never logged, never echoed back in an API response, and
never written to a config file anywhere in this repo. If you ever see a
guide, script, or prompt asking you to put a broker password in a text
file, that's not this project's flow — stop and use the UI panel instead.

Everything else in this document — API keys, shared secrets, alerting
credentials — is a different kind of value and *does* live in `.env` or a
platform's own secret store, as described below.

---

## 1. Core backend config (`.env`, `TB_` prefix)

These are fields of `Settings` in `backend/src/shared/config/settings.py`,
loaded from the repo-root `.env` file (via `pydantic-settings`, prefix
`TB_`). `.env.example` is the template — copy it to `.env` and fill in what
you need; every field has a working default unless noted otherwise.

| Variable | Purpose | Required? | Default | How to obtain a value | Platforms |
|---|---|---|---|---|---|
| `TB_DATABASE_URL` | SQLAlchemy async database URL | Optional | `sqlite+aiosqlite:///./data/trading.db` | Leave as-is unless you've moved to a different database — self-chosen | All |
| `TB_GATEWAY_URL` | Base URL of the primary account's gateway | Optional | `http://127.0.0.1:8787` | Self-set — must match wherever that account's gateway process actually listens | All |
| `TB_GATEWAY_SHARED_SECRET` | Shared secret the backend sends as the `X-Gateway-Secret` header to the primary account's gateway | Required for any non-localhost-only setup | none (empty = gateway skips the check, local dev only) | **Self-generated.** The installer generates one automatically (`secrets.token_hex(32)`); to make your own: `openssl rand -hex 32` (also what `make env` uses) | All |
| `TB_GATEWAY_SHARED_SECRET_<ACCOUNT>` (e.g. `TB_GATEWAY_SHARED_SECRET_DEMO_1`) | Same, one per additional MT5 account beyond the first | Required per additional account | none | Same as above — generate a fresh random value per account, never reuse one | All |
| `TB_FRONTEND_PORT` | Dev/production frontend port | Optional | `3000` | Self-set | Linux (Makefile, systemd unit), Windows (`.cmd` launcher), not read by Docker (port is fixed in `docker-compose.prod.yml`) |
| `TB_WINEPREFIX` | Wine prefix holding the MT5 terminal + Windows Python | Optional | `$(HOME)/.mt5` | Self-set, or accept the installer's prompt | Linux/Wine only |
| `TB_ANTHROPIC_API_KEY` | Anthropic API key, for the `claude` AI provider | Optional (only if `claude` is selected in `configs/ai.yaml` and no Settings-page key is set) | none | [console.anthropic.com](https://console.anthropic.com) | All (backend only) |
| `TB_OPENAI_API_KEY` | OpenAI API key, for the `openai` provider | Optional | none | [platform.openai.com](https://platform.openai.com) | All |
| `TB_GEMINI_API_KEY` | Google Generative Language API key, for the `gemini` provider | Optional | none | Google AI Studio (or a GCP project with the Generative Language API enabled) | All |
| `TB_MISTRAL_API_KEY` | Mistral API key, for the `mistral` provider | Optional | none | Mistral's "La Plateforme" console | All |
| `TB_GROQ_API_KEY` | Groq API key, for the `groq` provider | Optional | none | Groq's console | All |
| `TB_DEEPSEEK_API_KEY` | DeepSeek API key, for the `deepseek` provider | Optional | none | DeepSeek's platform console | All |
| `TB_XAI_API_KEY` | xAI (Grok) API key, for the `xai` provider | Optional | none | xAI's console | All |
| `TB_OLLAMA_URL` | URL of a local Ollama server, for the `ollama`/"Hermes Agent" provider | Optional | `http://127.0.0.1:11434` | Self-set — points at your own Ollama install, no signup needed | All |
| `TB_CLAUDE_CODE_BINARY` | Path/name of the `claude` CLI binary, for the `claude_code` provider | Optional | `claude` | Self-set — relies on your own `claude login` subscription, **not** `TB_ANTHROPIC_API_KEY` | All |
| `TB_CLAUDE_CODE_EXTRA_ARGS` | Extra CLI flags for the `claude_code` provider (e.g. `--agent`) | Optional | empty | Self-set | All |
| `TB_CLAUDE_CODE_TIMEOUT_S` | Timeout, in seconds, for `claude_code` provider calls | Optional | `480.0` | Self-set — raise it if your `claude_code` calls are getting cut off; note the frontend's proxy timeout must stay above this value (`frontend/next.config.ts`) | All |
| `TB_OPENCLAW_URL` | Base URL of an OpenClaw instance (unverified/beta integration) | Optional (required together with the key below if `openclaw` is selected) | none | Self-set — points at your own OpenClaw deployment | All |
| `TB_OPENCLAW_API_KEY` | API key for that OpenClaw instance | Optional, paired with the URL above | none | Whatever issues keys for your OpenClaw deployment — operator-specific | All |
| `TB_FOREXFACTORY_CALENDAR_URL` | ForexFactory news-calendar base URL | Optional | `https://nfs.faireconomy.media` | Self-set — rarely changed | All |
| `TB_FINNHUB_CALENDAR_URL` | Finnhub economic-calendar API base URL | Optional | `https://finnhub.io/api/v1` | Self-set — rarely changed | All |
| `TB_FINNHUB_API_KEY` | Finnhub API key | Optional — only required if `configs/news.yaml: calendar.source` is set to `finnhub` (default `forexfactory` needs no key) | none | [finnhub.io](https://finnhub.io) — free tier key from their dashboard after signup | All |
| `TB_APP_PASSWORD` | Single app-wide password gating every route except `/health` and `/auth/*` | Optional, but **strongly recommended** once the app is reachable from anywhere but localhost | empty (no login required) | You choose it yourself — a plain password, not something you look up externally. **See the Security notes section below — the installer does not prompt for this.** | All |
| `TB_LOG_FORMAT` | Log line format, `human` or `json` | Optional | `human` | Self-set | All |
| `TB_TELEGRAM_BOT_TOKEN` | Telegram bot token for alerting | Optional — only used if `configs/alerting.yaml: telegram.enabled` is `true` | none | Telegram — create a bot via **BotFather** in Telegram and copy the token it gives you | All |
| `TB_TELEGRAM_CHAT_ID` | Destination chat id for Telegram alerts | Optional, paired with the token above | none | Telegram — send your bot a message, then use any "get my chat id" bot/tool, or Telegram's own API, to read it back | All |
| `TB_SMTP_USERNAME` | SMTP auth username for email alerting | Optional — only used if `configs/alerting.yaml: email.enabled` is `true` | none | Your own SMTP provider account (the shipped default host is `smtp.gmail.com`, but any SMTP provider works) | All |
| `TB_SMTP_PASSWORD` | SMTP auth password for email alerting | Optional, paired with the username above | none | Your SMTP provider. **If using Gmail**, this must be a Gmail **App Password**, not your normal account password — Gmail rejects normal passwords for SMTP auth once 2FA is on | All |

**A Settings-page key always wins.** Every AI provider key above (except
`ollama`, which needs none, and `claude_code`, which uses your CLI login)
can also be entered on the app's Settings page instead of `.env`. A
Settings-page key is Fernet-encrypted at rest, takes effect immediately with
no restart, and is never written back into `.env`. If both are set, the
Settings-page value wins.

---

## 2. Per-account gateway process env vars (not `TB_`-prefixed)

The MT5 gateway is a separate process with its own environment — it does
**not** read the repo-root `.env` file itself (only the backend's `Settings`
class does that). These variables are supplied directly to each gateway
process's environment — by a systemd unit's `Environment=` lines, a
Windows `.cmd` launcher's `set VAR=` lines, or `make dev-gateway`'s inline
env prefix for manual dev use.

| Variable | Purpose | Required? | Default | How to obtain a value | Platforms |
|---|---|---|---|---|---|
| `GATEWAY_HOST` | Bind host for the gateway's own server | Optional | `127.0.0.1` | Self-set (installer prompt, or Makefile default) | Linux, Windows, `make dev-gateway` |
| `GATEWAY_PORT` | Listen port for the gateway | Optional | `8787` (auto-incremented per additional account: `8788`, `8789`, …) | Self-set | Same |
| `GATEWAY_SHARED_SECRET` | The gateway-side half of the shared-secret pair — must exactly match the backend's `TB_GATEWAY_SHARED_SECRET[_<ACCOUNT>]` for that same account | Same as `TB_GATEWAY_SHARED_SECRET` above | none | Same value, copied from `.env` at service-generation/launch time — see §1 | All |
| `MT5_TERMINAL_PATH` | Absolute path to this account's own `terminal64.exe` | Optional for a single/primary account; **required** for every additional concurrent account | none (unset = attach to whatever terminal is already running) | Self-determined — wherever you installed/Wine-installed that account's MT5 terminal | All |
| `MT5_TERMINAL_SUBPATH` | Legacy alias — path relative to the Wine prefix's `drive_c/` | Optional, legacy (`MT5_TERMINAL_PATH` takes precedence when both are set) | none | Self-determined | Linux/Wine only |

`GATEWAY_HOST`/`GATEWAY_PORT` per account aren't separate `.env` lines —
they live in that account's `gateway_url` field in `configs/accounts.yaml`
and get re-derived into the gateway process's own environment at
service-generation/launch time.

---

## 3. Broker credentials (MT5 login / password / server)

Covered in full at the top of this document. Summary for reference: these
three values come from your MT5 broker (demo or live account), are entered
only through the app's **MT5 Account** UI panel
(`backend/src/broker/api/routes.py`), and are stored — if you check
"remember" — as a Fernet-encrypted file per account
(`backend/src/broker/adapters/credential_store.py`), with the encryption
key itself held in the OS keyring, never in a file next to the ciphertext.
They are never environment variables and never appear in `installer/`,
`.env`, `.env.example`, or any Docker file.

---

## 4. Other OS-keyring-held secrets

Two more secrets live purely in your OS's keyring (GNOME Keyring/KWallet on
Linux, Credential Manager on Windows) under the service name
`"trading-bot"` — never in `.env`, never something you choose or type in.
Both are auto-generated the first time they're needed
(`backend/src/shared/security/keyring_store.py`):

| Item | Purpose | Keyring key name |
|---|---|---|
| Credential-encryption Fernet key | Encrypts both your MT5 broker credentials and any AI provider API keys saved via the Settings page — one shared key for both stores | `credential-encryption-key` |
| Session-signing Fernet key | Signs/encrypts the session token issued when `TB_APP_PASSWORD` is set | `session-signing-key` |

There's nothing to configure here — just be aware that losing access to the
OS keyring (e.g. a fresh OS profile, or a keyring reset) invalidates every
stored credential and API key silently; you'd just re-enter them via the
UI.

---

## 5. `configs/*.yaml` — no secret values, only switches

No YAML file under `configs/` holds an actual secret value. A few fields
*name* which `.env` variable to use, or *select* whether a secret is
needed at all:

- `configs/accounts.yaml` — each account's `gateway_shared_secret_env`
  field names the `.env` variable holding that account's secret (e.g.
  `TB_GATEWAY_SHARED_SECRET_DEMO_1`) — never the secret's value itself. The
  file's own header comment states this rule explicitly.
- `configs/news.yaml` — `calendar.source: forexfactory | finnhub` decides
  whether `TB_FINNHUB_API_KEY` is required at all (the default,
  `forexfactory`, needs no key).
- `configs/alerting.yaml` — `telegram.enabled` / `email.enabled` gate
  whether the corresponding `TB_TELEGRAM_*` / `TB_SMTP_*` variables are
  actually used; the file itself only holds non-secret host/port/address
  settings.
- `configs/ai.yaml` — `provider_per_task` names which provider/model
  handles each AI task; its comments point at which `TB_*_API_KEY` each
  provider needs, but the file carries no key values.

---

## 6. Frontend variables

| Variable | Purpose | Required? | Default | Platforms |
|---|---|---|---|---|
| `BACKEND_URL` | Backend base URL the Next.js server proxies `/api/*` rewrites to (server-side only — not exposed to the browser) | Optional | `http://127.0.0.1:8000` | Self-set; only Docker overrides it (`docker-compose.prod.yml` sets `BACKEND_URL=http://backend:8000` for container-to-container networking) |
| `NEXT_PUBLIC_WS_URL` | Socket.IO base URL for live market-data streaming — bypasses the Next.js rewrite proxy, since it doesn't support WebSockets | Optional | `http://127.0.0.1:8000` | Self-set only if your backend isn't at the default dev address |

**Known gap on `NEXT_PUBLIC_WS_URL`**: there is currently no configuration
path for this variable in Docker — it isn't in `.env.example`, isn't set in
either Compose file's `environment:` block, and isn't wired by the
installer. In the Docker production topology (frontend and backend in
separate containers on separate hostnames), the compiled-in default of
`http://127.0.0.1:8000` is wrong from the browser's point of view. Note
also that `NEXT_PUBLIC_*` variables are inlined into the JavaScript bundle
at Next.js **build time** — setting it as a Compose `environment:` entry on
the already-built image would have no effect; it would need to become a
Docker build `ARG` instead. If you hit this, the workaround today is to
rebuild the frontend image with the value baked in at build time, or run
the frontend natively where the default resolves correctly.

---

## 7. Docker Hub CI secrets (repo owner only)

These aren't runtime secrets — they're GitHub repository secrets/variables
that let `.github/workflows/docker-publish.yml` push images to Docker Hub.
See §"GitHub Actions" below for exactly where to set them.

| Name | Kind | Purpose |
|---|---|---|
| `DOCKERHUB_USERNAME` | Repo secret | Docker Hub login username |
| `DOCKERHUB_TOKEN` | Repo secret | Docker Hub **access token** — generate this from Docker Hub's account settings, not your account password |
| `DOCKERHUB_NAMESPACE` | Repo variable (not secret) | Which Docker Hub org/user to publish images under; falls back to the placeholder `tradingbot` if unset |

Get a Docker Hub access token from [hub.docker.com](https://hub.docker.com)
→ Account Settings → Security → New Access Token.

---

## Platform-specific: how do I actually set this?

### (a) Linux native

1. `cp .env.example .env`, then edit `.env` with any editor — it's a plain
   `KEY=value` file, one variable per line.
2. If you registered autostart services via the installer, each systemd
   `--user` unit under `~/.config/systemd/user/` (`trading-bot-backend.service`,
   `trading-bot-gateway@<id>.service`, etc.) has the account's gateway
   secret and terminal path **baked directly into its `Environment=` lines**
   at generation time (`installer/services_linux.py`'s `render_all()`) — it
   does not re-read `.env` on every start. The backend and frontend units
   read most of the rest of `.env` normally at process startup (via
   `pydantic-settings`), since they're plain Python/Node processes with
   `.env` in their working directory.
3. After editing `.env`, restart the affected service(s) —
   `python3 installer/manage.py restart` (or `systemctl --user restart
   trading-bot-backend.service` directly) — a plain file edit doesn't
   propagate to an already-running process.

### (b) Windows native

1. `.env` lives at the repo root, same format as Linux — edit it directly.
2. Autostart on Windows works differently: `install_services.ps1` reads
   `.env` **once, at generation time**, via its `Read-DotEnvValue` helper (a
   minimal parser that returns the first uncommented `Name=value` line for
   a given key), and writes the resolved value directly into a generated
   `.cmd` launcher under `installer\services\windows\generated\` (e.g.
   `set GATEWAY_SHARED_SECRET=...`). The Scheduled Task then just runs that
   `.cmd` file — it never reads `.env` itself at task-run time.
3. Because of that, editing `.env` after services are registered does
   **not** update an already-generated `.cmd` launcher — see "Rotating a
   leaked secret" below.
4. The Windows service scripts are explicitly marked **UNVERIFIED** in
   their own header comments (written against documented PowerShell
   `ScheduledTasks` behavior, not yet run against a real Windows machine) —
   test on a disposable VM before relying on them for a live account.

### (c) Docker / Compose

1. `cp .env.example .env`, edit it, then `docker compose -f
   docker-compose.prod.yml up -d`. Both the `backend` and `frontend`
   services declare `env_file: .env`, so Compose reads the whole file and
   injects every variable into both containers' environments at container
   start — no per-service filtering happens.
2. **Known limitation, not something to fix by editing the compose file**:
   the frontend container currently receives the *entire* backend `.env` —
   every AI provider key, every gateway shared secret, `TB_APP_PASSWORD`,
   alerting credentials — even though the frontend only actually needs
   `BACKEND_URL`. This is a real blast-radius consideration: if the
   frontend container were ever compromised (a dependency vulnerability, a
   supply-chain issue in an npm package, etc.), everything in your `.env`
   would be reachable from inside it. Until this is tightened, treat
   anything you put in `.env` as reachable from both containers, and avoid
   putting anything in `.env` you wouldn't want exposed that way.
3. The gateway is never containerized (see `gateway/README.md`) — a
   Docker-based deployment reaches an externally-run gateway via
   `TB_GATEWAY_URL` in `.env`, same as any other backend config value.

### (d) GitHub Actions

Repo owner setup only — go to **Settings → Secrets and variables →
Actions** on GitHub.

| Workflow | Trigger | Consumes |
|---|---|---|
| `.github/workflows/ci.yml` | Every push to `main`, every pull request | Nothing secret — the installer-smoke-test job copies `.env.example` → `.env` verbatim; no real credential is ever needed for lint/test/build |
| `.github/workflows/release.yml` | Push of a `v*` tag, or manual `workflow_dispatch` | `GITHUB_TOKEN` only — auto-provided by GitHub Actions, no setup needed beyond the workflow's own `permissions:` block |
| `.github/workflows/docker-publish.yml` | Push of a `v*` tag, or manual `workflow_dispatch` | Repo **secrets** `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (under "Secrets"); repo **variable** `DOCKERHUB_NAMESPACE` (under "Variables", not "Secrets" — it's not sensitive) |

`docker-publish.yml` is explicitly self-documented as **inert** until both
Docker Hub secrets are configured — until then it will simply fail at the
login step rather than publish anything.

---

## Security notes

**Never commit `.env`.** It's already listed in `.gitignore` (root
`.gitignore`, `.env` entry) — verify with `git check-ignore .env` if you're
ever unsure. Only `.env.example` (containing no real values) is meant to be
committed.

**Rotating a leaked secret is not just an `.env` edit.** If
`TB_GATEWAY_SHARED_SECRET` (or a per-account variant) leaks, editing `.env`
alone is not enough:
- On **Linux**, the value is baked into an already-generated systemd unit
  file's `Environment=` line — you need to re-run the installer/service
  generation (`installer/services_linux.py`, or
  `python3 installer/install.py --reconfigure`) so the unit is rewritten,
  then restart the service.
- On **Windows**, the value is baked into an already-generated `.cmd`
  launcher under `installer\services\windows\generated\` — re-run
  `install_services.ps1` to regenerate it.
- Also update the matching `.env` value on the gateway side (or the
  corresponding `GATEWAY_SHARED_SECRET` env var wherever that gateway
  process runs) so both halves match again.

A leaked AI provider key or alerting credential is simpler — just replace
it in `.env` (or on the Settings page, for AI provider keys) and restart
the backend; nothing else has a baked-in copy.

**`TB_APP_PASSWORD` gap — set this yourself before exposing the app.** The
installer wizard (`installer/wizard.py`) does **not** currently prompt for
`TB_APP_PASSWORD`, even though it's the single control gating every route
except `/health` and `/auth/*` on an app that can place live trades. If you
only ever run the CLI installer and never manually open `.env` afterward,
you can end up with an internet-reachable, live-trading instance with
**zero login required**. Before making this app reachable from anywhere
beyond `127.0.0.1` — a VPS, a port forward, a reverse proxy — open `.env`
yourself and set `TB_APP_PASSWORD` to a password of your own choosing, then
restart the backend.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Gateway calls return `401` | `TB_GATEWAY_SHARED_SECRET[_<ACCOUNT>]` in `.env` doesn't match the `GATEWAY_SHARED_SECRET` the gateway process actually launched with — check whether you edited `.env` after services were already generated (see "Rotating a leaked secret" above) |
| Gateway calls return `502` | Wrong `TB_GATEWAY_URL` (or per-account `gateway_url` in `configs/accounts.yaml`), or the gateway process/terminal isn't running at all |
| App loads with no login prompt, even remotely | `TB_APP_PASSWORD` is unset — see the security note above |
| An AI task returns `503` mentioning a `TB_*_API_KEY` name | That provider's key is missing/empty in both `.env` and the Settings page |
| An AI task returns `502` or an adapter error | The key is present but invalid, or the model id in `configs/ai.yaml`/Settings isn't one the provider recognizes |
| News calendar fails when `configs/news.yaml: calendar.source: finnhub` | `TB_FINNHUB_API_KEY` is missing — the default `forexfactory` source needs no key, `finnhub` does |
| Telegram/email alerts never arrive | Either the channel is `enabled: false` in `configs/alerting.yaml`, or the matching `TB_TELEGRAM_*`/`TB_SMTP_*` variables are empty/wrong |
| `docker compose ... pull` fails with image not found | `DOCKERHUB_NAMESPACE` still points at the placeholder `tradingbot` namespace, or the repo owner hasn't configured `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` yet — build the images locally instead (see `INSTALL.md`) |
| Editing `.env` doesn't change a running autostart service's behavior | Expected — see the platform sections above; restart (Linux) or regenerate the launcher (Windows) after editing `.env` |
