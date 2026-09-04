> 🇫🇷 Version française : [INSTALL.fr.md](INSTALL.fr.md)

# Installation Guide — AI Trading Bot

This is the full reference for getting the bot running on a machine that
isn't your own dev checkout — a home PC, a VPS, or a server you administer
for someone else. The root [`README.md`](README.md#installing-on-another-machine)
has the short version; this file is what it points to.

## Overview

There are three ways to get this running:

- **CLI installer** (`installer/install.py`) — sets up the full stack,
  including the MT5 gateway, on any OS. Recommended for most people.
- **Docker** (`docker-compose.prod.yml`) — backend + frontend only, pulled
  as prebuilt images, for when the MT5 gateway already runs somewhere else.
- **Manual dev setup** — for people contributing to the code itself; see
  [`LAUNCH.md`](LAUNCH.md).

| Your situation | Path |
|---|---|
| I want the whole thing running on my machine or a VPS, gateway included | **Path 1 — CLI installer** |
| I already have (or will separately set up) an MT5 gateway elsewhere and just want the app | **Path 2 — Docker** |
| I'm going to work on the code itself | **Path 3 — manual dev setup** (`LAUNCH.md`) |

## Requirements

### Operating system

| OS | CLI installer (Path 1) | Docker (Path 2) |
|---|---|---|
| **Windows 10/11** | Full stack, natively — no Wine. `MetaTrader5`'s Python package only ships Windows wheels, so this is the simplest way to run the gateway. | backend + frontend only |
| **Linux** | Full stack — the installer can auto-provision Wine and a Wine-hosted MT5 terminal for you (opt-in prompt; installs `wine64`/`wine32`/`winetricks` via `apt`). | backend + frontend only |
| **macOS** | Not a supported target: the installer's autostart/uninstall logic only branches on "Linux" vs. "everything else is Windows" (`platform.system()`), so macOS gets Windows-worded prompts and the wrong uninstall instructions. Use Docker instead, or the manual gateway steps on a different machine. | backend + frontend only — Docker Desktop for Mac works fine; the gateway itself still has to run on Windows, on Wine (Linux), or on a VPS reachable over the network |

### Software prerequisites

| Path | Needs |
|---|---|
| CLI installer | Python 3.12+, [`uv`](https://docs.astral.sh/uv), Node.js + `pnpm`. The installer script itself is stdlib-only and runs before any of this exists — but it checks `uv`/`node`/`pnpm` are on PATH up front and prints an install command for whatever's missing before it lets you proceed. |
| Docker | Docker Engine + the `docker compose` plugin (v2 syntax — invoked as `docker compose -f docker-compose.prod.yml ...`). Nothing else; the images bundle everything. |
| Every path | A broker **MT5 demo account** (login number, password, server name — e.g. `MetaQuotes-Demo`), free from any MT5 broker. Entered later through the app UI's **MT5 Account** panel — never in `.env`, an installer prompt, or a Docker file; credentials go straight to the OS keyring. |

### Hardware & network

- **Disk space**: the backend's dependencies include `torch`, so budget a
  few GB free just for `uv sync`. A Wine prefix + MT5 terminal install
  (Path 1 on Linux) needs its own space on top of that, comparable to
  installing MT5 natively on Windows.
- **Network/latency**: live trading wants low latency to your broker's
  trade server — same guidance as `gateway/README.md`'s VPS option: ping
  the server name shown in the MT5 login dialog and pick a VPS region with
  the lowest round-trip time before switching out of paper mode.

## Path 1: CLI installer (recommended for most people)

`installer/install.py` is a stdlib-only Python script — it runs before
`uv sync`/`pnpm install` have ever happened. It writes `.env` and
`configs/accounts.yaml`, optionally provisions Wine + an MT5 terminal on
Linux, and optionally registers autostart services.

### 1. Get the code

- Download a source archive from the repo's **GitHub Releases** page
  (published by `.github/workflows/release.yml` on a version tag —
  `trading-bot-<version>.zip` / `.tar.gz`) and extract it, or
- `git clone` the repository.

### 2. Run the installer

```bash
python3 installer/install.py      # Linux/macOS
python installer\install.py       # Windows
```

It first checks `uv`, `node`, and `pnpm` are on PATH, printing an
OS-specific install command for whatever's missing (and exiting, unless
`--dry-run`).

Flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Prints every file it would write and every command it would run, without writing or running anything. |
| `--reconfigure` | Re-runs the wizard with every prompt's default pre-filled from `installer/state.json` and the existing `.env`/`configs/accounts.yaml`, instead of the hardcoded defaults. Use this later to add another account or change an earlier answer. |
| `--non-interactive` | Accepts every default without prompting — for scripted/CI use. |
| `--uninstall` | See [Uninstalling](#uninstalling) below. |

### 3. Answer the prompts

In order:

1. **Install directory** — defaults to wherever you extracted/cloned the
   repo.
2. **Per MT5 account** (asks "Add an MT5 account?", then "Add another MT5
   account?" until you decline):
   - **Account id** (slug) — enter `default` for the primary account
     `configs/accounts.yaml` already ships with; anything else becomes a
     new entry. Reusing an id that already exists is safe — it's skipped
     with a warning, never overwritten.
   - **Label** — a human-readable name.
   - **Mode** — `paper` or `live` (defaults to `paper`; anything else also
     falls back to `paper`).
   - **Gateway host** — defaults to `127.0.0.1`.
   - **Gateway port** — defaults to `8787` for the first account,
     auto-incrementing to the next free port for each additional one.
   - **Path to this account's `terminal64.exe`** — leave blank to attach
     to whatever terminal is already running (fine for a single account).
     Every account beyond the first needs its own path here, or its
     gateway can silently log another account's terminal out.
3. **Wine + MT5 terminal auto-provisioning** — Linux only, and only asked
   if some account still needs a terminal path. Installs
   `wine64`/`wine32`/`winetricks` via `apt` and runs `winetricks
   corefonts` (needs `sudo`). Say no and it just prints the same commands
   for you to run by hand — `gateway/README.md`'s Option A.
4. **Autostart on boot** — on Linux this registers and enables systemd
   `--user` services for the backend, frontend, and every account's
   gateway + terminal, plus `loginctl enable-linger` so they survive
   logout/reboot. On Windows it only records your intent into
   `installer/state.json` — actually registering Scheduled Tasks needs a
   separate PowerShell step (printed at the end), since it isn't safe to
   auto-elevate from a script.
5. **Final confirmation** — a summary screen listing every `.env`/
   `configs/accounts.yaml` change, any Wine commands, and the autostart
   plan. Defaults to **no** — you must type `y` to proceed.
   (`--non-interactive` skips this and proceeds automatically;
   `--dry-run` never proceeds.)

### 4. What gets written

- **`.env`** — created from `.env.example` if it doesn't exist yet, with a
  freshly generated `TB_GATEWAY_SHARED_SECRET`. Every account beyond the
  true first one gets its own generated `TB_GATEWAY_SHARED_SECRET_<ID>`.
  For everything else you might want to set in `.env` (AI provider keys,
  alerting credentials, `TB_APP_PASSWORD`) and how to obtain each value,
  see [`SECRETS.md`](SECRETS.md).
- **`configs/accounts.yaml`** — one new block appended per new account,
  matching the file's existing formatting exactly. An id that already
  exists is never modified.
- **If you opted in**: the Wine prefix + MT5 terminal, and/or autostart
  services —
  - *Linux*: systemd `--user` units under `~/.config/systemd/user/`:
    `trading-bot.target`, `trading-bot-backend.service`,
    `trading-bot-frontend.service`, and per account
    `trading-bot-gateway@<id>.service` / `trading-bot-terminal@<id>.service`.
  - *Windows*: run the command the installer prints at the end:
    ```powershell
    powershell -ExecutionPolicy Bypass -File installer\services\windows\install_services.ps1 -RepoRoot <install dir> -StateJsonPath installer\state.json
    ```
    This registers `TradingBot-*` Scheduled Tasks, each pointing at a
    generated `.cmd` launcher under
    `installer\services\windows\generated\`.

    Note: the Windows service scripts are marked **UNVERIFIED** in their
    own header comments — written against documented PowerShell
    `ScheduledTasks` behavior but not yet run on a real Windows machine.
    Test on a disposable VM before relying on them for a live account.

### 5. Finish setup and check it worked

```bash
make setup       # uv sync + pnpm install, if you haven't already
make db-upgrade  # apply database migrations
python3 installer/manage.py status   # status of every registered service
```

`installer/manage.py status|start|stop|restart` drives `systemctl --user`
on Linux, or the `TradingBot-*` Scheduled Tasks on Windows; if nothing is
registered yet it says so instead of a raw error.

Didn't opt into autostart? Start everything by hand instead: `make dev`.

### 6. The one manual step: logging in to MT5

The installer never touches broker credentials — that's deliberate (see
`CLAUDE.md`'s "Installer & distribution" rule). Once the stack is running,
open the app UI's **MT5 Account** panel and log in with your demo (or
live) login/password/server. That's also when you enable **Algo Trading**
and add your symbols to Market Watch inside the terminal itself — see
`gateway/README.md`'s "Terminal configuration" section for the exact
steps, since none of that is scriptable from outside MT5.

## Path 2: Docker (backend + frontend only)

For when the MT5 gateway runs somewhere else already — a separate Windows
VPS, or a Wine-hosted gateway you set up by hand or via Path 1's
account/Wine steps. `docker-compose.prod.yml` never runs the gateway
itself: it's deliberately not containerized (Wine-in-a-container
GUI/rendering/sleep issues aren't worth it).

```bash
cp .env.example .env
# edit .env: set TB_GATEWAY_URL to wherever your gateway actually runs
docker compose -f docker-compose.prod.yml up -d
# or: make docker-up-prod
```

This pulls `${DOCKERHUB_NAMESPACE:-tradingbot}/backend` and
`.../frontend` at `${IMAGE_TAG:-latest}` — backend on port 8000, frontend
on 3000, with `./configs` mounted read-only and `./data` read-write into
the backend container.

`tradingbot` is currently a **placeholder** Docker Hub namespace (see the
`<!-- TODO -->` markers in `docker-compose.prod.yml` and
`.github/workflows/docker-publish.yml`) — until the real namespace is
confirmed and `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` are configured as
repo secrets, `docker compose ... pull` may 404. Build locally instead in
the meantime:

```bash
docker build -f backend/Dockerfile -t tradingbot/backend:local backend
docker build -f frontend/Dockerfile -t tradingbot/frontend:local frontend
DOCKERHUB_NAMESPACE=tradingbot IMAGE_TAG=local docker compose -f docker-compose.prod.yml up -d
```

Setting up the gateway itself is a separate step — follow
`gateway/README.md`, or run Path 1's installer just for the account/Wine
prompts. If you do use the installer that way, **decline the autostart
prompt**: the installer's autostart registers backend and frontend
services too, which would fight the containers for the same ports.

## Path 3: manual dev setup

For contributing to the code itself. Follow [`LAUNCH.md`](LAUNCH.md)'s
step-by-step walkthrough — it's the same steps the installer runs for
you, spelled out one at a time, with its own prerequisites table and
troubleshooting section. Not duplicated here.

## Configuring multiple accounts, paper vs. live

Each entry in `configs/accounts.yaml` is one MT5 login, reached through
its own gateway process — `MetaTrader5`'s Python package only supports
one logged-in account per OS process, so N accounts means N gateway
processes and, beyond the first, N separate terminal installs. Per
account:

- `mode: paper | live` — paper mode simulates orders in-memory and never
  reaches MT5; live sends real orders. The repo ships with `default` in
  `live` mode and `demo-1` in `paper` mode — check `configs/accounts.yaml`
  before you go live on the wrong one.
- its own `gateway_url` (host:port) and `gateway_shared_secret_env` (a
  distinct `.env` variable name — never a secret shared across accounts).
- `mt5_terminal_path` (preferred — works for both native Windows and
  Wine) or the legacy `mt5_terminal_subpath` (resolved against a Wine
  prefix). Every account beyond the first needs one of these set, or its
  gateway attaches to whatever terminal happens to be running and can
  silently log another account out.

The installer's wizard handles all of this for you in the accounts loop
(step 2 of Path 1 above): each additional account gets the next free
gateway port, its own generated secret variable, and its own
terminal-path prompt. Re-run `installer/install.py --reconfigure` any
time to add more accounts — it pre-fills previous answers and never
touches an existing account with a matching id.

## Uninstalling

```bash
python3 installer/install.py --uninstall
```

- **Linux**: stops, disables, and removes every `trading-bot.target` /
  `trading-bot-*.service` systemd `--user` unit the installer registered.
- **Windows**: doesn't uninstall directly (removing Scheduled Tasks isn't
  safe to auto-elevate from a script) — it prints the command to run
  instead:
  ```powershell
  powershell -ExecutionPolicy Bypass -File installer\services\windows\uninstall_services.ps1 -RepoRoot <install dir>
  ```
  which removes every `TradingBot-*` task and, with `-RepoRoot`, the
  generated `.cmd` launchers under
  `installer\services\windows\generated\` — some of those embed a copy of
  a gateway secret, so it's worth cleaning them up.

Either way, **`.env`, `configs/accounts.yaml`, and your data**
(`backend/data/`, journal, generated strategies, trained models) **are
never touched by uninstall.** Remove them by hand if you want a genuinely
clean slate.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Installer exits with "Missing required tooling" | `uv`/`node`/`pnpm` aren't on PATH — it prints the exact install command for your OS (e.g. `curl -LsSf https://astral.sh/uv/install.sh \| sh` on Linux, `winget install --id astral-sh.uv -e` on Windows, `brew install uv` on macOS). Install, then re-run `installer/install.py`. |
| I declined the Wine auto-provision prompt — now what? | `.env`/`configs/accounts.yaml` were still written, nothing's lost. Either follow `gateway/README.md`'s Option A by hand, or re-run `installer/install.py --reconfigure` and say yes this time (your other answers are pre-filled). |
| A Windows Scheduled Task is registered but the process isn't running | Check the generated launcher under `installer\services\windows\generated\<name>.cmd` — that's the actual command the task runs. Run it directly in a terminal to see the real error. `python installer\manage.py status` shows every task's state. |
| `terminal_connected: false` (from the gateway's `/health`, or `make doctor`) | The MT5 terminal isn't open or isn't logged in — start it and log in; the gateway reconnects on the next `/login`. Same root cause and fix as `LAUNCH.md`'s troubleshooting table. |
| `docker compose ... pull` fails / image not found | `tradingbot` is still a placeholder Docker Hub namespace — the real images may not be published yet. Build locally instead (see the Docker Hub note in Path 2 above), or set `DOCKERHUB_NAMESPACE`/`IMAGE_TAG` to point at images that do exist. |
| `502`/`401` errors right after install | Same causes as `gateway/README.md`'s troubleshooting table — wrong login/password/server name, or `TB_GATEWAY_SHARED_SECRET` not matching between backend and gateway. |
| Order calls fail with retcode `10027` | Algo Trading isn't enabled in the terminal — see `gateway/README.md`'s "Terminal configuration" step 2. This is never something the installer can do for you; it has to happen inside the MT5 terminal UI. |

## FAQ

**Is my broker password safe?** It never goes in `.env`, an installer
prompt, or a Docker file. It's entered only through the app UI's MT5
Account panel, encrypted with a key held in the OS keyring, and forwarded
to the gateway only at connect time.

**Can I run this unattended on a VPS?** Yes — that's what the autostart
services are for. Each one restarts on crash (Windows: up to 999 restarts,
one minute apart; Linux: systemd's restart-on-failure behavior) and comes
back after a reboot; `loginctl enable-linger` (Linux) keeps user services
running without an active login session.

**Windows or Linux — which should I pick?** Native Windows avoids Wine
entirely — the simplest path, no prefix quirks. Linux works fine too
(this is how the project itself was developed) and the installer
auto-provisions Wine for you if you opt in; it just adds a moving part.
Either way, for live trading a VPS near your broker's trade servers
matters more than the OS choice — see `gateway/README.md`'s Option B.

**Do I need `git`?** No — download a release archive (Path 1, step 1) and
extract it if you'd rather not clone.

**Can I add a second (or third) broker account later?** Yes — re-run
`installer/install.py --reconfigure` any time; see "Configuring multiple
accounts" above.
