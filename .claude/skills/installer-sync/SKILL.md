---
name: installer-sync
description: Checklist to keep installer/, Docker images, and setup docs in sync whenever a change touches .env.example, configs/accounts.yaml's schema, Makefile setup/dev-gateway targets, or gateway bootstrapping (Wine/terminal-path resolution). Run this before declaring such a change done.
---

# Installer Sync

This repo is installable on a machine other than a developer's own checkout
via `installer/install.py` (CLI wizard, any OS) and
`docker-compose.prod.yml` (Docker Hub pull, backend+frontend only). Both are
**generated/scripted equivalents** of what a developer does by hand — if the
manual path changes and these don't, a fresh install silently breaks in a
way `make dev` on an existing dev checkout won't catch, because the
installer's defaults/templates drift out of sync with what the app actually
needs.

Trigger: before declaring done any change that touches one of —
`.env.example`, `configs/accounts.yaml`'s schema (a new/renamed/removed
field on an account entry), the root `Makefile`'s `setup`/`setup-wine`/
`dev-gateway`/`env` targets, or `gateway/src/gateway/mt5_client.py`'s
terminal-path/env-var resolution.

## Checklist

1. **New/changed `.env` variable** (`.env.example`) — does
   `installer/env_writer.py` need to write/update it? Does
   `installer/wizard.py` need a new prompt, or does it stay a manual
   post-install edit? If it's needed at container runtime, does
   `docker-compose.prod.yml` pass it through (it already forwards the whole
   `.env` via `env_file:` — only touch it if the var needs a *different*
   value inside the container, e.g. `BACKEND_URL` for the frontend).

2. **New/changed `configs/accounts.yaml` field** — update
   `backend/src/broker/domain/account.py`'s `AccountConfig` and
   `backend/src/shared/config/loaders.py`'s `load_accounts_config` first
   (the source of truth), then thread it through everywhere the installer
   touches an account: `installer/wizard.py`'s `Account` dataclass +
   prompts, `installer/accounts_writer.py`'s appended-entry shape,
   `installer/services_linux.py`'s `render_all()` (systemd unit
   `Environment=` lines), and `installer/services/windows/install_services.ps1`'s
   per-account launcher `.cmd` generation. Check
   `backend/scripts/print_account_gateway_env.py` too if the field needs to
   reach the `Makefile`'s `dev-gateway` target for non-installer dev use.

3. **New Makefile setup/dev-gateway step** — does `installer/wine_setup.py`
   (Linux) need the equivalent command? Does
   `installer/services/windows/install_services.ps1` need an equivalent
   native-Windows step? A step that only makes sense for a hand-built dev
   checkout (not a fresh install) doesn't need this — say so in the
   Makefile target's own comment so the next person doesn't have to
   rediscover that.

4. **Gateway terminal-path/env-var resolution changes**
   (`mt5_client.py`) — this is the account-isolation mechanism (one MT5
   terminal per account; get it wrong and a second account's gateway
   silently attaches to whatever terminal is already running and can log
   another account out). Any new/renamed env var here needs matching
   updates in **all three** places that set it: the root `Makefile`'s
   `dev-gateway` target, `installer/services_linux.py` (systemd unit
   template), and `installer/services/windows/install_services.ps1`
   (Scheduled Task launcher `.cmd`). Add a regression test in
   `gateway/tests/test_mt5_client.py` for the resolution logic itself, and
   one in `installer/tests/test_services_linux.py` asserting the rendered
   unit actually carries the right env line — a test that only checks
   *unrelated* fields (host/port/secret) will pass even if the terminal
   path silently regresses to blank, which is exactly the bug class this
   guards against.

5. **Docs** — `LAUNCH.md`'s "Dev setup (contributors)" section is the
   manual version of whatever you just automated; update both together, not
   just the automated path. `gateway/README.md`'s callout at the top
   references `installer/install.py` as automating what follows — no
   content change usually needed there unless the manual steps themselves
   changed. `CLAUDE.md`'s "Installer & distribution" section states the
   binding rule this skill operationalizes — update it only if the rule
   itself changes, not for every individual sync. If the change introduces
   a new required env var/secret, `SECRETS.md`/`SECRETS.fr.md` also need a
   new row/entry (purpose, required/default, how to obtain a value, which
   platforms) — that's the canonical secrets reference and drifts out of
   sync just like the installer templates do if it's skipped.

## Verify

- `python3 -m unittest discover -s installer/tests -v` (stdlib-only, no
  third-party deps needed) from the repo root.
- `cd backend && uv run ruff check ../installer/*.py` — the installer
  package isn't covered by `make lint`/`make check` yet, so ruff on it has
  to be run explicitly.
- If `configs/accounts.yaml`'s schema changed: `cd backend && uv run pytest
  tests/unit/shared/test_loaders.py tests/unit/scripts/` and
  `cd gateway && uv run pytest tests/test_mt5_client.py`.
