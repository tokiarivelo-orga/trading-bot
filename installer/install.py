#!/usr/bin/env python3
"""AI Trading Bot -- setup wizard entrypoint.

Stdlib-only (argparse, pathlib, subprocess, platform, secrets, getpass,
shutil, textwrap, sys, os -- no third-party dependencies) so it can run
*before* `uv sync` / `pnpm install` have ever happened. Run as:

    python3 installer/install.py          # Linux/macOS
    python installer\\install.py           # Windows

Works under either the `python` or `python3` name -- it doesn't matter which
one the user typed, since this file itself makes no assumption about PATH
beyond "whatever interpreter is currently running me".

See the repo root CLAUDE.md for the project's binding rules: this installer
never touches configs/risk.yaml or backend/src/engine/, and never writes MT5
broker credentials anywhere -- those stay in the app UI -> OS keyring flow,
entirely out of scope here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Running as `python3 installer/install.py` already puts this file's own
# directory first on sys.path (CPython's documented startup behavior), but
# we insert it explicitly too so `import wizard` etc. below also work if
# this file is ever invoked in some other way (e.g. `python -c`).
_INSTALLER_DIR = Path(__file__).resolve().parent
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

import accounts_writer  # noqa: E402
import deps  # noqa: E402
import env_writer  # noqa: E402
import wine_setup  # noqa: E402
import wizard  # noqa: E402

if platform.system() == "Linux":
    import services_linux  # noqa: E402
else:
    services_linux = None  # type: ignore[assignment]

STATE_FILE_NAME = "state.json"


def _state_path() -> Path:
    return _INSTALLER_DIR / STATE_FILE_NAME


def load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(data: dict) -> None:
    _state_path().write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description=(
            "Interactive setup wizard for the AI Trading Bot: writes .env and "
            "configs/accounts.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print every file it would write and every command it would run, "
            "without writing or running anything."
        ),
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help=(
            "Re-run the wizard, pre-filling every prompt's default from "
            "installer/state.json and the existing .env/configs/accounts.yaml "
            "instead of the hardcoded defaults."
        ),
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help=(
            "Remove registered autostart services (systemd --user units on Linux; "
            "prints the Scheduled Task removal command on Windows). Leaves .env and "
            "configs/accounts.yaml untouched."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Accept every default without prompting (for scripted/CI testing of "
            "the installer itself)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.uninstall:
        if platform.system() == "Linux":
            print("Removing systemd --user services (trading-bot.target and its units)...")
            services_linux.uninstall(dry_run=args.dry_run)
            print(
                "\nServices removed. .env and configs/accounts.yaml are left untouched --"
                "\nremove the account block(s)/secret vars by hand if you also want those gone."
            )
        else:
            print(
                "Windows autostart isn't uninstalled from here -- run:\n"
                "  powershell -ExecutionPolicy Bypass -File "
                "installer\\services\\windows\\uninstall_services.ps1\n\n"
                "That removes every TradingBot-* Scheduled Task and its generated launcher.\n"
                ".env and configs/accounts.yaml are left untouched either way."
            )
        return 0

    repo_root = _INSTALLER_DIR.parent

    print("Checking for uv / node / pnpm on PATH...")
    deps.check(exit_on_missing=not args.dry_run)
    print()

    prior_state = load_state() if args.reconfigure else {}

    result = wizard.run(
        repo_root,
        non_interactive=args.non_interactive,
        reconfigure=args.reconfigure,
        dry_run=args.dry_run,
        prior_state=prior_state,
    )

    if args.dry_run:
        print("\nDry run complete -- nothing was written or run.")
        return 0

    if not result.proceed:
        print("\nAborted -- nothing was written.")
        return 1

    env_writer.apply(result.answers.install_dir, repo_root / ".env.example", result.env_changes)
    accounts_writer.apply(
        result.answers.install_dir / "configs" / "accounts.yaml", result.account_changes
    )

    for change in result.account_changes:
        if change.skipped:
            print(f"warning: {change.skip_reason}")

    if result.answers.run_wine_setup and result.answers.wine_prefix:
        wine_setup.run(Path(result.answers.wine_prefix))

    state = wizard.answers_to_state(result.answers)
    save_state(state)

    if result.answers.autostart_on_boot and platform.system() == "Linux":
        print("\nRegistering systemd --user autostart services...")
        services_linux.install(state)

    print("\nDone. Next steps:")
    print("  make setup       # uv sync + pnpm install (if not already run)")
    print("  make db-upgrade  # apply database migrations")
    if result.answers.autostart_on_boot and platform.system() == "Linux":
        print("  python3 installer/manage.py status   # check the registered services")
    elif result.answers.autostart_on_boot:
        print(
            "  powershell -ExecutionPolicy Bypass -File "
            "installer\\services\\windows\\install_services.ps1 "
            f"-RepoRoot {result.answers.install_dir} -StateJsonPath installer\\state.json"
        )
    else:
        print("  make dev         # start backend + frontend + the default account's gateway")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
