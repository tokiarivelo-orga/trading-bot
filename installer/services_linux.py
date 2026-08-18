"""Generates + registers systemd **user** units (no root needed) for the
backend, frontend, and one MT5 gateway+terminal pair per account, from a
loaded `installer/state.json` (the dict shape `wizard.answers_to_state()`
writes -- see that function for the exact schema this module reads).

This is the Linux half of the "Autonomous: installer registers OS-level
services... with restart-on-failure" decision -- the Windows half lives in
`installer/services/windows/*.ps1`, and `installer/manage.py` drives both
from one `status|start|stop|restart` CLI.

Design, mirroring the pattern already used in `installer/wine_setup.py`:
  - Every function here is a plain call -- nothing runs at import time.
  - Rendering (turning state.json + resolved tool paths into unit-file text)
    is pure and side-effect-free (`render_all` / the `render_*` helpers) --
    tests exercise these directly and never touch this machine's real
    systemd user session.
  - Anything that touches disk or shells out to `systemctl`/`loginctl`
    (`generate`, `install`, `uninstall`) takes an explicit `dry_run: bool`
    that, when True, only prints what it would do.

Unit files are systemd *template* units for the per-account gateway/
terminal pair (`trading-bot-gateway@.service.tmpl` /
`trading-bot-terminal@.service.tmpl`, the "@" suffix) -- we render one fully
resolved, concretely-named instance file per account (e.g.
`trading-bot-gateway@ftmo-1.service`) rather than relying on systemd to
substitute per-instance values at load time, since values like GATEWAY_HOST/
GATEWAY_PORT/the secret differ per account and systemd's `%i` specifier only
ever expands to the instance *name* itself. Where `%i` *is* genuinely useful
(cross-referencing an account's gateway unit to that same account's terminal
unit, and unit Descriptions) it's left untouched in the templates for
systemd itself to resolve natively at load time -- see the comments in the
.tmpl files.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shlex
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_INSTALLER_DIR = Path(__file__).resolve().parent
if str(_INSTALLER_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR))

import env_writer  # noqa: E402 -- read-only use of env_writer.read_existing()

_TEMPLATES_DIR = _INSTALLER_DIR / "services" / "linux"

UNIT_DIR = Path.home() / ".config" / "systemd" / "user"

TARGET_NAME = "trading-bot.target"
BACKEND_UNIT_NAME = "trading-bot-backend.service"
FRONTEND_UNIT_NAME = "trading-bot-frontend.service"

DEFAULT_BACKEND_PORT = 8000  # matches the root Makefile's `BACKEND_PORT ?= 8000`
DEFAULT_FRONTEND_PORT = 3000  # matches the root Makefile's TB_FRONTEND_PORT fallback
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8787
DEFAULT_WINE_PREFIX = str(Path.home() / ".mt5")  # matches the root Makefile's WINEPREFIX fallback
DEFAULT_WINEDEBUG = "fixme-all"  # matches the root Makefile's WINEDEBUG default
DEFAULT_PRIMARY_SECRET_VAR = "TB_GATEWAY_SHARED_SECRET"


class MissingToolError(RuntimeError):
    """Raised by `resolve_tool_paths()` when `uv`/`pnpm`/`wine` aren't on
    PATH -- these are hard requirements to render a working unit file, in
    dry-run or not, since ExecStart= needs a real absolute path either way."""


@dataclass(frozen=True)
class ToolPaths:
    uv_bin: str
    pnpm_bin: str
    wine_bin: str
    wine_python: str


def resolve_tool_paths(wine_prefix: str, *, which=None) -> ToolPaths:
    """Resolve `uv`/`pnpm`/`wine` to absolute paths via `shutil.which` (or
    the injected `which` callable, for tests). Raises `MissingToolError`
    listing everything missing rather than generating a unit file with a
    bare command name that would silently fail under systemd's minimal
    PATH."""
    import shutil

    which = which or shutil.which
    uv_bin = which("uv")
    pnpm_bin = which("pnpm")
    wine_bin = which("wine")
    found = (("uv", uv_bin), ("pnpm", pnpm_bin), ("wine", wine_bin))
    missing = [name for name, path in found if not path]
    if missing:
        raise MissingToolError(
            "Can't generate systemd units -- not found on PATH: "
            + ", ".join(missing)
            + ". Install them first (see the root Makefile's `setup`/`setup-wine` targets), "
            "then re-run."
        )
    wine_python = str(
        Path(wine_prefix)
        / "drive_c"
        / "users"
        / getpass.getuser()
        / "AppData"
        / "Local"
        / "Programs"
        / "Python"
        / "Python312"
        / "python.exe"
    )
    return ToolPaths(uv_bin=uv_bin, pnpm_bin=pnpm_bin, wine_bin=wine_bin, wine_python=wine_python)


def _render_template(name: str, **values: str) -> str:
    text = (_TEMPLATES_DIR / name).read_text()
    # .substitute() (not safe_substitute) on purpose: fail loudly if a
    # template gains a placeholder this module doesn't know how to fill,
    # rather than silently leaving `${...}` text in a live unit file.
    return string.Template(text).substitute(**values)


def default_terminal_path(wine_prefix: str) -> str:
    """Fallback MT5 terminal path when an account didn't record one in the
    wizard (blank = "attach to whatever terminal is already running") --
    matches the root Makefile's `dev-gateway` fallback exactly."""
    return f"{wine_prefix}/drive_c/Program Files/MetaTrader 5/terminal64.exe"


def gateway_unit_name(account_id: str) -> str:
    return f"trading-bot-gateway@{account_id}.service"


def terminal_unit_name(account_id: str) -> str:
    return f"trading-bot-terminal@{account_id}.service"


def render_all(
    state: dict, tools: ToolPaths, *, env_values: dict[str, str] | None = None
) -> dict[str, str]:
    """Pure rendering: `state` (the installer/state.json dict shape) +
    already-resolved `tools` -> `{unit_filename: unit_file_text}`. No disk
    writes, no subprocess calls -- safe to call from tests. `env_values`
    lets tests inject a fake `.env` dict instead of this module reading a
    real one; when omitted, the install dir's real `.env` is read (secret
    *values*, not just names, are needed here to bake them into the
    generated gateway unit -- see the .tmpl file's comment on why)."""
    install_dir = Path(state["install_dir"])
    backend_dir = install_dir / "backend"
    frontend_dir = install_dir / "frontend"
    gateway_dir = install_dir / "gateway"
    env_path = install_dir / ".env"

    wine_prefix = state.get("wine_prefix") or DEFAULT_WINE_PREFIX

    if env_values is None:
        env_values = env_writer.read_existing(env_path) if env_path.exists() else {}

    frontend_port = env_values.get("TB_FRONTEND_PORT") or str(DEFAULT_FRONTEND_PORT)

    rendered: dict[str, str] = {}

    rendered[BACKEND_UNIT_NAME] = _render_template(
        "trading-bot-backend.service.tmpl",
        BACKEND_DIR=str(backend_dir),
        UV_BIN=tools.uv_bin,
        BACKEND_PORT=str(DEFAULT_BACKEND_PORT),
    )
    rendered[FRONTEND_UNIT_NAME] = _render_template(
        "trading-bot-frontend.service.tmpl",
        FRONTEND_DIR=str(frontend_dir),
        PNPM_BIN=tools.pnpm_bin,
        FRONTEND_PORT=frontend_port,
    )

    wants: list[str] = [BACKEND_UNIT_NAME, FRONTEND_UNIT_NAME]

    for account in state.get("accounts") or []:
        account_id = account["id"]
        terminal_path = account.get("terminal_path") or default_terminal_path(wine_prefix)

        rendered[terminal_unit_name(account_id)] = _render_template(
            "trading-bot-terminal@.service.tmpl",
            WINE_PREFIX=wine_prefix,
            WINEDEBUG=DEFAULT_WINEDEBUG,
            WINE_BIN=tools.wine_bin,
            TERMINAL_PATH=terminal_path,
        )

        secret_env_name = account.get("gateway_secret_env") or DEFAULT_PRIMARY_SECRET_VAR
        secret_value = env_values.get(secret_env_name, "")

        rendered[gateway_unit_name(account_id)] = _render_template(
            "trading-bot-gateway@.service.tmpl",
            GATEWAY_DIR=str(gateway_dir),
            WINE_PREFIX=wine_prefix,
            WINEDEBUG=DEFAULT_WINEDEBUG,
            GATEWAY_HOST=str(account.get("gateway_host") or DEFAULT_GATEWAY_HOST),
            GATEWAY_PORT=str(account.get("gateway_port") or DEFAULT_GATEWAY_PORT),
            MT5_TERMINAL_PATH=terminal_path,
            GATEWAY_SECRET_ENV_NAME=secret_env_name,
            GATEWAY_SHARED_SECRET=secret_value,
            WINE_BIN=tools.wine_bin,
            WINE_PYTHON=tools.wine_python,
        )

        wants.extend([terminal_unit_name(account_id), gateway_unit_name(account_id)])

    rendered[TARGET_NAME] = _render_template(
        "trading-bot.target.tmpl",
        WANTS_LINE=" ".join(wants),
    )
    return rendered


def generate(state: dict, *, dry_run: bool = False, unit_dir: Path = UNIT_DIR) -> dict[str, str]:
    """Resolve tool paths, render every unit, and (unless `dry_run`) write
    them to `unit_dir`. Gateway unit files get 0600 permissions since they
    carry a real secret value -- see the .tmpl file's comment."""
    tools = resolve_tool_paths(state.get("wine_prefix") or DEFAULT_WINE_PREFIX)
    rendered = render_all(state, tools)

    if dry_run:
        print(f"[dry-run] would write {len(rendered)} unit file(s) to {unit_dir}:")
        for filename in sorted(rendered):
            print(f"  {unit_dir / filename}")
        return rendered

    unit_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in rendered.items():
        out_path = unit_dir / filename
        out_path.write_text(content)
        if filename.startswith("trading-bot-gateway@"):
            out_path.chmod(0o600)
        print(f"wrote {out_path}")
    return rendered


def _run(cmd: list[str], *, dry_run: bool, check: bool = True) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] $ {printable}")
        return
    print(f"$ {printable}")
    subprocess.run(cmd, check=check)


def _systemctl(args: list[str], *, dry_run: bool, check: bool = True) -> None:
    _run(["systemctl", "--user", *args], dry_run=dry_run, check=check)


def enable_linger(*, dry_run: bool = False) -> None:
    """`loginctl enable-linger $USER` lets this user's systemd --user
    instance -- and therefore `trading-bot.target` and everything it
    Wants= -- keep running with no active login session (e.g. over SSH
    after the SSH session closes, or on a headless VPS with no auto-login).
    Without linger, systemd-logind tears down every unit in the user's
    systemd instance the moment their last session ends, which would defeat
    the entire point of registering these as autostart-on-boot services.
    This can prompt for polkit consent depending on the distro's rules for
    loginctl -- kept as its own function (rather than an inline step) so it
    can be dry-run/tested independently of the rest of install()'s
    sequence, even though install() below does call it as part of the full
    "make autostart actually survive a reboot" sequence the wizard's own
    top-level confirmation screen already gates."""
    user = os.environ.get("USER") or getpass.getuser()
    _run(["loginctl", "enable-linger", user], dry_run=dry_run)


def install(state: dict, *, dry_run: bool = False, unit_dir: Path = UNIT_DIR) -> None:
    """generate() the unit files, then daemon-reload, enable --now the
    target, and enable-linger so it survives reboots/logouts."""
    generate(state, dry_run=dry_run, unit_dir=unit_dir)
    _systemctl(["daemon-reload"], dry_run=dry_run)
    _systemctl(["enable", "--now", TARGET_NAME], dry_run=dry_run)
    enable_linger(dry_run=dry_run)


def uninstall(*, dry_run: bool = False, unit_dir: Path = UNIT_DIR) -> None:
    """Inverse of install(): stop + disable the target, delete every unit
    file this module could have generated, and daemon-reload. Works without
    needing installer/state.json (it just globs whatever's actually on disk
    under unit_dir), so it can clean up even a partial/stale install."""
    _systemctl(["stop", TARGET_NAME], dry_run=dry_run, check=False)
    _systemctl(["disable", TARGET_NAME], dry_run=dry_run, check=False)

    existing = sorted(unit_dir.glob("trading-bot-*.service")) + sorted(unit_dir.glob(TARGET_NAME))
    for path in existing:
        if dry_run:
            print(f"[dry-run] would remove {path}")
        else:
            path.unlink(missing_ok=True)
            print(f"removed {path}")

    _systemctl(["daemon-reload"], dry_run=dry_run)


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="services_linux.py",
        description="Generate + register systemd --user units for the AI Trading Bot "
        "(backend, frontend, one MT5 gateway+terminal pair per account).",
    )
    parser.add_argument(
        "action",
        choices=["generate", "install", "uninstall"],
        help="generate: render+write unit files only. install: generate + daemon-reload + "
        "enable --now + enable-linger. uninstall: stop/disable/remove everything.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=_INSTALLER_DIR / "state.json",
        help="Path to installer/state.json (default: installer/state.json next to this file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without touching disk or running systemctl/loginctl.",
    )
    args = parser.parse_args(argv)

    if platform.system() != "Linux":
        print(
            f"services_linux.py only supports Linux (detected: {platform.system()}).",
            file=sys.stderr,
        )
        return 1

    if args.action == "uninstall":
        uninstall(dry_run=args.dry_run)
        return 0

    if not args.state.exists():
        print(
            f"No installer state found at {args.state} -- run installer/install.py first.",
            file=sys.stderr,
        )
        return 1
    state = json.loads(args.state.read_text())

    if args.action == "generate":
        generate(state, dry_run=args.dry_run)
    else:
        install(state, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
