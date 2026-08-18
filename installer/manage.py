#!/usr/bin/env python3
"""Cross-platform manage CLI for the AI Trading Bot's autostart services.

    python3 installer/manage.py status
    python3 installer/manage.py start
    python3 installer/manage.py stop
    python3 installer/manage.py restart

Stdlib-only (argparse, platform, subprocess, sys -- no third-party deps),
matching the rest of `installer/`. Detects the OS via `platform.system()`:

  - Linux: shells out to `systemctl --user {status,start,stop,restart}
    trading-bot.target` (registered by `installer/services_linux.py`), plus
    a per-unit status listing via `systemctl --user status 'trading-bot-*'`.
  - Windows: shells out to the `ScheduledTasks` PowerShell module against
    every `TradingBot-*` task (registered by
    `installer/services/windows/install_services.ps1`).

If nothing is registered yet on either OS, this prints a message pointing
at `installer/install.py` (and the relevant service-registration step)
instead of a raw systemctl/PowerShell error.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys

LINUX_TARGET = "trading-bot.target"
LINUX_UNIT_GLOB = "trading-bot-*"
WINDOWS_TASK_GLOB = "TradingBot-*"

_NOT_REGISTERED_LINUX = (
    "No trading-bot systemd --user services are registered yet.\n"
    "Run installer/install.py first, then:\n"
    "  python3 installer/services_linux.py install"
)
_NOT_REGISTERED_WINDOWS = (
    "No TradingBot-* Scheduled Tasks are registered yet.\n"
    "Run installer/install.py first, then:\n"
    "  powershell -ExecutionPolicy Bypass -File installer\\services\\windows\\install_services.ps1 "
    "-RepoRoot <repo> -StateJsonPath installer\\state.json"
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """subprocess.run wrapper that turns 'binary not found' into a
    CompletedProcess with a non-zero return code instead of an uncaught
    FileNotFoundError -- e.g. running this on a Linux box with no systemd
    (a minimal container) shouldn't crash, just report 'not registered'."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            cmd, returncode=127, stdout="", stderr=f"{cmd[0]}: not found"
        )


def _print_result(result: subprocess.CompletedProcess) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def _linux_registered() -> bool:
    result = _run(["systemctl", "--user", "list-unit-files", LINUX_TARGET])
    return result.returncode == 0 and LINUX_TARGET in result.stdout


def linux_manage(action: str) -> int:
    if not _linux_registered():
        print(_NOT_REGISTERED_LINUX)
        return 1

    if action == "status":
        _print_result(_run(["systemctl", "--user", "status", LINUX_TARGET]))
        print()
        _print_result(_run(["systemctl", "--user", "status", LINUX_UNIT_GLOB]))
        return 0

    result = _run(["systemctl", "--user", action, LINUX_TARGET])
    _print_result(result)
    return result.returncode


_WINDOWS_ACTION_COMMANDS = {
    "status": (
        f"Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' | Format-Table TaskName,State -AutoSize"
    ),
    "start": f"Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' | Start-ScheduledTask",
    "stop": f"Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' | Stop-ScheduledTask",
    "restart": (
        f"Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' | Stop-ScheduledTask; "
        "Start-Sleep -Seconds 2; "
        f"Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' | Start-ScheduledTask"
    ),
}


def _windows_registered() -> bool:
    result = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-ScheduledTask -TaskName '{WINDOWS_TASK_GLOB}' -ErrorAction SilentlyContinue "
            "| Measure-Object).Count",
        ]
    )
    return result.returncode == 0 and result.stdout.strip() not in ("", "0")


def windows_manage(action: str) -> int:
    if not _windows_registered():
        print(_NOT_REGISTERED_WINDOWS)
        return 1

    result = _run(["powershell", "-NoProfile", "-Command", _WINDOWS_ACTION_COMMANDS[action]])
    _print_result(result)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Status/start/stop/restart the AI Trading Bot's OS-registered autostart "
        "services.",
    )
    parser.add_argument("action", choices=["status", "start", "stop", "restart"])
    args = parser.parse_args(argv)

    system = platform.system()
    if system == "Linux":
        return linux_manage(args.action)
    if system == "Windows":
        return windows_manage(args.action)

    print(
        f"manage.py: unsupported platform '{system}' -- only Linux (systemd --user) and "
        "Windows (Scheduled Tasks) autostart services are supported.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
