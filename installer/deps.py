"""Checks whether the external tooling the rest of the stack needs (`uv`,
`node`, `pnpm`) is on PATH.

This installer itself is stdlib-only and doesn't need any of these to run —
but `make setup` / `make dev` do, so we check up front and point the user at
the exact install command for their OS rather than silently trying to
install system packages ourselves (that would be a much bigger blast radius
than this installer is meant to have).
"""

from __future__ import annotations

import platform
import shutil
import sys

REQUIRED_TOOLS: tuple[str, ...] = ("uv", "node", "pnpm")

# Debian/Ubuntu is the reference distro used by gateway/README.md and the
# Makefile's `setup-wine` target, so that's what the Linux hints target too.
_LINUX_HINTS = {
    "uv": "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "node": "sudo apt update && sudo apt install -y nodejs npm  # or use nvm: https://github.com/nvm-sh/nvm",
    "pnpm": "corepack enable && corepack prepare pnpm@latest --activate  # needs node first",
}
_WINDOWS_HINTS = {
    "uv": (
        'winget install --id astral-sh.uv -e   # or: powershell -c '
        '"irm https://astral.sh/uv/install.ps1 | iex"'
    ),
    "node": "winget install OpenJS.NodeJS.LTS",
    "pnpm": "corepack enable && corepack prepare pnpm@latest --activate  # needs node first",
}
_MACOS_HINTS = {
    "uv": "brew install uv",
    "node": "brew install node",
    "pnpm": "corepack enable && corepack prepare pnpm@latest --activate  # needs node first",
}


def _hints_for(os_name: str) -> dict[str, str]:
    if os_name == "Windows":
        return _WINDOWS_HINTS
    if os_name == "Darwin":
        return _MACOS_HINTS
    return _LINUX_HINTS


def find_missing(tools: tuple[str, ...] = REQUIRED_TOOLS) -> list[str]:
    """Return the subset of `tools` not found on PATH."""
    return [tool for tool in tools if shutil.which(tool) is None]


def report_missing(missing: list[str], *, os_name: str | None = None) -> None:
    os_name = os_name or platform.system()
    hints = _hints_for(os_name)
    print("Missing required tooling:")
    for tool in missing:
        hint = hints.get(tool, "(no install hint available — see gateway/README.md)")
        print(f"  - {tool}: {hint}")


def check(*, exit_on_missing: bool = True) -> bool:
    """Check for `uv`/`node`/`pnpm` on PATH, printing install hints for
    anything missing.

    Returns True if everything is present. If `exit_on_missing` is set and
    something is missing, calls `sys.exit(1)` instead of returning — used
    for a real (non-dry-run) install run, since `make setup`/`make dev`
    can't work without these. `--dry-run` passes `exit_on_missing=False` so
    the plan preview still prints in full even on a machine that hasn't
    installed them yet.
    """
    missing = find_missing()
    if not missing:
        print("uv, node, pnpm: all found on PATH.")
        return True
    report_missing(missing)
    if exit_on_missing:
        print("\nInstall the missing tooling above, then re-run this installer.")
        sys.exit(1)
    return False


if __name__ == "__main__":
    check()
