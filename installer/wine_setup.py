"""Wraps the root Makefile's `setup-wine` target for the installer.

Linux-only, and only ever invoked after the wizard's explicit final
confirmation (installer/wizard.py step 6) — never run automatically. This
module intentionally does nothing but shell out to the exact commands the
Makefile already runs, streaming their output, so the two never drift
apart.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def commands(wine_prefix: Path) -> list[list[str]]:
    """The exact argv lists this would run, in order — shared by the
    dry-run/confirmation preview (`describe`) and the real run (`run`) so
    they can never disagree with each other."""
    return [
        ["sudo", "dpkg", "--add-architecture", "i386"],
        ["sudo", "apt", "update"],
        ["sudo", "apt", "install", "--install-recommends", "-y", "wine64", "wine32", "winetricks"],
        # run with WINEPREFIX set in the environment, see run() below
        ["winetricks", "-q", "corefonts"],
    ]


def describe(wine_prefix: Path) -> list[str]:
    """Human-readable `$ ...` lines for the confirmation/dry-run screen."""
    lines = [" ".join(cmd) for cmd in commands(wine_prefix)]
    lines[-1] = f"WINEPREFIX={wine_prefix} {lines[-1]}"
    return lines


def run(wine_prefix: Path) -> None:
    """Actually provision Wine + winetricks corefonts for `wine_prefix`,
    streaming subprocess output. Requires sudo for the apt steps — the user
    will be prompted for their password by sudo itself."""
    print(f"Setting up Wine + winetricks (prefix: {wine_prefix}) — sudo required...")
    env = os.environ.copy()
    env["WINEPREFIX"] = str(wine_prefix)
    for cmd in commands(wine_prefix):
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

    print()
    print("Host setup done. Remaining manual steps (see gateway/README.md):")
    print(
        f"  1. WINEPREFIX={wine_prefix} wine mt5setup.exe                 "
        "# install the MT5 terminal"
    )
    print(
        f"  2. WINEPREFIX={wine_prefix} wine python-3.12.x-amd64.exe /quiet "
        "InstallAllUsers=0 PrependPath=1"
    )
    print(
        f"  3. WINEPREFIX={wine_prefix} wine python -m pip install "
        "MetaTrader5 fastapi uvicorn pydantic"
    )
    print("  4. Start the terminal, log in to your MT5 demo account, enable Algo Trading,")
    print("     and add your symbols to Market Watch — see 'Terminal configuration' in")
    print("     gateway/README.md.")
