"""Reads and writes the install directory's `.env` file.

Pure text-based read/modify/write (readlines, look for `^VARNAME=`, replace
or append) — no dotenv/YAML dependency, matching the Makefile's `env`
target philosophy: create `.env` from `.env.example` if missing (with a
fresh random primary secret, exactly like the Makefile's
`openssl rand -hex 32` step), and never clobber a value that's already
there. Extended here to also add a distinct secret var per additional MT5
account, and to record a custom Wine prefix when the wizard collects one.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, no runtime import
    from wizard import Account

_VAR_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

PRIMARY_SECRET_VAR = "TB_GATEWAY_SHARED_SECRET"
WINEPREFIX_VAR = "TB_WINEPREFIX"


@dataclass
class EnvChange:
    """One pending edit to `.env`. `description` is what dry-run/confirmation
    screens print — it is built to never contain a real secret value."""

    kind: str  # "create_file" | "set_var"
    var_name: str = ""
    value: str = ""
    description: str = ""


def account_secret_var(account_id: str) -> str:
    """`ftmo-1` -> `TB_GATEWAY_SHARED_SECRET_FTMO_1`."""
    slug = re.sub(r"[^A-Za-z0-9]", "_", account_id.strip()).upper()
    return f"TB_GATEWAY_SHARED_SECRET_{slug}"


def read_existing(env_path: Path) -> dict[str, str]:
    """`KEY -> value` for every uncommented `KEY=value` line in `env_path`.

    Only used for non-secret prefill/lookup (e.g. TB_WINEPREFIX) and for
    presence checks — callers must never print a value read from here back
    to the user for a secret-looking key.
    """
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        m = _VAR_LINE_RE.match(line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def existing_keys(env_path: Path) -> set[str]:
    return set(read_existing(env_path).keys())


def plan(
    install_dir: Path,
    env_example: Path,
    accounts: list[Account],
    wine_prefix: str | None,
) -> list[EnvChange]:
    """Compute the `.env` edits this run would make, without touching disk.

    Each account already carries its decided `gateway_secret_env` var name
    (wizard.py works this out, accounting for whether configs/accounts.yaml
    already had a primary account before this session — deliberately *not*
    just "first in this list", since a fresh custom id entered as the first
    account in a session must never end up sharing the plain
    `TB_GATEWAY_SHARED_SECRET` with a pre-existing primary account). This
    function only decides *how* to realize that: `TB_GATEWAY_SHARED_SECRET`
    itself is only ever generated fresh when `.env` is newly created here
    (never overwritten if `.env` already exists, matching `make env`'s
    "leave existing .env untouched" rule); every other named var is added
    only if that exact variable name isn't already present.
    """
    env_path = install_dir / ".env"
    changes: list[EnvChange] = []
    creating = not env_path.exists()

    if creating:
        changes.append(EnvChange(kind="create_file", description=".env: create from .env.example"))
        existing = dict(read_existing(env_example)) if env_example.exists() else {}
        changes.append(
            EnvChange(
                kind="set_var",
                var_name=PRIMARY_SECRET_VAR,
                value=secrets.token_hex(32),
                description=(
                    f".env: set {PRIMARY_SECRET_VAR}=<generated> (fresh random secret, "
                    "like `make env`)"
                ),
            )
        )
        existing[PRIMARY_SECRET_VAR] = "<generated above>"
    else:
        existing = dict(read_existing(env_path))

    for account in accounts:
        var = getattr(account, "gateway_secret_env", "") or PRIMARY_SECRET_VAR
        if var == PRIMARY_SECRET_VAR:
            # The primary var is only ever (re)established by the
            # create_file branch above — an account that resolves to it is
            # either the true from-scratch primary (already handled) or is
            # deliberately reusing the existing primary account's secret
            # (e.g. its id matched an existing entry and accounts_writer
            # will skip writing a new block for it); either way, nothing
            # more to do here.
            continue
        if var in existing:
            continue
        changes.append(
            EnvChange(
                kind="set_var",
                var_name=var,
                value=secrets.token_hex(32),
                description=f".env: append {var}=<generated>",
            )
        )
        # guards against double-adding a duplicate id within this same run
        existing[var] = "<generated>"

    if wine_prefix and existing.get(WINEPREFIX_VAR) != wine_prefix:
        changes.append(
            EnvChange(
                kind="set_var",
                var_name=WINEPREFIX_VAR,
                value=wine_prefix,
                description=f".env: set {WINEPREFIX_VAR}={wine_prefix}",
            )
        )

    return changes


def apply(install_dir: Path, env_example: Path, changes: list[EnvChange]) -> None:
    """Write the changes computed by `plan()` to `.env`, creating it from
    `.env.example` first if that was part of the plan. Idempotent: re-running
    with the same inputs after a partial run only fills in what's still
    missing, since every `set_var` either replaces an existing line in place
    or appends — it never duplicates a variable."""
    env_path = install_dir / ".env"

    if any(c.kind == "create_file" for c in changes):
        if not env_example.exists():
            raise FileNotFoundError(f".env.example not found at {env_example}")
        install_dir.mkdir(parents=True, exist_ok=True)
        env_path.write_text(env_example.read_text())

    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path} (no create step was planned)")

    lines = env_path.read_text().splitlines(keepends=True)
    for change in changes:
        if change.kind != "set_var":
            continue
        lines = _set_var(
            lines,
            change.var_name,
            change.value,
            try_uncomment=(change.var_name == WINEPREFIX_VAR),
        )
    env_path.write_text("".join(lines))


def _set_var(lines: list[str], name: str, value: str, *, try_uncomment: bool = False) -> list[str]:
    """Replace an existing `NAME=...` line in place; if none exists and
    `try_uncomment`, replace a commented `#NAME=...` line instead (e.g. the
    commented-out `# TB_WINEPREFIX=...` template line in `.env.example`);
    otherwise append a new line."""
    live_re = re.compile(rf"^{re.escape(name)}=")
    for i, line in enumerate(lines):
        if live_re.match(line):
            lines[i] = f"{name}={value}\n"
            return lines
    if try_uncomment:
        commented_re = re.compile(rf"^#\s*{re.escape(name)}=")
        for i, line in enumerate(lines):
            if commented_re.match(line):
                lines[i] = f"{name}={value}\n"
                return lines
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    lines.append(f"{name}={value}\n")
    return lines
