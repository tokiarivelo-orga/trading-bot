"""Reads and appends entries to `configs/accounts.yaml`.

Reading (existing ids / gateway ports, to avoid collisions) uses PyYAML when
it's importable in the environment running the installer, falling back to a
regex scan otherwise — this module has zero *required* third-party
dependencies either way (see `_yaml_existing_ids` / `_regex_existing_ids`).

Writing is always a targeted text append matching the file's existing
2-space-list-item / 4-space-key indentation exactly — never a full YAML
dump, which would blow away the file's header comments.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, no runtime import
    from wizard import Account

_ID_LINE_RE = re.compile(r"^\s*-\s*id:\s*(\S+)\s*$")
_GATEWAY_URL_PORT_RE = re.compile(r"^\s*gateway_url:\s*\S*:(\d+)\s*$")


def _yaml_existing_ids(accounts_path: Path) -> list[str] | None:
    """Ids read via PyYAML, or None if PyYAML isn't importable / parsing
    failed — callers fall back to `_regex_existing_ids` in that case."""
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml  # only imported once we know the module is actually there

    try:
        data = yaml.safe_load(accounts_path.read_text()) or {}
    except Exception:
        return None
    accounts = data.get("accounts") or []
    return [str(a["id"]) for a in accounts if isinstance(a, dict) and "id" in a]


def _regex_existing_ids(accounts_path: Path) -> list[str]:
    ids: list[str] = []
    for line in accounts_path.read_text().splitlines():
        m = _ID_LINE_RE.match(line)
        if m:
            ids.append(m.group(1).strip("'\""))
    return ids


def existing_ids(accounts_path: Path) -> list[str]:
    if not accounts_path.exists():
        return []
    ids = _yaml_existing_ids(accounts_path)
    return ids if ids is not None else _regex_existing_ids(accounts_path)


def existing_gateway_ports(accounts_path: Path) -> list[int]:
    if not accounts_path.exists():
        return []
    ports: list[int] = []
    for line in accounts_path.read_text().splitlines():
        m = _GATEWAY_URL_PORT_RE.match(line)
        if m:
            ports.append(int(m.group(1)))
    return ports


@dataclass
class AccountChange:
    account_id: str
    block_text: str
    description: str
    skipped: bool = False
    skip_reason: str = ""


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_entry(account: Account) -> str:
    """Matches configs/accounts.yaml's exact field order/shape: id, label,
    gateway_url, gateway_shared_secret_env, mode, enabled,
    risk_override_file, then mt5_terminal_path only if the wizard collected
    one (2-space list item, 4-space keys, mode's trailing comment)."""
    lines = [
        f"  - id: {account.id}",
        f"    label: {_quote(account.label)}",
        f"    gateway_url: {account.gateway_url}",
        f"    gateway_shared_secret_env: {account.gateway_secret_env}",
        f"    mode: {account.mode} # paper | live",
        "    enabled: true",
        "    risk_override_file: null",
    ]
    terminal_path = getattr(account, "terminal_path", None)
    if terminal_path:
        lines.append(f"    mt5_terminal_path: {_quote(terminal_path)}")
    return "\n".join(lines) + "\n"


def plan(accounts_path: Path, accounts: list[Account]) -> list[AccountChange]:
    """Compute the accounts.yaml edits this run would make, without touching
    disk. Never modifies or removes an existing entry with a matching id —
    such accounts come back as a `skipped` AccountChange with a warning
    instead."""
    already = set(existing_ids(accounts_path))
    seen_this_run: set[str] = set()
    changes: list[AccountChange] = []
    for account in accounts:
        if account.id in already or account.id in seen_this_run:
            reason = f"account '{account.id}' already exists in accounts.yaml, leaving it untouched"
            changes.append(
                AccountChange(
                    account_id=account.id,
                    block_text="",
                    description=f"configs/accounts.yaml: SKIP '{account.id}' (already exists)",
                    skipped=True,
                    skip_reason=reason,
                )
            )
            continue
        seen_this_run.add(account.id)
        changes.append(
            AccountChange(
                account_id=account.id,
                block_text=_format_entry(account),
                description=(
                    f"configs/accounts.yaml: append account '{account.id}' "
                    f"({account.mode}, {account.gateway_url})"
                ),
            )
        )
    return changes


def apply(accounts_path: Path, changes: list[AccountChange]) -> None:
    to_append = [c for c in changes if not c.skipped]
    if not to_append:
        return
    if not accounts_path.exists():
        raise FileNotFoundError(f"configs/accounts.yaml not found at {accounts_path}")
    text = accounts_path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    for change in to_append:
        text += "\n" + change.block_text
    accounts_path.write_text(text)
