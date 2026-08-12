"""Adds a symbol to `configs/app.yaml`'s `symbols:` list at runtime — the
durable half of activating a symbol for live automated trading from the UI
(`SkillAssignmentService.assign`, §6.6), so it survives a restart without a
human ever hand-editing the file.

Deliberately does NOT do a full `yaml.safe_load` + `yaml.safe_dump`
round-trip of the whole file: PyYAML drops comments on dump, and this file
carries a load-bearing one (`mode: live  # ... NEVER switch to live before
Phase 9 criteria`). Instead this rewrites only the single `symbols:` line via
a targeted regex, leaving every other line — including both comments —
byte-for-byte untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.shared.config.atomic_write import atomic_write_text

_SYMBOLS_LINE = re.compile(r"^symbols:\s*\[.*\]\s*$", re.MULTILINE)


def add_symbol_to_app_config(symbol: str, configs_dir: Path) -> bool:
    """Appends `symbol` to app.yaml's `symbols:` list if it isn't already
    there. Returns whether the file was changed — idempotent, safe to call
    on every assign(), not just the first time a symbol is activated.

    Raises RuntimeError if app.yaml has no single-line `symbols: [...]` to
    rewrite (e.g. hand-reformatted to YAML block-sequence style) — fails
    loud rather than silently leaving the symbol un-persisted.
    """
    path = configs_dir / "app.yaml"
    text = path.read_text()
    match = _SYMBOLS_LINE.search(text)
    if match is None:
        raise RuntimeError(
            f"{path} has no single-line `symbols: [...]` list to update — "
            "can't persist the new symbol automatically, edit it by hand"
        )
    current: list[str] = yaml.safe_load(match.group(0))["symbols"]
    if symbol in current:
        return False
    updated = [*current, symbol]
    new_line = "symbols: " + yaml.safe_dump(updated, default_flow_style=True).strip()
    new_text = text[: match.start()] + new_line + text[match.end() :]
    atomic_write_text(path, new_text)
    return True

def remove_symbol_from_app_config(symbol: str, configs_dir: Path) -> bool:
    """Removes `symbol` from app.yaml's `symbols:` list if it is there."""
    path = configs_dir / "app.yaml"
    text = path.read_text()
    match = _SYMBOLS_LINE.search(text)
    if match is None:
        raise RuntimeError(f"{path} has no single-line `symbols: [...]` list to update")
    current: list[str] = yaml.safe_load(match.group(0))["symbols"]
    if symbol not in current:
        return False
    updated = [s for s in current if s != symbol]
    new_line = "symbols: " + yaml.safe_dump(updated, default_flow_style=True).strip()
    new_text = text[: match.start()] + new_line + text[match.end() :]
    atomic_write_text(path, new_text)
    return True
