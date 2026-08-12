"""configs/app.yaml is edited at runtime when a symbol is activated for live
trading (§6.6, SkillAssignmentService.assign) — this must never destroy the
file's comments (one of which is a load-bearing paper/live safety warning),
must be idempotent, and must fail loudly rather than silently corrupt the
file if its `symbols:` line isn't in the expected single-line form."""

from __future__ import annotations

import pytest
import yaml

from src.shared.config.app_config_writer import add_symbol_to_app_config

APP_YAML = (
    "# Global app configuration. Hot-reloadable.\n"
    "mode: live              # paper | live  — NEVER switch to live before Phase 9 criteria\n"
    'timezone: "Indian/Antananarivo"\n'
    'symbols: [XAUUSD, XAGUSD]\n'
    "engine:\n"
    "  enabled: true\n"
    "  entry_timeframe: M5\n"
    "  confirmation_timeframes: [H1, H4]\n"
)


@pytest.fixture
def configs_dir(tmp_path):
    (tmp_path / "app.yaml").write_text(APP_YAML)
    return tmp_path


def test_appends_new_symbol(configs_dir):
    changed = add_symbol_to_app_config("Volatility 75 Index", configs_dir)

    assert changed is True
    text = (configs_dir / "app.yaml").read_text()
    assert "Volatility 75 Index" in text
    data = yaml.safe_load(text)
    assert data["symbols"] == [
        "XAUUSD",
        "XAGUSD",
        "Volatility 75 Index",
    ]


def test_preserves_every_other_line_byte_for_byte(configs_dir):
    add_symbol_to_app_config("Volatility 75 Index", configs_dir)

    text = (configs_dir / "app.yaml").read_text()
    lines = text.splitlines()
    original_lines = APP_YAML.splitlines()
    # Every line except the `symbols:` one is untouched, including both
    # comments — the mode-safety warning must survive verbatim.
    assert lines[0] == original_lines[0]
    assert lines[1] == original_lines[1]
    assert lines[2] == original_lines[2]
    assert lines[4:] == original_lines[4:]


def test_idempotent_on_already_present_symbol(configs_dir):
    first = add_symbol_to_app_config("XAUUSD", configs_dir)

    assert first is False
    assert (configs_dir / "app.yaml").read_text() == APP_YAML


def test_second_call_for_same_new_symbol_does_not_duplicate(configs_dir):
    add_symbol_to_app_config("Volatility 75 Index", configs_dir)
    changed_again = add_symbol_to_app_config("Volatility 75 Index", configs_dir)

    assert changed_again is False
    data = yaml.safe_load((configs_dir / "app.yaml").read_text())
    assert data["symbols"].count("Volatility 75 Index") == 1


def test_no_leftover_temp_file(configs_dir):
    add_symbol_to_app_config("Volatility 75 Index", configs_dir)

    assert not (configs_dir / "app.yaml.tmp").exists()


def test_raises_on_malformed_symbols_line(tmp_path):
    (tmp_path / "app.yaml").write_text("mode: paper\nsymbols:\n  - XAUUSD\n")

    with pytest.raises(RuntimeError):
        add_symbol_to_app_config("EURUSD", tmp_path)
