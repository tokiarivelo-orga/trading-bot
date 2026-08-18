"""Covers `scripts/print_account_gateway_env.py`: resolving one
`accounts.yaml` entry into the `export` statements the Makefile evals,
in particular that both `mt5_terminal_path` (new, absolute/OS-native) and
`mt5_terminal_subpath` (legacy, Wine-prefix-relative) round-trip correctly
whether set or left unset — see CLAUDE.md's account-portability rules.
"""

from scripts.print_account_gateway_env import main
from src.broker.domain.account import AccountConfig


def make_account(**overrides) -> AccountConfig:
    defaults = dict(
        id="default",
        label="Primary MT5 account",
        gateway_url="http://127.0.0.1:8787",
        gateway_shared_secret_env="TB_GATEWAY_SHARED_SECRET",
        mode="live",
        enabled=True,
    )
    defaults.update(overrides)
    return AccountConfig(**defaults)


def test_main_emits_both_terminal_path_exports_when_set(monkeypatch, capsys):
    accounts = [
        make_account(
            mt5_terminal_path="/home/user/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe",
            mt5_terminal_subpath="MT5-demo-1/terminal64.exe",
        )
    ]
    monkeypatch.setattr(
        "scripts.print_account_gateway_env.load_accounts_config", lambda configs_dir: accounts
    )

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert (
        "export TB_RESOLVED_TERMINAL_PATH="
        "'/home/user/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe'" in out
    )
    # shlex.quote only wraps in quotes when needed — no spaces here, so it's bare.
    assert "export TB_RESOLVED_TERMINAL_SUBPATH=MT5-demo-1/terminal64.exe" in out


def test_main_emits_empty_terminal_path_exports_when_unset(monkeypatch, capsys):
    accounts = [make_account()]
    monkeypatch.setattr(
        "scripts.print_account_gateway_env.load_accounts_config", lambda configs_dir: accounts
    )

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "export TB_RESOLVED_TERMINAL_PATH=''" in out
    assert "export TB_RESOLVED_TERMINAL_SUBPATH=''" in out


def test_main_prefers_mt5_terminal_path_when_only_it_is_set(monkeypatch, capsys):
    accounts = [
        make_account(
            mt5_terminal_path="/home/user/.mt5/drive_c/MT5-demo-1/terminal64.exe",
        )
    ]
    monkeypatch.setattr(
        "scripts.print_account_gateway_env.load_accounts_config", lambda configs_dir: accounts
    )

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    # shlex.quote only wraps in quotes when needed — no spaces here, so it's bare.
    assert (
        "export TB_RESOLVED_TERMINAL_PATH=/home/user/.mt5/drive_c/MT5-demo-1/terminal64.exe"
        in out
    )
    assert "export TB_RESOLVED_TERMINAL_SUBPATH=''" in out
