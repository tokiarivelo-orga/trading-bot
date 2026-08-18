"""Broker account domain: credentials and account state. Pure values, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class Mt5Credentials:
    login: int
    password: str = field(repr=False)  # keep passwords out of repr/logs
    server: str


@dataclass(frozen=True, kw_only=True)
class AccountInfo:
    login: int
    server: str
    name: str
    currency: str
    balance: float
    equity: float
    leverage: int


@dataclass(frozen=True, kw_only=True)
class GatewayHealth:
    gateway_up: bool
    terminal_connected: bool
    account: AccountInfo | None = None


@dataclass(frozen=True, kw_only=True)
class AccountConfig:
    """One entry from `configs/accounts.yaml` — a broker account this
    backend can run against, reached through its own gateway process.

    `id` is a short slug, not the MT5 login number — it's the identity used
    downstream in API paths, DB rows, and credential file names.
    """

    id: str
    label: str
    gateway_url: str
    gateway_shared_secret_env: str
    mode: str  # "paper" | "live"
    enabled: bool = True
    risk_override_file: str | None = None
    # Absolute, OS-native path to this account's own terminal64.exe — a
    # plain Windows path on a native Windows install, or an absolute Linux
    # path for Wine (e.g.
    # "/home/user/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe").
    # This is the new primary field going forward: it works for both native
    # Windows installs (no Wine prefix at all) and Wine setups where the
    # prefix location varies per machine. None for the primary account
    # (attaches to whichever terminal is already running, unchanged
    # pre-multi-account behavior) — required for any additional concurrent
    # account, since MetaTrader5 allows one login per terminal. Takes
    # precedence over `mt5_terminal_subpath` when both are set.
    mt5_terminal_path: str | None = None
    # Legacy/still-supported form: path to this account's own terminal64.exe,
    # relative to the Wine prefix's drive_c/ (e.g.
    # "MT5-demo-1/terminal64.exe"), resolved against whatever Wine prefix the
    # invoking Makefile/shell has configured. Keeps working unchanged for
    # existing Linux/Wine setups; prefer `mt5_terminal_path` for new entries.
    mt5_terminal_subpath: str | None = None


class BrokerUnavailable(Exception):
    """Gateway unreachable or the terminal rejected the request."""


class LoginRejected(Exception):
    """MT5 refused the credentials (bad login/password/server)."""
