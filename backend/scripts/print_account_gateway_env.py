"""Resolve one `configs/accounts.yaml` entry into shell `export` statements,
for the Makefile's per-account gateway targets (`make dev-gateway
ACCOUNT=<id>`, `make mt5-login ACCOUNT=<id> ...`) to `eval` — this is the one
place that parses `accounts.yaml`, so the Makefile never re-implements YAML
parsing in shell.

Run from `backend/`:

    uv run python -m scripts.print_account_gateway_env [account_id]
    uv run python -m scripts.print_account_gateway_env --list-ids

With no `account_id`, resolves the first *enabled* account (the Makefile's
single-account targets — `dev`, `dev-gateway`, `mt5-login` with no
ACCOUNT= — default to that account). `--list-ids` prints every enabled
account's id, one per line, for `dev-gateway-all` to loop over.

On any resolution failure (no such account, no enabled accounts at all),
prints an `echo ... >&2; exit 1` line to stdout instead of exports, so
`eval "$(...)"` in the Makefile fails loudly in the invoking shell rather
than silently continuing with empty variables.
"""

from __future__ import annotations

import shlex
import sys
from urllib.parse import urlparse

from src.broker.domain.account import AccountConfig
from src.shared.config.loaders import load_accounts_config
from src.shared.config.settings import CONFIGS_DIR


def _bail(message: str) -> None:
    print(f"echo {shlex.quote(message)} >&2; exit 1")


def _resolve(account_id: str | None, accounts: list[AccountConfig]) -> AccountConfig | None:
    enabled = [a for a in accounts if a.enabled]
    if account_id:
        return next((a for a in accounts if a.id == account_id), None)
    if not enabled:
        return None
    return enabled[0]


def main(argv: list[str]) -> int:
    accounts = load_accounts_config(CONFIGS_DIR)

    if argv[:1] == ["--list-ids"]:
        for account in accounts:
            if account.enabled:
                print(account.id)
        return 0

    account_id = argv[0] if argv else None
    account = _resolve(account_id, accounts)
    if account is None:
        if account_id:
            _bail(f"no account '{account_id}' in configs/accounts.yaml")
        else:
            _bail("no enabled accounts in configs/accounts.yaml")
        return 1
    if not account.enabled:
        _bail(f"account '{account.id}' is disabled in configs/accounts.yaml")
        return 1

    parsed = urlparse(account.gateway_url)
    if not parsed.hostname or not parsed.port:
        _bail(
            f"account '{account.id}': gateway_url '{account.gateway_url}' "
            "must include a host and port"
        )
        return 1

    print(f"export TB_RESOLVED_ACCOUNT_ID={shlex.quote(account.id)}")
    print(f"export TB_RESOLVED_GATEWAY_HOST={shlex.quote(parsed.hostname)}")
    print(f"export TB_RESOLVED_GATEWAY_PORT={shlex.quote(str(parsed.port))}")
    print(f"export TB_RESOLVED_GATEWAY_SECRET_ENV={shlex.quote(account.gateway_shared_secret_env)}")
    print(f"export TB_RESOLVED_TERMINAL_PATH={shlex.quote(account.mt5_terminal_path or '')}")
    print(f"export TB_RESOLVED_TERMINAL_SUBPATH={shlex.quote(account.mt5_terminal_subpath or '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
