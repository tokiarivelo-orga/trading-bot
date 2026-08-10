"""Walk-forward backtest CLI (OBSERVABILITY_PLAN.md Phase 6 Pass B).

    uv run python -m src.backtest.walk_forward_cli <strategy> <symbol> <period> [fold_months]
    uv run python -m src.backtest.walk_forward_cli breakout_v1 XAUUSD 2025-01:2025-06
    uv run python -m src.backtest.walk_forward_cli breakout_v1 XAUUSD 2025-01:2025-12 3

Run from `backend/`. Splits `period` into `fold_months`-sized (default 1,
monthly) consecutive windows and runs each as an independent out-of-sample
backtest via `run_walk_forward` — see `application/walk_forward.py`'s
docstring for why this is not classic in-sample-refit walk-forward
optimization. Reads candle history already persisted by
`market_data.CandleRepository` (via `POST /market-data/backfill`) — if a
fold has none, it is skipped (not fatal), same as `cli.py` exits with a
message telling you to backfill first if NO fold has any history at all.

This takes roughly `fold_count` times as long as `cli.py` for the same
period — mirror this CLI's own warning before running it on a long period on
a fine timeframe.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from src.backtest.application.period import InvalidPeriod
from src.backtest.application.run_backtest import NoSymbolSpecError
from src.backtest.application.walk_forward import run_walk_forward
from src.backtest.reports.walk_forward_writer import (
    render_walk_forward_summary,
    write_walk_forward_report,
)
from src.shared.config.settings import Settings


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(
            "usage: python -m src.backtest.walk_forward_cli <strategy> <symbol> <period> "
            "[fold_months]",
            file=sys.stderr,
        )
        print(
            "e.g.:  python -m src.backtest.walk_forward_cli breakout_v1 XAUUSD "
            "2025-01:2025-12 3",
            file=sys.stderr,
        )
        return 2

    strategy_name, symbol, period = argv[:3]
    try:
        fold_months = int(argv[3]) if len(argv) == 4 else 1
    except ValueError:
        print(f"error: fold_months must be an integer, got {argv[3]!r}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()

    try:
        report = asyncio.run(
            run_walk_forward(
                strategy_name,
                symbol,
                period,
                fold_months=fold_months,
                database_url=settings.database_url,
            )
        )
    except (InvalidPeriod, NoSymbolSpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    path = write_walk_forward_report(report)
    print(render_walk_forward_summary(report))
    print(f"\nFull report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
