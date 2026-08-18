"""One-off standalone backtest runner — deliberately NOT the API job path.

Runs entirely as its own OS process (no FastAPI, no shared event loop), so it
cannot block the live dev-backend the way `POST /backtest/run` did on
2026-08-18 (13 queued jobs ran as `BackgroundTasks` on the same single-worker
uvicorn event loop the live `trade_loop` uses, stalling live position
management for hours). Reuses the exact same strategy-resolution logic the
API job path uses (`_resolve_strategy_name`/`_build_full_registry` from
`src.backtest.api.routes`) so a validated-but-not-active version id resolves
correctly, without going anywhere near the running server.

Usage (from backend/):
    uv run python -m scripts.run_isolated_backtest <strategy_id> <symbol> <period>
    uv run python -m scripts.run_isolated_backtest 0224533a-... XAUUSD 2026-05:2026-08
"""

from __future__ import annotations

import asyncio
import logging
import sys

from src.backtest.api.routes import _build_full_registry, _resolve_strategy_name
from src.backtest.application.period import InvalidPeriod
from src.backtest.application.run_backtest import NoHistoryError, NoSymbolSpecError, run_backtest
from src.backtest.reports.writer import render_summary, write_report
from src.shared.config.settings import Settings


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: run_isolated_backtest <strategy_id> <symbol> <period>", file=sys.stderr)
        return 2

    strategy_id, symbol, period = argv
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()

    try:
        strategy_name = _resolve_strategy_name(strategy_id, settings.database_url)
        preferred_version_id = None if strategy_id == "breakout_v1" else strategy_id
        registry = _build_full_registry(settings.database_url, preferred_version_id)
        report = asyncio.run(
            run_backtest(
                strategy_name,
                symbol,
                period,
                database_url=settings.database_url,
                strategy_source=registry,
            )
        )
    except (InvalidPeriod, NoHistoryError, NoSymbolSpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    path = write_report(report)
    print(render_summary(report))
    print(f"\nFull report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
