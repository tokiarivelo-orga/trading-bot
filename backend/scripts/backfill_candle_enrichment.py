"""One-off backfill: populate `atr_14`/`day_of_week` on every existing candle
row for an account (see migration `c82842a85999_add_atr_and_day_of_week_to_
candles.py`, which adds the columns nullable and defers population to this
script).

Run from `backend/`:

    uv run python -m scripts.backfill_candle_enrichment --account default

Safely re-runnable: `CandleRepository.enrich_missing` is idempotent (it only
ever writes rows that are still NULL), so re-running this script after it has
already caught a symbol/timeframe up is a cheap no-op — useful both for a
first-time backfill of existing history and for periodically catching up any
symbol/timeframe whose enrichment fell behind (e.g. a `poll_once`/`backfill`
call site whose own single enrich pass didn't clear the whole backlog in one
go).
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.orm import CandleRow
from src.market_data.domain.models import Timeframe
from src.shared.config.settings import Settings
from src.shared.db.base import make_session_factory

logger = logging.getLogger(__name__)


def distinct_symbol_timeframes(
    session_factory, account_id: str
) -> list[tuple[str, Timeframe]]:
    """Every `(symbol, timeframe)` combination stored for `account_id`."""
    query = (
        select(CandleRow.symbol, CandleRow.timeframe)
        .where(CandleRow.account_id == account_id)
        .distinct()
    )
    with session_factory() as session:
        rows = session.execute(query).all()
    return [(symbol, Timeframe(timeframe)) for symbol, timeframe in rows]


def backfill_account(repository: CandleRepository, session_factory, account_id: str) -> int:
    """Runs `enrich_missing` to exhaustion for every symbol/timeframe stored
    under `account_id`. Returns the total number of rows enriched."""
    pairs = distinct_symbol_timeframes(session_factory, account_id)
    logger.info("account %s: %d symbol/timeframe combination(s) to check", account_id, len(pairs))

    total = 0
    for symbol, timeframe in pairs:
        while True:
            updated = repository.enrich_missing(symbol, timeframe, account_id=account_id)
            if updated == 0:
                break
            total += updated
            logger.info(
                "account=%s symbol=%s timeframe=%s: enriched %d row(s)",
                account_id,
                symbol,
                timeframe.value,
                updated,
            )
    logger.info("account %s: done, %d row(s) enriched total", account_id, total)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account", default="default", help="Account id to backfill (default: 'default')."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings()
    session_factory = make_session_factory(settings.database_url)
    repository = CandleRepository(session_factory)

    backfill_account(repository, session_factory, args.account)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
