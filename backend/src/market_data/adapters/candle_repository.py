"""Persistence for candle history (sync SQLAlchemy; call via asyncio.to_thread)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import Row, bindparam, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session, sessionmaker

from src.market_data.adapters.orm import CandleRow
from src.market_data.domain.enrichment import compute_atr_series, day_of_week_for
from src.market_data.domain.models import Candle, Timeframe

# Read queries select these columns as plain tuples instead of materializing
# ORM `CandleRow` instances — candle reads are the backtest runner's single
# biggest fixed cost (hundreds of thousands of rows per run), and identity-map
# bookkeeping buys nothing for immutable history rows that are never updated
# through the session.
_CANDLE_COLUMNS = (
    CandleRow.time,
    CandleRow.open,
    CandleRow.high,
    CandleRow.low,
    CandleRow.close,
    CandleRow.tick_volume,
    CandleRow.spread_points,
    CandleRow.real_volume,
    CandleRow.atr_14,
    CandleRow.day_of_week,
)


class CandleRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def upsert_many(self, candles: Iterable[Candle], account_id: str = "default") -> int:
        """Insert or refresh bars (re-downloads and the forming bar overwrite).

        `atr_14`/`day_of_week` are always written NULL here, insert or
        conflict-update alike — enrichment needs trailing history a single
        upsert batch doesn't carry, so it's a separate write pass
        (`enrich_missing`) rather than something this method can compute
        itself. Resetting them to NULL on every upsert (rather than only on
        insert) is deliberate: a forming bar's OHLC changes on every poll,
        which changes its true range too, so a stale ATR would otherwise
        survive a bar whose high/low just moved. Every caller of
        `upsert_many` in this codebase follows up with `enrich_missing` for
        the same symbol/timeframe, which is what fills these back in."""
        rows = [
            {
                "account_id": account_id,
                "symbol": c.symbol,
                "timeframe": c.timeframe.value,
                "time": int(c.time.timestamp()),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "tick_volume": c.tick_volume,
                "spread_points": c.spread_points,
                "real_volume": c.real_volume,
                "atr_14": None,
                "day_of_week": None,
            }
            for c in candles
        ]
        if not rows:
            return 0
        # SQLite-dialect upsert; swap for postgresql.insert when the DB moves.
        statement = insert(CandleRow)
        statement = statement.on_conflict_do_update(
            index_elements=["account_id", "symbol", "timeframe", "time"],
            set_={
                col: statement.excluded[col]
                for col in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread_points",
                    "real_volume",
                    "atr_14",
                    "day_of_week",
                )
            },
        )
        with self._session_factory() as session:
            session.execute(statement, rows)
            session.commit()
        return len(rows)

    def get_latest(
        self, symbol: str, timeframe: Timeframe, count: int, account_id: str = "default"
    ) -> list[Candle]:
        """Most recent `count` stored bars, oldest first."""
        query = (
            select(*_CANDLE_COLUMNS)
            .where(
                CandleRow.symbol == symbol,
                CandleRow.timeframe == timeframe.value,
                CandleRow.account_id == account_id,
            )
            .order_by(CandleRow.time.desc())
            .limit(count)
        )
        with self._session_factory() as session:
            rows = session.execute(query).all()
        return _to_domain_many(symbol, timeframe, reversed(rows))

    def get_before(
        self,
        symbol: str,
        timeframe: Timeframe,
        before: datetime,
        count: int,
        account_id: str = "default",
    ) -> list[Candle]:
        """`count` stored bars with open time strictly before `before`, oldest
        first — for paging further back in history than `get_latest`."""
        query = (
            select(*_CANDLE_COLUMNS)
            .where(
                CandleRow.symbol == symbol,
                CandleRow.timeframe == timeframe.value,
                CandleRow.time < int(before.timestamp()),
                CandleRow.account_id == account_id,
            )
            .order_by(CandleRow.time.desc())
            .limit(count)
        )
        with self._session_factory() as session:
            rows = session.execute(query).all()
        return _to_domain_many(symbol, timeframe, reversed(rows))

    def get_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        account_id: str = "default",
    ) -> list[Candle]:
        """Stored bars with open time in `[start, end)`, oldest first — used
        by the backtest replay adapter to load a bounded historical window."""
        query = (
            select(*_CANDLE_COLUMNS)
            .where(
                CandleRow.symbol == symbol,
                CandleRow.timeframe == timeframe.value,
                CandleRow.time >= int(start.timestamp()),
                CandleRow.time < int(end.timestamp()),
                CandleRow.account_id == account_id,
            )
            .order_by(CandleRow.time.asc())
        )
        with self._session_factory() as session:
            rows = session.execute(query).all()
        return _to_domain_many(symbol, timeframe, rows)

    def enrich_missing(
        self,
        symbol: str,
        timeframe: Timeframe,
        account_id: str = "default",
        atr_period: int = 14,
        batch: int = 500,
    ) -> int:
        """Backfills `atr_14`/`day_of_week` for the oldest stretch of stored
        bars that don't have them yet. `upsert_many` always writes both NULL
        (see its docstring) since ATR needs trailing history a single upsert
        batch doesn't carry — this is the separate write pass that fills
        them back in, called after every `upsert_many` in `application/
        history.py::backfill` and `application/candle_stream.py::poll_once`.

        Finds the oldest row with `atr_14 IS NULL`, reads up to `atr_period`
        bars of real trailing context immediately before it (so the ATR
        rolling window isn't computed off a cold NaN warm-up) plus up to
        `batch` rows from the gap onward, computes ATR/day-of-week over that
        window, and writes back only the rows that were actually NULL in the
        gap — the context rows are read-only, and any row in the window that
        already has a value is left untouched. That makes this idempotent:
        calling it again once a symbol/timeframe is fully caught up (or once
        the leading bars of its history are too close to the start to ever
        have `atr_period` bars of context) returns 0 both times.

        Returns the number of rows actually updated."""
        with self._session_factory() as session:
            first = session.execute(
                select(CandleRow.time)
                .where(
                    CandleRow.account_id == account_id,
                    CandleRow.symbol == symbol,
                    CandleRow.timeframe == timeframe.value,
                    CandleRow.atr_14.is_(None),
                )
                .order_by(CandleRow.time.asc())
                .limit(1)
            ).scalar_one_or_none()
            if first is None:
                return 0

            context_times = [
                row[0]
                for row in session.execute(
                    select(CandleRow.time)
                    .where(
                        CandleRow.account_id == account_id,
                        CandleRow.symbol == symbol,
                        CandleRow.timeframe == timeframe.value,
                        CandleRow.time < first,
                    )
                    .order_by(CandleRow.time.desc())
                    .limit(atr_period)
                ).all()
            ]
            window_start = min(context_times) if context_times else first

            window = session.execute(
                select(
                    CandleRow.time,
                    CandleRow.high,
                    CandleRow.low,
                    CandleRow.close,
                    CandleRow.atr_14,
                )
                .where(
                    CandleRow.account_id == account_id,
                    CandleRow.symbol == symbol,
                    CandleRow.timeframe == timeframe.value,
                    CandleRow.time >= window_start,
                )
                .order_by(CandleRow.time.asc())
                .limit(len(context_times) + batch)
            ).all()
            if not window:
                return 0

            highs = np.array([row.high for row in window], dtype=float)
            lows = np.array([row.low for row in window], dtype=float)
            closes = np.array([row.close for row in window], dtype=float)
            atr_values = compute_atr_series(highs, lows, closes, atr_period)

            updates = [
                {
                    "b_time": row.time,
                    "atr_14": float(atr_value),
                    "day_of_week": day_of_week_for(datetime.fromtimestamp(row.time, tz=UTC)),
                }
                for row, atr_value in zip(window, atr_values, strict=True)
                # Skip warm-up context rows (before the gap), rows already
                # enriched (never overwrite), and any bar too early in its
                # own history to have `atr_period` bars of trailing context
                # (rolling ATR is still NaN there).
                if row.time >= first and row.atr_14 is None and not np.isnan(atr_value)
            ]
            if not updates:
                return 0

            # Core (table-level, not ORM-mapped-class) `update` — this
            # repository never materializes `CandleRow` ORM instances on
            # reads (see `_CANDLE_COLUMNS`'s docstring), and executing an
            # executemany-style bulk update against the mapped class itself
            # would push SQLAlchemy's "ORM bulk UPDATE by primary key"
            # strategy, which requires every dict to carry all four PK
            # columns rather than the plain WHERE-by-time match used here.
            statement = (
                update(CandleRow.__table__)
                .where(
                    CandleRow.account_id == account_id,
                    CandleRow.symbol == symbol,
                    CandleRow.timeframe == timeframe.value,
                    CandleRow.time == bindparam("b_time"),
                )
                .values(atr_14=bindparam("atr_14"), day_of_week=bindparam("day_of_week"))
            )
            session.execute(statement, updates)
            session.commit()
            return len(updates)


def _to_domain_many(symbol: str, timeframe: Timeframe, rows: Iterable[Row]) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            time=datetime.fromtimestamp(time, tz=UTC),
            open=open_,
            high=high,
            low=low,
            close=close,
            tick_volume=tick_volume,
            spread_points=spread_points,
            # NULL for rows stored before real_volume existed on the wire.
            real_volume=real_volume if real_volume is not None else 0,
            # NULL until the enrichment pass reaches this bar.
            atr_14=atr_14,
            day_of_week=day_of_week,
        )
        for (
            time,
            open_,
            high,
            low,
            close,
            tick_volume,
            spread_points,
            real_volume,
            atr_14,
            day_of_week,
        ) in rows
    ]
