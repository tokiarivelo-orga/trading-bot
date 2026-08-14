"""Persistence for the economic-calendar event history (`news_events`
table; sync SQLAlchemy, call via `asyncio.to_thread` — same convention as
`market_data.adapters.candle_repository`/`order_book.adapters.repository`).

No `account_id` dimension: `NewsWindowService` (this module's one
application-layer instance) is process-wide, not per-account — one
calendar, shared across every account's engine (see
`news/application/news_window_service.py`'s module docstring and
`container.py`'s `_FanOutEventBus`) — so the calendar it persists has no
per-account split either.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session, sessionmaker

from src.news.adapters.orm import NewsEventRow
from src.news.domain.models import ImpactLevel, NewsEvent

# One persisted row, with the two fields `NewsEvent` (the pure calendar-fact
# domain type) has no room for: the row's own id, and the Phase 6 Part C
# balance snapshots. `(id, event, balance_before, balance_after)` — same
# "tuple alongside the domain object" convention
# `order_book.adapters.repository.list_for_account` already uses for its own
# id-not-on-the-domain-type case.
NewsEventRecord = tuple[int, NewsEvent, float | None, float | None]


class NewsEventRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_many(self, events: Iterable[NewsEvent]) -> None:
        """Idempotent upsert keyed on `(name, time)`.

        `NewsWindowService.refresh()` re-fetches the same upcoming events on
        every poll (calendars change rarely — see its docstring), so this
        must not insert a duplicate row each cycle. `balance_before`/
        `balance_after` are deliberately left out of the conflict `set_`
        clause: they're written later, one at a time, by
        `set_balance_before`/`set_balance_after` as `NewsWindowEntered`/
        `NewsWindowExited` fire, and a later calendar refresh re-upserting
        the same event must not clobber them back to NULL.
        """
        rows = [
            {
                "name": e.name,
                "time": int(e.time.timestamp()),
                "impact": e.impact.value,
                "currency": e.currency,
                "forecast": e.forecast,
                "previous": e.previous,
                "actual": e.actual,
            }
            for e in events
        ]
        if not rows:
            return
        # SQLite-dialect upsert; swap for postgresql.insert when the DB moves
        # (same note as `market_data.adapters.candle_repository.upsert_many`).
        statement = insert(NewsEventRow)
        statement = statement.on_conflict_do_update(
            index_elements=["name", "time"],
            set_={
                col: statement.excluded[col]
                for col in ("impact", "currency", "forecast", "previous", "actual")
            },
        )
        with self._session_factory() as session:
            session.execute(statement, rows)
            session.commit()

    def list_between(
        self, start: datetime, end: datetime, impact: ImpactLevel | None = None
    ) -> list[NewsEvent]:
        """Persisted events with scheduled time in `[start, end]`
        (inclusive), oldest first, optionally filtered to one `impact`
        level. Pure domain objects — no row id or balance, see
        `list_records_between` for the richer shape the HTTP API uses."""
        with self._session_factory() as session:
            rows = session.scalars(self._query(start, end, impact)).all()
        return [_to_domain(row) for row in rows]

    def list_records_between(
        self, start: datetime, end: datetime, impact: ImpactLevel | None = None
    ) -> list[NewsEventRecord]:
        """Same filter as `list_between`, but each entry also carries the
        row's `id` and `balance_before`/`balance_after` (Phase 6 Part C) —
        what `GET /accounts/{account_id}/news/events` actually renders."""
        with self._session_factory() as session:
            rows = session.scalars(self._query(start, end, impact)).all()
        return [(row.id, _to_domain(row), row.balance_before, row.balance_after) for row in rows]

    def set_balance_before(self, name: str, time: datetime, balance: float) -> None:
        """Stamps the account balance captured on `NewsWindowEntered` for
        the `(name, time)` event row. A no-op (not an error) when that row
        isn't persisted yet — the next `refresh()` upsert will insert it,
        permanently unstamped, which is acceptable (enrichment, not core
        data; see module/orm docstrings)."""
        self._set_balance(name, time, balance_before=balance)

    def set_balance_after(self, name: str, time: datetime, balance: float) -> None:
        """Same as `set_balance_before`, for `NewsWindowExited`."""
        self._set_balance(name, time, balance_after=balance)

    def _query(self, start: datetime, end: datetime, impact: ImpactLevel | None):
        query = select(NewsEventRow).where(
            NewsEventRow.time >= int(start.timestamp()),
            NewsEventRow.time <= int(end.timestamp()),
        )
        if impact is not None:
            query = query.where(NewsEventRow.impact == impact.value)
        return query.order_by(NewsEventRow.time.asc())

    def _set_balance(
        self,
        name: str,
        time: datetime,
        *,
        balance_before: float | None = None,
        balance_after: float | None = None,
    ) -> None:
        with self._session_factory() as session:
            row = session.scalar(
                select(NewsEventRow).where(
                    NewsEventRow.name == name, NewsEventRow.time == int(time.timestamp())
                )
            )
            if row is None:
                return
            if balance_before is not None:
                row.balance_before = balance_before
            if balance_after is not None:
                row.balance_after = balance_after
            session.commit()


def _to_domain(row: NewsEventRow) -> NewsEvent:
    return NewsEvent(
        name=row.name,
        time=datetime.fromtimestamp(row.time, tz=UTC),
        impact=ImpactLevel(row.impact),
        currency=row.currency,
        forecast=row.forecast,
        previous=row.previous,
        actual=row.actual,
    )
