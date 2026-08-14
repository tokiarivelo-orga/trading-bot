"""Persisted economic-calendar events (`news_events` table) — Phase 6 Part B.

`NewsWindowService._events` (see its module docstring) is deliberately
in-memory only, for the zero-I/O hot path `active_window_for()` needs — but
that also means a restart loses all calendar history, and nothing else in
this codebase can reconstruct it. This table exists specifically to stop
losing it: `NewsEventRepository.save_many` upserts every fetched batch here
in addition to (never instead of) the in-memory cache.

`balance_before`/`balance_after` (Phase 6 Part C) are enrichment, not core
calendar data: the account balance immediately before/after this event's
news window opened/closed, stamped by `NewsWindowService` when it publishes
`NewsWindowEntered`/`NewsWindowExited`. Both stay null until stamped, and
`balance_after` can stay null forever if the window never cleanly exits
(e.g. a process restart mid-window) — that is expected, not a data-quality
bug.
"""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.db.base import Base


class NewsEventRow(Base):
    __tablename__ = "news_events"
    __table_args__ = (
        # (name, time) is the calendar's natural dedup key — the same
        # upcoming event reappears on every `NewsWindowService.refresh()`
        # poll until it's released, and this is the conflict target
        # `save_many`'s upsert targets so re-fetching never duplicates rows.
        Index("ix_news_events_name_time", "name", "time", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    time: Mapped[int] = mapped_column(Integer, index=True)  # epoch seconds UTC
    impact: Mapped[str] = mapped_column(String(16), index=True)  # ImpactLevel value
    currency: Mapped[str] = mapped_column(String(16), default="")
    forecast: Mapped[str | None] = mapped_column(String(64), nullable=True)
    previous: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actual: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Phase 6 Part C — see module docstring.
    balance_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    balance_after: Mapped[float | None] = mapped_column(Float, nullable=True)
