"""`NewsEventRepository` (Phase 6 Part B): idempotent upsert, date-range +
impact filtering, and balance stamping (Phase 6 Part C)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.news.adapters.repository import NewsEventRepository
from src.news.domain.models import ImpactLevel, NewsEvent
from src.shared.db.base import Base


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path) -> NewsEventRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return NewsEventRepository(sessionmaker(bind=engine, expire_on_commit=False))


def make_event(name="Non-Farm Payrolls", time=None, impact=ImpactLevel.HIGH, **kw) -> NewsEvent:
    defaults = dict(name=name, time=time or utc(2026, 8, 14, 12, 30), impact=impact, currency="USD")
    return NewsEvent(**{**defaults, **kw})


# ── save_many / list_between ────────────────────────────────────────────────


def test_list_between_empty_when_nothing_persisted(repository):
    assert repository.list_between(utc(2026, 8, 1), utc(2026, 8, 31)) == []


def test_save_many_and_list_between_roundtrip(repository):
    event = make_event(forecast="8.5%", previous="8.0%", actual=None)
    repository.save_many([event])

    results = repository.list_between(utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59))

    assert len(results) == 1
    stored = results[0]
    assert stored.name == "Non-Farm Payrolls"
    assert stored.time == utc(2026, 8, 14, 12, 30)
    assert stored.impact == ImpactLevel.HIGH
    assert stored.currency == "USD"
    assert stored.forecast == "8.5%"
    assert stored.previous == "8.0%"
    assert stored.actual is None


def test_save_many_upserts_idempotently_on_name_and_time(repository):
    """The calendar re-fetches the same upcoming events on every refresh —
    calling `save_many` twice with overlapping events must not duplicate
    rows, and the second call's fresher fields (e.g. a released `actual`
    value) should win."""
    event_v1 = make_event(actual=None)
    event_v2 = make_event(actual="8.7%")  # same (name, time), released since

    repository.save_many([event_v1])
    repository.save_many([event_v1, event_v2])  # overlapping batch

    results = repository.list_between(utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59))

    assert len(results) == 1
    assert results[0].actual == "8.7%"


def test_save_many_with_empty_list_is_a_no_op(repository):
    repository.save_many([])
    assert repository.list_between(utc(2026, 1, 1), utc(2026, 12, 31)) == []


def test_list_between_filters_by_impact(repository):
    repository.save_many(
        [
            make_event(name="High Impact Release", impact=ImpactLevel.HIGH),
            make_event(
                name="Low Impact Release", time=utc(2026, 8, 14, 13, 0), impact=ImpactLevel.LOW
            ),
        ]
    )

    high_only = repository.list_between(
        utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59), impact=ImpactLevel.HIGH
    )

    assert [e.name for e in high_only] == ["High Impact Release"]


def test_list_between_excludes_events_outside_the_range(repository):
    repository.save_many(
        [
            make_event(name="Inside", time=utc(2026, 8, 14, 12, 0)),
            make_event(name="Before", time=utc(2026, 8, 1, 12, 0)),
            make_event(name="After", time=utc(2026, 9, 1, 12, 0)),
        ]
    )

    results = repository.list_between(utc(2026, 8, 10), utc(2026, 8, 20))

    assert [e.name for e in results] == ["Inside"]


def test_list_between_sorted_oldest_first(repository):
    repository.save_many(
        [
            make_event(name="Later", time=utc(2026, 8, 14, 14, 0)),
            make_event(name="Earlier", time=utc(2026, 8, 14, 8, 0)),
        ]
    )

    results = repository.list_between(utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59))

    assert [e.name for e in results] == ["Earlier", "Later"]


# ── list_records_between (id + balance) ─────────────────────────────────────


def test_list_records_between_carries_id_and_null_balances_by_default(repository):
    repository.save_many([make_event()])

    records = repository.list_records_between(utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59))

    assert len(records) == 1
    row_id, event, balance_before, balance_after = records[0]
    assert isinstance(row_id, int)
    assert event.name == "Non-Farm Payrolls"
    assert balance_before is None
    assert balance_after is None


# ── set_balance_before / set_balance_after ──────────────────────────────────


def test_set_balance_before_and_after_stamp_the_matching_row(repository):
    event_time = utc(2026, 8, 14, 12, 30)
    repository.save_many([make_event(time=event_time)])

    repository.set_balance_before("Non-Farm Payrolls", event_time, 10_000.0)
    repository.set_balance_after("Non-Farm Payrolls", event_time, 10_050.0)

    records = repository.list_records_between(utc(2026, 8, 14, 0, 0), utc(2026, 8, 14, 23, 59))
    _, _, balance_before, balance_after = records[0]
    assert balance_before == 10_000.0
    assert balance_after == 10_050.0


def test_set_balance_on_unpersisted_row_is_a_no_op(repository):
    """The event hasn't been saved yet (e.g. a window fired between two
    refreshes) — stamping must not raise or create a row."""
    repository.set_balance_before("Never Persisted", utc(2026, 8, 14, 12, 30), 10_000.0)

    assert repository.list_between(utc(2026, 1, 1), utc(2026, 12, 31)) == []
