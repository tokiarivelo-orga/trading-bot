from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.order_book.adapters.repository import OrderBookSnapshotRepository
from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot
from src.shared.db.base import Base


@pytest.fixture
def repository(tmp_path) -> OrderBookSnapshotRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return OrderBookSnapshotRepository(sessionmaker(bind=engine, expire_on_commit=False))


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_get_for_signal_returns_none_when_never_captured(repository):
    assert repository.get_for_signal("sig-1") is None


def test_save_and_get_roundtrip(repository):
    snapshot = OrderBookSnapshot(
        symbol="XAUUSD",
        time=utc(2026, 8, 13, 12, 0),
        levels=(
            BookLevel(side=BookSide.BID, price=2400.10, volume=5.5),
            BookLevel(side=BookSide.ASK, price=2400.35, volume=3.0),
        ),
    )

    repository.save("sig-1", snapshot)
    stored = repository.get_for_signal("sig-1")

    assert stored is not None
    assert stored.symbol == "XAUUSD"
    assert stored.time == utc(2026, 8, 13, 12, 0)
    assert stored.levels == snapshot.levels


def test_get_for_signal_is_scoped_to_account(repository):
    snapshot = OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 12, 0))
    repository.save("sig-1", snapshot, account_id="ftmo-1")

    assert repository.get_for_signal("sig-1", account_id="ftmo-1") is not None
    assert repository.get_for_signal("sig-1", account_id="ftmo-2") is None
    assert repository.get_for_signal("sig-1", account_id="default") is None


def test_get_for_signal_does_not_match_other_signals(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 12, 0)))
    assert repository.get_for_signal("sig-2") is None


# ── list_for_account (bulk export) ──────────────────────────────────────────


def test_list_for_account_returns_empty_list_when_nothing_captured(repository):
    assert repository.list_for_account("default") == []


def test_list_for_account_returns_all_snapshots_newest_first(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 10, 0)))
    repository.save("sig-2", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 12, 0)))
    repository.save("sig-3", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 11, 0)))

    rows = repository.list_for_account("default")

    assert [signal_id for signal_id, _ in rows] == ["sig-2", "sig-3", "sig-1"]


def test_list_for_account_filters_by_symbol(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 10, 0)))
    repository.save("sig-2", OrderBookSnapshot(symbol="VIX75", time=utc(2026, 8, 13, 11, 0)))

    rows = repository.list_for_account("default", symbol="XAUUSD")

    assert [signal_id for signal_id, _ in rows] == ["sig-1"]


def test_list_for_account_filters_by_since_and_until(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 9, 0)))
    repository.save("sig-2", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 10, 0)))
    repository.save("sig-3", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 11, 0)))

    since_only = repository.list_for_account("default", since=utc(2026, 8, 13, 10, 0))
    assert {signal_id for signal_id, _ in since_only} == {"sig-2", "sig-3"}

    until_only = repository.list_for_account("default", until=utc(2026, 8, 13, 10, 0))
    assert {signal_id for signal_id, _ in until_only} == {"sig-1", "sig-2"}

    windowed = repository.list_for_account(
        "default", since=utc(2026, 8, 13, 10, 0), until=utc(2026, 8, 13, 10, 0)
    )
    assert [signal_id for signal_id, _ in windowed] == ["sig-2"]


def test_list_for_account_is_scoped_to_account(repository):
    snapshot = OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 12, 0))
    repository.save("sig-1", snapshot, account_id="ftmo-1")
    repository.save("sig-2", snapshot, account_id="ftmo-2")

    assert [s for s, _ in repository.list_for_account("ftmo-1")] == ["sig-1"]
    assert [s for s, _ in repository.list_for_account("ftmo-2")] == ["sig-2"]
    assert repository.list_for_account("default") == []
