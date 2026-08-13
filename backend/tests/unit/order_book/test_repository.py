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
