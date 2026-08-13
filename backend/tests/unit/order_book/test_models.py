from datetime import UTC, datetime

from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot


def test_book_side_values():
    assert BookSide.BID == "bid"
    assert BookSide.ASK == "ask"


def test_snapshot_defaults_to_no_levels():
    snapshot = OrderBookSnapshot(symbol="XAUUSD", time=datetime.now(UTC))
    assert snapshot.levels == ()


def test_snapshot_holds_levels():
    level = BookLevel(side=BookSide.BID, price=2400.10, volume=5.5)
    snapshot = OrderBookSnapshot(symbol="XAUUSD", time=datetime.now(UTC), levels=(level,))
    assert snapshot.levels == (level,)
    assert snapshot.levels[0].side is BookSide.BID
