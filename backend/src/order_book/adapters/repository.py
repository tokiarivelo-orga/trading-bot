"""Order-book snapshot persistence (sync SQLAlchemy; call via asyncio.to_thread)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.order_book.adapters.orm import OrderBookSnapshotRow
from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot


class OrderBookSnapshotRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(
        self, signal_id: str, snapshot: OrderBookSnapshot, account_id: str = "default"
    ) -> None:
        row = OrderBookSnapshotRow(
            account_id=account_id,
            signal_id=signal_id,
            symbol=snapshot.symbol,
            captured_at=int(snapshot.time.timestamp()),
            levels=_levels_to_json(snapshot.levels),
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()

    def get_for_signal(
        self, signal_id: str, account_id: str = "default"
    ) -> OrderBookSnapshot | None:
        query = select(OrderBookSnapshotRow).where(
            OrderBookSnapshotRow.signal_id == signal_id,
            OrderBookSnapshotRow.account_id == account_id,
        )
        with self._session_factory() as session:
            row = session.scalar(query)
        return _to_domain(row) if row else None


def _levels_to_json(levels: tuple[BookLevel, ...]) -> list[list]:
    return [[level.side.value, level.price, level.volume] for level in levels]


def _levels_from_json(data: list[list] | None) -> tuple[BookLevel, ...]:
    if not data:
        return ()
    return tuple(
        BookLevel(side=BookSide(side), price=price, volume=volume) for side, price, volume in data
    )


def _to_domain(row: OrderBookSnapshotRow) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=row.symbol,
        time=datetime.fromtimestamp(row.captured_at, tz=UTC),
        levels=_levels_from_json(row.levels),
    )
