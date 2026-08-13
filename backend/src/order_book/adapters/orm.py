"""Order-book snapshot table — one row per signal that had market depth to
capture; symbols/brokers with no depth simply have no row (see
`OrderBookCaptureService`)."""

from __future__ import annotations

from sqlalchemy import JSON, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.db.base import Base


class OrderBookSnapshotRow(Base):
    __tablename__ = "order_book_snapshots"
    __table_args__ = (
        # `get_for_signal` always filters on both together — the natural
        # lookup key for "the snapshot captured for this signal on this
        # account".
        Index("ix_order_book_snapshots_account_signal", "account_id", "signal_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    captured_at: Mapped[int] = mapped_column(Integer)  # epoch seconds UTC
    # Flat list of [side, price, volume] triples — same convention as
    # `journal.adapters.orm.TradeRow.indicators` for tuple-shaped JSON.
    levels: Mapped[list] = mapped_column(JSON, default=list)
