"""Trade journal table — F5/F7 source of truth."""

from __future__ import annotations

from sqlalchemy import JSON, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.db.base import Base


class TradeRow(Base):
    __tablename__ = "trades"
    __table_args__ = (
        # Every hot query (`get_last_n`, `get_markers`, `get_open`, `count_closed`,
        # `search`) filters on account_id AND symbol together; the previous
        # single-column indexes on each let SQLite use only one and row-filter
        # the rest. This composite index (with close_time trailing, since several
        # of those queries also filter/order on it) serves the whole filter shape.
        Index("ix_trades_account_symbol_close", "account_id", "symbol", "close_time"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    volume: Mapped[float] = mapped_column(Float)
    open_price: Mapped[float] = mapped_column(Float)
    open_time: Mapped[int] = mapped_column(Integer)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_points_at_entry: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(String(255), default="")
    strategy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    skill: Mapped[str | None] = mapped_column(String(64), nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    m5_entry_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    h1_entry_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    m5_exit_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    h1_exit_snapshot: Mapped[list] = mapped_column(JSON, default=list)
    reason: Mapped[str] = mapped_column(String(500), default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    zone_price_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_price_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_time_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zone_time_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zone_pattern: Mapped[str | None] = mapped_column(String(16), nullable=True)
    pattern: Mapped[str | None] = mapped_column(String(64), nullable=True)
    structure: Mapped[list] = mapped_column(JSON, default=list)
    indicators: Mapped[list] = mapped_column(JSON, default=list)
    # Execution telemetry + excursion (OBSERVABILITY_PLAN.md Phase 3).
    # Nullable throughout: every trade journaled before Phase 3 has no such
    # measurement, and "unknown" must stay distinguishable from "zero
    # slippage" / "never moved" in the analytics averages.
    requested_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    broker_retcode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mfe: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Regime tagging + transaction cost (OBSERVABILITY_PLAN.md Phase 6).
    # Nullable throughout: every trade journaled before Phase 6 has no such
    # measurement.
    regime_volatility: Mapped[str | None] = mapped_column(String(16), nullable=True)
    regime_volatility_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_trend: Mapped[str | None] = mapped_column(String(16), nullable=True)
    regime_adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_session: Mapped[str | None] = mapped_column(String(16), nullable=True)
    transaction_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
