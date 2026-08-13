"""Candle history table — backtests and AI snapshots read from here."""

from __future__ import annotations

from sqlalchemy import BigInteger, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.db.base import Base


class CandleRow(Base):
    __tablename__ = "candles"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(4), primary_key=True)
    # Bar open time as epoch seconds UTC — matches the wire format and
    # lightweight-charts, and keeps SQLite/PostgreSQL behavior identical.
    time: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    tick_volume: Mapped[int] = mapped_column(Integer)
    spread_points: Mapped[int] = mapped_column(Integer)
    # Nullable/go-forward-only: historical rows were written before MT5's
    # real_volume field was captured, and never had it on the wire.
    real_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Populated by a separate enrichment pass after the OHLC upsert (see
    # `candle_repository.py::enrich_missing`), not by the upsert itself —
    # ATR needs trailing history a single upsert batch doesn't carry. NULL
    # until that pass reaches this bar.
    atr_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SymbolSpecRow(Base):
    """Static broker facts snapshotted from the gateway's MT5 `symbol_info`
    at backfill time — lets backtests replay any symbol offline without a
    hand-authored `configs/symbols/<symbol>.yaml` (see
    `market_data/application/history.py::sync_symbol_spec` and
    `backtest/application/run_backtest.py`)."""

    __tablename__ = "symbol_specs"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    point: Mapped[float] = mapped_column(Float)
    digits: Mapped[int] = mapped_column(Integer)
    stops_level: Mapped[int] = mapped_column(Integer)
    contract_size: Mapped[float] = mapped_column(Float)
    volume_min: Mapped[float] = mapped_column(Float)
    volume_max: Mapped[float] = mapped_column(Float)
    volume_step: Mapped[float] = mapped_column(Float)
    # Epoch seconds UTC of the last successful sync — surfaced so the UI/logs
    # can flag a spec that hasn't been refreshed in a long time.
    updated_at: Mapped[int] = mapped_column(BigInteger)
