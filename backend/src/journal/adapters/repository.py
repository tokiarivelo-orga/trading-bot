"""Trade record persistence (sync SQLAlchemy; call via asyncio.to_thread)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.orm import Session, sessionmaker

from src.journal.adapters.orm import TradeRow
from src.journal.domain.models import (
    CandleSnapshot,
    OpenTradeExcursion,
    TradeAnalyticsRecord,
    TradeRecord,
)

Outcome = Literal["win", "loss", "breakeven", "open"]
OrderField = Literal["open_time", "close_time", "profit"]
_ORDER_COLUMNS: dict[OrderField, ColumnElement] = {
    "open_time": TradeRow.open_time,
    "close_time": TradeRow.close_time,
    "profit": TradeRow.profit,
}


class JournalRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, record: TradeRecord, account_id: str = "default") -> None:
        row = _to_row(record, account_id)
        with self._session_factory() as session:
            session.merge(row)
            session.commit()

    def get(self, trade_id: str) -> TradeRecord | None:
        with self._session_factory() as session:
            row = session.get(TradeRow, trade_id)
        return _to_domain(row) if row else None

    def get_by_id(self, trade_id: str, account_id: str = "default") -> TradeRecord | None:
        """Single trade scoped to `account_id` — unlike `get`, which looks up
        by primary key alone. Backs `GET .../trades/{trade_id}/decision-context`."""
        query = select(TradeRow).where(
            TradeRow.id == trade_id, TradeRow.account_id == account_id
        )
        with self._session_factory() as session:
            row = session.scalar(query)
        return _to_domain(row) if row else None

    def get_last_n(
        self, symbol: str, count: int, account_id: str = "default"
    ) -> list[TradeRecord]:
        query = (
            select(TradeRow)
            .where(TradeRow.symbol == symbol, TradeRow.account_id == account_id)
            .order_by(TradeRow.open_time.desc())
            .limit(count)
        )
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]

    def get_markers(
        self,
        symbol: str,
        frm: int | None = None,
        to: int | None = None,
        skill: str | None = None,
        limit: int = 1000,
        account_id: str = "default",
    ) -> list[TradeRecord]:
        """`skill`, when given, scopes markers to one bot's own trades — lets
        the chart show a single bot's positions in isolation instead of every
        trade (any bot, or manual) ever placed on the symbol. `limit` bounds
        the query to the most recent `limit` trades by open_time — the chart
        polls this every few seconds for the life of a session, so an
        unbounded scan here would grow with the account's entire trade
        history instead of staying constant."""
        query = (
            select(TradeRow)
            .where(TradeRow.symbol == symbol, TradeRow.account_id == account_id)
            .order_by(TradeRow.open_time.desc())
            .limit(limit)
        )
        if frm is not None:
            query = query.where(TradeRow.open_time >= frm)
        if to is not None:
            query = query.where(TradeRow.open_time <= to)
        if skill is not None:
            query = query.where(TradeRow.skill == skill)
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        # Re-ascending: callers (chart markers) expect oldest-first, same as
        # the previous unbounded `order_by(open_time)` query returned.
        return [_to_domain(row) for row in reversed(rows)]

    def get_all(self, account_id: str = "default") -> list[TradeRecord]:
        """Every trade (open and closed) for the account — used for the
        analytics dashboard's aggregation, which needs the full history to
        group by symbol/bot rather than a bounded recent window."""
        query = select(TradeRow).where(TradeRow.account_id == account_id)
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]

    def get_all_for_analytics(
        self,
        open_from: int | None = None,
        open_to: int | None = None,
        account_id: str = "default",
    ) -> list[TradeAnalyticsRecord]:
        """Same rows as `get_all`, but selects only the columns
        `domain/analytics.py`'s aggregation reads (id, symbol, volume,
        open/close time, profit, skill, strategy_version) instead of the
        whole row — SQLAlchemy never fetches or deserializes the four JSON
        snapshot/structure columns (`m5/h1_entry/exit_snapshot`, `structure`)
        this way. Backs `get_symbol_analytics`/`get_bot_analytics`, which
        never look at chart snapshots or swing structure."""
        query = select(
            TradeRow.id,
            TradeRow.symbol,
            TradeRow.volume,
            TradeRow.open_time,
            TradeRow.close_time,
            TradeRow.profit,
            TradeRow.skill,
            TradeRow.strategy_version,
            TradeRow.slippage,
            TradeRow.execution_latency_ms,
            TradeRow.broker_retcode,
            TradeRow.mfe,
            TradeRow.mfe_time,
            TradeRow.mae,
            TradeRow.mae_time,
            TradeRow.regime_volatility,
            TradeRow.regime_trend,
            TradeRow.regime_session,
            TradeRow.transaction_cost,
        ).where(TradeRow.account_id == account_id)
        if open_from is not None:
            query = query.where(TradeRow.open_time >= open_from)
        if open_to is not None:
            query = query.where(TradeRow.open_time <= open_to)
        with self._session_factory() as session:
            rows = session.execute(query).all()
        return [
            TradeAnalyticsRecord(
                id=row.id,
                symbol=row.symbol,
                volume=row.volume,
                open_time=datetime.fromtimestamp(row.open_time, tz=UTC),
                close_time=datetime.fromtimestamp(row.close_time, tz=UTC)
                if row.close_time
                else None,
                profit=row.profit,
                skill=row.skill,
                strategy_version=row.strategy_version,
                slippage=row.slippage,
                execution_latency_ms=row.execution_latency_ms,
                broker_retcode=row.broker_retcode,
                mfe=row.mfe,
                mfe_time=datetime.fromtimestamp(row.mfe_time, tz=UTC) if row.mfe_time else None,
                mae=row.mae,
                mae_time=datetime.fromtimestamp(row.mae_time, tz=UTC) if row.mae_time else None,
                regime_volatility=row.regime_volatility,
                regime_trend=row.regime_trend,
                regime_session=row.regime_session,
                transaction_cost=row.transaction_cost,
            )
            for row in rows
        ]

    def get_open(
        self, symbol: str | None = None, account_id: str = "default"
    ) -> list[TradeRecord]:
        """Trades journaled as opened but never journaled as closed —
        candidates for reconciliation on startup/reconnect (Phase 9)."""
        query = select(TradeRow).where(
            TradeRow.close_time.is_(None), TradeRow.account_id == account_id
        )
        if symbol is not None:
            query = query.where(TradeRow.symbol == symbol)
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]

    def get_open_excursions(
        self, symbol: str, account_id: str = "default"
    ) -> list[OpenTradeExcursion]:
        """Still-open trades on `symbol`, projected down to what the MFE/MAE
        accumulator reads (Phase 3). Runs once per closed candle per symbol,
        hence the narrow column list rather than `get_open`'s full rows."""
        query = select(
            TradeRow.id, TradeRow.side, TradeRow.open_price, TradeRow.mfe, TradeRow.mfe_time, TradeRow.mae, TradeRow.mae_time
        ).where(
            TradeRow.close_time.is_(None),
            TradeRow.symbol == symbol,
            TradeRow.account_id == account_id,
        )
        with self._session_factory() as session:
            rows = session.execute(query).all()
        return [
            OpenTradeExcursion(
                id=row.id, side=row.side, open_price=row.open_price, mfe=row.mfe, mfe_time=datetime.fromtimestamp(row.mfe_time, tz=UTC) if row.mfe_time else None, mae=row.mae, mae_time=datetime.fromtimestamp(row.mae_time, tz=UTC) if row.mae_time else None
            )
            for row in rows
        ]

    def update_excursion(
        self, trade_id: str, mfe: float, mae: float, mfe_time: datetime | None, mae_time: datetime | None, account_id: str = "default"
    ) -> None:
        """Writes just the two excursion columns of one trade.

        A targeted UPDATE rather than `save()`: the accumulator fires on every
        closed candle for every open position, and re-merging the whole row
        (four JSON snapshot columns included) that often would rewrite far
        more than it changes — and would race a concurrent close that is
        rewriting the same row's exit fields."""
        with self._session_factory() as session:
            session.execute(
                update(TradeRow)
                .where(TradeRow.id == trade_id, TradeRow.account_id == account_id)
                .values(mfe=mfe, mae=mae, mfe_time=int(mfe_time.timestamp()) if mfe_time else None, mae_time=int(mae_time.timestamp()) if mae_time else None)
            )
            session.commit()

    def get_last_n_closed(
        self, symbol: str, count: int, skill: str | None = None, account_id: str = "default"
    ) -> list[TradeRecord]:
        query = (
            select(TradeRow)
            .where(
                TradeRow.symbol == symbol,
                TradeRow.close_time.is_not(None),
                TradeRow.account_id == account_id,
            )
            .order_by(TradeRow.close_time.desc())
            .limit(count)
        )
        if skill is not None:
            query = query.where(TradeRow.skill == skill)
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]

    def count_closed(
        self, symbol: str, skill: str | None = None, account_id: str = "default"
    ) -> int:
        """`skill`, when given, scopes the count to one bot's trades on
        `symbol` — several bots trading the same symbol concurrently each
        get reviewed on their own 10-trade cadence, not a shared one."""
        query = (
            select(func.count())
            .select_from(TradeRow)
            .where(
                TradeRow.symbol == symbol,
                TradeRow.close_time.is_not(None),
                TradeRow.account_id == account_id,
            )
        )
        if skill is not None:
            query = query.where(TradeRow.skill == skill)
        with self._session_factory() as session:
            return session.scalar(query) or 0

    def search(
        self,
        *,
        symbol: str | None = None,
        side: str | None = None,
        strategy_version: str | None = None,
        skill: str | None = None,
        outcome: Outcome | None = None,
        open_from: int | None = None,
        open_to: int | None = None,
        close_from: int | None = None,
        close_to: int | None = None,
        order_by: OrderField = "open_time",
        order_dir: Literal["asc", "desc"] = "desc",
        limit: int = 50,
        offset: int = 0,
        account_id: str = "default",
    ) -> tuple[list[TradeRecord], int, float]:
        """Filterable, paginated trade history query (any symbol, any field
        combination) — backs `GET /journal/history`."""
        filters: list[ColumnElement] = [TradeRow.account_id == account_id]
        if symbol is not None:
            filters.append(TradeRow.symbol == symbol)
        if side is not None:
            filters.append(TradeRow.side == side)
        if strategy_version is not None:
            filters.append(TradeRow.strategy_version.contains(strategy_version))
        if skill is not None:
            filters.append(TradeRow.skill.contains(skill))
        if outcome == "open":
            filters.append(TradeRow.close_time.is_(None))
        elif outcome == "win":
            filters.extend([TradeRow.close_time.is_not(None), TradeRow.profit > 0])
        elif outcome == "loss":
            filters.extend([TradeRow.close_time.is_not(None), TradeRow.profit < 0])
        elif outcome == "breakeven":
            filters.extend([TradeRow.close_time.is_not(None), TradeRow.profit == 0])
        if open_from is not None:
            filters.append(TradeRow.open_time >= open_from)
        if open_to is not None:
            filters.append(TradeRow.open_time <= open_to)
        if close_from is not None:
            filters.append(TradeRow.close_time >= close_from)
        if close_to is not None:
            filters.append(TradeRow.close_time <= close_to)

        count_query = (
            select(func.count(), func.sum(TradeRow.profit)).select_from(TradeRow).where(*filters)
        )
        order_column = _ORDER_COLUMNS[order_by]
        order_clause = order_column.desc() if order_dir == "desc" else order_column.asc()
        page_query = (
            select(TradeRow).where(*filters).order_by(order_clause).limit(limit).offset(offset)
        )
        with self._session_factory() as session:
            res = session.execute(count_query).first()
            total = res[0] or 0 if res else 0
            total_profit = float(res[1] or 0.0) if res else 0.0
            rows = session.scalars(page_query).all()
        return [_to_domain(row) for row in rows], total, total_profit


def _snapshot_to_json(snapshot: tuple[CandleSnapshot, ...]) -> list[dict]:
    return [
        {
            "time": int(c.time.timestamp()),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
        }
        for c in snapshot
    ]


def _snapshot_from_json(data: list[dict] | None) -> tuple[CandleSnapshot, ...]:
    if not data:
        return ()
    return tuple(
        CandleSnapshot(
            time=datetime.fromtimestamp(d["time"], tz=UTC),
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            tick_volume=d["tick_volume"],
        )
        for d in data
    )


def _structure_to_json(structure: tuple[tuple[str, float, datetime], ...]) -> list[dict]:
    return [
        {"label": label, "price": price, "time": int(time.timestamp())}
        for label, price, time in structure
    ]


def _structure_from_json(data: list[dict] | None) -> tuple[tuple[str, float, datetime], ...]:
    if not data:
        return ()
    return tuple(
        (d["label"], d["price"], datetime.fromtimestamp(d["time"], tz=UTC)) for d in data
    )


def _indicators_to_json(
    indicators: tuple[tuple[str, float, float, str, bool], ...],
) -> list[dict]:
    return [
        {
            "name": name,
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "passed": passed,
        }
        for name, value, threshold, comparison, passed in indicators
    ]


def _indicators_from_json(
    data: list[dict] | None,
) -> tuple[tuple[str, float, float, str, bool], ...]:
    if not data:
        return ()
    return tuple(
        (d["name"], d["value"], d["threshold"], d["comparison"], d["passed"]) for d in data
    )


def _to_row(record: TradeRecord, account_id: str) -> TradeRow:
    return TradeRow(
        id=record.id,
        account_id=account_id,
        symbol=record.symbol,
        side=record.side,
        volume=record.volume,
        open_price=record.open_price,
        open_time=int(record.open_time.timestamp()),
        sl=record.sl,
        tp=record.tp,
        spread_points_at_entry=record.spread_points_at_entry,
        comment=record.comment,
        strategy_version=record.strategy_version,
        skill=record.skill,
        close_price=record.close_price,
        close_time=int(record.close_time.timestamp()) if record.close_time else None,
        profit=record.profit,
        close_reason=record.close_reason,
        m5_entry_snapshot=_snapshot_to_json(record.m5_entry_snapshot),
        h1_entry_snapshot=_snapshot_to_json(record.h1_entry_snapshot),
        m5_exit_snapshot=_snapshot_to_json(record.m5_exit_snapshot),
        h1_exit_snapshot=_snapshot_to_json(record.h1_exit_snapshot),
        reason=record.reason,
        confidence=record.confidence,
        zone_kind=record.zone_kind,
        zone_price_low=record.zone_price_low,
        zone_price_high=record.zone_price_high,
        zone_time_start=int(record.zone_time_start.timestamp())
        if record.zone_time_start
        else None,
        zone_time_end=int(record.zone_time_end.timestamp()) if record.zone_time_end else None,
        zone_pattern=record.zone_pattern,
        pattern=record.pattern,
        structure=_structure_to_json(record.structure),
        indicators=_indicators_to_json(record.indicators),
        requested_price=record.requested_price,
        slippage=record.slippage,
        execution_latency_ms=record.execution_latency_ms,
        broker_retcode=record.broker_retcode,
        mfe=record.mfe,
        mfe_time=int(record.mfe_time.timestamp()) if record.mfe_time else None,
        mae=record.mae,
        mae_time=int(record.mae_time.timestamp()) if record.mae_time else None,
        regime_volatility=record.regime_volatility,
        regime_volatility_percentile=record.regime_volatility_percentile,
        regime_trend=record.regime_trend,
        regime_adx=record.regime_adx,
        regime_session=record.regime_session,
        transaction_cost=record.transaction_cost,
    )


def _to_domain(row: TradeRow) -> TradeRecord:
    return TradeRecord(
        id=row.id,
        symbol=row.symbol,
        side=row.side,
        volume=row.volume,
        open_price=row.open_price,
        open_time=datetime.fromtimestamp(row.open_time, tz=UTC),
        sl=row.sl,
        tp=row.tp,
        spread_points_at_entry=row.spread_points_at_entry,
        comment=row.comment,
        strategy_version=row.strategy_version,
        skill=row.skill,
        close_price=row.close_price,
        close_time=datetime.fromtimestamp(row.close_time, tz=UTC) if row.close_time else None,
        profit=row.profit,
        close_reason=row.close_reason,
        m5_entry_snapshot=_snapshot_from_json(row.m5_entry_snapshot),
        h1_entry_snapshot=_snapshot_from_json(row.h1_entry_snapshot),
        m5_exit_snapshot=_snapshot_from_json(row.m5_exit_snapshot),
        h1_exit_snapshot=_snapshot_from_json(row.h1_exit_snapshot),
        reason=row.reason,
        confidence=row.confidence,
        zone_kind=row.zone_kind,
        zone_price_low=row.zone_price_low,
        zone_price_high=row.zone_price_high,
        zone_time_start=datetime.fromtimestamp(row.zone_time_start, tz=UTC)
        if row.zone_time_start
        else None,
        zone_time_end=datetime.fromtimestamp(row.zone_time_end, tz=UTC)
        if row.zone_time_end
        else None,
        zone_pattern=row.zone_pattern,
        pattern=row.pattern,
        structure=_structure_from_json(row.structure),
        indicators=_indicators_from_json(row.indicators),
        requested_price=row.requested_price,
        slippage=row.slippage,
        execution_latency_ms=row.execution_latency_ms,
        broker_retcode=row.broker_retcode,
        mfe=row.mfe,
        mfe_time=datetime.fromtimestamp(row.mfe_time, tz=UTC) if row.mfe_time else None,
        mae=row.mae,
        mae_time=datetime.fromtimestamp(row.mae_time, tz=UTC) if row.mae_time else None,
        regime_volatility=row.regime_volatility,
        regime_volatility_percentile=row.regime_volatility_percentile,
        regime_trend=row.regime_trend,
        regime_adx=row.regime_adx,
        regime_session=row.regime_session,
        transaction_cost=row.transaction_cost,
    )
