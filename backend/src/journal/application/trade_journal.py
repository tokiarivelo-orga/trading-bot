"""Trade journal use cases (§6.8): record entries/exits with market-context
snapshots, serve chart markers and trade history, trigger the 10-trade review.

Subscribes to PositionOpened/PositionClosed on the event bus rather than being
called directly by the broker module — see §4 "event-driven core".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Literal

from src.journal.adapters.repository import JournalRepository, OrderField, Outcome
from src.journal.domain.analytics import (
    BotAnalytics,
    DailyPnl,
    RegimeAnalytics,
    SymbolAnalytics,
    compute_bot_analytics,
    compute_daily_pnl,
    compute_regime_analytics,
    compute_symbol_analytics,
)
from src.journal.domain.excursion import Excursion, extend_excursion, finalize_excursion
from src.journal.domain.models import TradeRecord
from src.journal.ports.market_context import MarketContextPort
from src.shared.events.bus import EventBus
from src.shared.events.definitions import (
    CandleClosed,
    PositionClosed,
    PositionOpened,
    TenTradesCompleted,
)

logger = logging.getLogger(__name__)

# Excursion is accumulated off one timeframe only. M5 is the engine's own
# clock (every symbol the engine trades publishes CandleClosed on it), so
# using it means the accumulator never needs a subscription the rest of the
# system doesn't already produce. A finer timeframe would only sharpen
# intra-bar precision; the exit price is folded in on close regardless.
EXCURSION_TIMEFRAME = "M5"


class TradeJournalService:
    def __init__(
        self,
        repository: JournalRepository,
        market_context: MarketContextPort,
        event_bus: EventBus,
        review_every_n_trades: int = 10,
        account_id: str = "default",
    ) -> None:
        self._repository = repository
        self._market_context = market_context
        self._event_bus = event_bus
        self._review_every_n_trades = review_every_n_trades
        self._account_id = account_id

    async def on_position_opened(self, event: PositionOpened) -> None:
        snapshot = await self._market_context.capture(event.symbol)
        record = TradeRecord(
            id=event.position_id,
            symbol=event.symbol,
            side=event.side,
            volume=event.volume,
            open_price=event.price,
            open_time=event.occurred_at,
            sl=event.sl,
            tp=event.tp,
            spread_points_at_entry=event.spread_points,
            comment=event.comment,
            strategy_version=event.strategy_version,
            skill=event.skill,
            m5_entry_snapshot=snapshot.m5,
            h1_entry_snapshot=snapshot.h1,
            reason=event.reason,
            confidence=event.confidence,
            zone_kind=event.zone_kind,
            zone_price_low=event.zone_price_low,
            zone_price_high=event.zone_price_high,
            zone_time_start=event.zone_time_start,
            zone_time_end=event.zone_time_end,
            zone_pattern=event.zone_pattern,
            pattern=event.pattern,
            structure=event.structure,
            indicators=event.indicators,
            requested_price=event.requested_price,
            slippage=event.slippage,
            execution_latency_ms=event.execution_latency_ms,
            broker_retcode=event.broker_retcode,
            transaction_cost=event.transaction_cost,
            regime_volatility=event.regime_volatility,
            regime_volatility_percentile=event.regime_volatility_percentile,
            regime_trend=event.regime_trend,
            regime_adx=event.regime_adx,
            regime_session=event.regime_session,
            signal_id=event.signal_id,
            # Excursion starts at zero, not None: the trade has now been
            # measured (the market simply hasn't moved yet), which is what
            # distinguishes it from a pre-Phase-3 row that never was.
            mfe=0.0,
            mae=0.0,
            mfe_time=event.occurred_at,
            mae_time=event.occurred_at,
        )
        await asyncio.to_thread(self._repository.save, record, self._account_id)
        logger.info(
            "trade journaled (open): id=%s %s %s %.2f lots @ %.5f "
            "requested=%s slippage=%s latency=%sms retcode=%s",
            record.id,
            event.side,
            event.symbol,
            event.volume,
            event.price,
            event.requested_price,
            event.slippage,
            event.execution_latency_ms,
            event.broker_retcode,
        )

    async def on_candle_closed(self, event: CandleClosed) -> None:
        """Extends every open trade's MFE/MAE with the candle that just
        closed (OBSERVABILITY_PLAN.md Phase 3).

        Only `EXCURSION_TIMEFRAME` bars count, so a symbol streaming several
        timeframes doesn't accumulate the same price action more than once —
        excursion is a running maximum, but double-counting would still make
        the work quadratic in the number of subscribed timeframes for no
        added information."""
        if event.timeframe != EXCURSION_TIMEFRAME:
            return
        open_trades = await asyncio.to_thread(
            self._repository.get_open_excursions, event.symbol, self._account_id
        )
        if not open_trades:
            return
        candle = await self._market_context.latest_candle(event.symbol, EXCURSION_TIMEFRAME)
        if candle is None:
            return
        for trade in open_trades:
            updated = extend_excursion(
                Excursion(
                    mfe=trade.mfe or 0.0,
                    mae=trade.mae or 0.0,
                    mfe_time=trade.mfe_time,
                    mae_time=trade.mae_time,
                ),
                side=trade.side,
                open_price=trade.open_price,
                high=candle.high,
                low=candle.low,
                time=candle.time,
            )
            await asyncio.to_thread(
                self._repository.update_excursion,
                trade.id,
                updated.mfe,
                updated.mae,
                updated.mfe_time,
                updated.mae_time,
                self._account_id,
            )

    async def on_position_closed(self, event: PositionClosed) -> None:
        existing = await asyncio.to_thread(self._repository.get, event.position_id)
        if existing is None:
            logger.warning(
                "position closed but no journaled trade found: id=%s symbol=%s",
                event.position_id,
                event.symbol,
            )
            return
        snapshot = await self._market_context.capture(event.symbol)
        # Fold the exit price into the excursion accumulated from closed
        # candles. This is also what gives a trade that opened and closed
        # inside a single candle real MFE/MAE numbers — no candle ever closed
        # during its life, so this is its only measurement.
        excursion = finalize_excursion(
            Excursion(
                mfe=existing.mfe or 0.0,
                mae=existing.mae or 0.0,
                mfe_time=existing.mfe_time,
                mae_time=existing.mae_time,
            ),
            side=existing.side,
            open_price=existing.open_price,
            close_price=event.close_price,
            time=event.occurred_at,
        )
        closed = replace(
            existing,
            close_price=event.close_price,
            close_time=event.occurred_at,
            profit=event.profit,
            close_reason=event.close_reason or None,
            m5_exit_snapshot=snapshot.m5,
            h1_exit_snapshot=snapshot.h1,
            mfe=excursion.mfe,
            mae=excursion.mae,
            mfe_time=excursion.mfe_time,
            mae_time=excursion.mae_time,
        )
        await asyncio.to_thread(self._repository.save, closed, self._account_id)
        logger.info(
            "trade journaled (close): id=%s %s profit=%.2f mfe=%.5f mae=%.5f",
            closed.id,
            closed.symbol,
            event.profit,
            excursion.mfe,
            excursion.mae,
        )

        if closed.skill is None:
            # No bot attributed this trade (manual/API-placed) — nothing to
            # review, and folding it into a shared per-symbol count would
            # misattribute it to whichever bot happens to run next.
            return
        closed_count = await asyncio.to_thread(
            self._repository.count_closed, event.symbol, closed.skill, self._account_id
        )
        if closed_count > 0 and closed_count % self._review_every_n_trades == 0:
            last_n = await asyncio.to_thread(
                self._repository.get_last_n_closed,
                event.symbol,
                self._review_every_n_trades,
                closed.skill,
                self._account_id,
            )
            logger.info(
                "%d trades completed for %s [%s] — triggering AI review",
                self._review_every_n_trades,
                event.symbol,
                closed.skill,
            )
            await self._event_bus.publish(
                TenTradesCompleted(
                    symbol=event.symbol,
                    skill=closed.skill,
                    trade_ids=tuple(t.id for t in reversed(last_n)),
                )
            )

    async def get_markers(
        self,
        symbol: str,
        frm: int | None = None,
        to: int | None = None,
        skill: str | None = None,
        limit: int = 1000,
    ) -> list[TradeRecord]:
        return await asyncio.to_thread(
            self._repository.get_markers, symbol, frm, to, skill, limit, self._account_id
        )

    async def get_last_n(self, symbol: str, count: int) -> list[TradeRecord]:
        return await asyncio.to_thread(
            self._repository.get_last_n, symbol, count, self._account_id
        )

    async def get_open_trades(self, symbol: str | None = None) -> list[TradeRecord]:
        return await asyncio.to_thread(self._repository.get_open, symbol, self._account_id)

    def get_trade(self, trade_id: str) -> TradeRecord | None:
        """Single trade by id, scoped to this account — backs the "why did
        the bot take this trade" decision-context endpoint. Sync (unlike the
        rest of this class's repository-backed methods): the route layer
        wraps this call in `asyncio.to_thread` itself rather than this
        method doing it internally."""
        return self._repository.get_by_id(trade_id, self._account_id)

    async def get_symbol_analytics(
        self, open_from: int | None = None, open_to: int | None = None
    ) -> list[SymbolAnalytics]:
        trades = await asyncio.to_thread(
            self._repository.get_all_for_analytics,
            open_from=open_from,
            open_to=open_to,
            account_id=self._account_id,
        )
        return compute_symbol_analytics(trades)

    async def get_bot_analytics(
        self, open_from: int | None = None, open_to: int | None = None
    ) -> list[BotAnalytics]:
        trades = await asyncio.to_thread(
            self._repository.get_all_for_analytics,
            open_from=open_from,
            open_to=open_to,
            account_id=self._account_id,
        )
        return compute_bot_analytics(trades)

    async def get_regime_analytics(
        self, open_from: int | None = None, open_to: int | None = None
    ) -> list[RegimeAnalytics]:
        trades = await asyncio.to_thread(
            self._repository.get_all_for_analytics,
            open_from=open_from,
            open_to=open_to,
            account_id=self._account_id,
        )
        return compute_regime_analytics(trades)

    async def get_daily_pnl(
        self, open_from: int | None = None, open_to: int | None = None
    ) -> list[DailyPnl]:
        trades = await asyncio.to_thread(
            self._repository.get_all_for_analytics,
            open_from=open_from,
            open_to=open_to,
            account_id=self._account_id,
        )
        return compute_daily_pnl(trades)

    async def search_trades(
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
    ) -> tuple[list[TradeRecord], int, float]:
        return await asyncio.to_thread(
            self._repository.search,
            symbol=symbol,
            side=side,
            strategy_version=strategy_version,
            skill=skill,
            outcome=outcome,
            open_from=open_from,
            open_to=open_to,
            close_from=close_from,
            close_to=close_to,
            order_by=order_by,
            order_dir=order_dir,
            limit=limit,
            offset=offset,
            account_id=self._account_id,
        )
