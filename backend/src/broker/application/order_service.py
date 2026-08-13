"""Order use cases: pre-trade spread/RR gate, fill, publish position events.

The engine (Phase 4) will call `open_position`/`close_position` from the trade
loop; for now these are also reachable manually via the broker API so the
plumbing can be exercised end-to-end before the engine exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.activity.domain.models import DecisionCheck
from src.activity.ports.signal_decisions import SignalDecisionSinkPort
from src.broker.application.spread_gate import DEFAULT_MIN_RR, SpreadGate
from src.broker.domain.trading import (
    ExecutionResult,
    OrderRejected,
    OrderRequest,
    OrderType,
    PendingOrder,
    PendingOrderRequest,
    Position,
    Side,
    execution_slippage,
)
from src.broker.ports.trading import BrokerPort
from src.market_data.ports.market_data import MarketDataPort
from src.shared.events.bus import EventBus
from src.shared.events.definitions import PositionClosed, PositionOpened
from src.shared.metrics.registry import record_signal_outcome

logger = logging.getLogger(__name__)

# MT5 TRADE_RETCODE_DONE — the "the deal went through" code. Used only as the
# threshold a recorded `broker_retcode` check is compared against, so the
# decision trail shows what the code should have been.
_RETCODE_DONE = 10009


class OrderService:
    def __init__(
        self,
        broker: BrokerPort,
        market_data: MarketDataPort,
        spread_gate: SpreadGate,
        event_bus: EventBus,
        signal_decisions: SignalDecisionSinkPort | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._broker = broker
        self._market_data = market_data
        self._spread_gate = spread_gate
        self._event_bus = event_bus
        # Injected so execution-latency measurement (Phase 3) is deterministic
        # under test; production passes wall-clock UTC.
        self._clock = clock
        # Typed decision trail (OBSERVABILITY_PLAN.md Phase 1) — used only to
        # stamp the outcome of a `signal_id` the engine already recorded.
        # Optional: manual/API orders carry no signal_id, and the backtest
        # runner wires no sink at all.
        self._signal_decisions = signal_decisions

    async def open_position(
        self,
        symbol: str,
        side: Side,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
        strategy_version: str | None = None,
        skill: str | None = None,
        magic: int = 0,
        max_spread_points: int | None = None,
        reason: str = "",
        confidence: float | None = None,
        zone_kind: str | None = None,
        zone_price_low: float | None = None,
        zone_price_high: float | None = None,
        zone_time_start: datetime | None = None,
        zone_time_end: datetime | None = None,
        zone_pattern: str | None = None,
        pattern: str | None = None,
        structure: tuple[tuple[str, float, datetime], ...] = (),
        indicators: tuple[tuple[str, float, float, str, bool], ...] = (),
        signal_id: str | None = None,
        signal_emitted_at: datetime | None = None,
        regime_volatility: str | None = None,
        regime_volatility_percentile: float | None = None,
        regime_trend: str | None = None,
        regime_adx: float | None = None,
        regime_session: str | None = None,
    ) -> ExecutionResult:
        """`max_spread_points`, when set, overrides the symbol's configured
        cap for this order only — used by news skills to widen (or, in
        principle, tighten) the allowance during a post-event window
        (`SkillDecision.max_spread_points`, §6.6). `magic`, when set, is the
        MT5 magic number identifying which bot placed the order — lets
        several bots trading the same symbol be told apart on open
        positions (`SkillDecision.magic`, §6.6); 0 for manual/API orders.
        `reason`/`confidence`/`zone_*`/`pattern`/`structure`/`indicators` are
        optional decision-context passthrough from the strategy's Signal (see
        strategies/domain/models.py) — kept as flat primitives here rather
        than importing that module's domain types, so the broker layer stays
        independent of the strategies module; they flow straight into the
        published `PositionOpened` event unused by order placement itself.
        `indicators` is `(name, value, threshold, comparison, passed)` tuples,
        flattened from `Signal.indicators` (`IndicatorReading`).
        `signal_id`, when set, is the `SignalDecision` the engine recorded for
        the signal this order came from — this method stamps that decision's
        terminal outcome (`opened` / `spread_veto` / `broker_rejected`).
        `None` for manual/API orders, which have no signal behind them.
        `signal_emitted_at` is that same `SignalDecision`'s `created_at` — the
        instant the strategy emitted the signal — passed down explicitly so
        measuring execution latency costs no database read on the order path;
        it is the emit end of the `signal_id` decision's signal→fill span
        (OBSERVABILITY_PLAN.md Phase 3). `regime_volatility`/
        `regime_volatility_percentile`/`regime_trend`/`regime_adx`/
        `regime_session` are the market-regime snapshot the engine computed
        at signal time (`engine.domain.regime.compute_entry_regime`,
        OBSERVABILITY_PLAN.md Phase 6) — passthrough only, like the
        `zone_*`/`pattern` fields above, straight into `PositionOpened`."""
        info = await self._market_data.get_symbol_info(symbol)
        reference_price = info.ask if side is Side.BUY else info.bid
        sl_distance = abs(reference_price - sl) if sl is not None else None
        tp_distance = abs(tp - reference_price) if tp is not None else None

        veto = self._spread_gate.check(
            symbol,
            info.spread_points,
            info.point,
            sl_distance,
            tp_distance,
            max_spread_override=max_spread_points,
        )
        if veto is not None:
            logger.info(
                "ENTRY REJECTED (spread/RR gate): %s %s spread=%dpts sl=%s tp=%s "
                "strategy=%s skill=%s — %s",
                side.value,
                symbol,
                info.spread_points,
                sl,
                tp,
                strategy_version,
                skill,
                veto.reason,
            )
            # The two gates are distinct outcomes on the decision trail: a
            # spread cap breach is a market-condition veto, an RR failure is
            # the strategy's own SL/TP geometry not clearing the floor.
            await self._record_outcome(
                signal_id,
                "rr_gate" if veto.kind == "rr" else "spread_veto",
                base_reason=reason,
                explanation=veto.reason,
                checks=(
                    DecisionCheck(
                        name="risk_reward" if veto.kind == "rr" else "spread_points",
                        value=veto.value,
                        threshold=veto.threshold,
                        comparison=">=" if veto.kind == "rr" else "<=",
                        passed=False,
                    ),
                ),
            )
            raise OrderRejected(veto.reason)

        await self._record_checks(
            signal_id,
            self._passed_gate_checks(
                symbol,
                spread_points=info.spread_points,
                point=info.point,
                max_spread_override=max_spread_points,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
            ),
        )

        order = OrderRequest(
            symbol=symbol, side=side, volume=volume, sl=sl, tp=tp, comment=comment, magic=magic
        )
        try:
            result = await self._broker.open_position(order)
        except OrderRejected as exc:
            # The broker's own numeric refusal code, recorded on the decision
            # trail (there is no TradeRecord for an order that never filled) —
            # a fleet dying on MT5 10016 "invalid stops" is otherwise only
            # visible as free text in a log line.
            retcode_checks = (
                (
                    DecisionCheck(
                        name="broker_retcode",
                        value=float(exc.retcode),
                        threshold=float(_RETCODE_DONE),
                        comparison="==",
                        passed=False,
                    ),
                )
                if exc.retcode is not None
                else ()
            )
            # Unlike the spread/RR veto above, this is the broker/MT5 itself
            # refusing the order (stops too close, market closed, filling
            # mode, price moved, ...) — without this log the rejection was
            # previously invisible: the caller only sees `OrderRejected`
            # propagate and silently gives up on the candle.
            logger.info(
                "ENTRY REJECTED (broker): %s %s %.2f lots sl=%s tp=%s strategy=%s skill=%s "
                "retcode=%s — %s",
                side.value,
                symbol,
                volume,
                sl,
                tp,
                strategy_version,
                skill,
                exc.retcode,
                exc,
            )
            await self._record_outcome(
                signal_id,
                "broker_rejected",
                base_reason=reason,
                explanation=str(exc),
                checks=retcode_checks,
            )
            raise
        # Execution telemetry, measured here because this is the only place
        # that sees both the price the order asked for and the broker's ack
        # (OBSERVABILITY_PLAN.md Phase 3). `reference_price` is the tradable
        # price the spread gate was just evaluated against, so it is exactly
        # the price this order was placed at.
        requested_price = reference_price
        slippage = execution_slippage(side, requested_price, result.price)
        execution_latency_ms = (
            (self._clock() - signal_emitted_at).total_seconds() * 1000.0
            if signal_emitted_at is not None
            else None
        )
        # Transaction cost (OBSERVABILITY_PLAN.md Phase 6, cost-as-%-of-edge):
        # spread paid plus signed slippage, converted from price units to
        # account currency via the symbol's contract size. `info` (fetched at
        # the top of this method) is still the same `SymbolInfo` the spread
        # gate evaluated against, so `spread_points`/`point`/`contract_size`
        # all describe this exact fill.
        transaction_cost = (
            (info.spread_points * info.point + slippage) * volume * info.contract_size
        )
        logger.info(
            "ENTRY OPENED: ticket=%d %s %s %.2f lots @ %.5f (requested %.5f, slippage %+.5f) "
            "sl=%s tp=%s spread=%dpts latency=%sms retcode=%s "
            "strategy=%s skill=%s magic=%d reason=%s",
            result.ticket,
            side.value,
            symbol,
            volume,
            result.price,
            requested_price,
            slippage,
            sl,
            tp,
            result.spread_points,
            f"{execution_latency_ms:.0f}" if execution_latency_ms is not None else "n/a",
            result.retcode,
            strategy_version,
            skill,
            magic,
            comment or "manual",
        )
        await self._record_outcome(signal_id, "opened", base_reason=reason)
        await self._event_bus.publish(
            PositionOpened(
                symbol=symbol,
                position_id=str(result.ticket),
                side=result.side.value,
                volume=result.volume,
                price=result.price,
                sl=result.sl,
                tp=result.tp,
                spread_points=result.spread_points,
                comment=result.comment,
                strategy_version=strategy_version,
                skill=skill,
                reason=reason,
                confidence=confidence,
                zone_kind=zone_kind,
                zone_price_low=zone_price_low,
                zone_price_high=zone_price_high,
                zone_time_start=zone_time_start,
                zone_time_end=zone_time_end,
                zone_pattern=zone_pattern,
                pattern=pattern,
                structure=structure,
                indicators=indicators,
                requested_price=requested_price,
                slippage=slippage,
                execution_latency_ms=execution_latency_ms,
                broker_retcode=result.retcode,
                transaction_cost=transaction_cost,
                regime_volatility=regime_volatility,
                regime_volatility_percentile=regime_volatility_percentile,
                regime_trend=regime_trend,
                regime_adx=regime_adx,
                regime_session=regime_session,
                signal_id=signal_id,
            )
        )
        return result

    def _passed_gate_checks(
        self,
        symbol: str,
        *,
        spread_points: int,
        point: float,
        max_spread_override: int | None,
        sl_distance: float | None,
        tp_distance: float | None,
    ) -> tuple[DecisionCheck, ...]:
        """The spread/RR numbers as seen on an order the gate let through —
        recorded so a filled signal's trail shows what it cleared by, not just
        that it cleared. Mirrors `SpreadGate.check`'s own arithmetic; a gate
        that didn't apply (no spread cap configured, or sl/tp not both set)
        contributes no check rather than a fabricated one."""
        config = self._spread_gate.get_config(symbol)
        checks: list[DecisionCheck] = []
        max_spread_points = (
            max_spread_override
            if max_spread_override is not None
            else (config.max_spread_points if config is not None else None)
        )
        if max_spread_points is not None:
            checks.append(
                DecisionCheck(
                    name="spread_points",
                    value=float(spread_points),
                    threshold=float(max_spread_points),
                    comparison="<=",
                    passed=True,
                )
            )
        if sl_distance is not None and tp_distance is not None:
            min_rr = config.min_rr if config is not None else DEFAULT_MIN_RR
            required_tp = min_rr * (sl_distance + spread_points * point)
            checks.append(
                DecisionCheck(
                    name="risk_reward",
                    value=tp_distance,
                    threshold=required_tp,
                    comparison=">=",
                    passed=True,
                )
            )
        return tuple(checks)

    async def _record_checks(
        self, signal_id: str | None, checks: tuple[DecisionCheck, ...]
    ) -> None:
        if signal_id is None or self._signal_decisions is None or not checks:
            return
        await self._signal_decisions.record_checks(signal_id, checks)

    async def _record_outcome(
        self,
        signal_id: str | None,
        outcome: str,
        *,
        base_reason: str,
        explanation: str | None = None,
        checks: tuple[DecisionCheck, ...] = (),
    ) -> None:
        """Stamps the engine-recorded `SignalDecision`'s terminal outcome.
        No-op unless this order came from a recorded signal and a sink is
        wired. `explanation` (veto/reject text) is appended to the strategy's
        own reason exactly as the legacy log-scraper merged it; a fill has
        none, so its reason is left untouched.

        Also increments `tradingbot_signal_outcomes_total` (OBSERVABILITY_PLAN.md
        Phase 5) whenever this order came from a recorded signal — manual/API
        orders (`signal_id is None`) have no signal to attribute an outcome
        to, so they're intentionally excluded from this metric, same as they
        already are from the decision trail itself."""
        if signal_id is None:
            return
        record_signal_outcome(outcome)
        if self._signal_decisions is None:
            return
        reason = None
        if explanation is not None and explanation not in base_reason:
            reason = f"{base_reason} — {explanation}" if base_reason else explanation
        await self._signal_decisions.record_outcome(
            signal_id, outcome, reason=reason, checks=checks
        )

    async def close_position(
        self, ticket: int, volume: float | None = None, reason: str = ""
    ) -> ExecutionResult:
        result = await self._broker.close_position(ticket, volume)
        logger.info(
            "position closed: ticket=%d %s %.2f lots @ %.5f profit=%.2f reason=%s",
            result.ticket,
            result.symbol,
            result.volume,
            result.price,
            result.profit or 0.0,
            reason or "manual",
        )
        await self._event_bus.publish(
            PositionClosed(
                symbol=result.symbol,
                position_id=str(result.ticket),
                close_price=result.price,
                profit=result.profit or 0.0,
                close_reason=reason,
            )
        )
        return result

    async def close_at_price(self, ticket: int, price: float, at: datetime) -> ExecutionResult:
        """Backtest-only: close at an explicit price (e.g. an SL/TP touch
        detected from a historical bar's high/low). Requires a broker
        adapter that supports it (`PaperBroker`); raises `AttributeError`
        against the live gateway broker, which has no such concept."""
        result = await self._broker.close_at_price(ticket, price, at)
        logger.info(
            "position closed (explicit price): ticket=%d %s %.2f lots @ %.5f profit=%.2f",
            result.ticket,
            result.symbol,
            result.volume,
            result.price,
            result.profit or 0.0,
        )
        await self._event_bus.publish(
            PositionClosed(
                symbol=result.symbol,
                position_id=str(result.ticket),
                close_price=result.price,
                profit=result.profit or 0.0,
                occurred_at=at,
            )
        )
        return result

    async def close_all_positions(self, symbol: str) -> list[ExecutionResult]:
        """Closes every currently open position on `symbol`, one broker call
        per ticket via `close_position` (so each still logs and publishes its
        own `PositionClosed`). A single ticket's `OrderRejected` (e.g. broker
        refuses that specific close) is logged and skipped rather than
        aborting the rest of the batch — the caller sees which tickets
        actually closed via the returned list's length against the symbol's
        position count."""
        positions = await self._broker.get_positions(symbol)
        results: list[ExecutionResult] = []
        for position in positions:
            try:
                results.append(await self.close_position(position.ticket))
            except OrderRejected as exc:
                logger.info(
                    "close-all skipped ticket=%d %s — %s", position.ticket, symbol, exc
                )
        return results

    async def modify_position(
        self, ticket: int, sl: float | None, tp: float | None, reason: str = ""
    ) -> None:
        await self._broker.modify_position(ticket, sl, tp)
        logger.info(
            "position modified: ticket=%d sl=%s tp=%s reason=%s", ticket, sl, tp, reason or "manual"
        )

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        return await self._broker.get_positions(symbol)

    async def place_pending_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        volume: float,
        price: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> PendingOrder:
        order = PendingOrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            comment=comment,
        )
        result = await self._broker.place_pending_order(order)
        logger.info(
            "pending order placed: ticket=%d %s %s %s %.2f lots @ %.5f sl=%s tp=%s",
            result.ticket,
            side.value,
            order_type.value,
            symbol,
            volume,
            price,
            sl,
            tp,
        )
        return result

    async def cancel_pending_order(self, ticket: int) -> None:
        await self._broker.cancel_pending_order(ticket)
        logger.info("pending order cancelled: ticket=%d", ticket)

    async def modify_pending_order(
        self, ticket: int, price: float | None, sl: float | None, tp: float | None
    ) -> None:
        await self._broker.modify_pending_order(ticket, price, sl, tp)
        logger.info("pending order modified: ticket=%d price=%s sl=%s tp=%s", ticket, price, sl, tp)

    async def get_pending_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        return await self._broker.get_pending_orders(symbol)

    @property
    def simulates_pending_fills(self) -> bool:
        return self._broker.simulates_pending_fills
