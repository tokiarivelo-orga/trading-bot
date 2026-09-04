"""Position reconciliation (Phase 9 §12): detects trades the broker closed
without the backend's involvement — a server-side SL/TP fill, or any close
that happened while the backend was down — and republishes `PositionClosed`
so the journal, risk-manager circuit breakers, and 10-trade review trigger
all see it exactly as if `OrderService.close_position` had been called.

Three entry points:
  - `reconcile_all()`: startup/reconnect, and now also `ReconciliationPoller`'s
    fast steady-state poll (every few seconds, independent of candle
    cadence — see `reconciliation_poller.py`) — diffs the journal's
    persisted open trades against the broker's current open positions,
    across all symbols. Catches closes that happened while the backend was
    completely down, and is what makes a live SL/TP fill show up in trade
    history within seconds instead of waiting on the next M5 candle.
  - `reconcile_vanished()`: mid-session — `PositionManager.on_candle_closed`
    already knows exactly which tickets disappeared between two M5 closes;
    this just resolves and republishes them. Kept as a belt-and-suspenders
    fallback alongside the poller above (e.g. if the poller task ever dies).
  - Both funnel through `_close_from_history`, which re-checks the journal
    immediately before publishing so the two triggers racing each other
    never double-publish `PositionClosed` for the same ticket.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.broker.domain.account import BrokerUnavailable
from src.broker.domain.trading import Side
from src.broker.ports.trading import BrokerPort
from src.journal.application.trade_journal import TradeJournalService
from src.shared.events.bus import EventBus
from src.shared.events.definitions import PositionClosed, PositionOpened

logger = logging.getLogger(__name__)

# How long a ticket logs its "no close history" WARNING loudly (persisted to
# the activity log) before `_log_unresolved` downgrades it to DEBUG, and how
# often it gets one WARNING reminder after that — see `_log_unresolved`.
_LOUD_RETRY_WINDOW = timedelta(minutes=10)
_QUIET_REMINDER_INTERVAL = timedelta(minutes=30)


class ReconciliationService:
    def __init__(
        self,
        broker: BrokerPort,
        journal: TradeJournalService,
        event_bus: EventBus,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._broker = broker
        self._journal = journal
        self._event_bus = event_bus
        self._clock = clock
        # ticket_id -> when it was first seen unresolved / last logged loudly.
        # Cleared whenever a ticket resolves (see `_close_from_history`) so a
        # ticket that goes stale, resolves, then later goes stale again
        # (unlikely but not impossible) gets a fresh loud window rather than
        # inheriting a stale timestamp from its first go-round.
        self._unresolved_since: dict[str, datetime] = {}
        self._last_loud_log: dict[str, datetime] = {}

    async def reconcile_all(self) -> None:
        open_trades = await self._journal.get_open_trades()
        if not open_trades:
            return
        try:
            open_tickets = {p.ticket for p in await self._broker.get_positions()}
        except BrokerUnavailable as exc:
            logger.warning("reconciliation skipped: broker unavailable: %s", exc)
            return
        stale = [t for t in open_trades if int(t.id) not in open_tickets]
        for trade in stale:
            try:
                await self._close_from_history(trade.symbol, trade.id)
            except BrokerUnavailable as exc:
                logger.warning(
                    "reconciliation: could not get close history for ticket=%s: %s",
                    trade.id,
                    exc,
                )

    async def reconcile_vanished(self, symbol: str, vanished_tickets: set[int]) -> None:
        for ticket in vanished_tickets:
            try:
                await self._close_from_history(symbol, str(ticket))
            except BrokerUnavailable as exc:
                logger.warning(
                    "reconciliation: could not reconcile vanished ticket=%d symbol=%s: %s",
                    ticket,
                    symbol,
                    exc,
                )

    async def reconcile_pending_fill(
        self, symbol: str, ticket: int, side: Side, volume: float
    ) -> bool:
        """A pending order we were tracking is no longer resting — either we
        cancelled it ourselves, or the broker triggered it. Look for the
        resulting open position (matching ticket first, since a triggered
        MT5 pending order typically keeps its order ticket; falling back to
        side+volume in case it doesn't) and publish `PositionOpened` for it
        so the journal/risk manager see the fill exactly as if
        `OrderService.open_position` had been called directly. Returns
        whether a match was found — `False` just means we cancelled it
        ourselves (nothing to reconcile), not an error."""
        try:
            positions = await self._broker.get_positions(symbol)
        except BrokerUnavailable as exc:
            logger.warning(
                "reconciliation: could not check pending fill for ticket=%d: %s",
                ticket,
                exc,
            )
            return False
        match = next((p for p in positions if p.ticket == ticket), None)
        if match is None:
            match = next((p for p in positions if p.side is side and p.volume == volume), None)
        if match is None:
            return False
        await self._event_bus.publish(
            PositionOpened(
                symbol=match.symbol,
                position_id=str(match.ticket),
                side=match.side.value,
                volume=match.volume,
                price=match.open_price,
                sl=match.sl,
                tp=match.tp,
                spread_points=0,
                comment=match.comment,
                occurred_at=match.open_time,
            )
        )
        logger.info(
            "reconciled pending-order fill: ticket=%d symbol=%s side=%s volume=%.2f @ %.5f",
            match.ticket,
            match.symbol,
            match.side.value,
            match.volume,
            match.open_price,
        )
        return True

    async def _close_from_history(self, symbol: str, ticket_id: str) -> None:
        # Idempotency guard: `reconcile_vanished` (M5-candle-gated, per
        # symbol) and `reconcile_all` (fast periodic poll, all symbols —
        # see `ReconciliationPoller`) can both observe the same vanished
        # ticket before either one's `PositionClosed` publish lands in the
        # journal. Re-publishing for an already-closed trade would double
        # `risk_manager.record_trade_closed` (double-counts P&L into the
        # daily-loss/consecutive-loss breakers) and could double-trigger the
        # 10-trade AI review, so re-check the journal immediately before
        # publishing rather than trusting the caller's vanished-ticket diff.
        existing = await asyncio.to_thread(self._journal.get_trade, ticket_id)
        if existing is not None and not existing.is_open:
            logger.debug(
                "reconciliation: ticket=%s symbol=%s already closed in journal, skipping",
                ticket_id,
                symbol,
            )
            self._unresolved_since.pop(ticket_id, None)
            self._last_loud_log.pop(ticket_id, None)
            return
        info = await self._broker.get_close_info(int(ticket_id))
        if info is None:
            self._log_unresolved(ticket_id, symbol)
            return
        self._unresolved_since.pop(ticket_id, None)
        self._last_loud_log.pop(ticket_id, None)
        await self._event_bus.publish(
            PositionClosed(
                symbol=symbol,
                position_id=ticket_id,
                close_price=info.price,
                profit=info.profit,
                occurred_at=info.time,
            )
        )
        logger.info(
            "reconciled broker-side close: ticket=%s symbol=%s profit=%.2f",
            ticket_id,
            symbol,
            info.profit,
        )

    def _log_unresolved(self, ticket_id: str, symbol: str) -> None:
        """Logs a ticket the broker still has no close history for.

        WARNING (persisted to the activity log) for the first
        `_LOUD_RETRY_WINDOW` after first seen — plenty of time to notice a
        genuine late-syncing MT5 deal history, which in practice always
        resolves within minutes of a gateway reconnect. Past that window,
        a ticket that still won't resolve isn't a transient sync race —
        it's permanently unresolvable (root-caused 2026-08-20: a stale
        paper-mode position id — see `PaperBroker`'s `itertools.count(1)`
        ticket counter — left open in the journal from before the account
        switched to its live gateway, which of course has no MT5 deal
        history for a ticket number MT5 never issued). Since `reconcile_all`
        retries every few seconds forever (`ReconciliationPoller`), logging
        every attempt at WARNING floods the activity log for as long as the
        ticket stays stuck — one real incident produced ~20k WARNING rows
        in under 4 hours from a single ticket. Downgraded to DEBUG (dropped
        by the default INFO log level) after the loud window, with one
        WARNING reminder every `_QUIET_REMINDER_INTERVAL` so a permanently
        stuck ticket doesn't go completely invisible — it still needs a
        human to close it out (delete/adjust the journal row) since it can
        never resolve on its own."""
        now = self._clock()
        first_seen = self._unresolved_since.setdefault(ticket_id, now)
        age = now - first_seen
        last_loud = self._last_loud_log.get(ticket_id)
        loud = (
            age < _LOUD_RETRY_WINDOW
            or last_loud is None
            or now - last_loud >= _QUIET_REMINDER_INTERVAL
        )
        log = logger.warning if loud else logger.debug
        if loud:
            self._last_loud_log[ticket_id] = now
        log(
            "reconciliation: no close history for ticket=%s symbol=%s (unresolved for %s) — "
            "still unresolved, will retry next reconciliation pass",
            ticket_id,
            symbol,
            age,
        )
