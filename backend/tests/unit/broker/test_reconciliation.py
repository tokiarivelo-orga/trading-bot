from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.broker.application.reconciliation import ReconciliationService
from src.broker.domain.trading import ClosedPositionInfo, Position, Side
from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import MarketSnapshot
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import PositionClosed, PositionOpened


class FakeMarketContext:
    async def capture(self, symbol):
        return MarketSnapshot(m5=(), h1=())


class FakeBroker:
    def __init__(self, open_positions: list[Position], close_info: dict[int, ClosedPositionInfo]):
        self._open_positions = open_positions
        self._close_info = close_info

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        if symbol is None:
            return list(self._open_positions)
        return [p for p in self._open_positions if p.symbol == symbol]

    async def get_close_info(self, ticket: int) -> ClosedPositionInfo | None:
        return self._close_info.get(ticket)


@pytest.fixture
def journal(tmp_path) -> TradeJournalService:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    repository = JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))
    return TradeJournalService(
        repository=repository, market_context=FakeMarketContext(), event_bus=EventBus()
    )


def _open_position(ticket=1, symbol="XAUUSD") -> Position:
    return Position(
        ticket=ticket,
        symbol=symbol,
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=2390.0,
        tp=2420.0,
        open_time=datetime.now(UTC),
        profit=0.0,
    )


async def test_reconcile_all_closes_journaled_trade_the_broker_no_longer_shows(journal):
    await journal.on_position_opened(
        PositionOpened(
            symbol="XAUUSD",
            position_id="1",
            side="buy",
            volume=0.1,
            price=2400.0,
            sl=2390.0,
            tp=2420.0,
            spread_points=25,
        )
    )
    close_info = ClosedPositionInfo(
        symbol="XAUUSD", price=2390.0, time=datetime.now(UTC), profit=-10.0
    )
    broker = FakeBroker(open_positions=[], close_info={1: close_info})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    await reconciliation.reconcile_all()

    assert len(published) == 1
    assert published[0].position_id == "1"
    assert published[0].profit == -10.0


async def test_reconcile_all_skips_trades_still_open_at_the_broker(journal):
    await journal.on_position_opened(
        PositionOpened(
            symbol="XAUUSD",
            position_id="1",
            side="buy",
            volume=0.1,
            price=2400.0,
            sl=2390.0,
            tp=2420.0,
            spread_points=25,
        )
    )
    broker = FakeBroker(open_positions=[_open_position(ticket=1)], close_info={})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    await reconciliation.reconcile_all()

    assert published == []


async def test_reconcile_vanished_logs_and_skips_when_no_close_history(journal):
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    await reconciliation.reconcile_vanished("XAUUSD", {99})

    assert published == []


async def test_reconcile_vanished_publishes_position_closed(journal):
    close_info = ClosedPositionInfo(
        symbol="XAUUSD", price=2405.0, time=datetime.now(UTC), profit=15.0
    )
    broker = FakeBroker(open_positions=[], close_info={7: close_info})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    await reconciliation.reconcile_vanished("XAUUSD", {7})

    assert len(published) == 1
    assert published[0].position_id == "7"
    assert published[0].profit == 15.0


async def test_reconcile_pending_fill_matches_by_ticket_and_publishes_opened(journal):
    broker = FakeBroker(open_positions=[_open_position(ticket=5)], close_info={})
    event_bus = EventBus()
    published: list[PositionOpened] = []

    async def on_opened(event: PositionOpened) -> None:
        published.append(event)

    event_bus.subscribe(PositionOpened, on_opened)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    filled = await reconciliation.reconcile_pending_fill("XAUUSD", 5, Side.BUY, 0.1)

    assert filled is True
    assert len(published) == 1
    assert published[0].position_id == "5"
    assert published[0].side == "buy"


async def test_reconcile_pending_fill_falls_back_to_side_and_volume_match(journal):
    # Ticket 5 was the pending order's ticket; the resulting position got a
    # different ticket (9) but matches side/volume.
    broker = FakeBroker(open_positions=[_open_position(ticket=9)], close_info={})
    event_bus = EventBus()
    published: list[PositionOpened] = []

    async def on_opened(event: PositionOpened) -> None:
        published.append(event)

    event_bus.subscribe(PositionOpened, on_opened)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    filled = await reconciliation.reconcile_pending_fill("XAUUSD", 5, Side.BUY, 0.1)

    assert filled is True
    assert published[0].position_id == "9"


async def test_reconcile_all_does_not_republish_a_trade_already_closed_in_the_journal(journal):
    # Simulates the real wiring (container.py: event_bus.subscribe(PositionClosed,
    # trade_journal.on_position_closed)) so the journal row actually gets a
    # close_time after the first reconciliation pass.
    await journal.on_position_opened(
        PositionOpened(
            symbol="XAUUSD",
            position_id="1",
            side="buy",
            volume=0.1,
            price=2400.0,
            sl=2390.0,
            tp=2420.0,
            spread_points=25,
        )
    )
    close_info = ClosedPositionInfo(
        symbol="XAUUSD", price=2390.0, time=datetime.now(UTC), profit=-10.0
    )
    broker = FakeBroker(open_positions=[], close_info={1: close_info})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)
        await journal.on_position_closed(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    # Two poll ticks in a row (as ReconciliationPoller would run), same
    # vanished ticket still absent from the broker's open-position list.
    await reconciliation.reconcile_all()
    await reconciliation.reconcile_all()

    assert len(published) == 1


async def test_reconcile_vanished_does_not_republish_after_reconcile_all_already_closed_it(
    journal,
):
    # Cross-trigger race: the fast poller's reconcile_all() and
    # PositionManager's M5-gated reconcile_vanished() both observing the
    # same vanished ticket must not double-publish PositionClosed (that
    # would double-count P&L in the risk manager's circuit breakers).
    await journal.on_position_opened(
        PositionOpened(
            symbol="XAUUSD",
            position_id="1",
            side="buy",
            volume=0.1,
            price=2400.0,
            sl=2390.0,
            tp=2420.0,
            spread_points=25,
        )
    )
    close_info = ClosedPositionInfo(
        symbol="XAUUSD", price=2390.0, time=datetime.now(UTC), profit=-10.0
    )
    broker = FakeBroker(open_positions=[], close_info={1: close_info})
    event_bus = EventBus()
    published: list[PositionClosed] = []

    async def on_closed(event: PositionClosed) -> None:
        published.append(event)
        await journal.on_position_closed(event)

    event_bus.subscribe(PositionClosed, on_closed)
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    await reconciliation.reconcile_all()
    await reconciliation.reconcile_vanished("XAUUSD", {1})

    assert len(published) == 1


async def test_reconcile_pending_fill_returns_false_when_no_match(journal):
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    reconciliation = ReconciliationService(broker=broker, journal=journal, event_bus=event_bus)

    filled = await reconciliation.reconcile_pending_fill("XAUUSD", 5, Side.BUY, 0.1)

    assert filled is False


async def test_unresolved_ticket_logs_warning_within_the_loud_window(journal, caplog):
    # A ticket that's been unresolved for a few seconds — well inside
    # `_LOUD_RETRY_WINDOW` — should still log at WARNING every pass, same as
    # before this feature existed.
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    now = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
    reconciliation = ReconciliationService(
        broker=broker, journal=journal, event_bus=event_bus, clock=lambda: now
    )

    with caplog.at_level("DEBUG"):
        await reconciliation.reconcile_vanished("XAUUSD", {1})

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "ticket=1" in warnings[0].message


async def test_unresolved_ticket_downgrades_to_debug_after_the_loud_window(journal, caplog):
    # Root cause of the 2026-08-20 incident: a permanently-unresolvable
    # ticket (a stale paper-mode position id, never known to the live
    # gateway) retried every few seconds forever, each attempt logging a
    # fresh WARNING — ~20k rows in under 4 hours. Past `_LOUD_RETRY_WINDOW`
    # the same ticket must fall back to DEBUG so it stops flooding the
    # activity log, while `reconcile_vanished` keeps actually retrying (a
    # ticket that resolves late, as real ones did that day, must still be
    # caught automatically).
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    current = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
    reconciliation = ReconciliationService(
        broker=broker, journal=journal, event_bus=event_bus, clock=lambda: current
    )

    with caplog.at_level("DEBUG"):
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # first sighting: loud
        current += timedelta(minutes=15)  # past the 10-minute loud window
        caplog.clear()
        await reconciliation.reconcile_vanished("XAUUSD", {1})

    levels = {r.levelname for r in caplog.records}
    assert levels == {"DEBUG"}


async def test_unresolved_ticket_gets_a_periodic_warning_reminder(journal, caplog):
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    current = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
    reconciliation = ReconciliationService(
        broker=broker, journal=journal, event_bus=event_bus, clock=lambda: current
    )

    with caplog.at_level("DEBUG"):
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # first sighting: loud
        current += timedelta(minutes=15)  # quiet (DEBUG) window
        caplog.clear()
        await reconciliation.reconcile_vanished("XAUUSD", {1})
        assert {r.levelname for r in caplog.records} == {"DEBUG"}

        current += timedelta(minutes=30)  # past the 30-minute reminder interval
        caplog.clear()
        await reconciliation.reconcile_vanished("XAUUSD", {1})

    assert [r.levelname for r in caplog.records] == ["WARNING"]


async def test_resolved_ticket_clears_its_unresolved_tracking(journal, caplog):
    # A ticket that goes stale, resolves, then later (unlikely, but a real
    # ticket id could in principle be reused) goes stale again must get a
    # fresh loud window rather than inheriting timing from its first
    # go-round — otherwise it would start out already downgraded to DEBUG.
    close_info = ClosedPositionInfo(
        symbol="XAUUSD", price=2390.0, time=datetime.now(UTC), profit=-10.0
    )
    broker = FakeBroker(open_positions=[], close_info={})
    event_bus = EventBus()
    current = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)
    reconciliation = ReconciliationService(
        broker=broker, journal=journal, event_bus=event_bus, clock=lambda: current
    )

    with caplog.at_level("DEBUG"):
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # unresolved, loud
        current += timedelta(minutes=15)
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # unresolved, quiet now

        broker._close_info[1] = close_info
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # resolves

        del broker._close_info[1]
        current += timedelta(seconds=1)
        caplog.clear()
        await reconciliation.reconcile_vanished("XAUUSD", {1})  # unresolved again

    assert [r.levelname for r in caplog.records] == ["WARNING"]
