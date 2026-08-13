from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import CandleSnapshot, MarketSnapshot
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import PositionClosed, PositionOpened, TenTradesCompleted


class FakeMarketContext:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.snapshot = MarketSnapshot(
            m5=(
                CandleSnapshot(
                    time=datetime(2026, 7, 10, 13, 55, tzinfo=UTC),
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    tick_volume=100,
                ),
            ),
            h1=(),
        )

    async def capture(self, symbol):
        self.calls.append(symbol)
        return self.snapshot


@pytest.fixture
def repository(tmp_path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))


@pytest.fixture
def market_context() -> FakeMarketContext:
    return FakeMarketContext()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def service(repository, market_context, event_bus) -> TradeJournalService:
    return TradeJournalService(
        repository=repository,
        market_context=market_context,
        event_bus=event_bus,
        review_every_n_trades=3,
    )


def opened_event(
    position_id="1", symbol="XAUUSD", skill="normal/xauusd/breakout_v1"
) -> PositionOpened:
    return PositionOpened(
        symbol=symbol,
        position_id=position_id,
        side="buy",
        volume=0.1,
        price=2400.35,
        sl=2390.0,
        tp=2420.0,
        spread_points=25,
        comment="",
        skill=skill,
    )


def closed_event(position_id="1", symbol="XAUUSD", profit=9.65, occurred_at=None) -> PositionClosed:
    kwargs = {"occurred_at": occurred_at} if occurred_at else {}
    return PositionClosed(
        symbol=symbol, position_id=position_id, close_price=2410.0, profit=profit, **kwargs
    )


async def test_on_position_opened_journals_entry_with_snapshot(service, repository, market_context):
    await service.on_position_opened(opened_event())

    record = repository.get("1")
    assert record is not None
    assert record.symbol == "XAUUSD"
    assert record.open_price == 2400.35
    assert record.is_open is True
    assert record.m5_entry_snapshot == market_context.snapshot.m5
    assert market_context.calls == ["XAUUSD"]


async def test_on_position_opened_journals_decision_context(service, repository):
    event = PositionOpened(
        symbol="XAUUSD",
        position_id="1",
        side="buy",
        volume=0.1,
        price=2400.35,
        sl=2390.0,
        tp=2420.0,
        spread_points=25,
        comment="RBR base retest",
        skill="normal/xauusd/breakout_v1",
        reason="RBR base retest + M15 bullish engulf confirmation",
        confidence=0.82,
        zone_kind="demand",
        zone_price_low=2395.0,
        zone_price_high=2398.5,
        zone_time_start=datetime(2026, 7, 10, 10, 0, tzinfo=UTC),
        zone_time_end=datetime(2026, 7, 10, 13, 45, tzinfo=UTC),
        pattern="bullish_engulfing",
        structure=(("HL", 2397.2, datetime(2026, 7, 10, 13, 30, tzinfo=UTC)),),
    )

    await service.on_position_opened(event)

    record = repository.get("1")
    assert record.reason == "RBR base retest + M15 bullish engulf confirmation"
    assert record.confidence == 0.82
    assert record.zone_kind == "demand"
    assert record.zone_price_low == 2395.0
    assert record.zone_price_high == 2398.5
    assert record.pattern == "bullish_engulfing"
    assert record.structure == (("HL", 2397.2, datetime(2026, 7, 10, 13, 30, tzinfo=UTC)),)


async def test_on_position_opened_stamps_signal_id_from_the_event(service, repository):
    """order_book/ Phase 5: `PositionOpened.signal_id` — the join key back
    to `signal_decisions`/`order_book_snapshots` — must land on the
    journaled `TradeRecord` unchanged."""
    event = PositionOpened(
        symbol="XAUUSD",
        position_id="1",
        side="buy",
        volume=0.1,
        price=2400.35,
        sl=2390.0,
        tp=2420.0,
        spread_points=25,
        skill="normal/xauusd/breakout_v1",
        signal_id="abc123signal",
    )

    await service.on_position_opened(event)

    record = repository.get("1")
    assert record.signal_id == "abc123signal"


async def test_on_position_opened_defaults_signal_id_to_none(service, repository):
    """Manual/API trades (and the pre-Phase-5 `opened_event()` fixture,
    which carries none) must not silently coerce a missing signal_id into
    something falsy-but-present."""
    await service.on_position_opened(opened_event())

    record = repository.get("1")
    assert record.signal_id is None


async def test_on_position_closed_updates_existing_record(service, repository):
    await service.on_position_opened(opened_event())
    await service.on_position_closed(closed_event())

    record = repository.get("1")
    assert record.is_open is False
    assert record.close_price == 2410.0
    assert record.profit == 9.65
    assert record.m5_exit_snapshot != ()


async def test_on_position_closed_without_matching_open_is_ignored(service, repository):
    await service.on_position_closed(closed_event(position_id="missing"))
    assert repository.get("missing") is None


async def test_ten_trade_review_fires_after_n_closed_trades(service, event_bus):
    published = []

    async def record(event):
        published.append(event)

    event_bus.subscribe(TenTradesCompleted, record)

    for i in range(3):
        await service.on_position_opened(opened_event(position_id=str(i)))
        await service.on_position_closed(
            closed_event(
                position_id=str(i),
                profit=float(i),
                occurred_at=datetime(2026, 7, 10, 15, i, tzinfo=UTC),
            )
        )

    assert len(published) == 1
    assert isinstance(published[0], TenTradesCompleted)
    assert published[0].symbol == "XAUUSD"
    assert published[0].skill == "normal/xauusd/breakout_v1"
    assert published[0].trade_ids == ("0", "1", "2")


async def test_no_review_event_before_threshold(service, event_bus):
    published = []

    async def record(event):
        published.append(event)

    event_bus.subscribe(TenTradesCompleted, record)

    await service.on_position_opened(opened_event(position_id="1"))
    await service.on_position_closed(closed_event(position_id="1"))

    assert published == []


async def test_manual_trade_with_no_skill_never_triggers_review(service, event_bus):
    published = []

    async def record(event):
        published.append(event)

    event_bus.subscribe(TenTradesCompleted, record)

    for i in range(3):
        await service.on_position_opened(opened_event(position_id=str(i), skill=None))
        await service.on_position_closed(
            closed_event(position_id=str(i), occurred_at=datetime(2026, 7, 10, 15, i, tzinfo=UTC))
        )

    assert published == []


async def test_two_bots_on_one_symbol_are_reviewed_on_independent_cadences(service, event_bus):
    published = []

    async def record(event):
        published.append(event)

    event_bus.subscribe(TenTradesCompleted, record)

    # Bot A's 3rd closed trade should trigger a review scoped to bot A only,
    # even though bot B has also been trading the same symbol concurrently.
    for i in range(2):
        await service.on_position_opened(opened_event(position_id=f"a{i}", skill="normal/xauusd/a"))
        await service.on_position_closed(
            closed_event(position_id=f"a{i}", occurred_at=datetime(2026, 7, 10, 15, i, tzinfo=UTC))
        )
    await service.on_position_opened(opened_event(position_id="b0", skill="normal/xauusd/b"))
    await service.on_position_closed(
        closed_event(position_id="b0", occurred_at=datetime(2026, 7, 10, 15, 10, tzinfo=UTC))
    )
    await service.on_position_opened(opened_event(position_id="a2", skill="normal/xauusd/a"))
    await service.on_position_closed(
        closed_event(position_id="a2", occurred_at=datetime(2026, 7, 10, 15, 11, tzinfo=UTC))
    )

    assert len(published) == 1
    assert published[0].skill == "normal/xauusd/a"
    assert published[0].trade_ids == ("a0", "a1", "a2")


async def test_get_markers_and_get_last_n_proxy_repository(service):
    await service.on_position_opened(opened_event())
    await service.on_position_closed(closed_event())

    markers = await service.get_markers("XAUUSD")
    assert len(markers) == 1

    last = await service.get_last_n("XAUUSD", 5)
    assert len(last) == 1


async def test_get_trade_returns_journaled_record(service, repository):
    await service.on_position_opened(opened_event())

    record = service.get_trade("1")

    assert record is not None
    assert record.id == "1"
    assert record.symbol == "XAUUSD"


def test_get_trade_returns_none_for_unknown_id(service):
    assert service.get_trade("missing") is None


async def test_get_markers_skill_filter_proxies_through(service):
    await service.on_position_opened(opened_event(position_id="a", skill="normal/xauusd/a"))
    await service.on_position_opened(opened_event(position_id="b", skill="normal/xauusd/b"))

    markers = await service.get_markers("XAUUSD", skill="normal/xauusd/a")

    assert [m.id for m in markers] == ["a"]


async def test_analytics_methods_use_slim_query_not_full_get_all(service, repository):
    """get_symbol_analytics/get_bot_analytics must go through the slim
    get_all_for_analytics query (which skips the JSON snapshot/structure
    columns) rather than the full get_all() — the optimization this test
    guards against regressing."""
    await service.on_position_opened(opened_event())
    await service.on_position_closed(closed_event())

    calls = {"get_all": 0, "get_all_for_analytics": 0}
    orig_get_all = repository.get_all
    orig_slim = repository.get_all_for_analytics

    def counted_get_all(*args, **kwargs):
        calls["get_all"] += 1
        return orig_get_all(*args, **kwargs)

    def counted_slim(*args, **kwargs):
        calls["get_all_for_analytics"] += 1
        return orig_slim(*args, **kwargs)

    repository.get_all = counted_get_all
    repository.get_all_for_analytics = counted_slim

    symbol_analytics = await service.get_symbol_analytics()
    bot_analytics = await service.get_bot_analytics()

    assert calls == {"get_all": 0, "get_all_for_analytics": 2}
    assert len(symbol_analytics) == 1
    assert len(bot_analytics) == 1
