"""MFE/MAE excursion (OBSERVABILITY_PLAN.md Phase 3): the pure arithmetic,
and its accumulation through `TradeJournalService` — candle by candle while a
position is open, finalized against the exit price on close.

The case that pins the design down is a position that opens and closes inside
a single candle: no candle ever closes during its life, so unless the exit
price itself extends the excursion, such a trade would report a flat 0/0 and
quietly bias every scalp bot's excursion stats toward nothing.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.excursion import Excursion, extend_excursion, finalize_excursion
from src.journal.domain.models import CandleSnapshot, MarketSnapshot
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed, PositionClosed, PositionOpened

ENTRY = 2400.00


class FakeMarketContext:
    """Serves one "latest candle" at a time, so a test can walk a position
    through a specific sequence of bars."""

    def __init__(self) -> None:
        self.candle: CandleSnapshot | None = None

    async def capture(self, symbol):
        return MarketSnapshot(m5=(), h1=())

    async def latest_candle(self, symbol, timeframe):
        return self.candle

    def serve(self, high: float, low: float) -> None:
        self.candle = CandleSnapshot(
            time=datetime(2026, 8, 5, 12, 0, tzinfo=UTC),
            open=(high + low) / 2,
            high=high,
            low=low,
            close=(high + low) / 2,
            tick_volume=100,
        )


@pytest.fixture
def repository(tmp_path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))


@pytest.fixture
def market_context() -> FakeMarketContext:
    return FakeMarketContext()


@pytest.fixture
def service(repository, market_context) -> TradeJournalService:
    return TradeJournalService(
        repository=repository,
        market_context=market_context,
        event_bus=EventBus(),
    )


def opened(side: str = "buy") -> PositionOpened:
    return PositionOpened(
        symbol="XAUUSD",
        position_id="1",
        side=side,
        volume=0.1,
        price=ENTRY,
        sl=2390.0,
        tp=2420.0,
        spread_points=25,
    )


# ── pure arithmetic ───────────────────────────────────────────────────────


def test_a_buys_excursion_is_high_above_entry_and_low_below_it():
    result = extend_excursion(Excursion(), side="buy", open_price=ENTRY, high=2405.0, low=2397.0)
    assert result.mfe == pytest.approx(5.0)
    assert result.mae == pytest.approx(3.0)


def test_a_sells_excursion_mirrors_a_buys():
    result = extend_excursion(Excursion(), side="sell", open_price=ENTRY, high=2405.0, low=2397.0)
    assert result.mfe == pytest.approx(3.0)
    assert result.mae == pytest.approx(5.0)


def test_a_candle_entirely_on_one_side_of_entry_never_reports_negative_excursion():
    """A buy whose candle never traded below entry has zero adverse
    excursion, not a negative one — MFE/MAE are magnitudes."""
    result = extend_excursion(Excursion(), side="buy", open_price=ENTRY, high=2406.0, low=2402.0)
    assert result.mfe == pytest.approx(6.0)
    assert result.mae == pytest.approx(0.0)


def test_a_quieter_later_candle_never_shrinks_an_earlier_spike():
    after_spike = extend_excursion(
        Excursion(), side="buy", open_price=ENTRY, high=2410.0, low=2390.0
    )
    after_quiet = extend_excursion(
        after_spike, side="buy", open_price=ENTRY, high=2401.0, low=2399.5
    )
    assert after_quiet.mfe == pytest.approx(10.0)
    assert after_quiet.mae == pytest.approx(10.0)


def test_finalizing_folds_in_the_exit_price():
    result = finalize_excursion(
        Excursion(mfe=2.0, mae=1.0), side="buy", open_price=ENTRY, close_price=2408.0
    )
    assert result.mfe == pytest.approx(8.0)
    assert result.mae == pytest.approx(1.0)


# ── accumulation through the journal service ──────────────────────────────


async def test_a_freshly_opened_trade_starts_measured_at_zero(service, repository):
    await service.on_position_opened(opened())

    record = repository.get("1")
    assert record is not None
    assert (record.mfe, record.mae) == (0.0, 0.0)


async def test_excursion_widens_over_successive_candles(service, repository, market_context):
    await service.on_position_opened(opened())

    market_context.serve(high=2403.0, low=2399.0)
    await service.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))
    market_context.serve(high=2412.0, low=2401.0)
    await service.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    record = repository.get("1")
    assert record is not None
    assert record.mfe == pytest.approx(12.0)
    assert record.mae == pytest.approx(1.0)  # from the first candle only


async def test_only_the_excursion_timeframe_is_accumulated(service, repository, market_context):
    """The same symbol may stream several timeframes; counting each of them
    would be redundant work for identical results."""
    await service.on_position_opened(opened())

    market_context.serve(high=2450.0, low=2350.0)
    await service.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M1"))

    record = repository.get("1")
    assert record is not None
    assert (record.mfe, record.mae) == (0.0, 0.0)


async def test_a_closed_trade_stops_accumulating(service, repository, market_context):
    await service.on_position_opened(opened())
    await service.on_position_closed(
        PositionClosed(symbol="XAUUSD", position_id="1", close_price=2402.0, profit=20.0)
    )

    market_context.serve(high=2500.0, low=2300.0)
    await service.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    record = repository.get("1")
    assert record is not None
    assert record.mfe == pytest.approx(2.0)


async def test_closing_folds_the_exit_price_into_the_accumulated_excursion(
    service, repository, market_context
):
    await service.on_position_opened(opened())
    market_context.serve(high=2403.0, low=2396.0)
    await service.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    await service.on_position_closed(
        PositionClosed(symbol="XAUUSD", position_id="1", close_price=2409.0, profit=90.0)
    )

    record = repository.get("1")
    assert record is not None
    assert record.mfe == pytest.approx(9.0)  # exit beat every candle high
    assert record.mae == pytest.approx(4.0)  # still the candle's low


async def test_a_trade_opened_and_closed_within_one_candle_still_gets_excursion(
    service, repository
):
    """No `CandleClosed` ever fires during this position's life — the exit
    price is the only measurement it will ever have, and it must be used."""
    await service.on_position_opened(opened())

    await service.on_position_closed(
        PositionClosed(symbol="XAUUSD", position_id="1", close_price=2396.5, profit=-35.0)
    )

    record = repository.get("1")
    assert record is not None
    assert record.mfe == pytest.approx(0.0)
    assert record.mae == pytest.approx(3.5)


async def test_a_sell_opened_and_closed_within_one_candle_measures_the_other_way(
    service, repository
):
    await service.on_position_opened(opened(side="sell"))

    await service.on_position_closed(
        PositionClosed(symbol="XAUUSD", position_id="1", close_price=2396.5, profit=35.0)
    )

    record = repository.get("1")
    assert record is not None
    assert record.mfe == pytest.approx(3.5)
    assert record.mae == pytest.approx(0.0)
