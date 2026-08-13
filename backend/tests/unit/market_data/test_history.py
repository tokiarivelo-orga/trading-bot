from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.market_data.adapters.symbol_spec_repository import SymbolSpecRepository
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import Candle, MarketDataUnavailable, SymbolInfo, Timeframe
from src.shared.db.base import Base


class FakeMarketData:
    """Stub `MarketDataPort` — only `get_symbol_info` is exercised here."""

    def __init__(self, info: SymbolInfo) -> None:
        self._info = info

    async def get_candles(self, symbol, timeframe, count, before=None):
        raise NotImplementedError

    async def get_tick(self, symbol):
        raise NotImplementedError

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self._info


class FakePagingMarketData:
    """Stub `MarketDataPort` that mimics the gateway's `before`-cursor paging:
    given a full ascending-by-time history, returns up to `count` bars
    strictly before `before` (or the most recent `count` bars if `before` is
    None), newest last — same contract as `GatewayMarketData.get_candles`."""

    def __init__(self, bars: list[Candle]) -> None:
        self._bars = bars  # ascending by time
        self.calls: list[datetime | None] = []

    async def get_candles(self, symbol, timeframe, count, before=None):
        self.calls.append(before)
        if before is None:
            return self._bars[-count:]
        cutoff = next((i for i, c in enumerate(self._bars) if c.time >= before), len(self._bars))
        return self._bars[:cutoff][-count:]

    async def get_tick(self, symbol):
        raise NotImplementedError

    async def get_symbol_info(self, symbol):
        raise NotImplementedError


class FakeCandleRepository:
    def __init__(self, bars: list[Candle] | None = None) -> None:
        self.stored: list[Candle] = []
        self._bars = bars or []  # ascending by time, as get_latest/get_before promise
        self.enrich_calls: list[tuple] = []

    def enrich_missing(self, symbol, timeframe, account_id: str = "default", **kwargs) -> int:
        self.enrich_calls.append((symbol, timeframe, account_id))
        return 0

    def upsert_many(self, candles, account_id: str = "default") -> int:
        candles = list(candles)
        self.stored.extend(candles)
        # Merge into the readable history the same way the real repository's
        # upsert does (overwrite in place, keyed on bar open) — `repair_gaps`
        # rescans right after downloading, so a fake whose reads ignore its
        # own writes would report every repaired hole as still missing.
        merged = {c.time: c for c in self._bars}
        merged.update({c.time: c for c in candles})
        self._bars = [merged[time] for time in sorted(merged)]
        return len(candles)

    def get_latest(self, symbol, timeframe, count, account_id: str = "default") -> list[Candle]:
        return self._bars[-count:]

    def get_before(
        self, symbol, timeframe, before, count, account_id: str = "default"
    ) -> list[Candle]:
        cutoff = next((i for i, c in enumerate(self._bars) if c.time >= before), len(self._bars))
        return self._bars[:cutoff][-count:]

    def get_range(self, symbol, timeframe, start, end, account_id: str = "default") -> list[Candle]:
        return [c for c in self._bars if start <= c.time < end]


class FakeUnavailableMarketData:
    """Stub `MarketDataPort` where every candle fetch raises, like a
    down/unreachable gateway."""

    async def get_candles(self, symbol, timeframe, count, before=None):
        raise MarketDataUnavailable("gateway down")

    async def get_tick(self, symbol):
        raise NotImplementedError

    async def get_symbol_info(self, symbol):
        raise NotImplementedError


def make_bars(count: int, *, start: datetime, timeframe: Timeframe = Timeframe.M5) -> list[Candle]:
    return [
        Candle(
            symbol="XAUUSD",
            timeframe=timeframe,
            time=start + i * timedelta(seconds=timeframe.seconds),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            tick_volume=1,
            spread_points=1,
        )
        for i in range(count)
    ]


def make_info(**overrides) -> SymbolInfo:
    defaults = dict(
        symbol="Volatility 75 Index",
        bid=245678.10,
        ask=245678.35,
        spread_points=25,
        point=0.01,
        digits=2,
        stops_level=0,
        contract_size=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    return SymbolInfo(**{**defaults, **overrides})


@pytest.fixture
def symbol_spec_repository(tmp_path) -> SymbolSpecRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return SymbolSpecRepository(sessionmaker(bind=engine, expire_on_commit=False))


async def test_sync_symbol_spec_persists_gateway_facts(symbol_spec_repository):
    service = CandleHistoryService(
        FakeMarketData(make_info()), repository=None, symbol_spec_repository=symbol_spec_repository
    )

    await service.sync_symbol_spec("Volatility 75 Index")

    stored = symbol_spec_repository.get("Volatility 75 Index")
    assert stored is not None
    assert stored.point == 0.01
    assert stored.digits == 2
    assert stored.contract_size == 1.0
    assert stored.volume_min == 0.01


async def test_sync_symbol_spec_without_repository_does_not_raise():
    service = CandleHistoryService(FakeMarketData(make_info()), repository=None)
    await service.sync_symbol_spec("Volatility 75 Index")  # no-ops, doesn't crash


async def test_backfill_without_start_fetches_only_most_recent_page():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository()
    service = CandleHistoryService(market_data, repository)

    stored = await service.backfill("XAUUSD", Timeframe.M5, 100)

    assert stored == 100
    assert market_data.calls == [None]
    assert repository.stored == bars[-100:]


async def test_backfill_with_start_pages_backward_until_range_is_covered():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)  # 250 * 5min spans well past `origin`
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository()
    service = CandleHistoryService(market_data, repository)

    stored = await service.backfill("XAUUSD", Timeframe.M5, 100, start=origin)

    # 3 pages of <=100 to cover all 250 bars back to `origin`.
    assert stored == 250
    assert len(market_data.calls) == 3
    assert sorted(repository.stored, key=lambda c: c.time) == bars


async def test_get_candles_serves_from_db_when_already_caught_up():
    """`CandleStreamService` keeps a watched symbol's DB copy current — once
    it has, a plain (no `before`) fetch should be a local read, not a live
    gateway round trip. Freshness is judged against wall-clock time (the
    service calls `datetime.now(UTC)` itself), so bars are anchored off the
    real last-closed M5 boundary rather than a fixed date."""
    last_closed = Timeframe.M5.last_closed_open(datetime.now(UTC))
    origin = last_closed - timedelta(minutes=5 * 100)
    bars = make_bars(101, start=origin)  # last bar's open == last_closed_open(now)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars)
    service = CandleHistoryService(market_data, repository)

    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=None)

    assert result == bars[-100:]
    assert market_data.calls == []  # never touched the gateway


async def test_get_candles_falls_through_to_gateway_when_db_is_stale():
    """DB hasn't caught up to the latest closed bar yet (e.g. symbol just
    opened on a chart, first poll hasn't run) — must not serve stale data,
    falls back to the live gateway like before this DB-first optimization."""
    last_closed = Timeframe.M5.last_closed_open(datetime.now(UTC))
    origin = last_closed - timedelta(minutes=5 * 100)
    bars = make_bars(101, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars[:-1])  # missing the most recent bar

    service = CandleHistoryService(market_data, repository)

    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=None)

    assert result == bars[-100:]
    assert market_data.calls == [None]


async def test_get_candles_serves_full_page_from_db_when_paging_older_history():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars)
    service = CandleHistoryService(market_data, repository)

    cursor = bars[150].time
    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=cursor)

    assert result == repository.get_before("XAUUSD", Timeframe.M5, cursor, 100)
    assert market_data.calls == []


async def test_get_candles_hits_gateway_when_db_page_is_short():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)
    market_data = FakePagingMarketData(bars)
    # DB only holds a handful of bars near the cursor — short of a full page.
    repository = FakeCandleRepository(bars[145:155])
    service = CandleHistoryService(market_data, repository)

    cursor = bars[150].time
    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=cursor)

    assert market_data.calls == [cursor]
    assert result == bars[:150][-100:]


async def test_get_candles_falls_through_to_gateway_when_db_has_internal_gap():
    """`cached[-1]` looks fresh and the DB returned a full page, but a hole
    was left in the middle by e.g. a gateway outage mid-session that ended
    before the current poll caught back up — `get_latest`'s plain
    `ORDER BY time DESC LIMIT` can't see that on its own. Must not serve it
    as if the history were contiguous."""
    last_closed = Timeframe.M5.last_closed_open(datetime.now(UTC))
    newer_half = make_bars(50, start=last_closed - timedelta(minutes=5 * 49))
    older_half = make_bars(50, start=newer_half[0].time - timedelta(days=4, minutes=5 * 49))
    bars = older_half + newer_half  # 4-day hole between the two halves

    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars)
    service = CandleHistoryService(market_data, repository)

    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=None)

    assert result == bars[-100:]
    assert market_data.calls == [None]  # fell through instead of trusting the gapped cache


async def test_get_candles_hits_gateway_when_db_page_has_internal_gap():
    """Same gap check applied to the `before`-cursor paging branch — a full
    `count`-sized page from the DB isn't enough if it has a hole in it."""
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    newer_half = make_bars(50, start=origin + timedelta(days=10))
    older_half = make_bars(50, start=origin)  # >4-day gap before newer_half
    bars = older_half + newer_half

    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars)
    service = CandleHistoryService(market_data, repository)

    cursor = bars[-1].time + timedelta(minutes=5)
    await service.get_candles("XAUUSD", Timeframe.M5, 100, before=cursor)

    assert market_data.calls == [cursor]


async def test_get_candles_falls_back_to_db_when_gateway_unavailable():
    now = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    origin = now - timedelta(minutes=5 * 100)
    bars = make_bars(101, start=origin)
    repository = FakeCandleRepository(bars[:-1])  # stale, forces the gateway attempt
    service = CandleHistoryService(FakeUnavailableMarketData(), repository)

    result = await service.get_candles("XAUUSD", Timeframe.M5, 100, before=None)

    assert result == bars[:-1][-100:]


async def test_reconcile_gaps_backfills_when_gap_exceeds_poll_lookback():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin, timeframe=Timeframe.M5)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars[:100])  # missing bars[100:250] — a 150-bar hole
    service = CandleHistoryService(market_data, repository)

    now = bars[-1].time + timedelta(minutes=5)
    stored = await service.reconcile_gaps(
        ["XAUUSD"], [Timeframe.M5], poll_lookback=lambda _tf: 20, count=100, now=now
    )

    assert stored["XAUUSD:M5"] > 0
    assert set(bars[100:250]) <= set(repository.stored)


async def test_reconcile_gaps_noop_when_gap_within_poll_lookback():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(30, start=origin, timeframe=Timeframe.M5)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository(bars)
    service = CandleHistoryService(market_data, repository)

    now = bars[-1].time + timedelta(minutes=5 * 10)  # 10-bar gap, under poll_lookback's 20
    stored = await service.reconcile_gaps(
        ["XAUUSD"], [Timeframe.M5], poll_lookback=lambda _tf: 20, now=now
    )

    assert stored == {}
    assert market_data.calls == []


async def test_reconcile_gaps_noop_when_no_stored_history():
    market_data = FakePagingMarketData([])
    repository = FakeCandleRepository()

    service = CandleHistoryService(market_data, repository)

    stored = await service.reconcile_gaps(["XAUUSD"], [Timeframe.M5], poll_lookback=lambda _tf: 20)

    assert stored == {}
    assert market_data.calls == []


async def test_reconcile_gaps_swallows_gateway_unavailable_per_pair():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(150, start=origin, timeframe=Timeframe.M5)
    repository = FakeCandleRepository(bars[:50])
    service = CandleHistoryService(FakeUnavailableMarketData(), repository)

    now = bars[-1].time + timedelta(minutes=5)
    stored = await service.reconcile_gaps(
        ["XAUUSD"], [Timeframe.M5], poll_lookback=lambda _tf: 20, now=now
    )

    assert stored == {}


async def test_backfill_without_start_calls_enrich_missing():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository()
    service = CandleHistoryService(market_data, repository)

    await service.backfill("XAUUSD", Timeframe.M5, 100)

    assert repository.enrich_calls == [("XAUUSD", Timeframe.M5, "default")]


async def test_backfill_with_start_calls_enrich_missing():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(250, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository()
    service = CandleHistoryService(market_data, repository)

    await service.backfill("XAUUSD", Timeframe.M5, 100, start=origin)

    # Wired once, after the paged loop completes (not once per page).
    assert repository.enrich_calls == [("XAUUSD", Timeframe.M5, "default")]


async def test_backfill_with_start_stops_when_broker_history_runs_out():
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    bars = make_bars(50, start=origin)
    market_data = FakePagingMarketData(bars)
    repository = FakeCandleRepository()
    service = CandleHistoryService(market_data, repository)

    # Requested start is far earlier than any real history — pagination must
    # stop once the broker returns a short (final) page, not loop forever.
    stored = await service.backfill("XAUUSD", Timeframe.M5, 100, start=origin - timedelta(days=365))

    assert stored == 50
    assert len(market_data.calls) == 1


# ── Gap scanning & repair (chart's "fill gaps" button) ──────────────────────
#
# A hole in stored history is invisible to get_latest/get_before, so the chart
# draws straight across it and every indicator/backtest over that window
# treats bars hours apart as adjacent. These cover finding those holes and
# closing them from the broker.


def holed_history(
    *, origin: datetime, before: int, missing: int, after: int
) -> tuple[list[Candle], list[Candle]]:
    """`(full, holed)` M5 histories — `holed` is `full` minus `missing` bars
    starting at index `before`, i.e. what a stream outage leaves behind."""
    full = make_bars(before + missing + after, start=origin)
    return full, full[:before] + full[before + missing :]


async def test_scan_gaps_reports_a_hole_between_stored_bars():
    origin = datetime(2026, 1, 5, tzinfo=UTC)  # a Monday
    full, holed = holed_history(origin=origin, before=10, missing=10, after=10)
    service = CandleHistoryService(FakeUnavailableMarketData(), FakeCandleRepository(holed))

    gaps = await service.scan_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert len(gaps) == 1
    assert gaps[0].missing_bars == 10
    assert gaps[0].start == full[10].time
    assert gaps[0].end == full[20].time
    assert gaps[0].weekend is False


async def test_scan_gaps_reports_the_whole_range_when_nothing_is_stored():
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full = make_bars(30, start=origin)
    service = CandleHistoryService(FakeUnavailableMarketData(), FakeCandleRepository([]))

    gaps = await service.scan_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert [gap.missing_bars for gap in gaps] == [29]


async def test_scan_gaps_reports_history_missing_before_the_range_starts():
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full = make_bars(30, start=origin)
    service = CandleHistoryService(FakeUnavailableMarketData(), FakeCandleRepository(full[10:]))

    gaps = await service.scan_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert [gap.missing_bars for gap in gaps] == [10]
    assert gaps[0].start == full[0].time


async def test_scan_gaps_tolerates_the_db_trailing_the_live_stream_by_a_bar():
    """The chart's newest bar is the forming one; the stream persists it on
    its own ~1/minute cadence, so a bar of lag is normal and must not light
    up a permanent "gaps found" badge."""
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full = make_bars(30, start=origin)
    service = CandleHistoryService(FakeUnavailableMarketData(), FakeCandleRepository(full[:-1]))

    assert await service.scan_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time) == []


async def test_scan_gaps_without_a_repository_is_empty():
    service = CandleHistoryService(FakeUnavailableMarketData(), repository=None)
    origin = datetime(2026, 1, 5, tzinfo=UTC)

    assert await service.scan_gaps("XAUUSD", Timeframe.M5, origin, origin + timedelta(days=1)) == []


async def test_repair_gaps_downloads_the_hole_and_closes_it():
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full, holed = holed_history(origin=origin, before=10, missing=10, after=10)
    market_data = FakePagingMarketData(full)
    service = CandleHistoryService(market_data, FakeCandleRepository(holed))

    report = await service.repair_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert len(report.found) == 1
    assert len(report.repaired) == 1
    assert report.remaining == []
    assert report.bars_recovered == 10
    assert report.bars_downloaded > 0


async def test_repair_gaps_asks_only_for_the_missing_stretch():
    """One backward-paged request per hole, cursored at the hole's end — not
    a re-download of the whole scanned window."""
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full, holed = holed_history(origin=origin, before=10, missing=10, after=10)
    market_data = FakePagingMarketData(full)
    service = CandleHistoryService(market_data, FakeCandleRepository(holed))

    await service.repair_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert market_data.calls == [full[20].time]


async def test_repair_gaps_reports_what_the_broker_cannot_fill():
    """The broker's own history has the same hole — a holiday, a halt, or a
    symbol listed later. Retrying will never fix it, so say so instead of
    inviting another click."""
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    _full, holed = holed_history(origin=origin, before=10, missing=10, after=10)
    service = CandleHistoryService(FakePagingMarketData(holed), FakeCandleRepository(holed))

    report = await service.repair_gaps("XAUUSD", Timeframe.M5, holed[0].time, holed[-1].time)

    assert report.repaired == []
    assert len(report.remaining) == 1
    assert report.bars_recovered == 0


async def test_repair_gaps_skips_weekend_closures_by_default():
    """Friday 20:59 to Sunday 22:00 on M1: the broker never had those bars,
    so asking for them burns a Wine/MT5 round trip for nothing."""
    friday = datetime(2026, 7, 17, 20, 59, tzinfo=UTC)
    stored = [
        make_bars(1, start=friday, timeframe=Timeframe.M1)[0],
        make_bars(1, start=friday + timedelta(days=2, hours=1, minutes=1), timeframe=Timeframe.M1)[
            0
        ],
    ]
    market_data = FakePagingMarketData(stored)
    service = CandleHistoryService(market_data, FakeCandleRepository(stored))

    report = await service.repair_gaps("XAUUSD", Timeframe.M1, stored[0].time, stored[-1].time)

    assert [gap.weekend for gap in report.found] == [True]
    assert report.repaired == []
    assert report.remaining == []
    assert market_data.calls == []


async def test_repair_gaps_asks_for_weekend_holes_when_told_to():
    """Crypto and synthetic indices really do trade through the weekend."""
    friday = datetime(2026, 7, 17, 20, 59, tzinfo=UTC)
    stored = [
        make_bars(1, start=friday, timeframe=Timeframe.M1)[0],
        make_bars(1, start=friday + timedelta(days=2, hours=1, minutes=1), timeframe=Timeframe.M1)[
            0
        ],
    ]
    market_data = FakePagingMarketData(stored)
    service = CandleHistoryService(market_data, FakeCandleRepository(stored))

    await service.repair_gaps(
        "BTCUSD", Timeframe.M1, stored[0].time, stored[-1].time, include_weekend=True
    )

    assert market_data.calls == [stored[-1].time]


async def test_repair_gaps_leaves_a_healthy_range_alone():
    origin = datetime(2026, 1, 5, tzinfo=UTC)
    full = make_bars(30, start=origin)
    market_data = FakePagingMarketData(full)
    service = CandleHistoryService(market_data, FakeCandleRepository(full))

    report = await service.repair_gaps("XAUUSD", Timeframe.M5, full[0].time, full[-1].time)

    assert report.found == []
    assert report.bars_downloaded == 0
    assert market_data.calls == []
