import time
from datetime import UTC, datetime, timedelta

import pytest

from src.market_data.application.candle_stream import CandleStreamService, poll_lookback_for
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import MarketDataUnavailable, Timeframe
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed


class FakeMarketData:
    """Serves whatever candles the test sets; last one may be forming.
    `candles[key]` doubles as the full ascending-by-time gateway history for
    `before`-cursor paging (mirrors `FakePagingMarketData` in
    test_history.py) — a plain `count`-only call returns its tail, a `before`
    call pages backward through it, same contract as the real gateway."""

    def __init__(self):
        self.candles: dict[tuple[str, Timeframe], list] = {}
        self.backfill_calls: list[tuple] = []

    async def get_candles(self, symbol, timeframe, count, before=None):
        bars = self.candles.get((symbol, timeframe), [])
        if before is None:
            return bars[-count:]
        self.backfill_calls.append((symbol, timeframe, count, before))
        cutoff = next((i for i, c in enumerate(bars) if c.time >= before), len(bars))
        return bars[:cutoff][-count:]


class FakeRepository:
    def __init__(self):
        self.stored = []
        self.enrich_calls: list[tuple] = []

    def upsert_many(self, candles, account_id: str = "default"):
        self.stored.extend(candles)
        return len(list(candles))

    def enrich_missing(self, symbol, timeframe, account_id: str = "default", **kwargs) -> int:
        self.enrich_calls.append((symbol, timeframe, account_id))
        return 0


class FakeBroadcaster:
    def __init__(self):
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture
def setup(candle_factory):
    market_data = FakeMarketData()
    repository = FakeRepository()
    broadcaster = FakeBroadcaster()
    bus = EventBus()
    events: list[CandleClosed] = []

    async def capture(event):
        events.append(event)

    bus.subscribe(CandleClosed, capture)
    service = CandleStreamService(
        market_data=market_data,
        repository=repository,
        event_bus=bus,
        broadcaster=broadcaster,
        symbols=["XAUUSD"],
        timeframes=[Timeframe.M5],
    )
    return service, market_data, repository, broadcaster, events


async def test_first_poll_baselines_without_emitting(setup, candle_factory):
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
        candle_factory(utc(2026, 7, 10, 14, 0)),  # forming at 14:03
    ]

    emitted = await service.poll_once(now=utc(2026, 7, 10, 14, 3))

    assert emitted == []
    assert events == []
    # ...but closed history is persisted right away.
    assert [c.time for c in repository.stored] == [utc(2026, 7, 10, 13, 55)]


async def test_new_closed_bar_is_emitted_persisted_broadcast(setup, candle_factory):
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
        candle_factory(utc(2026, 7, 10, 14, 0), close=2405.0),
    ]
    emitted = await service.poll_once(now=utc(2026, 7, 10, 14, 5, 2))

    assert [c.time for c in emitted] == [utc(2026, 7, 10, 14, 0)]
    assert events == [
        CandleClosed(symbol="XAUUSD", timeframe="M5", occurred_at=events[0].occurred_at)
    ]
    (message,) = broadcaster.messages
    assert message["type"] == "candle_closed"
    assert message["candle"]["close"] == 2405.0
    assert message["candle"]["time"] == int(utc(2026, 7, 10, 14, 0).timestamp())


async def test_poll_once_calls_enrich_missing_for_polled_symbol(setup, candle_factory):
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
    ]

    await service.poll_once(now=utc(2026, 7, 10, 14, 3))

    assert repository.enrich_calls == [("XAUUSD", Timeframe.M5, "default")]


async def test_poll_once_populates_recent_candle_cache_with_latest_bar(candle_factory):
    # Shared with LiveCandleService (OPTIMIZATION_CHECKLIST.md §2) so its own
    # ~1.5s poll can reuse this fetch instead of double-hitting the gateway
    # for the same symbol/timeframe right at a bar close.
    market_data = FakeMarketData()
    repository = FakeRepository()
    broadcaster = FakeBroadcaster()
    bus = EventBus()
    cache: dict = {}
    service = CandleStreamService(
        market_data=market_data,
        repository=repository,
        event_bus=bus,
        broadcaster=broadcaster,
        symbols=["XAUUSD"],
        timeframes=[Timeframe.M5],
        recent_candle_cache=cache,
    )
    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
        candle_factory(utc(2026, 7, 10, 14, 0)),  # still forming at 14:03
    ]

    before = time.monotonic()
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))

    fetched_at, cached_candle = cache[("XAUUSD", Timeframe.M5)]
    assert fetched_at >= before
    # The cache holds the raw fetch's last element (the still-forming bar) —
    # exactly what LiveCandleService's own count=1 poll wants, not just the
    # last *closed* bar.
    assert cached_candle.time == utc(2026, 7, 10, 14, 0)


async def test_same_bar_is_not_emitted_twice(setup, candle_factory):
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 14, 0)),
    ]
    await service.poll_once(now=utc(2026, 7, 10, 14, 5, 2))
    again = await service.poll_once(now=utc(2026, 7, 10, 14, 6))

    assert again == []
    assert len(events) == 1


async def test_forming_bar_is_never_emitted(setup, candle_factory):
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
        candle_factory(utc(2026, 7, 10, 14, 0)),  # still forming at 14:04
    ]
    emitted = await service.poll_once(now=utc(2026, 7, 10, 14, 4))

    assert emitted == []
    assert all(c.time != utc(2026, 7, 10, 14, 0) for c in repository.stored)


def test_poll_scheduling_targets_just_after_finest_boundary(setup):
    service, *_ = setup  # setup's fixture timeframes=[Timeframe.M5]
    delay = service._seconds_until_next_poll(utc(2026, 7, 10, 14, 3, 30))
    assert delay == pytest.approx(90 + 2.0)
    # Right after a boundary, wait for the next one — no tight loop.
    delay = service._seconds_until_next_poll(utc(2026, 7, 10, 14, 5, 0))
    assert delay >= 5.0


def test_poll_scheduling_uses_finest_configured_timeframe():
    service = CandleStreamService(
        market_data=FakeMarketData(),
        repository=FakeRepository(),
        event_bus=EventBus(),
        broadcaster=FakeBroadcaster(),
        symbols=["XAUUSD"],
        timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.H1],
    )
    # M1 (60s) is the finest timeframe, so the next boundary is at 14:04:00.
    delay = service._seconds_until_next_poll(utc(2026, 7, 10, 14, 3, 30))
    assert delay == pytest.approx(30 + 2.0)


def test_poll_lookback_scales_up_for_fast_timeframes():
    # A flat 20-bar lookback was a 20-minute buffer on M1 — too thin to
    # bridge even a short gateway hiccup without falling into the explicit
    # gap-backfill path every time.
    assert poll_lookback_for(Timeframe.M1) > 20
    assert poll_lookback_for(Timeframe.M5) >= 20


def test_poll_lookback_scales_down_for_slow_timeframes():
    # A flat 20-bar lookback was 20 days of D1 (or ~4.6 months of W1) fetched
    # on every poll tick for no benefit — floored, not zero, so there's still
    # a little overlap to catch a repainted/corrected bar.
    for timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1, Timeframe.MN):
        assert 1 <= poll_lookback_for(timeframe) < 20


def test_poll_lookback_never_exceeds_engine_context_bars_cap():
    for timeframe in Timeframe:
        assert poll_lookback_for(timeframe) <= 200


def test_watch_adds_ad_hoc_symbol_to_active_symbols(setup):
    service, *_ = setup
    assert service.active_symbols == ["XAUUSD"]

    service.watch("EURUSD")

    assert service.active_symbols == ["XAUUSD", "EURUSD"]


def test_watch_is_noop_for_already_configured_symbol(setup):
    service, *_ = setup

    service.watch("XAUUSD")

    assert service.active_symbols == ["XAUUSD"]


def test_unwatch_removes_symbol_once_last_watcher_leaves(setup):
    service, *_ = setup
    service.watch("EURUSD")  # e.g. two browser tabs charting the same symbol
    service.watch("EURUSD")

    service.unwatch("EURUSD")
    assert service.active_symbols == ["XAUUSD", "EURUSD"]

    service.unwatch("EURUSD")
    assert service.active_symbols == ["XAUUSD"]


def test_unwatch_never_removes_a_configured_symbol(setup):
    service, *_ = setup

    service.unwatch("XAUUSD")

    assert service.active_symbols == ["XAUUSD"]


async def test_ad_hoc_watched_symbol_is_polled_and_broadcast(setup, candle_factory):
    # The bug this guards against: a chart browsing a symbol outside
    # configs/app.yaml (e.g. via the symbol picker) must still receive live
    # candle_closed updates for as long as it's being watched, not just the
    # engine's configured universe.
    service, market_data, repository, broadcaster, events = setup
    service.watch("EURUSD")
    market_data.candles[("EURUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55), symbol="EURUSD")
    ]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    market_data.candles[("EURUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55), symbol="EURUSD"),
        candle_factory(utc(2026, 7, 10, 14, 0), symbol="EURUSD", close=1.09),
    ]
    emitted = await service.poll_once(now=utc(2026, 7, 10, 14, 5, 2))

    assert [c.time for c in emitted] == [utc(2026, 7, 10, 14, 0)]
    assert any(m["candle"]["symbol"] == "EURUSD" for m in broadcaster.messages)


def test_add_symbol_permanently_activates_it(setup):
    service, *_ = setup

    added = service.add_symbol("EURUSD")

    assert added is True
    assert service.active_symbols == ["XAUUSD", "EURUSD"]


def test_add_symbol_is_idempotent(setup):
    service, *_ = setup
    service.add_symbol("EURUSD")

    added_again = service.add_symbol("EURUSD")

    assert added_again is False
    assert service.active_symbols == ["XAUUSD", "EURUSD"]


def test_add_symbol_is_noop_for_already_permanent_symbol(setup):
    service, *_ = setup

    added = service.add_symbol("XAUUSD")

    assert added is False
    assert service.active_symbols == ["XAUUSD"]


def test_add_symbol_folds_an_ad_hoc_watch_into_permanent(setup):
    # A chart tab was watching EURUSD (ref-counted); activating it for
    # automated trading should promote it, not double-track it, and a later
    # unwatch() from that same chart tab must still no-op cleanly rather than
    # corrupting a refcount that no longer exists.
    service, *_ = setup
    service.watch("EURUSD")

    added = service.add_symbol("EURUSD")

    assert added is True
    assert service.active_symbols == ["XAUUSD", "EURUSD"]
    service.unwatch("EURUSD")
    assert service.active_symbols == ["XAUUSD", "EURUSD"]


async def test_poll_once_heals_gap_left_by_mid_session_outage(candle_factory):
    # The bug this guards against: the gateway/broker connection drops for
    # longer than poll_lookback_for(timeframe) bars while the backend process
    # itself keeps running (no restart) — CandleHistoryService.reconcile_gaps
    # only ever runs once at startup and never gets a chance to see this.
    # Without a fix, the hole between the last bar seen before the outage and
    # `now - poll_lookback_for(timeframe) bars` is permanent.
    market_data = FakeMarketData()
    repository = FakeRepository()
    broadcaster = FakeBroadcaster()
    bus = EventBus()
    candle_history = CandleHistoryService(market_data, repository)
    service = CandleStreamService(
        market_data=market_data,
        repository=repository,
        event_bus=bus,
        broadcaster=broadcaster,
        symbols=["XAUUSD"],
        timeframes=[Timeframe.M5],
        candle_history=candle_history,
    )
    key = ("XAUUSD", Timeframe.M5)

    market_data.candles[key] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline, previous=13:55

    # ~6 hours pass with the gateway unreachable; once it recovers, the
    # broker's full history (including every bar missed during the outage)
    # is available again via `before`-cursor paging.
    all_bars = [
        candle_factory(utc(2026, 7, 10, 13, 55) + timedelta(minutes=5 * i)) for i in range(73)
    ]  # 13:55 .. 19:55, every 5 minutes
    market_data.candles[key] = all_bars

    recovery_now = all_bars[-1].time + timedelta(minutes=5, seconds=2)
    emitted = await service.poll_once(now=recovery_now)

    # The gap between the last-seen bar and the recovery poll's fetch window
    # exceeds poll_lookback_for(M5) bars, so a targeted backfill must have
    # run and every bar missed during the outage must end up persisted — not
    # just the tail the normal poll window would have covered.
    stored_times = {c.time for c in repository.stored}
    assert all(bar.time in stored_times for bar in all_bars)
    # The tail still gets emitted as a normal live close, same as any poll.
    assert emitted


async def test_poll_once_gap_backfill_failure_does_not_crash_the_poll(candle_factory):
    # The gateway can flake again mid-backfill (e.g. still recovering) —
    # that must not blow up the poll that's healing the gap; it should just
    # log and let the normal tail-emit logic continue, retrying the backfill
    # on a later tick.
    class FlakyCandleHistory:
        async def backfill(self, symbol, timeframe, count, start=None):
            raise MarketDataUnavailable("still down")

    market_data = FakeMarketData()
    repository = FakeRepository()
    broadcaster = FakeBroadcaster()
    bus = EventBus()
    service = CandleStreamService(
        market_data=market_data,
        repository=repository,
        event_bus=bus,
        broadcaster=broadcaster,
        symbols=["XAUUSD"],
        timeframes=[Timeframe.M5],
        candle_history=FlakyCandleHistory(),
    )
    key = ("XAUUSD", Timeframe.M5)

    market_data.candles[key] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    all_bars = [
        candle_factory(utc(2026, 7, 10, 13, 55) + timedelta(minutes=5 * i)) for i in range(73)
    ]
    market_data.candles[key] = all_bars
    recovery_now = all_bars[-1].time + timedelta(minutes=5, seconds=2)

    emitted = await service.poll_once(now=recovery_now)  # must not raise

    assert emitted  # tail still emitted despite the failed backfill attempt


async def test_multiple_closed_bars_since_last_poll_are_all_emitted(setup, candle_factory):
    # If a poll is delayed, more than one bar of a fine timeframe (e.g. M1)
    # can close in between — every one of them must still be emitted, not
    # just the newest.
    service, market_data, repository, broadcaster, events = setup
    market_data.candles[("XAUUSD", Timeframe.M5)] = [candle_factory(utc(2026, 7, 10, 13, 55))]
    await service.poll_once(now=utc(2026, 7, 10, 14, 3))  # baseline

    market_data.candles[("XAUUSD", Timeframe.M5)] = [
        candle_factory(utc(2026, 7, 10, 13, 55)),
        candle_factory(utc(2026, 7, 10, 14, 0), close=2405.0),
        candle_factory(utc(2026, 7, 10, 14, 5), close=2410.0),
    ]
    emitted = await service.poll_once(now=utc(2026, 7, 10, 14, 10, 2))

    assert [c.time for c in emitted] == [utc(2026, 7, 10, 14, 0), utc(2026, 7, 10, 14, 5)]
    assert [e.occurred_at for e in events] and len(events) == 2
    assert len(broadcaster.messages) == 2
