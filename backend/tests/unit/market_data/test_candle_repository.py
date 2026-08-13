from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.domain.models import Timeframe
from src.shared.db.base import Base
from src.shared.domain.indicators import atr as shared_atr


@pytest.fixture
def repository(tmp_path) -> CandleRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return CandleRepository(sessionmaker(bind=engine, expire_on_commit=False))


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_roundtrip_preserves_candle(repository, candle_factory):
    candle = candle_factory(utc(2026, 7, 10, 14, 0))
    assert repository.upsert_many([candle]) == 1
    assert repository.get_latest("XAUUSD", Timeframe.M5, 10) == [candle]


def test_roundtrip_preserves_real_volume(repository, candle_factory):
    candle = candle_factory(utc(2026, 7, 10, 14, 0), real_volume=1234)
    assert repository.upsert_many([candle]) == 1

    (stored,) = repository.get_latest("XAUUSD", Timeframe.M5, 10)
    assert stored.real_volume == 1234


def test_upsert_refreshes_real_volume_on_conflict(repository, candle_factory):
    time = utc(2026, 7, 10, 14, 0)
    repository.upsert_many([candle_factory(time, real_volume=100)])
    repository.upsert_many([candle_factory(time, real_volume=200)])

    (stored,) = repository.get_latest("XAUUSD", Timeframe.M5, 10)
    assert stored.real_volume == 200


def test_upsert_overwrites_same_bar(repository, candle_factory):
    time = utc(2026, 7, 10, 14, 0)
    repository.upsert_many([candle_factory(time, close=2400.5)])
    repository.upsert_many([candle_factory(time, close=2410.0)])

    (stored,) = repository.get_latest("XAUUSD", Timeframe.M5, 10)
    assert stored.close == 2410.0


def test_get_latest_returns_most_recent_oldest_first(repository, candle_factory):
    times = [utc(2026, 7, 10, 14, m) for m in (0, 5, 10, 15)]
    repository.upsert_many([candle_factory(t) for t in times])

    stored = repository.get_latest("XAUUSD", Timeframe.M5, 2)
    assert [c.time for c in stored] == times[-2:]


def test_get_before_returns_bars_strictly_before_cutoff_oldest_first(repository, candle_factory):
    times = [utc(2026, 7, 10, 14, m) for m in (0, 5, 10, 15)]
    repository.upsert_many([candle_factory(t) for t in times])

    stored = repository.get_before("XAUUSD", Timeframe.M5, utc(2026, 7, 10, 14, 10), 10)
    assert [c.time for c in stored] == times[:2]


def test_get_before_respects_count(repository, candle_factory):
    times = [utc(2026, 7, 10, 14, m) for m in (0, 5, 10, 15)]
    repository.upsert_many([candle_factory(t) for t in times])

    stored = repository.get_before("XAUUSD", Timeframe.M5, utc(2026, 7, 10, 14, 15), 2)
    assert [c.time for c in stored] == times[1:3]


def test_get_range_bounds_are_start_inclusive_end_exclusive(repository, candle_factory):
    times = [utc(2026, 7, 10, 14, m) for m in (0, 5, 10, 15)]
    repository.upsert_many([candle_factory(t) for t in times])

    stored = repository.get_range(
        "XAUUSD", Timeframe.M5, utc(2026, 7, 10, 14, 5), utc(2026, 7, 10, 14, 15)
    )
    assert [c.time for c in stored] == times[1:3]


def test_get_range_returns_oldest_first(repository, candle_factory):
    times = [utc(2026, 7, 10, 14, m) for m in (10, 0, 5)]
    repository.upsert_many([candle_factory(t) for t in times])

    stored = repository.get_range(
        "XAUUSD", Timeframe.M5, utc(2026, 7, 10, 13, 0), utc(2026, 7, 10, 15, 0)
    )
    assert [c.time for c in stored] == sorted(times)


def test_symbols_and_timeframes_are_isolated(repository, candle_factory):
    time = utc(2026, 7, 10, 14, 0)
    repository.upsert_many(
        [
            candle_factory(time, symbol="XAUUSD"),
            candle_factory(time, symbol="BTCUSD"),
            candle_factory(utc(2026, 7, 10, 14, 0), timeframe=Timeframe.H1),
        ]
    )
    assert len(repository.get_latest("XAUUSD", Timeframe.M5, 10)) == 1
    assert len(repository.get_latest("BTCUSD", Timeframe.M5, 10)) == 1
    assert len(repository.get_latest("XAUUSD", Timeframe.H1, 10)) == 1


def test_bars_are_keyed_per_account(repository, candle_factory):
    """Different brokers quote different spreads/prices for a nominally
    identical symbol (e.g. `XAUUSD` vs `XAUUSD.a`) — a shared cache keyed
    only on (symbol, timeframe, time) would silently mix them."""
    time = utc(2026, 7, 10, 14, 0)
    repository.upsert_many([candle_factory(time, close=2400.5)], account_id="ftmo-1")
    repository.upsert_many([candle_factory(time, close=2410.0)], account_id="ftmo-2")

    (ftmo_1,) = repository.get_latest("XAUUSD", Timeframe.M5, 10, account_id="ftmo-1")
    (ftmo_2,) = repository.get_latest("XAUUSD", Timeframe.M5, 10, account_id="ftmo-2")
    assert ftmo_1.close == 2400.5
    assert ftmo_2.close == 2410.0
    assert repository.get_latest("XAUUSD", Timeframe.M5, 10, account_id="default") == []


def test_upsert_writes_atr_and_day_of_week_as_null(repository, candle_factory):
    candle = candle_factory(utc(2026, 8, 3, 14, 0))
    repository.upsert_many([candle])

    (stored,) = repository.get_latest("XAUUSD", Timeframe.M5, 10)
    assert stored.atr_14 is None
    assert stored.day_of_week is None


def test_enrich_missing_with_no_stored_candles_returns_zero(repository):
    assert repository.enrich_missing("XAUUSD", Timeframe.M5) == 0


def test_enrich_missing_fills_trailing_window_and_is_idempotent(repository, candle_factory):
    """Constant-range OHLC (high - low always dominates the gap-based terms
    of true range) keeps the hand-computed expectation simple: true range is
    3.0 on every bar, so ATR is exactly 3.0 wherever a full `period`-bar
    trailing window exists, and NaN (left NULL) everywhere it doesn't."""
    period = 14
    n_first = period + 5  # 19 bars: first 13 never get a full window, last 6 do
    start = utc(2026, 8, 3, 0, 0)  # a Monday

    def bar(i: int) -> tuple[datetime, float, float, float]:
        low = 2400.0 + i * 0.9
        return start + i * timedelta(minutes=5), low + 3.0, low, low + 1.5

    first_batch = []
    for i in range(n_first):
        t, high, low, close = bar(i)
        first_batch.append(candle_factory(t, high=high, low=low, close=close))
    repository.upsert_many(first_batch)

    updated_first = repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=period)
    assert updated_first == n_first - (period - 1)

    # Nothing new to enrich yet — the leading (period - 1) bars can never
    # get a full trailing window from this history alone.
    assert repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=period) == 0

    # A second batch (simulating the next poll's upsert) — these have real
    # trailing context now (the first batch's tail), so all of them should
    # get enriched in one pass.
    n_second = 6
    second_batch = []
    for i in range(n_first, n_first + n_second):
        t, high, low, close = bar(i)
        second_batch.append(candle_factory(t, high=high, low=low, close=close))
    repository.upsert_many(second_batch)

    updated_second = repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=period)
    assert updated_second == n_second

    all_bars = [bar(i) for i in range(n_first + n_second)]
    highs = np.array([b[1] for b in all_bars])
    lows = np.array([b[2] for b in all_bars])
    closes = np.array([b[3] for b in all_bars])
    expected_atr = shared_atr(highs, lows, closes, period).to_numpy()

    stored = repository.get_latest("XAUUSD", Timeframe.M5, n_first + n_second)
    for candle, expected in zip(stored, expected_atr, strict=True):
        if np.isnan(expected):
            assert candle.atr_14 is None
            assert candle.day_of_week is None
        else:
            assert candle.atr_14 == pytest.approx(expected)
            assert candle.day_of_week == candle.time.weekday()

    # Idempotent: calling again with nothing left to enrich is a no-op.
    assert repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=period) == 0


def test_enrich_missing_never_overwrites_already_enriched_row(repository, candle_factory):
    time = utc(2026, 8, 3, 14, 0)
    repository.upsert_many([candle_factory(time, high=2402.0, low=2399.0, close=2400.5)])
    repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=1)
    (before,) = repository.get_latest("XAUUSD", Timeframe.M5, 1)
    assert before.atr_14 is not None

    # Re-upserting the same bar (e.g. a forming-bar overwrite) resets it to
    # NULL per `upsert_many`'s contract — enrich_missing should then pick it
    # back up rather than leaving it stuck, but should never blindly clobber
    # a row that still holds a value from a previous enrichment pass while
    # doing so.
    repository.upsert_many([candle_factory(time, high=2402.0, low=2399.0, close=2400.5)])
    (reset,) = repository.get_latest("XAUUSD", Timeframe.M5, 1)
    assert reset.atr_14 is None

    updated = repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=1)
    assert updated == 1
    (after,) = repository.get_latest("XAUUSD", Timeframe.M5, 1)
    assert after.atr_14 == pytest.approx(3.0)
