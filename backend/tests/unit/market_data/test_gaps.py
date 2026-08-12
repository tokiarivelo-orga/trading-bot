"""Candle-history gap detection (`domain/gaps.py`).

Reference week: 2026-07-17 is a Friday, 2026-07-18 Saturday, 2026-07-19
Sunday, 2026-07-20 Monday — every weekend assertion below is anchored to it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.market_data.domain.gaps import bars_between, find_gaps, is_weekend_closure
from src.market_data.domain.models import Candle, Timeframe

FRIDAY = datetime(2026, 7, 17, tzinfo=UTC)


def bars(times: list[datetime], timeframe: Timeframe = Timeframe.M5) -> list[Candle]:
    return [
        Candle(
            symbol="XAUUSD",
            timeframe=timeframe,
            time=t,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            tick_volume=1,
            spread_points=1,
        )
        for t in times
    ]


def series(count: int, *, start: datetime, timeframe: Timeframe = Timeframe.M5) -> list[Candle]:
    return bars([start + i * timedelta(seconds=timeframe.seconds) for i in range(count)], timeframe)


def test_no_gaps_in_contiguous_history():
    assert find_gaps(series(50, start=FRIDAY.replace(hour=8)), Timeframe.M5) == []


def test_finds_hole_between_stored_bars():
    start = FRIDAY.replace(hour=8)
    before = series(10, start=start)
    # Two hours of M5 bars (24 of them) never made it into the DB.
    after = series(10, start=before[-1].time + timedelta(hours=2, seconds=300))

    gaps = find_gaps(before + after, Timeframe.M5)

    assert len(gaps) == 1
    assert gaps[0].start == before[-1].time + timedelta(minutes=5)
    assert gaps[0].end == after[0].time
    assert gaps[0].missing_bars == 24
    assert gaps[0].weekend is False


def test_finds_every_hole_in_one_pass():
    start = FRIDAY.replace(hour=8)
    first = series(5, start=start)
    second = series(5, start=first[-1].time + timedelta(minutes=30))
    third = series(5, start=second[-1].time + timedelta(minutes=15))

    gaps = find_gaps(first + second + third, Timeframe.M5)

    assert [gap.missing_bars for gap in gaps] == [5, 2]


def test_friday_close_to_sunday_reopen_is_a_weekend_closure():
    """The M1 stream stops at Friday 20:59 UTC and resumes Sunday 22:00 —
    ~49h of missing bars that the broker never had. Reported, but flagged so
    a repair doesn't waste a round trip asking for them."""
    stored = bars(
        [FRIDAY.replace(hour=20, minute=59), FRIDAY + timedelta(days=2, hours=22)], Timeframe.M1
    )

    gaps = find_gaps(stored, Timeframe.M1)

    assert len(gaps) == 1
    assert gaps[0].weekend is True


def test_daily_bars_across_the_weekend_are_a_weekend_closure():
    """D1: Friday's bar is followed by Monday's, leaving Saturday and Sunday
    missing. `end` is Monday — past the weekend window — so only the last
    *missing* bar's open (Sunday) may be compared against it."""
    gaps = find_gaps(bars([FRIDAY, FRIDAY + timedelta(days=3)], Timeframe.D1), Timeframe.D1)

    assert len(gaps) == 1
    assert gaps[0].missing_bars == 2
    assert gaps[0].weekend is True


def test_outage_starting_before_the_close_is_not_a_weekend_closure():
    """A stream that dies Friday lunchtime and comes back Sunday night loses
    real trading hours. It covers a whole Saturday, so a naive "does it span
    a weekend?" test would wave it through — this must stay reported."""
    stored = bars([FRIDAY.replace(hour=12), FRIDAY + timedelta(days=2, hours=22)], Timeframe.M1)

    gaps = find_gaps(stored, Timeframe.M1)

    assert gaps[0].weekend is False


def test_outage_running_into_monday_is_not_a_weekend_closure():
    stored = bars(
        [FRIDAY.replace(hour=20, minute=59), FRIDAY + timedelta(days=3, hours=9)], Timeframe.M1
    )

    assert find_gaps(stored, Timeframe.M1)[0].weekend is False


def test_multi_day_outage_over_a_weekend_is_not_a_weekend_closure():
    stored = bars([FRIDAY.replace(hour=21), FRIDAY + timedelta(days=5)], Timeframe.M1)

    assert find_gaps(stored, Timeframe.M1)[0].weekend is False


@pytest.mark.parametrize("timeframe", [Timeframe.W1, Timeframe.MN])
def test_weekly_and_monthly_bars_are_never_scanned(timeframe):
    """Their spacing already spans calendar closures — what counts as a
    missing W1/MN bar is the broker's call, not ours."""
    stored = bars([FRIDAY, FRIDAY + timedelta(days=90)], timeframe)

    assert find_gaps(stored, timeframe) == []


def test_bars_between_counts_calendar_months_for_mn():
    assert (
        bars_between(
            Timeframe.MN, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)
        )
        == 3
    )


def test_bars_between_is_zero_for_an_empty_or_inverted_range():
    assert bars_between(Timeframe.M5, FRIDAY, FRIDAY) == 0
    assert bars_between(Timeframe.M5, FRIDAY, FRIDAY - timedelta(hours=1)) == 0


def test_weekend_window_covers_a_broker_server_time_offset():
    """Brokers on UTC+2/+3 close and reopen a couple of hours either side of
    the nominal 21:00 UTC — the window is deliberately wide enough that their
    weekend still classifies as one."""
    assert is_weekend_closure(
        Timeframe.M1, FRIDAY.replace(hour=21), FRIDAY + timedelta(days=2, hours=21)
    )
    assert is_weekend_closure(
        Timeframe.M1, FRIDAY.replace(hour=20), FRIDAY + timedelta(days=2, hours=23)
    )
