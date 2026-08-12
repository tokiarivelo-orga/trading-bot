"""Holes in candle history: finding them, and telling a real one from a
normal market closure.

Pure values — no I/O, no framework imports. A "gap" here is a stretch of bar
slots that *should* exist between two stored bars but don't: the stream was
down, the backend was restarted for longer than `poll_lookback` covers, or a
backfill was interrupted. Those holes are invisible to
`CandleRepository.get_latest`/`get_before` (plain `ORDER BY time DESC LIMIT`
only ever sees the newest end of a window), so a chart happily renders across
one — and every indicator, zone detector and backtest computed over that
window is silently wrong, because bars that never happened look adjacent.

The hard part is that not every hole is a defect: markets close. Weekend
closures are classified here (`CandleGap.weekend`) so callers can skip
re-downloading a stretch the broker was never going to have. Everything else
is reported, and `CandleHistoryService.repair_gaps` lets the broker be the
final arbiter — whatever it still can't fill after a re-download is a genuine
closure (holiday, halt, symbol not yet listed), not a hole in our copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.market_data.domain.models import Candle, Timeframe

# The weekend closure window, deliberately wider than any real broker's:
# retail FX/metals stop around 21:00-22:00 UTC Friday and reopen around
# 21:00-23:00 UTC Sunday, and a broker's own server-time offset (commonly
# UTC+2/+3) shifts those edges. A hole that fits entirely inside Friday 20:00
# -> Sunday 23:00 UTC is therefore explainable by the weekend alone; one that
# starts Friday lunchtime or runs into Monday is not, and gets reported.
_WEEKEND_OPEN_HOUR = 20  # Friday, UTC
_WEEKEND_SPAN = timedelta(days=2, hours=3)  # Friday 20:00 -> Sunday 23:00
_MAX_WEEKEND_GAP = timedelta(days=3)


@dataclass(frozen=True, kw_only=True)
class CandleGap:
    """One stretch of missing bars, in `[start, end)` bar-open terms.

    `start` is the open time the first missing bar *would* have had (i.e. the
    close time of the last bar before the hole) and `end` is the open time of
    the first bar present after it, so `[start, end)` is exactly what a repair
    needs to re-download.
    """

    start: datetime
    end: datetime
    missing_bars: int
    weekend: bool

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass(frozen=True, kw_only=True)
class GapRepairReport:
    """Outcome of one `CandleHistoryService.repair_gaps` pass over a range.

    `remaining` is the honest part: gaps the broker itself couldn't fill on a
    fresh download are genuine market closures (holiday, halt, symbol listed
    later) and no amount of re-clicking will change them — the UI says so
    instead of leaving the user to retry forever.

    `repaired` holds only the gaps that closed *completely*, but a hole often
    just shrinks: an outage that ran into the broker's nightly break gives
    back its trading-hours bars and keeps the closure. `bars_recovered`
    therefore counts bars, not holes — it's the one number that reflects that
    case (see `CandleHistoryService.repair_gaps`, which computes it).
    """

    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    found: list[CandleGap]
    repaired: list[CandleGap]
    remaining: list[CandleGap]
    bars_downloaded: int
    bars_recovered: int


def bars_between(timeframe: Timeframe, start: datetime, end: datetime) -> int:
    """How many bar slots of `timeframe` fit in `[start, end)`. Calendar-correct
    for MN's variable month length; plain arithmetic otherwise, since MT5 bar
    opens are aligned to the timeframe."""
    if end <= start:
        return 0
    if timeframe is Timeframe.MN:
        return (end.year - start.year) * 12 + (end.month - start.month)
    return int((end - start).total_seconds()) // timeframe.seconds


def is_weekend_closure(timeframe: Timeframe, start: datetime, end: datetime) -> bool:
    """True if every missing bar in `[start, end)` falls inside one weekend
    closure — see `_WEEKEND_OPEN_HOUR`. Compared against the *last missing
    bar's open* rather than `end` itself, because `end` is the first bar back
    after the reopen: a D1 hole runs Saturday 00:00 -> Monday 00:00, whose
    only missing opens are Saturday and Sunday."""
    if end - start > _MAX_WEEKEND_GAP:
        return False
    days_since_friday = (start.weekday() - 4) % 7
    weekend_open = (start - timedelta(days=days_since_friday)).replace(
        hour=_WEEKEND_OPEN_HOUR, minute=0, second=0, microsecond=0
    )
    if start < weekend_open:
        return False
    last_missing_open = end - timedelta(seconds=timeframe.seconds)
    return last_missing_open <= weekend_open + _WEEKEND_SPAN


def find_gaps(candles: list[Candle], timeframe: Timeframe) -> list[CandleGap]:
    """Holes between consecutive bars of `candles` (oldest first, as every
    `CandleRepository` read returns them). Only looks *between* stored bars —
    whether history is missing before the first or after the last one isn't
    knowable from the window alone, so callers that care (see
    `CandleHistoryService.scan_gaps`) handle the empty-range case themselves.

    W1/MN are skipped: their bar spacing already spans calendar closures by
    design, so the notion of a "missing bar" there is the broker's to define.
    """
    if timeframe in (Timeframe.W1, Timeframe.MN) or len(candles) < 2:
        return []
    gaps: list[CandleGap] = []
    for earlier, later in zip(candles, candles[1:], strict=False):
        expected = timeframe.close_of(earlier.time)
        missing = bars_between(timeframe, expected, later.time)
        if missing <= 0:
            continue
        gaps.append(
            CandleGap(
                start=expected,
                end=later.time,
                missing_bars=missing,
                weekend=is_weekend_closure(timeframe, expected, later.time),
            )
        )
    return gaps
