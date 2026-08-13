"""Market data domain: timeframes, candles, ticks, symbol specs.

Pure values — no I/O, no framework imports. All times are UTC; candle `time`
is the bar's open time, matching MT5 and lightweight-charts conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN": 2592000,  # approximate (30-day average); use bar_open/close_of for exact month alignment
}


class Timeframe(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN = "MN"

    @property
    def seconds(self) -> int:
        return _TIMEFRAME_SECONDS[self.value]

    def next_up(self) -> Timeframe | None:
        """The next-larger timeframe (declaration order above is the
        hierarchy), or `None` for `MN` — there is nothing above it."""
        members = list(Timeframe)
        index = members.index(self) + 1
        return members[index] if index < len(members) else None

    def bar_open(self, moment: datetime) -> datetime:
        """Open time of the bar containing `moment` (UTC-aligned; broker
        server-time offsets for D1 are a Phase 4 concern). W1 bars open on
        Monday 00:00 UTC; MN bars open on the 1st of the calendar month."""
        if self is Timeframe.MN:
            return datetime(moment.year, moment.month, 1, tzinfo=UTC)
        epoch = int(moment.timestamp())
        if self is Timeframe.W1:
            # Unix epoch (1970-01-01) was a Thursday; shift 3 days so Monday
            # boundaries align to a multiple of `seconds`.
            shifted = epoch + 3 * 86400
            return datetime.fromtimestamp(shifted - shifted % self.seconds - 3 * 86400, tz=UTC)
        return datetime.fromtimestamp(epoch - epoch % self.seconds, tz=UTC)

    def close_of(self, bar_open_time: datetime) -> datetime:
        """Open time of the bar following the one opening at `bar_open_time`
        (i.e. this bar's close time). Calendar-correct for MN's variable
        month length."""
        if self is Timeframe.MN:
            year, month = bar_open_time.year, bar_open_time.month
            return datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
        return datetime.fromtimestamp(int(bar_open_time.timestamp()) + self.seconds, tz=UTC)

    def last_closed_open(self, now: datetime) -> datetime:
        """Open time of the most recent fully closed bar at `now`."""
        current = self.bar_open(now)
        if self is Timeframe.MN:
            year, month = current.year, current.month
            prev_month = 12 if month == 1 else month - 1
            prev_year = year - 1 if month == 1 else year
            return datetime(prev_year, prev_month, 1, tzinfo=UTC)
        return datetime.fromtimestamp(int(current.timestamp()) - self.seconds, tz=UTC)


@dataclass(frozen=True, kw_only=True)
class Candle:
    symbol: str
    timeframe: Timeframe
    time: datetime  # bar open, UTC
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread_points: int
    # Actual traded volume during the bar (distinct from tick_volume, a tick
    # count) — 0 when the broker doesn't report it for this symbol, and NULL
    # in the DB for any bar stored before this field existed.
    real_volume: int = 0
    # Trailing 14-period ATR and UTC day-of-week (0=Monday..6=Sunday),
    # computed by a separate enrichment pass after the bar is stored (see
    # `adapters/candle_repository.py::enrich_missing`) — NULL until that pass
    # reaches this bar, since ATR needs trailing history the OHLC upsert
    # alone doesn't have.
    atr_14: float | None = None
    day_of_week: int | None = None

    @property
    def close_time(self) -> datetime:
        return self.timeframe.close_of(self.time)

    def is_closed(self, now: datetime) -> bool:
        return now >= self.close_time


@dataclass(frozen=True, kw_only=True)
class Tick:
    symbol: str
    time: datetime
    bid: float
    ask: float


@dataclass(frozen=True, kw_only=True)
class BrokerSymbol:
    """One entry in the broker's tradable-symbol catalog (chart/watchlist
    browsing only — adding one here does not configure it for the engine;
    see configs/app.yaml: symbols for that)."""

    name: str
    description: str
    path: str  # broker's Market Watch group, e.g. "Forex\\Majors"
    visible: bool  # already in Market Watch


@dataclass(frozen=True, kw_only=True)
class SymbolPage:
    """One page of the broker's symbol catalog. `total` is the count after
    search filtering but before paging, so callers know whether more pages
    remain (`offset + len(items) < total`)."""

    items: list[BrokerSymbol]
    total: int


@dataclass(frozen=True, kw_only=True)
class SymbolInfo:
    symbol: str
    bid: float
    ask: float
    spread_points: int
    point: float
    digits: int
    stops_level: int
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float


class MarketDataUnavailable(Exception):
    """Gateway unreachable, not logged in, or the terminal rejected the call."""
