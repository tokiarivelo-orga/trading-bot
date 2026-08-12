"""Wire schemas for the `/market-data` HTTP API.

Mirrors `market_data/domain/models.py`. Candle/tick times are always epoch
seconds UTC on the wire (matching MT5 and `lightweight-charts` conventions) —
see `application/candle_stream.py:candle_message` for the REST/WS shared shape.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.market_data.domain.models import Timeframe


class CandleOut(BaseModel):
    """One OHLC bar. Identical shape over REST (`GET /candles`) and the
    `candle_closed` Socket.IO event."""

    symbol: str
    timeframe: str = Field(
        description="One of 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN'."
    )
    time: int = Field(description="Bar open time, epoch seconds UTC.")
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = Field(description="Number of ticks during the bar.")
    spread_points: int = Field(description="Spread in points, as recorded on the bar.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "time": 1_732_000_800,
                "open": 2400.12,
                "high": 2401.50,
                "low": 2399.80,
                "close": 2400.90,
                "tick_volume": 842,
                "spread_points": 25,
            }
        }
    }


class SymbolInfoOut(BaseModel):
    """Live tradable-instrument spec, as reported by the broker terminal."""

    symbol: str
    bid: float
    ask: float
    spread_points: int = Field(description="Live spread in points.")
    point: float = Field(description="Smallest price increment for this symbol.")
    digits: int = Field(description="Number of decimal digits in the quoted price.")
    stops_level: int = Field(description="Broker's minimum SL/TP distance, in points.")
    contract_size: float
    volume_min: float = Field(description="Minimum order volume in lots.")
    volume_max: float = Field(description="Maximum order volume in lots.")
    volume_step: float = Field(description="Volume increment step in lots.")


class BrokerSymbolOut(BaseModel):
    """One entry in the broker's tradable-symbol catalog — for browsing/
    charting only. Adding one to the chart does not configure it for the
    automated engine; that's a separate, deliberate step (`configs/app.yaml:
    symbols` + `configs/symbols/<sym>.yaml`, currently XAUUSD/XAGUSD)."""

    name: str
    description: str = Field(description="Broker's human-readable name for the instrument.")
    path: str = Field(description="Broker's Market Watch group, e.g. 'Forex\\\\Majors'.")
    visible: bool = Field(description="Whether the symbol is already in Market Watch.")


class BrokerSymbolPageOut(BaseModel):
    """One page of the broker's symbol catalog, for paging through the full
    list in the chart's symbol picker (as well as filtering it by `search`)."""

    items: list[BrokerSymbolOut]
    total: int = Field(
        description=(
            "Count of symbols matching `search` (or the full catalog if unset), "
            "before `limit`/`offset` are applied — use it to know whether more "
            "pages remain."
        )
    )


class BackfillRequest(BaseModel):
    symbols: list[str] | None = Field(
        default=None, description="Symbols to backfill; defaults to `configs/app.yaml: symbols`."
    )
    timeframes: list[Timeframe] | None = Field(
        default=None,
        description="Timeframes to backfill; defaults to all of M1/M5/M15/M30/H1/H4/D1/W1/MN.",
    )
    count: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description=(
            "Number of bars per symbol/timeframe. Without `start`, this is the "
            "total fetched (most recent bars). With `start`, this is the page "
            "size used while paging backward — a single gateway call is capped "
            "at 5000 bars."
        ),
    )
    start: datetime | None = Field(
        default=None,
        description=(
            "If set, pages backward from now in `count`-sized chunks until "
            "candle history reaches this date (or the broker's history runs "
            "out), instead of only fetching the most recent `count` bars — use "
            "this to seed a full date range (e.g. a year of M5 bars) for "
            "`POST /backtest/run`. Can take a while for a large range/fine "
            "timeframe combination."
        ),
    )


class BackfillResponse(BaseModel):
    stored: dict[str, int] = Field(
        description="Bars written per '<symbol>:<timeframe>' key, e.g. {'XAUUSD:M5': 1000}."
    )


class CandleGapOut(BaseModel):
    """One stretch of missing bars in stored candle history — the chart draws
    straight across these, and any indicator/backtest over the window treats
    the bars either side as adjacent when they can be hours or days apart."""

    start: int = Field(
        description=(
            "Open time the first missing bar would have had, epoch seconds UTC "
            "(i.e. the close time of the last bar before the hole)."
        )
    )
    end: int = Field(
        description="Open time of the first bar present after the hole, epoch seconds UTC."
    )
    missing_bars: int = Field(description="Bar slots of this timeframe that fit in the hole.")
    weekend: bool = Field(
        description=(
            "True when the hole fits entirely inside a weekend closure (Friday "
            "20:00 -> Sunday 23:00 UTC, wide enough to cover any broker's "
            "server-time offset) and is therefore expected, not damage. "
            "`/candle-gaps/repair` skips these unless asked otherwise."
        )
    )


class CandleGapScanOut(BaseModel):
    """Result of scanning a range of stored history for holes."""

    symbol: str
    timeframe: str = Field(description="One of 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'.")
    start: int = Field(description="Start of the scanned range, epoch seconds UTC (inclusive).")
    end: int = Field(description="End of the scanned range, epoch seconds UTC (inclusive).")
    gaps: list[CandleGapOut] = Field(description="Holes found, oldest first.")
    missing_bars: int = Field(
        description="Total bars missing across every reported hole, weekend closures included."
    )


class GapRepairRequest(BaseModel):
    symbol: str = Field(description="Trading symbol to repair, e.g. 'XAUUSD'.")
    timeframe: Timeframe = Field(description="Bar size to repair.")
    start: datetime = Field(
        description=(
            "Start of the range to repair (inclusive) — typically the chart's oldest loaded bar."
        )
    )
    end: datetime = Field(
        description=(
            "End of the range to repair (inclusive) — typically the chart's newest loaded bar."
        )
    )
    count: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description=(
            "Page size for each hole's backward-paged download. A single "
            "gateway call is capped at 5000 bars; wider holes are paged."
        ),
    )
    include_weekend: bool = Field(
        default=False,
        description=(
            "Re-download holes that look like a normal weekend closure too. Off "
            "by default (the broker has no bars there); turn it on for symbols "
            "that genuinely trade through the weekend, e.g. crypto or synthetics."
        ),
    )


class GapRepairResponse(BaseModel):
    """What the repair pass found, fixed, and could not fix."""

    symbol: str
    timeframe: str = Field(description="One of 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'.")
    found: list[CandleGapOut] = Field(
        description="Holes present before the repair ran, oldest first."
    )
    repaired: list[CandleGapOut] = Field(
        description=(
            "Holes from `found` that the fresh download closed completely. A "
            "hole that only shrank stays in `remaining` — read `bars_recovered` "
            "for what that partial fix actually gave back."
        )
    )
    remaining: list[CandleGapOut] = Field(
        description=(
            "Holes still present afterwards. The broker has no bars there "
            "either, so these are genuine market closures (holiday, halt, "
            "symbol listed later) — retrying will not change them."
        )
    )
    bars_downloaded: int = Field(
        description=(
            "Bars fetched from the gateway and upserted, including the "
            "deliberate overlap either side of each hole."
        )
    )
    bars_recovered: int = Field(
        description=(
            "Bars that were missing before and are stored now — counted across "
            "every hole the repair targeted, so a hole that merely shrank "
            "(an outage running into the broker's nightly break) still reports "
            "the bars it gave back. This is the number worth showing a user."
        )
    )
