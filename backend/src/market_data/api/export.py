"""Bulk candle-history export for downstream AI-training pipelines.

Split out of `api/routes.py` (mirroring how `journal/api/export.py` keeps its
own export endpoint separate from that module's main CRUD routes) since this
introduces a pattern — `StreamingResponse` — nothing else in this codebase
uses yet: a chunked read straight from `CandleRepository` via
`CandleHistoryService.export_range`, so a symbol's entire M1 history can be
downloaded as CSV without materializing the whole range as one in-memory
list first.
"""

from __future__ import annotations

import csv
import io
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from src.market_data.api.routes import TimeframeParam
from src.market_data.api.schemas import CandleExportOut, CandleOut
from src.market_data.application.candle_stream import candle_message
from src.market_data.domain.models import Candle, Timeframe
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/market-data", tags=["market-data"])

# Hard cap on the JSON path only — a typed `list[CandleOut]` response has to
# be built and serialized as one object, unlike the CSV path's page-at-a-time
# stream, so this keeps that path from being asked to hold, say, a year of M1
# bars (>500k rows) in memory/response body at once. `format=csv` has no such
# cap: it's the tool this route deliberately offers for exactly that case.
_MAX_JSON_EXPORT_ROWS = 20_000

_CSV_HEADER = (
    "symbol",
    "timeframe",
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread_points",
    "real_volume",
    "atr_14",
    "day_of_week",
)


def _csv_row(candle: Candle) -> list[object]:
    return [
        candle.symbol,
        candle.timeframe.value,
        int(candle.time.timestamp()),
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.tick_volume,
        candle.spread_points,
        candle.real_volume,
        candle.atr_14,
        candle.day_of_week,
    ]


async def _csv_stream(candles: AsyncIterator[Candle]) -> AsyncIterator[str]:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    async for candle in candles:
        writer.writerow(_csv_row(candle))
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)


@router.get(
    "/candles/export",
    summary="Bulk-export stored candle history (CSV or JSON)",
    description=(
        "Downloads every stored bar for `symbol`/`timeframe` in `[from_time, to_time)` "
        "(epoch seconds UTC), including the `real_volume`/`atr_14`/`day_of_week` columns "
        "enrichment adds on top of the raw OHLCV bar — this is the API-reachable "
        "replacement for reading `candles` straight out of the SQLite file, for a "
        "downstream AI training pipeline. Reads local storage only, never the live "
        "gateway; run `POST /backfill` first if the range isn't fully downloaded yet.\n\n"
        "`from_time`/`to_time` are **required** — there is no 'export everything' "
        "default, since an unbounded pull over a symbol's full M1 history is exactly "
        "the request size that DoSes the export itself or the DB.\n\n"
        "`format=csv` streams the range page by page as it's read from storage "
        "(`Content-Disposition: attachment`), so even a multi-year M1 range downloads "
        "without the server holding it all in memory at once — this is the format to "
        "use for a wide range. `format=json` returns a single typed "
        f"`CandleExportOut` envelope instead, capped at {_MAX_JSON_EXPORT_ROWS:,} rows "
        "(the request fails with 400 if the range holds more — narrow it or switch to "
        "`format=csv`), since a JSON response has to be built as one object rather than "
        "streamed row by row.\n\n"
        "A range with no stored candles is not an error: CSV returns just the header "
        "row, JSON returns `count: 0` and an empty `candles` list, both with a 200."
    ),
    response_model=CandleExportOut,
    responses={
        200: {
            "description": (
                "CSV attachment (`format=csv`) or a `CandleExportOut` JSON envelope "
                "(`format=json`). Either can be empty when the range holds no stored bars."
            ),
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/CandleExportOut"}},
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": (
                            "Header row `symbol,timeframe,time,open,high,low,close,"
                            "tick_volume,spread_points,real_volume,atr_14,day_of_week` "
                            "followed by one row per candle, oldest first."
                        ),
                    }
                },
            },
        },
        400: {
            "description": (
                "`to_time` is not after `from_time`, or `format=json` was requested for a "
                "range whose row count exceeds the JSON export cap."
            )
        },
    },
)
async def export_candles(
    account: AccountRuntimeDep,
    symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'."),
    timeframe: TimeframeParam = Timeframe.M5,
    from_time: int = Query(description="Start of the export range, epoch seconds UTC (inclusive)."),
    to_time: int = Query(description="End of the export range, epoch seconds UTC (exclusive)."),
    format: Annotated[  # noqa: A002 - "format" is the clearest name for this query param
        Literal["csv", "json"], Query(description="'csv' streams; 'json' returns a typed envelope.")
    ] = "csv",
) -> StreamingResponse | CandleExportOut:
    if to_time <= from_time:
        raise HTTPException(status_code=400, detail="to_time must be after from_time")
    start = datetime.fromtimestamp(from_time, tz=UTC)
    end = datetime.fromtimestamp(to_time, tz=UTC)

    if format == "csv":
        filename = f"{symbol}_{timeframe.value}_{from_time}_{to_time}.csv"
        return StreamingResponse(
            _csv_stream(account.candle_history.export_range(symbol, timeframe, start, end)),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    candles = await account.candle_history.get_range(symbol, timeframe, start, end)
    if len(candles) > _MAX_JSON_EXPORT_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"range holds {len(candles)} rows, over the {_MAX_JSON_EXPORT_ROWS} "
                "format=json cap — narrow from_time/to_time or use format=csv"
            ),
        )
    return CandleExportOut(
        symbol=symbol,
        timeframe=timeframe.value,
        from_time=from_time,
        to_time=to_time,
        count=len(candles),
        candles=[CandleOut(**candle_message(c)) for c in candles],
    )
