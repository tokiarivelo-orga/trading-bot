"""Market data REST endpoints (chart + tooling). Live streaming is Socket.IO —
see `src.market_data.api.ws`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from src.market_data.api.schemas import (
    BackfillRequest,
    BackfillResponse,
    BrokerSymbolOut,
    BrokerSymbolPageOut,
    CandleGapOut,
    CandleGapScanOut,
    CandleOut,
    GapRepairRequest,
    GapRepairResponse,
    SymbolInfoOut,
)
from src.market_data.application.candle_stream import candle_message
from src.market_data.domain.gaps import CandleGap
from src.market_data.domain.models import MarketDataUnavailable, Timeframe
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/market-data", tags=["market-data"])

TimeframeParam = Annotated[
    Timeframe, Query(description="Bar size: M1, M5, M15, M30, H1, H4, D1, W1, or MN.")
]

_UNAVAILABLE = {503: {"description": "The MT5 gateway is unreachable or not logged in."}}


@router.get(
    "/candles",
    response_model=list[CandleOut],
    summary="Get historical candles",
    description=(
        "Returns up to `count` closed bars for `symbol`/`timeframe`, newest last. "
        "Serves from the live gateway when connected; falls back to the local "
        "database (populated by the background candle stream and `/backfill`) "
        "when the gateway is unreachable, so the chart keeps working across "
        "MT5 disconnects. Pass `before` to page further back than the most "
        "recent `count` bars, e.g. when the chart is panned to the left edge "
        "of its currently loaded history."
    ),
)
async def get_candles(
    account: AccountRuntimeDep,
    symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'."),
    timeframe: TimeframeParam = Timeframe.M5,
    count: Annotated[int, Query(ge=1, le=5000, description="Number of bars to return.")] = 300,
    before: Annotated[
        int | None,
        Query(
            description=(
                "Epoch seconds UTC. When set, returns `count` bars with open "
                "time strictly before this instant instead of the most recent "
                "ones — for loading older history on demand."
            )
        ),
    ] = None,
) -> list[CandleOut]:
    before_dt = datetime.fromtimestamp(before, tz=UTC) if before is not None else None
    candles = await account.candle_history.get_candles(symbol, timeframe, count, before_dt)
    return [CandleOut(**candle_message(c)) for c in candles]


@router.get(
    "/symbol-info",
    response_model=SymbolInfoOut,
    summary="Get live symbol spec",
    description="Live bid/ask, spread, and broker-side volume/price constraints for a symbol.",
    responses=_UNAVAILABLE,
)
async def get_symbol_info(
    account: AccountRuntimeDep, symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'.")
) -> SymbolInfoOut:
    try:
        info = await account.market_data.get_symbol_info(symbol)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SymbolInfoOut(
        symbol=info.symbol,
        bid=info.bid,
        ask=info.ask,
        spread_points=info.spread_points,
        point=info.point,
        digits=info.digits,
        stops_level=info.stops_level,
        contract_size=info.contract_size,
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
    )


@router.get(
    "/broker-symbols",
    response_model=BrokerSymbolPageOut,
    summary="Browse the connected broker's tradable symbols",
    description=(
        "A page of the symbols the broker offers (optionally filtered by a "
        "case-insensitive substring match on name/description), for the chart's "
        "symbol picker — pass `offset` to page through the full catalog when no "
        "`search` is given. This is browsing only — it never modifies "
        "`configs/app.yaml` or `configs/symbols/`, so picking one shows its chart "
        "on demand (including live `candle_closed` WebSocket updates for as long "
        "as a client has it open) but does not add it to the automated engine's "
        "traded universe (currently XAUUSD/XAGUSD)."
    ),
    responses=_UNAVAILABLE,
)
async def get_broker_symbols(
    account: AccountRuntimeDep,
    search: str | None = Query(
        default=None,
        max_length=64,
        description="Case-insensitive substring match on name/description.",
    ),
    limit: Annotated[int, Query(ge=1, le=500, description="Maximum symbols to return.")] = 200,
    offset: Annotated[
        int, Query(ge=0, description="Number of matching symbols to skip, for paging.")
    ] = 0,
) -> BrokerSymbolPageOut:
    try:
        page = await account.market_data.list_symbols(search, limit, offset)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BrokerSymbolPageOut(
        items=[
            BrokerSymbolOut(name=s.name, description=s.description, path=s.path, visible=s.visible)
            for s in page.items
        ],
        total=page.total,
    )


@router.post(
    "/backfill",
    response_model=BackfillResponse,
    summary="Backfill candle history into the local database",
    description=(
        "Fetches bars per symbol/timeframe from the gateway and upserts them "
        "into the local database, so `GET /candles` has data to fall back to "
        "when the gateway later goes offline, and so `POST /backtest/run` has "
        "history to replay. Without `start`, fetches only the most recent "
        "`count` bars. With `start`, pages backward until history reaches "
        "that date — use this before backtesting a multi-month/year period, "
        "since a backtest can only replay candles already in the database. "
        "Safe to call repeatedly — existing bars are overwritten in place, "
        "not duplicated. Also snapshots "
        "each symbol's broker facts (point, digits, stops_level, contract_size, "
        "volume min/max/step) from the gateway's live `symbol_info` into the "
        "`symbol_specs` table — this is what lets `POST /backtest/run` replay "
        "any symbol offline afterward without a hand-authored "
        "`configs/symbols/<symbol>.yaml`."
    ),
    responses=_UNAVAILABLE,
)
async def backfill(account: AccountRuntimeDep, body: BackfillRequest) -> BackfillResponse:
    symbols = body.symbols or account.symbols
    timeframes = body.timeframes or list(Timeframe)
    stored: dict[str, int] = {}
    try:
        for symbol in symbols:
            await account.candle_history.sync_symbol_spec(symbol)
            for timeframe in timeframes:
                bars = await account.candle_history.backfill(
                    symbol, timeframe, body.count, body.start
                )
                stored[f"{symbol}:{timeframe.value}"] = bars
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BackfillResponse(stored=stored)


@router.get(
    "/candle-gaps",
    response_model=CandleGapScanOut,
    summary="Find holes in stored candle history",
    description=(
        "Scans locally stored bars for `symbol`/`timeframe` between `start` and "
        "`end` (both inclusive, epoch seconds UTC) and reports every stretch of "
        "missing bars. A hole left by a stream outage or an interrupted backfill "
        "is invisible to `GET /candles` — the chart draws straight across it, and "
        "indicators, zone detection and `POST /backtest/run` all treat the bars "
        "either side as adjacent when they can be hours apart, which quietly "
        "falsifies every result computed over that window. Holes that fit inside "
        "a normal weekend closure are flagged `weekend: true` rather than hidden, "
        "so a caller can show them separately. Read-only — use "
        "`POST /candle-gaps/repair` to actually fill what's missing. W1/MN always "
        "return no gaps: their bar spacing spans calendar closures by design."
    ),
)
async def get_candle_gaps(
    account: AccountRuntimeDep,
    symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'."),
    start: int = Query(description="Start of the range to scan, epoch seconds UTC (inclusive)."),
    end: int = Query(description="End of the range to scan, epoch seconds UTC (inclusive)."),
    timeframe: TimeframeParam = Timeframe.M5,
) -> CandleGapScanOut:
    gaps = await account.candle_history.scan_gaps(
        symbol,
        timeframe,
        datetime.fromtimestamp(start, tz=UTC),
        datetime.fromtimestamp(end, tz=UTC),
    )
    return CandleGapScanOut(
        symbol=symbol,
        timeframe=timeframe.value,
        start=start,
        end=end,
        gaps=[_gap_out(gap) for gap in gaps],
        missing_bars=sum(gap.missing_bars for gap in gaps),
    )


@router.post(
    "/candle-gaps/repair",
    response_model=GapRepairResponse,
    summary="Re-download missing candles to close gaps in history",
    description=(
        "Finds the holes `GET /candle-gaps` reports over the same range, asks the "
        "gateway for each missing stretch (one backward-paged request per hole, "
        "not the whole window), upserts what comes back, then rescans so the "
        "response says which holes actually closed. This is the chart's "
        '"fill gaps" action, and the fix for a window whose indicators or '
        "backtest results are skewed by bars that were never downloaded. "
        "Safe to call repeatedly — bars are overwritten in place, never "
        "duplicated. Weekend closures are skipped unless `include_weekend` is "
        "set. Anything still listed in `remaining` afterwards is a hole the "
        "broker itself cannot fill (holiday, halt, symbol listed later), so "
        "retrying will not change it."
    ),
    responses=_UNAVAILABLE,
)
async def repair_candle_gaps(
    account: AccountRuntimeDep, body: GapRepairRequest
) -> GapRepairResponse:
    try:
        report = await account.candle_history.repair_gaps(
            body.symbol,
            body.timeframe,
            body.start,
            body.end,
            body.count,
            body.include_weekend,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return GapRepairResponse(
        symbol=report.symbol,
        timeframe=report.timeframe.value,
        found=[_gap_out(gap) for gap in report.found],
        repaired=[_gap_out(gap) for gap in report.repaired],
        remaining=[_gap_out(gap) for gap in report.remaining],
        bars_downloaded=report.bars_downloaded,
        bars_recovered=report.bars_recovered,
    )


def _gap_out(gap: CandleGap) -> CandleGapOut:
    return CandleGapOut(
        start=int(gap.start.timestamp()),
        end=int(gap.end.timestamp()),
        missing_bars=gap.missing_bars,
        weekend=gap.weekend,
    )
