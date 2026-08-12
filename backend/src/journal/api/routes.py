"""Trade journal endpoints — chart markers + trade history (F7)."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Path, Query

from src.journal.api.schemas import (
    BotAnalyticsOut,
    CandleOut,
    DecisionContextOut,
    EquityPointOut,
    IndicatorReadingOut,
    RegimeAnalyticsOut,
    RetcodeCountOut,
    StructurePointOut,
    SymbolAnalyticsOut,
    TradeHistoryPage,
    TradeRecordOut,
    ZoneOut,
)
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.analytics import BotAnalytics, RegimeAnalytics, SymbolAnalytics
from src.journal.domain.models import CandleSnapshot, TradeRecord
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/journal", tags=["journal"])

from src.journal.api.export import router as export_router
router.include_router(export_router)


def _service(account: AccountRuntimeDep) -> TradeJournalService:
    return account.trade_journal


def _zone_out(record: TradeRecord) -> ZoneOut | None:
    if (
        record.zone_kind is None
        or record.zone_price_low is None
        or record.zone_price_high is None
        or record.zone_time_start is None
        or record.zone_time_end is None
    ):
        return None
    return ZoneOut(
        kind=record.zone_kind,
        price_low=record.zone_price_low,
        price_high=record.zone_price_high,
        time_start=int(record.zone_time_start.timestamp()),
        time_end=int(record.zone_time_end.timestamp()),
        pattern=record.zone_pattern,
    )


def _candles_out(snapshot: tuple[CandleSnapshot, ...]) -> list[CandleOut]:
    return [
        CandleOut(
            time=int(c.time.timestamp()),
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            tick_volume=c.tick_volume,
        )
        for c in snapshot
    ]


def _trade_out(record: TradeRecord) -> TradeRecordOut:
    zone = _zone_out(record)
    return TradeRecordOut(
        id=record.id,
        symbol=record.symbol,
        side=record.side,
        volume=record.volume,
        open_price=record.open_price,
        open_time=int(record.open_time.timestamp()),
        sl=record.sl,
        tp=record.tp,
        close_price=record.close_price,
        close_time=int(record.close_time.timestamp()) if record.close_time else None,
        profit=record.profit,
        close_reason=record.close_reason,
        comment=record.comment,
        strategy_version=record.strategy_version,
        skill=record.skill,
        reason=record.reason,
        confidence=record.confidence,
        zone=zone,
        pattern=record.pattern,
        structure=[
            StructurePointOut(label=label, price=price, time=int(time.timestamp()))
            for label, price, time in record.structure
        ],
        indicators=[
            IndicatorReadingOut(name=n, value=v, threshold=t, comparison=c, passed=p)
            for n, v, t, c, p in record.indicators
        ],
        regime_volatility=record.regime_volatility,
        regime_volatility_percentile=record.regime_volatility_percentile,
        regime_trend=record.regime_trend,
        regime_adx=record.regime_adx,
        regime_session=record.regime_session,
        transaction_cost=record.transaction_cost,
        mfe=record.mfe,
        mfe_time=int(record.mfe_time.timestamp()) if record.mfe_time else None,
        mae=record.mae,
        mae_time=int(record.mae_time.timestamp()) if record.mae_time else None,
    )


@router.get(
    "/markers",
    response_model=list[TradeRecordOut],
    summary="Get trade markers for the chart",
    description=(
        "Returns trades for `symbol` whose open time falls within `[from, to)` "
        "(epoch seconds UTC, both optional). Used to plot entry/exit markers on "
        "the `lightweight-charts` panel — the frontend queries this per visible "
        "chart range. Pass `skill` (a bot's full id from `GET /skills/normal`, e.g. "
        "'normal/xauusd/breakout_v1') to scope markers to one bot's own trades instead of "
        "every trade (any bot, or manual) on the symbol. Capped to the most recent `limit` "
        "trades by open_time so a long-running chart session doesn't re-fetch the entire "
        "trade history on every poll."
    ),
)
async def get_markers(
    account: AccountRuntimeDep,
    symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'."),
    frm: int | None = Query(
        default=None, alias="from", description="Range start, epoch seconds UTC (inclusive)."
    ),
    to: int | None = Query(default=None, description="Range end, epoch seconds UTC (exclusive)."),
    skill: str | None = Query(
        default=None, description="Scope to one bot's own trades, e.g. 'normal/xauusd/breakout_v1'."
    ),
    limit: int = Query(
        default=1000, ge=1, le=5000, description="Maximum number of most-recent trades to return."
    ),
) -> list[TradeRecordOut]:
    records = await _service(account).get_markers(symbol, frm, to, skill, limit)
    return [_trade_out(r) for r in records]


@router.get(
    "/trades",
    response_model=list[TradeRecordOut],
    summary="Get recent trade history",
    description="Returns the most recent `limit` trades for `symbol`, newest first.",
)
async def get_trades(
    account: AccountRuntimeDep,
    symbol: str = Query(description="Trading symbol, e.g. 'XAUUSD'."),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of trades to return."),
) -> list[TradeRecordOut]:
    records = await _service(account).get_last_n(symbol, limit)
    return [_trade_out(r) for r in records]


@router.get(
    "/trades/{trade_id}/decision-context",
    response_model=DecisionContextOut,
    summary="Get the entry chart snapshot and decision annotations for one trade",
    description=(
        "Returns the M5 and H1 candle snapshot captured once, at the moment this trade's "
        "`PositionOpened` event fired, plus the zone/pattern/structure/confluence-indicator "
        "annotations the strategy reported for it. This is a **frozen snapshot from the moment "
        "of entry** — not live/refetched market data — so it renders identically no matter how "
        "much later you look at it, even after the symbol's live candle history has since aged "
        "the original bars out. Powers the 'why did the bot take this trade' chart view. Never "
        "includes exit-time snapshots or any other AI-review-only data."
    ),
    responses={404: {"description": "No trade with this `trade_id` exists for this account."}},
)
async def get_decision_context(
    account: AccountRuntimeDep,
    trade_id: str = Path(description="Broker position ticket, as a string."),
) -> DecisionContextOut:
    record = await asyncio.to_thread(_service(account).get_trade, trade_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    return DecisionContextOut(
        trade_id=record.id,
        symbol=record.symbol,
        side=record.side,
        open_price=record.open_price,
        open_time=int(record.open_time.timestamp()),
        entry_candles=_candles_out(record.m5_entry_snapshot),
        higher_tf_candles=_candles_out(record.h1_entry_snapshot),
        zone=_zone_out(record),
        pattern=record.pattern,
        structure=[
            StructurePointOut(label=label, price=price, time=int(time.timestamp()))
            for label, price, time in record.structure
        ],
        indicators=[
            IndicatorReadingOut(name=n, value=v, threshold=t, comparison=c, passed=p)
            for n, v, t, c, p in record.indicators
        ],
        reason=record.reason,
        confidence=record.confidence,
    )


@router.get(
    "/history",
    response_model=TradeHistoryPage,
    summary="Search and paginate trade history",
    description=(
        "Returns a filtered, paginated page of journaled trades across any symbol. "
        "Unlike `/trades` (single symbol, most-recent-N, no filters), this endpoint "
        "supports filtering by symbol, side, strategy version, skill, outcome, and "
        "open/close time ranges, plus sorting and offset pagination — it backs the "
        "trade history UI's filter and category controls."
    ),
)
async def get_history(
    account: AccountRuntimeDep,
    symbol: str | None = Query(default=None, description="Exact symbol match, e.g. 'XAUUSD'."),
    side: Literal["buy", "sell"] | None = Query(default=None, description="Trade direction."),
    strategy_version: str | None = Query(
        default=None,
        description="Substring match on strategy version, e.g. 'breakout' or 'breakout_v1:v1'.",
    ),
    skill: str | None = Query(
        default=None, description="Substring match on bot skill, e.g. 'normal' or 'xauusd'."
    ),
    outcome: Literal["win", "loss", "breakeven", "open"] | None = Query(
        default=None,
        description=(
            "'open' = not yet closed; 'win'/'loss'/'breakeven' = closed with profit >0 / <0 / ==0."
        ),
    ),
    open_from: int | None = Query(
        default=None, description="Only trades opened at/after this epoch-seconds UTC."
    ),
    open_to: int | None = Query(
        default=None, description="Only trades opened at/before this epoch-seconds UTC."
    ),
    close_from: int | None = Query(
        default=None, description="Only trades closed at/after this epoch-seconds UTC."
    ),
    close_to: int | None = Query(
        default=None, description="Only trades closed at/before this epoch-seconds UTC."
    ),
    order_by: Literal["open_time", "close_time", "profit"] = Query(
        default="open_time", description="Field to sort by."
    ),
    order_dir: Literal["asc", "desc"] = Query(default="desc", description="Sort direction."),
    limit: int = Query(default=50, ge=1, le=500, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Number of matching trades to skip."),
) -> TradeHistoryPage:
    records, total, total_profit = await _service(account).search_trades(
        symbol=symbol,
        side=side,
        strategy_version=strategy_version,
        skill=skill,
        outcome=outcome,
        open_from=open_from,
        open_to=open_to,
        close_from=close_from,
        close_to=close_to,
        order_by=order_by,
        order_dir=order_dir,
        limit=limit,
        offset=offset,
    )
    return TradeHistoryPage(
        items=[_trade_out(r) for r in records], total=total, total_profit=total_profit
    )


def _symbol_analytics_out(a: SymbolAnalytics) -> SymbolAnalyticsOut:
    return SymbolAnalyticsOut(
        symbol=a.symbol,
        trade_count=a.trade_count,
        open_count=a.open_count,
        closed_count=a.closed_count,
        win_count=a.win_count,
        loss_count=a.loss_count,
        breakeven_count=a.breakeven_count,
        win_rate=a.win_rate,
        total_profit=a.total_profit,
        gross_profit=a.gross_profit,
        gross_loss=a.gross_loss,
        profit_factor=a.profit_factor,
        avg_win=a.avg_win,
        avg_loss=a.avg_loss,
        avg_profit_per_trade=a.avg_profit_per_trade,
        largest_win=a.largest_win,
        largest_loss=a.largest_loss,
        total_volume=a.total_volume,
        bot_count=a.bot_count,
        first_trade_time=a.first_trade_time,
        last_trade_time=a.last_trade_time,
    )


def _bot_analytics_out(a: BotAnalytics) -> BotAnalyticsOut:
    return BotAnalyticsOut(
        skill=a.skill,
        bot_name=a.bot_name,
        symbol=a.symbol,
        strategy_version=a.strategy_version,
        trade_count=a.trade_count,
        open_count=a.open_count,
        closed_count=a.closed_count,
        win_count=a.win_count,
        loss_count=a.loss_count,
        breakeven_count=a.breakeven_count,
        win_rate=a.win_rate,
        total_profit=a.total_profit,
        gross_profit=a.gross_profit,
        gross_loss=a.gross_loss,
        profit_factor=a.profit_factor,
        avg_win=a.avg_win,
        avg_loss=a.avg_loss,
        expectancy=a.expectancy,
        largest_win=a.largest_win,
        largest_loss=a.largest_loss,
        max_drawdown=a.max_drawdown,
        avg_trade_duration_seconds=a.avg_trade_duration_seconds,
        first_trade_time=a.first_trade_time,
        last_trade_time=a.last_trade_time,
        equity_curve=[
            EquityPointOut(
                trade_id=p.trade_id,
                close_time=p.close_time,
                profit=p.profit,
                cumulative_profit=p.cumulative_profit,
            )
            for p in a.equity_curve
        ],
        avg_slippage=a.avg_slippage,
        measured_slippage_count=a.measured_slippage_count,
        avg_execution_latency_ms=a.avg_execution_latency_ms,
        retcode_histogram=[
            RetcodeCountOut(retcode=code, count=count) for code, count in a.retcode_histogram
        ],
        avg_mfe=a.avg_mfe,
        avg_mae=a.avg_mae,
        mfe_mae_ratio=a.mfe_mae_ratio,
        avg_mfe_on_losers=a.avg_mfe_on_losers,
        avg_mae_on_winners=a.avg_mae_on_winners,
        total_transaction_cost=a.total_transaction_cost,
        avg_transaction_cost_per_trade=a.avg_transaction_cost_per_trade,
        cost_pct_of_gross_edge=a.cost_pct_of_gross_edge,
    )


def _regime_analytics_out(a: RegimeAnalytics) -> RegimeAnalyticsOut:
    return RegimeAnalyticsOut(
        skill=a.skill,
        bot_name=a.bot_name,
        dimension=a.dimension,
        bucket=a.bucket,
        trade_count=a.trade_count,
        closed_count=a.closed_count,
        win_count=a.win_count,
        loss_count=a.loss_count,
        win_rate=a.win_rate,
        profit_factor=a.profit_factor,
        expectancy=a.expectancy,
        total_profit=a.total_profit,
    )


@router.get(
    "/analytics/symbols",
    response_model=list[SymbolAnalyticsOut],
    summary="Get per-symbol trading analytics",
    description=(
        "Aggregates every journaled trade (any bot, or manual) grouped by symbol: trade "
        "counts, win rate, profit factor, gross/net profit, average win/loss, volume, and how "
        "many distinct bots have traded it. Sorted by total_profit descending. Powers the "
        "analytics dashboard's symbol comparison view — use `GET /journal/analytics/bots` "
        "instead to compare individual bots rather than symbols."
    ),
)
async def get_symbol_analytics(
    account: AccountRuntimeDep,
    open_from: int | None = Query(
        default=None, description="Only trades opened at/after this epoch-seconds UTC."
    ),
    open_to: int | None = Query(
        default=None, description="Only trades opened at/before this epoch-seconds UTC."
    ),
) -> list[SymbolAnalyticsOut]:
    analytics = await _service(account).get_symbol_analytics(
        open_from=open_from, open_to=open_to
    )
    return [_symbol_analytics_out(a) for a in analytics]


@router.get(
    "/analytics/bots",
    response_model=list[BotAnalyticsOut],
    summary="Get per-bot trading analytics and equity curves",
    description=(
        "Aggregates every journaled trade grouped by bot (`skill`): trade counts, win rate, "
        "profit factor, expectancy, max drawdown, average trade duration, and a full "
        "cumulative-profit equity curve. Sorted by total_profit descending, so the best- and "
        "worst-performing bots sort to the ends. Trades placed manually or via the API (no "
        "`skill`) are excluded, since they aren't attributable to any bot. Also reports "
        "execution quality per bot — average slippage, average signal-to-fill latency, a "
        "broker-return-code histogram, and MFE/MAE excursion aggregates, which answer whether "
        "a bot's take-profits sit past where price turns and whether its stops are too tight. "
        "Also reports total/average transaction cost and cost_pct_of_gross_edge — what fraction "
        "of the bot's edge before costs is spent on spread + slippage. Powers the analytics "
        "dashboard's bot comparison and equity-curve charts."
    ),
)
async def get_bot_analytics(
    account: AccountRuntimeDep,
    open_from: int | None = Query(
        default=None, description="Only trades opened at/after this epoch-seconds UTC."
    ),
    open_to: int | None = Query(
        default=None, description="Only trades opened at/before this epoch-seconds UTC."
    ),
) -> list[BotAnalyticsOut]:
    analytics = await _service(account).get_bot_analytics(
        open_from=open_from, open_to=open_to
    )
    return [_bot_analytics_out(a) for a in analytics]


@router.get(
    "/analytics/regimes",
    response_model=list[RegimeAnalyticsOut],
    summary="Get per-bot trading analytics split by market regime",
    description=(
        "Aggregates every journaled trade grouped by bot (`skill`) and, separately, by each of "
        "three market-regime dimensions the trade was tagged with at entry "
        "(OBSERVABILITY_PLAN.md Phase 6): volatility bucket ('low'/'normal'/'high'/'extreme'), "
        "trend/range ('trending'/'ranging'), and trading session ('asian'/'london'/'overlap'/"
        "'new_york'/'off_session'). One entry per (bot, dimension, bucket) with at least one "
        "attributable trade — win rate, profit factor, expectancy, and total profit, sliced to "
        "that bucket only. Reports each dimension independently rather than the full cross "
        "product, so a bot's edge (or lack of one) in a specific market condition doesn't get "
        "averaged away by every other condition it also traded through. Trades with no `skill` "
        "or whose regime tag is missing (journaled before Phase 6, or the entry timeframe had "
        "no candles to classify) are excluded rather than fabricating an 'unknown' bucket. Use "
        "this to answer 'is this bot's edge (or a scalp's cost burden) regime-dependent', "
        "alongside `GET .../analytics/bots` for the bot's overall numbers."
    ),
)
async def get_regime_analytics(
    account: AccountRuntimeDep,
    open_from: int | None = Query(
        default=None, description="Only trades opened at/after this epoch-seconds UTC."
    ),
    open_to: int | None = Query(
        default=None, description="Only trades opened at/before this epoch-seconds UTC."
    ),
) -> list[RegimeAnalyticsOut]:
    analytics = await _service(account).get_regime_analytics(
        open_from=open_from, open_to=open_to
    )
    return [_regime_analytics_out(a) for a in analytics]
