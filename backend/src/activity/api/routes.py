"""Activity log endpoints — filterable, paginated history of what the bot did
and why (every `src.*` INFO+ log line, persisted)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.activity.api.schemas import (
    BotFunnelOut,
    BotSignalOut,
    DecisionCheckOut,
    FunnelDropOut,
    LogDeleteByFilterRequest,
    LogDeleteByIdsRequest,
    LogDeleteResult,
    LogEntryOut,
    LogHistoryPage,
)
from src.activity.application.activity_log_service import ActivityLogService
from src.activity.domain.funnel import BotFunnel
from src.activity.domain.models import BotSignal, LogEntry
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/activity", tags=["activity"])


def _service(account: AccountRuntimeDep) -> ActivityLogService:
    return account.activity_log


def _log_out(entry: LogEntry) -> LogEntryOut:
    return LogEntryOut(
        id=entry.id or 0,
        created_at=int(entry.created_at.timestamp()),
        level=entry.level,
        logger=entry.logger,
        message=entry.message,
        signal_id=entry.signal_id,
    )


@router.get(
    "/history",
    response_model=LogHistoryPage,
    summary="Search and paginate the bot's activity log",
    description=(
        "Returns a filtered, paginated page of persisted log lines from every backend "
        "module (engine decisions, broker fills, skill routing, risk vetoes, spread gate, "
        "circuit breaker) — the durable answer to 'what is the bot doing and why', beyond "
        "what scrolls past in stdout. Newest first by default. Filtering by `signal_id` "
        "(OBSERVABILITY_PLAN.md Phase 5) reads one signal's whole life in order: signal -> "
        "sizing -> order -> fill -> journal."
    ),
)
async def get_history(
    account: AccountRuntimeDep,
    level: str | None = Query(
        default=None, description="Exact level match, e.g. 'INFO', 'WARNING', 'ERROR'."
    ),
    logger_contains: str | None = Query(
        default=None,
        description="Substring match on the logger name, e.g. 'trade_loop' or 'broker'.",
    ),
    q: str | None = Query(
        default=None, description="Substring match on the message text, e.g. a symbol or reason."
    ),
    created_from: int | None = Query(
        default=None, description="Only entries at/after this epoch-seconds UTC."
    ),
    created_to: int | None = Query(
        default=None, description="Only entries at/before this epoch-seconds UTC."
    ),
    limit: int = Query(default=100, ge=1, le=1000, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Number of matching entries to skip."),
    signal_id: str | None = Query(
        default=None,
        description="Exact match on the correlation id (OBSERVABILITY_PLAN.md Phase 5) — "
        "every log line from one signal's life, in one call.",
    ),
) -> LogHistoryPage:
    entries, total = await _service(account).search(
        level=level,
        logger_contains=logger_contains,
        q=q,
        created_from=created_from,
        created_to=created_to,
        limit=limit,
        offset=offset,
        signal_id=signal_id,
    )
    return LogHistoryPage(items=[_log_out(e) for e in entries], total=total)


@router.get(
    "/signals",
    response_model=list[BotSignalOut],
    summary="Get one bot's signal→outcome trail for the chart",
    description=(
        "Returns `skill`'s own signal decisions — every setup the strategy saw, whether it "
        "became a trade or was vetoed/rejected, in the order it happened. Served from the "
        "`signal_decisions` table the engine writes directly; the part of the requested "
        "window that predates that table's oldest row falls back to reconstructing the "
        "trail from persisted decision-trail log lines (legacy path), and the two halves "
        "are concatenated at that boundary without overlapping. `skill` is the full bot id "
        "from `GET "
        "/skills/normal` (e.g. 'normal/xauusd/breakout_v1'), passed as a query param rather "
        "than a path segment since it contains '/'. Defaults to the last 14 days if `from` is "
        "omitted, to bound the query. This is the live analog of a backtest report's "
        "`signals` field — same shape, powers the same chart overlay."
    ),
)
async def get_bot_signals(
    account: AccountRuntimeDep,
    skill: str = Query(description="Full bot id, e.g. 'normal/xauusd/breakout_v1'."),
    frm: int | None = Query(
        default=None, alias="from", description="Range start, epoch seconds UTC (inclusive)."
    ),
    to: int | None = Query(default=None, description="Range end, epoch seconds UTC (inclusive)."),
) -> list[BotSignalOut]:
    signals: list[BotSignal] = await _service(account).get_bot_signals(
        skill=skill, created_from=frm, created_to=to
    )
    return [
        BotSignalOut(
            time=int(s.time.timestamp()),
            direction=s.direction,
            outcome=s.outcome,
            reason=s.reason,
            price=s.price,
            checks=[
                DecisionCheckOut(
                    name=c.name,
                    value=c.value,
                    threshold=c.threshold,
                    comparison=c.comparison,
                    passed=c.passed,
                )
                for c in s.checks
            ],
        )
        for s in signals
    ]


@router.get(
    "/signals/funnel",
    response_model=list[BotFunnelOut],
    summary="Get the per-bot veto funnel for a period",
    description=(
        "Answers 'of N signals this bot fired, why did only M trade?' — for each bot that "
        "fired at least one signal in the window, the count surviving each engine gate "
        "(HTF confirmation and the pre-trade risk gate, then the "
        "open-position cap / lot sizing, then the broker's spread cap and spread-adjusted "
        "risk-reward floor, then the fill) plus the grouped reasons everything else "
        "dropped out. Ordered busiest bot first. Built purely from the typed "
        "`signal_decisions` table: unlike `GET /activity/signals` there is deliberately no "
        "legacy log-scrape fallback, because the old log vocabulary collapsed every risk "
        "block into a single bucket — the exact ambiguity this funnel exists to remove — "
        "so a window that entirely predates that table returns an empty list. Defaults to "
        "the last 14 days when `from` is omitted, to bound the query. Read-only; publishes "
        "no events."
    ),
    responses={404: {"description": "No account with this `account_id` is configured."}},
)
async def get_signal_funnel(
    account: AccountRuntimeDep,
    skill: str | None = Query(
        default=None,
        description="Restrict to one bot by full id, e.g. 'normal/xauusd/breakout_v1'. "
        "Omit for every bot that fired in the window.",
    ),
    frm: int | None = Query(
        default=None, alias="from", description="Range start, epoch seconds UTC (inclusive)."
    ),
    to: int | None = Query(default=None, description="Range end, epoch seconds UTC (inclusive)."),
) -> list[BotFunnelOut]:
    funnels: list[BotFunnel] = await _service(account).get_signal_funnel(
        skill=skill, created_from=frm, created_to=to
    )
    return [
        BotFunnelOut(
            bot=f.bot,
            symbols=list(f.symbols),
            fired=f.fired,
            passed_htf=f.passed_htf,
            sized_ok=f.sized_ok,
            passed_spread=f.passed_spread,
            filled=f.filled,
            drops=[
                FunnelDropOut(
                    stage=d.stage,
                    outcome=d.outcome,
                    count=d.count,
                    example_reason=d.example_reason,
                )
                for d in f.drops
            ],
        )
        for f in funnels
    ]


@router.post(
    "/history/delete-by-ids",
    response_model=LogDeleteResult,
    summary="Delete specific activity log entries",
    description=(
        "Hard-deletes the given log rows by id. Backs both single-row delete and "
        "multi-select bulk delete in the activity log UI. This cannot be undone."
    ),
)
async def delete_by_ids(account: AccountRuntimeDep, body: LogDeleteByIdsRequest) -> LogDeleteResult:
    deleted = await _service(account).delete_by_ids(body.ids)
    return LogDeleteResult(deleted=deleted)


@router.post(
    "/history/delete-by-filter",
    response_model=LogDeleteResult,
    summary="Delete every activity log entry matching a filter",
    description=(
        "Hard-deletes every log row matching the given filters, using the same filter "
        "semantics as `GET /activity/history` (omitting all fields deletes every row). "
        "Backs 'delete all matching' in the activity log UI. This cannot be undone."
    ),
)
async def delete_by_filter(
    account: AccountRuntimeDep, body: LogDeleteByFilterRequest
) -> LogDeleteResult:
    deleted = await _service(account).delete_by_filter(
        level=body.level,
        logger_contains=body.logger_contains,
        q=body.q,
        created_from=body.created_from,
        created_to=body.created_to,
    )
    return LogDeleteResult(deleted=deleted)
