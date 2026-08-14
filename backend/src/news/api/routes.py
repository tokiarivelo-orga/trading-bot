"""Economic calendar REST endpoints (§6.7, §8). Read-only status for the UI —
the engine reacts to news windows internally via `NewsSkillSelector` and the
`NewsWindowEntered`/`NewsWindowExited` events; nothing here can trigger a
flatten or change trading behavior.

Two routers: `router` (unprefixed `/news/...`) is the original live/
process-wide status API — `NewsWindowService` has no account dimension (one
calendar, shared across every account, see its module docstring), matching
how `main.py`'s `accounts` tag documents "news" as one of the few
process-wide, unprefixed route groups. `events_router` (Phase 6 Part B,
`/accounts/{account_id}/news/...`) is the new persisted-history endpoint;
its path still takes `account_id` (purely to reuse `AccountRuntimeDep`'s
existence check — the query itself is not filtered by it, since the
underlying `news_events` table has no per-account rows either).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from src.news.api.schemas import NewsEventOut, NewsEventRecordOut, NewsWindowOut
from src.news.domain.models import ImpactLevel
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/news", tags=["news"])
events_router = APIRouter(prefix="/accounts/{account_id}/news", tags=["news"])


def _container(request: Request) -> Any:
    return request.app.state.container


def _event_out(event: Any, skill: str | None) -> NewsEventOut:
    return NewsEventOut(
        name=event.name,
        time=int(event.time.timestamp()),
        impact=event.impact.value,
        currency=event.currency,
        skill=skill,
        forecast=event.forecast,
        previous=event.previous,
        actual=event.actual,
    )


@router.get(
    "/upcoming",
    response_model=list[NewsEventOut],
    summary="List upcoming calendar events",
    description=(
        "Events from the configured calendar source (`configs/news.yaml: "
        "calendar.source`) within `days_ahead`, from the last background "
        "refresh — this never fetches the calendar live, so it returns "
        "instantly. Each event's `skill` field shows which news skill (if "
        "any) will activate a trading window around it, resolved from "
        "`configs/news.yaml: tracked_events`."
    ),
)
async def get_upcoming(
    request: Request,
    days_ahead: Annotated[
        int, Query(ge=1, le=30, description="How many days ahead to include.")
    ] = 7,
) -> list[NewsEventOut]:
    service = _container(request).news_window_service
    return [_event_out(event, service.skill_for(event)) for event in service.upcoming(days_ahead)]


@router.get(
    "/active-windows",
    response_model=list[NewsWindowOut],
    summary="List currently active news windows",
    description=(
        "News windows active right now — i.e. `configs/app.yaml`-configured "
        "symbols currently trading under a news skill instead of their "
        "normal one, per `NewsSkillSelector`'s priority "
        "'news skill > symbol normal skill > global default' (§6.6). Empty "
        "outside any tracked event's before/after window."
    ),
)
async def get_active_windows(request: Request) -> list[NewsWindowOut]:
    service = _container(request).news_window_service
    now = datetime.now(UTC)
    out = []
    for window in service.active_windows(now):
        out.append(
            NewsWindowOut(
                event=_event_out(window.event, window.skill),
                skill=window.skill,
                window_start=int(window.window_start.timestamp()),
                window_end=int(window.window_end.timestamp()),
                phase="pre" if window.is_pre(now) else "post",
                symbols=list(service.symbols_for(window.skill)),
            )
        )
    return out


def _record_out(record: tuple[int, Any, float | None, float | None]) -> NewsEventRecordOut:
    row_id, event, balance_before, balance_after = record
    return NewsEventRecordOut(
        id=row_id,
        name=event.name,
        time=int(event.time.timestamp()),
        impact=event.impact.value,
        currency=event.currency,
        forecast=event.forecast,
        previous=event.previous,
        actual=event.actual,
        balance_before=balance_before,
        balance_after=balance_after,
    )


@events_router.get(
    "/events",
    response_model=list[NewsEventRecordOut],
    summary="Get persisted calendar event history",
    description=(
        "Persisted economic-calendar events (Phase 6 Part B) with scheduled time in "
        "`[start, end]` (epoch seconds UTC, both inclusive), optionally filtered to one "
        "`impact` level. Unlike `GET /news/upcoming` (which reads `NewsWindowService`'s "
        "in-memory cache — always the current 7-day-ahead window, and lost on every "
        "restart), this reads the durable `news_events` table, so it covers any "
        "historical range and survives process restarts. Each row also reports "
        "`balance_before`/`balance_after` (Phase 6 Part C) — the account balance "
        "immediately before/after this event's news window opened/closed, captured on "
        "`NewsWindowEntered`/`NewsWindowExited` — null when the event never activated a "
        "tracked news window, or its window hasn't opened/closed yet."
    ),
    responses={
        404: {"description": "Unknown account_id — see `GET /accounts` for valid values."}
    },
)
async def get_news_events(
    account: AccountRuntimeDep,
    request: Request,
    start: Annotated[
        int, Query(description="Only events scheduled at/after this epoch-seconds UTC.")
    ],
    end: Annotated[
        int, Query(description="Only events scheduled at/before this epoch-seconds UTC.")
    ],
    impact: Annotated[
        ImpactLevel | None,
        Query(description="Restrict to one impact level ('low'/'medium'/'high')."),
    ] = None,
) -> list[NewsEventRecordOut]:
    del account  # only used for AccountRuntimeDep's account_id existence check
    service = _container(request).news_window_service
    start_dt = datetime.fromtimestamp(start, tz=UTC)
    end_dt = datetime.fromtimestamp(end, tz=UTC)
    records = await service.list_events(start_dt, end_dt, impact)
    return [_record_out(r) for r in records]
