"""Wire schemas for the `/news` HTTP API. Mirrors `news/domain/models.py`;
times are epoch seconds UTC, matching the market-data/candle convention.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NewsEventOut(BaseModel):
    """One upcoming calendar release."""

    name: str
    time: int = Field(description="Scheduled release time, epoch seconds UTC.")
    impact: str = Field(description="'low', 'medium', or 'high'.")
    currency: str = Field(description="Affected currency/country code, e.g. 'USD'.")
    skill: str | None = Field(
        default=None,
        description="Matched news skill name from `configs/news.yaml: tracked_events`, "
        "if any — null means this event never activates a news window.",
    )
    forecast: str | None = Field(
        default=None,
        description="Consensus estimate, formatted as the source publishes it "
        "(e.g. '8.5%', '1950B'). Null if the source doesn't provide one.",
    )
    previous: str | None = Field(
        default=None, description="Prior period's reading, same formatting caveat as `forecast`."
    )
    actual: str | None = Field(
        default=None,
        description="Released value, if the event has already happened and the source "
        "reports it. ForexFactory's calendar never populates this; Finnhub does once "
        "released.",
    )


class NewsEventRecordOut(BaseModel):
    """One persisted calendar event, read from the durable `news_events`
    table (Phase 6 Part B) — the historical counterpart to `NewsEventOut`,
    which serves the live, in-memory `/news/upcoming` cache (lost on every
    restart) and carries a resolved `skill` instead of balance context.
    Backs `GET /accounts/{account_id}/news/events`."""

    id: int = Field(description="Row id in `news_events`.")
    name: str = Field(description="Calendar event name, e.g. 'Non-Farm Payrolls'.")
    time: int = Field(description="Scheduled release time, epoch seconds UTC.")
    impact: str = Field(description="'low', 'medium', or 'high'.")
    currency: str = Field(description="Affected currency/country code, e.g. 'USD'.")
    forecast: str | None = Field(
        default=None,
        description="Consensus estimate, formatted as the source publishes it "
        "(e.g. '8.5%', '1950B'). Null if the source doesn't provide one.",
    )
    previous: str | None = Field(
        default=None, description="Prior period's reading, same formatting caveat as `forecast`."
    )
    actual: str | None = Field(
        default=None,
        description="Released value, if the event has already happened and the source "
        "reports it. Null before release, or if the source never populates it.",
    )
    balance_before: float | None = Field(
        default=None,
        description="Account balance immediately before this event's news window opened "
        "(captured on `NewsWindowEntered`, Phase 6 Part C). Null when this event never "
        "activated a tracked news window, or the window hasn't opened yet.",
    )
    balance_after: float | None = Field(
        default=None,
        description="Account balance immediately after this event's news window closed "
        "(captured on `NewsWindowExited`). Null when the window hasn't closed yet, or "
        "never formally exited (e.g. a process restart mid-window) — enrichment, not "
        "something callers should treat as a data-quality error.",
    )


class NewsWindowOut(BaseModel):
    """A currently-active before/after window around one calendar event."""

    event: NewsEventOut
    skill: str = Field(description="The news skill whose activation window is active.")
    window_start: int = Field(description="Epoch seconds UTC.")
    window_end: int = Field(description="Epoch seconds UTC.")
    phase: str = Field(description="'pre' (before the release) or 'post' (after it).")
    symbols: list[str] = Field(
        description="Symbols this window affects (`skills/news/<skill>.yaml: "
        "activation.symbols`) — the chart uses this to shade only the symbols "
        "actually under a news skill right now."
    )
