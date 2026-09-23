"""Wire schema for the `/activity` HTTP API. Mirrors `activity/domain/models.py`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LogEntryOut(BaseModel):
    """One persisted activity-log line — a bot decision, fill, veto, or
    status change, in the order it happened."""

    id: int = Field(description="Autoincrement row id, also the natural newest-first sort key.")
    created_at: int = Field(description="Epoch seconds UTC when the log line was emitted.")
    level: str = Field(description="Python logging level name, e.g. 'INFO', 'WARNING', 'ERROR'.")
    logger: str = Field(
        description="Originating logger name, e.g. 'src.engine.application.trade_loop' — "
        "identifies which module made the decision."
    )
    message: str = Field(description="The formatted log message, e.g. why a signal was vetoed.")
    signal_id: str | None = Field(
        default=None,
        description="Correlation id (OBSERVABILITY_PLAN.md Phase 5) joining every line from "
        "signal -> sizing -> order -> fill -> journal for one signal. `None` for lines emitted "
        "outside any signal's processing window (health checks, candle polling, ...). Filter "
        "`GET /activity/history?signal_id=...` by one to read that signal's whole life in order.",
    )


class LogHistoryPage(BaseModel):
    """One page of the filtered activity log history (`GET /activity/history`)."""

    items: list[LogEntryOut] = Field(description="Log entries matching the filters, one page.")
    total: int = Field(description="Total entries matching the filters, across all pages.")


class LogDeleteByIdsRequest(BaseModel):
    """Request body for deleting specific activity log rows — backs single-row
    delete and multi-select bulk delete in the activity log UI."""

    ids: list[int] = Field(
        description="Row ids to delete, as returned by `GET /activity/history`.", min_length=1
    )


class LogDeleteByFilterRequest(BaseModel):
    """Request body for deleting every activity log row matching a filter —
    backs "delete all matching" in the activity log UI. Mirrors the query
    filters of `GET /activity/history`; omitting all fields deletes every row."""

    level: str | None = Field(
        default=None, description="Exact level match, e.g. 'INFO', 'WARNING', 'ERROR'."
    )
    logger_contains: str | None = Field(
        default=None,
        description="Substring match on the logger name, e.g. 'trade_loop' or 'broker'.",
    )
    q: str | None = Field(
        default=None, description="Substring match on the message text, e.g. a symbol or reason."
    )
    created_from: int | None = Field(
        default=None, description="Only entries at/after this epoch-seconds UTC."
    )
    created_to: int | None = Field(
        default=None, description="Only entries at/before this epoch-seconds UTC."
    )


class LogDeleteResult(BaseModel):
    """Result of a bulk or single activity-log delete."""

    deleted: int = Field(description="Number of log entries removed.")


class DecisionCheckOut(BaseModel):
    """One gate the engine evaluated for a signal, with the numbers it saw.
    Same five-field shape as a trade's `indicators` readings, so the two read
    identically in the UI."""

    name: str = Field(
        description="Gate id: 'htf_confirm', 'open_positions', 'position_volume', "
        "'spread_points', or 'risk_reward' ('volatility_percentile' on historical rows only)."
    )
    value: float = Field(
        description="What the gate measured, e.g. the live spread in points. Boolean gates "
        "use 1.0/0.0."
    )
    threshold: float = Field(description="What it was measured against, e.g. the spread cap.")
    comparison: str = Field(
        description="The operator applied between value and threshold, e.g. '<=', '>=', '=='."
    )
    passed: bool = Field(description="Whether the signal cleared this gate.")


class FunnelDropOut(BaseModel):
    """One reason signals stopped at a given funnel stage, with how often."""

    stage: str = Field(
        description="The stage these signals failed to reach: 'passed_htf', 'sized_ok', "
        "'passed_spread', or 'filled'."
    )
    outcome: str = Field(
        description="The decision outcome that stopped them, e.g. 'htf_veto', "
        "'max_positions', 'risk_sizing', 'spread_veto', 'rr_gate', "
        "'broker_rejected', 'daily_loss_breaker', 'risk_rejected', 'skipped'."
    )
    count: int = Field(description="How many of this bot's signals dropped out this way.")
    example_reason: str = Field(
        description="The reason text of one dropped signal, so the count is actionable "
        "without a second query."
    )


class BotFunnelOut(BaseModel):
    """One bot's signal→fill funnel over the queried period. The counts are
    monotonically non-increasing and follow the engine's real gate order:
    HTF confirmation, then position cap / lot sizing, then
    the broker's spread + risk-reward gate, then the fill."""

    bot: str = Field(description="Full bot id, e.g. 'normal/xauusd/breakout_v1'.")
    symbols: list[str] = Field(
        description="Every symbol this bot fired a signal on in the period, sorted."
    )
    fired: int = Field(description="Signals the strategy emitted.")
    passed_htf: int = Field(
        description="Of those, how many cleared the higher-timeframe confirmation and the "
        "pre-trade risk gate."
    )
    sized_ok: int = Field(
        description="Of those, how many cleared the open-position cap and produced a "
        "tradable lot size."
    )
    passed_spread: int = Field(
        description="Of those, how many cleared the broker spread cap and the "
        "spread-adjusted risk-reward floor."
    )
    filled: int = Field(description="Of those, how many the broker actually filled.")
    drops: list[FunnelDropOut] = Field(
        description="Why the rest stopped, grouped by stage then outcome, earliest stage first."
    )


class BotSignalOut(BaseModel):
    """One strategy signal a live bot emitted — including signals that never
    became a trade (vetoed or rejected), so the chart can show every setup the
    strategy saw for this bot, not only its fills. Read from the
    `signal_decisions` table, with a legacy log-scrape fallback for the window
    that predates it (see `GET /activity/signals`)."""

    time: int = Field(description="Epoch seconds UTC — when the strategy emitted this signal.")
    direction: str = Field(description="'buy' or 'sell'.")
    outcome: str = Field(
        description="What the engine did with it: 'opened' (became a trade), 'htf_veto' "
        "(higher-timeframe trend opposed it), 'volatility_guard' (historical rows only: the "
        "removed EXTREME-ATR entry block), "
        "'max_positions' (open-position cap), 'risk_sizing' (no tradable lot size), "
        "'spread_veto' (live spread over the cap), 'rr_gate' (spread-adjusted risk-reward "
        "floor), 'daily_loss_breaker' (a circuit breaker had the engine paused), "
        "'broker_rejected' (the broker/MT5 itself refused the order), 'risk_rejected' "
        "(any other pre-trade risk block — also every pre-Phase-2 row, where the named "
        "buckets above were collapsed into this one), or 'skipped' (no outcome yet)."
    )
    reason: str = Field(
        description="The strategy's own reason string — pattern matched, zone rectangle, "
        "entry/SL/TP lines, confirmations — with the outcome line's own explanation "
        "(veto reason, sizing failure) appended after an em dash when there is one."
    )
    price: float | None = Field(
        default=None,
        description="Reference price the engine saw when the signal fired (ask for a buy, "
        "bid for a sell), so the chart can place the marker at the signal's own level. "
        "Null for log lines written before the price was recorded.",
    )
    checks: list[DecisionCheckOut] = Field(
        default_factory=list,
        description="Every gate the engine evaluated for this signal, in evaluation order, "
        "with the numbers it saw. Empty for signals recorded before this was captured and "
        "for rows recovered from the legacy log-scrape path.",
    )
