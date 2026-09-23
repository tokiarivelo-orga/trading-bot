"""Activity log domain: one persisted line of what the bot did and why.

Every `logging.getLogger("src.*")` call at INFO+ ends up here too (see
`shared/logging/adapters/handler.py`) — this is what backs the "why did/didn't
it take a position" question after the fact, not just live stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, kw_only=True)
class LogEntry:
    id: int | None
    created_at: datetime
    level: str
    logger: str
    message: str
    signal_id: str | None = None
    """Correlation id (OBSERVABILITY_PLAN.md Phase 5) — the signal this line
    was emitted while processing, `None` for lines outside any signal's
    processing window. See `shared/logging/account_context.py`."""


@dataclass(frozen=True, kw_only=True)
class DecisionCheck:
    """One gate the engine evaluated on its way from signal to fill, recorded
    with the numbers it actually saw (OBSERVABILITY_PLAN.md Phase 2).

    Deliberately the same five-field shape as `TradeRecord.indicators`
    (`journal/domain/models.py`) — `(name, value, threshold, comparison,
    passed)` — so "what did the strategy see" and "what did the engine check"
    read identically in the UI. `value`/`threshold` are floats even for
    boolean-ish gates (1.0/0.0), and `comparison` is the operator that was
    applied, e.g. `"<="`, `">"`, `"=="`.
    """

    name: str
    """Gate id, e.g. "htf_confirm", "spread_points", "risk_reward",
    "open_positions", "position_volume" ("volatility_percentile" appears on
    historical rows only)."""
    value: float
    threshold: float
    comparison: str
    passed: bool


# The closed outcome vocabulary of `SignalDecision.outcome`, in the order the
# engine evaluates the gates. Phase 2 split the old collapsed `risk_rejected`
# bucket into named reasons; `risk_rejected` itself stays in the vocabulary as
# the catch-all for a pre-trade risk block that isn't one of the named ones
# (and so historical rows keep rendering). `volatility_guard` is the same kind
# of legacy value: the engine's volatility guard was removed and nothing emits
# it any more, but rows recorded while it existed still carry it.
SIGNAL_OUTCOMES: tuple[str, ...] = (
    "skipped",
    "daily_loss_breaker",
    "risk_rejected",
    "htf_veto",
    "volatility_guard",
    "max_positions",
    "risk_sizing",
    "spread_veto",
    "rr_gate",
    "broker_rejected",
    "opened",
)


@dataclass(frozen=True, kw_only=True)
class SignalDecision:
    """One strategy signal and what the engine decided to do with it, recorded
    as a first-class row at the moment it happens — the typed replacement for
    regex-scraping `SIGNAL:`/`ENTRY *` log lines back out of `LogEntry`
    (`application/bot_signals.py`, now legacy/backfill-only).

    The engine writes one of these when a signal fires (`outcome="skipped"`,
    i.e. no terminal outcome yet) and updates `outcome` once the entry is
    filled, vetoed, or rejected. Human-readable log lines are still emitted —
    they are just no longer the source of truth.
    """

    signal_id: str
    """UUID4 hex assigned by the engine when the signal fires — the join key
    every later outcome update targets."""
    account_id: str
    bot: str  # full skill id, e.g. "normal/xauusd/breakout_v1"
    strategy: str  # StrategySpec.name
    symbol: str
    timeframe: str  # the bot's own entry timeframe, e.g. "M5"
    direction: str  # "buy" | "sell"
    price: float | None
    """Reference price the engine saw when the signal fired — ask for a buy,
    bid for a sell."""
    created_at: datetime
    outcome: str
    """One of `SIGNAL_OUTCOMES` — the closed vocabulary above."""
    reason: str
    confidence: float | None = None
    checks: tuple[DecisionCheck, ...] = ()
    """Every gate evaluated for this signal, in evaluation order, appended as
    the engine walks them. Empty for decisions recorded before Phase 2 and for
    signals that never reached a gate."""
    # ── Regime tagging (OBSERVABILITY_PLAN.md Phase 6) ─────────────────────
    # Snapshotted once, at the same moment the decision itself is recorded
    # (`TradeEngine._enter_for_bot`, via `engine.domain.regime.
    # compute_entry_regime`) — every recorded signal is tagged, not only the
    # ones that fill, so regime-split analytics can answer "what regime were
    # the *vetoed* signals in" too. None for decisions recorded before Phase
    # 6 and for signals whose entry timeframe had no candles to classify.
    regime_volatility: str | None = None
    """`VolatilityRegime` value at signal time — 'low'/'normal'/'high'/
    'extreme'."""
    regime_volatility_percentile: float | None = None
    """The ATR percentile rank behind `regime_volatility` (0-100)."""
    regime_trend: str | None = None
    """`TrendRegime` value at signal time — 'trending'/'ranging'."""
    regime_adx: float | None = None
    """Raw ADX reading behind `regime_trend`."""
    regime_session: str | None = None
    """`TradingSession` value at signal time — 'asian'/'london'/'overlap'/
    'new_york'/'off_session'."""


@dataclass(frozen=True, kw_only=True)
class BotSignal:
    """One strategy signal a live bot emitted — including ones that never
    became a trade (vetoed or rejected). Reconstructed from this bot's own
    `LogEntry` decision-trail lines (see `application/bot_signals.py`), the
    live analog of `backtest.domain.models.BacktestSignal` — kept as a
    separate type rather than importing that one so `activity` doesn't reach
    into the `backtest` module's internals (see CLAUDE.md)."""

    time: datetime
    direction: str  # "buy" | "sell"
    outcome: str
    """"opened" | "htf_veto" | "risk_rejected" | "spread_veto" | "broker_rejected"
    | "skipped" (no outcome line followed within the queried window)."""
    reason: str  # the strategy's own Signal.reason (pattern, zone, entry/sl/tp lines)
    price: float | None = None
    """Reference price the engine saw when the signal fired — ask for a buy,
    bid for a sell. `None` for legacy log lines logged before the price was
    added to the `SIGNAL:` line, so the chart must treat it as optional."""
    checks: tuple[DecisionCheck, ...] = ()
    """The gates evaluated for this signal (Phase 2). Always empty on the
    legacy log-scrape path, which has no structured checks to recover."""
