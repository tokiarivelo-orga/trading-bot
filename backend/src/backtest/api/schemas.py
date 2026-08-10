"""Wire schema for the `/backtest` HTTP API. Mirrors `backtest/domain/models.py`;
these endpoints only ever read reports written by `python -m src.backtest.cli`
(see `backtest/reports/writer.py`) — they never run a backtest themselves."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneOut(BaseModel):
    """The supply/demand rectangle the strategy identified before entering
    this trade, for drawing on the chart. Only present for strategies that
    report one (e.g. `pob_price_action_snd`)."""

    kind: str = Field(description="'demand' (buy zone) or 'supply' (sell zone).")
    price_low: float = Field(description="Lower bound of the zone rectangle.")
    price_high: float = Field(description="Upper bound of the zone rectangle.")
    time_start: int = Field(description="Epoch seconds UTC — left edge of the zone rectangle.")
    time_end: int = Field(
        description="Epoch seconds UTC — right edge of the zone rectangle (the entry candle)."
    )
    pattern: str | None = Field(
        default=None,
        description=(
            "The zone's own subtype, e.g. 'RBR', 'DBD', 'RBD', 'DBR' — distinct from the "
            "trade's own `pattern` (confirming candlestick pattern). Null for strategies "
            "that don't label their zone detector's setup type, or reports predating "
            "this field."
        ),
    )


class StructurePointOut(BaseModel):
    """A single labeled swing point from the window the strategy used to
    validate this trade's zone — chart annotation only, not used to gate
    the trade itself."""

    label: str = Field(description="Swing-structure label: 'HH', 'HL', 'LH', or 'LL'.")
    price: float = Field(description="Price of the swing high/low.")
    time: int = Field(description="Epoch seconds UTC of the swing bar.")


class BacktestTradeOut(BaseModel):
    side: str = Field(description="'buy' or 'sell'.")
    volume: float
    open_time: int = Field(description="Epoch seconds UTC.")
    open_price: float
    sl: float | None
    tp: float | None
    close_time: int = Field(description="Epoch seconds UTC.")
    close_price: float
    profit: float
    r_multiple: float | None = Field(
        description="Profit / initial risk in account currency; null if the trade had no SL."
    )
    zone: ZoneOut | None = Field(
        default=None,
        description="Supply/demand zone this trade was taken from, if the strategy reports one.",
    )
    pattern: str | None = Field(
        default=None,
        description="Confirming candlestick pattern, e.g. 'bullish_engulfing', if reported.",
    )
    structure: list[StructurePointOut] = Field(
        default_factory=list,
        description=(
            "Labeled swing points (HH/HL/LH/LL) from the window the strategy used to "
            "validate this trade's zone, for chart annotation."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "The strategy's own Signal.reason that led to this trade — pattern, zone, "
            "entry/SL/TP lines. Empty for report files predating this field."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description="Strategy's confidence in this signal, 0..1. Null if not reported.",
    )


class EquityPointOut(BaseModel):
    time: int = Field(description="Epoch seconds UTC.")
    balance: float


class ActivityLogEntryOut(BaseModel):
    """One decision-trail line captured while replaying — a signal, an HTF
    veto, a risk-sizing rejection, a fill, a circuit breaker — so a report
    with zero trades still explains why, without needing server stdout."""

    time: int = Field(
        description="Epoch seconds UTC — the simulated bot clock (bar close "
        "time) when this was logged, not wall-clock time."
    )
    level: str = Field(description="Python logging level name, e.g. 'INFO', 'WARNING'.")
    logger: str = Field(
        description="Originating logger name, e.g. 'src.engine.application.trade_loop' — "
        "identifies which module made the decision."
    )
    message: str = Field(description="The formatted log message, e.g. why a signal was vetoed.")


class BacktestSignalOut(BaseModel):
    """One strategy signal emitted during the replay — including signals that
    never became trades (vetoed or rejected by the engine). Lets the report
    page and chart show every valid setup the strategy saw, not only fills."""

    time: int = Field(
        description="Epoch seconds UTC — the simulated bot clock (bar close time) "
        "when the strategy emitted this signal."
    )
    direction: str = Field(description="'buy' or 'sell'.")
    outcome: str = Field(
        description="What the engine did with it, from the same closed vocabulary the "
        "live decision trail uses: 'opened' (became a trade), 'htf_veto' (higher-timeframe "
        "trend opposed it), 'volatility_guard', 'max_positions', 'risk_sizing', "
        "'spread_veto', 'rr_gate', 'broker_rejected' (the broker refused the order — e.g. "
        "stops closer than the symbol's stops_level), 'daily_loss_breaker', or 'skipped'. "
        "Reports written before backtests recorded structured decisions instead collapse "
        "every risk block into the legacy 'risk_rejected' value."
    )
    reason: str = Field(
        description="The strategy's own reason string — pattern matched, zone rectangle, "
        "entry/SL/TP lines, confirmations."
    )
    price: float | None = Field(
        default=None,
        description="Reference price the engine saw when the signal fired — ask for a buy, "
        "bid for a sell. Null in reports produced before the backtest recorded structured "
        "decisions, which never captured it.",
    )


class BacktestRejectionOut(BaseModel):
    """How many entries the simulated broker refused for one reason, and why.

    A rejection is a signal the strategy produced, the risk manager sized and
    the spread gate cleared — that a real broker would then have thrown away.
    A large count here means the backtest's headline numbers describe trades
    that could never have been placed."""

    reason: str = Field(
        description="Which broker rule refused it: 'stops_level' (SL or TP closer to price "
        "than the symbol's minimum distance), 'volume_below_min' or 'volume_above_max' "
        "(lot size off the broker's volume grid)."
    )
    count: int = Field(description="Number of entries refused for this reason during the run.")
    retcode: int = Field(
        description="The MT5 return code a live server would have produced — 10016 for "
        "invalid stops, 10014 for an invalid volume."
    )
    example: str = Field(
        description="The first refusal's own message, including the concrete distances or "
        "lot sizes involved, so the count is actionable."
    )


class BrokerRealismOut(BaseModel):
    """Which broker constraints the run simulated, and what they cost.

    Backtests before this existed filled every order at the bar's closing
    quote in exactly the requested lot size, which is why an M1 scalp whose
    stops sat inside the symbol's `stops_level` could report a profitable
    equity curve while every live order was refused with retcode 10016."""

    enabled: bool = Field(
        default=False,
        description="Whether broker constraints were simulated at all. False for reports "
        "written before this feature existed and for runs that explicitly disabled it — "
        "their trade lists may include entries a live broker would have refused.",
    )
    stops_level_enforced: bool = Field(
        default=False,
        description="Whether entries with an SL/TP closer to price than the symbol's "
        "stops_level were refused.",
    )
    volume_grid_enforced: bool = Field(
        default=False,
        description="Whether lot sizes were rounded down to volume_step and refused when "
        "outside volume_min..volume_max.",
    )
    clamp_stops: bool = Field(
        default=False,
        description="Research mode: a too-close SL/TP was widened to the broker minimum "
        "instead of the entry being refused. True means the reported trades risked more "
        "than the risk manager sized them for, so their R multiples are not comparable "
        "with a normal run's.",
    )
    spread_widening_factor: float = Field(
        default=1.0,
        description="Multiplier applied to each bar's recorded closing spread, since real "
        "entries do not happen at the close. 1.0 means the historical spread was used "
        "as-is.",
    )
    slippage_mean: float = Field(
        default=0.0,
        description="Mean slippage applied per fill, in price units, positive = it cost "
        "the trader.",
    )
    slippage_stddev: float = Field(
        default=0.0, description="Standard deviation of the slippage distribution, in price units."
    )
    slippage_source: str = Field(
        default="none",
        description="'live' — calibrated from real measured fills on this symbol; "
        "'fallback' — not enough live fills yet, so a documented pessimistic default was "
        "used and these numbers are a guess; 'none' — no slippage was simulated.",
    )
    slippage_sample_count: int = Field(
        default=0,
        description="How many real measured fills the slippage model was calibrated from.",
    )
    accepted_count: int = Field(
        default=0, description="Entries the simulated broker filled."
    )
    clamped_count: int = Field(
        default=0,
        description="Entries whose SL/TP was widened to the broker minimum rather than "
        "refused. Always 0 unless clamp_stops is true.",
    )
    rejected_count: int = Field(
        default=0, description="Entries the simulated broker refused, across all reasons."
    )
    rejections: list[BacktestRejectionOut] = Field(
        default_factory=list,
        description="Per-reason breakdown of the refusals, most frequent first.",
    )


class BacktestReportSummaryOut(BaseModel):
    """One report file's headline stats — used by the report list view."""

    id: str = Field(
        description="Report identifier; fetch full detail at GET /backtest/reports/{id}."
    )
    strategy: str
    symbol: str
    period: str = Field(description="'YYYY-MM:YYYY-MM' as passed to the CLI.")
    trade_count: int
    win_rate: float = Field(description="Fraction of trades with positive profit, 0..1.")
    profit_factor: float | None = Field(
        description="Gross profit / gross loss; null means no losing trades (infinite)."
    )
    max_drawdown_pct: float
    avg_r: float
    worst_losing_streak: int
    starting_balance: float
    ending_balance: float
    min_rr: float = Field(
        default=1.0,
        description="The spread-adjusted minimum reward:risk ratio SpreadGate actually "
        "enforced for this run — configs/symbols/<symbol>.yaml's value, this run's own "
        "min_rr override, or 1.0 (SpreadGate.DEFAULT_MIN_RR) if the symbol has no config "
        "file. A run parameter like starting_balance, not a fixed strategy property — "
        "older report files predating this field default to 1.0.",
    )
    risk_per_trade_pct: float = Field(
        default=0.5,
        description="% of balance risked per trade — configs/risk.yaml's value, or this "
        "run's own override. A run parameter, not a fixed strategy property.",
    )
    daily_loss_limit_pct: float = Field(
        default=2.0,
        description="Circuit breaker: the engine pauses once a trading day's realized loss "
        "reaches this — and never auto-resumes (see RiskManager.record_trade_closed), so a "
        "trade count far lower than the period would suggest often means this tripped early "
        "and the rest of the run saw every entry blocked, not that no more setups occurred.",
    )
    max_open_positions: int = Field(
        default=100, description="Circuit breaker: cap on simultaneous positions for this run."
    )
    max_trades_per_day_enabled: bool = Field(
        default=False,
        description="Manual daily kill switch active for this run — not a count. True means "
        "every new trade was rejected for the rest of the trading day once flipped on; false "
        "means entries were unlimited. Older report files predating this field default to "
        "false.",
    )
    consecutive_loss_pause: int = Field(
        default=10,
        description="Circuit breaker: the engine pauses (same never-auto-resumes caveat as "
        "daily_loss_limit_pct) after this many losing trades in a row.",
    )
    min_lot_fallback_enabled: bool = Field(
        default=False,
        description="Whether the broker-minimum-lot sizing fallback was enabled for this "
        "run (see RiskManager.size_position) — configs/risk.yaml's value, this run's own "
        "override, or the live engine override if neither was passed.",
    )
    max_risk_per_trade_pct: float | None = Field(
        default=None,
        description="Ceiling (%) for the minimum-lot fallback's effective risk, for this "
        "run. Only matters when min_lot_fallback_enabled is true. Null means the fallback "
        "(when enabled) used risk_per_trade_pct itself as the ceiling.",
    )
    broker_realism: BrokerRealismOut = Field(
        default_factory=BrokerRealismOut,
        description="Which broker constraints (stops_level, lot grid, spread widening, "
        "slippage) this run simulated and how many entries they refused. Reports predating "
        "this field report enabled=false, which correctly describes them: they simulated "
        "nothing and may show trades a live broker would have refused.",
    )


class BacktestReportDetailOut(BacktestReportSummaryOut):
    """Full report: headline stats plus every trade and the equity curve, for
    the report detail page's trade table and `lightweight-charts` equity plot."""

    trades: list[BacktestTradeOut]
    equity_curve: list[EquityPointOut]
    activity_log: list[ActivityLogEntryOut] = Field(
        default_factory=list,
        description="The bot's decision trail during the replay (signals, HTF vetoes, "
        "sizing rejections, fills, circuit breakers), oldest first — explains a "
        "zero-trade report. Older report files predating this field return an "
        "empty list.",
    )
    signals: list[BacktestSignalOut] = Field(
        default_factory=list,
        description="Every signal the strategy emitted during the replay (taken or "
        "vetoed), oldest first — the structured counterpart of the activity log's "
        "SIGNAL lines. Older report files predating this field return an empty list.",
    )


class ImportBacktestReportIn(BacktestReportSummaryOut):
    """Request body for `POST /backtest/reports/import` — the same shape as
    `BacktestReportDetailOut`. The exact JSON produced by downloading an
    existing report (`GET /backtest/reports/{id}`) validates against this
    as-is — including its `id`, which is accepted but ignored, since a
    fresh id is always assigned from the new file's name — so the trader
    can always get a valid example by downloading one."""

    id: str | None = Field(
        default=None,
        exclude=True,
        description="Ignored if present — a new id is always assigned from the import.",
    )
    trades: list[BacktestTradeOut]
    equity_curve: list[EquityPointOut]
    activity_log: list[ActivityLogEntryOut] = Field(
        default_factory=list,
        description="Same meaning as BacktestReportDetailOut.activity_log.",
    )
    signals: list[BacktestSignalOut] = Field(
        default_factory=list,
        description="Same meaning as BacktestReportDetailOut.signals.",
    )


class BacktestReportListOut(BaseModel):
    """One page of the saved-report list, newest first."""

    items: list[BacktestReportSummaryOut] = Field(
        description="Report summaries for this page, newest first."
    )
    total: int = Field(description="Total number of saved report files, across all pages.")
    limit: int = Field(description="Page size that was applied.")
    offset: int = Field(description="Number of newest reports skipped before this page.")


# ── Walk-forward harness (OBSERVABILITY_PLAN.md Phase 6 Pass B) ──────────────
#
# Mirrors the single-backtest job/report schemas above field-by-field, for a
# harness that runs run_backtest() independently over consecutive
# `fold_months`-sized sub-periods instead of one window — see
# `application/walk_forward.py`'s module docstring for why this is NOT
# classic in-sample-refit walk-forward optimization.


class WalkForwardRunIn(BaseModel):
    """Request body for `POST /backtest/walk-forward/run`. Same fields as
    `RunBacktestIn` (the single-backtest launcher) plus `fold_months`."""

    strategy_id: str = Field(description="A bot `id` from GET /backtest/bots.")
    symbol: str = Field(description="Symbol to backtest, e.g. 'XAUUSD'.")
    period: str = Field(
        description="'YYYY-MM:YYYY-MM' — the FULL range to split into folds. Must match "
        "candle history already in the DB, or be backfillable from the gateway."
    )
    fold_months: int = Field(
        default=1,
        ge=1,
        description="Size of each out-of-sample fold, in whole months. Default 1 (monthly "
        "folds). E.g. a 12-month period with fold_months=3 runs 4 independent quarterly "
        "backtests; the last fold is shorter than fold_months if the period doesn't divide "
        "evenly.",
    )
    starting_balance: float = Field(
        default=10_000.0,
        gt=0,
        description="Simulated account balance EVERY fold starts fresh from — folds do not "
        "compound into one another.",
    )
    min_lot_fallback_enabled: bool | None = Field(
        default=None,
        description="Same meaning as RunBacktestIn's field, applied to every fold. Null uses "
        "whatever is currently configured.",
    )
    max_risk_per_trade_pct: float | None = Field(
        default=None,
        gt=0,
        le=100,
        description="Same meaning as RunBacktestIn's field, applied to every fold. Null uses "
        "whatever is currently configured.",
    )
    min_rr: float | None = Field(
        default=None,
        gt=0,
        description="Same meaning as RunBacktestIn's field, applied to every fold. Null uses "
        "whatever is currently configured.",
    )


class WalkForwardRunOut(BaseModel):
    """Returned immediately by `POST /backtest/walk-forward/run` — the job
    runs in the background, poll `GET /backtest/walk-forward/run/{job_id}`."""

    job_id: str = Field(description="Pass this to GET /backtest/walk-forward/run/{job_id}.")
    status: str = Field(description="Always 'pending' at creation.")


class WalkForwardJobStatusOut(BaseModel):
    """Poll response for a walk-forward job. Same shape as the single-backtest
    JobStatusOut."""

    job_id: str = Field(description="Echoes the id from POST /backtest/walk-forward/run.")
    status: str = Field(description="'pending', 'running', 'done', or 'error'.")
    report_id: str | None = Field(
        default=None,
        description="Set once status is 'done' — fetch the full report at "
        "GET /backtest/walk-forward/reports/{report_id}.",
    )
    error: str | None = Field(
        default=None, description="Set when status is 'error' — human-readable failure reason."
    )


class FoldResultOut(BaseModel):
    """One fold's headline stats within a walk-forward report."""

    period: str = Field(description="This fold's own 'YYYY-MM:YYYY-MM' sub-range.")
    trade_count: int = Field(description="Trades this fold's independent backtest produced.")
    win_rate: float = Field(
        description="Fraction of this fold's trades with positive profit, 0..1."
    )
    profit_factor: float | None = Field(
        description="Gross profit / gross loss for this fold. Null when undefined: no losing "
        "trades, or no trades at all (unlike the single-backtest report, an empty fold is "
        "reported as null here, not 0, so it doesn't read as 'this fold lost everything')."
    )
    expectancy: float = Field(
        description="Average profit per trade in this fold, account currency. 0.0 if the "
        "fold had no trades."
    )
    broker_acceptance_rate: float = Field(
        description="Filled / (filled + refused) for this fold's simulated broker, 0..1."
    )


class WalkForwardReportSummaryOut(BaseModel):
    """Headline aggregate stats for one walk-forward run — used by the report
    list view. Does not include the per-fold breakdown; see
    WalkForwardReportDetailOut for that."""

    id: str = Field(
        description="Report identifier; fetch full detail at "
        "GET /backtest/walk-forward/reports/{id}."
    )
    strategy: str = Field(description="Strategy family name.")
    symbol: str = Field(description="Symbol this walk-forward run backtested, e.g. 'XAUUSD'.")
    period: str = Field(description="The full requested range, before splitting into folds.")
    fold_months: int = Field(description="Fold size, in whole months, this run used.")
    starting_balance: float = Field(description="Balance every fold started fresh from.")
    fold_count: int = Field(
        description="Number of folds that actually ran. Can be less than the number of "
        "'YYYY-MM' windows the period splits into if any were skipped for missing candle "
        "history — a bad/unbackfilled month is skipped, not fatal."
    )
    folds_with_zero_trades: int = Field(
        description="Of the folds that ran, how many produced zero trades."
    )
    losing_fold_count: int = Field(
        description="Of the folds that ran, how many had negative expectancy."
    )
    mean_profit_factor: float | None = Field(
        description="Mean profit factor across folds that had one (excludes folds with no "
        "trades or an infinite/undefined value). Null if no fold had one."
    )
    stddev_profit_factor: float | None = Field(
        description="Population standard deviation of the same set — a large value alongside "
        "a healthy mean means the edge is inconsistent across time, not just consistently "
        "positive."
    )
    worst_fold_profit_factor: float | None = Field(
        description="The lowest profit factor among folds that had one — the single worst "
        "window, which a single-period backtest's headline number can hide."
    )
    mean_expectancy: float = Field(
        description="Mean of every fold's expectancy (including zero-trade folds, which "
        "contribute 0.0). 0.0 if every fold was skipped."
    )


class WalkForwardReportDetailOut(WalkForwardReportSummaryOut):
    """Full walk-forward report: aggregate stats plus the per-fold breakdown."""

    folds: list[FoldResultOut] = Field(
        description="One entry per fold that actually ran, oldest first. A fold skipped for "
        "missing history is simply absent — see fold_count's description."
    )


class WalkForwardReportListOut(BaseModel):
    """One page of the saved walk-forward report list, newest first."""

    items: list[WalkForwardReportSummaryOut] = Field(
        description="Report summaries for this page, newest first."
    )
    total: int = Field(description="Total number of saved walk-forward report files.")
    limit: int = Field(description="Page size that was applied.")
    offset: int = Field(description="Number of newest reports skipped before this page.")


class DivergenceMetricOut(BaseModel):
    """One measurement compared between live trading and a backtest."""

    name: str = Field(
        description="Metric id: 'fill_rate', 'avg_slippage', 'win_rate', 'avg_profit', "
        "'avg_r', or 'avg_volume'."
    )
    kind: str = Field(
        description="'execution' — how the order was filled (fill rate, slippage, lot "
        "size); or 'outcome' — what the trade then earned (win rate, profit, R). Execution "
        "metrics diverging points at the simulator, outcome metrics alone at the edge."
    )
    live_value: float | None = Field(
        description="Value measured from live journalled trades. Null when no live sample "
        "carries this metric (e.g. slippage on trades journalled before it was recorded)."
    )
    backtest_value: float | None = Field(
        description="Same metric measured from the backtest report. Null when unavailable."
    )
    delta: float | None = Field(
        description="live_value - backtest_value. Null when either side is null."
    )
    relative_delta: float | None = Field(
        description="delta divided by the absolute backtest value — the size of the gap "
        "relative to what was predicted. Null when the backtest value is zero or missing."
    )
    significant: bool = Field(
        description="True when the relative gap exceeds the report's tolerance and both "
        "sides have enough samples for that to mean something."
    )
    live_sample_count: int = Field(description="Live observations behind live_value.")
    backtest_sample_count: int = Field(description="Backtest observations behind backtest_value.")
    note: str = Field(description="What a gap in this particular metric implies.")


class DivergenceReportOut(BaseModel):
    """Live-vs-backtest comparison for one strategy on one symbol.

    Answers the question a profitable backtest and a losing account raise:
    is the simulator lying about fills, or has the edge decayed?"""

    strategy: str = Field(description="Strategy name compared, as reported by the backtest.")
    symbol: str
    report_id: str = Field(description="Backtest report the live trades were compared against.")
    live_trade_count: int = Field(description="Closed live trades matched for this comparison.")
    backtest_trade_count: int = Field(description="Trades in the backtest report.")
    comparable: bool = Field(
        description="False when either side has too few trades for a comparison to mean "
        "anything. The metrics are still returned so the UI can show what exists, but no "
        "conclusion should be drawn from them."
    )
    verdict: str = Field(
        description="'aligned' — live matches the backtest; 'simulator_optimistic' — fills "
        "are worse live than simulated, so the backtest's numbers are not achievable; "
        "'edge_decayed' — fills agree but results do not; 'both' — both diverge, so the "
        "outcome gap cannot yet be attributed; 'insufficient_data'."
    )
    summary: str = Field(description="One-paragraph plain-language reading of the verdict.")
    metrics: list[DivergenceMetricOut] = Field(
        description="Every metric compared, execution metrics first."
    )
