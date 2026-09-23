"""Backtest domain: a completed trade, an equity point, and the final report.

Pure values — no I/O. Deliberately not the `journal` module's `TradeRecord`:
a backtest doesn't touch the journal DB and modules don't reach into each
other's internals (see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, kw_only=True)
class BacktestZone:
    """A supply/demand rectangle the strategy identified before entering —
    chart-annotation data. Kept local to this module (not importing
    strategies.domain.models.PriceZone) so backtest doesn't reach into
    another module's internals; see CLAUDE.md."""

    kind: str  # "demand" | "supply"
    price_low: float
    price_high: float
    time_start: datetime
    time_end: datetime
    # The zone's own subtype, e.g. "RBR"/"DBD"/"RBD"/"DBR"/"QML" — distinct
    # from BacktestTrade.pattern (the confirming candlestick pattern).
    pattern: str | None = None


@dataclass(frozen=True, kw_only=True)
class BacktestTrade:
    side: str  # "buy" | "sell"
    volume: float
    open_time: datetime
    open_price: float
    sl: float | None
    tp: float | None
    close_time: datetime
    close_price: float
    profit: float
    r_multiple: float | None  # profit / initial risk in account currency; None if sl was unset
    zone: BacktestZone | None = None
    pattern: str | None = None  # confirming candlestick pattern, e.g. "bullish_engulfing"
    structure: tuple[tuple[str, float, datetime], ...] = ()
    """Swing points as (label, price, time), label one of HH/HL/LH/LL."""
    # The strategy's own Signal.reason/confidence that led to this trade — same
    # fields the live journal's TradeRecord carries (see PositionOpened),
    # threaded through BacktestBookkeeper so the report/UI can always show why
    # a trade was taken, not just its zone/pattern chart annotations. Defaults
    # keep report JSON files predating this field loadable.
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True, kw_only=True)
class BacktestSignal:
    """One strategy signal emitted during the replay — including the ones
    that never became trades. Extracted from the engine's decision-trail log
    lines (see `application/signals.py`), so the report can show every valid
    setup the strategy saw and what the engine did with it."""

    time: datetime  # simulated bot clock (the M5 bar's close time)
    direction: str  # "buy" | "sell"
    outcome: str
    """One of `activity.domain.models.SIGNAL_OUTCOMES` — the same closed
    vocabulary the live decision trail uses since Phase 2 ("opened",
    "htf_veto", "max_positions", "risk_sizing", "spread_veto", "rr_gate",
    "broker_rejected", "daily_loss_breaker", "skipped"; "volatility_guard"
    only in reports written before the volatility guard was removed).
    Backtests produced the older collapsed vocabulary until
    Phase 4 wired the engine's own decision sink into the replay; the legacy
    `risk_rejected` value still appears in report JSON files written before
    that, which is why it remains renderable."""
    reason: str  # the strategy's own Signal.reason (pattern, zone, entry/sl/tp lines)
    price: float | None = None
    """Reference price the engine saw when the signal fired — ask for a buy,
    bid for a sell. `None` in reports produced by the legacy log-scrape path,
    which never recovered it."""


@dataclass(frozen=True, kw_only=True)
class EquityPoint:
    time: datetime
    balance: float


@dataclass(frozen=True, kw_only=True)
class ActivityLogEntry:
    """One decision-trail line captured from the engine/broker/risk-manager's
    own `logging` calls while replaying — signals, HTF vetoes, sizing
    rejections, fills, circuit breakers — so a report with zero trades still
    explains why, without needing the server's stdout at the time it ran."""

    time: datetime  # simulated bot clock (the M5 bar's close time), not wall clock
    level: str  # "INFO" | "WARNING" | "ERROR"
    logger: str  # originating logger name, e.g. "src.engine.application.trade_loop"
    message: str


@dataclass(frozen=True, kw_only=True)
class BacktestRejection:
    """How many entries the simulated broker refused for one reason
    (OBSERVABILITY_PLAN.md Phase 4).

    A rejection is a signal the strategy produced, the risk manager sized, and
    the spread gate cleared — that a real broker would then have thrown away.
    Counting them by reason is the whole point: a fleet dying on `stops_level`
    used to look like a profitable backtest and an empty live account."""

    reason: str  # e.g. "stops_level", "volume_below_min", "volume_above_max"
    count: int
    retcode: int  # the MT5 code a live server would have returned, e.g. 10016
    example: str  # first refusal's message, with the concrete numbers


@dataclass(frozen=True, kw_only=True)
class BrokerRealism:
    """What the simulated broker enforced during a run, recorded so a report's
    numbers can be read against the assumptions that produced them.

    Reports written before Phase 4 have `enabled=False` and no rejections —
    their trade lists include entries a live broker would have refused."""

    enabled: bool = False
    stops_level_enforced: bool = False
    volume_grid_enforced: bool = False
    clamp_stops: bool = False
    spread_widening_factor: float = 1.0
    slippage_mean: float = 0.0
    slippage_stddev: float = 0.0
    slippage_source: str = "none"  # "live" | "fallback" | "none"
    slippage_sample_count: int = 0
    accepted_count: int = 0
    clamped_count: int = 0
    rejected_count: int = 0
    rejections: tuple[BacktestRejection, ...] = ()

    @property
    def acceptance_rate(self) -> float:
        """Filled / (filled + refused). 1.0 when nothing was refused; 0.0 for a
        strategy the broker would refuse outright, which is the number this
        phase exists to make visible."""
        total = self.accepted_count + self.rejected_count
        return self.accepted_count / total if total else 0.0


@dataclass(frozen=True, kw_only=True)
class BacktestReport:
    strategy: str
    symbol: str
    period: str
    starting_balance: float
    ending_balance: float
    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    avg_r: float
    worst_losing_streak: int
    activity_log: tuple[ActivityLogEntry, ...] = ()
    # Every signal the strategy emitted (taken or vetoed), oldest first —
    # empty for report files predating this field.
    signals: tuple[BacktestSignal, ...] = ()
    # The spread-adjusted minimum reward:risk ratio SpreadGate actually
    # enforced for this run — configs/symbols/<symbol>.yaml's value, the
    # run's own min_rr override, or SpreadGate.DEFAULT_MIN_RR if the symbol
    # has no config file at all. Recorded per-report since it's a run
    # parameter (like starting_balance), not a fixed strategy property.
    min_rr: float = 1.0
    # The full RiskCaps actually enforced for this run — configs/risk.yaml's
    # values, any of this run's own min_lot_fallback_enabled/
    # max_risk_per_trade_pct overrides, or the live engine override if
    # neither was passed. Recorded per-report for the same reason min_rr is:
    # it's a run parameter that can change what the report's trade count
    # means (e.g. explains a circuit-breaker pause cutting a run short — see
    # RiskManager.record_trade_closed, which never auto-resumes).
    risk_per_trade_pct: float = 0.5
    daily_loss_limit_pct: float = 2.0
    max_open_positions: int = 100
    max_trades_per_day_enabled: bool = False
    consecutive_loss_pause: int = 10
    min_lot_fallback_enabled: bool = False
    max_risk_per_trade_pct: float | None = None
    # The broker constraints simulated for this run and what they cost
    # (Phase 4). Defaulted so report JSON files predating it still load, with
    # `enabled=False` correctly describing them: they simulated nothing.
    broker_realism: BrokerRealism = BrokerRealism()
