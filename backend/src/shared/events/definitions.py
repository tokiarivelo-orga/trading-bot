"""Core event types exchanged between modules.

Phase 0 defines the shapes the whole system is built around; later phases
fill in richer payloads as the domain models land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, kw_only=True)
class Event:
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, kw_only=True)
class CandleClosed(Event):
    symbol: str
    timeframe: str  # "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1" | "W1" | "MN"


@dataclass(frozen=True, kw_only=True)
class PositionOpened(Event):
    symbol: str
    position_id: str
    side: str  # "buy" | "sell"
    volume: float
    price: float
    sl: float | None
    tp: float | None
    spread_points: int
    comment: str = ""
    strategy_version: str | None = None
    skill: str | None = None
    # Full signal reason and confidence, for the journal's "why did the bot
    # take this trade" record — unlike `comment`, not truncated to MT5's
    # 29-char comment limit (see engine/application/trade_loop.py).
    reason: str = ""
    confidence: float | None = None  # Signal.confidence, 0..1; None for manual/API orders
    # Optional chart-annotation data from the strategy's Signal (see
    # strategies/domain/models.py: PriceZone/StructurePoint). Flattened to
    # primitives rather than importing those domain types, same as `side`
    # above being a plain str instead of the broker's Side enum — shared
    # events stay framework/module-independent so any module can subscribe
    # without importing another module's internals.
    zone_kind: str | None = None  # "demand" | "supply"
    zone_price_low: float | None = None
    zone_price_high: float | None = None
    zone_time_start: datetime | None = None
    zone_time_end: datetime | None = None
    zone_pattern: str | None = None  # the zone's own subtype, e.g. "RBR"/"DBD"/"QML"
    pattern: str | None = None
    structure: tuple[tuple[str, float, datetime], ...] = ()
    """Swing points as (label, price, time), label one of HH/HL/LH/LL."""
    indicators: tuple[tuple[str, float, float, str, bool], ...] = ()
    """Confluence-check readings as (name, value, threshold, comparison, passed)."""
    # Execution telemetry (OBSERVABILITY_PLAN.md Phase 3) — measured by
    # `broker/application/order_service.py` around the broker call and carried
    # to the journal, which is the only subscriber that stores them.
    requested_price: float | None = None
    """Tradable price the order service saw when it sent the order (ask to
    buy, bid to sell). None for fills whose caller had no reference price."""
    slippage: float | None = None
    """`price` minus `requested_price`, signed so a POSITIVE number always
    means the fill cost the trader (bought higher / sold lower). None
    whenever `requested_price` is None."""
    execution_latency_ms: float | None = None
    """Milliseconds from the strategy signal being emitted (the `created_at`
    of its `SignalDecision`) to the broker acknowledging the fill. None for
    manual/API orders, which have no signal behind them."""
    broker_retcode: int | None = None
    """Broker return code for the fill (MT5 10009 = done). None for brokers
    with no such concept (paper)."""
    # ── Regime tagging (Phase 6) ─────────────────────────────────────────
    # Snapshotted by `engine/application/trade_loop.py` at signal time
    # (`engine.domain.regime.compute_entry_regime`) and carried to the
    # journal, which is the only subscriber that stores them.
    transaction_cost: float | None = None
    """Spread + slippage cost of this fill, in account currency — see
    `broker.application.order_service.OrderService.open_position`. None for
    fills whose caller had no reference price to compute it from."""
    regime_volatility: str | None = None
    """`VolatilityRegime` value at signal time — 'low'/'normal'/'high'/
    'extreme'. None when the entry timeframe had no candles to classify."""
    regime_volatility_percentile: float | None = None
    """The ATR percentile rank behind `regime_volatility` (0-100)."""
    regime_trend: str | None = None
    """`TrendRegime` value at signal time — 'trending'/'ranging'."""
    regime_adx: float | None = None
    """Raw ADX reading behind `regime_trend`."""
    regime_session: str | None = None
    """`TradingSession` value at signal time — 'asian'/'london'/'overlap'/
    'new_york'/'off_session'."""
    signal_id: str | None = None
    """The `SignalDecision.signal_id` (order_book/ Phase 5) that led to this
    fill — the join key back to `signal_decisions` and, if the symbol
    reported depth, `order_book_snapshots`. None for manual/API orders,
    which have no signal behind them."""


@dataclass(frozen=True, kw_only=True)
class PositionClosed(Event):
    symbol: str
    position_id: str
    close_price: float
    profit: float
    close_reason: str = ""


@dataclass(frozen=True, kw_only=True)
class TenTradesCompleted(Event):
    """Emitted by the journal every 10 closed trades for one bot → triggers
    AI review. Scoped by `skill` (a bot's unique `NormalSkill.name`), not
    just `symbol`, since several bots may trade the same symbol
    concurrently, each on its own 10-trade cadence."""

    symbol: str
    skill: str
    trade_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class NewsWindowEntered(Event):
    event_name: str
    symbols: tuple[str, ...]
    close_all: bool = False
    """Whether the matched news skill's `pre_event.close_all` requests
    flattening open positions in `symbols` before the event (§6.6)."""


@dataclass(frozen=True, kw_only=True)
class NewsWindowExited(Event):
    event_name: str
    symbols: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class CircuitBreakerTripped(Event):
    """Emitted when the engine pauses — either a risk-manager circuit breaker
    (consecutive losses, daily loss limit) or the manual kill switch (§11).
    Alerting subscribes to this; nothing else does."""

    reason: str


@dataclass(frozen=True, kw_only=True)
class RefinementCompleted(Event):
    """Emitted after the 10-trade AI review loop finishes (§8.2), regardless
    of whether it proposed a refinement. Alerting subscribes to this."""

    symbol: str
    verdict: str
    proposal_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class GatewayHealthChanged(Event):
    """Emitted by `GatewayHealthMonitor` on a gateway-up/terminal-connected
    state transition (Phase 9 reconnect/resume). Alerting subscribes to this."""

    gateway_up: bool
    terminal_connected: bool


@dataclass(frozen=True, kw_only=True)
class BotWentSilent(Event):
    """Emitted by `SilenceMonitor` (OBSERVABILITY_PLAN.md Phase 5) when a
    bot's gap since its last signal exceeds `multiplier` times its own
    median inter-signal interval — a dead bot otherwise looks identical to a
    quiet market. Alerting subscribes to this."""

    bot: str
    """Full skill id, e.g. 'normal/xauusd/breakout_v1'."""
    elapsed_s: float
    median_interval_s: float
    threshold_s: float
    last_signal_at: datetime | None
