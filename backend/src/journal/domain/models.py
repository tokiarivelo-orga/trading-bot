"""Journal domain: the trade record and its market-context snapshots (§6.8).

Pure values — no I/O. `id` is the broker position ticket (as a string), the
natural unique key shared with the broker/engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, kw_only=True)
class CandleSnapshot:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int


@dataclass(frozen=True, kw_only=True)
class MarketSnapshot:
    m5: tuple[CandleSnapshot, ...] = ()
    h1: tuple[CandleSnapshot, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TradeRecord:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    volume: float
    open_price: float
    open_time: datetime
    sl: float | None
    tp: float | None
    spread_points_at_entry: int
    comment: str = ""
    # Filled in by later phases (strategies/skills don't exist yet in Phase 3).
    strategy_version: str | None = None
    skill: str | None = None
    close_price: float | None = None
    close_time: datetime | None = None
    profit: float | None = None
    close_reason: str | None = None
    m5_entry_snapshot: tuple[CandleSnapshot, ...] = ()
    h1_entry_snapshot: tuple[CandleSnapshot, ...] = ()
    m5_exit_snapshot: tuple[CandleSnapshot, ...] = ()
    h1_exit_snapshot: tuple[CandleSnapshot, ...] = ()
    # "Why" the bot took this trade — the strategy's Signal.reason/confidence
    # and chart-annotation data (see strategies/domain/models.py), passed
    # through PositionOpened. Empty/None for manually- or API-placed trades.
    reason: str = ""
    confidence: float | None = None
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
    # ── Execution telemetry (OBSERVABILITY_PLAN.md Phase 3) ────────────────
    # Measured by `broker/application/order_service.py` around the broker call
    # and carried here on `PositionOpened`. All None for trades journaled
    # before Phase 3, and for fills whose caller had no reference price.
    requested_price: float | None = None
    """Tradable price the order asked for (ask to buy, bid to sell)."""
    slippage: float | None = None
    """Fill minus requested, signed so POSITIVE always means it cost the
    trader (bought higher / sold lower). See `broker.domain.trading.
    execution_slippage`."""
    execution_latency_ms: float | None = None
    """Milliseconds from the strategy emitting the signal to the broker
    acknowledging the fill. None for manual/API trades (no signal)."""
    broker_retcode: int | None = None
    """Broker return code on the fill — MT5 10009 is a clean deal. None for
    the paper broker, which has no such concept."""
    # ── Excursion (MFE/MAE), also Phase 3 ──────────────────────────────────
    # Accumulated while the position is open by `TradeJournalService.
    # on_candle_closed` and finalized against the close price on
    # `on_position_closed` — see `journal/domain/excursion.py` for the
    # (pure) arithmetic and the rationale for where it lives.
    mfe: float | None = None
    """Maximum favorable excursion in price units — the furthest the market
    ever moved in the trade's favor from entry. Non-negative."""
    mfe_time: datetime | None = None
    """Time at which the maximum favorable excursion occurred."""
    mae: float | None = None
    """Maximum adverse excursion in price units — the furthest the market
    ever moved against the trade from entry. Non-negative."""
    mae_time: datetime | None = None
    """Time at which the maximum adverse excursion occurred."""
    # ── Regime tagging (OBSERVABILITY_PLAN.md Phase 6) ─────────────────────
    # Snapshotted by `engine/application/trade_loop.py` at the moment the
    # signal fired (`engine.domain.regime.compute_entry_regime`) and carried
    # here on `PositionOpened`. All None for trades journaled before Phase 6,
    # and for fills whose entry timeframe had no candles to classify.
    regime_volatility: str | None = None
    """`VolatilityRegime` value at entry — 'low'/'normal'/'high'/'extreme'."""
    regime_volatility_percentile: float | None = None
    """The ATR percentile rank behind `regime_volatility` (0-100)."""
    regime_trend: str | None = None
    """`TrendRegime` value at entry — 'trending'/'ranging'."""
    regime_adx: float | None = None
    """Raw ADX reading behind `regime_trend`."""
    regime_session: str | None = None
    """`TradingSession` value at entry — 'asian'/'london'/'overlap'/
    'new_york'/'off_session'."""
    transaction_cost: float | None = None
    """Spread + slippage cost of this fill, in account currency: `(spread_points
    * point + slippage) * volume * contract_size`. Computed by
    `broker/application/order_service.py` around the broker call — the same
    place execution telemetry (Phase 3) is measured. None for trades
    journaled before Phase 6."""
    # ── Signal correlation (order_book/ Phase 5) ────────────────────────────
    signal_id: str | None = None
    """The `SignalDecision.signal_id` that led to this trade, carried
    through `PositionOpened` — the join key back to `signal_decisions` and,
    if the symbol reported depth, `order_book_snapshots`. None for trades
    journaled before this field existed, or opened manually/via the API."""

    @property
    def is_open(self) -> bool:
        return self.close_time is None


@dataclass(frozen=True, kw_only=True)
class OpenTradeExcursion:
    """The only fields the per-candle MFE/MAE accumulation needs from an open
    trade. Backs `JournalRepository.get_open_excursions`, which selects just
    these columns — the accumulator runs on every closed candle for every open
    position, so loading whole `TradeRecord`s (with their four JSON snapshot
    columns) on that path would be pure waste."""

    id: str
    side: str  # "buy" | "sell"
    open_price: float
    mfe: float | None
    mfe_time: datetime | None = None
    mae: float | None
    mae_time: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class TradeAnalyticsRecord:
    """Slim projection of `TradeRecord` carrying only the fields
    `domain/analytics.py`'s aggregation actually reads (id, symbol, volume,
    open/close time, profit, skill, strategy_version, the Phase 3
    execution-telemetry/excursion columns, and the Phase 6 regime-bucket +
    transaction-cost columns). Backs
    `JournalRepository.get_all_for_analytics`, which selects just these
    columns so SQLAlchemy never deserializes the four JSON snapshot/structure
    columns (`m5/h1_entry/exit_snapshot`, `structure`) that analytics never
    touches. `compute_symbol_analytics`/`compute_bot_analytics` accept either
    this or a full `TradeRecord` — they only use attributes both share."""

    id: str
    symbol: str
    volume: float
    open_time: datetime
    close_time: datetime | None
    profit: float | None
    skill: str | None
    strategy_version: str | None
    slippage: float | None = None
    execution_latency_ms: float | None = None
    broker_retcode: int | None = None
    mfe: float | None = None
    mfe_time: datetime | None = None
    mae: float | None = None
    mae_time: datetime | None = None
    # Regime tagging (OBSERVABILITY_PLAN.md Phase 6) — bucket strings + cost
    # only; the two raw floats (`regime_volatility_percentile`, `regime_adx`)
    # aren't read by any analytics aggregation, so they're left off this slim
    # projection the same way the other unused `TradeRecord` fields are.
    regime_volatility: str | None = None
    regime_trend: str | None = None
    regime_session: str | None = None
    transaction_cost: float | None = None

    @property
    def is_open(self) -> bool:
        return self.close_time is None
