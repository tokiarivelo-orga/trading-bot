"""Strategy contract — the interface every generated strategy implements.

Generated code receives a MarketContext and returns a Signal or None.
It has no access to the broker, filesystem, or network: the engine is the
only component that executes trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Direction(StrEnum):
    BUY = "buy"
    SELL = "sell"


class ZoneKind(StrEnum):
    DEMAND = "demand"  # buy zone (support) — price expected to rise from it
    SUPPLY = "supply"  # sell zone (resistance) — price expected to fall from it


@dataclass(frozen=True)
class PriceZone:
    """A supply/demand rectangle a strategy identified before entering —
    chart-annotation data, not used by the engine for execution."""

    kind: ZoneKind
    price_low: float
    price_high: float
    time_start: datetime
    time_end: datetime
    # The zone's own subtype, e.g. "RBR"/"DBD"/"RBD"/"DBR"/"DZ"/"SZ" — distinct
    # from Signal.pattern (the confirming candlestick pattern). None for
    # strategies that don't label their zone detector's setup type.
    pattern: str | None = None


class StructureLabel(StrEnum):
    HH = "HH"  # higher high
    HL = "HL"  # higher low
    LH = "LH"  # lower high
    LL = "LL"  # lower low


@dataclass(frozen=True)
class StructurePoint:
    """A single labeled swing point (HH/HL/LH/LL) — chart-annotation data."""

    time: datetime
    price: float
    label: StructureLabel


@dataclass(frozen=True)
class IndicatorReading:
    """A single confluence-check indicator value — chart/journal-annotation
    data showing why a vote passed or failed, not used by the engine."""

    name: str
    value: float
    threshold: float
    comparison: str  # ">" or "<" — the direction that makes this reading "pass"
    passed: bool


@dataclass(frozen=True)
class Signal:
    direction: Direction
    sl_points: float
    tp_points: float
    confidence: float = 1.0  # 0..1
    reason: str = ""
    # Optional chart-annotation data a strategy may supply. Not read by the
    # engine — purely for backtest reports / chart drawing, so existing
    # strategies that don't set these keep working unchanged.
    zone: PriceZone | None = None
    pattern: str | None = None
    structure: tuple[StructurePoint, ...] = field(default_factory=tuple)
    indicators: tuple[IndicatorReading, ...] = field(default_factory=tuple)
    # Bounded per-signal RISK-AMOUNT multiplier for high-conviction setups —
    # not a lot-count multiplier and not a way to open more positions; the
    # engine folds this into the existing per-position risk_multiplier before
    # configs/risk.yaml's caps are applied, so it stays bounded. Default 1.0
    # preserves today's behaviour for every strategy that doesn't set it.
    size_multiplier: float = 1.0


class ExitActionKind(StrEnum):
    # Closes this bot's own open position outright — for thesis
    # invalidation (e.g. the zone/FVG the entry was built on just got
    # violated). Distinct from `PositionManager`'s generic give-back/
    # volatility exits, which react to realised profit state, not setup
    # knowledge only the strategy has.
    CLOSE = "close"
    # Moves this bot's own open position's SL to its entry price, subject
    # to the engine's usual never-loosen guard — for setup-specific
    # continuation confirmation, not a substitute for the generic +1R
    # breakeven rule `PositionManager` already runs for every position.
    BREAKEVEN = "breakeven"
    # Moves this bot's own open position's SL to `ExitDecision.target_price`
    # — same never-loosen guard as BREAKEVEN, but to an arbitrary price
    # instead of only entry. For setups that want to lock in more than
    # breakeven once a milestone clears (e.g. a sibling leg's TP), without
    # closing the position outright. `target_price` is required for this
    # action; the engine ignores the action (logs nothing, moves nothing)
    # if it's `None`.
    SET_SL = "set_sl"


@dataclass(frozen=True)
class ExitDecision:
    """A strategy's request to act on its OWN already-open position —
    returned from `evaluate()` alongside/instead of a `Signal`. Matched to
    a position by the engine via magic number, the same way
    `close_on_opposite_signal` is; see `Strategy.evaluate`."""

    action: ExitActionKind
    reason: str = ""
    # Only read for `ExitActionKind.SET_SL` — the absolute price to move SL
    # to, still subject to the engine's `_improves()` never-loosen guard.
    target_price: float | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    """What a strategy is allowed to see about its own open position on
    this symbol, populated by the engine into `MarketContext.own_position`
    before `evaluate()` runs. `None` when the bot has no open position on
    this symbol. Deliberately narrow — no ticket/broker fields — since a
    strategy has no business addressing the broker directly; it only ever
    expresses intent via `ExitDecision`."""

    direction: Direction
    entry_price: float
    sl: float | None
    tp: float | None
    opened_at: datetime


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: int
    # The set of broker symbol names this strategy is designed for.
    # An empty tuple means the strategy accepts *any* symbol — used when
    # symbols are configured dynamically at runtime rather than hard-coded
    # at codegen time. A non-empty tuple acts as an explicit allowlist:
    # the engine and backtest runner will skip/reject any symbol not in it.
    symbols: tuple[str, ...]
    entry_timeframe: str  # the bar size this strategy evaluates on, e.g. "M1", "M5"
    confirmation_timeframes: tuple[str, ...]
    params: dict[str, Any]
    # Whether the engine's HTF veto (the timeframe above entry_timeframe,
    # see trade_loop._veto_timeframe) applies to this bot. True for every
    # existing strategy; a strategy can opt out when it does its own
    # trend/quality gating and the engine veto would be redundant or unwanted.
    htf_veto: bool = True
    # When True, a fresh signal whose direction opposes this bot's currently
    # open position (matched by OrderRequest.magic, so only this bot's own
    # position on the symbol) closes that position before the normal entry
    # pipeline runs, instead of waiting for SL/TP/time-stop — see
    # trade_loop._close_opposite_position. False (default) preserves today's
    # behavior for every existing strategy.
    close_on_opposite_signal: bool = False


@dataclass(frozen=True)
class MarketContext:
    """Everything a strategy may look at. Populated by the engine (Phase 4).

    candles maps timeframe → OHLCV frame (pandas DataFrame at runtime; typed
    loosely here so domain code stays import-light).
    """

    symbol: str
    candles: dict[str, Any]
    spread_points: float
    # This bot's own open position on `symbol`, if any — see
    # `PositionSnapshot`. `None` for every existing strategy's context until
    # the engine is told to populate it (opt-in via reading it at all;
    # strategies that never look at this field behave exactly as before).
    own_position: PositionSnapshot | None = None


@runtime_checkable
class Strategy(Protocol):
    spec: StrategySpec

    def evaluate(
        self, ctx: MarketContext
    ) -> (
        Signal
        | ExitDecision
        | tuple[Signal | ExitDecision, ...]
        | list[Signal | ExitDecision]
        | None
    ): ...
