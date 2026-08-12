"""Broker order-acceptance rules, as pure arithmetic (OBSERVABILITY_PLAN.md
Phase 4).

These are the checks a real MT5 server applies *before* it fills anything.
Until Phase 4 they existed nowhere in this codebase: the live path let the
broker enforce them (and reported the refusal only as free text), and the
backtest never modelled them at all — so an M1 scalp whose stop-loss was
closer than the symbol's `stops_level` produced a tidy winning backtest and
retcode 10016 on every single live order.

Two rules matter for entries:

* **`stops_level`** — the minimum distance, in points, that SL and TP must
  keep from the current price. Some synthetic indices (e.g. Deriv VIX75) report
  `stops_level` such that a stop must sit at least 107.70 price units away; an
  M1 scalp risking 30-70 units is rejected outright, every time.
* **volume granularity** — `volume_min` / `volume_step` / `volume_max`. MT5
  rounds nothing for you; a 0.037-lot request on a 0.01-step symbol is
  refused, and a lot that rounds below `volume_min` cannot be sent at all.

Everything here is a pure function over primitives so it can be used by the
paper broker, the backtest simulator, and tests alike without any of them
importing each other. Money-touching, so it is deliberately explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.broker.domain.trading import Side

# MT5 `TRADE_RETCODE_*` values the simulated broker reports, so a simulated
# refusal is indistinguishable from the live one on the decision trail (the
# retcode is what Phase 3 records).
RETCODE_INVALID_VOLUME = 10014
RETCODE_INVALID_STOPS = 10016

# Rejection reason ids reported by the backtest. Stable strings — they are the
# keys of the per-reason counts in `BacktestReport.rejections` and are rendered
# by the report UI.
REASON_STOPS_LEVEL = "stops_level"
REASON_VOLUME_BELOW_MIN = "volume_below_min"
REASON_VOLUME_ABOVE_MAX = "volume_above_max"

REJECTION_REASONS: tuple[str, ...] = (
    REASON_STOPS_LEVEL,
    REASON_VOLUME_BELOW_MIN,
    REASON_VOLUME_ABOVE_MAX,
)

# Floating-point slack when comparing a distance against the stops_level
# threshold, in multiples of `point`. A stop placed at exactly the minimum
# distance is legal; binary rounding must not turn it into a rejection.
_DISTANCE_EPSILON_POINTS = 1e-6


@dataclass(frozen=True, kw_only=True)
class StopsLevelViolation:
    """Which leg was too close, and by how much."""

    leg: str  # "sl" | "tp"
    distance: float  # actual distance from the reference price, in price units
    required: float  # minimum legal distance, in price units


@dataclass(frozen=True, kw_only=True)
class VolumeViolation:
    """A lot size the broker cannot accept even after step rounding."""

    reason: str  # REASON_VOLUME_BELOW_MIN | REASON_VOLUME_ABOVE_MAX
    requested: float
    rounded: float
    limit: float


@dataclass(frozen=True, kw_only=True)
class SimulatedEntry:
    """What a simulated broker decided an accepted entry actually becomes:
    the broker-legal lot size and the price it filled at, plus the modelled
    slippage that moved it there (positive always means it cost the trader,
    same convention as `broker.domain.trading.execution_slippage`)."""

    volume: float
    fill_price: float
    slippage: float
    sl: float | None
    tp: float | None
    """The stops actually registered with the broker. Identical to the ones
    requested unless the simulator is configured to *clamp* a too-close leg out
    to `stops_level` instead of rejecting the order — in which case the
    position really does get the wider stop, and its backtested P&L must be
    computed against that wider stop rather than the one the strategy asked
    for."""


def min_stop_distance(stops_level: int, point: float) -> float:
    """The minimum SL/TP distance from price, in price units.

    MT5 reports `stops_level` in *points*, so it only becomes a price
    distance once multiplied by the symbol's `point`. For a synthetic index
    with `point` 0.01 and `stops_level` 10770, this is 107.70 price units —
    the figure that silently killed the M1 scalp fleet."""
    return max(0, stops_level) * point


def check_stops_level(
    *,
    side: Side,
    price: float,
    sl: float | None,
    tp: float | None,
    stops_level: int,
    point: float,
) -> StopsLevelViolation | None:
    """`None` when both legs clear the broker's minimum distance from `price`.

    Only the *distance* is checked here, not which side of price each leg sits
    on — a wrong-sided stop is a strategy bug the engine's own RR gate already
    catches, and MT5 reports it under the same retcode anyway. Absent legs
    (`None`) trivially pass: an order with no stop has nothing to be too close.
    `stops_level == 0` means the broker imposes no minimum, so nothing is
    rejected."""
    required = min_stop_distance(stops_level, point)
    if required <= 0.0:
        return None
    tolerance = required - _DISTANCE_EPSILON_POINTS * point
    # `side` does not change the arithmetic (distance is absolute), but it is
    # part of the signature so callers cannot forget which price they are
    # measuring from — the reference price must be the tradable one for the
    # side (ask for a buy, bid for a sell).
    _ = side
    if sl is not None:
        distance = abs(price - sl)
        if distance < tolerance:
            return StopsLevelViolation(leg="sl", distance=distance, required=required)
    if tp is not None:
        distance = abs(tp - price)
        if distance < tolerance:
            return StopsLevelViolation(leg="tp", distance=distance, required=required)
    return None


def clamp_stops(
    *,
    side: Side,
    price: float,
    sl: float | None,
    tp: float | None,
    stops_level: int,
    point: float,
) -> tuple[float | None, float | None]:
    """Push any leg closer than `stops_level` out to exactly that distance,
    keeping it on its correct side of `price`.

    The alternative to rejecting. It answers a different question — "what would
    this strategy have made if its stops were legal" — and must never be the
    default, because a widened SL is a *different, riskier* trade than the one
    the strategy sized: the position now risks more account currency than the
    risk manager approved. Useful for research, dishonest as a headline number.

    A buy's SL sits below price and TP above; a sell's the other way round.
    Legs already far enough away are returned unchanged."""
    required = min_stop_distance(stops_level, point)
    if required <= 0.0:
        return sl, tp
    sl_sign = -1.0 if side is Side.BUY else 1.0
    new_sl = sl
    if sl is not None and abs(price - sl) < required - _DISTANCE_EPSILON_POINTS * point:
        new_sl = price + sl_sign * required
    new_tp = tp
    if tp is not None and abs(tp - price) < required - _DISTANCE_EPSILON_POINTS * point:
        new_tp = price - sl_sign * required
    return new_sl, new_tp


def round_volume(
    volume: float, *, volume_min: float, volume_max: float, volume_step: float
) -> tuple[float, VolumeViolation | None]:
    """Snap a requested lot size onto the broker's volume grid.

    Returns `(rounded_volume, violation)`. Rounding is **down** to the nearest
    step — never up, because rounding up would size a position larger than the
    risk manager approved, and risk caps are user-owned (CLAUDE.md). A volume
    that rounds below `volume_min` is a rejection, not a silent bump to the
    minimum: live, MT5 refuses it with retcode 10014, and pretending otherwise
    is exactly the kind of lie this phase exists to remove. (The engine's own
    `min_lot_fallback_enabled` cap decides *before* this whether an undersized
    trade should be taken at `volume_min`; by the time an order reaches the
    broker that decision is already made.)

    A volume above `volume_max` is likewise a rejection rather than a clamp."""
    if volume_step <= 0.0:
        rounded = volume
    else:
        steps = math.floor(volume / volume_step + 1e-9)
        rounded = steps * volume_step
        # Re-round to the step's own decimal precision so 3 * 0.01 is 0.03 and
        # not 0.030000000000000002 — the value is logged and compared in tests.
        rounded = round(rounded, _step_decimals(volume_step))
    if volume_max > 0.0 and rounded > volume_max:
        return rounded, VolumeViolation(
            reason=REASON_VOLUME_ABOVE_MAX,
            requested=volume,
            rounded=rounded,
            limit=volume_max,
        )
    if rounded < volume_min:
        return rounded, VolumeViolation(
            reason=REASON_VOLUME_BELOW_MIN,
            requested=volume,
            rounded=rounded,
            limit=volume_min,
        )
    return rounded, None


def _step_decimals(volume_step: float) -> int:
    """Decimal places implied by a lot step (0.01 -> 2, 1.0 -> 0). Capped at 8
    so a pathological step can't produce nonsense precision."""
    if volume_step <= 0.0:
        return 8
    decimals = max(0, -math.floor(math.log10(volume_step)))
    return min(8, int(decimals))


def widened_spread_points(spread_points: int, widening_factor: float) -> int:
    """Historical spread scaled by a configured factor, floored at the
    recorded value.

    Stored M5 candles carry the spread observed at the *close* of the bar,
    which is a best case: real entries happen mid-bar, during the news spikes
    and rollover windows the bar's closing quote has already recovered from.
    A factor > 1 makes the backtest pay a spread closer to what live fills
    actually paid; 1.0 (the default) reproduces the old behaviour exactly."""
    if widening_factor <= 1.0:
        return spread_points
    return int(math.ceil(spread_points * widening_factor))
