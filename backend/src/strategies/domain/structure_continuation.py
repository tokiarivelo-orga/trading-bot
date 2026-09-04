"""Trend-continuation and structure-weakness/reversal detectors, sharing one
fractal swing-pivot algorithm.

WHY THIS EXISTS
────────────────────────────────────────────────────────────────────────
The pure supply/demand zone-reversal strategies (`xauusd_snd_qm_structure_*`)
only ever fire when price is *currently touching* a live zone. After a sharp
directional move, price can go a long time without coming back to touch one
— the bots sit idle even though the market is trading cleanly. This module
adds independent setup types these strategies can layer in, none of them
needing a marked zone:

  * `detect_trend_continuation` — a trend already in motion, that just
    finished a shallow pullback and has resumed.
  * `detect_structure_reversal` — the opposite case: a trend whose most
    recent impulse leg is measurably weaker (smaller, lower-volume) than
    its prior leg, confirmed by a break of the swing point between them
    (a change of character / CHoCH). This is the module's own definition
    of "the trend is showing weakness" — momentum and participation both
    fading before the structural break, not just the break itself.

Neither loosens or removes any existing gate; both are new coverage,
evaluated only after a strategy's own zone-touch scan (if it has one)
finds nothing (see each strategy file's own integration for how priority
is preserved).

────────────────────────────────────────────────────────────────────────
WHY THIS LIVES IN `domain/`
────────────────────────────────────────────────────────────────────────
Sibling to `fatigue.py`/`online_learning.py`: pure, sandbox-safe (`math`,
`numpy`, `pandas`, `src.strategies.domain.models`, plus the one
already-allowlisted exception below), no I/O, no adapters, no broker
access, so every generated strategy can share one copy instead of
re-deriving it. See `strategies/sandbox.py`'s `ALLOWED_IMPORT_MODULES`.

Two building blocks:

1. **Trend direction + strength** — `src.engine.domain.regime.
   latest_trend_regime` (already sandbox-allowlisted: proven, tested, ADX-
   based) says trending-vs-ranging and how strongly. It does not say which
   way, so this module pairs it with a lightweight EMA(12)/EMA(34)
   separation-sign check purely to pick *direction* once TRENDING is
   confirmed.
2. **Fractal swing pivots** — the same algorithm as `engine/domain/
   structure_pivot.py::detect_swing_pivots` (a bar is a pivot if strictly
   extreme vs `pivot_bars` neighbours each side, labeled HL/HH/LH/LL the
   same way `strategies.domain.models.StructureLabel` already names them).
   That module is *not* sandbox-allowlisted (it lives under `engine/`), so
   the algorithm is reimplemented here rather than imported — deliberately
   kept bit-for-bit identical so a pivot found by this module means the
   same thing the engine's own structure-pivot ratchet rule means. See
   `tests/unit/strategies/test_structure_continuation.py` for a direct
   cross-check against `tests/unit/engine/test_structure_pivot.py`'s own
   zigzag fixture.

────────────────────────────────────────────────────────────────────────
WHAT "FIRES" MEANS
────────────────────────────────────────────────────────────────────────
`detect_trend_continuation()` returns a `ContinuationSetup` only when, in
sequence:

  1. `latest_trend_regime` reports TRENDING, and the EMA(12)/EMA(34)
     separation sign is unambiguous (a direction).
  2. The single most recent fractal pivot (of either kind) is a fresh HL
     for an uptrend / LH for a downtrend — the pullback that's expected to
     hold, not some older or opposite-kind swing point.
  3. Its pullback depth — measured from the prior swing extreme of the
     opposite kind (the last swing high before an HL, the last swing low
     before an LH) down/up to this pivot, in ATR units — sits inside
     `[min_pullback_atr, max_pullback_atr]` (0.5 to 2.5 by default). The
     floor excludes noise; the ceiling is what keeps this from just
     re-finding the same reversals the zone detectors already catch — a
     pullback deeper than 2.5 ATR looks like a reversal, not a
     continuation.
  4. Momentum has actually resumed, not just paused: the latest closed
     bar's close is already beyond the pivot bar's own high (uptrend) /
     low (downtrend) by at least `momentum_confirm_atr_mult` (0.1) ATR.

Every threshold is `params`-overridable, the same convention every other
helper in these files already follows (see `fatigue.py`'s `_param`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.engine.domain.regime import TrendRegime, latest_trend_regime
from src.strategies.domain.models import Direction, StructureLabel, StructurePoint

__all__ = [
    "DEFAULT_EMA_FAST_SPAN",
    "DEFAULT_EMA_SLOW_SPAN",
    "DEFAULT_MAX_PULLBACK_ATR",
    "DEFAULT_MIN_PULLBACK_ATR",
    "DEFAULT_MIN_WEAKNESS_RATIO",
    "DEFAULT_MOMENTUM_CONFIRM_ATR_MULT",
    "DEFAULT_PIVOT_BARS",
    "DEFAULT_TREND_ADX_PERIOD",
    "DEFAULT_TREND_ADX_THRESHOLD",
    "ContinuationSetup",
    "ReversalSetup",
    "SwingPivot",
    "detect_structure_reversal",
    "detect_swing_pivots",
    "detect_trend_continuation",
]

DEFAULT_PIVOT_BARS = 2  # bars required strictly beyond the pivot on each side to confirm it
DEFAULT_TREND_ADX_PERIOD = 14
DEFAULT_TREND_ADX_THRESHOLD = 25.0
DEFAULT_EMA_FAST_SPAN = 12
DEFAULT_EMA_SLOW_SPAN = 34
DEFAULT_MIN_PULLBACK_ATR = 0.5
DEFAULT_MAX_PULLBACK_ATR = 2.5
DEFAULT_MOMENTUM_CONFIRM_ATR_MULT = 0.1
# A weakening leg must be <= this fraction of its predecessor's size, in ATR
# units, AND its mean volume must be <= this same fraction of the
# predecessor leg's mean volume — both required, since either alone (a
# smaller candle, or a quiet one) is noise a single-condition gate would
# false-positive on constantly.
DEFAULT_MIN_WEAKNESS_RATIO = 0.7

# Which StructureLabel values belong to a low-type pivot vs a high-type one
# — used both to pick "the most recent pivot must be HL/LH" and to walk
# back for "the prior swing extreme of the opposite kind".
_LOW_LABELS = (StructureLabel.HL, StructureLabel.LL)
_HIGH_LABELS = (StructureLabel.HH, StructureLabel.LH)


@dataclass(frozen=True)
class SwingPivot:
    """One labeled fractal swing point, mirroring `engine.domain.
    structure_pivot.SwingPivot` field-for-field (see module docstring for
    why this is a reimplementation rather than an import)."""

    index: int
    time: datetime
    price: float
    label: StructureLabel


@dataclass(frozen=True, kw_only=True)
class ContinuationSetup:
    """One qualifying trend-continuation setup, ready for a strategy to
    build a `Signal` from. `structure_points` mirrors the chart-annotation
    convention every zone-touch `Signal.structure` already uses — the most
    recent pivots found, oldest first."""

    direction: Direction
    pivot_index: int
    pivot_price: float
    pullback_depth_atr: float
    trend_strength: float
    structure_points: tuple[StructurePoint, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ReversalSetup:
    """One qualifying structure-weakness/CHoCH reversal setup. `direction`
    is the *new* direction being entered (opposite the trend that just
    broke) — mirrors `ContinuationSetup`'s shape so a strategy can build a
    `Signal` from either the same way."""

    direction: Direction
    pivot_index: int
    pivot_price: float
    weakness_score: float  # 0..1, higher = a bigger momentum+volume drop-off
    trend_strength: float
    structure_points: tuple[StructurePoint, ...] = ()


def _as_array(values) -> np.ndarray:
    """Anything array-like (numpy array, pandas Series, list) -> float64,
    mirroring `fatigue.py`'s `_as_array` so callers can hand over whatever
    they already have on hand (a strategy's `df["high"].to_numpy()` or a
    plain list built in a test)."""
    if values is None:
        return np.empty(0, dtype=float)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float, copy=False)
    return np.asarray(values, dtype=float)


def _param(params: dict | None, key: str, default: float) -> float:
    if params is not None and key in params:
        return params[key]
    return default


def _latest_finite(values) -> float | None:
    """The last non-NaN reading in an array-like series, or `None` if it
    has none — an ATR series commonly carries leading NaNs during its
    warm-up window; this walks back from the tail rather than assuming the
    very last element is populated."""
    arr = _as_array(values)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return None
    return float(arr[np.flatnonzero(finite_mask)[-1]])


def detect_swing_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    times: Sequence[datetime],
    *,
    pivot_bars: int = DEFAULT_PIVOT_BARS,
) -> list[SwingPivot]:
    """Standard fractal swing-pivot detector — bit-for-bit the same
    algorithm as `engine.domain.structure_pivot.detect_swing_pivots`,
    reimplemented here because that module is not sandbox-allowlisted (see
    module docstring). Bar `i` is a swing low if its low is *strictly*
    below every one of the `pivot_bars` bars on each side of it (a tie
    disqualifies it); a swing high analogously on highs. Consecutive swing
    points of the same kind are labeled HL/LL (lows) or HH/LH (highs)
    against the previous swing of that kind; the first swing of each kind
    has no predecessor and is not labeled. Returns pivots in chronological
    order (ascending `index`/`time`)."""
    n = len(highs)
    if len(lows) != n or len(times) != n:
        raise ValueError("highs, lows, and times must be the same length")
    if pivot_bars < 1:
        raise ValueError("pivot_bars must be >= 1")

    pivots: list[SwingPivot] = []
    prev_low: float | None = None
    prev_high: float | None = None
    for i in range(pivot_bars, n - pivot_bars):
        left_low = lows[i - pivot_bars : i]
        right_low = lows[i + 1 : i + pivot_bars + 1]
        if lows[i] < left_low.min() and lows[i] < right_low.min():
            if prev_low is not None:
                label = StructureLabel.HL if lows[i] > prev_low else StructureLabel.LL
                pivots.append(SwingPivot(index=i, time=times[i], price=float(lows[i]), label=label))
            prev_low = float(lows[i])

        left_high = highs[i - pivot_bars : i]
        right_high = highs[i + 1 : i + pivot_bars + 1]
        if highs[i] > left_high.max() and highs[i] > right_high.max():
            if prev_high is not None:
                label = StructureLabel.HH if highs[i] > prev_high else StructureLabel.LH
                pivots.append(
                    SwingPivot(index=i, time=times[i], price=float(highs[i]), label=label)
                )
            prev_high = float(highs[i])

    pivots.sort(key=lambda p: p.index)
    return pivots


def _ema_direction(closes: np.ndarray, fast_span: int, slow_span: int) -> int:
    """+1 if EMA(fast) sits above EMA(slow) on the latest bar, -1 if below,
    0 if equal/indeterminate (not enough bars, or a NaN/zero separation) —
    a 0 means "no unambiguous direction", handled by the caller as "cannot
    fire", never guessed."""
    if closes.size < slow_span:
        return 0
    series = pd.Series(closes)
    ema_fast = series.ewm(span=fast_span, adjust=False).mean().to_numpy()
    ema_slow = series.ewm(span=slow_span, adjust=False).mean().to_numpy()
    separation = float(ema_fast[-1] - ema_slow[-1])
    if not math.isfinite(separation) or separation == 0.0:
        return 0
    return 1 if separation > 0.0 else -1


def detect_trend_continuation(
    *,
    highs,
    lows,
    closes,
    times: Sequence[datetime],
    atr_series,
    params: dict | None = None,
) -> ContinuationSetup | None:
    """Pure detector — see module docstring for the full "what fires"
    sequence. `highs`/`lows`/`closes` and `times`/`atr_series` must all be
    the same length and aligned to the same bars (a strategy's already-
    resampled zone timeframe, typically). Returns `None` at the first gate
    that isn't satisfied; never raises for "not enough history" or "no
    pivot found" — only for genuinely mismatched input shapes.

    Recognized `params` keys (all optional, defaults per the module-level
    `DEFAULT_*` constants): `pivot_bars`, `trend_adx_period`,
    `trend_adx_threshold`, `trend_ema_fast_span`, `trend_ema_slow_span`,
    `min_pullback_atr`, `max_pullback_atr`, `momentum_confirm_atr_mult`.
    """
    highs_arr = _as_array(highs)
    lows_arr = _as_array(lows)
    closes_arr = _as_array(closes)
    n = highs_arr.size
    if lows_arr.size != n or closes_arr.size != n or len(times) != n:
        raise ValueError("highs, lows, closes, and times must be the same length")
    if n == 0:
        return None

    pivot_bars = int(_param(params, "pivot_bars", DEFAULT_PIVOT_BARS))
    trend_adx_period = int(_param(params, "trend_adx_period", DEFAULT_TREND_ADX_PERIOD))
    trend_adx_threshold = float(_param(params, "trend_adx_threshold", DEFAULT_TREND_ADX_THRESHOLD))
    ema_fast_span = int(_param(params, "trend_ema_fast_span", DEFAULT_EMA_FAST_SPAN))
    ema_slow_span = int(_param(params, "trend_ema_slow_span", DEFAULT_EMA_SLOW_SPAN))
    min_pullback_atr = float(_param(params, "min_pullback_atr", DEFAULT_MIN_PULLBACK_ATR))
    max_pullback_atr = float(_param(params, "max_pullback_atr", DEFAULT_MAX_PULLBACK_ATR))
    momentum_mult = float(
        _param(params, "momentum_confirm_atr_mult", DEFAULT_MOMENTUM_CONFIRM_ATR_MULT)
    )

    # ── Gate 1: trending, with an unambiguous direction ──
    regime, adx_value = latest_trend_regime(
        highs_arr,
        lows_arr,
        closes_arr,
        adx_period=trend_adx_period,
        adx_trend_threshold=trend_adx_threshold,
    )
    if regime is not TrendRegime.TRENDING:
        return None

    ema_dir = _ema_direction(closes_arr, ema_fast_span, ema_slow_span)
    if ema_dir == 0:
        return None
    direction = Direction.BUY if ema_dir > 0 else Direction.SELL

    atr_now = _latest_finite(atr_series)
    if atr_now is None or atr_now <= 0.0:
        return None

    # ── Gate 2: most recent pivot is a fresh HL (up) / LH (down) ──
    pivots = detect_swing_pivots(highs_arr, lows_arr, list(times), pivot_bars=pivot_bars)
    if not pivots:
        return None

    latest_pivot = pivots[-1]
    wanted_label = StructureLabel.HL if direction is Direction.BUY else StructureLabel.LH
    if latest_pivot.label is not wanted_label:
        return None

    # ── Gate 3: pullback depth from the prior opposite-kind swing extreme ──
    opposite_labels = _HIGH_LABELS if direction is Direction.BUY else _LOW_LABELS
    prior_extreme: SwingPivot | None = None
    for pivot in reversed(pivots):
        if pivot.index >= latest_pivot.index:
            continue
        if pivot.label in opposite_labels:
            prior_extreme = pivot
            break
    if prior_extreme is None:
        return None

    pullback_depth_atr = abs(latest_pivot.price - prior_extreme.price) / atr_now
    if not (min_pullback_atr <= pullback_depth_atr <= max_pullback_atr):
        return None

    # ── Gate 4: momentum has resumed beyond the pivot bar's own extreme ──
    pivot_bar_high = float(highs_arr[latest_pivot.index])
    pivot_bar_low = float(lows_arr[latest_pivot.index])
    latest_close = float(closes_arr[-1])
    if direction is Direction.BUY:
        if latest_close < pivot_bar_high + momentum_mult * atr_now:
            return None
    else:
        if latest_close > pivot_bar_low - momentum_mult * atr_now:
            return None

    structure_points = tuple(
        StructurePoint(time=p.time, price=p.price, label=p.label) for p in pivots[-4:]
    )

    return ContinuationSetup(
        direction=direction,
        pivot_index=latest_pivot.index,
        pivot_price=latest_pivot.price,
        pullback_depth_atr=float(pullback_depth_atr),
        trend_strength=float(adx_value),
        structure_points=structure_points,
    )


def _try_reversal_direction(
    *,
    prevailing: Direction,
    pivots: list[SwingPivot],
    closes_arr: np.ndarray,
    volumes_arr: np.ndarray,
    atr_now: float,
    adx_value: float,
    min_weakness_ratio: float,
    momentum_mult: float,
) -> ReversalSetup | None:
    """One direction hypothesis for `detect_structure_reversal` — see that
    function for what "extreme"/"neckline" mean. Structural, not EMA-based:
    trying both directions here (rather than picking one via an EMA read on
    the latest bar, which the break itself is actively dragging down) is
    what keeps this reliable exactly when it matters — during the break."""
    extreme_labels = _HIGH_LABELS if prevailing is Direction.BUY else _LOW_LABELS
    neckline_labels = _LOW_LABELS if prevailing is Direction.BUY else _HIGH_LABELS

    extremes = [p for p in pivots if p.label in extreme_labels]
    necklines = [p for p in pivots if p.label in neckline_labels]
    if len(extremes) < 2:
        return None
    extreme2 = extremes[-1]
    extreme1 = extremes[-2]

    neckline2_candidates = [p for p in necklines if extreme1.index < p.index < extreme2.index]
    if not neckline2_candidates:
        return None
    neckline2 = max(neckline2_candidates, key=lambda p: p.index)

    neckline1_candidates = [p for p in necklines if p.index < extreme1.index]
    if not neckline1_candidates:
        return None
    neckline1 = max(neckline1_candidates, key=lambda p: p.index)

    leg_prior_size = abs(extreme1.price - neckline1.price)
    leg_latest_size = abs(extreme2.price - neckline2.price)
    if leg_prior_size <= 0.0:
        return None
    momentum_ratio = leg_latest_size / leg_prior_size
    if momentum_ratio > min_weakness_ratio:
        return None

    vol_prior = float(volumes_arr[neckline1.index : extreme1.index + 1].mean())
    vol_latest = float(volumes_arr[neckline2.index : extreme2.index + 1].mean())
    if vol_prior <= 0.0:
        return None
    volume_ratio = vol_latest / vol_prior
    if volume_ratio > min_weakness_ratio:
        return None

    reversal_direction = Direction.SELL if prevailing is Direction.BUY else Direction.BUY
    latest_close = float(closes_arr[-1])
    if reversal_direction is Direction.SELL:
        if latest_close > neckline2.price - momentum_mult * atr_now:
            return None
    else:
        if latest_close < neckline2.price + momentum_mult * atr_now:
            return None

    weakness_score = float(np.clip(1.0 - max(momentum_ratio, volume_ratio), 0.0, 1.0))
    structure_points = tuple(
        StructurePoint(time=p.time, price=p.price, label=p.label)
        for p in sorted((neckline1, extreme1, neckline2, extreme2), key=lambda p: p.index)
    )

    return ReversalSetup(
        direction=reversal_direction,
        pivot_index=neckline2.index,
        pivot_price=neckline2.price,
        weakness_score=weakness_score,
        trend_strength=float(adx_value),
        structure_points=structure_points,
    )


def detect_structure_reversal(
    *,
    highs,
    lows,
    closes,
    volumes,
    times: Sequence[datetime],
    atr_series,
    params: dict | None = None,
) -> ReversalSetup | None:
    """Pure detector for a trend that has shown structural weakness and then
    broken. See module docstring for the concept; the sequence checked here:

      1. `latest_trend_regime` reports TRENDING — a trend-strength floor,
         direction-agnostic (unlike `detect_trend_continuation`'s gate 1,
         this function does not also require an EMA-direction read on the
         latest bar: that read is exactly what a fresh break is dragging
         toward the new direction, so using it to pick which prevailing
         trend to test would be unreliable right when this function is
         supposed to fire). Both directions are tried instead — see
         `_try_reversal_direction`.
      2. For whichever direction has one: its two most recent same-kind
         "extreme" pivots (swing highs in an uptrend, swing lows in a
         downtrend) and the opposite-kind "neckline" pivot between them and
         before them — four points bracketing the two most recent impulse
         legs. The latest leg must be <= `min_weakness_ratio` of the prior
         leg's size *and* mean volume (both, in ATR/participation terms) —
         a real momentum+participation divergence, not just a smaller
         candle.
      3. A change of character: the latest closed bar's close breaks the
         neckline between the two legs by at least
         `momentum_confirm_atr_mult` ATR, in the direction opposite that
         hypothesis's prevailing trend.

    Returns `None` at the first gate not satisfied; never raises except for
    mismatched input shapes. `volumes` is any array-like of the same length
    as the other series (e.g. a candle frame's `tick_volume` column) — real
    traded volume is not available for most CFD/FX symbols, so this is
    participation as the broker reports it, not exchange volume.

    Recognized `params` keys (all optional, defaults per the module-level
    `DEFAULT_*` constants): `pivot_bars`, `trend_adx_period`,
    `trend_adx_threshold`, `min_weakness_ratio`, `momentum_confirm_atr_mult`.
    """
    highs_arr = _as_array(highs)
    lows_arr = _as_array(lows)
    closes_arr = _as_array(closes)
    volumes_arr = _as_array(volumes)
    n = highs_arr.size
    if lows_arr.size != n or closes_arr.size != n or volumes_arr.size != n or len(times) != n:
        raise ValueError("highs, lows, closes, volumes, and times must be the same length")
    if n == 0:
        return None

    pivot_bars = int(_param(params, "pivot_bars", DEFAULT_PIVOT_BARS))
    trend_adx_period = int(_param(params, "trend_adx_period", DEFAULT_TREND_ADX_PERIOD))
    trend_adx_threshold = float(_param(params, "trend_adx_threshold", DEFAULT_TREND_ADX_THRESHOLD))
    min_weakness_ratio = float(_param(params, "min_weakness_ratio", DEFAULT_MIN_WEAKNESS_RATIO))
    momentum_mult = float(
        _param(params, "momentum_confirm_atr_mult", DEFAULT_MOMENTUM_CONFIRM_ATR_MULT)
    )

    regime, adx_value = latest_trend_regime(
        highs_arr,
        lows_arr,
        closes_arr,
        adx_period=trend_adx_period,
        adx_trend_threshold=trend_adx_threshold,
    )
    if regime is not TrendRegime.TRENDING:
        return None

    atr_now = _latest_finite(atr_series)
    if atr_now is None or atr_now <= 0.0:
        return None

    pivots = detect_swing_pivots(highs_arr, lows_arr, list(times), pivot_bars=pivot_bars)
    if len(pivots) < 3:
        return None

    for prevailing in (Direction.BUY, Direction.SELL):
        setup = _try_reversal_direction(
            prevailing=prevailing,
            pivots=pivots,
            closes_arr=closes_arr,
            volumes_arr=volumes_arr,
            atr_now=atr_now,
            adx_value=adx_value,
            min_weakness_ratio=min_weakness_ratio,
            momentum_mult=momentum_mult,
        )
        if setup is not None:
            return setup
    return None
