"""Unit tests for `src/strategies/domain/structure_continuation.py` — the
trend-continuation / breakout structure detector shared by the M1 S&D
strategy family (see that module's docstring for the full "why").

Two groups:

  * `detect_swing_pivots` cross-checked directly against `engine.domain.
    structure_pivot.detect_swing_pivots`'s own zigzag fixture (imported
    from `tests/unit/engine/test_structure_pivot.py`'s module so both
    implementations are proven to agree on the exact same input, not just
    "look similar") — this module deliberately reimplements that engine
    algorithm rather than importing it (sandbox rules), so this is the
    guarantee that reimplementation didn't drift.
  * `detect_trend_continuation` end-to-end: a clean up/down trend with a
    qualifying pullback fires; choppy, too-shallow, too-deep, wrong-type-
    pivot, and momentum-not-yet-resumed inputs each correctly return
    `None`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.engine.domain.regime import latest_trend_regime
from src.engine.domain.structure_pivot import detect_swing_pivots as engine_detect_swing_pivots
from src.engine.domain.zone_detection import atr
from src.strategies.domain.models import Direction, StructureLabel
from src.strategies.domain.structure_continuation import (
    DEFAULT_MAX_PULLBACK_ATR,
    DEFAULT_MIN_PULLBACK_ATR,
    DEFAULT_MIN_WEAKNESS_RATIO,
    detect_structure_reversal,
    detect_swing_pivots,
    detect_trend_continuation,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def _times(n: int) -> list[datetime]:
    return [T0 + i * STEP for i in range(n)]


# ── detect_swing_pivots: cross-check against engine/domain/structure_pivot.py ──


def test_pivot_labeling_matches_engine_on_the_uptrend_zigzag_fixture() -> None:
    """Same lows/highs `test_structure_pivot.py` uses for its own uptrend
    zigzag assertion (see that test's docstring) — both implementations
    must agree bar-for-bar on index/label/price."""
    lows = [10, 9, 8, 9, 8.5, 9.5, 9, 10, 9.5, 11, 10.5, 12]
    highs = [10.5, 9.5, 8.5, 9.5, 9, 10, 9.5, 10.5, 10, 11.5, 11, 12.5]
    times = _times(len(lows))

    engine_pivots = engine_detect_swing_pivots(np.array(highs), np.array(lows), times, pivot_bars=1)
    ours = detect_swing_pivots(np.array(highs), np.array(lows), times, pivot_bars=1)

    assert [(p.index, p.label, p.price) for p in ours] == [
        (p.index, p.label, p.price) for p in engine_pivots
    ]
    # And matches the literal expectation asserted by the engine's own test.
    assert [(p.index, p.label, p.price) for p in ours] == [
        (4, StructureLabel.HL, 8.5),
        (5, StructureLabel.HH, 10.0),
        (6, StructureLabel.HL, 9.0),
        (7, StructureLabel.HH, 10.5),
        (8, StructureLabel.HL, 9.5),
        (9, StructureLabel.HH, 11.5),
        (10, StructureLabel.HL, 10.5),
    ]


def test_pivot_labeling_matches_engine_on_the_downtrend_zigzag_fixture() -> None:
    """Mirror of the above, on `test_structure_pivot.py`'s downtrend
    zigzag fixture."""
    highs = [10, 9, 8, 9, 8.5, 7.5, 8, 7, 7.5, 6, 6.5, 5]
    lows = [9.5, 8.5, 7.5, 8.5, 8, 7, 7.5, 6.5, 7, 5.5, 6, 4.5]
    times = _times(len(lows))

    engine_pivots = engine_detect_swing_pivots(np.array(highs), np.array(lows), times, pivot_bars=1)
    ours = detect_swing_pivots(np.array(highs), np.array(lows), times, pivot_bars=1)

    assert [(p.index, p.label, p.price) for p in ours] == [
        (p.index, p.label, p.price) for p in engine_pivots
    ]


def test_flat_candles_produce_no_pivots() -> None:
    n = 20
    highs = np.full(n, 100.6)
    lows = np.full(n, 99.4)
    assert detect_swing_pivots(highs, lows, _times(n), pivot_bars=2) == []


def test_confirmation_lags_pivot_bars_on_each_edge() -> None:
    n = 10
    rng = np.random.default_rng(7)
    highs = 100 + rng.random(n)
    lows = highs - 1 - rng.random(n)
    pivots = detect_swing_pivots(highs, lows, _times(n), pivot_bars=3)
    assert all(3 <= p.index < n - 3 for p in pivots)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        detect_swing_pivots(np.array([1.0, 2.0]), np.array([1.0]), _times(2))


# ── detect_trend_continuation: synthetic OHLC scenarios ──────────────────


def _build_zigzag(
    *,
    n_legs: int,
    leg: float,
    pull: float,
    bars_per_leg: int,
    tail_bars: int,
    tail_step: float,
    offset: float = 0.05,
    up: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """A clean alternating zigzag: `n_legs` impulse-then-pullback cycles
    (impulse size `leg`, pullback size `pull`, each spread over
    `bars_per_leg` bars), then `tail_bars` extra bars moving `tail_step`
    per bar in the trend direction — the "resumption" after the final
    pullback. `up=True` builds an uptrend (impulses up, pullbacks down);
    `up=False` mirrors it into a downtrend."""
    sign = 1.0 if up else -1.0
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for _ in range(n_legs):
        for _ in range(bars_per_leg):
            price += sign * leg / bars_per_leg
            highs.append(price + offset)
            lows.append(price - offset)
        for _ in range(bars_per_leg):
            price -= sign * pull / bars_per_leg
            highs.append(price + offset)
            lows.append(price - offset)
    for _ in range(tail_bars):
        price += sign * tail_step
        highs.append(price + offset)
        lows.append(price - offset)
    return np.array(highs), np.array(lows)


def _setup(highs: np.ndarray, lows: np.ndarray, params: dict | None = None):
    closes = (highs + lows) / 2.0
    n = len(highs)
    atr_series = atr(highs, lows, closes, 14)
    return detect_trend_continuation(
        highs=highs,
        lows=lows,
        closes=closes,
        times=_times(n),
        atr_series=atr_series,
        params=params or {},
    )


def test_fires_on_clean_uptrend_with_qualifying_pullback_and_resumption() -> None:
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=3, tail_step=0.5, up=True
    )
    setup = _setup(highs, lows)
    assert setup is not None
    assert setup.direction is Direction.BUY
    assert DEFAULT_MIN_PULLBACK_ATR <= setup.pullback_depth_atr <= DEFAULT_MAX_PULLBACK_ATR
    assert setup.trend_strength > 0.0
    assert setup.structure_points  # chart-annotation data populated


def test_fires_on_clean_downtrend_with_qualifying_pullback_and_resumption() -> None:
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=3, tail_step=0.5, up=False
    )
    setup = _setup(highs, lows)
    assert setup is not None
    assert setup.direction is Direction.SELL
    assert DEFAULT_MIN_PULLBACK_ATR <= setup.pullback_depth_atr <= DEFAULT_MAX_PULLBACK_ATR


def test_none_when_market_is_choppy_not_trending() -> None:
    """A tight two-bar alternation nets zero directional movement — ADX
    reads ~0, `latest_trend_regime` reports RANGING, and the detector must
    bail before ever looking at pivots."""
    n = 90
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for i in range(n):
        price += 0.5 if i % 2 == 0 else -0.5
        highs.append(price + 0.3)
        lows.append(price - 0.3)
    highs_arr, lows_arr = np.array(highs), np.array(lows)
    closes_arr = (highs_arr + lows_arr) / 2.0
    regime, _adx = latest_trend_regime(highs_arr, lows_arr, closes_arr)
    assert regime.value == "ranging"
    assert _setup(highs_arr, lows_arr) is None


def test_none_when_pullback_shallower_than_floor() -> None:
    highs, lows = _build_zigzag(
        n_legs=11,
        leg=3.0,
        pull=0.1,
        bars_per_leg=4,
        tail_bars=3,
        tail_step=0.5,
        offset=0.02,
        up=True,
    )
    assert _setup(highs, lows) is None
    # Confirm it's specifically the floor doing the vetoing, not some other
    # gate — bypassing the floor alone must let it fire.
    bypassed = _setup(highs, lows, params={"min_pullback_atr": -100.0})
    assert bypassed is not None
    assert bypassed.pullback_depth_atr < DEFAULT_MIN_PULLBACK_ATR


def test_none_when_pullback_deeper_than_ceiling() -> None:
    highs, lows = _build_zigzag(
        n_legs=11, leg=6.0, pull=3.0, bars_per_leg=4, tail_bars=3, tail_step=0.5, up=True
    )
    assert _setup(highs, lows) is None
    bypassed = _setup(highs, lows, params={"max_pullback_atr": 100.0})
    assert bypassed is not None
    assert bypassed.pullback_depth_atr > DEFAULT_MAX_PULLBACK_ATR


def test_none_when_momentum_has_not_resumed() -> None:
    """A qualifying HL pivot (pullback depth safely inside the band) whose
    resumption bars are too small to clear the pivot bar's own high by
    `momentum_confirm_atr_mult` ATR yet — the pullback held, but price
    hasn't actually pushed back through it."""
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=0.6, bars_per_leg=4, tail_bars=2, tail_step=0.02, up=True
    )
    assert _setup(highs, lows) is None
    bypassed = _setup(highs, lows, params={"momentum_confirm_atr_mult": -100.0})
    assert bypassed is not None
    assert DEFAULT_MIN_PULLBACK_ATR <= bypassed.pullback_depth_atr <= DEFAULT_MAX_PULLBACK_ATR


def test_none_when_latest_pivot_is_the_wrong_type() -> None:
    """Same uptrend as the firing test, cut short right after a fresh HH
    confirms (not an HL) — an uptrend continuation must not fire off a
    high pivot."""
    highs_full, lows_full = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=3, tail_step=0.5, up=True
    )
    highs, lows = highs_full[:86], lows_full[:86]
    pivots = detect_swing_pivots(highs, lows, _times(len(highs)), pivot_bars=2)
    assert pivots[-1].label is StructureLabel.HH  # confirms the fixture is set up as intended
    assert _setup(highs, lows) is None


def test_no_pivot_at_all_returns_none() -> None:
    n = 10
    highs = 100.0 + np.arange(n, dtype=float) * 0.1
    lows = highs - 0.5
    assert _setup(highs, lows) is None


def test_mismatched_input_lengths_raise() -> None:
    highs = np.array([1.0, 2.0, 3.0])
    lows = np.array([1.0, 2.0])
    closes = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        detect_trend_continuation(
            highs=highs, lows=lows, closes=closes, times=_times(3), atr_series=closes, params={}
        )


# ── detect_structure_reversal: synthetic OHLCV scenarios ─────────────────


def _build_weakening_reversal(
    *,
    n_warmup_legs: int = 8,
    leg: float = 3.0,
    pull: float = 1.0,
    weak_leg: float = 1.0,
    bars_per_leg: int = 4,
    break_bars: int = 5,
    break_step: float = 1.0,
    offset: float = 0.05,
    base_volume: float = 1000.0,
    weak_volume: float = 300.0,
    up: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`n_warmup_legs` normal impulse/pullback cycles (establishes TRENDING
    and supplies the "prior" leg the weakening check compares against),
    then one smaller, lower-volume `weak_leg` impulse, then a `break_bars`
    decline past the neckline (the low — or high, if `up=False` — right
    before that weak leg) at reduced volume, mirroring a real CHoCH: the
    push into the reversal was already weak before it broke."""
    sign = 1.0 if up else -1.0
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    price = 100.0

    def _emit(step: float, vol: float) -> None:
        nonlocal price
        price += sign * step
        highs.append(price + offset)
        lows.append(price - offset)
        volumes.append(vol)

    for _ in range(n_warmup_legs):
        for _ in range(bars_per_leg):
            _emit(leg / bars_per_leg, base_volume)
        for _ in range(bars_per_leg):
            _emit(-pull / bars_per_leg, base_volume)

    for _ in range(bars_per_leg):
        _emit(weak_leg / bars_per_leg, weak_volume)

    for _ in range(break_bars):
        _emit(-break_step, weak_volume)

    return np.array(highs), np.array(lows), np.array(volumes)


def _setup_reversal(
    highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray, params: dict | None = None
):
    closes = (highs + lows) / 2.0
    n = len(highs)
    atr_series = atr(highs, lows, closes, 14)
    return detect_structure_reversal(
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        times=_times(n),
        atr_series=atr_series,
        params=params or {},
    )


def test_reversal_fires_on_weakening_uptrend_broken_by_choch() -> None:
    highs, lows, volumes = _build_weakening_reversal(up=True)
    setup = _setup_reversal(highs, lows, volumes)
    assert setup is not None
    assert setup.direction is Direction.SELL  # reversing OUT of the uptrend
    assert 0.0 < setup.weakness_score <= 1.0
    assert setup.trend_strength > 0.0
    assert setup.structure_points


def test_reversal_fires_on_weakening_downtrend_broken_by_choch() -> None:
    highs, lows, volumes = _build_weakening_reversal(up=False)
    setup = _setup_reversal(highs, lows, volumes)
    assert setup is not None
    assert setup.direction is Direction.BUY  # reversing OUT of the downtrend


def test_reversal_none_when_final_leg_is_not_weaker() -> None:
    """Same size/volume as every prior leg — a normal continuation impulse,
    not a weakening one — even though the tail still breaks back down."""
    highs, lows, volumes = _build_weakening_reversal(
        weak_leg=3.0, weak_volume=1000.0, break_bars=6, break_step=1.2
    )
    assert _setup_reversal(highs, lows, volumes) is None
    bypassed = _setup_reversal(highs, lows, volumes, params={"min_weakness_ratio": 100.0})
    assert bypassed is not None


def test_reversal_none_when_leg_shrinks_but_volume_does_not() -> None:
    """The momentum condition alone (smaller leg) must not be enough —
    volume has to confirm the weakness too."""
    highs, lows, volumes = _build_weakening_reversal(weak_leg=1.0, weak_volume=1000.0)
    assert _setup_reversal(highs, lows, volumes) is None


def test_reversal_none_when_volume_drops_but_leg_does_not_shrink() -> None:
    """The volume condition alone must not be enough either — both must
    hold together (see module docstring for why: either alone is noise)."""
    highs, lows, volumes = _build_weakening_reversal(weak_leg=3.0, weak_volume=200.0)
    assert _setup_reversal(highs, lows, volumes) is None


def test_reversal_none_when_break_has_not_reached_the_neckline_yet() -> None:
    """Leg and volume both weaken, but the pullback is too shallow to have
    actually broken the neckline — no CHoCH yet, so no signal yet."""
    highs, lows, volumes = _build_weakening_reversal(break_bars=2, break_step=0.1)
    assert _setup_reversal(highs, lows, volumes) is None
    bypassed = _setup_reversal(highs, lows, volumes, params={"momentum_confirm_atr_mult": -100.0})
    assert bypassed is not None


def test_reversal_none_when_market_is_choppy_not_trending() -> None:
    n = 90
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    price = 100.0
    for i in range(n):
        price += 0.5 if i % 2 == 0 else -0.5
        highs.append(price + 0.3)
        lows.append(price - 0.3)
        volumes.append(1000.0)
    assert _setup_reversal(np.array(highs), np.array(lows), np.array(volumes)) is None


def test_reversal_mismatched_input_lengths_raise() -> None:
    highs = np.array([1.0, 2.0, 3.0])
    lows = np.array([1.0, 2.0])
    closes = np.array([1.0, 2.0, 3.0])
    volumes = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        detect_structure_reversal(
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            times=_times(3),
            atr_series=closes,
            params={},
        )


def test_reversal_default_min_weakness_ratio_is_documented_value() -> None:
    # Guards the module docstring's claim (0.7 = at least a 30% drop) from
    # silently drifting without the docs being updated to match.
    assert DEFAULT_MIN_WEAKNESS_RATIO == 0.7
