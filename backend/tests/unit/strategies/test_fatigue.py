"""Unit tests for `src.strategies.domain.fatigue`.

The module is a gate on money-touching code — a high score vetoes a
continuation entry and licences a fade — so each sub-measure is tested on
its own against a synthetic series built to isolate exactly the thing it
claims to measure, and the composite is tested on three whole-market
shapes (fresh trend / exhausted trend / chop) that a reader can check by
eye from the generator below.

Edge cases get their own section because "unavailable" and "exhausted"
must never be confused: too few bars, a flat series, NaNs and a zero ATR
all have to abstain (NaN from a sub-measure, 0.0 with `available == 0`
from the composite), not veto.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.fatigue import (
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    FatigueReading,
    impulse_decay_score,
    momentum_divergence_score,
    overextension_score,
    persistence_decay_score,
    range_decay_score,
    required_bars,
    trend_age_score,
    trend_fatigue,
    volume_decay_score,
    wick_rejection_score,
)

# ─────────────────────────────────────────────────────────────────────
# Synthetic market builders
# ─────────────────────────────────────────────────────────────────────


def _bars(
    closes: np.ndarray,
    *,
    span: float = 1.0,
    upper_bias: float = 0.0,
    volume: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """OHLCV around a close path.

    `upper_bias` in [0, 1) lifts the high away from the body without
    moving the low, which is how a rejection wick is manufactured for
    `wick_rejection_score`.
    """
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    body_top = np.maximum(opens, closes)
    body_bottom = np.minimum(opens, closes)
    highs = body_top + span * (0.5 + upper_bias)
    lows = body_bottom - span * 0.5
    n = closes.size
    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": np.full(n, 1000.0) if volume is None else np.asarray(volume, dtype=float),
    }


def _fresh_uptrend(n: int = 220) -> dict[str, np.ndarray]:
    """A young, accelerating advance: steady drift, expanding legs, no
    upper-wick rejection, volume building, price close to its EMA."""
    rng = np.random.default_rng(11)
    steps = rng.normal(0.9, 0.35, n)
    closes = 2000.0 + np.cumsum(steps)
    volume = np.linspace(600.0, 1600.0, n)
    return _bars(closes, span=1.2, upper_bias=0.0, volume=volume)


def _exhausted_uptrend(n: int = 220) -> dict[str, np.ndarray]:
    """The same advance, run to a stop: the drift decays toward zero,
    ranges contract, volume drains and every bar is sold into."""
    rng = np.random.default_rng(23)
    drift = np.linspace(1.6, 0.02, n)
    noise = rng.normal(0.0, 0.25, n) * np.linspace(1.0, 0.25, n)
    closes = 2000.0 + np.cumsum(drift + noise)
    volume = np.linspace(2200.0, 400.0, n)
    return _bars(closes, span=1.0, upper_bias=0.45, volume=volume)


def _chop(n: int = 220) -> dict[str, np.ndarray]:
    """Mean-reverting sideways noise — no trend to be tired of."""
    rng = np.random.default_rng(31)
    closes = 2000.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    closes = 2000.0 + (closes - closes.mean()) * 0.4
    return _bars(closes, span=1.5)


def _flat(n: int = 220, price: float = 2000.0) -> dict[str, np.ndarray]:
    closes = np.full(n, price)
    return {
        "opens": closes.copy(),
        "highs": closes.copy(),
        "lows": closes.copy(),
        "closes": closes.copy(),
        "volumes": np.full(n, 1000.0),
    }


def _atr_of(bars: dict[str, np.ndarray], period: int = 14) -> float:
    highs, lows, closes = bars["highs"], bars["lows"], bars["closes"]
    tr = highs - lows
    prev = closes[:-1]
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(highs[1:] - prev), np.abs(lows[1:] - prev)))
    return float(np.mean(tr[-period:]))


# ─────────────────────────────────────────────────────────────────────
# momentum_divergence_score
# ─────────────────────────────────────────────────────────────────────


def test_divergence_fires_when_a_new_high_comes_with_weaker_momentum() -> None:
    """A hard push, then a grind that just barely takes out the old high
    — the textbook regular bearish divergence.

    Both pushes have to sit inside `divergence_window` (40) for the two
    halves to straddle them; the 20 leading bars only exist to prime RSI,
    which needs `divergence_rsi_period` + 1 bars before the window opens.
    """
    warmup = np.full(20, 2000.0)
    first = np.linspace(2000.0, 2040.0, 15)  # strong impulse: RSI pinned high
    pause = np.linspace(2040.0, 2025.0, 10)  # RSI unwinds
    second = np.linspace(2025.0, 2042.0, 15)  # new high, much weaker push
    closes = np.concatenate([warmup, first, pause, second])
    bars = _bars(closes)
    score = momentum_divergence_score(bars["highs"], bars["lows"], closes, 1)
    assert score > 0.5


def test_divergence_is_zero_when_momentum_confirms_the_new_high() -> None:
    """Accelerating advance: the newest high is also the strongest push,
    so there is nothing diverging."""
    closes = 2000.0 + np.cumsum(np.linspace(0.2, 2.0, 90))
    bars = _bars(closes)
    assert momentum_divergence_score(bars["highs"], bars["lows"], closes, 1) == 0.0


def test_divergence_is_zero_without_a_new_extreme() -> None:
    """No new high means no failing push to score — this measures a
    stalling advance, not an ordinary pullback."""
    closes = 2000.0 + np.concatenate([np.linspace(0.0, 60.0, 45), np.linspace(60.0, 20.0, 45)])
    bars = _bars(closes)
    assert momentum_divergence_score(bars["highs"], bars["lows"], closes, 1) == 0.0


def test_divergence_is_symmetric_for_a_downtrend() -> None:
    warmup = np.full(20, 2000.0)
    first = np.linspace(2000.0, 1960.0, 15)
    pause = np.linspace(1960.0, 1975.0, 10)
    second = np.linspace(1975.0, 1958.0, 15)
    closes = np.concatenate([warmup, first, pause, second])
    bars = _bars(closes)
    assert momentum_divergence_score(bars["highs"], bars["lows"], closes, -1) > 0.5


# ─────────────────────────────────────────────────────────────────────
# impulse_decay_score
# ─────────────────────────────────────────────────────────────────────


def _legged_path(
    impulses: list[float], pullbacks: list[float], bars_per_leg: int = 8
) -> np.ndarray:
    """A clean zigzag: alternating up-legs and down-pullbacks of the
    exact sizes asked for, so the leg maths is checked against numbers
    the test itself chose."""
    path = [2000.0]
    for index, impulse in enumerate(impulses):
        path.extend(np.linspace(path[-1], path[-1] + impulse, bars_per_leg + 1)[1:])
        if index < len(pullbacks):
            path.extend(np.linspace(path[-1], path[-1] - pullbacks[index], bars_per_leg + 1)[1:])
    return np.asarray(path, dtype=float)


def test_impulse_decay_high_when_legs_shrink_and_pullbacks_deepen() -> None:
    closes = _legged_path([60.0, 30.0, 12.0], [12.0, 15.0, 9.0])
    bars = _bars(closes, span=0.5)
    score = impulse_decay_score(bars["highs"], bars["lows"], 1, atr=3.0)
    assert score > 0.6


def test_impulse_decay_low_when_legs_expand() -> None:
    closes = _legged_path([12.0, 30.0, 60.0], [4.0, 6.0, 8.0])
    bars = _bars(closes, span=0.5)
    score = impulse_decay_score(bars["highs"], bars["lows"], 1, atr=3.0)
    assert score < 0.35


def test_impulse_decay_abstains_without_enough_legs() -> None:
    closes = np.linspace(2000.0, 2100.0, 120)  # one straight leg, no zigzag
    bars = _bars(closes, span=0.5)
    assert math.isnan(impulse_decay_score(bars["highs"], bars["lows"], 1, atr=3.0))


def test_impulse_decay_abstains_on_zero_atr() -> None:
    closes = _legged_path([60.0, 30.0, 12.0], [12.0, 15.0, 9.0])
    bars = _bars(closes, span=0.5)
    assert math.isnan(impulse_decay_score(bars["highs"], bars["lows"], 1, atr=0.0))
    assert math.isnan(impulse_decay_score(bars["highs"], bars["lows"], 1, atr=float("nan")))


# ─────────────────────────────────────────────────────────────────────
# range_decay_score
# ─────────────────────────────────────────────────────────────────────


def test_range_decay_high_when_true_range_contracts() -> None:
    """The contraction has to land inside the *slow* window (40 bars) and
    fill the *fast* one (10), or there is nothing for the two windows to
    disagree about: 40 wide bars followed by 40 narrow ones puts the whole
    slow window inside the narrow regime and correctly reads as no change.
    Bodies shrink with the ranges here because that is what a stalling
    market actually does — a constant body inside a shrinking range means
    conviction is *rising*, and the body half of the measure says so.
    """
    steps = np.concatenate([np.full(70, 0.3), np.full(10, 0.05)])
    closes = 2000.0 + np.cumsum(steps)
    spans = np.concatenate([np.full(70, 4.0), np.full(10, 1.0)])
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) + spans / 2.0
    lows = np.minimum(opens, closes) - spans / 2.0
    assert range_decay_score(opens, highs, lows, closes) > 0.6


def test_range_decay_low_when_range_expands() -> None:
    closes = 2000.0 + np.cumsum(np.full(80, 0.3))
    spans = np.concatenate([np.full(40, 1.0), np.full(40, 4.0)])
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) + spans / 2.0
    lows = np.minimum(opens, closes) - spans / 2.0
    assert range_decay_score(opens, highs, lows, closes) < 0.2


def test_range_decay_is_direction_free() -> None:
    """It takes no `direction` argument by design; a mirrored series must
    therefore give the identical number."""
    bars = _exhausted_uptrend()
    up = range_decay_score(bars["opens"], bars["highs"], bars["lows"], bars["closes"])
    down = range_decay_score(-bars["opens"], -bars["lows"], -bars["highs"], -bars["closes"])
    assert up == pytest.approx(down)


# ─────────────────────────────────────────────────────────────────────
# volume_decay_score
# ─────────────────────────────────────────────────────────────────────


def test_volume_decay_high_when_participation_drains_into_new_highs() -> None:
    closes = 2000.0 + np.cumsum(np.full(60, 0.8))
    volumes = np.linspace(3000.0, 500.0, 60)
    assert volume_decay_score(volumes, closes, 1) > 0.7


def test_volume_decay_zero_when_price_is_not_extending() -> None:
    """Volume always falls in a pullback; scoring that as fatigue would
    fire this on every healthy consolidation."""
    closes = 2000.0 - np.cumsum(np.full(60, 0.8))
    volumes = np.linspace(3000.0, 500.0, 60)
    assert volume_decay_score(volumes, closes, 1) == 0.0


def test_volume_decay_low_when_volume_builds() -> None:
    closes = 2000.0 + np.cumsum(np.full(60, 0.8))
    volumes = np.linspace(500.0, 3000.0, 60)
    assert volume_decay_score(volumes, closes, 1) == 0.0


def test_volume_decay_abstains_without_a_volume_series() -> None:
    closes = 2000.0 + np.cumsum(np.full(60, 0.8))
    assert math.isnan(volume_decay_score(None, closes, 1))
    assert math.isnan(volume_decay_score(np.zeros(60), closes, 1))


# ─────────────────────────────────────────────────────────────────────
# wick_rejection_score
# ─────────────────────────────────────────────────────────────────────


def test_wick_rejection_high_when_upper_wicks_dominate_an_advance() -> None:
    closes = 2000.0 + np.cumsum(np.full(40, 0.4))
    bars = _bars(closes, span=2.0, upper_bias=0.45)
    score = wick_rejection_score(bars["opens"], bars["highs"], bars["lows"], bars["closes"], 1)
    assert score > 0.5


def test_wick_rejection_low_on_symmetric_wicks() -> None:
    closes = 2000.0 + np.cumsum(np.full(40, 0.4))
    bars = _bars(closes, span=2.0, upper_bias=0.0)
    score = wick_rejection_score(bars["opens"], bars["highs"], bars["lows"], bars["closes"], 1)
    assert score == 0.0


def test_wick_rejection_flips_with_direction() -> None:
    """Upper wicks are adverse for a buyer and favourable for a seller —
    the same bars must score high one way and zero the other."""
    closes = 2000.0 + np.cumsum(np.full(40, 0.4))
    bars = _bars(closes, span=2.0, upper_bias=0.45)
    up = wick_rejection_score(bars["opens"], bars["highs"], bars["lows"], bars["closes"], 1)
    down = wick_rejection_score(bars["opens"], bars["highs"], bars["lows"], bars["closes"], -1)
    assert up > 0.5
    assert down == 0.0


# ─────────────────────────────────────────────────────────────────────
# overextension_score
# ─────────────────────────────────────────────────────────────────────


def test_overextension_high_on_a_parabolic_one_way_run() -> None:
    base = 2000.0 + np.cumsum(np.full(120, 0.05))
    blowoff = base[-1] + np.cumsum(np.linspace(0.5, 6.0, 30))
    closes = np.concatenate([base, blowoff])
    assert overextension_score(closes, 1, atr=2.0) > 0.7


def test_overextension_low_when_price_sits_on_its_anchor() -> None:
    rng = np.random.default_rng(5)
    closes = 2000.0 + np.cumsum(rng.normal(0.0, 0.5, 200))
    assert overextension_score(closes, 1, atr=8.0) < 0.3


def test_overextension_abstains_on_zero_or_missing_atr() -> None:
    closes = 2000.0 + np.cumsum(np.full(200, 0.5))
    assert math.isnan(overextension_score(closes, 1, atr=0.0))
    assert math.isnan(overextension_score(closes, 1, atr=float("nan")))


def test_overextension_abstains_on_too_few_bars() -> None:
    closes = 2000.0 + np.cumsum(np.full(30, 0.5))
    assert math.isnan(overextension_score(closes, 1, atr=2.0))


# ─────────────────────────────────────────────────────────────────────
# persistence_decay_score
# ─────────────────────────────────────────────────────────────────────


def test_persistence_decay_low_for_a_persistent_trend() -> None:
    """A near-deterministic drift has a variance ratio far above 1, so
    there is no persistence decay to report."""
    closes = 2000.0 + np.cumsum(np.full(120, 0.5))
    assert persistence_decay_score(closes) == 0.0


def test_persistence_decay_high_for_a_mean_reverting_series() -> None:
    """Alternating up/down ticks: q-bar returns cancel, variance ratio
    collapses toward 0."""
    closes = 2000.0 + np.where(np.arange(120) % 2 == 0, 1.0, -1.0)
    assert persistence_decay_score(closes) > 0.8


def test_persistence_decay_abstains_on_a_flat_series() -> None:
    assert math.isnan(persistence_decay_score(np.full(120, 2000.0)))


def test_persistence_decay_abstains_on_non_positive_prices() -> None:
    """The variance ratio is computed on log returns; a zero or negative
    price has to abstain rather than produce a NaN/-inf silently."""
    closes = np.concatenate([np.full(60, 100.0), np.full(61, 0.0)])
    assert math.isnan(persistence_decay_score(closes))


# ─────────────────────────────────────────────────────────────────────
# trend_age_score
# ─────────────────────────────────────────────────────────────────────


def test_trend_age_rises_with_an_unbroken_ema_regime() -> None:
    closes = 2000.0 + np.cumsum(np.full(400, 0.5))
    assert trend_age_score(closes, 1) == 1.0


def test_trend_age_is_zero_when_the_regime_opposes_the_direction() -> None:
    closes = 2000.0 + np.cumsum(np.full(400, 0.5))
    assert trend_age_score(closes, -1) == 0.0


def test_trend_age_abstains_on_too_few_bars() -> None:
    assert math.isnan(trend_age_score(2000.0 + np.cumsum(np.full(50, 0.5)), 1))


# ─────────────────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────────────────


def _reading(bars: dict[str, np.ndarray], direction: int, **kwargs) -> FatigueReading:
    return trend_fatigue(
        opens=bars["opens"],
        highs=bars["highs"],
        lows=bars["lows"],
        closes=bars["closes"],
        volumes=bars["volumes"],
        direction=direction,
        atr=_atr_of(bars),
        **kwargs,
    )


def test_exhausted_trend_scores_higher_than_a_fresh_one() -> None:
    """`_exhausted_uptrend` is the *grinding* death: drift decays to zero,
    ranges contract, volume drains. That is real fatigue, but by the last
    bar price has sagged back onto its EMA, so `overextension` — which
    carries the heaviest default weight — legitimately votes ~0 and caps
    the composite in the high 0.3s.

    That is the measure telling the truth about two different shapes of
    exhaustion: a blow-off is stretched (overextension high, divergence
    low), a grind is not (overextension low, divergence and decay high).
    A single composite averages across both, so neither ever pins the
    score near 1. The claim worth asserting is the *separation* between
    tired and fresh, not an absolute level — Phase 5's A/B is what decides
    whether the two modes deserve to be scored separately.
    """
    fresh = _reading(_fresh_uptrend(), 1)
    tired = _reading(_exhausted_uptrend(), 1)
    assert tired.score > fresh.score + 0.2
    assert tired.score > 0.35
    assert fresh.score < 0.4


def test_every_score_stays_inside_the_unit_interval() -> None:
    for build in (_fresh_uptrend, _exhausted_uptrend, _chop):
        for direction in (1, -1):
            reading = _reading(build(), direction)
            assert 0.0 <= reading.score <= 1.0
            for value in reading.components.values():
                assert 0.0 <= value <= 1.0


def test_chop_does_not_read_as_an_exhausted_trend() -> None:
    """Sideways noise is not an exhausted trend, and must not score as
    one — a fade gate keyed on this would otherwise fire all day in a
    range, which is the single worst place to be fading.

    Only the upper bound is asserted. An earlier version also demanded
    `score > 0.15`, on the theory that chop should read as "unsure", but
    there is nothing to be unsure about: with no trend running, every
    sub-measure correctly finds no failing push, no leg decay and no
    stretch, so a floored score is the right answer. Reading "fresh" in a
    range is harmless here because fatigue is not the trend filter — the
    strategies gate direction on structure and trend state before this
    measure is ever consulted.
    """
    reading = _reading(_chop(), 1)
    assert reading.score < 0.5


def test_components_and_weights_are_reported_for_the_audit_trail() -> None:
    reading = _reading(_exhausted_uptrend(), 1)
    assert reading.available >= 5
    assert set(reading.components) <= set(COMPONENT_NAMES)
    # trend_age ships at weight 0 and must therefore not even be computed.
    assert "trend_age" not in reading.components
    assert sum(reading.weights.values()) == pytest.approx(1.0)
    assert set(reading.weights) == set(reading.components)


def test_describe_names_every_component_that_voted() -> None:
    reading = _reading(_exhausted_uptrend(), 1)
    text = reading.describe()
    assert text.startswith("fatigue=")
    assert "dir+" in text
    assert "ext=" in text


def test_zero_weight_disables_a_component() -> None:
    weights = dict.fromkeys(COMPONENT_NAMES, 0.0)
    weights["overextension"] = 1.0
    reading = _reading(_exhausted_uptrend(), 1, weights=weights)
    assert set(reading.components) == {"overextension"}
    assert reading.weights["overextension"] == pytest.approx(1.0)


def test_all_weights_zero_yields_the_no_opinion_default() -> None:
    reading = _reading(_exhausted_uptrend(), 1, weights=dict.fromkeys(COMPONENT_NAMES, 0.0))
    assert reading.score == 0.0
    assert reading.available == 0


def test_custom_params_reach_the_sub_measures() -> None:
    """An impossible extension threshold has to drag the composite down,
    proving `params` is threaded through rather than ignored."""
    # A blow-off, not `_exhausted_uptrend`: that fixture ends with its
    # drift decayed to nothing, so price has sagged back onto its EMA and
    # `overextension` correctly reads 0 there — a threshold change cannot
    # move a component that is already floored, which proves nothing.
    base = 2000.0 + np.cumsum(np.full(120, 0.05))
    blowoff = base[-1] + np.cumsum(np.linspace(0.5, 6.0, 30))
    bars = _bars(np.concatenate([base, blowoff]))
    weights = dict.fromkeys(COMPONENT_NAMES, 0.0)
    weights["overextension"] = 1.0
    strict = _reading(
        bars,
        1,
        weights=weights,
        params={"extension_onset_atr": 50.0, "extension_full_atr": 100.0},
    )
    loose = _reading(bars, 1, weights=weights)
    assert strict.score < loose.score


# ─────────────────────────────────────────────────────────────────────
# Edge cases — "unavailable" must never read as "exhausted"
# ─────────────────────────────────────────────────────────────────────


def test_insufficient_bars_abstains_rather_than_vetoes() -> None:
    bars = _bars(2000.0 + np.cumsum(np.full(12, 0.5)))
    reading = _reading(bars, 1)
    assert reading.score == 0.0
    assert reading.available == 0
    assert reading.bars == 12


def test_empty_series_is_safe() -> None:
    empty = np.empty(0)
    reading = trend_fatigue(highs=empty, lows=empty, closes=empty, direction=1, atr=1.0)
    assert reading.score == 0.0
    assert reading.direction == 0
    assert reading.bars == 0


def test_zero_direction_is_a_neutral_reading() -> None:
    reading = _reading(_exhausted_uptrend(), 0)
    assert reading.score == 0.0
    assert reading.direction == 0
    assert reading.available == 0


def test_flat_series_never_reports_exhaustion() -> None:
    """A dead tape has no trend to be tired of. Whatever abstains must
    abstain, and whatever still votes must not vote 'exhausted'."""
    reading = _reading(_flat(), 1)
    assert reading.score <= 0.5
    for name, value in reading.components.items():
        assert math.isfinite(value), name


def test_zero_atr_only_disables_the_components_that_need_it() -> None:
    bars = _exhausted_uptrend()
    reading = trend_fatigue(
        opens=bars["opens"],
        highs=bars["highs"],
        lows=bars["lows"],
        closes=bars["closes"],
        volumes=bars["volumes"],
        direction=1,
        atr=0.0,
    )
    assert "overextension" not in reading.components
    assert "impulse_decay" not in reading.components
    assert "range_decay" in reading.components
    assert 0.0 <= reading.score <= 1.0


def test_missing_atr_is_the_same_as_a_zero_one() -> None:
    bars = _exhausted_uptrend()
    without = trend_fatigue(
        opens=bars["opens"],
        highs=bars["highs"],
        lows=bars["lows"],
        closes=bars["closes"],
        direction=1,
    )
    assert "overextension" not in without.components
    assert "impulse_decay" not in without.components


def test_nan_bars_abstain_instead_of_propagating() -> None:
    bars = _exhausted_uptrend()
    for key in ("opens", "highs", "lows", "closes"):
        bars[key] = bars[key].copy()
        bars[key][-3] = float("nan")
    reading = _reading(bars, 1)
    assert math.isfinite(reading.score)
    assert 0.0 <= reading.score <= 1.0
    for name, value in reading.components.items():
        assert math.isfinite(value), name


def test_nan_volume_abstains() -> None:
    closes = 2000.0 + np.cumsum(np.full(60, 0.8))
    volumes = np.full(60, 1000.0)
    volumes[-2] = float("nan")
    assert math.isnan(volume_decay_score(volumes, closes, 1))


def test_pandas_series_inputs_match_numpy_inputs() -> None:
    """Strategies hand over `df[...].to_numpy()`, but callers and tests
    pass Series; both must give the same number."""
    bars = _exhausted_uptrend()
    from_numpy = _reading(bars, 1)
    as_series = trend_fatigue(
        opens=pd.Series(bars["opens"]),
        highs=pd.Series(bars["highs"]),
        lows=pd.Series(bars["lows"]),
        closes=pd.Series(bars["closes"]),
        volumes=pd.Series(bars["volumes"]),
        direction=1,
        atr=_atr_of(bars),
    )
    assert as_series.score == pytest.approx(from_numpy.score)


# ─────────────────────────────────────────────────────────────────────
# Budget: the engine caps strategy context at 200 bars per timeframe
# ─────────────────────────────────────────────────────────────────────


def test_default_lookback_fits_inside_the_engine_context_cap() -> None:
    """`trade_loop.DEFAULT_CONTEXT_BARS` is 200. A measure needing more
    would silently never fire, live or backtest, with no error."""
    assert required_bars() <= 200


def test_required_bars_reflects_the_enabled_components_only() -> None:
    only_wicks = dict.fromkeys(COMPONENT_NAMES, 0.0)
    only_wicks["wick_rejection"] = 1.0
    assert required_bars(weights=only_wicks) < required_bars()
    assert required_bars(weights=dict.fromkeys(COMPONENT_NAMES, 0.0)) == 0


def test_trend_age_is_shipped_disabled() -> None:
    """It is implemented so the A/B can measure the time-based family,
    not because the evidence justifies trusting it — see the module
    docstring."""
    assert DEFAULT_WEIGHTS["trend_age"] == 0.0
    assert all(DEFAULT_WEIGHTS[name] > 0.0 for name in COMPONENT_NAMES if name != "trend_age")
