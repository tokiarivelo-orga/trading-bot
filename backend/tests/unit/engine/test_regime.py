"""Unit tests for the engine-side market-regime tagger (OBSERVABILITY_PLAN.md
Phase 6, Pass A): trend/range (ADX) classification, trading-session
bucketing, and the combined `compute_entry_regime` snapshot. Volatility
classification itself is `engine.domain.volatility`'s own module and already
covered by `test_volatility.py` — this file only exercises what `regime.py`
adds on top of it."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.engine.domain.regime import (
    RegimeConfig,
    TradingSession,
    TrendRegime,
    classify_trend_regime,
    compute_entry_regime,
    latest_trend_regime,
    session_for,
)
from src.engine.domain.volatility import VolatilityRegime


def _trending_up_candles(n: int, *, step: float = 1.0, rng: float = 0.5):
    """`n` bars closing `step` higher each bar with a fixed high/low band
    `rng` around the close — a clean, one-directional trend so +DM dominates
    -DM and ADX should climb well above the trend threshold."""
    closes = 100.0 + np.arange(n) * step
    highs = closes + rng / 2
    lows = closes - rng / 2
    return highs, lows, closes


def _choppy_candles(n: int, *, amplitude: float = 1.0, rng: float = 1.0):
    """`n` bars oscillating between two fixed levels bar-to-bar — up moves
    and down moves alternate in exact antiphase, so +DM and -DM average out
    to roughly equal and ADX should stay near zero (RANGING)."""
    closes = 100.0 + amplitude * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    highs = closes + rng / 2
    lows = closes - rng / 2
    return highs, lows, closes


# ── trend/range (ADX) classification ────────────────────────────────────────


def _latest(highs, lows, closes, **kwargs):
    return latest_trend_regime(highs, lows, closes, **kwargs)


def test_trending_series_classifies_as_trending():
    highs, lows, closes = _trending_up_candles(60)

    regime, adx = _latest(highs, lows, closes)

    assert regime == TrendRegime.TRENDING
    assert adx >= 20.0


def test_choppy_series_classifies_as_ranging():
    highs, lows, closes = _choppy_candles(60)

    regime, adx = _latest(highs, lows, closes)

    assert regime == TrendRegime.RANGING
    assert adx < 20.0


def test_trend_classification_boundary_is_inclusive_of_the_threshold():
    """`>= threshold` -> TRENDING (spec's fixed-threshold rule): setting the
    threshold to exactly the measured ADX must still classify TRENDING;
    nudging the threshold just above it must flip to RANGING."""
    highs, lows, closes = _trending_up_candles(60)
    _regime, adx = _latest(highs, lows, closes)

    at_threshold, _ = _latest(highs, lows, closes, adx_trend_threshold=adx)
    just_above, _ = _latest(highs, lows, closes, adx_trend_threshold=adx + 0.01)

    assert at_threshold == TrendRegime.TRENDING
    assert just_above == TrendRegime.RANGING


def test_insufficient_history_returns_ranging_not_raise():
    highs, lows, closes = _trending_up_candles(5)

    regimes = classify_trend_regime(highs, lows, closes, adx_period=14)
    assert len(regimes) == 5
    assert all(r == TrendRegime.RANGING for r in regimes)

    regime, adx = _latest(highs, lows, closes, adx_period=14)
    assert regime == TrendRegime.RANGING
    assert math.isnan(adx)


def test_empty_arrays_return_ranging_not_raise():
    empty = np.array([])

    regime, adx = _latest(empty, empty, empty)

    assert regime == TrendRegime.RANGING
    assert math.isnan(adx)


def test_latest_trend_regime_matches_last_element_of_vectorized_series():
    highs, lows, closes = _trending_up_candles(60)

    regimes = classify_trend_regime(highs, lows, closes)
    regime, adx = _latest(highs, lows, closes)

    assert regime == regimes.iloc[-1]
    assert not math.isnan(adx)


# ── trading session bucketing ───────────────────────────────────────────────


def _at(hour: int) -> datetime:
    return datetime(2026, 8, 7, hour, 0, tzinfo=UTC)


def test_session_boundaries_with_default_config():
    config = RegimeConfig()
    cases = [
        (6, TradingSession.ASIAN),  # wraps past midnight
        (7, TradingSession.LONDON),  # asian end exclusive, london start inclusive
        (11, TradingSession.LONDON),
        (12, TradingSession.OVERLAP),  # overlap start inclusive
        (15, TradingSession.OVERLAP),
        (16, TradingSession.NEW_YORK),  # overlap/london end exclusive, NY start inclusive
        (20, TradingSession.NEW_YORK),
        (21, TradingSession.OFF_SESSION),  # NY end exclusive, gap before Asian
        (22, TradingSession.ASIAN),  # asian start inclusive
        (23, TradingSession.ASIAN),
        (0, TradingSession.ASIAN),
        (3, TradingSession.ASIAN),
    ]
    for hour, expected in cases:
        assert session_for(_at(hour), config) == expected, f"hour={hour}"


def test_overlap_is_checked_before_london_and_new_york():
    """With a config where OVERLAP genuinely intersects both neighboring
    sessions (unlike the default, whose overlap sits inside London's tail
    only), OVERLAP still wins — proving the check order, not just a
    boundary that happens to land there anyway."""
    config = RegimeConfig(
        session_london_start_hour=7,
        session_london_end_hour=16,
        session_new_york_start_hour=15,
        session_new_york_end_hour=21,
        session_overlap_start_hour=15,
        session_overlap_end_hour=16,
    )
    assert session_for(_at(15), config) == TradingSession.OVERLAP


def test_asian_session_wraps_past_midnight():
    config = RegimeConfig()
    for hour in (22, 23, 0, 1, 6):
        assert session_for(_at(hour), config) == TradingSession.ASIAN, f"hour={hour}"
    assert session_for(_at(7), config) != TradingSession.ASIAN


# ── compute_entry_regime ─────────────────────────────────────────────────────


def test_compute_entry_regime_returns_none_for_missing_or_empty_frame():
    now = _at(13)
    kwargs = dict(now=now, regime_config=RegimeConfig())

    assert compute_entry_regime(None, **kwargs) is None
    assert compute_entry_regime(pd.DataFrame(columns=["high", "low", "close"]), **kwargs) is None


def test_compute_entry_regime_combines_volatility_trend_and_session():
    highs, lows, closes = _trending_up_candles(60)
    frame = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    now = _at(13)  # OVERLAP under the default config

    regime = compute_entry_regime(frame, now=now, regime_config=RegimeConfig())

    assert regime is not None
    assert regime.trend == TrendRegime.TRENDING
    assert regime.session == TradingSession.OVERLAP
    assert isinstance(regime.volatility, VolatilityRegime)
    assert not math.isnan(regime.adx)
    assert isinstance(regime.volatility_percentile, float)
