"""Volatility-regime classification, used only to tag signals and trades.

Ranks the current ATR reading against its own trailing history (percentile
rank) to bucket the market into a `VolatilityRegime`. `engine.domain.regime`
attaches that bucket to every `SignalDecision`/`TradeRecord` for analytics;
nothing gates, sizes, or exits a trade on it.

No I/O — pure functions over OHLC/ATR arrays, matching this module's
hexagonal `domain/` placement. Thresholds come from `configs/regime.yaml`
(`volatility_*` keys, see `RegimeConfig`).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from src.engine.domain.zone_detection import atr

DEFAULT_ATR_PERIOD = 14
DEFAULT_REGIME_LOOKBACK_BARS = 100
DEFAULT_LOW_PERCENTILE = 20.0
DEFAULT_HIGH_PERCENTILE = 70.0
DEFAULT_EXTREME_PERCENTILE = 90.0


class VolatilityRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


def _percentile_rank(window: np.ndarray, value: float) -> float:
    """Percentile rank of `value` against `window` (0-100) using the
    tie-aware "average rank" convention: ties count as half a step rather
    than a full step. This matters a lot here — a naive "fraction at or
    below `value`" definition pins a perfectly flat/calm window (a common,
    unremarkable market state) to the 100th percentile purely because every
    trailing reading ties the current one, which would misclassify ordinary
    calm markets as EXTREME. Averaging ties instead puts a flat window at
    ~50 (NORMAL), which is the sane answer."""
    if window.size == 0:
        return np.nan
    below = float(np.sum(window < value))
    tied = float(np.sum(window == value))
    return (below + 0.5 * tied) / window.size * 100.0


def _classify(
    percentile: float,
    *,
    low_percentile: float,
    high_percentile: float,
    extreme_percentile: float,
) -> VolatilityRegime:
    if np.isnan(percentile):
        return VolatilityRegime.NORMAL
    if percentile > extreme_percentile:
        return VolatilityRegime.EXTREME
    if percentile > high_percentile:
        return VolatilityRegime.HIGH
    if percentile < low_percentile:
        return VolatilityRegime.LOW
    return VolatilityRegime.NORMAL


def classify_volatility_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    atr_period: int = DEFAULT_ATR_PERIOD,
    regime_lookback_bars: int = DEFAULT_REGIME_LOOKBACK_BARS,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE,
) -> pd.Series:
    """Vectorized regime classification aligned to the OHLC arrays, for
    backtest use. For each bar, ranks that bar's ATR against the trailing
    `regime_lookback_bars` ATR readings (excluding the current bar) and
    buckets the percentile into LOW/NORMAL/HIGH/EXTREME. Bars without enough
    ATR or lookback history (warm-up window) default to NORMAL rather than
    raising, matching the "insufficient history" guard used by
    `zone_detection.detect_bases`."""
    atr_values = atr(highs, lows, closes, atr_period)
    n = len(atr_values)
    regimes = [VolatilityRegime.NORMAL] * n
    atr_arr = atr_values.to_numpy()

    for i in range(n):
        current = atr_arr[i]
        if np.isnan(current):
            continue
        window_start = max(0, i - regime_lookback_bars)
        window = atr_arr[window_start:i]
        window = window[~np.isnan(window)]
        if window.size == 0:
            continue
        percentile = _percentile_rank(window, current)
        regimes[i] = _classify(
            percentile,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            extreme_percentile=extreme_percentile,
        )

    return pd.Series(regimes, index=atr_values.index)


def latest_volatility_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    atr_period: int = DEFAULT_ATR_PERIOD,
    regime_lookback_bars: int = DEFAULT_REGIME_LOOKBACK_BARS,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE,
) -> tuple[VolatilityRegime, float, float]:
    """Convenience "latest regime" read for live entry tagging: classifies only the most recent bar
    and also returns its percentile rank and raw ATR value. Returns
    `(VolatilityRegime.NORMAL, nan, nan)` when there isn't enough history yet
    instead of raising."""
    atr_values = atr(highs, lows, closes, atr_period)
    valid_atr = atr_values.dropna()
    if valid_atr.empty:
        return VolatilityRegime.NORMAL, float("nan"), float("nan")

    atr_arr = atr_values.to_numpy()
    current = float(atr_arr[-1])
    if np.isnan(current):
        return VolatilityRegime.NORMAL, float("nan"), float("nan")

    window_start = max(0, len(atr_arr) - 1 - regime_lookback_bars)
    window = atr_arr[window_start : len(atr_arr) - 1]
    window = window[~np.isnan(window)]
    if window.size == 0:
        return VolatilityRegime.NORMAL, float("nan"), current

    percentile = _percentile_rank(window, current)
    regime = _classify(
        percentile,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        extreme_percentile=extreme_percentile,
    )
    return regime, percentile, current
