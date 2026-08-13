from __future__ import annotations

import numpy as np
import pytest

from src.shared.domain.indicators import atr


def test_atr_period_1_equals_true_range_each_bar() -> None:
    # With period=1, ATR degenerates to the true range of each bar itself.
    highs = np.array([10.0, 12.0, 11.0])
    lows = np.array([8.0, 9.0, 9.0])
    closes = np.array([9.0, 11.0, 10.0])

    result = atr(highs, lows, closes, period=1)

    # TR[0] = high-low = 2
    # TR[1] = max(12-9, |12-9|, |9-9|) = 3
    # TR[2] = max(11-9, |11-11|, |9-11|) = 2
    expected = [2.0, 3.0, 2.0]
    assert result.tolist() == pytest.approx(expected)


def test_atr_period_3_rolling_mean_of_true_range() -> None:
    highs = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
    lows = np.array([8.0, 9.0, 9.0, 10.0, 11.0])
    closes = np.array([9.0, 11.0, 10.0, 12.0, 13.0])

    result = atr(highs, lows, closes, period=3)

    # True range series: [2, 3, 2, 3, 3]
    # Rolling mean(period=3, min_periods=3):
    #   idx 0,1 -> NaN (not enough periods)
    #   idx 2 -> mean(2, 3, 2) = 2.3333...
    #   idx 3 -> mean(3, 2, 3) = 2.6666...
    #   idx 4 -> mean(2, 3, 3) = 2.6666...
    assert result.iloc[0] != result.iloc[0]  # NaN
    assert result.iloc[1] != result.iloc[1]  # NaN
    assert result.iloc[2] == pytest.approx(7.0 / 3.0)
    assert result.iloc[3] == pytest.approx(8.0 / 3.0)
    assert result.iloc[4] == pytest.approx(8.0 / 3.0)
