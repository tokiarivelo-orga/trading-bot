from datetime import UTC, datetime

import numpy as np
import pytest

from src.market_data.domain.enrichment import compute_atr_series, day_of_week_for


def test_compute_atr_series_matches_hand_computed_values():
    # True range per bar: bar0 = high-low = 1.0 (no prior close);
    # bar1 = max(11-9.5, |11-9.5|, |9.5-9.5|) = 1.5;
    # bar2 = max(12-10.5, |12-10.5|, |10.5-10.5|) = 1.5;
    # bar3 = max(11.5-10, |11.5-11.5|, |10-11.5|) = 1.5;
    # bar4 = max(13-11, |13-10.5|, |11-10.5|) = 2.5.
    # 3-period rolling mean (min_periods=3): first two NaN, then
    # (1+1.5+1.5)/3, (1.5+1.5+1.5)/3, (1.5+1.5+2.5)/3.
    highs = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
    lows = np.array([9.0, 9.5, 10.5, 10.0, 11.0])
    closes = np.array([9.5, 10.5, 11.5, 10.5, 12.5])

    result = compute_atr_series(highs, lows, closes, period=3)

    assert isinstance(result, np.ndarray)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[2:], [4.0 / 3, 4.5 / 3, 5.5 / 3])


def test_compute_atr_series_single_bar_has_no_prior_close():
    # With one bar, true range is just high - low.
    result = compute_atr_series(
        np.array([10.0]), np.array([9.0]), np.array([9.5]), period=1
    )
    np.testing.assert_allclose(result, [1.0])


@pytest.mark.parametrize(
    "date_args, expected",
    [
        ((2026, 8, 10), 0),  # Monday
        ((2026, 8, 11), 1),  # Tuesday
        ((2026, 8, 12), 2),  # Wednesday
        ((2026, 8, 13), 3),  # Thursday
        ((2026, 8, 14), 4),  # Friday
        ((2026, 8, 15), 5),  # Saturday
        ((2026, 8, 16), 6),  # Sunday
    ],
)
def test_day_of_week_for_matches_calendar(date_args, expected):
    dt = datetime(*date_args, tzinfo=UTC)
    assert day_of_week_for(dt) == expected
