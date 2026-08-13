"""Pure technical-indicator primitives shared across modules.

No I/O — pure functions over OHLC arrays. Framework-free (only
`numpy`/`pandas`), so both `engine/` and `market_data/` (and anything else)
may import from here without violating the `engine -> market_data`
dependency direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range_values(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    tr = highs - lows
    if len(tr) > 1:
        gap_high = np.abs(highs[1:] - closes[:-1])
        gap_low = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(tr[1:], np.maximum(gap_high, gap_low))
    return tr


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> pd.Series:
    tr = pd.Series(_true_range_values(highs, lows, closes))
    return tr.rolling(period, min_periods=period).mean()
