"""Per-candle enrichment: trailing ATR and day-of-week.

Pure domain — numpy plus stdlib `datetime` only (ATR itself is computed by
`shared.domain.indicators.atr`, this module doesn't reimplement it), no I/O.
See `market_data/adapters/candle_repository.py::enrich_missing` for how these
get applied to stored rows: enrichment needs trailing history a single OHLC
upsert batch doesn't carry, so it's a separate write pass, not part of the
OHLC upsert path (`upsert_many` always writes `atr_14`/`day_of_week` as NULL —
see its docstring).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.shared.domain.indicators import atr


def compute_atr_series(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """`period`-bar ATR over `highs`/`lows`/`closes`, aligned index-for-index
    with the inputs. Thin wrapper over `shared.domain.indicators.atr` that
    hands back a plain array rather than a pandas `Series` — callers here
    index by array position, and a pandas index that doesn't correspond to
    anything meaningful outside this one call would just invite bugs."""
    return atr(highs, lows, closes, period).to_numpy()


def day_of_week_for(dt: datetime) -> int:
    """0=Monday..6=Sunday, matching `datetime.weekday()`. `dt` must already be
    UTC-aware — this codebase is UTC-only throughout (see `market_data/
    domain/models.py`'s module docstring); `Candle.time`/`close_time` are
    already built via `datetime.fromtimestamp(..., tz=UTC)`, so callers
    should pass those straight through rather than re-deriving from epoch
    seconds."""
    return dt.weekday()
