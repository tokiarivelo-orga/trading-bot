from datetime import datetime, timedelta

import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.trend_structure_v2 import PIVOT_WING, TrendStructureV2

# Zigzag control points (bar index -> price): 5 alternating swings, spaced
# 15 bars apart (> 2*PIVOT_WING per leg, so each pivot is an unambiguous
# fractal) with enough lead room for MIN_HISTORY (60) and ATR(14) warmup.
#   lead-in -> s1 high @15 (90) -> s2 low @30 (80) -> s3 high @45 (96)
#   -> s4 low @60 (85, a genuine higher low vs s2) -> s5 high @75 (105, HH
#   vs s3, amplitude 9 well above 0.5xATR) -> tail
ALIGNED_BUY = [
    (0, 85.0),
    (15, 90.0),
    (30, 80.0),
    (45, 96.0),
    (60, 85.0),
    (75, 105.0),
    (90, 95.0),
]

# Same shape, but s4 (60) is a LOWER low than s2 (30): the HH at 75 is real,
# but it isn't preceded by a genuine higher low, so v2 should reject it.
UNALIGNED_BUY = [
    (0, 85.0),
    (15, 90.0),
    (30, 80.0),
    (45, 96.0),
    (60, 75.0),
    (75, 105.0),
    (90, 95.0),
]

# Same shape, alignment holds, but s5 barely beats s3 (96.3 vs 96): with a
# per-bar range of 2 (high/low offsets of +/-1), ATR(14) settles near 2, so
# a 0.3 amplitude is well below the 0.5xATR (~1.0) floor — v2 should reject
# it as noise.
SMALL_AMPLITUDE_BUY = [
    (0, 85.0),
    (15, 90.0),
    (30, 80.0),
    (45, 96.0),
    (60, 85.0),
    (75, 96.3),
    (90, 95.0),
]

# Mirror image for the short side: s1 low, s2 high, s3 low, s4 high (a
# genuine lower high vs s2), s5 low (LL vs s3, amplitude well above floor).
ALIGNED_SELL = [
    (0, 115.0),
    (15, 110.0),
    (30, 120.0),
    (45, 104.0),
    (60, 115.0),
    (75, 95.0),
    (90, 105.0),
]


def _make_path(control_points: list[tuple[int, float]]) -> list[float]:
    path: list[float] = []
    for (i0, p0), (i1, p1) in zip(control_points, control_points[1:], strict=False):
        steps = i1 - i0
        for step in range(steps):
            path.append(p0 + (p1 - p0) * step / steps)
    path.append(control_points[-1][1])
    return path


def _make_ctx(control_points: list[tuple[int, float]], bars: int) -> MarketContext:
    prices = _make_path(control_points)[:bars]
    start = datetime(2026, 1, 1)
    df = pd.DataFrame(
        {
            "time": [start + timedelta(minutes=5 * i) for i in range(len(prices))],
            "open": prices,
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "tick_volume": [100] * len(prices),
        }
    )
    return MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=25.0)


def _confirm_length(pivot_index: int) -> int:
    return pivot_index + PIVOT_WING + 1


def test_buy_signal_when_aligned_and_above_amplitude_floor():
    ctx = _make_ctx(ALIGNED_BUY, _confirm_length(75))
    signal = TrendStructureV2().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * TrendStructureV2().spec.params["tp_rr"]


def test_no_signal_when_prior_low_is_not_higher():
    ctx = _make_ctx(UNALIGNED_BUY, _confirm_length(75))
    assert TrendStructureV2().evaluate(ctx) is None


def test_no_signal_when_swing_amplitude_below_atr_floor():
    ctx = _make_ctx(SMALL_AMPLITUDE_BUY, _confirm_length(75))
    assert TrendStructureV2().evaluate(ctx) is None


def test_sell_signal_when_aligned_and_above_amplitude_floor():
    ctx = _make_ctx(ALIGNED_SELL, _confirm_length(75))
    signal = TrendStructureV2().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.SELL


def test_no_signal_with_insufficient_history():
    ctx = _make_ctx(ALIGNED_BUY, 20)
    assert TrendStructureV2().evaluate(ctx) is None


def test_spec_covers_base_symbols():
    spec = TrendStructureV2().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
    assert spec.version == 2
