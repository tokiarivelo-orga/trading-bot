from datetime import datetime, timedelta

import pandas as pd

from src.strategies.domain.models import Direction, MarketContext, StructureLabel
from src.strategies.generated.trend_structure_v1 import PIVOT_WING, TrendStructureV1

# Zigzag control points (bar index -> price), linearly interpolated between
# them so every leg is long enough (> 2*PIVOT_WING) to produce an unambiguous
# fractal pivot exactly at each control point:
#   0 -> 100 -> 90 (low1=90 @7) -> 96 (high1=96 @14) -> 92 (low2=92 @21,
#   higher than low1: HL, not LL) -> 100 (high2=100 @28, higher than high1: HH)
#   -> 85 (low3=85 @35, lower than low2: LL) -> 90 (tail @42, gives PIVOT_WING
#   bars of confirmation room after low3)
CONTROL_POINTS = [
    (0, 100.0),
    (7, 90.0),
    (14, 96.0),
    (21, 92.0),
    (28, 100.0),
    (35, 85.0),
    (42, 90.0),
]


def _make_path() -> list[float]:
    path: list[float] = []
    for (i0, p0), (i1, p1) in zip(CONTROL_POINTS, CONTROL_POINTS[1:], strict=False):
        steps = i1 - i0
        for step in range(steps):
            path.append(p0 + (p1 - p0) * step / steps)
    path.append(CONTROL_POINTS[-1][1])
    return path


def _make_ctx(bars: int) -> MarketContext:
    path = _make_path()
    prices = path[:bars]
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


def test_buy_signal_on_higher_high():
    # high2 (index 28, price 100) beats high1 (index 14, price 96) -> HH.
    ctx = _make_ctx(_confirm_length(28))
    signal = TrendStructureV1().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * TrendStructureV1().spec.params["tp_rr"]
    assert len(signal.structure) == 1
    assert signal.structure[0].label is StructureLabel.HH


def test_sell_signal_on_lower_low():
    # low3 (index 35, price 85) beats low2 (index 21, price 92) -> LL.
    ctx = _make_ctx(_confirm_length(35))
    signal = TrendStructureV1().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.SELL
    assert signal.structure[0].label is StructureLabel.LL


def test_no_signal_on_higher_low():
    # low2 (index 21, price 92) is higher than low1 (index 7, price 90):
    # a higher low, not a lower low, so no continuation signal fires.
    ctx = _make_ctx(_confirm_length(21))
    assert TrendStructureV1().evaluate(ctx) is None


def test_no_signal_before_pivot_confirms():
    # One bar short of the HH's confirmation bar.
    ctx = _make_ctx(_confirm_length(28) - 1)
    assert TrendStructureV1().evaluate(ctx) is None


def test_no_signal_with_insufficient_history():
    ctx = _make_ctx(10)
    assert TrendStructureV1().evaluate(ctx) is None


def test_spec_covers_base_symbols():
    spec = TrendStructureV1().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
