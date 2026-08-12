from datetime import datetime, timedelta

import pandas as pd

from src.strategies.domain.models import Direction, MarketContext, StructureLabel
from src.strategies.generated.mean_reversion_v1 import PIVOT_WING, MeanReversionV1

# Zigzag control points (bar index -> price), linearly interpolated between
# them so every leg is long enough (> 2*PIVOT_WING) to produce an unambiguous
# fractal pivot exactly at each control point:
#   0 -> 100 -> 70 (low1=70 @20) -> 90 (high1=90 @40) -> 80 (low2=80 @60,
#   higher than low1) -> 110 (high2=110 @80, higher than high1: HH -> fade
#   -> SELL) -> 60 (low3=60 @100, lower than low2: LL -> fade -> BUY)
#   -> 80 (tail @120)
CONTROL_POINTS = [
    (0, 100.0),
    (20, 70.0),
    (40, 90.0),
    (60, 80.0),
    (80, 110.0),
    (100, 60.0),
    (120, 80.0),
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


def test_sell_signal_fades_higher_high():
    # high2 (index 80, price 110) beats high1 (index 40, price 90) -> HH,
    # which this strategy fades with a SELL rather than following with a BUY.
    ctx = _make_ctx(_confirm_length(80))
    signal = MeanReversionV1().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.SELL
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * MeanReversionV1().spec.params["tp_rr"]
    assert signal.structure[0].label is StructureLabel.HH


def test_buy_signal_fades_lower_low():
    # low3 (index 100, price 60) beats low2 (index 60, price 80) -> LL,
    # which this strategy fades with a BUY rather than following with a SELL.
    ctx = _make_ctx(_confirm_length(100))
    signal = MeanReversionV1().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.structure[0].label is StructureLabel.LL


def test_no_signal_on_higher_low():
    # low2 (index 60, price 80) is higher than low1 (index 20, price 70):
    # a higher low, not a lower low, so no fade signal fires.
    ctx = _make_ctx(_confirm_length(60))
    assert MeanReversionV1().evaluate(ctx) is None


def test_no_signal_before_pivot_confirms():
    ctx = _make_ctx(_confirm_length(80) - 1)
    assert MeanReversionV1().evaluate(ctx) is None


def test_no_signal_with_insufficient_history():
    ctx = _make_ctx(10)
    assert MeanReversionV1().evaluate(ctx) is None


def test_spec_covers_base_symbols():
    spec = MeanReversionV1().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
