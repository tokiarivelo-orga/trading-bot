import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.breakout_v1 import BreakoutV1


def make_ctx(closes: list[float]) -> MarketContext:
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})
    return MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=25.0)


def test_buy_signal_on_breakout_above_range():
    closes = [2400.0] * 20 + [2410.0]  # last bar breaks above the 20-bar high (2400.5)
    strategy = BreakoutV1()
    signal = strategy.evaluate(make_ctx(closes))

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * strategy.spec.params["tp_rr"]


def test_sell_signal_on_breakout_below_range():
    closes = [2400.0] * 20 + [2390.0]  # last bar breaks below the 20-bar low (2399.5)
    signal = BreakoutV1().evaluate(make_ctx(closes))

    assert signal is not None
    assert signal.direction is Direction.SELL


def test_no_signal_inside_range():
    closes = [2400.0] * 21
    assert BreakoutV1().evaluate(make_ctx(closes)) is None


def test_no_signal_with_insufficient_history():
    closes = [2400.0] * 5
    assert BreakoutV1().evaluate(make_ctx(closes)) is None


def test_spec_covers_base_symbols():
    spec = BreakoutV1().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
