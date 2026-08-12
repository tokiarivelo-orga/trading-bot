import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.scalp_ema_cross_v1_v1 import ScalpEmaCrossV1


def make_ctx(closes: list[float]) -> MarketContext:
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [100.0] * len(closes),
        }
    )
    return MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=25.0)


def test_buy_signal_on_bullish_ema_cross():
    # Flat for exactly MIN_HISTORY-1 bars, then a single ramp bar: EMA5
    # (fast) crosses above EMA13 (slow) precisely on this last bar.
    closes = [2400.0] * 41 + [2403.0]
    strategy = ScalpEmaCrossV1()
    signal = strategy.evaluate(make_ctx(closes))

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.sl_points > 0
    spread_distance = 25.0 * 0.01  # ctx.spread_points * XAUUSD point size
    assert signal.tp_points == (signal.sl_points + spread_distance) * strategy.spec.params["tp_rr"]


def test_sell_signal_on_bearish_ema_cross():
    closes = [2400.0] * 41 + [2397.0]
    signal = ScalpEmaCrossV1().evaluate(make_ctx(closes))

    assert signal is not None
    assert signal.direction is Direction.SELL


def test_no_signal_when_emas_already_aligned_no_fresh_cross():
    # Long, smooth ramp: by the last bar EMA5 has been above EMA13 for many
    # bars already, so there's no fresh crossover on this exact bar.
    closes = [2400.0 + i * 1.0 for i in range(50)]
    assert ScalpEmaCrossV1().evaluate(make_ctx(closes)) is None


def test_no_signal_with_insufficient_history():
    closes = [2400.0] * 10
    assert ScalpEmaCrossV1().evaluate(make_ctx(closes)) is None


def test_spec_covers_base_symbols_and_m5_entry():
    spec = ScalpEmaCrossV1().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
    assert spec.confirmation_timeframes == ("M1", "H1")
    assert spec.params["ema_fast"] == 5
    assert spec.params["ema_slow"] == 13
