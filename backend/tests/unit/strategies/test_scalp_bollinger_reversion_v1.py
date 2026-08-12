import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.scalp_bollinger_reversion_v1_v1 import ScalpBollingerReversionV1

N = 43  # MIN_HISTORY: period(20) + width_lookback(20) + 3
BASE = 100.0
NOISE = 0.2  # small, stable alternating noise keeps band width contracted
SPREAD_DISTANCE = 25.0 * 0.01  # ctx.spread_points * XAUUSD point size


def _ctx(opens, highs, lows, closes, volumes) -> MarketContext:
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": volumes}
    )
    return MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=25.0)


def test_sell_signal_on_upper_band_touch_and_reversal():
    closes = [BASE + (NOISE if i % 2 == 0 else -NOISE) for i in range(N - 2)]
    closes.append(BASE + 0.4)  # touches/pierces the upper band
    closes.append(BASE + 0.1)  # closes back inside -> reversal confirmed
    opens = list(closes)
    highs = [c + 0.05 for c in closes]
    highs[-2] = closes[-2] + 0.1  # the touch bar's wick clears the band
    lows = [c - 0.05 for c in closes]
    volumes = [100.0] * N

    strategy = ScalpBollingerReversionV1()
    signal = strategy.evaluate(_ctx(opens, highs, lows, closes, volumes))

    assert signal is not None
    assert signal.direction is Direction.SELL
    assert signal.sl_points > 0
    assert signal.tp_points == (signal.sl_points + SPREAD_DISTANCE) * strategy.spec.params["tp_rr"]


def test_buy_signal_on_lower_band_touch_and_reversal():
    closes = [BASE + (NOISE if i % 2 == 0 else -NOISE) for i in range(N - 2)]
    closes.append(BASE - 0.4)
    closes.append(BASE - 0.1)
    opens = list(closes)
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    lows[-2] = closes[-2] - 0.1

    signal = ScalpBollingerReversionV1().evaluate(
        _ctx(opens, highs, lows, closes, [100.0] * N)
    )

    assert signal is not None
    assert signal.direction is Direction.BUY


def test_no_signal_when_market_is_trending_not_ranging():
    # Steady monotonic ramp -- band width keeps expanding, so the range
    # filter (width <= 1.1x its own recent average) blocks every bar.
    closes = [BASE + i * 0.5 for i in range(N)]
    opens = list(closes)
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signal = ScalpBollingerReversionV1().evaluate(
        _ctx(opens, highs, lows, closes, [100.0] * N)
    )
    assert signal is None


def test_no_signal_with_insufficient_history():
    closes = [BASE] * 10
    opens = list(closes)
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signal = ScalpBollingerReversionV1().evaluate(
        _ctx(opens, highs, lows, closes, [100.0] * 10)
    )
    assert signal is None


def test_spec_covers_base_symbols_and_carries_no_confirmation_timeframes():
    spec = ScalpBollingerReversionV1().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
    assert spec.confirmation_timeframes == ()
