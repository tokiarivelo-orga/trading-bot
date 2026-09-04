import math

import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.scalp_bollinger_reversion_v1_v2 import ScalpBollingerReversionV2

N = 43  # MIN_HISTORY: period(20) + width_lookback(20) + 3


def _ctx(opens, highs, lows, closes, volumes) -> MarketContext:
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": volumes}
    )
    return MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=25.0)


def _oscillating_touch_and_reversal(
    sign: int, base: float = 100.0, amp: float = 0.3, period_bars: int = 8
) -> MarketContext:
    """A steady sine-wave oscillation (not a one-off spike) keeps band width
    stable leading into the touch, since the touch itself is then a normal
    recurring feature of the series rather than a novel outlier -- a one-off
    spike-then-recover fixture can't clear v2's tightened width filter (a
    genuine touch mechanically elevates its own rolling std momentarily; see
    the strategy's module docstring). `sign=1` builds an upper-band
    touch+reversal (SELL setup), `sign=-1` a lower-band one (BUY setup)."""
    closes = [base + sign * amp * math.sin(2 * math.pi * i / period_bars) for i in range(N)]
    closes[-2] = base + sign * amp * 1.5  # clears the band
    closes[-1] = base  # pulls back to the mean
    opens = list(closes)
    highs = [c + 0.03 for c in closes]
    lows = [c - 0.03 for c in closes]
    if sign > 0:
        highs[-2] = closes[-2] + 0.05
    else:
        lows[-2] = closes[-2] - 0.05
    volumes = [100.0] * N
    return _ctx(opens, highs, lows, closes, volumes)


def test_sell_signal_on_upper_band_touch_with_strong_pullback():
    strategy = ScalpBollingerReversionV2()
    signal = strategy.evaluate(_oscillating_touch_and_reversal(sign=1))

    assert signal is not None
    assert signal.direction is Direction.SELL
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * strategy.spec.params["tp_rr"]


def test_buy_signal_on_lower_band_touch_with_strong_pullback():
    signal = ScalpBollingerReversionV2().evaluate(_oscillating_touch_and_reversal(sign=-1))

    assert signal is not None
    assert signal.direction is Direction.BUY


def test_no_signal_when_market_is_trending_not_ranging():
    closes = [100.0 + i * 0.5 for i in range(N)]
    opens = list(closes)
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signal = ScalpBollingerReversionV2().evaluate(_ctx(opens, highs, lows, closes, [100.0] * N))
    assert signal is None


def test_no_signal_with_insufficient_history():
    closes = [100.0] * 10
    opens = list(closes)
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signal = ScalpBollingerReversionV2().evaluate(_ctx(opens, highs, lows, closes, [100.0] * 10))
    assert signal is None


def test_spec_is_version_two_with_tightened_and_new_filters():
    spec = ScalpBollingerReversionV2().spec
    assert spec.version == 2
    assert spec.name == "scalp_bollinger_reversion_v1"
    assert spec.confirmation_timeframes == ()
    assert spec.params["width_range_threshold"] == 1.05
    assert spec.params["reversal_fraction"] == 0.25
    assert spec.params["min_sl_atr_mult"] == 0.5
