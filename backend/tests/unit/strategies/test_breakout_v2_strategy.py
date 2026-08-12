import pandas as pd

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.breakout_v2 import ATR_PERIOD, TREND_SMA_PERIOD, BreakoutV2

# M5 history long enough for the ATR(14) warmup plus the 20-bar range: a
# gently rising/falling series (so ATR settles to a stable, non-noise value)
# followed by a clear breakout bar.
_M5_BARS = max(ATR_PERIOD, 20) + 5


def _m5_df(base: float, drift: float, breakout: float) -> pd.DataFrame:
    closes = [base + drift * i for i in range(_M5_BARS)] + [breakout]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})


def _htf_df(closes: list[float]) -> pd.DataFrame:
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes})


def _uptrend_htf() -> pd.DataFrame:
    # Rising SMA with the last close above it.
    return _htf_df([100.0 + i * 0.5 for i in range(TREND_SMA_PERIOD)])


def _downtrend_htf() -> pd.DataFrame:
    return _htf_df([200.0 - i * 0.5 for i in range(TREND_SMA_PERIOD)])


def make_ctx(m5: pd.DataFrame, h1: pd.DataFrame | None, h4: pd.DataFrame | None) -> MarketContext:
    candles: dict[str, pd.DataFrame] = {"M5": m5}
    if h1 is not None:
        candles["H1"] = h1
    if h4 is not None:
        candles["H4"] = h4
    return MarketContext(symbol="XAUUSD", candles=candles, spread_points=25.0)


def test_buy_signal_when_breakout_aligned_with_uptrend():
    m5 = _m5_df(base=2400.0, drift=0.1, breakout=2420.0)
    ctx = make_ctx(m5, _uptrend_htf(), _uptrend_htf())
    strategy = BreakoutV2()
    signal = strategy.evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.BUY
    assert signal.sl_points > 0
    assert signal.tp_points == signal.sl_points * strategy.spec.params["tp_rr"]


def test_sell_signal_when_breakout_aligned_with_downtrend():
    m5 = _m5_df(base=2400.0, drift=-0.1, breakout=2380.0)
    ctx = make_ctx(m5, _downtrend_htf(), _downtrend_htf())
    signal = BreakoutV2().evaluate(ctx)

    assert signal is not None
    assert signal.direction is Direction.SELL


def test_no_signal_when_breakout_against_htf_trend():
    # Upward M5 breakout, but H1/H4 are in a downtrend — filtered out.
    m5 = _m5_df(base=2400.0, drift=0.1, breakout=2420.0)
    ctx = make_ctx(m5, _downtrend_htf(), _downtrend_htf())
    assert BreakoutV2().evaluate(ctx) is None


def test_no_signal_when_htf_disagree():
    m5 = _m5_df(base=2400.0, drift=0.1, breakout=2420.0)
    ctx = make_ctx(m5, _uptrend_htf(), _downtrend_htf())
    assert BreakoutV2().evaluate(ctx) is None


def test_no_signal_without_htf_history():
    m5 = _m5_df(base=2400.0, drift=0.1, breakout=2420.0)
    ctx = make_ctx(m5, None, None)
    assert BreakoutV2().evaluate(ctx) is None


def test_no_signal_inside_range():
    m5 = _m5_df(base=2400.0, drift=0.0, breakout=2400.0)
    ctx = make_ctx(m5, _uptrend_htf(), _uptrend_htf())
    assert BreakoutV2().evaluate(ctx) is None


def test_no_signal_with_insufficient_m5_history():
    closes = [2400.0] * 5
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
        }
    )
    ctx = make_ctx(df, _uptrend_htf(), _uptrend_htf())
    assert BreakoutV2().evaluate(ctx) is None


def test_spec_covers_base_symbols():
    spec = BreakoutV2().spec
    assert spec.symbols == ()  # empty = accepts all dynamically configured symbols
    assert spec.entry_timeframe == "M5"
    assert spec.version == 2
