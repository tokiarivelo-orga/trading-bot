"""Unit tests for `ict_order_block_xauusd_v1.py` — ICT Order Block strategy
for XAUUSD (Derek Salvon's "ICT Order Block Trading").

Fixture layout (H1, 60 bars):
  idx 0-19   flat ATR warmup at 100.0
  idx 20     swing high at 102.0
  idx 21-24  decline toward the swing low
  idx 25     swing low / the Order Block candle itself (down-close, [94.0, 95.0])
  idx 26-29  green basing/lift candles
  idx 30     break candle: close 103.2 > 102.0 -> bullish MSS, break_idx=30
  idx 31-59  flat continuation at 103.5 (no new structure)

M5 (20 bars) retraces from ~103 down into the [94.0, 95.0] Order Block,
forming a bullish Fair Value Gap at idx 12 (gap [94.8, 94.95], inside the
OB +/- tolerance) before the final bar wicks into the zone and closes back
above it — the retest+rejection entry trigger. The last M5 bar's UTC hour
(07:xx) sits inside the London Kill Zone (07:00-10:00 UTC).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.strategies.domain.models import Direction, MarketContext, ZoneKind
from src.strategies.generated.ict_order_block_xauusd_v1 import (
    IctOrderBlockXauusd,
    _atr,
    _detect_fvgs,
    _detect_swing_points,
    _find_mss,
    _find_order_block,
)

H1_START = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
M5_START = datetime(2026, 1, 5, 6, 0, tzinfo=UTC)  # last M5 bar lands at 07:35 UTC
SPREAD_POINTS = 20.0

H1_OHLC: list[tuple[float, float, float, float]] = (
    [(100.0, 100.3, 99.7, 100.0)] * 20
    + [
        (100.0, 102.0, 99.9, 101.0),  # idx20 swing high
        (101.0, 101.2, 99.5, 100.0),  # idx21
        (100.0, 100.2, 97.0, 97.5),  # idx22
        (97.5, 98.0, 95.5, 96.0),  # idx23
        (96.0, 96.5, 94.2, 94.5),  # idx24
        (94.9, 95.0, 94.0, 94.2),  # idx25 swing low / Order Block (down-close)
        (94.2, 96.0, 94.1, 95.8),  # idx26 up
        (95.8, 98.0, 95.7, 97.8),  # idx27 up
        (97.8, 100.0, 97.7, 99.8),  # idx28 up
        (99.8, 101.5, 99.7, 101.3),  # idx29 up
        (101.3, 103.5, 101.2, 103.2),  # idx30 break candle (close > 102.0)
    ]
    + [(103.5, 103.8, 103.2, 103.5)] * 29
)

M5_OHLC: list[tuple[float, float, float, float]] = [
    (103.0, 103.2, 102.5, 102.6),
    (102.6, 102.7, 101.8, 101.9),
    (101.9, 102.0, 100.9, 101.0),
    (101.0, 101.1, 99.9, 100.0),
    (100.0, 100.1, 98.9, 99.0),
    (99.0, 99.1, 97.9, 98.0),
    (98.0, 98.1, 97.3, 97.4),
    (97.4, 97.5, 96.6, 96.7),
    (96.7, 96.8, 95.9, 96.0),
    (96.0, 96.1, 95.2, 95.3),
    (95.3, 95.4, 94.6, 94.7),
    (94.7, 94.8, 94.3, 94.4),  # idx11 FVG candle 1 (high=94.8)
    (94.4, 95.3, 94.2, 95.1),  # idx12 FVG displacement candle (up)
    (95.1, 95.6, 94.95, 95.5),  # idx13 FVG candle 3 (low=94.95 > 94.8 -> gap)
    (95.5, 95.6, 95.2, 95.3),
    (95.3, 95.4, 95.0, 95.1),
    (95.1, 95.2, 94.8, 94.9),
    (94.9, 95.0, 94.6, 94.7),
    (94.7, 94.8, 94.5, 94.6),
    (94.6, 94.9, 94.2, 94.8),  # idx19 rejection candle: wicks to 94.2, closes 94.8 > open
]


def _bars(
    ohlc: list[tuple[float, float, float, float]], start: datetime, step: timedelta
) -> list[dict]:
    return [
        {
            "time": start + i * step,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "tick_volume": 1000,
        }
        for i, (o, h, low, c) in enumerate(ohlc)
    ]


def _h1_frame(ohlc: list[tuple[float, float, float, float]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(_bars(ohlc or H1_OHLC, H1_START, timedelta(hours=1)))


def _m5_frame(
    ohlc: list[tuple[float, float, float, float]] | None = None, start: datetime = M5_START
) -> pd.DataFrame:
    return pd.DataFrame(_bars(ohlc or M5_OHLC, start, timedelta(minutes=5)))


def _ctx(h1: pd.DataFrame, m5: pd.DataFrame, h4: pd.DataFrame | None = None) -> MarketContext:
    candles: dict[str, pd.DataFrame] = {"H1": h1, "M5": m5}
    if h4 is not None:
        candles["H4"] = h4
    return MarketContext(symbol="XAUUSD", candles=candles, spread_points=SPREAD_POINTS)


def _strategy() -> IctOrderBlockXauusd:
    return IctOrderBlockXauusd()


def test_spec_shape() -> None:
    strategy = _strategy()
    assert strategy.spec.name == "ict_order_block_xauusd"
    assert strategy.spec.symbols == ("XAUUSD",)
    assert strategy.spec.entry_timeframe == "M5"
    assert strategy.spec.confirmation_timeframes == ("H1", "H4")
    assert strategy.spec.htf_veto is False
    assert strategy.spec.close_on_opposite_signal is True
    params = strategy.spec.params
    assert params["tp1_target_rr"] < params["tp2_target_rr"] < params["tp3_target_rr"]
    assert params["tp1_target_rr"] > 1.5  # clears XAUUSD's configured min_rr with headroom


def test_no_signal_on_short_history() -> None:
    short_h1 = _h1_frame(H1_OHLC[:20])
    assert _strategy().evaluate(_ctx(short_h1, _m5_frame())) is None


def test_buy_signal_on_full_ict_setup() -> None:
    result = _strategy().evaluate(_ctx(_h1_frame(), _m5_frame()))

    assert result is not None
    assert len(result) == 3
    sig1, sig2, sig3 = result
    for sig in result:
        assert sig.direction is Direction.BUY
        assert sig.zone is not None
        assert sig.zone.kind is ZoneKind.DEMAND
        assert sig.zone.pattern == "BuOB"
        assert sig.zone.price_low == pytest.approx(94.0)
        assert sig.zone.price_high == pytest.approx(95.0)
        assert sig.sl_points > 0
        assert "kz_hour=7" in sig.reason
        assert "FVG[" in sig.reason

    assert sig1.sl_points == pytest.approx(sig2.sl_points)
    assert sig2.sl_points == pytest.approx(sig3.sl_points)
    assert sig1.tp_points < sig2.tp_points < sig3.tp_points
    assert "(Scalp)" in sig1.reason
    assert "(Zone)" in sig2.reason
    assert "(Runner)" in sig3.reason

    # tp1 clears XAUUSD's min_rr=1.5 spread-adjusted floor with headroom.
    risk = sig1.sl_points + SPREAD_POINTS * strategy_params()["point_value"]
    assert sig1.tp_points >= 1.5 * risk


def strategy_params() -> dict:
    return IctOrderBlockXauusd().spec.params


def test_no_signal_outside_kill_zone() -> None:
    off_hours_start = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)  # last bar lands ~01:35 UTC (Asia)
    m5 = _m5_frame(start=off_hours_start)
    assert _strategy().evaluate(_ctx(_h1_frame(), m5)) is None


def test_no_signal_without_fvg_confirmation() -> None:
    ohlc = list(M5_OHLC)
    o, h, low, c = ohlc[13]
    ohlc[13] = (o, h, 94.7, c)  # candle 3's low now overlaps candle 1's high -> no gap
    assert _strategy().evaluate(_ctx(_h1_frame(), _m5_frame(ohlc))) is None


def test_no_signal_when_order_block_invalidated() -> None:
    ohlc = list(H1_OHLC)
    ohlc[40] = (90.0, 90.3, 89.7, 90.0)  # a later H1 close well below ob_low -> invalidated
    assert _strategy().evaluate(_ctx(_h1_frame(ohlc), _m5_frame())) is None


def test_no_signal_against_opposing_h4_bias() -> None:
    declining = [
        (100.0 - i * 0.5, 100.3 - i * 0.5, 99.7 - i * 0.5, 100.0 - i * 0.5) for i in range(25)
    ]
    h4 = pd.DataFrame(_bars(declining, H1_START, timedelta(hours=4)))
    assert _strategy().evaluate(_ctx(_h1_frame(), _m5_frame(), h4=h4)) is None


def test_no_signal_on_wrong_direction_wick() -> None:
    ohlc = list(M5_OHLC)
    # Replace the rejection candle with a bearish one that closes below its open.
    ohlc[19] = (94.9, 95.0, 94.2, 94.3)
    assert _strategy().evaluate(_ctx(_h1_frame(), _m5_frame(ohlc))) is None


def test_find_mss_detects_bullish_break() -> None:
    highs = _h1_frame()["high"].to_numpy()
    lows = _h1_frame()["low"].to_numpy()
    closes = _h1_frame()["close"].to_numpy()
    swings = _detect_swing_points(highs, lows, lookback=5)
    mss = _find_mss(closes, swings, max_bars_since_break=40)
    assert mss is not None
    assert mss["direction"] == "buy"
    assert mss["pivot_price"] == pytest.approx(94.0)
    assert mss["structure_price"] == pytest.approx(102.0)
    assert mss["break_idx"] == 30


def test_find_order_block_picks_last_down_close_candle() -> None:
    df = _h1_frame()
    ob = _find_order_block(
        df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
        leg_start_idx=25, break_idx=30, bullish=True,
    )
    assert ob is not None
    assert ob["idx"] == 25
    assert ob["price_low"] == pytest.approx(94.0)
    assert ob["price_high"] == pytest.approx(95.0)


def test_detect_fvgs_finds_bullish_gap_near_ob() -> None:
    df = _m5_frame()
    gaps = _detect_fvgs(
        df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(),
        start=7, end=19, bullish=True,
    )
    assert any(
        idx == 12 and low == pytest.approx(94.8) and high == pytest.approx(94.95)
        for idx, low, high in gaps
    )


def test_atr_warms_up_after_period() -> None:
    df = _h1_frame()
    atr_series = _atr(df, 14)
    assert atr_series.iloc[:13].isna().all()
    assert atr_series.iloc[13:].notna().all()
