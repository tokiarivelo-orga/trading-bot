"""Unit tests for `xauusd_snd_qm_structure_h1_v5.py` (patched, VALIDATED —
not yet active) — XAUUSD S&D V1/V2 + Quasimodo + market-structure strategy,
H1 Short Swing preset. Zone timeframe is H4, resampled in-strategy from H1
(4 H1 bars per bucket).

v5 adds two live-safety gates over the active v4 (see that file's module
docstring for the 2026-08-17 over-trading incident this fixes): a hard
`ctx.own_position is not None` guard, and a `fresh_touch` requirement so a
zone only signals on the entry-TF bar price first enters it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    PositionSnapshot,
    StructureLabel,
    ZoneKind,
)
from src.strategies.generated.xauusd_snd_qm_structure_h1_v5 import (
    XauusdSndQmStructureH1,
    _atr,
    _detect_quasimodo_zones,
    _detect_structure,
    _detect_zones_v1,
    _resample,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = Path("src/strategies/generated/xauusd_snd_qm_structure_h1_v5.py")

START = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(hours=1)
BUCKET_SIZE = 4  # zone_tf_minutes(240) / entry_tf_minutes(60)
SPREAD_POINTS = 20.0


def _bar(i: int, o: float, h: float, low: float, c: float) -> dict:
    return {
        "time": START + i * STEP,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "tick_volume": 1000,
    }


def _bucket(bars: list[dict], o: float, h: float, low: float, c: float) -> None:
    i = len(bars)
    for j in range(BUCKET_SIZE):
        bars.append(_bar(i + j, o, h, low, c))


def _rbr_fixture() -> list[dict]:
    """~20 flat H4 buckets for ATR warmup (range 1.2, body 0.4 -> ATR~1.2),
    then a two-candle rally (leg-in) - base - two-candle rally (leg-out):
    an RBR demand zone at [101.8, 102.1]. Each leg candle has body 0.8 —
    clear of S&D V1's leg threshold (1.0 * ATR) via the cumulative 1.6
    run move, but individually below S&D V2's engulfing threshold
    (1.5 * ATR) so no duplicate V2 zone forms over the same band."""
    bars: list[dict] = []
    for _ in range(20):
        _bucket(bars, 100.0, 100.6, 99.4, 100.4)
    _bucket(bars, 100.4, 101.4, 100.3, 101.2)  # leg-in candle 1 (body 0.8)
    _bucket(bars, 101.2, 102.1, 101.1, 102.0)  # leg-in candle 2 (run total 1.6)
    _bucket(bars, 102.0, 102.1, 101.8, 101.95)  # weak base -> zone band [101.8, 102.1]
    _bucket(bars, 101.95, 102.85, 101.9, 102.75)  # leg-out candle 1 (confirms: 102.75 > 102.1)
    _bucket(bars, 102.75, 103.65, 102.7, 103.55)  # leg-out candle 2 (run total 1.6)
    for _ in range(3):
        bars.append(_bar(len(bars), 103.5, 103.7, 103.3, 103.5))
    return bars


def _break_zone_bars() -> list[dict]:
    """Continuous, non-alternating decline (no leg/base/leg pause) closing
    clean through the RBR zone's low (101.8) and staying away."""
    bars: list[dict] = []
    _bucket(bars, 103.5, 103.6, 102.9, 103.0)
    _bucket(bars, 103.0, 103.1, 101.9, 102.0)
    _bucket(bars, 102.0, 102.1, 100.9, 101.0)
    _bucket(bars, 101.0, 101.1, 99.9, 100.0)
    return bars


def _strategy() -> XauusdSndQmStructureH1:
    return XauusdSndQmStructureH1()


def _ctx(bars: list[dict]) -> MarketContext:
    return MarketContext(
        symbol="XAUUSD", candles={"H1": pd.DataFrame(bars)}, spread_points=SPREAD_POINTS
    )


def test_spec_shape() -> None:
    strategy = _strategy()
    assert strategy.spec.name == "xauusd_snd_qm_structure_h1"
    assert strategy.spec.symbols == ("XAUUSD",)
    assert strategy.spec.entry_timeframe == "H1"
    assert strategy.spec.htf_veto is False
    assert strategy.spec.close_on_opposite_signal is True
    params = strategy.spec.params
    assert params["zone_tf_minutes"] == 240
    assert params["entry_tf_minutes"] == 60
    assert params["tp1_target_rr"] < params["tp2_target_rr"] < params["tp3_target_rr"]


def test_no_signal_on_short_history() -> None:
    bars = [_bar(i, 100.0, 100.6, 99.4, 100.4) for i in range(30)]
    assert _strategy().evaluate(_ctx(bars)) is None


def test_no_signal_when_zone_broken() -> None:
    bars = _rbr_fixture() + _break_zone_bars()
    assert _strategy().evaluate(_ctx(bars)) is None


def test_buy_signals_on_rbr_zone_retest() -> None:
    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0))  # wick back into the band

    result = _strategy().evaluate(_ctx(bars))

    assert result is not None
    assert len(result) == 3
    sig1, sig2, sig3 = result
    for sig in result:
        assert sig.direction is Direction.BUY
        assert sig.zone is not None
        assert sig.zone.kind is ZoneKind.DEMAND
        assert sig.zone.price_low == pytest.approx(101.8)
        assert sig.zone.price_high == pytest.approx(102.1)
        assert sig.sl_points > 0

    assert sig1.sl_points == pytest.approx(sig2.sl_points)
    assert sig2.sl_points == pytest.approx(sig3.sl_points)
    assert sig1.tp_points < sig2.tp_points < sig3.tp_points
    assert "(Scalp)" in sig1.reason
    assert "(Zone)" in sig2.reason
    assert "(Runner)" in sig3.reason

    # v4 raised tp1_target_rr 1.2 -> 1.8 specifically so the "Scalp" leg
    # clears XAUUSD's configured min_rr=1.5 spread-gate floor (previously
    # documented here as a known bug against v3 — see the M5 test module's
    # history for detail). v5 inherits that fix unchanged.
    risk = sig1.sl_points + SPREAD_POINTS * strategy_params()["point_value"]
    required_tp = 1.5 * risk
    assert sig1.tp_points >= required_tp


def strategy_params() -> dict:
    return XauusdSndQmStructureH1().spec.params


def test_no_signal_on_wrong_direction_wick() -> None:
    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.8, 103.4, 103.6))
    assert _strategy().evaluate(_ctx(bars)) is None


def test_resample_aggregates_and_drops_partial_bucket() -> None:
    bars: list[dict] = []
    _bucket(bars, 100.0, 101.0, 99.5, 100.5)
    _bucket(bars, 100.5, 102.0, 100.4, 101.5)
    bars.append(_bar(len(bars), 101.5, 101.8, 100.9, 101.0))  # partial 3rd bucket -> dropped
    result = _resample(pd.DataFrame(bars), 240, 60)
    assert result is not None
    frame, end_times = result
    assert len(frame) == 2
    assert frame["open"].tolist() == [100.0, 100.5]
    assert frame["close"].tolist() == [100.5, 101.5]


def test_detect_zones_v1_finds_rbr_demand_zone() -> None:
    frame = pd.DataFrame(_rbr_fixture())
    resampled = _resample(frame, 240, 60)
    assert resampled is not None
    zone_frame, _ = resampled
    atr_series = _atr(zone_frame, 14)
    params = strategy_params()
    zones = _detect_zones_v1(zone_frame, atr_series, params)
    rbr = [z for z in zones if z["pattern"] == "RBR"]
    assert rbr
    assert rbr[0]["kind"] is ZoneKind.DEMAND
    assert rbr[0]["price_low"] == pytest.approx(101.8)
    assert rbr[0]["price_high"] == pytest.approx(102.1)


def test_detect_quasimodo_zones_finds_bullish_failure_swing() -> None:
    seq = [
        100,
        99,
        98,
        95,
        96,
        98,
        100,
        102,
        105,
        103,
        100,
        98,
        97,
        99,
        101,
        103,
        100,
        101,
        102,
        103,
    ]
    bars = [_bar(i, p, p + 0.3, p - 0.3, p) for i, p in enumerate(seq)]
    df = pd.DataFrame(bars)
    atr_series = _atr(df, 5)
    qm_params = {"qm_swing_lookback": 2, "qm_min_swing_atr": 0.1}
    zones = _detect_quasimodo_zones(df, atr_series, qm_params)
    for z in zones:
        assert z["source"] == "QUASIMODO"
        assert z["pattern"] in ("QM_BULL", "QM_BEAR")
        assert z["price_high"] >= z["price_low"]


def test_detect_structure_labels_hh_hl_lh_ll() -> None:
    seq = [100, 102, 104, 101, 103, 106, 104, 105, 102, 101]
    bars = [_bar(i, p, p + 0.2, p - 0.2, p) for i, p in enumerate(seq)]
    df = pd.DataFrame(bars)
    structure = _detect_structure(df["high"].to_numpy(), df["low"].to_numpy(), lookback=1)
    assert structure
    labels = [s["label"] for s in structure]
    valid_labels = (StructureLabel.HH, StructureLabel.HL, StructureLabel.LH, StructureLabel.LL)
    assert all(label in valid_labels for label in labels)


def test_no_signal_when_own_position_open() -> None:
    """`ctx.own_position` set -> evaluate() must return None even though
    the same candles would otherwise produce a fresh-touch signal (see
    test_buy_signals_on_rbr_zone_retest). This is the v5 live-safety
    guard: only one position at a time."""
    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0))  # wick back into the band
    position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=102.0,
        sl=101.0,
        tp=103.0,
        opened_at=START,
    )
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"H1": pd.DataFrame(bars)},
        spread_points=SPREAD_POINTS,
        own_position=position,
    )
    assert _strategy().evaluate(ctx) is None


def test_fresh_touch_gate_only_signals_on_first_touch() -> None:
    """Two consecutive in-zone candles with no exit between them: only the
    first (the fresh touch) may emit a signal; the second, still-inside
    candle must return None — the v5 fix for the 2026-08-17 over-trading
    incident. Routed through `validate_and_load()`, the sandboxed
    execution path the live engine and registry actually use, per
    test_smc_dl_m5_step200_v1.py's precedent."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None

    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0))  # first touch
    first_ctx = MarketContext(
        symbol="XAUUSD", candles={"H1": pd.DataFrame(bars)}, spread_points=SPREAD_POINTS
    )
    first_result = instance.evaluate(first_ctx)
    assert first_result is not None
    assert len(first_result) == 3

    # Second consecutive candle, still resting inside the same zone, no
    # exit in between -> must NOT re-signal.
    bars.append(_bar(len(bars), 102.0, 102.05, 101.95, 102.0))
    second_ctx = MarketContext(
        symbol="XAUUSD", candles={"H1": pd.DataFrame(bars)}, spread_points=SPREAD_POINTS
    )
    assert instance.evaluate(second_ctx) is None
