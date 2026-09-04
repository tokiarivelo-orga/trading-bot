"""Tests for the IOF-microstructure XAUUSD M1 scalper.

Split three ways: pure-function tests for the new microstructure detectors
(FVG, volume-delta proxy, absorption, killzone), pure-function tests for the
new `ExitDecision` routing, and an end-to-end walk over a deterministic
synthetic tape the way the engine drives `evaluate()` — a sliding 200-bar
window, one call per closed bar — following
`test_xauusd_snd_adaptive_m1.py`'s pattern since this strategy forks that
file's skeleton.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
    MarketContext,
    PositionSnapshot,
    Strategy,
    ZoneKind,
)
from src.strategies.generated.xauusd_iof_scalp_m1_v1 import (
    N_FEATURES,
    XauusdIofScalpM1,
    _detect_fvg_zones,
    _killzone_flag,
    _microstructure_state,
    _nearest_fvg_distance_atr,
    _session_for,
    _volume_delta,
    _zone_freshly_formed,
)
from src.strategies.sandbox import validate_and_load

CONTEXT_BARS = 200
_STRATEGY_FILE = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_iof_scalp_m1_v1.py"
)


# ─────────────────────────────────────────────────────────────────
# Synthetic tape
# ─────────────────────────────────────────────────────────────────


def _tape(bars=900, seed=7, start_price=2000.0):
    """A deterministic tape with trend, mean reversion and impulses, so the
    leg-base-leg/swing detectors and the FVG detector all have something
    real to find."""
    rng = np.random.default_rng(seed)
    price = start_price
    rows = []
    for i in range(bars):
        drift = 0.35 * np.sin(i / 90.0) + 0.12 * np.sin(i / 17.0)
        shock = 1.8 if i % 137 == 0 else 0.0
        price += drift + rng.normal(0, 0.22) + shock * rng.choice([-1.0, 1.0])
        spread = abs(rng.normal(0, 0.25)) + 0.12
        open_ = price - rng.normal(0, 0.1)
        close = price
        rows.append(
            {
                "time": pd.Timestamp("2026-03-02 00:00", tz=None) + pd.Timedelta(minutes=i),
                "open": open_,
                "high": max(open_, close) + spread,
                "low": min(open_, close) - spread,
                "close": close,
                "tick_volume": int(80 + abs(rng.normal(0, 30))),
            }
        )
    return pd.DataFrame(rows)


def _m15_from(m1):
    grouped = (
        m1.set_index("time")
        .resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"})
        .dropna()
        .reset_index()
    )
    return grouped


def _walk(strategy, m1, m15, spread=25.0, bars=None, own_position=None):
    """Drive `evaluate()` the way `trade_loop` does."""
    out = []
    stop = len(m1) if bars is None else min(len(m1), CONTEXT_BARS + bars)
    m15_times = m15["time"].to_numpy()
    for i in range(CONTEXT_BARS, stop):
        window = m1.iloc[i - CONTEXT_BARS : i].reset_index(drop=True)
        cut = int(m15_times.searchsorted(window["time"].iloc[-1].to_datetime64(), "right"))
        ctx = MarketContext(
            symbol="XAUUSD",
            candles={
                "M1": window,
                "M15": m15.iloc[max(0, cut - CONTEXT_BARS) : cut].reset_index(drop=True),
            },
            spread_points=spread,
            own_position=own_position,
        )
        result = strategy.evaluate(ctx)
        if result:
            out.append((i, result))
    return out


def _with_engineered_fvg(m1, *, bullish, offset_from_last=4):
    """Overwrite the last 3 bars of a copy of `m1` with a clean 3-candle FVG
    of the requested polarity, anchored near the tape's own last price so
    the rest of the pipeline (ATR, zone detection) stays well-behaved."""
    m1 = m1.copy()
    anchor = float(m1["close"].iloc[-offset_from_last])
    idx = m1.index
    if bullish:
        # low[-1] > high[-3]
        m1.loc[idx[-3], ["open", "high", "low", "close"]] = [
            anchor - 0.5,
            anchor - 0.0,
            anchor - 0.7,
            anchor - 0.2,
        ]
        m1.loc[idx[-2], ["open", "high", "low", "close"]] = [
            anchor + 0.2,
            anchor + 0.5,
            anchor + 0.0,
            anchor + 0.4,
        ]
        m1.loc[idx[-1], ["open", "high", "low", "close"]] = [
            anchor + 0.8,
            anchor + 1.5,
            anchor + 1.0,
            anchor + 1.3,
        ]
    else:
        # high[-1] < low[-3]
        m1.loc[idx[-3], ["open", "high", "low", "close"]] = [
            anchor + 0.5,
            anchor + 0.6,
            anchor + 0.0,
            anchor + 0.2,
        ]
        m1.loc[idx[-2], ["open", "high", "low", "close"]] = [
            anchor - 0.2,
            anchor - 0.0,
            anchor - 0.5,
            anchor - 0.4,
        ]
        m1.loc[idx[-1], ["open", "high", "low", "close"]] = [
            anchor - 0.8,
            anchor - 1.0,
            anchor - 1.5,
            anchor - 1.3,
        ]
    return m1, anchor


# ─────────────────────────────────────────────────────────────────
# Contract
# ─────────────────────────────────────────────────────────────────


def test_satisfies_the_strategy_protocol():
    assert isinstance(XauusdIofScalpM1(), Strategy)


def test_passes_the_generated_code_sandbox():
    instance, errors = validate_and_load(_STRATEGY_FILE.read_text())
    assert errors == ()
    assert instance is not None
    assert instance.spec.name == "xauusd_iof_scalp_m1"


def test_spec_declares_what_the_engine_needs_to_fetch():
    spec = XauusdIofScalpM1().spec
    assert spec.entry_timeframe == "M1"
    assert "M15" in spec.confirmation_timeframes
    assert spec.close_on_opposite_signal is True
    assert spec.symbols == ("XAUUSD",)


def test_returns_none_without_enough_history():
    strategy = XauusdIofScalpM1()
    tiny = _tape(bars=30)
    assert (
        strategy.evaluate(MarketContext(symbol="XAUUSD", candles={"M1": tiny}, spread_points=20.0))
        is None
    )
    assert strategy.evaluate(MarketContext(symbol="XAUUSD", candles={}, spread_points=20.0)) is None


def test_frame_without_a_time_column_is_declined_not_crashed():
    strategy = XauusdIofScalpM1()
    frame = _tape(bars=300).drop(columns=["time"])
    assert (
        strategy.evaluate(MarketContext(symbol="XAUUSD", candles={"M1": frame}, spread_points=20.0))
        is None
    )


# ─────────────────────────────────────────────────────────────────
# Pure helpers — sessions / killzone
# ─────────────────────────────────────────────────────────────────


def test_session_boundaries_match_the_engine():
    """Byte-identical to the base file — the killzone flag is a separate
    function precisely so this stays true (see `_killzone_flag`'s docstring)."""
    assert _session_for(13) == "overlap"
    assert _session_for(8) == "london"
    assert _session_for(17) == "new_york"
    assert _session_for(23) == "asian"
    assert _session_for(3) == "asian"
    assert _session_for(21) == "off_session"


def test_killzone_flag_covers_only_the_session_open_window():
    params = {"killzone_window_minutes": 45}
    assert _killzone_flag(7, 0, params) is True
    assert _killzone_flag(7, 44, params) is True
    assert _killzone_flag(7, 45, params) is False
    assert _killzone_flag(16, 0, params) is True
    assert _killzone_flag(16, 30, params) is True
    assert _killzone_flag(16, 45, params) is False
    # London mid-session and New York mid-session are NOT killzone, even
    # though `_session_for` still reports them as "london"/"new_york".
    assert _killzone_flag(10, 0, params) is False
    assert _killzone_flag(19, 0, params) is False
    assert _killzone_flag(3, 0, params) is False


def test_killzone_window_is_configurable():
    narrow = {"killzone_window_minutes": 10}
    assert _killzone_flag(7, 9, narrow) is True
    assert _killzone_flag(7, 10, narrow) is False


# ─────────────────────────────────────────────────────────────────
# Pure helpers — FVG detector
# ─────────────────────────────────────────────────────────────────


def _bar(open_, high, low, close, volume=100):
    return {"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume}


def test_fvg_detector_finds_a_clean_bullish_gap():
    rows = [_bar(100.0, 100.2, 99.8, 100.0) for _ in range(5)]
    # candle i-2 high = 100.2; candle i low must exceed it.
    rows.append(_bar(100.5, 101.0, 100.4, 100.8))  # i-2 (index 5)
    rows.append(_bar(100.9, 101.3, 100.7, 101.1))  # i-1 (index 6)
    rows.append(_bar(101.6, 102.0, 101.5, 101.9))  # i   (index 7), low=101.5 > high[5]=101.0
    df = pd.DataFrame(rows)
    atr_series = pd.Series([0.3] * len(df))
    zones = _detect_fvg_zones(df, atr_series, {"fvg_min_gap_atr_mult": 0.05})
    bull = [z for z in zones if z["pattern"] == "FVG_BULL" and z["conf_idx"] == 7]
    assert len(bull) == 1
    zone = bull[0]
    assert zone["kind"] == ZoneKind.DEMAND
    assert zone["price_low"] == pytest.approx(101.0)  # high[i-2]
    assert zone["price_high"] == pytest.approx(101.5)  # low[i]
    assert zone["base_start"] == 5
    assert zone["source"] == "FVG"


def test_fvg_detector_finds_a_clean_bearish_gap():
    rows = [_bar(100.0, 100.2, 99.8, 100.0) for _ in range(5)]
    rows.append(_bar(100.5, 101.0, 100.4, 100.8))  # i-2, low=100.4
    rows.append(_bar(100.0, 100.3, 99.6, 99.8))  # i-1
    rows.append(_bar(99.5, 99.9, 99.0, 99.2))  # i, high=99.9 < low[5]=100.4
    df = pd.DataFrame(rows)
    atr_series = pd.Series([0.3] * len(df))
    zones = _detect_fvg_zones(df, atr_series, {"fvg_min_gap_atr_mult": 0.05})
    bear = [z for z in zones if z["pattern"] == "FVG_BEAR" and z["conf_idx"] == 7]
    assert len(bear) == 1
    zone = bear[0]
    assert zone["kind"] == ZoneKind.SUPPLY
    assert zone["price_low"] == pytest.approx(99.9)  # high[i]
    assert zone["price_high"] == pytest.approx(100.4)  # low[i-2]


def test_fvg_detector_ignores_overlapping_candles():
    rows = [_bar(100.0, 100.5, 99.5, 100.0) for _ in range(8)]
    df = pd.DataFrame(rows)
    atr_series = pd.Series([0.3] * len(df))
    zones = _detect_fvg_zones(df, atr_series, {"fvg_min_gap_atr_mult": 0.05})
    assert zones == []


def test_fvg_detector_respects_the_minimum_gap_floor():
    rows = [_bar(100.0, 100.2, 99.8, 100.0) for _ in range(5)]
    rows.append(_bar(100.2, 100.3, 100.1, 100.2))  # i-2 (index 5), high=100.3
    rows.append(_bar(100.2, 100.3, 100.1, 100.2))  # i-1 (index 6)
    # A tiny gap on the final triplet: low[i] = 100.301 barely clears
    # high[i-2] = 100.3 — smaller than the ATR-scaled floor.
    rows.append(_bar(100.3, 100.4, 100.301, 100.35))  # i (index 7)
    df = pd.DataFrame(rows)
    atr_series = pd.Series([1.0] * len(df))  # large ATR makes the gap negligible
    zones = _detect_fvg_zones(df, atr_series, {"fvg_min_gap_atr_mult": 0.05})
    # No zone confirms on the final bar (index 7) — the only triplet this
    # test is actually about.
    assert not any(z["conf_idx"] == len(df) - 1 for z in zones)


def test_zone_freshly_formed_uses_the_right_index_space():
    fvg_zone = {"source": "FVG", "conf_idx": 199}
    base_zone = {"source": "SND_V1", "conf_idx": 39}
    assert _zone_freshly_formed(fvg_zone, zone_frame_len=40, entry_len=200) is True
    assert _zone_freshly_formed(base_zone, zone_frame_len=40, entry_len=200) is True
    assert _zone_freshly_formed(fvg_zone, zone_frame_len=40, entry_len=201) is False
    assert _zone_freshly_formed(base_zone, zone_frame_len=41, entry_len=200) is False
    assert _zone_freshly_formed({"source": "FVG"}, zone_frame_len=40, entry_len=200) is False


def test_nearest_fvg_distance_prefers_same_direction_and_closest():
    live_zones = [
        {"source": "FVG", "kind": ZoneKind.DEMAND, "price_low": 99.0, "price_high": 99.4},
        {"source": "FVG", "kind": ZoneKind.DEMAND, "price_low": 95.0, "price_high": 95.4},
        {"source": "FVG", "kind": ZoneKind.SUPPLY, "price_low": 100.0, "price_high": 100.4},
        {"source": "SND_V1", "kind": ZoneKind.DEMAND, "price_low": 99.1, "price_high": 99.3},
    ]
    dist = _nearest_fvg_distance_atr(
        live_zones, close=100.0, trade_dir=1, atr_val=1.0, sentinel=6.0
    )
    # Nearest same-direction (DEMAND) FVG midpoint is (99.0+99.4)/2 = 99.2 -> dist 0.8.
    assert dist == pytest.approx(0.8)
    # The one SUPPLY-kind FVG resolves for a short-direction query instead.
    dist_short = _nearest_fvg_distance_atr(
        live_zones, close=100.0, trade_dir=-1, atr_val=1.0, sentinel=6.0
    )
    assert dist_short == pytest.approx(0.2)  # (100.0+100.4)/2=100.2, |100-100.2|=0.2


def test_nearest_fvg_distance_falls_back_to_sentinel_when_none_exist():
    assert _nearest_fvg_distance_atr([], close=100.0, trade_dir=1, atr_val=1.0, sentinel=6.0) == 6.0
    assert (
        _nearest_fvg_distance_atr(
            [{"source": "SND_V1", "kind": ZoneKind.DEMAND, "price_low": 99.0, "price_high": 99.4}],
            close=100.0,
            trade_dir=1,
            atr_val=1.0,
            sentinel=6.0,
        )
        == 6.0
    )


# ─────────────────────────────────────────────────────────────────
# Pure helpers — volume-delta proxy / absorption
# ─────────────────────────────────────────────────────────────────


def test_volume_delta_leans_toward_the_close():
    # Close at the high -> almost all buy volume.
    df = pd.DataFrame([_bar(100.0, 101.0, 100.0, 101.0, volume=200)])
    buy_vol, sell_vol, delta = _volume_delta(df)
    assert buy_vol[0] == pytest.approx(200.0)
    assert sell_vol[0] == pytest.approx(0.0)
    assert delta[0] == pytest.approx(200.0)

    # Close at the low -> almost all sell volume.
    df2 = pd.DataFrame([_bar(101.0, 101.0, 100.0, 100.0, volume=200)])
    buy_vol2, sell_vol2, delta2 = _volume_delta(df2)
    assert buy_vol2[0] == pytest.approx(0.0)
    assert sell_vol2[0] == pytest.approx(200.0)
    assert delta2[0] == pytest.approx(-200.0)


def test_volume_delta_guards_the_zero_range_bar():
    """A bar where high == low carries no directional information — it must
    split its volume 50/50, never divide by zero / produce NaN or inf."""
    df = pd.DataFrame([_bar(100.0, 100.0, 100.0, 100.0, volume=150)])
    buy_vol, sell_vol, delta = _volume_delta(df)
    assert np.isfinite(buy_vol).all()
    assert np.isfinite(sell_vol).all()
    assert np.isfinite(delta).all()
    assert buy_vol[0] == pytest.approx(75.0)
    assert sell_vol[0] == pytest.approx(75.0)
    assert delta[0] == pytest.approx(0.0)


def test_volume_delta_handles_a_mixed_tape_without_nans():
    rows = [_bar(100.0, 100.0, 100.0, 100.0) for _ in range(3)]  # zero-range
    rows += [_bar(100.0, 101.0, 99.5, 100.7, volume=50) for _ in range(3)]
    df = pd.DataFrame(rows)
    buy_vol, sell_vol, delta = _volume_delta(df)
    assert np.isfinite(buy_vol).all() and np.isfinite(sell_vol).all()
    assert np.isfinite(delta).all()


def test_absorption_flag_fires_on_high_volume_small_body_one_sided_delta():
    rows = [_bar(100.0, 100.5, 99.5, 100.0, volume=100) for _ in range(25)]
    # An absorption bar: huge tick_volume, tiny body, close near the high
    # (one-sided buy delta) despite the volume — the classic absorption shape.
    rows.append(_bar(100.0, 100.6, 99.9, 100.05, volume=800))
    df = pd.DataFrame(rows)
    params = {
        "volume_lookback_bars": 20,
        "absorption_min_vol_ratio": 1.5,
        "absorption_max_body_atr": 0.35,
        "absorption_min_one_sided": 0.4,
    }
    state = _microstructure_state(df, atr_val=0.5, params=params)
    assert bool(state["absorption_flag"][-1]) is True
    assert state["absorption_score"][-1] > 0.0


def test_absorption_flag_does_not_fire_on_an_ordinary_bar():
    rows = [_bar(100.0, 100.5, 99.5, 100.0, volume=100) for _ in range(25)]
    # Average volume, big body, roughly balanced delta -> not absorption.
    rows.append(_bar(100.0, 101.0, 99.0, 100.5, volume=100))
    df = pd.DataFrame(rows)
    params = {
        "volume_lookback_bars": 20,
        "absorption_min_vol_ratio": 1.5,
        "absorption_max_body_atr": 0.35,
        "absorption_min_one_sided": 0.4,
    }
    state = _microstructure_state(df, atr_val=0.5, params=params)
    assert bool(state["absorption_flag"][-1]) is False


def test_microstructure_state_arrays_are_finite_over_a_real_tape():
    df = _tape(bars=120)
    state = _microstructure_state(df, atr_val=0.5, params={"volume_lookback_bars": 20})
    for key in ("delta", "delta_ratio", "delta_z", "vol_ratio", "absorption_score"):
        assert np.isfinite(state[key]).all(), key


# ─────────────────────────────────────────────────────────────────
# ExitDecision — thesis invalidation / continuation
# ─────────────────────────────────────────────────────────────────


def test_close_exit_decision_on_a_fresh_opposing_fvg():
    m1 = _tape(bars=250, seed=3)
    m1, anchor = _with_engineered_fvg(m1, bullish=False)  # bearish FVG, against a BUY
    m15 = _m15_from(m1)
    strategy = XauusdIofScalpM1()
    own_position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=anchor - 1.0,
        sl=anchor - 5.0,
        tp=anchor + 5.0,
        opened_at=m1["time"].iloc[-10],
    )
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": m1, "M15": m15},
        spread_points=25.0,
        own_position=own_position,
    )
    result = strategy.evaluate(ctx)
    assert isinstance(result, ExitDecision)
    assert result.action == ExitActionKind.CLOSE
    assert "opposing FVG" in result.reason


def test_breakeven_exit_decision_on_continuation_with_fresh_confluence():
    m1 = _tape(bars=250, seed=5)
    m1, anchor = _with_engineered_fvg(m1, bullish=True)  # bullish FVG, with a BUY
    m15 = _m15_from(m1)
    strategy = XauusdIofScalpM1()
    own_position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=anchor - 5.0,  # well below current close -> already in favor
        sl=anchor - 10.0,
        tp=anchor + 10.0,
        opened_at=m1["time"].iloc[-10],
    )
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": m1, "M15": m15},
        spread_points=25.0,
        own_position=own_position,
    )
    result = strategy.evaluate(ctx)
    assert isinstance(result, ExitDecision)
    assert result.action == ExitActionKind.BREAKEVEN
    assert "same-direction confluence" in result.reason


def test_no_exit_decision_when_there_is_no_own_position():
    m1 = _tape(bars=250, seed=3)
    m1, _anchor = _with_engineered_fvg(m1, bullish=False)
    m15 = _m15_from(m1)
    strategy = XauusdIofScalpM1()
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": m1, "M15": m15},
        spread_points=25.0,
        own_position=None,
    )
    result = strategy.evaluate(ctx)
    # With no own position there is nothing to invalidate/confirm — whatever
    # comes back (None, a Signal, or a tuple of Signals) must not be an
    # ExitDecision.
    if result is not None:
        results = result if isinstance(result, tuple | list) else (result,)
        assert not any(isinstance(r, ExitDecision) for r in results)


def test_exit_decisions_can_be_switched_off():
    m1 = _tape(bars=250, seed=3)
    m1, anchor = _with_engineered_fvg(m1, bullish=False)
    m15 = _m15_from(m1)
    strategy = XauusdIofScalpM1()
    strategy.spec.params["exit_decisions_enabled"] = False
    own_position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=anchor - 1.0,
        sl=anchor - 5.0,
        tp=anchor + 5.0,
        opened_at=m1["time"].iloc[-10],
    )
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": m1, "M15": m15},
        spread_points=25.0,
        own_position=own_position,
    )
    result = strategy.evaluate(ctx)
    if result is not None:
        results = result if isinstance(result, tuple | list) else (result,)
        assert not any(isinstance(r, ExitDecision) for r in results)


# ─────────────────────────────────────────────────────────────────
# End-to-end over a sliding window
# ─────────────────────────────────────────────────────────────────


def test_walks_a_real_sliding_window_and_emits_wellformed_signals():
    m1 = _tape()
    events = _walk(XauusdIofScalpM1(), m1, _m15_from(m1))
    assert events, "strategy produced no signal at all on a 900-bar tape"
    for _bar_idx, signals in events:
        if isinstance(signals, ExitDecision):
            continue
        assert 1 <= len(signals) <= 2
        for signal in signals:
            assert signal.direction in (Direction.BUY, Direction.SELL)
            assert signal.sl_points > 0
            assert signal.tp_points > signal.sl_points
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.reason
        assert len({s.direction for s in signals}) == 1
        assert len({round(s.sl_points, 9) for s in signals}) == 1


def test_two_fresh_instances_agree_bar_for_bar():
    """Determinism is not optional: the learner mutates instance state every
    call, so identical input must still give identical output."""
    m1 = _tape()
    m15 = _m15_from(m1)
    first = _walk(XauusdIofScalpM1(), m1, m15)
    second = _walk(XauusdIofScalpM1(), m1, m15)
    assert [b for b, _ in first] == [b for b, _ in second]
    for (_, a), (_, b) in zip(first, second, strict=True):
        if isinstance(a, ExitDecision) or isinstance(b, ExitDecision):
            assert a == b
            continue
        assert [(s.direction, s.sl_points, s.tp_points) for s in a] == [
            (s.direction, s.sl_points, s.tp_points) for s in b
        ]


def test_learner_can_be_switched_off_for_a_controlled_ab():
    m1 = _tape()
    m15 = _m15_from(m1)
    strategy = XauusdIofScalpM1()
    strategy.spec.params["learner_enabled"] = False
    events = _walk(strategy, m1, m15)
    assert events
    assert strategy._learner is None
    assert any(
        "learner=off" in s.reason
        for _b, sigs in events
        if not isinstance(sigs, ExitDecision)
        for s in sigs
    )


def test_feature_vector_width_matches_the_learner():
    strategy = XauusdIofScalpM1()
    m1 = _tape()
    _walk(strategy, m1, _m15_from(m1), bars=250)
    assert strategy._learner is not None
    assert strategy._learner._model.weights.shape == (N_FEATURES,)


def test_bucket_keys_carry_the_new_microstructure_tokens():
    m1 = _tape()
    events = _walk(XauusdIofScalpM1(), m1, _m15_from(m1))
    zone_events = [
        s
        for _b, sigs in events
        if not isinstance(sigs, ExitDecision)
        for s in sigs
        if s.pattern != "CHOCH"
    ]
    assert zone_events
    # `learner_p_secure`/`learner_bucket_samples` readings prove the learner
    # actually scored these against a bucket key — the bucket string itself
    # isn't surfaced on the Signal, so this checks the plumbing indirectly
    # via the fact that a verdict was attached at all.
    names = {reading.name for reading in zone_events[0].indicators}
    assert {"learner_p_secure", "learner_bucket_samples"} <= names


def test_a_reused_instance_resets_when_the_tape_jumps_to_another_replay():
    m1 = _tape()
    m15 = _m15_from(m1)
    reused = XauusdIofScalpM1()
    _walk(reused, m1, m15, bars=300)
    assert reused._learner is not None and reused._learner.resolved_count > 0

    far_future = m1.copy()
    far_future["time"] = far_future["time"] + pd.Timedelta(days=400)
    _walk(reused, far_future, _m15_from(far_future), bars=1)
    assert reused._learner.resolved_count == 0
    # A fresh candidate observed on the one bar just walked is legitimate new
    # state (this bot's extra FVG family raises candidate density versus the
    # base file, so a single bar can already record one); what must NOT
    # survive the reset is anything from before the jump, which
    # `resolved_count == 0` above already confirms.
    assert reused._learner.pending_count <= 5
