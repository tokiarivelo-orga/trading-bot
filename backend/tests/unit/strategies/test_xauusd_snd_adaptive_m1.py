"""Tests for the adaptive XAUUSD M1 strategy.

Split in two: pure helpers get exact assertions on hand-built inputs, and the
whole `evaluate()` path gets driven over a deterministic synthetic tape the
way the engine drives it — a sliding 200-bar window, one call per closed bar.
The sliding window is the part worth exercising, because every piece of
learned state has to survive bars dropping off the back of it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    Strategy,
    StructureLabel,
    ZoneKind,
)
from src.strategies.generated.xauusd_snd_adaptive_m1_v1 import (
    DEFAULT_SURVIVAL,
    MFE_SURVIVAL,
    N_FEATURES,
    PATTERN_PRIOR,
    SURVIVAL_GRID,
    XauusdSndAdaptiveM1,
    _blended_expectancy,
    _detect_choch,
    _pattern_key,
    _prior_logit,
    _session_for,
    _survival_at,
    _track_zone,
    _volume_state,
)
from src.strategies.sandbox import validate_and_load

CONTEXT_BARS = 200
_STRATEGY_FILE = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_snd_adaptive_m1_v1.py"
)


# ─────────────────────────────────────────────────────────────────
# Synthetic tape
# ─────────────────────────────────────────────────────────────────


def _tape(bars=900, seed=7):
    """A deterministic tape with trend, mean reversion and impulses, so the
    leg-base-leg and swing detectors all have something real to find."""
    rng = np.random.default_rng(seed)
    price = 2000.0
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


def _walk(strategy, m1, m15, spread=25.0, bars=None):
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
        )
        result = strategy.evaluate(ctx)
        if result:
            out.append((i, result))
    return out


# ─────────────────────────────────────────────────────────────────
# Contract
# ─────────────────────────────────────────────────────────────────


def test_satisfies_the_strategy_protocol():
    assert isinstance(XauusdSndAdaptiveM1(), Strategy)


def test_passes_the_generated_code_sandbox():
    """The file has to load under the real sandbox, not just import — that is
    what proves the `online_learning` allowlist entry works and that no
    forbidden import or dunder access crept in."""
    instance, errors = validate_and_load(_STRATEGY_FILE.read_text())
    assert errors == ()
    assert instance is not None
    assert instance.spec.name == "xauusd_snd_adaptive_m1"


def test_spec_declares_what_the_engine_needs_to_fetch():
    spec = XauusdSndAdaptiveM1().spec
    assert spec.entry_timeframe == "M1"
    assert "M15" in spec.confirmation_timeframes  # HTF confirmation + features
    assert spec.close_on_opposite_signal is True  # the only exit lever available
    assert spec.symbols == ("XAUUSD",)


def test_returns_none_without_enough_history():
    strategy = XauusdSndAdaptiveM1()
    tiny = _tape(bars=30)
    assert (
        strategy.evaluate(MarketContext(symbol="XAUUSD", candles={"M1": tiny}, spread_points=20.0))
        is None
    )
    assert strategy.evaluate(MarketContext(symbol="XAUUSD", candles={}, spread_points=20.0)) is None


def test_frame_without_a_time_column_is_declined_not_crashed():
    strategy = XauusdSndAdaptiveM1()
    frame = _tape(bars=300).drop(columns=["time"])
    assert (
        strategy.evaluate(MarketContext(symbol="XAUUSD", candles={"M1": frame}, spread_points=20.0))
        is None
    )


# ─────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────


def test_session_boundaries_match_the_engine():
    assert _session_for(13) == "overlap"
    assert _session_for(8) == "london"
    assert _session_for(17) == "new_york"
    assert _session_for(23) == "asian"
    assert _session_for(3) == "asian"
    assert _session_for(21) == "off_session"


def test_priors_are_additive_and_penalise_unknown_families():
    known = _prior_logit("RBR", "new_york", "normal", "trending", True)
    assert known == pytest.approx(0.6391 + 0.4999 + 0.1786 + 0.0785)
    unknown = _prior_logit("RBR", "new_york", "normal", "trending", False)
    assert unknown < known
    # The worst live combination must land well below the best one.
    worst = _prior_logit("DZ_V2", "overlap", "high", "ranging", True)
    assert worst < -1.0 < known


def test_survival_curve_is_monotone_and_interpolates():
    for pattern, curve in MFE_SURVIVAL.items():
        assert len(curve) == len(SURVIVAL_GRID)
        assert all(a >= b for a, b in zip(curve, curve[1:], strict=False)), pattern
    assert _survival_at("RBR", 1.0) == pytest.approx(MFE_SURVIVAL["RBR"][0])
    mid = _survival_at("RBR", 1.125)
    assert MFE_SURVIVAL["RBR"][1] < mid < MFE_SURVIVAL["RBR"][0]
    # An unseen pattern falls back to the pooled curve, never to zero.
    assert _survival_at("NOT_A_PATTERN", 1.5) == pytest.approx(DEFAULT_SURVIVAL[2])


def test_pattern_key_maps_families_onto_journal_patterns():
    assert _pattern_key({"pattern": "RBR"}) == "RBR"
    assert _pattern_key({"pattern": "SZ_V2"}) == "DZ_V2"
    assert _pattern_key({"pattern": "BREAKER_RBR"}) is None  # unproven, not RBR
    assert _pattern_key({"pattern": "SWEEP_LOW"}) is None
    assert _pattern_key({}) is None


def test_every_prior_pattern_has_a_survival_curve():
    assert set(PATTERN_PRIOR) == set(MFE_SURVIVAL)


def test_choch_fires_only_against_the_prevailing_sequence():
    up = [
        {"index": 0, "price": 100.0, "label": StructureLabel.HL},
        {"index": 1, "price": 110.0, "label": StructureLabel.HH},
        {"index": 2, "price": 105.0, "label": StructureLabel.HL},
    ]
    params = {"choch_break_atr_mult": 0.1}
    # Still above the last higher low: character unchanged.
    assert _detect_choch(up, np.array([106.0]), 1.0, params) == 0
    # Closed below it: bearish change of character.
    assert _detect_choch(up, np.array([104.0]), 1.0, params) == -1
    # The margin must actually bite.
    assert _detect_choch(up, np.array([104.95]), 1.0, params) == 0

    down = [
        {"index": 0, "price": 110.0, "label": StructureLabel.LH},
        {"index": 1, "price": 100.0, "label": StructureLabel.LL},
        {"index": 2, "price": 105.0, "label": StructureLabel.LH},
    ]
    assert _detect_choch(down, np.array([106.0]), 1.0, params) == 1
    assert _detect_choch(down, np.array([104.0]), 1.0, params) == 0
    assert _detect_choch([], np.array([104.0]), 1.0, params) == 0


def test_track_zone_reports_a_fresh_touch_only_on_the_entry_bar():
    zone = {"kind": ZoneKind.DEMAND, "price_low": 99.0, "price_high": 100.0}
    times = np.arange(6, dtype=np.int64) * 60_000_000_000
    # Price above the zone, then dips in and stays in.
    highs = np.array([102.0, 102.0, 100.5, 100.4, 100.3, 100.2])
    lows = np.array([101.0, 101.0, 99.5, 99.6, 99.7, 99.8])
    closes = np.array([101.5, 101.5, 100.0, 100.0, 100.0, 100.0])

    at_entry = _track_zone(zone, times[0], times[:3], highs[:3], lows[:3], closes[:3])
    assert at_entry["fresh_touch"] is True
    assert at_entry["touches"] == 1

    later = _track_zone(zone, times[0], times, highs, lows, closes)
    assert later["in_zone"] is True
    assert later["fresh_touch"] is False  # same episode, no new event
    assert later["touches"] == 1


def test_volume_state_is_scale_free_and_survives_a_missing_column():
    frame = _tape(bars=60)
    state = _volume_state(frame, {"volume_lookback_bars": 20})
    assert state["ratio"] > 0 and state["burst"] > 0
    flat = _volume_state(frame.drop(columns=["tick_volume"]), {})
    assert flat == {"ratio": 1.0, "burst": 1.0, "trend": 1.0}


# ─────────────────────────────────────────────────────────────────
# End-to-end over a sliding window
# ─────────────────────────────────────────────────────────────────


def test_walks_a_real_sliding_window_and_emits_wellformed_signals():
    m1 = _tape()
    events = _walk(XauusdSndAdaptiveM1(), m1, _m15_from(m1))
    assert events, "strategy produced no signal at all on a 900-bar tape"
    for _bar, signals in events:
        assert 1 <= len(signals) <= 2
        for signal in signals:
            assert signal.direction in (Direction.BUY, Direction.SELL)
            assert signal.sl_points > 0
            assert signal.tp_points > signal.sl_points  # RR floor, else SpreadGate drops it
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.reason
        # All legs of one decision agree on direction and stop.
        assert len({s.direction for s in signals}) == 1
        assert len({round(s.sl_points, 9) for s in signals}) == 1


def test_targets_always_clear_the_brokers_min_rr_floor():
    """XAUUSD is configured min_rr=1.5 and SpreadGate silently rejects any leg
    below it after spread adjustment — a leg that never opens is worse than
    no leg, because the bot then runs one target short without saying so."""
    m1 = _tape()
    strategy = XauusdSndAdaptiveM1()
    floor = strategy.spec.params["tp_min_rr_floor"]
    spread_price = 25.0 * strategy.spec.params["point_value"]
    for _bar, signals in _walk(strategy, m1, _m15_from(m1)):
        for signal in signals:
            rr = signal.tp_points / (signal.sl_points + spread_price)
            assert rr >= floor - 1e-9


def test_two_fresh_instances_agree_bar_for_bar():
    """Determinism is not optional: the learner mutates instance state every
    call, so identical input must still give identical output or no backtest
    number can be reproduced."""
    m1 = _tape()
    m15 = _m15_from(m1)
    first = _walk(XauusdSndAdaptiveM1(), m1, m15)
    second = _walk(XauusdSndAdaptiveM1(), m1, m15)
    assert [b for b, _ in first] == [b for b, _ in second]
    for (_, a), (_, b) in zip(first, second, strict=True):
        assert [(s.direction, s.sl_points, s.tp_points) for s in a] == [
            (s.direction, s.sl_points, s.tp_points) for s in b
        ]


def test_a_reused_instance_resets_when_the_tape_jumps_to_another_replay():
    """A sweep harness builds one instance per variant, but a caller that
    reuses one across two backtests would otherwise carry the first run's
    learning into the second — silently, and visible only as an
    unreproducible number."""
    m1 = _tape()
    m15 = _m15_from(m1)
    reused = XauusdSndAdaptiveM1()
    _walk(reused, m1, m15, bars=300)
    assert reused._learner is not None and reused._learner.resolved_count > 0

    far_future = m1.copy()
    far_future["time"] = far_future["time"] + pd.Timedelta(days=400)
    _walk(reused, far_future, _m15_from(far_future), bars=1)
    assert reused._learner.resolved_count == 0
    assert reused._learner.pending_count == 0


def test_a_weekend_gap_does_not_reset_learned_state():
    m1 = _tape()
    m15 = _m15_from(m1)
    strategy = XauusdSndAdaptiveM1()
    _walk(strategy, m1, m15, bars=400)
    resolved = strategy._learner.resolved_count
    assert resolved > 0

    shifted = m1.copy()
    shifted["time"] = shifted["time"] + pd.Timedelta(days=2, hours=12)
    _walk(strategy, shifted, _m15_from(shifted), bars=1)
    assert strategy._learner.resolved_count >= resolved


def test_learner_can_be_switched_off_for_a_controlled_ab():
    m1 = _tape()
    m15 = _m15_from(m1)
    strategy = XauusdSndAdaptiveM1()
    strategy.spec.params["learner_enabled"] = False
    events = _walk(strategy, m1, m15)
    assert events
    assert strategy._learner is None
    assert any("learner=off" in s.reason for _b, sigs in events for s in sigs)


def test_denylisted_families_never_produce_a_signal():
    m1 = _tape()
    m15 = _m15_from(m1)
    strategy = XauusdSndAdaptiveM1()
    strategy.spec.params["family_denylist"] = ("SND_V2", "BREAKER", "SWEEP")
    for _bar, signals in _walk(strategy, m1, m15):
        for signal in signals:
            assert not signal.reason.startswith(("SND_V2", "BREAKER", "SWEEP"))


def test_skipping_every_session_and_regime_silences_zone_entries():
    """The hard regime gates have to actually gate. CHoCH exits are exempt by
    design — closing a position that is going wrong must not depend on the
    regime being one the bot likes to open in."""
    m1 = _tape()
    m15 = _m15_from(m1)
    strategy = XauusdSndAdaptiveM1()
    strategy.spec.params["skip_volatility_buckets"] = (0, 1, 2)
    for _bar, signals in _walk(strategy, m1, m15):
        assert all(s.pattern == "CHOCH" for s in signals)


def test_signals_carry_learner_reasoning_as_indicator_readings():
    m1 = _tape()
    events = _walk(XauusdSndAdaptiveM1(), m1, _m15_from(m1))
    zone_events = [s for _b, sigs in events for s in sigs if s.pattern != "CHOCH"]
    assert zone_events
    names = {reading.name for reading in zone_events[0].indicators}
    assert {"learner_p_secure", "target_expected_r", "target_p_hit"} <= names


def test_feature_vector_width_matches_the_learner():
    strategy = XauusdSndAdaptiveM1()
    m1 = _tape()
    _walk(strategy, m1, _m15_from(m1), bars=250)
    assert strategy._learner is not None
    assert strategy._learner._model.weights.shape == (N_FEATURES,)


def test_blended_expectancy_models_the_engines_three_outcome_exit():
    """Reproduces a real logged decision: p_hit 0.064 at a 1.75R target with
    p_secure 0.826. A two-outcome bracket model calls this -0.823 and would
    refuse it; the three-outcome model returns +0.090, and the backtest that
    produced this decision measured avg R of 0.07-0.09."""
    value = _blended_expectancy(p_hit=0.064, r_target=1.75, p_secure=0.826, secure_r=0.2)
    assert value == pytest.approx(0.0902, abs=5e-4)

    # A setup that never secures is worth -1R however good its target looks.
    assert _blended_expectancy(
        p_hit=0.9, r_target=3.0, p_secure=0.0, secure_r=0.2
    ) == pytest.approx(-1.0)

    # p_hit cannot exceed p_secure: reaching the target without first
    # travelling secure_r is impossible, and an unclamped prior would
    # manufacture expectancy out of arithmetic.
    assert _blended_expectancy(
        p_hit=0.99, r_target=3.0, p_secure=0.5, secure_r=0.2
    ) == _blended_expectancy(p_hit=0.5, r_target=3.0, p_secure=0.5, secure_r=0.2)


def test_expectancy_gate_rejects_setups_the_probability_gate_would_pass():
    m1 = _tape()
    m15 = _m15_from(m1)
    permissive = XauusdSndAdaptiveM1()
    permissive.spec.params["min_expectancy_r"] = -99.0
    strict = XauusdSndAdaptiveM1()
    strict.spec.params["min_expectancy_r"] = 0.5  # far above anything realistic

    loose_events = _walk(permissive, m1, m15)
    strict_events = _walk(strict, m1, m15)
    loose_zone = [s for _b, sigs in loose_events for s in sigs if s.pattern != "CHOCH"]
    strict_zone = [s for _b, sigs in strict_events for s in sigs if s.pattern != "CHOCH"]
    assert loose_zone, "permissive run should still trade"
    assert len(strict_zone) < len(loose_zone)


def test_emitted_signals_clear_the_expectancy_floor():
    m1 = _tape()
    strategy = XauusdSndAdaptiveM1()
    floor = strategy.spec.params["min_expectancy_r"]
    for _bar, signals in _walk(strategy, m1, _m15_from(m1)):
        for signal in signals:
            if signal.pattern == "CHOCH":
                continue
            reading = next(r for r in signal.indicators if r.name == "target_expected_r")
            assert reading.value >= floor - 1e-9
