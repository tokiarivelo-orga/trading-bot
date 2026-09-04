"""Unit tests for `xauusd_snd_qm_structure_m1_v7.py` (VALIDATED — not
ACTIVE). v7 is additive over v6 (see that file's own module docstring for
the full v1-v6 history): it layers in a new trend-continuation setup
(`src.strategies.domain.structure_continuation.detect_trend_continuation`)
that fires only when the existing zone-touch scan finds nothing — the zone-
reversal path itself is untouched code, copied verbatim from v6.

This file covers three things `test_xauusd_snd_qm_structure_m1.py` (the v6
test module) does not need to:

  1. Zone-touch regression — the same fixtures/assertions as the v6 test
     module, proving the untouched path still behaves identically (and,
     structurally, that zone-touch still wins whenever both a zone-touch
     and a continuation setup would qualify: `evaluate()`'s if/else only
     ever calls `detect_trend_continuation` when the zone scan found
     nothing).
  2. The new continuation path firing on its own synthetic trend+pullback
     fixture, with a `pattern` (`TC_UP`) that is deliberately *not* in
     `_KEPT_PATTERNS` — proving that zone-specific filter is skipped for
     this path, exactly as the v7 module docstring says.
  3. The continuation path still passing through the same volatility/
     session/own-position gates the zone-touch path already runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    PositionSnapshot,
    ZoneKind,
)
from src.strategies.generated.xauusd_snd_qm_structure_m1_v7 import (
    _KEPT_PATTERNS,
    XauusdSndQmStructureM1,
    _high_conviction_demand_size_multiplier,
    _high_conviction_supply_size_multiplier,
    _session_label,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = Path("src/strategies/generated/xauusd_snd_qm_structure_m1_v7.py")

START = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=1)
BUCKET_SIZE = 5  # zone_tf_minutes(5) / entry_tf_minutes(1)
SPREAD_POINTS = 20.0


def _bar(i: int, o: float, h: float, low: float, c: float, *, start: datetime = START) -> dict:
    return {
        "time": start + i * STEP,
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "tick_volume": 1000,
    }


def _bucket(
    bars: list[dict], o: float, h: float, low: float, c: float, *, start: datetime = START
) -> None:
    i = len(bars)
    for j in range(BUCKET_SIZE):
        bars.append(_bar(i + j, o, h, low, c, start=start))


def _rbr_fixture() -> list[dict]:
    """Identical to `test_xauusd_snd_qm_structure_m1.py`'s own `_rbr_fixture`
    — same zone-touch regression fixture, so a divergent result here would
    mean the untouched zone-touch code path actually changed."""
    bars: list[dict] = []
    for _ in range(20):
        _bucket(bars, 100.0, 100.6, 99.4, 100.4)
    _bucket(bars, 100.4, 101.4, 100.3, 101.2)
    _bucket(bars, 101.2, 102.1, 101.1, 102.0)
    _bucket(bars, 102.0, 102.1, 101.8, 101.95)
    _bucket(bars, 101.95, 102.85, 101.9, 102.75)
    _bucket(bars, 102.75, 103.65, 102.7, 103.55)
    for _ in range(3):
        bars.append(_bar(len(bars), 103.5, 103.7, 103.3, 103.5))
    return bars


def _strategy() -> XauusdSndQmStructureM1:
    return XauusdSndQmStructureM1()


def _ctx(bars: list[dict], *, own_position: PositionSnapshot | None = None) -> MarketContext:
    return MarketContext(
        symbol="XAUUSD",
        candles={"M1": pd.DataFrame(bars)},
        spread_points=SPREAD_POINTS,
        own_position=own_position,
    )


def _build_zigzag_bars(
    *,
    n_legs: int,
    leg: float,
    pull: float,
    tail_bars: int,
    tail_step: float,
    offset: float = 0.1,
    base: float = 2000.0,
    start: datetime = START,
) -> list[dict]:
    """A clean M5-bucket-level uptrend zigzag, expressed as M1 bars (5
    identical M1 bars per M5 bucket, the same construction
    `test_xauusd_snd_qm_structure_m1.py`'s own `_bucket` helper uses) —
    `n_legs` impulse-then-pullback cycles, then `tail_bars` resumption
    buckets. Tuned (leg=6.0, pull=2.0, tail_step=1.0) to satisfy every
    `detect_trend_continuation` gate at its module defaults: ADX confirms
    TRENDING, the final pivot is a fresh HL, its pullback sits inside
    [0.5, 2.5] ATR, and momentum has resumed beyond the pivot bar's high."""
    bars: list[dict] = []
    price = base
    for _ in range(n_legs):
        for _ in range(4):
            price += leg / 4
            _bucket(
                bars, price - offset, price + offset, price - offset, price + offset, start=start
            )
        for _ in range(4):
            price -= pull / 4
            _bucket(
                bars, price - offset, price + offset, price - offset, price + offset, start=start
            )
    for _ in range(tail_bars):
        price += tail_step
        _bucket(bars, price - offset, price + offset, price - offset, price + offset, start=start)
    return bars


def _continuation_fixture() -> list[dict]:
    return _build_zigzag_bars(n_legs=11, leg=6.0, pull=2.0, tail_bars=3, tail_step=1.0)


# ── Zone-touch regression (untouched code path) ──────────────────────────


def test_spec_shape_carries_both_zone_and_continuation_params() -> None:
    strategy = _strategy()
    assert strategy.spec.name == "xauusd_snd_qm_structure_m1"
    assert strategy.spec.symbols == ("XAUUSD",)
    assert strategy.spec.entry_timeframe == "M1"
    assert strategy.spec.htf_veto is False
    assert strategy.spec.close_on_opposite_signal is True
    params = strategy.spec.params
    assert params["zone_tf_minutes"] == 5
    assert params["tp1_target_rr"] < params["tp2_target_rr"] < params["tp3_target_rr"]
    # v7's new, additive continuation params — unprefixed, matching
    # structure_continuation.detect_trend_continuation's own params.get keys.
    for key in (
        "pivot_bars",
        "trend_adx_period",
        "trend_adx_threshold",
        "min_pullback_atr",
        "max_pullback_atr",
        "momentum_confirm_atr_mult",
    ):
        assert key in params


def test_no_signal_on_short_history() -> None:
    bars = [_bar(i, 100.0, 100.6, 99.4, 100.4) for i in range(30)]
    assert _strategy().evaluate(_ctx(bars)) is None


def _break_zone_bars() -> list[dict]:
    """Same as `test_xauusd_snd_qm_structure_m1.py`'s own
    `_break_zone_bars` — a continuous, non-alternating decline closing
    clean through the RBR zone's low (101.8) and staying away."""
    bars: list[dict] = []
    _bucket(bars, 103.5, 103.6, 102.9, 103.0)
    _bucket(bars, 103.0, 103.1, 101.9, 102.0)
    _bucket(bars, 102.0, 102.1, 100.9, 101.0)
    _bucket(bars, 101.0, 101.1, 99.9, 100.0)
    return bars


def test_no_signal_when_zone_broken() -> None:
    bars = _rbr_fixture() + _break_zone_bars()
    assert _strategy().evaluate(_ctx(bars)) is None


def test_buy_signals_on_rbr_zone_retest_match_v6_exactly() -> None:
    """Same fixture and assertions as
    `test_xauusd_snd_qm_structure_m1.py::test_buy_signals_on_rbr_zone_retest`
    — proves the untouched zone-touch path is unchanged, and (since
    `evaluate()`'s if/else only reaches `detect_trend_continuation` when the
    zone scan finds nothing) that zone-touch still wins priority over the
    new path whenever a zone-touch setup exists."""
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
        assert sig.pattern in _KEPT_PATTERNS  # zone pattern, not TC_UP/TC_DOWN

    assert sig1.sl_points == pytest.approx(sig2.sl_points)
    assert sig2.sl_points == pytest.approx(sig3.sl_points)
    assert sig1.tp_points < sig2.tp_points < sig3.tp_points

    risk = sig1.sl_points + SPREAD_POINTS * strategy_params()["point_value"]
    required_tp = 1.5 * risk
    assert sig1.tp_points >= required_tp


def strategy_params() -> dict:
    return XauusdSndQmStructureM1().spec.params


def test_no_signal_on_wrong_direction_wick() -> None:
    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.8, 103.4, 103.6))
    assert _strategy().evaluate(_ctx(bars)) is None


def test_pattern_filter_rejects_dbd_supply_zone() -> None:
    """DBD is kept out of `_KEPT_PATTERNS` — must still be vetoed on the
    zone path exactly like v6 (this filter is untouched)."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None

    bars: list[dict] = []
    for _ in range(20):
        _bucket(bars, 100.0, 100.6, 99.4, 100.4)
    _bucket(bars, 100.4, 100.5, 99.6, 99.8)
    _bucket(bars, 99.8, 99.9, 98.8, 99.0)
    _bucket(bars, 99.0, 99.2, 98.9, 99.05)
    _bucket(bars, 99.05, 99.15, 98.25, 98.35)
    _bucket(bars, 98.35, 98.45, 97.45, 97.55)
    for _ in range(3):
        bars.append(_bar(len(bars), 97.5, 97.6, 97.4, 97.5))
    bars.append(_bar(len(bars), 99.1, 99.15, 99.0, 99.05))

    ctx = _ctx(bars)
    result = instance.evaluate(ctx)
    assert result is None, "DBD is not in _KEPT_PATTERNS and must be vetoed"


def test_no_signal_when_own_position_open_on_zone_touch_fixture() -> None:
    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0))
    position = PositionSnapshot(
        direction=Direction.BUY, entry_price=102.0, sl=101.0, tp=103.0, opened_at=START
    )
    assert _strategy().evaluate(_ctx(bars, own_position=position)) is None


def test_fresh_touch_gate_only_signals_on_first_touch() -> None:
    """Same v5-era live-safety gate as v6, routed through
    `validate_and_load()` (the sandboxed path the live engine/registry
    actually use)."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None

    bars = _rbr_fixture()
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0))  # first touch
    first_result = instance.evaluate(_ctx(bars))
    assert first_result is not None
    assert len(first_result) == 3

    bars.append(_bar(len(bars), 102.0, 102.05, 101.95, 102.0))  # still inside, no exit
    assert instance.evaluate(_ctx(bars)) is None


# ── New in v7: trend-continuation path ────────────────────────────────────


def test_continuation_fires_when_zone_touch_finds_nothing() -> None:
    bars = _continuation_fixture()
    result = _strategy().evaluate(_ctx(bars))

    assert result is not None
    assert len(result) == 3
    sig1, sig2, sig3 = result
    for sig in result:
        assert sig.direction is Direction.BUY
        # zone=None is the proof this is the continuation path, not a
        # zone-touch signal that happened to also qualify — the zone-touch
        # branch always populates `zone` (see the regression test above).
        assert sig.zone is None
        assert sig.pattern == "TC_UP"
        assert sig.pattern not in _KEPT_PATTERNS  # proves that filter is skipped here
        assert sig.sl_points > 0
        assert sig.structure  # chart-annotation pivots populated
        assert "TREND_CONTINUATION" in sig.reason

    assert sig1.sl_points == pytest.approx(sig2.sl_points)
    assert sig2.sl_points == pytest.approx(sig3.sl_points)
    assert sig1.tp_points < sig2.tp_points < sig3.tp_points

    risk = sig1.sl_points + SPREAD_POINTS * strategy_params()["point_value"]
    required_tp = 1.5 * risk
    assert sig1.tp_points >= required_tp  # clears XAUUSD's SpreadGate min_rr=1.5 floor


def test_continuation_respects_volatility_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `_volatility_ok` gate the zone-touch path already runs, forced
    to veto — the continuation path must not bypass it."""
    import src.strategies.generated.xauusd_snd_qm_structure_m1_v7 as v7

    monkeypatch.setattr(v7, "_volatility_ok", lambda *_a, **_kw: False)
    bars = _continuation_fixture()
    assert v7.XauusdSndQmStructureM1().evaluate(_ctx(bars)) is None


def test_continuation_respects_session_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same `_session_ok` gate the zone-touch path already runs, forced to
    veto — the continuation path must not bypass it."""
    import src.strategies.generated.xauusd_snd_qm_structure_m1_v7 as v7

    monkeypatch.setattr(v7, "_session_ok", lambda *_a, **_kw: False)
    bars = _continuation_fixture()
    assert v7.XauusdSndQmStructureM1().evaluate(_ctx(bars)) is None


def test_continuation_respects_own_position_guard() -> None:
    bars = _continuation_fixture()
    position = PositionSnapshot(
        direction=Direction.BUY, entry_price=2000.0, sl=1990.0, tp=2020.0, opened_at=START
    )
    assert _strategy().evaluate(_ctx(bars, own_position=position)) is None


def test_continuation_signal_survives_the_sandbox() -> None:
    """Routed through `validate_and_load()` like the zone-touch tests above
    — the new path must work under the same restricted-exec sandbox the
    live engine and registry actually use, not just as a plain class."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None

    bars = _continuation_fixture()
    result = instance.evaluate(_ctx(bars))
    assert result is not None
    assert result[0].pattern == "TC_UP"


# ── v7 (extended, 2026-08-27): sizing tier ────────────────────────────────
#
# Pure-function tests for `_high_conviction_supply_size_multiplier()` and
# `_high_conviction_demand_size_multiplier()`, mirroring the test style used
# for the sibling `xauusd_snd_qm_structure_fixed_m1_v4` sizing tier. Both
# qualifying combinations here (RBD/new_york, RBR/london) are reachable
# through the entry filter — unlike fixed_m1_v4's QM_BEAR/asian branch,
# neither is dead code.

SUPPLY_PARAMS = {
    "sizing_high_conviction_supply_multiplier": 1.3,
    "sizing_high_conviction_supply_min_confidence": 0.75,
}
DEMAND_PARAMS = {
    "sizing_high_conviction_demand_multiplier": 1.3,
    "sizing_high_conviction_demand_min_confidence": 0.75,
}


def test_session_label_boundaries() -> None:
    assert _session_label(22) == "asian"
    assert _session_label(9) == "london"
    assert _session_label(13) == "overlap"
    assert _session_label(18) == "new_york"
    assert _session_label(21) == "off_session"


def test_rbd_supply_new_york_high_confidence_gets_elevated_multiplier() -> None:
    mult = _high_conviction_supply_size_multiplier(
        "RBD", ZoneKind.SUPPLY, "new_york", 0.75, SUPPLY_PARAMS
    )
    assert mult == 1.3


def test_rbr_demand_london_high_confidence_gets_elevated_multiplier() -> None:
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.DEMAND, "london", 0.75, DEMAND_PARAMS
    )
    assert mult == 1.3


def test_supply_function_never_fires_on_demand_zone() -> None:
    mult = _high_conviction_supply_size_multiplier(
        "RBD", ZoneKind.DEMAND, "new_york", 0.90, SUPPLY_PARAMS
    )
    assert mult == 1.0


def test_demand_function_never_fires_on_supply_zone() -> None:
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.SUPPLY, "london", 0.90, DEMAND_PARAMS
    )
    assert mult == 1.0


def test_non_qualifying_pattern_or_session_stays_default() -> None:
    # RBR is kept by the entry filter and can hit supply zones too, but
    # only RBD/new_york was validated for the supply-side elevated tier.
    assert (
        _high_conviction_supply_size_multiplier(
            "RBR", ZoneKind.SUPPLY, "new_york", 0.90, SUPPLY_PARAMS
        )
        == 1.0
    )
    assert (
        _high_conviction_supply_size_multiplier(
            "RBD", ZoneKind.SUPPLY, "asian", 0.90, SUPPLY_PARAMS
        )
        == 1.0
    )
    assert (
        _high_conviction_demand_size_multiplier(
            "QM_BULL", ZoneKind.DEMAND, "london", 0.90, DEMAND_PARAMS
        )
        == 1.0
    )
    assert (
        _high_conviction_demand_size_multiplier(
            "RBR", ZoneKind.DEMAND, "asian", 0.90, DEMAND_PARAMS
        )
        == 1.0
    )


def test_confidence_below_threshold_stays_default() -> None:
    assert (
        _high_conviction_supply_size_multiplier(
            "RBD", ZoneKind.SUPPLY, "new_york", 0.70, SUPPLY_PARAMS
        )
        == 1.0
    )
    assert (
        _high_conviction_demand_size_multiplier(
            "RBR", ZoneKind.DEMAND, "london", 0.70, DEMAND_PARAMS
        )
        == 1.0
    )


def test_multiplier_and_threshold_are_param_overridable() -> None:
    supply_params = {
        "sizing_high_conviction_supply_multiplier": 1.5,
        "sizing_high_conviction_supply_min_confidence": 0.9,
    }
    assert (
        _high_conviction_supply_size_multiplier(
            "RBD", ZoneKind.SUPPLY, "new_york", 0.85, supply_params
        )
        == 1.0
    )
    assert (
        _high_conviction_supply_size_multiplier(
            "RBD", ZoneKind.SUPPLY, "new_york", 0.9, supply_params
        )
        == 1.5
    )


def test_qualifying_signal_carries_elevated_size_multiplier_through_evaluate() -> None:
    """End-to-end: a real RBR/demand zone-touch signal at a london-hour
    timestamp comes out of `evaluate()` with `size_multiplier` already
    applied on TP1 (confidence 0.75) and left at the default on TP2/TP3
    (confidence 0.70/0.65, below the 0.75 gate) — not just the pure
    function in isolation above."""
    london_start = START.replace(hour=9)
    bars = _rbr_fixture_at(london_start)
    bars.append(_bar(len(bars), 103.5, 103.6, 101.9, 102.0, start=london_start))

    result = _strategy().evaluate(_ctx(bars))

    assert result is not None
    sig1, sig2, sig3 = result
    assert sig1.pattern == "RBR"
    assert sig1.confidence == 0.75
    assert sig1.size_multiplier == 1.3
    assert sig2.size_multiplier == 1.0
    assert sig3.size_multiplier == 1.0


def _rbr_fixture_at(start: datetime) -> list[dict]:
    """Same shape as `_rbr_fixture()` above, timestamped from `start` so
    the zone-touch resolves inside a chosen UTC session hour instead of
    always `START`'s (2026-01-01 00:00, asian)."""
    bars: list[dict] = []
    for _ in range(20):
        _bucket(bars, 100.0, 100.6, 99.4, 100.4, start=start)
    _bucket(bars, 100.4, 101.4, 100.3, 101.2, start=start)
    _bucket(bars, 101.2, 102.1, 101.1, 102.0, start=start)
    _bucket(bars, 102.0, 102.1, 101.8, 101.95, start=start)
    _bucket(bars, 101.95, 102.85, 101.9, 102.75, start=start)
    _bucket(bars, 102.75, 103.65, 102.7, 103.55, start=start)
    for _ in range(3):
        bars.append(_bar(len(bars), 103.5, 103.7, 103.3, 103.5, start=start))
    return bars
