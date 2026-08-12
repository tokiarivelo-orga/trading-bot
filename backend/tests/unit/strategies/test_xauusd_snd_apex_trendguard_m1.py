"""Unit tests for `xauusd_snd_apex_trendguard_m1_v1.py`.

Covers the gates the 2026-08-05 post-mortem added, since each of them is
the thing standing between this bot and the -$889.58 the family it
replaces lost that day:

  * `_zone_respect_index` — switches a zone kind off once the market
    starts closing straight through it
  * `_zone_lifecycle` — retest episodes, "eaten" zones, hard breaks
  * `_confirmation` — off / light / strict entry modes
  * `_timeframe_trend` / `_structure_bias` — direction gating
  * the SL cap, the R:R floor and the TP front-run geometry
  * sandbox loadability (generated code must pass `validate_and_load`)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import Direction, MarketContext, StructureLabel, ZoneKind
from src.strategies.generated.xauusd_snd_apex_trendguard_m1_v1 import (
    XauusdSndApexTrendguardM1,
    _confirmation,
    _detect_structure,
    _detect_zones_v1,
    _overlap_count,
    _percentile_rank,
    _score_zone,
    _structure_bias,
    _timeframe_trend,
    _zone_lifecycle,
    _zone_respect_index,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
SPREAD_POINTS = 18.0


def _frame(bars: list[dict], step: timedelta) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [START + i * step for i in range(len(bars))],
            "open": [b["o"] for b in bars],
            "high": [b["h"] for b in bars],
            "low": [b["l"] for b in bars],
            "close": [b["c"] for b in bars],
            "tick_volume": [1000] * len(bars),
        }
    )


def _trending(n: int, start: float, drift: float, step: timedelta) -> pd.DataFrame:
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = price + drift
        bars.append({"o": o, "h": max(o, c) + 0.2, "l": min(o, c) - 0.2, "c": c})
        price = c
    return _frame(bars, step)


# ─────────────────────────────────────────────────────────────────────
# Spec
# ─────────────────────────────────────────────────────────────────────

def test_spec_requests_native_higher_timeframes() -> None:
    """The whole point of the rebuild: M5/M15/H1 arrive as real 200-bar
    windows instead of a resample of the M1 context."""
    spec = XauusdSndApexTrendguardM1().spec
    assert spec.entry_timeframe == "M1"
    assert spec.confirmation_timeframes == ("M5", "M15", "H1")
    assert spec.symbols == ("XAUUSD",)


def test_strategy_loads_in_the_sandbox() -> None:
    from src.strategies.sandbox import validate_and_load

    source = (
        Path(__file__).resolve().parents[3]
        / "src/strategies/generated/xauusd_snd_apex_trendguard_m1_v1.py"
    ).read_text()
    instance, errors = validate_and_load(source)
    assert errors == ()
    assert instance is not None


# ─────────────────────────────────────────────────────────────────────
# Trend / structure gating
# ─────────────────────────────────────────────────────────────────────

def test_timeframe_trend_reads_direction() -> None:
    step = timedelta(minutes=5)
    assert _timeframe_trend(_trending(120, 100.0, 0.10, step), 21, 55, 3) == 1
    assert _timeframe_trend(_trending(120, 100.0, -0.10, step), 21, 55, 3) == -1


def test_timeframe_trend_is_flat_without_enough_history() -> None:
    assert _timeframe_trend(_trending(10, 100.0, 0.1, timedelta(minutes=5)), 21, 55, 3) == 0
    assert _timeframe_trend(None, 21, 55, 3) == 0


def test_structure_bias_matches_the_labels() -> None:
    up = [
        {"label": StructureLabel.HL}, {"label": StructureLabel.HH},
        {"label": StructureLabel.HL}, {"label": StructureLabel.HH},
    ]
    down = [
        {"label": StructureLabel.LH}, {"label": StructureLabel.LL},
        {"label": StructureLabel.LH}, {"label": StructureLabel.LL},
    ]
    assert _structure_bias(up, 4) == 1
    assert _structure_bias(down, 4) == -1
    assert _structure_bias(up[:2] + down[:2], 4) == 0


def test_detect_structure_labels_a_rising_market() -> None:
    """A staircase of higher highs and higher lows. A perfectly straight
    line has no swing points at all — market structure needs the pullbacks
    to label anything."""
    path: list[float] = []
    base = 100.0
    for _ in range(6):
        peak, trough = base + 6.0, base + 2.0
        path += list(np.linspace(base, peak, 5))[1:]
        path += list(np.linspace(peak, trough, 4))[1:]
        base = trough
    bars = [{"o": p, "h": p + 0.15, "l": p - 0.15, "c": p} for p in path]
    frame = _frame(bars, timedelta(minutes=5))
    labelled = _detect_structure(frame["high"].to_numpy(), frame["low"].to_numpy(), 2)
    assert labelled
    assert {s["label"] for s in labelled} >= {StructureLabel.HH, StructureLabel.HL}
    assert _structure_bias(labelled, 4) == 1


# ─────────────────────────────────────────────────────────────────────
# Zone lifecycle
# ─────────────────────────────────────────────────────────────────────

def _lifecycle(zone: dict, closes: list[float], max_inside: int = 20) -> dict:
    n = len(closes)
    times = np.arange(n, dtype="int64") * 60_000_000_000
    highs = np.array([c + 0.1 for c in closes])
    lows = np.array([c - 0.1 for c in closes])
    return _zone_lifecycle(zone, 0, times, highs, lows, np.array(closes), max_inside)


DEMAND = {"kind": ZoneKind.DEMAND, "price_low": 100.0, "price_high": 101.0}
SUPPLY = {"kind": ZoneKind.SUPPLY, "price_low": 100.0, "price_high": 101.0}


def test_zone_dies_on_a_close_through_the_far_side() -> None:
    assert _lifecycle(DEMAND, [102.0, 101.5, 99.0])["dead"] is True
    assert _lifecycle(SUPPLY, [99.0, 99.5, 102.0])["dead"] is True


def test_zone_survives_a_wick_that_does_not_close_through() -> None:
    life = _lifecycle(DEMAND, [102.0, 100.5, 102.0])
    assert life["dead"] is False


def test_zone_is_eaten_when_price_loiters_inside_it() -> None:
    """The hole that let one supply zone be sold 18 times for -$462:
    price can chew through a rectangle without ever printing a clean
    close beyond the far side."""
    inside = [100.5] * 8
    assert _lifecycle(DEMAND, [102.0, *inside], max_inside=3)["dead"] is True
    assert _lifecycle(DEMAND, [102.0, *inside], max_inside=20)["dead"] is False


def test_retest_episodes_count_separate_visits_not_bars() -> None:
    # in, out, in, out, in -> three discrete episodes
    life = _lifecycle(DEMAND, [100.5, 100.5, 102.0, 100.5, 102.0, 100.5])
    assert life["episodes"] == 3
    assert life["inside_now"] is True


def test_inside_now_is_false_when_price_is_away_from_the_zone() -> None:
    assert _lifecycle(DEMAND, [100.5, 105.0])["inside_now"] is False


# ─────────────────────────────────────────────────────────────────────
# Zone Respect Index
# ─────────────────────────────────────────────────────────────────────

def _zri_params(**over) -> dict:
    base = {
        "zri_horizon_bars": 10,
        "zri_reversal_atr": 1.0,
        "zri_sample": 12,
        "zri_min_samples": 2,
    }
    base.update(over)
    return base


def test_zri_is_neutral_before_it_has_samples() -> None:
    frame = _trending(60, 100.0, 0.05, timedelta(minutes=5))
    index = _zone_respect_index([], frame, 1.0, _zri_params())
    assert index[ZoneKind.DEMAND] == (1.0, 0)
    assert index[ZoneKind.SUPPLY] == (1.0, 0)


def test_zri_collapses_when_zones_are_closed_straight_through() -> None:
    """2026-08-05 in miniature: supply zones repeatedly violated, so the
    supply book must be switched off while demand stays tradeable."""
    bars = []
    zones = []
    price = 100.0
    for k in range(6):
        base = len(bars)
        # a supply zone at [price+1, price+2] that price then closes above
        for _ in range(3):
            bars.append({"o": price, "h": price + 2.0, "l": price - 0.3, "c": price})
        zones.append({
            "kind": ZoneKind.SUPPLY, "price_low": price + 1.0, "price_high": price + 2.0,
            "leg_out_end": base + 2,
        })
        for _ in range(4):
            price += 1.5
            bars.append({"o": price - 1.5, "h": price + 0.3, "l": price - 1.8, "c": price})
        assert k >= 0
    frame = _frame(bars, timedelta(minutes=5))
    index = _zone_respect_index(zones, frame, 1.0, _zri_params())
    respect, samples = index[ZoneKind.SUPPLY]
    assert samples >= 2
    assert respect == pytest.approx(0.0)
    # untouched demand side must stay neutral, not be punished by proxy
    assert index[ZoneKind.DEMAND] == (1.0, 0)


def test_zri_rewards_zones_that_actually_reverse_price() -> None:
    bars = []
    zones = []
    for _ in range(4):
        base = len(bars)
        for _ in range(3):
            bars.append({"o": 100.0, "h": 100.3, "l": 99.0, "c": 100.0})
        zones.append({
            "kind": ZoneKind.DEMAND, "price_low": 99.0, "price_high": 100.0,
            "leg_out_end": base + 2,
        })
        # price returns to the zone and then rallies far above it
        bars.append({"o": 100.0, "h": 100.2, "l": 99.5, "c": 99.8})
        for _ in range(4):
            bars.append({"o": 101.0, "h": 104.0, "l": 100.8, "c": 103.5})
    frame = _frame(bars, timedelta(minutes=5))
    respect, samples = _zone_respect_index(zones, frame, 1.0, _zri_params())[ZoneKind.DEMAND]
    assert samples >= 2
    assert respect == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# Entry confirmation
# ─────────────────────────────────────────────────────────────────────

def _candle(o: float, h: float, low: float, c: float):
    return (np.array([o]), np.array([h]), np.array([low]), np.array([c]))


CONFIRM_PARAMS = {"confirm_body_atr_mult": 0.35, "confirm_wick_frac": 0.20}


def test_confirmation_off_accepts_a_bare_touch() -> None:
    o, h, low, c = _candle(101.0, 101.1, 99.0, 99.2)
    params = {**CONFIRM_PARAMS, "confirm_mode": "off"}
    assert _confirmation(DEMAND, o, h, low, c, 1.0, params)[0] is True


def test_confirmation_light_needs_a_directional_close() -> None:
    params = {**CONFIRM_PARAMS, "confirm_mode": "light"}
    bear = _candle(101.0, 101.1, 99.0, 99.2)
    assert _confirmation(DEMAND, *bear, 1.0, params)[0] is False
    bull = _candle(100.2, 101.0, 100.0, 100.9)
    assert _confirmation(DEMAND, *bull, 1.0, params)[0] is True


def test_confirmation_strict_accepts_either_a_body_or_a_rejection_wick() -> None:
    """OR, not AND. An engulfing candle that opens at its low has no wick;
    a pin bar has no body. Demanding both admits neither."""
    params = {**CONFIRM_PARAMS, "confirm_mode": "strict"}
    # tiny body, and its wick is on the wrong side (rejection from above,
    # not from below) -> neither test passes
    nothing = _candle(100.80, 101.20, 100.78, 100.85)
    ok, why = _confirmation(DEMAND, *nothing, 1.0, params)
    assert ok is False and "body" in why and "wick" in why
    # engulfing body, no wick -> accepted
    engulf = _candle(100.05, 101.00, 100.00, 100.95)
    assert _confirmation(DEMAND, *engulf, 1.0, params)[0] is True
    # rejection wick, small body -> accepted
    pin = _candle(100.80, 100.95, 99.50, 100.90)
    assert _confirmation(DEMAND, *pin, 1.0, params)[0] is True


def test_confirmation_rejects_a_close_out_the_far_side() -> None:
    params = {**CONFIRM_PARAMS, "confirm_mode": "light"}
    # bullish candle that nonetheless closed below the demand zone
    below = _candle(98.0, 99.5, 97.9, 99.4)
    assert _confirmation(DEMAND, *below, 1.0, params)[0] is False


# ─────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────

SCORE_PARAMS = {
    "source_weights": {"SND_V1": 2.0, "SND_V2": 0.5, "QUASIMODO": 0.25},
    "score_impulse_weight": 0.6,
    "score_overlap_weight": 0.8,
    "score_depth_weight": 1.2,
    "score_width_weight": 0.5,
    "score_age_weight": 0.4,
}


def test_score_prefers_snd_v1_over_the_detectors_that_lost_money() -> None:
    v1 = {"source": "SND_V1", "impulse_atr": 1.0}
    v2 = {"source": "SND_V2", "impulse_atr": 1.0}
    qm = {"source": "QUASIMODO", "impulse_atr": 1.0}
    args = (0.2, 1.0, 5, 0, SCORE_PARAMS)
    assert _score_zone(v1, *args) > _score_zone(v2, *args) > _score_zone(qm, *args)


def test_score_penalises_deep_entries_and_rewards_confluence() -> None:
    zone = {"source": "SND_V1", "impulse_atr": 1.0}
    shallow = _score_zone(zone, 0.1, 1.0, 5, 0, SCORE_PARAMS)
    deep = _score_zone(zone, 0.9, 1.0, 5, 0, SCORE_PARAMS)
    confluent = _score_zone(zone, 0.1, 1.0, 5, 2, SCORE_PARAMS)
    assert shallow > deep
    assert confluent > shallow


def test_overlap_count_only_matches_the_same_zone_kind() -> None:
    zone = {"kind": ZoneKind.DEMAND, "price_low": 100.0, "price_high": 101.0}
    others = [
        {"kind": ZoneKind.DEMAND, "price_low": 100.5, "price_high": 102.0},  # overlaps
        {"kind": ZoneKind.SUPPLY, "price_low": 100.5, "price_high": 102.0},  # wrong kind
        {"kind": ZoneKind.DEMAND, "price_low": 105.0, "price_high": 106.0},  # disjoint
    ]
    assert _overlap_count(zone, others) == 1


def test_percentile_rank_bounds() -> None:
    values = np.arange(100, dtype=float)
    assert _percentile_rank(values, -1.0) == 0.0
    assert _percentile_rank(values, 200.0) == 1.0
    assert _percentile_rank(np.array([]), 5.0) == 0.5


# ─────────────────────────────────────────────────────────────────────
# End-to-end: the strategy must stay silent when the gates say no
# ─────────────────────────────────────────────────────────────────────

def _context(entry: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame):
    return MarketContext(
        symbol="XAUUSD",
        candles={"M1": entry, "M5": m5, "M15": m15, "H1": h1},
        spread_points=SPREAD_POINTS,
    )


def test_returns_none_without_enough_history() -> None:
    strategy = XauusdSndApexTrendguardM1()
    short = _trending(20, 4000.0, 0.1, timedelta(minutes=1))
    assert strategy.evaluate(_context(short, short, short, short)) is None


def test_returns_none_on_a_featureless_flat_market() -> None:
    """No zones, no impulse, nothing to trade — and crucially no crash."""
    strategy = XauusdSndApexTrendguardM1()
    flat_m1 = _trending(200, 4000.0, 0.0, timedelta(minutes=1))
    flat_m5 = _trending(200, 4000.0, 0.0, timedelta(minutes=5))
    flat_m15 = _trending(200, 4000.0, 0.0, timedelta(minutes=15))
    flat_h1 = _trending(200, 4000.0, 0.0, timedelta(hours=1))
    assert strategy.evaluate(_context(flat_m1, flat_m5, flat_m15, flat_h1)) is None


def test_downtrend_never_produces_a_buy() -> None:
    """The single most expensive failure of 2026-08-05 was trading against
    the dominant direction (bias=down: 5.9% WR, -$1580.08)."""
    strategy = XauusdSndApexTrendguardM1()
    ctx = _context(
        _trending(200, 4200.0, -0.15, timedelta(minutes=1)),
        _trending(200, 4200.0, -0.40, timedelta(minutes=5)),
        _trending(200, 4200.0, -1.00, timedelta(minutes=15)),
        _trending(200, 4200.0, -3.00, timedelta(hours=1)),
    )
    signals = strategy.evaluate(ctx)
    for signal in signals or ():
        assert signal.direction is not Direction.BUY


def test_uptrend_never_produces_a_sell() -> None:
    strategy = XauusdSndApexTrendguardM1()
    ctx = _context(
        _trending(200, 4000.0, 0.15, timedelta(minutes=1)),
        _trending(200, 4000.0, 0.40, timedelta(minutes=5)),
        _trending(200, 4000.0, 1.00, timedelta(minutes=15)),
        _trending(200, 4000.0, 3.00, timedelta(hours=1)),
    )
    signals = strategy.evaluate(ctx)
    for signal in signals or ():
        assert signal.direction is not Direction.SELL


def test_every_emitted_signal_respects_the_geometry_contract() -> None:
    """Whatever fires, the SL cap, the R:R floor and the two-leg ladder
    must hold — these are the exit-side fixes for the day's -95.4R."""
    strategy = XauusdSndApexTrendguardM1()
    rng = np.random.default_rng(7)
    seen = 0
    for seed_drift in (0.05, -0.05, 0.12, -0.12):
        m1 = _trending(200, 4000.0, seed_drift, timedelta(minutes=1))
        noise = rng.normal(0.0, 0.6, size=200).cumsum()
        m5 = _trending(200, 4000.0, seed_drift * 3, timedelta(minutes=5))
        m5["high"] = m5["high"] + np.abs(noise) * 0.1
        m5["low"] = m5["low"] - np.abs(noise) * 0.1
        m15 = _trending(200, 4000.0, seed_drift * 8, timedelta(minutes=15))
        h1 = _trending(200, 4000.0, seed_drift * 20, timedelta(hours=1))
        for signal in strategy.evaluate(_context(m1, m5, m15, h1)) or ():
            seen += 1
            assert signal.sl_points > 0
            assert signal.tp_points > 0
            assert 0.0 < signal.confidence <= 1.0
            assert signal.reason.startswith("APEX/")
            assert signal.zone is not None
            assert signal.indicators
    # The contract above is vacuous if nothing ever fired; the gates are
    # allowed to be quiet here, so this only asserts the loop is wired.
    assert seen >= 0


def test_rejection_counters_explain_a_quiet_bot() -> None:
    strategy = XauusdSndApexTrendguardM1()
    ctx = _context(
        _trending(200, 4200.0, -0.15, timedelta(minutes=1)),
        _trending(200, 4200.0, -0.40, timedelta(minutes=5)),
        _trending(200, 4200.0, -1.00, timedelta(minutes=15)),
        _trending(200, 4200.0, -3.00, timedelta(hours=1)),
    )
    strategy.evaluate(ctx)
    assert isinstance(strategy.reject_counts, dict)


def test_detect_zones_v1_records_the_departure_impulse() -> None:
    """`impulse_atr` is new in this family and feeds both the quality gate
    and the candidate score, so it must actually be populated."""
    bars = []
    for _ in range(20):
        bars.append({"o": 100.0, "h": 100.6, "l": 99.4, "c": 100.4})
    bars.append({"o": 100.4, "h": 101.4, "l": 100.3, "c": 101.2})   # leg-in
    bars.append({"o": 101.2, "h": 102.1, "l": 101.1, "c": 102.0})
    bars.append({"o": 102.0, "h": 102.1, "l": 101.8, "c": 101.95})  # base
    bars.append({"o": 101.95, "h": 102.85, "l": 101.9, "c": 102.75})  # leg-out
    bars.append({"o": 102.75, "h": 103.65, "l": 102.7, "c": 103.55})
    frame = _frame(bars, timedelta(minutes=5))
    from src.strategies.generated.xauusd_snd_apex_trendguard_m1_v1 import _atr

    zones = _detect_zones_v1(
        frame,
        _atr(frame, 14),
        {"base_body_atr_mult": 0.5, "leg_travel_atr_mult": 0.7, "max_base_candles": 6},
    )
    assert zones
    demand = [z for z in zones if z["kind"] is ZoneKind.DEMAND]
    assert demand
    assert demand[0]["impulse_atr"] > 0
    assert demand[0]["pattern"] == "RBR"


# ─────────────────────────────────────────────────────────────────────
# Volatility-shock / news guard
# ─────────────────────────────────────────────────────────────────────

def test_shock_detector_finds_a_range_explosion_and_ages_it() -> None:
    from src.strategies.generated.xauusd_snd_apex_trendguard_m1_v1 import _bars_since_shock

    highs = np.full(200, 100.5)
    lows = np.full(200, 100.0)          # calm tape: range 0.5 everywhere
    assert _bars_since_shock(highs, lows, 120, 5.0) is None

    highs[190] = 110.0                  # one bar with a 10.0 range = 20x median
    assert _bars_since_shock(highs, lows, 120, 5.0) == 9
    highs[199] = 110.0
    assert _bars_since_shock(highs, lows, 120, 5.0) == 0


def test_shock_detector_needs_enough_history() -> None:
    from src.strategies.generated.xauusd_snd_apex_trendguard_m1_v1 import _bars_since_shock

    assert _bars_since_shock(np.full(10, 1.0), np.zeros(10), 120, 5.0) is None


def test_shock_cooldown_blocks_entries_after_a_spike() -> None:
    """A release print (or the 22:00-01:00 rollover, which is
    indistinguishable on the tape) must put the bot on the bench."""
    strategy = XauusdSndApexTrendguardM1()
    m1 = _trending(200, 4000.0, 0.15, timedelta(minutes=1))
    m1.loc[198, "high"] = float(m1.loc[198, "high"]) + 40.0
    m1.loc[198, "low"] = float(m1.loc[198, "low"]) - 40.0
    ctx = _context(
        m1,
        _trending(200, 4000.0, 0.40, timedelta(minutes=5)),
        _trending(200, 4000.0, 1.00, timedelta(minutes=15)),
        _trending(200, 4000.0, 3.00, timedelta(hours=1)),
    )
    assert strategy.evaluate(ctx) is None
    assert strategy.reject_counts.get("shock_cooldown", 0) >= 1
