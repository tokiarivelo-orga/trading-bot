"""Unit tests for the v3 sizing-tier gate on
`xauusd_snd_qm_structure_adaptive_m1_v3.py`.

v3 is a copy of v2 (which added the `ctx.own_position`/`fresh_touch`
re-entry-spam fix — see v2's module docstring for the 2026-08-17
over-trading incident) that additionally sets `Signal.size_multiplier`
above the default 1.0 for high-conviction SELL signals. The tier is
validated ONLY for pattern in {QM_BEAR, QMC}, zone_kind=supply, the
London+NY overlap session, and confidence >= 0.75 — see this file's
own module docstring ("v3 — ...") for the data-mined numbers. These
tests exercise `_qm_supply_overlap_size_multiplier()` directly against
hand-built zone dicts rather than driving the full `evaluate()` pipeline,
matching the pure-gate-function test style used for the sibling
`xauusd_snd_qm_structure_m1_v6._session_ok()`/`_volatility_ok()` gates.
"""

from __future__ import annotations

from pathlib import Path

from src.strategies.domain.models import ZoneKind
from src.strategies.generated.xauusd_snd_qm_structure_adaptive_m1_v3 import (
    _qm_supply_overlap_size_multiplier,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/strategies/generated/xauusd_snd_qm_structure_adaptive_m1_v3.py"
)

PARAMS = {
    "sizing_qm_supply_overlap_multiplier": 2.0,
    "sizing_qm_supply_overlap_min_confidence": 0.75,
}


def _zone(kind: ZoneKind, pattern: str) -> dict:
    return {"kind": kind, "pattern": pattern}


def test_sandbox_accepts_v3_source():
    # v3 is still sandbox-safe (math/numpy/pandas only, no I/O/broker access)
    # — a strategy file that fails this never gets registered as a version.
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


def test_qm_bear_supply_overlap_high_confidence_gets_elevated_multiplier():
    zone = _zone(ZoneKind.SUPPLY, "QM_BEAR")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.75, PARAMS)
    assert mult == 2.0


def test_qmc_supply_overlap_high_confidence_gets_elevated_multiplier():
    zone = _zone(ZoneKind.SUPPLY, "QMC")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.80, PARAMS)
    assert mult == 2.0


def test_demand_side_never_gets_elevated_even_with_matching_pattern_label():
    # No equivalent buy/demand-side finding was validated — this must never
    # fire regardless of pattern/session/confidence.
    zone = _zone(ZoneKind.DEMAND, "QM_BEAR")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.90, PARAMS)
    assert mult == 1.0


def test_non_qualifying_pattern_stays_default():
    zone = _zone(ZoneKind.SUPPLY, "QMR")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.90, PARAMS)
    assert mult == 1.0


def test_outside_overlap_session_stays_default():
    zone = _zone(ZoneKind.SUPPLY, "QM_BEAR")
    mult = _qm_supply_overlap_size_multiplier(zone, False, 0.90, PARAMS)
    assert mult == 1.0


def test_confidence_below_threshold_stays_default():
    zone = _zone(ZoneKind.SUPPLY, "QM_BEAR")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.70, PARAMS)
    assert mult == 1.0


def test_confidence_exactly_at_threshold_qualifies():
    zone = _zone(ZoneKind.SUPPLY, "QMC")
    mult = _qm_supply_overlap_size_multiplier(zone, True, 0.75, PARAMS)
    assert mult == 2.0


def test_multiplier_and_threshold_are_param_overridable():
    zone = _zone(ZoneKind.SUPPLY, "QM_BEAR")
    params = {
        "sizing_qm_supply_overlap_multiplier": 1.5,
        "sizing_qm_supply_overlap_min_confidence": 0.9,
    }
    assert _qm_supply_overlap_size_multiplier(zone, True, 0.85, params) == 1.0
    assert _qm_supply_overlap_size_multiplier(zone, True, 0.9, params) == 1.5
