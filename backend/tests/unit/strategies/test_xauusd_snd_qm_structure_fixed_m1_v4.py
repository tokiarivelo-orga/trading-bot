"""Unit tests for the v4 sizing-tier gate on
`xauusd_snd_qm_structure_fixed_m1_v4.py`.

v4 is a copy of v3 (which added the pattern/volatility/session entry
filter — see v3's module docstring) that additionally sets
`Signal.size_multiplier` above the default 1.0 for high-conviction SELL
signals. The tier is validated for two combinations mined from v1's
closed trades — pattern=RBD/session=new_york and pattern=QM_BEAR/
session=asian, both zone_kind=supply, confidence >= 0.75 — see this
file's own module docstring ("v4 — ...") for the data-mined numbers and,
importantly, the QM_BEAR dead-code caveat: v3/v4's own `_KEPT_PATTERNS`
entry filter excludes QM_BEAR, so a QM_BEAR candidate never reaches the
signal builder in this file — the QM_BEAR branch of
`_high_conviction_supply_size_multiplier()` is only reachable as a pure
function in these tests, not through `evaluate()`.

These tests exercise `_high_conviction_supply_size_multiplier()` and
`_session_label()` directly against hand-built inputs rather than driving
the full `evaluate()` pipeline, matching the pure-gate-function test style
used for the sibling `xauusd_snd_qm_structure_adaptive_m1_v3.
_qm_supply_overlap_size_multiplier()` tests.

2026-08-27 extension: v4 also gained a DEMAND-side sibling,
`_high_conviction_demand_size_multiplier()`, for pattern=RBR/session=
london/zone_kind=demand — see the module docstring's "v4 (extended,
2026-08-27)" entry for the data-mining writeup. Unlike QM_BEAR/asian on
the supply side, this combination is reachable through the entry filter,
so it is exercised through evaluate() live/in backtest too.
"""

from __future__ import annotations

from pathlib import Path

from src.strategies.domain.models import ZoneKind
from src.strategies.generated.xauusd_snd_qm_structure_fixed_m1_v4 import (
    _KEPT_PATTERNS,
    _high_conviction_demand_size_multiplier,
    _high_conviction_supply_size_multiplier,
    _session_label,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/strategies/generated/xauusd_snd_qm_structure_fixed_m1_v4.py"
)

PARAMS = {
    "sizing_high_conviction_supply_multiplier": 1.3,
    "sizing_high_conviction_supply_min_confidence": 0.75,
}

DEMAND_PARAMS = {
    "sizing_high_conviction_demand_multiplier": 1.3,
    "sizing_high_conviction_demand_min_confidence": 0.75,
}


def test_sandbox_accepts_v4_source():
    # v4 is still sandbox-safe (math/numpy/pandas only, no I/O/broker access)
    # — a strategy file that fails this never gets registered as a version.
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


# ── _session_label ──────────────────────────────────────────────────────


def test_session_label_asian_hours():
    assert _session_label(22) == "asian"
    assert _session_label(0) == "asian"
    assert _session_label(6) == "asian"


def test_session_label_london_hour():
    assert _session_label(9) == "london"


def test_session_label_overlap_hour():
    assert _session_label(13) == "overlap"


def test_session_label_new_york_hour():
    assert _session_label(18) == "new_york"


def test_session_label_off_session_gap_hour():
    assert _session_label(21) == "off_session"


# ── _high_conviction_supply_size_multiplier ─────────────────────────────


def test_rbd_supply_new_york_high_confidence_gets_elevated_multiplier():
    mult = _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "new_york", 0.75, PARAMS)
    assert mult == 1.3


def test_qm_bear_supply_asian_high_confidence_gets_elevated_multiplier():
    # Reachable only as a pure-function call — see module docstring above:
    # `_KEPT_PATTERNS` filters QM_BEAR out before evaluate() ever calls this.
    mult = _high_conviction_supply_size_multiplier(
        "QM_BEAR", ZoneKind.SUPPLY, "asian", 0.80, PARAMS
    )
    assert mult == 1.3


def test_demand_side_never_gets_elevated_even_with_matching_pattern_and_session():
    # No equivalent buy/demand-side finding was validated — this must never
    # fire regardless of pattern/session/confidence.
    mult = _high_conviction_supply_size_multiplier("RBD", ZoneKind.DEMAND, "new_york", 0.90, PARAMS)
    assert mult == 1.0


def test_non_qualifying_pattern_stays_default():
    # RBR is kept by the entry filter and can hit supply zones too, but was
    # never validated for the elevated tier — only RBD/QM_BEAR were.
    mult = _high_conviction_supply_size_multiplier("RBR", ZoneKind.SUPPLY, "new_york", 0.90, PARAMS)
    assert mult == 1.0


def test_rbd_outside_new_york_session_stays_default():
    mult = _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "asian", 0.90, PARAMS)
    assert mult == 1.0


def test_qm_bear_outside_asian_session_stays_default():
    mult = _high_conviction_supply_size_multiplier(
        "QM_BEAR", ZoneKind.SUPPLY, "new_york", 0.90, PARAMS
    )
    assert mult == 1.0


def test_rbd_new_york_pattern_swapped_for_qm_bear_session_stays_default():
    # The two qualifying combinations are pattern+session PAIRS, not an
    # independent OR of pattern and session — RBD only qualifies in
    # new_york, QM_BEAR only qualifies in asian, not any cross-match.
    mult = _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "asian", 0.90, PARAMS)
    assert mult == 1.0
    mult2 = _high_conviction_supply_size_multiplier(
        "QM_BEAR", ZoneKind.SUPPLY, "new_york", 0.90, PARAMS
    )
    assert mult2 == 1.0


def test_confidence_below_threshold_stays_default():
    mult = _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "new_york", 0.70, PARAMS)
    assert mult == 1.0


def test_confidence_exactly_at_threshold_qualifies():
    mult = _high_conviction_supply_size_multiplier(
        "QM_BEAR", ZoneKind.SUPPLY, "asian", 0.75, PARAMS
    )
    assert mult == 1.3


def test_multiplier_and_threshold_are_param_overridable():
    params = {
        "sizing_high_conviction_supply_multiplier": 1.5,
        "sizing_high_conviction_supply_min_confidence": 0.9,
    }
    assert (
        _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "new_york", 0.85, params)
        == 1.0
    )
    assert (
        _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "new_york", 0.9, params)
        == 1.5
    )


def test_qm_bear_excluded_by_entry_filter_kept_patterns():
    # Documents the dead-code caveat from the module docstring: as long as
    # _KEPT_PATTERNS excludes QM_BEAR, evaluate() can never build a QM_BEAR
    # candidate signal, so the QM_BEAR branch above never actually fires
    # live/in backtest — only RBD does. If this assertion ever breaks (a
    # future version relaxes _KEPT_PATTERNS to include QM_BEAR), the module
    # docstring's "DEAD CODE HERE" note needs to be revisited too.
    assert "QM_BEAR" not in _KEPT_PATTERNS
    assert "RBD" in _KEPT_PATTERNS


# ── _high_conviction_demand_size_multiplier (2026-08-27 extension) ──────
#
# Mirrors the supply-side tests above for the DEMAND-side sibling: pattern=
# RBR/session=london, zone_kind=demand, confidence >= 0.75 — both RBR and
# london are reachable through the entry filter (unlike QM_BEAR/asian on
# the supply side), so unlike the QM_BEAR tests this one is exercised
# through evaluate() live/in backtest too, not just as a pure function.


def test_rbr_demand_london_high_confidence_gets_elevated_multiplier():
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.DEMAND, "london", 0.75, DEMAND_PARAMS
    )
    assert mult == 1.3


def test_supply_side_never_gets_elevated_by_demand_function():
    # The demand function must never fire for a supply-side candidate,
    # regardless of pattern/session/confidence match.
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.SUPPLY, "london", 0.90, DEMAND_PARAMS
    )
    assert mult == 1.0


def test_demand_non_qualifying_pattern_stays_default():
    # RBD/QM_BULL are kept by the entry filter and can hit demand zones
    # too, but only RBR/london was validated for the elevated tier.
    mult = _high_conviction_demand_size_multiplier(
        "QM_BULL", ZoneKind.DEMAND, "london", 0.90, DEMAND_PARAMS
    )
    assert mult == 1.0


def test_rbr_outside_london_session_stays_default():
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.DEMAND, "asian", 0.90, DEMAND_PARAMS
    )
    assert mult == 1.0


def test_demand_confidence_below_threshold_stays_default():
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.DEMAND, "london", 0.70, DEMAND_PARAMS
    )
    assert mult == 1.0


def test_demand_confidence_exactly_at_threshold_qualifies():
    mult = _high_conviction_demand_size_multiplier(
        "RBR", ZoneKind.DEMAND, "london", 0.75, DEMAND_PARAMS
    )
    assert mult == 1.3


def test_demand_multiplier_and_threshold_are_param_overridable():
    params = {
        "sizing_high_conviction_demand_multiplier": 1.5,
        "sizing_high_conviction_demand_min_confidence": 0.9,
    }
    assert (
        _high_conviction_demand_size_multiplier("RBR", ZoneKind.DEMAND, "london", 0.85, params)
        == 1.0
    )
    assert (
        _high_conviction_demand_size_multiplier("RBR", ZoneKind.DEMAND, "london", 0.9, params)
        == 1.5
    )


def test_supply_and_demand_functions_are_independent_params():
    # The two functions read distinct param keys (sizing_high_conviction_
    # supply_* vs sizing_high_conviction_demand_*) — overriding one's
    # params must not affect the other's default behavior.
    assert (
        _high_conviction_supply_size_multiplier("RBD", ZoneKind.SUPPLY, "new_york", 0.75, PARAMS)
        == 1.3
    )
    assert (
        _high_conviction_demand_size_multiplier(
            "RBR", ZoneKind.DEMAND, "london", 0.75, DEMAND_PARAMS
        )
        == 1.3
    )
