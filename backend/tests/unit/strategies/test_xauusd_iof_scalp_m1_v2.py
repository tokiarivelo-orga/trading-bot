"""Unit tests for the v2 sizing-tier gate on `xauusd_iof_scalp_m1_v2.py`.

v2 is a copy of v1 (the IOF-microstructure XAUUSD M1 scalper — see
`test_xauusd_iof_scalp_m1.py` for the full pipeline tests, unchanged here)
that additionally sets `Signal.size_multiplier` above the default 1.0 for
high-conviction SELL/supply FVG_BEAR signals in the new_york session. The
tier is validated on this strategy's OWN closed trades for
`xauusd_iof_scalp_m1:v1` (side=sell, pattern=FVG_BEAR, zone_kind=supply,
session=new_york): n=106, win rate flat at 35.8% either side of the median
confidence split, but profit factor improves 1.27 -> 1.48 above confidence
0.7 — a payoff-ratio edge, not a hit-rate edge. See this file's own module
docstring ("v2 — ...") for the full data-mining writeup and the floor-lock
check against the live account.

These tests exercise `_high_conviction_fvg_bear_size_multiplier()` directly
against hand-built inputs rather than driving the full `evaluate()`
pipeline, matching the pure-gate-function test style used for the sibling
`xauusd_snd_qm_structure_fixed_m1_v4.test_high_conviction_supply_size_
multiplier()` tests.
"""

from __future__ import annotations

from pathlib import Path

from src.strategies.domain.models import ZoneKind
from src.strategies.generated.xauusd_iof_scalp_m1_v2 import (
    _high_conviction_fvg_bear_size_multiplier,
    _session_for,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_iof_scalp_m1_v2.py"
)

PARAMS = {
    "sizing_fvg_bear_supply_ny_multiplier": 1.2,
    "sizing_fvg_bear_supply_ny_min_confidence": 0.7,
}


def test_sandbox_accepts_v2_source():
    # v2 is still sandbox-safe (math/numpy/pandas only, no I/O/broker access)
    # — a strategy file that fails this never gets registered as a version.
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


# ── _session_for (byte-identical to v1 — sanity-check the boundary the
# sizing gate is keyed to, since it decides new_york reachability) ──


def test_session_for_new_york_hours():
    assert _session_for(16) == "new_york"
    assert _session_for(18) == "new_york"
    assert _session_for(20) == "new_york"


def test_session_for_new_york_boundary_exclusive():
    assert _session_for(21) != "new_york"
    assert _session_for(12) != "new_york"


# ── _high_conviction_fvg_bear_size_multiplier ───────────────────────────


def test_fvg_bear_supply_new_york_high_confidence_gets_elevated_multiplier():
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.75, PARAMS
    )
    assert mult == 1.2


def test_demand_side_never_gets_elevated_even_with_matching_pattern_and_session():
    # No equivalent buy/demand-side finding was validated — this must never
    # fire regardless of pattern/session/confidence. FVG_BEAR zones are
    # always SUPPLY by construction (see _detect_fvg_zones), but the gate
    # still checks zone_kind defensively.
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.DEMAND, "new_york", 0.90, PARAMS
    )
    assert mult == 1.0


def test_non_qualifying_pattern_stays_default():
    # A supply zone from any other detector (S&D, Quasimodo, breaker, ...)
    # was never validated for the elevated tier — only FVG_BEAR was.
    mult = _high_conviction_fvg_bear_size_multiplier(
        "RBD", ZoneKind.SUPPLY, "new_york", 0.90, PARAMS
    )
    assert mult == 1.0


def test_fvg_bear_outside_new_york_session_stays_default():
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "london", 0.90, PARAMS
    )
    assert mult == 1.0
    mult2 = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "asian", 0.90, PARAMS
    )
    assert mult2 == 1.0
    mult3 = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "overlap", 0.90, PARAMS
    )
    assert mult3 == 1.0


def test_confidence_below_threshold_stays_default():
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.65, PARAMS
    )
    assert mult == 1.0


def test_confidence_exactly_at_threshold_qualifies():
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.7, PARAMS
    )
    assert mult == 1.2


def test_multiplier_and_threshold_are_param_overridable():
    params = {
        "sizing_fvg_bear_supply_ny_multiplier": 1.5,
        "sizing_fvg_bear_supply_ny_min_confidence": 0.9,
    }
    assert (
        _high_conviction_fvg_bear_size_multiplier(
            "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.85, params
        )
        == 1.0
    )
    assert (
        _high_conviction_fvg_bear_size_multiplier(
            "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.9, params
        )
        == 1.5
    )


def test_defaults_used_when_params_missing_keys():
    # An empty params dict must fall back to the module's own defaults
    # (1.2x / 0.7) rather than raising or silently disabling the tier.
    mult = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.75, {}
    )
    assert mult == 1.2
    mult_below = _high_conviction_fvg_bear_size_multiplier(
        "FVG_BEAR", ZoneKind.SUPPLY, "new_york", 0.65, {}
    )
    assert mult_below == 1.0
