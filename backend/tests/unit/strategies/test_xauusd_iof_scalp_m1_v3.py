"""Unit tests for the v3 BREAKER_FVG_BEAR/BREAKER_FVG_BULL exclusion on
`xauusd_iof_scalp_m1_v3.py`.

v3 is a copy of v2 (see `test_xauusd_iof_scalp_m1_v2.py` for the sizing-tier
tests, unchanged here; `test_xauusd_iof_scalp_m1.py` for the full pipeline)
that additionally retires two BREAKER subtypes entirely:
`BREAKER_FVG_BEAR` (a broken FVG_BEAR/supply zone flipped to DEMAND) and
`BREAKER_FVG_BULL` (a broken FVG_BULL/demand zone flipped to SUPPLY). Both
were data-mined from this strategy's OWN closed trades for strategy_version
`xauusd_iof_scalp_m1:v1`, filtered to genuinely independent samples via
distinct 5-minute open-time buckets: BREAKER_FVG_BEAR n=21/11 episodes,
38.1% WR, -$26.18; BREAKER_FVG_BULL n=21/11 episodes, 28.6% WR, -$106.25 —
real, multi-day, multi-episode losers, not a re-entry-spam artifact (unlike
e.g. `BREAKER_RBD`'s spam-inflated 100% raw win rate, correctly left
untouched). See this file's own module docstring ("v3 — ...") for the full
writeup.

These tests exercise `_is_disabled_breaker_signal()` directly — the single
gate `_decide`'s candidate-scoring loop checks right alongside the existing
`family_denylist` gate — against zone shapes built by the real, unmodified
`_flip_to_breaker()` (the same function that actually constructs
`BREAKER_FVG_BEAR`/`BREAKER_FVG_BULL` zones in production), matching the
pure-gate-function test style used for the sibling v2 sizing-tier tests. An
empirical check confirmed the two disabled patterns essentially never win
the per-bar ranking against competing plain-FVG candidates on the repo's
standard synthetic tape (`test_xauusd_iof_scalp_m1.py`'s `_tape()`) even
before this fix — they exist but are consistently outranked — so a
full `evaluate()` walk is not a reliable way to exercise this path
end-to-end; testing the gate function directly against real
`_flip_to_breaker()` output is the precise, deterministic alternative.
"""

from __future__ import annotations

from pathlib import Path

from src.strategies.domain.models import ZoneKind
from src.strategies.generated.xauusd_iof_scalp_m1_v3 import (
    _flip_to_breaker,
    _is_disabled_breaker_signal,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_iof_scalp_m1_v3.py"
)


def test_sandbox_accepts_v3_source():
    # v3 is still sandbox-safe (math/numpy/pandas only, no I/O/broker access)
    # — a strategy file that fails this never gets registered as a version.
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


def test_decide_calls_the_gate_right_after_the_family_denylist_check():
    # A "helper function exists but never gets called" regression would pass
    # every test below yet ship completely inert — assert it's actually
    # wired into the candidate loop, in the intended spot.
    code = STRATEGY_PATH.read_text()
    denylist_idx = code.index('if blocked or zone["family"] in tuple(params.get("family_denylist"')
    gate_idx = code.index('if _is_disabled_breaker_signal(zone.get("pattern"), zone["kind"]):')
    choch_idx = code.index("if choch == -trade_dir:")
    assert denylist_idx < gate_idx < choch_idx


# ── _is_disabled_breaker_signal — direct pattern/zone_kind cases ────────


def test_breaker_fvg_bear_demand_is_disabled():
    assert _is_disabled_breaker_signal("BREAKER_FVG_BEAR", ZoneKind.DEMAND) is True


def test_breaker_fvg_bull_supply_is_disabled():
    assert _is_disabled_breaker_signal("BREAKER_FVG_BULL", ZoneKind.SUPPLY) is True


def test_breaker_fvg_bear_wrong_kind_is_not_disabled():
    # BREAKER_FVG_BEAR is always DEMAND by construction (see
    # `_flip_to_breaker`), but the gate checks zone_kind defensively —
    # confirm it doesn't over-fire on a mismatched kind.
    assert _is_disabled_breaker_signal("BREAKER_FVG_BEAR", ZoneKind.SUPPLY) is False


def test_breaker_fvg_bull_wrong_kind_is_not_disabled():
    assert _is_disabled_breaker_signal("BREAKER_FVG_BULL", ZoneKind.DEMAND) is False


def test_other_breaker_subtypes_stay_untouched():
    for pattern, kind in (
        ("BREAKER_RBR", ZoneKind.DEMAND),
        ("BREAKER_RBD", ZoneKind.SUPPLY),
        ("BREAKER_DBD", ZoneKind.DEMAND),
        ("BREAKER_DBR", ZoneKind.SUPPLY),
        ("BREAKER_SWEEP_LOW", ZoneKind.DEMAND),
        ("BREAKER_SWEEP_HIGH", ZoneKind.SUPPLY),
        ("BREAKER_QM_BULL", ZoneKind.DEMAND),
    ):
        assert _is_disabled_breaker_signal(pattern, kind) is False


def test_plain_fvg_and_other_families_stay_untouched():
    # Non-BREAKER patterns must never match, including the very patterns
    # BREAKER_FVG_BEAR/BULL are derived from.
    for pattern, kind in (
        ("FVG_BEAR", ZoneKind.SUPPLY),
        ("FVG_BULL", ZoneKind.DEMAND),
        ("SWEEP_LOW", ZoneKind.DEMAND),
        ("SWEEP_HIGH", ZoneKind.SUPPLY),
        ("RBR", ZoneKind.DEMAND),
        ("RBD", ZoneKind.SUPPLY),
        (None, ZoneKind.DEMAND),
    ):
        assert _is_disabled_breaker_signal(pattern, kind) is False


# ── Integration with the real `_flip_to_breaker` — the function that
# actually constructs these patterns in production ──────────────────────


def test_a_broken_fvg_bear_zone_flips_into_the_disabled_pattern():
    # Exactly what `_select_candidates` does to a broken FVG_BEAR/supply
    # zone: flip it, and this is the zone+life combination that would have
    # produced a `Signal` in v2 and now must be skipped in v3.
    fvg_bear_zone = {
        "source": "FVG",
        "family": "FVG",
        "pattern": "FVG_BEAR",
        "kind": ZoneKind.SUPPLY,
        "price_high": 2001.0,
        "price_low": 2000.5,
        "base_start": 10,
        "conf_idx": 12,
        "leg_out_end": 12,
        "impulse_atr": 1.2,
    }
    flipped = _flip_to_breaker(fvg_bear_zone, break_ns=123)
    assert flipped["pattern"] == "BREAKER_FVG_BEAR"
    assert flipped["kind"] == ZoneKind.DEMAND
    assert _is_disabled_breaker_signal(flipped["pattern"], flipped["kind"]) is True


def test_a_broken_fvg_bull_zone_flips_into_the_disabled_pattern():
    fvg_bull_zone = {
        "source": "FVG",
        "family": "FVG",
        "pattern": "FVG_BULL",
        "kind": ZoneKind.DEMAND,
        "price_high": 2000.5,
        "price_low": 2000.0,
        "base_start": 10,
        "conf_idx": 12,
        "leg_out_end": 12,
        "impulse_atr": 1.2,
    }
    flipped = _flip_to_breaker(fvg_bull_zone, break_ns=123)
    assert flipped["pattern"] == "BREAKER_FVG_BULL"
    assert flipped["kind"] == ZoneKind.SUPPLY
    assert _is_disabled_breaker_signal(flipped["pattern"], flipped["kind"]) is True


def test_a_broken_non_fvg_zone_still_produces_a_signal_normally():
    # An otherwise-identical broken zone from a different detector (e.g. the
    # classic leg-base-leg RBR/supply family) must flip and pass the gate
    # exactly as it did in v2 — only the two named FVG-derived patterns are
    # excluded.
    rbr_zone = {
        "source": "SND_V1",
        "family": "SND_V1_RBR",
        "pattern": "RBR",
        "kind": ZoneKind.DEMAND,
        "price_high": 2001.0,
        "price_low": 2000.5,
        "base_start": 10,
        "conf_idx": 12,
        "leg_out_end": 12,
        "impulse_atr": 1.2,
    }
    flipped = _flip_to_breaker(rbr_zone, break_ns=123)
    assert flipped["pattern"] == "BREAKER_RBR"
    assert _is_disabled_breaker_signal(flipped["pattern"], flipped["kind"]) is False


def test_a_broken_sweep_zone_still_produces_a_signal_normally():
    sweep_zone = {
        "source": "SWEEP",
        "family": "SWEEP",
        "pattern": "SWEEP_LOW",
        "kind": ZoneKind.DEMAND,
        "price_high": 2001.0,
        "price_low": 2000.5,
        "base_start": 10,
        "conf_idx": 12,
        "leg_out_end": 12,
        "impulse_atr": 1.2,
        "needs_htf": False,
    }
    flipped = _flip_to_breaker(sweep_zone, break_ns=123)
    assert flipped["pattern"] == "BREAKER_SWEEP_LOW"
    assert _is_disabled_breaker_signal(flipped["pattern"], flipped["kind"]) is False
