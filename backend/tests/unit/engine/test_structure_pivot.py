"""Pure decision-logic tests for the structure-pivot profit-lock rule
(`src/engine/domain/structure_pivot.py`). Wiring through `PositionManager`
(the `_improves` merge, the config-off no-op) is covered by
`test_position_manager_structure_pivot.py`, mirroring how
`test_exit_policy.py`/`test_position_manager_giveback.py` split the same
give-back rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.engine.domain.structure_pivot import (
    StructurePivotConfig,
    SwingPivot,
    decide_structure_pivot_exit,
    detect_swing_pivots,
    select_ratchet_pivot,
)
from src.strategies.domain.models import StructureLabel

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _times(n: int) -> list[datetime]:
    return [T0 + i * timedelta(minutes=5) for i in range(n)]


# ---- detect_swing_pivots ----------------------------------------------------


def test_detects_alternating_hl_hh_in_a_clean_uptrend_zigzag() -> None:
    """A textbook uptrend zigzag: each swing low is higher than the last
    (HL) and each swing high is higher than the last (HH). The very first
    swing low and first swing high found have no predecessor and are
    skipped (documented in `detect_swing_pivots`)."""
    lows = [10, 9, 8, 9, 8.5, 9.5, 9, 10, 9.5, 11, 10.5, 12]
    highs = [10.5, 9.5, 8.5, 9.5, 9, 10, 9.5, 10.5, 10, 11.5, 11, 12.5]
    pivots = detect_swing_pivots(
        np.array(highs), np.array(lows), _times(len(lows)), pivot_bars=1
    )
    labels = [(p.index, p.label, p.price) for p in pivots]
    assert labels == [
        (4, StructureLabel.HL, 8.5),
        (5, StructureLabel.HH, 10.0),
        (6, StructureLabel.HL, 9.0),
        (7, StructureLabel.HH, 10.5),
        (8, StructureLabel.HL, 9.5),
        (9, StructureLabel.HH, 11.5),
        (10, StructureLabel.HL, 10.5),
    ]


def test_detects_alternating_lh_ll_in_a_clean_downtrend_zigzag() -> None:
    """Mirror of the uptrend case: each swing low lower than the last is
    LL, each swing high lower than the last is LH."""
    highs = [10, 9, 8, 9, 8.5, 7.5, 8, 7, 7.5, 6, 6.5, 5]
    lows = [9.5, 8.5, 7.5, 8.5, 8, 7, 7.5, 6.5, 7, 5.5, 6, 4.5]
    pivots = detect_swing_pivots(
        np.array(highs), np.array(lows), _times(len(lows)), pivot_bars=1
    )
    labels = [(p.index, p.label, p.price) for p in pivots]
    # The first swing low (index 2) and first swing high (index 3) each have
    # no predecessor of their own kind yet, so both are left unlabeled and
    # do not appear below — labeling starts from the second swing of each
    # kind, same as the uptrend case above.
    assert labels == [
        (5, StructureLabel.LL, 7.0),
        (6, StructureLabel.LH, 8.0),
        (7, StructureLabel.LL, 6.5),
        (8, StructureLabel.LH, 7.5),
        (9, StructureLabel.LL, 5.5),
        (10, StructureLabel.LH, 6.5),
    ]


def test_flat_candles_produce_no_pivots() -> None:
    """Identical bars never satisfy the strict-extreme test — no ties are
    resolved as pivots, matching how manual/warmup flat data must not
    fabricate structure."""
    n = 20
    highs = np.full(n, 100.6)
    lows = np.full(n, 99.4)
    pivots = detect_swing_pivots(highs, lows, _times(n), pivot_bars=2)
    assert pivots == []


def test_confirmation_lags_pivot_bars_on_each_edge() -> None:
    """The first/last `pivot_bars` bars can never be confirmed pivots — a
    swing pivot needs `pivot_bars` bars of history on *both* sides."""
    n = 10
    rng = np.random.default_rng(7)
    highs = 100 + rng.random(n)
    lows = highs - 1 - rng.random(n)
    pivots = detect_swing_pivots(highs, lows, _times(n), pivot_bars=3)
    assert all(3 <= p.index < n - 3 for p in pivots)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        detect_swing_pivots(np.array([1.0, 2.0]), np.array([1.0]), _times(2))


# ---- select_ratchet_pivot ----------------------------------------------------


def _pivot(index: int, minutes: int, price: float, label: StructureLabel) -> SwingPivot:
    return SwingPivot(index=index, time=T0 + timedelta(minutes=minutes), price=price, label=label)


def test_select_ratchet_pivot_picks_most_recent_hl_for_a_buy() -> None:
    pivots = [
        _pivot(1, 5, 2402.0, StructureLabel.HL),
        _pivot(2, 10, 2405.0, StructureLabel.HH),
        _pivot(3, 15, 2408.0, StructureLabel.HL),
    ]
    chosen = select_ratchet_pivot(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        pivots=pivots,
        entered_after=T0,
    )
    assert chosen is not None
    assert chosen.price == pytest.approx(2408.0)


def test_select_ratchet_pivot_picks_most_recent_lh_for_a_sell() -> None:
    pivots = [
        _pivot(1, 5, 2398.0, StructureLabel.LH),
        _pivot(2, 10, 2395.0, StructureLabel.LL),
        _pivot(3, 15, 2394.0, StructureLabel.LH),
    ]
    chosen = select_ratchet_pivot(
        is_buy=False,
        entry_price=2400.0,
        risk=10.0,
        pivots=pivots,
        entered_after=T0,
    )
    assert chosen is not None
    assert chosen.price == pytest.approx(2394.0)


def test_select_ratchet_pivot_ignores_pivots_before_entry() -> None:
    """A pivot that predates the position is not 'a new swing-structure
    point in the trade's direction' — it must be excluded even if it is the
    most recent qualifying label in the full series."""
    entered_after = T0 + timedelta(minutes=20)
    pivots = [
        _pivot(1, 5, 2412.0, StructureLabel.HL),  # before entry — must be ignored
        _pivot(2, 12, 2401.0, StructureLabel.HL),  # before entry — must be ignored
    ]
    chosen = select_ratchet_pivot(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        pivots=pivots,
        entered_after=entered_after,
    )
    assert chosen is None


def test_select_ratchet_pivot_respects_min_pivot_r() -> None:
    pivots = [_pivot(1, 5, 2401.0, StructureLabel.HL)]  # only 0.1R beyond entry
    assert (
        select_ratchet_pivot(
            is_buy=True,
            entry_price=2400.0,
            risk=10.0,
            pivots=pivots,
            entered_after=T0,
            min_pivot_r=0.5,
        )
        is None
    )
    assert (
        select_ratchet_pivot(
            is_buy=True,
            entry_price=2400.0,
            risk=10.0,
            pivots=pivots,
            entered_after=T0,
            min_pivot_r=0.05,
        )
        is not None
    )


def test_select_ratchet_pivot_zero_risk_returns_none() -> None:
    pivots = [_pivot(1, 5, 2412.0, StructureLabel.HL)]
    assert (
        select_ratchet_pivot(
            is_buy=True, entry_price=2400.0, risk=0.0, pivots=pivots, entered_after=T0
        )
        is None
    )


# ---- decide_structure_pivot_exit --------------------------------------------


def test_decide_returns_none_when_disabled() -> None:
    pivots = [_pivot(1, 5, 2420.0, StructureLabel.HL)]
    candidate = decide_structure_pivot_exit(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        peak_r_value=2.0,
        pivots=pivots,
        entered_after=T0,
        config=StructurePivotConfig(enabled=False),
    )
    assert candidate is None


def test_decide_returns_none_below_arm_r() -> None:
    pivots = [_pivot(1, 5, 2420.0, StructureLabel.HL)]
    candidate = decide_structure_pivot_exit(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        peak_r_value=0.5,  # below default arm_r=1.0
        pivots=pivots,
        entered_after=T0,
        config=StructurePivotConfig(),
    )
    assert candidate is None


def test_decide_returns_pivot_price_minus_buffer_for_a_buy() -> None:
    pivots = [_pivot(1, 5, 2420.0, StructureLabel.HL)]
    candidate = decide_structure_pivot_exit(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        peak_r_value=1.5,
        pivots=pivots,
        entered_after=T0,
        config=StructurePivotConfig(arm_r=1.0, buffer_r_mult=0.1),
    )
    # buffer = 0.1 * risk(10.0) = 1.0 -> 2420.0 - 1.0 = 2419.0
    assert candidate == pytest.approx(2419.0)


def test_decide_returns_pivot_price_plus_buffer_for_a_sell() -> None:
    pivots = [_pivot(1, 5, 2380.0, StructureLabel.LH)]
    candidate = decide_structure_pivot_exit(
        is_buy=False,
        entry_price=2400.0,
        risk=10.0,
        peak_r_value=1.5,
        pivots=pivots,
        entered_after=T0,
        config=StructurePivotConfig(arm_r=1.0, buffer_r_mult=0.1),
    )
    # buffer = 0.1 * risk(10.0) = 1.0 -> 2380.0 + 1.0 = 2381.0
    assert candidate == pytest.approx(2381.0)


def test_decide_returns_none_when_no_qualifying_pivot_exists() -> None:
    pivots = [_pivot(1, 5, 2420.0, StructureLabel.LH)]  # wrong direction for a buy
    candidate = decide_structure_pivot_exit(
        is_buy=True,
        entry_price=2400.0,
        risk=10.0,
        peak_r_value=2.0,
        pivots=pivots,
        entered_after=T0,
        config=StructurePivotConfig(),
    )
    assert candidate is None
