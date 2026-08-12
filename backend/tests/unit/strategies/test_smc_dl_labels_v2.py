"""Label semantics — the two-stage stop and the pessimism rule.

The label is the thing the model optimises, so a label that does not
describe this engine's actual behaviour produces a model that optimises the
wrong quantity however well it trains. That is the defect this label set
replaces: ``smc_dl_labels.generate_labels`` predicted a pure TP-or-SL
bracket, under which expected R is negative at every target for every
pattern on this account.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.generated.smc_dl_labels_v2 import (
    LabelConfig,
    breakeven_secure_probability,
    expected_r,
    generate_labels_v2,
)


def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"])


def _atr(n: int, value: float = 1.0) -> pd.Series:
    return pd.Series([value] * n)


CONFIG = LabelConfig(secure_r=0.2, target_r=1.75, sl_atr_mult=1.0, max_holding_bars=5)


def test_stop_before_secure_is_a_loss() -> None:
    bars = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),  # entry at 100, risk 1.0, stop 99.0
            (100.0, 100.1, 98.5, 98.6),  # low pierces the stop
            (98.6, 98.7, 98.4, 98.5),
            (98.5, 98.6, 98.3, 98.4),
            (98.4, 98.5, 98.2, 98.3),
            (98.3, 98.4, 98.1, 98.2),
        ]
    )
    labels = generate_labels_v2(bars, _atr(len(bars)), CONFIG)
    assert labels["secure_long"].iloc[0] == 0.0
    assert labels["target_long"].iloc[0] == 0.0
    assert labels["bars_to_stop_long"].iloc[0] == 1


def test_secure_requires_a_close_not_a_wick() -> None:
    """``PositionManager`` only modifies stops at candle close, so a wick
    through +0.2R that reverses inside the bar must not count as secured.

    Crediting the wick inflates p_secure — measured at 0.779 with a
    high-based trigger versus 0.619 with the close-based one, against a live
    secure-rate of 0.62. The close-based number is the one that matches
    reality.
    """
    wick_only = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.5, 99.9, 100.0),  # high clears +0.2R, close does not
            (100.0, 100.1, 98.9, 98.9),  # then the stop is hit
            (98.9, 99.0, 98.8, 98.9),
            (98.9, 99.0, 98.8, 98.9),
            (98.9, 99.0, 98.8, 98.9),
        ]
    )
    assert generate_labels_v2(wick_only, _atr(6), CONFIG)["secure_long"].iloc[0] == 0.0

    closed_above = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.5, 99.9, 100.3),  # closes above +0.2R
            (100.3, 100.4, 98.9, 98.9),
            (98.9, 99.0, 98.8, 98.9),
            (98.9, 99.0, 98.8, 98.9),
            (98.9, 99.0, 98.8, 98.9),
        ]
    )
    assert generate_labels_v2(closed_above, _atr(6), CONFIG)["secure_long"].iloc[0] == 1.0


def test_target_is_a_subset_of_secure() -> None:
    """Price cannot reach the far barrier without crossing the near one, so
    ``p_target <= p_secure`` must hold by construction. The expected-R
    formula depends on it."""
    rng = np.random.default_rng(11)
    closes = 100.0 + rng.normal(0.0, 0.5, 800).cumsum()
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes + np.abs(rng.normal(0.4, 0.2, 800)),
            "low": closes - np.abs(rng.normal(0.4, 0.2, 800)),
            "close": closes,
        }
    )
    labels = generate_labels_v2(frame, _atr(800), LabelConfig(max_holding_bars=40))
    for side in ("long", "short"):
        violations = (labels[f"target_{side}"] == 1) & (labels[f"secure_{side}"] == 0)
        assert not violations.any()


def test_ambiguous_bar_resolves_pessimistically() -> None:
    """A bar that touches both the stop and the target has an unknown
    intrabar path. It resolves as a stop, because the optimistic reading
    flatters exactly the marginal setups a gate exists to remove."""
    bars = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 102.0, 98.5, 101.0),  # reaches +1.75R AND -1R in one bar
            (101.0, 101.1, 100.9, 101.0),
            (101.0, 101.1, 100.9, 101.0),
            (101.0, 101.1, 100.9, 101.0),
            (101.0, 101.1, 100.9, 101.0),
        ]
    )
    labels = generate_labels_v2(bars, _atr(6), CONFIG)
    assert labels["target_long"].iloc[0] == 0.0
    assert labels["secure_long"].iloc[0] == 0.0


def test_labels_only_look_at_bars_after_the_entry_bar() -> None:
    """Changing the entry bar's own high/low must not change its label —
    only bars strictly after it may resolve it."""
    bars = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.4, 99.8, 100.3),
            (100.3, 102.0, 100.2, 101.9),
            (101.9, 102.1, 101.7, 102.0),
            (102.0, 102.2, 101.8, 102.1),
            (102.1, 102.3, 101.9, 102.2),
        ]
    )
    baseline = generate_labels_v2(bars, _atr(6), CONFIG).iloc[0]

    widened = bars.copy()
    widened.loc[0, "high"] = 150.0
    widened.loc[0, "low"] = 50.0
    after = generate_labels_v2(widened, _atr(6), CONFIG).iloc[0]
    pd.testing.assert_series_equal(baseline, after)


def test_unresolvable_tail_rows_are_dropped_not_guessed() -> None:
    bars = _frame([(100.0, 100.2, 99.8, 100.0)] * 4)
    labels = generate_labels_v2(bars, _atr(4), CONFIG)
    assert labels["secure_long"].isna().iloc[-1]


def test_revert_class_marks_ambiguous_bars_for_exclusion() -> None:
    bars = _frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),  # both +/-0.6 ATR barriers in one bar
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
        ]
    )
    labels = generate_labels_v2(bars, _atr(6), LabelConfig(revert_bars=3, max_holding_bars=3))
    assert labels["revert_class"].iloc[0] == -1


# ---------------------------------------------------------------------------
# expected-R arithmetic
# ---------------------------------------------------------------------------
def test_expected_r_matches_the_logged_live_decision() -> None:
    """The worked example this whole label redesign was checked against:
    p_target 0.064, target 1.75R, p_secure 0.826 -> +0.090, against a
    measured backtest avg R of 0.07-0.09 for the run that produced it."""
    assert expected_r(0.826, 0.064, 1.75, 0.2) == pytest.approx(0.090, abs=0.002)


def test_expected_r_clamps_impossible_probability_pairs() -> None:
    """An uncalibrated net will emit p_target > p_secure. Clamping lives in
    one place so the strategy gate, the trainer and the API cannot disagree."""
    clamped = expected_r(0.5, 0.9, 1.75, 0.2)
    assert clamped == expected_r(0.5, 0.5, 1.75, 0.2)


def test_expected_r_subtracts_cost() -> None:
    assert expected_r(0.9, 0.3, 1.75, 0.2, cost_r=0.05) == pytest.approx(
        expected_r(0.9, 0.3, 1.75, 0.2) - 0.05
    )


def test_breakeven_probability_is_the_engine_arithmetic() -> None:
    """1/(1+0.2) = 0.8333 — the win rate at which 0.2R-for-1R trading breaks
    even. Live secure-rate is 0.62, which is why the M1 bots lose."""
    assert breakeven_secure_probability(0.2) == pytest.approx(0.83333, abs=1e-5)
