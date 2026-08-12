"""Give-back exit policy — pure decision logic.

The invariants that matter here are safety ones: the policy must never
loosen a stop, never act before a position has earned something, and never
depend on the model to work.
"""

from __future__ import annotations

import pytest

from src.engine.domain.exit_policy import (
    ExitAction,
    ExitPolicyConfig,
    decide_exit,
    peak_r,
)

# BUY at 100 with a 1.0 stop distance: 1R = 1.0 price unit.
BUY = dict(is_buy=True, entry_price=100.0, risk=1.0)
SELL = dict(is_buy=False, entry_price=100.0, risk=1.0)
CONFIG = ExitPolicyConfig(arm_r=0.5, keep_fraction=0.5, min_lock_r=0.2)


def test_does_nothing_before_the_position_has_earned_anything() -> None:
    decision = decide_exit(
        **BUY, current_sl=99.0, mark=100.2, extreme_favorable=100.3, config=CONFIG
    )
    assert decision.action is ExitAction.NONE


def test_arms_once_peak_clears_arm_r_and_locks_the_keep_fraction() -> None:
    # Peak +1.0R, keep 0.5 -> lock +0.5R -> stop at 100.5.
    decision = decide_exit(
        **BUY, current_sl=99.0, mark=100.8, extreme_favorable=101.0, config=CONFIG
    )
    assert decision.action is ExitAction.TIGHTEN_SL
    assert decision.stop_price == pytest.approx(100.5)
    assert "peak 1.00R" in decision.reason


def test_lock_never_falls_below_min_lock_r() -> None:
    """Peak 0.5R x keep 0.5 = 0.25R, but `min_lock_r` is 0.2R here, so the
    floor does not bind. With a smaller keep it must."""
    tight = ExitPolicyConfig(arm_r=0.5, keep_fraction=0.1, min_lock_r=0.2)
    decision = decide_exit(
        **BUY, current_sl=99.0, mark=100.4, extreme_favorable=100.5, config=tight
    )
    assert decision.stop_price == pytest.approx(100.2)


def test_never_loosens_an_already_tighter_stop() -> None:
    """The existing rules may already have moved the stop past what this
    policy would propose. It must yield, not fight."""
    decision = decide_exit(
        **BUY, current_sl=100.9, mark=101.0, extreme_favorable=101.2, config=CONFIG
    )
    assert decision.action is ExitAction.NONE


def test_sell_side_mirrors_the_buy_side() -> None:
    # SELL from 100, best price 99.0 -> peak 1.0R -> lock 0.5R -> stop 99.5.
    decision = decide_exit(
        **SELL, current_sl=101.0, mark=99.2, extreme_favorable=99.0, config=CONFIG
    )
    assert decision.action is ExitAction.TIGHTEN_SL
    assert decision.stop_price == pytest.approx(99.5)


def test_closes_when_the_locked_level_is_already_behind_price() -> None:
    """Sending a stop the market has already passed either gets rejected or
    fills at an arbitrary later price. Closing is the honest action."""
    decision = decide_exit(
        **BUY, current_sl=99.0, mark=100.3, extreme_favorable=102.0, config=CONFIG
    )
    assert decision.action is ExitAction.CLOSE


def test_change_of_character_tightens_a_profitable_position() -> None:
    decision = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=100.3,
        extreme_favorable=100.35,
        choch_against=True,
        config=CONFIG,
    )
    assert decision.action is ExitAction.TIGHTEN_SL
    assert decision.stop_price == pytest.approx(100.2)
    assert "change of character" in decision.reason


def test_change_of_character_is_ignored_when_never_in_profit() -> None:
    decision = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=99.5,
        extreme_favorable=100.0,
        choch_against=True,
        config=CONFIG,
    )
    assert decision.action is ExitAction.NONE


def test_model_modulator_is_off_by_default_because_the_head_has_no_skill() -> None:
    """With no ``revert_probability`` supplied the policy behaves identically.
    The proven state rule must not depend on the unproven model."""
    without = decide_exit(
        **BUY, current_sl=99.0, mark=100.8, extreme_favorable=101.0, config=CONFIG
    )
    with_chance_level = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=100.8,
        extreme_favorable=101.0,
        revert_probability=0.55,
        config=CONFIG,
    )
    assert without == with_chance_level


def test_model_modulator_closes_only_when_far_past_chance_and_in_profit() -> None:
    decision = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=100.4,
        extreme_favorable=100.5,
        revert_probability=0.95,
        config=CONFIG,
    )
    assert decision.action is ExitAction.CLOSE
    assert "model reversal" in decision.reason

    # Same call while the position is under water must not fire.
    losing = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=99.5,
        extreme_favorable=100.05,
        revert_probability=0.95,
        config=CONFIG,
    )
    assert losing.action is not ExitAction.CLOSE


def test_take_profit_is_pulled_in_to_the_expected_excursion() -> None:
    decision = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=100.1,
        extreme_favorable=100.15,
        take_profit=103.0,  # 3.0R away
        expected_mfe_r=1.2,
        config=CONFIG,
    )
    assert decision.action is ExitAction.REDUCE_TP
    assert decision.take_profit == pytest.approx(101.2)


def test_take_profit_is_not_pulled_behind_current_price() -> None:
    decision = decide_exit(
        **BUY,
        current_sl=99.0,
        mark=101.5,
        extreme_favorable=101.6,
        take_profit=103.0,
        expected_mfe_r=1.2,  # 101.2, already behind the 101.5 mark
        config=CONFIG,
    )
    assert decision.action is not ExitAction.REDUCE_TP


def test_disabled_config_reproduces_previous_behaviour_exactly() -> None:
    disabled = ExitPolicyConfig(enabled=False)
    decision = decide_exit(
        **BUY, current_sl=99.0, mark=102.0, extreme_favorable=103.0, config=disabled
    )
    assert decision.action is ExitAction.NONE


def test_zero_risk_is_not_a_division_by_zero() -> None:
    decision = decide_exit(
        is_buy=True,
        entry_price=100.0,
        risk=0.0,
        current_sl=100.0,
        mark=101.0,
        extreme_favorable=101.0,
        config=CONFIG,
    )
    assert decision.action is ExitAction.NONE


def test_peak_r_floors_at_flat() -> None:
    assert peak_r(is_buy=True, entry_price=100.0, extreme_favorable=98.0, risk=1.0) == 0.0
    assert peak_r(is_buy=False, entry_price=100.0, extreme_favorable=98.0, risk=1.0) == 2.0


# ─────────────────────────────────────────────────────────────────
# Per-bot resolution
# ─────────────────────────────────────────────────────────────────

def _settings(**per_strategy):
    from src.engine.domain.exit_policy import ExitPolicySettings
    return ExitPolicySettings(
        default=ExitPolicyConfig(enabled=True, arm_r=0.3),
        per_strategy=per_strategy,
    )


def test_per_bot_disable_turns_the_policy_off_for_that_bot_only():
    settings = _settings(bad_bot=ExitPolicyConfig(enabled=False))
    assert settings.resolve("bad_bot") is None
    assert settings.resolve("good_bot") is not None
    assert settings.resolve("good_bot").arm_r == 0.3


def test_unknown_bot_gets_the_default_rather_than_being_exempt():
    """A manual trade or a retired bot has no entry. Falling back to the
    default is the safe direction: unknown must mean "treat like everything
    else", never "silently skip the exit policy"."""
    settings = _settings(bad_bot=ExitPolicyConfig(enabled=False))
    assert settings.resolve(None) is not None
    assert settings.resolve("never_seen") is not None


def test_per_bot_entry_can_retune_instead_of_disabling():
    settings = _settings(
        cautious=ExitPolicyConfig(enabled=True, arm_r=0.8, keep_fraction=0.7)
    )
    resolved = settings.resolve("cautious")
    assert resolved is not None
    assert (resolved.arm_r, resolved.keep_fraction) == (0.8, 0.7)


def test_disabled_default_disables_everything_not_explicitly_enabled():
    from src.engine.domain.exit_policy import ExitPolicySettings
    settings = ExitPolicySettings(
        default=ExitPolicyConfig(enabled=False),
        per_strategy={"opt_in": ExitPolicyConfig(enabled=True)},
    )
    assert settings.resolve("anything") is None
    assert settings.resolve("opt_in") is not None


# ─────────────────────────────────────────────────────────────────
# Fixed profit target
# ─────────────────────────────────────────────────────────────────

def _fixed(target=0.3, **kw):
    return ExitPolicyConfig(enabled=True, fixed_target_r=target, **kw)


def test_fixed_target_is_off_by_default():
    """It rescues M1 and does nothing for M15, so it must be opted into
    per bot rather than applied fleet-wide."""
    assert ExitPolicyConfig().fixed_target_r is None
    decision = decide_exit(
        is_buy=True, entry_price=100.0, current_sl=99.0, mark=100.05,
        extreme_favorable=100.05, risk=1.0, take_profit=103.0,
        config=ExitPolicyConfig(enabled=True),
    )
    assert decision.action is not ExitAction.REDUCE_TP


def test_fixed_target_pulls_the_take_profit_to_the_configured_r():
    decision = decide_exit(
        is_buy=True, entry_price=100.0, current_sl=99.0, mark=100.02,
        extreme_favorable=100.02, risk=1.0, take_profit=103.0,
        config=_fixed(0.3),
    )
    assert decision.action is ExitAction.REDUCE_TP
    assert decision.take_profit == pytest.approx(100.3)


def test_fixed_target_mirrors_for_a_sell():
    decision = decide_exit(
        is_buy=False, entry_price=100.0, current_sl=101.0, mark=99.98,
        extreme_favorable=99.98, risk=1.0, take_profit=97.0,
        config=_fixed(0.3),
    )
    assert decision.action is ExitAction.REDUCE_TP
    assert decision.take_profit == pytest.approx(99.7)


def test_fixed_target_never_pushes_a_take_profit_further_out():
    """A TP already tighter than the target must be left alone — this rule
    only ever pulls in, like every other rule in this module."""
    decision = decide_exit(
        is_buy=True, entry_price=100.0, current_sl=99.0, mark=100.02,
        extreme_favorable=100.02, risk=1.0, take_profit=100.1,  # 0.1R, tighter
        config=_fixed(0.3),
    )
    assert decision.action is not ExitAction.REDUCE_TP

    sell = decide_exit(
        is_buy=False, entry_price=100.0, current_sl=101.0, mark=99.98,
        extreme_favorable=99.98, risk=1.0, take_profit=99.9,  # 0.1R, tighter
        config=_fixed(0.3),
    )
    assert sell.action is not ExitAction.REDUCE_TP


def test_fixed_target_applies_when_no_take_profit_is_set():
    decision = decide_exit(
        is_buy=True, entry_price=100.0, current_sl=99.0, mark=100.0,
        extreme_favorable=100.0, risk=1.0, take_profit=None,
        config=_fixed(0.5),
    )
    assert decision.action is ExitAction.REDUCE_TP
    assert decision.take_profit == pytest.approx(100.5)


def test_fixed_target_is_ignored_when_the_policy_is_disabled():
    decision = decide_exit(
        is_buy=True, entry_price=100.0, current_sl=99.0, mark=100.02,
        extreme_favorable=100.02, risk=1.0, take_profit=103.0,
        config=ExitPolicyConfig(enabled=False, fixed_target_r=0.3),
    )
    assert decision.action is ExitAction.NONE


def test_fixed_target_resolves_per_bot_through_settings():
    from src.engine.domain.exit_policy import ExitPolicySettings
    settings = ExitPolicySettings(
        default=ExitPolicyConfig(enabled=True),
        per_strategy={"scalper": _fixed(0.3)},
    )
    assert settings.resolve("scalper").fixed_target_r == 0.3
    assert settings.resolve("swing_bot").fixed_target_r is None
