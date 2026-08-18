import dataclasses
from datetime import UTC, datetime

import pytest

from src.engine.application.risk_manager import RiskManager, apply_risk_override
from src.engine.domain.models import RiskCaps

CAPS = RiskCaps(
    risk_per_trade_pct=0.5,
    daily_loss_limit_pct=2.0,
    max_open_positions=2,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=3,
)


def make_manager(caps: RiskCaps = CAPS) -> RiskManager:
    return RiskManager(caps=caps, timezone="UTC")


def test_size_position_computes_lots_from_risk_pct():
    manager = make_manager()
    decision = manager.size_position(
        balance=10_000.0,
        sl_distance_price=5.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert decision.approved
    assert decision.volume == 0.1  # (10000*0.005) / (5*100)


def test_size_position_rejects_when_fallback_disabled():
    manager = make_manager()  # CAPS has min_lot_fallback_enabled=False (the default)
    decision = manager.size_position(
        balance=10.0,  # tiny risk budget -> below volume_min
        sl_distance_price=50.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved
    assert "minimum" in decision.reason


def test_size_position_falls_back_to_min_lot_within_ceiling():
    caps = RiskCaps(
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=2.0,
        max_open_positions=2,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=3,
        min_lot_fallback_enabled=True,
        max_risk_per_trade_pct=10.0,  # generous ceiling for a small account
    )
    manager = make_manager(caps)
    decision = manager.size_position(
        balance=100.0,  # 0.5% risk budget can't reach volume_min at this sl distance
        sl_distance_price=50.0,
        contract_size=10.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert decision.approved
    # min-lot risk = 0.01*50*10 = $5 = 5% of $100, under the 10% ceiling
    assert decision.volume == 0.01


def test_size_position_rejects_when_min_lot_exceeds_ceiling():
    caps = RiskCaps(
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=2.0,
        max_open_positions=2,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=3,
        min_lot_fallback_enabled=True,
        max_risk_per_trade_pct=2.0,  # min-lot risk (5%) exceeds this
    )
    manager = make_manager(caps)
    decision = manager.size_position(
        balance=10.0,
        sl_distance_price=50.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved
    assert "ceiling" in decision.reason


def test_size_position_ignores_ceiling_when_fallback_disabled():
    """A configured max_risk_per_trade_pct has no effect unless
    min_lot_fallback_enabled is also true — the two are independent toggles."""
    caps = RiskCaps(
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=2.0,
        max_open_positions=2,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=3,
        min_lot_fallback_enabled=False,
        max_risk_per_trade_pct=50.0,  # generous, but fallback is off
    )
    manager = make_manager(caps)
    decision = manager.size_position(
        balance=100.0,
        sl_distance_price=50.0,
        contract_size=10.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved


def test_set_min_lot_fallback_updates_caps_live():
    manager = make_manager()  # starts disabled
    decision = manager.size_position(
        balance=100.0,
        sl_distance_price=50.0,
        contract_size=10.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved

    manager.set_min_lot_fallback(enabled=True, max_risk_per_trade_pct=10.0)
    decision = manager.size_position(
        balance=100.0,
        sl_distance_price=50.0,
        contract_size=10.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert decision.approved
    assert decision.volume == 0.01
    # Every other cap is untouched by the live update.
    assert manager.caps.risk_per_trade_pct == CAPS.risk_per_trade_pct
    assert manager.caps.max_open_positions == CAPS.max_open_positions

    manager.set_min_lot_fallback(enabled=False, max_risk_per_trade_pct=None)
    decision = manager.size_position(
        balance=100.0,
        sl_distance_price=50.0,
        contract_size=10.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved


def test_size_position_rejects_non_positive_balance():
    manager = make_manager()
    decision = manager.size_position(
        balance=0.0,
        sl_distance_price=5.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved
    assert "balance" in decision.reason


def test_size_position_rejects_non_positive_sl_distance():
    manager = make_manager()
    decision = manager.size_position(
        balance=10_000.0,
        sl_distance_price=0.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    assert not decision.approved


def test_check_pretrade_blocks_at_max_open_positions():
    manager = make_manager()
    decision = manager.check_pretrade(open_positions_count=2)
    assert not decision.approved
    assert "max open positions" in decision.reason


def test_check_pretrade_allows_when_daily_kill_switch_disabled():
    manager = make_manager()  # CAPS has max_trades_per_day_enabled=False
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    for _ in range(50):
        manager.record_trade_opened(now)
    decision = manager.check_pretrade(open_positions_count=0, now=now)
    assert decision.approved  # CAPS.max_trades_per_day is None -> no numeric cap configured


def test_check_pretrade_blocks_when_daily_kill_switch_enabled():
    caps = dataclasses.replace(CAPS, max_trades_per_day_enabled=True)
    manager = make_manager(caps)
    decision = manager.check_pretrade(open_positions_count=0)
    assert not decision.approved
    assert "max_trades_per_day_enabled" in decision.reason


def test_check_pretrade_blocks_at_max_trades_per_day():
    caps = dataclasses.replace(CAPS, max_trades_per_day=2)
    manager = make_manager(caps)
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    manager.record_trade_opened(now)
    manager.record_trade_opened(now)
    decision = manager.check_pretrade(open_positions_count=0, now=now)
    assert not decision.approved
    assert decision.code == "max_trades_per_day"
    assert "max trades per day" in decision.reason


def test_check_pretrade_allows_below_max_trades_per_day():
    caps = dataclasses.replace(CAPS, max_trades_per_day=2)
    manager = make_manager(caps)
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    manager.record_trade_opened(now)
    decision = manager.check_pretrade(open_positions_count=0, now=now)
    assert decision.approved


def test_max_trades_per_day_resets_on_new_day():
    caps = dataclasses.replace(CAPS, max_trades_per_day=1)
    manager = make_manager(caps)
    day_one = datetime(2026, 7, 11, 23, 55, tzinfo=UTC)
    day_two = datetime(2026, 7, 12, 0, 5, tzinfo=UTC)
    manager.record_trade_opened(day_one)
    decision = manager.check_pretrade(open_positions_count=0, now=day_one)
    assert not decision.approved
    assert decision.code == "max_trades_per_day"
    decision = manager.check_pretrade(open_positions_count=0, now=day_two)
    assert decision.approved  # new day rolled the counter back to 0


def test_consecutive_losses_trip_circuit_breaker():
    manager = make_manager()
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    for _ in range(2):
        manager.record_trade_closed(-10.0, now=now)
    assert not manager.paused
    manager.record_trade_closed(-10.0, now=now)
    assert manager.paused
    assert "consecutive losses" in manager.status.pause_reason


def test_a_win_resets_consecutive_loss_counter():
    manager = make_manager()
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    manager.record_trade_closed(-10.0, now=now)
    manager.record_trade_closed(-10.0, now=now)
    manager.record_trade_closed(50.0, now=now)
    assert manager.status.consecutive_losses == 0
    manager.record_trade_closed(-10.0, now=now)
    manager.record_trade_closed(-10.0, now=now)
    assert not manager.paused  # only 2 in a row since the win


def test_daily_loss_limit_trips_circuit_breaker():
    manager = make_manager()
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    manager.record_trade_closed(-250.0, balance=10_000.0, now=now)  # 2.5% > 2% cap
    assert manager.paused
    assert "daily loss" in manager.status.pause_reason


def test_pretrade_blocked_while_paused():
    manager = make_manager()
    manager.kill("test pause")
    decision = manager.check_pretrade(open_positions_count=0)
    assert not decision.approved
    assert "engine paused" in decision.reason


def test_resume_clears_pause_and_consecutive_losses():
    manager = make_manager()
    now = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    for _ in range(3):
        manager.record_trade_closed(-10.0, now=now)
    assert manager.paused
    manager.resume()
    assert not manager.paused
    assert manager.status.consecutive_losses == 0


def test_daily_counters_reset_on_new_day():
    manager = make_manager()
    day_one = datetime(2026, 7, 11, 23, 55, tzinfo=UTC)
    day_two = datetime(2026, 7, 12, 0, 5, tzinfo=UTC)
    manager.record_trade_opened(day_one)
    assert manager.status.trades_today == 1
    manager.record_trade_opened(day_two)
    assert manager.status.trades_today == 1  # reset, then incremented once


GLOBAL_CAPS = RiskCaps(
    risk_per_trade_pct=0.5,
    daily_loss_limit_pct=50.0,
    max_open_positions=100,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=10,
    min_lot_fallback_enabled=True,
    max_risk_per_trade_pct=2.0,
)


def test_apply_risk_override_tightens_numeric_caps():
    result = apply_risk_override(
        GLOBAL_CAPS,
        {"risk_per_trade_pct": 0.25, "max_open_positions": 5},
    )
    assert result.risk_per_trade_pct == 0.25
    assert result.max_open_positions == 5
    # untouched fields keep the global value
    assert result.daily_loss_limit_pct == GLOBAL_CAPS.daily_loss_limit_pct
    assert result.consecutive_loss_pause == GLOBAL_CAPS.consecutive_loss_pause


@pytest.mark.parametrize(
    "field",
    [
        "risk_per_trade_pct",
        "daily_loss_limit_pct",
        "max_open_positions",
        "consecutive_loss_pause",
    ],
)
def test_apply_risk_override_rejects_loosening_numeric_caps(field):
    looser_value = getattr(GLOBAL_CAPS, field) * 2 + 1
    with pytest.raises(ValueError, match="cannot loosen"):
        apply_risk_override(GLOBAL_CAPS, {field: looser_value})


def test_apply_risk_override_can_enable_daily_kill_switch():
    result = apply_risk_override(GLOBAL_CAPS, {"max_trades_per_day_enabled": True})
    assert result.max_trades_per_day_enabled is True


def test_apply_risk_override_rejects_disabling_daily_kill_switch_when_globally_enabled():
    enabled_global = dataclasses.replace(GLOBAL_CAPS, max_trades_per_day_enabled=True)
    with pytest.raises(ValueError, match="cannot loosen"):
        apply_risk_override(enabled_global, {"max_trades_per_day_enabled": False})


def test_apply_risk_override_can_disable_min_lot_fallback():
    result = apply_risk_override(GLOBAL_CAPS, {"min_lot_fallback_enabled": False})
    assert result.min_lot_fallback_enabled is False


def test_apply_risk_override_rejects_enabling_min_lot_fallback_when_globally_disabled():
    disabled_global = dataclasses.replace(GLOBAL_CAPS, min_lot_fallback_enabled=False)
    with pytest.raises(ValueError, match="cannot loosen"):
        apply_risk_override(disabled_global, {"min_lot_fallback_enabled": True})


def test_apply_risk_override_tightens_max_risk_per_trade_pct():
    result = apply_risk_override(GLOBAL_CAPS, {"max_risk_per_trade_pct": 1.0})
    assert result.max_risk_per_trade_pct == 1.0


def test_apply_risk_override_rejects_loosening_max_risk_per_trade_pct():
    with pytest.raises(ValueError, match="cannot loosen"):
        apply_risk_override(GLOBAL_CAPS, {"max_risk_per_trade_pct": 5.0})


def test_apply_risk_override_rejects_null_max_risk_per_trade_pct():
    with pytest.raises(ValueError, match="cannot loosen"):
        apply_risk_override(GLOBAL_CAPS, {"max_risk_per_trade_pct": None})


def test_apply_risk_override_empty_dict_returns_equivalent_caps():
    result = apply_risk_override(GLOBAL_CAPS, {})
    assert result == GLOBAL_CAPS
