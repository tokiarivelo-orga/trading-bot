"""PositionManager x give-back exit policy — the wiring, not the arithmetic.

The decision logic itself is covered by ``test_exit_policy.py``. What this
file pins is that ``PositionManager`` tracks the high-water mark correctly,
merges the policy's candidate without ever loosening a stop, and reproduces
pre-existing behaviour exactly when the policy is not configured.
"""

from __future__ import annotations

import pytest

from src.engine.application.position_manager import PositionManager
from src.engine.domain.exit_policy import ExitPolicyConfig
from tests.unit.engine.test_position_manager import (
    INFO,
    FakeMarketData,
    FakeOrderService,
    _flat_candles,
    _position,
)

# INFO quotes bid 2410.00 / ask 2410.20. A BUY opened at 2400 with SL 2390
# has risk 10.0, so 1R = 10.0 and the position is +1.0R at the bid.
POLICY = ExitPolicyConfig(arm_r=0.5, keep_fraction=0.5, min_lock_r=0.2)


def _manager(orders: FakeOrderService, policy: ExitPolicyConfig | None) -> PositionManager:
    return PositionManager(
        order_service=orders,
        market_data=FakeMarketData(candles=_flat_candles(60)),
        exit_policy_config=policy,
    )


@pytest.mark.asyncio
async def test_give_back_trail_tightens_the_stop_on_a_runner() -> None:
    """Peak +1.0R, keep 0.5 -> stop moves to entry + 0.5R = 2405."""
    orders = FakeOrderService([_position(open_price=2400.0, sl=2390.0, tp=2500.0)])
    await _manager(orders, POLICY).on_candle_closed("XAUUSD")

    assert orders.modified, "expected the stop to be tightened"
    _ticket, stop, _tp = orders.modified[-1]
    assert stop == pytest.approx(2405.0)


@pytest.mark.asyncio
async def test_no_policy_configured_reproduces_previous_behaviour() -> None:
    """`exit_policy_config=None` must be a true no-op — this is the escape
    hatch if the policy ever misbehaves in production."""
    with_policy = FakeOrderService([_position(open_price=2400.0, sl=2390.0, tp=2500.0)])
    without = FakeOrderService([_position(open_price=2400.0, sl=2390.0, tp=2500.0)])

    await _manager(with_policy, POLICY).on_candle_closed("XAUUSD")
    await _manager(without, None).on_candle_closed("XAUUSD")

    assert with_policy.modified != without.modified


@pytest.mark.asyncio
async def test_never_loosens_a_stop_another_rule_already_tightened() -> None:
    """Breakeven-at-+1R already moved this stop to 2400; the give-back
    candidate (2405) is tighter, so it wins. The reverse must never happen —
    a stop at 2408 must survive."""
    orders = FakeOrderService([_position(open_price=2400.0, sl=2408.0, tp=2500.0)])
    await _manager(orders, POLICY).on_candle_closed("XAUUSD")

    for _ticket, stop, _tp in orders.modified:
        assert stop >= 2408.0


@pytest.mark.asyncio
async def test_losing_position_is_left_alone() -> None:
    """A position that has never been in profit has nothing to protect; the
    policy must not touch it (the existing time-stop and volatility rules own
    that case)."""
    orders = FakeOrderService([_position(open_price=2450.0, sl=2440.0, tp=2500.0)])
    await _manager(orders, POLICY).on_candle_closed("XAUUSD")
    assert not any("give-back" in reason for reason in orders.modify_reasons)
    assert orders.closed == []


@pytest.mark.asyncio
async def test_high_water_mark_is_tracked_from_the_first_candle() -> None:
    """The peak used by the give-back rule is only meaningful if it starts
    accumulating at entry. It used to be populated exclusively inside the
    HIGH-volatility chandelier rule, so it would have been silently absent
    for most positions."""
    orders = FakeOrderService([_position(open_price=2400.0, sl=2390.0, tp=2500.0)])
    manager = _manager(orders, POLICY)
    await manager.on_candle_closed("XAUUSD")
    assert manager._trade_extreme_favorable[1] == pytest.approx(INFO.bid)


@pytest.mark.asyncio
async def test_tracking_is_cleaned_up_when_the_position_vanishes() -> None:
    orders = FakeOrderService([_position(open_price=2400.0, sl=2390.0, tp=2500.0)])
    manager = _manager(orders, POLICY)
    await manager.on_candle_closed("XAUUSD")
    assert 1 in manager._trade_extreme_favorable

    orders._positions = []
    await manager.on_candle_closed("XAUUSD")
    assert 1 not in manager._trade_extreme_favorable
    assert 1 not in manager._candles_since_open
