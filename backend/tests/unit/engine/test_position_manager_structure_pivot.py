"""PositionManager x structure-pivot profit-lock rule — the wiring, not the
pivot arithmetic (that's `test_structure_pivot.py`). Mirrors
`test_position_manager_giveback.py`'s split for the give-back rule: this
file pins that `PositionManager` reuses its existing `secure_timeframe`
candle fetch to compute pivots, arms only once peak R clears `arm_r`,
selects only pivots formed after the position's own entry, merges the
candidate through `_improves` (never loosens a stop another rule already
set), and reproduces pre-existing behaviour exactly when
`structure_pivot_config=None`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.broker.domain.trading import Side
from src.engine.application.position_manager import PositionManager
from src.engine.domain.structure_pivot import StructurePivotConfig
from tests.unit.engine.test_position_manager import (
    INFO,
    FakeMarketData,
    FakeOrderService,
    _candle,
    _flat_candles,
    _position,
)

CONFIG = StructurePivotConfig(pivot_bars=1, arm_r=1.0, buffer_r_mult=0.1, min_pivot_r=0.0)


def _uptrend_pivot_candles():
    """39 flat warmup bars (indices 0-38, no pivots possible — every bar is
    identical, so nothing is ever strictly the extreme of its neighbours),
    then a small zigzag that forms exactly one Higher-Low at index 42
    (2402.0 at index 40, unlabeled — no predecessor yet — then 2406.0 at
    index 42, labeled HL because it's higher than 2402.0). Returns
    `(candles, open_time)` where `open_time` is the last warmup bar's time,
    so the whole zigzag counts as 'formed after entry'."""
    bars = _flat_candles(39)
    open_time = bars[-1].time
    bars.append(_candle(39, 2400.4, 2410.0, 2405.0, 2408.0))
    bars.append(_candle(40, 2408.0, 2409.0, 2402.0, 2403.0))  # swing low #1 (unlabeled)
    bars.append(_candle(41, 2403.0, 2415.0, 2410.0, 2413.0))
    bars.append(_candle(42, 2413.0, 2411.0, 2406.0, 2407.0))  # swing low #2 -> HL @ 2406.0
    bars.append(_candle(43, 2407.0, 2420.0, 2415.0, 2418.0))
    return bars, open_time


def _downtrend_pivot_candles():
    """Mirror of `_uptrend_pivot_candles` for a sell: a Lower-High forms at
    index 42 (2394.0), lower than the first (unlabeled) swing high at index
    40 (2398.0)."""
    bars = _flat_candles(39)
    open_time = bars[-1].time
    bars.append(_candle(39, 2399.6, 2395.0, 2390.0, 2392.0))
    bars.append(_candle(40, 2392.0, 2398.0, 2391.0, 2397.0))  # swing high #1 (unlabeled)
    bars.append(_candle(41, 2397.0, 2390.0, 2385.0, 2387.0))
    bars.append(_candle(42, 2387.0, 2394.0, 2389.0, 2393.0))  # swing high #2 -> LH @ 2394.0
    bars.append(_candle(43, 2393.0, 2380.0, 2375.0, 2378.0))
    return bars, open_time


def _manager(
    orders: FakeOrderService, candles, config: StructurePivotConfig | None
) -> PositionManager:
    return PositionManager(
        order_service=orders,
        market_data=FakeMarketData(candles=candles),
        structure_pivot_config=config,
    )


# INFO quotes bid 2410.00 / ask 2410.20 (see test_position_manager.py). A BUY
# opened at 2400 with SL 2390 has risk 10.0, so peak R at the bid is exactly
# 1.0 -- equal to CONFIG.arm_r, and Rule 1 (breakeven at +1R) also fires,
# putting the floor this rule must beat at 2400.21 (entry + the
# spread/stops_level buffer -- see test_position_manager.py).


@pytest.mark.asyncio
async def test_ratchets_sl_to_just_beyond_a_fresh_hl_for_a_buy() -> None:
    candles, open_time = _uptrend_pivot_candles()
    position = _position(open_price=2400.0, sl=2390.0, tp=2500.0, open_time=open_time)
    orders = FakeOrderService([position])

    await _manager(orders, candles, CONFIG).on_candle_closed("XAUUSD")

    assert orders.modified, "expected the stop to be tightened"
    _ticket, stop, _tp = orders.modified[-1]
    # pivot 2406.0 - buffer(0.1 * risk 10.0 = 1.0) = 2405.0, beating the
    # +1R breakeven floor of 2400.21.
    assert stop == pytest.approx(2405.0)
    assert orders.modify_reasons[-1] == "structure-pivot profit lock"


@pytest.mark.asyncio
async def test_ratchets_sl_to_just_beyond_a_fresh_lh_for_a_sell() -> None:
    candles, open_time = _downtrend_pivot_candles()
    # A SELL at 2400 with SL 2410 has risk 10.0; INFO's ask (2410.20) makes
    # the mark for a sell 2410.20 -- worse than entry, so this test uses an
    # explicit lower ask via a custom SymbolInfo so the position is +1R too.
    info = replace(INFO, bid=2389.80, ask=2390.00)
    position = _position(
        open_price=2400.0, sl=2410.0, tp=2300.0, side=Side.SELL, open_time=open_time
    )
    orders = FakeOrderService([position])
    manager = PositionManager(
        order_service=orders,
        market_data=FakeMarketData(info=info, candles=candles),
        structure_pivot_config=CONFIG,
    )

    await manager.on_candle_closed("XAUUSD")

    assert orders.modified, "expected the stop to be tightened"
    _ticket, stop, _tp = orders.modified[-1]
    # pivot 2394.0 + buffer(0.1 * risk 10.0 = 1.0) = 2395.0, beating the
    # +1R breakeven floor of 2400.21.
    assert stop == pytest.approx(2395.0)
    assert orders.modify_reasons[-1] == "structure-pivot profit lock"


@pytest.mark.asyncio
async def test_no_config_reproduces_previous_behaviour() -> None:
    """`structure_pivot_config=None` must be a true no-op — the same
    escape hatch `exit_policy_config=None`/`volatility_config=None` already
    give the other optional rules."""
    candles, open_time = _uptrend_pivot_candles()
    with_config = FakeOrderService(
        [_position(open_price=2400.0, sl=2390.0, tp=2500.0, open_time=open_time)]
    )
    without = FakeOrderService(
        [_position(open_price=2400.0, sl=2390.0, tp=2500.0, open_time=open_time)]
    )

    await _manager(with_config, candles, CONFIG).on_candle_closed("XAUUSD")
    await _manager(without, candles, None).on_candle_closed("XAUUSD")

    assert with_config.modified != without.modified
    # Without the rule, only the +1R breakeven fires (SL -> entry + the
    # spread/stops_level buffer, 2400.21 — see
    # test_position_manager.test_moves_sl_to_breakeven_once_risk_is_covered).
    assert without.modified == [(1, 2400.21, 2500.0)]


@pytest.mark.asyncio
async def test_does_not_arm_below_arm_r() -> None:
    """Peak R below `arm_r` (the TP1-reached proxy) must not ratchet to
    structure yet, even when a qualifying pivot exists."""
    candles, open_time = _uptrend_pivot_candles()
    position = _position(open_price=2350.0, sl=2340.0, tp=2500.0, open_time=open_time)
    # risk = 10.0, mark = INFO.bid = 2410.0 -> peak R = 6.0, comfortably
    # above arm_r=1.0, so instead raise arm_r past that to prove the gate.
    config = StructurePivotConfig(pivot_bars=1, arm_r=100.0, buffer_r_mult=0.1)
    orders = FakeOrderService([position])

    await _manager(orders, candles, config).on_candle_closed("XAUUSD")

    assert not any("structure-pivot" in reason for reason in orders.modify_reasons)


@pytest.mark.asyncio
async def test_ignores_a_pivot_that_predates_entry() -> None:
    """A position opened *after* the zigzag already formed must not ratchet
    to that stale pivot — only pivots formed since this position's own
    entry qualify (`select_ratchet_pivot`'s `entered_after` filter)."""
    candles, _ = _uptrend_pivot_candles()
    late_open_time = candles[-1].time  # opened after every pivot already formed
    position = _position(open_price=2400.0, sl=2390.0, tp=2500.0, open_time=late_open_time)
    orders = FakeOrderService([position])

    await _manager(orders, candles, CONFIG).on_candle_closed("XAUUSD")

    assert not any("structure-pivot" in reason for reason in orders.modify_reasons)
    # The +1R breakeven rule still fires on its own (SL -> entry + the
    # spread/stops_level buffer, 2400.21).
    assert orders.modified == [(1, 2400.21, 2500.0)]


@pytest.mark.asyncio
async def test_never_loosens_a_stop_another_rule_already_tightened() -> None:
    """A stop already tightened past this rule's own candidate (2405.0)
    must survive untouched — the same never-loosen invariant every other
    SL-tightening rule obeys via `_improves`."""
    candles, open_time = _uptrend_pivot_candles()
    position = _position(open_price=2400.0, sl=2408.0, tp=2500.0, open_time=open_time)
    orders = FakeOrderService([position])

    await _manager(orders, candles, CONFIG).on_candle_closed("XAUUSD")

    for _ticket, stop, _tp in orders.modified:
        assert stop >= 2408.0


@pytest.mark.asyncio
async def test_structure_pivot_candidate_looser_than_existing_sl_does_not_win() -> None:
    """The mirror case: this rule's own candidate (2405.0) is *less*
    protective than a stop another rule (breakeven, here simulated by an
    already-tighter manual SL) already set at 2406.0 -- the merge must keep
    2406.0, not fall back to the looser 2405.0."""
    candles, open_time = _uptrend_pivot_candles()
    position = _position(open_price=2400.0, sl=2406.0, tp=2500.0, open_time=open_time)
    orders = FakeOrderService([position])

    await _manager(orders, candles, CONFIG).on_candle_closed("XAUUSD")

    for _ticket, stop, _tp in orders.modified:
        assert stop >= 2406.0
