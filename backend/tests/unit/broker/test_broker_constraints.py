"""Broker order-acceptance rules (OBSERVABILITY_PLAN.md Phase 4).

The `stops_level` numbers here are the **real** ones this project measured
from its own broker (`symbol_specs`): Volatility 75 Index reports
`stops_level=10770` on a `point` of 0.01, i.e. a minimum SL/TP distance of
107.70 price units. Every M1 scalp stop of 30-70 units placed on that symbol
was refused live with MT5 retcode 10016 while the backtest reported the same
trades as winners. These tests are what stop that from being possible again.
"""

from __future__ import annotations

import pytest

from src.broker.domain.broker_constraints import (
    REASON_VOLUME_ABOVE_MAX,
    REASON_VOLUME_BELOW_MIN,
    RETCODE_INVALID_STOPS,
    check_stops_level,
    clamp_stops,
    min_stop_distance,
    round_volume,
    widened_spread_points,
)
from src.broker.domain.trading import Side

# The broker's real facts for Volatility 75 Index.
VIX75_POINT = 0.01
VIX75_STOPS_LEVEL = 10770
VIX75_MIN_DISTANCE = 107.70
VIX75_PRICE = 100_000.0


def test_vix75_min_stop_distance_is_107_70_price_units() -> None:
    assert min_stop_distance(VIX75_STOPS_LEVEL, VIX75_POINT) == pytest.approx(VIX75_MIN_DISTANCE)


@pytest.mark.parametrize("sl_distance", [30.0, 45.0, 70.0, 107.69])
def test_m1_scalp_stop_inside_vix75_stops_level_is_rejected(sl_distance: float) -> None:
    """The exact failure mode that killed the VIX75 fleet: an M1 scalp risking
    30-70 units on a symbol demanding 107.70."""
    violation = check_stops_level(
        side=Side.BUY,
        price=VIX75_PRICE,
        sl=VIX75_PRICE - sl_distance,
        tp=VIX75_PRICE + 500.0,
        stops_level=VIX75_STOPS_LEVEL,
        point=VIX75_POINT,
    )
    assert violation is not None
    assert violation.leg == "sl"
    assert violation.distance == pytest.approx(sl_distance)
    assert violation.required == pytest.approx(VIX75_MIN_DISTANCE)


def test_stop_exactly_at_the_minimum_distance_is_accepted() -> None:
    """A stop placed at exactly `stops_level` is legal — floating-point
    rounding must not turn a boundary-legal order into a rejection."""
    assert (
        check_stops_level(
            side=Side.SELL,
            price=VIX75_PRICE,
            sl=VIX75_PRICE + VIX75_MIN_DISTANCE,
            tp=VIX75_PRICE - VIX75_MIN_DISTANCE,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        is None
    )


def test_too_close_take_profit_is_rejected_too() -> None:
    violation = check_stops_level(
        side=Side.BUY,
        price=VIX75_PRICE,
        sl=VIX75_PRICE - 500.0,
        tp=VIX75_PRICE + 50.0,
        stops_level=VIX75_STOPS_LEVEL,
        point=VIX75_POINT,
    )
    assert violation is not None
    assert violation.leg == "tp"


def test_absent_stops_and_zero_stops_level_never_violate() -> None:
    assert (
        check_stops_level(
            side=Side.BUY,
            price=VIX75_PRICE,
            sl=None,
            tp=None,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        is None
    )
    # XAUUSD-style symbols with a tiny stops_level, and brokers with none at
    # all, must not start rejecting perfectly normal scalps.
    assert (
        check_stops_level(
            side=Side.BUY,
            price=2400.0,
            sl=2399.0,
            tp=2402.0,
            stops_level=0,
            point=0.01,
        )
        is None
    )


def test_retcode_for_invalid_stops_is_the_mt5_one() -> None:
    assert RETCODE_INVALID_STOPS == 10016


class TestClampStops:
    def test_widens_a_too_close_buy_stop_to_the_minimum_on_the_right_side(self) -> None:
        sl, tp = clamp_stops(
            side=Side.BUY,
            price=VIX75_PRICE,
            sl=VIX75_PRICE - 40.0,
            tp=VIX75_PRICE + 60.0,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        assert sl == pytest.approx(VIX75_PRICE - VIX75_MIN_DISTANCE)
        assert tp == pytest.approx(VIX75_PRICE + VIX75_MIN_DISTANCE)

    def test_widens_a_sell_the_other_way_round(self) -> None:
        sl, tp = clamp_stops(
            side=Side.SELL,
            price=VIX75_PRICE,
            sl=VIX75_PRICE + 40.0,
            tp=VIX75_PRICE - 60.0,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        assert sl == pytest.approx(VIX75_PRICE + VIX75_MIN_DISTANCE)
        assert tp == pytest.approx(VIX75_PRICE - VIX75_MIN_DISTANCE)

    def test_leaves_already_legal_stops_untouched(self) -> None:
        sl, tp = clamp_stops(
            side=Side.BUY,
            price=VIX75_PRICE,
            sl=VIX75_PRICE - 500.0,
            tp=VIX75_PRICE + 900.0,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        assert (sl, tp) == (VIX75_PRICE - 500.0, VIX75_PRICE + 900.0)

    def test_clamped_stops_then_pass_the_check(self) -> None:
        sl, tp = clamp_stops(
            side=Side.BUY,
            price=VIX75_PRICE,
            sl=VIX75_PRICE - 40.0,
            tp=VIX75_PRICE + 60.0,
            stops_level=VIX75_STOPS_LEVEL,
            point=VIX75_POINT,
        )
        assert (
            check_stops_level(
                side=Side.BUY,
                price=VIX75_PRICE,
                sl=sl,
                tp=tp,
                stops_level=VIX75_STOPS_LEVEL,
                point=VIX75_POINT,
            )
            is None
        )


class TestRoundVolume:
    def test_rounds_down_onto_the_step_grid(self) -> None:
        # Never up: rounding up would size the position larger than the risk
        # manager approved.
        rounded, violation = round_volume(
            0.037, volume_min=0.01, volume_max=10.0, volume_step=0.01
        )
        assert rounded == pytest.approx(0.03)
        assert violation is None

    def test_step_is_respected_exactly_not_approximately(self) -> None:
        rounded, _ = round_volume(0.03, volume_min=0.01, volume_max=10.0, volume_step=0.01)
        assert rounded == 0.03  # not 0.030000000000000002

    def test_vix75_thousandth_lot_step(self) -> None:
        # Volatility 75 Index really does have volume_step 0.001.
        rounded, violation = round_volume(
            0.0257, volume_min=0.01, volume_max=15.0, volume_step=0.001
        )
        assert rounded == pytest.approx(0.025)
        assert violation is None

    def test_volume_rounding_below_minimum_is_a_rejection_not_a_bump(self) -> None:
        rounded, violation = round_volume(
            0.004, volume_min=0.01, volume_max=10.0, volume_step=0.01
        )
        assert rounded == pytest.approx(0.0)
        assert violation is not None
        assert violation.reason == REASON_VOLUME_BELOW_MIN
        assert violation.limit == pytest.approx(0.01)

    def test_volume_above_maximum_is_a_rejection_not_a_clamp(self) -> None:
        _, violation = round_volume(12.0, volume_min=0.01, volume_max=10.0, volume_step=0.01)
        assert violation is not None
        assert violation.reason == REASON_VOLUME_ABOVE_MAX

    def test_coarse_minimum_lot_is_rejected(self) -> None:
        # A synthetic index with a coarser minimum than the forex-sized default
        # is refused outright, which is easy to miss on such a symbol where
        # every other instrument accepts 0.01.
        _, violation = round_volume(0.1, volume_min=0.2, volume_max=80.0, volume_step=0.01)
        assert violation is not None
        assert violation.reason == REASON_VOLUME_BELOW_MIN

    def test_a_zero_step_leaves_the_volume_alone(self) -> None:
        rounded, violation = round_volume(
            0.037, volume_min=0.01, volume_max=10.0, volume_step=0.0
        )
        assert rounded == pytest.approx(0.037)
        assert violation is None


class TestSpreadWidening:
    def test_factor_of_one_reproduces_the_recorded_spread_exactly(self) -> None:
        assert widened_spread_points(37, 1.0) == 37

    def test_factor_below_one_never_narrows_the_spread(self) -> None:
        assert widened_spread_points(37, 0.5) == 37

    def test_widening_rounds_up_so_a_widened_spread_is_never_the_original(self) -> None:
        assert widened_spread_points(37, 1.5) == 56
        assert widened_spread_points(1, 1.1) == 2
