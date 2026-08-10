from datetime import UTC, datetime

import pytest

from src.backtest.application import metrics
from src.backtest.domain.models import BacktestTrade, EquityPoint

T0 = datetime(2025, 1, 1, tzinfo=UTC)


def make_trade(profit: float, r_multiple: float | None = None) -> BacktestTrade:
    return BacktestTrade(
        side="buy",
        volume=1.0,
        open_time=T0,
        open_price=100.0,
        sl=99.0,
        tp=102.0,
        close_time=T0,
        close_price=100.0 + profit,
        profit=profit,
        r_multiple=r_multiple,
    )


def test_win_rate_empty_is_zero():
    assert metrics.win_rate(()) == 0.0


def test_win_rate_counts_positive_profit_only():
    trades = (make_trade(10.0), make_trade(-5.0), make_trade(0.0))
    assert metrics.win_rate(trades) == 1 / 3


def test_profit_factor_no_losses_is_infinite():
    trades = (make_trade(10.0), make_trade(5.0))
    assert metrics.profit_factor(trades) == float("inf")


def test_profit_factor_no_trades_is_zero():
    assert metrics.profit_factor(()) == 0.0


def test_profit_factor_ratio_of_gross_profit_to_gross_loss():
    trades = (make_trade(20.0), make_trade(-10.0))
    assert metrics.profit_factor(trades) == 2.0


def test_max_drawdown_pct_tracks_worst_peak_to_trough():
    curve = (
        EquityPoint(time=T0, balance=100.0),
        EquityPoint(time=T0, balance=150.0),
        EquityPoint(time=T0, balance=120.0),
        EquityPoint(time=T0, balance=180.0),
    )
    # worst drop: 150 -> 120 = 20%
    assert metrics.max_drawdown_pct(curve) == 20.0


def test_max_drawdown_pct_empty_is_zero():
    assert metrics.max_drawdown_pct(()) == 0.0


def test_avg_r_ignores_trades_with_no_r_multiple():
    trades = (make_trade(10.0, r_multiple=2.0), make_trade(-5.0, r_multiple=-1.0), make_trade(3.0))
    assert metrics.avg_r(trades) == 0.5


def test_avg_r_no_trades_is_zero():
    assert metrics.avg_r(()) == 0.0


def test_worst_losing_streak_counts_consecutive_losses():
    trades = (
        make_trade(-1.0),
        make_trade(-1.0),
        make_trade(1.0),
        make_trade(-1.0),
        make_trade(-1.0),
        make_trade(-1.0),
    )
    assert metrics.worst_losing_streak(trades) == 3


def test_worst_losing_streak_no_losses_is_zero():
    assert metrics.worst_losing_streak((make_trade(1.0), make_trade(2.0))) == 0


def test_expectancy_no_trades_is_zero():
    assert metrics.expectancy(()) == 0.0


def test_expectancy_averages_profit_across_all_trades():
    trades = (make_trade(20.0), make_trade(-10.0), make_trade(5.0))
    assert metrics.expectancy(trades) == pytest.approx(5.0)


def test_expectancy_negative_when_losing_on_average():
    trades = (make_trade(1.0), make_trade(-10.0))
    assert metrics.expectancy(trades) == pytest.approx(-4.5)
