"""Unit tests for the walk-forward harness (OBSERVABILITY_PLAN.md Phase 6
Pass B). `run_backtest` is ALWAYS mocked here — a real run takes ~7 minutes,
and these tests must run in milliseconds regardless of fold count."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from src.backtest.application import walk_forward as walk_forward_module
from src.backtest.application.run_backtest import NoHistoryError
from src.backtest.application.walk_forward import run_walk_forward
from src.backtest.domain.models import BacktestReport, BacktestTrade, BrokerRealism

T0 = datetime(2025, 1, 1, tzinfo=UTC)


def make_trade(profit: float) -> BacktestTrade:
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
        r_multiple=None,
    )


def make_report(
    period: str,
    trades: tuple[BacktestTrade, ...] = (),
    profit_factor: float = 0.0,
    win_rate: float = 0.0,
    broker_realism: BrokerRealism | None = None,
) -> BacktestReport:
    return BacktestReport(
        strategy="breakout_v1",
        symbol="XAUUSD",
        period=period,
        starting_balance=10_000.0,
        ending_balance=10_000.0 + sum(t.profit for t in trades),
        trades=trades,
        equity_curve=(),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=0.0,
        avg_r=0.0,
        worst_losing_streak=0,
        broker_realism=broker_realism if broker_realism is not None else BrokerRealism(),
    )


async def test_run_walk_forward_runs_one_fold_per_split_period(monkeypatch):
    calls: list[str] = []

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        calls.append(period)
        return make_report(period, trades=(make_trade(10.0),), profit_factor=2.0, win_rate=1.0)

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-03", fold_months=1)

    assert calls == ["2025-01:2025-01", "2025-02:2025-02", "2025-03:2025-03"]
    assert report.fold_count == 3
    assert [f.period for f in report.folds] == calls
    for fold in report.folds:
        assert fold.trade_count == 1
        assert fold.win_rate == 1.0
        assert fold.profit_factor == 2.0
        assert fold.expectancy == pytest.approx(10.0)


async def test_run_walk_forward_passes_kwargs_through_to_every_fold(monkeypatch):
    captured: list[dict] = []

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        captured.append(kwargs)
        return make_report(period)

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    await run_walk_forward(
        "breakout_v1",
        "XAUUSD",
        "2025-01:2025-02",
        fold_months=1,
        starting_balance=5_000.0,
        database_url="sqlite:///./data/test.db",
        min_lot_fallback_enabled=True,
        max_risk_per_trade_pct=3.0,
        min_rr=1.5,
        simulate_broker_constraints=False,
        clamp_stops=True,
        spread_widening_factor=2.0,
        slippage_samples=(0.1, 0.2),
        slippage_seed=99,
    )

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["starting_balance"] == 5_000.0
        assert kwargs["database_url"] == "sqlite:///./data/test.db"
        assert kwargs["min_lot_fallback_enabled"] is True
        assert kwargs["max_risk_per_trade_pct"] == 3.0
        assert kwargs["min_rr"] == 1.5
        assert kwargs["simulate_broker_constraints"] is False
        assert kwargs["clamp_stops"] is True
        assert kwargs["spread_widening_factor"] == 2.0
        assert kwargs["slippage_samples"] == (0.1, 0.2)
        assert kwargs["slippage_seed"] == 99


async def test_run_walk_forward_skips_fold_with_no_history_not_fatal(monkeypatch):
    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        if period == "2025-02:2025-02":
            raise NoHistoryError(f"no history for {period}")
        return make_report(period, trades=(make_trade(5.0),), profit_factor=1.5, win_rate=1.0)

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-03", fold_months=1)

    assert report.fold_count == 2
    assert [f.period for f in report.folds] == ["2025-01:2025-01", "2025-03:2025-03"]


async def test_run_walk_forward_all_folds_skipped_yields_empty_report_no_crash(monkeypatch):
    async def always_no_history(strategy_name, symbol, period, **kwargs):
        raise NoHistoryError("no history at all")

    monkeypatch.setattr(walk_forward_module, "run_backtest", always_no_history)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-03", fold_months=1)

    assert report.fold_count == 0
    assert report.folds == ()
    assert report.folds_with_zero_trades == 0
    assert report.losing_fold_count == 0
    assert report.mean_profit_factor is None
    assert report.stddev_profit_factor is None
    assert report.worst_fold_profit_factor is None
    assert report.mean_expectancy == 0.0


async def test_run_walk_forward_all_zero_trade_folds_no_crash(monkeypatch):
    """Every fold runs (no NoHistoryError) but produces zero trades — must
    not divide by zero computing the aggregate."""

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        return make_report(period)  # no trades

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-03", fold_months=1)

    assert report.fold_count == 3
    assert report.folds_with_zero_trades == 3
    assert report.losing_fold_count == 0
    assert report.mean_profit_factor is None
    assert report.stddev_profit_factor is None
    assert report.worst_fold_profit_factor is None
    assert report.mean_expectancy == 0.0
    assert all(f.profit_factor is None for f in report.folds)


async def test_run_walk_forward_normalizes_infinite_and_nan_profit_factor_to_none(monkeypatch):
    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        if period == "2025-01:2025-01":
            return make_report(
                period, trades=(make_trade(10.0),), profit_factor=float("inf"), win_rate=1.0
            )
        return make_report(
            period, trades=(make_trade(-5.0),), profit_factor=float("nan"), win_rate=0.0
        )

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-02", fold_months=1)

    assert report.fold_count == 2
    assert all(f.profit_factor is None for f in report.folds)
    # Both folds had a real (nonzero) trade, so they don't count as zero-trade folds
    # even though their profit factor was undefined.
    assert report.folds_with_zero_trades == 0
    assert report.mean_profit_factor is None
    assert report.stddev_profit_factor is None
    assert report.worst_fold_profit_factor is None


async def test_run_walk_forward_aggregation_math_mixed_folds(monkeypatch):
    reports_by_period = {
        "2025-01:2025-01": make_report(
            "2025-01:2025-01", trades=(make_trade(5.0),), profit_factor=2.0, win_rate=1.0
        ),
        "2025-02:2025-02": make_report("2025-02:2025-02"),  # zero trades -> PF None
        "2025-03:2025-03": make_report(
            "2025-03:2025-03", trades=(make_trade(-3.0),), profit_factor=4.0, win_rate=0.0
        ),
        "2025-04:2025-04": make_report(
            "2025-04:2025-04", trades=(make_trade(10.0),), profit_factor=float("inf"), win_rate=1.0
        ),
    }

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        return reports_by_period[period]

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-04", fold_months=1)

    assert report.fold_count == 4
    assert report.folds_with_zero_trades == 1
    assert report.losing_fold_count == 1  # fold 3, expectancy -3.0
    # Only folds 1 and 3 have a finite, defined profit factor (2.0 and 4.0).
    assert report.mean_profit_factor == pytest.approx(3.0)
    assert report.stddev_profit_factor == pytest.approx(1.0)  # pstdev([2.0, 4.0])
    assert report.worst_fold_profit_factor == pytest.approx(2.0)
    assert report.mean_expectancy == pytest.approx((5.0 + 0.0 - 3.0 + 10.0) / 4)


async def test_run_walk_forward_broker_acceptance_rate_from_report(monkeypatch):
    realism = BrokerRealism(enabled=True, accepted_count=8, rejected_count=2)

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        return make_report(period, trades=(make_trade(1.0),), broker_realism=realism)

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-01", fold_months=1)

    (fold,) = report.folds
    assert fold.broker_acceptance_rate == pytest.approx(0.8)


async def test_run_walk_forward_propagates_non_no_history_errors(monkeypatch):
    calls = {"n": 0}

    async def raises_value_error(strategy_name, symbol, period, **kwargs):
        calls["n"] += 1
        raise ValueError("unknown strategy: 'nope'")

    monkeypatch.setattr(walk_forward_module, "run_backtest", raises_value_error)

    with pytest.raises(ValueError, match="unknown strategy"):
        await run_walk_forward("nope", "XAUUSD", "2025-01:2025-03", fold_months=1)

    # Fails fast on the first fold rather than trying every fold.
    assert calls["n"] == 1


async def test_run_walk_forward_single_fold_covering_whole_period(monkeypatch):
    calls: list[str] = []

    async def fake_run_backtest(strategy_name, symbol, period, **kwargs):
        calls.append(period)
        return make_report(period, trades=(make_trade(1.0),), profit_factor=1.0, win_rate=1.0)

    monkeypatch.setattr(walk_forward_module, "run_backtest", fake_run_backtest)

    report = await run_walk_forward("breakout_v1", "XAUUSD", "2025-01:2025-06", fold_months=6)

    assert calls == ["2025-01:2025-06"]
    assert report.fold_count == 1


def test_normalized_profit_factor_helper_directly():
    finite = make_report("2025-01:2025-01", trades=(make_trade(1.0),), profit_factor=1.5)
    infinite = make_report("2025-01:2025-01", trades=(make_trade(1.0),), profit_factor=math.inf)
    zero_trades = make_report("2025-01:2025-01", profit_factor=0.0)

    assert walk_forward_module._normalized_profit_factor(finite) == 1.5
    assert walk_forward_module._normalized_profit_factor(infinite) is None
    assert walk_forward_module._normalized_profit_factor(zero_trades) is None
