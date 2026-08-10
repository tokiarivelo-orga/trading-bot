"""Pure JSON round-trip tests for the walk-forward report writer
(OBSERVABILITY_PLAN.md Phase 6 Pass B) — no backtest run involved at all."""

from __future__ import annotations

import dataclasses
import json
import math

from src.backtest.application.walk_forward import FoldResult, WalkForwardReport
from src.backtest.reports.walk_forward_writer import (
    render_walk_forward_summary,
    write_walk_forward_report,
)


def make_report() -> WalkForwardReport:
    folds = (
        FoldResult(
            period="2025-01:2025-01",
            trade_count=4,
            win_rate=0.75,
            profit_factor=2.5,
            expectancy=12.5,
            broker_acceptance_rate=0.9,
        ),
        FoldResult(
            period="2025-02:2025-02",
            trade_count=0,
            win_rate=0.0,
            profit_factor=None,
            expectancy=0.0,
            broker_acceptance_rate=0.0,
        ),
    )
    return WalkForwardReport(
        strategy="breakout_v1",
        symbol="XAUUSD",
        period="2025-01:2025-02",
        fold_months=1,
        starting_balance=10_000.0,
        folds=folds,
        fold_count=2,
        folds_with_zero_trades=1,
        losing_fold_count=0,
        mean_profit_factor=2.5,
        stddev_profit_factor=0.0,
        worst_fold_profit_factor=2.5,
        mean_expectancy=6.25,
    )


def test_write_walk_forward_report_round_trips_all_fields(tmp_path):
    report = make_report()
    path = write_walk_forward_report(report, tmp_path)

    assert path.is_file()
    assert path.parent == tmp_path

    data = json.loads(path.read_text())
    assert data["strategy"] == "breakout_v1"
    assert data["symbol"] == "XAUUSD"
    assert data["period"] == "2025-01:2025-02"
    assert data["fold_months"] == 1
    assert data["starting_balance"] == 10_000.0
    assert data["fold_count"] == 2
    assert data["folds_with_zero_trades"] == 1
    assert data["losing_fold_count"] == 0
    assert data["mean_profit_factor"] == 2.5
    assert data["stddev_profit_factor"] == 0.0
    assert data["worst_fold_profit_factor"] == 2.5
    assert data["mean_expectancy"] == 6.25

    assert len(data["folds"]) == 2
    first, second = data["folds"]
    assert first == {
        "period": "2025-01:2025-01",
        "trade_count": 4,
        "win_rate": 0.75,
        "profit_factor": 2.5,
        "expectancy": 12.5,
        "broker_acceptance_rate": 0.9,
    }
    assert second["profit_factor"] is None
    assert second["trade_count"] == 0


def test_write_walk_forward_report_creates_directory(tmp_path):
    reports_dir = tmp_path / "nested" / "walk_forward"
    assert not reports_dir.exists()

    write_walk_forward_report(make_report(), reports_dir)

    assert reports_dir.is_dir()


def test_write_walk_forward_report_filename_matches_strategy_symbol_period(tmp_path):
    path = write_walk_forward_report(make_report(), tmp_path)
    assert path.name == "breakout_v1_XAUUSD_2025-01_2025-02.json"


def test_write_walk_forward_report_serializes_none_aggregate_fields_as_null(tmp_path):
    report = dataclasses.replace(
        make_report(),
        mean_profit_factor=None,
        stddev_profit_factor=None,
        worst_fold_profit_factor=None,
    )
    path = write_walk_forward_report(report, tmp_path)

    data = json.loads(path.read_text())
    assert data["mean_profit_factor"] is None
    assert data["stddev_profit_factor"] is None
    assert data["worst_fold_profit_factor"] is None


def test_write_walk_forward_report_serializes_infinite_float_as_null(tmp_path):
    """Defensive: to_jsonable normalizes inf/nan even though
    `_normalized_profit_factor` in walk_forward.py is supposed to have
    already converted them to None before a FoldResult is ever built."""
    report = dataclasses.replace(make_report(), mean_profit_factor=math.inf)
    path = write_walk_forward_report(report, tmp_path)

    data = json.loads(path.read_text())
    assert data["mean_profit_factor"] is None


def test_render_walk_forward_summary_includes_headline_stats():
    summary = render_walk_forward_summary(make_report())

    assert "breakout_v1" in summary
    assert "XAUUSD" in summary
    assert "2025-01:2025-02" in summary
    assert "Folds run:            2" in summary
    assert "Folds w/ zero trades: 1" in summary
    assert "Mean profit factor:   2.50" in summary
    assert "2025-01:2025-01" in summary
    assert "2025-02:2025-02" in summary


def test_render_walk_forward_summary_handles_none_aggregate_values():
    report = dataclasses.replace(
        make_report(),
        mean_profit_factor=None,
        stddev_profit_factor=None,
        worst_fold_profit_factor=None,
    )
    summary = render_walk_forward_summary(report)
    assert "n/a" in summary


def test_render_walk_forward_summary_handles_empty_folds():
    report = dataclasses.replace(
        make_report(),
        folds=(),
        fold_count=0,
        folds_with_zero_trades=0,
        losing_fold_count=0,
        mean_profit_factor=None,
        stddev_profit_factor=None,
        worst_fold_profit_factor=None,
        mean_expectancy=0.0,
    )
    summary = render_walk_forward_summary(report)
    assert "every fold was skipped" in summary
