from datetime import UTC, datetime

import pytest

from src.backtest.application.period import InvalidPeriod, parse_period, split_into_folds


def test_parse_period_single_month():
    start, end = parse_period("2025-01:2025-01")
    assert start == datetime(2025, 1, 1, tzinfo=UTC)
    assert end == datetime(2025, 2, 1, tzinfo=UTC)


def test_parse_period_multi_month_handles_month_lengths():
    start, end = parse_period("2025-01:2025-02")
    assert start == datetime(2025, 1, 1, tzinfo=UTC)
    assert end == datetime(2025, 3, 1, tzinfo=UTC)


def test_parse_period_leap_year_february():
    start, end = parse_period("2024-02:2024-02")
    assert start == datetime(2024, 2, 1, tzinfo=UTC)
    assert end == datetime(2024, 3, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "period",
    ["2025-01", "2025-01:2025-02:2025-03", "not-a-period", "2025-13:2025-14", ""],
)
def test_parse_period_invalid_raises(period):
    with pytest.raises(InvalidPeriod):
        parse_period(period)


def test_parse_period_end_before_start_raises():
    with pytest.raises(InvalidPeriod):
        parse_period("2025-06:2025-01")


# ── split_into_folds (OBSERVABILITY_PLAN.md Phase 6 Pass B) ──────────────────


def test_split_into_folds_even_monthly_split():
    assert split_into_folds("2025-01:2025-03", fold_months=1) == [
        "2025-01:2025-01",
        "2025-02:2025-02",
        "2025-03:2025-03",
    ]


def test_split_into_folds_even_multi_month_split():
    assert split_into_folds("2025-01:2025-06", fold_months=2) == [
        "2025-01:2025-02",
        "2025-03:2025-04",
        "2025-05:2025-06",
    ]


def test_split_into_folds_uneven_remainder_shortens_last_fold():
    assert split_into_folds("2025-01:2025-05", fold_months=2) == [
        "2025-01:2025-02",
        "2025-03:2025-04",
        "2025-05:2025-05",
    ]


def test_split_into_folds_spans_year_boundary():
    assert split_into_folds("2025-11:2026-02", fold_months=1) == [
        "2025-11:2025-11",
        "2025-12:2025-12",
        "2026-01:2026-01",
        "2026-02:2026-02",
    ]


def test_split_into_folds_single_fold_covering_whole_period():
    assert split_into_folds("2025-01:2025-06", fold_months=6) == ["2025-01:2025-06"]


def test_split_into_folds_fold_months_larger_than_period_yields_one_shorter_fold():
    assert split_into_folds("2025-01:2025-03", fold_months=12) == ["2025-01:2025-03"]


def test_split_into_folds_default_fold_months_is_one():
    assert split_into_folds("2025-01:2025-02") == ["2025-01:2025-01", "2025-02:2025-02"]


@pytest.mark.parametrize("fold_months", [0, -1, -12])
def test_split_into_folds_fold_months_below_one_raises(fold_months):
    with pytest.raises(InvalidPeriod):
        split_into_folds("2025-01:2025-06", fold_months=fold_months)


@pytest.mark.parametrize(
    "period",
    ["2025-01", "2025-01:2025-02:2025-03", "not-a-period", "2025-13:2025-14", ""],
)
def test_split_into_folds_malformed_period_raises(period):
    with pytest.raises(InvalidPeriod):
        split_into_folds(period)


def test_split_into_folds_end_before_start_raises():
    with pytest.raises(InvalidPeriod):
        split_into_folds("2025-06:2025-01")
