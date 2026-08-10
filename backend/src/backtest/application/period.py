"""Parses the CLI's `YYYY-MM:YYYY-MM` period argument into a UTC [start, end) range."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime


class InvalidPeriod(Exception):
    pass


def parse_period(period: str) -> tuple[datetime, datetime]:
    """`"2025-01:2025-06"` -> (2025-01-01, 2025-07-01), both UTC midnight —
    `end` is exclusive, one day past the last day of the end month."""
    parts = period.split(":")
    if len(parts) != 2:
        raise InvalidPeriod(f"expected 'YYYY-MM:YYYY-MM', got {period!r}")
    try:
        start = _month_start(parts[0])
        end_month_start = _month_start(parts[1])
    except ValueError as exc:
        raise InvalidPeriod(f"expected 'YYYY-MM:YYYY-MM', got {period!r}: {exc}") from exc
    end = _next_month(end_month_start)
    if end <= start:
        raise InvalidPeriod(f"end of period must be after start: {period!r}")
    return start, end


def _month_start(token: str) -> datetime:
    year_str, month_str = token.split("-")
    return datetime(int(year_str), int(month_str), 1, tzinfo=UTC)


def _next_month(month_start: datetime) -> datetime:
    days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
    epoch = int(month_start.timestamp()) + days_in_month * 86400
    return datetime.fromtimestamp(epoch, tz=UTC)


def _add_months(month_start: datetime, months: int) -> datetime:
    """`month_start` advanced by `months` whole calendar months, always
    landing on a month's first day — used to size walk-forward folds
    (OBSERVABILITY_PLAN.md Phase 6 Pass B) without reimplementing month
    arithmetic `_next_month` already does one step at a time."""
    zero_based = month_start.month - 1 + months
    year = month_start.year + zero_based // 12
    month = zero_based % 12 + 1
    return datetime(year, month, 1, tzinfo=UTC)


def split_into_folds(period: str, fold_months: int = 1) -> list[str]:
    """Splits `period` (`"YYYY-MM:YYYY-MM"`) into consecutive
    `fold_months`-sized `"YYYY-MM:YYYY-MM"` windows covering the full range,
    oldest first. The last fold is shorter than `fold_months` when the range
    doesn't divide evenly — e.g. `split_into_folds("2025-01:2025-05", 2)` ->
    `["2025-01:2025-02", "2025-03:2025-04", "2025-05:2025-05"]`.

    This is the fold boundary used by the walk-forward harness
    (`application/walk_forward.py`) — see that module's docstring for why
    "walk-forward" here means independent out-of-sample folds, not classic
    in-sample-refit walk-forward optimization.

    Raises `InvalidPeriod` for `fold_months < 1` or a malformed `period` —
    reuses `parse_period`'s validation rather than duplicating it, so the
    same error type/message shape covers both entry points.
    """
    if fold_months < 1:
        raise InvalidPeriod(f"fold_months must be >= 1, got {fold_months}")
    start, end = parse_period(period)  # validates shape; end is exclusive

    folds: list[str] = []
    cursor = start
    while cursor < end:
        fold_end_exclusive = min(_add_months(cursor, fold_months), end)
        # fold_end_exclusive is the first day of the month *after* the fold's
        # last month — subtract one month to get the last month itself, since
        # the "YYYY-MM:YYYY-MM" format wants the last month inclusive.
        last_month = _add_months(fold_end_exclusive, -1)
        folds.append(f"{_fmt(cursor)}:{_fmt(last_month)}")
        cursor = fold_end_exclusive
    return folds


def _fmt(month_start: datetime) -> str:
    return f"{month_start.year:04d}-{month_start.month:02d}"
