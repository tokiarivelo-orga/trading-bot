"""Writes a `WalkForwardReport` to `backend/src/backtest/reports/walk_forward/`
as JSON, and renders the human-readable summary block the walk-forward CLI
prints (OBSERVABILITY_PLAN.md Phase 6 Pass B).

Mirrors `reports/writer.py`'s shape for the single-backtest report, but
writes ONE aggregate JSON file per walk-forward run (not one file per fold —
per-fold detail lives inside that one file's `folds` array) into its own
`walk_forward/` subdirectory. Keeping it out of `writer.REPORTS_DIR` means
`GET /backtest/reports`'s glob over that directory never has to special-case
a differently-shaped file — the two report kinds are listed by two separate
endpoints (`GET /backtest/reports` vs `GET /backtest/walk-forward/reports`).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.backtest.application.walk_forward import WalkForwardReport
from src.backtest.reports.writer import to_jsonable

WALK_FORWARD_REPORTS_DIR = Path(__file__).resolve().parent / "walk_forward"


def write_walk_forward_report(
    report: WalkForwardReport, reports_dir: Path = WALK_FORWARD_REPORTS_DIR
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report.strategy}_{report.symbol}_{report.period.replace(':', '_')}.json"
    path.write_text(json.dumps(to_jsonable(report), indent=2))
    return path


def render_walk_forward_summary(report: WalkForwardReport) -> str:
    """Mirrors `writer.render_summary`'s plain-text block, for the CLI and
    for a quick terminal read without opening the JSON file."""

    def _fmt_pf(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        f"Walk-forward: {report.strategy} on {report.symbol}, {report.period} "
        f"({report.fold_months}-month folds)",
        f"  Folds run:            {report.fold_count}",
        f"  Folds w/ zero trades: {report.folds_with_zero_trades}",
        f"  Losing folds (exp<0): {report.losing_fold_count}",
        f"  Mean profit factor:   {_fmt_pf(report.mean_profit_factor)}",
        f"  Stddev profit factor: {_fmt_pf(report.stddev_profit_factor)}",
        f"  Worst fold PF:        {_fmt_pf(report.worst_fold_profit_factor)}",
        f"  Mean expectancy:      {report.mean_expectancy:.2f}",
        f"  Starting balance:     {report.starting_balance:.2f} (each fold starts fresh here)",
        "",
        "  Per-fold breakdown:",
    ]
    for fold in report.folds:
        lines.append(
            f"    {fold.period}: {fold.trade_count} trades, "
            f"win rate {fold.win_rate * 100:.1f}%, PF {_fmt_pf(fold.profit_factor)}, "
            f"expectancy {fold.expectancy:.2f}, "
            f"broker acceptance {fold.broker_acceptance_rate * 100:.1f}%"
        )
    if not report.folds:
        lines.append("    (every fold was skipped — no candle history for this period yet)")
    return "\n".join(lines)
