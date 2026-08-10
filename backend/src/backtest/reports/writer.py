"""Writes a `BacktestReport` to `backend/src/backtest/reports/` as JSON, and
renders the human-readable summary block `/backtest`'s SKILL.md expects."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from src.backtest.domain.models import BacktestReport

REPORTS_DIR = Path(__file__).resolve().parent


def write_report(report: BacktestReport, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{report.strategy}_{report.symbol}_{report.period.replace(':', '_')}.json"
    path.write_text(json.dumps(to_jsonable(report), indent=2))
    return path


def render_summary(report: BacktestReport) -> str:
    profit_factor = "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.2f}"
    lines = [
        f"Backtest: {report.strategy} on {report.symbol}, {report.period}",
        f"  Trades:              {len(report.trades)}",
        f"  Win rate:            {report.win_rate * 100:.1f}%",
        f"  Profit factor:       {profit_factor}",
        f"  Max drawdown:        {report.max_drawdown_pct:.2f}%",
        f"  Avg R:               {report.avg_r:.2f}",
        f"  Worst losing streak: {report.worst_losing_streak}",
        f"  Starting balance:    {report.starting_balance:.2f}",
        f"  Ending balance:      {report.ending_balance:.2f}",
        f"  Signals:             {len(report.signals)} emitted "
        f"({sum(1 for s in report.signals if s.outcome == 'opened')} opened, rest "
        "vetoed/rejected — see the report's signals list)",
        f"  Activity log lines:  {len(report.activity_log)} (signals, vetoes, sizing "
        "rejections, fills — see the report file for the full trail)",
    ]
    lines.extend(_realism_lines(report))
    return "\n".join(lines)


def _realism_lines(report: BacktestReport) -> list[str]:
    """The broker-constraint block (OBSERVABILITY_PLAN.md Phase 4). Says
    explicitly when nothing was simulated, because a reader who does not see
    the block would otherwise assume the numbers above are broker-legal."""
    realism = report.broker_realism
    if not realism.enabled:
        return [
            "  Broker realism:      NOT SIMULATED — these trades were filled at the bar's "
            "closing quote in the exact lot size requested, so some may be entries a real "
            "broker would refuse (stops_level, lot grid).",
        ]
    lines = [
        f"  Broker realism:      {realism.accepted_count} filled, "
        f"{realism.rejected_count} refused "
        f"({realism.acceptance_rate * 100:.1f}% accepted)",
        f"  Slippage model:      mean {realism.slippage_mean:+.5f} "
        f"sd {realism.slippage_stddev:.5f} "
        f"({realism.slippage_source}, {realism.slippage_sample_count} live samples)",
    ]
    if realism.clamped_count:
        lines.append(
            f"  Stops clamped:       {realism.clamped_count} entries had their SL/TP widened "
            "to the broker minimum — they risked MORE than the risk manager sized them for"
        )
    for rejection in realism.rejections:
        lines.append(
            f"    refused {rejection.count}x {rejection.reason} "
            f"(retcode {rejection.retcode}): {rejection.example}"
        )
    return lines


def to_jsonable(value: Any) -> Any:
    """Recursively converts a dataclass tree (datetimes -> ISO strings, `inf`/
    `nan` floats -> `null`, dataclasses -> plain dicts) into something
    `json.dumps` accepts. Public (not `_to_jsonable`) because
    `reports/walk_forward_writer.py` (OBSERVABILITY_PLAN.md Phase 6 Pass B)
    reuses it for `WalkForwardReport`, which is its own dataclass tree, not a
    `BacktestReport`."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        # `Infinity`/`NaN` aren't valid JSON — a profit factor with no losing
        # trades is mathematically infinite; represent it as null instead
        # (the API schema documents null as "no losing trades").
        return None
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {f: to_jsonable(getattr(value, f)) for f in value.__dataclass_fields__}
    return value
