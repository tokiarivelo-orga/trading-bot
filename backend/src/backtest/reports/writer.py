"""Writes `BacktestReport`s to JSON files under this directory and renders a
human-readable summary for the CLI. The only place a `BacktestReport`
becomes a file — `backtest/api/routes.py`'s read endpoints, and
`ai/api/routes_refinement.py`'s analysis-report lookup, both read back
exactly the shape written here (see each module's own docstring)."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.backtest.domain.models import BacktestReport

REPORTS_DIR = Path(__file__).resolve().parent


def write_report(report: BacktestReport, directory: Path | None = None) -> Path:
    """Serializes `report` to a JSON file and returns its path. Datetimes are
    stored as ISO 8601 strings (read back via `datetime.fromisoformat` by the
    API layer's `_epoch`/`_iso` helpers); `profit_factor=inf` (no losing
    trades) is stored as `null` since JSON has no infinity literal — the API
    layer treats a null profit_factor as "infinite", not "unknown"."""
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{report.strategy}_{report.symbol}_{report.period.replace(':', '_')}"
        f"_{uuid.uuid4().hex[:8]}"
    )
    path = directory / f"{filename}.json"
    path.write_text(json.dumps(_serialize(report), indent=2))
    return path


def render_summary(report: BacktestReport) -> str:
    """Human-readable headline stats for the CLI's stdout output."""
    profit_factor = "inf" if math.isinf(report.profit_factor) else f"{report.profit_factor:.2f}"
    return (
        f"{report.strategy} on {report.symbol}, {report.period}\n"
        f"  balance:        {report.starting_balance:.2f} -> {report.ending_balance:.2f}\n"
        f"  trades:         {len(report.trades)}\n"
        f"  win rate:       {report.win_rate:.1%}\n"
        f"  profit factor:  {profit_factor}\n"
        f"  max drawdown:   {report.max_drawdown_pct:.2f}%\n"
        f"  avg R:          {report.avg_r:.2f}\n"
        f"  worst streak:   {report.worst_losing_streak} losses in a row"
    )


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _serialize(report: BacktestReport) -> dict[str, Any]:
    return {
        "strategy": report.strategy,
        "symbol": report.symbol,
        "period": report.period,
        "starting_balance": report.starting_balance,
        "ending_balance": report.ending_balance,
        "win_rate": report.win_rate,
        "profit_factor": None if math.isinf(report.profit_factor) else report.profit_factor,
        "max_drawdown_pct": report.max_drawdown_pct,
        "avg_r": report.avg_r,
        "worst_losing_streak": report.worst_losing_streak,
        "min_rr": report.min_rr,
        "risk_per_trade_pct": report.risk_per_trade_pct,
        "daily_loss_limit_pct": report.daily_loss_limit_pct,
        "max_open_positions": report.max_open_positions,
        "max_trades_per_day_enabled": report.max_trades_per_day_enabled,
        "consecutive_loss_pause": report.consecutive_loss_pause,
        "min_lot_fallback_enabled": report.min_lot_fallback_enabled,
        "max_risk_per_trade_pct": report.max_risk_per_trade_pct,
        "trades": [_trade(t) for t in report.trades],
        "equity_curve": [
            {"time": _iso(p.time), "balance": p.balance} for p in report.equity_curve
        ],
        "activity_log": [
            {
                "time": _iso(e.time),
                "level": e.level,
                "logger": e.logger,
                "message": e.message,
            }
            for e in report.activity_log
        ],
        "signals": [
            {
                "time": _iso(s.time),
                "direction": s.direction,
                "outcome": s.outcome,
                "reason": s.reason,
            }
            for s in report.signals
        ],
    }


def _trade(trade: Any) -> dict[str, Any]:
    return {
        "side": trade.side,
        "volume": trade.volume,
        "open_time": _iso(trade.open_time),
        "open_price": trade.open_price,
        "sl": trade.sl,
        "tp": trade.tp,
        "close_time": _iso(trade.close_time),
        "close_price": trade.close_price,
        "profit": trade.profit,
        "r_multiple": trade.r_multiple,
        "zone": _zone(trade.zone) if trade.zone is not None else None,
        "pattern": trade.pattern,
        "structure": [[label, price, _iso(time)] for label, price, time in trade.structure],
        "reason": trade.reason,
        "confidence": trade.confidence,
    }


def _zone(zone: Any) -> dict[str, Any]:
    return {
        "kind": zone.kind,
        "price_low": zone.price_low,
        "price_high": zone.price_high,
        "time_start": _iso(zone.time_start),
        "time_end": _iso(zone.time_end),
        "pattern": zone.pattern,
    }
