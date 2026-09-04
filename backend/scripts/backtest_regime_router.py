"""Phase 5 real-history backtest for `xauusd_regime_router_m1` (DB version id
2a0de4bb-c1ad-4690-8add-66c196a262c3, `xauusd_regime_router_m1_v3.py`,
status VALIDATED — NOT activated by this script, and never will be: it
never calls `activate_version`). v3 supersedes v2 (now archived): Mode C
(range_reversion) is unchanged code-wise but gated behind
`params["enable_range_reversion"]`, which now defaults to `False` after the
first real-history run (this same script, against v2) found Mode C alone at
PF 0.37 (-$2,452) while Modes A+B together ran PF 4.31. This run uses v3's
default params — Mode C stays off, `enable_range_reversion` is NOT
overridden to `True`.

Runs the exact `TradeEngine`/`RiskManager`/`PositionManager`/`OrderService`
pipeline live trading uses (via `run_backtest()`), against `PaperBroker`,
replaying real XAUUSD M1 candles already sitting in `backend/data/trading.db`
(no backfill, no network, no writes to the live journal). The strategy is
injected into a hand-built `StrategyRegistry` (bypassing the DB's
ACTIVE-only registry gate) — the same mechanism
`tests/integration/test_regime_router_flow.py` already proved works.

Usage (from backend/):
    uv run python scripts/backtest_regime_router.py

Writes the full structured results (full-period / train / holdout reports,
per-mode and per-regime-bucket breakdowns, a compact trade list, and the
signal-outcome tally) to
`backend/scripts/regime_router_backtest_results.json`.

WHY THE PERIOD BOUNDARIES ARE WHAT THEY ARE
────────────────────────────────────────────────────────────────────────
`backend/src/backtest/application/period.py`'s `parse_period()` only accepts
`"YYYY-MM:YYYY-MM"` — always a calendar-month start, never a precise day.
Real XAUUSD M1 history in `trading.db` begins 2026-04-06 (mid-month, verified
below at runtime), and `run_backtest()` itself refuses a period whose start
is more than 4 days before the first available candle (`NoHistoryError` —
its own guard against a silently-truncated report). A `"2026-04:..."` start
(month boundary = 2026-04-01) is ~5.9 days before the first real M1 candle,
so it trips that guard. The full-period run therefore starts 2026-05-01,
which drops the partial 2026-04-06..2026-04-30 stretch (~25 days, ~17% of
the raw ~150-day window) — an accepted, documented loss, not a silent one.

For the chronological train/holdout split, 2026-08-01 is the calendar-month
boundary closest to a 70/30 split of the ~126 real days actually usable
(2026-05-01..2026-09-03): train = May+Jun+Jul (92 real days, ~73%), holdout
= Aug 1..Sep 3 (34 real days available, ~27%) — train and holdout periods
are contiguous and non-overlapping (holdout starts exactly where train ends),
so there is no lookahead leakage between them.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from src.backtest.application import metrics
from src.backtest.application.run_backtest import run_backtest
from src.backtest.domain.models import BacktestReport, BacktestTrade
from src.strategies.generated.xauusd_regime_router_m1_v3 import XauusdRegimeRouterM1
from src.strategies.registry import StrategyRegistry

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
OUT_PATH = Path(__file__).resolve().parent / "regime_router_backtest_results.json"

STRATEGY_NAME = "xauusd_regime_router_m1"
STRATEGY_VERSION_ID = "2a0de4bb-c1ad-4690-8add-66c196a262c3"
SYMBOL = "XAUUSD"

PERIODS = {
    "full": "2026-05:2026-09",
    "train": "2026-05:2026-07",
    "holdout": "2026-08:2026-09",
}

# Strategy embeds the regime bucket key straight into Signal.reason as
# "... session=<session> trend=<trend> vol=<vol> ..." — see
# xauusd_regime_router_m1_v3.py's evaluate(), the `reason = (...)` f-string.
_REASON_RE = re.compile(r"session=(?P<session>\S+)\s+trend=(?P<trend>\S+)")

# Fleet reference numbers from the plan (backend/*.md research pass,
# 15,669 live trades 2026-07-12..2026-09-03) — NOT recomputed here, just
# quoted for comparison.
FLEET_REFERENCE = {
    "best_live_performers": [
        {"strategy": "xauusd_snd_qm_structure_adaptive_m1", "profit_factor": 1.64, "n": 533},
        {"strategy": "xauusd_apex_turbo_m1", "profit_factor": 1.44, "n": 546},
    ],
    "worst_live_performers": [
        {"strategy": "xauusd_snd_qm_structure_h1", "profit_factor_range": [0.24, 0.37]},
        {"strategy": "xauusd_snd_qm_structure_fixed_h1", "profit_factor_range": [0.24, 0.37]},
    ],
}


def _verify_data_range() -> dict:
    """Confirms the real M1 history bounds this script's period choices rely
    on, straight from the DB — read-only, no writes."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            "SELECT MIN(time), MAX(time), COUNT(*) FROM candles "
            "WHERE symbol = ? AND timeframe = 'M1'",
            (SYMBOL,),
        )
        min_ts, max_ts, count = cur.fetchone()
    finally:
        conn.close()
    if min_ts is None:
        raise RuntimeError(f"no M1 candles for {SYMBOL} in {DB_PATH}")
    return {
        "min_time": datetime.fromtimestamp(min_ts, tz=UTC).isoformat(),
        "max_time": datetime.fromtimestamp(max_ts, tz=UTC).isoformat(),
        "candle_count": count,
    }


def _pf(value: float) -> float | str:
    """`profit_factor` is `float('inf')` when there are zero losing trades —
    valid Python `json` output (non-standard but round-trips via
    `json.loads`), but stringifying it keeps the file trivially parseable by
    stricter JSON readers downstream."""
    if math.isinf(value):
        return "inf"
    return round(value, 4)


def _parse_reason(reason: str) -> tuple[str | None, str | None]:
    match = _REASON_RE.search(reason or "")
    if match is None:
        return None, None
    return match.group("session"), match.group("trend")


def _trade_dict(trade: BacktestTrade) -> dict:
    session, trend = _parse_reason(trade.reason)
    return {
        "mode": trade.pattern,
        "direction": trade.side,
        "open_time": trade.open_time.isoformat(),
        "close_time": trade.close_time.isoformat(),
        "profit": round(trade.profit, 4),
        "r_multiple": round(trade.r_multiple, 4) if trade.r_multiple is not None else None,
        "session": session,
        "trend": trend,
    }


def _subset_metrics(trades: tuple[BacktestTrade, ...]) -> dict:
    total_profit = sum(t.profit for t in trades)
    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit < 0]
    return {
        "count": len(trades),
        "win_rate": round(metrics.win_rate(trades), 4),
        "profit_factor": _pf(metrics.profit_factor(trades)),
        "avg_r": round(metrics.avg_r(trades), 4),
        "total_profit": round(total_profit, 2),
        "win_count": len(wins),
        "loss_count": len(losses),
    }


def _mode_and_bucket_breakdown(trades: tuple[BacktestTrade, ...]) -> dict:
    by_mode: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        by_mode[t.pattern or "unknown"].append(t)

    result: dict[str, dict] = {}
    for mode, mode_trades in sorted(by_mode.items()):
        entry = _subset_metrics(tuple(mode_trades))
        by_bucket: dict[str, list[BacktestTrade]] = defaultdict(list)
        for t in mode_trades:
            session, trend = _parse_reason(t.reason)
            by_bucket[f"trend={trend}|session={session}"].append(t)
        entry["regime_buckets"] = {
            key: _subset_metrics(tuple(ts)) for key, ts in sorted(by_bucket.items())
        }
        result[mode] = entry
    return result


def _signal_outcome_tally(report: BacktestReport) -> dict:
    tally: dict[str, int] = defaultdict(int)
    for s in report.signals:
        tally[s.outcome] += 1
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]))


def _report_summary(report: BacktestReport) -> dict:
    total_profit = sum(t.profit for t in report.trades)
    return {
        "period": report.period,
        "trade_count": len(report.trades),
        "profit_factor": _pf(report.profit_factor),
        "win_rate": round(report.win_rate, 4),
        "max_drawdown_pct": round(report.max_drawdown_pct, 4),
        "avg_r": round(report.avg_r, 4),
        "worst_losing_streak": report.worst_losing_streak,
        "starting_balance": report.starting_balance,
        "ending_balance": round(report.ending_balance, 2),
        "total_profit": round(total_profit, 2),
        "broker_realism": {
            "enabled": report.broker_realism.enabled,
            "accepted_count": report.broker_realism.accepted_count,
            "clamped_count": report.broker_realism.clamped_count,
            "rejected_count": report.broker_realism.rejected_count,
            "rejections": [
                {
                    "reason": r.reason,
                    "count": r.count,
                    "retcode": r.retcode,
                    "example": r.example,
                }
                for r in report.broker_realism.rejections
            ],
        },
        "signal_outcome_tally": _signal_outcome_tally(report),
        "trades": [_trade_dict(t) for t in report.trades],
    }


def _build_registry() -> tuple[StrategyRegistry, XauusdRegimeRouterM1]:
    """A fresh strategy instance (and therefore a cold `AdaptiveLearner`) per
    run — full/train/holdout are independent backtests, not one continuous
    replay, so nothing should carry learner state between them."""
    registry = StrategyRegistry()
    strategy = XauusdRegimeRouterM1()
    registry.register(strategy.spec.name, strategy)
    return registry, strategy


async def _run_one(period: str) -> BacktestReport:
    registry, strategy = _build_registry()
    return await run_backtest(
        strategy.spec.name,
        SYMBOL,
        period,
        strategy_source=registry,
        database_url=DATABASE_URL,
    )


async def main() -> None:
    data_range = _verify_data_range()
    print(f"XAUUSD M1 candles in DB: {data_range}")

    reports: dict[str, BacktestReport] = {}
    for key, period in PERIODS.items():
        print(f"\nRunning {key!r} period={period!r} ...")
        report = await _run_one(period)
        reports[key] = report
        print(
            f"  trades={len(report.trades)} pf={_pf(report.profit_factor)} "
            f"win_rate={report.win_rate:.3f} max_dd={report.max_drawdown_pct:.2f}% "
            f"avg_r={report.avg_r:.3f} worst_losing_streak={report.worst_losing_streak} "
            f"ending_balance={report.ending_balance:.2f}"
        )
        if not report.trades:
            print(f"  ZERO TRADES — signal outcome tally: {_signal_outcome_tally(report)}")

    full_report = reports["full"]
    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": STRATEGY_NAME,
        "strategy_version_id": STRATEGY_VERSION_ID,
        "strategy_module": "xauusd_regime_router_m1_v3",
        "strategy_status": "VALIDATED (not activated by this script)",
        "enable_range_reversion": "default (False) — Mode C not force-enabled",
        "symbol": SYMBOL,
        "database_url": DATABASE_URL,
        "xauusd_m1_data_range": data_range,
        "period_boundaries": {
            **PERIODS,
            "note": (
                "parse_period() only supports YYYY-MM:YYYY-MM calendar-month-start "
                "boundaries. Real XAUUSD M1 history begins 2026-04-06 (mid-month); "
                "run_backtest()'s own >4-day data-gap guard rejects a 2026-04 period "
                "start (~5.9 day gap from 2026-04-01), so 'full' starts 2026-05-01, "
                "dropping ~25 days (~17%) of real April history. train/holdout split "
                "at 2026-08-01 is the closest month boundary to a 70/30 split of the "
                "~126 real days available May-Sep3: train=May-Jul (92 real days, "
                "~73%), holdout=Aug1-Sep3 (34 real days, ~27%); contiguous, "
                "non-overlapping, no lookahead leakage."
            ),
        },
        "reports": {key: _report_summary(r) for key, r in reports.items()},
        "mode_and_bucket_breakdown_full_period": _mode_and_bucket_breakdown(full_report.trades),
        "fleet_comparison_reference": FLEET_REFERENCE,
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
