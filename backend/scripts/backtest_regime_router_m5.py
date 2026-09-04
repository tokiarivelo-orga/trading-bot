"""Phase 5-equivalent real-history backtest for `xauusd_regime_router_m5`
(the M5 sibling of `xauusd_regime_router_m1`; DB version id and status
printed at the end of the coordinator's `save_generated_code` call —
`xauusd_regime_router_m5_v1.py`, status VALIDATED, never activated by this
script or by the build step: neither ever calls `activate_version`).

Runs the exact `TradeEngine`/`RiskManager`/`PositionManager`/`OrderService`
pipeline live trading uses (via `run_backtest()`), against `PaperBroker`,
replaying real XAUUSD M5 candles already sitting in
`backend/data/trading.db` (no backfill, no network, no writes to the live
journal). The strategy is injected into a hand-built `StrategyRegistry`
(bypassing the DB's ACTIVE-only registry gate) — the same mechanism
`backtest_regime_router.py` (the M1 sibling's own script) and
`tests/integration/test_regime_router_flow.py` already use.

Usage (from backend/):
    uv run python scripts/backtest_regime_router_m5.py

Writes the full structured results (full-period / train / holdout reports,
per-mode and per-regime-bucket breakdowns, a compact trade list, and the
signal-outcome tally) to
`backend/scripts/regime_router_m5_backtest_results.json`.

WHY THE PERIOD BOUNDARIES ARE WHAT THEY ARE
────────────────────────────────────────────────────────────────────────
Real XAUUSD M5 history in `trading.db` begins 2025-02-16 (mid-month,
verified below at runtime) and runs through 2026-09-03 — a much longer
window than the M1 sibling's ~150 real days, because M5 candles were
backfilled far deeper than M1 ever was. `parse_period()` only accepts
`"YYYY-MM:YYYY-MM"` (always a calendar-month start), and `run_backtest()`
refuses a period whose start is more than 4 days before the first available
candle (`NoHistoryError`). A `"2025-02"` start (month boundary = 2025-02-01)
is ~16 days before the first real M5 candle, so it trips that guard; the
next month boundary, `"2025-03"` (2025-03-01), is AFTER the first real
candle (2025-02-16), so it's always safe regardless of the 4-day tolerance.
The full-period run therefore starts 2025-03-01, dropping the partial
2025-02-16..2025-02-28 stretch (~13 days, ~2.3% of the ~565-day raw window)
— a much smaller loss than the M1 script's own ~17% April drop, since M5's
mid-month gap falls much earlier relative to the window's total length.

For the chronological train/holdout split, calendar-month boundaries were
checked directly against the real elapsed-day count (2025-03-01..2026-09-03,
551 real days) for the boundary closest to 70/30: 2026-04-01 gives
train=396 days (71.9%) / holdout=155 days (28.1%) — closer to 70/30 than
either neighboring boundary (2026-03-01: 66.2%/33.8%; 2026-05-01:
77.3%/22.7%). train = 2025-03..2026-03 (13 months), holdout = 2026-04..2026-09
(6 months, through the 2026-09-03 data cutoff) — contiguous, non-overlapping
(holdout starts exactly where train ends), so there is no lookahead leakage
between them. Same discipline the M1 sibling's own script documents.
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
from src.strategies.generated.xauusd_regime_router_m5_v1 import XauusdRegimeRouterM5
from src.strategies.registry import StrategyRegistry

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
OUT_PATH = Path(__file__).resolve().parent / "regime_router_m5_backtest_results.json"

STRATEGY_NAME = "xauusd_regime_router_m5"
SYMBOL = "XAUUSD"

PERIODS = {
    "full": "2025-03:2026-09",
    "train": "2025-03:2026-03",
    "holdout": "2026-04:2026-09",
}

# Strategy embeds the regime bucket key straight into Signal.reason as
# "... session=<session> trend=<trend> vol=<vol> ..." — see
# xauusd_regime_router_m5_v1.py's evaluate(), the `reason = (...)` f-string.
_REASON_RE = re.compile(r"session=(?P<session>\S+)\s+trend=(?P<trend>\S+)")

# Reference numbers to compare this M5 run against, both quoted verbatim
# from the coordinator's brief — NOT recomputed here, just quoted:
#   1. xauusd_regime_router_m1's own real backtest (its
#      regime_router_backtest_results.json): full PF 6.90, train PF 7.65,
#      holdout PF 6.11, n=465.
#   2. The H1-variant cautionary tale for this same
#      structure_continuation-based family: xauusd_snd_qm_structure_h1 /
#      _fixed_h1, PF 0.24-0.37, losing in every session, zero regime
#      gating.
COMPARISON_REFERENCE = {
    "xauusd_regime_router_m1_own_real_backtest": {
        "full_profit_factor": 6.90,
        "train_profit_factor": 7.65,
        "holdout_profit_factor": 6.11,
        "n": 465,
    },
    "h1_variant_cautionary_tale": {
        "strategies": ["xauusd_snd_qm_structure_h1", "xauusd_snd_qm_structure_fixed_h1"],
        "profit_factor_range": [0.24, 0.37],
        "note": "losing in every session, zero regime gating",
    },
}


def _verify_data_range() -> dict:
    """Confirms the real M5 history bounds this script's period choices rely
    on, straight from the DB — read-only, no writes."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.execute(
            "SELECT MIN(time), MAX(time), COUNT(*) FROM candles "
            "WHERE symbol = ? AND timeframe = 'M5'",
            (SYMBOL,),
        )
        min_ts, max_ts, count = cur.fetchone()
    finally:
        conn.close()
    if min_ts is None:
        raise RuntimeError(f"no M5 candles for {SYMBOL} in {DB_PATH}")
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


def _build_registry() -> tuple[StrategyRegistry, XauusdRegimeRouterM5]:
    """A fresh strategy instance (and therefore a cold `AdaptiveLearner`) per
    run — full/train/holdout are independent backtests, not one continuous
    replay, so nothing should carry learner state between them."""
    registry = StrategyRegistry()
    strategy = XauusdRegimeRouterM5()
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
    print(f"XAUUSD M5 candles in DB: {data_range}")

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
        "strategy_module": "xauusd_regime_router_m5_v1",
        "strategy_status": "VALIDATED (not activated by this script)",
        "enable_range_reversion": "default (False) — Mode C not force-enabled",
        "symbol": SYMBOL,
        "database_url": DATABASE_URL,
        "xauusd_m5_data_range": data_range,
        "period_boundaries": {
            **PERIODS,
            "note": (
                "parse_period() only supports YYYY-MM:YYYY-MM calendar-month-start "
                "boundaries. Real XAUUSD M5 history begins 2025-02-16 (mid-month); "
                "run_backtest()'s own >4-day data-gap guard rejects a 2025-02 period "
                "start, so 'full' starts 2025-03-01, dropping ~13 days (~2.3%) of real "
                "February history. train/holdout split at 2026-04-01 is the calendar "
                "month boundary closest to a 70/30 split of the ~551 real days "
                "available Mar'25-Sep3'26: train=Mar'25-Mar'26 (396 real days, "
                "~71.9%), holdout=Apr'26-Sep3'26 (155 real days, ~28.1%); contiguous, "
                "non-overlapping, no lookahead leakage."
            ),
        },
        "reports": {key: _report_summary(r) for key, r in reports.items()},
        "mode_and_bucket_breakdown_full_period": _mode_and_bucket_breakdown(full_report.trades),
        "mode_and_bucket_breakdown_train_period": _mode_and_bucket_breakdown(
            reports["train"].trades
        ),
        "mode_and_bucket_breakdown_holdout_period": _mode_and_bucket_breakdown(
            reports["holdout"].trades
        ),
        "comparison_reference": COMPARISON_REFERENCE,
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
