"""Per-bot validation of the give-back exit policy.

`eval_giveback_exit.py` answers "does the policy help on aggregate". This
answers the question that actually decides whether to run it fleet-wide:
**does it help every bot, or does it help some and quietly hurt others?**

A policy that adds +80R on one bot and takes -30R from three others is not a
fleet-wide win, and the aggregate number hides that. Each bot is replayed
over its own real M1 price path — same entries, same stops, same spreads the
broker actually filled; only the exit rule is counterfactual.

Reported with a chronological split per bot, because an in-sample-only number
on this data has repeatedly proven worthless (a per-hour entry filter tested
PF 2.70 in-sample and PF 0.12 out-of-sample).

    uv run python scripts/eval_giveback_per_bot.py --symbol XAUUSD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3  # noqa: E402

import pandas as pd  # noqa: E402

from scripts.eval_giveback_exit import (  # noqa: E402
    load_live_trades,
    replay_live_trades,
    summarise,
)
from src.engine.domain.exit_policy import ExitPolicyConfig  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "trading.db"
MIN_TRADES = 25


def _load_m1(db_path: Path, symbol: str) -> pd.DataFrame:
    """M1 path candles. Same query `run_live_mode` uses — the archive starts
    2026-04, so trades older than that have no path and get skipped."""
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(
            "SELECT time, high, low, close FROM candles "
            "WHERE symbol = ? AND timeframe = 'M1' ORDER BY time",
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()


def _fmt(label: str, stats: dict[str, float] | None, n: int) -> str:
    if stats is None or n == 0:
        return f"  {label:<12} n=0"
    return (
        f"  {label:<12} n={n:<5} sumR={stats.get('sum_r', 0.0):>+8.1f} "
        f"avgR={stats.get('avg_r', 0.0):>+8.4f} "
        f"PF={stats.get('profit_factor', 0.0):>5.2f} "
        f"maxDD={stats.get('max_dd_r', 0.0):>6.1f}R"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", type=float, default=0.6, help="in-sample fraction")
    parser.add_argument("--arm-r", type=float, default=0.3)
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DEFAULT_DB
    trades = load_live_trades(db_path, args.symbol)
    m1 = _load_m1(db_path, args.symbol)
    policy = ExitPolicyConfig(
        enabled=True, arm_r=args.arm_r, keep_fraction=args.keep
    )

    print(f"symbol={args.symbol}  live trades with a stop={len(trades)}  M1 bars={len(m1)}")
    print(f"policy: arm_r={args.arm_r} keep_fraction={args.keep}  split={args.split:.0%}\n")

    bots = [b for b in trades["strategy_version"].dropna().unique()]
    rows = []
    for bot in sorted(bots):
        subset = trades[trades["strategy_version"] == bot].reset_index(drop=True)
        base = replay_live_trades(subset, m1, None)
        with_policy = replay_live_trades(subset, m1, policy)
        if len(base) < MIN_TRADES or len(base) != len(with_policy):
            continue

        cut = int(len(base) * args.split)
        rows.append((bot, base, with_policy, cut))

    if not rows:
        print(f"No bot had >= {MIN_TRADES} replayable trades "
              f"(M1 coverage starts 2026-04, so older trades are skipped).")
        return 1

    helped = hurt = 0
    total_delta_oos = 0.0
    summary: list = []
    for bot, base, with_policy, cut in rows:
        print(f"── {bot} ──")
        for label, lo, hi in (("IN-SAMPLE", 0, cut), ("OUT-SAMPLE", cut, len(base))):
            b = summarise(base[lo:hi])
            p = summarise(with_policy[lo:hi])
            n = hi - lo
            print(f"  {label}")
            print(_fmt("baseline", b, n))
            print(_fmt("give-back", p, n))
            delta = p.get("sum_r", 0.0) - b.get("sum_r", 0.0)
            dd = b.get("max_dd_r", 0.0) - p.get("max_dd_r", 0.0)
            print(f"  {'delta':<12} sumR={delta:>+8.1f}   drawdown reduced by {dd:>6.1f}R")
            if label == "OUT-SAMPLE":
                total_delta_oos += delta
                summary.append((delta, dd, bot, b, p, n))
                if delta > 0:
                    helped += 1
                else:
                    hurt += 1
        print()

    print("═" * 78)
    print("OUT-OF-SAMPLE PER BOT, best delta first")
    print(f"{'bot':<40}{'n':>5}{'base':>9}{'policy':>9}{'delta':>9}{'DD cut':>9}")
    for delta, dd, bot, b, p, n in sorted(summary, reverse=True):
        print(f"{bot:<40}{n:>5}{b.get('sum_r', 0.0):>+9.1f}"
              f"{p.get('sum_r', 0.0):>+9.1f}{delta:>+9.1f}{dd:>9.1f}")

    print("═" * 78)
    print(f"across {len(rows)} bots: helped {helped}, hurt {hurt}, "
          f"net {total_delta_oos:+.1f}R")
    if hurt:
        print("\nSuggested configs/exits.yaml per_bot block (disable where it costs "
              "more than it saves):")
        for delta, dd, bot, _b, _p, _n in sorted(summary):
            # A bot is only worth disabling when the expectancy it loses is
            # not bought back by a meaningful drawdown reduction.
            if delta < 0 and dd < abs(delta):
                print(f"  {bot.split(':')[0]}: {{ enabled: false }}   "
                      f"# {delta:+.1f}R, only {dd:.1f}R drawdown saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
