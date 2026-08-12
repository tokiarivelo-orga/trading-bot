"""Would a fixed profit-taking exit help — and which timeframes?

The give-back policy already in place caps how much of a *peak* is handed
back. This asks a different question: would simply closing at a fixed +X R,
whenever price touches it, do better?

That is worth asking because the numbers say the current risk/reward is
inverted. The engine risks 1R to bank 0.2R via secure-base trailing, which
needs 1/(1+0.2) = 83.3% to break even, and the live secure-rate is 0.62 —
while the mean favourable excursion is over 1R. The market does move; the
exit books almost none of it.

Note this cannot be done with a broker take-profit: `SpreadGate` rejects any
TP below spread-adjusted `min_rr` (1.5 on XAUUSD), so a 0.5R target would
never reach the broker. As a *exit rule* inside `PositionManager` — where the
give-back policy already lives — there is no such floor.

HOW THE COUNTERFACTUAL WORKS, AND WHY IT IS CONSERVATIVE
────────────────────────────────────────────────────────────────────────
For each real closed trade:

    MFE >= X  ->  the fixed exit would have triggered; book X, minus this
                  trade's own realised transaction cost in R
    MFE <  X  ->  it never triggered; the trade plays out exactly as it did

`profit` in the `trades` table is already net of `transaction_cost`, so the
counterfactual subtracts that cost explicitly rather than quietly booking a
gross number against net baselines. Booking exactly X also ignores slippage
on the exit fill, which makes the result mildly optimistic — treat the shape
of the curve, not its peak, as the finding.

Split chronologically per timeframe, because on this data an in-sample-only
number has repeatedly proven worthless.

    uv run python scripts/eval_fixed_target_by_tf.py
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "trading.db"
CONTRACT = 100.0
GRID = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5)
MIN_TRADES = 40


def load(db_path: Path, symbol: str) -> dict[str, list[dict]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT skill, open_time, open_price, sl, volume, profit, mfe, "
            "       transaction_cost "
            "FROM trades WHERE symbol = ? AND close_time IS NOT NULL "
            "  AND sl IS NOT NULL AND mfe IS NOT NULL ORDER BY open_time",
            (symbol,),
        ))
    finally:
        conn.close()

    by_tf: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        risk_px = abs(row["open_price"] - row["sl"])
        if risk_px <= 0 or row["volume"] <= 0:
            continue
        risk_money = risk_px * row["volume"] * CONTRACT
        if risk_money <= 0:
            continue
        match = re.search(r"_(m1|m5|m15|h1|h4|d1)\b", (row["skill"] or "").lower())
        by_tf[match.group(1).upper() if match else "other"].append({
            "r": row["profit"] / risk_money,
            "mfe_r": row["mfe"] / risk_px,
            # Cost the counterfactual must also pay. `profit` is already net,
            # so booking a gross X against it would be comparing two different
            # things.
            "cost_r": (row["transaction_cost"] or 0.0) / risk_money,
        })
    return by_tf


def stats(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    wins = values[values > 0].sum()
    losses = -values[values <= 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return float(values.sum()), float(values.mean()), float(pf)


def booked_at(trades: list[dict], target: float) -> np.ndarray:
    return np.array([
        (target - t["cost_r"]) if t["mfe_r"] >= target else t["r"]
        for t in trades
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    by_tf = load(Path(args.db) if args.db else DEFAULT_DB, args.symbol)

    for tf in ("M1", "M5", "M15", "H1", "other"):
        trades = by_tf.get(tf)
        if not trades or len(trades) < MIN_TRADES:
            continue
        cut = int(len(trades) * args.split)
        ins, oos = trades[:cut], trades[cut:]

        base_is = np.array([t["r"] for t in ins])
        base_oos = np.array([t["r"] for t in oos])
        b_is, _, pf_is = stats(base_is)
        b_oos, avg_oos, pf_oos = stats(base_oos)

        print(f"\n══ {tf}  (n={len(trades)}, IS={len(ins)} OOS={len(oos)}) ══")
        print(f"  {'target':>8}{'IS sumR':>10}{'IS PF':>8}"
              f"{'OOS sumR':>11}{'OOS avgR':>10}{'OOS PF':>8}{'fires':>8}")
        print(f"  {'baseline':>8}{b_is:>+10.1f}{pf_is:>8.2f}"
              f"{b_oos:>+11.1f}{avg_oos:>+10.4f}{pf_oos:>8.2f}{'—':>8}")

        for target in GRID:
            s_is, _, p_is = stats(booked_at(ins, target))
            oos_values = booked_at(oos, target)
            s_oos, a_oos, p_oos = stats(oos_values)
            fires = float(np.mean([t["mfe_r"] >= target for t in oos]))
            print(f"  {target:>8.2f}{s_is:>+10.1f}{p_is:>8.2f}"
                  f"{s_oos:>+11.1f}{a_oos:>+10.4f}{p_oos:>8.2f}{fires:>7.0%}")

    print("\nReading this: a target only counts as real if it improves OOS,")
    print("and if the improvement holds across neighbouring targets rather")
    print("than spiking at one value. A single-point peak is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
