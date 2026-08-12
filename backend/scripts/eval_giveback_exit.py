"""Measure the give-back exit policy against the engine's current exit rules.

Run from ``backend/``::

    uv run python scripts/eval_giveback_exit.py --symbol XAUUSD
    uv run python scripts/eval_giveback_exit.py --arm-r 0.5 --keep 0.5

WHAT IT COMPARES
────────────────────────────────────────────────────────────────────────
Two exit stacks, replayed bar by bar over real M5 candles from the same
entry set, so the only difference is the exit:

  baseline   what the engine does today — initial stop at -1R, stop moved to
             +``secure_r`` once a close clears that level, resting take-profit
             at ``target_r``.
  policy     the same, plus ``engine.domain.exit_policy.decide_exit``.

Both call the *same* ``decide_exit`` the live ``PositionManager`` calls, so
a result here is a statement about production code, not about a
reimplementation of it.

HOW IT IS SCORED
────────────────────────────────────────────────────────────────────────
Chronological split, never random: the first ``--split`` fraction of the
period is in-sample, the remainder out-of-sample, and both are reported. A
parameter whose in-sample and out-of-sample optima disagree is reported as
not tunable rather than fitted — a per-hour entry filter on this same data
measured PF 2.70 in-sample and PF 0.12 out-of-sample.

Entries are non-overlapping (one position at a time) and every trade pays
the spread recorded on its entry bar. Win rate is not reported: with a
0.2R trailing rule it is an artifact of the rule, not of the strategy.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.domain.exit_policy import (  # noqa: E402
    ExitAction,
    ExitPolicyConfig,
    decide_exit,
)
from src.strategies.generated.smc_dl_features_v2 import atr  # noqa: E402
from src.strategies.generated.smc_dl_labels_v2 import LabelConfig  # noqa: E402

logger = logging.getLogger("eval_giveback_exit")
BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
POINT_VALUE = 0.01


@dataclass(frozen=True)
class TradeResult:
    index: int
    realised_r: float
    bars_held: int
    peak_r: float


def load_m5(db_path: Path, symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            "SELECT time, open, high, low, close, tick_volume, spread_points "
            "FROM candles WHERE symbol = ? AND timeframe = 'M5' ORDER BY time",
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame.set_index("time", drop=False)


def simulate(
    frame: pd.DataFrame,
    entries: np.ndarray,
    is_buy: np.ndarray,
    label_cfg: LabelConfig,
    policy: ExitPolicyConfig | None,
) -> list[TradeResult]:
    """Replay every entry to resolution, one position at a time.

    Intrabar ordering is unknowable, so when a bar touches both the stop and
    a favourable level the stop is taken first — the same pessimistic
    convention the label generator uses. Anything else flatters the policy in
    exactly the cases it is meant to be judged on.
    """
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr14 = atr(frame, 14).to_numpy(dtype=float)
    spread_r = frame["spread_points"].to_numpy(dtype=float) * POINT_VALUE

    n = len(frame)
    results: list[TradeResult] = []
    blocked_until = -1

    for i in entries:
        if i <= blocked_until or i + 1 >= n:
            continue
        risk = label_cfg.sl_atr_mult * atr14[i]
        if not np.isfinite(risk) or risk <= 0:
            continue
        direction = 1.0 if is_buy[i] else -1.0
        entry_price = close[i]
        stop = entry_price - direction * risk
        take_profit = entry_price + direction * label_cfg.target_r * risk
        extreme = entry_price
        secured = False
        cost = spread_r[i] / risk

        outcome: float | None = None
        bars = 0
        for step in range(1, label_cfg.max_holding_bars + 1):
            j = i + step
            if j >= n:
                break
            bars = step
            # Stop and take-profit are resting broker orders: they fill
            # intrabar. Stop first when both are touched.
            hit_stop = low[j] <= stop if is_buy[i] else high[j] >= stop
            if hit_stop:
                outcome = (stop - entry_price) * direction / risk
                break
            hit_tp = high[j] >= take_profit if is_buy[i] else low[j] <= take_profit
            if hit_tp:
                outcome = (take_profit - entry_price) * direction / risk
                break

            extreme = max(extreme, high[j]) if is_buy[i] else min(extreme, low[j])
            mark = close[j]

            # Engine rule 2, close-triggered: PositionManager only modifies
            # stops at candle close, so a wick through the secure level does
            # not arm it.
            secure_level = entry_price + direction * label_cfg.secure_r * risk
            if not secured and ((mark >= secure_level) if is_buy[i] else (mark <= secure_level)):
                secured = True
                if (secure_level - stop) * direction > 0:
                    stop = secure_level

            if policy is not None:
                decision = decide_exit(
                    is_buy=bool(is_buy[i]),
                    entry_price=entry_price,
                    current_sl=stop,
                    mark=mark,
                    extreme_favorable=extreme,
                    risk=risk,
                    take_profit=take_profit,
                    config=policy,
                )
                if decision.action is ExitAction.CLOSE:
                    outcome = (mark - entry_price) * direction / risk
                    break
                if decision.action is ExitAction.TIGHTEN_SL and decision.stop_price is not None:
                    if (decision.stop_price - stop) * direction > 0:
                        stop = decision.stop_price
                elif decision.action is ExitAction.REDUCE_TP and decision.take_profit is not None:
                    take_profit = decision.take_profit

        if outcome is None:
            outcome = (
                (close[min(i + label_cfg.max_holding_bars, n - 1)] - entry_price) * direction / risk
            )

        results.append(
            TradeResult(
                index=int(i),
                realised_r=float(outcome - cost),
                bars_held=bars,
                peak_r=float((extreme - entry_price) * direction / risk),
            )
        )
        blocked_until = i + bars
    return results


def summarise(results: list[TradeResult]) -> dict[str, float]:
    if not results:
        return {"trades": 0, "sum_r": 0.0, "avg_r": 0.0, "profit_factor": 0.0, "max_dd_r": 0.0}
    values = np.array([r.realised_r for r in results])
    equity = np.cumsum(values)
    wins = values[values > 0].sum()
    losses = -values[values <= 0].sum()
    return {
        "trades": len(values),
        "sum_r": round(float(values.sum()), 2),
        "avg_r": round(float(values.mean()), 4),
        "profit_factor": round(float(wins / losses), 3) if losses > 0 else float("inf"),
        "max_dd_r": round(float(np.max(np.maximum.accumulate(equity) - equity)), 2),
        "giveback_r": round(float(sum(max(r.peak_r - r.realised_r, 0.0) for r in results)), 1),
    }


def build_entries(frame: pd.DataFrame, every: int) -> tuple[np.ndarray, np.ndarray]:
    """A neutral, strategy-independent entry set.

    Every ``every``-th bar, direction taken from the prevailing 50-bar trend.
    The point of this harness is to isolate the *exit*: using a real
    strategy's entries would confound the exit's contribution with that
    strategy's edge, and using random directions would produce an entry set
    no bot resembles.
    """
    trend = frame["close"] - frame["close"].rolling(50, min_periods=50).mean()
    indices = np.arange(200, len(frame) - 1, every)
    directions = trend.to_numpy()[indices] > 0
    return indices, np.repeat(directions, 1)


def load_live_trades(db_path: Path, symbol: str) -> pd.DataFrame:
    """Closed live trades with a usable stop distance, oldest first."""
    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            "SELECT id, side, open_price, sl, tp, close_price, open_time, close_time, "
            "       strategy_version, spread_points_at_entry "
            "FROM trades WHERE symbol = ? AND close_time IS NOT NULL AND sl IS NOT NULL "
            "  AND close_price IS NOT NULL ORDER BY open_time",
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()
    frame = frame[(frame["open_price"] - frame["sl"]).abs() > 0].reset_index(drop=True)
    return frame


def replay_live_trades(
    trades: pd.DataFrame, m1: pd.DataFrame, policy: ExitPolicyConfig | None
) -> list[TradeResult]:
    """Re-run each real trade's own price path under a different exit stack.

    This is the check that matters most here, because backtests on this
    project are not predictive — one bot backtests at PF 2.41 and runs at PF
    0.96 live. The entries, stops, spreads and timings below are the ones the
    broker actually filled; only the exit rule is counterfactual.

    The path comes from M1 candles between the trade's own open and close
    times, so the sequence of highs and lows is real rather than modelled.
    Trades whose window has no M1 coverage (the M1 archive starts 2026-04)
    are skipped, not guessed at.
    """
    # Epoch seconds, not datetimes: the trades table stores integer epochs and
    # mixing those with tz-aware Timestamps only invites comparison errors.
    times = m1["time"].to_numpy(dtype=np.int64)
    high = m1["high"].to_numpy(dtype=float)
    low = m1["low"].to_numpy(dtype=float)
    close = m1["close"].to_numpy(dtype=float)
    results: list[TradeResult] = []

    for row in trades.itertuples():
        entry_price = float(row.open_price)
        risk = abs(entry_price - float(row.sl))
        is_buy = row.side == "buy"
        direction = 1.0 if is_buy else -1.0
        start = int(np.searchsorted(times, int(row.open_time)))
        end = int(np.searchsorted(times, int(row.close_time)))
        if end <= start or start >= len(times):
            continue

        stop = float(row.sl)
        take_profit = float(row.tp) if row.tp else None
        extreme = entry_price
        actual_r = (float(row.close_price) - entry_price) * direction / risk

        outcome: float | None = None
        for j in range(start, min(end + 1, len(times))):
            hit_stop = low[j] <= stop if is_buy else high[j] >= stop
            if hit_stop:
                outcome = (stop - entry_price) * direction / risk
                break
            if take_profit is not None:
                hit_tp = high[j] >= take_profit if is_buy else low[j] <= take_profit
                if hit_tp:
                    outcome = (take_profit - entry_price) * direction / risk
                    break
            extreme = max(extreme, high[j]) if is_buy else min(extreme, low[j])
            if policy is None:
                continue
            decision = decide_exit(
                is_buy=is_buy,
                entry_price=entry_price,
                current_sl=stop,
                mark=close[j],
                extreme_favorable=extreme,
                risk=risk,
                take_profit=take_profit,
                config=policy,
            )
            if decision.action is ExitAction.CLOSE:
                outcome = (close[j] - entry_price) * direction / risk
                break
            if (
                decision.action is ExitAction.TIGHTEN_SL
                and decision.stop_price is not None
                and (decision.stop_price - stop) * direction > 0
            ):
                stop = decision.stop_price

        # No counterfactual exit fired: the trade ends exactly as it really did.
        if outcome is None:
            outcome = actual_r
        results.append(
            TradeResult(
                index=start,
                realised_r=float(outcome),
                bars_held=max(end - start, 0),
                peak_r=float((extreme - entry_price) * direction / risk),
            )
        )
    return results


def run_live_mode(db_path: Path, symbol: str, split: float) -> None:
    trades = load_live_trades(db_path, symbol)
    conn = sqlite3.connect(str(db_path))
    try:
        m1 = pd.read_sql_query(
            "SELECT time, high, low, close FROM candles "
            "WHERE symbol = ? AND timeframe = 'M1' ORDER BY time",
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()
    first_epoch, last_epoch = int(m1["time"].iloc[0]), int(m1["time"].iloc[-1])
    covered = trades[trades["open_time"] >= first_epoch]
    logger.info(
        "%d closed live trades, %d inside M1 coverage (%s -> %s)",
        len(trades),
        len(covered),
        pd.to_datetime(first_epoch, unit="s", utc=True),
        pd.to_datetime(last_epoch, unit="s", utc=True),
    )
    split_at = int(len(covered) * split)
    parts = {"IS": covered.iloc[:split_at], "OOS": covered.iloc[split_at:]}

    grid = [
        ("baseline (as traded)", None),
        *[
            (f"arm={arm} keep={keep}", ExitPolicyConfig(arm_r=arm, keep_fraction=keep))
            for arm in (0.3, 0.5, 0.8)
            for keep in (0.4, 0.5, 0.6, 0.75)
        ],
    ]
    print(
        f"\n{'variant':<24}{'IS sumR':>10}{'IS avgR':>10}{'IS PF':>8}"
        f"{'OOS sumR':>11}{'OOS avgR':>10}{'OOS PF':>8}{'OOS DD':>9}{'OOS n':>7}"
    )
    print("-" * 97)
    rows = []
    for name, policy in grid:
        stats = {
            key: summarise(replay_live_trades(part, m1, policy)) for key, part in parts.items()
        }
        rows.append((name, stats["IS"], stats["OOS"]))
        print(
            f"{name:<24}{stats['IS']['sum_r']:>10.1f}{stats['IS']['avg_r']:>10.4f}"
            f"{stats['IS']['profit_factor']:>8.2f}{stats['OOS']['sum_r']:>11.1f}"
            f"{stats['OOS']['avg_r']:>10.4f}{stats['OOS']['profit_factor']:>8.2f}"
            f"{stats['OOS']['max_dd_r']:>9.1f}{stats['OOS']['trades']:>7}"
        )
    tuned = [r for r in rows if r[0] != "baseline (as traded)"]
    correlation = float(
        np.corrcoef([r[1]["avg_r"] for r in tuned], [r[2]["avg_r"] for r in tuned])[0, 1]
    )
    print(f"\n  IS/OOS avg-R correlation across the grid: r={correlation:.2f}")
    print(
        "  verdict: "
        + (
            "parameters transfer — safe to tune"
            if correlation > 0.5
            else "IS and OOS disagree — treat these parameters as NOT tunable"
        )
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--every", type=int, default=37, help="entry every Nth M5 bar")
    parser.add_argument("--split", type=float, default=0.6, help="in-sample fraction")
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--live-trades",
        action="store_true",
        help="replay real closed trades from the trades table instead of synthetic entries",
    )
    args = parser.parse_args()
    if args.live_trades:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        run_live_mode(Path(args.db) if args.db else DB_PATH, args.symbol, args.split)
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frame = load_m5(Path(args.db) if args.db else DB_PATH, args.symbol)
    logger.info("%d M5 bars  %s -> %s", len(frame), frame.index[0], frame.index[-1])

    label_cfg = LabelConfig()
    indices, directions = build_entries(frame, args.every)
    is_buy = np.zeros(len(frame), dtype=bool)
    is_buy[indices] = directions

    split_bar = int(len(frame) * args.split)
    in_sample = indices[indices < split_bar]
    out_sample = indices[indices >= split_bar]
    logger.info(
        "IS %d entries (%s -> %s) | OOS %d entries (%s -> %s)",
        len(in_sample),
        frame.index[in_sample[0]],
        frame.index[in_sample[-1]],
        len(out_sample),
        frame.index[out_sample[0]],
        frame.index[out_sample[-1]],
    )

    grid = [
        ("baseline (engine today)", None),
        *[
            (f"arm={arm} keep={keep}", ExitPolicyConfig(arm_r=arm, keep_fraction=keep))
            for arm in (0.3, 0.5, 0.8, 1.2)
            for keep in (0.4, 0.5, 0.6, 0.75)
        ],
    ]

    print(
        f"\n{'variant':<26}{'IS sumR':>10}{'IS avgR':>10}{'IS PF':>8}"
        f"{'OOS sumR':>11}{'OOS avgR':>10}{'OOS PF':>8}{'OOS DD':>9}{'OOS n':>7}"
    )
    print("-" * 99)
    rows = []
    for name, policy in grid:
        is_stats = summarise(simulate(frame, in_sample, is_buy, label_cfg, policy))
        oos_stats = summarise(simulate(frame, out_sample, is_buy, label_cfg, policy))
        rows.append((name, is_stats, oos_stats))
        print(
            f"{name:<26}{is_stats['sum_r']:>10.1f}{is_stats['avg_r']:>10.4f}"
            f"{is_stats['profit_factor']:>8.2f}{oos_stats['sum_r']:>11.1f}"
            f"{oos_stats['avg_r']:>10.4f}{oos_stats['profit_factor']:>8.2f}"
            f"{oos_stats['max_dd_r']:>9.1f}{oos_stats['trades']:>7}"
        )

    tuned = [r for r in rows if r[1]["trades"] > 0 and r[0] != "baseline (engine today)"]
    if tuned:
        best_is = max(tuned, key=lambda r: r[1]["avg_r"])
        best_oos = max(tuned, key=lambda r: r[2]["avg_r"])
        is_ranks = [r[1]["avg_r"] for r in tuned]
        oos_ranks = [r[2]["avg_r"] for r in tuned]
        correlation = float(np.corrcoef(is_ranks, oos_ranks)[0, 1])
        print(f"\n  best in-sample:     {best_is[0]}  (OOS avg R {best_is[2]['avg_r']:+.4f})")
        print(f"  best out-of-sample: {best_oos[0]}  (IS avg R {best_oos[1]['avg_r']:+.4f})")
        print(f"  IS/OOS avg-R correlation across the grid: r={correlation:.2f}")
        print(
            "  verdict: "
            + (
                "parameters transfer — safe to tune"
                if correlation > 0.5
                else "IS and OOS disagree — treat these parameters as NOT tunable"
            )
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
