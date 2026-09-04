"""Measure the structure-pivot profit-lock rule against real closed trades.

Run from ``backend/``::

    uv run python scripts/eval_structure_pivot_exit.py --symbol XAUUSD

WHAT IT COMPARES
────────────────────────────────────────────────────────────────────────
Same replay principle as ``eval_giveback_exit.py --live-trades`` (see that
module's docstring): each real closed trade's own recorded entry/SL/TP is
re-run over its own M1 price path — the entries, stops, spreads and timings
are exactly what the broker filled; only the exit rule is counterfactual.

  baseline  the trade exactly as it actually closed (no counterfactual
            exit at all — this is deliberately *not* "current engine
            behaviour with every rule on", because isolating the marginal
            effect of the structure-pivot rule specifically is the point;
            see the module-level "SCOPE" note below).
  policy    baseline, plus ``engine.domain.structure_pivot.
            decide_structure_pivot_exit`` applied every M1 step, using the
            same swing-pivot detector production `PositionManager` calls
            (`detect_swing_pivots`), computed from the symbol's real M5
            candle history.

SCOPE: isolated, not composed
────────────────────────────────────────────────────────────────────────
`PositionManager` runs this rule alongside four existing ones (breakeven,
secure-base, zone-contraire, give-back), each of which can tighten the stop
independently. Replaying "all five rules together vs today's four" would
conflate this rule's own contribution with those other rules' already-
measured effects (`eval_giveback_exit.py`, `eval_giveback_per_bot.py`).
This script isolates the new rule specifically: baseline is the trade as it
actually happened (i.e. under whatever rules were live at the time), policy
is that same path with *only* the structure-pivot candidate additionally
applied. That answers "what does this rule add on top of what already
happened" — the right question for a rule being evaluated for activation,
not "what would zero rules vs five rules produce."

HOW PIVOTS ARE DETECTED AND WHEN THEY BECOME KNOWABLE
────────────────────────────────────────────────────────────────────────
`detect_swing_pivots` is run once over the symbol's whole M5 history (pure
function, no lookahead concern in itself), but a pivot at bar index `i` is
only *knowable* once its confirming bars (`pivot_bars` bars after `i`) have
closed — so each pivot is additionally stamped with a `confirmed_epoch`
(the close of that confirming bar) and excluded from selection at any M1
timestamp before that. This mirrors `PositionManager`'s own natural lag: it
can only see a pivot once enough live M5 candles have closed.

HOW IT IS SCORED
────────────────────────────────────────────────────────────────────────
Chronological split, never random — see `eval_giveback_exit.py`'s docstring
for why. Win rate is not reported for the same reason it is not reported
there: with several 0.1-0.2R trailing rules already live, it is an artifact
of exit mechanics, not of the entry.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.eval_giveback_exit import TradeResult, load_live_trades, summarise  # noqa: E402
from src.engine.domain.structure_pivot import (  # noqa: E402
    StructurePivotConfig,
    detect_swing_pivots,
)
from src.strategies.domain.models import StructureLabel  # noqa: E402

logger = logging.getLogger("eval_structure_pivot_exit")
BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
M5_CONFIRM_SECONDS = 300  # one M5 bar's own duration, added to a pivot's confirming-bar close


def _load_candles(db_path: Path, symbol: str, timeframe: str, columns: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(
            f"SELECT {columns} FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY time",
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class PivotSeries:
    """Pivots detected on a symbol's full M5 history, split by label and
    flattened to parallel numpy arrays (price / own-bar epoch / the epoch
    each becomes knowable) — precomputed once per symbol, independent of
    any one trade, since detection doesn't depend on the trade being
    replayed. Plain arrays rather than a list of `SwingPivot` so the
    per-trade replay below can walk them with a monotonic integer cursor
    instead of re-scanning `SwingPivot.time` (a `Timestamp`) on every bar —
    see `replay_with_structure_pivot`'s docstring for why that rescan was
    the harness's actual bottleneck, not the production rule."""

    hl_price: np.ndarray
    hl_epoch: np.ndarray
    hl_confirmed: np.ndarray
    lh_price: np.ndarray
    lh_epoch: np.ndarray
    lh_confirmed: np.ndarray


def build_pivot_series(m5: pd.DataFrame, pivot_bars: int) -> PivotSeries:
    highs = m5["high"].to_numpy(dtype=float)
    lows = m5["low"].to_numpy(dtype=float)
    epochs = m5["time"].to_numpy(dtype=np.int64)
    times = pd.to_datetime(epochs, unit="s", utc=True)
    pivots = detect_swing_pivots(highs, lows, times, pivot_bars=pivot_bars)

    def _confirmed(index: int) -> int:
        return int(epochs[min(index + pivot_bars, len(epochs) - 1)]) + M5_CONFIRM_SECONDS

    hl = [p for p in pivots if p.label is StructureLabel.HL]
    lh = [p for p in pivots if p.label is StructureLabel.LH]
    return PivotSeries(
        hl_price=np.array([p.price for p in hl], dtype=float),
        hl_epoch=np.array([epochs[p.index] for p in hl], dtype=np.int64),
        hl_confirmed=np.array([_confirmed(p.index) for p in hl], dtype=np.int64),
        lh_price=np.array([p.price for p in lh], dtype=float),
        lh_epoch=np.array([epochs[p.index] for p in lh], dtype=np.int64),
        lh_confirmed=np.array([_confirmed(p.index) for p in lh], dtype=np.int64),
    )


def replay_with_structure_pivot(
    trades: pd.DataFrame,
    m1: pd.DataFrame,
    series: PivotSeries | None,
    config: StructurePivotConfig | None,
) -> list[TradeResult]:
    """Re-run each real trade's own price path with the structure-pivot
    rule as the only counterfactual (see module docstring's SCOPE note —
    `config=None` replays the trade exactly as it actually closed, not
    under a hypothetical zero-rules baseline).

    Pivot selection is a two-pointer sweep, not a per-bar search: `cursor`
    only advances forward as `now_epoch` increases across a trade's own bar
    loop (bars are visited in increasing time order), consuming each
    knowable pivot at most once and keeping `best_price` as the most recent
    one seen so far that clears `min_pivot_r`. An earlier version of this
    function re-scanned the pivot list backwards from scratch on every
    single bar — correct, but O(bars_in_trade x pivots_after_entry) per
    trade, which is why the first run of this script (12k+ trades, ~7,500
    HL pivots) did not finish inside a 10-minute budget. This version is
    O(bars_in_trade + pivots_after_entry) per trade."""
    times = m1["time"].to_numpy(dtype=np.int64)
    high = m1["high"].to_numpy(dtype=float)
    low = m1["low"].to_numpy(dtype=float)
    results: list[TradeResult] = []
    use_pivots = config is not None and config.enabled and series is not None

    for row in trades.itertuples():
        entry_price = float(row.open_price)
        risk = abs(entry_price - float(row.sl))
        if risk <= 0:
            continue
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
        entry_epoch = int(row.open_time)

        cursor = 0
        n_pivots = 0
        pivot_price = pivot_epoch = pivot_confirmed = None
        best_price: float | None = None
        if use_pivots:
            assert config is not None and series is not None
            pivot_price = series.hl_price if is_buy else series.lh_price
            pivot_epoch = series.hl_epoch if is_buy else series.lh_epoch
            pivot_confirmed = series.hl_confirmed if is_buy else series.lh_confirmed
            n_pivots = len(pivot_confirmed)
            # Only pivots formed strictly after this trade's own entry ever
            # qualify (a "NEW swing-structure point in the trade's
            # direction" — see `select_ratchet_pivot`'s docstring), so the
            # sweep starts there rather than at 0.
            cursor = int(np.searchsorted(pivot_epoch, entry_epoch, side="right"))

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
            if not use_pivots:
                continue
            assert config is not None
            peak_r_value = max((extreme - entry_price) * direction / risk, 0.0)
            if peak_r_value < config.arm_r:
                continue
            now_epoch = int(times[j])
            while cursor < n_pivots and pivot_confirmed[cursor] <= now_epoch:  # type: ignore[index]
                progress_r = (pivot_price[cursor] - entry_price) * direction / risk  # type: ignore[index]
                if progress_r >= config.min_pivot_r:
                    best_price = float(pivot_price[cursor])  # type: ignore[index]
                cursor += 1
            if best_price is None:
                continue
            candidate = best_price - direction * config.buffer_r_mult * risk
            if (candidate - stop) * direction > 0:
                stop = candidate

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


def run(
    db_path: Path,
    symbol: str,
    split: float,
    configs: list[tuple[str, StructurePivotConfig | None]],
) -> None:
    trades = load_live_trades(db_path, symbol)
    m1 = _load_candles(db_path, symbol, "M1", "time, high, low, close")
    m5 = _load_candles(db_path, symbol, "M5", "time, high, low, close")
    if m1.empty or m5.empty:
        logger.error("no M1/M5 candles for %s", symbol)
        return
    first_epoch, last_epoch = int(m1["time"].iloc[0]), int(m1["time"].iloc[-1])
    covered = trades[trades["open_time"] >= first_epoch]
    logger.info(
        "%d closed live trades, %d inside M1 coverage (%s -> %s), %d M5 bars",
        len(trades),
        len(covered),
        pd.to_datetime(first_epoch, unit="s", utc=True),
        pd.to_datetime(last_epoch, unit="s", utc=True),
        len(m5),
    )
    split_at = int(len(covered) * split)
    parts = {"IS": covered.iloc[:split_at], "OOS": covered.iloc[split_at:]}

    # Cache one PivotSeries per distinct pivot_bars value across the grid,
    # since detection is independent of arm_r/buffer_r_mult/min_pivot_r and
    # is the most expensive step (a full pass over the M5 history).
    series_cache: dict[int, PivotSeries] = {}

    def _series_for(pivot_bars: int) -> PivotSeries:
        if pivot_bars not in series_cache:
            series_cache[pivot_bars] = build_pivot_series(m5, pivot_bars)
        return series_cache[pivot_bars]

    print(
        f"\n{'variant':<28}{'IS sumR':>10}{'IS avgR':>10}{'IS PF':>8}"
        f"{'OOS sumR':>11}{'OOS avgR':>10}{'OOS PF':>8}{'OOS DD':>9}{'OOS n':>7}"
    )
    print("-" * 100)
    rows = []
    for name, config in configs:
        series = _series_for(config.pivot_bars) if config is not None else None
        stats = {}
        for key, part in parts.items():
            replayed = replay_with_structure_pivot(part, m1, series, config)
            stats[key] = summarise(replayed)
        rows.append((name, stats["IS"], stats["OOS"]))
        print(
            f"{name:<28}{stats['IS']['sum_r']:>10.1f}{stats['IS']['avg_r']:>10.4f}"
            f"{stats['IS']['profit_factor']:>8.2f}{stats['OOS']['sum_r']:>11.1f}"
            f"{stats['OOS']['avg_r']:>10.4f}{stats['OOS']['profit_factor']:>8.2f}"
            f"{stats['OOS']['max_dd_r']:>9.1f}{stats['OOS']['trades']:>7}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", type=float, default=0.6, help="in-sample fraction")
    parser.add_argument("--db", default=None)
    parser.add_argument("--pivot-bars", type=int, default=2)
    parser.add_argument("--arm-r", type=float, default=1.0)
    parser.add_argument("--buffer-r-mult", type=float, default=0.1)
    parser.add_argument("--min-pivot-r", type=float, default=0.0)
    parser.add_argument(
        "--grid", action="store_true", help="also sweep a small arm_r/pivot_bars grid"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    configs: list[tuple[str, StructurePivotConfig | None]] = [("baseline (as traded)", None)]
    configs.append(
        (
            f"pivot_bars={args.pivot_bars} arm_r={args.arm_r} buf={args.buffer_r_mult}",
            StructurePivotConfig(
                pivot_bars=args.pivot_bars,
                arm_r=args.arm_r,
                buffer_r_mult=args.buffer_r_mult,
                min_pivot_r=args.min_pivot_r,
            ),
        )
    )
    if args.grid:
        for pivot_bars in (2, 3):
            for arm_r in (0.5, 1.0, 1.5):
                configs.append(
                    (
                        f"pivot_bars={pivot_bars} arm_r={arm_r} buf={args.buffer_r_mult}",
                        StructurePivotConfig(
                            pivot_bars=pivot_bars,
                            arm_r=arm_r,
                            buffer_r_mult=args.buffer_r_mult,
                            min_pivot_r=args.min_pivot_r,
                        ),
                    )
                )

    run(Path(args.db) if args.db else DB_PATH, args.symbol, args.split, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
