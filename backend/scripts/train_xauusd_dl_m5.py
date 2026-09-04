"""Train the XAUUSD M5 learned-entry gate on `dl_features.compute_feature_frame`.

Run from `backend/`::

    uv run python scripts/train_xauusd_dl_m5.py
    uv run python scripts/train_xauusd_dl_m5.py --symbol XAUUSD --c-grid 0.01,0.03,0.1,0.3,1.0

WHY THIS LABEL, NOT A TP-VS-SL BRACKET
────────────────────────────────────────────────────────────────────────
`smc_dl_m5_v2`'s "revert" head predicted market reversal directly and scored
50.4-50.8% out-of-sample accuracy on a ~50/50 base rate across four
walk-forward folds — no skill (see `engine/domain/exit_policy.py`'s module
docstring). Separately, a naive "did TP come before SL" label is simply
wrong on this engine: `PositionManager` rule 2 (secure-base trailing,
`DEFAULT_SECURE_BUFFER_R_MULT = 0.2`) takes ~94% of trades out at exactly
+0.2R within a couple of minutes of entry, so the strategy's own take-profit
almost never decides the outcome (see `smc_dl_labels_v2.py`'s docstring for
the measured numbers, and `domain/online_learning.py`'s for the mechanism).

So this trainer reuses `domain.online_learning.AdaptiveLearner`'s own,
already-validated label definition verbatim rather than re-deriving (or
re-failing) one from scratch:

    y = 1  price's HIGH/LOW wick travels `secure_r` x risk in the trade's
           favour before its wick travels `risk` (1R) against it
           -> the engine's secure-base trailing would lock in ~+0.2R
    y = 0  the adverse move (1R) happens first, OR both happen on the same
           bar (intrabar order unknown -> resolved pessimistically as a
           stop, exactly as `AdaptiveLearner._advance_one` does)

A pending sample that resolves neither way within `max_holding_bars` is
graded by which excursion — favourable or adverse — was larger by the
deadline, at half sample weight (`AdaptiveLearner`'s `expiry_sample_weight
= 0.5`). `risk` is `sl_atr_mult x ATR(14)`, matching how the S&D/apex bots
in this codebase size a stop; `secure_r = 0.2` and `max_holding_bars = 60`
M5 bars (5 hours) mirror `LearnerConfig`'s and `smc_dl_labels_v2.LabelConfig`'s
defaults respectively. `_barrier_labels` below is a vectorised, batch-mode
re-implementation of `AdaptiveLearner._advance_one`'s exact per-sample
control flow (first-hit index comparison, same-bar tie -> stop, expiry
tie-break by mfe vs mae) — not an approximation of it.

Two independent logistic regressions are fit — one per side (long/short) —
sharing the same `dl_features.FEATURE_NAMES` input. Two linear heads is the
numpy-only, honestly-simple equivalent of `smc_dl_model_v2.SmcExitAwareNet`'s
2-sigmoid `secure` head, appropriate because `strategies/sandbox.py`'s
allowlist is meant to admit numpy-only inference for this bot (unlike the
smc_dl_* bots, which additionally carry `torch` on the allowlist).

CHRONOLOGICAL TRAIN / VALIDATION / TEST, NOT RANDOM SHUFFLE
────────────────────────────────────────────────────────────────────────
This project's established bar (`entry-filter-oos-validation-hour-overfits`,
and `train_smc_dl_v2.py`'s walk-forward harness): a per-hour entry filter
here measured PF 2.70 in-sample and PF 0.12 out-of-sample, and a gate
threshold has shown IS/OOS optima with r = 0.00. A single chronological cut
is used (train -> validation -> test, in that order), each boundary purged
by `max_holding_bars` bars so no training or validation label's barrier walk
reads into the following split's price action. `C` (L2 strength) is
selected on validation only; the reported OOS numbers are the test split,
touched exactly once.

THE GATE
────────────────────────────────────────────────────────────────────────
`train()` reports Brier score and ROC-AUC against a constant predictor (the
*training* base rate — the only base rate known at decision time in a real
deployment) on the untouched test split, for both sides. If neither side
clears `MIN_AUC_EDGE` above chance with a non-trivial Brier improvement,
`train()` sets `meta["has_edge"] = False`, prints the verdict, and returns
without writing weights — mirroring `smc_dl_m5_v2`'s honest "no model: rule
fallback" posture rather than shipping a coin flip.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.domain.dl_features import (  # noqa: E402
    FEATURE_NAMES,
    atr,
    compute_feature_frame,
)

logger = logging.getLogger("train_xauusd_dl_m5")

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
MODELS_DIR = BACKEND_DIR / "data" / "ml_models"
HTF_TIMEFRAMES = ("M15", "H1", "H4")
DEFAULT_POINT_VALUE = 0.01

# Mirrors `domain.online_learning.DEFAULT_SECURE_R` /
# `LearnerConfig.secure_r` — the engine's real secure-base trailing buffer.
DEFAULT_SECURE_R = 0.2
# Stop distance the label (and the served strategy's own SL) is measured
# against, matching `smc_dl_labels_v2.DEFAULT_SL_ATR_MULT`.
DEFAULT_SL_ATR_MULT = 1.0
# Mirrors `smc_dl_labels_v2.DEFAULT_MAX_HOLDING_BARS` (5 hours of M5) — the
# bar-count translation of `AdaptiveLearner`'s wall-clock `deadline_ns`.
DEFAULT_MAX_HOLDING_BARS = 60
# `AdaptiveLearner.LearnerConfig.expiry_sample_weight` — a time-barrier
# resolution is genuine information but weaker than a clean barrier touch.
EXPIRY_SAMPLE_WEIGHT = 0.5

# A model is only reported as having an edge if AUC clears chance by this
# margin on the untouched test split, for at least one side. 0.55 is a
# conventional soft floor for a weak-signal financial gate — well short of
# "impressive" on purpose, so a model that clears it barely is reported as
# exactly that, not oversold.
MIN_AUC_EDGE = 0.55
# ...and the Brier improvement over the train-base-rate constant predictor
# must also be a non-trivial fraction of the constant predictor's own score,
# not a rounding-noise improvement.
MIN_BRIER_RELATIVE_IMPROVEMENT = 0.02


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    symbol: str = "XAUUSD"
    model_name: str = "xauusd_dl_m5_v1"
    secure_r: float = DEFAULT_SECURE_R
    sl_atr_mult: float = DEFAULT_SL_ATR_MULT
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS
    point_value: float = DEFAULT_POINT_VALUE
    train_frac: float = 0.6
    val_frac: float = 0.2
    # test_frac is whatever remains.
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
    seed: int = 7


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_candles(db_path: Path, symbol: str) -> dict[str, pd.DataFrame]:
    """OHLCV per timeframe, `time` kept as raw epoch-seconds int64 — the
    shape `dl_features.compute_feature_frame` requires (it does its own
    `pd.to_datetime(df["time"], unit="s")` internally and joins HTF context
    on close-time arithmetic in seconds)."""
    conn = sqlite3.connect(str(db_path))
    try:
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in ("M5", *HTF_TIMEFRAMES):
            frame = pd.read_sql_query(
                "SELECT time, open, high, low, close, tick_volume, spread_points "
                "FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY time",
                conn,
                params=(symbol, timeframe),
            )
            frames[timeframe] = frame
        return frames
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# labels — vectorised re-implementation of AdaptiveLearner._advance_one
# ---------------------------------------------------------------------------
def _shift_forward(values: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """`values[i + offset]` plus an explicit "that bar exists" mask — a NaN
    pad alone would read as "barrier not hit" under `>=`, silently biasing
    every label near the end of the series. Same convention as
    `smc_dl_labels_v2._shift_forward`."""
    n = len(values)
    out = np.empty(n, dtype=float)
    exists = np.zeros(n, dtype=bool)
    if offset < n:
        out[: n - offset] = values[offset:]
        exists[: n - offset] = True
    out[~exists] = np.nan
    return out, exists


def barrier_labels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_values: np.ndarray,
    *,
    long_side: bool,
    secure_r: float,
    sl_atr_mult: float,
    max_holding_bars: int,
    expiry_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`(label, weight)` for every bar, one side, matching
    `AdaptiveLearner._advance_one` bar-for-bar:

    * `risk = sl_atr_mult x ATR(14)`, `secure_dist = secure_r x risk` — the
      same two levels `observe()`/`_advance_one` are given by a caller.
    * Both barriers are checked against **wicks** (high/low), never close —
      `_advance_one` uses `bar_highs`/`bar_lows` throughout, unlike
      `smc_dl_labels_v2._barrier_walk`'s close-triggered secure event. That
      difference is deliberate: this trainer's job is to reproduce
      `online_learning.py`'s validated label exactly, not to improve on it.
    * First-hit bar index is tracked independently for each barrier; a
      same-bar hit on both, or a stop index at-or-before the secure index,
      resolves as a stop (`stop_idx <= secure_idx` in `_advance_one`).
    * A sample that hits neither within `max_holding_bars` is graded by
      `mfe > mae` at `expiry_weight`, mirroring the time-barrier branch.
    * A sample whose horizon runs past the end of the series is dropped
      (unlabelable), not guessed at.
    """
    n = len(close)
    entry = close
    risk = sl_atr_mult * atr_values
    valid = np.isfinite(risk) & (risk > 0)
    secure_dist = secure_r * risk

    secure_idx = np.full(n, -1, dtype=np.int64)
    stop_idx = np.full(n, -1, dtype=np.int64)
    mfe = np.zeros(n, dtype=float)
    mae = np.zeros(n, dtype=float)

    for step in range(1, max_holding_bars + 1):
        hi, exists = _shift_forward(high, step)
        lo, _ = _shift_forward(low, step)

        favourable = (hi - entry) if long_side else (entry - lo)
        adverse = (entry - lo) if long_side else (hi - entry)

        fav_now = np.where(exists, favourable, -np.inf)
        adv_now = np.where(exists, adverse, -np.inf)
        mfe = np.maximum(mfe, fav_now)
        mae = np.maximum(mae, adv_now)

        secure_now = exists & (favourable >= secure_dist)
        stop_now = exists & (adverse >= risk)
        secure_idx[secure_now & (secure_idx == -1)] = step
        stop_idx[stop_now & (stop_idx == -1)] = step

    _, horizon_exists = _shift_forward(high, max_holding_bars)

    stop_first = (stop_idx != -1) & ((secure_idx == -1) | (stop_idx <= secure_idx))
    secure_first = (~stop_first) & (secure_idx != -1)
    neither = (secure_idx == -1) & (stop_idx == -1)
    expired = neither & horizon_exists
    incomplete = neither & ~horizon_exists

    label = np.full(n, np.nan)
    weight = np.ones(n)
    label[stop_first] = 0.0
    label[secure_first] = 1.0
    label[expired] = (mfe[expired] > mae[expired]).astype(float)
    weight[expired] = expiry_weight

    drop = incomplete | ~valid
    label[drop] = np.nan

    return label, weight


def build_dataset(
    candles: dict[str, pd.DataFrame], cfg: TrainConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aligned `(features, labels, weights)`, chronologically ordered,
    warmup/incomplete-horizon rows dropped."""
    m5 = candles["M5"]
    if m5.empty:
        raise RuntimeError(f"no M5 candles for {cfg.symbol}")

    features = compute_feature_frame(candles)
    atr14 = atr(m5.reset_index(drop=True), 14)
    high = m5["high"].to_numpy(dtype=float)
    low = m5["low"].to_numpy(dtype=float)
    close = m5["close"].to_numpy(dtype=float)
    atr_values = atr14.to_numpy(dtype=float)

    label_long, weight_long = barrier_labels(
        high,
        low,
        close,
        atr_values,
        long_side=True,
        secure_r=cfg.secure_r,
        sl_atr_mult=cfg.sl_atr_mult,
        max_holding_bars=cfg.max_holding_bars,
        expiry_weight=EXPIRY_SAMPLE_WEIGHT,
    )
    label_short, weight_short = barrier_labels(
        high,
        low,
        close,
        atr_values,
        long_side=False,
        secure_r=cfg.secure_r,
        sl_atr_mult=cfg.sl_atr_mult,
        max_holding_bars=cfg.max_holding_bars,
        expiry_weight=EXPIRY_SAMPLE_WEIGHT,
    )
    labels = pd.DataFrame(
        {
            "secure_long": label_long,
            "weight_long": weight_long,
            "secure_short": label_short,
            "weight_short": weight_short,
        },
        index=features.index,
    )

    valid = (
        features.notna().all(axis=1)
        & labels["secure_long"].notna()
        & labels["secure_short"].notna()
    )
    features = features.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)
    # Cost-in-R, useful context in the report even though the entry gate
    # itself is threshold-on-probability, not EV-based, for this v1.
    spread_price = (
        m5["spread_points"].astype(float) * cfg.point_value
    ).reset_index(drop=True)
    risk = (cfg.sl_atr_mult * atr14).reset_index(drop=True)
    cost_r = (spread_price / risk.replace(0.0, np.nan)).clip(0.0, 1.0)
    cost_r = cost_r.loc[valid].reset_index(drop=True)
    return features, labels, cost_r.to_frame(name="cost_r")


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    train: slice
    val: slice
    test: slice


def chronological_split(n: int, cfg: TrainConfig, purge: int) -> Split:
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)
    val_end = min(val_end, n)
    # Purge bars whose label horizon overlaps the next split — without this
    # the last `purge` labels of a split are a function of the following
    # split's price action, the same leak `train_smc_dl_v2.py` purges.
    return Split(
        train=slice(0, max(train_end - purge, 1)),
        val=slice(train_end, max(val_end - purge, train_end)),
        test=slice(val_end, n),
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def _fit_side(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[LogisticRegression, float, float]:
    """Sweep `C` on validation Brier, return `(best_model, best_c,
    val_brier)`. No `class_weight='balanced'`: the served gate compares the
    model's own probability to an absolute break-even threshold
    (`1/(1+secure_r)`), so rebalancing would silently change what the
    output number means."""
    best: tuple[LogisticRegression, float, float] | None = None
    for c in c_grid:
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        model.fit(x_train, y_train, sample_weight=w_train)
        p_val = model.predict_proba(x_val)[:, 1]
        score = brier(p_val, y_val)
        if best is None or score < best[2]:
            best = (model, c, score)
    assert best is not None
    return best


def train(cfg: TrainConfig, db_path: Path | None = None) -> dict:
    db = db_path or DB_PATH
    if not db.exists():
        raise FileNotFoundError(f"database not found at {db}")

    logger.info("loading candles for %s", cfg.symbol)
    candles = load_candles(db, cfg.symbol)
    features, labels, cost = build_dataset(candles, cfg)
    n = len(features)
    logger.info("dataset: %d usable M5 bars", n)
    if n < 5000:
        raise RuntimeError(f"only {n} usable bars — too little data to train/validate/test")

    split = chronological_split(n, cfg, purge=cfg.max_holding_bars)
    logger.info(
        "split: train=%d val=%d test=%d (purge=%d bars at each boundary)",
        split.train.stop - split.train.start,
        split.val.stop - split.val.start,
        split.test.stop - split.test.start,
        cfg.max_holding_bars,
    )

    x_all = features.to_numpy(dtype=np.float64)
    mean = x_all[split.train].mean(axis=0)
    scale = x_all[split.train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_std = (x_all - mean) / scale

    results: dict[str, dict] = {}
    fitted: dict[str, LogisticRegression] = {}
    for side, label_col, weight_col in (
        ("long", "secure_long", "weight_long"),
        ("short", "secure_short", "weight_short"),
    ):
        y_all = labels[label_col].to_numpy(dtype=np.float64)
        w_all = labels[weight_col].to_numpy(dtype=np.float64)

        model, best_c, val_brier = _fit_side(
            x_std[split.train],
            y_all[split.train],
            w_all[split.train],
            x_std[split.val],
            y_all[split.val],
            w_all[split.val],
            cfg.c_grid,
            cfg.seed,
        )

        train_base_rate = float(np.average(y_all[split.train], weights=w_all[split.train]))
        p_test = model.predict_proba(x_std[split.test])[:, 1]
        y_test = y_all[split.test]
        w_test = w_all[split.test]

        model_brier_test = brier(p_test, y_test)
        baseline_brier_test = brier(np.full_like(p_test, train_base_rate), y_test)
        relative_improvement = (
            (baseline_brier_test - model_brier_test) / baseline_brier_test
            if baseline_brier_test > 0
            else 0.0
        )
        try:
            auc_test = float(roc_auc_score(y_test, p_test, sample_weight=w_test))
        except ValueError:
            # test split has only one class present — AUC undefined.
            auc_test = 0.5

        p_val = model.predict_proba(x_std[split.val])[:, 1]
        try:
            auc_val = float(roc_auc_score(y_all[split.val], p_val, sample_weight=w_all[split.val]))
        except ValueError:
            auc_val = 0.5

        results[side] = {
            "best_c": best_c,
            "val_brier": round(val_brier, 6),
            "val_auc": round(auc_val, 4),
            "train_base_rate": round(train_base_rate, 4),
            "test_base_rate": round(float(np.average(y_test, weights=w_test)), 4),
            "test_brier_model": round(model_brier_test, 6),
            "test_brier_baseline": round(baseline_brier_test, 6),
            "test_brier_relative_improvement": round(relative_improvement, 4),
            "test_auc": round(auc_test, 4),
            "n_train": int(split.train.stop - split.train.start),
            "n_val": int(split.val.stop - split.val.start),
            "n_test": int(split.test.stop - split.test.start),
        }
        fitted[side] = model

    edge_sides = [
        side
        for side, r in results.items()
        if r["test_auc"] >= MIN_AUC_EDGE
        and r["test_brier_relative_improvement"] >= MIN_BRIER_RELATIVE_IMPROVEMENT
    ]
    has_edge = len(edge_sides) > 0

    meta = {
        "model_name": cfg.model_name,
        "symbol": cfg.symbol,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "label_config": {
            "secure_r": cfg.secure_r,
            "sl_atr_mult": cfg.sl_atr_mult,
            "max_holding_bars": cfg.max_holding_bars,
            "expiry_sample_weight": EXPIRY_SAMPLE_WEIGHT,
        },
        "train_config": asdict(cfg),
        "n_bars_usable": n,
        "gate": {
            "min_auc_edge": MIN_AUC_EDGE,
            "min_brier_relative_improvement": MIN_BRIER_RELATIVE_IMPROVEMENT,
        },
        "results": results,
        "has_edge": has_edge,
        "edge_sides": edge_sides,
    }

    _print_report(meta)

    if not has_edge:
        logger.warning(
            "NO OUT-OF-SAMPLE EDGE on %s: neither side cleared AUC>=%.2f with "
            "Brier improvement>=%.1f%% on the untouched test split. Stopping "
            "here per the task's gate — not saving weights, not building a "
            "strategy around this model.",
            cfg.symbol,
            MIN_AUC_EDGE,
            MIN_BRIER_RELATIVE_IMPROVEMENT * 100,
        )
        return meta

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"{cfg.model_name}.npz"
    meta_path = MODELS_DIR / f"{cfg.model_name}_meta.json"

    long_model = fitted["long"]
    short_model = fitted["short"]
    np.savez(
        weights_path,
        mean=mean,
        scale=scale,
        coef_long=long_model.coef_.reshape(-1),
        intercept_long=long_model.intercept_.reshape(-1),
        coef_short=short_model.coef_.reshape(-1),
        intercept_short=short_model.intercept_.reshape(-1),
        secure_r=np.array([cfg.secure_r]),
        sl_atr_mult=np.array([cfg.sl_atr_mult]),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("saved %s and %s", weights_path.name, meta_path.name)
    return meta


def _print_report(meta: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {meta['model_name']}  —  {meta['symbol']}  ({meta['n_bars_usable']} usable M5 bars)")
    print("=" * 78)
    for side, r in meta["results"].items():
        print(f"\n  --- {side.upper()} ---")
        print(f"    best C (selected on validation): {r['best_c']}")
        print(f"    validation: Brier={r['val_brier']}  AUC={r['val_auc']}")
        print(
            f"    train base rate (secure_r={meta['label_config']['secure_r']}): "
            f"{r['train_base_rate']}"
        )
        print(
            f"    TEST (out-of-sample, touched once): base_rate={r['test_base_rate']}  "
            f"n={r['n_test']}"
        )
        print(
            f"      Brier   model={r['test_brier_model']}   "
            f"baseline(constant={r['train_base_rate']})={r['test_brier_baseline']}   "
            f"relative improvement={r['test_brier_relative_improvement'] * 100:.2f}%"
        )
        print(f"      AUC     {r['test_auc']}  (0.5 = chance)")
    print("\n  " + "-" * 74)
    gate = meta["gate"]
    verdict = "HAS OUT-OF-SAMPLE EDGE" if meta["has_edge"] else "NO OUT-OF-SAMPLE EDGE"
    print(
        f"  GATE: AUC >= {gate['min_auc_edge']} AND Brier relative improvement >= "
        f"{gate['min_brier_relative_improvement'] * 100:.0f}%  ->  {verdict}"
    )
    if meta["has_edge"]:
        print(f"  side(s) clearing the gate: {meta['edge_sides']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=os.environ.get("SYMBOL", "XAUUSD"))
    parser.add_argument("--model-name", default="xauusd_dl_m5_v1")
    parser.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--secure-r", type=float, default=DEFAULT_SECURE_R)
    parser.add_argument("--sl-atr-mult", type=float, default=DEFAULT_SL_ATR_MULT)
    parser.add_argument("--max-holding-bars", type=int, default=DEFAULT_MAX_HOLDING_BARS)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    c_grid = tuple(float(x) for x in args.c_grid.split(","))
    cfg = TrainConfig(
        symbol=args.symbol,
        model_name=args.model_name,
        secure_r=args.secure_r,
        sl_atr_mult=args.sl_atr_mult,
        max_holding_bars=args.max_holding_bars,
        c_grid=c_grid,
    )
    train(cfg, Path(args.db) if args.db else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
