"""Train the SMC exit-aware model for a given entry timeframe, with
walk-forward chronological validation.

Run from ``backend/``::

    uv run python scripts/train_smc_dl_v3.py --timeframe M5
    uv run python scripts/train_smc_dl_v3.py --timeframe M1
    uv run python scripts/train_smc_dl_v3.py --timeframe M1 --folds 5 --epochs 12

This is ``scripts/train_smc_dl_v2.py`` generalized to any entry timeframe via
``smc_dl_features_v3.compute_features_batch``'s rung-relative HTF interface
(``htf_candles=(rung1, rung2, rung3)`` rather than v2's fixed
``m15/h1/h4`` kwargs). Everything v2 already validated is reused verbatim:
walk-forward expanding-window folds (never a single split), purge-by-
``max_holding_bars`` at each fold boundary, temperature calibration fit only
on a training-tail slice (never on OOS), Brier + reliability tables,
permutation importance, and an economic score (non-overlapping simulated
trades, cost-charged, PF/sum-R/avg-R/drawdown/count — deliberately no win
rate, since ~94% of trades on this engine close at exactly +0.2R regardless
of entry quality) plus the ``threshold_tunable`` verdict that reports,
rather than resolves, whether the in-sample-optimal EV threshold predicts
the out-of-sample-optimal one.

TIMEFRAME CONFIG
────────────────────────────────────────────────────────────────────────
``--timeframe {M1,M5}`` selects both the confirmation-rung ladder and the
label horizon:

  * M1  entry -> confirmation rungs (M5, M15, H1) — the same triple
        ``xauusd_snd_apex_trendguard_m1_v5.py`` already declares for its own
        M1 confirmation ladder.
  * M5  entry -> confirmation rungs (M15, H1, H4) — unchanged from v2.

``LabelConfig.max_holding_bars``/``revert_bars`` encode real wall-clock
time (5 hours / 1 hour respectively, per ``smc_dl_labels_v2``'s own
docstring), so they are rescaled by bars-per-minute rather than reused
verbatim across timeframes: M5's 60/12 bars stay 60/12 (5h/1h at 5 min per
bar); M1's equivalent is 300/60 bars (5h/1h at 1 min per bar). The R-multiple
and ATR-ratio fields (``secure_r``, ``target_r``, ``sl_atr_mult``,
``revert_atr_mult``) are dimensionless and stay at v2's defaults for both.

Reported win rate is deliberately absent, for the same reason as v2: on this
engine ~94% of trades close at exactly +0.20R, so win rate is a property of
the trailing rule, not of the model. PF / sum R / avg R / drawdown / count
are the metrics.
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
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.generated.smc_dl_features_v3 import (  # noqa: E402
    FEATURE_NAMES,
    atr,
    compute_features_batch,
)
from src.strategies.generated.smc_dl_labels_v2 import (  # noqa: E402
    LabelConfig,
    breakeven_secure_probability,
    generate_labels_v2,
)
from src.strategies.generated.smc_dl_model_v2 import SmcExitAwareNet  # noqa: E402

logger = logging.getLogger("train_smc_dl_v3")

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
MODELS_DIR = BACKEND_DIR / "data" / "ml_models"
# XAUUSD quotes to 2 decimals, so one point is 0.01 of a dollar.
DEFAULT_POINT_VALUE = 0.01

# Entry-timeframe -> (confirmation rung ladder, minutes per bar). The rung
# ladder is what gets passed to `compute_features_batch(htf_candles=...)`;
# `bar_per_minute` is only used to rescale LabelConfig's bar-count fields so
# they keep meaning the same real-time window across timeframes (see module
# docstring, "TIMEFRAME CONFIG").
TIMEFRAME_CONFIG: dict[str, dict[str, object]] = {
    "M1": {"confirmation": ("M5", "M15", "H1"), "bar_per_minute": 1},
    "M5": {"confirmation": ("M15", "H1", "H4"), "bar_per_minute": 5},
}

# Real-time windows LabelConfig's default bar counts encode, per
# smc_dl_labels_v2's own docstring: max_holding_bars = 5 hours, revert_bars =
# 1 hour. Rescaled by bar_per_minute so an M1 run keeps the same real-time
# meaning as M5's 60/12 defaults, rather than reusing the bar counts verbatim.
_MAX_HOLDING_MINUTES = 300
_REVERT_MINUTES = 60


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    symbol: str = "XAUUSD"
    model_name: str = "smc_dl_m5_v3"
    entry_timeframe: str = "M5"
    confirmation_timeframes: tuple[str, str, str] = ("M15", "H1", "H4")
    hidden_dim: int = 128
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 12
    batch_size: int = 512
    folds: int = 5
    min_train_fraction: float = 0.4
    point_value: float = DEFAULT_POINT_VALUE
    # Gate threshold in expected-R used when scoring a fold economically.
    ev_threshold: float = 0.02
    seed: int = 7


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_candles(
    db_path: Path, symbol: str, timeframes: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(str(db_path))
    try:
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            frame = pd.read_sql_query(
                "SELECT time, open, high, low, close, tick_volume, spread_points "
                "FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY time",
                conn,
                params=(symbol, timeframe),
            )
            if not frame.empty:
                frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
                frame = frame.set_index("time", drop=False)
            frames[timeframe] = frame
        return frames
    finally:
        conn.close()


def build_dataset(
    candles: dict[str, pd.DataFrame], cfg: TrainConfig, labels: LabelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Aligned (features, labels, spread-cost-in-R), chronologically ordered."""
    entry = candles[cfg.entry_timeframe]
    if entry.empty:
        raise RuntimeError(f"no {cfg.entry_timeframe} candles for {cfg.symbol}")

    rungs = tuple(candles.get(tf) for tf in cfg.confirmation_timeframes)
    features = compute_features_batch(
        entry,
        htf_candles=rungs,
        spread_points=entry["spread_points"].to_numpy(dtype=float),
        point_value=cfg.point_value,
    )
    atr14 = atr(entry, 14)
    label_frame = generate_labels_v2(entry, atr14, labels)

    # Round-trip spread expressed in R (risk = sl_atr_mult x ATR), the unit
    # the expected-R formula works in. Entry pays the spread; the exit leg is
    # a resting order, so one crossing is charged, not two.
    spread_price = entry["spread_points"].astype(float) * cfg.point_value
    cost_r = (spread_price / (labels.sl_atr_mult * atr14)).clip(0.0, 1.0)

    valid = (
        features.notna().all(axis=1)
        & label_frame[["secure_long", "secure_short", "target_long", "target_short"]]
        .notna()
        .all(axis=1)
        & cost_r.notna()
    )
    return features[valid], label_frame[valid], cost_r[valid]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _tensors(
    features: pd.DataFrame, labels: pd.DataFrame, mean: np.ndarray, scale: np.ndarray
) -> dict[str, torch.Tensor]:
    x = (features.to_numpy(dtype=np.float32) - mean) / scale
    return {
        "x": torch.from_numpy(x.astype(np.float32)),
        "secure": torch.from_numpy(
            labels[["secure_long", "secure_short"]].to_numpy(dtype=np.float32)
        ),
        "target": torch.from_numpy(
            labels[["target_long", "target_short"]].to_numpy(dtype=np.float32)
        ),
        "revert": torch.from_numpy(labels["revert_class"].to_numpy(dtype=np.int64)),
        "mfe": torch.from_numpy(labels[["mfe_long_r", "mfe_short_r"]].to_numpy(dtype=np.float32)),
    }


def fit(train: dict[str, torch.Tensor], cfg: TrainConfig, input_dim: int) -> SmcExitAwareNet:
    torch.manual_seed(cfg.seed)
    model = SmcExitAwareNet(input_dim=input_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    # `revert_class == -1` marks bars whose intrabar order is unknown or whose
    # horizon ran past the series end. ignore_index drops them from the loss
    # rather than inventing a class for them.
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    huber = nn.SmoothL1Loss()

    n = train["x"].shape[0]
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(n)
        total = 0.0
        batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = permutation[start : start + cfg.batch_size]
            optimiser.zero_grad()
            secure_logits, target_logits, revert_logits, mfe = model(train["x"][idx])
            loss = (
                bce(secure_logits, train["secure"][idx])
                + bce(target_logits, train["target"][idx])
                + ce(revert_logits, train["revert"][idx])
                + 0.2 * huber(mfe, train["mfe"][idx])
            )
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        if epoch % 4 == 0 or epoch == 1:
            logger.info("    epoch %2d/%d  train loss %.4f", epoch, cfg.epochs, total / batches)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: SmcExitAwareNet, x: torch.Tensor, temperature: float = 1.0
) -> dict[str, np.ndarray]:
    secure_logits, target_logits, revert_logits, mfe = model(x)
    return {
        "p_secure": torch.sigmoid(secure_logits / temperature).numpy(),
        "p_target": torch.sigmoid(target_logits / temperature).numpy(),
        "p_revert": torch.softmax(revert_logits, dim=-1).numpy(),
        "mfe_r": mfe.numpy(),
    }


def fit_temperature(model: SmcExitAwareNet, holdout: dict[str, torch.Tensor]) -> float:
    """Single scalar temperature on the secure/target logits, fitted on a
    chronologically held-out slice of the training window (never on the OOS
    block being scored). Miscalibration here propagates straight into the
    expected-R gate, so it is fitted, not assumed."""
    with torch.no_grad():
        secure_logits, target_logits, _revert, _mfe = model(holdout["x"])
    logits = torch.cat([secure_logits, target_logits], dim=1)
    targets = torch.cat([holdout["secure"], holdout["target"]], dim=1)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = bce(logits / torch.exp(log_temperature), targets)
        loss.backward()
        return loss

    optimiser.step(closure)  # type: ignore[arg-type]
    return float(torch.exp(log_temperature).item())


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def reliability_table(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "predicted": round(float(probabilities[mask].mean()), 4),
                "actual": round(float(outcomes[mask].mean()), 4),
            }
        )
    return rows


def expected_r_arrays(
    predictions: dict[str, np.ndarray], cost_r: np.ndarray, label_cfg: LabelConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``expected_r`` for both sides, with the same clamp."""
    result = []
    for side in (0, 1):
        p_secure = np.clip(predictions["p_secure"][:, side], 0.0, 1.0)
        p_target = np.clip(predictions["p_target"][:, side], 0.0, 1.0)
        p_target = np.minimum(p_target, p_secure)
        result.append(
            p_target * label_cfg.target_r
            + (p_secure - p_target) * label_cfg.secure_r
            - (1.0 - p_secure)
            - cost_r
        )
    return result[0], result[1]


def economic_score(
    predictions: dict[str, np.ndarray],
    labels: pd.DataFrame,
    cost_r: np.ndarray,
    label_cfg: LabelConfig,
    ev_threshold: float,
) -> dict[str, float]:
    """Realised R of the trades this model's gate would actually have taken.

    Two properties make this comparable to a backtest rather than to a
    fantasy:

    * **Non-overlapping.** A naive per-bar mask counts a trade on every bar
      above threshold — 73% of bars in one measured run — while a 60-bar
      holding horizon means those are almost all the same move, booked
      dozens of times. Here the scan is chronological and a new trade is
      only opened when the previous one has resolved, mirroring the engine's
      one-position-per-bot reality.
    * **Cost is charged.** Transaction cost measured 116% of gross profit on
      this account, so an evaluation that ignores spread ranks strategies by
      an irrelevant quantity.

    Payoff comes from the labels, i.e. the same three-outcome accounting the
    model was trained on: ``target_r`` if the target was reached,
    ``secure_r`` if it secured then stopped, ``-1`` otherwise.
    """
    ev_long, ev_short = expected_r_arrays(predictions, cost_r, label_cfg)

    payoff_long = np.where(
        labels["target_long"].to_numpy() > 0,
        label_cfg.target_r,
        np.where(labels["secure_long"].to_numpy() > 0, label_cfg.secure_r, -1.0),
    )
    payoff_short = np.where(
        labels["target_short"].to_numpy() > 0,
        label_cfg.target_r,
        np.where(labels["secure_short"].to_numpy() > 0, label_cfg.secure_r, -1.0),
    )
    bars_long = np.nan_to_num(
        labels["bars_to_stop_long"].to_numpy(), nan=float(label_cfg.max_holding_bars)
    )
    bars_short = np.nan_to_num(
        labels["bars_to_stop_short"].to_numpy(), nan=float(label_cfg.max_holding_bars)
    )

    realised: list[float] = []
    blocked_until = -1
    for i in range(len(ev_long)):
        if i <= blocked_until:
            continue
        long_better = ev_long[i] >= ev_short[i]
        ev = ev_long[i] if long_better else ev_short[i]
        if ev < ev_threshold:
            continue
        payoff = payoff_long[i] if long_better else payoff_short[i]
        held = bars_long[i] if long_better else bars_short[i]
        realised.append(float(payoff - cost_r[i]))
        blocked_until = i + int(max(held, 1))

    if not realised:
        return {
            "trades": 0,
            "sum_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
        }

    values = np.asarray(realised)
    equity = np.cumsum(values)
    drawdown = float(np.max(np.maximum.accumulate(equity) - equity))
    wins = values[values > 0].sum()
    losses = -values[values <= 0].sum()
    return {
        "trades": int(values.size),
        "sum_r": round(float(values.sum()), 2),
        "avg_r": round(float(values.mean()), 4),
        "profit_factor": round(float(wins / losses), 3) if losses > 0 else float("inf"),
        "max_drawdown_r": round(drawdown, 2),
    }


# Thresholds swept on every fold, in and out of sample. The point is not to
# pick the best one: if the IS and OOS optima anti-correlate, the parameter
# is not tunable and must be reported as such rather than fitted. That has
# already happened once on this data (IS peak 0.60, OOS peak 0.75 on a gate
# threshold) and once catastrophically (per-hour filter, PF 2.70 IS / 0.12
# OOS).
EV_SWEEP = (-0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def threshold_sweep(
    predictions: dict[str, np.ndarray],
    labels: pd.DataFrame,
    cost_r: np.ndarray,
    label_cfg: LabelConfig,
) -> list[dict]:
    return [
        {
            "threshold": threshold,
            **economic_score(predictions, labels, cost_r, label_cfg, threshold),
        }
        for threshold in EV_SWEEP
    ]


def permutation_importance(
    model: SmcExitAwareNet,
    tensors: dict[str, torch.Tensor],
    temperature: float,
    top_k: int = 15,
) -> list[dict]:
    """Drop in secure-head Brier when a single feature column is shuffled."""
    baseline = predict(model, tensors["x"], temperature)
    base_score = brier(baseline["p_secure"], tensors["secure"].numpy())
    rng = np.random.default_rng(0)
    scores: list[tuple[str, float]] = []
    x = tensors["x"].clone()
    for index, name in enumerate(FEATURE_NAMES):
        column = x[:, index].clone()
        x[:, index] = column[torch.from_numpy(rng.permutation(len(column)))]
        shuffled = predict(model, x, temperature)
        scores.append((name, brier(shuffled["p_secure"], tensors["secure"].numpy()) - base_score))
        x[:, index] = column
    scores.sort(key=lambda item: item[1], reverse=True)
    return [{"feature": name, "brier_increase": round(delta, 6)} for name, delta in scores[:top_k]]


# ---------------------------------------------------------------------------
# walk-forward driver
# ---------------------------------------------------------------------------
def walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cost_r: pd.Series,
    cfg: TrainConfig,
    label_cfg: LabelConfig,
) -> list[dict]:
    n = len(features)
    first = int(n * cfg.min_train_fraction)
    block = max((n - first) // cfg.folds, 1)
    results: list[dict] = []

    for fold in range(cfg.folds):
        train_end = first + fold * block
        test_end = min(train_end + block, n)
        if test_end - train_end < 500 or train_end < 2000:
            continue

        # Purge the bars whose label horizon overlaps the OOS block. Without
        # this the last `max_holding_bars` training labels are functions of
        # OOS price action — a subtle leak that inflates every fold.
        purge = label_cfg.max_holding_bars
        train_slice = slice(0, max(train_end - purge, 1))
        # The temperature is fitted on the tail of the training window, never
        # on the OOS block it will be scored against.
        calib_start = max(int(train_slice.stop * 0.9), 1)

        x_train = features.iloc[train_slice]
        mean = x_train.to_numpy(dtype=np.float64).mean(axis=0)
        scale = x_train.to_numpy(dtype=np.float64).std(axis=0)
        scale[scale < 1e-8] = 1.0

        train_tensors = _tensors(
            features.iloc[:calib_start], labels.iloc[:calib_start], mean, scale
        )
        calib_tensors = _tensors(
            features.iloc[calib_start : train_slice.stop],
            labels.iloc[calib_start : train_slice.stop],
            mean,
            scale,
        )
        test_tensors = _tensors(
            features.iloc[train_end:test_end], labels.iloc[train_end:test_end], mean, scale
        )

        logger.info(
            "  fold %d/%d  train=%d  calib=%d  oos=%d  [%s -> %s]",
            fold + 1,
            cfg.folds,
            train_tensors["x"].shape[0],
            calib_tensors["x"].shape[0],
            test_tensors["x"].shape[0],
            features.index[train_end],
            features.index[test_end - 1],
        )
        model = fit(train_tensors, cfg, input_dim=len(FEATURE_NAMES))
        temperature = (
            fit_temperature(model, calib_tensors) if calib_tensors["x"].shape[0] > 100 else 1.0
        )

        in_sample = predict(model, train_tensors["x"], temperature)
        out_sample = predict(model, test_tensors["x"], temperature)
        cost_is = cost_r.iloc[:calib_start].to_numpy()
        cost_oos = cost_r.iloc[train_end:test_end].to_numpy()

        results.append(
            {
                "fold": fold + 1,
                "oos_start": str(features.index[train_end]),
                "oos_end": str(features.index[test_end - 1]),
                "temperature": round(temperature, 4),
                "brier_secure_is": round(
                    brier(in_sample["p_secure"], train_tensors["secure"].numpy()), 5
                ),
                "brier_secure_oos": round(
                    brier(out_sample["p_secure"], test_tensors["secure"].numpy()), 5
                ),
                "brier_target_oos": round(
                    brier(out_sample["p_target"], test_tensors["target"].numpy()), 5
                ),
                "revert_acc_oos": _revert_accuracy(out_sample["p_revert"], test_tensors["revert"]),
                "base_rate_secure_oos": round(float(test_tensors["secure"].numpy().mean()), 4),
                "economics_is": economic_score(
                    in_sample,
                    labels.iloc[:calib_start],
                    cost_is,
                    label_cfg,
                    cfg.ev_threshold,
                ),
                "economics_oos": economic_score(
                    out_sample,
                    labels.iloc[train_end:test_end],
                    cost_oos,
                    label_cfg,
                    cfg.ev_threshold,
                ),
                "sweep_is": threshold_sweep(
                    in_sample, labels.iloc[:calib_start], cost_is, label_cfg
                ),
                "sweep_oos": threshold_sweep(
                    out_sample, labels.iloc[train_end:test_end], cost_oos, label_cfg
                ),
                "reliability_secure_oos": reliability_table(
                    out_sample["p_secure"][:, 0], test_tensors["secure"].numpy()[:, 0]
                ),
                "importance_oos": permutation_importance(model, test_tensors, temperature),
            }
        )
    return results


def _revert_accuracy(p_revert: np.ndarray, targets: torch.Tensor) -> float:
    y = targets.numpy()
    mask = y >= 0
    if not mask.any():
        return 0.0
    return round(float((p_revert[mask].argmax(axis=1) == y[mask]).mean()), 4)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def train(
    cfg: TrainConfig, label_cfg: LabelConfig | None = None, db_path: Path | None = None
) -> dict:
    label_cfg = label_cfg or LabelConfig()
    db = db_path or DB_PATH
    if not db.exists():
        raise FileNotFoundError(f"database not found at {db}")

    logger.info("loading candles for %s (%s entry)", cfg.symbol, cfg.entry_timeframe)
    candles = load_candles(
        db, cfg.symbol, (cfg.entry_timeframe, *cfg.confirmation_timeframes)
    )
    features, labels, cost_r = build_dataset(candles, cfg, label_cfg)
    logger.info(
        "dataset: %d usable bars, %s -> %s", len(features), features.index[0], features.index[-1]
    )
    logger.info(
        "base rates: p_secure(long)=%.4f p_target(long)=%.4f  break-even p_secure=%.4f",
        float(labels["secure_long"].mean()),
        float(labels["target_long"].mean()),
        breakeven_secure_probability(label_cfg.secure_r),
    )

    logger.info("walk-forward validation (%d folds)", cfg.folds)
    folds = walk_forward(features, labels, cost_r, cfg, label_cfg)

    # Final model: everything, minus a purge and a calibration tail.
    logger.info("fitting final model on the full history")
    n = len(features)
    calib_start = int(n * 0.9)
    mean = features.iloc[:calib_start].to_numpy(dtype=np.float64).mean(axis=0)
    scale = features.iloc[:calib_start].to_numpy(dtype=np.float64).std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_tensors = _tensors(features.iloc[:calib_start], labels.iloc[:calib_start], mean, scale)
    calib_tensors = _tensors(features.iloc[calib_start:], labels.iloc[calib_start:], mean, scale)
    model = fit(train_tensors, cfg, input_dim=len(FEATURE_NAMES))
    temperature = fit_temperature(model, calib_tensors)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"{cfg.model_name}.pt"
    scaler_path = MODELS_DIR / f"{cfg.model_name}_scaler.npz"
    meta_path = MODELS_DIR / f"{cfg.model_name}_meta.json"
    torch.save(model.state_dict(), weights_path)
    np.savez(scaler_path, mean=mean, scale=scale)

    summary = _summarise(folds)
    meta = {
        "model_name": cfg.model_name,
        "symbol": cfg.symbol,
        "entry_timeframe": cfg.entry_timeframe,
        "confirmation_timeframes": list(cfg.confirmation_timeframes),
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": cfg.hidden_dim,
        "temperature": round(temperature, 4),
        "label_config": asdict(label_cfg),
        "train_config": asdict(cfg),
        "n_bars": int(n),
        "data_start": str(features.index[0]),
        "data_end": str(features.index[-1]),
        "walk_forward": folds,
        "summary": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("saved %s, %s, %s", weights_path.name, scaler_path.name, meta_path.name)
    _print_report(meta)
    return meta


def _summarise(folds: list[dict]) -> dict:
    if not folds:
        return {"folds": 0}
    oos = [f["economics_oos"] for f in folds]
    is_ = [f["economics_is"] for f in folds]
    avg_r = [f["avg_r"] for f in oos]
    return {
        "folds": len(folds),
        "oos_avg_r_mean": round(float(np.mean(avg_r)), 4),
        "oos_avg_r_min": round(float(np.min(avg_r)), 4),
        "oos_avg_r_max": round(float(np.max(avg_r)), 4),
        "oos_sum_r_total": round(float(np.sum([f["sum_r"] for f in oos])), 2),
        "oos_trades_total": int(np.sum([f["trades"] for f in oos])),
        "oos_pf_mean": round(
            float(np.mean([f["profit_factor"] for f in oos if np.isfinite(f["profit_factor"])])), 3
        )
        if any(np.isfinite(f["profit_factor"]) for f in oos)
        else 0.0,
        "is_avg_r_mean": round(float(np.mean([f["avg_r"] for f in is_])), 4),
        "brier_secure_is_mean": round(float(np.mean([f["brier_secure_is"] for f in folds])), 5),
        "brier_secure_oos_mean": round(float(np.mean([f["brier_secure_oos"] for f in folds])), 5),
        "folds_positive_oos": int(sum(1 for f in oos if f["avg_r"] > 0)),
        "threshold_tunable": _threshold_verdict(folds),
    }


def _threshold_verdict(folds: list[dict]) -> str:
    """Does the in-sample best EV threshold predict the out-of-sample one?

    Reported rather than silently resolved. If the per-fold IS and OOS optima
    anti-correlate, the honest conclusion is "this parameter is not tunable
    on this data", not "here is the flattering number".
    """
    is_best, oos_best = [], []
    for fold in folds:
        is_best.append(EV_SWEEP[int(np.argmax([s["avg_r"] for s in fold["sweep_is"]]))])
        oos_best.append(EV_SWEEP[int(np.argmax([s["avg_r"] for s in fold["sweep_oos"]]))])
    if len(is_best) < 3:
        return "too few folds to judge"
    if len(set(is_best)) == 1 or len(set(oos_best)) == 1:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(is_best, oos_best)[0, 1])
    verdict = "TUNABLE" if correlation > 0.5 else "NOT TUNABLE — do not fit this on IS"
    return f"{verdict} (IS optima {is_best}, OOS optima {oos_best}, r={correlation:.2f})"


def _print_report(meta: dict) -> None:
    summary = meta["summary"]
    print("\n" + "=" * 72)
    print(f"  {meta['model_name']}  —  {meta['symbol']} ({meta['entry_timeframe']} entry)")
    print(f"  {meta['n_bars']} bars   {meta['data_start']} -> {meta['data_end']}")
    print("=" * 72)
    print("\n  WALK-FORWARD (chronological, purged, temperature fitted in-window)")
    print(
        f"  {'fold':<5}{'oos start':<28}{'trades':>8}{'sum R':>9}{'avg R':>9}"
        f"{'PF':>7}{'maxDD':>8}{'Brier':>8}"
    )
    for fold in meta["walk_forward"]:
        economics = fold["economics_oos"]
        print(
            f"  {fold['fold']:<5}{fold['oos_start'][:19]:<28}{economics['trades']:>8}"
            f"{economics['sum_r']:>9.2f}{economics['avg_r']:>9.4f}"
            f"{economics['profit_factor']:>7.2f}{economics['max_drawdown_r']:>8.1f}"
            f"{fold['brier_secure_oos']:>8.4f}"
        )
    print("\n  IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"    avg R   IS {summary.get('is_avg_r_mean')}   OOS {summary.get('oos_avg_r_mean')}")
    print(
        f"    Brier   IS {summary.get('brier_secure_is_mean')}   "
        f"OOS {summary.get('brier_secure_oos_mean')}"
    )
    print(
        f"    folds with positive OOS avg R: "
        f"{summary.get('folds_positive_oos')}/{summary.get('folds')}"
    )
    if meta["walk_forward"]:
        print("\n  EV-THRESHOLD SWEEP — avg R, in-sample vs out-of-sample, per fold")
        print(
            f"    {'thr':>6} "
            + " ".join(f"{'f' + str(f['fold']) + ' IS':>9}{'OOS':>9}" for f in meta["walk_forward"])
        )
        for index, threshold in enumerate(EV_SWEEP):
            cells = []
            for fold in meta["walk_forward"]:
                cells.append(
                    f"{fold['sweep_is'][index]['avg_r']:>9.4f}{fold['sweep_oos'][index]['avg_r']:>9.4f}"
                )
            print(f"    {threshold:>6.2f} " + " ".join(cells))
        verdict = summary.get("threshold_tunable")
        print(f"    -> IS/OOS optimum agreement: {verdict}")
        print("\n  TOP FEATURES (permutation, last fold, secure head)")
        for row in meta["walk_forward"][-1]["importance_oos"][:10]:
            print(f"    {row['feature']:<24}{row['brier_increase']:+.6f}")
        print("\n  CALIBRATION — secure head, long, last fold OOS")
        for row in meta["walk_forward"][-1]["reliability_secure_oos"]:
            print(
                f"    {row['bin']:<12}n={row['n']:<8}predicted {row['predicted']:.3f}"
                f"   actual {row['actual']:.3f}"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframe",
        required=True,
        choices=sorted(TIMEFRAME_CONFIG),
        help="Entry timeframe to train on; selects the confirmation-rung ladder and rescales "
        "the label horizon (see module docstring).",
    )
    parser.add_argument("--symbol", default=os.environ.get("SYMBOL", "XAUUSD"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-r", type=float, default=None)
    parser.add_argument("--ev-threshold", type=float, default=0.02)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    tf_config = TIMEFRAME_CONFIG[args.timeframe]
    confirmation = tf_config["confirmation"]
    bar_per_minute = tf_config["bar_per_minute"]

    model_name = args.model_name or f"smc_dl_{args.timeframe.lower()}_v3"
    cfg = TrainConfig(
        symbol=args.symbol,
        model_name=model_name,
        entry_timeframe=args.timeframe,
        confirmation_timeframes=confirmation,
        folds=args.folds,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        ev_threshold=args.ev_threshold,
    )
    label_kwargs = {
        "max_holding_bars": round(_MAX_HOLDING_MINUTES / bar_per_minute),
        "revert_bars": round(_REVERT_MINUTES / bar_per_minute),
    }
    if args.target_r:
        label_kwargs["target_r"] = args.target_r
    label_cfg = LabelConfig(**label_kwargs)
    train(cfg, label_cfg, Path(args.db) if args.db else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
