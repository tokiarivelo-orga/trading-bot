"""Train the regime-router priors used to seed `AdaptiveLearner.score(...,
prior_logit=...)` for the upcoming regime-switching XAUUSD strategy (Phase 1
of a 6-phase plan; this script and its JSON artifact are the whole of Phase 1
— no generated strategy file is produced here).

Run from `backend/`::

    uv run python scripts/train_regime_router_priors.py
    uv run python scripts/train_regime_router_priors.py --db /path/to/trading.db

WHERE THE LABEL COMES FROM, AND WHY IT IS AN APPROXIMATION
────────────────────────────────────────────────────────────────────────
`domain.online_learning.AdaptiveLearner`'s label (see that module's "WHAT
WIN MEANS HERE" docstring section) is *not* "did TP beat SL" — on this
engine `PositionManager`'s secure-base trailing takes ~94% of trades out at
exactly +0.2R within a couple of minutes, so the strategy's own TP almost
never decides the outcome. The real label is:

    y = 1  price travels `secure_r` x risk in the trade's favour **before**
           touching the stop distance               -> "secure hit"
    y = 0  the stop distance is touched first, or neither happens

`AdaptiveLearner` computes this bar-by-bar from the live candle stream
(`_advance_one`'s first-hit-index comparison). `trades` in `trading.db` does
not store the bar-by-bar path — only the trade's *final* MFE/MAE magnitudes
(non-negative, price-unit, running-maximum excursions from the entry price;
see `journal/domain/excursion.py`) plus `mfe_time`/`mae_time`, the instants
those running maxima were last extended. This script approximates the same
label from those four numbers:

    sl_dist     = abs(open_price - sl)
    secure_dist = secure_r * sl_dist            (secure_r = 0.2)
    hit_secure  = mfe >= secure_dist
    hit_stop    = mae >= sl_dist

    both False        -> y = 0   (never secured; pessimistic, see below)
    only hit_secure    -> y = 1
    only hit_stop       -> y = 0
    both True           -> whichever of mfe_time / mae_time is EARLIER
                            decided it first; equal or either missing
                            -> y = 0 (pessimistic tie-break)

This is a coarse proxy, not a claim of bar-level precision: a trade whose
MFE and MAE both cleared their thresholds only tells us *that* both
barriers were touched at some point in the trade's life, not the intrabar
sequence within the bar each was set on, and `mfe_time`/`mae_time` are
themselves only as fine-grained as the candle timeframe that produced them.
It intentionally mirrors `AdaptiveLearner`'s own documented convention of
resolving genuine ambiguity (same-bar ties) pessimistically as a stop,
because the alternative flatters exactly the marginal setups a gate exists
to filter out. The output of this script (`prior_logit` per bucket) is
explicitly a *starting point* for a cold `AdaptiveLearner` instance, not a
fixed authority — `AdaptiveLearner.score()` blends it with the bucket's own
accumulating live evidence and progressively outweighs a stale prior as real
observations arrive (see that module's "EVIDENCE BEFORE AUTHORITY" section).
Any error introduced by this label approximation is exactly the kind of
thing live evidence is designed to correct.

Rows with a NULL/zero `sl`, NULL `open_price`, or NULL `profit` (not a
closed trade, or `sl_dist` undefined) are dropped — they cannot be labelled
at all.

UNTAGGED ROWS ARE DROPPED, NOT KEPT AS AN "UNKNOWN" CATEGORY
────────────────────────────────────────────────────────────────────────
`regime_session`/`regime_trend`/`regime_volatility_percentile` are NULL on
~3,960 rows in this dataset — every one of them predates this engine's
regime-tagging rollout (tagging is atomic: verified that on this dataset
every row with a NULL `regime_session` also has NULL `regime_trend` and
NULL `regime_volatility_percentile`, and every row with any one of the
three present has all three present). Those rows are a historical
instrumentation artifact, **not** a genuine "untagged" case the live
strategy will ever encounter — every trade going forward gets a real
session/trend/volatility tag from `engine.domain.regime`. An earlier
version of this script kept them, folded into an explicit "unknown"
one-hot category per feature. That was a real mistake, caught by the
required chronological split: those ~3,960 rows are also the
chronologically OLDEST rows in the table, so a 70/30 time-ordered split put
every one of them in TRAIN and *none* in HOLDOUT. The `*_unknown` one-hot
columns then looked strongly predictive on train (they correlate with "old
trade, pre-secure-base-trailing-tuning" far more than with any real regime
effect) while being constant-zero and uninformative on holdout — and
collapsed holdout AUC to 0.492 (chance) in that version. `filter_tagged_
regime_rows()` now drops every row missing any of the three regime fields
*before* labelling and splitting. The `unknown` one-hot slot is kept in
`SESSION_CATEGORIES`/`TREND_CATEGORIES`/`VOLATILITY_CATEGORIES` (and thus
`FEATURE_ORDER`) purely for schema forward-compatibility — a downstream
consumer's feature vector never changes shape even in a hypothetical future
edge case that produces an untagged live trade — but on this training run
it is guaranteed never to fire: every row that could set it has already
been dropped.

FEATURES AND NO-LOOKAHEAD SPLIT
────────────────────────────────────────────────────────────────────────
`regime_session` (5 real values), `regime_trend` (2 real values) and
`regime_volatility_percentile` (continuous) are one-hot/derived per trade;
see above for why every row reaching this stage always has all three.
Volatility is bucketed into terciles (low/mid/high); per this project's
documented lesson (an hour-of-day gate that measured great in-sample and
collapsed chronologically-validated, see
`entry-filter-oos-validation-hour-overfits`), the tercile cut points are
fit on the TRAIN split's distribution only, then applied unchanged to the
holdout split — exactly the same discipline already applied to the feature
standardizer's mean/std. Both sets of fitted constants
(`volatility_tercile_bounds`, `standardizer_mean`/`standardizer_std`) are
persisted in the output artifact so a downstream consumer buckets a live
trade's volatility percentile identically to how training did.

The chronological 70/30 train/holdout split is by `open_time`, not shuffled
— required by that same lesson. No purge window is needed at the split
boundary (unlike a rolling-bar-window trainer unlike this one): each row is
an already-closed, independent trade with its own final MFE/MAE, so one
trade's label cannot leak across the boundary into another's features.

THE MODEL
────────────────────────────────────────────────────────────────────────
A 2-layer MLP (input -> 8 hidden, tanh -> 1 output, sigmoid), hand-rolled in
plain numpy — forward pass, backward pass, L2-regularised full-batch
gradient descent — mirroring `online_learning.OnlineLogisticModel`'s
hand-rolled style rather than pulling in a training framework this offline
script does not need. (This script itself runs outside the strategy
sandbox, so scikit-learn/torch are available if wanted — see
`backend/pyproject.toml` — but the task this seeds explicitly wants a small,
inspectable, dependency-light net, so none is used here.) `AUC` is a plain
rank-based (Mann-Whitney) implementation via `pandas.Series.rank`, no
scikit-learn needed either.

THE EDGE GATE — WHY A CHANCE-LEVEL RESULT IS NOT SHIPPED AS A LEARNED NET
────────────────────────────────────────────────────────────────────────
After dropping untagged rows (see above), `XAUUSD` history in `trading.db`
still only spans ~27 days end to end, split into an 11-day TRAIN window and
a 16-day HOLDOUT window. That is a short horizon to ask "does a
(trend, session) bucket's secure-hit rate hold up over time" — and
empirically here it does not: the *raw per-bucket empirical rate itself*
(no model at all) is **negatively** correlated between the two windows
(measured Pearson r ~ -0.24 across the 8 `(trend, session)` buckets), and
the aggregate secure rate drops from ~0.82 (train window) to ~0.74 (holdout
window) across every bucket. That is consistent with a population-level
shift in the live bot fleet over those weeks (new strategy versions/tuning
going live) rather than a stable regime effect — and no amount of hidden
layer width or L2 strength moves holdout AUC out of the ~0.49-0.53 band (
swept `hidden_dim` in 2/4/8 and `l2` in 1e-3/5e-2/2e-1 while building this
script; all landed there). So this is treated as a genuine "not enough
stable signal yet" finding, not a bug to keep tuning away.

Exactly like `train_xauusd_dl_m5.py`'s `MIN_AUC_EDGE`/
`MIN_BRIER_RELATIVE_IMPROVEMENT` gate, this script does not ship a model
that failed to demonstrate an out-of-sample edge as if it had one: when
holdout AUC and Brier improvement (over a constant-train-base-rate
predictor) both fail to clear `MIN_AUC_EDGE`/`MIN_BRIER_RELATIVE_
IMPROVEMENT`, `has_edge=False` is recorded in the artifact and the
*shipped* `layer1_weights`/`layer1_bias`/`layer2_weights`/`layer2_bias` are
zeroed to a literal no-op (`sigmoid(0) = 0.5` regardless of input, so
`logit(0.5) = 0` contributes nothing to a downstream `prior_logit`) —
never the measured-not-generalizing trained weights. `train_metrics`/
`holdout_metrics` in the artifact still report the *actual trained net's*
measured performance (so the "why" stays visible), separate from what
ships. `bucket_table` (full-dataset, heavily shrunk toward the global rate
by any sane downstream consumer per `AdaptiveLearner`'s own
`bucket_prior_strength` convention) remains the honest, low-capacity
source of whatever real signal exists — a weak-but-not-random prior from
it is fine to seed `AdaptiveLearner` with; a chance-level net dressed up as
a confident correction is not.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("train_regime_router_priors")

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "trading.db"
MODELS_DIR = BACKEND_DIR / "data" / "ml_models"
DEFAULT_OUTPUT_NAME = "regime_router_priors_v1.json"

# Mirrors `domain.online_learning.DEFAULT_SECURE_R` / `LearnerConfig.secure_r`
# — duplicated rather than imported so this offline script has no runtime
# coupling to the sandbox-importable domain module, matching
# `train_xauusd_dl_m5.py`'s convention for the same constant.
DEFAULT_SECURE_R = 0.2

# A trained net is only shipped as-is if it clears BOTH gates on the
# untouched holdout split — otherwise `has_edge=False` is recorded and the
# shipped weights are zeroed to a no-op (see module docstring's "THE EDGE
# GATE" section). Values mirror `train_xauusd_dl_m5.py`'s
# `MIN_AUC_EDGE`/`MIN_BRIER_RELATIVE_IMPROVEMENT` for the same reason that
# script gives: 0.55 AUC is a conventional soft floor for a weak financial
# signal, and Brier improvement must be a non-trivial fraction of the
# constant-predictor baseline, not rounding noise.
MIN_AUC_EDGE = 0.55
MIN_BRIER_RELATIVE_IMPROVEMENT = 0.02

UNKNOWN = "unknown"
SESSION_CATEGORIES = ("asian", "london", "overlap", "new_york", "off_session", UNKNOWN)
TREND_CATEGORIES = ("trending", "ranging", UNKNOWN)
VOLATILITY_CATEGORIES = ("low", "mid", "high", UNKNOWN)
DOW_CATEGORIES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

FEATURE_ORDER: tuple[str, ...] = (
    *(f"session_{s}" for s in SESSION_CATEGORIES),
    *(f"trend_{t}" for t in TREND_CATEGORIES),
    *(f"vol_{v}" for v in VOLATILITY_CATEGORIES),
    "hour_sin",
    "hour_cos",
    *(f"dow_{d}" for d in DOW_CATEGORIES),
)


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    symbol: str = "XAUUSD"
    secure_r: float = DEFAULT_SECURE_R
    train_frac: float = 0.7
    hidden_dim: int = 8
    epochs: int = 500
    learning_rate: float = 0.05
    l2: float = 1e-3
    seed: int = 7


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_trades(db_path: Path, symbol: str) -> pd.DataFrame:
    """Closed trades for `symbol`, read-only, chronologically ordered. Never
    writes to `db_path` — this script's only relationship with the live
    trading DB is a `SELECT`."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            """
            SELECT open_time, side, open_price, sl, profit, mfe, mae,
                   mfe_time, mae_time, regime_session, regime_trend,
                   regime_volatility_percentile
            FROM trades
            WHERE symbol = ? AND profit IS NOT NULL
            ORDER BY open_time
            """,
            conn,
            params=[symbol],
        )
        return df
    finally:
        conn.close()


def filter_tagged_regime_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a REAL regime tag on all three regime fields.

    See the module docstring's "UNTAGGED ROWS ARE DROPPED" section for why:
    the ~3,960 rows with NULL `regime_session`/`regime_trend`/
    `regime_volatility_percentile` predate this engine's regime-tagging
    rollout and are chronologically the oldest rows in the table, so
    keeping them let a chronological split turn "untagged" into a spurious
    time-boundary feature. Called before labelling/splitting, on the raw
    loaded frame."""
    tagged = (
        df["regime_session"].notna()
        & df["regime_trend"].notna()
        & df["regime_volatility_percentile"].notna()
    )
    dropped = int((~tagged).sum())
    if dropped:
        logger.info(
            "dropping %d rows with no real regime tag (pre-regime-tagging historical "
            "artifact — not a genuine live 'untagged' case, see module docstring)",
            dropped,
        )
    return df.loc[tagged].reset_index(drop=True)


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def compute_labels(df: pd.DataFrame, secure_r: float) -> pd.DataFrame:
    """Drops unlabelable rows, then attaches `sl_dist` and the `y` secure-hit
    label per this script's module-docstring recipe."""
    valid = df["sl"].notna() & (df["sl"] != 0) & df["open_price"].notna() & df["profit"].notna()
    dropped = int((~valid).sum())
    if dropped:
        logger.info("dropping %d rows with missing/zero sl, open_price, or profit", dropped)
    out = df.loc[valid].copy()

    sl_dist = (out["open_price"] - out["sl"]).abs()
    secure_dist = secure_r * sl_dist
    # `fillna(0.0)` is a defensive fallback, not an expected code path: by
    # the time this runs, `filter_tagged_regime_rows` has already dropped
    # every row that (on this dataset) ever had a NULL mfe/mae, so this
    # never actually fires — kept in case a future/different dataset has a
    # tagged row with a genuinely un-set excursion, matching `Excursion`'s
    # own 0.0 default for "no candle has extended this trade's excursion
    # yet" rather than crashing on it.
    mfe = out["mfe"].fillna(0.0)
    mae = out["mae"].fillna(0.0)

    hit_secure = mfe >= secure_dist
    hit_stop = mae >= sl_dist

    mfe_time = out["mfe_time"]
    mae_time = out["mae_time"]
    times_known = mfe_time.notna() & mae_time.notna()
    secure_first = hit_secure & hit_stop & times_known & (mfe_time < mae_time)

    y = np.where(hit_secure & ~hit_stop, 1, np.where(secure_first, 1, 0))

    out["sl_dist"] = sl_dist
    out["y"] = y.astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _one_hot(series: pd.Series, categories: tuple[str, ...], prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(series.fillna(UNKNOWN), categories=list(categories))
    dummies = pd.get_dummies(cat, prefix=prefix)
    # `pd.Categorical` with fixed `categories` guarantees every category gets
    # a column even if it never occurs in this slice (e.g. `off_session`
    # never appears in the current XAUUSD history but is a real value the
    # live regime tagger can emit) — a downstream consumer's feature vector
    # then always has the same shape as `feature_order`.
    return dummies.astype(np.float64)


def _volatility_bucket(percentile: pd.Series, low_cut: float, high_cut: float) -> pd.Series:
    bucket = pd.Series(UNKNOWN, index=percentile.index, dtype=object)
    known = percentile.notna()
    bucket.loc[known & (percentile <= low_cut)] = "low"
    bucket.loc[known & (percentile > low_cut) & (percentile <= high_cut)] = "mid"
    bucket.loc[known & (percentile > high_cut)] = "high"
    return bucket


def build_features(
    df: pd.DataFrame, *, low_cut: float, high_cut: float
) -> pd.DataFrame:
    """Builds every column in `FEATURE_ORDER`, in order. `low_cut`/`high_cut`
    are volatility-tercile boundaries — callers must fit them on the TRAIN
    split only and pass the same two numbers when building holdout
    features (see `main()`)."""
    # `.loc[:, col]` rather than `df[col]`: a plain bracket-indexed getitem
    # on a `DataFrame` type-checks as `Series | DataFrame` under pyright
    # (the overload keyed on a single string label vs. a list of labels is
    # not narrowed), which then fails every downstream call expecting a
    # `Series`. `.loc[:, single_label]` is the unambiguous one-column form.
    session = _one_hot(df.loc[:, "regime_session"], SESSION_CATEGORIES, "session")
    trend = _one_hot(df.loc[:, "regime_trend"], TREND_CATEGORIES, "trend")
    vol_bucket = _volatility_bucket(df.loc[:, "regime_volatility_percentile"], low_cut, high_cut)
    vol = _one_hot(vol_bucket, VOLATILITY_CATEGORIES, "vol")

    open_dt = pd.to_datetime(df["open_time"], unit="s", utc=True)
    hour = open_dt.dt.hour.to_numpy(dtype=np.float64)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_labels = pd.Categorical(
        open_dt.dt.weekday.map(dict(enumerate(DOW_CATEGORIES))),
        categories=list(DOW_CATEGORIES),
    )
    dow = pd.get_dummies(dow_labels, prefix="dow").astype(np.float64)

    features = pd.concat(
        [
            session,
            trend,
            vol,
            pd.DataFrame({"hour_sin": hour_sin, "hour_cos": hour_cos}, index=df.index),
            dow,
        ],
        axis=1,
    )
    features = features.reindex(columns=list(FEATURE_ORDER), fill_value=0.0)
    return features


# ---------------------------------------------------------------------------
# model — plain numpy 2-layer MLP, mirrors OnlineLogisticModel's style
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class SmallMLP:
    """input -> hidden (tanh) -> 1 (sigmoid), trained by full-batch gradient
    descent with L2 regularisation. Deliberately tiny — this is a prior for a
    cold `AdaptiveLearner`, not the model that decides live trades."""

    def __init__(self, n_features: int, hidden_dim: int, *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # He-ish scaling: small enough that tanh starts near-linear.
        self.w1 = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=(n_features, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        p = _sigmoid(z2)
        return a1, p

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, p = self.forward(x)
        return p.reshape(-1)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
    ) -> None:
        n = x.shape[0]
        y_col = y.reshape(-1, 1).astype(np.float64)
        for _ in range(epochs):
            a1, p = self.forward(x)
            dz2 = (p - y_col) / n
            dw2 = a1.T @ dz2 + l2 * self.w2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ self.w2.T
            dz1 = da1 * (1.0 - a1**2)
            dw1 = x.T @ dz1 + l2 * self.w1
            db1 = dz1.sum(axis=0)

            self.w1 -= learning_rate * dw1
            self.b1 -= learning_rate * db1
            self.w2 -= learning_rate * dw2
            self.b2 -= learning_rate * db2


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold).astype(np.int64) == y))


def rank_auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Plain rank-based (Mann-Whitney U) AUC, ties handled via average rank —
    no scikit-learn dependency. `None` when one class is entirely absent
    (AUC undefined)."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    sum_ranks_pos = float(ranks[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(p: np.ndarray, y: np.ndarray) -> dict:
    auc = rank_auc(p, y)
    return {
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "accuracy": round(accuracy(p, y), 4),
        "brier": round(brier_score(p, y), 6),
        "auc": None if auc is None else round(auc, 4),
    }


# ---------------------------------------------------------------------------
# bucket table — the live strategy's actual (trend, session) bucket keys
# ---------------------------------------------------------------------------
def unknown_regime_diagnostics(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    """How many rows in each split have no `regime_session` tag at all.

    Verification check, not a live concern: `filter_tagged_regime_rows` now
    drops every untagged row before this script ever splits the data, so
    both counts are expected to be exactly 0 on every run — this function
    exists to make that visible in the report rather than assumed. A
    non-zero count here would mean the upstream filter regressed."""
    return {
        "train_unknown_session_rows": int(train_df.loc[:, "regime_session"].isna().sum()),
        "holdout_unknown_session_rows": int(holdout_df.loc[:, "regime_session"].isna().sum()),
    }


def bucket_table(df: pd.DataFrame) -> list[dict]:
    # `.fillna(UNKNOWN)` is defensive only — `filter_tagged_regime_rows` has
    # already dropped every row with a NULL regime_trend/regime_session by
    # the time this runs, so no NULLs are actually expected here.
    grouped = df.assign(
        regime_trend=df.loc[:, "regime_trend"].fillna(UNKNOWN),
        regime_session=df.loc[:, "regime_session"].fillna(UNKNOWN),
    ).groupby(["regime_trend", "regime_session"], observed=True)["y"]
    table = grouped.agg(count="count", secure_rate="mean").reset_index()
    table["secure_rate"] = table["secure_rate"].round(4)
    table = table.sort_values(["regime_trend", "regime_session"]).reset_index(drop=True)
    return table.to_dict(orient="records")


def bucket_rate_stability(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> dict:
    """Does a `(regime_trend, regime_session)` bucket's secure-hit rate rank
    the same way in TRAIN as in HOLDOUT? This is the plainest possible
    check of "is there real, temporally stable regime signal here at all" —
    computed on the raw empirical rate, with no model in between, so a
    negative correlation here means even a perfect per-bucket lookup table
    fit on TRAIN would rank buckets backwards on HOLDOUT. See module
    docstring's "THE EDGE GATE" section for the measured result."""
    train_table = {
        (r["regime_trend"], r["regime_session"]): r["secure_rate"] for r in bucket_table(train_df)
    }
    holdout_table = {
        (r["regime_trend"], r["regime_session"]): r["secure_rate"]
        for r in bucket_table(holdout_df)
    }
    shared_keys = sorted(set(train_table) & set(holdout_table))
    if len(shared_keys) < 3:
        return {"n_shared_buckets": len(shared_keys), "pearson_r": None}
    train_rates = np.array([train_table[k] for k in shared_keys])
    holdout_rates = np.array([holdout_table[k] for k in shared_keys])
    r = float(np.corrcoef(train_rates, holdout_rates)[0, 1])
    return {
        "n_shared_buckets": len(shared_keys),
        "pearson_r": None if not np.isfinite(r) else round(r, 4),
        "train_period_rates": {"|".join(k): v for k, v in train_table.items()},
        "holdout_period_rates": {"|".join(k): v for k, v in holdout_table.items()},
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _print_report(meta: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  regime_router_priors  —  {meta['symbol']}  (n={meta['n_train'] + meta['n_holdout']})")
    print("=" * 78)
    for split_name in ("train_metrics", "holdout_metrics"):
        m = meta[split_name]
        label = "TRAIN" if split_name == "train_metrics" else "HOLDOUT (out-of-sample)"
        print(f"\n  --- {label} --- n={m['n']}  base_rate={m['base_rate']}")
        print(f"    accuracy={m['accuracy']}   brier={m['brier']}   auc={m['auc']}")

    diag = meta["diagnostics"]
    print(
        f"\n  unknown-regime (pre-tagging) rows: train={diag['train_unknown_session_rows']}  "
        f"holdout={diag['holdout_unknown_session_rows']}"
    )

    train_m, hold_m = meta["train_metrics"], meta["holdout_metrics"]
    if hold_m["auc"] is not None and train_m["auc"] is not None:
        auc_drop = train_m["auc"] - hold_m["auc"]
        if auc_drop > 0.05:
            cause = ""
            if diag["train_unknown_session_rows"] > 0 and diag["holdout_unknown_session_rows"] == 0:
                cause = (
                    "\n  Likely cause: every pre-regime-tagging ('unknown') trade falls in "
                    "TRAIN and none in HOLDOUT (see counts above) — the unknown_* one-hot "
                    "features look predictive on train (correlating with 'old trade' rather "
                    "than a real regime effect) but are constant-zero and uninformative on "
                    "holdout. This is a real generalization failure, not a training bug."
                )
            print(
                f"\n  ** WARNING: holdout AUC ({hold_m['auc']}) is {auc_drop:.3f} below "
                f"train AUC ({train_m['auc']}) — meaningful overfitting, treat this prior "
                f"with extra caution / shrink it further downstream. **{cause}"
            )
        else:
            print(
                f"\n  holdout AUC ({hold_m['auc']}) tracks train AUC ({train_m['auc']}) "
                f"within {auc_drop:.3f} — no strong overfitting signal."
            )
    brier_gap = hold_m["brier"] - train_m["brier"]
    if brier_gap > 0.01:
        print(
            f"  ** WARNING: holdout Brier ({hold_m['brier']}) is {brier_gap:.4f} worse than "
            f"train Brier ({train_m['brier']}). **"
        )

    stability = meta["bucket_rate_stability"]
    print(
        "\n  --- bucket-rate stability: does TRAIN's per-bucket secure rate rank the same "
        "way in HOLDOUT? ---"
    )
    if stability["pearson_r"] is None:
        print(f"    too few shared buckets ({stability['n_shared_buckets']}) to correlate")
    else:
        verdict = "STABLE" if stability["pearson_r"] > 0.3 else "NOT STABLE"
        print(
            f"    pearson r across {stability['n_shared_buckets']} shared "
            f"(trend, session) buckets = {stability['pearson_r']}  ->  {verdict}"
        )
        if stability["pearson_r"] <= 0:
            print(
                "    ** a non-positive correlation means even a perfect train-period lookup "
                "table would rank buckets BACKWARDS on holdout — this is a real signal-"
                "stability problem at this sample size, not a model-capacity problem. **"
            )

    gate = meta["edge_gate"]
    verdict = "HAS OUT-OF-SAMPLE EDGE" if meta["has_edge"] else "NO OUT-OF-SAMPLE EDGE"
    improvement_pct = gate["brier_relative_improvement"] * 100
    print(
        f"\n  GATE: holdout AUC >= {gate['min_auc_edge']} AND Brier relative improvement "
        f">= {gate['min_brier_relative_improvement'] * 100:.0f}%  "
        f"(measured: AUC={hold_m['auc']}, improvement={improvement_pct:.2f}%)  ->  {verdict}"
    )
    if not meta["has_edge"]:
        print(
            "  Shipped layer1/layer2 weights are ZEROED (neutral no-op, always outputs the "
            "base rate) — see module docstring's 'THE EDGE GATE' section. train_metrics/"
            "holdout_metrics above still report the actual trained (non-neutralized) net's "
            "measured performance, for transparency about why the gate failed."
        )

    print("\n  --- empirical secure-hit-rate by (regime_trend, regime_session), full dataset ---")
    print(f"  {'trend':<10} {'session':<12} {'count':>7}   {'secure_rate':>11}")
    for row in meta["bucket_table"]:
        print(
            f"  {row['regime_trend']:<10} {row['regime_session']:<12} {row['count']:>7}   "
            f"{row['secure_rate']:>11.4f}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=os.environ.get("SYMBOL", "XAUUSD"))
    parser.add_argument("--db", default=None, help="override path to trading.db")
    parser.add_argument("--output", default=None, help="override output JSON path")
    parser.add_argument("--secure-r", type=float, default=DEFAULT_SECURE_R)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = TrainConfig(
        symbol=args.symbol,
        secure_r=args.secure_r,
        train_frac=args.train_frac,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    db_path = Path(args.db) if args.db else DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"database not found at {db_path}")

    logger.info("loading closed %s trades from %s (read-only)", cfg.symbol, db_path)
    raw = load_trades(db_path, cfg.symbol)
    logger.info("%d closed trades with profit recorded", len(raw))
    if raw.empty:
        raise RuntimeError(f"no closed {cfg.symbol} trades with profit recorded")

    # Drop pre-regime-tagging rows BEFORE labelling/splitting — see module
    # docstring's "UNTAGGED ROWS ARE DROPPED" section for why keeping them
    # (as a training-time "unknown" category) collapsed holdout AUC to
    # chance in an earlier version of this script.
    tagged = filter_tagged_regime_rows(raw)
    logger.info("%d rows retained with a real regime tag (session+trend+volatility)", len(tagged))
    if tagged.empty:
        raise RuntimeError(f"no {cfg.symbol} trades with a real regime tag")

    labelled = compute_labels(tagged, cfg.secure_r)
    n = len(labelled)
    logger.info("%d labelled trades after dropping unlabelable rows", n)
    if n < 100:
        raise RuntimeError(f"only {n} labelled trades — too little data to train a prior")

    # Chronological split — first train_frac by open_time (already sorted by
    # the SQL query's ORDER BY open_time), NOT a random shuffle.
    split_at = int(n * cfg.train_frac)
    split_at = max(1, min(split_at, n - 1))
    train_df = labelled.iloc[:split_at].reset_index(drop=True)
    holdout_df = labelled.iloc[split_at:].reset_index(drop=True)
    logger.info(
        "chronological split: train=%d (up to %s) holdout=%d (from %s)",
        len(train_df),
        pd.to_datetime(train_df["open_time"].iloc[-1], unit="s", utc=True),
        len(holdout_df),
        pd.to_datetime(holdout_df["open_time"].iloc[0], unit="s", utc=True),
    )

    # Volatility tercile cut points fit on TRAIN split's non-null
    # distribution only — see module docstring's no-lookahead discussion.
    train_vol = train_df["regime_volatility_percentile"].dropna()
    if len(train_vol) >= 10:
        low_cut = float(train_vol.quantile(1.0 / 3.0))
        high_cut = float(train_vol.quantile(2.0 / 3.0))
    else:
        # Too little train-split volatility data to fit real terciles —
        # every trade falls into "unknown" for this feature rather than
        # guessing cut points from too few samples.
        low_cut, high_cut = float("-inf"), float("-inf")
    logger.info("volatility tercile bounds (train-fit): low<=%.3f, mid<=%.3f", low_cut, high_cut)

    train_x = build_features(train_df, low_cut=low_cut, high_cut=high_cut).to_numpy(
        dtype=np.float64
    )
    holdout_x = build_features(holdout_df, low_cut=low_cut, high_cut=high_cut).to_numpy(
        dtype=np.float64
    )
    train_y = train_df["y"].to_numpy(dtype=np.int64)
    holdout_y = holdout_df["y"].to_numpy(dtype=np.int64)

    # Standardize — mean/std fit on TRAIN split only, applied to both.
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_x_std = (train_x - mean) / std
    holdout_x_std = (holdout_x - mean) / std

    logger.info(
        "training %d -> %d (tanh) -> 1 (sigmoid) MLP, %d epochs, lr=%s, l2=%s",
        train_x_std.shape[1],
        cfg.hidden_dim,
        cfg.epochs,
        cfg.learning_rate,
        cfg.l2,
    )
    model = SmallMLP(train_x_std.shape[1], cfg.hidden_dim, seed=cfg.seed)
    model.fit(
        train_x_std,
        train_y,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
    )

    train_p = model.predict_proba(train_x_std)
    holdout_p = model.predict_proba(holdout_x_std)
    train_metrics = evaluate(train_p, train_y)
    holdout_metrics = evaluate(holdout_p, holdout_y)

    if not np.all(np.isfinite(train_p)) or not np.all(np.isfinite(holdout_p)):
        raise RuntimeError("model produced non-finite predictions — aborting, not writing artifact")

    # Edge gate — see module docstring's "THE EDGE GATE" section. Brier
    # baseline is a constant predictor at the TRAIN base rate (the only base
    # rate known at decision time in a real deployment), matching
    # `train_xauusd_dl_m5.py`'s convention.
    train_base_rate = float(train_y.mean())
    baseline_holdout_brier = brier_score(np.full_like(holdout_p, train_base_rate), holdout_y)
    model_holdout_brier = holdout_metrics["brier"]
    brier_relative_improvement = (
        (baseline_holdout_brier - model_holdout_brier) / baseline_holdout_brier
        if baseline_holdout_brier > 0
        else 0.0
    )
    holdout_auc = holdout_metrics["auc"] if holdout_metrics["auc"] is not None else 0.5
    has_edge = (
        holdout_auc >= MIN_AUC_EDGE and brier_relative_improvement >= MIN_BRIER_RELATIVE_IMPROVEMENT
    )
    edge_gate = {
        "min_auc_edge": MIN_AUC_EDGE,
        "min_brier_relative_improvement": MIN_BRIER_RELATIVE_IMPROVEMENT,
        "brier_relative_improvement": round(brier_relative_improvement, 4),
        "baseline_holdout_brier": round(baseline_holdout_brier, 6),
    }
    if not has_edge:
        logger.warning(
            "NO OUT-OF-SAMPLE EDGE: holdout AUC=%.4f (need >=%.2f) / Brier relative "
            "improvement=%.2f%% (need >=%.0f%%) — shipping ZEROED (neutral) weights, "
            "not the trained-but-non-generalizing net. See printed report for the "
            "bucket-rate stability diagnostic.",
            holdout_auc,
            MIN_AUC_EDGE,
            brier_relative_improvement * 100,
            MIN_BRIER_RELATIVE_IMPROVEMENT * 100,
        )
        # Ship a literal no-op net (always outputs 0.5, i.e. contributes zero
        # correction in log-odds) rather than a net that measurably does not
        # generalize — never silently ship the trained-but-overfit weights.
        shipped_w1 = np.zeros_like(model.w1)
        shipped_b1 = np.zeros_like(model.b1)
        shipped_w2 = np.zeros_like(model.w2)
        shipped_b2 = np.zeros_like(model.b2)
    else:
        shipped_w1, shipped_b1, shipped_w2, shipped_b2 = model.w1, model.b1, model.w2, model.b2

    table = bucket_table(labelled)
    diagnostics = unknown_regime_diagnostics(train_df, holdout_df)
    stability = bucket_rate_stability(train_df, holdout_df)

    meta = {
        "symbol": cfg.symbol,
        "secure_r": cfg.secure_r,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_order": list(FEATURE_ORDER),
        "volatility_tercile_bounds": {"low_upper": low_cut, "mid_upper": high_cut},
        "standardizer_mean": mean.tolist(),
        "standardizer_std": std.tolist(),
        "hidden_activation": "tanh",
        "output_activation": "sigmoid",
        "layer1_weights": shipped_w1.tolist(),
        "layer1_bias": shipped_b1.tolist(),
        "layer2_weights": shipped_w2.tolist(),
        "layer2_bias": shipped_b2.tolist(),
        "has_edge": has_edge,
        "edge_gate": edge_gate,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "diagnostics": diagnostics,
        "bucket_rate_stability": stability,
        "bucket_table": table,
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "train_config": asdict(cfg),
        "label_definition_note": (
            "Approximated from final mfe/mae + mfe_time/mae_time, not the true "
            "bar-by-bar path (trading.db stores only final excursion magnitudes). "
            "See this script's module docstring for the exact recipe and its "
            "documented limitations. This is a seed prior for a cold "
            "AdaptiveLearner instance, not a fixed authority — live evidence "
            "corrects it as each bucket accumulates real observations."
        ),
    }

    _print_report(meta)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else MODELS_DIR / DEFAULT_OUTPUT_NAME
    output_path.write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
