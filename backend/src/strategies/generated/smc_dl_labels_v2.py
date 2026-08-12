"""Three-outcome + give-back labels for the SMC deep-learning M5 model.

WHY THE OLD LABEL WAS WRONG
────────────────────────────────────────────────────────────────────────
``smc_dl_labels.generate_labels`` emits ``hit_tp_before_sl`` — a pure
TP-or-SL bracket outcome. Measured over 3,105 closed live XAUUSD trades,
expected R under that bracket model is **negative at every target from 1.0R
to 3.0R, for all seven zone patterns** (best: RBR at -0.008R). So the old
head was trained to predict a quantity that, whatever its accuracy, cannot
select profitable trades on this engine.

What actually books money here is ``PositionManager`` rule 2: secure-base
trailing at ``DEFAULT_SECURE_BUFFER_R_MULT = 0.2`` closes the position at
roughly +0.2R long before any take-profit is touched. The take-profit is
almost never the deciding event.

THE THREE-OUTCOME MODEL
────────────────────────────────────────────────────────────────────────
A trade on this engine resolves into exactly one of three states, so it
needs two probabilities, not one:

    p_secure  P(price travels +secure_r x risk in favour before touching the stop)
    p_target  P(price travels +target_r x risk in favour before touching the stop)

with ``p_target <= p_secure`` by construction (you cannot reach the far
barrier without passing the near one). Expected R is then

    E[R] = p_target * target_r
         + (p_secure - p_target) * secure_r
         - (1 - p_secure) * 1.0

On a real logged decision (p_target 0.064, target 1.75R, p_secure 0.826)
this evaluates to **+0.090**, against a measured backtest avg R of
0.07-0.09 for the run that produced it. The bracket model scores the same
decision as a loser.

Break-even on the secure leg alone is ``1 / (1 + secure_r)`` = **0.833** at
the engine's 0.2 buffer, versus a **live secure-rate of 0.62**. Closing that
gap — not raising a TP hit-rate — is the job.

THE GIVE-BACK / EXIT LABEL
────────────────────────────────────────────────────────────────────────
89.5% of losing live XAUUSD trades (1,949 of 2,178 with excursion data) were
in profit at some point first, giving back **3,579R** in aggregate. That is
several times the entire entry-selection edge, and no entry filter can
recover it: the trades were *right*, then died.

``revert_class`` is the head that attacks it. From each bar, over a short
horizon, it races a symmetric ``+/- revert_atr_mult x ATR`` barrier:

    0  down barrier first   -> a long running here is about to give back
    1  up barrier first     -> a short running here is about to give back
    2  neither within the horizon
   -1  both inside one bar  -> intrabar order unknown, EXCLUDED from training

Direction-agnostic and position-state-free by design, so one head serves
every open position regardless of when or why it was opened, and it can be
learned from candles alone.

NO LOOKAHEAD IN THE FEATURES THESE LABELS ATTACH TO
────────────────────────────────────────────────────────────────────────
Labels look forward — that is what a label is. The contract is that the
*features* for bar ``i`` use only bars ``<= i`` while the label for bar
``i`` uses only bars ``> i``. Every barrier walk below starts at ``i + 1``.
``tests/unit/strategies/test_smc_dl_labels_v2.py`` pins that boundary.

AMBIGUITY IS RESOLVED PESSIMISTICALLY
────────────────────────────────────────────────────────────────────────
When a bar's high reaches a favourable barrier and its low reaches the stop,
the intrabar path is unknown. The stop is assumed to have come first. This
matches ``domain/online_learning``'s convention and is deliberate: the
optimistic reading flatters exactly the marginal setups a gate exists to
remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_SECURE_R = 0.2  # mirrors engine PositionManager DEFAULT_SECURE_BUFFER_R_MULT
DEFAULT_TARGET_R = 1.75
DEFAULT_SL_ATR_MULT = 1.0
DEFAULT_MAX_HOLDING_BARS = 60  # 5 hours of M5
DEFAULT_REVERT_ATR_MULT = 0.6
DEFAULT_REVERT_BARS = 12  # 1 hour of M5

LABEL_COLUMNS: tuple[str, ...] = (
    "secure_long",
    "secure_short",
    "target_long",
    "target_short",
    "mfe_long_r",
    "mfe_short_r",
    "revert_class",
    "bars_to_stop_long",
    "bars_to_stop_short",
)


@dataclass(frozen=True, kw_only=True)
class LabelConfig:
    secure_r: float = DEFAULT_SECURE_R
    target_r: float = DEFAULT_TARGET_R
    sl_atr_mult: float = DEFAULT_SL_ATR_MULT
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS
    revert_atr_mult: float = DEFAULT_REVERT_ATR_MULT
    revert_bars: int = DEFAULT_REVERT_BARS


def expected_r(
    p_secure: float,
    p_target: float,
    target_r: float = DEFAULT_TARGET_R,
    secure_r: float = DEFAULT_SECURE_R,
    cost_r: float = 0.0,
) -> float:
    """Expected R of one trade under the three-outcome model.

    ``p_target`` is clamped to ``p_secure``: the far barrier cannot be
    reached without crossing the near one, and an uncalibrated net will
    occasionally emit the impossible pair. Clamping here rather than trusting
    the heads keeps every caller (strategy gate, backtest, API) on the same
    arithmetic.
    """
    p_secure = float(min(max(p_secure, 0.0), 1.0))
    p_target = float(min(max(p_target, 0.0), p_secure))
    return p_target * target_r + (p_secure - p_target) * secure_r - (1.0 - p_secure) * 1.0 - cost_r


def breakeven_secure_probability(secure_r: float = DEFAULT_SECURE_R) -> float:
    """Win rate at which secure-only trading breaks even: ``1/(1+secure_r)``."""
    return 1.0 / (1.0 + max(secure_r, 1e-9))


def _shift_forward(values: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """``values[i + offset]`` with an explicit "this bar exists" mask.

    Padding with NaN alone is not enough: NaN comparisons silently evaluate
    False, which would read as "barrier not hit" rather than "unknown", and
    quietly bias every label near the end of the series.
    """
    n = len(values)
    out = np.empty(n, dtype=float)
    exists = np.zeros(n, dtype=bool)
    if offset < n:
        out[: n - offset] = values[offset:]
        exists[: n - offset] = True
    out[~exists] = np.nan
    return out, exists


def _barrier_walk(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry: np.ndarray,
    risk: np.ndarray,
    *,
    long_side: bool,
    secure_r: float,
    target_r: float,
    max_holding_bars: int,
) -> dict[str, np.ndarray]:
    """Two-stage barrier walk that reproduces this engine's actual stop path.

    A one-stage "target before stop" race is the wrong simulation and gets
    the answer badly wrong. Under a driftless random walk, P(+1.75R before
    -1R) = 1/(1+1.75) = 0.364, and plugging 0.364 into the expected-R formula
    yields **+0.56R per trade on random entries** — obvious nonsense, and
    exactly what a one-stage label produces on this data (measured: 0.3641).

    The engine does not hold a fixed stop. ``PositionManager`` rule 2 moves
    the stop up to ``entry + secure_r x risk`` once a base is cleared, so a
    position that has run in profit can no longer lose, and it is usually
    taken out at ~+0.2R rather than continuing to a distant target. That is
    why ~94% of backtest trades close at exactly +0.20R, and why the logged
    live decision this design was checked against had p_target = 0.064, not
    0.364.

    So the walk has two states:

      pre-secure   stop at ``entry - risk``. Reaching ``secure`` flips the
                   state; reaching ``target`` in this state still counts,
                   because the stop has not moved yet.
      post-secure  stop at ``entry + secure_r x risk``. Only ``target``
                   beats it now.

    Which price triggers what is not cosmetic, because the engine's three
    events have genuinely different mechanics:

      stop    a resting broker order — fills intrabar, so it reads ``low``
              (long) / ``high`` (short).
      target  also a resting broker order — reads ``high`` / ``low``.
      secure  **not** a resting order. It is ``PositionManager`` deciding, at
              an M5 candle close, to modify the stop. A wick that spikes
              through +0.2R and reverses within the bar never triggers it.
              So it reads ``close``, and the tighter stop is only in force
              from the *following* bar.

    Using ``high`` for the secure trigger — the obvious first implementation
    — inflates p_secure by crediting every wick, which is precisely the
    direction that flatters the gate.

    Vectorised over bars, looped over the horizon, so a 105k-bar M5 series
    labels in well under a second.
    """
    n = len(entry)
    sign = 1.0 if long_side else -1.0
    stop_initial = entry - sign * risk
    secure_level = entry + sign * secure_r * risk
    target_level = entry + sign * target_r * risk

    valid = np.isfinite(risk) & (risk > 0)
    alive = valid.copy()
    secured = np.zeros(n, dtype=bool)  # secure event has happened
    stop_armed = np.zeros(n, dtype=bool)  # ...and the tighter stop is in force
    hit_target = np.zeros(n, dtype=bool)
    mfe = np.zeros(n, dtype=float)
    bars_to_stop = np.full(n, np.nan)

    for step in range(1, max_holding_bars + 1):
        if not alive.any():
            break
        hi, exists = _shift_forward(high, step)
        lo, _ = _shift_forward(low, step)
        cl, _ = _shift_forward(close, step)
        live = alive & exists

        stop_now = np.where(stop_armed, secure_level, stop_initial)
        stopped = live & ((lo <= stop_now) if long_side else (hi >= stop_now))
        # Pessimistic: a bar that touched the stop contributes no favourable
        # progress, because the intrabar order is unknown.
        progressing = live & ~stopped

        favourable = (hi - entry) if long_side else (entry - lo)
        with np.errstate(invalid="ignore", divide="ignore"):
            excursion_r = np.where(progressing, favourable / risk, 0.0)
        mfe = np.maximum(mfe, np.nan_to_num(excursion_r, nan=0.0))

        reached_secure = progressing & ((cl >= secure_level) if long_side else (cl <= secure_level))
        reached_target = progressing & ((hi >= target_level) if long_side else (lo <= target_level))
        secured |= reached_secure
        hit_target |= reached_target

        bars_to_stop[stopped & np.isnan(bars_to_stop)] = step
        alive &= ~stopped
        # Arm the tighter stop only from the next iteration — see docstring.
        stop_armed |= reached_secure

    # Never resolved inside the horizon and never secured: price sat between
    # -1R and +0.2R for the whole window. Rare, and unlabellable either way,
    # so it is dropped rather than guessed at.
    unresolved = valid & alive & ~secured
    # A bar whose horizon runs past the end of the series cannot be labelled.
    _, horizon_exists = _shift_forward(high, max_holding_bars)
    incomplete = valid & ~horizon_exists & alive
    drop = unresolved | incomplete | ~valid

    secure_out = secured.astype(float)
    target_out = hit_target.astype(float)
    mfe_out = np.clip(mfe, 0.0, 10.0)
    secure_out[drop] = np.nan
    target_out[drop] = np.nan
    mfe_out[drop] = np.nan

    return {
        "secure": secure_out,
        "target": target_out,
        "mfe_r": mfe_out,
        "bars_to_stop": bars_to_stop,
    }


def _revert_labels(
    high: np.ndarray,
    low: np.ndarray,
    entry: np.ndarray,
    atr_values: np.ndarray,
    *,
    revert_atr_mult: float,
    revert_bars: int,
) -> np.ndarray:
    """Symmetric race: which side of ``+/- revert_atr_mult x ATR`` breaks first.

    Classes are 0 = down first, 1 = up first, 2 = neither, -1 = ambiguous
    (both inside one bar) or unresolvable (horizon runs past the series end).
    -1 rows are excluded from the loss rather than guessed at.
    """
    n = len(entry)
    band = revert_atr_mult * atr_values
    upper = entry + band
    lower = entry - band

    out = np.full(n, 2.0)
    decided = ~(np.isfinite(band) & (band > 0))
    out[decided] = -1.0
    complete = np.zeros(n, dtype=bool)

    for step in range(1, revert_bars + 1):
        hi, exists = _shift_forward(high, step)
        lo, _ = _shift_forward(low, step)
        complete |= exists
        live = ~decided & exists

        up_hit = live & (hi >= upper)
        down_hit = live & (lo <= lower)
        both = up_hit & down_hit
        out[both] = -1.0
        out[down_hit & ~both] = 0.0
        out[up_hit & ~both] = 1.0
        decided |= up_hit | down_hit

    # Ran out of series before the horizon closed: "neither" is unverified.
    _, horizon_exists = _shift_forward(high, revert_bars)
    out[~decided & ~horizon_exists] = -1.0
    return out


def generate_labels_v2(
    candles: pd.DataFrame,
    atr_series: pd.Series,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Three-outcome + give-back labels for every bar of ``candles``.

    Entry price is the bar's close — the first price a strategy evaluating on
    that closed bar could actually transact at. Risk is
    ``sl_atr_mult x ATR``, matching how the strategies here size stops.
    """
    cfg = config or LabelConfig()
    high = pd.to_numeric(candles["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(candles["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(candles["close"], errors="coerce").to_numpy(dtype=float)
    entry = close
    atr_values = pd.to_numeric(atr_series, errors="coerce").to_numpy(dtype=float)
    atr_values = np.where(np.isfinite(atr_values) & (atr_values > 0), atr_values, np.nan)
    risk = cfg.sl_atr_mult * atr_values

    long_side = _barrier_walk(
        high,
        low,
        close,
        entry,
        risk,
        long_side=True,
        secure_r=cfg.secure_r,
        target_r=cfg.target_r,
        max_holding_bars=cfg.max_holding_bars,
    )
    short_side = _barrier_walk(
        high,
        low,
        close,
        entry,
        risk,
        long_side=False,
        secure_r=cfg.secure_r,
        target_r=cfg.target_r,
        max_holding_bars=cfg.max_holding_bars,
    )
    revert = _revert_labels(
        high,
        low,
        entry,
        atr_values,
        revert_atr_mult=cfg.revert_atr_mult,
        revert_bars=cfg.revert_bars,
    )

    return pd.DataFrame(
        {
            "secure_long": long_side["secure"],
            "secure_short": short_side["secure"],
            "target_long": long_side["target"],
            "target_short": short_side["target"],
            "mfe_long_r": long_side["mfe_r"],
            "mfe_short_r": short_side["mfe_r"],
            "revert_class": revert,
            "bars_to_stop_long": long_side["bars_to_stop"],
            "bars_to_stop_short": short_side["bars_to_stop"],
        },
        index=candles.index,
    )
