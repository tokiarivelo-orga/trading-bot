"""XAUUSD REGIME ROUTER — M5 regime-switching meta-strategy.

A separate per-timeframe sibling family to `xauusd_regime_router_m1`
(distinct DB family name, own version ladder — same convention the fleet
already uses for `xauusd_snd_apex_trendguard_m1`/`_m5`/`_h1` and
`xauusd_snd_qm_structure_m1`/`_m5`, never a new version *within* the M1
family). Re-timeframed to `entry_timeframe="M5"`, same three-mode
architecture, same `AdaptiveLearner` gating, same prior artifact
(`backend/data/ml_models/regime_router_priors_v1.json` — session/trend/
volatility bucket rates, timeframe-agnostic: it was trained on tagged live
trades across the whole fleet, not specifically M1, so it is reused here
as-is with no retraining, exactly as the M1 file already does).

────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM THE M1 FILE, AND WHY (read this before touching params)
────────────────────────────────────────────────────────────────────────
1. `entry_timeframe="M5"`.
2. `confirmation_timeframes=("H1", "H4")` instead of the M1 file's
   `("M5", "H1")` — one and two rungs up from the entry timeframe, per this
   fleet's own convention for an M5-entry family (checked directly against
   two real sibling pairs before choosing: `xauusd_snd_apex_trendguard_m1`
   -> `_m5` moves `("M5","M15","H1")` -> `("M15","H1","H4")`, i.e. every
   rung shifts up one; `xauusd_snd_qm_structure_adaptive_m1` -> `_m5` moves
   `("H1",)` unchanged in *name* but is still "the rung(s) above entry".
   Reusing the M1 pair's literal `("M5","H1")` would leave `M5` as both the
   entry AND a confirmation timeframe, which no sibling pair in the fleet
   does). Same as the M1 file, these are declared solely so the engine's
   own HTF-veto pipeline (`trade_loop._veto_timeframe`, the rung
   immediately above entry_timeframe) has native H1 to check against —
   `evaluate()` below only ever reads its own M5 entry frame, exactly like
   the M1 file only ever reads its own M1 entry frame.
3. `entry_tf_minutes=5` (new param, absent from the M1 file — see #4).
4. **The one real code change beyond metadata**: `learner_horizon_bars`
   converts to real elapsed time via `* entry_tf_minutes` now, not via a
   bare `* _NS_PER_MINUTE` (which silently assumed 1-minute bars — true
   only for the M1 file). Checked against real fleet precedent rather than
   guessed: `xauusd_snd_qm_structure_adaptive_m1_v3`/`_m5_v2` is the one
   sibling pair in this fleet whose learner also converts a bar-count
   horizon to real time, and it keeps `learner_horizon_bars` LITERALLY
   IDENTICAL (120 in both files) while varying only `entry_tf_minutes`
   (1 vs 5) in the same multiplication this file now does. That is the
   exact pattern reproduced here.
5. **Every other param is left numerically IDENTICAL to the M1 file** —
   `atr_period`, `vol_lookback_bars`, the SL/TP ATR multiples, `tp_r_grid`,
   the range-reversion channel params, and every `learner_*` sample/
   half-life threshold. This was a reasoned call, not a default taken by
   omission: checked against the one real M1/M5 sibling pair in the fleet
   whose params are otherwise directly comparable,
   `xauusd_snd_apex_trendguard_m1_v6`/`_m5_v3` — its M5 file changes ONLY
   the timeframe-name literals (`zone_timeframe`, `htf_zone_timeframe`,
   `trend_weights` keys); every numeric threshold (`trend_min_score`,
   `max_extension_atr`, `fatigue_max`, `zone_height_atr_min/max`, ...) is
   byte-identical between the two files. `atr_period=14` itself is also
   identical in that pair despite M5 bars being 5x coarser — ATR already
   auto-normalizes to whatever bar size computes it, so a fixed-period ATR
   doesn't need re-tuning per timeframe the way an absolute price distance
   would. `vol_lookback_bars=100` matches `engine.domain.volatility.
   VolatilityConfig`'s own single default (`DEFAULT_REGIME_LOOKBACK_BARS`),
   which is itself timeframe-agnostic — the same constant regardless of
   which timeframe's bars are fed into it. This is the file's own
   documented instance of "if unsure, keep params identical to M1 and let
   `AdaptiveLearner`'s own evidence-gating adapt" — it is unsure by
   default, but here it isn't blind: two real sibling pairs in the fleet
   independently corroborate identical-params as the fleet's actual
   convention for a same-architecture M1->M5 retimeframe, not a guess.
6. Mode C (`range_reversion`) stays **disabled by default**
   (`enable_range_reversion=False`), inherited directly from the M1 file's
   v3 real-backtest finding (Mode C alone: PF 0.37, losing in every regime
   bucket, dragging the blended three-mode PF down from 4.31 to 1.51 — see
   the M1 file's own module docstring). That finding was measured on M1
   bars specifically; this file carries the same conservative default
   forward for M5 pending this family's OWN real-history M5 backtest (see
   `backend/scripts/regime_router_m5_backtest.py` /
   `backend/scripts/regime_router_m5_backtest_results.json`) rather than
   assuming Mode C is untested here — if that backtest's own per-mode
   breakdown clears Mode C on M5 bars, re-enabling it is a one-flag change
   exactly as the M1 file's own comment already describes, not a rewrite.

Everything else below — the three detectors, the mode-selection order, the
bucket-key format, the prior_logit sourcing, the size_multiplier formula,
and the regime-flip BREAKEVEN exit — mirrors `xauusd_regime_router_m1_v3`
exactly; see that file's own module docstring for the full design
rationale (duplicated only where genuinely needed below, not restated
wholesale).

────────────────────────────────────────────────────────────────────────
THREE MODES
────────────────────────────────────────────────────────────────────────
`src.strategies.domain.structure_continuation` supplies the first two:

  * Mode A ("trend_continuation") — `detect_trend_continuation()`: a
    pullback-and-resumption entry inside a trend the module's own ADX read
    (threshold 25, its own default) still calls TRENDING.
  * Mode B ("structure_reversal") — `detect_structure_reversal()`: a
    CHoCH-confirmed reversal off a weakening impulse leg. Its own gate 1
    ALSO requires the same ADX-TRENDING read (by construction — a reversal
    needs a still-measurably-trending leg to be reversing FROM; see that
    module's docstring) at its *own* internal default threshold (25.0).

  MODE ROUTING, AND WHY IT IS "TRY A, THEN B" RATHER THAN A HARD PRE-GATE
  ──────────────────────────────────────────────────────────────────────
  The top-level regime read every bar (`latest_trend_regime(highs, lows,
  closes)`, bare defaults, threshold 20.0 — the SAME convention `engine.
  domain.regime.compute_entry_regime` used to tag every trade the artifact's
  prior was trained on, so the bucket-key/feature lookups below stay
  apples-to-apples with the artifact regardless of entry timeframe) is a
  *lower* ADX bar for "trending" than `detect_structure_reversal`'s own
  internal gate (25.0). The M1 sibling verified directly, against that
  detector's own passing test fixture, that on every input where
  `detect_structure_reversal` can possibly fire, the top-level read is also
  TRENDING, never RANGING — making a strict "only call
  `detect_structure_reversal` when top-level reads RANGING" gate provably
  unreachable, not just rare. That argument is about the relationship
  between two ADX thresholds applied to the SAME candle series, independent
  of what timeframe those candles are sampled at, so it holds here
  unchanged. Mode selection below tries Mode A first (only when top-level
  reads TRENDING, exactly as specified), and falls back to attempting
  Mode B whenever Mode A does not produce a setup — covering both a genuine
  top-level RANGING bar and a TRENDING bar with no qualifying continuation
  pullback. `detect_structure_reversal`'s own internal gate is what
  actually decides whether it fires; this fallback only removes the
  redundant outer gate that made it structurally dead.

  * Mode C ("range_reversion") — `_detect_range_reversion()` (inline,
    below; deliberately NOT a shared `structure_continuation` addition —
    it needs no fractal-pivot machinery, and keeping it self-contained
    means no `sandbox.py` allowlist change). Gates strictly on top-level
    `TrendRegime.RANGING` (so, unlike Mode B, this mode's own bucket rows
    are genuinely `ranging`-tagged) and is tried only after both A and B
    have found nothing for the bar. A classic Bollinger-band mean-reversion
    fade: the latest closed bar's high pierces a rolling mean +/-
    `range_band_mult` x stddev channel and its close comes back inside it
    (a rejection) -> fade SELL toward the mean; symmetric on the lower
    band -> fade BUY. Channel period (20) and band width (2.0 stddev)
    mirror the M1 file's own constants, which in turn mirror
    `scalp_bollinger_reversion_v1`'s `PERIOD`/`STD_MULT` — see point 6
    above for why Mode C stays off by default here regardless.

  DISABLED BY DEFAULT — see "WHAT CHANGED" point 6 above.

Selection order per bar: Mode A (if TRENDING) -> Mode B (fallback,
regardless of what fired first, still gated by its own internal ADX>=25
check) -> Mode C (only if top-level RANGING, neither A nor B fired, AND
`enable_range_reversion` is `True` — off by default). No fourth mode, no
extra microstructure signals stacked on top of any of the three.

────────────────────────────────────────────────────────────────────────
LOSS PREVENTION / AUTO-ADAPT
────────────────────────────────────────────────────────────────────────
`AdaptiveLearner` (`domain.online_learning`) gates and sizes every setup,
seeded with a `prior_logit` built from two pieces, both computed fresh every
`evaluate()` call, never parsed from JSON at runtime (the sandbox has no
`json` import — see `strategies/sandbox.py`'s `ALLOWED_IMPORT_MODULES`, so
the artifact's numbers are embedded below as literal module-scope
constants, identical to the M1 file's):

  1. `empirical_offset` — `logit(bucket_secure_rate) - logit(overall_rate)`,
     read straight from the JSON's `bucket_table` for this bar's
     `(regime_trend, regime_session)`.
  2. `net_correction` — the artifact's tiny embedded MLP (22 -> 8 tanh -> 1
     sigmoid), forward-passed on this bar's own regime feature vector, but
     only ever ADDED when `HAS_EDGE` (embedded below as `False`, matching
     the artifact) is `True`. Every embedded weight is a literal zero (the
     artifact shipped a neutral no-op net, per its own edge gate), so this
     term is 0.0 today either way — wired and gated explicitly so a future
     retrain that flips `HAS_EDGE` and ships real weights turns this on
     with no strategy-code change beyond re-embedding the new constants.

Bucket key: `f"{mode}|{session.value}|{trend.value}"`. Gate: trade a cold
bucket (fewer than `min_bucket_samples` observations this instance has
personally seen) on the structure detector's own confirmation alone, or a
warmed bucket only when `verdict.expectancy_r > 0` — no permanent blacklist
on any bucket, since `AdaptiveLearner`'s half-life decay is what "auto-
adapt" means here.

`Signal.size_multiplier` scales with `verdict.edge_over(learner.breakeven_p)`
(log-odds distance above the secure-hit break-even probability), clamped to
the engine's enforced `[1.0, 2.0]` band (`trade_loop._clamp_size_multiplier`)
— cold or barely-positive setups stay at 1.0x, confident warmed buckets size
up modestly.

────────────────────────────────────────────────────────────────────────
EXIT
────────────────────────────────────────────────────────────────────────
`ExitDecision(BREAKEVEN)` only, and only for a Mode-A (trend_continuation)
position of this bot's own (`ctx.own_position`), when the regime has flipped
TRENDING -> RANGING since entry. No `SET_SL`/`CLOSE` — this reuses the one
exit action every strategy already has, per the plan.

Sandbox-safe: imports limited to `math`, `numpy`, `pandas`,
`src.strategies.domain.models`, `src.strategies.domain.online_learning`,
`src.strategies.domain.structure_continuation`, `src.engine.domain.regime`
— all on `strategies/sandbox.py`'s `ALLOWED_IMPORT_MODULES`. No I/O, no
network, no broker access.
"""

import math

import numpy as np
import pandas as pd

from src.engine.domain.regime import (
    RegimeConfig,
    TradingSession,
    TrendRegime,
    latest_trend_regime,
    session_for,
)
from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
    IndicatorReading,
    Signal,
    StrategySpec,
)
from src.strategies.domain.online_learning import AdaptiveLearner, LearnerConfig
from src.strategies.domain.structure_continuation import (
    detect_structure_reversal,
    detect_trend_continuation,
)

# ─────────────────────────────────────────────────────────────────────
# Embedded artifact — backend/data/ml_models/regime_router_priors_v1.json,
# trained_at 2026-09-03T09:32:34Z, symbol XAUUSD, n=11413 tagged trades
# (n_train=7989, n_holdout=3424). Baked in as literal constants: the
# sandbox has no `json` import (see ALLOWED_IMPORT_MODULES), so this file
# cannot load the artifact at runtime — a future retrain re-embeds these
# constants by hand (or via a small generator script), it does not change
# any of the code that consumes them below. Identical to the M1 sibling's
# copy — the artifact is timeframe-agnostic (session/trend/vol buckets
# over the whole tagged fleet), not specific to any one entry timeframe.
# ─────────────────────────────────────────────────────────────────────

ARTIFACT_TRAINED_AT = "2026-09-03T09:32:34.520946+00:00"
ARTIFACT_SECURE_R = 0.2  # matches online_learning.DEFAULT_SECURE_R

FEATURE_ORDER = (
    "session_asian",
    "session_london",
    "session_overlap",
    "session_new_york",
    "session_off_session",
    "session_unknown",
    "trend_trending",
    "trend_ranging",
    "trend_unknown",
    "vol_low",
    "vol_mid",
    "vol_high",
    "vol_unknown",
    "hour_sin",
    "hour_cos",
    "dow_mon",
    "dow_tue",
    "dow_wed",
    "dow_thu",
    "dow_fri",
    "dow_sat",
    "dow_sun",
)
N_FEATURES = len(FEATURE_ORDER)  # 22

# Fit on the artifact's TRAIN split only (chronological, no lookahead) —
# the tercile cut points a live bar's volatility percentile rank (0-100,
# tie-aware, same definition `engine.domain.regime`/`engine.domain.
# volatility` uses) is bucketed against below, so live buckets line up with
# the trained bucket_table's own volatility convention.
VOL_TERCILE_LOW_UPPER = 17.0
VOL_TERCILE_MID_UPPER = 55.00000000000001

STANDARDIZER_MEAN = np.array(
    [
        0.36212291901364374,
        0.17674302165477532,
        0.18525472524721492,
        0.275879334084366,
        0.0,
        0.0,
        0.8505444986856928,
        0.14945550131430718,
        0.0,
        0.3343347102265615,
        0.34347227437726874,
        0.32219301539616974,
        0.0,
        -0.0555918645196984,
        -0.0478832664945336,
        0.2801351858805858,
        0.12867693077982226,
        0.14657654274627613,
        0.1585930654650144,
        0.26924521216672925,
        0.0,
        0.01677306296157216,
    ],
    dtype=float,
)
STANDARDIZER_STD = np.array(
    [
        0.48061409731580057,
        0.3814510793681346,
        0.3885040695035173,
        0.44695629216908084,
        1.0,
        1.0,
        0.35653689071566813,
        0.35653689071566813,
        1.0,
        0.4717573653524476,
        0.47486742477387833,
        0.46731646260975357,
        1.0,
        0.7235734968597443,
        0.6863367482739056,
        0.44906509941458983,
        0.33484201986146706,
        0.3536832761989534,
        0.36529618811510217,
        0.4435676136645034,
        1.0,
        0.12842012038796452,
    ],
    dtype=float,
)

# `has_edge: false` in the artifact -> its own train script ships a literal
# no-op net (every weight zeroed) rather than the trained-but-not-
# generalizing one (holdout AUC 0.5049, chance level) — see module
# docstring. `np.zeros((22, 8))`/`np.zeros((8, 1))` IS the literal shipped
# value here, not a placeholder; shapes match `feature_order` (22) -> 8
# hidden units -> 1 output exactly, so a future retrain only has to replace
# these four arrays and flip `HAS_EDGE`.
HAS_EDGE = False
LAYER1_WEIGHTS = np.zeros((N_FEATURES, 8), dtype=float)
LAYER1_BIAS = np.zeros(8, dtype=float)
LAYER2_WEIGHTS = np.zeros((8, 1), dtype=float)
LAYER2_BIAS = np.zeros(1, dtype=float)

# `bucket_table` from the artifact: empirical secure-hit rate per
# (regime_trend, regime_session), full dataset, keyed exactly as
# `TrendRegime.value`/`TradingSession.value` emit them. No `off_session`
# row exists in the artifact (never occurred in this training window) —
# `_empirical_offset` below falls back to a zero offset for any bucket not
# present here, rather than guessing.
BUCKET_TABLE = {
    ("ranging", "asian"): 0.7684,
    ("ranging", "london"): 0.7689,
    ("ranging", "new_york"): 0.7955,
    ("ranging", "overlap"): 0.8072,
    ("trending", "asian"): 0.8108,
    ("trending", "london"): 0.7661,
    ("trending", "new_york"): 0.7720,
    ("trending", "overlap"): 0.8612,
}
# Count-weighted average secure_rate across every `bucket_table` row
# (583+489+401+498+3283+2159+2566+1434 = 11413 trades) — the reference rate
# `_empirical_offset` measures each bucket against, in log-odds.
OVERALL_SECURE_RATE = 0.7952974239901865

_LOGIT_EPS = 1e-6
_NS_PER_MINUTE = 60 * 1_000_000_000
# A candle stream jumping backwards, or gapping more than this, means a
# fresh backtest/replay is reusing this strategy instance — reset the
# learner rather than let one run's learning leak into the next. Mirrors
# `xauusd_snd_qm_structure_adaptive_m1_v2`'s `_RESET_GAP_NS` convention.
# Absolute wall-clock threshold, not bar-count-based — unaffected by the
# M1->M5 retimeframe, so left identical to the M1 file's own value.
_RESET_GAP_NS = 30 * 24 * 60 * _NS_PER_MINUTE

MIN_HISTORY_BARS = 60


def _logit(p: float) -> float:
    clipped = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return math.log(clipped / (1.0 - clipped))


def _true_range(df) -> pd.Series:
    prev_close = df["close"].shift(1)
    # `pd.Series(...)` wrap: `DataFrame.max(axis=1)` is unambiguously a
    # `Series` at runtime but pandas-stubs' overloads leave pyright unable
    # to narrow it — the explicit wrap (not a cast, a real no-op
    # reconstruction) is what actually silences `reportReturnType` here.
    return pd.Series(
        pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
    )


def _atr(df, period) -> pd.Series:
    return pd.Series(_true_range(df).rolling(period, min_periods=period).mean())


class RangeReversionSetup:
    """One qualifying Bollinger-band mean-reversion fade — mirrors
    `structure_continuation.ContinuationSetup`/`ReversalSetup`'s shape
    (`direction`, `pivot_price`, `structure_points`) so `evaluate()`'s
    downstream SL/TP/Signal-building code handles all three modes with the
    same lines, no per-mode branching. `pivot_price` is the rejection
    extreme (the bar's high for a SELL fade, low for a BUY fade) — the
    level that invalidates the fade if price closes back through it, the
    same role `ContinuationSetup`/`ReversalSetup`'s own pivot plays for
    Modes A/B. `structure_points` is always empty — this mode has no
    fractal swing annotations to chart.

    Deliberately a plain class, not `@dataclass` — `sandbox.py`'s own
    `from __future__ import annotations` is inherited by `compile()`'s
    default future-flag inheritance into every generated-strategy source
    it compiles (confirmed directly on the M1 sibling: even a single-field
    `@dataclass` defined inside sandboxed code crashes sandbox loading with
    `AttributeError: 'NoneType' object has no attribute '__dict__'`,
    because `dataclasses._process_class` then sees stringified
    annotations and does an unguarded `sys.modules[cls.__module__].
    __dict__` lookup for a module name — `"sandboxed_strategy"` — that was
    never registered in `sys.modules`). A plain `__init__` sidesteps the
    dataclass machinery entirely and needs no fix upstream."""

    def __init__(
        self, *, direction: Direction, pivot_price: float, mean_price: float, band_price: float
    ) -> None:
        self.direction = direction
        self.pivot_price = pivot_price
        self.mean_price = mean_price
        self.band_price = band_price
        self.structure_points: tuple = ()


def _detect_range_reversion(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, *, lookback: int, band_mult: float
) -> RangeReversionSetup | None:
    """Mode C's detector — inline and self-contained (no `sandbox.py`
    change, no new shared domain module; see module docstring's "THREE
    MODES" section for why). Channel = mean +/- `band_mult` x stddev of
    closes over the trailing `lookback` bars, computed as of the bar
    BEFORE the latest one (so the pierce/rejection bar itself is never
    part of the channel it's being tested against — avoids the channel
    quietly absorbing the very extreme it's supposed to flag). Fires only
    when the latest CLOSED bar's high pierces the upper band and its close
    comes back inside it (a rejection, not just a touch) -> SELL fade
    toward the mean; symmetric on the lower band -> BUY fade. Returns
    `None` on too-short history or a degenerate (zero-variance) channel —
    never raises."""
    n = len(closes)
    if n < lookback + 2:
        return None
    closes_series = pd.Series(closes)
    rolling_mean = pd.Series(closes_series.rolling(lookback, min_periods=lookback).mean())
    rolling_std = pd.Series(closes_series.rolling(lookback, min_periods=lookback).std(ddof=0))
    mean_prev = float(rolling_mean.iloc[-2])
    std_prev = float(rolling_std.iloc[-2])
    if not math.isfinite(mean_prev) or not math.isfinite(std_prev) or std_prev <= 0.0:
        return None

    upper = mean_prev + band_mult * std_prev
    lower = mean_prev - band_mult * std_prev
    latest_high = float(highs[-1])
    latest_low = float(lows[-1])
    latest_close = float(closes[-1])

    if latest_high > upper and latest_close <= upper:
        return RangeReversionSetup(
            direction=Direction.SELL,
            pivot_price=latest_high,
            mean_price=mean_prev,
            band_price=upper,
        )
    if latest_low < lower and latest_close >= lower:
        return RangeReversionSetup(
            direction=Direction.BUY,
            pivot_price=latest_low,
            mean_price=mean_prev,
            band_price=lower,
        )
    return None


def _volatility_percentile_rank(atr_values: np.ndarray, lookback: int) -> float:
    """Tie-aware percentile rank (0-100) of the latest ATR reading against
    its own trailing `lookback` window, excluding the current bar —
    bit-for-bit the same definition `engine.domain.volatility.
    latest_volatility_regime`/`_percentile_rank` uses (that module is not
    sandbox-importable, see `strategies/sandbox.py`'s
    `ALLOWED_IMPORT_MODULES`, so this is a deliberate reimplementation, not
    a divergent one) — the artifact's `volatility_tercile_bounds` cutoffs
    were fit on exactly this quantity, so live buckets only line up with
    the trained ones if this matches precisely."""
    valid = atr_values[np.isfinite(atr_values)]
    if valid.size < 2:
        return float("nan")
    current = float(valid[-1])
    window = valid[max(0, valid.size - 1 - lookback) : valid.size - 1]
    if window.size == 0:
        return float("nan")
    below = float(np.sum(window < current))
    tied = float(np.sum(window == current))
    return (below + 0.5 * tied) / window.size * 100.0


def _volatility_tercile(percentile: float) -> str:
    if not math.isfinite(percentile):
        return "unknown"
    if percentile <= VOL_TERCILE_LOW_UPPER:
        return "low"
    if percentile <= VOL_TERCILE_MID_UPPER:
        return "mid"
    return "high"


_SESSION_INDEX = {
    TradingSession.ASIAN: 0,
    TradingSession.LONDON: 1,
    TradingSession.OVERLAP: 2,
    TradingSession.NEW_YORK: 3,
    TradingSession.OFF_SESSION: 4,
}
_TREND_INDEX = {TrendRegime.TRENDING: 6, TrendRegime.RANGING: 7}
_VOL_INDEX = {"low": 9, "mid": 10, "high": 11}


def _regime_features(
    session: TradingSession, trend: TrendRegime, vol_bucket: str, hour_utc: float, weekday: int
) -> np.ndarray:
    """Builds one length-`N_FEATURES` vector matching `FEATURE_ORDER`
    exactly — used both as the embedded net's own input (see
    `_prior_net_correction`) and as `AdaptiveLearner`'s online feature
    vector, so the two models see the same regime description of "now"."""
    x = np.zeros(N_FEATURES, dtype=float)
    session_idx = _SESSION_INDEX.get(session)
    x[session_idx if session_idx is not None else 5] = 1.0
    trend_idx = _TREND_INDEX.get(trend)
    x[trend_idx if trend_idx is not None else 8] = 1.0
    vol_idx = _VOL_INDEX.get(vol_bucket)
    x[vol_idx if vol_idx is not None else 12] = 1.0
    x[13] = math.sin(2.0 * math.pi * hour_utc / 24.0)
    x[14] = math.cos(2.0 * math.pi * hour_utc / 24.0)
    if 0 <= weekday <= 6:
        x[15 + weekday] = 1.0
    return x


def _prior_net_correction(features: np.ndarray) -> float:
    """Embedded-MLP forward pass (22 -> 8 tanh -> 1 sigmoid), matching the
    artifact's `layer1_weights/bias`, `layer2_weights/bias`,
    `standardizer_mean/std` exactly. Every weight here is a literal zero
    (see module docstring), so this returns 0.0 regardless of `features`
    today — but is a real, general forward pass, ready for a future retrain
    that re-embeds non-zero weights with no other code change."""
    std = np.where(STANDARDIZER_STD < 1e-8, 1.0, STANDARDIZER_STD)
    x_std = (features - STANDARDIZER_MEAN) / std
    hidden = np.tanh(x_std @ LAYER1_WEIGHTS + LAYER1_BIAS)
    z2 = float(hidden @ LAYER2_WEIGHTS[:, 0] + LAYER2_BIAS[0])
    output = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z2))))
    return float(np.clip(_logit(output) - _logit(0.5), -1.5, 1.5))


def _empirical_offset(trend: TrendRegime, session: TradingSession) -> float:
    rate = BUCKET_TABLE.get((trend.value, session.value))
    if rate is None:
        return 0.0
    return _logit(rate) - _logit(OVERALL_SECURE_RATE)


def _prior_logit(trend: TrendRegime, session: TradingSession, features: np.ndarray) -> float:
    offset = _empirical_offset(trend, session)
    correction = _prior_net_correction(features) if HAS_EDGE else 0.0
    return offset + correction


def _confidence(verdict) -> float:
    if verdict is None or not verdict.ready:
        return 0.65
    return float(np.clip(verdict.p_secure, 0.05, 0.99))


def _size_multiplier(verdict, breakeven_p: float, edge_scale: float) -> float:
    """Bounded [1.0, 2.0] per-signal risk multiplier (the engine clamps to
    this band regardless, `trade_loop._clamp_size_multiplier` — this keeps
    the strategy's own output already inside it). A cold or barely-positive
    bucket stays at 1.0x; `edge_scale` log-odds of edge over break-even maps
    to the full +1.0 of headroom."""
    if verdict is None or not verdict.ready:
        return 1.0
    edge = verdict.edge_over(breakeven_p)
    return float(np.clip(1.0 + max(edge, 0.0) / edge_scale, 1.0, 2.0))


class XauusdRegimeRouterM5:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="xauusd_regime_router_m5",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            # One and two rungs up from the M5 entry timeframe — see module
            # docstring's "WHAT CHANGED FROM THE M1 FILE" point 2 for the
            # sibling-pair precedent this follows. Declared solely so the
            # engine's own HTF-veto pipeline has native H1 to check
            # against; evaluate() below only ever reads its own M5 entry
            # frame, exactly like the M1 file only ever reads its own M1
            # entry frame.
            confirmation_timeframes=("H1", "H4"),
            params={
                "atr_period": 14,
                # Matches `engine.domain.volatility.VolatilityConfig`'s own
                # defaults (atr_period=14, regime_lookback_bars=100) — the
                # convention the artifact's volatility tercile was trained
                # against. Kept identical to the M1 file — see module
                # docstring point 5 (this default is itself timeframe-
                # agnostic, not something that needs re-tuning per entry
                # timeframe).
                "vol_lookback_bars": 100,
                # SL: pivot/neckline price +/- this many ATR, floored/capped
                # in ATR terms — mirrors the ATR-multiple-beyond-invalidation
                # convention `xauusd_snd_apex_trendguard_m1`'s
                # `sl_zone_buffer_atr_mult`/`sl_min_atr_mult`/
                # `sl_max_atr_mult` params use for the same purpose. ATR
                # self-normalizes to whatever bar size computed it, so these
                # multiples are kept identical to the M1 file (see module
                # docstring point 5).
                "sl_buffer_atr_mult": 0.25,
                "sl_min_atr_mult": 0.50,
                "sl_max_atr_mult": 2.50,
                # `configs/symbols/xauusd.yaml`'s `min_rr: 1.5` is the hard
                # SpreadGate floor; the grid starts with real margin above
                # it (mirrors `breakout_v1`'s TP_RR=2.2 against the same
                # 1.5 floor, and apex_trendguard's tp1_target_rr=1.80) so a
                # signal is never silently vetoed at the broker layer.
                "broker_min_rr": 1.5,
                "tp_r_grid": (1.8, 2.2, 2.6, 3.0, 3.5),
                # Mode C (range_reversion) channel — mirrors
                # `scalp_bollinger_reversion_v1`'s own PERIOD=20/STD_MULT=2.0
                # (the same constants the M1 file's Mode C uses).
                "range_lookback_bars": 20,
                "range_band_mult": 2.0,
                # Inherited OFF from the M1 file's own real-backtest finding
                # (Mode C alone: PF 0.37, losing in every regime bucket —
                # see module docstring "WHAT CHANGED" point 6). That finding
                # was measured on M1 bars; kept off here as the conservative
                # default pending this family's own M5 real-history
                # backtest. The detector and its params above are left
                # fully in place, not deleted, so this is a one-flag
                # re-enable once a fresh M5 backtest validates it, not an
                # architecture change.
                "enable_range_reversion": False,
                # AdaptiveLearner — secure_r matches the artifact's own
                # secure_r (0.2) so the bucket_table prior and this
                # instance's own live evidence are measuring the same
                # label.
                "learner_secure_r": ARTIFACT_SECURE_R,
                "learner_min_bucket_samples": 25.0,
                "learner_min_model_samples": 150,
                "learner_min_quantile_samples": 30,
                "learner_half_life_samples": 750.0,
                "learner_max_pending": 400,
                # Minutes per entry-timeframe bar — new vs. the M1 file (see
                # module docstring point 3/4): converts the bar-count
                # horizon below into real elapsed time. The M1 file hardcoded
                # 1 implicitly (`* _NS_PER_MINUTE` with no multiplier); this
                # file makes it explicit rather than silently assuming M1's
                # bar size, the same way
                # `xauusd_snd_qm_structure_adaptive_m1`/`_m5` already does.
                "entry_tf_minutes": 5,
                # Time-barrier horizon for a pending sample: 180 bars, kept
                # numerically IDENTICAL to the M1 file (bar-count, not real
                # time) per the `qm_structure_adaptive` M1/M5 sibling
                # precedent cited in module docstring point 4. At 5 min/bar
                # that is 900 minutes (15h) of real time to either secure or
                # stall out, vs. the M1 file's 180 minutes (3h) — a wider
                # real-time window is the correct, not accidental,
                # consequence of coarser bars.
                "learner_horizon_bars": 180,
                # `edge_over()` is in log-odds; +2.0 log-odds of edge above
                # break-even maps to the full +1.0 of size headroom.
                "size_edge_scale": 2.0,
            },
        )
        self._learner: AdaptiveLearner | None = None
        self._last_bar_ns: int | None = None
        # Which mode produced the currently-open position, so the exit
        # rule below only ever acts on a Mode-A (trend_continuation)
        # position — set on every emitted Signal, read only while
        # `ctx.own_position` is not None.
        self._entry_mode: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def _ensure_learner(self, params) -> AdaptiveLearner:
        if self._learner is None:
            self._learner = AdaptiveLearner(
                N_FEATURES,
                LearnerConfig(
                    secure_r=float(params["learner_secure_r"]),
                    min_bucket_samples=float(params["learner_min_bucket_samples"]),
                    min_model_samples=int(params["learner_min_model_samples"]),
                    min_quantile_samples=int(params["learner_min_quantile_samples"]),
                    half_life_samples=float(params["learner_half_life_samples"]),
                    max_pending=int(params["learner_max_pending"]),
                ),
            )
        return self._learner

    def reset_state(self) -> None:
        if self._learner is not None:
            self._learner.reset()
        self._last_bar_ns = None
        self._entry_mode = None

    def _check_continuity(self, first_ns: int, last_ns: int) -> None:
        if self._last_bar_ns is None:
            return
        if last_ns < self._last_bar_ns or first_ns > self._last_bar_ns + _RESET_GAP_NS:
            self.reset_state()

    # ── exit management (own position only) ────────────────────────────

    def _maybe_exit(self, ctx, trend: TrendRegime) -> ExitDecision | None:
        if ctx.own_position is None:
            return None
        if self._entry_mode != "trend_continuation":
            return None
        if trend is not TrendRegime.RANGING:
            return None
        return ExitDecision(
            action=ExitActionKind.BREAKEVEN,
            reason=(
                "Mode A (trend_continuation) position open and the regime has "
                "flipped TRENDING -> RANGING since entry — moving stop to "
                "entry to protect the continuation thesis, per the plan's "
                "regime-flip exit rule."
            ),
        )

    # ── entry point ──────────────────────────────────────────────────

    def evaluate(self, ctx):
        params = self.spec.params
        entry = ctx.candles.get(self.spec.entry_timeframe)
        if entry is None or len(entry) < MIN_HISTORY_BARS or "time" not in entry.columns:
            return None

        highs = entry["high"].to_numpy()
        lows = entry["low"].to_numpy()
        closes = entry["close"].to_numpy()
        volumes = (
            entry["tick_volume"].to_numpy()
            if "tick_volume" in entry.columns
            else np.ones_like(closes)
        )
        times = list(entry["time"])
        now = entry["time"].iloc[-1]

        atr_series = _atr(entry, int(params["atr_period"]))
        atr_clean = atr_series.dropna()
        if atr_clean.empty or float(atr_clean.iloc[-1]) <= 0.0:
            return None
        atr_now = float(atr_clean.iloc[-1])

        entry_t_ns = pd.DatetimeIndex(entry["time"]).as_unit("ns").asi8
        self._check_continuity(int(entry_t_ns[0]), int(entry_t_ns[-1]))

        # ── regime read, every bar ──────────────────────────────────
        trend, _adx = latest_trend_regime(highs, lows, closes)
        session = session_for(now, RegimeConfig())
        vol_pct = _volatility_percentile_rank(
            atr_series.to_numpy(), int(params["vol_lookback_bars"])
        )
        vol_bucket = _volatility_tercile(vol_pct)
        hour_utc = float(now.hour) + float(now.minute) / 60.0
        weekday = int(now.weekday())
        features = _regime_features(session, trend, vol_bucket, hour_utc, weekday)

        learner = self._ensure_learner(params)
        learner.advance(entry_t_ns, highs, lows)
        self._last_bar_ns = int(entry_t_ns[-1])

        exit_decision = self._maybe_exit(ctx, trend)
        if ctx.own_position is not None:
            # One position at a time for this bot: manage the open one,
            # never stack a fresh entry on top of it.
            return exit_decision

        # ── mode selection + setup detection ─────────────────────────
        # Try Mode A first (only while top-level regime reads TRENDING);
        # fall back to attempting Mode B whenever Mode A found nothing —
        # see module docstring's "MODE ROUTING" section for why a strict
        # RANGING-only pre-gate on Mode B would make it unreachable. Modes
        # A and B stay unconditionally on. Mode C is tried last, only while
        # top-level regime reads RANGING, AND only when
        # `enable_range_reversion` is True — OFF by default (see module
        # docstring "WHAT CHANGED" point 6). The detector itself is
        # untouched; this is purely a call-site gate so re-enabling it
        # later is a one-param flip, not a rewrite.
        mode = None
        setup = None
        if trend is TrendRegime.TRENDING:
            setup = detect_trend_continuation(
                highs=highs, lows=lows, closes=closes, times=times, atr_series=atr_series
            )
            if setup is not None:
                mode = "trend_continuation"
        if setup is None:
            setup = detect_structure_reversal(
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                times=times,
                atr_series=atr_series,
            )
            if setup is not None:
                mode = "structure_reversal"
        if (
            setup is None
            and trend is TrendRegime.RANGING
            and bool(params["enable_range_reversion"])
        ):
            setup = _detect_range_reversion(
                highs,
                lows,
                closes,
                lookback=int(params["range_lookback_bars"]),
                band_mult=float(params["range_band_mult"]),
            )
            if setup is not None:
                mode = "range_reversion"
        if setup is None or mode is None:
            return None

        direction = setup.direction
        close_price = float(closes[-1])
        pivot_price = float(setup.pivot_price)
        buffer_dist = float(params["sl_buffer_atr_mult"]) * atr_now
        if direction is Direction.BUY:
            sl_price = pivot_price - buffer_dist
            sl_points = close_price - sl_price
        else:
            sl_price = pivot_price + buffer_dist
            sl_points = sl_price - close_price
        if sl_points <= 0.0:
            return None

        sl_min = float(params["sl_min_atr_mult"]) * atr_now
        sl_max = float(params["sl_max_atr_mult"]) * atr_now
        if sl_points < sl_min:
            sl_points = sl_min
        if sl_points > sl_max:
            # Oversized stop off an oversized pivot swing — skip rather
            # than widen past the cap (mirrors apex_trendguard's
            # `sl_max_atr_mult` convention).
            return None

        bucket = f"{mode}|{session.value}|{trend.value}"
        prior_logit = _prior_logit(trend, session, features)
        verdict = learner.score(features, bucket, prior_logit=prior_logit)

        gate_ok = (not verdict.ready) or verdict.expectancy_r > 0.0
        if not gate_ok:
            return None

        grid = tuple(float(x) for x in params["tp_r_grid"])
        r_target, _expected_r, p_hit = learner.best_target(bucket, grid=grid, prior_curve=None)
        tp_points = sl_points * r_target
        min_rr = float(params["broker_min_rr"])
        if tp_points < min_rr * sl_points:
            # Defensive — the grid is already built with margin above
            # `broker_min_rr`, so this should not trigger, but a signal
            # this close to the SpreadGate floor is worth skipping outright
            # rather than shipping one that gets silently vetoed downstream.
            return None

        now_ns = int(entry_t_ns[-1])
        horizon_ns = (
            int(params["learner_horizon_bars"]) * int(params["entry_tf_minutes"]) * _NS_PER_MINUTE
        )
        learner.observe(
            features=features,
            bucket=bucket,
            entry_ns=now_ns,
            entry_price=close_price,
            direction=1 if direction is Direction.BUY else -1,
            sl_dist=sl_points,
            atr=atr_now,
            deadline_ns=now_ns + horizon_ns,
        )

        breakeven_p = learner.breakeven_p
        size_multiplier = _size_multiplier(verdict, breakeven_p, float(params["size_edge_scale"]))

        readings = (
            IndicatorReading(
                name="learner_p_secure",
                value=round(verdict.p_secure, 4),
                threshold=round(breakeven_p, 4),
                comparison=">",
                passed=gate_ok,
            ),
            IndicatorReading(
                name="learner_bucket_samples",
                value=round(verdict.bucket_samples, 1),
                threshold=float(params["learner_min_bucket_samples"]),
                comparison=">",
                passed=verdict.ready,
            ),
            IndicatorReading(
                name="prior_logit",
                value=round(prior_logit, 4),
                threshold=0.0,
                comparison=">",
                passed=prior_logit > 0.0,
            ),
        )

        reason = (
            f"{mode} {direction.value} pivot={pivot_price:.2f} close={close_price:.2f} "
            f"sl={sl_points:.2f} tp={tp_points:.2f}(rr={r_target:.2f},p_hit={p_hit:.2f}) "
            f"session={session.value} trend={trend.value} vol={vol_bucket} "
            f"prior_logit={prior_logit:.3f}(empirical={_empirical_offset(trend, session):.3f}"
            f",net={'on' if HAS_EDGE else 'off'}) "
            f"p_secure={verdict.p_secure:.3f} breakeven={breakeven_p:.3f} "
            f"bucket_n={verdict.bucket_samples:.0f}{'' if verdict.ready else '(cold)'} "
            f"size={size_multiplier:.2f}x"
        )

        self._entry_mode = mode
        return Signal(
            direction=direction,
            sl_points=sl_points,
            tp_points=tp_points,
            confidence=_confidence(verdict),
            reason=reason,
            pattern=mode,
            structure=setup.structure_points,
            indicators=readings,
            size_multiplier=size_multiplier,
        )
