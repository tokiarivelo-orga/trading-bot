"""XAUUSD REGIME ROUTER — M1 regime-switching meta-strategy.

Phase 2 of the 6-phase "XAUUSD regime-switching meta-strategy, DL-guided"
plan. Phase 1 trained `backend/data/ml_models/regime_router_priors_v1.json`
on ~11,400 tagged live trades: the offline neural net's holdout AUC came
back at chance (0.5049) given only ~27 days of tagged regime history, so it
honestly self-gated to `has_edge: false` and ZEROED weights rather than
shipping a falsely-confident correction — but the artifact's `bucket_table`
(the real, empirical secure-hit rate per `(regime_trend, regime_session)`
bucket) is a legitimate, non-overfit statistic and is what actually seeds
this bot's `AdaptiveLearner` prior below. See that JSON's
`label_definition_note`/`bucket_rate_stability` fields for the full caveat.

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
  domain.regime.compute_entry_regime` used to tag every trade this
  strategy's prior was trained on, so the bucket-key/feature lookups below
  stay apples-to-apples with the artifact) is a *lower* ADX bar for
  "trending" than `detect_structure_reversal`'s own internal gate (25.0).
  Verified directly against that detector's own passing test fixture
  (`test_reversal_fires_on_weakening_uptrend_broken_by_choch`): its ADX
  reads ~39, comfortably above BOTH thresholds — so on every input where
  `detect_structure_reversal` can possibly fire, this bot's own top-level
  read is *also* TRENDING, never RANGING. A strict "only call
  `detect_structure_reversal` when top-level reads RANGING" gate would
  therefore make Mode B provably unreachable, not just rare — confirmed
  empirically, not assumed. So mode selection below tries Mode A first
  (only when top-level reads TRENDING, exactly as specified), and falls
  back to attempting Mode B whenever Mode A does not produce a setup —
  covering both a genuine top-level RANGING bar and a TRENDING bar with no
  qualifying continuation pullback. `detect_structure_reversal`'s own
  internal gate is what actually decides whether it fires; this fallback
  only removes the redundant outer gate that made it structurally dead.

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
    band -> fade BUY. Grounded in real fleet evidence, not a guess: the
    #2 all-time live performer by total profit across the whole 61-bot
    fleet analyzed for this plan is `scalp_bollinger_reversion_v1`
    (+$1283, PF 1.78) — a completely different method family from every
    S&D/structure-zone bot in the fleet, and the natural RANGING-regime
    complement to Modes A/B (both of which need a still-elevated ADX read
    to fire at all, per the "MODE ROUTING" section above). Channel period
    (20) and band width (2.0 stddev) mirror that strategy's own
    `PERIOD`/`STD_MULT` constants.

Selection order per bar: Mode A (if TRENDING) -> Mode B (fallback,
regardless of what fired first, still gated by its own internal ADX>=25
check) -> Mode C (only if top-level RANGING and neither A nor B fired). No
fourth mode, no extra microstructure signals stacked on top of any of the
three — the plan explicitly names `xauusd_iof_scalp_m1`'s FVG/volume-delta/
absorption stack as the cautionary example of that mistake, and this
strategy deliberately keeps to three narrow, regime-distinct setup types
sharing one downstream SL/TP/learner pipeline.

────────────────────────────────────────────────────────────────────────
LOSS PREVENTION / AUTO-ADAPT
────────────────────────────────────────────────────────────────────────
`AdaptiveLearner` (`domain.online_learning`) gates and sizes every setup,
seeded with a `prior_logit` built from two pieces, both computed fresh every
`evaluate()` call, never parsed from JSON at runtime (the sandbox has no
`json` import — see `strategies/sandbox.py`'s `ALLOWED_IMPORT_MODULES` — so
the artifact's numbers are embedded below as literal module-scope
constants):

  1. `empirical_offset` — `logit(bucket_secure_rate) - logit(overall_rate)`,
     read straight from the JSON's `bucket_table` for this bar's
     `(regime_trend, regime_session)`.
  2. `net_correction` — the artifact's tiny embedded MLP (22 -> 8 tanh -> 1
     sigmoid), forward-passed on this bar's own regime feature vector, but
     only ever ADDED when `HAS_EDGE` (embedded below as `False`, matching
     the artifact) is `True`. Because every embedded weight is a literal
     zero (the artifact shipped a neutral no-op net, per its own gate — see
     module docstring above), this term is 0.0 today either way; it is
     wired and gated explicitly (not "implicitly inert because the weights
     happen to be zero") so a future retrain that flips `HAS_EDGE` to `True`
     and ships real weights turns this on with no strategy-code change
     beyond re-embedding the new constants.

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
# any of the code that consumes them below.
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
    it compiles (confirmed directly: even a single-field `@dataclass`
    defined inside sandboxed code crashes sandbox loading with
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


class XauusdRegimeRouterM1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="xauusd_regime_router_m1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            # Native higher-timeframe candles (no internal M1->M5 resample,
            # see module docstring) — declared so the engine's own HTF veto
            # pipeline has native M5/H1 to check against.
            confirmation_timeframes=("M5", "H1"),
            params={
                "atr_period": 14,
                # Matches `engine.domain.volatility.VolatilityConfig`'s own
                # defaults (atr_period=14, regime_lookback_bars=100) — the
                # convention the artifact's volatility tercile was trained
                # against.
                "vol_lookback_bars": 100,
                # SL: pivot/neckline price +/- this many ATR, floored/capped
                # in ATR terms — mirrors the ATR-multiple-beyond-invalidation
                # convention `xauusd_snd_apex_trendguard_m1`'s
                # `sl_zone_buffer_atr_mult`/`sl_min_atr_mult`/
                # `sl_max_atr_mult` params use for the same purpose.
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
                # (the fleet's #2 all-time performer by total profit, the
                # empirical grounding for this mode — see module docstring).
                "range_lookback_bars": 20,
                "range_band_mult": 2.0,
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
                # Time-barrier horizon for a pending sample: 180 M1 bars
                # (3h) — long enough for a scalp setup to either secure or
                # stall out.
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
        # RANGING-only pre-gate on Mode B would make it unreachable. Mode C
        # is tried last and only while top-level regime reads RANGING —
        # unlike Mode B it needs no such fallback (its own detector has no
        # conflicting internal ADX gate), so it stays a straightforward
        # regime-gated third branch.
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
        if setup is None and trend is TrendRegime.RANGING:
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
        horizon_ns = int(params["learner_horizon_bars"]) * _NS_PER_MINUTE
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
