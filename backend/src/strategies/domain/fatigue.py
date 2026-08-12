"""Trend fatigue — how much breath is left in the move currently running.

The question this answers is the one a discretionary trader asks before
clicking: *is this trend still fresh, or is it out of breath?* A bot that
can answer it stops buying the last leg of a vertical move, and knows when
a fade is worth taking instead of always being the wrong side of one.

    score 0.0  → the trend is fresh: impulses still expanding, momentum
                 confirming every new extreme, price near its anchor
    score 1.0  → the trend is exhausted: shrinking legs, deeper pullbacks,
                 momentum diverging, price stretched far from its anchor
                 and being rejected at the extreme

────────────────────────────────────────────────────────────────────────
WHY THIS LIVES IN `domain/`
────────────────────────────────────────────────────────────────────────
`strategies/sandbox.py` lets generated strategy code import exactly
`math`, `statistics`, `numpy`, `pandas`, `src.strategies.domain.models`
and this module. Anything a strategy needs must therefore either be
copy-pasted into every strategy file (which is how the Apex ladder's
timeframe siblings already have to work, and it is a maintenance tax) or
live in a whitelisted pure-domain module. This is that module: no I/O, no
FastAPI, no SQLAlchemy, no adapters, no broker access — only `math`,
`numpy` and `pandas`, per CLAUDE.md's hexagonal rule.

────────────────────────────────────────────────────────────────────────
SHAPE OF THE MEASURE
────────────────────────────────────────────────────────────────────────
Eight independent sub-measures, each a pure function returning a float in
[0, 1] (or NaN when it cannot be computed from the bars it was given).
`trend_fatigue()` folds the ones with a non-zero weight into a weighted
mean and returns a `FatigueReading` carrying the full component
breakdown, so a veto is always explainable in one INFO line rather than
being an opaque number ("money-touching code paths: explicit over
clever", CLAUDE.md).

    momentum_divergence  price prints a new extreme, RSI does not
    impulse_decay        successive same-direction legs shrink (ATR-
                         normalised) while pullbacks get deeper
    range_decay          true range and candle bodies contract while the
                         trend keeps going
    volume_decay         tick_volume falls away as price extends
    wick_rejection       adverse wicks build at the trend extreme
    overextension        ATR-distance from an EMA anchor, one-way bar
                         streak, and slope acceleration (parabolic tell)
    persistence_decay    variance ratio drops toward / below 1: the walk
                         has stopped being persistent
    trend_age            bars since the EMA regime flipped

Every component is bounded well inside the engine's hard 200-bar context
cap (`trade_loop.DEFAULT_CONTEXT_BARS`): the deepest default lookback is
`persistence_decay` at 61 bars and `impulse_decay` at ~120. A measure
that needed more than 200 would silently never fire, live or backtest,
with no error anywhere — so `required_bars()` reports the real number and
`trend_fatigue()` simply drops components it cannot compute.

────────────────────────────────────────────────────────────────────────
UNAVAILABLE IS NOT EXHAUSTED
────────────────────────────────────────────────────────────────────────
A component that lacks bars, sees a flat series, gets a zero/NaN ATR, or
has no volume column returns NaN and is dropped from the weighted mean,
with the remaining weights renormalised. If *nothing* can be computed the
score is 0.0 — "fresh" — deliberately: an unknown must never veto a trade
or licence a fade on its own. `FatigueReading.available` says how many
components actually voted, so a caller that wants to insist on evidence
can require a minimum.

────────────────────────────────────────────────────────────────────────
EVIDENCE, HONESTLY
────────────────────────────────────────────────────────────────────────
The default weights are not equal, because the underlying ideas are not
equally supported:

  * `overextension` and `momentum_divergence` carry the most weight.
    Distance-from-mean in ATR/σ units is the standard, well-studied
    mean-reversion setup, and regular momentum divergence is the
    textbook exhaustion tell with the clearest programmatic definition.
  * `impulse_decay` and `range_decay` are structural and cheap and agree
    with the "expansion = conviction, contraction = absorption" reading
    that the impulse/retracement literature is built on.
  * `volume_decay` is downweighted because `tick_volume` on an FX/CFD
    feed is a tick-count proxy for real traded volume, not real volume.
  * `trend_age` — the bars-since-break / TD-Sequential-style "9 and 13
    count" family — ships at weight 0.0. The one serious published test
    of DeMark indicators (Lissandrin, Daly & Sornette, *Statistical
    Testing of DeMark Technical Indicators on Commodity Futures*, 2015)
    does find statistically significant predictive power on commodity
    futures 2004-2014, but only over a narrow band of holding periods
    and with the effect sensitive to transaction costs — which is a much
    weaker claim than the indicator's popular reputation. Pure elapsed
    time also says nothing about *this* move. It is implemented so the
    A/B can measure it rather than argue about it, and defaults off.

Hidden divergence (higher low in price against a lower low in momentum
during an uptrend) is deliberately *not* folded in: it is a continuation
tell, the mirror of the regular divergence this module scores, so adding
it would mean mixing "trend has fuel" into a fatigue number. It belongs
in a freshness measure, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_PARAMS",
    "DEFAULT_WEIGHTS",
    "FatigueReading",
    "COMPONENT_NAMES",
    "impulse_decay_score",
    "momentum_divergence_score",
    "overextension_score",
    "persistence_decay_score",
    "range_decay_score",
    "required_bars",
    "trend_age_score",
    "trend_fatigue",
    "volume_decay_score",
    "wick_rejection_score",
]

NAN = float("nan")

COMPONENT_NAMES: tuple[str, ...] = (
    "momentum_divergence",
    "impulse_decay",
    "range_decay",
    "volume_decay",
    "wick_rejection",
    "overextension",
    "persistence_decay",
    "trend_age",
)

# Short tags for the one-line audit string that ends up in a Signal.reason
# and in the INFO log — full names would blow the MT5 comment budget and
# make the veto-funnel `example_reason` unreadable.
_TAGS: dict[str, str] = {
    "momentum_divergence": "div",
    "impulse_decay": "imp",
    "range_decay": "rng",
    "volume_decay": "vol",
    "wick_rejection": "wick",
    "overextension": "ext",
    "persistence_decay": "vr",
    "trend_age": "age",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum_divergence": 1.00,
    "impulse_decay": 1.00,
    "range_decay": 0.75,
    "volume_decay": 0.50,
    "wick_rejection": 0.75,
    "overextension": 1.25,
    "persistence_decay": 0.75,
    # See "EVIDENCE, HONESTLY" above — implemented, measured by the A/B,
    # not trusted by default.
    "trend_age": 0.00,
}

DEFAULT_PARAMS: dict[str, float] = {
    # ── momentum_divergence ──
    "divergence_window": 40,  # bars split into two halves to find the two extremes
    "divergence_rsi_period": 14,
    "divergence_full_gap": 10.0,  # RSI points of divergence that score a full 1.0
    # ── impulse_decay ──
    "impulse_pivot_wing": 3,  # fractal wing; 3 => a 7-bar confirmation window
    "impulse_legs": 3,  # same-direction legs compared
    "impulse_full_shrink": 0.60,  # newest leg 60% smaller than the biggest earlier one
    "impulse_full_deepening": 0.35,  # pullback depth growing by 35 pts of the impulse
    # ── range_decay ──
    "range_fast": 10,
    "range_slow": 40,
    "range_full_contraction": 0.45,
    "range_full_body_decay": 0.35,
    # ── volume_decay ──
    "volume_fast": 10,
    "volume_slow": 40,
    "volume_full_decay": 0.40,
    # ── wick_rejection ──
    "wick_window": 10,
    "wick_onset_asymmetry": 0.05,
    "wick_full_asymmetry": 0.35,
    # ── overextension ──
    "extension_ema_span": 21,
    # The absolute stretch is deliberately a *generous* ramp: any real
    # trend sits several ATR above a 21-EMA simply because the EMA lags,
    # so only a genuinely extreme distance should register here.
    "extension_onset_atr": 2.50,
    "extension_full_atr": 6.00,
    "streak_onset_bars": 4,
    "streak_full_bars": 10,
    "accel_window": 8,  # slope of the last N bars vs the N before them
    "accel_onset": 0.50,  # +50% steeper
    "accel_full": 2.00,  # 3x steeper — the parabolic tell
    # The three parts' split inside overextension. Distance-from-anchor
    # carries most of the weight because it is the only one of the three
    # that is scale-free; streak and acceleration are shape tells that
    # confirm it rather than stand on their own.
    "extension_weight": 0.70,
    "streak_weight": 0.15,
    "accel_weight": 0.15,
    # ── persistence_decay ──
    "persistence_window": 60,
    "persistence_q": 5,
    "persistence_full_drop": 0.50,  # variance ratio 0.5 => fully mean-reverting
    # ── trend_age ──
    "age_ema_fast": 21,
    "age_ema_slow": 55,
    "age_onset_bars": 40,
    "age_full_bars": 120,
}


@dataclass(frozen=True)
class FatigueReading:
    """One evaluation of trend fatigue, with the breakdown that produced it.

    `components` holds only the sub-measures that actually voted (weight
    above zero *and* computable from the bars supplied), so it doubles as
    the audit trail: a strategy logs `describe()` at INFO next to its
    entry decision and the number is never unexplained.
    """

    score: float
    """Composite fatigue in [0, 1]. 0 = fresh trend, 1 = exhausted. 0.0
    when nothing could be computed — unavailable is not exhausted."""

    direction: int
    """+1 if the fatigue of an up-move was measured, -1 for a down-move,
    0 when the caller had no trend direction (score is then always 0.0)."""

    components: dict[str, float] = field(default_factory=dict)
    """Sub-measure name → its own [0, 1] score. Only the ones that voted."""

    weights: dict[str, float] = field(default_factory=dict)
    """The renormalised weight each voting component carried, summing to 1."""

    bars: int = 0
    """How many bars the reading was computed from."""

    @property
    def available(self) -> int:
        """How many sub-measures actually voted. Zero means the score is
        the 'no opinion' default, not a measurement."""
        return len(self.components)

    def describe(self) -> str:
        """Compact one-line audit string for logs and `Signal.reason`."""
        if not self.components:
            return f"fatigue={self.score:.2f}(n=0)"
        parts = " ".join(
            f"{_TAGS.get(name, name)}={value:.2f}"
            for name, value in sorted(self.components.items())
        )
        sign = "+" if self.direction > 0 else ("-" if self.direction < 0 else "0")
        return f"fatigue={self.score:.2f}dir{sign}[{parts}]"


# ─────────────────────────────────────────────────────────────────────
# Small numeric helpers
# ─────────────────────────────────────────────────────────────────────

def _ramp(value: float, onset: float, full: float) -> float:
    """Linear 0→1 ramp: 0 at or below `onset`, 1 at or above `full`.

    Every sub-measure ends in one of these so the raw quantity (RSI
    points, ATR multiples, a ratio) is turned into the same [0, 1]
    currency and the composite can average them without one unit
    quietly dominating.
    """
    if not math.isfinite(value) or not math.isfinite(onset) or not math.isfinite(full):
        return NAN
    if full <= onset:
        return 1.0 if value >= full else 0.0
    return float(min(1.0, max(0.0, (value - onset) / (full - onset))))


def _finite(values: np.ndarray) -> bool:
    return bool(values.size) and bool(np.all(np.isfinite(values)))


def _as_array(values) -> np.ndarray:
    """Anything array-like (numpy array, pandas Series, list) → float64.

    Strategies hand us `df["close"].to_numpy()` on the hot path, but the
    tests and the odd caller pass a Series; converting once here keeps
    every sub-measure free of isinstance checks.
    """
    if values is None:
        return np.empty(0, dtype=float)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float, copy=False)
    return np.asarray(values, dtype=float)


def _param(params: dict | None, key: str) -> float:
    if params is not None and key in params:
        return float(params[key])
    return float(DEFAULT_PARAMS[key])


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI, vectorised through pandas' exponential mean.

    `ewm(alpha=1/period)` is Wilder's smoothing exactly; a Python loop
    over bars here would be re-run on every bar of a 100k-bar M1 replay.
    """
    if closes.size < period + 1:
        return np.full(closes.size, NAN)
    delta = np.diff(closes, prepend=closes[0])
    gain = pd.Series(np.where(delta > 0.0, delta, 0.0))
    loss = pd.Series(np.where(delta < 0.0, -delta, 0.0))
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0.0, avg_gain / avg_loss, np.inf)
        rsi = 100.0 - 100.0 / (1.0 + rs)
    # A stretch with no losses at all is RS=inf -> RSI 100; a stretch with
    # neither gains nor losses (a genuinely flat series) is 0/0 and has no
    # meaningful RSI, so it stays NaN rather than being called neutral.
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), NAN, rsi)
    return rsi


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    tr = highs - lows
    if tr.size > 1:
        prev_close = closes[:-1]
        tr[1:] = np.maximum(
            tr[1:],
            np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
        )
    return tr


def _pivots(highs: np.ndarray, lows: np.ndarray, wing: int) -> list[tuple[int, float, int]]:
    """Alternating zigzag pivots as (index, price, +1 for a high / -1 low).

    Fractal detection is one vectorised sliding-window pass; only the
    alternation walk afterwards is a Python loop, and it runs over the
    handful of pivots found, never over bars.
    """
    n = highs.size
    window = 2 * wing + 1
    if n < window:
        return []
    window_max = np.lib.stride_tricks.sliding_window_view(highs, window).max(axis=1)
    window_min = np.lib.stride_tricks.sliding_window_view(lows, window).min(axis=1)
    centre = np.arange(wing, n - wing)
    is_high = highs[centre] == window_max
    is_low = lows[centre] == window_min

    raw: list[tuple[int, float, int]] = []
    for offset in np.flatnonzero(is_high | is_low):
        index = int(centre[offset])
        if is_high[offset]:
            raw.append((index, float(highs[index]), 1))
        if is_low[offset]:
            raw.append((index, float(lows[index]), -1))
    raw.sort(key=lambda pivot: pivot[0])

    # Collapse same-kind runs to their extreme so the sequence alternates
    # high/low/high/... — a leg between two same-kind pivots is not a leg.
    zigzag: list[tuple[int, float, int]] = []
    for pivot in raw:
        if zigzag and zigzag[-1][2] == pivot[2]:
            better = pivot[1] > zigzag[-1][1] if pivot[2] > 0 else pivot[1] < zigzag[-1][1]
            if better:
                zigzag[-1] = pivot
            continue
        zigzag.append(pivot)
    return zigzag


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 1 — momentum divergence
# ─────────────────────────────────────────────────────────────────────

def momentum_divergence_score(
    highs, lows, closes, direction: int, params: dict | None = None
) -> float:
    """Regular divergence: a new price extreme that RSI refuses to confirm.

    The window is split in half; the prior extreme is the best bar of the
    older half, the current extreme the best bar of the newer half. If
    price has *not* made a new extreme there is nothing to diverge from
    and the score is 0 — this measures a failing push, not a pullback.
    Otherwise the score ramps with how many RSI points the momentum
    extreme fell short by.

    Both extremes must fall inside `divergence_window`, which is the one
    real limitation of splitting a fixed window rather than tracking swing
    pivots: a window that lands mid-pullback can compare the top of a
    retracement against the top of a fresh push, and read a genuine
    divergence as 0. A pivot-based version was tried and is worse in
    practice — on anything smoother than live tick data the fractal
    detector finds fewer than two same-kind pivots in 40 bars and the
    measure abstains entirely. The window version's failure mode is a
    false negative (fatigue missed, entry allowed under the strategy's
    other gates); the pivot version's is no reading at all. For a gate
    that vetoes trades, the false negative is the safer way to be wrong.

    Cost: `divergence_window + divergence_rsi_period + 1` bars (54 by
    default).
    """
    if direction not in (1, -1):
        return NAN
    highs = _as_array(highs)
    lows = _as_array(lows)
    closes = _as_array(closes)
    window = int(_param(params, "divergence_window"))
    period = int(_param(params, "divergence_rsi_period"))
    full_gap = _param(params, "divergence_full_gap")
    half = window // 2
    if half < 2 or closes.size < window + period + 1:
        return NAN
    if not (_finite(closes[-window:]) and _finite(highs[-window:]) and _finite(lows[-window:])):
        return NAN

    rsi = _rsi(closes, period)[-window:]
    extreme = highs[-window:] if direction > 0 else lows[-window:]
    if direction > 0:
        prior = int(np.argmax(extreme[:half]))
        recent = half + int(np.argmax(extreme[half:]))
        made_new_extreme = extreme[recent] > extreme[prior]
    else:
        prior = int(np.argmin(extreme[:half]))
        recent = half + int(np.argmin(extreme[half:]))
        made_new_extreme = extreme[recent] < extreme[prior]

    if not made_new_extreme:
        return 0.0
    if not (math.isfinite(rsi[prior]) and math.isfinite(rsi[recent])):
        return NAN
    # Up-move: the fresh high should come with a *higher* RSI. Down-move:
    # the fresh low should come with a *lower* RSI. Either way the gap is
    # "how far short of confirming the momentum extreme fell".
    gap = (rsi[prior] - rsi[recent]) if direction > 0 else (rsi[recent] - rsi[prior])
    return _ramp(float(gap), 0.0, full_gap)


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 2 — impulse / leg decay
# ─────────────────────────────────────────────────────────────────────

def impulse_decay_score(
    highs, lows, direction: int, atr: float, params: dict | None = None
) -> float:
    """Successive same-direction legs shrinking, pullbacks deepening.

    Two halves, averaged:

      * shrink — the newest impulse leg measured against the largest of
        the earlier ones, both in ATR units so the number survives a
        volatility regime change.
      * deepening — each impulse's following pullback as a fraction of
        that impulse; the newest pullback against the oldest. A trend
        that gives back 30% then 40% then 60% of each leg is running out
        of buyers even if the legs themselves stay the same size.

    Cost: enough bars to find `2 * impulse_legs + 1` alternating fractal
    pivots — ~120 bars at the defaults on a normally-swinging series,
    which is why the strategies feed this their zone timeframe rather
    than the entry timeframe.
    """
    if direction not in (1, -1):
        return NAN
    if not math.isfinite(atr) or atr <= 0.0:
        return NAN
    highs = _as_array(highs)
    lows = _as_array(lows)
    if highs.size != lows.size or not (_finite(highs) and _finite(lows)):
        return NAN

    wing = int(_param(params, "impulse_pivot_wing"))
    want_legs = int(_param(params, "impulse_legs"))
    full_shrink = _param(params, "impulse_full_shrink")
    full_deepening = _param(params, "impulse_full_deepening")

    zigzag = _pivots(highs, lows, wing)
    if len(zigzag) < 3:
        return NAN

    # An impulse in `direction` runs from an opposite-kind pivot to a
    # same-kind one: for an up-move, low -> high.
    impulses: list[float] = []
    pullbacks: list[float] = []
    for i in range(len(zigzag) - 1):
        start, end = zigzag[i], zigzag[i + 1]
        travel = end[1] - start[1]
        if travel * direction > 0.0:
            impulses.append(abs(travel) / atr)
            # The pullback that follows this impulse, if it has happened.
            if i + 2 < len(zigzag):
                give_back = abs(zigzag[i + 2][1] - end[1]) / atr
                pullbacks.append(give_back / impulses[-1] if impulses[-1] > 0.0 else NAN)

    impulses = impulses[-want_legs:]
    pullbacks = pullbacks[-want_legs:]
    if len(impulses) < 2:
        return NAN

    earlier = max(impulses[:-1])
    shrink = 1.0 - (impulses[-1] / earlier) if earlier > 0.0 else NAN
    shrink_score = _ramp(shrink, 0.0, full_shrink)

    clean = [depth for depth in pullbacks if math.isfinite(depth)]
    deepening_score = (
        _ramp(clean[-1] - clean[0], 0.0, full_deepening) if len(clean) >= 2 else NAN
    )

    scores = [s for s in (shrink_score, deepening_score) if math.isfinite(s)]
    if not scores:
        return NAN
    return float(sum(scores) / len(scores))


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 3 — range / body decay
# ─────────────────────────────────────────────────────────────────────

def range_decay_score(opens, highs, lows, closes, params: dict | None = None) -> float:
    """Volatility drying up underneath a trend that is still going.

    Two halves, averaged: true range over a fast window against a slow
    one, and the mean body/range ratio over the same two windows. The
    first says the market is covering less ground per bar; the second
    says that what ground it does cover is being fought over rather than
    taken. Direction-free by construction — a contracting range is
    contracting whichever way the trend points.

    Cost: `range_slow` bars (40 by default).
    """
    opens = _as_array(opens)
    highs = _as_array(highs)
    lows = _as_array(lows)
    closes = _as_array(closes)
    fast = int(_param(params, "range_fast"))
    slow = int(_param(params, "range_slow"))
    if fast < 2 or slow <= fast or closes.size < slow:
        return NAN
    if not (_finite(highs[-slow:]) and _finite(lows[-slow:]) and _finite(closes[-slow:])):
        return NAN

    tr = _true_range(highs[-slow:], lows[-slow:], closes[-slow:])
    tr_slow = float(np.mean(tr))
    tr_fast = float(np.mean(tr[-fast:]))
    contraction = 1.0 - (tr_fast / tr_slow) if tr_slow > 0.0 else NAN
    range_part = _ramp(contraction, 0.0, _param(params, "range_full_contraction"))

    spans = highs[-slow:] - lows[-slow:]
    bodies = np.abs(closes[-slow:] - opens[-slow:])
    usable = spans > 0.0
    if int(np.count_nonzero(usable[-fast:])) < 2 or int(np.count_nonzero(usable)) < fast:
        body_part = NAN
    else:
        ratio = np.where(usable, bodies / np.where(usable, spans, 1.0), NAN)
        body_slow = float(np.nanmean(ratio))
        body_fast = float(np.nanmean(ratio[-fast:]))
        decay = 1.0 - (body_fast / body_slow) if body_slow > 0.0 else NAN
        body_part = _ramp(decay, 0.0, _param(params, "range_full_body_decay"))

    scores = [s for s in (range_part, body_part) if math.isfinite(s)]
    if not scores:
        return NAN
    return float(sum(scores) / len(scores))


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 4 — volume decay
# ─────────────────────────────────────────────────────────────────────

def volume_decay_score(volumes, closes, direction: int, params: dict | None = None) -> float:
    """Participation falling away while price keeps extending.

    The "while extending" half matters: volume always drops in a
    pullback, and scoring that as fatigue would fire this on every
    healthy consolidation. So if price has not actually travelled in
    `direction` across the slow window, the score is a flat 0.

    We only have `tick_volume` (a tick-count proxy, not traded volume),
    which is why the default weight for this component is the lowest of
    the set.

    Cost: `volume_slow` bars (40 by default).
    """
    if direction not in (1, -1):
        return NAN
    volumes = _as_array(volumes)
    closes = _as_array(closes)
    fast = int(_param(params, "volume_fast"))
    slow = int(_param(params, "volume_slow"))
    if fast < 2 or slow <= fast or volumes.size < slow or closes.size < slow:
        return NAN
    recent = volumes[-slow:]
    if not _finite(recent) or float(np.max(recent)) <= 0.0:
        return NAN
    if not _finite(closes[-slow:]):
        return NAN

    extending = (closes[-1] - closes[-slow]) * direction > 0.0
    if not extending:
        return 0.0

    slow_mean = float(np.mean(recent))
    fast_mean = float(np.mean(recent[-fast:]))
    if slow_mean <= 0.0:
        return NAN
    return _ramp(1.0 - fast_mean / slow_mean, 0.0, _param(params, "volume_full_decay"))


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 5 — wick / rejection asymmetry
# ─────────────────────────────────────────────────────────────────────

def wick_rejection_score(
    opens, highs, lows, closes, direction: int, params: dict | None = None
) -> float:
    """Adverse wicks building at the trend extreme.

    For an up-move the adverse wick is the upper one — price reached
    higher and was sold back before the close. Scoring the *asymmetry*
    (adverse minus favourable, each as a fraction of the bar's range)
    rather than the adverse wick alone means a market that is simply
    choppy on both sides does not register as exhaustion.

    Cost: `wick_window` bars (10 by default).
    """
    if direction not in (1, -1):
        return NAN
    opens = _as_array(opens)
    highs = _as_array(highs)
    lows = _as_array(lows)
    closes = _as_array(closes)
    window = int(_param(params, "wick_window"))
    if window < 2 or closes.size < window:
        return NAN

    o = opens[-window:]
    h = highs[-window:]
    low = lows[-window:]
    c = closes[-window:]
    if not (_finite(o) and _finite(h) and _finite(low) and _finite(c)):
        return NAN
    spans = h - low
    usable = spans > 0.0
    if int(np.count_nonzero(usable)) < 2:
        return NAN

    body_top = np.maximum(o, c)
    body_bottom = np.minimum(o, c)
    upper = h - body_top
    lower = body_bottom - low
    adverse, favourable = (upper, lower) if direction > 0 else (lower, upper)
    safe = np.where(usable, spans, 1.0)
    asymmetry = float(np.mean(np.where(usable, (adverse - favourable) / safe, np.nan)[usable]))
    return _ramp(
        asymmetry,
        _param(params, "wick_onset_asymmetry"),
        _param(params, "wick_full_asymmetry"),
    )


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 6 — overextension
# ─────────────────────────────────────────────────────────────────────

def overextension_score(
    closes, direction: int, atr: float, params: dict | None = None
) -> float:
    """How stretched the move is, from three angles.

      * extension — signed distance from an EMA anchor in ATR units. The
        Apex family already has a cruder version of exactly this
        (`max_extension_atr`); this one is ramped rather than a cliff, so
        it contributes proportionally instead of all-or-nothing.
      * streak — trailing run of closes that all moved in `direction`.
        One-way bars are how a blow-off actually looks bar by bar.
      * acceleration — slope of the last `accel_window` closes against
        the slope of the `accel_window` before them. A trend getting
        steeper is the parabolic tell; a linear trend scores 0 here.

    Cost: `max(extension_ema_span * 3, accel_window * 2, streak_full_bars)`
    bars — ~63 at the defaults.
    """
    if direction not in (1, -1):
        return NAN
    if not math.isfinite(atr) or atr <= 0.0:
        return NAN
    closes = _as_array(closes)
    span = int(_param(params, "extension_ema_span"))
    accel_window = int(_param(params, "accel_window"))
    streak_full = int(_param(params, "streak_full_bars"))
    needed = max(span * 3, accel_window * 2 + 1, streak_full + 1)
    if closes.size < needed or not _finite(closes[-needed:]):
        return NAN

    ema = pd.Series(closes).ewm(span=span, adjust=False).mean().to_numpy()
    extension = (closes[-1] - ema[-1]) * direction / atr
    extension_part = _ramp(
        float(extension),
        _param(params, "extension_onset_atr"),
        _param(params, "extension_full_atr"),
    )

    steps = np.diff(closes[-(streak_full + 1):]) * direction
    # Length of the trailing run of strictly favourable closes.
    against = np.flatnonzero(steps <= 0.0)
    streak = steps.size if against.size == 0 else steps.size - 1 - int(against[-1])
    streak_part = _ramp(
        float(streak),
        _param(params, "streak_onset_bars"),
        float(streak_full),
    )

    recent_slope = (closes[-1] - closes[-1 - accel_window]) * direction / accel_window
    prior_slope = (
        closes[-1 - accel_window] - closes[-1 - 2 * accel_window]
    ) * direction / accel_window
    if prior_slope > 0.0:
        accel_part = _ramp(
            float(recent_slope / prior_slope - 1.0),
            _param(params, "accel_onset"),
            _param(params, "accel_full"),
        )
    else:
        # The move only just started going this way; "accelerating from
        # nothing" is freshness, not fatigue.
        accel_part = NAN

    pairs = [
        (value, weight)
        for value, weight in (
            (extension_part, _param(params, "extension_weight")),
            (streak_part, _param(params, "streak_weight")),
            (accel_part, _param(params, "accel_weight")),
        )
        if math.isfinite(value) and weight > 0.0
    ]
    if not pairs:
        return NAN
    total = sum(weight for _, weight in pairs)
    return float(sum(value * weight for value, weight in pairs) / total)


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 7 — persistence decay (variance ratio)
# ─────────────────────────────────────────────────────────────────────

def persistence_decay_score(closes, params: dict | None = None) -> float:
    """Variance ratio: has the walk stopped being persistent?

    VR(q) = Var(q-bar returns) / (q * Var(1-bar returns)). A pure random
    walk sits at 1; a trending (positively autocorrelated) series sits
    above it; a mean-reverting one below. Fatigue ramps as VR falls below
    1 — the same regime read the Hurst-exponent literature uses, computed
    the cheap way, with no rescaled-range fitting on the hot path.

    Direction-free: persistence is a property of the series, not of which
    way it happens to be pointing.

    Cost: `persistence_window + 1` bars (61 by default).
    """
    closes = _as_array(closes)
    window = int(_param(params, "persistence_window"))
    q = int(_param(params, "persistence_q"))
    if q < 2 or window < q * 4 or closes.size < window + 1:
        return NAN
    segment = closes[-(window + 1):]
    if not _finite(segment) or float(np.min(segment)) <= 0.0:
        return NAN

    returns = np.diff(np.log(segment))
    single = float(np.var(returns, ddof=1))
    if single <= 0.0:
        return NAN
    usable = (returns.size // q) * q
    if usable < q * 2:
        return NAN
    aggregated = returns[-usable:].reshape(-1, q).sum(axis=1)
    if aggregated.size < 2:
        return NAN
    multi = float(np.var(aggregated, ddof=1))
    variance_ratio = multi / (q * single)
    if not math.isfinite(variance_ratio):
        return NAN
    return _ramp(1.0 - variance_ratio, 0.0, _param(params, "persistence_full_drop"))


# ─────────────────────────────────────────────────────────────────────
# Sub-measure 8 — trend age
# ─────────────────────────────────────────────────────────────────────

def trend_age_score(closes, direction: int, params: dict | None = None) -> float:
    """Bars since the EMA regime last agreed with `direction`.

    The time-based family (bars-since-structure-break, TD-Sequential
    9/13 counting). Ships at weight 0.0 — see the module docstring for
    why the evidence does not justify trusting elapsed time on its own.

    Cost: `age_ema_slow * 3` bars (165 by default) — the deepest lookback
    in the module and still inside the engine's 200-bar cap, but only
    just, which is another reason it is off by default.
    """
    if direction not in (1, -1):
        return NAN
    closes = _as_array(closes)
    fast = int(_param(params, "age_ema_fast"))
    slow = int(_param(params, "age_ema_slow"))
    needed = slow * 3
    if closes.size < needed or not _finite(closes[-needed:]):
        return NAN
    series = pd.Series(closes)
    spread = (
        series.ewm(span=fast, adjust=False).mean()
        - series.ewm(span=slow, adjust=False).mean()
    ).to_numpy()
    agreeing = (spread * direction) > 0.0
    if not bool(agreeing[-1]):
        return 0.0
    disagreeing = np.flatnonzero(~agreeing)
    age = agreeing.size if disagreeing.size == 0 else agreeing.size - 1 - int(disagreeing[-1])
    return _ramp(
        float(age),
        _param(params, "age_onset_bars"),
        _param(params, "age_full_bars"),
    )


# ─────────────────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────────────────

def required_bars(params: dict | None = None, weights: dict | None = None) -> int:
    """Bars needed for every enabled component to be able to vote.

    A caller that hands over fewer still gets a reading — the components
    that cannot be computed simply abstain — but this is the number to
    compare against the engine's 200-bar context cap when choosing which
    timeframe to measure on.
    """
    active = _resolve_weights(weights)
    needs = {
        "momentum_divergence": int(_param(params, "divergence_window"))
        + int(_param(params, "divergence_rsi_period"))
        + 1,
        # A leg needs a confirmed pivot at each end, and pivots are
        # confirmed `wing` bars late; 2 legs + their pullbacks is ~8
        # alternating pivots, which on a normal series is this many bars.
        "impulse_decay": int(_param(params, "impulse_pivot_wing")) * 2
        * (2 * int(_param(params, "impulse_legs")) + 2)
        + 20,
        "range_decay": int(_param(params, "range_slow")),
        "volume_decay": int(_param(params, "volume_slow")),
        "wick_rejection": int(_param(params, "wick_window")),
        "overextension": max(
            int(_param(params, "extension_ema_span")) * 3,
            int(_param(params, "accel_window")) * 2 + 1,
            int(_param(params, "streak_full_bars")) + 1,
        ),
        "persistence_decay": int(_param(params, "persistence_window")) + 1,
        "trend_age": int(_param(params, "age_ema_slow")) * 3,
    }
    enabled = [needs[name] for name, weight in active.items() if weight > 0.0]
    return max(enabled) if enabled else 0


def _resolve_weights(weights: dict | None) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    resolved = {}
    for name in COMPONENT_NAMES:
        resolved[name] = float(weights.get(name, DEFAULT_WEIGHTS[name]))
    return resolved


def trend_fatigue(
    *,
    opens=None,
    highs,
    lows,
    closes,
    volumes=None,
    direction: int,
    atr: float | None = None,
    params: dict | None = None,
    weights: dict | None = None,
    min_components: int = 3,
) -> FatigueReading:
    """Composite trend fatigue in [0, 1] for a move running in `direction`.

    `direction` is the direction of the *trend being measured*, not of
    the trade being considered: +1 for an up-move, -1 for a down-move.
    A continuation entry wants a low score; a fade wants a high one.

    `atr` gates the two components that need a volatility unit
    (`impulse_decay`, `overextension`); pass the same ATR the caller
    already computed for its stops so the numbers agree. Omit it and
    those two abstain rather than guess.

    Components with weight 0, or that cannot be computed from the bars
    supplied, abstain and the remaining weights are renormalised. With
    nothing left the score is 0.0 — fresh — because an unknown must not
    veto a trade on its own.

    `min_components` is the same principle one step further: a score
    assembled from one or two surviving sub-measures is not a consensus,
    it is whichever measure happened to have a short enough lookback
    (`wick_rejection` needs 10 bars; the rest need 40-120). Below the
    floor the reading abstains outright rather than letting that one
    measure speak for the whole composite. Pass 1 to opt out.
    """
    highs = _as_array(highs)
    lows = _as_array(lows)
    closes = _as_array(closes)
    opens = closes if opens is None else _as_array(opens)
    active = _resolve_weights(weights)
    bars = int(closes.size)

    if direction not in (1, -1) or bars == 0:
        return FatigueReading(score=0.0, direction=0, bars=bars)

    atr_value = NAN if atr is None else float(atr)
    raw: dict[str, float] = {}
    for name, weight in active.items():
        if weight <= 0.0:
            continue
        if name == "momentum_divergence":
            value = momentum_divergence_score(highs, lows, closes, direction, params)
        elif name == "impulse_decay":
            value = impulse_decay_score(highs, lows, direction, atr_value, params)
        elif name == "range_decay":
            value = range_decay_score(opens, highs, lows, closes, params)
        elif name == "volume_decay":
            value = volume_decay_score(volumes, closes, direction, params)
        elif name == "wick_rejection":
            value = wick_rejection_score(opens, highs, lows, closes, direction, params)
        elif name == "overextension":
            value = overextension_score(closes, direction, atr_value, params)
        elif name == "persistence_decay":
            value = persistence_decay_score(closes, params)
        else:
            value = trend_age_score(closes, direction, params)
        if math.isfinite(value):
            raw[name] = float(min(1.0, max(0.0, value)))

    enabled_count = sum(1 for weight in active.values() if weight > 0.0)
    if len(raw) < min(min_components, enabled_count):
        return FatigueReading(score=0.0, direction=direction, bars=bars)

    total = sum(active[name] for name in raw)
    normalised = {name: active[name] / total for name in raw}
    score = sum(raw[name] * normalised[name] for name in raw)
    return FatigueReading(
        score=float(min(1.0, max(0.0, score))),
        direction=direction,
        components=raw,
        weights=normalised,
        bars=bars,
    )
