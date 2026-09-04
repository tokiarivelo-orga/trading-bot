"""XAUUSD adaptive S&D / Quasimodo / SNRC — M1, self-learning.

A superset of `xauusd_snd_qm_structure_fixed_m1` that (a) detects every entry
family in the PoB doctrine rather than three of them, and (b) learns online
which of them are working *right now*, in *this* volatility and trend regime,
and adjusts its own stop buffer and take-profit ladder to match.

The baseline bot is left untouched so the two can be A/B'd on the same period.

────────────────────────────────────────────────────────────────────────
ENTRY FAMILIES
────────────────────────────────────────────────────────────────────────
Every detector below produces the same shape of thing — a rectangle with a
polarity (demand = buy, supply = sell), the bar it became valid on, and a
family label. The label matters: it is part of the learner's bucket key, so
each family accumulates its own expectancy and a family that stops working
gets gated out without touching the others.

  SND_V1     classic leg-base-leg: RBR, DBR (demand) / DBD, RBD (supply).
  SND_V2     engulfing-base departure block.
  QM_LEGACY  the baseline bot's generic swing-failure Quasimodo. Kept —
             under its own label — so the A/B is apples-to-apples.
  QMR / QML  the doctrinal Quasimodo: shoulder → neckline → higher head →
             close back through the neckline; trade the retest of the
             *left shoulder*. Reversal-only, hence high risk.
  QMC        the same structure occurring *with* the prevailing trend
             (continuation) and confirmed by a trendline touch.
  QM2P       QMR plus a two-point trendline anchored at the head.
  QMM        a QMR that *failed*: price closed through the left-shoulder
             line, which flips that line's polarity. Entered only with a
             higher-timeframe engulfing candle, per doctrine.
  SNRC1      an RBR/DBD (continuation) base sitting on a strong S/R level,
             confirmed by an HTF engulfing candle.
  SNRC2      the same, for DBR/RBD (reversal) bases.
  SNRC3      the "engulfing war": a failed bullish engulfing at strong
             support becomes a sell on the next bearish engulfing at that
             line (and the mirror image for buys).
  SWEEP      liquidity sweep / stop hunt: price wicks through a prior swing
             extreme and closes back inside it.
  BREAKER    a zone that got broken. Instead of being discarded it flips
             polarity and is traded on the retest from the other side.

SNRC1/2/3 are *upgrades* of a zone that already exists, not extra rectangles
— a base that qualifies is re-labelled to the most specific family it
matches (SNRC3 > SNRC2 > SNRC1 > raw pattern) so one setup is never counted
twice.

────────────────────────────────────────────────────────────────────────
WHAT THE LEARNER DOES, AND WHAT IT IS NOT ALLOWED TO DO
────────────────────────────────────────────────────────────────────────
`src.strategies.domain.online_learning` holds the machinery; see its
docstring for why the label is "did price secure +0.2R before the stop"
rather than "did TP come before SL", and for the no-lookahead argument.
Here is what this strategy wires into it:

  * Every candidate touch is recorded once per touch episode — **including
    ones the gate declines**. Labels come from price, not from fills, so
    declining a setup costs no information. Without that, the gate would
    only ever see the setups it already likes and could never learn that a
    family has recovered.
  * The gate: skip when the learner has enough evidence *and* puts the
    setup's secure-probability below break-even plus a margin. Break-even is
    0.833 at the engine's 0.2R trailing buffer, so this is a demanding test
    by construction, and it is aimed at the losing tail — the full stops —
    which is the part of this engine's P&L that entry selection can actually
    move.
  * The stop buffer: widened toward a high quantile of the adverse
    excursion that *winning* setups in this bucket actually survived.
  * The TP ladder: TP2/TP3 pulled toward quantiles of realised favourable
    excursion for this bucket, instead of fixed R multiples.

Everything learned is clamped into a band around the base parameter
(`AdaptiveLearner.blend`), and every accessor returns the base value until
its own evidence threshold is met. A cold instance is therefore exactly the
un-adapted strategy: the failure mode is "no adaptation", never "confident
nonsense from four samples". `learner_enabled=False` turns the whole thing
off for a controlled A/B.

Note that generated strategies cannot import `logging` (sandbox allowlist),
so a *declined* setup is silent, exactly as `return None` is for every other
strategy here. Signals that are emitted carry the learner's reasoning as
`IndicatorReading` annotations, which is what surfaces in the journal and on
the chart.

v1 — initial implementation.
"""

import numpy as np
import pandas as pd

from src.strategies.domain.models import (
    Direction,
    IndicatorReading,
    MarketContext,
    PriceZone,
    Signal,
    StrategySpec,
    StructureLabel,
    StructurePoint,
    ZoneKind,
)
from src.strategies.domain.online_learning import AdaptiveLearner, LearnerConfig

_NS_PER_MINUTE = 60 * 1_000_000_000
# A forward jump larger than this is a different replay, not a market gap
# (the longest real XAUUSD close is a ~2.5-day weekend, plus holidays).
_RESET_GAP_NS = 30 * 24 * 60 * _NS_PER_MINUTE


# ─────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────


def _true_range(df):
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df, period):
    return _true_range(df).rolling(period, min_periods=period).mean()


def _ema(values, period):
    return pd.Series(values).ewm(span=period, adjust=False).mean().to_numpy()


def _adx(highs, lows, closes, period):
    """Average Directional Index with simple rolling-mean smoothing — the
    same simplification `engine.domain.regime._adx` makes, kept identical so
    a regime tag computed here means the same thing as one computed there."""
    n = len(closes)
    if n < period + 2:
        return np.full(n, np.nan)
    tr = pd.concat(
        [
            pd.Series(highs) - pd.Series(lows),
            (pd.Series(highs) - pd.Series(closes).shift(1)).abs(),
            (pd.Series(lows) - pd.Series(closes).shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_values = tr.rolling(period, min_periods=period).mean().to_numpy()
    up_move = np.diff(highs)
    down_move = -np.diff(lows)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate(([np.nan], plus_dm))
    minus_dm = np.concatenate(([np.nan], minus_dm))
    smooth_plus = pd.Series(plus_dm).rolling(period, min_periods=period).mean().to_numpy()
    smooth_minus = pd.Series(minus_dm).rolling(period, min_periods=period).mean().to_numpy()
    safe_atr = np.where((atr_values > 0) & np.isfinite(atr_values), atr_values, np.nan)
    plus_di = 100.0 * smooth_plus / safe_atr
    minus_di = 100.0 * smooth_minus / safe_atr
    denominator = plus_di + minus_di
    denominator = np.where(denominator > 0, denominator, np.nan)
    dx = 100.0 * np.abs(plus_di - minus_di) / denominator
    return pd.Series(dx).rolling(period, min_periods=period).mean().to_numpy()


def _engulfing_flags(opens, closes):
    """Bullish/bearish engulfing masks. `RBR, DBD, RBD, DBR are actually
    Engulfing` (PoB) — the leg-out candle engulfing the base is the same
    event these masks find, which is why confirmation reuses them."""
    n = len(closes)
    if n < 2:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    prev_open = np.concatenate(([np.nan], opens[:-1]))
    prev_close = np.concatenate(([np.nan], closes[:-1]))
    up = closes > opens
    prev_down = prev_close < prev_open
    prev_up = prev_close > prev_open
    # The leading NaN makes every comparison on bar 0 False, so bar 0 can
    # never be an engulfing — which is correct, it has nothing to engulf.
    bull = up & prev_down & (closes >= prev_open) & (opens <= prev_close)
    bear = (~up) & prev_up & (closes <= prev_open) & (opens >= prev_close)
    return bull, bear


# ─────────────────────────────────────────────────────────────────────
# Resampling: bucket entry-TF candles into zone-TF bars
# ─────────────────────────────────────────────────────────────────────


def _resample(df, tf_minutes, entry_tf_minutes):
    """Bucket entry-TF rows into tf_minutes OHLC bars (numpy reduceat).
    Returns (zone_frame, zone_end_ns) or None. The trailing bucket is dropped
    unless the entry frame proves it is complete, so a zone can never be
    detected off a half-formed bar."""
    if "time" not in df.columns:
        return None
    t_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
    step = np.int64(tf_minutes) * _NS_PER_MINUTE
    entry_ns = np.int64(entry_tf_minutes) * _NS_PER_MINUTE
    bucket = t_ns // step
    starts = np.flatnonzero(np.concatenate(([True], bucket[1:] != bucket[:-1])))
    if len(starts) < 2:
        return None
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    ends = np.concatenate((starts[1:], [len(t_ns)])) - 1
    frame = pd.DataFrame(
        {
            "time": df["time"].iloc[starts].reset_index(drop=True),
            "open": opens[starts],
            "high": np.maximum.reduceat(highs, starts),
            "low": np.minimum.reduceat(lows, starts),
            "close": closes[ends],
        }
    )
    end_times = (bucket[starts] + 1) * step
    if t_ns[ends[-1]] + entry_ns < end_times[-1]:
        frame = frame.iloc[:-1]
        end_times = end_times[:-1]
    if len(frame) < 2:
        return None
    return frame, end_times


# ─────────────────────────────────────────────────────────────────────
# Supply & Demand V1: classic leg-base-leg (RBR/DBD/RBD/DBR)
# ─────────────────────────────────────────────────────────────────────


def _classify_bars(closes, opens, atr_filled, base_mult):
    body = np.abs(closes - opens)
    return np.where(body <= base_mult * atr_filled, 0, np.where(closes >= opens, 1, -1))


def _build_runs(classes):
    runs = []
    for i in range(len(classes)):
        cls = int(classes[i])
        if runs and runs[-1][0] == cls:
            runs[-1][2] = i
        else:
            runs.append([cls, i, i])
    return runs


def _make_is_leg(closes, opens, atr_filled, leg_mult):
    def is_leg(run):
        cls, start, end = run
        return cls != 0 and abs(closes[end] - opens[start]) >= leg_mult * atr_filled[end]

    return is_leg


def _merge_weak_runs(runs, is_leg, max_base):
    merged = True
    while merged:
        merged = False
        for k in range(len(runs) - 2):
            d1, pause, d2 = runs[k], runs[k + 1], runs[k + 2]
            if d1[0] == 0 or pause[0] != 0 or d2[0] != d1[0]:
                continue
            if pause[2] - pause[1] + 1 > max_base:
                continue
            if is_leg(d1) and is_leg(d2):
                continue
            runs[k : k + 3] = [[d1[0], d1[1], d2[2]]]
            merged = True
            break
    return runs


def _atr_filled(atr_series):
    valid = atr_series.dropna()
    if valid.empty:
        return None
    return atr_series.fillna(valid.iloc[0]).to_numpy()


def _detect_zones_v1(df, atr_series, params):
    """Supply & Demand V1: classic leg-base-leg geometry (RBR/DBD/RBD/DBR)."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    base_mult = params["base_body_atr_mult"]
    leg_mult = params["leg_travel_atr_mult"]
    max_base = int(params["max_base_candles"])

    classes = _classify_bars(closes, opens, atr_filled, base_mult)
    is_leg = _make_is_leg(closes, opens, atr_filled, leg_mult)
    runs = _merge_weak_runs([list(r) for r in _build_runs(classes)], is_leg, max_base)

    legs = [r for r in runs if is_leg(r)]
    zones = []
    for k in range(len(legs) - 1):
        leg_in, leg_out = legs[k], legs[k + 1]
        base_start = leg_in[2] + 1
        base_end = leg_out[1] - 1
        base_count = base_end - base_start + 1
        if base_count < 1 or base_count > max_base:
            continue
        price_high = float(highs[base_start : base_end + 1].max())
        price_low = float(lows[base_start : base_end + 1].min())
        leg_out_up = leg_out[0] == 1
        conf_idx = None
        for j in range(leg_out[1], leg_out[2] + 1):
            cleared = (closes[j] > price_high) if leg_out_up else (closes[j] < price_low)
            if cleared:
                conf_idx = j
                break
        if conf_idx is None:
            continue
        if leg_in[0] == 1:
            pattern = "RBR" if leg_out_up else "RBD"
        else:
            pattern = "DBR" if leg_out_up else "DBD"
        impulse = abs(closes[leg_out[2]] - opens[leg_out[1]])
        zones.append(
            {
                "source": "SND_V1",
                "family": "SND_V1_" + pattern,
                "pattern": pattern,
                "kind": ZoneKind.DEMAND if leg_out_up else ZoneKind.SUPPLY,
                "price_high": price_high,
                "price_low": price_low,
                "base_start": base_start,
                "conf_idx": conf_idx,
                "leg_out_end": leg_out[2],
                "impulse_atr": float(impulse / max(atr_filled[leg_out[2]], 1e-9)),
            }
        )
    return zones


# ─────────────────────────────────────────────────────────────────────
# Supply & Demand V2: engulfing-base departure zones
# ─────────────────────────────────────────────────────────────────────


def _detect_zones_v2(df, atr_series, params):
    """A strong engulfing candle creates the zone edge, small-body candles
    form the base, then a departure candle closes clear of the base in the
    same direction."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)

    base_mult = params["base_body_atr_mult"]
    max_base = int(params.get("v2_max_base_candles", params["max_base_candles"]))
    min_engulf_atr = params.get("v2_engulf_min_atr_mult", 1.0)

    zones = []
    i = 1
    while i < n - 2:
        body = abs(closes[i] - opens[i])
        if body < min_engulf_atr * atr_filled[i]:
            i += 1
            continue
        bullish = closes[i] > opens[i]

        base_start = i + 1
        base_end = base_start
        while base_end < n - 1 and (base_end - base_start) < max_base:
            if abs(closes[base_end] - opens[base_end]) > base_mult * atr_filled[base_end]:
                break
            base_end += 1

        if base_end - base_start < 1:
            i += 1
            continue

        dep_idx = base_end
        if dep_idx >= n:
            break

        price_high = float(max(highs[i], highs[base_start:base_end].max()))
        price_low = float(min(lows[i], lows[base_start:base_end].min()))
        impulse = float(body / max(atr_filled[i], 1e-9))

        if bullish and closes[dep_idx] > price_high:
            zones.append(
                {
                    "source": "SND_V2",
                    "family": "SND_V2",
                    "pattern": "DZ_V2",
                    "kind": ZoneKind.DEMAND,
                    "price_high": price_high,
                    "price_low": price_low,
                    "base_start": base_start,
                    "conf_idx": dep_idx,
                    "leg_out_end": dep_idx,
                    "impulse_atr": impulse,
                }
            )
        elif (not bullish) and closes[dep_idx] < price_low:
            zones.append(
                {
                    "source": "SND_V2",
                    "family": "SND_V2",
                    "pattern": "SZ_V2",
                    "kind": ZoneKind.SUPPLY,
                    "price_high": price_high,
                    "price_low": price_low,
                    "base_start": base_start,
                    "conf_idx": dep_idx,
                    "leg_out_end": dep_idx,
                    "impulse_atr": impulse,
                }
            )
        i = dep_idx + 1

    return zones


# ─────────────────────────────────────────────────────────────────────
# Swing points & market structure (HH / HL / LH / LL)
# ─────────────────────────────────────────────────────────────────────


def _detect_swing_points(highs, lows, lookback):
    """Swing highs/lows by strict N-bar dominance. Strict (a unique extreme
    in the window) so a flat shelf does not print a swing on every bar."""
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] == window_highs.max() and sum(window_highs == highs[i]) == 1:
            swings.append((i, float(highs[i]), "high"))
        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] == window_lows.min() and sum(window_lows == lows[i]) == 1:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda x: x[0])
    return swings


def _detect_structure(swings):
    """Label swing points HH/HL/LH/LL relative to the previous swing of the
    same type."""
    structure = []
    last_high = None
    last_low = None
    for idx, price, typ in swings:
        if typ == "high":
            higher = last_high is None or price > last_high
            label = StructureLabel.HH if higher else StructureLabel.LH
            last_high = price
        else:
            lower = last_low is not None and price < last_low
            label = StructureLabel.LL if lower else StructureLabel.HL
            last_low = price
        structure.append({"index": idx, "price": price, "label": label})
    return structure


def _structure_bias(structure):
    """+1 when the last two labelled swings read HH/HL, -1 for LL/LH, 0 when
    they disagree — the cheapest honest read of "which way is structure
    pointing"."""
    if len(structure) < 2:
        return 0
    labels = [s["label"] for s in structure[-2:]]
    bullish = sum(1 for a in labels if a in (StructureLabel.HH, StructureLabel.HL))
    bearish = sum(1 for a in labels if a in (StructureLabel.LL, StructureLabel.LH))
    if bullish == 2:
        return 1
    if bearish == 2:
        return -1
    return 0


def _band_around(idx, highs, lows, ext, atr_at, max_atr_mult):
    """A zone rectangle centred on one bar, widened by its neighbours and
    capped so a single wide bar cannot produce an untradeably tall zone."""
    start = max(0, idx - ext)
    end = min(len(highs), idx + ext + 1)
    high = float(highs[start:end].max())
    low = float(lows[start:end].min())
    max_height = max_atr_mult * atr_at
    if high - low > max_height:
        mid = (high + low) / 2.0
        high = mid + max_height / 2.0
        low = mid - max_height / 2.0
    return low, high


# ─────────────────────────────────────────────────────────────────────
# Strong support / resistance levels (SnR)
# ─────────────────────────────────────────────────────────────────────


def _strong_levels(swings, atr_val, tol_mult, min_touches):
    """Cluster swing extremes that sit within `tol_mult` x ATR of each other;
    a cluster of `min_touches` or more is a *strong* level. This is the
    "strong Support/Resistance" every SNRC setup is built on."""
    if not swings:
        return []
    tol = max(atr_val * tol_mult, 1e-9)
    prices = sorted(s[1] for s in swings)
    clusters = [[prices[0]]]
    for price in prices[1:]:
        if price - clusters[-1][-1] <= tol:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= min_touches]


def _at_strong_level(levels, price_low, price_high, tol):
    """Does a zone's band overlap a strong level (within tolerance)?"""
    return any(price_low - tol <= level <= price_high + tol for level, _strength in levels)


# ─────────────────────────────────────────────────────────────────────
# Two-point trendline confluence
# ─────────────────────────────────────────────────────────────────────


def _trendline_touch(swings, kind, target_idx, price_low, price_high, tol, max_pairs):
    """True when a line drawn through two recent same-kind swings passes
    through a zone's band at that zone's bar.

    This is the trendline half of CK1 confluence and what separates QM2P/QMC
    from a plain QMR. Only the most recent `max_pairs` swing pairs are tried
    — an old line that happens to intersect is coincidence, not structure."""
    points = [(i, p) for i, p, t in swings if t == kind]
    if len(points) < 2:
        return False
    checked = 0
    for a in range(len(points) - 1, 0, -1):
        for b in range(a - 1, -1, -1):
            i2, p2 = points[a]
            i1, p1 = points[b]
            if i2 <= i1 or target_idx <= i2:
                continue
            slope = (p2 - p1) / float(i2 - i1)
            projected = p2 + slope * (target_idx - i2)
            if price_low - tol <= projected <= price_high + tol:
                return True
            checked += 1
            if checked >= max_pairs:
                return False
    return False


# ─────────────────────────────────────────────────────────────────────
# Quasimodo family: QMR / QML, QMC, QM2P, QMM
# ─────────────────────────────────────────────────────────────────────


def _detect_quasimodo_legacy(df, atr_series, params):
    """The baseline bot's swing-failure Quasimodo, kept verbatim under its
    own family label so an A/B against `xauusd_snd_qm_structure_fixed_m1`
    compares like with like rather than silently dropping one of its three
    signal sources."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    lookback = int(params.get("qm_swing_lookback", 3))
    min_swing_atr = params.get("qm_min_swing_atr", 0.5)
    max_height_mult = params.get("qm_zone_max_atr_mult", 2.0)
    swings = _detect_swing_points(highs, lows, lookback)
    if len(swings) < 4:
        return []

    ext = max(1, lookback // 2)
    zones = []
    for idx in range(len(swings) - 3):
        s1, s2, s3, s4 = swings[idx], swings[idx + 1], swings[idx + 2], swings[idx + 3]
        atr_at = atr_filled[min(s3[0], len(atr_filled) - 1)]

        bullish = (
            s1[2] == "low"
            and s2[2] == "high"
            and s3[2] == "low"
            and s4[2] == "high"
            and s3[1] < s1[1]
            and s4[1] < s2[1]
            and abs(s2[1] - s3[1]) >= min_swing_atr * atr_at
        )
        bearish = (
            s1[2] == "high"
            and s2[2] == "low"
            and s3[2] == "high"
            and s4[2] == "low"
            and s3[1] > s1[1]
            and s4[1] > s2[1]
            and abs(s3[1] - s2[1]) >= min_swing_atr * atr_at
        )
        if not (bullish or bearish):
            continue
        low, high = _band_around(s3[0], highs, lows, ext, atr_at, max_height_mult)
        zones.append(
            {
                "source": "QUASIMODO",
                "family": "QM_LEGACY",
                "pattern": "QM_BULL" if bullish else "QM_BEAR",
                "kind": ZoneKind.DEMAND if bullish else ZoneKind.SUPPLY,
                "price_high": high,
                "price_low": low,
                "base_start": s3[0],
                "conf_idx": s4[0],
                "leg_out_end": s4[0],
                "impulse_atr": float(abs(s3[1] - s2[1]) / max(atr_at, 1e-9)),
            }
        )
    return zones


def _detect_quasimodo_family(df, atr_series, params, swings, trend_dir):
    """Doctrinal Quasimodo, plus its QMC / QM2P / QMM variants.

    Bearish QMR: left shoulder (high) → neckline (low) → head (higher high)
    → close back through the neckline. The setup is the *retest of the left
    shoulder*, which is where the imbalance left by the head sits. Bullish
    QML is the mirror image.

    Variant tagging is by specificity, most specific first, because a setup
    must land in exactly one learner bucket:

        QM2P  trendline anchored at the head passes through the shoulder
        QMC   the structure runs *with* the prevailing trend (continuation)
        QMR   / QML — the plain reversal case

    QMM is separate: it is the *failure* of a QMR. Once price closes through
    the left-shoulder line, that line flips polarity, and doctrine allows the
    trade only with a higher-timeframe engulfing candle — enforced by the
    caller via `needs_htf`."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None or len(swings) < 4:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)

    lookback = int(params.get("qm_swing_lookback", 3))
    ext = max(1, lookback // 2)
    max_height_mult = params.get("qm_zone_max_atr_mult", 2.0)
    min_head_atr = params.get("qm_min_head_atr_mult", 0.4)
    tl_tol_mult = params.get("trendline_tol_atr_mult", 0.35)
    max_pairs = int(params.get("trendline_max_pairs", 6))

    zones = []
    for idx in range(len(swings) - 3):
        shoulder, neck, head, breaker = (
            swings[idx],
            swings[idx + 1],
            swings[idx + 2],
            swings[idx + 3],
        )
        atr_at = atr_filled[min(head[0], len(atr_filled) - 1)]
        if atr_at <= 0:
            continue

        bearish = (
            shoulder[2] == "high"
            and neck[2] == "low"
            and head[2] == "high"
            and breaker[2] == "low"
            and head[1] > shoulder[1]  # head above the left shoulder
            and breaker[1] < neck[1]  # neckline broken
            and (head[1] - neck[1]) >= min_head_atr * atr_at
        )
        bullish = (
            shoulder[2] == "low"
            and neck[2] == "high"
            and head[2] == "low"
            and breaker[2] == "high"
            and head[1] < shoulder[1]
            and breaker[1] > neck[1]
            and (neck[1] - head[1]) >= min_head_atr * atr_at
        )
        if not (bearish or bullish):
            continue

        low, high = _band_around(shoulder[0], highs, lows, ext, atr_at, max_height_mult)
        kind = ZoneKind.SUPPLY if bearish else ZoneKind.DEMAND
        trade_dir = -1 if bearish else 1

        tl_tol = atr_at * tl_tol_mult
        head_line = _trendline_touch(
            swings, "high" if bearish else "low", head[0], low, high, tl_tol, max_pairs
        )
        continuation = trend_dir == trade_dir
        if head_line:
            family = "QM2P"
        elif continuation:
            family = "QMC"
        else:
            family = "QMR" if bearish else "QML"

        zones.append(
            {
                "source": "QUASIMODO",
                "family": family,
                "pattern": family,
                "kind": kind,
                "price_high": high,
                "price_low": low,
                "base_start": shoulder[0],
                "conf_idx": breaker[0],
                "leg_out_end": breaker[0],
                "impulse_atr": float(abs(head[1] - neck[1]) / max(atr_at, 1e-9)),
                "needs_htf": False,
            }
        )

        # ── QMM: the same structure once the shoulder line has failed ──
        # A close beyond the shoulder band in the direction the QMR said it
        # would not go invalidates the QMR and flips that line's polarity.
        fail_from = breaker[0] + 1
        if fail_from >= n:
            continue
        if bearish:
            failed = np.flatnonzero(closes[fail_from:] > high)
        else:
            failed = np.flatnonzero(closes[fail_from:] < low)
        if len(failed) == 0:
            continue
        fail_idx = int(failed[0]) + fail_from
        zones.append(
            {
                "source": "QUASIMODO",
                "family": "QMM",
                "pattern": "QMM",
                "kind": ZoneKind.DEMAND if bearish else ZoneKind.SUPPLY,
                "price_high": high,
                "price_low": low,
                "base_start": shoulder[0],
                "conf_idx": fail_idx,
                "leg_out_end": fail_idx,
                "impulse_atr": float(abs(head[1] - neck[1]) / max(atr_at, 1e-9)),
                "needs_htf": True,
            }
        )
    return zones


# ─────────────────────────────────────────────────────────────────────
# SNRC — SnR continuation setups
# ─────────────────────────────────────────────────────────────────────


def _upgrade_snrc(zones, levels, atr_val, params):
    """Re-label V1 bases that sit on a strong SnR level.

        SNRC1  entry point is RBR / DBD (the continuation pair)
        SNRC2  entry point is DBR / RBD (the reversal pair)

    Both additionally require higher-timeframe confirmation, which the
    caller enforces through `needs_htf` — doctrine is explicit that the
    setup is only the plan, and confirmation comes from switching timeframe.
    Mutates in place and returns the same list: an upgraded zone must remain
    one zone, not become a second copy under a new label."""
    tol = atr_val * params.get("level_tol_atr_mult", 0.3)
    for zone in zones:
        if zone["source"] != "SND_V1":
            continue
        if not _at_strong_level(levels, zone["price_low"], zone["price_high"], tol):
            continue
        if zone["pattern"] in ("RBR", "DBD"):
            zone["family"] = "SNRC1"
            zone["needs_htf"] = True
        elif zone["pattern"] in ("DBR", "RBD"):
            zone["family"] = "SNRC2"
            zone["needs_htf"] = True
    return zones


def _detect_snrc3(df, atr_series, params, levels, bull_engulf, bear_engulf):
    """SNRC3 — the "war" between engulfings at a strong level.

    To sell: a strong support where a *bullish* engulfing formed and then
    failed (a later close back below that engulfing's low); the entry is the
    bearish engulfing printed on the same line. To buy, the mirror.

    Deliberately ignores the S&D reading, per doctrine: at this point the
    only thing being judged is which engulfing won."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)
    tol_mult = params.get("level_tol_atr_mult", 0.3)
    max_wait = int(params.get("snrc3_max_wait_bars", 12))
    max_height_mult = params.get("qm_zone_max_atr_mult", 2.0)

    zones = []
    for i in range(1, n - 2):
        atr_at = atr_filled[i]
        if atr_at <= 0:
            continue
        tol = atr_at * tol_mult
        for failed_bull in (True, False):
            if failed_bull and not bull_engulf[i]:
                continue
            if (not failed_bull) and not bear_engulf[i]:
                continue
            if not _at_strong_level(levels, lows[i], highs[i], tol):
                continue
            stop = min(n, i + 1 + max_wait)
            # Failure = a close back through the engulfing candle's far side.
            if failed_bull:
                broke = np.flatnonzero(closes[i + 1 : stop] < lows[i])
            else:
                broke = np.flatnonzero(closes[i + 1 : stop] > highs[i])
            if len(broke) == 0:
                continue
            fail_idx = int(broke[0]) + i + 1
            # The entry is the opposite engulfing on the same line.
            counter = bear_engulf if failed_bull else bull_engulf
            candidates = np.flatnonzero(counter[fail_idx : min(n, fail_idx + max_wait)])
            if len(candidates) == 0:
                continue
            entry_idx = int(candidates[0]) + fail_idx
            if not _at_strong_level(levels, lows[entry_idx], highs[entry_idx], tol):
                continue
            low, high = _band_around(entry_idx, highs, lows, 0, atr_at, max_height_mult)
            zones.append(
                {
                    "source": "SNRC3",
                    "family": "SNRC3",
                    "pattern": "SNRC3",
                    "kind": ZoneKind.SUPPLY if failed_bull else ZoneKind.DEMAND,
                    "price_high": high,
                    "price_low": low,
                    "base_start": i,
                    "conf_idx": entry_idx,
                    "leg_out_end": entry_idx,
                    "impulse_atr": float(abs(closes[entry_idx] - closes[i]) / max(atr_at, 1e-9)),
                    "needs_htf": False,
                }
            )
    return zones


# ─────────────────────────────────────────────────────────────────────
# Liquidity sweep / stop hunt
# ─────────────────────────────────────────────────────────────────────


def _detect_sweep_zones(df, atr_series, params, swings):
    """Price wicks through a prior swing extreme — taking the stops resting
    beyond it — and closes back inside. The zone is the swept band itself:
    the liquidity has been collected, so the level is expected to hold."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)
    min_pierce = params.get("sweep_min_pierce_atr_mult", 0.15)
    max_wait = int(params.get("sweep_max_wait_bars", 20))

    zones = []
    for swing_idx, swing_price, kind in swings:
        stop = min(n, swing_idx + 1 + max_wait)
        for j in range(swing_idx + 1, stop):
            atr_at = atr_filled[j]
            if atr_at <= 0:
                continue
            pierce = min_pierce * atr_at
            if kind == "low" and lows[j] < swing_price - pierce and closes[j] > swing_price:
                zones.append(
                    {
                        "source": "SWEEP",
                        "family": "SWEEP",
                        "pattern": "SWEEP_LOW",
                        "kind": ZoneKind.DEMAND,
                        "price_high": float(swing_price),
                        "price_low": float(lows[j]),
                        "base_start": swing_idx,
                        "conf_idx": j,
                        "leg_out_end": j,
                        "impulse_atr": float((swing_price - lows[j]) / atr_at),
                        "needs_htf": False,
                    }
                )
                break
            if kind == "high" and highs[j] > swing_price + pierce and closes[j] < swing_price:
                zones.append(
                    {
                        "source": "SWEEP",
                        "family": "SWEEP",
                        "pattern": "SWEEP_HIGH",
                        "kind": ZoneKind.SUPPLY,
                        "price_high": float(highs[j]),
                        "price_low": float(swing_price),
                        "base_start": swing_idx,
                        "conf_idx": j,
                        "leg_out_end": j,
                        "impulse_atr": float((highs[j] - swing_price) / atr_at),
                        "needs_htf": False,
                    }
                )
                break
    return zones


# ─────────────────────────────────────────────────────────────────────
# Zone lifecycle on the entry timeframe
# ─────────────────────────────────────────────────────────────────────


def _track_zone(zone, start_ns, entry_t_ns, highs, lows, closes):
    """Follow one zone forward on entry-TF bars from the moment it became
    valid. Returns a lifecycle dict, or None if it is not yet in the window.

    `touches` counts *episodes*, not bars: price entering, leaving and
    returning is two touches, while sitting inside for thirty bars is one.
    Zone freshness is a first-class S&D idea (a level that has already been
    tested twice is not the level it was), and it is also what keeps the
    learner from recording the same setup once per bar."""
    start = int(np.searchsorted(entry_t_ns, start_ns, side="left"))
    n = len(entry_t_ns)
    if start >= n:
        return None

    demand = zone["kind"] == ZoneKind.DEMAND
    breached = closes[start:] < zone["price_low"] if demand else closes[start:] > zone["price_high"]
    break_off = int(np.argmax(breached)) if bool(breached.any()) else -1
    end = start + break_off if break_off >= 0 else n

    inside = np.zeros(0, dtype=bool)
    if end > start:
        window_high = highs[start:end]
        window_low = lows[start:end]
        inside = (window_low <= zone["price_high"]) & (window_high >= zone["price_low"])
    if len(inside) == 0:
        touches = 0
    else:
        entries = np.flatnonzero(inside & ~np.concatenate(([False], inside[:-1])))
        touches = int(len(entries))

    live_now = bool(len(inside) > 0 and inside[-1] and end == n)
    return {
        "broken": break_off >= 0,
        "break_ns": int(entry_t_ns[start + break_off]) if break_off >= 0 else None,
        "touches": touches,
        "in_zone": live_now,
        # True only on the bar price *enters* the zone. A retest is an event,
        # not a state: without this the bot re-signals every bar price sits
        # inside a zone, which on real tape fired on 28.8% of all bars. With
        # live transaction cost already at 116% of gross profit, paying a
        # round trip per bar of a slow retest is how an edge gets spent.
        "fresh_touch": bool(live_now and (len(inside) < 2 or not inside[-2])),
        "age_bars": int(n - 1 - start),
    }


def _flip_to_breaker(zone, break_ns):
    """A broken zone does not stop existing — it changes sign. Demand that
    price closed below is now supply on the retest from underneath. This is
    the breaker / swap block, the one PoB setup with no entry point of its
    own ("Blindspot"), so it is traded only on the retest and carries its own
    family label for the learner to judge separately."""
    flipped = dict(zone)
    flipped["kind"] = ZoneKind.SUPPLY if zone["kind"] == ZoneKind.DEMAND else ZoneKind.DEMAND
    flipped["source"] = "BREAKER"
    flipped["family"] = "BREAKER"
    flipped["pattern"] = "BREAKER_" + str(zone.get("pattern", ""))
    flipped["needs_htf"] = True
    flipped["valid_from_ns"] = break_ns
    return flipped


# ─────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────


def _volatility_bucket(atr_values, params):
    """Where current ATR sits in its own trailing distribution — 0 quiet,
    1 normal, 2 expanded. Percentile-ranked against recent history rather
    than an absolute threshold, so it means the same thing in any price
    regime."""
    valid = atr_values[np.isfinite(atr_values)]
    if len(valid) < int(params.get("regime_min_atr_history", 20)):
        return 1
    current = valid[-1]
    low_p = np.quantile(valid, params.get("vol_low_percentile", 0.33))
    high_p = np.quantile(valid, params.get("vol_high_percentile", 0.67))
    if current <= low_p:
        return 0
    if current >= high_p:
        return 2
    return 1


def _trend_direction(closes, params):
    """+1 up, -1 down, 0 undecided — EMA fast vs slow, gated by a minimum
    separation in ATR so a flat crossover does not read as a trend."""
    fast_period = int(params.get("trend_ema_fast", 12))
    slow_period = int(params.get("trend_ema_slow", 34))
    if len(closes) < slow_period + 2:
        return 0, 0.0
    fast = _ema(closes, fast_period)
    slow = _ema(closes, slow_period)
    separation = float(fast[-1] - slow[-1])
    return (1 if separation > 0 else -1 if separation < 0 else 0), separation


def _trend_bucket(adx_value, trend_dir, trade_dir, params):
    """0 = against a real trend, 1 = no trend worth the name, 2 = with it.
    The single most load-bearing conditioner in the bucket key: the same
    zone is a different proposition depending on which of these it is."""
    threshold = params.get("adx_trend_threshold", 20.0)
    if not np.isfinite(adx_value) or adx_value < threshold or trend_dir == 0:
        return 1
    return 2 if trend_dir == trade_dir else 0


# ─────────────────────────────────────────────────────────────────────
# Priors distilled from live trading history
# ─────────────────────────────────────────────────────────────────────
#
# Derived from 3,105 closed live XAUUSD M1 trades (all `*_m1` skills, the
# only ones with MFE/MAE recorded), labelled "did favourable excursion reach
# +0.2R before the stop". Each number is a log-odds offset against the
# overall live base rate of 0.620, empirical-Bayes shrunk toward zero by 60
# pseudo-observations so a thin bucket cannot shout.
#
# These are what a cold instance starts from. Without them a fresh bot
# treats a setup with a known-awful record exactly like a known-good one and
# has to lose real money to rediscover the difference; with them, the online
# learner starts in roughly the right place and spends its evidence
# correcting the prior rather than building one from scratch.
#
# IMPORTANT — hour-of-day is deliberately NOT here. Per-hour offsets look
# enormously profitable in-sample (PF 2.70) and collapse out-of-sample
# (PF 0.12 on a chronological 70/30 split): with ~20 buckets over ~2,200
# fitting trades it is fitting noise. Time-of-day still reaches the model,
# but only as `hour_sin`/`hour_cos` features the online learner may weigh
# for itself against everything else — never as a hard gate.
PATTERN_PRIOR = {
    "RBR": 0.6391,
    "RBD": 0.2663,
    "DBD": -0.1153,
    "QM_BEAR": -0.2061,
    "QM_BULL": -0.3678,
    "DBR": -0.4093,
    "DZ_V2": -0.8874,
}
SESSION_PRIOR = {
    "new_york": 0.4999,
    "asian": 0.0108,
    "overlap": -0.1925,
    "london": -0.2132,
}
VOLATILITY_PRIOR = {"normal": 0.1786, "low": 0.0903, "high": -0.2756}
TREND_PRIOR = {"trending": 0.0785, "ranging": -0.2086}

# Families this bot detects that have no live track record yet. They start
# slightly in the red so they have to earn their place rather than trade
# freely on the strength of never having lost yet.
UNPROVEN_FAMILY_PRIOR = -0.30

# P(MFE reaches r x risk), per pattern, measured on the same live trades.
# The take-profit decision is a bet on this curve, so it is measured rather
# than assumed. Read it and note how fast it falls: no pattern reaches 2.0R
# more than a quarter of the time, which is why the inherited 2.5R/4.0R
# ladder almost never filled and the trade just rode until it reversed.
SURVIVAL_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
MFE_SURVIVAL = {
    "RBR": (0.496, 0.386, 0.311, 0.227, 0.147, 0.098, 0.063),
    "RBD": (0.484, 0.410, 0.377, 0.326, 0.251, 0.143, 0.092),
    "DBD": (0.342, 0.259, 0.191, 0.108, 0.059, 0.025, 0.018),
    "QM_BULL": (0.256, 0.194, 0.186, 0.155, 0.031, 0.016, 0.008),
    "DBR": (0.235, 0.172, 0.120, 0.093, 0.067, 0.028, 0.011),
    "DZ_V2": (0.213, 0.191, 0.154, 0.105, 0.059, 0.034, 0.022),
    "QM_BEAR": (0.074, 0.041, 0.041, 0.017, 0.001, 0.001, 0.001),
}
DEFAULT_SURVIVAL = (0.374, 0.294, 0.241, 0.180, 0.119, 0.068, 0.043)


def _survival_for(pattern):
    return MFE_SURVIVAL.get(pattern, DEFAULT_SURVIVAL)


def _survival_at(pattern, r_level):
    """Linearly interpolate the measured survival curve at an arbitrary R."""
    curve = _survival_for(pattern)
    return float(np.interp(r_level, SURVIVAL_GRID, curve))


def _pattern_key(zone):
    """Map a detector family onto the pattern name the live journal recorded,
    or None when this bot detects something the journal has never seen. None
    is meaningful — it routes the setup to `UNPROVEN_FAMILY_PRIOR` instead of
    silently borrowing another pattern's track record."""
    pattern = str(zone.get("pattern") or "")
    if pattern.startswith("BREAKER_"):
        return None
    if pattern == "SZ_V2":
        return "DZ_V2"  # same detector, the journal only ever labelled one side
    if pattern in PATTERN_PRIOR:
        return pattern
    return None


def _prior_logit(pattern, session, volatility, trend, known_family):
    """Additive log-odds prior for one setup.

    Additive across dimensions rather than a joint lookup on purpose: 3,105
    trades split by pattern x session x volatility x hour is far too sparse
    to estimate jointly, and a joint table would memorise noise. Marginal
    offsets that add are the standard robust decomposition."""
    total = 0.0
    total += PATTERN_PRIOR.get(pattern, 0.0)
    total += SESSION_PRIOR.get(session, 0.0)
    total += VOLATILITY_PRIOR.get(volatility, 0.0)
    total += TREND_PRIOR.get(trend, 0.0)
    if not known_family:
        total += UNPROVEN_FAMILY_PRIOR
    return float(total)


# ─────────────────────────────────────────────────────────────────────
# Sessions and volume
# ─────────────────────────────────────────────────────────────────────


def _session_for(hour):
    """UTC-hour trading session, matching `engine.domain.regime.session_for`
    boundaries so a session name means the same thing here as in the journal
    the priors above were measured from."""
    if 12 <= hour < 16:
        return "overlap"
    if 7 <= hour < 12:
        return "london"
    if 16 <= hour < 21:
        return "new_york"
    if hour >= 22 or hour < 7:
        return "asian"
    return "off_session"


def _volume_state(df, params):
    """Participation, as ratios rather than raw tick counts.

    Volume matters twice over for this bot: a departure leg on thin volume is
    a weaker zone, and a retest arriving on heavy volume is more likely to be
    a break than a bounce. Both are ratios against the bar's own trailing
    average so they mean the same thing across sessions, where absolute tick
    counts differ by an order of magnitude."""
    if "tick_volume" not in df.columns or len(df) < 25:
        return {"ratio": 1.0, "burst": 1.0, "trend": 1.0}
    volume = df["tick_volume"].to_numpy(dtype=float)
    window = int(params.get("volume_lookback_bars", 20))
    baseline = float(np.mean(volume[-window:]))
    if baseline <= 0:
        return {"ratio": 1.0, "burst": 1.0, "trend": 1.0}
    recent = float(np.mean(volume[-5:]))
    return {
        "ratio": float(volume[-1] / baseline),
        "burst": float(np.max(volume[-3:]) / baseline),
        "trend": float(recent / baseline),
    }


# ─────────────────────────────────────────────────────────────────────
# Change of character (CHoCH)
# ─────────────────────────────────────────────────────────────────────


def _detect_choch(structure, closes, atr_val, params):
    """Has the market just changed character against the prevailing swing
    sequence?

    A break of structure (BOS) continues a trend: price takes out the last
    high while making higher lows. A *change of character* is the first break
    the other way — in an up-sequence, a close below the most recent higher
    low. That is the earliest structural evidence that the move which
    justified the entry has stopped being that move.

    Returns +1 (turned bullish), -1 (turned bearish) or 0.

    This matters more than any entry tweak on this bot: measured on live
    trades, 88.7% of losers were in profit first, and they gave back 2,579R
    in aggregate. The losers peak a median 354s after entry and then decay
    for another ~1,200s before dying, so there is a real window in which a
    structural exit can act."""
    if len(structure) < 3 or len(closes) == 0:
        return 0
    close = float(closes[-1])
    margin = atr_val * float(params.get("choch_break_atr_mult", 0.1))

    last_low = None
    last_high = None
    for point in reversed(structure):
        if last_low is None and point["label"] in (StructureLabel.HL, StructureLabel.LL):
            last_low = point
        if last_high is None and point["label"] in (StructureLabel.HH, StructureLabel.LH):
            last_high = point
        if last_low is not None and last_high is not None:
            break

    bias = _structure_bias(structure)
    if bias > 0 and last_low is not None and close < last_low["price"] - margin:
        return -1
    if bias < 0 and last_high is not None and close > last_high["price"] + margin:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────

# Names are the contract between `_build_features` and the learner's weight
# vector — order matters and must not be reshuffled without resetting the
# learner, since a learned weight is bound to a position, not a name.
FEATURE_NAMES = (
    "zone_height_atr",
    "depth_into_zone",
    "zone_age",
    "touch_count",
    "impulse_atr",
    "atr_pct",
    "atr_ratio",
    "adx_norm",
    "trend_align",
    "htf_align",
    "extension_atr",
    "structure_align",
    "rr_room",
    "spread_atr",
    "cost_over_target",
    "hour_sin",
    "hour_cos",
    "session_quality",
    "body_ratio",
    "wick_reject",
    "range_pos",
    "htf_engulf",
    "volume_ratio",
    "volume_burst",
    "volume_trend",
    "choch_against",
)
N_FEATURES = len(FEATURE_NAMES)


def _build_features(
    *,
    zone,
    life,
    close,
    atr_val,
    atr_slow,
    adx_value,
    trend_dir,
    htf_dir,
    htf_engulf,
    structure_bias,
    trade_dir,
    rr_room,
    spread_price,
    sl_points,
    hour,
    session,
    last_bar,
    range_pos,
    extension_atr,
    volume,
    choch,
):
    """Twenty-six scale-free numbers describing one candidate.

    Every one is a ratio, a bounded count, or a sign — nothing carries a
    price or a point value. That is what lets the same learned weights stay
    meaningful as gold moves from 1,900 to 4,000, and it is why the model can
    be this small: the features already carry the invariance a bigger model
    would otherwise have to learn.

    `cost_over_target` earns its place from the live numbers: transaction
    cost came to 116% of gross profit, so the bot was profitable before costs
    and lost money after them. A setup whose target barely clears its own
    round-trip cost is a different proposition from an identical-looking one
    that clears it comfortably, and nothing else in this vector says so."""
    zone_height = max(zone["price_high"] - zone["price_low"], 1e-9)
    if zone["kind"] == ZoneKind.DEMAND:
        depth = (zone["price_high"] - close) / zone_height
    else:
        depth = (close - zone["price_low"]) / zone_height

    body = abs(last_bar["close"] - last_bar["open"])
    bar_range = max(last_bar["high"] - last_bar["low"], 1e-9)
    # The rejection wick on the side that favours the trade: for a long, the
    # tail below the body (sellers pushed down and failed).
    if trade_dir == 1:
        wick = min(last_bar["open"], last_bar["close"]) - last_bar["low"]
    else:
        wick = last_bar["high"] - max(last_bar["open"], last_bar["close"])
    wick = max(wick, 0.0)

    return np.array(
        [
            zone_height / atr_val,
            float(np.clip(depth, -0.5, 1.5)),
            float(np.log1p(max(life["age_bars"], 0)) / 5.0),
            float(min(life["touches"], 5)) / 5.0,
            float(np.clip(zone.get("impulse_atr", 0.0), 0.0, 6.0)),
            atr_val / max(close, 1e-9) * 1000.0,
            atr_val / max(atr_slow, 1e-9),
            float(np.clip(adx_value / 50.0, 0.0, 2.0)) if np.isfinite(adx_value) else 0.4,
            float(trend_dir * trade_dir),
            float(htf_dir * trade_dir),
            float(np.clip(extension_atr, -6.0, 6.0)),
            float(structure_bias * trade_dir),
            float(np.clip(rr_room, 0.0, 8.0)),
            spread_price / atr_val,
            float(np.clip(spread_price / max(sl_points, 1e-9), 0.0, 2.0)),
            float(np.sin(2.0 * np.pi * hour / 24.0)),
            float(np.cos(2.0 * np.pi * hour / 24.0)),
            float(SESSION_PRIOR.get(session, 0.0)),
            float(body / bar_range),
            float(np.clip(wick / atr_val, 0.0, 4.0)),
            float(np.clip(range_pos, 0.0, 1.0)),
            float(htf_engulf),
            float(np.clip(volume["ratio"], 0.0, 5.0)),
            float(np.clip(volume["burst"], 0.0, 6.0)),
            float(np.clip(volume["trend"], 0.0, 4.0)),
            float(1.0 if choch == -trade_dir else 0.0),
        ],
        dtype=float,
    )


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────


class XauusdSndAdaptiveM1:
    def __init__(self):
        self.spec = StrategySpec(
            name="xauusd_snd_adaptive_m1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            # M15 is fetched purely for confirmation and the HTF feature — the
            # doctrine ("multi timeframe analysis is behind every great
            # setup") and the learner both want a frame the entry noise
            # cannot dominate.
            confirmation_timeframes=("M15",),
            htf_veto=False,  # this bot does its own trend gating
            close_on_opposite_signal=True,
            params={
                # ── Timeframes ──
                "zone_tf_minutes": 5,
                "entry_tf_minutes": 1,
                "htf_key": "M15",
                # ── Zone detection ──
                "atr_period": 14,
                "atr_slow_period": 50,
                "base_body_atr_mult": 0.5,
                "leg_travel_atr_mult": 0.7,
                "max_base_candles": 6,
                "v2_max_base_candles": 4,
                "v2_engulf_min_atr_mult": 1.0,
                # ── Quasimodo ──
                "qm_swing_lookback": 3,
                "qm_min_swing_atr": 0.5,
                "qm_min_head_atr_mult": 0.4,
                "qm_zone_max_atr_mult": 2.0,
                # ── Structure / levels / trendlines ──
                "structure_swing_lookback": 3,
                "level_cluster_atr_mult": 0.35,
                "level_min_touches": 2,
                "level_tol_atr_mult": 0.3,
                "trendline_tol_atr_mult": 0.35,
                "trendline_max_pairs": 6,
                # ── SNRC3 / sweep ──
                "snrc3_max_wait_bars": 12,
                "sweep_min_pierce_atr_mult": 0.15,
                "sweep_max_wait_bars": 20,
                # ── HTF confirmation ──
                "htf_engulf_lookback": 4,
                # ── Regime ──
                "adx_period": 14,
                "adx_trend_threshold": 20.0,
                "trend_ema_fast": 12,
                "trend_ema_slow": 34,
                "vol_low_percentile": 0.33,
                "vol_high_percentile": 0.67,
                "regime_min_atr_history": 20,
                "range_lookback_bars": 40,
                # ── Stop loss ──
                "sl_zone_buffer_atr_mult": 0.15,
                "sl_zone_buffer_zone_frac": 0.10,
                "sl_min_atr_mult": 0.5,
                # ── Take profit ladder ──
                # Targets are *chosen*, not fixed: `_target_grid` maximises
                # expected R against the measured MFE survival curve for this
                # setup's bucket. The grid starts at the broker floor because
                # SpreadGate silently drops any leg below spread-adjusted
                # min_rr (XAUUSD: 1.5) — see broker/spread_gate.py. The old
                # fixed 2.5R/4.0R ladder sat beyond the p85 MFE of every
                # pattern measured live, so those legs almost never filled.
                "tp_min_rr_floor": 1.75,
                "tp_target_grid": (1.75, 2.0, 2.5, 3.0),
                "tp_runner_extra_rr": 1.0,
                "tp_buffer_points": 20,
                "tp_buffer_atr_mult": 0.3,
                "point_value": 0.01,
                # ── Volume ──
                "volume_lookback_bars": 20,
                # ── Change of character ──
                # Live evidence: 88.7% of losing trades were in profit first
                # and gave back 2,579R in aggregate. `engine/` is off-limits
                # per CLAUDE.md, so the only sanctioned exit lever is
                # `close_on_opposite_signal` — a CHoCH against an open
                # position emits the opposing signal, which closes it.
                "choch_exit_enabled": True,
                "choch_break_atr_mult": 0.1,
                "choch_sl_atr_mult": 1.0,
                # ── Regime gates (validated out-of-sample) ──
                # A chronological 70/30 split of 3,105 live trades: these two
                # gates take out-of-sample PF from 1.02 to ~1.40 while keeping
                # ~31% of trades. Hour-of-day is deliberately absent — it
                # tested at PF 2.70 in-sample and 0.12 out-of-sample.
                "skip_volatility_buckets": (2,),
                "skip_sessions": ("overlap",),
                "family_denylist": ("SND_V2",),
                # ── Learner ──
                "learner_enabled": True,
                "learner_secure_r": 0.2,
                "learner_horizon_bars": 120,
                "learner_min_bucket_samples": 25.0,
                "learner_min_model_samples": 150,
                "learner_min_quantile_samples": 30,
                "learner_half_life_samples": 750.0,
                # 0.620 is the measured live secure-rate across 3,105 XAUUSD
                # M1 trades — the honest base rate a cold bucket should start
                # from. The library default (0.85) would wave everything
                # through until local evidence caught up.
                "learner_global_prior_rate": 0.62,
                "learner_gate_p": 0.65,
                # Second gate, on expected R under the engine's real
                # three-outcome exit process (see `_blended_expectancy`). The
                # probability gate alone let through setups whose own numbers
                # said they were negative-EV.
                "min_expectancy_r": 0.02,
                "learner_max_pending": 400,
                # Learned values are clamped to [base * lo, base * hi].
                "sl_buffer_learned_quantile": 0.8,
                "sl_buffer_band_lo": 0.6,
                "sl_buffer_band_hi": 2.2,
                "tp2_learned_quantile": 0.6,
                "tp3_learned_quantile": 0.85,
                "tp_band_lo": 0.6,
                "tp_band_hi": 1.6,
            },
        )
        self._learner = None
        self._last_bar_ns = None
        self._observed = {}
        self._last_choch = 0
        self._swings = []
        self._levels = []
        self._trend_dir = 0
        self._htf = {"dir": 0, "bull": False, "bear": False}

    # ── learner lifecycle ─────────────────────────────────────────────

    def _ensure_learner(self, params):
        if self._learner is None:
            self._learner = AdaptiveLearner(
                N_FEATURES,
                LearnerConfig(
                    secure_r=float(params["learner_secure_r"]),
                    min_bucket_samples=float(params["learner_min_bucket_samples"]),
                    min_model_samples=int(params["learner_min_model_samples"]),
                    min_quantile_samples=int(params["learner_min_quantile_samples"]),
                    half_life_samples=float(params["learner_half_life_samples"]),
                    global_prior_rate=float(params["learner_global_prior_rate"]),
                    max_pending=int(params["learner_max_pending"]),
                ),
            )
        return self._learner

    def reset_state(self):
        """Forget everything learned. Public because a sweep harness building
        one instance per variant needs a guaranteed-clean slate, and because
        `evaluate` calls it itself whenever the candle stream shows it is
        looking at a different replay."""
        if self._learner is not None:
            self._learner.reset()
        self._last_bar_ns = None
        self._observed = {}
        self._last_choch = 0

    def _check_continuity(self, first_ns, last_ns):
        """Reuse of one strategy instance across two backtests is the normal
        case in a parameter sweep, and it would otherwise carry the first
        run's learning into the second — silently, and only visible as an
        unreproducible number. A stream that moves backwards, or jumps
        further forward than any real market close, is a different replay."""
        if self._last_bar_ns is None:
            return
        if last_ns < self._last_bar_ns or first_ns > self._last_bar_ns + _RESET_GAP_NS:
            self.reset_state()

    # ── main entry point ──────────────────────────────────────────────

    def evaluate(self, ctx: MarketContext):
        params = self.spec.params
        df = ctx.candles.get(self.spec.entry_timeframe)
        if df is None or len(df) < 60 or "time" not in df.columns:
            return None

        entry_t_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
        self._check_continuity(int(entry_t_ns[0]), int(entry_t_ns[-1]))

        resampled = _resample(df, int(params["zone_tf_minutes"]), int(params["entry_tf_minutes"]))
        if resampled is None:
            return None
        zone_frame, zone_end_ns = resampled

        atr_period = int(params["atr_period"])
        if len(zone_frame) < atr_period + 6:
            return None
        atr_series = _atr(zone_frame, atr_period)
        atr_clean = atr_series.dropna()
        if atr_clean.empty or float(atr_clean.iloc[-1]) <= 0:
            return None
        atr_val = float(atr_clean.iloc[-1])

        entry_highs = df["high"].to_numpy()
        entry_lows = df["low"].to_numpy()
        entry_closes = df["close"].to_numpy()

        learner_on = bool(params.get("learner_enabled", True))
        learner = self._ensure_learner(params) if learner_on else None
        # Grade whatever the new bars resolved *before* asking about a new
        # setup — this ordering is what guarantees the gate is a function of
        # already-determined labels only.
        if learner is not None:
            learner.advance(entry_t_ns, entry_highs, entry_lows)
        self._last_bar_ns = int(entry_t_ns[-1])

        zones = self._detect_all(zone_frame, atr_series, atr_val, params, ctx)
        if not zones:
            return None

        candidates, live_zones = self._select_candidates(
            zones, zone_end_ns, entry_t_ns, entry_highs, entry_lows, entry_closes
        )
        if not candidates:
            return None

        return self._decide(
            ctx=ctx,
            params=params,
            df=df,
            zone_frame=zone_frame,
            atr_series=atr_series,
            atr_val=atr_val,
            candidates=candidates,
            live_zones=live_zones,
            learner=learner,
            entry_t_ns=entry_t_ns,
        )

    # ── detection ─────────────────────────────────────────────────────

    def _detect_all(self, zone_frame, atr_series, atr_val, params, ctx):
        highs = zone_frame["high"].to_numpy()
        lows = zone_frame["low"].to_numpy()
        closes = zone_frame["close"].to_numpy()
        opens = zone_frame["open"].to_numpy()

        swings = _detect_swing_points(highs, lows, int(params["structure_swing_lookback"]))
        trend_dir, _separation = _trend_direction(closes, params)
        levels = _strong_levels(
            swings,
            atr_val,
            params.get("level_cluster_atr_mult", 0.35),
            int(params.get("level_min_touches", 2)),
        )
        bull_engulf, bear_engulf = _engulfing_flags(opens, closes)

        zones = []
        zones.extend(_detect_zones_v1(zone_frame, atr_series, params))
        zones.extend(_detect_zones_v2(zone_frame, atr_series, params))
        zones.extend(_detect_quasimodo_legacy(zone_frame, atr_series, params))
        zones.extend(_detect_quasimodo_family(zone_frame, atr_series, params, swings, trend_dir))
        zones.extend(
            _detect_snrc3(zone_frame, atr_series, params, levels, bull_engulf, bear_engulf)
        )
        zones.extend(_detect_sweep_zones(zone_frame, atr_series, params, swings))
        _upgrade_snrc(zones, levels, atr_val, params)

        for zone in zones:
            zone.setdefault("needs_htf", False)
        # Cached on the instance so the decision step does not recompute what
        # detection already needed.
        self._swings = swings
        self._levels = levels
        self._trend_dir = trend_dir
        self._htf = self._htf_state(ctx, params)
        return zones

    def _htf_state(self, ctx, params):
        """Higher-timeframe direction and whether it has just printed an
        engulfing candle. Doctrine treats the engulfing body as the best
        confirmation there is, and `needs_htf` setups are refused without
        one."""
        htf = ctx.candles.get(params.get("htf_key", "M15"))
        if htf is None or len(htf) < int(params["trend_ema_slow"]) + 2:
            return {"dir": 0, "bull": False, "bear": False}
        closes = htf["close"].to_numpy()
        opens = htf["open"].to_numpy()
        direction, _sep = _trend_direction(closes, params)
        bull, bear = _engulfing_flags(opens, closes)
        window = int(params.get("htf_engulf_lookback", 4))
        return {
            "dir": direction,
            "bull": bool(bull[-window:].any()),
            "bear": bool(bear[-window:].any()),
        }

    def _select_candidates(self, zones, zone_end_ns, entry_t_ns, highs, lows, closes):
        """Walk every zone forward, keep the ones still alive, flip the ones
        that broke into breakers, and collect those price is touching now."""
        live = []
        candidates = []
        for zone in zones:
            leg_out_end = zone["leg_out_end"]
            if leg_out_end >= len(zone_end_ns):
                continue
            start_ns = int(zone_end_ns[leg_out_end])
            life = _track_zone(zone, start_ns, entry_t_ns, highs, lows, closes)
            if life is None:
                continue

            if not life["broken"]:
                live.append(zone)
                if life["fresh_touch"]:
                    candidates.append((zone, life))
                continue

            # Broken: the original is dead, but its band flips polarity.
            if zone["source"] == "BREAKER":
                continue
            flipped = _flip_to_breaker(zone, life["break_ns"])
            flipped_life = _track_zone(flipped, life["break_ns"], entry_t_ns, highs, lows, closes)
            if flipped_life is None or flipped_life["broken"]:
                continue
            live.append(flipped)
            if flipped_life["fresh_touch"]:
                candidates.append((flipped, flipped_life))
        return candidates, live

    # ── decision ──────────────────────────────────────────────────────

    def _decide(
        self,
        *,
        ctx,
        params,
        df,
        zone_frame,
        atr_series,
        atr_val,
        candidates,
        live_zones,
        learner,
        entry_t_ns,
    ):
        last_i = len(df) - 1
        close = float(df["close"].iloc[last_i])
        now = df["time"].iloc[last_i]
        last_bar = {
            "open": float(df["open"].iloc[last_i]),
            "high": float(df["high"].iloc[last_i]),
            "low": float(df["low"].iloc[last_i]),
            "close": close,
        }
        spread_price = float(ctx.spread_points) * float(params.get("point_value", 0.01))

        zone_closes = zone_frame["close"].to_numpy()
        zone_highs = zone_frame["high"].to_numpy()
        zone_lows = zone_frame["low"].to_numpy()
        adx_values = _adx(zone_highs, zone_lows, zone_closes, int(params["adx_period"]))
        adx_value = float(adx_values[-1]) if len(adx_values) else float("nan")
        atr_slow_series = _atr(zone_frame, int(params.get("atr_slow_period", 50)))
        atr_slow_clean = atr_slow_series.dropna()
        atr_slow = float(atr_slow_clean.iloc[-1]) if not atr_slow_clean.empty else atr_val
        vol_bucket = _volatility_bucket(atr_series.to_numpy(), params)
        structure = _detect_structure(self._swings)
        structure_bias = _structure_bias(structure)

        slow_period = int(params.get("trend_ema_slow", 34))
        if len(zone_closes) >= slow_period + 2:
            anchor = float(_ema(zone_closes, slow_period)[-1])
        else:
            anchor = close

        lookback = int(params.get("range_lookback_bars", 40))
        window_high = float(zone_highs[-lookback:].max())
        window_low = float(zone_lows[-lookback:].min())
        span = max(window_high - window_low, 1e-9)

        hour = int(now.hour)
        session = _session_for(hour)
        volume = _volume_state(df, params)
        choch = _detect_choch(structure, zone_closes, atr_val, params)
        trend_name = (
            "trending"
            if np.isfinite(adx_value) and adx_value >= params.get("adx_trend_threshold", 20.0)
            else "ranging"
        )
        vol_name = ("low", "normal", "high")[vol_bucket]

        horizon_ns = (
            int(params["learner_horizon_bars"]) * int(params["entry_tf_minutes"]) * _NS_PER_MINUTE
        )
        now_ns = int(entry_t_ns[-1])
        threshold = self._gate_threshold(params)

        # ── Regime gates, validated out-of-sample ──
        # Unlike everything else here these are hard skips, because the
        # measured effect is not marginal: high volatility ran PF 0.34 over
        # 508 live trades and the London/NY overlap PF 0.31 over 321. A
        # learner could rediscover both, but only by paying for them again.
        blocked = vol_bucket in tuple(
            params.get("skip_volatility_buckets", ())
        ) or session in tuple(params.get("skip_sessions", ()))

        scored = []
        for zone, life in candidates:
            demand = zone["kind"] == ZoneKind.DEMAND
            trade_dir = 1 if demand else -1

            if blocked or zone["family"] in tuple(params.get("family_denylist", ())):
                continue

            # A change of character against the intended direction says the
            # move this setup is built on has already broken. Structural, and
            # prior to any probability estimate.
            if choch == -trade_dir:
                continue

            # Doctrine: a breaker, a QMM, or an SNRC base is only tradable
            # with higher-timeframe confirmation. Enforced before anything is
            # measured, because an unconfirmed one is not a setup at all.
            if zone.get("needs_htf", False):
                confirmed = self._htf["bull"] if demand else self._htf["bear"]
                if not confirmed:
                    continue

            plan = self._build_plan(
                zone=zone,
                life=life,
                params=params,
                close=close,
                atr_val=atr_val,
                spread_price=spread_price,
                live_zones=live_zones,
                learner=learner,
                vol_bucket=vol_bucket,
                adx_value=adx_value,
                trade_dir=trade_dir,
                session=session,
            )
            if plan is None:
                continue

            extension_atr = (close - anchor) / atr_val * trade_dir
            range_pos = (close - window_low) / span
            if trade_dir == -1:
                range_pos = 1.0 - range_pos
            features = _build_features(
                zone=zone,
                life=life,
                close=close,
                atr_val=atr_val,
                atr_slow=atr_slow,
                adx_value=adx_value,
                trend_dir=self._trend_dir,
                htf_dir=self._htf["dir"],
                htf_engulf=1.0 if (self._htf["bull"] if demand else self._htf["bear"]) else 0.0,
                structure_bias=structure_bias,
                trade_dir=trade_dir,
                rr_room=plan["rr_room"],
                spread_price=spread_price,
                sl_points=plan["sl_points"],
                hour=float(hour) + float(now.minute) / 60.0,
                session=session,
                last_bar=last_bar,
                range_pos=range_pos,
                extension_atr=extension_atr,
                volume=volume,
                choch=choch,
            )

            verdict = None
            if learner is not None:
                prior = _prior_logit(
                    plan["pattern"], session, vol_name, trend_name, plan["pattern"] is not None
                )
                verdict = learner.score(features, plan["bucket"], prior)
                self._record_candidate(
                    learner=learner,
                    zone=zone,
                    life=life,
                    features=features,
                    bucket=plan["bucket"],
                    now_ns=now_ns,
                    close=close,
                    trade_dir=trade_dir,
                    sl_points=plan["sl_points"],
                    atr_val=atr_val,
                    horizon_ns=horizon_ns,
                    params=params,
                )
                # Gated from the first bar, not only once the bucket is
                # `ready`: with historical priors seeded, a cold estimate is
                # already informative, and waiting for local evidence would
                # mean paying live for what the journal already recorded.
                if verdict.p_secure < threshold:
                    continue
                plan["expected_r"] = _blended_expectancy(
                    p_hit=plan["p_hit"],
                    r_target=plan["r_target"],
                    p_secure=verdict.p_secure,
                    secure_r=float(params["learner_secure_r"]),
                )
                if plan["expected_r"] < float(params.get("min_expectancy_r", 0.02)):
                    continue

            scored.append((zone, life, plan, verdict, features))

        if scored:
            scored.sort(key=lambda item: _rank_key(item[3], item[1]), reverse=True)
            zone, life, plan, verdict, _features = scored[0]
            self._last_choch = choch
            return self._build_signals(
                zone=zone,
                life=life,
                plan=plan,
                verdict=verdict,
                params=params,
                zone_frame=zone_frame,
                structure=structure,
                now=now,
                threshold=threshold,
                atr_val=atr_val,
            )

        exit_signal = self._choch_exit_signal(
            choch=choch,
            params=params,
            atr_val=atr_val,
            spread_price=spread_price,
            structure=structure,
            zone_frame=zone_frame,
            now=now,
        )
        self._last_choch = choch
        return exit_signal

    def _choch_exit_signal(
        self, *, choch, params, atr_val, spread_price, structure, zone_frame, now
    ):
        """Emit the opposing signal when structure turns, so the engine's
        `close_on_opposite_signal` path closes a position that is going wrong.

        This is the only exit lever a strategy has here: `PositionManager` is
        engine code and CLAUDE.md puts `backend/src/engine/` off-limits to
        strategy work. Note the side effect, which is real and deliberate —
        the same signal that closes the losing position also *opens* one in
        the new direction. A change of character is a reversal thesis, so
        that is coherent rather than accidental, but it is why this is gated
        behind `choch_exit_enabled` and fires only on the transition bar
        rather than on every bar structure stays broken.

        Fires only on a state change (`_last_choch`) because repeating it
        every bar would reopen the reversal position immediately after each
        stop-out."""
        if choch == 0 or not params.get("choch_exit_enabled", True):
            return None
        if choch == self._last_choch:
            return None

        sl_points = atr_val * float(params.get("choch_sl_atr_mult", 1.0))
        if sl_points <= 0:
            return None
        risk_price = sl_points + spread_price
        floor_rr = float(params.get("tp_min_rr_floor", 1.75))
        tp_points = risk_price * floor_rr

        structure_points = tuple(
            StructurePoint(
                time=zone_frame["time"].iloc[min(max(int(s["index"]), 0), len(zone_frame) - 1)],
                price=s["price"],
                label=s["label"],
            )
            for s in structure[-4:]
        )
        return (
            Signal(
                direction=Direction.BUY if choch == 1 else Direction.SELL,
                sl_points=float(sl_points),
                tp_points=float(tp_points),
                confidence=0.55,
                reason=(
                    f"CHoCH {'bullish' if choch == 1 else 'bearish'} — structure broke "
                    f"against the prevailing sequence; closes an opposing position "
                    f"sl={sl_points:.2f} tp={tp_points:.2f}"
                ),
                pattern="CHOCH",
                structure=structure_points,
                indicators=(
                    IndicatorReading(
                        name="choch_direction",
                        value=float(choch),
                        threshold=0.0,
                        comparison=">",
                        passed=True,
                    ),
                ),
            ),
        )

    def _gate_threshold(self, params):
        """The probability a setup must clear to be worth taking.

        Note what this is *not*: `breakeven_probability(0.2)` = 0.833, the
        rate at which pure secure-base trailing breaks even. Gating there
        would reject nearly everything, because the measured live secure rate
        is 0.620 — which is precisely why the bot has been losing money, and
        is a fact about the engine's exit behaviour rather than about any one
        setup.

        So the gate is relative instead: take only setups the evidence rates
        meaningfully above the average setup this family of bots has
        historically taken. That is a comparison the priors and the online
        table can both make honestly, and it degrades safely — if everything
        looks average, nothing trades."""
        return float(np.clip(params.get("learner_gate_p", 0.65), 0.05, 0.99))

    def _build_plan(
        self,
        *,
        zone,
        life,
        params,
        close,
        atr_val,
        spread_price,
        live_zones,
        learner,
        vol_bucket,
        adx_value,
        trade_dir,
        session,
    ):
        """Stop, take-profit ladder and learner bucket for one candidate."""
        demand = zone["kind"] == ZoneKind.DEMAND
        zone_height = zone["price_high"] - zone["price_low"]
        if zone_height <= 0:
            return None

        pattern = _pattern_key(zone)
        trend_slot = _trend_bucket(adx_value, self._trend_dir, trade_dir, params)
        # Session is in the key because the live split is large (new_york PF
        # 2.01 vs overlap PF 0.31) and it is stable enough to condition on —
        # unlike hour-of-day, which is in the feature vector only.
        bucket = f"{zone['family']}|v{vol_bucket}|t{trend_slot}|{session}"

        # ── Stop: zone height + a buffer that the learner may widen ──
        base_buffer_mult = float(params.get("sl_zone_buffer_atr_mult", 0.15))
        buffer_mult = base_buffer_mult
        if learner is not None:
            learned = learner.adverse_excursion_atr(
                bucket, float(params.get("sl_buffer_learned_quantile", 0.8))
            )
            if learned is not None:
                # The learned quantile is total adverse excursion in ATR; the
                # buffer only has to cover what sits *beyond* the zone, so the
                # zone's own height is netted off before clamping.
                beyond = max(learned - zone_height / atr_val, 0.0)
                buffer_mult = AdaptiveLearner.blend(
                    base_buffer_mult,
                    beyond,
                    lo_frac=float(params.get("sl_buffer_band_lo", 0.6)),
                    hi_frac=float(params.get("sl_buffer_band_hi", 2.2)),
                )
        buffer = max(
            atr_val * buffer_mult,
            zone_height * float(params.get("sl_zone_buffer_zone_frac", 0.10)),
        )

        if demand:
            sl_points = (close - zone["price_low"]) + buffer
        else:
            sl_points = (zone["price_high"] - close) + buffer
        sl_points = max(sl_points, atr_val * float(params.get("sl_min_atr_mult", 0.5)))
        if sl_points <= 0:
            return None
        risk_price = sl_points + spread_price

        # ── Targets: nearest opposite zones, then R multiples as fallback ──
        tp_buffer = max(
            float(params.get("tp_buffer_points", 20)) * float(params.get("point_value", 0.01)),
            atr_val * float(params.get("tp_buffer_atr_mult", 0.3)),
        )
        opposite = [z for z in live_zones if z["kind"] != zone["kind"]]
        zone_target_1 = None
        zone_target_2 = None
        if demand:
            above = sorted(
                [z for z in opposite if z["price_low"] > close], key=lambda z: z["price_low"]
            )
            if len(above) >= 1:
                zone_target_1 = above[0]["price_low"] - close - tp_buffer
            if len(above) >= 2:
                zone_target_2 = above[1]["price_low"] - close - tp_buffer
        else:
            below = sorted(
                [z for z in opposite if z["price_high"] < close],
                key=lambda z: z["price_high"],
                reverse=True,
            )
            if len(below) >= 1:
                zone_target_1 = close - below[0]["price_high"] - tp_buffer
            if len(below) >= 2:
                zone_target_2 = close - below[1]["price_high"] - tp_buffer

        # ── The target is chosen, not assumed ──
        # Expected R for a bracket is p(reach r) * r - (1 - p(reach r)), and
        # p(reach r) is *measured*: the live MFE survival curve for this
        # pattern, updated by whatever this instance has since observed. The
        # grid is floored at the broker's spread-adjusted min_rr because
        # SpreadGate silently rejects anything below it.
        floor_rr = float(params.get("tp_min_rr_floor", 1.75))
        grid = tuple(
            float(g)
            for g in params.get("tp_target_grid", (1.75, 2.0, 2.5, 3.0))
            if float(g) >= floor_rr
        ) or (floor_rr,)

        def prior_curve(r_level):
            return _survival_at(pattern, r_level)

        if learner is not None:
            r_target, expected_r, p_hit = learner.best_target(
                bucket, grid=grid, prior_curve=prior_curve
            )
        else:
            r_target = max(grid, key=lambda r: prior_curve(r) * r - (1.0 - prior_curve(r)))
            p_hit = prior_curve(r_target)
            expected_r = p_hit * r_target - (1.0 - p_hit)

        tp1_points = risk_price * r_target
        # A nearer opposite zone caps the target: price is far likelier to
        # turn at the next real level than to run through it to a round R.
        if zone_target_1 is not None and risk_price * floor_rr <= zone_target_1 < tp1_points:
            tp1_points = zone_target_1

        # One runner, not two. Every extra leg is another round-trip cost, and
        # the live books show transaction cost at 116% of gross profit — the
        # inherited third leg at 4.0R was beyond the p85 MFE of every pattern
        # measured, so it paid spread to never fill.
        runner_extra = float(params.get("tp_runner_extra_rr", 1.0))
        tp2_points = max(tp1_points + risk_price * runner_extra, risk_price * (r_target + 0.5))
        if zone_target_2 is not None and zone_target_2 > tp1_points:
            tp2_points = min(tp2_points, zone_target_2)

        return {
            "bucket": bucket,
            "pattern": pattern,
            "sl_points": float(sl_points),
            "risk_price": float(risk_price),
            "tp1": float(tp1_points),
            "tp2": float(tp2_points),
            "buffer_mult": float(buffer_mult),
            "r_target": float(r_target),
            "p_hit": float(p_hit),
            "expected_r": float(expected_r),
            "rr_room": float(zone_target_1 / risk_price if zone_target_1 is not None else 0.0),
        }

    def _record_candidate(
        self,
        *,
        learner,
        zone,
        life,
        features,
        bucket,
        now_ns,
        close,
        trade_dir,
        sl_points,
        atr_val,
        horizon_ns,
        params,
    ):
        """Record this touch for grading — once per touch episode, whether or
        not it is traded.

        Recording declined setups is what stops the gate becoming a
        self-fulfilling filter: labels come from price, so a family the gate
        has stopped trading keeps accumulating honest evidence and can be
        readmitted when it starts working again."""
        key = f"{zone['family']}|{zone['price_low']:.2f}|{zone['price_high']:.2f}|{life['touches']}"
        if key in self._observed:
            return
        self._observed[key] = now_ns
        limit = int(params.get("learner_max_pending", 400)) * 4
        if len(self._observed) > limit:
            for stale in sorted(self._observed, key=lambda k: self._observed[k])[: limit // 2]:
                del self._observed[stale]
        learner.observe(
            features=features,
            bucket=bucket,
            entry_ns=now_ns,
            entry_price=close,
            direction=trade_dir,
            sl_dist=sl_points,
            atr=atr_val,
            deadline_ns=now_ns + horizon_ns,
        )

    def _build_signals(
        self,
        *,
        zone,
        life,
        plan,
        verdict,
        params,
        zone_frame,
        structure,
        now,
        threshold,
        atr_val,
    ):
        demand = zone["kind"] == ZoneKind.DEMAND
        direction = Direction.BUY if demand else Direction.SELL

        base_idx = min(max(int(zone.get("base_start", 0)), 0), len(zone_frame) - 1)
        annotation = PriceZone(
            kind=zone["kind"],
            pattern=zone.get("pattern"),
            price_low=zone["price_low"],
            price_high=zone["price_high"],
            time_start=zone_frame["time"].iloc[base_idx],
            time_end=now,
        )
        structure_points = tuple(
            StructurePoint(
                time=zone_frame["time"].iloc[min(max(int(s["index"]), 0), len(zone_frame) - 1)],
                price=s["price"],
                label=s["label"],
            )
            for s in structure[-4:]
        )

        readings = [
            IndicatorReading(
                name="target_expected_r",
                value=round(plan["expected_r"], 4),
                threshold=0.0,
                comparison=">",
                passed=plan["expected_r"] > 0.0,
            ),
            IndicatorReading(
                name="target_p_hit",
                value=round(plan["p_hit"], 4),
                threshold=round(1.0 / (1.0 + plan["r_target"]), 4),
                comparison=">",
                passed=plan["p_hit"] > 1.0 / (1.0 + plan["r_target"]),
            ),
            IndicatorReading(
                name="sl_buffer_atr_mult",
                value=plan["buffer_mult"],
                threshold=float(params["sl_zone_buffer_atr_mult"]),
                comparison=">",
                passed=True,
            ),
            IndicatorReading(
                name="zone_touches",
                value=float(life["touches"]),
                threshold=1.0,
                comparison="<",
                passed=life["touches"] <= 1,
            ),
        ]
        if verdict is not None:
            readings.insert(
                0,
                IndicatorReading(
                    name="learner_p_secure",
                    value=round(verdict.p_secure, 4),
                    threshold=round(threshold, 4),
                    comparison=">",
                    passed=verdict.p_secure >= threshold,
                ),
            )
            readings.insert(
                1,
                IndicatorReading(
                    name="learner_bucket_samples",
                    value=round(verdict.bucket_samples, 1),
                    threshold=float(params["learner_min_bucket_samples"]),
                    comparison=">",
                    passed=verdict.ready,
                ),
            )
            learn_note = (
                f" p={verdict.p_secure:.3f}/{threshold:.3f} n={verdict.bucket_samples:.0f}"
                f"{'' if verdict.ready else ' (prior)'}"
            )
        else:
            learn_note = " learner=off"

        base_reason = (
            f"{zone['family']} [{zone['price_low']:.2f},{zone['price_high']:.2f}] "
            f"sl={plan['sl_points']:.2f} atr={atr_val:.2f} touch={life['touches']}"
            f" target={plan['r_target']:.2f}R pHit={plan['p_hit']:.3f}"
            f" E[R]={plan['expected_r']:+.3f}{learn_note}"
        )
        readings_tuple = tuple(readings)

        # Two legs, not three: every extra leg pays another round-trip, and
        # live transaction cost already runs at 116% of gross profit.
        return (
            Signal(
                direction=direction,
                sl_points=plan["sl_points"],
                tp_points=plan["tp1"],
                confidence=_confidence(verdict, 0.75),
                reason=f"{base_reason} tp={plan['tp1']:.2f} (Target)",
                zone=annotation,
                pattern=zone.get("pattern"),
                structure=structure_points,
                indicators=readings_tuple,
            ),
            Signal(
                direction=direction,
                sl_points=plan["sl_points"],
                tp_points=plan["tp2"],
                confidence=_confidence(verdict, 0.65),
                reason=f"{base_reason} tp={plan['tp2']:.2f} (Runner)",
                zone=annotation,
                pattern=zone.get("pattern"),
                structure=structure_points,
                indicators=readings_tuple,
            ),
        )


def _blended_expectancy(*, p_hit, r_target, p_secure, secure_r):
    """Expected R under the exit process this engine actually runs.

    A two-outcome bracket model (target or stop) is the wrong model here, and
    following it would refuse every trade: measured on live history, expected
    R is negative at every target level from 1.0R to 3.0R for all seven
    patterns. That is not because the setups are worthless — it is because
    almost nothing closes at its take-profit. `PositionManager`'s secure-base
    trailing books roughly `secure_r` long before the target is reached.

    So there are three outcomes, not two:

        reaches the target        p_hit                  -> +r_target
        secures but stalls        p_secure - p_hit       -> +secure_r
        stopped                   1 - p_secure           -> -1

        E[R] = p_hit*r_target + (p_secure - p_hit)*secure_r - (1 - p_secure)

    Sanity check against a real logged decision (p_hit 0.064, target 1.75R,
    p_secure 0.826): this returns +0.090, and the backtest that produced it
    measured avg R of 0.07-0.09. The model predicts what the engine does.

    `p_hit` is clamped below `p_secure` because reaching the target without
    first travelling `secure_r` is impossible; a prior-driven estimate that
    violated it would otherwise manufacture expectancy out of arithmetic."""
    p_hit = min(max(p_hit, 0.0), max(p_secure, 0.0))
    return p_hit * r_target + (p_secure - p_hit) * secure_r - (1.0 - p_secure)


def _confidence(verdict, base):
    """Confidence tracks the learner's probability once it has evidence, and
    falls back to the leg's fixed prior while it is still warming up."""
    if verdict is None or not verdict.ready:
        return base
    return float(np.clip(verdict.p_secure, 0.05, 0.99))


def _rank_key(verdict, life):
    """Best candidate first: highest probability, then the freshest zone.

    Fresh means untested and recently formed — fewer touches first, then
    younger. Freshness breaks ties while the learner is still running on
    priors alone, which keeps behaviour stable and sensible before it has
    local evidence of its own."""
    probability = verdict.p_secure if verdict is not None else 0.0
    return (probability, -life["touches"], -life["age_bars"])
