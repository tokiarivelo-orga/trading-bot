"""XAUUSD Institutional-Order-Flow microstructure scalper — M1, self-learning,
strategy-driven exits.

A fork of `xauusd_snd_adaptive_m1_v1` (same zone/Quasimodo/SNRC/sweep/breaker
detectors, same `AdaptiveLearner` wiring, same `_session_for`/feature-vector
skeleton — copied rather than imported because the sandbox only allows
importing `src.strategies.domain.{models,online_learning,fatigue}`, see
`strategies/sandbox.py::ALLOWED_IMPORT_MODULES`) that adds the genuine
microstructure signals nothing else in this repo covers yet:

  FVG / IMBALANCE   classic 3-candle fair value gap. A rectangle the same
                    shape as every other zone here (DEMAND/SUPPLY, high/low,
                    a confirming bar), tagged `pattern="FVG_BULL"/"FVG_BEAR"`,
                    fed into the same lifecycle tracker so an FVG retest is a
                    tradable candidate exactly like an S&D or Quasimodo zone.
  VOLUME DELTA      There is no real bid/ask volume in this data — only
                    `tick_volume`, a broker tick count (confirmed against
                    `market_data/adapters/candle_repository.py`). `_volume_
                    delta` is therefore a PROXY, not order flow: it splits
                    each bar's tick_volume by where its close sits in its own
                    high/low range. A close near the high implies buyers won
                    that bar, near the low implies sellers did. Treat the
                    resulting delta as a directional lean, never a fill count.
  ABSORPTION        High participation (tick_volume well above its trailing
                    average) that failed to move price (small body vs ATR)
                    while the volume-delta proxy above was strongly one-sided
                    — the OHLC-only stand-in for "one side's resting orders
                    absorbed the other side's aggression".
  KILLZONE          A narrow flag for the opening ~45 minutes of the London
                    and New York sessions, layered on top of (not replacing)
                    `_session_for`'s existing four-way session bucket, which
                    stays byte-identical to the base file so its priors keep
                    meaning what they were measured to mean.

All three become (a) new tokens on the learner's bucket-key string and
(b) new dimensions in the fixed-length feature vector `AdaptiveLearner.score
()`/`.observe()` sees — zero learner-side change, per its own design (a cold
bucket/dimension just falls back to the base rate/weight of zero).

────────────────────────────────────────────────────────────────────────
EXIT DECISIONS — the other genuinely new capability (Phase A of this plan)
────────────────────────────────────────────────────────────────────────
`ctx.own_position` (a `PositionSnapshot`) is populated by the engine with
this bot's own open position on the symbol, if any. `evaluate()` reads it
every call and may emit an `ExitDecision` instead of (or alongside) a
`Signal`:

  CLOSE       a fresh *opposing* FVG just printed — a brand-new imbalance in
              the direction against the open position. Restricted to FVG
              specifically (not any zone family) because a freshly-formed
              imbalance is the one unambiguous "a new opposite-direction move
              just happened on this very bar" signal available here — exactly
              the thesis-invalidation case this hook exists for, per
              `engine/domain/exit_policy.py`'s docstring on why exits must be
              deterministic setup-invalidation, not a prediction.
  BREAKEVEN   price has already moved in the position's favor AND a fresh
              same-direction zone (an FVG, or any of the S&D/Quasimodo/SNRC/
              sweep families) has just formed — read as continuation
              confirmation, so the stop moves to entry. Never a generic
              profit-lock: `PositionManager`'s give-back trail and volatility
              guard already run underneath this, on every position, and this
              hook is deliberately narrower than either.

`learner_enabled` and `exit_decisions_enabled` are both clean on/off params
for a controlled A/B, same pattern as the base file.

v1 — initial implementation.

v2 — adds a bounded, confidence-gated SIZING multiplier on top of v1 — not a
  new entry path, and it never opens an extra position or bypasses
  `configs/risk.yaml`'s caps (see CLAUDE.md — risk caps stay user-owned;
  this only changes the `risk_multiplier` this bot's own signals hand to
  `RiskManager.size_position()`). Data-mined from this file's OWN closed
  trades for strategy_version `xauusd_iof_scalp_m1:v1` (side=sell):
  pattern=FVG_BEAR, zone_kind=supply, session=new_york (UTC 16:00-21:00,
  see `_session_for` below) — n=106 closed trades spanning 52 independent
  5-minute open-time episodes (~2 trades/episode, not a re-entry-spam
  artifact) over a real 6-day date spread (2026-08-12 to 2026-08-18).
  Splitting at the median confidence (>=0.7 vs 0.6-0.7, 53/53 trades): win
  rate is FLAT at 35.8% in both buckets, but profit factor improves 1.27 ->
  1.48 (avg win $15.39 vs avg loss $6.18 in the high-confidence bucket,
  worst single trade -$15.90 — no fat-tail risk in the sample). This is a
  PAYOFF-RATIO edge (bigger winners at higher confidence), not a hit-rate
  edge — re-verified directly against `trades` before shipping this
  version, not taken on faith from the original data-mining pass.
  REACHABLE: `_detect_fvg_zones` actively constructs FVG_BEAR/SUPPLY zones
  every bar (see below); confidence here is a genuine graduated score from
  `_prior_logit`/`AdaptiveLearner.score()` (additive log-odds over
  pattern/session/volatility/trend priors), NOT a fixed per-leg constant —
  unlike the `xauusd_snd_qm_structure_fixed_m1`/`_adaptive_m1` sizing
  attempts earlier tonight, so a confidence-tiered gate is meaningful here,
  and both the TP1 ("Target", base confidence 0.75) and TP2 ("Runner", base
  confidence 0.65) legs `_build_signals` emits are checked independently
  against the gate (each leg's own `_confidence()` value is what is
  compared, not a shared bucket-level number) — TP1 clears >=0.7 even with
  the learner off/cold (its static base is 0.75), TP2 only clears once the
  learner is warm and pushes that leg's own `p_secure` to >=0.7.
  Floor-lock check (interactive, against the live account, 2026-08-19,
  using the real `RiskManager.size_position` math against this bucket's
  actual historical SL distances — not assumption): at the account's
  balance ($5,142.82) and XAUUSD's contract_size=100/volume_min=0.01/
  volume_step=0.01, NONE of the 106 qualifying trades' SL distances
  (0.5%-risk lots range 0.03-0.25, avg 0.1286 at the baseline 1.0x
  multiplier) round down to the 0.01-lot broker floor — this bucket's SL
  distances (~$1-8 in XAUUSD price, i.e. ~100-800 points at a $100
  contract) are simply too large relative to this account's risk budget to
  ever floor-lock, unlike the `xauusd_snd_qm_structure_adaptive_m1`/M1
  case earlier tonight where the multiplier was fully inert. At 1.2x every
  qualifying trade's lot size scales up proportionally (avg lot
  0.1286 -> 0.1552, +20.7%, no floor-locking at either multiplier) — a
  real, visible effect.
  Chosen multiplier: 1.2x — deliberately more conservative than the
  `xauusd_snd_qm_structure_fixed_m1_v4` sibling's 1.3x, because that
  finding had a much larger win-rate gradient (42.2%->71.8%) where this one
  has a flat win rate and only a moderate profit-factor gradient (a real
  but smaller edge), and well under the 1.5x conservative ceiling for a
  payoff-ratio-only finding. `evaluate()`'s signal builder
  (`_build_signals`) sets `Signal.size_multiplier` to
  `sizing_fvg_bear_supply_ny_multiplier` (default 1.2) for a SELL/SUPPLY
  signal whose `pattern == "FVG_BEAR"`, whose session is `new_york`, and
  whose own leg confidence is >=
  `sizing_fvg_bear_supply_ny_min_confidence` (default 0.7, matching the
  mined median split) — see
  `_high_conviction_fvg_bear_size_multiplier()`. Every other signal (every
  BUY/demand signal, every other pattern/session combination, the CHoCH
  opposing-exit `Signal` in `_choch_exit_signal` which carries no
  `pattern`) is unaffected and keeps the default 1.0 multiplier.
"""

import numpy as np
import pandas as pd

from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
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
# Indicator helpers (copied verbatim from xauusd_snd_adaptive_m1_v1 — the
# sandbox allowlist has no room for a shared indicators module, see the
# module docstring)
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
    a cluster of `min_touches` or more is a *strong* level."""
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
    through a zone's band at that zone's bar."""
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
    own family label."""
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
    """Doctrinal Quasimodo, plus its QMC / QM2P / QMM variants."""
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
            and head[1] > shoulder[1]
            and breaker[1] < neck[1]
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
    """Re-label V1 bases that sit on a strong SnR level. Mutates in place."""
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
    """SNRC3 — the "war" between engulfings at a strong level."""
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
            if failed_bull:
                broke = np.flatnonzero(closes[i + 1 : stop] < lows[i])
            else:
                broke = np.flatnonzero(closes[i + 1 : stop] > highs[i])
            if len(broke) == 0:
                continue
            fail_idx = int(broke[0]) + i + 1
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
    """Price wicks through a prior swing extreme and closes back inside."""
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
# NEW — Fair Value Gap / imbalance (detected on the entry timeframe)
# ─────────────────────────────────────────────────────────────────────


def _detect_fvg_zones(df, atr_series, params):
    """Classic 3-candle fair value gap / imbalance: bullish when
    `low[i] > high[i-2]` (the middle candle's impulse left a void between
    candles i-2 and i that price never traded through), bearish for the
    mirror `high[i] < low[i-2]`.

    Represented as the same rectangle shape every other detector here
    produces — DEMAND for a bullish gap (it acts as support on a retest),
    SUPPLY for bearish — tagged `pattern="FVG_BULL"/"FVG_BEAR"`, so it flows
    through the exact same lifecycle tracker (`_track_zone`), candidate
    selection and learner pipeline as an S&D or Quasimodo zone. `conf_idx`/
    `leg_out_end`/`base_start` are indices into `df` (the entry timeframe)
    directly — unlike every other detector in this file, which indexes into
    the resampled `zone_frame` — because an FVG is inherently an
    entry-timeframe microstructure event, not a higher-timeframe structure
    one. Callers that need to know which index space a zone belongs to check
    `zone["source"] == "FVG"` (see `_select_candidates`/`_zone_freshly_
    formed`/`_build_signals`)."""
    atr_filled = _atr_filled(atr_series)
    if atr_filled is None:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)
    min_gap_atr = float(params.get("fvg_min_gap_atr_mult", 0.05))

    zones = []
    for i in range(2, n):
        atr_at = atr_filled[i]
        if atr_at <= 0:
            continue
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if gap < min_gap_atr * atr_at:
                continue
            zones.append(
                {
                    "source": "FVG",
                    "family": "FVG",
                    "pattern": "FVG_BULL",
                    "kind": ZoneKind.DEMAND,
                    "price_high": float(lows[i]),
                    "price_low": float(highs[i - 2]),
                    "base_start": i - 2,
                    "conf_idx": i,
                    "leg_out_end": i,
                    "impulse_atr": float(gap / atr_at),
                    "needs_htf": False,
                }
            )
        elif highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if gap < min_gap_atr * atr_at:
                continue
            zones.append(
                {
                    "source": "FVG",
                    "family": "FVG",
                    "pattern": "FVG_BEAR",
                    "kind": ZoneKind.SUPPLY,
                    "price_high": float(lows[i - 2]),
                    "price_low": float(highs[i]),
                    "base_start": i - 2,
                    "conf_idx": i,
                    "leg_out_end": i,
                    "impulse_atr": float(gap / atr_at),
                    "needs_htf": False,
                }
            )
    return zones


def _zone_freshly_formed(zone, zone_frame_len, entry_len):
    """Whether `zone`'s confirming bar is the most recent bar in whichever
    index space it was detected in — FVGs are detected directly on the entry
    timeframe (see `_detect_fvg_zones`), everything else on the resampled
    zone timeframe."""
    conf_idx = zone.get("conf_idx")
    if conf_idx is None:
        return False
    if zone.get("source") == "FVG":
        return conf_idx == entry_len - 1
    return conf_idx == zone_frame_len - 1


def _nearest_fvg_distance_atr(live_zones, close, trade_dir, atr_val, sentinel):
    """Distance in ATR from `close` to the nearest still-live (unfilled) FVG
    whose polarity matches `trade_dir` — the continuous "is there FVG
    confluence nearby" signal that feeds both the feature vector and the
    bucket-key confluence flag. `sentinel` (a large ATR distance) stands in
    for "no such FVG exists", the same convention `extension_atr`/`rr_room`
    use elsewhere in this file's feature vector."""
    if atr_val <= 0:
        return float(sentinel)
    same_kind = ZoneKind.DEMAND if trade_dir == 1 else ZoneKind.SUPPLY
    best = float(sentinel)
    for zone in live_zones:
        if zone.get("source") != "FVG" or zone["kind"] != same_kind:
            continue
        mid = (zone["price_low"] + zone["price_high"]) / 2.0
        dist = abs(close - mid) / atr_val
        if dist < best:
            best = dist
    return float(best)


# ─────────────────────────────────────────────────────────────────────
# NEW — Volume-delta proxy & absorption
# ─────────────────────────────────────────────────────────────────────


def _volume_delta(df):
    """PROXY volume delta, not real order flow. There is no bid/ask volume in
    this data — `tick_volume` is a broker tick count. Each bar's tick_volume
    is split by where its close sits within its own high/low range: a close
    near the high implies buyers dominated that bar (`buy_vol`), near the low
    implies sellers did (`sell_vol`). `delta = buy_vol - sell_vol` is
    therefore a directional lean inferred from OHLC shape, never a measured
    fill count. Zero-range bars (`high == low`, e.g. an illiquid tick) are
    guarded explicitly: they carry no directional information, so they split
    their volume 50/50 rather than producing a NaN/inf from a zero
    denominator."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    volume = (
        df["tick_volume"].to_numpy(dtype=float) if "tick_volume" in df.columns else np.ones(len(df))
    )
    rng = highs - lows
    safe_rng = np.where(rng > 0, rng, 1.0)
    buy_frac = np.where(rng > 0, (closes - lows) / safe_rng, 0.5)
    buy_frac = np.clip(buy_frac, 0.0, 1.0)
    buy_vol = volume * buy_frac
    sell_vol = volume * (1.0 - buy_frac)
    delta = buy_vol - sell_vol
    return buy_vol, sell_vol, delta


def _microstructure_state(df, atr_val, params):
    """Per-bar volume-delta ratio/z-score plus the absorption proxy — high
    participation (`tick_volume` well above its trailing average) that failed
    to move price (small body vs ATR) while the delta proxy above leaned
    strongly one way. In real order-flow reading this shape means one side's
    resting orders absorbed the other side's aggression; here it is inferred
    purely from OHLC + tick_volume (see `_volume_delta`), so treat it as a
    lean, not a certainty."""
    buy_vol, sell_vol, delta = _volume_delta(df)
    total = buy_vol + sell_vol
    safe_total = np.where(total > 0, total, 1.0)
    delta_ratio = np.where(total > 0, delta / safe_total, 0.0)

    window = int(params.get("volume_lookback_bars", 20))
    min_periods = max(5, window // 2)
    delta_series = pd.Series(delta)
    mean = delta_series.rolling(window, min_periods=min_periods).mean()
    std = delta_series.rolling(window, min_periods=min_periods).std(ddof=0)
    delta_z = ((delta_series - mean) / std.replace(0.0, np.nan)).fillna(0.0).to_numpy()

    volume = (
        df["tick_volume"].to_numpy(dtype=float) if "tick_volume" in df.columns else np.ones(len(df))
    )
    vol_baseline = pd.Series(volume).rolling(window, min_periods=min_periods).mean().to_numpy()
    safe_baseline = np.where((vol_baseline > 0) & np.isfinite(vol_baseline), vol_baseline, 1.0)
    vol_ratio = np.where(
        (vol_baseline > 0) & np.isfinite(vol_baseline), volume / safe_baseline, 1.0
    )

    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    body_atr = np.abs(closes - opens) / max(atr_val, 1e-9)
    absorption_score = vol_ratio * np.abs(delta_ratio) / (1.0 + body_atr)

    min_vol_ratio = float(params.get("absorption_min_vol_ratio", 1.5))
    max_body_atr = float(params.get("absorption_max_body_atr", 0.35))
    min_one_sided = float(params.get("absorption_min_one_sided", 0.4))
    absorption_flag = (
        (vol_ratio >= min_vol_ratio)
        & (body_atr <= max_body_atr)
        & (np.abs(delta_ratio) >= min_one_sided)
    )
    return {
        "delta": delta,
        "delta_ratio": delta_ratio,
        "delta_z": delta_z,
        "vol_ratio": vol_ratio,
        "absorption_score": absorption_score,
        "absorption_flag": absorption_flag,
    }


# ─────────────────────────────────────────────────────────────────────
# NEW — Killzone (session-open) flag
# ─────────────────────────────────────────────────────────────────────


def _killzone_flag(hour, minute, params):
    """True during the opening window (default 45 minutes) of the London or
    New York session — layered on top of, not replacing, `_session_for`'s
    four-way session bucket below. Kept as a separate function rather than
    folded into `_session_for` itself so that function's string output stays
    byte-identical to the base file's, which matters because `SESSION_PRIOR`
    was measured against exactly those four session names; changing what
    `_session_for` returns would silently break that mapping."""
    minute_of_day = int(hour) * 60 + int(minute)
    window = int(params.get("killzone_window_minutes", 45))
    london_start = 7 * 60
    ny_start = 16 * 60
    return bool(
        london_start <= minute_of_day < london_start + window
        or ny_start <= minute_of_day < ny_start + window
    )


# ─────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────


def _volatility_bucket(atr_values, params):
    """Where current ATR sits in its own trailing distribution — 0 quiet,
    1 normal, 2 expanded."""
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
    separation."""
    fast_period = int(params.get("trend_ema_fast", 12))
    slow_period = int(params.get("trend_ema_slow", 34))
    if len(closes) < slow_period + 2:
        return 0, 0.0
    fast = _ema(closes, fast_period)
    slow = _ema(closes, slow_period)
    separation = float(fast[-1] - slow[-1])
    return (1 if separation > 0 else -1 if separation < 0 else 0), separation


def _trend_bucket(adx_value, trend_dir, trade_dir, params):
    """0 = against a real trend, 1 = no trend worth the name, 2 = with it."""
    threshold = params.get("adx_trend_threshold", 20.0)
    if not np.isfinite(adx_value) or adx_value < threshold or trend_dir == 0:
        return 1
    return 2 if trend_dir == trade_dir else 0


# ─────────────────────────────────────────────────────────────────────
# Priors distilled from live trading history (same measured numbers the base
# bot uses — this family shares the live journal until it earns its own)
# ─────────────────────────────────────────────────────────────────────
#
# IMPORTANT — FVG has no live track record: it is not in PATTERN_PRIOR, so
# `_pattern_key` returns None for it and it falls through to
# UNPROVEN_FAMILY_PRIOR, exactly like every other family this bot detects
# that the journal has never seen. It has to earn its place from the
# learner's own online evidence rather than trade freely on an assumed edge.
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

UNPROVEN_FAMILY_PRIOR = -0.30

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
    or None when this bot detects something the journal has never seen."""
    pattern = str(zone.get("pattern") or "")
    if pattern.startswith("BREAKER_"):
        return None
    if pattern == "SZ_V2":
        return "DZ_V2"
    if pattern in PATTERN_PRIOR:
        return pattern
    return None


def _prior_logit(pattern, session, volatility, trend, known_family):
    """Additive log-odds prior for one setup."""
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
    the priors above were measured from. Byte-identical to the base file —
    see `_killzone_flag`'s docstring for why the new narrow-window signal is
    a separate function instead of changing this one."""
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
    """Participation, as ratios rather than raw tick counts (unchanged from
    the base file — `_microstructure_state` above is the new, finer-grained
    volume-delta/absorption reading; this coarser ratio/burst/trend triple
    stays because it still feeds the same feature-vector slots the base
    file's learner weights were shaped around)."""
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
    sequence? Returns +1 (turned bullish), -1 (turned bearish) or 0."""
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
# learner. The first 26 are unchanged from the base file; the last four are
# this strategy's new microstructure dimensions.
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
    "volume_delta_z",
    "absorption_score",
    "fvg_proximity_atr",
    "killzone_flag",
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
    volume_delta_z,
    absorption_score,
    fvg_proximity_atr,
    killzone_flag,
):
    """Thirty scale-free numbers describing one candidate — the base file's
    26, plus this strategy's volume-delta z-score, absorption score, FVG
    proximity and killzone flag."""
    zone_height = max(zone["price_high"] - zone["price_low"], 1e-9)
    if zone["kind"] == ZoneKind.DEMAND:
        depth = (zone["price_high"] - close) / zone_height
    else:
        depth = (close - zone["price_low"]) / zone_height

    body = abs(last_bar["close"] - last_bar["open"])
    bar_range = max(last_bar["high"] - last_bar["low"], 1e-9)
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
            float(np.clip(volume_delta_z, -5.0, 5.0)),
            float(np.clip(absorption_score, 0.0, 10.0)),
            float(np.clip(fvg_proximity_atr, 0.0, 6.0)),
            float(killzone_flag),
        ],
        dtype=float,
    )


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────


class XauusdIofScalpM1:
    def __init__(self):
        self.spec = StrategySpec(
            name="xauusd_iof_scalp_m1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            confirmation_timeframes=("M15",),
            htf_veto=False,
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
                "tp_min_rr_floor": 1.75,
                "tp_target_grid": (1.75, 2.0, 2.5, 3.0),
                "tp_runner_extra_rr": 1.0,
                "tp_buffer_points": 20,
                "tp_buffer_atr_mult": 0.3,
                "point_value": 0.01,
                # ── Volume (base ratio/burst/trend triple) ──
                "volume_lookback_bars": 20,
                # ── NEW — FVG / imbalance ──
                "fvg_min_gap_atr_mult": 0.05,
                "fvg_confluence_atr_mult": 1.0,
                "fvg_sentinel_atr": 6.0,
                # ── NEW — Volume-delta proxy / absorption ──
                "vd_agree_threshold": 0.15,
                "absorption_min_vol_ratio": 1.5,
                "absorption_max_body_atr": 0.35,
                "absorption_min_one_sided": 0.4,
                # ── NEW — Killzone ──
                "killzone_window_minutes": 45,
                # ── Change of character ──
                "choch_exit_enabled": True,
                "choch_break_atr_mult": 0.1,
                "choch_sl_atr_mult": 1.0,
                # ── NEW — Strategy-driven exit decisions (Phase A channel) ──
                # Clean A/B kill switch, same pattern as `learner_enabled`.
                "exit_decisions_enabled": True,
                # ── v2 — high-conviction FVG_BEAR/supply/new_york sizing
                # tier (see module docstring for the data-mining writeup
                # and the floor-lock check) ──
                "sizing_fvg_bear_supply_ny_multiplier": 1.2,
                "sizing_fvg_bear_supply_ny_min_confidence": 0.7,
                # ── Regime gates (validated out-of-sample on the base bot;
                # inherited here since the underlying detectors are the
                # same — re-validate independently once this bot has its
                # own live history, per the plan's Phase C) ──
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
                "learner_global_prior_rate": 0.62,
                "learner_gate_p": 0.65,
                "min_expectancy_r": 0.02,
                "learner_max_pending": 400,
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
        """Forget everything learned."""
        if self._learner is not None:
            self._learner.reset()
        self._last_bar_ns = None
        self._observed = {}
        self._last_choch = 0

    def _check_continuity(self, first_ns, last_ns):
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

        # Entry-timeframe ATR — the FVG detector and the volume-delta/
        # absorption/killzone features all operate at M1 scale, not the
        # 5-minute zone_frame scale everything else here uses.
        entry_atr_series = _atr(df, atr_period)
        entry_atr_clean = entry_atr_series.dropna()
        entry_atr_val = float(entry_atr_clean.iloc[-1]) if not entry_atr_clean.empty else atr_val

        entry_highs = df["high"].to_numpy()
        entry_lows = df["low"].to_numpy()
        entry_closes = df["close"].to_numpy()

        learner_on = bool(params.get("learner_enabled", True))
        learner = self._ensure_learner(params) if learner_on else None
        if learner is not None:
            learner.advance(entry_t_ns, entry_highs, entry_lows)
        self._last_bar_ns = int(entry_t_ns[-1])

        zones = self._detect_all(zone_frame, atr_series, atr_val, params, ctx, df, entry_atr_series)

        # ── strategy-driven exit decision on this bot's OWN open position ──
        # Runs every bar regardless of whether a new entry candidate exists —
        # thesis invalidation/continuation is about the position already
        # open, not about today's candidate list. See module docstring.
        if ctx.own_position is not None and bool(params.get("exit_decisions_enabled", True)):
            exit_decision = self._thesis_exit_decision(
                own_position=ctx.own_position,
                zones=zones,
                zone_frame_len=len(zone_frame),
                entry_len=len(df),
                close=float(entry_closes[-1]),
            )
            if exit_decision is not None:
                return exit_decision

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
            entry_atr_val=entry_atr_val,
            candidates=candidates,
            live_zones=live_zones,
            learner=learner,
            entry_t_ns=entry_t_ns,
        )

    # ── strategy-driven exit decisions ──────────────────────────────────

    def _thesis_exit_decision(self, *, own_position, zones, zone_frame_len, entry_len, close):
        """Narrow, setup-specific exit for this bot's OWN open position (see
        module docstring for the CLOSE/BREAKEVEN rules). Never a substitute
        for `PositionManager`'s generic give-back/volatility rules, which
        keep running underneath whatever this leaves behind."""
        dir_sign = 1 if own_position.direction == Direction.BUY else -1
        opposing_kind = ZoneKind.SUPPLY if dir_sign == 1 else ZoneKind.DEMAND
        same_kind = ZoneKind.DEMAND if dir_sign == 1 else ZoneKind.SUPPLY

        fresh_opposing_fvg = any(
            zone["kind"] == opposing_kind
            and zone.get("source") == "FVG"
            and _zone_freshly_formed(zone, zone_frame_len, entry_len)
            for zone in zones
        )
        if fresh_opposing_fvg:
            return ExitDecision(
                action=ExitActionKind.CLOSE,
                reason=(
                    f"fresh opposing FVG printed against the open "
                    f"{own_position.direction.value} position — thesis invalidated"
                ),
            )

        in_favor = (close - own_position.entry_price) * dir_sign > 0
        if in_favor:
            fresh_same_dir = any(
                zone["kind"] == same_kind and _zone_freshly_formed(zone, zone_frame_len, entry_len)
                for zone in zones
            )
            if fresh_same_dir:
                return ExitDecision(
                    action=ExitActionKind.BREAKEVEN,
                    reason=(
                        f"price continuing in favor of the open "
                        f"{own_position.direction.value} position with a fresh "
                        f"same-direction confluence — moving stop to entry"
                    ),
                )
        return None

    # ── detection ─────────────────────────────────────────────────────

    def _detect_all(self, zone_frame, atr_series, atr_val, params, ctx, df, entry_atr_series):
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

        # ── NEW: FVG/imbalance, detected directly on the entry timeframe ──
        zones.extend(_detect_fvg_zones(df, entry_atr_series, params))

        for zone in zones:
            zone.setdefault("needs_htf", False)
        self._swings = swings
        self._levels = levels
        self._trend_dir = trend_dir
        self._htf = self._htf_state(ctx, params)
        return zones

    def _htf_state(self, ctx, params):
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
        that broke into breakers, and collect those price is touching now.

        Two index spaces meet here: every zone_frame-based detector needs
        `zone_end_ns` to translate its own index into an entry-timeframe
        timestamp; FVGs are already indexed directly into `entry_t_ns`."""
        live = []
        candidates = []
        for zone in zones:
            leg_out_end = zone["leg_out_end"]
            if zone.get("source") == "FVG":
                if leg_out_end >= len(entry_t_ns):
                    continue
                start_ns = int(entry_t_ns[leg_out_end])
            else:
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
        entry_atr_val,
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
        minute = int(now.minute)
        session = _session_for(hour)
        killzone = _killzone_flag(hour, minute, params)
        volume = _volume_state(df, params)
        micro = _microstructure_state(df, entry_atr_val, params)
        delta_z_last = float(micro["delta_z"][-1])
        delta_ratio_last = float(micro["delta_ratio"][-1])
        absorption_score_last = float(micro["absorption_score"][-1])
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

        blocked = vol_bucket in tuple(
            params.get("skip_volatility_buckets", ())
        ) or session in tuple(params.get("skip_sessions", ()))

        fvg_sentinel = float(params.get("fvg_sentinel_atr", 6.0))
        fvg_confluence_mult = float(params.get("fvg_confluence_atr_mult", 1.0))
        vd_threshold = float(params.get("vd_agree_threshold", 0.15))

        scored = []
        for zone, life in candidates:
            demand = zone["kind"] == ZoneKind.DEMAND
            trade_dir = 1 if demand else -1

            if blocked or zone["family"] in tuple(params.get("family_denylist", ())):
                continue

            if choch == -trade_dir:
                continue

            if zone.get("needs_htf", False):
                confirmed = self._htf["bull"] if demand else self._htf["bear"]
                if not confirmed:
                    continue

            fvg_proximity_atr = _nearest_fvg_distance_atr(
                live_zones, close, trade_dir, entry_atr_val, fvg_sentinel
            )
            fvg_confluence = 1 if fvg_proximity_atr <= fvg_confluence_mult else 0
            delta_agree = delta_ratio_last * trade_dir
            if delta_agree >= vd_threshold:
                vd_bucket = 1
            elif delta_agree <= -vd_threshold:
                vd_bucket = -1
            else:
                vd_bucket = 0

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
                fvg_confluence=fvg_confluence,
                vd_bucket=vd_bucket,
                killzone_flag=int(killzone),
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
                volume_delta_z=delta_z_last,
                absorption_score=absorption_score_last,
                fvg_proximity_atr=fvg_proximity_atr,
                killzone_flag=1.0 if killzone else 0.0,
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
                df=df,
                structure=structure,
                now=now,
                threshold=threshold,
                atr_val=atr_val,
                session=session,
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
        `close_on_opposite_signal` path closes a position that is going
        wrong. Unrelated to the new `ExitDecision` channel above — this is
        the base file's original exit lever (a fresh opposite `Signal`),
        kept because it still catches the "no candidate scored this bar, but
        structure just broke" case the FVG-driven `ExitDecision` checks do
        not cover."""
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
        fvg_confluence,
        vd_bucket,
        killzone_flag,
    ):
        """Stop, take-profit ladder and learner bucket for one candidate.

        The bucket key gains three new tokens over the base file's
        `family|v{vol}|t{trend}|{session}`: `fvg{0/1}` (is there a live,
        same-direction FVG within `fvg_confluence_atr_mult` ATR of price
        right now), `vd{-1/0/1}` (does the volume-delta proxy agree with the
        trade direction), `kz{0/1}` (killzone). All three are cheap,
        genuinely new conditioning dimensions the learner did not have
        before — a cold combination just falls back to the base rate, per
        `AdaptiveLearner`'s own design."""
        demand = zone["kind"] == ZoneKind.DEMAND
        zone_height = zone["price_high"] - zone["price_low"]
        if zone_height <= 0:
            return None

        pattern = _pattern_key(zone)
        trend_slot = _trend_bucket(adx_value, self._trend_dir, trade_dir, params)
        bucket = (
            f"{zone['family']}|v{vol_bucket}|t{trend_slot}|{session}"
            f"|fvg{fvg_confluence}|vd{vd_bucket}|kz{killzone_flag}"
        )

        base_buffer_mult = float(params.get("sl_zone_buffer_atr_mult", 0.15))
        buffer_mult = base_buffer_mult
        if learner is not None:
            learned = learner.adverse_excursion_atr(
                bucket, float(params.get("sl_buffer_learned_quantile", 0.8))
            )
            if learned is not None:
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
        if zone_target_1 is not None and risk_price * floor_rr <= zone_target_1 < tp1_points:
            tp1_points = zone_target_1

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
        df,
        structure,
        now,
        threshold,
        atr_val,
        session,
    ):
        demand = zone["kind"] == ZoneKind.DEMAND
        direction = Direction.BUY if demand else Direction.SELL

        # FVG zones index into `df` (entry timeframe); every other detector
        # indexes into `zone_frame` — see `_detect_fvg_zones`'s docstring.
        annotation_frame = df if zone.get("source") == "FVG" else zone_frame
        base_idx = min(max(int(zone.get("base_start", 0)), 0), len(annotation_frame) - 1)
        annotation = PriceZone(
            kind=zone["kind"],
            pattern=zone.get("pattern"),
            price_low=zone["price_low"],
            price_high=zone["price_high"],
            time_start=annotation_frame["time"].iloc[base_idx],
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

        # v2 — high-conviction FVG_BEAR/supply/new_york sizing tier (see
        # module docstring / _high_conviction_fvg_bear_size_multiplier).
        # Each TP leg's own confidence decides independently whether it
        # qualifies — TP1's static 0.75 base clears the default 0.7 gate
        # even with the learner off/cold; TP2's 0.65 base only clears once
        # the learner pushes that leg's own p_secure to >=0.7.
        confidence_tp1 = _confidence(verdict, 0.75)
        confidence_tp2 = _confidence(verdict, 0.65)
        mult_tp1 = _high_conviction_fvg_bear_size_multiplier(
            zone.get("pattern"), zone["kind"], session, confidence_tp1, params
        )
        mult_tp2 = _high_conviction_fvg_bear_size_multiplier(
            zone.get("pattern"), zone["kind"], session, confidence_tp2, params
        )

        return (
            Signal(
                direction=direction,
                sl_points=plan["sl_points"],
                tp_points=plan["tp1"],
                confidence=confidence_tp1,
                reason=f"{base_reason} tp={plan['tp1']:.2f} (Target)",
                zone=annotation,
                pattern=zone.get("pattern"),
                structure=structure_points,
                indicators=readings_tuple,
                size_multiplier=mult_tp1,
            ),
            Signal(
                direction=direction,
                sl_points=plan["sl_points"],
                tp_points=plan["tp2"],
                confidence=confidence_tp2,
                reason=f"{base_reason} tp={plan['tp2']:.2f} (Runner)",
                zone=annotation,
                pattern=zone.get("pattern"),
                structure=structure_points,
                indicators=readings_tuple,
                size_multiplier=mult_tp2,
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# Zone lifecycle on the entry timeframe (shared by both zone_frame-based
# zones and entry-tf FVGs — see `_select_candidates`)
# ─────────────────────────────────────────────────────────────────────


def _track_zone(zone, start_ns, entry_t_ns, highs, lows, closes):
    """Follow one zone forward on entry-TF bars from the moment it became
    valid. Returns a lifecycle dict, or None if it is not yet in the window."""
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
        "fresh_touch": bool(live_now and (len(inside) < 2 or not inside[-2])),
        "age_bars": int(n - 1 - start),
    }


def _flip_to_breaker(zone, break_ns):
    """A broken zone does not stop existing — it changes sign."""
    flipped = dict(zone)
    flipped["kind"] = ZoneKind.SUPPLY if zone["kind"] == ZoneKind.DEMAND else ZoneKind.DEMAND
    flipped["source"] = "BREAKER"
    flipped["family"] = "BREAKER"
    flipped["pattern"] = "BREAKER_" + str(zone.get("pattern", ""))
    flipped["needs_htf"] = True
    flipped["valid_from_ns"] = break_ns
    return flipped


def _blended_expectancy(*, p_hit, r_target, p_secure, secure_r):
    """Expected R under the exit process this engine actually runs — see the
    base file (`xauusd_snd_adaptive_m1_v1`) for the full derivation, which
    this is a verbatim copy of."""
    p_hit = min(max(p_hit, 0.0), max(p_secure, 0.0))
    return p_hit * r_target + (p_secure - p_hit) * secure_r - (1.0 - p_secure)


def _confidence(verdict, base):
    if verdict is None or not verdict.ready:
        return base
    return float(np.clip(verdict.p_secure, 0.05, 0.99))


def _high_conviction_fvg_bear_size_multiplier(pattern, zone_kind, session, confidence, params):
    """Bounded per-signal risk-amount multiplier (v2) — see the module
    docstring for the full data-mining writeup and floor-lock check.
    Validated ONLY for SELL/supply FVG_BEAR signals in the new_york
    session, mined from this strategy's OWN closed trades
    (`xauusd_iof_scalp_m1:v1`). Every other pattern/session/direction
    combination — including every BUY/demand signal, for which no
    equivalent finding was validated — keeps the default 1.0 multiplier
    from `Signal`."""
    if zone_kind != ZoneKind.SUPPLY:
        return 1.0
    if pattern != "FVG_BEAR" or session != "new_york":
        return 1.0
    min_confidence = float(params.get("sizing_fvg_bear_supply_ny_min_confidence", 0.7))
    if confidence < min_confidence:
        return 1.0
    return float(params.get("sizing_fvg_bear_supply_ny_multiplier", 1.2))


def _rank_key(verdict, life):
    probability = verdict.p_secure if verdict is not None else 0.0
    return (probability, -life["touches"], -life["age_bars"])
