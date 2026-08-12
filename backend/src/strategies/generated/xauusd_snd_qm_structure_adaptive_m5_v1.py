"""XAUUSD adaptive S&D / Quasimodo / SNRC — M1, self-learning with
multi-regime awareness and confluence scoring.

An enhanced superset of `xauusd_snd_qm_structure_fixed_m1` that learns
online which entry families and market conditions are working *right now*
and dynamically adjusts stop buffers, take-profit ladders, and entry
gating to match the current market regime.

────────────────────────────────────────────────────────────────────────
WHAT MAKES THIS DIFFERENT FROM `xauusd_snd_adaptive_m1`
────────────────────────────────────────────────────────────────────────

The first adaptive strategy (v1) added online learning with volatility
and trend regime bucketing. This version goes further with:

  1. **Session-aware learning**: Gold behaves very differently in Asian,
     London, NY, and overlap sessions. The learner bucket key now
     includes a session dimension (s0–s4), so a zone family that works
     in London but fails in Asia is gated in Asia without affecting
     London's acceptance rate.

  2. **Momentum filtering via RSI**: Entries are checked against a
     14-period RSI. Buying into an overbought zone or selling into an
     oversold one costs confluence points and may be declined entirely.

  3. **Consolidation awareness via Bollinger Band width**: The strategy
     measures whether the market is ranging (narrow bands) or expanding
     (wide bands). S&D zones are stronger in consolidation; structure
     breaks matter more in expansion.

  4. **Confluence scoring**: Before the learner gate, every candidate
     accumulates a score from structure alignment, HTF agreement, zone
     freshness, momentum, and session quality. Candidates below a
     minimum confluence threshold (default 2) are skipped entirely,
     regardless of what the learner thinks — no amount of good history
     can rescue a setup with zero supporting evidence right now.

  5. **Expanded feature vector (25 features)**: Five new scale-free
     features feed the online logistic model: cyclical session encoding
     (sin/cos), normalized RSI, ATR-normalized Bollinger width, and
     zone confluence count. These give the model information about
     time-of-day effects and compression/expansion that the 20-feature
     version was blind to.

────────────────────────────────────────────────────────────────────────
ENTRY FAMILIES
────────────────────────────────────────────────────────────────────────
All entry families from the PoB doctrine are detected:

  SND_V1     classic leg-base-leg: RBR, DBR (demand) / DBD, RBD (supply)
  SND_V2     engulfing-base departure block
  QM_LEGACY  baseline bot's generic swing-failure Quasimodo
  QMR / QML  doctrinal Quasimodo reversal
  QMC        Quasimodo continuation (with prevailing trend)
  QM2P       Quasimodo with two-point trendline confluence
  QMM        failed Quasimodo polarity flip
  SNRC1/2    S&R continuation (requires HTF confirmation)
  SNRC3      engulfing war at strong level
  SWEEP      liquidity sweep / stop hunt
  BREAKER    broken zone polarity flip (swap block)

────────────────────────────────────────────────────────────────────────
ADAPTATION MECHANICS
────────────────────────────────────────────────────────────────────────
Same self-labeling triple-barrier learner as the base adaptive strategy.
The label is "did price reach +0.2R before touching the stop" — matching
the engine's PositionManager secure-base trailing. Break-even probability
is 0.833 at the default buffer.

The learner bucket key is now `{family}|v{vol}|t{trend}|s{session}`,
giving each entry family independent performance tracking across all
three regime dimensions simultaneously.

Evidence before authority: every accessor returns the base parameter
until its evidence threshold is met. Cold = un-adapted. Failure mode
is always "no adaptation", never "confident nonsense from 4 samples".

v1 — initial implementation.
"""

import math
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
from src.strategies.domain.online_learning import (
    AdaptiveLearner,
    LearnerConfig,
    breakeven_probability,
)

_NS_PER_MINUTE = 60 * 1_000_000_000
_RESET_GAP_NS = 30 * 24 * 60 * _NS_PER_MINUTE

N_FEATURES = 31
FEATURE_NAMES = (
    "zone_height_atr", "depth_into_zone", "zone_age", "touch_count",
    "impulse_atr", "atr_pct", "atr_ratio", "adx_norm", "trend_align",
    "htf_align", "extension_atr", "structure_align", "rr_room",
    "spread_atr", "hour_sin", "hour_cos", "body_ratio", "wick_reject",
    "range_pos", "htf_engulf",
    "session_sin", "session_cos", "momentum_rsi", "bb_width_norm",
    "zone_confluence",
    "ix_trend_mom", "ix_session_spread", "ix_struct_htf", "ix_height_bb", "ix_adx_rr",
    "hmm_regime_state",
)

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

def _rsi(closes, period):
    n = len(closes)
    if n < period + 1:
        return np.full(n, np.nan)
    diff = np.diff(closes)
    diff = np.concatenate(([np.nan], diff))
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    up_ewm = pd.Series(up).ewm(alpha=1.0/period, adjust=False).mean().to_numpy()
    down_ewm = pd.Series(down).ewm(alpha=1.0/period, adjust=False).mean().to_numpy()
    rs = np.zeros_like(up_ewm)
    mask = down_ewm != 0
    rs[mask] = up_ewm[mask] / down_ewm[mask]
    rsi = np.full_like(up_ewm, 100.0)
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[:period] = np.nan
    return rsi

def _bollinger_width(closes, period, std_mult):
    if len(closes) < period:
        return np.full(len(closes), np.nan)
    s = pd.Series(closes)
    rolling_std = s.rolling(period).std(ddof=0)
    width = rolling_std * 2.0 * std_mult
    return width.to_numpy()

def _adx(highs, lows, closes, period):
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
# Resampling
# ─────────────────────────────────────────────────────────────────────

def _resample(df, tf_minutes, entry_tf_minutes):
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
# Zone Detection V1
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
            runs[k: k + 3] = [[d1[0], d1[1], d2[2]]]
            merged = True
            break
    return runs

def _atr_filled(atr_series):
    valid = atr_series.dropna()
    if valid.empty:
        return None
    return atr_series.fillna(valid.iloc[0]).to_numpy()

def _detect_zones_v1(df, atr_series, params):
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
        price_high = float(highs[base_start: base_end + 1].max())
        price_low = float(lows[base_start: base_end + 1].min())
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
        zones.append({
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
        })
    return zones

# ─────────────────────────────────────────────────────────────────────
# Zone Detection V2
# ─────────────────────────────────────────────────────────────────────

def _detect_zones_v2(df, atr_series, params):
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
        price_high = float(max(highs[i], highs[base_start: base_end].max()))
        price_low = float(min(lows[i], lows[base_start: base_end].min()))
        impulse = float(body / max(atr_filled[i], 1e-9))
        if bullish and closes[dep_idx] > price_high:
            zones.append({
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
            })
        elif (not bullish) and closes[dep_idx] < price_low:
            zones.append({
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
            })
        i = dep_idx + 1
    return zones

# ─────────────────────────────────────────────────────────────────────
# Swing points & Structure
# ─────────────────────────────────────────────────────────────────────

def _detect_swing_points(highs, lows, lookback):
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_highs.max() and sum(window_highs == highs[i]) == 1:
            swings.append((i, float(highs[i]), "high"))
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_lows.min() and sum(window_lows == lows[i]) == 1:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda x: x[0])
    return swings

def _detect_structure(swings):
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

def _strong_levels(swings, atr_val, tol_mult, min_touches):
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
    return [
        (float(np.mean(c)), len(c))
        for c in clusters
        if len(c) >= min_touches
    ]

def _at_strong_level(levels, price_low, price_high, tol):
    for level, _strength in levels:
        if price_low - tol <= level <= price_high + tol:
            return True
    return False

def _trendline_touch(swings, kind, target_idx, price_low, price_high, tol, max_pairs):
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
# Quasimodo
# ─────────────────────────────────────────────────────────────────────

def _detect_quasimodo_legacy(df, atr_series, params):
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
            s1[2] == "low" and s2[2] == "high" and s3[2] == "low" and s4[2] == "high"
            and s3[1] < s1[1] and s4[1] < s2[1]
            and abs(s2[1] - s3[1]) >= min_swing_atr * atr_at
        )
        bearish = (
            s1[2] == "high" and s2[2] == "low" and s3[2] == "high" and s4[2] == "low"
            and s3[1] > s1[1] and s4[1] > s2[1]
            and abs(s3[1] - s2[1]) >= min_swing_atr * atr_at
        )
        if not (bullish or bearish):
            continue
        low, high = _band_around(s3[0], highs, lows, ext, atr_at, max_height_mult)
        zones.append({
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
        })
    return zones

def _detect_quasimodo_family(df, atr_series, params, swings, trend_dir):
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
        shoulder, neck, head, breaker = swings[idx], swings[idx + 1], swings[idx + 2], swings[idx + 3]
        atr_at = atr_filled[min(head[0], len(atr_filled) - 1)]
        if atr_at <= 0:
            continue
        bearish = (
            shoulder[2] == "high" and neck[2] == "low"
            and head[2] == "high" and breaker[2] == "low"
            and head[1] > shoulder[1]
            and breaker[1] < neck[1]
            and (head[1] - neck[1]) >= min_head_atr * atr_at
        )
        bullish = (
            shoulder[2] == "low" and neck[2] == "high"
            and head[2] == "low" and breaker[2] == "high"
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
        head_line = _trendline_touch(swings, "high" if bearish else "low", head[0], low, high, tl_tol, max_pairs)
        continuation = trend_dir == trade_dir
        if head_line:
            family = "QM2P"
        elif continuation:
            family = "QMC"
        else:
            family = "QMR" if bearish else "QML"
        zones.append({
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
        })
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
        zones.append({
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
        })
    return zones

def _upgrade_snrc(zones, levels, atr_val, params):
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
                broke = np.flatnonzero(closes[i + 1: stop] < lows[i])
            else:
                broke = np.flatnonzero(closes[i + 1: stop] > highs[i])
            if len(broke) == 0:
                continue
            fail_idx = int(broke[0]) + i + 1
            counter = bear_engulf if failed_bull else bull_engulf
            candidates = np.flatnonzero(counter[fail_idx: min(n, fail_idx + max_wait)])
            if len(candidates) == 0:
                continue
            entry_idx = int(candidates[0]) + fail_idx
            if not _at_strong_level(levels, lows[entry_idx], highs[entry_idx], tol):
                continue
            low, high = _band_around(entry_idx, highs, lows, 0, atr_at, max_height_mult)
            zones.append({
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
            })
    return zones

def _detect_sweep_zones(df, atr_series, params, swings):
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
                zones.append({
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
                })
                break
            if kind == "high" and highs[j] > swing_price + pierce and closes[j] < swing_price:
                zones.append({
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
                })
                break
    return zones

# ─────────────────────────────────────────────────────────────────────
# Zone lifecycle
# ─────────────────────────────────────────────────────────────────────

def _track_zone(zone, start_ns, entry_t_ns, highs, lows, closes):
    start = int(np.searchsorted(entry_t_ns, start_ns, side="left"))
    n = len(entry_t_ns)
    if start >= n:
        return None
    demand = zone["kind"] == ZoneKind.DEMAND
    if demand:
        breached = closes[start:] < zone["price_low"]
    else:
        breached = closes[start:] > zone["price_high"]
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
    return {
        "broken": break_off >= 0,
        "break_ns": int(entry_t_ns[start + break_off]) if break_off >= 0 else None,
        "touches": touches,
        "in_zone": bool(len(inside) > 0 and inside[-1] and end == n),
        "age_bars": int(n - 1 - start),
    }

def _flip_to_breaker(zone, break_ns):
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
    fast_period = int(params.get("trend_ema_fast", 12))
    slow_period = int(params.get("trend_ema_slow", 34))
    if len(closes) < slow_period + 2:
        return 0, 0.0
    fast = _ema(closes, fast_period)
    slow = _ema(closes, slow_period)
    separation = float(fast[-1] - slow[-1])
    return (1 if separation > 0 else -1 if separation < 0 else 0), separation

def _trend_bucket(adx_value, trend_dir, trade_dir, params):
    threshold = params.get("adx_trend_threshold", 20.0)
    if not np.isfinite(adx_value) or adx_value < threshold or trend_dir == 0:
        return 1
    return 2 if trend_dir == trade_dir else 0

def _session_bucket(hour_utc, params):
    h = hour_utc
    if 13 <= h < 16:
        return 4
    elif 8 <= h < 13:
        return 2
    elif 16 <= h < 21:
        return 3
    elif 0 <= h < 8:
        return 1
    else:
        return 0

def _momentum_bucket(rsi_value, params):
    if not np.isfinite(rsi_value):
        return 1
    if rsi_value < params.get("rsi_oversold", 30.0):
        return 0
    if rsi_value > params.get("rsi_overbought", 70.0):
        return 2
    return 1

def _spread_quality(spread_price, atr_val, params):
    if atr_val <= 0:
        return 1
    ratio = spread_price / atr_val
    if ratio < params.get("spread_tight_atr_mult", 0.15):
        return 0
    if ratio > params.get("spread_wide_atr_mult", 0.45):
        return 2
    return 1

def _confluence_score(
    structure_align, htf_align, touches, rsi_val, session_bucket, trade_dir, params
):
    score = 0
    if structure_align == trade_dir:
        score += 1
    if htf_align == trade_dir:
        score += 1
    if touches == 0:
        score += 1
    elif touches >= 2:
        score -= 1
    rsi_overbought = params.get("rsi_overbought", 70.0)
    rsi_oversold = params.get("rsi_oversold", 30.0)
    if trade_dir == 1 and rsi_val < rsi_overbought:
        score += 1
    elif trade_dir == -1 and rsi_val > rsi_oversold:
        score += 1
    if session_bucket == 4:
        score += 1
    elif session_bucket in (0, 1):
        score -= 1
    return score

# ─────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────

def _detect_hmm_regime(returns, lookback=100):
    if len(returns) < lookback:
        return 0
    recent = returns[-lookback:]
    vars = pd.Series(recent).rolling(10).var().bfill().to_numpy()
    if not np.isfinite(vars).all():
        vars = np.nan_to_num(vars)
    mu0, mu1 = np.percentile(vars, 25), np.percentile(vars, 75)
    for _ in range(3):
        d0 = np.abs(vars - mu0)
        d1 = np.abs(vars - mu1)
        c0 = vars[d0 <= d1]
        c1 = vars[d0 > d1]
        mu0 = float(c0.mean()) if len(c0) > 0 else mu0
        mu1 = float(c1.mean()) if len(c1) > 0 else mu1
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
    current_var = vars[-1]
    return 0 if abs(current_var - mu0) < abs(current_var - mu1) else 1

def _build_features(
    *, zone, life, close, atr_val, atr_slow, adx_value, trend_dir, htf_dir,
    htf_engulf, structure_bias, trade_dir, rr_room, spread_price, hour,
    last_bar, range_pos, extension_atr, rsi_val, bb_width, zone_confluence, 
    hmm_regime, params
):
    zone_height = max(zone["price_high"] - zone["price_low"], 1e-9)
    if zone["kind"] == ZoneKind.DEMAND:
        depth = (zone["price_high"] - close) / zone_height
    else:
        depth = (close - zone["price_low"]) / zone_height

    body = abs(last_bar["close"] - last_bar["open"])
    bar_range = max(last_bar["high"] - last_bar["low"], 1e-9)
    if trade_dir == 1:
        wick = last_bar["low"] < close and (min(last_bar["open"], last_bar["close"]) - last_bar["low"]) or 0.0
    else:
        wick = last_bar["high"] > close and (last_bar["high"] - max(last_bar["open"], last_bar["close"])) or 0.0

    ix_trend_mom = float(trend_dir * trade_dir) * (float(rsi_val / 100.0) if np.isfinite(rsi_val) else 0.5)
    ix_session_spread = float(np.sin(2.0 * np.pi * hour / 24.0)) * (spread_price / atr_val)
    ix_struct_htf = float(structure_bias * trade_dir) * float(htf_dir * trade_dir)
    ix_height_bb = (zone_height / atr_val) * (float(bb_width / max(atr_val, 1e-9)) if np.isfinite(bb_width) else 1.0)
    ix_adx_rr = (float(np.clip(adx_value / 50.0, 0.0, 2.0)) if np.isfinite(adx_value) else 0.4) * float(np.clip(rr_room, 0.0, 8.0))

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
            float(np.sin(2.0 * np.pi * hour / 24.0)),
            float(np.cos(2.0 * np.pi * hour / 24.0)),
            float(body / bar_range),
            float(np.clip(wick / atr_val, 0.0, 4.0)),
            float(np.clip(range_pos, 0.0, 1.0)),
            float(htf_engulf),
            float(np.sin(2.0 * np.pi * hour / 24.0)),
            float(np.cos(2.0 * np.pi * hour / 24.0)),
            float(rsi_val / 100.0) if np.isfinite(rsi_val) else 0.5,
            float(bb_width / max(atr_val, 1e-9)) if np.isfinite(bb_width) else 1.0,
            float(zone_confluence),
            ix_trend_mom,
            ix_session_spread,
            ix_struct_htf,
            ix_height_bb,
            ix_adx_rr,
            float(hmm_regime),
        ],
        dtype=float,
    )

# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────

class XauusdSndQmStructureAdaptiveM1:
    def __init__(self):
        self.spec = StrategySpec(
            name="xauusd_snd_qm_structure_adaptive_m5",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            confirmation_timeframes=("M15",),
            htf_veto=False,
            close_on_opposite_signal=True,
            params={
                # ── Timeframes ──
                "zone_tf_minutes": 15,
                "entry_tf_minutes": 5,
                "htf_key": "H1",

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

                # Session regime
                "session_asian_start_utc": 0,
                "session_asian_end_utc": 8,
                "session_london_start_utc": 8,
                "session_london_end_utc": 16,
                "session_ny_start_utc": 13,
                "session_ny_end_utc": 21,

                # Momentum
                "rsi_period": 14,
                "rsi_overbought": 70.0,
                "rsi_oversold": 30.0,

                # Consolidation
                "bb_period": 20,
                "bb_std_mult": 2.0,

                # Confluence
                "min_confluence_score": 2,
                "zone_confluence_atr_radius": 1.0,

                # Spread quality
                "spread_tight_atr_mult": 0.15,
                "spread_wide_atr_mult": 0.45,

                # ── Stop loss ──
                "sl_zone_buffer_atr_mult": 0.15,
                "sl_zone_buffer_zone_frac": 0.10,
                "sl_min_atr_mult": 0.5,

                # ── Take profit ladder ──
                "tp1_target_rr": 1.8,
                "tp2_target_rr": 2.5,
                "tp3_target_rr": 4.0,
                "tp_min_rr_floor": 1.75,
                "tp_buffer_points": 20,
                "tp_buffer_atr_mult": 0.3,
                "point_value": 0.01,

                # ── Learner ──
                "learner_enabled": True,
                "learner_secure_r": 0.2,
                "learner_horizon_bars": 120,
                "learner_min_bucket_samples": 25.0,
                "learner_min_model_samples": 150,
                "learner_min_quantile_samples": 30,
                "learner_half_life_samples": 750.0,
                "learner_edge_margin": 0.15,
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
                    max_pending=int(params["learner_max_pending"]),
                ),
            )
        return self._learner

    def reset_state(self):
        if self._learner is not None:
            self._learner.reset()
        self._last_bar_ns = None
        self._observed = {}

    def _check_continuity(self, first_ns, last_ns):
        if self._last_bar_ns is None:
            return
        if last_ns < self._last_bar_ns or first_ns > self._last_bar_ns + _RESET_GAP_NS:
            self.reset_state()

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
        zones.extend(_detect_snrc3(zone_frame, atr_series, params, levels, bull_engulf, bear_engulf))
        zones.extend(_detect_sweep_zones(zone_frame, atr_series, params, swings))
        _upgrade_snrc(zones, levels, atr_val, params)

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
                if life["in_zone"]:
                    candidates.append((zone, life))
                continue

            if zone["source"] == "BREAKER":
                continue
            flipped = _flip_to_breaker(zone, life["break_ns"])
            flipped_life = _track_zone(flipped, life["break_ns"], entry_t_ns, highs, lows, closes)
            if flipped_life is None or flipped_life["broken"]:
                continue
            live.append(flipped)
            if flipped_life["in_zone"]:
                candidates.append((flipped, flipped_life))
        return candidates, live

    def _count_nearby_zones(self, zone, live_zones, atr_val, params):
        radius = params.get("zone_confluence_atr_radius", 1.0) * atr_val
        count = 0
        zone_mid = (zone["price_high"] + zone["price_low"]) / 2.0
        for z in live_zones:
            if z is zone or z["kind"] != zone["kind"]:
                continue
            z_mid = (z["price_high"] + z["price_low"]) / 2.0
            if abs(z_mid - zone_mid) <= radius:
                count += 1
        return count

    def _decide(
        self, *, ctx, params, df, zone_frame, atr_series, atr_val,
        candidates, live_zones, learner, entry_t_ns,
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
        
        rsi_values = _rsi(zone_closes, int(params.get("rsi_period", 14)))
        rsi_val = float(rsi_values[-1]) if len(rsi_values) else float("nan")
        
        bb_widths = _bollinger_width(zone_closes, int(params.get("bb_period", 20)), float(params.get("bb_std_mult", 2.0)))
        bb_width = float(bb_widths[-1]) if len(bb_widths) else float("nan")

        vol_bucket = _volatility_bucket(atr_series.to_numpy(), params)
        hour_utc = float(now.hour) + float(now.minute) / 60.0
        session_bucket = _session_bucket(hour_utc, params)

        returns = np.diff(zone_closes) / zone_closes[:-1] if len(zone_closes) > 1 else np.array([])
        hmm_regime = _detect_hmm_regime(returns, lookback=int(params.get("range_lookback_bars", 40)))

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

        horizon_ns = int(params["learner_horizon_bars"]) * int(params["entry_tf_minutes"]) * _NS_PER_MINUTE
        now_ns = int(entry_t_ns[-1])
        threshold = self._gate_threshold(params)
        min_confluence_score = int(params.get("min_confluence_score", 2))

        scored = []
        for zone, life in candidates:
            demand = zone["kind"] == ZoneKind.DEMAND
            trade_dir = 1 if demand else -1

            if zone.get("needs_htf", False):
                confirmed = self._htf["bull"] if demand else self._htf["bear"]
                if not confirmed:
                    continue
                    
            htf_align = self._htf["dir"]
            
            confluence_score = _confluence_score(
                structure_bias, htf_align, life["touches"], rsi_val, session_bucket, trade_dir, params
            )
            
            if confluence_score < min_confluence_score:
                continue

            plan = self._build_plan(
                zone=zone, life=life, params=params, close=close, atr_val=atr_val,
                spread_price=spread_price, live_zones=live_zones, learner=learner,
                vol_bucket=vol_bucket, adx_value=adx_value, trade_dir=trade_dir,
                session_bucket=session_bucket,
            )
            if plan is None:
                continue

            extension_atr = (close - anchor) / atr_val * trade_dir
            range_pos = (close - window_low) / span
            if trade_dir == -1:
                range_pos = 1.0 - range_pos
                
            zone_confluence = self._count_nearby_zones(zone, live_zones, atr_val, params)

            features = _build_features(
                zone=zone, life=life, close=close, atr_val=atr_val, atr_slow=atr_slow,
                adx_value=adx_value, trend_dir=self._trend_dir, htf_dir=self._htf["dir"],
                htf_engulf=1.0 if (self._htf["bull"] if demand else self._htf["bear"]) else 0.0,
                structure_bias=structure_bias, trade_dir=trade_dir,
                rr_room=plan["rr_room"], spread_price=spread_price,
                hour=hour_utc, last_bar=last_bar,
                range_pos=range_pos, extension_atr=extension_atr,
                rsi_val=rsi_val, bb_width=bb_width, zone_confluence=zone_confluence,
                hmm_regime=hmm_regime, params=params,
            )

            verdict = None
            if learner is not None:
                verdict = learner.score(features, plan["bucket"])
                self._record_candidate(
                    learner=learner, zone=zone, life=life, features=features,
                    bucket=plan["bucket"], now_ns=now_ns, close=close,
                    trade_dir=trade_dir, sl_points=plan["sl_points"], atr_val=atr_val,
                    horizon_ns=horizon_ns, params=params,
                )
                if verdict.ready and verdict.edge_over(threshold) <= 0.0:
                    continue

            scored.append((zone, life, plan, verdict, features, confluence_score))

        if not scored:
            return None

        scored.sort(key=lambda item: _rank_key(item[3], item[1], item[5]), reverse=True)
        zone, life, plan, verdict, _features, confluence_score = scored[0]
        return self._build_signals(
            zone=zone, life=life, plan=plan, verdict=verdict, params=params,
            zone_frame=zone_frame, structure=structure, now=now, threshold=threshold,
            atr_val=atr_val, confluence_score=confluence_score
        )

    def _gate_threshold(self, params):
        base = breakeven_probability(float(params["learner_secure_r"]))
        margin = float(params.get("learner_edge_margin", 0.15))
        odds = base / (1.0 - base) * float(np.exp(margin))
        return float(odds / (1.0 + odds))

    def _build_plan(
        self, *, zone, life, params, close, atr_val, spread_price, live_zones,
        learner, vol_bucket, adx_value, trade_dir, session_bucket
    ):
        demand = zone["kind"] == ZoneKind.DEMAND
        zone_height = zone["price_high"] - zone["price_low"]
        if zone_height <= 0:
            return None

        bucket = "{0}|v{1}|t{2}|s{3}".format(
            zone["family"],
            vol_bucket,
            _trend_bucket(adx_value, self._trend_dir, trade_dir, params),
            session_bucket
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
                    base_buffer_mult, beyond,
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
            above = sorted([z for z in opposite if z["price_low"] > close], key=lambda z: z["price_low"])
            if len(above) >= 1:
                zone_target_1 = above[0]["price_low"] - close - tp_buffer
            if len(above) >= 2:
                zone_target_2 = above[1]["price_low"] - close - tp_buffer
        else:
            below = sorted([z for z in opposite if z["price_high"] < close], key=lambda z: z["price_high"], reverse=True)
            if len(below) >= 1:
                zone_target_1 = close - below[0]["price_high"] - tp_buffer
            if len(below) >= 2:
                zone_target_2 = close - below[1]["price_high"] - tp_buffer

        rr2 = float(params.get("tp2_target_rr", 2.5))
        rr3 = float(params.get("tp3_target_rr", 4.0))
        if learner is not None:
            rr2 = AdaptiveLearner.blend(
                rr2,
                learner.favourable_excursion_r(bucket, float(params.get("tp2_learned_quantile", 0.6))),
                lo_frac=float(params.get("tp_band_lo", 0.6)),
                hi_frac=float(params.get("tp_band_hi", 1.6)),
            )
            rr3 = AdaptiveLearner.blend(
                rr3,
                learner.favourable_excursion_r(bucket, float(params.get("tp3_learned_quantile", 0.85))),
                lo_frac=float(params.get("tp_band_lo", 0.6)),
                hi_frac=float(params.get("tp_band_hi", 1.6)),
            )

        floor_rr = float(params.get("tp_min_rr_floor", 1.75))
        tp1_points = risk_price * max(float(params.get("tp1_target_rr", 1.8)), floor_rr)
        if zone_target_1 is not None and zone_target_1 > risk_price * 0.8:
            tp1_points = min(tp1_points, max(zone_target_1 * 0.5, risk_price * floor_rr))

        if zone_target_1 is not None and zone_target_1 > tp1_points:
            tp2_points = zone_target_1
        else:
            tp2_points = max(tp1_points + risk_price * 0.5, risk_price * rr2)
        if zone_target_2 is not None and zone_target_2 > tp2_points:
            tp3_points = zone_target_2
        else:
            tp3_points = max(tp2_points + risk_price * 0.5, risk_price * rr3)

        return {
            "bucket": bucket,
            "sl_points": float(sl_points),
            "risk_price": float(risk_price),
            "tp1": float(tp1_points),
            "tp2": float(tp2_points),
            "tp3": float(tp3_points),
            "buffer_mult": float(buffer_mult),
            "rr2": float(rr2),
            "rr3": float(rr3),
            "rr_room": float(zone_target_1 / risk_price if zone_target_1 is not None else 0.0),
        }

    def _record_candidate(
        self, *, learner, zone, life, features, bucket, now_ns, close, trade_dir,
        sl_points, atr_val, horizon_ns, params,
    ):
        key = "{0}|{1:.2f}|{2:.2f}|{3}".format(
            zone["family"], zone["price_low"], zone["price_high"], life["touches"]
        )
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
        self, *, zone, life, plan, verdict, params, zone_frame, structure, now,
        threshold, atr_val, confluence_score
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
                name="sl_buffer_atr_mult", value=plan["buffer_mult"],
                threshold=float(params["sl_zone_buffer_atr_mult"]),
                comparison=">", passed=True,
            ),
            IndicatorReading(
                name="zone_touches", value=float(life["touches"]),
                threshold=1.0, comparison="<", passed=life["touches"] <= 1,
            ),
        ]
        if verdict is not None:
            readings.insert(
                0,
                IndicatorReading(
                    name="learner_p_secure", value=round(verdict.p_secure, 4),
                    threshold=round(threshold, 4), comparison=">",
                    passed=(not verdict.ready) or verdict.edge_over(threshold) > 0.0,
                ),
            )
            readings.insert(
                1,
                IndicatorReading(
                    name="learner_bucket_samples", value=round(verdict.bucket_samples, 1),
                    threshold=float(params["learner_min_bucket_samples"]),
                    comparison=">", passed=verdict.ready,
                ),
            )
            learn_note = " p={0:.3f}/{1:.3f} n={2:.0f}{3}".format(
                verdict.p_secure, threshold, verdict.bucket_samples,
                "" if verdict.ready else " (warmup)",
            )
        else:
            learn_note = " learner=off"

        base_reason = "{0} [{1:.2f},{2:.2f}] sl={3:.2f} atr={4:.2f} touch={5} conf={6}{7}".format(
            zone["family"], zone["price_low"], zone["price_high"],
            plan["sl_points"], atr_val, life["touches"], confluence_score, learn_note,
        )
        readings_tuple = tuple(readings)

        return (
            Signal(
                direction=direction, sl_points=plan["sl_points"], tp_points=plan["tp1"],
                confidence=_confidence(verdict, 0.75),
                reason=base_reason + " tp1={0:.2f} (Scalp)".format(plan["tp1"]),
                zone=annotation, pattern=zone.get("pattern"),
                structure=structure_points, indicators=readings_tuple,
            ),
            Signal(
                direction=direction, sl_points=plan["sl_points"], tp_points=plan["tp2"],
                confidence=_confidence(verdict, 0.70),
                reason=base_reason + " tp2={0:.2f} (Zone rr={1:.2f})".format(plan["tp2"], plan["rr2"]),
                zone=annotation, pattern=zone.get("pattern"),
                structure=structure_points, indicators=readings_tuple,
            ),
            Signal(
                direction=direction, sl_points=plan["sl_points"], tp_points=plan["tp3"],
                confidence=_confidence(verdict, 0.65),
                reason=base_reason + " tp3={0:.2f} (Runner rr={1:.2f})".format(plan["tp3"], plan["rr3"]),
                zone=annotation, pattern=zone.get("pattern"),
                structure=structure_points, indicators=readings_tuple,
            ),
        )

def _confidence(verdict, base):
    if verdict is None or not verdict.ready:
        return base
    return float(np.clip(verdict.p_secure, 0.05, 0.99))

def _rank_key(verdict, life, confluence):
    probability = verdict.p_secure if verdict is not None else 0.0
    return (probability, confluence, -life["touches"], life["age_bars"])
