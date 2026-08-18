"""XAUUSD APEX — S&D + Quasimodo + Trend-Guard + Zone-Respect, H1 entries.

The H1 sibling of `xauusd_snd_apex_trendguard_m1_v5.py`. The algorithm is
that file byte-for-byte apart from the timeframe wiring below; every
threshold, gate and default still traces to the forensic post-mortem of the
2026-08-05 XAUUSD session (608 positions, 41.2% win rate, profit factor 0.73,
-$889.58 on the day) documented in full there, plus the v2-v5 calibration
passes on top of it (trend fatigue gate included but off by default — see
`fatigue_max`/`fatigue_fade_min` below). Read the M1 file for *why* each
number is what it is — this docstring only covers what changes.

Sandbox rules (CLAUDE.md) limit a generated strategy to math / statistics /
numpy / pandas, so it cannot import the M1 file and share the code; the
siblings are full copies produced by an exact-substitution generator that
fails rather than emit a drifted copy.

────────────────────────────────────────────────────────────────────────
TIMEFRAME LADDER
────────────────────────────────────────────────────────────────────────
    entry / trigger      H1
    zone detection       H4
    higher-TF zones      D1      (confluence + Quasimodo confirmation)
    trend votes          H1 (w 1.0), H4 (w 1.5), D1 (w 2.0)

The engine hands every declared `confirmation_timeframes` entry its own
200-bar window (`trade_loop.DEFAULT_CONTEXT_BARS`), so these are native
candles, not a resample of the entry frame: ~33 days of H4 and ~200 sessions of D1.

────────────────────────────────────────────────────────────────────────
WHY THIS LADDER AND NOT THE OBVIOUS ONE
────────────────────────────────────────────────────────────────────────
The trend ladder starts at the entry timeframe itself rather than at the
zone timeframe: above H4/D1 the only rungs left are W1 and MN, and a
backtest's 60-day context buffer holds ~8 W1 bars, so a W1 vote would
sit at 0 for most of any replay while being alive live. A vote that
means different things in backtest and live is worse than no vote. The
EMA pair is 13/34 for the same reason the M15 sibling uses it — D1 has
only ~43 bars of pre-roll.

────────────────────────────────────────────────────────────────────────
THE ADAPTIVE CORE: ZONE RESPECT INDEX (ZRI)
────────────────────────────────────────────────────────────────────────
Unchanged from the M1 file. The bot measures, separately for demand and for
supply, whether zones of that kind are actually holding *right now* on this
symbol: of the recently resolved zones of that kind, what fraction produced
a real reversal rather than being closed straight through. A kind whose ZRI
falls under `zri_min` is switched off until it earns its way back. Nothing
is hardcoded to a direction — it is a live read of which side of the book
the market is currently honouring.

Sandbox-safe: only math / numpy / pandas, no I/O, no broker access.
"""

import math

import numpy as np
import pandas as pd

from src.strategies.domain.fatigue import trend_fatigue
from src.strategies.domain.models import (
    Direction,
    IndicatorReading,
    PriceZone,
    Signal,
    StrategySpec,
    StructureLabel,
    StructurePoint,
    ZoneKind,
)

# ─────────────────────────────────────────────────────────────────────
# ATR / EMA helpers
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


def _ema_series(frame, span):
    """EMA of the close series. Vectorised via pandas — this runs on every
    zone-timeframe close across a 100k-bar replay, so a Python loop here
    dominates backtest runtime."""
    return frame["close"].ewm(span=span, adjust=False).mean()


def _percentile_rank(series_values, current):
    """Fraction of `series_values` strictly below `current`, in 0..1."""
    if len(series_values) == 0:
        return 0.5
    below = float(np.sum(series_values < current))
    return below / float(len(series_values))


# ─────────────────────────────────────────────────────────────────────
# Volatility-shock guard (the calendar proxy)
# ─────────────────────────────────────────────────────────────────────

def _bars_since_shock(highs, lows, lookback, mult):
    """How many entry-TF bars ago the last abnormal range expansion was,
    or None if there hasn't been one in the visible window.

    This is the strategy's stand-in for an economic calendar. The engine
    does have a real one (`configs/news.yaml`, the `news/` module and
    `skills/news/*.yaml`), but it only helps when the feed is configured
    and the event is scheduled, and its `max_spread_points` rules lean on
    a spread that on this broker's XAUUSD feed sat at a flat 15 points
    (max 17) for the whole 2026-08-05 session — so on the day being fixed
    here they could not have fired at all.

    Measured over 2026-04-06..2026-08-05 (118,564 M1 bars), range shocks
    of >= 5x the trailing 4h median cluster about 2x above chance in known
    release quarter-hours (21.7% vs 11.5% expected), with 12:30 UTC — US
    NFP / CPI / PPI / claims — the largest scheduled cluster at 36 shocks.
    The single biggest cluster overall is 22:00-01:00 UTC, which is the
    daily rollover rather than news. Both are thin-book conditions an M1
    zone bot should sit out, and both look identical on the tape, so one
    detector covers them without needing to know which it is.

    Blast radius, measured on this strategy's own 2,136-trade replay:
    inside -5/+15 min of a shock its win rate falls to 90.9% against 96.6%
    outside, so the cooldown is sized to that window rather than to the
    much wider -30/+60 the calendar config uses — over that wider window
    results were, if anything, better than baseline once the dust settled.
    """
    n = len(highs)
    if n < lookback + 2:
        return None
    ranges = highs - lows
    median = float(np.median(ranges[n - lookback: n]))
    if median <= 0:
        return None
    threshold = mult * median
    for back in range(n - 1, max(-1, n - 1 - lookback), -1):
        if float(ranges[back]) >= threshold:
            return (n - 1) - back
    return None


# ─────────────────────────────────────────────────────────────────────
# Trend state — one signed vote per timeframe
# ─────────────────────────────────────────────────────────────────────

def _timeframe_trend(frame, fast_span, slow_span, slope_bars):
    """+1 up / -1 down / 0 flat for one timeframe.

    A vote needs both agreement of the two EMAs *and* a slope on the fast
    EMA of the same sign — the 2026-08-05 losses were dominated by zones
    taken while a fast EMA had already rolled over but still sat on the
    "right" side of the slow one.
    """
    if frame is None or len(frame) < slow_span + slope_bars + 1:
        return 0
    fast_ema = _ema_series(frame, fast_span).to_numpy()
    slow_ema = _ema_series(frame, slow_span).to_numpy()
    last = len(fast_ema) - 1
    fast_now = float(fast_ema[last])
    slow_now = float(slow_ema[last])
    fast_prev = float(fast_ema[last - slope_bars])
    if math.isnan(fast_now) or math.isnan(slow_now) or math.isnan(fast_prev):
        return 0
    rising = fast_now > fast_prev
    falling = fast_now < fast_prev
    if fast_now > slow_now and rising:
        return 1
    if fast_now < slow_now and falling:
        return -1
    return 0


def _trend_state(ctx, params):
    """Aggregate signed trend score across the confirmation timeframes.

    Returns (score, votes) where score is the sum of the per-timeframe
    votes weighted by `trend_weights`, and votes is a dict for the
    signal's audit trail.

    The ladder itself *is* `trend_weights` — its keys are the timeframes
    voted on, in order. That is what lets the M1/M5/M15/H1/D1 siblings of
    this family run the identical algorithm on a shifted timeframe ladder
    with no code change, only a params change.
    """
    fast = int(params["trend_ema_fast"])
    slow = int(params["trend_ema_slow"])
    slope_bars = int(params["trend_slope_bars"])
    weights = params["trend_weights"]

    votes = {}
    score = 0.0
    for timeframe in weights:
        vote = _timeframe_trend(ctx.candles.get(timeframe), fast, slow, slope_bars)
        votes[timeframe] = vote
        score += vote * float(weights[timeframe])
    return score, votes


# ─────────────────────────────────────────────────────────────────────
# Supply & Demand V1: classic leg-base-leg (RBR / DBD / RBD / DBR)
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


def _detect_zones_v1(df, atr_series, params):
    """Classic leg-base-leg geometry, with the departure impulse recorded.

    `impulse_atr` (how hard the leg-out left the base, in ATR) is new: on
    2026-08-05 the zones that held were the ones price left violently, and
    it now feeds both the quality gate and the candidate score.
    """
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    base_mult = float(params["base_body_atr_mult"])
    leg_mult = float(params["leg_travel_atr_mult"])
    max_base = int(params["max_base_candles"])

    classes = _classify_bars(closes, opens, atr_filled, base_mult)
    runs = _build_runs(classes)
    is_leg = _make_is_leg(closes, opens, atr_filled, leg_mult)
    runs = _merge_weak_runs([list(r) for r in runs], is_leg, max_base)

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
        atr_at = float(atr_filled[leg_out[2]])
        impulse = abs(float(closes[leg_out[2]]) - float(opens[leg_out[1]]))
        if leg_in[0] == 1:
            pattern = "RBR" if leg_out_up else "RBD"
        else:
            pattern = "DBR" if leg_out_up else "DBD"
        zones.append({
            "source": "SND_V1",
            "pattern": pattern,
            "kind": ZoneKind.DEMAND if leg_out_up else ZoneKind.SUPPLY,
            "price_high": price_high,
            "price_low": price_low,
            "base_start": base_start,
            "conf_idx": conf_idx,
            "leg_out_end": leg_out[2],
            "impulse_atr": (impulse / atr_at) if atr_at > 0 else 0.0,
            "base_count": base_count,
        })
    return zones


# ─────────────────────────────────────────────────────────────────────
# Supply & Demand V2: engulfing-base departure zones
# ─────────────────────────────────────────────────────────────────────

def _detect_zones_v2(df, atr_series, params):
    """Engulf -> base -> departure order blocks.

    Demoted relative to the old family: V2 supply zones alone lost $796.12
    on 2026-08-05 at a 4.5% win rate, so `_score_zone` weights this source
    below SND_V1 and `v2_engulf_min_atr_mult` is raised, making the
    detector fire only on genuinely violent engulfing candles.
    """
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)

    base_mult = float(params["base_body_atr_mult"])
    max_base = int(params.get("v2_max_base_candles", params["max_base_candles"]))
    min_engulf_atr = float(params.get("v2_engulf_min_atr_mult", 1.6))

    zones = []
    i = 1
    while i < n - 2:
        body = abs(closes[i] - opens[i])
        atr_at = float(atr_filled[i])
        if atr_at <= 0 or body < min_engulf_atr * atr_at:
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
        impulse = abs(float(closes[dep_idx]) - float(opens[dep_idx]))
        common = {
            "source": "SND_V2",
            "price_high": price_high,
            "price_low": price_low,
            "base_start": base_start,
            "conf_idx": dep_idx,
            "leg_out_end": dep_idx,
            "impulse_atr": (impulse / atr_at) if atr_at > 0 else 0.0,
            "base_count": base_end - base_start,
        }
        if bullish and closes[dep_idx] > price_high:
            zones.append({**common, "pattern": "DZ_V2", "kind": ZoneKind.DEMAND})
        elif (not bullish) and closes[dep_idx] < price_low:
            zones.append({**common, "pattern": "SZ_V2", "kind": ZoneKind.SUPPLY})
        i = dep_idx + 1

    return zones


# ─────────────────────────────────────────────────────────────────────
# Quasimodo
# ─────────────────────────────────────────────────────────────────────

def _detect_swing_points(highs, lows, lookback):
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_highs.max() and int(np.sum(window_highs == highs[i])) == 1:
            swings.append((i, float(highs[i]), "high"))
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_lows.min() and int(np.sum(window_lows == lows[i])) == 1:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda s: s[0])
    return swings


def _detect_quasimodo_zones(df, atr_series, params):
    """Quasimodo failure zones.

    QM_BEAR went 0-for-37 (-$225.59) on 2026-08-05 fighting the trend, so
    Quasimodo is kept as a *confluence* source only — `_score_zone` never
    lets it be a standalone primary candidate unless it overlaps an
    SND_V1 zone (see `qm_requires_confluence`).
    """
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    lookback = int(params.get("qm_swing_lookback", 3))
    min_swing_atr = float(params.get("qm_min_swing_atr", 0.8))

    swings = _detect_swing_points(highs, lows, lookback)
    if len(swings) < 4:
        return []

    zones = []
    for idx in range(len(swings) - 3):
        s1, s2, s3, s4 = swings[idx], swings[idx + 1], swings[idx + 2], swings[idx + 3]
        bull = (s1[2] == "low" and s2[2] == "high" and s3[2] == "low" and s4[2] == "high"
                and s3[1] < s1[1] and s4[1] < s2[1])
        bear = (s1[2] == "high" and s2[2] == "low" and s3[2] == "high" and s4[2] == "low"
                and s3[1] > s1[1] and s4[1] > s2[1])
        if not (bull or bear):
            continue
        swing_range = abs(s2[1] - s3[1])
        zone_idx = s3[0]
        atr_at = float(atr_filled[min(zone_idx, len(atr_filled) - 1)])
        if atr_at <= 0 or swing_range < min_swing_atr * atr_at:
            continue

        ext = max(1, lookback // 2)
        start = max(0, zone_idx - ext)
        end = min(len(lows), zone_idx + ext + 1)
        zone_low = float(lows[start:end].min())
        zone_high = float(highs[start:end].max())
        max_height = 2.0 * atr_at
        if zone_high - zone_low > max_height:
            mid = (zone_high + zone_low) / 2.0
            zone_high = mid + max_height / 2.0
            zone_low = mid - max_height / 2.0

        zones.append({
            "source": "QUASIMODO",
            "pattern": "QM_BULL" if bull else "QM_BEAR",
            "kind": ZoneKind.DEMAND if bull else ZoneKind.SUPPLY,
            "price_high": zone_high,
            "price_low": zone_low,
            "base_start": zone_idx,
            "conf_idx": s4[0],
            "leg_out_end": s4[0],
            "impulse_atr": swing_range / atr_at,
            "base_count": 1,
        })
    return zones


# ─────────────────────────────────────────────────────────────────────
# Market structure (HH / HL / LH / LL)
# ─────────────────────────────────────────────────────────────────────

def _detect_structure(highs, lows, lookback):
    swings = _detect_swing_points(highs, lows, lookback)
    labelled = []
    last_high = None
    last_low = None
    for index, price, kind in swings:
        if kind == "high":
            higher = last_high is None or price > last_high
            label = StructureLabel.HH if higher else StructureLabel.LH
            last_high = price
        else:
            higher = last_low is not None and price > last_low
            label = StructureLabel.HL if higher else StructureLabel.LL
            last_low = price
        labelled.append({"index": index, "price": price, "label": label})
    return labelled


def _structure_bias(labelled, depth):
    """Signed structure read over the last `depth` labelled swings.

    This is the exact annotation the old family already attached to every
    signal and then ignored: bias=up scored 63.7% WR / +$709.81 on
    2026-08-05 while bias=down scored 5.9% WR / -$1580.08.
    """
    recent = labelled[len(labelled) - depth:] if len(labelled) > depth else labelled
    up = sum(1 for s in recent if s["label"] in (StructureLabel.HH, StructureLabel.HL))
    down = sum(1 for s in recent if s["label"] in (StructureLabel.LL, StructureLabel.LH))
    if up > down:
        return 1
    if down > up:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────────
# Zone lifecycle on the entry timeframe
# ─────────────────────────────────────────────────────────────────────

def _zone_lifecycle(zone, zone_end_time, entry_times, highs, lows, closes, max_bars_inside):
    """Life story of one zone since it was created.

    Returns a dict with:
      dead          — the zone must never be traded again
      dead_reason   — why, for the audit trail
      inside_now    — price is interacting with the zone on this bar
      episodes      — how many *discrete* retest episodes have happened,
                      counting the current one
      bars_inside   — total entry-TF closes that sat inside the zone

    The old family only checked "has a candle closed through the far
    side", which let one supply zone be sold 18 times for -$462 while
    price chewed through it without ever printing a clean close beyond
    the top. `episodes` and `bars_inside` are what close that hole.
    """
    start = int(np.searchsorted(entry_times, zone_end_time, side="left"))
    if start >= len(entry_times):
        return {"dead": False, "dead_reason": "", "inside_now": False,
                "episodes": 0, "bars_inside": 0}

    demand = zone["kind"] == ZoneKind.DEMAND
    low = zone["price_low"]
    high = zone["price_high"]

    seg_high = highs[start:]
    seg_low = lows[start:]
    seg_close = closes[start:]

    # Hard break: a close beyond the far side.
    broke = (seg_close < low) if demand else (seg_close > high)
    if bool(np.any(broke)):
        return {"dead": True, "dead_reason": "closed through far side", "inside_now": False,
                "episodes": 0, "bars_inside": 0}

    # Interaction = the bar overlapped the rectangle at all.
    touching = (seg_high >= low) & (seg_low <= high)
    bars_inside = int(np.sum((seg_close >= low) & (seg_close <= high)))

    # Discrete episodes: count rising edges of `touching`.
    if len(touching) == 0:
        episodes = 0
    else:
        prev = np.concatenate(([False], touching[: len(touching) - 1]))
        episodes = int(np.sum(touching & (~prev)))

    inside_now = bool(touching[len(touching) - 1]) if len(touching) else False

    # Eaten: price has loitered inside the rectangle so long that whatever
    # unfilled interest created it is gone, even without a clean break.
    if bars_inside > max_bars_inside:
        return {"dead": True, "dead_reason": f"eaten ({bars_inside} closes inside)",
                "inside_now": inside_now, "episodes": episodes, "bars_inside": bars_inside}

    return {"dead": False, "dead_reason": "", "inside_now": inside_now,
            "episodes": episodes, "bars_inside": bars_inside}


# ─────────────────────────────────────────────────────────────────────
# Zone Respect Index — the adaptive regime read
# ─────────────────────────────────────────────────────────────────────

def _zone_respect_index(zones, frame, atr_value, params):
    """Are zones of each kind actually holding on this symbol right now?

    For every zone already resolved by history, replay what happened on
    the zone timeframe when price first came back to it:

      respected — price turned and travelled `zri_reversal_atr` x ATR in
                  the zone's favour before any close through the far side
      violated  — price closed clean through the far side

    ZRI(kind) = respected / (respected + violated) over the most recent
    `zri_sample` resolved cases.  Untested and inconclusive zones are not
    samples, so the index measures conviction, not mere frequency.

    Returns {ZoneKind: (index, sample_count)}.  Below `zri_min_samples`
    the index is neutral (1.0) so a cold start never blocks everything.

    This is what would have switched gold's short book off on
    2026-08-05: supply zones were being closed straight through from
    ~05:00 UTC (05h 0% WR, 10h 4% WR, 11h 0% WR), so supply-ZRI collapses
    while demand-ZRI stays high.
    """
    horizon = int(params["zri_horizon_bars"])
    reversal = float(params["zri_reversal_atr"]) * atr_value
    sample_cap = int(params["zri_sample"])
    min_samples = int(params["zri_min_samples"])

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()
    n = len(closes)

    outcomes = {ZoneKind.DEMAND: [], ZoneKind.SUPPLY: []}

    for zone in zones:
        created = int(zone["leg_out_end"])
        demand = zone["kind"] == ZoneKind.DEMAND
        low = zone["price_low"]
        high = zone["price_high"]

        # First return to the rectangle.
        touch = None
        limit = min(n, created + 1 + horizon)
        for i in range(created + 1, limit):
            if highs[i] >= low and lows[i] <= high:
                touch = i
                break
        if touch is None:
            continue

        verdict = None
        end = min(n, touch + horizon + 1)
        for i in range(touch, end):
            # A close through the far side counts from the touch bar
            # itself. The favourable-excursion test does not: the bar that
            # first reaches a zone is frequently the same bar that blows
            # through it, and its far wick would otherwise be scored as a
            # rejection the zone never actually produced.
            if demand:
                if closes[i] < low:
                    verdict = False
                    break
                if i > touch and highs[i] - high >= reversal:
                    verdict = True
                    break
            else:
                if closes[i] > high:
                    verdict = False
                    break
                if i > touch and low - lows[i] >= reversal:
                    verdict = True
                    break
        if verdict is None:
            continue
        outcomes[zone["kind"]].append((created, verdict))

    index = {}
    for kind in (ZoneKind.DEMAND, ZoneKind.SUPPLY):
        cases = sorted(outcomes[kind], key=lambda c: c[0])
        cases = cases[len(cases) - sample_cap:] if len(cases) > sample_cap else cases
        if len(cases) < min_samples:
            index[kind] = (1.0, len(cases))
            continue
        respected = sum(1 for _, ok in cases if ok)
        index[kind] = (respected / float(len(cases)), len(cases))
    return index


# ─────────────────────────────────────────────────────────────────────
# Entry confirmation on the H1 candle
# ─────────────────────────────────────────────────────────────────────

def _confirmation(zone, opens, highs, lows, closes, atr_value, params):
    """Does the last H1 candle agree with the trade?

    The old family entered on `high >= zone_low` — a bare touch, which is
    indistinguishable from price arriving at speed on its way through, and
    is a direct contributor to its 357 stop-outs at a full -1.04R.

    Three modes, because confirmation trades win rate against trade count
    and the right point on that curve differs per bot:

      "off"    — bare touch, the old behaviour.
      "light"  — the candle must merely be closing in the trade's
                 direction and not have closed out the far side of the
                 zone. Cheap, keeps H1 frequency high. Default.
      "strict" — additionally demands a real confirmation candle: either
                 an engulfing-style body of `confirm_body_atr_mult` x ATR
                 *or* a rejection wick back into the zone worth
                 `confirm_wick_frac` of the bar's range. Deliberately OR,
                 not AND: a momentum engulfing candle that opens at its low
                 and runs has almost no wick by construction, so requiring
                 both admits neither shape — an AND version of this rule
                 took zero trades across a whole session.

    Returns (confirmed, description).
    """
    mode = params.get("confirm_mode", "light")
    if mode == "off":
        return True, "off"

    last = len(closes) - 1
    o = float(opens[last])
    h = float(highs[last])
    low_ = float(lows[last])
    c = float(closes[last])
    demand = zone["kind"] == ZoneKind.DEMAND

    bar_range = h - low_
    if bar_range <= 0:
        return False, "zero-range candle"

    if demand:
        if c <= o:
            return False, "not a bullish close"
        if c < zone["price_low"]:
            return False, "closed below demand zone"
        wick = min(o, c) - low_
    else:
        if c >= o:
            return False, "not a bearish close"
        if c > zone["price_high"]:
            return False, "closed above supply zone"
        wick = h - max(o, c)

    body = abs(c - o)
    if mode != "strict":
        return True, f"light body={body / atr_value:.2f}atr"

    min_body = float(params["confirm_body_atr_mult"]) * atr_value
    wick_frac = wick / bar_range
    min_wick = float(params["confirm_wick_frac"])
    if body < min_body and wick_frac < min_wick:
        return False, (
            f"body {body:.2f}<{min_body:.2f} and wick {wick_frac:.2f}<{min_wick:.2f}"
        )

    return True, f"strict body={body / atr_value:.2f}atr wick={wick_frac:.2f}"


# ─────────────────────────────────────────────────────────────────────
# Candidate scoring
# ─────────────────────────────────────────────────────────────────────

def _score_zone(zone, depth, height_atr, age_bars, overlaps, params):
    """Rank surviving candidates so the best one trades, not the first one.

    The old family took `candidate = z` on the first zone its iteration
    happened to reach (V1 list, then V2, then Quasimodo) — an arbitrary
    pick that let the day's worst sources front-run the best ones.  Source
    weights come straight from the 2026-08-05 P&L by method: SND_V1
    +$98.36, SND_V2 -$730.43, QUASIMODO -$225.59.
    """
    weights = params["source_weights"]
    score = float(weights.get(zone["source"], 0.0))
    score += float(params["score_impulse_weight"]) * min(zone.get("impulse_atr", 0.0), 3.0)
    score += float(params["score_overlap_weight"]) * float(overlaps)
    score -= float(params["score_depth_weight"]) * depth
    score -= float(params["score_width_weight"]) * height_atr
    score -= float(params["score_age_weight"]) * (age_bars / 50.0)
    return score


def _overlap_count(zone, others):
    """How many zones from other detectors/timeframes cover this one.

    Confluence: a H4 RBR sitting inside a D1 demand block is a much
    better trade than either alone.
    """
    count = 0
    for other in others:
        if other["kind"] != zone["kind"]:
            continue
        if other["price_low"] <= zone["price_high"] and other["price_high"] >= zone["price_low"]:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────

class XauusdSndApexTrendguardH1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="xauusd_snd_apex_trendguard_h1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="H1",
            # Native higher-timeframe candles: the engine gives every entry
            # here its own 200-bar window, so zones are found on real H4
            # and D1 bars instead of a resample of the H1 context.
            confirmation_timeframes=("H4", "D1"),
            # The engine's own veto is EMA20/50 on H4 only. This
            # strategy runs a strictly stronger multi-timeframe gate plus the
            # Zone Respect Index internally, and records both in `reason`, so
            # the decision stays in one auditable place. Flip it on per-bot
            # with `htf_veto_override` in the skill YAML if you want belt
            # and braces.
            htf_veto=False,
            close_on_opposite_signal=True,
            params={
                # ── Zone timeframes (native, not resampled) ──
                "zone_timeframe": "H4",
                "htf_zone_timeframe": "D1",
                "atr_period": 14,

                # ── Zone detection ──
                "base_body_atr_mult": 0.5,
                "leg_travel_atr_mult": 0.8,
                "max_base_candles": 6,
                "v2_max_base_candles": 4,
                # Raised from 1.0: V2 supply zones went 4.5% WR / -$796.12
                # on 2026-08-05, so only violent engulfings qualify now.
                "v2_engulf_min_atr_mult": 1.6,
                "qm_swing_lookback": 3,
                # Raised from 0.5 for the same reason (QM_BEAR 0-for-37).
                "qm_min_swing_atr": 0.8,
                "qm_requires_confluence": True,
                "structure_swing_lookback": 3,
                "structure_depth": 4,

                # ── Candidate ranking ──
                # Source weights are the 2026-08-05 P&L by detector, ranked:
                # SND_V1 +$98.36, QUASIMODO -$225.59, SND_V2 -$730.43. The
                # old family took whichever zone its loop reached first, so
                # the worst detectors regularly front-ran the best one.
                "source_weights": {"SND_V1": 2.0, "SND_V2": 0.5, "QUASIMODO": 0.25},
                "score_impulse_weight": 0.60,
                "score_overlap_weight": 0.80,
                "score_depth_weight": 1.20,
                "score_width_weight": 0.50,
                "score_age_weight": 0.40,

                # ── Trend gate ──
                "trend_ema_fast": 13,
                "trend_ema_slow": 34,
                "trend_slope_bars": 3,
                "trend_weights": {"H1": 1.0, "H4": 1.5, "D1": 2.0},
                # Of a possible +/-4.5. Requires real multi-timeframe
                # agreement, not one timeframe's opinion.
                "trend_min_score": 2.0,
                # When no timeframe set reaches `trend_min_score` the market
                # is judged directionless and trades are still allowed — but
                # the aggregate must not lean *against* the trade. Without
                # this floor the first working build of the M1 sibling still
                # took 8 losing shorts into a mildly bullish tape.
                "neutral_min_score": 0.5,
                # Structure on the zone timeframe must not oppose the trade.
                "require_structure_alignment": True,
                # Exhaustion guard. A trend gate on its own will happily buy
                # the top of a vertical move: the first build of this
                # strategy took exactly two trades on 2026-08-05 and both
                # were longs at 4197-4200, the literal high of the session,
                # right before the drop back to 4186. Refuse entries more
                # than this many ATRs away from the zone timeframe's fast
                # EMA — buying strength is fine, chasing it is not.
                "max_extension_atr": 2.20,
                # Trend fatigue (see Gate 3c). Both default to no-op so
                # the ladder's measured baseline is unchanged until the
                # A/B says a threshold earns its keep:
                #   fatigue_max > 1.0 can never fire (score is in [0, 1])
                #   fatigue_fade_min 0.0 is always satisfied
                "fatigue_max": 1.01,
                "fatigue_fade_min": 0.0,
                "fatigue_min_components": 3,
                # Counter-trend trades off fresh opposing zones. Off by
                # default: on 2026-08-05 they were the entire loss.
                "allow_counter_trend": False,
                "counter_trend_min_rr": 2.5,
                "counter_trend_max_age_bars": 12,
                "counter_trend_zri_min": 0.55,

                # ── Zone Respect Index (adaptive regime read) ──
                "zri_horizon_bars": 12,
                "zri_reversal_atr": 1.0,
                "zri_sample": 12,
                "zri_min_samples": 4,
                "zri_min": 0.40,

                # ── Zone quality gates ──
                # 2026-08-05: height 3-4 USD 53.7% WR / +$366 vs height
                # 6-9 USD 10.4% WR / -$1074. Expressed in ATR so it travels
                # across symbols and volatility regimes.
                "zone_height_atr_min": 0.25,
                "zone_height_atr_max": 2.00,
                "max_zone_age_bars": 120,
                # Depth 0.50-0.75 into the zone: 19.0% WR / -$529.89.
                "max_entry_depth": 0.75,
                "min_impulse_atr": 0.7,
                # One supply zone was sold 18 separate times for -$462 on
                # 2026-08-05, so a zone still gets a hard episode budget.
                # But the counterfactuals on that same day showed the
                # winning filter set kept 200-300 trades: re-entry per se
                # was not the problem, re-entry into a zone the market had
                # already rejected was. 8 kills the spam without killing
                # the H1 trade count.
                "max_touch_episode": 8,
                # In *entry*-timeframe bars. On the M1 sibling 3 killed
                # essentially every zone within minutes (16,456 dead-zone
                # rejections in one session); price legitimately works
                # inside a H4-sized rectangle for a good while before
                # resolving.
                "max_bars_inside": 20,

                # ── H1 entry confirmation ──
                # Both thresholds are measured against the *zone*-timeframe
                # ATR, so 0.35 of a H4 ATR is a real rejection candle on
                # the H1 chart, not the bare touch the old family entered
                # on.
                "confirm_mode": "light",
                "confirm_body_atr_mult": 0.35,
                "confirm_wick_frac": 0.20,

                # ── Stop loss ──
                "sl_zone_buffer_atr_mult": 0.20,
                "sl_zone_buffer_zone_frac": 0.10,
                "sl_min_atr_mult": 0.60,
                # SL 4-5 USD: 22.5% WR / -$516.14. Past this the setup is
                # skipped rather than sized down into a bad stop.
                "sl_max_atr_mult": 2.20,

                # ── Take profit ──
                # The broker-side spread/RR gate (`SpreadGate.check`, symbol
                # config `min_rr`, default 1.5) rejects any order whose TP
                # distance is under min_rr x (SL + spread). The first build
                # set TP1 at 1.0R and every single Scalp leg was refused —
                # 1,169 rejections over a 3-month replay, so the bot silently
                # ran as a one-leg strategy. Keep TP1 above that floor with
                # margin for the volatility tp_scale.
                "broker_min_rr": 1.60,
                "tp1_target_rr": 1.80,
                "tp2_target_rr": 3.50,
                "tp2_min_rr": 2.20,
                # Keeps every target clear of PositionManager rule 3, which
                # arms break-even within 0.5R of an opposing base — the old
                # TP2 sat *on* the opposing zone and so was scratched
                # before filling 80% of the time.
                "tp_frontrun_r_mult": 0.60,
                "tp_buffer_atr_mult": 0.30,

                # ── Volatility adaptation ──
                "vol_lookback": 100,
                "vol_low_pct": 0.20,
                "vol_high_pct": 0.60,
                "vol_extreme_pct": 0.98,
                "vol_sl_scale_low": 0.90,
                "vol_sl_scale_high": 1.35,
                "vol_tp_scale_low": 0.90,
                "vol_tp_scale_high": 1.35,
                "vol_extreme_block": True,
                # Dead-tape filter: skip when the zone timeframe's ATR has
                # collapsed relative to its own recent median. Replaces a
                # hardcoded session window, so it keeps working when the
                # session that matters moves.
                "min_atr_median_frac": 0.55,

                # ── Volatility-shock / news guard ──
                "shock_guard_enabled": True,
                "shock_lookback_bars": 120,
                "shock_range_mult": 5.0,
                # The M1 sibling measured a -5/+15 minute blast radius
                # around a range shock and sat out 15 M1 bars for it;
                # 3 H1 bars is the same window here.
                "shock_cooldown_bars": 3,

                # ── Execution ──
                "max_spread_r_frac": 0.12,
                "point_value": 0.01,
                "max_signals_per_bar": 1,
                "emit_runner_leg": True,
            },
        )
        # Everything derived from the zone timeframe — zones, ATR regime,
        # Zone Respect Index, structure, trend votes — is invariant until a
        # new zone-TF bar closes, so it is computed once per H4 bar instead
        # of once per H1 bar. Without this a long H1 replay would redo the
        # whole 200-bar scan on every single bar.
        self._zone_cache = {}
        self._regime_cache = None
        # Which gate rejected how many candidates, since construction. Pure
        # counters (never unbounded — one key per gate), read by the tuning
        # harness and by the signal-funnel view to answer "why is this bot
        # quiet?" without re-deriving the pipeline by hand.
        self.reject_counts = {}

    # ── internals ────────────────────────────────────────────────────

    def _reject(self, gate):
        self.reject_counts[gate] = self.reject_counts.get(gate, 0) + 1

    def _zones_for(self, frame, timeframe, params):
        atr_period = int(params["atr_period"])
        if frame is None or len(frame) < atr_period + 10:
            return [], None
        stamp = int(pd.Timestamp(frame["time"].iloc[len(frame) - 1]).value)
        key = (stamp, len(frame))
        cached = self._zone_cache.get(timeframe)
        if cached is not None and cached[0] == key:
            return cached[1]

        atr_series = _atr(frame, atr_period)
        valid = atr_series.dropna()
        if valid.empty:
            return [], None
        atr_value = float(valid.iloc[len(valid) - 1])
        if atr_value <= 0:
            return [], None

        zones = (
            _detect_zones_v1(frame, atr_series, params)
            + _detect_zones_v2(frame, atr_series, params)
            + _detect_quasimodo_zones(frame, atr_series, params)
        )
        for zone in zones:
            zone["timeframe"] = timeframe
        result = (zones, {"atr": atr_value, "series": atr_series})
        self._zone_cache[timeframe] = (key, result)
        return result

    def _regime(self, ctx, params, zone_frame, zones, atr_value, atr_series):
        """Zone-timeframe regime bundle: volatility percentile, trend votes,
        market structure and the Zone Respect Index. Cached to the newest
        zone-TF bar (see `__init__`)."""
        stamp = int(pd.Timestamp(zone_frame["time"].iloc[len(zone_frame) - 1]).value)
        key = (stamp, len(zone_frame))
        if self._regime_cache is not None and self._regime_cache[0] == key:
            return self._regime_cache[1]

        history_all = atr_series.dropna().to_numpy()
        lookback = int(params["vol_lookback"])
        history = history_all[max(0, len(history_all) - lookback - 1): len(history_all) - 1]
        vol_pct = _percentile_rank(history, atr_value)
        median_atr = float(np.median(history)) if len(history) > 0 else 0.0

        trend_score, votes = _trend_state(ctx, params)
        fast_ema = _ema_series(zone_frame, int(params["trend_ema_fast"])).to_numpy()
        ema_anchor = float(fast_ema[len(fast_ema) - 1])
        structure = _detect_structure(
            zone_frame["high"].to_numpy(),
            zone_frame["low"].to_numpy(),
            int(params["structure_swing_lookback"]),
        )
        bias = _structure_bias(structure, int(params["structure_depth"]))
        zri = _zone_respect_index(zones, zone_frame, atr_value, params)

        bundle = {
            "vol_pct": vol_pct, "median_atr": median_atr, "trend_score": trend_score,
            "votes": votes, "structure": structure, "bias": bias, "zri": zri,
            "ema_anchor": ema_anchor,
        }
        self._regime_cache = (key, bundle)
        return bundle

    # ── entry point ──────────────────────────────────────────────────

    def evaluate(self, ctx):
        params = self.spec.params

        entry = ctx.candles.get(self.spec.entry_timeframe)
        if entry is None or len(entry) < 60 or "time" not in entry.columns:
            return None

        # ── Shock / news cooldown ─────────────────────────────────────
        # First gate of the pipeline: it is the cheapest check here and it
        # is logically prior to everything else — if the tape just printed
        # a release or a rollover gap, no amount of zone quality makes an
        # H1 entry a good idea.
        if params["shock_guard_enabled"]:
            since = _bars_since_shock(
                entry["high"].to_numpy(), entry["low"].to_numpy(),
                int(params["shock_lookback_bars"]), float(params["shock_range_mult"]),
            )
            if since is not None and since < int(params["shock_cooldown_bars"]):
                self._reject("shock_cooldown")
                return None

        zone_tf = params["zone_timeframe"]
        htf_tf = params["htf_zone_timeframe"]
        zone_frame = ctx.candles.get(zone_tf)
        if zone_frame is None or "time" not in zone_frame.columns:
            return None

        zones, zone_meta = self._zones_for(zone_frame, zone_tf, params)
        if not zones or zone_meta is None:
            return None
        atr_value = zone_meta["atr"]

        htf_frame = ctx.candles.get(htf_tf)
        htf_zones, _ = self._zones_for(htf_frame, htf_tf, params)

        regime_bundle = self._regime(
            ctx, params, zone_frame, zones, atr_value, zone_meta["series"]
        )

        # ── Volatility regime (adaptive sizing + stand-down) ──────────
        vol_pct = regime_bundle["vol_pct"]
        if params["vol_extreme_block"] and vol_pct >= float(params["vol_extreme_pct"]):
            return None
        median_atr = regime_bundle["median_atr"]
        if median_atr > 0 and atr_value < float(params["min_atr_median_frac"]) * median_atr:
            return None

        sl_scale = 1.0
        tp_scale = 1.0
        if vol_pct <= float(params["vol_low_pct"]):
            sl_scale = float(params["vol_sl_scale_low"])
            tp_scale = float(params["vol_tp_scale_low"])
            regime = "LOW"
        elif vol_pct >= float(params["vol_high_pct"]):
            sl_scale = float(params["vol_sl_scale_high"])
            tp_scale = float(params["vol_tp_scale_high"])
            regime = "HIGH"
        else:
            regime = "NORMAL"

        # ── Trend + structure gate ────────────────────────────────────
        trend_score = regime_bundle["trend_score"]
        votes = regime_bundle["votes"]
        trend_min = float(params["trend_min_score"])
        structure = regime_bundle["structure"]
        bias = regime_bundle["bias"]

        # ── Zone Respect Index ────────────────────────────────────────
        zri = regime_bundle["zri"]
        zri_min = float(params["zri_min"])

        # ── Candidate selection ───────────────────────────────────────
        entry_times = pd.DatetimeIndex(entry["time"]).as_unit("ns").asi8
        opens = entry["open"].to_numpy()
        highs = entry["high"].to_numpy()
        lows = entry["low"].to_numpy()
        closes = entry["close"].to_numpy()
        last = len(closes) - 1
        close = float(closes[last])
        now = entry["time"].iloc[last]

        zone_times = pd.DatetimeIndex(zone_frame["time"]).as_unit("ns").asi8
        bar_ns = int(zone_times[1] - zone_times[0]) if len(zone_times) > 1 else 0
        newest_bar = len(zone_frame) - 1

        live = []
        candidates = []
        for zone in zones:
            created = int(zone["leg_out_end"])
            if created >= len(zone_times):
                continue
            created_end_ns = int(zone_times[created]) + bar_ns
            life = _zone_lifecycle(
                zone, created_end_ns, entry_times, highs, lows, closes,
                int(params["max_bars_inside"]),
            )
            if life["dead"]:
                self._reject("zone_dead")
                continue
            live.append(zone)
            if not life["inside_now"]:
                continue
            if life["episodes"] > int(params["max_touch_episode"]):
                self._reject("retest_episode")
                continue

            height = zone["price_high"] - zone["price_low"]
            if height <= 0:
                continue
            height_atr = height / atr_value
            if not (float(params["zone_height_atr_min"]) <= height_atr
                    <= float(params["zone_height_atr_max"])):
                self._reject("zone_height")
                continue

            age_bars = newest_bar - created
            if age_bars > int(params["max_zone_age_bars"]):
                self._reject("zone_age")
                continue
            if zone.get("impulse_atr", 0.0) < float(params["min_impulse_atr"]):
                self._reject("weak_impulse")
                continue

            demand = zone["kind"] == ZoneKind.DEMAND
            if demand:
                depth = (zone["price_high"] - close) / height
            else:
                depth = (close - zone["price_low"]) / height
            if depth > float(params["max_entry_depth"]):
                self._reject("entry_depth")
                continue

            overlaps = _overlap_count(zone, htf_zones)
            if params["qm_requires_confluence"] and zone["source"] == "QUASIMODO" and overlaps == 0:
                self._reject("qm_no_confluence")
                continue

            candidates.append({
                "zone": zone, "depth": max(depth, 0.0), "height": height,
                "height_atr": height_atr, "age_bars": age_bars, "overlaps": overlaps,
                "score": _score_zone(zone, max(depth, 0.0), height_atr, age_bars, overlaps, params),
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: -c["score"])

        signals = []
        for candidate in candidates:
            built = self._build_signals(
                candidate, live, ctx, params, close, now, atr_value, sl_scale, tp_scale,
                trend_score, trend_min, votes, bias, zri, zri_min, regime, vol_pct,
                opens, highs, lows, closes, structure, zone_frame,
                regime_bundle["ema_anchor"],
            )
            if built:
                signals.extend(built)
            if len(signals) >= 2 * int(params["max_signals_per_bar"]):
                break

        return tuple(signals) if signals else None

    def _build_signals(self, candidate, live, ctx, params, close, now, atr_value,
                       sl_scale, tp_scale, trend_score, trend_min, votes, bias,
                       zri, zri_min, regime, vol_pct, opens, highs, lows, closes,
                       structure, zone_frame, ema_anchor):
        zone = candidate["zone"]
        demand = zone["kind"] == ZoneKind.DEMAND
        direction = Direction.BUY if demand else Direction.SELL
        want = 1 if demand else -1

        # ── Gate 1: Zone Respect Index for this side of the book ──────
        zri_value, zri_n = zri.get(zone["kind"], (1.0, 0))
        counter_trend = (trend_score * want) <= -trend_min
        effective_zri_min = (
            float(params["counter_trend_zri_min"]) if counter_trend else zri_min
        )
        if zri_value < effective_zri_min:
            self._reject("zone_respect_index")
            return None

        # ── Gate 2: multi-timeframe trend ─────────────────────────────
        with_trend = (trend_score * want) >= trend_min
        if not with_trend:
            if counter_trend:
                if not params["allow_counter_trend"]:
                    self._reject("counter_trend")
                    return None
                if candidate["age_bars"] > int(params["counter_trend_max_age_bars"]):
                    self._reject("counter_trend_age")
                    return None
            elif abs(trend_score) >= trend_min:
                self._reject("trend_gate")
                return None
            elif (trend_score * want) < float(params["neutral_min_score"]):
                self._reject("neutral_trend_lean")
                return None

        # ── Gate 3: structure on the zone timeframe ───────────────────
        if params["require_structure_alignment"] and bias != 0 and bias != want:
            self._reject("structure_bias")
            return None

        # ── Gate 3b: exhaustion — do not chase a vertical move ────────
        extension = (close - ema_anchor) * want / atr_value
        if extension > float(params["max_extension_atr"]):
            self._reject("overextended")
            return None

        # ── Gate 3c: trend fatigue (essoufflement) ────────────────────
        # Gate 3b asks one question — "is price far from its anchor?" —
        # and answers it with a cliff. This asks the broader one: is the
        # move we would be joining still breathing? Momentum divergence,
        # leg decay, range/volume drain, rejection wicks and stretch all
        # vote, so a trend that is dying *without* being stretched (drift
        # decaying, ranges contracting) is caught too, which 3b cannot
        # see by construction.
        #
        # Two independent uses, both off by default so the A/B measures
        # this gate alone and nothing else:
        #   fatigue_max      veto a continuation entry into a tired move
        #   fatigue_fade_min require the move being faded to *be* tired
        #                    before a counter-trend entry is allowed
        # Measured on the zone timeframe, not M1: the trend being joined
        # is the one the zone was found in, and `atr_value` is already
        # that frame's ATR, so the ATR-normalised parts stay consistent.
        fatigue_max = float(params["fatigue_max"])
        fatigue_fade_min = float(params["fatigue_fade_min"])
        if fatigue_max <= 1.0 or fatigue_fade_min > 0.0:
            # Fading a move running against us reads the fatigue of *that*
            # move, so the direction flips for a counter-trend entry.
            measured = -want if counter_trend else want
            reading = trend_fatigue(
                opens=zone_frame["open"].to_numpy(),
                highs=zone_frame["high"].to_numpy(),
                lows=zone_frame["low"].to_numpy(),
                closes=zone_frame["close"].to_numpy(),
                volumes=zone_frame["tick_volume"].to_numpy(),
                direction=measured,
                atr=atr_value,
                min_components=int(params["fatigue_min_components"]),
            )
            # `available == 0` is "could not measure", never "exhausted" —
            # an unknown must not veto a trade on its own.
            if reading.available:
                if counter_trend:
                    if reading.score < fatigue_fade_min:
                        self._reject("fade_needs_fatigue")
                        return None
                elif reading.score > fatigue_max:
                    self._reject("trend_fatigue")
                    return None
                fatigue_note = reading.describe()
            else:
                fatigue_note = ""
        else:
            fatigue_note = ""

        # ── Gate 4: the H1 candle must actually reject the zone ───────
        confirmed, confirm_note = _confirmation(
            zone, opens, highs, lows, closes, atr_value, params
        )
        if not confirmed:
            self._reject(f"confirm:{confirm_note.split()[0]}")
            return None

        # ── Stop loss: zone-anchored, buffered, ATR-capped ────────────
        height = candidate["height"]
        buffer = max(
            atr_value * float(params["sl_zone_buffer_atr_mult"]),
            height * float(params["sl_zone_buffer_zone_frac"]),
        )
        if demand:
            sl_points = (close - zone["price_low"]) + buffer
        else:
            sl_points = (zone["price_high"] - close) + buffer
        sl_points = max(sl_points, atr_value * float(params["sl_min_atr_mult"])) * sl_scale
        sl_cap = atr_value * float(params["sl_max_atr_mult"])
        if sl_points <= 0 or sl_points > sl_cap:
            self._reject("sl_too_wide")
            return None

        # ── Spread sanity ─────────────────────────────────────────────
        spread_price = float(ctx.spread_points) * float(params["point_value"])
        if spread_price > float(params["max_spread_r_frac"]) * sl_points:
            self._reject("spread")
            return None
        risk = sl_points + spread_price

        # ── Take profit: front-run the opposing zone by >= 0.6R ───────
        frontrun = float(params["tp_frontrun_r_mult"]) * risk
        tp_buffer = max(frontrun, atr_value * float(params["tp_buffer_atr_mult"]))
        opposing = [z for z in live if z["kind"] != zone["kind"]]
        if demand:
            ahead = sorted(
                [z["price_low"] - close for z in opposing if z["price_low"] > close]
            )
        else:
            ahead = sorted(
                [close - z["price_high"] for z in opposing if z["price_high"] < close]
            )
        ceiling = (ahead[0] - tp_buffer) if ahead else None

        tp1 = risk * float(params["tp1_target_rr"]) * tp_scale
        tp2 = risk * float(params["tp2_target_rr"]) * tp_scale
        if ceiling is not None:
            tp1 = min(tp1, ceiling)
            tp2 = min(tp2, ceiling)

        # Anything under the broker gate is an order that will be refused,
        # not a conservative target — so the runner must clear it, and the
        # scalp leg is dropped rather than emitted unfillable.
        broker_floor = risk * float(params["broker_min_rr"])
        min_rr = (
            float(params["counter_trend_min_rr"]) if counter_trend
            else float(params["tp2_min_rr"])
        )
        if tp2 < max(risk * min_rr, broker_floor):
            self._reject("min_rr")
            return None
        emit_scalp = tp1 >= broker_floor

        # ── Annotations ───────────────────────────────────────────────
        base_index = min(max(int(zone["base_start"]), 0), len(zone_frame) - 1)
        annotation = PriceZone(
            kind=zone["kind"],
            pattern=zone["pattern"],
            price_low=zone["price_low"],
            price_high=zone["price_high"],
            time_start=zone_frame["time"].iloc[base_index],
            time_end=now,
        )
        points = tuple(
            StructurePoint(
                time=zone_frame["time"].iloc[min(max(int(s["index"]), 0), len(zone_frame) - 1)],
                price=s["price"],
                label=s["label"],
            )
            for s in structure[len(structure) - 4:]
        )
        readings = (
            IndicatorReading(
                name="trend_score", value=round(trend_score, 2), threshold=trend_min,
                comparison=">" if want > 0 else "<", passed=True,
            ),
            IndicatorReading(
                name=f"zri_{zone['kind'].value}", value=round(zri_value, 2),
                threshold=effective_zri_min, comparison=">", passed=True,
            ),
            IndicatorReading(
                name="zone_height_atr", value=round(candidate["height_atr"], 2),
                threshold=float(params["zone_height_atr_max"]), comparison="<", passed=True,
            ),
            IndicatorReading(
                name="entry_depth", value=round(candidate["depth"], 2),
                threshold=float(params["max_entry_depth"]), comparison="<", passed=True,
            ),
            IndicatorReading(
                name="vol_pct", value=round(vol_pct, 2),
                threshold=float(params["vol_extreme_pct"]), comparison="<", passed=True,
            ),
        )

        mode = "counter" if counter_trend else ("trend" if with_trend else "neutral")
        head = (
            f"APEX/{zone['source']}/{zone['pattern']}@{zone['timeframe']} "
            f"[{zone['price_low']:.2f},{zone['price_high']:.2f}] "
            f"trend={trend_score:+.1f}{votes} bias={bias:+d} mode={mode} "
            f"zri={zri_value:.2f}(n={zri_n}) vol={regime}({vol_pct:.2f}) "
            f"h={candidate['height_atr']:.2f}atr depth={candidate['depth']:.2f} "
            f"ext={extension:+.2f}atr "
            # Empty unless the fatigue gate is enabled, so the reason line
            # of a default-configured run is byte-identical to before.
            f"{fatigue_note + ' ' if fatigue_note else ''}"
            f"conf[{confirm_note}] overlap={candidate['overlaps']} sl={sl_points:.2f}"
        )

        signals = []
        if emit_scalp:
            signals.append(
                Signal(
                    direction=direction,
                    sl_points=float(sl_points),
                    tp_points=float(tp1),
                    confidence=0.80,
                    reason=f"{head} tp1={tp1:.2f} rr={tp1 / risk:.2f} (Scalp)",
                    zone=annotation,
                    pattern=zone["pattern"],
                    structure=points,
                    indicators=readings,
                )
            )
        if params["emit_runner_leg"] or not signals:
            signals.append(
                Signal(
                    direction=direction,
                    sl_points=float(sl_points),
                    tp_points=float(tp2),
                    confidence=0.70,
                    reason=f"{head} tp2={tp2:.2f} rr={tp2 / risk:.2f} (Runner)",
                    zone=annotation,
                    pattern=zone["pattern"],
                    structure=points,
                    indicators=readings,
                )
            )
        return signals
