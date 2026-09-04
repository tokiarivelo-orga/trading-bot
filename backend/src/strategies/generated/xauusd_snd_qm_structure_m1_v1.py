"""XAUUSD S&D + Quasimodo + Market Structure — M1 Scalping preset.

Trades XAUUSD using three zone-detection methods and market structure:

  1. Supply & Demand V1 (classic leg-base-leg: RBR/DBD/RBD/DBR)
  2. Supply & Demand V2 (engulfing-base: strong candle creates the zone
     boundary, followed by a consolidation base, then a departure)
  3. Quasimodo patterns (the shoulder-head-shoulder variant where the
     second shoulder fails to reach the first's level, creating a zone
     at the failure point)

Entry: price touches/enters a valid, unbroken zone → trade fires.
Stop Loss: exactly the zone height below (buy) or above (sell) entry.
Take Profit: next opposite zone minus a configurable buffer.
Break-even: via market structure — BUY waits for confirmed HH then
  moves SL to latest HL; SELL waits for confirmed LL then moves SL
  to latest LH. The engine's own PositionManager handles the actual
  SL modification; this strategy's Signal carries the structure points
  as annotation data.

Position rules: one trade at a time, no hedging, no pyramiding.
No higher-timeframe trend filter, no R:R filter — every valid signal
trades, per the prompt spec.

This is the M1 Scalping preset: fast entries on 1-minute candles,
zones detected on a 5-minute resampled timeframe, tight parameters
for quick resolution.

v1 — initial implementation from the xauusd_trading_bot_prompt spec.
"""

import numpy as np
import pandas as pd

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    PriceZone,
    Signal,
    StrategySpec,
    StructureLabel,
    StructurePoint,
    ZoneKind,
)

# ─────────────────────────────────────────────────────────────────────
# ATR helpers
# ─────────────────────────────────────────────────────────────────────


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=period).mean()


# ─────────────────────────────────────────────────────────────────────
# Resampling: bucket entry-TF candles into zone-TF bars
# ─────────────────────────────────────────────────────────────────────


def _resample(df: pd.DataFrame, tf_minutes: int, entry_tf_minutes: int) -> tuple:
    """Bucket entry-TF rows into tf_minutes OHLC bars (numpy reduceat).
    Returns (zone_frame, zone_end_ns) or None."""
    if "time" not in df.columns:
        return None
    t_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
    step = np.int64(tf_minutes) * 60 * 1_000_000_000
    entry_ns = np.int64(entry_tf_minutes) * 60 * 1_000_000_000
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


def _build_runs(classes, start=0, stop=None):
    end_pos = len(classes) if stop is None else stop
    runs = []
    for i in range(start, end_pos):
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


def _detect_zones_v1(df, atr_series, params):
    """Supply & Demand V1: classic leg-base-leg geometry (RBR/DBD/RBD/DBR)."""
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    base_mult = params["base_body_atr_mult"]
    leg_mult = params["leg_travel_atr_mult"]
    max_base = int(params["max_base_candles"])

    classes = _classify_bars(closes, opens, atr_filled, base_mult)
    runs = _build_runs(classes, 0)
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
        zones.append(
            {
                "source": "SND_V1",
                "pattern": pattern,
                "kind": ZoneKind.DEMAND if leg_out_up else ZoneKind.SUPPLY,
                "price_high": price_high,
                "price_low": price_low,
                "base_start": base_start,
                "conf_idx": conf_idx,
                "leg_out_end": leg_out[2],
            }
        )
    return zones


# ─────────────────────────────────────────────────────────────────────
# Supply & Demand V2: engulfing-base departure zones
# ─────────────────────────────────────────────────────────────────────


def _detect_zones_v2(df, atr_series, params):
    """Supply & Demand V2: a strong engulfing candle creates the zone edge,
    followed by small-body consolidation candles (the base), then a
    departure candle that closes clear of the base in the same direction.

    The zone rectangle spans from the engulfing candle's extreme to the
    base band, capturing the institutional order block."""
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(closes)

    base_mult = params["base_body_atr_mult"]
    max_base = int(params.get("v2_max_base_candles", params["max_base_candles"]))
    min_engulf_atr = params.get("v2_engulf_min_atr_mult", 1.2)

    zones = []
    i = 1
    while i < n - 2:
        body = abs(closes[i] - opens[i])
        if body < min_engulf_atr * atr_filled[i]:
            i += 1
            continue
        bullish = closes[i] > opens[i]

        # Look for base candles after the engulfing
        base_start = i + 1
        base_end = base_start
        while base_end < n - 1 and (base_end - base_start) < max_base:
            if abs(closes[base_end] - opens[base_end]) > base_mult * atr_filled[base_end]:
                break
            base_end += 1

        base_count = base_end - base_start
        if base_count < 1:
            i += 1
            continue

        # Check departure candle
        dep_idx = base_end
        if dep_idx >= n:
            break

        price_high = float(max(highs[i], highs[base_start:base_end].max()))
        price_low = float(min(lows[i], lows[base_start:base_end].min()))

        # Demand zone: departure must close above base high
        if bullish and closes[dep_idx] > price_high:
            zones.append(
                {
                    "source": "SND_V2",
                    "pattern": "DZ_V2",
                    "kind": ZoneKind.DEMAND,
                    "price_high": price_high,
                    "price_low": price_low,
                    "base_start": base_start,
                    "conf_idx": dep_idx,
                    "leg_out_end": dep_idx,
                }
            )
        else:
            # Supply zone: departure must close below base low
            if closes[dep_idx] < price_low:
                zones.append(
                    {
                        "source": "SND_V2",
                        "pattern": "SZ_V2",
                        "kind": ZoneKind.SUPPLY,
                        "price_high": price_high,
                        "price_low": price_low,
                        "base_start": base_start,
                        "conf_idx": dep_idx,
                        "leg_out_end": dep_idx,
                    }
                )
        i = dep_idx + 1

    return zones


# ─────────────────────────────────────────────────────────────────────
# Quasimodo Pattern Detection
# ─────────────────────────────────────────────────────────────────────


def _detect_swing_points(highs, lows, lookback):
    """Detect swing highs and swing lows using a simple N-bar lookback.
    Returns list of (index, price, 'high'|'low')."""
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        # Swing high: highest in the window
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] == window_highs.max() and sum(window_highs == highs[i]) == 1:
            swings.append((i, float(highs[i]), "high"))
        # Swing low: lowest in the window
        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] == window_lows.min() and sum(window_lows == lows[i]) == 1:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda x: x[0])
    return swings


def _detect_quasimodo_zones(df, atr_series, params):
    """Quasimodo pattern: a variant of head-and-shoulders where the second
    shoulder fails to reach the first shoulder's level, creating an
    imbalance zone.

    Bullish Quasimodo (demand): HL → HH → LL → (price fails to make new HH)
      Zone at the LL area — the failure swing creates demand.

    Bearish Quasimodo (supply): LH → LL → HH → (price fails to make new LL)
      Zone at the HH area — the failure swing creates supply."""
    valid_atr = atr_series.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_series.fillna(valid_atr.iloc[0]).to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    df["close"].to_numpy()

    lookback = int(params.get("qm_swing_lookback", 3))
    min_swing_atr = params.get("qm_min_swing_atr", 0.5)

    swings = _detect_swing_points(highs, lows, lookback)
    if len(swings) < 4:
        return []

    zones = []
    for idx in range(len(swings) - 3):
        s1, s2, s3, s4 = swings[idx], swings[idx + 1], swings[idx + 2], swings[idx + 3]

        # Bullish QM: swing low → swing high → lower low → lower high
        # Pattern: the second low goes below the first, but the subsequent
        # high fails to reach the previous high → demand zone at the lower low
        # s3 must be lower than s1 (lower low)
        # s4 must fail to reach s2 (lower high, or just fail)
        if (
            s1[2] == "low"
            and s2[2] == "high"
            and s3[2] == "low"
            and s4[2] == "high"
            and s3[1] < s1[1]
            and s4[1] < s2[1]
        ):
            # Swing magnitude check
            swing_range = abs(s2[1] - s3[1])
            atr_at = atr_filled[min(s3[0], len(atr_filled) - 1)]
            if swing_range >= min_swing_atr * atr_at:
                # Zone around the lower low (s3)
                zone_idx = s3[0]
                zone_low = float(lows[zone_idx])
                zone_high = float(highs[zone_idx])
                # Extend zone slightly using surrounding bars
                ext = max(1, lookback // 2)
                start = max(0, zone_idx - ext)
                end = min(len(lows), zone_idx + ext + 1)
                zone_low = float(lows[start:end].min())
                zone_high = max(zone_high, float(highs[start:end].max()))
                # But keep zone height reasonable
                max_height = 2.0 * atr_at
                if zone_high - zone_low > max_height:
                    mid = (zone_high + zone_low) / 2
                    zone_high = mid + max_height / 2
                    zone_low = mid - max_height / 2

                zones.append(
                    {
                        "source": "QUASIMODO",
                        "pattern": "QM_BULL",
                        "kind": ZoneKind.DEMAND,
                        "price_high": zone_high,
                        "price_low": zone_low,
                        "base_start": zone_idx,
                        "conf_idx": s4[0],
                        "leg_out_end": s4[0],
                    }
                )

        # Bearish QM: swing high → swing low → higher high → higher low
        # Pattern: the second high goes above the first, but the subsequent
        # low fails to reach the previous low → supply zone at the higher high
        # s3 must be higher than s1 (higher high)
        # s4 must fail to reach s2 (higher low)
        if (
            s1[2] == "high"
            and s2[2] == "low"
            and s3[2] == "high"
            and s4[2] == "low"
            and s3[1] > s1[1]
            and s4[1] > s2[1]
        ):
            swing_range = abs(s3[1] - s2[1])
            atr_at = atr_filled[min(s3[0], len(atr_filled) - 1)]
            if swing_range >= min_swing_atr * atr_at:
                zone_idx = s3[0]
                zone_high = float(highs[zone_idx])
                zone_low = float(lows[zone_idx])
                ext = max(1, lookback // 2)
                start = max(0, zone_idx - ext)
                end = min(len(highs), zone_idx + ext + 1)
                zone_high = float(highs[start:end].max())
                zone_low = min(zone_low, float(lows[start:end].min()))
                max_height = 2.0 * atr_at
                if zone_high - zone_low > max_height:
                    mid = (zone_high + zone_low) / 2
                    zone_high = mid + max_height / 2
                    zone_low = mid - max_height / 2

                zones.append(
                    {
                        "source": "QUASIMODO",
                        "pattern": "QM_BEAR",
                        "kind": ZoneKind.SUPPLY,
                        "price_high": zone_high,
                        "price_low": zone_low,
                        "base_start": zone_idx,
                        "conf_idx": s4[0],
                        "leg_out_end": s4[0],
                    }
                )

    return zones


# ─────────────────────────────────────────────────────────────────────
# Market Structure Detection (HH / HL / LH / LL)
# ─────────────────────────────────────────────────────────────────────


def _detect_structure(highs, lows, lookback):
    """Label swing points as HH/HL/LH/LL based on their relationship to
    the previous swing of the same type. Uses only closed candles.

    Returns list of dicts: {index, price, label (StructureLabel)}."""
    swings = _detect_swing_points(highs, lows, lookback)
    if len(swings) < 2:
        return []

    structure = []
    last_high = None
    last_low = None

    for idx, price, typ in swings:
        if typ == "high":
            if last_high is not None:
                label = StructureLabel.HH if price > last_high[1] else StructureLabel.LH
            else:
                label = StructureLabel.HH  # first high, default
            structure.append({"index": idx, "price": price, "label": label})
            last_high = (idx, price)
        else:  # low
            if last_low is not None:
                label = StructureLabel.LL if price < last_low[1] else StructureLabel.HL
            else:
                label = StructureLabel.HL  # first low, default
            structure.append({"index": idx, "price": price, "label": label})
            last_low = (idx, price)

    return structure


# ─────────────────────────────────────────────────────────────────────
# Zone tracking on entry-TF candles
# ─────────────────────────────────────────────────────────────────────


def _track_zone_on_entry_tf(zone, zone_end_ns, entry_t_ns, entry_highs, entry_lows, entry_closes):
    """Track whether a zone has been broken and find retest episodes
    on the entry timeframe. Returns (is_broken, is_price_in_zone_now)."""
    start = int(np.searchsorted(entry_t_ns, zone_end_ns, side="left"))
    if start >= len(entry_t_ns):
        return False, False

    demand = zone["kind"] == ZoneKind.DEMAND

    # Check if zone is broken (close through the far side)
    if demand:
        broke = entry_closes[start:] < zone["price_low"]
    else:
        broke = entry_closes[start:] > zone["price_high"]

    is_broken = bool(np.any(broke))

    # Check if current price is touching/in the zone
    last_idx = len(entry_closes) - 1
    if demand:
        in_zone = entry_lows[last_idx] <= zone["price_high"]
    else:
        in_zone = entry_highs[last_idx] >= zone["price_low"]

    return is_broken, in_zone


# ─────────────────────────────────────────────────────────────────────
# Main strategy class — M1 Scalping Preset
# ─────────────────────────────────────────────────────────────────────


class XauusdSndQmStructureM1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="xauusd_snd_qm_structure_m1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            confirmation_timeframes=(),
            htf_veto=False,  # No HTF trend confirmation per prompt spec
            close_on_opposite_signal=True,
            params={
                # ── Zone timeframe (resample entry candles into this) ──
                "zone_tf_minutes": 5,
                "entry_tf_minutes": 1,
                # ── Zone detection common params ──
                "atr_period": 14,
                "base_body_atr_mult": 0.5,
                "leg_travel_atr_mult": 0.7,
                "max_base_candles": 6,
                # ── S&D V2 specific ──
                "v2_max_base_candles": 4,
                "v2_engulf_min_atr_mult": 1.0,
                # ── Quasimodo specific ──
                "qm_swing_lookback": 3,
                "qm_min_swing_atr": 0.5,
                # ── Market structure ──
                "structure_swing_lookback": 3,
                # ── Take Profit ──
                "tp_buffer_points": 20,
                "point_value": 0.01,
                # ── Optimized SL & Tiered Multi-TP parameters ──
                "sl_zone_buffer_atr_mult": 0.15,  # Stop-hunt & spread survival buffer
                "sl_zone_buffer_zone_frac": 0.10,  # Additional zone headroom fraction
                "sl_min_atr_mult": 0.5,  # Minimum ATR stop loss floor
                "tp1_target_rr": 1.2,  # Quick scalp target (locks in early profit)
                "tp2_target_rr": 2.5,  # Primary zone target fallback
                "tp3_target_rr": 4.0,  # Structural runner target fallback
                # Dynamic ATR front-run distance before liquidity rejection
                "tp_buffer_atr_mult": 0.3,
                # ── Break Even ──
                "be_buffer_points": 10,
                # ── Position management ──
                "one_trade_at_a_time": True,
            },
        )

    def evaluate(self, ctx: MarketContext) -> tuple[Signal, ...] | None:
        params = self.spec.params
        entry_tf = self.spec.entry_timeframe
        df = ctx.candles.get(entry_tf)

        if df is None or len(df) < 60:
            return None
        if "time" not in df.columns:
            return None

        zone_tf = int(params["zone_tf_minutes"])
        entry_tf_min = int(params["entry_tf_minutes"])
        atr_period = int(params["atr_period"])

        # Resample entry-TF candles to zone-TF for zone detection
        resampled = _resample(df, zone_tf, entry_tf_min)
        if resampled is None:
            return None
        zone_frame, zone_end_ns = resampled

        if len(zone_frame) < atr_period + 6:
            return None

        atr_series = _atr(zone_frame, atr_period)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return None
        atr_val = float(atr_val)

        # ── Detect all zone types ──
        zones_v1 = _detect_zones_v1(zone_frame, atr_series, params)
        zones_v2 = _detect_zones_v2(zone_frame, atr_series, params)
        zones_qm = _detect_quasimodo_zones(zone_frame, atr_series, params)

        all_zones = zones_v1 + zones_v2 + zones_qm
        if not all_zones:
            return None

        # ── Track zones on entry TF and find valid candidates ──
        entry_t_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
        entry_highs = df["high"].to_numpy()
        entry_lows = df["low"].to_numpy()
        entry_closes = df["close"].to_numpy()

        last_i = len(df) - 1
        close = float(df["close"].iloc[last_i])
        t = df["time"].iloc[last_i]

        live_zones = []
        candidate = None
        for z in all_zones:
            leg_out_end = z["leg_out_end"]
            if leg_out_end >= len(zone_end_ns):
                continue
            z_end_ns = zone_end_ns[leg_out_end]
            is_broken, is_in_zone = _track_zone_on_entry_tf(
                z, z_end_ns, entry_t_ns, entry_highs, entry_lows, entry_closes
            )
            if is_broken:
                continue
            live_zones.append(z)

            # Entry: price touches or enters the zone, zone still valid
            if is_in_zone and candidate is None:
                candidate = z

        if candidate is None:
            return None

        demand = candidate["kind"] == ZoneKind.DEMAND
        direction = Direction.BUY if demand else Direction.SELL

        # ── Optimized Stop Loss (anchored to execution price + dynamic
        # buffer + ATR floor) ──
        zone_height = candidate["price_high"] - candidate["price_low"]
        if zone_height <= 0:
            return None

        atr_val = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 1.0
        buffer = max(
            atr_val * float(params.get("sl_zone_buffer_atr_mult", 0.15)),
            zone_height * float(params.get("sl_zone_buffer_zone_frac", 0.10)),
        )

        if demand:
            sl_points = (close - candidate["price_low"]) + buffer
        else:
            sl_points = (candidate["price_high"] - close) + buffer

        # Enforce minimum ATR floor so normal Gold volatility doesn't stop out narrow bases
        sl_points = max(sl_points, atr_val * float(params.get("sl_min_atr_mult", 0.5)))
        if sl_points <= 0:
            return None

        # Calculate true risk including spread
        spread_price = float(ctx.spread_points) * float(params.get("point_value", 0.01))
        risk_price = sl_points + spread_price

        # ── Tiered Take Profit calculation for 3 target positions (TP1, TP2, TP3) ──
        tp_buffer = max(
            float(params.get("tp_buffer_points", 20)) * float(params.get("point_value", 0.01)),
            atr_val * float(params.get("tp_buffer_atr_mult", 0.3)),
        )
        opposite_zones = [z for z in live_zones if z["kind"] != candidate["kind"]]

        # 1. Locate primary and secondary liquidity zones
        zone_target_1 = None
        zone_target_2 = None
        if demand:
            supply_above = sorted(
                [z for z in opposite_zones if z["price_low"] > close], key=lambda z: z["price_low"]
            )
            if len(supply_above) >= 1:
                zone_target_1 = supply_above[0]["price_low"] - close - tp_buffer
            if len(supply_above) >= 2:
                zone_target_2 = supply_above[1]["price_low"] - close - tp_buffer
        else:
            demand_below = sorted(
                [z for z in opposite_zones if z["price_high"] < close],
                key=lambda z: z["price_high"],
                reverse=True,
            )
            if len(demand_below) >= 1:
                zone_target_1 = close - demand_below[0]["price_high"] - tp_buffer
            if len(demand_below) >= 2:
                zone_target_2 = close - demand_below[1]["price_high"] - tp_buffer

        # 2. Determine TP1 (Quick Scalp / Momentum target): short distance
        # to pay spread and trigger breakeven
        tp1_points = risk_price * float(params.get("tp1_target_rr", 1.2))
        if zone_target_1 is not None and zone_target_1 > risk_price * 0.8:
            tp1_points = min(tp1_points, zone_target_1 * 0.5)
        tp1_points = max(tp1_points, risk_price * 0.8)

        # 3. Determine TP2 (Primary Zone target): target the primary opposite S&D zone
        if zone_target_1 is not None and zone_target_1 > tp1_points:
            tp2_points = zone_target_1
        else:
            tp2_points = max(
                tp1_points + risk_price * 1.0, risk_price * float(params.get("tp2_target_rr", 2.5))
            )

        # 4. Determine TP3 (Structural Runner target): target secondary zone or trend extension
        if zone_target_2 is not None and zone_target_2 > tp2_points:
            tp3_points = zone_target_2
        else:
            tp3_points = max(
                tp2_points + risk_price * 1.5, risk_price * float(params.get("tp3_target_rr", 4.0))
            )

        # ── Market Structure for annotation ──
        struct_lookback = int(params["structure_swing_lookback"])
        zone_highs = zone_frame["high"].to_numpy()
        zone_lows = zone_frame["low"].to_numpy()
        structure = _detect_structure(zone_highs, zone_lows, struct_lookback)

        structure_points = tuple(
            StructurePoint(
                time=zone_frame["time"].iloc[min(max(int(s["index"]), 0), len(zone_frame) - 1)],
                price=s["price"],
                label=s["label"],
            )
            for s in structure[-4:]
        )

        base_idx = min(max(int(candidate.get("base_start", 0)), 0), len(zone_frame) - 1)
        zone_annotation = PriceZone(
            kind=candidate["kind"],
            pattern=candidate.get("pattern"),
            price_low=candidate["price_low"],
            price_high=candidate["price_high"],
            time_start=zone_frame["time"].iloc[base_idx],
            time_end=t,
        )

        base_reason = (
            f"{candidate['source']}/{candidate['pattern']} "
            f"[{candidate['price_low']:.2f},{candidate['price_high']:.2f}] "
            f"sl={sl_points:.2f}"
        )

        sig_tp1 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp1_points),
            confidence=0.75,
            reason=f"{base_reason} tp1={tp1_points:.2f} (Scalp)",
            zone=zone_annotation,
            pattern=candidate.get("pattern"),
            structure=structure_points,
        )
        sig_tp2 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp2_points),
            confidence=0.70,
            reason=f"{base_reason} tp2={tp2_points:.2f} (Zone)",
            zone=zone_annotation,
            pattern=candidate.get("pattern"),
            structure=structure_points,
        )
        sig_tp3 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp3_points),
            confidence=0.65,
            reason=f"{base_reason} tp3={tp3_points:.2f} (Runner)",
            zone=zone_annotation,
            pattern=candidate.get("pattern"),
            structure=structure_points,
        )

        return (sig_tp1, sig_tp2, sig_tp3)
