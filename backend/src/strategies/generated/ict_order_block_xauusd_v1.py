"""ICT Order Block strategy for XAUUSD (Derek Salvon's "ICT Order Block
Trading" course).

Multi-timeframe cascade, mirroring the book's weekly-order-flow /
H4-zone / lower-TF-refinement structure while respecting this project's
fixed M5 entry timeframe (project rule F6):

  H4  -> directional bias only (close vs SMA) — stands in for the book's
         "measure order flow on a higher timeframe" step.
  H1  -> market-structure shift (MSS/MSB) detection + Order Block
         identification + the 50% Fibonacci discount/premium filter.
  M5  -> ICT Kill Zone time filter, the retest-and-rejection entry
         trigger, and Fair Value Gap (FVG) confirmation.

Order Block: after an MSS (price breaking a swing high following a swing
low = bullish shift, and the mirror for bearish), the Order Block is the
last opposite-colour candle in the leg that produced the break — a
down-close candle for a bullish OB, an up-close candle for a bearish OB.

Fair Value Gap: the classic 3-candle imbalance (candle 1's high/low does
not overlap candle 3's low/high), used here as the entry-refinement
signal the book layers on top of the OB retest.

Kill Zones: London (07:00-10:00 UTC = 02:00-05:00 EST) and New York
(12:00-15:00 UTC = 07:00-10:00 EST) — the book's high-probability windows.
DST is ignored deliberately: this is a rule-of-thumb session filter, not
an exact broker-time gate.

v1 — initial implementation from the ICT Order Block Trading PDF.
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
# Swing points (rolling-window fractal highs/lows)
# ─────────────────────────────────────────────────────────────────────

def _detect_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int) -> list[tuple]:
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_highs.max() and int(np.sum(window_highs == highs[i])) == 1:
            swings.append((i, float(highs[i]), "high"))
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_lows.min() and int(np.sum(window_lows == lows[i])) == 1:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda x: x[0])
    return swings


# ─────────────────────────────────────────────────────────────────────
# Market Structure Shift (MSS/MSB) detection
# ─────────────────────────────────────────────────────────────────────

def _find_mss(
    closes: np.ndarray, swings: list[tuple], max_bars_since_break: int
) -> dict | None:
    """Finds the most recent break-of-structure: a swing high followed by a
    swing low then a close back above that swing high (bullish), or a swing
    low followed by a swing high then a close back below that swing low
    (bearish). Returns the candidate anchored to the most recent pivot pair,
    or None if no fresh break exists within `max_bars_since_break` bars."""
    n = len(closes)
    last_idx = n - 1
    best = None
    for i in range(len(swings) - 1):
        a_idx, a_price, a_type = swings[i]
        b_idx, b_price, b_type = swings[i + 1]
        if a_type == "high" and b_type == "low":
            direction = "buy"
        elif a_type == "low" and b_type == "high":
            direction = "sell"
        else:
            continue

        break_idx = None
        for j in range(b_idx + 1, n):
            if direction == "buy" and closes[j] > a_price:
                break_idx = j
                break
            if direction == "sell" and closes[j] < a_price:
                break_idx = j
                break
        if break_idx is None or (last_idx - break_idx) > max_bars_since_break:
            continue

        candidate = {
            "direction": direction,
            "pivot_idx": b_idx,
            "pivot_price": b_price,
            "structure_idx": a_idx,
            "structure_price": a_price,
            "break_idx": break_idx,
        }
        if best is None or b_idx > best["pivot_idx"]:
            best = candidate
    return best


# ─────────────────────────────────────────────────────────────────────
# Order Block identification
# ─────────────────────────────────────────────────────────────────────

def _find_order_block(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    leg_start_idx: int, break_idx: int, bullish: bool,
) -> dict | None:
    """The last opposite-colour candle inside [leg_start_idx, break_idx) —
    the last down-close candle before a bullish impulse, or the last
    up-close candle before a bearish impulse."""
    ob_idx = None
    for k in range(break_idx - 1, leg_start_idx - 1, -1):
        is_down = closes[k] < opens[k]
        if bullish and is_down:
            ob_idx = k
            break
        if not bullish and not is_down and closes[k] > opens[k]:
            ob_idx = k
            break
    if ob_idx is None:
        ob_idx = leg_start_idx
    return {
        "idx": ob_idx,
        "price_low": float(lows[ob_idx]),
        "price_high": float(highs[ob_idx]),
    }


# ─────────────────────────────────────────────────────────────────────
# Fair Value Gap detection
# ─────────────────────────────────────────────────────────────────────

def _detect_fvgs(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    start: int, end: int, bullish: bool,
) -> list[tuple]:
    """3-candle imbalance: candle i is the displacement candle; a bullish
    FVG needs candle (i-1)'s high below candle (i+1)'s low, with candle i
    itself a same-direction (up) candle — and the mirror for bearish."""
    gaps = []
    lo = max(start, 1)
    hi = min(end, len(closes) - 1)
    for i in range(lo, hi):
        if bullish:
            if closes[i] > opens[i] and lows[i + 1] > highs[i - 1]:
                gaps.append((i, float(highs[i - 1]), float(lows[i + 1])))
        else:
            if closes[i] < opens[i] and highs[i + 1] < lows[i - 1]:
                gaps.append((i, float(highs[i + 1]), float(lows[i - 1])))
    return gaps


def _ranges_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return low_a <= high_b and low_b <= high_a


# ─────────────────────────────────────────────────────────────────────
# Structure labels (chart annotation only)
# ─────────────────────────────────────────────────────────────────────

def _label_structure(swings: list[tuple]) -> list[dict]:
    structure = []
    last_high = None
    last_low = None
    for idx, price, typ in swings:
        if typ == "high":
            is_new_high = last_high is None or price > last_high
            label = StructureLabel.HH if is_new_high else StructureLabel.LH
            structure.append({"index": idx, "price": price, "label": label})
            last_high = price
        else:
            is_new_low = last_low is None or price > last_low
            label = StructureLabel.HL if is_new_low else StructureLabel.LL
            structure.append({"index": idx, "price": price, "label": label})
            last_low = price
    return structure


# ─────────────────────────────────────────────────────────────────────
# Kill Zones (UTC hours, half-open [start, end))
# ─────────────────────────────────────────────────────────────────────

def _in_kill_zone(hour: int, kill_zones_utc: tuple) -> bool:
    return any(start <= hour < end for start, end in kill_zones_utc)


# ─────────────────────────────────────────────────────────────────────
# Main strategy
# ─────────────────────────────────────────────────────────────────────

class IctOrderBlockXauusd:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="ict_order_block_xauusd",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            confirmation_timeframes=("H1", "H4"),
            htf_veto=False,
            close_on_opposite_signal=True,
            params={
                # ── H1 structure / order block ──
                "swing_lookback": 5,
                "atr_period": 14,
                "max_bars_since_break": 40,
                "invalidation_buffer_atr_mult": 0.1,

                # ── Fibonacci discount/premium filter ──
                "fib_discount_max": 0.5,
                "fib_premium_min": 0.5,

                # ── H4 directional bias ──
                "h4_bias_sma_period": 20,

                # ── M5 Fair Value Gap confirmation ──
                "require_fvg_confirmation": True,
                "fvg_lookback_bars": 12,
                "fvg_zone_tolerance_atr_mult": 0.25,

                # ── ICT Kill Zones, UTC hours [start, end) ──
                # London 07:00-10:00 UTC (02:00-05:00 EST),
                # New York 12:00-15:00 UTC (07:00-10:00 EST).
                "kill_zones_utc": ((7, 10), (12, 15)),

                # ── Stop loss: beyond the Order Block + ATR-scaled buffer ──
                "sl_zone_buffer_atr_mult": 0.15,
                "sl_zone_buffer_zone_frac": 0.10,
                "sl_min_atr_mult": 0.5,

                # ── Tiered take profit — floors chosen with headroom over
                # XAUUSD's configured min_rr=1.5 (the xauusd_snd_qm_structure
                # family had to raise tp1 from 1.2 to 1.8 after discovering
                # 1.2 never cleared the spread gate; start from that lesson). ──
                "tp1_target_rr": 1.8,
                "tp2_target_rr": 2.6,
                "tp3_target_rr": 4.0,
                "point_value": 0.01,
            },
        )

    def evaluate(self, ctx: MarketContext) -> tuple[Signal, ...] | None:
        params = self.spec.params
        m5 = ctx.candles.get("M5")
        h1 = ctx.candles.get("H1")
        h4 = ctx.candles.get("H4")

        swing_lookback = int(params["swing_lookback"])
        atr_period = int(params["atr_period"])
        min_h1_len = atr_period + swing_lookback * 2 + 10
        fvg_lookback = int(params["fvg_lookback_bars"])

        if m5 is None or h1 is None or len(h1) < min_h1_len or len(m5) < fvg_lookback + 3:
            return None
        if "time" not in m5.columns or "time" not in h1.columns:
            return None

        h1_opens = h1["open"].to_numpy()
        h1_highs = h1["high"].to_numpy()
        h1_lows = h1["low"].to_numpy()
        h1_closes = h1["close"].to_numpy()

        atr_series = _atr(h1, atr_period)
        valid_atr = atr_series.dropna()
        if valid_atr.empty:
            return None
        atr_val = float(valid_atr.iloc[-1])
        if atr_val <= 0:
            return None

        swings = _detect_swing_points(h1_highs, h1_lows, swing_lookback)
        if len(swings) < 2:
            return None

        mss = _find_mss(h1_closes, swings, int(params["max_bars_since_break"]))
        if mss is None:
            return None

        bullish = mss["direction"] == "buy"
        leg_start_idx = mss["pivot_idx"]
        break_idx = mss["break_idx"]

        ob = _find_order_block(
            h1_opens, h1_highs, h1_lows, h1_closes, leg_start_idx, break_idx, bullish
        )
        ob_low, ob_high = ob["price_low"], ob["price_high"]
        if ob_high <= ob_low:
            return None

        # ── Fibonacci discount/premium filter ──
        if bullish:
            leg_low = mss["pivot_price"]
            leg_high = float(h1_highs[leg_start_idx: break_idx + 1].max())
        else:
            leg_high = mss["pivot_price"]
            leg_low = float(h1_lows[leg_start_idx: break_idx + 1].min())
        leg_range = leg_high - leg_low
        if leg_range <= 0:
            return None
        ob_mid = (ob_low + ob_high) / 2
        fib_ratio = (ob_mid - leg_low) / leg_range
        if bullish and fib_ratio > float(params["fib_discount_max"]):
            return None
        if not bullish and fib_ratio < float(params["fib_premium_min"]):
            return None

        # ── Invalidation: the Order Block must not have been broken since ──
        inval_buffer = atr_val * float(params["invalidation_buffer_atr_mult"])
        last_h1_idx = len(h1_closes) - 1
        if bullish:
            if last_h1_idx > ob["idx"] and bool(
                np.any(h1_closes[ob["idx"] + 1:] < ob_low - inval_buffer)
            ):
                return None
        else:
            if last_h1_idx > ob["idx"] and bool(
                np.any(h1_closes[ob["idx"] + 1:] > ob_high + inval_buffer)
            ):
                return None

        # ── H4 directional bias (skip the filter if there isn't enough H4 history) ──
        if h4 is not None and "close" in h4.columns:
            bias_period = int(params["h4_bias_sma_period"])
            if len(h4) >= bias_period + 1:
                h4_sma = float(h4["close"].tail(bias_period).mean())
                h4_close = float(h4["close"].iloc[-1])
                if bullish and h4_close < h4_sma:
                    return None
                if not bullish and h4_close > h4_sma:
                    return None

        # ── ICT Kill Zone time filter (M5 bar's UTC hour) ──
        last_m5_time = m5["time"].iloc[-1]
        hour = int(pd.Timestamp(last_m5_time).hour)
        if not _in_kill_zone(hour, tuple(params["kill_zones_utc"])):
            return None

        # ── M5 retest + rejection entry trigger ──
        m5_opens = m5["open"].to_numpy()
        m5_highs = m5["high"].to_numpy()
        m5_lows = m5["low"].to_numpy()
        m5_closes = m5["close"].to_numpy()
        last_m5 = len(m5_closes) - 1
        entry_open = float(m5_opens[last_m5])
        entry_high = float(m5_highs[last_m5])
        entry_low = float(m5_lows[last_m5])
        entry_close = float(m5_closes[last_m5])

        if bullish:
            touched = entry_low <= ob_high
            rejected = entry_close > ob_low and entry_close > entry_open
            if not (touched and rejected):
                return None
        else:
            touched = entry_high >= ob_low
            rejected = entry_close < ob_high and entry_close < entry_open
            if not (touched and rejected):
                return None

        # ── Fair Value Gap confirmation ──
        fvg_note = "no FVG required"
        if bool(params["require_fvg_confirmation"]):
            window_start = max(0, last_m5 - fvg_lookback)
            fvgs = _detect_fvgs(
                m5_opens, m5_highs, m5_lows, m5_closes, window_start, last_m5, bullish
            )
            tol = atr_val * float(params["fvg_zone_tolerance_atr_mult"])
            matching = [g for g in fvgs if _ranges_overlap(g[1], g[2], ob_low - tol, ob_high + tol)]
            if not matching:
                return None
            g_idx, g_low, g_high = matching[-1]
            fvg_note = f"FVG[{g_low:.2f},{g_high:.2f}]@{g_idx}"

        direction = Direction.BUY if bullish else Direction.SELL

        # ── Stop loss: beyond the Order Block + ATR-scaled buffer ──
        zone_height = ob_high - ob_low
        buffer = max(
            atr_val * float(params["sl_zone_buffer_atr_mult"]),
            zone_height * float(params["sl_zone_buffer_zone_frac"]),
        )
        sl_points = (
            entry_close - (ob_low - buffer) if bullish else (ob_high + buffer) - entry_close
        )
        sl_points = max(sl_points, atr_val * float(params["sl_min_atr_mult"]))
        if sl_points <= 0:
            return None

        spread_price = float(ctx.spread_points) * float(params["point_value"])
        risk_price = sl_points + spread_price

        fib_extension = (leg_high - entry_close) if bullish else (entry_close - leg_low)

        tp1_points = risk_price * float(params["tp1_target_rr"])
        tp2_points = max(
            tp1_points + risk_price * 0.8, risk_price * float(params["tp2_target_rr"])
        )
        if fib_extension > tp2_points:
            tp3_points = fib_extension
        else:
            tp3_points = max(
                tp2_points + risk_price * 1.2, risk_price * float(params["tp3_target_rr"])
            )

        # ── Chart annotations ──
        zone_annotation = PriceZone(
            kind=ZoneKind.DEMAND if bullish else ZoneKind.SUPPLY,
            pattern="BuOB" if bullish else "BeOB",
            price_low=ob_low,
            price_high=ob_high,
            time_start=h1["time"].iloc[ob["idx"]],
            time_end=m5["time"].iloc[last_m5],
        )
        structure = _label_structure(swings)
        structure_points = tuple(
            StructurePoint(
                time=h1["time"].iloc[min(max(int(s["index"]), 0), len(h1) - 1)],
                price=s["price"],
                label=s["label"],
            )
            for s in structure[-4:]
        )

        base_reason = (
            f"{'BuOB' if bullish else 'BeOB'} [{ob_low:.2f},{ob_high:.2f}] "
            f"fib={fib_ratio:.2f} sl={sl_points:.2f} kz_hour={hour} {fvg_note}"
        )

        sig_tp1 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp1_points),
            confidence=0.75,
            reason=f"{base_reason} tp1={tp1_points:.2f} (Scalp)",
            zone=zone_annotation,
            pattern=zone_annotation.pattern,
            structure=structure_points,
        )
        sig_tp2 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp2_points),
            confidence=0.70,
            reason=f"{base_reason} tp2={tp2_points:.2f} (Zone)",
            zone=zone_annotation,
            pattern=zone_annotation.pattern,
            structure=structure_points,
        )
        sig_tp3 = Signal(
            direction=direction,
            sl_points=float(sl_points),
            tp_points=float(tp3_points),
            confidence=0.65,
            reason=f"{base_reason} tp3={tp3_points:.2f} (Runner)",
            zone=zone_annotation,
            pattern=zone_annotation.pattern,
            structure=structure_points,
        )
        return (sig_tp1, sig_tp2, sig_tp3)
