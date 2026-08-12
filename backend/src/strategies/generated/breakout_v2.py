"""Breakout v2: M5 range breakout, filtered by higher-TF trend and sized by
volatility instead of raw range width.

Refines `breakout_v1`, which takes every M5 close beyond the prior 20-bar
high/low with no regard for the higher-TF context, and sizes its stop as
the full width of the range just broken — so a wide consolidation produces
an oversized SL/TP pair even though that width says nothing about current
volatility. v2 adds two changes on top of the same core breakout signal:

1. Higher-TF trend filter — a BUY only fires when both H1 and H4 closes sit
   above their own `TREND_SMA_PERIOD`-bar SMA (an uptrend on both
   confirmation timeframes already declared in the spec but never read by
   v1); SELL is the mirror image below both SMAs. Cuts breakouts taken
   against the dominant trend.
2. ATR-sized stop — SL is `SL_ATR_MULT` x ATR(14) on M5, not the breakout
   range's width, so stop distance tracks recent volatility instead of
   whatever the consolidation happened to be wide enough to be.

TP:SL is left at the same 2.2 used by v1 so a before/after backtest isolates
these two filters' effect rather than also moving the reward ratio. Sandbox-
safe: only `numpy`/`pandas` — no I/O, no broker access, no external
indicator libraries (ATR here is derived from the same OHLC series, not a
signal input on its own).
"""

import numpy as np
import pandas as pd

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

LOOKBACK = 20  # M5 bars defining the range to break out of
TREND_SMA_PERIOD = 50  # H1/H4 bars averaged for the higher-TF trend filter
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) with headroom — see breakout_v1.py for the same constraint.
TP_RR = 2.2


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    tr = highs - lows
    if len(tr) > 1:
        gap_high = np.abs(highs[1:] - closes[:-1])
        gap_low = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(tr[1:], np.maximum(gap_high, gap_low))
    return tr


def _atr(m5: pd.DataFrame, period: int) -> float | None:
    highs = m5["high"].to_numpy()
    lows = m5["low"].to_numpy()
    closes = m5["close"].to_numpy()
    tr = pd.Series(_true_range(highs, lows, closes))
    atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
    return None if pd.isna(atr) or atr <= 0 else float(atr)


def _htf_trend(ctx: MarketContext, timeframe: str, sma_period: int) -> int | None:
    """+1 for an uptrend (last close above the SMA), -1 for a downtrend,
    None if the timeframe isn't available or has too little history to form
    the SMA yet."""
    candles = ctx.candles.get(timeframe)
    if candles is None or len(candles) < sma_period:
        return None
    closes = candles["close"]
    sma = closes.iloc[-sma_period:].mean()
    last_close = closes.iloc[-1]
    if last_close > sma:
        return 1
    if last_close < sma:
        return -1
    return None


class BreakoutV2:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="breakout_v2",
            version=2,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=("H1", "H4"),
            params={
                "lookback": LOOKBACK,
                "trend_sma_period": TREND_SMA_PERIOD,
                "atr_period": ATR_PERIOD,
                "sl_atr_mult": SL_ATR_MULT,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        lookback = self.spec.params["lookback"]
        atr_period = self.spec.params["atr_period"]
        sl_atr_mult = self.spec.params["sl_atr_mult"]
        tp_rr = self.spec.params["tp_rr"]
        trend_sma_period = self.spec.params["trend_sma_period"]
        if m5 is None or len(m5) < max(lookback + 1, atr_period + 1):
            return None

        # Exclude the still-forming last bar; the prior `lookback` bars set
        # the range this bar must break out of.
        window = m5.iloc[-(lookback + 1) : -1]
        last_close = m5.iloc[-1]["close"]
        highest_high = window["high"].max()
        lowest_low = window["low"].min()

        if last_close > highest_high:
            direction = Direction.BUY
            reason = f"M5 close {last_close:.5f} broke {lookback}-bar high {highest_high:.5f}"
        elif last_close < lowest_low:
            direction = Direction.SELL
            reason = f"M5 close {last_close:.5f} broke {lookback}-bar low {lowest_low:.5f}"
        else:
            return None

        h1_trend = _htf_trend(ctx, "H1", trend_sma_period)
        h4_trend = _htf_trend(ctx, "H4", trend_sma_period)
        if h1_trend is None or h4_trend is None:
            return None
        wanted_trend = 1 if direction is Direction.BUY else -1
        if h1_trend != wanted_trend or h4_trend != wanted_trend:
            return None  # breakout runs against the H1/H4 trend — skip it

        atr_val = _atr(m5, atr_period)
        if atr_val is None:
            return None
        sl_points = atr_val * sl_atr_mult

        return Signal(
            direction=direction,
            sl_points=sl_points,
            tp_points=sl_points * tp_rr,
            confidence=0.6,
            reason=f"{reason}; aligned with H1/H4 trend; SL {sl_atr_mult}xATR({atr_period})",
        )
