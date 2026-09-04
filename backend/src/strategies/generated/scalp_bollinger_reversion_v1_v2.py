"""Scalping strategy: Bollinger Band (20, 2) mean reversion on M5 — fades a
touch of the outer band back toward the middle band, only while band width
is contracted relative to its own recent average.

Deliberately carries NO `confirmation_timeframes`, same reasoning as
`scalp_vwap_reversion_v1` — see that file's docstring.

v2: revised after the v1 backtest (XAUUSD 2026-05:2026-07, only 25 trades
but an 8% win rate and PF 0.20 — catastrophic, far below the ~31% breakeven
at TP_RR=2.2) showed the setup was mostly catching the *start* of breakouts,
not genuine reversions. Three changes: (1) `width_range_threshold` 1.1 ->
1.05 — tightened, but not all the way to "at or below average": a touch of
the band is, by construction, a local-volatility event, so the rolling
std/width computed *at the touch itself* is almost always somewhat elevated
versus its own trailing average — testing confirmed a genuine touch+
immediate-reversal setup can't clear a threshold much under ~1.04, so 0.95
(the first value tried here) would have silently rejected nearly every real
setup rather than fixing anything; (2) a new `reversal_fraction` requires
the close to have pulled back at least 25% of the way from the touch
extreme toward the mean, not just barely inside the band; (3) a new
ATR-based minimum stop floor (`min_sl_atr_mult`) rejects setups whose SL
would be smaller than half an ATR — too tight to survive ordinary
noise/spread. Sandbox-safe: only `numpy`/`pandas`.
"""

import numpy as np

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

PERIOD = 20
STD_MULT = 2.0
WIDTH_LOOKBACK = 20  # bars averaged to judge whether current band width is contracted
WIDTH_RANGE_THRESHOLD = 1.05  # v2: tightened from 1.1 (0.95 tested as unachievable — see docstring)
REVERSAL_FRACTION = 0.25  # v2 new: close must pull back at least this far toward the mean
ATR_PERIOD = 14  # v2 new
MIN_SL_ATR_MULT = 0.5  # v2 new: reject stops tighter than this many ATRs
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) with headroom — see breakout_v1.py for the same constraint.
TP_RR = 2.2
MIN_HISTORY = PERIOD + WIDTH_LOOKBACK + 3


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float | None:
    if len(highs) < period + 1:
        return None
    tr = highs[1:] - lows[1:]
    gap_high = np.abs(highs[1:] - closes[:-1])
    gap_low = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr, np.maximum(gap_high, gap_low))
    return float(np.mean(tr[-period:]))


class ScalpBollingerReversionV2:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="scalp_bollinger_reversion_v1",
            version=2,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params={
                "period": PERIOD,
                "std_mult": STD_MULT,
                "width_lookback": WIDTH_LOOKBACK,
                "width_range_threshold": WIDTH_RANGE_THRESHOLD,
                "reversal_fraction": REVERSAL_FRACTION,
                "atr_period": ATR_PERIOD,
                "min_sl_atr_mult": MIN_SL_ATR_MULT,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        period = int(self.spec.params["period"])
        std_mult = self.spec.params["std_mult"]
        width_lookback = int(self.spec.params["width_lookback"])
        width_range_threshold = self.spec.params["width_range_threshold"]
        reversal_fraction = self.spec.params["reversal_fraction"]
        atr_period = int(self.spec.params["atr_period"])
        min_sl_atr_mult = self.spec.params["min_sl_atr_mult"]
        tp_rr = self.spec.params["tp_rr"]
        if m5 is None or len(m5) < MIN_HISTORY:
            return None

        closes = m5["close"]
        sma = closes.rolling(period).mean()
        std = closes.rolling(period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        width = (upper - lower) / sma
        avg_width = width.rolling(width_lookback).mean()

        last_width = float(width.iloc[-1])
        last_avg_width = float(avg_width.iloc[-1])
        if not np.isfinite(last_width) or not np.isfinite(last_avg_width) or last_avg_width <= 0:
            return None
        if last_width > width_range_threshold * last_avg_width:
            return None  # bands are expanding — likely trending, skip mean reversion

        prev_upper, prev_lower = upper.iloc[-2], lower.iloc[-2]
        last_sma = float(sma.iloc[-1])
        if not np.isfinite(prev_upper) or not np.isfinite(prev_lower) or not np.isfinite(last_sma):
            return None

        atr = _atr(m5["high"].to_numpy(), m5["low"].to_numpy(), m5["close"].to_numpy(), atr_period)
        if atr is None or atr <= 0:
            return None

        prev_high = float(m5["high"].iloc[-2])
        prev_low = float(m5["low"].iloc[-2])
        last_close = float(closes.iloc[-1])
        last_upper = float(upper.iloc[-1])
        last_lower = float(lower.iloc[-1])

        # Prior bar touched/pierced the upper band; this bar's close must
        # have pulled back at least `reversal_fraction` of the way from that
        # extreme toward the mean (not just barely inside the band).
        sell_reversal_line = prev_high - reversal_fraction * (prev_high - last_sma)
        if prev_high >= prev_upper and last_close < last_upper and last_close <= sell_reversal_line:
            sl_distance = prev_high - last_close
            if sl_distance < min_sl_atr_mult * atr:
                return None
            return Signal(
                direction=Direction.SELL,
                sl_points=sl_distance,
                tp_points=sl_distance * tp_rr,
                confidence=0.55,
                reason=(
                    f"M5 touched upper band {prev_upper:.5f} (high {prev_high:.5f}), closed "
                    f"at {last_close:.5f} (>= {reversal_fraction:.0%} pullback toward mean "
                    f"{last_sma:.5f}); width {last_width:.4f} <= "
                    f"{width_range_threshold}x avg {last_avg_width:.4f}; SL "
                    f"{sl_distance:.5f} >= {min_sl_atr_mult}x ATR({atr_period})={atr:.5f}"
                ),
            )
        buy_reversal_line = prev_low + reversal_fraction * (last_sma - prev_low)
        if prev_low <= prev_lower and last_close > last_lower and last_close >= buy_reversal_line:
            sl_distance = last_close - prev_low
            if sl_distance < min_sl_atr_mult * atr:
                return None
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_distance,
                tp_points=sl_distance * tp_rr,
                confidence=0.55,
                reason=(
                    f"M5 touched lower band {prev_lower:.5f} (low {prev_low:.5f}), closed "
                    f"at {last_close:.5f} (>= {reversal_fraction:.0%} pullback toward mean "
                    f"{last_sma:.5f}); width {last_width:.4f} <= "
                    f"{width_range_threshold}x avg {last_avg_width:.4f}; SL "
                    f"{sl_distance:.5f} >= {min_sl_atr_mult}x ATR({atr_period})={atr:.5f}"
                ),
            )
        return None
