"""Scalping strategy: Bollinger Band (20, 2) mean reversion on M5 — fades a
touch of the outer band back toward the middle band, but only while band
width is contracted relative to its own recent average (a proxy for "range,
not trending" — bands widen sharply once price starts trending, and fading
the outer band during a genuine trend is the classic way this setup loses,
per the French scalping notes this batch was built from).

Deliberately carries NO `confirmation_timeframes`, same reasoning as
`scalp_vwap_reversion_v1`: this is a reversion strategy, and
`engine/application/mtf_confirm.py`'s automatic EMA-trend veto on a declared
confirmation timeframe would oppose most of its setups by construction. The
band-width range filter is this strategy's own trend/range gate instead
(mirrors how `mean_reversion_v1`, the existing swing-fade baseline, also
carries no confirmation timeframes).

Sandbox-safe: only `numpy`/`pandas`.
"""

import numpy as np

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

PERIOD = 20
STD_MULT = 2.0
WIDTH_LOOKBACK = 20  # bars averaged to judge whether current band width is contracted
WIDTH_RANGE_THRESHOLD = 1.1  # current width must be <= this * its own recent average
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) — enforced explicitly below via POINT_VALUES + ctx.spread_points
# (tp_points = (sl_distance + spread) * TP_RR), the same formula SpreadGate
# applies at the broker gate.
TP_RR = 2.2
MIN_HISTORY = PERIOD + WIDTH_LOOKBACK + 3
# Point size per traded symbol (configs/symbols/*.yaml) — converts
# ctx.spread_points (raw broker points) into a price distance.
POINT_VALUES = {
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
    "BTCUSD": 0.01,
    "Boom 1000 Index": 0.0001,
    "Volatility 75 Index": 0.01,
}


class ScalpBollingerReversionV1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="scalp_bollinger_reversion_v1",
            version=1,
            symbols=("XAUUSD", "XAGUSD", "BTCUSD", "Boom 1000 Index", "Volatility 75 Index"),
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params={
                "period": PERIOD,
                "std_mult": STD_MULT,
                "width_lookback": WIDTH_LOOKBACK,
                "width_range_threshold": WIDTH_RANGE_THRESHOLD,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        period = int(self.spec.params["period"])
        std_mult = self.spec.params["std_mult"]
        width_lookback = int(self.spec.params["width_lookback"])
        width_range_threshold = self.spec.params["width_range_threshold"]
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
        if not np.isfinite(prev_upper) or not np.isfinite(prev_lower):
            return None

        prev_high = float(m5["high"].iloc[-2])
        prev_low = float(m5["low"].iloc[-2])
        last_close = float(closes.iloc[-1])
        last_upper = float(upper.iloc[-1])
        last_lower = float(lower.iloc[-1])
        spread_distance = float(ctx.spread_points) * POINT_VALUES.get(ctx.symbol, 0.01)

        # Prior bar touched/pierced the upper band, this bar's close has
        # already pulled back inside it -> reversal toward the mean, sell.
        if prev_high >= prev_upper and last_close < last_upper:
            sl_distance = prev_high - last_close
            if sl_distance <= 0:
                return None
            return Signal(
                direction=Direction.SELL,
                sl_points=sl_distance,
                tp_points=(sl_distance + spread_distance) * tp_rr,
                confidence=0.55,
                reason=(
                    f"M5 touched upper band {prev_upper:.5f} (high {prev_high:.5f}) and closed "
                    f"back inside at {last_close:.5f}; width {last_width:.4f} <= "
                    f"{width_range_threshold}x avg {last_avg_width:.4f} (range-bound)"
                ),
            )
        # Prior bar touched/pierced the lower band, this bar's close has
        # already pulled back inside it -> reversal toward the mean, buy.
        if prev_low <= prev_lower and last_close > last_lower:
            sl_distance = last_close - prev_low
            if sl_distance <= 0:
                return None
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_distance,
                tp_points=(sl_distance + spread_distance) * tp_rr,
                confidence=0.55,
                reason=(
                    f"M5 touched lower band {prev_lower:.5f} (low {prev_low:.5f}) and closed "
                    f"back inside at {last_close:.5f}; width {last_width:.4f} <= "
                    f"{width_range_threshold}x avg {last_avg_width:.4f} (range-bound)"
                ),
            )
        return None
