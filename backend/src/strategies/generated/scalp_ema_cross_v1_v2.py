"""Scalping strategy: EMA(5)/EMA(13) crossover on M5, gated by the engine's
automatic H1 trend confirmation and M1 momentum alignment (see
`confirmation_timeframes` — `engine/application/mtf_confirm.py` vetoes any
signal whose direction opposes either timeframe's EMA(20/50) trend).

v2: revised after the v1 backtest (XAUUSD 2026-05:2026-07, 383 trades, 26%
win rate, PF 0.978 — essentially breakeven but slightly negative, needs ~31%
at TP_RR=2.2) showed the raw cross alone doesn't filter enough marginal/
whipsaw crosses. Two new filters: (1) `min_vol_ratio` requires ATR(14) as a
fraction of price to clear a floor, skipping crosses that fire during dead,
choppy conditions where EMA5/13 crosses constantly without going anywhere;
(2) a price-confirmation check requires the close to already be on the
correct side of EMA_SLOW at the moment of the cross (not just the EMA lines
themselves crossing) — filters crosses where price hasn't actually
confirmed the new direction yet. `atr_mult` widened 1.2 -> 1.4 for a bit
more room. Sandbox-safe: only `numpy`/`pandas`.
"""

import numpy as np

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

EMA_FAST = 5
EMA_SLOW = 13
ATR_PERIOD = 14
ATR_MULT = 1.4  # v2: widened from 1.2
MIN_VOL_RATIO = 0.0006  # v2 new: ATR(14)/close must clear this to trade (skips dead chop)
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) with headroom — see breakout_v1.py for the same constraint.
TP_RR = 2.2
MIN_HISTORY = EMA_SLOW * 2 + ATR_PERIOD + 2  # a couple of EMA_SLOW spans to settle + ATR warmup


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float | None:
    if len(highs) < period + 1:
        return None
    tr = highs[1:] - lows[1:]
    gap_high = np.abs(highs[1:] - closes[:-1])
    gap_low = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr, np.maximum(gap_high, gap_low))
    return float(np.mean(tr[-period:]))


class ScalpEmaCrossV2:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="scalp_ema_cross_v1",
            version=2,
            symbols=("XAUUSD", "XAGUSD", "BTCUSD", "Boom 1000 Index", "Volatility 75 Index"),
            entry_timeframe="M5",
            confirmation_timeframes=("M1", "H1"),
            params={
                "ema_fast": EMA_FAST,
                "ema_slow": EMA_SLOW,
                "atr_period": ATR_PERIOD,
                "atr_mult": ATR_MULT,
                "min_vol_ratio": MIN_VOL_RATIO,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        fast = int(self.spec.params["ema_fast"])
        slow = int(self.spec.params["ema_slow"])
        atr_period = int(self.spec.params["atr_period"])
        atr_mult = self.spec.params["atr_mult"]
        min_vol_ratio = self.spec.params["min_vol_ratio"]
        tp_rr = self.spec.params["tp_rr"]
        if m5 is None or len(m5) < MIN_HISTORY:
            return None

        closes = m5["close"]
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        prev_fast, prev_slow = float(ema_fast.iloc[-2]), float(ema_slow.iloc[-2])
        last_fast, last_slow = float(ema_fast.iloc[-1]), float(ema_slow.iloc[-1])

        crossed_up = prev_fast <= prev_slow and last_fast > last_slow
        crossed_down = prev_fast >= prev_slow and last_fast < last_slow
        if not crossed_up and not crossed_down:
            return None

        last_close = float(closes.iloc[-1])
        if crossed_up and last_close <= last_slow:
            return None  # EMAs crossed but price hasn't confirmed above EMA_SLOW yet
        if crossed_down and last_close >= last_slow:
            return None

        atr = _atr(
            m5["high"].to_numpy(), m5["low"].to_numpy(), m5["close"].to_numpy(), atr_period
        )
        if atr is None or atr <= 0 or last_close <= 0:
            return None
        if (atr / last_close) < min_vol_ratio:
            return None  # too quiet/choppy — skip low-conviction crosses
        sl_distance = atr * atr_mult

        if crossed_up:
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_distance,
                tp_points=sl_distance * tp_rr,
                confidence=0.55,
                reason=(
                    f"EMA{fast} crossed above EMA{slow} on M5 "
                    f"({last_fast:.5f} > {last_slow:.5f}), price confirmed at "
                    f"{last_close:.5f}; ATR({atr_period})={atr:.5f} "
                    f"({atr / last_close:.5f} of price)"
                ),
            )
        return Signal(
            direction=Direction.SELL,
            sl_points=sl_distance,
            tp_points=sl_distance * tp_rr,
            confidence=0.55,
            reason=(
                f"EMA{fast} crossed below EMA{slow} on M5 "
                f"({last_fast:.5f} < {last_slow:.5f}), price confirmed at "
                f"{last_close:.5f}; ATR({atr_period})={atr:.5f} "
                f"({atr / last_close:.5f} of price)"
            ),
        )
