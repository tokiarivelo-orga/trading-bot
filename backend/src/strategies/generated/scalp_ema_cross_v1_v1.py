"""Scalping strategy: EMA(5)/EMA(13) crossover on M5, gated by the engine's
automatic H1 trend confirmation and M1 momentum alignment (see
`confirmation_timeframes` — `engine/application/mtf_confirm.py` vetoes any
signal whose direction opposes either timeframe's EMA(20/50) trend). Classic
scalp playbook: a fast/slow EMA cross filtered by a higher timeframe trend so
only crosses that agree with the prevailing direction get taken.
Sandbox-safe: only `numpy`/`pandas`.
"""

import numpy as np

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

EMA_FAST = 5
EMA_SLOW = 13
ATR_PERIOD = 14
ATR_MULT = 1.2
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) — enforced explicitly below via POINT_VALUES + ctx.spread_points
# (tp_points = (sl_distance + spread) * TP_RR), the same formula SpreadGate
# applies at the broker gate.
TP_RR = 2.2
MIN_HISTORY = EMA_SLOW * 2 + ATR_PERIOD + 2  # a couple of EMA_SLOW spans to settle + ATR warmup
# Point size per traded symbol (configs/symbols/*.yaml) — converts
# ctx.spread_points (raw broker points) into a price distance.
POINT_VALUES = {
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
}


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float | None:
    if len(highs) < period + 1:
        return None
    tr = highs[1:] - lows[1:]
    gap_high = np.abs(highs[1:] - closes[:-1])
    gap_low = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr, np.maximum(gap_high, gap_low))
    return float(np.mean(tr[-period:]))


class ScalpEmaCrossV1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="scalp_ema_cross_v1",
            version=1,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=("M1", "H1"),
            params={
                "ema_fast": EMA_FAST,
                "ema_slow": EMA_SLOW,
                "atr_period": ATR_PERIOD,
                "atr_mult": ATR_MULT,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        fast = int(self.spec.params["ema_fast"])
        slow = int(self.spec.params["ema_slow"])
        atr_period = int(self.spec.params["atr_period"])
        atr_mult = self.spec.params["atr_mult"]
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

        atr = _atr(
            m5["high"].to_numpy(), m5["low"].to_numpy(), m5["close"].to_numpy(), atr_period
        )
        if atr is None or atr <= 0:
            return None
        sl_distance = atr * atr_mult
        spread_distance = float(ctx.spread_points) * POINT_VALUES.get(ctx.symbol, 0.01)

        if crossed_up:
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_distance,
                tp_points=(sl_distance + spread_distance) * tp_rr,
                confidence=0.55,
                reason=(
                    f"EMA{fast} crossed above EMA{slow} on M5 "
                    f"({last_fast:.5f} > {last_slow:.5f}); ATR({atr_period})={atr:.5f}"
                ),
            )
        return Signal(
            direction=Direction.SELL,
            sl_points=sl_distance,
            tp_points=(sl_distance + spread_distance) * tp_rr,
            confidence=0.55,
            reason=(
                f"EMA{fast} crossed below EMA{slow} on M5 "
                f"({last_fast:.5f} < {last_slow:.5f}); ATR({atr_period})={atr:.5f}"
            ),
        )
