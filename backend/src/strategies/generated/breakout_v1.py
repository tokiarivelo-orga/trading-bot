"""Baseline hand-written strategy (Phase 4): M5 range breakout.

Proves the engine pipe end-to-end before the AI codegen loop (Phase 6)
exists. Sandbox-safe: only `math`/`pandas` — no I/O, no broker access.
No `from __future__ import annotations` — the sandbox's import whitelist
rejects it, and every generated strategy must pass that whitelist.
"""

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec

LOOKBACK = 20
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) with headroom — the broker's spread-adjusted gate requires
# tp_distance >= min_rr * (sl_distance + spread), so a strategy tp_rr equal to
# min_rr would fail that check on every single trade.
TP_RR = 2.2


class BreakoutV1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="breakout_v1",
            version=1,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=("H1", "H4"),
            params={"lookback": LOOKBACK, "tp_rr": TP_RR},
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        lookback = self.spec.params["lookback"]
        tp_rr = self.spec.params["tp_rr"]
        if m5 is None or len(m5) < lookback + 1:
            return None

        # Exclude the still-forming last bar; the prior `lookback` bars set
        # the range this bar must break out of.
        window = m5.iloc[-(lookback + 1) : -1]
        last_close = m5.iloc[-1]["close"]
        highest_high = window["high"].max()
        lowest_low = window["low"].min()

        if last_close > highest_high:
            sl_distance = last_close - lowest_low
            if sl_distance <= 0:
                return None
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_distance,
                tp_points=sl_distance * tp_rr,
                confidence=0.6,
                reason=f"M5 close {last_close:.5f} broke {lookback}-bar high {highest_high:.5f}",
            )
        if last_close < lowest_low:
            sl_distance = highest_high - last_close
            if sl_distance <= 0:
                return None
            return Signal(
                direction=Direction.SELL,
                sl_points=sl_distance,
                tp_points=sl_distance * tp_rr,
                confidence=0.6,
                reason=f"M5 close {last_close:.5f} broke {lookback}-bar low {lowest_low:.5f}",
            )
        return None
