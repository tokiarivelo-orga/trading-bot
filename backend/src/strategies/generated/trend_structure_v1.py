"""Pure price-structure trend-following strategy: trades HH/LL continuation.

No indicators — only swing-pivot (fractal) detection on the M5 candle series.
Enters long the bar a confirmed swing high prints above the prior confirmed
swing high (HH), short the bar a confirmed swing low prints below the prior
confirmed swing low (LL). Higher lows / lower highs (HL/LH) are structure but
not entries here — this strategy only trades trend continuation, not the
reversal legs. Sandbox-safe: only `numpy` — no I/O, no broker access.

Perf note: swing detection runs on numpy arrays with vectorized windowed
max/min instead of per-bar pandas `.iloc` scans — identical pivots, but a
backtest calls evaluate() on every bar and the old scan dominated its
runtime.
"""

import numpy as np

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    Signal,
    StrategySpec,
    StructureLabel,
    StructurePoint,
)

PIVOT_WING = 3  # bars required on each side of a candidate to confirm a swing
MIN_HISTORY = 30  # enough bars for at least 3 alternating swing pivots to form
# Must clear every symbol's configs/symbols/<sym>.yaml min_rr (highest: XAGUSD
# at 1.8) with headroom — see breakout_v1.py for the same constraint.
TP_RR = 2.2


def _swing_flags(highs: np.ndarray, lows: np.ndarray, wing: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractal swing highs/lows: a bar whose high (low) is the max (min) of
    the `wing`-bar window on each side. Windowed max/min are computed in one
    vector pass; equality against the center bar matches the old per-bar
    scan exactly."""
    n = len(highs)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    window = 2 * wing + 1
    if n >= window:
        window_max = np.lib.stride_tricks.sliding_window_view(highs, window).max(axis=1)
        window_min = np.lib.stride_tricks.sliding_window_view(lows, window).min(axis=1)
        is_high[wing : n - wing] = highs[wing : n - wing] == window_max
        is_low[wing : n - wing] = lows[wing : n - wing] == window_min
    return is_high, is_low


def _push_swing(swings: list[tuple[int, float, str]], index: int, price: float, kind: str) -> None:
    # Collapses runs of same-kind pivots (e.g. two fractal highs in a row
    # before the next low prints) down to the single most extreme one, so the
    # resulting sequence strictly alternates high/low/high/low — a zigzag.
    if swings and swings[-1][2] == kind:
        _, prev_price, _ = swings[-1]
        if (kind == "high" and price > prev_price) or (kind == "low" and price < prev_price):
            swings[-1] = (index, price, kind)
        return
    swings.append((index, price, kind))


def _zigzag_swings(highs: np.ndarray, lows: np.ndarray, wing: int) -> list[tuple[int, float, str]]:
    is_high, is_low = _swing_flags(highs, lows, wing)
    swings: list[tuple[int, float, str]] = []
    # Only flagged bars can push a swing; a bar flagged as both pushes the
    # high first then the low, matching the old full-range scan.
    for i in np.flatnonzero(is_high | is_low):
        index = int(i)
        if is_high[index]:
            _push_swing(swings, index, float(highs[index]), "high")
        if is_low[index]:
            _push_swing(swings, index, float(lows[index]), "low")
    return swings


class TrendStructureV1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="trend_structure_v1",
            version=1,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params={"pivot_wing": PIVOT_WING, "tp_rr": TP_RR},
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        wing = int(self.spec.params["pivot_wing"])
        tp_rr = self.spec.params["tp_rr"]
        if m5 is None or len(m5) < MIN_HISTORY:
            return None

        highs = m5["high"].to_numpy()
        lows = m5["low"].to_numpy()

        swings = _zigzag_swings(highs, lows, wing)
        if len(swings) < 3:
            return None

        # A fractal at index i only confirms once `wing` bars have closed to
        # its right. Only act on the bar that confirmation lands on, so the
        # same pivot doesn't re-fire a signal on every later bar.
        last_index, last_price, last_kind = swings[-1]
        if last_index != len(m5) - 1 - wing:
            return None

        prior_index, prior_price, prior_kind = swings[-3]
        if prior_kind != last_kind:
            return None
        _, sl_reference, _ = swings[-2]  # opposite-kind pivot between them: structure invalidation

        if last_kind == "high" and last_price > prior_price:
            direction, label = Direction.BUY, StructureLabel.HH
        elif last_kind == "low" and last_price < prior_price:
            direction, label = Direction.SELL, StructureLabel.LL
        else:
            return None

        entry_price = float(m5["close"].iloc[-1])
        sl_points = abs(entry_price - sl_reference)
        if sl_points <= 0:
            return None
        tp_points = sl_points * tp_rr

        structure: tuple[StructurePoint, ...] = ()
        if "time" in m5.columns:
            structure = (
                StructurePoint(time=m5["time"].iloc[last_index], price=last_price, label=label),
            )

        return Signal(
            direction=direction,
            sl_points=sl_points,
            tp_points=tp_points,
            confidence=0.55,
            reason=(
                f"{label.value} at {last_price:.5f} (bar {last_index}) beat prior swing "
                f"{prior_price:.5f} (bar {prior_index}); SL anchored to swing {sl_reference:.5f}"
            ),
            structure=structure,
        )
