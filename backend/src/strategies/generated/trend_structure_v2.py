"""Trend-structure v2: HH/LL continuation, filtered for quality.

Refines `trend_structure_v1` after a 33-run backtest matrix (Sep 2025-Jul
2026, XAUUSD/XAGUSD) showed a 21.6% blended win rate against v1's
2.2:1 TP:SL — well under the 31.2% breakeven win rate that ratio requires.
The loss cluster was concentrated in choppy stretches where a fractal beat
the prior swing by a noise-level amount, or where price poked a fraction
above/below a prior extreme mid-range rather than during an established
trend. v2 adds two price-structure-only filters (no external indicators —
ATR here is a volatility measure derived from the same OHLC series, used
only to size "is this move bigger than noise", not as a signal input):

1. Structural alignment — a HH is only traded if the swing low right before
   it is itself a higher low (confirms both legs of the zigzag agree on
   direction); symmetric LL/lower-high check for shorts. Filters "one new
   high inside an otherwise ranging/choppy market."
2. Minimum swing amplitude — the new extreme must beat the prior same-kind
   swing by at least `MIN_SWING_ATR_MULT` x ATR(14), so a 1-tick technical
   HH/LL doesn't qualify as trend continuation.

Both filters cut trade frequency; that's the intended trade — fewer, higher
quality entries over chasing volume. TP:SL is left at the same 2.2 used by
v1 so a before/after backtest isolates the filters' effect. Sandbox-safe:
only `numpy`/`pandas` — no I/O, no broker access.

Perf note: swing detection and ATR run on numpy arrays with vectorized
windowed max/min instead of per-bar pandas `.iloc` scans — identical pivots
and values, but a backtest calls evaluate() on every bar and the old scan
dominated its runtime.
"""

import numpy as np
import pandas as pd

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    Signal,
    StrategySpec,
    StructureLabel,
    StructurePoint,
)

PIVOT_WING = 3  # bars required on each side of a candidate to confirm a swing
ATR_PERIOD = 14
MIN_SWING_ATR_MULT = 0.5  # new swing must beat the prior same-kind swing by this much ATR
MIN_HISTORY = 60  # room for ATR(14) warmup plus at least 5 alternating swing pivots
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


def _true_range_values(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    tr = highs - lows
    if len(tr) > 1:
        # Bar 0 has no previous close; its TR stays high-low, matching the
        # old concat().max(axis=1) which skipped the NaN gap columns there.
        gap_high = np.abs(highs[1:] - closes[:-1])
        gap_low = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(tr[1:], np.maximum(gap_high, gap_low))
    return tr


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> pd.Series:
    # Rolling mean stays in pandas (not a cumsum shortcut) so ATR values are
    # bit-identical to the previous implementation.
    tr = pd.Series(_true_range_values(highs, lows, closes))
    return tr.rolling(period, min_periods=period).mean()


class TrendStructureV2:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="trend_structure_v2",
            version=2,
            symbols=(),  # empty = accepts all dynamically configured symbols
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params={
                "pivot_wing": PIVOT_WING,
                "atr_period": ATR_PERIOD,
                "min_swing_atr_mult": MIN_SWING_ATR_MULT,
                "tp_rr": TP_RR,
            },
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        m5 = ctx.candles.get("M5")
        wing = int(self.spec.params["pivot_wing"])
        atr_period = int(self.spec.params["atr_period"])
        min_swing_atr_mult = self.spec.params["min_swing_atr_mult"]
        tp_rr = self.spec.params["tp_rr"]
        if m5 is None or len(m5) < MIN_HISTORY:
            return None

        highs = m5["high"].to_numpy()
        lows = m5["low"].to_numpy()

        swings = _zigzag_swings(highs, lows, wing)
        if len(swings) < 5:
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
        _, context_reference, context_kind = swings[-4]  # same kind as swings[-2]: alignment check
        if context_kind != swings[-2][2]:
            return None

        if last_kind == "high" and last_price > prior_price:
            # Only a HH inside an established uptrend: the low right before
            # it (swings[-2]) must itself be a higher low than the one
            # before that (swings[-4]) — both legs of the zigzag agree.
            if sl_reference <= context_reference:
                return None
            direction, label = Direction.BUY, StructureLabel.HH
        elif last_kind == "low" and last_price < prior_price:
            # Symmetric: the high right before this LL must be a lower high.
            if sl_reference >= context_reference:
                return None
            direction, label = Direction.SELL, StructureLabel.LL
        else:
            return None

        closes = m5["close"].to_numpy()
        atr = _atr(highs, lows, closes, atr_period)
        atr_val = atr.iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return None
        if abs(last_price - prior_price) < atr_val * min_swing_atr_mult:
            return None  # new extreme barely beat the prior one — noise, not continuation

        entry_price = float(closes[-1])
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
            confidence=0.6,
            reason=(
                f"{label.value} at {last_price:.5f} (bar {last_index}) beat prior swing "
                f"{prior_price:.5f} (bar {prior_index}) by >= {min_swing_atr_mult}xATR, "
                f"aligned with prior {'HL' if label is StructureLabel.HH else 'LH'}; "
                f"SL anchored to swing {sl_reference:.5f}"
            ),
            structure=structure,
        )
