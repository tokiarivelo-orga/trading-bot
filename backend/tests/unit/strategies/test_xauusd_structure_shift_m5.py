"""Unit tests for `xauusd_structure_shift_m5_v1.py` — the M5 sibling of
`xauusd_structure_shift_m1_v1.py`. Kept short: the logic is identical (see
`test_xauusd_structure_shift_m1.py` for the full detector/gate/anti-spam
coverage), so this file only confirms the spec differs correctly by
timeframe and that the same synthetic setup still fires end-to-end on this
file specifically — cheap insurance against a copy/paste slip between the
two files, not a re-test of shared logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
    MarketContext,
    PositionSnapshot,
)
from src.strategies.generated.xauusd_structure_shift_m5_v1 import XauusdStructureShiftM5
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_structure_shift_m5_v1.py"
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def test_sandbox_accepts_source():
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


def test_spec_matches_m5_convention():
    strategy = XauusdStructureShiftM5()
    assert strategy.spec.name == "xauusd_structure_shift_m5"
    assert strategy.spec.symbols == ("XAUUSD",)
    assert strategy.spec.entry_timeframe == "M5"
    assert strategy.spec.confirmation_timeframes == ("H1",)


def _build_reversal_frame(
    *,
    n_warmup_legs: int = 8,
    leg: float = 3.0,
    pull: float = 1.0,
    weak_leg: float = 1.0,
    bars_per_leg: int = 4,
    break_bars: int = 5,
    break_step: float = 1.0,
    offset: float = 0.05,
    base_volume: float = 1000.0,
    weak_volume: float = 300.0,
    up: bool = True,
) -> pd.DataFrame:
    sign = 1.0 if up else -1.0
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    price = 100.0

    def _emit(step: float, vol: float) -> None:
        nonlocal price
        price += sign * step
        highs.append(price + offset)
        lows.append(price - offset)
        volumes.append(vol)

    for _ in range(n_warmup_legs):
        for _ in range(bars_per_leg):
            _emit(leg / bars_per_leg, base_volume)
        for _ in range(bars_per_leg):
            _emit(-pull / bars_per_leg, base_volume)
    for _ in range(bars_per_leg):
        _emit(weak_leg / bars_per_leg, weak_volume)
    for _ in range(break_bars):
        _emit(-break_step, weak_volume)

    n = len(highs)
    times = [T0 + i * STEP for i in range(n)]
    closes = [(h + lo) / 2.0 for h, lo in zip(highs, lows, strict=True)]
    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n)]
    return pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes,
        }
    )


def test_fires_choch_reversal_signal_on_weakening_uptrend():
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM5()
    signals = strategy.evaluate(
        MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=20.0)
    )

    assert signals
    assert len(signals) == 1  # cold learner -> 1 leg
    assert signals[0].direction is Direction.SELL
    assert signals[0].pattern == "CHOCH_REV"


def test_leg_count_scales_up_once_the_learner_is_ready_and_confident():
    from src.strategies.domain.online_learning import _BucketState

    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM5()
    strategy._learner = strategy._ensure_learner(strategy.spec.params)
    strategy._learner._buckets["reversal_sell"] = _BucketState(count=120.0, wins=118.0)
    strategy._learner._global_count = 120.0
    strategy._learner._global_wins = 118.0

    signals = strategy.evaluate(
        MarketContext(symbol="XAUUSD", candles={"M5": df}, spread_points=20.0)
    )
    assert signals
    assert len(signals) == 3
    assert {s.sl_points for s in signals} == {signals[0].sl_points}


def test_sl_ratchet_wired_on_m5_too():
    # Full state-machine coverage lives in test_xauusd_structure_shift_m1.py
    # (identical logic); this is the "did the copy/paste bring it over"
    # check for this file specifically.
    strategy = XauusdStructureShiftM5()
    strategy._batch_direction = Direction.SELL
    strategy._batch_tp_prices = [100.0]
    strategy._batch_ratcheted = [False]
    strategy._batch_atr = 1.0

    assert strategy._sl_ratchet(Direction.SELL, 101.0) is None
    result = strategy._sl_ratchet(Direction.SELL, 99.0)
    assert isinstance(result, ExitDecision)
    assert result.action is ExitActionKind.SET_SL
    assert result.target_price == pytest.approx(100.3)  # tp + 0.3 ATR, sell side
    assert strategy._batch_ratcheted == [True]


def test_evaluate_routes_to_sl_ratchet_when_position_open():
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM5()
    close = float(df["close"].iloc[-1])
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M5": df},
        spread_points=20.0,
        own_position=PositionSnapshot(
            direction=Direction.SELL,
            entry_price=100.0,
            sl=None,
            tp=None,
            opened_at=datetime.now(UTC),
        ),
    )
    strategy._batch_direction = Direction.SELL
    strategy._batch_tp_prices = [close + 0.01]
    strategy._batch_ratcheted = [False]
    strategy._batch_atr = 1.0

    result = strategy.evaluate(ctx)
    assert isinstance(result, ExitDecision)
    assert result.action is ExitActionKind.SET_SL
