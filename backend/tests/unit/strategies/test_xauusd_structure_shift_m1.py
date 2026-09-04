"""Unit tests for `xauusd_structure_shift_m1_v1.py` — sandbox acceptance,
a synthetic end-to-end fire on a clean CHoCH weakening-reversal setup, the
anti-spam pivot dedupe, and the volume-confirmation gate.

Logic tests instantiate `XauusdStructureShiftM1` directly (this is trusted
test code, not arbitrary generated code) rather than through
`validate_and_load` — that avoids any doubt about whether its own internal
smoke test (which calls `evaluate()` once against generic synthetic
candles as part of validation) leaves state on the instance this file's
own assertions would then depend on.
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
from src.strategies.generated.xauusd_structure_shift_m1_v1 import XauusdStructureShiftM1
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_structure_shift_m1_v1.py"
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=1)


def test_sandbox_accepts_source():
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


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
    """Same shape as `test_structure_continuation.py`'s
    `_build_weakening_reversal` fixture, packaged as the OHLCV frame a
    strategy's `evaluate()` actually receives via `MarketContext.candles`."""
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


def _build_continuation_frame(
    *,
    n_legs: int = 11,
    leg: float = 3.0,
    pull: float = 1.0,
    bars_per_leg: int = 4,
    tail_bars: int = 3,
    tail_step: float = 0.5,
    offset: float = 0.05,
    volume: float = 1000.0,
    up: bool = True,
) -> pd.DataFrame:
    """Continuation-side counterpart to `_build_reversal_frame`, matching
    `test_structure_continuation.py`'s own `_build_zigzag` shape (clean
    impulse/pullback cycles, then a resumption tail) packaged as an OHLCV
    frame with uniform volume — the volume-confirmation gate only applies
    to this setup kind, so it needs its own fixture to exercise."""
    sign = 1.0 if up else -1.0
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for _ in range(n_legs):
        for _ in range(bars_per_leg):
            price += sign * leg / bars_per_leg
            highs.append(price + offset)
            lows.append(price - offset)
        for _ in range(bars_per_leg):
            price -= sign * pull / bars_per_leg
            highs.append(price + offset)
            lows.append(price - offset)
    for _ in range(tail_bars):
        price += sign * tail_step
        highs.append(price + offset)
        lows.append(price - offset)

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
            "tick_volume": [volume] * n,
        }
    )


def _ctx(df: pd.DataFrame, spread_points: float = 20.0) -> MarketContext:
    return MarketContext(symbol="XAUUSD", candles={"M1": df}, spread_points=spread_points)


def test_fires_choch_reversal_signal_on_weakening_uptrend():
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    signals = strategy.evaluate(_ctx(df))

    assert signals
    assert len(signals) == 1  # cold learner -> 1 leg, see the leg-count test below
    signal = signals[0]
    assert signal.direction is Direction.SELL  # reversing OUT of the uptrend
    assert signal.sl_points > 0.0
    assert signal.tp_points > 0.0
    assert signal.pattern == "CHOCH_REV"
    assert signal.structure  # chart-annotation data populated


def test_cold_learner_uses_the_conservative_default_target_not_the_grid_max():
    """A cold bucket's `best_target()` would always return the grid's
    largest R (its expected-value math has nothing real to optimise
    against yet, so it's monotonically increasing in r) — that is a
    cold-start artifact, not a learned target, so `evaluate()` must not
    call `best_target()` until the bucket has enough resolved samples for
    its MFE-survival reservoir to mean something. A fresh instance's very
    first fire must land on a single leg at `tp_grid_r[0]`, the same
    conservative default the un-adapted gate already falls back to."""
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    signals = strategy.evaluate(_ctx(df))

    assert signals
    assert len(signals) == 1
    grid = strategy.spec.params["tp_grid_r"]
    assert f"r={grid[0]:.2f}" in signals[0].reason
    assert f"r={grid[-1]:.2f}" not in signals[0].reason


def test_fires_choch_reversal_signal_on_weakening_downtrend():
    df = _build_reversal_frame(up=False)
    strategy = XauusdStructureShiftM1()
    signals = strategy.evaluate(_ctx(df))

    assert signals
    assert signals[0].direction is Direction.BUY  # reversing OUT of the downtrend


def test_anti_spam_guard_suppresses_repeat_fire_on_same_pivot():
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    first = strategy.evaluate(_ctx(df))
    assert first

    # Same candle stream again — same setup, same broken pivot. Without the
    # `_observed` dedupe this would re-fire (and re-grade the learner)
    # every single call, exactly the "in_zone, not fresh_touch" overtrading
    # bug this guard exists to avoid.
    second = strategy.evaluate(_ctx(df))
    assert second is None


def test_fires_continuation_signal_when_no_reversal_and_volume_confirms():
    df = _build_continuation_frame(up=True)
    strategy = XauusdStructureShiftM1()
    signals = strategy.evaluate(_ctx(df))

    assert signals
    assert signals[0].direction is Direction.BUY
    assert signals[0].pattern == "TREND_CONT"


def test_volume_confirmation_gate_suppresses_low_participation_continuation():
    # Same qualifying continuation structure, but the resumption tail
    # carries participation far below the rolling mean — a drift-through,
    # not a confirmed resumption. Reversal setups are deliberately exempt
    # from this gate (see the strategy's own module docstring for why), so
    # this must be exercised on the continuation path specifically.
    df = _build_continuation_frame(up=True)
    df.loc[df.index[-3:], "tick_volume"] = 1.0
    strategy = XauusdStructureShiftM1()
    signals = strategy.evaluate(_ctx(df))
    assert signals is None


def test_leg_count_scales_up_only_once_the_learner_is_ready_and_confident():
    """A cold or barely-passing verdict must stay at 1 leg; only a *ready*
    bucket whose p_secure clears the 3-leg threshold gets more legs — and
    every extra leg shares the exact same SL, only adding a farther TP on
    top of it, never loosening risk."""
    from src.strategies.domain.online_learning import _BucketState

    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    strategy._learner = strategy._ensure_learner(strategy.spec.params)

    # `evaluate()` fires a SELL reversal on this fixture (see the uptrend
    # firing test above), so the bucket it will actually score against is
    # "reversal_sell". Seed that bucket's decayed win/count state directly
    # — `score()` only ever reads count/wins plus the global rate, so this
    # is an honest, minimal way to put the learner in a "seen this before,
    # worked" state without depending on evaluate()'s own gating to get a
    # bucket there one resolved trade at a time.
    bucket = "reversal_sell"
    strategy._learner._buckets[bucket] = _BucketState(count=120.0, wins=118.0)
    strategy._learner._global_count = 120.0
    strategy._learner._global_wins = 118.0

    signals = strategy.evaluate(_ctx(df))
    assert signals
    assert len(signals) == 3
    assert {s.sl_points for s in signals} == {signals[0].sl_points}  # shared SL
    grid = strategy.spec.params["tp_grid_r"]
    reasons = [s.reason for s in signals]
    assert any(f"r={grid[0]:.2f}" in r for r in reasons)
    assert any(f"r={grid[1]:.2f}" in r for r in reasons)
    assert any(f"r={grid[2]:.2f}" in r for r in reasons)


def test_leg_count_stays_at_one_for_a_barely_passing_confidence():
    """Just past the trade gate but short of the 2-leg threshold: still a
    single leg — sizing up is earned, not the default once a setup merely
    clears the bar to be traded at all."""
    from src.strategies.domain.online_learning import _BucketState

    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    strategy._learner = strategy._ensure_learner(strategy.spec.params)

    bucket = "reversal_sell"
    # count=wins=100 -> empirical rate ~1.0, shrunk by the bucket prior
    # toward global_rate; tuned here to land just above the 0.65 gate but
    # below the 0.75 two-leg threshold.
    strategy._learner._buckets[bucket] = _BucketState(count=100.0, wins=68.0)
    strategy._learner._global_count = 100.0
    strategy._learner._global_wins = 68.0

    signals = strategy.evaluate(_ctx(df))
    assert signals
    assert len(signals) == 1


def test_spec_matches_m1_convention():
    strategy = XauusdStructureShiftM1()
    assert strategy.spec.name == "xauusd_structure_shift_m1"
    assert strategy.spec.symbols == ("XAUUSD",)
    assert strategy.spec.entry_timeframe == "M1"
    assert strategy.spec.confirmation_timeframes == ("M15",)


def test_no_candles_returns_none():
    strategy = XauusdStructureShiftM1()
    ctx = MarketContext(symbol="XAUUSD", candles={}, spread_points=20.0)
    assert strategy.evaluate(ctx) is None


def test_own_position_blocks_new_signal():
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": df},
        spread_points=20.0,
        own_position=PositionSnapshot(
            direction=Direction.SELL,
            entry_price=100.0,
            sl=None,
            tp=None,
            opened_at=datetime.now(UTC),
        ),
    )
    assert strategy.evaluate(ctx) is None


def test_sl_ratchet_progresses_through_legs_directly():
    """Unit test of `_sl_ratchet` itself, bypassing evaluate()'s detection
    plumbing — this is about the ratchet's own state machine, not setup
    detection."""
    strategy = XauusdStructureShiftM1()
    strategy._batch_direction = Direction.SELL
    strategy._batch_tp_prices = [100.0, 95.0]  # nearest-first
    strategy._batch_ratcheted = [False, False]
    strategy._batch_atr = 1.0  # ratchet_buffer_atr_mult default 0.3 -> buffer 0.3

    # Price hasn't reached the first level yet.
    assert strategy._sl_ratchet(Direction.SELL, 101.0) is None

    # Price clears leg 1's TP (a SELL favours price going down). Target is
    # a little *over* TP1 (100.0 + 0.3 buffer), per the SELL-side rule.
    result = strategy._sl_ratchet(Direction.SELL, 99.5)
    assert isinstance(result, ExitDecision)
    assert result.action is ExitActionKind.SET_SL
    assert result.target_price == pytest.approx(100.3)
    assert strategy._batch_ratcheted == [True, False]

    # Same price again: leg 1 already ratcheted, leg 2 not yet reached —
    # must not re-fire.
    assert strategy._sl_ratchet(Direction.SELL, 99.5) is None

    # Price clears leg 2's TP too — target ratchets further, to 95.0 + 0.3.
    result2 = strategy._sl_ratchet(Direction.SELL, 94.0)
    assert isinstance(result2, ExitDecision)
    assert result2.action is ExitActionKind.SET_SL
    assert result2.target_price == pytest.approx(95.3)
    assert strategy._batch_ratcheted == [True, True]

    # Nothing left to ratchet — both legs already reacted to.
    assert strategy._sl_ratchet(Direction.SELL, 90.0) is None


def test_sl_ratchet_returns_none_with_no_active_batch():
    strategy = XauusdStructureShiftM1()
    assert strategy._sl_ratchet(Direction.BUY, 100.0) is None


def test_evaluate_routes_to_sl_ratchet_when_position_open():
    """Wiring test: with `ctx.own_position` set, `evaluate()` must consult
    `_sl_ratchet` rather than generate a fresh Signal — one trade at a time
    still holds; the ratchet is the only thing that can fire."""
    df = _build_reversal_frame(up=True)
    strategy = XauusdStructureShiftM1()
    close = float(df["close"].iloc[-1])
    own_position = PositionSnapshot(
        direction=Direction.SELL,
        entry_price=100.0,
        sl=None,
        tp=None,
        opened_at=datetime.now(UTC),
    )
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": df}, spread_points=20.0, own_position=own_position
    )

    # Armed, but far from reached — no Signal, no ExitDecision either.
    strategy._batch_direction = Direction.SELL
    strategy._batch_tp_prices = [close - 1000.0]
    strategy._batch_ratcheted = [False]
    strategy._batch_atr = 1.0
    assert strategy.evaluate(ctx) is None

    # Armed at a level this bar's own close has already cleared.
    strategy._batch_tp_prices = [close + 0.01]
    strategy._batch_ratcheted = [False]
    result = strategy.evaluate(ctx)
    assert isinstance(result, ExitDecision)
    assert result.action is ExitActionKind.SET_SL
