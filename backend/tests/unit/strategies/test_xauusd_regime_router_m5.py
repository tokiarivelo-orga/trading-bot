"""Unit tests for `xauusd_regime_router_m5_v1.py` — the M5 sibling of
`xauusd_regime_router_m1_v3.py` (see that file's module docstring for the
full design, and this file's own module docstring for exactly what changed
in the M5 retimeframe: `entry_tf_minutes` and the `learner_horizon_bars`
real-time conversion).

Same coverage shape as `test_xauusd_regime_router_m1.py`: sandbox
acceptance, mode routing (trending -> Mode A, the Mode-A-absent fallback to
Mode B, Mode C firing strictly on a genuine RANGING read AND an explicit
`enable_range_reversion=True` opt-in, off by default), the regime-flip
BREAKEVEN exit, the artifact-sourced prior_logit plumbing, the
size_multiplier clamp, and — the core "prevent loss" proof — the
learner-veto-on-negative-expectancy test. Fixtures are the same
zigzag/weakening-reversal/choppy-range constructions the M1 test module
uses (M5 vs M1 makes no difference to these synthetic candle shapes; the
strategy reads whatever `entry_timeframe` frame it's handed), just fed
through `XauusdRegimeRouterM5` instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.domain.regime import TradingSession, TrendRegime
from src.strategies.domain.models import (
    Direction,
    ExitDecision,
    MarketContext,
    PositionSnapshot,
    Signal,
)
from src.strategies.domain.online_learning import LearnerVerdict
from src.strategies.generated.xauusd_regime_router_m5_v1 import (
    BUCKET_TABLE,
    HAS_EDGE,
    N_FEATURES,
    OVERALL_SECURE_RATE,
    XauusdRegimeRouterM5,
    _detect_range_reversion,
    _empirical_offset,
    _logit,
    _prior_net_correction,
    _regime_features,
    _size_multiplier,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_regime_router_m5_v1.py"
)

STEP = timedelta(minutes=5)
# T0 is chosen so the 90-bar zigzag/band-rejection fixtures below (index 89
# or 90, ~7h25m-7h30m of M5 bars after T0) land their LAST bar — the only
# one `evaluate()` reads a session from (`session_for(entry["time"].iloc[-1],
# ...)`) — inside the overlap session (12:00-16:00 UTC per
# `RegimeConfig`'s defaults). The M1 test module's T0=13:00 doesn't
# transfer here: at 1-minute spacing the same 90-bar fixture only spans 90
# minutes and stays inside whatever session it started in, but at 5-minute
# spacing it spans ~7.5 hours, so T0 has to sit *before* the overlap window
# by roughly that much instead of inside it.
T0 = datetime(2026, 8, 3, 5, 0, tzinfo=UTC)  # a Monday, 05:00 UTC


def _frame(highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray | None = None) -> pd.DataFrame:
    n = len(highs)
    closes = (highs + lows) / 2.0
    times = [T0 + i * STEP for i in range(n)]
    vol = np.full(n, 500.0) if volumes is None else volumes
    return pd.DataFrame(
        {
            "time": times,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": vol,
        }
    )


def _build_zigzag(
    *, n_legs: int, leg: float, pull: float, bars_per_leg: int, tail_bars: int, tail_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clean uptrend zigzag — impulse/pullback cycles then a resumption
    tail. Same construction the M1 test module (and
    `test_structure_continuation.py`) uses for its own
    `detect_trend_continuation` firing fixture — the detector reads bar
    shape, not wall-clock spacing, so this is unchanged for M5."""
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for _ in range(n_legs):
        for _ in range(bars_per_leg):
            price += leg / bars_per_leg
            highs.append(price + 0.05)
            lows.append(price - 0.05)
        for _ in range(bars_per_leg):
            price -= pull / bars_per_leg
            highs.append(price + 0.05)
            lows.append(price - 0.05)
    for _ in range(tail_bars):
        price += tail_step
        highs.append(price + 0.05)
        lows.append(price - 0.05)
    return np.array(highs), np.array(lows)


def _build_weakening_reversal(
    *, break_bars: int, break_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warmup uptrend legs, then one smaller/lower-volume impulse, then a
    break back through the neckline — same construction as the M1 test
    module's own helper."""
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    price = 100.0

    def _emit(step: float, vol: float) -> None:
        nonlocal price
        price += step
        highs.append(price + 0.05)
        lows.append(price - 0.05)
        volumes.append(vol)

    for _ in range(8):
        for _ in range(4):
            _emit(3.0 / 4, 1000.0)
        for _ in range(4):
            _emit(-1.0 / 4, 1000.0)
    for _ in range(4):
        _emit(1.0 / 4, 300.0)
    for _ in range(break_bars):
        _emit(-break_step, 300.0)

    return np.array(highs), np.array(lows), np.array(volumes)


def _build_choppy_range(*, n: int, step: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Tight two-bar alternation -> ADX ~0 -> RANGING. Same construction
    the M1 test module uses."""
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for i in range(n):
        price += step if i % 2 == 0 else -step
        highs.append(price + offset)
        lows.append(price - offset)
    return np.array(highs), np.array(lows)


def test_sandbox_accepts_m5_source() -> None:
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


def test_spec_is_retimeframed_to_m5() -> None:
    strategy = XauusdRegimeRouterM5()
    assert strategy.spec.name == "xauusd_regime_router_m5"
    assert strategy.spec.entry_timeframe == "M5"
    # One and two rungs up from M5 — see module docstring point 2, not the
    # M1 file's own ("M5", "H1") pair.
    assert strategy.spec.confirmation_timeframes == ("H1", "H4")
    assert strategy.spec.params["entry_tf_minutes"] == 5


def test_returns_none_on_short_history() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows = _build_zigzag(
        n_legs=2, leg=1.0, pull=0.3, bars_per_leg=4, tail_bars=1, tail_step=0.1
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_mode_a_fires_on_clean_uptrend_continuation() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)

    assert isinstance(signal, Signal)
    assert signal.direction is Direction.BUY
    assert signal.pattern == "trend_continuation"
    assert signal.sl_points > 0.0
    assert signal.tp_points / signal.sl_points >= strategy.spec.params["broker_min_rr"]
    assert 1.0 <= signal.size_multiplier <= 2.0
    assert strategy._entry_mode == "trend_continuation"
    # Bucket key format per the plan: f"{mode}|{session.value}|{trend.value}"
    assert "session=overlap trend=trending" in signal.reason


def test_mode_b_fires_when_continuation_absent_but_reversal_present() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows, volumes = _build_weakening_reversal(break_bars=4, break_step=0.35)
    frame = _frame(highs, lows, volumes)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)

    assert isinstance(signal, Signal)
    assert signal.direction is Direction.SELL
    assert signal.pattern == "structure_reversal"
    assert signal.sl_points > 0.0
    assert strategy._entry_mode == "structure_reversal"


def test_oversized_reversal_break_is_skipped_not_widened() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows, volumes = _build_weakening_reversal(break_bars=5, break_step=1.0)
    frame = _frame(highs, lows, volumes)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_detect_range_reversion_buy_fade_on_lower_band_rejection() -> None:
    lookback, band_mult = 20, 2.0
    highs, lows = _build_choppy_range(n=90, step=0.5, offset=0.3)
    closes = (highs + lows) / 2.0
    mean_recent = float(np.mean(closes[-lookback:]))
    std_recent = float(np.std(closes[-lookback:], ddof=0))
    lower = mean_recent - band_mult * std_recent
    spike_low = lower - 0.2
    spike_close = lower + 0.05
    highs = np.append(highs, spike_close + 0.05)
    lows = np.append(lows, spike_low)
    closes = np.append(closes, spike_close)

    setup = _detect_range_reversion(highs, lows, closes, lookback=lookback, band_mult=band_mult)
    assert setup is not None
    assert setup.direction is Direction.BUY
    assert setup.pivot_price == float(lows[-1])


def test_detect_range_reversion_none_without_a_rejection_back_inside() -> None:
    lookback, band_mult = 20, 2.0
    highs, lows = _build_choppy_range(n=90, step=0.5, offset=0.3)
    closes = (highs + lows) / 2.0
    mean_recent = float(np.mean(closes[-lookback:]))
    std_recent = float(np.std(closes[-lookback:], ddof=0))
    upper = mean_recent + band_mult * std_recent
    spike_close = upper + 0.3
    highs = np.append(highs, spike_close + 0.05)
    lows = np.append(lows, spike_close - 0.05)
    closes = np.append(closes, spike_close)

    setup = _detect_range_reversion(highs, lows, closes, lookback=lookback, band_mult=band_mult)
    assert setup is None


def _build_band_rejection_frame(
    lookback: int, band_mult: float
) -> tuple[np.ndarray, np.ndarray]:
    """Choppy RANGING base plus one bar that pierces the upper Bollinger
    band and closes back inside it. Shared by the on/off
    `enable_range_reversion` tests below."""
    highs, lows = _build_choppy_range(n=90, step=0.5, offset=0.3)
    closes = (highs + lows) / 2.0
    mean_recent = float(np.mean(closes[-lookback:]))
    std_recent = float(np.std(closes[-lookback:], ddof=0))
    upper = mean_recent + band_mult * std_recent
    spike_high = upper + 0.2
    spike_close = upper - 0.05
    spike_low = 2.0 * spike_close - spike_high
    highs = np.append(highs, spike_high)
    lows = np.append(lows, spike_low)
    return highs, lows


def test_mode_c_disabled_by_default_even_on_a_qualifying_rejection() -> None:
    """`enable_range_reversion` defaults to `False`, inherited from the M1
    file's own real-backtest finding — see module docstring "WHAT CHANGED"
    point 6. The exact same fixture that fires below once opted in must
    produce no Signal on a fresh, unmodified instance."""
    strategy = XauusdRegimeRouterM5()
    assert strategy.spec.params["enable_range_reversion"] is False

    lookback = int(strategy.spec.params["range_lookback_bars"])
    band_mult = float(strategy.spec.params["range_band_mult"])
    highs, lows = _build_band_rejection_frame(lookback, band_mult)

    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_mode_c_fires_on_band_rejection_when_ranging_and_explicitly_enabled() -> None:
    strategy = XauusdRegimeRouterM5()
    strategy.spec.params["enable_range_reversion"] = True
    lookback = int(strategy.spec.params["range_lookback_bars"])
    band_mult = float(strategy.spec.params["range_band_mult"])

    highs, lows = _build_band_rejection_frame(lookback, band_mult)

    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)

    assert isinstance(signal, Signal)
    assert signal.direction is Direction.SELL
    assert signal.pattern == "range_reversion"
    assert signal.sl_points > 0.0
    assert 1.0 <= signal.size_multiplier <= 2.0
    assert strategy._entry_mode == "range_reversion"
    assert "session=overlap trend=ranging" in signal.reason


def test_no_new_entry_while_own_position_is_open() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    own_position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=float(frame["close"].iloc[-1]),
        sl=float(frame["close"].iloc[-1]) - 2.0,
        tp=float(frame["close"].iloc[-1]) + 5.0,
        opened_at=frame["time"].iloc[-1],
    )
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M5": frame, "H1": frame, "H4": frame},
        spread_points=15.0,
        own_position=own_position,
    )
    assert strategy.evaluate(ctx) is None


def test_breakeven_exit_on_mode_a_position_when_regime_flips_to_ranging() -> None:
    strategy = XauusdRegimeRouterM5()
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    trend_frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M5": trend_frame, "H1": trend_frame, "H4": trend_frame},
        spread_points=15.0,
    )
    entry_signal = strategy.evaluate(ctx)
    assert isinstance(entry_signal, Signal)
    assert strategy._entry_mode == "trend_continuation"

    last_price = float(trend_frame["close"].iloc[-1])
    flat_n = 150
    flat_highs = []
    flat_lows = []
    price = last_price
    for i in range(flat_n):
        price += 0.05 if i % 2 == 0 else -0.05
        flat_highs.append(price + 0.03)
        flat_lows.append(price - 0.03)
    flat_frame = _frame(np.array(flat_highs), np.array(flat_lows))
    flat_frame["time"] = [
        trend_frame["time"].iloc[-1] + (i + 1) * STEP for i in range(flat_n)
    ]
    full = pd.concat([trend_frame, flat_frame], ignore_index=True).iloc[-200:].reset_index(
        drop=True
    )

    own_position = PositionSnapshot(
        direction=Direction.BUY,
        entry_price=last_price,
        sl=last_price - 2.0,
        tp=last_price + 5.0,
        opened_at=trend_frame["time"].iloc[-1],
    )
    ctx2 = MarketContext(
        symbol="XAUUSD",
        candles={"M5": full, "H1": full, "H4": full},
        spread_points=15.0,
        own_position=own_position,
    )
    result = strategy.evaluate(ctx2)
    assert isinstance(result, ExitDecision)
    assert result.action.value == "breakeven"
    assert "TRENDING -> RANGING" in result.reason


def test_empirical_offset_matches_bucket_table() -> None:
    offset = _empirical_offset(TrendRegime.TRENDING, TradingSession.OVERLAP)
    expected = _logit(BUCKET_TABLE[("trending", "overlap")]) - _logit(OVERALL_SECURE_RATE)
    assert offset == expected
    assert offset > 0.0

    assert _empirical_offset(TrendRegime.TRENDING, TradingSession.OFF_SESSION) == 0.0


def test_prior_net_correction_is_inert_while_artifact_has_no_edge() -> None:
    assert HAS_EDGE is False
    features = _regime_features(TradingSession.OVERLAP, TrendRegime.TRENDING, "mid", 13.5, 0)
    assert features.shape == (N_FEATURES,)
    assert _prior_net_correction(features) == 0.0


def test_size_multiplier_clamped_to_engine_band() -> None:
    breakeven_p = 0.8333
    cold_verdict = LearnerVerdict(
        p_secure=0.5, expectancy_r=-0.5, bucket_samples=3.0, model_samples=0, ready=False
    )
    assert _size_multiplier(cold_verdict, breakeven_p, edge_scale=2.0) == 1.0

    confident_verdict = LearnerVerdict(
        p_secure=0.999, expectancy_r=0.5, bucket_samples=500.0, model_samples=500, ready=True
    )
    sized = _size_multiplier(confident_verdict, breakeven_p, edge_scale=2.0)
    assert 1.0 <= sized <= 2.0

    barely_warm_verdict = LearnerVerdict(
        p_secure=breakeven_p, expectancy_r=0.0, bucket_samples=25.0, model_samples=0, ready=True
    )
    assert _size_multiplier(barely_warm_verdict, breakeven_p, edge_scale=2.0) == 1.0


def test_learner_vetoes_qualifying_setup_after_bucket_shows_negative_expectancy() -> None:
    """The central "prevent loss" mechanism the whole plan is built on: drive
    `AdaptiveLearner` through enough RESOLVED trades that a bucket goes
    `ready=True` with negative expectancy, then confirm a subsequent,
    otherwise-qualifying setup in that SAME bucket gets vetoed
    (`evaluate()` returns `None`, not a `Signal`).

    Uses 5-minute spacing/deadlines (this file's own `entry_tf_minutes=5`
    and `STEP`), not the M1 test module's 1-minute ones — the mechanics
    under test (`AdaptiveLearner.observe`/`advance`/`score`) don't care
    about wall-clock units, only that entry/resolution/deadline timestamps
    are internally consistent, which they are here.

    Bucket = `trend_continuation|overlap|trending` — the exact bucket
    `test_mode_a_fires_on_clean_uptrend_continuation` above lands in (T0 is
    a Monday 13:00 UTC == overlap, and a clean uptrend zigzag reads
    TRENDING), so the fresh setup fed in below is guaranteed to land in the
    same poisoned bucket."""
    strategy = XauusdRegimeRouterM5()
    learner = strategy._ensure_learner(strategy.spec.params)

    bucket = "trend_continuation|overlap|trending"
    features = _regime_features(TradingSession.OVERLAP, TrendRegime.TRENDING, "mid", 13.5, 0)
    entry_price = 100.0
    sl_dist = 2.0
    atr_now = 1.0
    secure_dist = sl_dist * float(strategy.spec.params["learner_secure_r"])

    ns_per_minute = 60_000_000_000
    entry_tf_minutes = int(strategy.spec.params["entry_tf_minutes"])
    bar_ns = entry_tf_minutes * ns_per_minute
    base_ns = int(pd.Timestamp(T0).value)
    spacing_ns = 10_000 * bar_ns  # keeps every sample's own resolution window independent

    # 40 pure stop-outs in this one bucket: the resolving bar's high barely
    # moves in favour (well under `secure_dist`) while its low blows straight
    # through the stop — an unambiguous label=0 (stop-first) every single
    # time, never a secure hit.
    for i in range(40):
        entry_ns = base_ns + i * spacing_ns
        recorded = learner.observe(
            features=features,
            bucket=bucket,
            entry_ns=entry_ns,
            entry_price=entry_price,
            direction=1,
            sl_dist=sl_dist,
            atr=atr_now,
            deadline_ns=entry_ns + 180 * bar_ns,
        )
        assert recorded
        resolved = learner.advance(
            times_ns=np.array([entry_ns + bar_ns]),
            highs=np.array([entry_price + 0.1 * secure_dist]),
            lows=np.array([entry_price - 1.25 * sl_dist]),
        )
        assert len(resolved) == 1
        assert resolved[0].label == 0

    prior_logit = _empirical_offset(TrendRegime.TRENDING, TradingSession.OVERLAP)
    verdict = learner.score(features, bucket, prior_logit=prior_logit)
    assert verdict.ready
    assert verdict.bucket_samples >= float(strategy.spec.params["learner_min_bucket_samples"])
    assert verdict.expectancy_r < 0.0

    # A fresh, otherwise-qualifying Mode A setup in the SAME bucket — but
    # this strategy instance's learner has now seen 40 straight stop-outs
    # in exactly this regime bucket. `evaluate()` must decline it.
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_learner_does_not_veto_a_qualifying_setup_after_bucket_shows_positive_expectancy() -> None:
    """Contrasting case: the same mechanism, but every resolved sample in the
    bucket secures instead of stopping out — the bucket should warm up to a
    clearly positive verdict, and the strategy must still take the trade."""
    strategy = XauusdRegimeRouterM5()
    learner = strategy._ensure_learner(strategy.spec.params)

    bucket = "trend_continuation|overlap|trending"
    features = _regime_features(TradingSession.OVERLAP, TrendRegime.TRENDING, "mid", 13.5, 0)
    entry_price = 100.0
    sl_dist = 2.0
    atr_now = 1.0
    secure_dist = sl_dist * float(strategy.spec.params["learner_secure_r"])

    ns_per_minute = 60_000_000_000
    entry_tf_minutes = int(strategy.spec.params["entry_tf_minutes"])
    bar_ns = entry_tf_minutes * ns_per_minute
    base_ns = int(pd.Timestamp(T0).value)
    spacing_ns = 10_000 * bar_ns

    # 40 pure secures: the resolving bar's high clears `secure_dist` while
    # its low never comes close to the stop — an unambiguous label=1 every
    # time.
    for i in range(40):
        entry_ns = base_ns + i * spacing_ns
        assert learner.observe(
            features=features,
            bucket=bucket,
            entry_ns=entry_ns,
            entry_price=entry_price,
            direction=1,
            sl_dist=sl_dist,
            atr=atr_now,
            deadline_ns=entry_ns + 180 * bar_ns,
        )
        resolved = learner.advance(
            times_ns=np.array([entry_ns + bar_ns]),
            highs=np.array([entry_price + 1.5 * secure_dist]),
            lows=np.array([entry_price - 0.1 * sl_dist]),
        )
        assert len(resolved) == 1
        assert resolved[0].label == 1

    prior_logit = _empirical_offset(TrendRegime.TRENDING, TradingSession.OVERLAP)
    verdict = learner.score(features, bucket, prior_logit=prior_logit)
    assert verdict.ready
    assert verdict.expectancy_r > 0.0

    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M5": frame, "H1": frame, "H4": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)
    assert isinstance(signal, Signal)
    assert signal.pattern == "trend_continuation"
