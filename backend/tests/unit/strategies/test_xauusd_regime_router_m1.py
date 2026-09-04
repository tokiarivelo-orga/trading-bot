"""Unit tests for `xauusd_regime_router_m1_v3.py` — the regime-switching
XAUUSD M1 meta-strategy (Phase 2 of the "regime-switching, DL-guided" plan;
see `backend/data/ml_models/regime_router_priors_v1.json` for Phase 1's
trained prior artifact).

Deeper coverage (a real backtest, an integration test through the full
`TradeEngine`/`PositionManager` path) is later phases of that plan — this
file is the skill's own stub + smoke-test requirement: sandbox acceptance,
mode routing (trending -> Mode A, the Mode-A-absent fallback to Mode B —
see the strategy's own module docstring "MODE ROUTING" section for why a
strict RANGING-only pre-gate on Mode B would make it unreachable — and
Mode C firing strictly on a genuine RANGING read AND an explicit
`enable_range_reversion=True` opt-in), the regime-flip BREAKEVEN exit, the
artifact-sourced prior_logit plumbing, and the size_multiplier clamp.

v3: a real XAUUSD backtest found Mode C (range_reversion) alone at PF 0.37
(losing in every regime bucket) while Modes A+B together ran PF 4.31 — so
`enable_range_reversion` now defaults to `False` (see the generated file's
own module docstring "DISABLED BY DEFAULT" section). Mode C's tests below
still exercise the real code, just opted in explicitly rather than relying
on the old always-on default; `test_mode_c_disabled_by_default_even_on_a_
qualifying_rejection` pins the new off default itself.
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
from src.strategies.generated.xauusd_regime_router_m1_v3 import (
    BUCKET_TABLE,
    HAS_EDGE,
    N_FEATURES,
    OVERALL_SECURE_RATE,
    XauusdRegimeRouterM1,
    _detect_range_reversion,
    _empirical_offset,
    _logit,
    _prior_net_correction,
    _regime_features,
    _size_multiplier,
)
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = (
    Path(__file__).resolve().parents[3] / "src/strategies/generated/xauusd_regime_router_m1_v3.py"
)

T0 = datetime(2026, 8, 3, 13, 0, tzinfo=UTC)  # a Monday, 13:00 UTC == overlap session
STEP = timedelta(minutes=1)


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
    tail. Same construction `test_structure_continuation.py` uses for its
    own `detect_trend_continuation` firing fixture."""
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
    break back through the neckline — same construction as
    `test_structure_continuation.py::_build_weakening_reversal`. Kept
    modest (`break_bars=4, break_step=0.35`, verified against the running
    strategy) so the resulting SL distance stays inside this strategy's
    own `sl_max_atr_mult` cap — a much larger break (e.g. the sibling
    module test's own default `break_bars=5, break_step=1.0`) still makes
    `detect_structure_reversal` fire, but lands a pivot-to-close distance
    this strategy correctly refuses as an oversized stop."""
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
    `test_structure_continuation.py::test_none_when_market_is_choppy_not_trending`
    uses, and this file's own regime-flip exit test below — the one shared
    fixture shape for "the top-level regime reads RANGING"."""
    highs: list[float] = []
    lows: list[float] = []
    price = 100.0
    for i in range(n):
        price += step if i % 2 == 0 else -step
        highs.append(price + offset)
        lows.append(price - offset)
    return np.array(highs), np.array(lows)


def test_sandbox_accepts_v3_source() -> None:
    code = STRATEGY_PATH.read_text()
    instance, errors = validate_and_load(code)
    assert instance is not None, errors


def test_returns_none_on_short_history() -> None:
    strategy = XauusdRegimeRouterM1()
    highs, lows = _build_zigzag(
        n_legs=2, leg=1.0, pull=0.3, bars_per_leg=4, tail_bars=1, tail_step=0.1
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_mode_a_fires_on_clean_uptrend_continuation() -> None:
    strategy = XauusdRegimeRouterM1()
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
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
    strategy = XauusdRegimeRouterM1()
    highs, lows, volumes = _build_weakening_reversal(break_bars=4, break_step=0.35)
    frame = _frame(highs, lows, volumes)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)

    assert isinstance(signal, Signal)
    assert signal.direction is Direction.SELL
    assert signal.pattern == "structure_reversal"
    assert signal.sl_points > 0.0
    assert strategy._entry_mode == "structure_reversal"


def test_oversized_reversal_break_is_skipped_not_widened() -> None:
    """Same setup type as the firing test above, but with the sibling
    structure_continuation test module's own larger default break
    (`break_bars=5, break_step=1.0`) — `detect_structure_reversal` still
    fires, but the resulting pivot-to-close distance exceeds this
    strategy's `sl_max_atr_mult` cap, so no Signal is emitted."""
    strategy = XauusdRegimeRouterM1()
    highs, lows, volumes = _build_weakening_reversal(break_bars=5, break_step=1.0)
    frame = _frame(highs, lows, volumes)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_detect_range_reversion_buy_fade_on_lower_band_rejection() -> None:
    # Exercises `_detect_range_reversion` directly, not through
    # `evaluate()` — unaffected by `enable_range_reversion` (v3's
    # off-by-default call-site gate only wraps where `evaluate()` calls
    # this function, see `test_mode_c_*` below for that path).
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
    # Direct detector call, same as above — unaffected by
    # `enable_range_reversion`.
    # High pierces the upper band but the close stays BEYOND it too (no
    # rejection back inside the channel) — must not fire.
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
    band and closes back inside it — the exact geometry
    `_detect_range_reversion` requires for a SELL fade. Shared by the
    on/off `enable_range_reversion` tests below so both exercise the
    identical setup, differing only in the flag."""
    highs, lows = _build_choppy_range(n=90, step=0.5, offset=0.3)
    closes = (highs + lows) / 2.0
    mean_recent = float(np.mean(closes[-lookback:]))
    std_recent = float(np.std(closes[-lookback:], ddof=0))
    upper = mean_recent + band_mult * std_recent
    spike_high = upper + 0.2
    spike_close = upper - 0.05
    # `_frame()` derives close as the bar's own (high + low) / 2 midpoint
    # (matching the fixture convention used throughout this file), so the
    # low must be solved from the intended close, not appended independently
    # — otherwise the rejection close silently lands on the wrong side of
    # the band.
    spike_low = 2.0 * spike_close - spike_high
    highs = np.append(highs, spike_high)
    lows = np.append(lows, spike_low)
    return highs, lows


def test_mode_c_disabled_by_default_even_on_a_qualifying_rejection() -> None:
    """v3: `enable_range_reversion` defaults to `False` — a real XAUUSD
    backtest found Mode C alone at PF 0.37 across every regime bucket
    while Modes A+B together ran PF 4.31 (see module docstring's
    "DISABLED BY DEFAULT" section). The exact same fixture that fires
    below once opted in must produce no Signal on a fresh, unmodified
    instance."""
    strategy = XauusdRegimeRouterM1()
    assert strategy.spec.params["enable_range_reversion"] is False

    lookback = int(strategy.spec.params["range_lookback_bars"])
    band_mult = float(strategy.spec.params["range_band_mult"])
    highs, lows = _build_band_rejection_frame(lookback, band_mult)

    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_mode_c_fires_on_band_rejection_when_ranging_and_explicitly_enabled() -> None:
    # v3: Mode C is off by default (see the disabled-by-default test
    # above) — this test opts in explicitly via
    # `enable_range_reversion=True` so the real `evaluate()` code path for
    # Mode C stays exercised, not just the standalone detector.
    strategy = XauusdRegimeRouterM1()
    strategy.spec.params["enable_range_reversion"] = True
    lookback = int(strategy.spec.params["range_lookback_bars"])
    band_mult = float(strategy.spec.params["range_band_mult"])

    highs, lows = _build_band_rejection_frame(lookback, band_mult)

    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)

    assert isinstance(signal, Signal)
    assert signal.direction is Direction.SELL
    assert signal.pattern == "range_reversion"
    assert signal.sl_points > 0.0
    assert 1.0 <= signal.size_multiplier <= 2.0
    assert strategy._entry_mode == "range_reversion"
    # Bucket key format per the coordinator's instruction:
    # f"range_reversion|{session.value}|{trend.value}" — genuinely
    # `ranging`-tagged (unlike Mode B, which mostly fires while the
    # top-level read is still TRENDING — see module docstring).
    assert "session=overlap trend=ranging" in signal.reason


def test_no_new_entry_while_own_position_is_open() -> None:
    strategy = XauusdRegimeRouterM1()
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
        candles={"M1": frame, "M5": frame, "H1": frame},
        spread_points=15.0,
        own_position=own_position,
    )
    # A qualifying Mode A setup exists in this fixture (see the firing test
    # above), but no entry_mode is tracked yet (no prior signal from this
    # instance) so the exit rule has nothing to act on either — the engine
    # must never see a fresh Signal stacked on an already-open position.
    assert strategy.evaluate(ctx) is None


def test_breakeven_exit_on_mode_a_position_when_regime_flips_to_ranging() -> None:
    strategy = XauusdRegimeRouterM1()
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    trend_frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={"M1": trend_frame, "M5": trend_frame, "H1": trend_frame},
        spread_points=15.0,
    )
    entry_signal = strategy.evaluate(ctx)
    assert isinstance(entry_signal, Signal)
    assert strategy._entry_mode == "trend_continuation"

    # Append a long flat/choppy tail so the top-level regime read flips
    # RANGING (ADX collapses) — same "tight two-bar alternation" construction
    # `test_structure_continuation.py::test_none_when_market_is_choppy_not_trending`
    # uses to force a RANGING read.
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
        candles={"M1": full, "M5": full, "H1": full},
        spread_points=15.0,
        own_position=own_position,
    )
    result = strategy.evaluate(ctx2)
    assert isinstance(result, ExitDecision)
    assert result.action.value == "breakeven"
    assert "TRENDING -> RANGING" in result.reason


def test_empirical_offset_matches_bucket_table() -> None:
    # trending|overlap has the highest secure_rate in the artifact (0.8612)
    # -> a positive offset vs the count-weighted overall rate.
    offset = _empirical_offset(TrendRegime.TRENDING, TradingSession.OVERLAP)
    expected = _logit(BUCKET_TABLE[("trending", "overlap")]) - _logit(OVERALL_SECURE_RATE)
    assert offset == expected
    assert offset > 0.0

    # off_session has no row in the artifact's bucket_table -> no offset.
    assert _empirical_offset(TrendRegime.TRENDING, TradingSession.OFF_SESSION) == 0.0


def test_prior_net_correction_is_inert_while_artifact_has_no_edge() -> None:
    # The artifact's own edge gate failed (holdout AUC at chance) so it
    # shipped literal-zero weights and has_edge=false — the embedded net
    # must therefore contribute nothing today, though it is a real forward
    # pass wired for a future retrain.
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
    """The central "prevent loss" mechanism the whole plan is built on, and
    the one gap none of the tests above cover: drive `AdaptiveLearner`
    through enough RESOLVED trades that a bucket goes `ready=True` with
    negative expectancy, then confirm a subsequent, otherwise-qualifying
    setup in that SAME bucket gets vetoed (`evaluate()` returns `None`, not
    a `Signal`).

    Drives the strategy's own `self._learner` directly via `observe()`/
    `advance()` with synthetic stop-out outcomes — dozens of full candle
    sequences that each organically resolve as a stop-out through
    `evaluate()` would be far less reliable and add nothing `advance()`'s
    own unit tests (`test_online_learning.py`) don't already cover; what's
    missing here is the strategy-level wiring, not the resolution mechanics.

    Bucket = `trend_continuation|overlap|trending` — the exact bucket
    `test_mode_a_fires_on_clean_uptrend_continuation` above lands in (T0 is
    a Monday 13:00 UTC == overlap, and a clean uptrend zigzag reads
    TRENDING), so the fresh setup fed in below is guaranteed to land in the
    same poisoned bucket."""
    strategy = XauusdRegimeRouterM1()
    learner = strategy._ensure_learner(strategy.spec.params)

    bucket = "trend_continuation|overlap|trending"
    features = _regime_features(TradingSession.OVERLAP, TrendRegime.TRENDING, "mid", 13.5, 0)
    entry_price = 100.0
    sl_dist = 2.0
    atr_now = 1.0
    secure_dist = sl_dist * float(strategy.spec.params["learner_secure_r"])

    ns_per_minute = 60_000_000_000
    base_ns = int(pd.Timestamp(T0).value)
    spacing_ns = 10_000 * ns_per_minute  # keeps every sample's own resolution window independent

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
            deadline_ns=entry_ns + 180 * ns_per_minute,
        )
        assert recorded
        resolved = learner.advance(
            times_ns=np.array([entry_ns + ns_per_minute]),
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

    # A fresh, otherwise-qualifying Mode A setup in the SAME bucket — same T0
    # overlap session, same clean-uptrend fixture
    # `test_mode_a_fires_on_clean_uptrend_continuation` above already proves
    # fires — but this strategy instance's learner has now seen 40 straight
    # stop-outs in exactly this regime bucket. `evaluate()` must decline it.
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=2, tail_step=0.3
    )
    frame = _frame(highs, lows)
    ctx = MarketContext(
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    assert strategy.evaluate(ctx) is None


def test_learner_does_not_veto_a_qualifying_setup_after_bucket_shows_positive_expectancy() -> None:
    """Contrasting case: the same mechanism, but every resolved sample in the
    bucket secures instead of stopping out — the bucket should warm up to a
    clearly positive verdict, and the strategy must still take the trade."""
    strategy = XauusdRegimeRouterM1()
    learner = strategy._ensure_learner(strategy.spec.params)

    bucket = "trend_continuation|overlap|trending"
    features = _regime_features(TradingSession.OVERLAP, TrendRegime.TRENDING, "mid", 13.5, 0)
    entry_price = 100.0
    sl_dist = 2.0
    atr_now = 1.0
    secure_dist = sl_dist * float(strategy.spec.params["learner_secure_r"])

    ns_per_minute = 60_000_000_000
    base_ns = int(pd.Timestamp(T0).value)
    spacing_ns = 10_000 * ns_per_minute

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
            deadline_ns=entry_ns + 180 * ns_per_minute,
        )
        resolved = learner.advance(
            times_ns=np.array([entry_ns + ns_per_minute]),
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
        symbol="XAUUSD", candles={"M1": frame, "M5": frame, "H1": frame}, spread_points=15.0
    )
    signal = strategy.evaluate(ctx)
    assert isinstance(signal, Signal)
    assert signal.pattern == "trend_continuation"
