"""Feature contract tests for the entry-TF-agnostic v3 feature module.

Mirrors ``test_smc_dl_features_v2.py``'s coverage (lookahead proof, scale
invariance, contract, warmup, session boundaries, sandbox allowlist, AST
guard) adapted for v3's generic ``htf1/htf2/htf3`` rungs, plus new cases for
the two column groups v3 adds that don't exist in v2: liquidity sweeps
(``sweep_high``/``sweep_low``) and BOS/CHoCH regime state
(``bos_streak``/``bars_since_choch``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.domain.regime import RegimeConfig
from src.strategies.generated.smc_dl_features_v2 import _ASIAN, _LONDON, _NY
from src.strategies.generated.smc_dl_features_v3 import (
    FEATURE_NAMES,
    MIN_ENTRY_BARS,
    _confirmed_pivots,
    _sweep_and_streak_features,
    compute_features_batch,
    compute_features_live,
)

_BARS_PER_DAY = 288  # 24h / 5min


def _series(n: int, start: float = 2000.0, seed: int = 0, freq: str = "5min") -> pd.DataFrame:
    """Deterministic pseudo-market data with real structure (trends, ranges)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, n).cumsum() + np.sin(np.arange(n) / 40.0) * 12.0
    close = start + steps
    high = close + np.abs(rng.normal(0.6, 0.3, n))
    low = close - np.abs(rng.normal(0.6, 0.3, n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    times = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "tick_volume": rng.integers(50, 500, n),
        },
        index=times,
    )


def _resample(base: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample `base` (must carry a `time` column) into a `minutes`-minute
    timeframe the way the broker would. Independent of `base`'s own
    frequency, unlike v2 test's `_htf`, so it works for an M1 or M5 base."""
    rule = f"{minutes}min"
    grouped = base.resample(rule, on="time")
    frame = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "tick_volume": grouped["tick_volume"].sum(),
        }
    ).dropna()
    frame["time"] = frame.index
    return frame


def _append_bar(
    df: pd.DataFrame, open_: float, high: float, low: float, close: float, volume: float = 200.0
) -> pd.DataFrame:
    """Append one hand-crafted bar right after `df`'s last bar, same freq."""
    freq = df.index[1] - df.index[0]
    next_time = df.index[-1] + freq
    new_row = pd.DataFrame(
        {
            "time": [next_time],
            "open": [open_],
            "high": [high],
            "low": [low],
            "close": [close],
            "tick_volume": [volume],
        },
        index=[next_time],
    )
    return pd.concat([df, new_row])


# ---------------------------------------------------------------------------
# lookahead
# ---------------------------------------------------------------------------
def test_features_never_use_future_bars() -> None:
    """Truncation invariance on the entry TF: the row for bar t must not
    change when every bar after t is deleted."""
    entry = _series(500)
    full = compute_features_batch(entry)

    for cut in (300, 380, 460):
        truncated = compute_features_batch(entry.iloc[:cut])
        assert not truncated.empty
        pd.testing.assert_series_equal(
            truncated.iloc[-1],
            full.iloc[cut - 1],
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
        )


def test_htf_features_never_use_an_unclosed_higher_timeframe_bar() -> None:
    """Truncating the HTF rungs as well must not change the entry-TF row —
    same guarantee v2 proves, now over generic htf1/htf2/htf3 rungs."""
    entry = _series(900)
    rung1, rung2, rung3 = _resample(entry, 15), _resample(entry, 60), _resample(entry, 240)
    full = compute_features_batch(entry, (rung1, rung2, rung3))

    cut = 700
    cut_time = entry["time"].iloc[cut - 1]
    truncated = compute_features_batch(
        entry.iloc[:cut],
        (
            rung1[rung1["time"] <= cut_time],
            rung2[rung2["time"] <= cut_time],
            rung3[rung3["time"] <= cut_time],
        ),
    )
    pd.testing.assert_series_equal(
        truncated.iloc[-1], full.iloc[cut - 1], check_names=False, rtol=1e-9, atol=1e-9
    )


def test_future_price_changes_do_not_alter_past_features() -> None:
    """Mutating bars after t must leave the row for bar t untouched."""
    entry = _series(400)
    baseline = compute_features_batch(entry).iloc[250]

    tampered = entry.copy()
    tampered.iloc[260:, tampered.columns.get_loc("high")] *= 1.5
    tampered.iloc[260:, tampered.columns.get_loc("low")] *= 0.5
    tampered.iloc[260:, tampered.columns.get_loc("close")] *= 1.2

    pd.testing.assert_series_equal(
        compute_features_batch(tampered).iloc[250], baseline, check_names=False
    )


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------
def test_columns_are_exactly_feature_names_in_order() -> None:
    frame = compute_features_batch(_series(400))
    assert list(frame.columns) == list(FEATURE_NAMES)


def test_live_vector_equals_last_batch_row() -> None:
    """The live path must be the batch path — this is what stops training and
    serving from drifting apart."""
    entry = _series(500)
    rungs = (_resample(entry, 15), _resample(entry, 60), _resample(entry, 240))
    batch = compute_features_batch(entry, rungs, spread_points=25.0)
    live = compute_features_live(entry, rungs, spread_points=25.0)
    assert live is not None
    np.testing.assert_allclose(live, batch.iloc[-1].to_numpy(dtype=np.float32), rtol=1e-5)


def test_features_are_scale_free() -> None:
    """Gold at 1,900 and gold at 4,000 must produce the same features,
    including the new sweep/streak columns (all are comparisons, counts or
    normalized bar positions — none carry a price unit)."""
    entry = _series(500, start=1900.0)
    scaled = entry.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * (4000.0 / 1900.0)

    base = compute_features_batch(entry).iloc[-1]
    lifted = compute_features_batch(scaled).iloc[-1]
    comparable = [n for n in FEATURE_NAMES if n not in {"atr_rel", "spread_atr"}]
    np.testing.assert_allclose(
        base[comparable].to_numpy(dtype=float),
        lifted[comparable].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_short_history_returns_empty_rather_than_garbage() -> None:
    frame = compute_features_batch(_series(MIN_ENTRY_BARS - 10))
    assert frame.empty
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert compute_features_live(_series(40)) is None


def test_warmup_rows_are_nan_not_zero() -> None:
    frame = compute_features_batch(_series(400))
    assert frame.iloc[: MIN_ENTRY_BARS - 1].isna().all(axis=1).all()
    assert frame.iloc[-1].notna().all()


def test_works_without_a_datetime_index() -> None:
    """The engine hands strategies frames with `time` as a column, not an
    index; training uses a DatetimeIndex. Both must produce identical
    features."""
    entry = _series(400)
    indexed = compute_features_batch(entry)
    columnar = compute_features_batch(entry.reset_index(drop=True))
    pd.testing.assert_frame_equal(indexed.reset_index(drop=True), columnar.reset_index(drop=True))


def test_missing_higher_timeframes_do_not_crash() -> None:
    frame = compute_features_batch(_series(400))
    assert frame.iloc[-1].notna().all()
    assert frame["htf1_trend"].iloc[-1] == 0.0
    assert frame["htf3_trend"].iloc[-1] == 0.0


@pytest.mark.parametrize(
    "bounded", ["pos_in_range_20", "rsi_14", "adx_14", "body_ratio", "bars_since_choch"]
)
def test_bounded_features_stay_bounded(bounded: str) -> None:
    frame = compute_features_batch(_series(600, seed=3)).dropna()
    assert frame[bounded].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# HTF-rung genericization — the point of this module
# ---------------------------------------------------------------------------
def test_htf_rungs_are_positional_not_tf_named() -> None:
    """v3 must not secretly assume rung1/2/3 correspond to particular
    absolute timeframes: swapping which frame is handed in which rung slot
    must simply move that frame's data to the correspondingly-named
    columns — proving the merge is purely positional, and that the same
    module genuinely serves an M1-entry and an M5-entry strategy."""
    m1 = _series(1800, freq="1min", seed=7)
    m5 = _resample(m1, 5)
    m15 = _resample(m1, 15)
    h1 = _resample(m1, 60)

    forward = compute_features_batch(m1, (m5, m15, h1))
    swapped = compute_features_batch(m1, (h1, m15, m5))

    assert list(forward.columns) == list(FEATURE_NAMES)
    assert list(swapped.columns) == list(FEATURE_NAMES)

    # htf1 is fed by M5 in `forward` and by H1 in `swapped` — different
    # underlying data landing in the same-named column purely by position.
    assert not forward["htf1_trend"].dropna().equals(swapped["htf1_trend"].dropna())
    # htf2 (the middle slot) got the same M15 frame both times, so it must
    # be bit-identical between the two calls.
    pd.testing.assert_series_equal(
        forward["htf2_trend"].dropna(), swapped["htf2_trend"].dropna(), check_names=False
    )

    # And an M5-entry variant (M15/H1/H4 as its rungs) runs through the
    # exact same code path — no special-casing by which TF is "entry".
    h4 = _resample(m1, 240)
    m5_entry = compute_features_batch(m5, (m15, h1, h4))
    assert list(m5_entry.columns) == list(FEATURE_NAMES)
    assert m5_entry.iloc[-1].notna().all()


# ---------------------------------------------------------------------------
# new in v3: liquidity sweeps
# ---------------------------------------------------------------------------
def test_sweep_high_fires_on_wick_through_and_reclaim() -> None:
    entry = _series(300)
    pivot_high, _pivot_low = _confirmed_pivots(entry)
    level = pivot_high.iloc[-1]
    assert np.isfinite(level)

    swept = _append_bar(
        entry, open_=level - 0.6, high=level + 1.0, low=level - 1.2, close=level - 0.5
    )
    frame = compute_features_batch(swept)
    last = frame.iloc[-1]
    assert last["sweep_high"] == 1.0
    assert last["bos_up"] == 0.0


def test_sweep_high_does_not_fire_on_a_clean_breakout() -> None:
    entry = _series(300)
    pivot_high, _pivot_low = _confirmed_pivots(entry)
    level = pivot_high.iloc[-1]
    assert np.isfinite(level)

    broke_out = _append_bar(
        entry, open_=level - 0.6, high=level + 1.0, low=level - 1.0, close=level + 0.5
    )
    frame = compute_features_batch(broke_out)
    last = frame.iloc[-1]
    assert last["sweep_high"] == 0.0
    assert last["bos_up"] == 1.0


def test_sweep_low_fires_on_wick_through_and_reclaim() -> None:
    entry = _series(300)
    _pivot_high, pivot_low = _confirmed_pivots(entry)
    level = pivot_low.iloc[-1]
    assert np.isfinite(level)

    swept = _append_bar(
        entry, open_=level + 0.6, high=level + 1.2, low=level - 1.0, close=level + 0.5
    )
    frame = compute_features_batch(swept)
    last = frame.iloc[-1]
    assert last["sweep_low"] == 1.0
    assert last["bos_down"] == 0.0


def test_sweep_low_does_not_fire_on_a_clean_breakdown() -> None:
    entry = _series(300)
    _pivot_high, pivot_low = _confirmed_pivots(entry)
    level = pivot_low.iloc[-1]
    assert np.isfinite(level)

    broke_down = _append_bar(
        entry, open_=level + 0.6, high=level + 1.0, low=level - 1.0, close=level - 0.5
    )
    frame = compute_features_batch(broke_down)
    last = frame.iloc[-1]
    assert last["sweep_low"] == 0.0
    assert last["bos_down"] == 1.0


# ---------------------------------------------------------------------------
# new in v3: BOS streak / bars-since-CHoCH regime state
# ---------------------------------------------------------------------------
def test_bos_streak_and_bars_since_choch_on_a_constructed_event_sequence() -> None:
    """Exercise `_sweep_and_streak_features` directly against a hand-built
    bos/choch event sequence — the precise, deterministic way to prove the
    reset-on-CHoCH and running-count semantics without needing raw OHLC to
    coincidentally produce the exact structural events under test."""
    n = 12
    idx = pd.RangeIndex(n)
    df = pd.DataFrame({"high": np.zeros(n), "low": np.zeros(n), "close": np.zeros(n)}, index=idx)
    pivot_high = pd.Series(np.nan, index=idx)
    pivot_low = pd.Series(np.nan, index=idx)

    # 0,1: two bullish BOS bars (streak builds 1, 2)
    # 2:   bearish CHoCH -> reset to 0
    # 3,4,5: three bearish BOS bars (streak -1, -2, -3)
    # 6:   quiet bar, streak holds at -3
    # 7:   bullish CHoCH -> reset to 0
    # 8:   bullish BOS (streak 1)
    # 9,10,11: quiet, streak holds at 1
    bos_up = pd.Series([1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], index=idx, dtype=float)
    bos_down = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    choch_up = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], index=idx, dtype=float)
    choch_down = pd.Series([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
    structure = pd.DataFrame(
        {"bos_up": bos_up, "bos_down": bos_down, "choch_up": choch_up, "choch_down": choch_down},
        index=idx,
    )

    result = _sweep_and_streak_features(df, pivot_high, pivot_low, structure, lookback=10)

    expected_streak = [1, 2, 0, -1, -2, -3, -3, 0, 1, 1, 1, 1]
    np.testing.assert_allclose(result["bos_streak"].to_numpy(), expected_streak)

    expected_bars_since = [1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(result["bars_since_choch"].to_numpy(), expected_bars_since)

    # Both CHoCH bars themselves read exactly 0 — the reset happens *at* the
    # event, not one bar late.
    assert result["bos_streak"].iloc[2] == 0.0
    assert result["bos_streak"].iloc[7] == 0.0
    assert result["bars_since_choch"].iloc[2] == 0.0
    assert result["bars_since_choch"].iloc[7] == 0.0

    # Monotonic increase within each CHoCH-delimited segment.
    segment_a = result["bars_since_choch"].iloc[2:7].to_numpy()
    segment_b = result["bars_since_choch"].iloc[7:12].to_numpy()
    assert np.all(np.diff(segment_a) >= 0)
    assert np.all(np.diff(segment_b) >= 0)


def test_bos_streak_is_clipped() -> None:
    """A very long unbroken run must clip to [-10, 10], the same way v2
    clips its own unbounded counts."""
    n = 40
    idx = pd.RangeIndex(n)
    df = pd.DataFrame({"high": np.zeros(n), "low": np.zeros(n), "close": np.zeros(n)}, index=idx)
    pivot_high = pd.Series(np.nan, index=idx)
    pivot_low = pd.Series(np.nan, index=idx)
    structure = pd.DataFrame(
        {
            "bos_up": pd.Series(1.0, index=idx),
            "bos_down": pd.Series(0.0, index=idx),
            "choch_up": pd.Series(0.0, index=idx),
            "choch_down": pd.Series(0.0, index=idx),
        },
        index=idx,
    )
    result = _sweep_and_streak_features(df, pivot_high, pivot_low, structure, lookback=10)
    assert result["bos_streak"].iloc[-1] == 10.0


# ---------------------------------------------------------------------------
# session boundaries — reconciled with engine.domain.regime.RegimeConfig,
# unchanged from v2 (v3 delegates session logic to v2's `_time_features`
# entirely rather than redefining it)
# ---------------------------------------------------------------------------
def test_session_boundaries_are_read_from_regime_config() -> None:
    cfg = RegimeConfig()
    assert (cfg.session_asian_start_hour, cfg.session_asian_end_hour) == _ASIAN
    assert (cfg.session_london_start_hour, cfg.session_london_end_hour) == _LONDON
    assert (cfg.session_new_york_start_hour, cfg.session_new_york_end_hour) == _NY


def test_session_columns_reflect_regime_config_boundaries() -> None:
    entry = _series(550)
    frame = compute_features_batch(entry)

    def at(day: int, hour: int) -> pd.Series:
        return frame.iloc[day * _BARS_PER_DAY + hour * 12]

    london_only = at(1, 10)  # 10:00 UTC -> inside London (7-16) only
    assert london_only["sess_london"] == 1.0
    assert london_only["sess_asian"] == 0.0
    assert london_only["sess_ny"] == 0.0

    ny_only = at(1, 18)  # 18:00 UTC -> inside New York (16-21) only
    assert ny_only["sess_ny"] == 1.0
    assert ny_only["sess_london"] == 0.0
    assert ny_only["sess_asian"] == 0.0

    off_session = at(1, 21)  # 21:00 UTC -> NY just ended, Asian not yet started
    assert off_session["sess_london"] == 0.0
    assert off_session["sess_ny"] == 0.0
    assert off_session["sess_asian"] == 0.0

    assert frame["sess_overlap"].dropna().eq(0.0).all()


def test_session_asian_handles_the_midnight_wrap() -> None:
    entry = _series(550)
    frame = compute_features_batch(entry)

    late_evening = frame.iloc[0 * _BARS_PER_DAY + 23 * 12]  # 23:00 UTC, day 1
    assert late_evening["sess_asian"] == 1.0

    early_morning = frame.iloc[1 * _BARS_PER_DAY + 3 * 12]  # 03:00 UTC, day 2
    assert early_morning["sess_asian"] == 1.0

    assert 0.4 < early_morning["minutes_into_session"] < 0.7


# ---------------------------------------------------------------------------
# sandbox regression
# ---------------------------------------------------------------------------
def test_features_v3_is_allowlisted_in_the_sandbox() -> None:
    from src.strategies.sandbox import ALLOWED_IMPORT_MODULES, _static_scan

    assert "src.strategies.generated.smc_dl_features_v3" in ALLOWED_IMPORT_MODULES
    assert (
        _static_scan(
            "from src.strategies.generated.smc_dl_features_v3 import compute_features_live\n"
        )
        == []
    )


# ---------------------------------------------------------------------------
# static guard against the specific defect v2 exists to fix
# ---------------------------------------------------------------------------
def test_no_negative_shift_or_centred_rolling_in_the_feature_source() -> None:
    """Fail if this file ever gains a `shift(-n)` or `rolling(center=True)` —
    see `test_smc_dl_features_v2.py`'s identical guard for the historical
    defect this catches."""
    source_path = Path("src/strategies/generated/smc_dl_features_v3.py")
    source = source_path.read_text()
    tree = ast.parse(source)

    offences: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else None

        if name == "shift":
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                is_negative_literal = (
                    isinstance(argument, ast.UnaryOp)
                    and isinstance(argument.op, ast.USub)
                    and isinstance(argument.operand, ast.Constant)
                )
                if is_negative_literal:
                    offences.append(f"line {node.lineno}: shift(-{argument.operand.value})")

        if name in {"rolling", "expanding"}:
            for keyword in node.keywords:
                if keyword.arg == "center" and getattr(keyword.value, "value", False) is True:
                    offences.append(f"line {node.lineno}: {name}(center=True)")

    assert offences == [], (
        "lookahead constructs found in " + str(source_path) + ": " + "; ".join(offences)
    )
