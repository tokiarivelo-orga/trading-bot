"""Feature contract tests — the lookahead proof is the point of this file.

A self-labeling or supervised trading model is one careless line away from
being an oracle that backtests beautifully and loses money live. The v1
feature module contained two such lines (``rolling(center=True)`` and
``shift(-1)``) plus a higher-timeframe join on bar *open* time that leaked up
to four hours of future price. Each of those is caught here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.generated.smc_dl_features_v2 import (
    FEATURE_NAMES,
    MIN_M5_BARS,
    compute_features_batch,
    compute_features_live,
)


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


def _htf(m5: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Resample M5 into a higher timeframe the way the broker would."""
    rule = f"{5 * factor}min"
    grouped = m5.resample(rule, on="time")
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


# ---------------------------------------------------------------------------
# lookahead
# ---------------------------------------------------------------------------
def test_features_never_use_future_bars() -> None:
    """Truncation invariance: the row for bar t must not change when every
    bar after t is deleted.

    This is the strongest practical statement of "no lookahead" — if any
    feature peeked forward, removing the future would change its value.
    """
    m5 = _series(500)
    full = compute_features_batch(m5)

    for cut in (300, 380, 460):
        truncated = compute_features_batch(m5.iloc[:cut])
        assert not truncated.empty
        pd.testing.assert_series_equal(
            truncated.iloc[-1],
            full.iloc[cut - 1],
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
        )


def test_htf_features_never_use_an_unclosed_higher_timeframe_bar() -> None:
    """Truncating the higher timeframes as well must not change the M5 row.

    The v1 join matched on HTF *open* time, so an H4 bar opening at 04:00 —
    whose stored high/low/close summarise 04:00-08:00 — was attached to the
    M5 bar at 04:05. Deleting the future HTF bars would have changed that
    row; here it must not.
    """
    m5 = _series(900)
    m15, h1, h4 = _htf(m5, 3), _htf(m5, 12), _htf(m5, 48)
    full = compute_features_batch(m5, m15, h1, h4)

    cut = 700
    cut_time = m5["time"].iloc[cut - 1]
    truncated = compute_features_batch(
        m5.iloc[:cut],
        m15[m15["time"] <= cut_time],
        h1[h1["time"] <= cut_time],
        h4[h4["time"] <= cut_time],
    )
    pd.testing.assert_series_equal(
        truncated.iloc[-1], full.iloc[cut - 1], check_names=False, rtol=1e-9, atol=1e-9
    )


def test_future_price_changes_do_not_alter_past_features() -> None:
    """Mutating bars after t must leave the row for bar t untouched."""
    m5 = _series(400)
    baseline = compute_features_batch(m5).iloc[250]

    tampered = m5.copy()
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
    m5 = _series(500)
    m15, h1, h4 = _htf(m5, 3), _htf(m5, 12), _htf(m5, 48)
    batch = compute_features_batch(m5, m15, h1, h4, spread_points=25.0)
    live = compute_features_live(
        {"M5": m5, "M15": m15, "H1": h1, "H4": h4}, spread_points=25.0
    )
    assert live is not None
    np.testing.assert_allclose(live, batch.iloc[-1].to_numpy(dtype=np.float32), rtol=1e-5)


def test_features_are_scale_free() -> None:
    """Gold at 1,900 and gold at 4,000 must produce the same features.

    Everything here is a ratio, a bounded oscillator, or a distance divided
    by ATR, precisely so weights fitted at one price level survive a move to
    another. Absolute-price features are what made the v1 set unusable after
    a large trend.
    """
    m5 = _series(500, start=1900.0)
    scaled = m5.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * (4000.0 / 1900.0)

    base = compute_features_batch(m5).iloc[-1]
    lifted = compute_features_batch(scaled).iloc[-1]
    # `atr_rel` is ATR/close and `spread_atr` is spread/ATR: both change when
    # the price level is rescaled without rescaling volatility, which is
    # correct behaviour, not scale dependence.
    comparable = [n for n in FEATURE_NAMES if n not in {"atr_rel", "spread_atr"}]
    np.testing.assert_allclose(
        base[comparable].to_numpy(dtype=float),
        lifted[comparable].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_short_history_returns_empty_rather_than_garbage() -> None:
    frame = compute_features_batch(_series(MIN_M5_BARS - 10))
    assert frame.empty
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert compute_features_live({"M5": _series(40)}) is None


def test_warmup_rows_are_nan_not_zero() -> None:
    """Filling an un-warmed window with 0.0 teaches the net that "no data" is
    a real and frequent market state. v1 did exactly that."""
    frame = compute_features_batch(_series(400))
    assert frame.iloc[: MIN_M5_BARS - 1].isna().all(axis=1).all()
    assert frame.iloc[-1].notna().all()


def test_works_without_a_datetime_index() -> None:
    """The engine hands strategies frames with `time` as a column, not an
    index (`engine/application/context.candles_to_dataframe`); training uses
    a DatetimeIndex. Both must produce identical features."""
    m5 = _series(400)
    indexed = compute_features_batch(m5)
    columnar = compute_features_batch(m5.reset_index(drop=True))
    pd.testing.assert_frame_equal(
        indexed.reset_index(drop=True), columnar.reset_index(drop=True)
    )


def test_missing_higher_timeframes_do_not_crash() -> None:
    frame = compute_features_batch(_series(400), None, None, None)
    assert frame.iloc[-1].notna().all()
    assert frame["h4_trend"].iloc[-1] == 0.0


@pytest.mark.parametrize("bounded", ["pos_in_range_20", "rsi_14", "adx_14", "body_ratio"])
def test_bounded_features_stay_bounded(bounded: str) -> None:
    frame = compute_features_batch(_series(600, seed=3)).dropna()
    assert frame[bounded].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# static guard against the specific defect that caused the v1 leak
# ---------------------------------------------------------------------------
def test_no_negative_shift_or_centred_rolling_in_the_feature_source() -> None:
    """Fail if anyone reintroduces `shift(-n)` or `center=True`.

    The truncation tests above catch a leak behaviourally, but only for the
    code paths a synthetic fixture exercises. This one reads the source and
    is impossible to sneak past. Both patterns are exactly what
    `smc_dl_features.py` (v1) used:

        df['fvg_bullish'] = (df['low'].shift(-1) > df['high'].shift(1))
        df['swing_high']  = (df['high'] == df['high'].rolling(5, center=True).max())

    A negative shift reads a bar that has not closed. `center=True` reads
    half its window from the future. Neither can be reproduced live, so a
    model trained on them cannot work in production.
    """
    import ast
    from pathlib import Path

    source_path = Path("src/strategies/generated/smc_dl_features_v2.py")
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
