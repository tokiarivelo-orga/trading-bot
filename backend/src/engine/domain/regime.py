"""Market-regime tagging for entry decisions (OBSERVABILITY_PLAN.md Phase 6,
Pass A): "what regime was the market in when this signal fired" — volatility
bucket, trend/range classification, and trading session — attached to every
`SignalDecision` and `TradeRecord` so PF/expectancy can be split per regime
instead of averaged across all of them.

Pure — no I/O, matching this module's hexagonal `domain/` placement.
Volatility classification is **reused** from `engine.domain.volatility`
(`classify_volatility_regime`/`latest_volatility_regime`), not duplicated
here; this module adds trend (ADX) and session on top and combines all three
into one `EntryRegime` snapshot via `compute_entry_regime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd

from src.engine.domain.volatility import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_EXTREME_PERCENTILE,
    DEFAULT_HIGH_PERCENTILE,
    DEFAULT_LOW_PERCENTILE,
    DEFAULT_REGIME_LOOKBACK_BARS,
    VolatilityRegime,
    latest_volatility_regime,
)
from src.engine.domain.zone_detection import atr

DEFAULT_ADX_PERIOD = 14
DEFAULT_ADX_TREND_THRESHOLD = 20.0


class TrendRegime(StrEnum):
    TRENDING = "trending"
    RANGING = "ranging"


class TradingSession(StrEnum):
    ASIAN = "asian"
    LONDON = "london"
    OVERLAP = "overlap"
    NEW_YORK = "new_york"
    OFF_SESSION = "off_session"


@dataclass(frozen=True, kw_only=True)
class RegimeConfig:
    """Mirrors `configs/regime.yaml` (see `shared.config.loaders.
    load_regime_config`). `adx_period`/`adx_trend_threshold` drive
    `classify_trend_regime`/`latest_trend_regime`; the `session_*_hour`
    fields are UTC-hour boundaries consumed by `session_for`; the
    `volatility_*` fields drive the ATR-percentile volatility bucket
    (`engine.domain.volatility.latest_volatility_regime`) — a tag only,
    nothing reads it to gate or size a trade."""

    volatility_atr_period: int = DEFAULT_ATR_PERIOD
    volatility_lookback_bars: int = DEFAULT_REGIME_LOOKBACK_BARS
    volatility_low_percentile: float = DEFAULT_LOW_PERCENTILE
    volatility_high_percentile: float = DEFAULT_HIGH_PERCENTILE
    volatility_extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE
    adx_period: int = DEFAULT_ADX_PERIOD
    adx_trend_threshold: float = DEFAULT_ADX_TREND_THRESHOLD
    session_overlap_start_hour: int = 12
    session_overlap_end_hour: int = 16
    session_london_start_hour: int = 7
    session_london_end_hour: int = 16
    session_new_york_start_hour: int = 16
    session_new_york_end_hour: int = 21
    session_asian_start_hour: int = 22
    session_asian_end_hour: int = 7  # wraps past midnight (22 -> 07 UTC)


def _directional_movement(highs: np.ndarray, lows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wilder's +DM/-DM per bar-to-bar step, length `n - 1` (one fewer than
    the input arrays — there is no directional move on the first bar)."""
    up_move = np.diff(highs)
    down_move = -np.diff(lows)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    return plus_dm, minus_dm


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> pd.Series:
    """Average Directional Index, smoothed with the same **simple
    rolling-mean** `zone_detection.atr()` already uses rather than textbook
    Wilder recursive smoothing — deliberate, for consistency with that one
    existing precedent in this codebase, and it's what makes this
    vectorizable with plain pandas `.rolling().mean()` instead of a
    bar-by-bar Python loop."""
    if len(highs) == 0:
        # `np.diff` of a zero-length array is itself zero-length (same as a
        # one-element array's diff), so the leading-NaN re-alignment below
        # would produce a length-1 series against `atr_values`'s genuine
        # length-0 series — an explicit empty-in/empty-out guard sidesteps
        # that mismatch rather than letting pandas raise on the length check.
        return pd.Series(dtype=float)
    atr_values = atr(highs, lows, closes, period)
    plus_dm, minus_dm = _directional_movement(highs, lows)
    # Re-align to length n: `_directional_movement` has no reading for the
    # first bar (nothing to diff against), matching how `atr_values` itself
    # carries no true-range reading before it either.
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    smoothed_plus = pd.Series(plus_dm).rolling(period, min_periods=period).mean()
    smoothed_minus = pd.Series(minus_dm).rolling(period, min_periods=period).mean()
    # A flat/warm-up window can divide 0/0 (zero ATR, zero DM) — silenced
    # rather than left to warn, since NaN is exactly the right answer there
    # (nothing directional to measure yet) and this is expected, not a bug.
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100 * smoothed_plus / atr_values.to_numpy()
        minus_di = 100 * smoothed_minus / atr_values.to_numpy()
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period, min_periods=period).mean()  # = ADX


def classify_trend_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_trend_threshold: float = DEFAULT_ADX_TREND_THRESHOLD,
) -> pd.Series:
    """Vectorized trend/range classification aligned to the OHLC arrays, for
    backtest use — mirrors `classify_volatility_regime`'s shape.

    Unlike volatility (percentile-ranked against its own trailing history),
    ADX is classified against a **fixed** threshold: it's already a
    normalized 0-100 trend-strength score by construction (not a raw,
    unbounded reading like ATR that only means something relative to its own
    history), so there is nothing to rank it against. Bars with no ADX
    reading yet (warm-up window, needs ~2x `adx_period` bars) default to
    RANGING — "nothing detected" — the same convention volatility uses,
    defaulting to NORMAL rather than raising."""
    adx = _adx(highs, lows, closes, adx_period)
    adx_arr = adx.to_numpy()
    regimes = [
        TrendRegime.TRENDING
        if not np.isnan(v) and v >= adx_trend_threshold
        else TrendRegime.RANGING
        for v in adx_arr
    ]
    return pd.Series(regimes, index=adx.index)


def latest_trend_regime(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    adx_period: int = DEFAULT_ADX_PERIOD,
    adx_trend_threshold: float = DEFAULT_ADX_TREND_THRESHOLD,
) -> tuple[TrendRegime, float]:
    """Convenience "latest regime" read for live use: classifies only the
    most recent bar and also returns its raw ADX value. Returns
    `(TrendRegime.RANGING, nan)` when there isn't enough history yet instead
    of raising, mirroring `latest_volatility_regime`'s "never raise" shape."""
    adx = _adx(highs, lows, closes, adx_period)
    if adx.empty:
        return TrendRegime.RANGING, float("nan")
    current = float(adx.to_numpy()[-1])
    if np.isnan(current):
        return TrendRegime.RANGING, float("nan")
    regime = TrendRegime.TRENDING if current >= adx_trend_threshold else TrendRegime.RANGING
    return regime, current


def session_for(utc_dt: datetime, config: RegimeConfig) -> TradingSession:
    """Which trading session `utc_dt` falls in, purely off its UTC hour.

    Checked in this order: OVERLAP first (it's nested inside both the
    London and New York ranges, so checking London/New York first would
    always shadow it), then London, then New York, then Asian (handling the
    session's wrap past midnight, e.g. 22:00 UTC -> 07:00 UTC), else
    OFF_SESSION for the gap hours the four named sessions don't cover.

    `utc_dt` must already be a UTC-aware (or naive-but-UTC) datetime — this
    function is a pure function of `.hour` and does no timezone conversion
    itself."""
    hour = utc_dt.hour
    if config.session_overlap_start_hour <= hour < config.session_overlap_end_hour:
        return TradingSession.OVERLAP
    if config.session_london_start_hour <= hour < config.session_london_end_hour:
        return TradingSession.LONDON
    if config.session_new_york_start_hour <= hour < config.session_new_york_end_hour:
        return TradingSession.NEW_YORK
    start, end = config.session_asian_start_hour, config.session_asian_end_hour
    wraps = start > end
    in_asian = (hour >= start or hour < end) if wraps else (start <= hour < end)
    if in_asian:
        return TradingSession.ASIAN
    return TradingSession.OFF_SESSION


@dataclass(frozen=True, kw_only=True)
class EntryRegime:
    """The full regime snapshot at one signal's entry moment — the flattened
    form threaded into `SignalDecision`/`TradeRecord`/`PositionOpened`
    (regime_volatility, regime_volatility_percentile, regime_trend,
    regime_adx, regime_session)."""

    volatility: VolatilityRegime
    volatility_percentile: float
    trend: TrendRegime
    adx: float
    session: TradingSession


def compute_entry_regime(
    entry_frame: pd.DataFrame | None,
    *,
    now: datetime,
    regime_config: RegimeConfig,
) -> EntryRegime | None:
    """One-shot regime read for the engine's entry path
    (`engine/application/trade_loop.py`): `None` when `entry_frame` is
    `None` or empty — never fabricate a regime for a bot with no candles.
    Otherwise pulls `high`/`low`/`close` off the bot's own entry-timeframe
    DataFrame and classifies volatility, trend, and session in one call."""
    if entry_frame is None or entry_frame.empty:
        return None
    highs = entry_frame["high"].to_numpy()
    lows = entry_frame["low"].to_numpy()
    closes = entry_frame["close"].to_numpy()

    volatility, volatility_percentile, _atr_value = latest_volatility_regime(
        highs,
        lows,
        closes,
        atr_period=regime_config.volatility_atr_period,
        regime_lookback_bars=regime_config.volatility_lookback_bars,
        low_percentile=regime_config.volatility_low_percentile,
        high_percentile=regime_config.volatility_high_percentile,
        extreme_percentile=regime_config.volatility_extreme_percentile,
    )
    trend, adx_value = latest_trend_regime(
        highs,
        lows,
        closes,
        adx_period=regime_config.adx_period,
        adx_trend_threshold=regime_config.adx_trend_threshold,
    )
    return EntryRegime(
        volatility=volatility,
        volatility_percentile=volatility_percentile,
        trend=trend,
        adx=adx_value,
        session=session_for(now, regime_config),
    )
