"""Scale-free, causal feature engineering for the XAUUSD M5 learned bot.

This module is the **single source of truth** for features: the offline
trainer (`scripts/train_xauusd_dl_m5.py`) and the live strategy
(`generated/xauusd_dl_m5_v1.py`) both call `compute_feature_frame` on the
same code path. Training computes it over the whole history and keeps every
row; serving computes it over the engine's 200-bar window and keeps the last
row. There is no second implementation to drift out of sync, which is the
single most common way a model that validates well goes on to lose money.

Why this exists next to `generated/smc_dl_features.py` rather than replacing
it in place: that module is load-bearing for the two live `smc_dl_*` bots and
has two defects that cannot be fixed without changing their trained weights'
meaning (documented here so the next reader does not "fix" one and silently
break a live bot):

  * **Lookahead.** `fvg_bullish = low.shift(-1) > high.shift(1)` and
    `swing_high = high == high.rolling(5, center=True).max()` both read bars
    that had not happened yet at decision time. `shift(-1)` is one bar of
    the future; `center=True` is two. At serving time those columns are NaN
    for the newest bar and get filled with 0.0, so the model is *also*
    served a different distribution than it was trained on.
  * **Scale dependence.** `atr_5`, `atr_14`, `atr_50`, `body_size`,
    `upper_wick`, `lower_wick`, `ob_size`, `fvg_size` and
    `distance_to_liquidity` are all in raw price units. A weight fitted while
    gold traded at 1,900 means something different at 4,000 — the model
    quietly decays as the instrument reprices, with no error and no alert.

Everything below is therefore built on two rules, both asserted in
`tests/unit/strategies/test_dl_features.py`:

  1. **Causal by construction.** Only `.shift(+k)` and trailing
     `.rolling(...)` (never `center=True`, never a negative shift). Swing
     pivots are published `pivot_lookback` bars *after* the bar they
     describe, because that is when they became knowable. Higher-timeframe
     context is joined on the HTF bar's **close** time, so an H1 bar stamped
     09:00 first reaches an M5 bar at 10:00 — not at 09:05.
  2. **Scale-free by construction.** Every feature is a ratio, a sign, a
     z-score, a bounded oscillator, or a distance divided by ATR. Gold can
     go 1,900 -> 4,000 without invalidating a single fitted weight.

`FEATURE_NAMES` is the contract between trainer and strategy. Order matters:
a trained weight is bound to a *position* in the vector, not to a name. The
saved artifact records the list it was trained with and the strategy refuses
to serve a model whose list does not match, so a reordering here can only
ever disable the bot, never silently mis-serve it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bar counts are in M5 bars unless the name says otherwise.
DEFAULT_ATR_PERIOD = 14
DEFAULT_PIVOT_LOOKBACK = 3
DEFAULT_ZONE_LOOKBACK = 60
# Kept comfortably under `trade_loop.DEFAULT_CONTEXT_BARS = 200`: the longest
# trailing window below is 50 bars, so a 200-bar context leaves ~150 bars of
# warmup and the newest row is always fully populated. Anything approaching
# 200 here would make the live bot silently never fire (see MEMORY: "Engine
# context_bars=200 hard cap").
MAX_WARMUP_BARS = 60

# UTC-hour session boundaries, mirroring `engine.domain.regime.session_for`
# so "london" means the same thing here as in the live journal the priors in
# `generated/xauusd_snd_adaptive_m1_v1.py` were measured from.
_SESSION_BOUNDS = {
    "london": (7, 12),
    "overlap": (12, 16),
    "new_york": (16, 21),
}

FEATURE_NAMES: tuple[str, ...] = (
    # --- volatility state (scale-free: ratios of ATR to price or to itself)
    "atr_pct",
    "atr_ratio_fast",
    "atr_ratio_slow",
    "vol_of_vol",
    # --- momentum, expressed in ATR units so 1.0 means "one typical bar"
    "ret_1_atr",
    "ret_3_atr",
    "ret_6_atr",
    "ret_12_atr",
    "ret_24_atr",
    # --- candle shape (already dimensionless)
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "range_atr",
    "close_pos_in_bar",
    # --- trend / location
    "ema_fast_dist_atr",
    "ema_slow_dist_atr",
    "ema_spread_atr",
    "adx_norm",
    "range_pos_50",
    "dist_high_50_atr",
    "dist_low_50_atr",
    # --- participation (ratios, never raw tick counts)
    "vol_ratio_1_20",
    "vol_ratio_5_20",
    "vol_z",
    # --- cost (the binding constraint: cost was 116% of gross profit live)
    "spread_atr",
    "spread_ratio_100",
    # --- time of day, as features the model may weigh, never as a hard gate
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "sess_asian",
    "sess_london",
    "sess_overlap",
    "sess_new_york",
    # --- market structure (causal pivots)
    "structure_bias",
    "bos_up",
    "bos_dn",
    "choch",
    "swing_range_atr",
    # --- supply & demand proximity
    "dist_demand_atr",
    "dist_supply_atr",
    "demand_height_atr",
    "supply_height_atr",
    "zone_balance",
    # --- higher timeframe context (joined on HTF close time)
    "m15_trend",
    "m15_dist_atr",
    "m15_adx_norm",
    "m15_range_pos",
    "h1_trend",
    "h1_dist_atr",
    "h1_adx_norm",
    "h1_range_pos",
    "h4_trend",
    "h4_dist_atr",
    "h4_adx_norm",
    "h4_range_pos",
    "htf_align",
)

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """Wilder-style ATR via an EMA of true range. `min_periods=period` keeps
    the warmup rows NaN rather than averaging two bars and calling it ATR —
    callers drop those rows instead of training on a fiction."""
    return _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index in [0, 100]. Trailing only."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100.0
        * pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
        / atr_s.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
        / atr_s.replace(0.0, np.nan)
    )
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# causal market structure
# ---------------------------------------------------------------------------


def causal_pivots(
    df: pd.DataFrame, lookback: int = DEFAULT_PIVOT_LOOKBACK
) -> tuple[pd.Series, pd.Series]:
    """Confirmed swing highs/lows, published only once they are knowable.

    A bar `i` is a swing high when its high is the maximum of the window
    `[i - lookback, i + lookback]`. That fact is not established until bar
    `i + lookback` has closed, so the returned series carries the pivot
    *price* at index `i + lookback`, not at `i`. This is the whole difference
    between a legitimate structure feature and the `center=True` lookahead in
    `generated/smc_dl_features.py`.

    Returns `(pivot_high_price, pivot_low_price)`, NaN where no pivot was
    confirmed on that bar.
    """
    window = 2 * lookback + 1
    # `rolling(window)` is trailing, so position `i + lookback` sees exactly
    # the window centred on `i` — the shift below re-labels it to the bar the
    # pivot describes only for price lookup, never for timing.
    roll_max = df["high"].rolling(window, min_periods=window).max()
    roll_min = df["low"].rolling(window, min_periods=window).min()
    centre_high = df["high"].shift(lookback)
    centre_low = df["low"].shift(lookback)
    is_high = centre_high >= roll_max
    is_low = centre_low <= roll_min
    return centre_high.where(is_high), centre_low.where(is_low)


def _structure_features(df: pd.DataFrame, atr_s: pd.Series, lookback: int) -> dict[str, pd.Series]:
    """Higher-high/lower-low bias, break of structure, and change of character
    — all derived from `causal_pivots`, so all lag reality by `lookback` bars
    exactly as a live reader of the chart would."""
    piv_high, piv_low = causal_pivots(df, lookback)

    last_high = piv_high.ffill()
    prev_high = piv_high.ffill().shift(1).where(piv_high.notna()).ffill()
    last_low = piv_low.ffill()
    prev_low = piv_low.ffill().shift(1).where(piv_low.notna()).ffill()

    higher_highs = (last_high > prev_high).astype(float)
    higher_lows = (last_low > prev_low).astype(float)
    # +1 when both highs and lows are rising, -1 when both are falling, 0 mixed.
    bias = higher_highs + higher_lows - 1.0

    close = df["close"]
    # A break of structure is a close beyond the most recent confirmed pivot.
    bos_up = (close > last_high).astype(float)
    bos_dn = (close < last_low).astype(float)

    # Change of character: the first break *against* the prevailing bias —
    # in an up-sequence, a close under the latest higher low. This is the
    # earliest structural evidence that the move justifying an entry has
    # stopped being that move, and it is what the exit head keys on.
    choch = pd.Series(0.0, index=df.index)
    choch = choch.mask((bias > 0) & (bos_dn > 0), -1.0)
    choch = choch.mask((bias < 0) & (bos_up > 0), 1.0)

    swing_range = _safe_div(last_high - last_low, atr_s)

    return {
        "structure_bias": bias,
        "bos_up": bos_up,
        "bos_dn": bos_dn,
        "choch": choch,
        "swing_range_atr": swing_range.clip(0.0, 20.0),
    }


def _zone_features(
    df: pd.DataFrame, atr_s: pd.Series, lookback: int, zone_lookback: int
) -> dict[str, pd.Series]:
    """Proximity to the nearest demand/supply shelf, in ATR units.

    A deliberately compact stand-in for the full RBR/DBD/RBD/DBR detector in
    `engine/domain/zone_detection.py`: that module belongs to another module's
    domain (CLAUDE.md forbids reaching into it from `strategies/`) and is not
    on the sandbox allowlist. What the model actually needs from a zone is
    "how far is price from the last shelf, and how thick is it" — both of
    which the confirmed pivots already carry.
    """
    piv_high, piv_low = causal_pivots(df, lookback)
    close = df["close"]

    # Nearest confirmed pivot low below / pivot high above, within the window.
    demand = piv_low.ffill(limit=zone_lookback)
    supply = piv_high.ffill(limit=zone_lookback)

    dist_demand = _safe_div(close - demand, atr_s)
    dist_supply = _safe_div(supply - close, atr_s)

    # Shelf thickness: the bar range at the pivot, carried forward with it.
    bar_range = df["high"] - df["low"]
    demand_height = _safe_div(bar_range.where(piv_low.notna()).ffill(limit=zone_lookback), atr_s)
    supply_height = _safe_div(bar_range.where(piv_high.notna()).ffill(limit=zone_lookback), atr_s)

    # Where price sits between the two shelves: -1 hugging demand, +1 hugging
    # supply. Bounded and scale-free by construction.
    span = (supply - demand).replace(0.0, np.nan)
    balance = (2.0 * (close - demand) / span - 1.0).clip(-2.0, 2.0)

    return {
        "dist_demand_atr": dist_demand.clip(-20.0, 20.0),
        "dist_supply_atr": dist_supply.clip(-20.0, 20.0),
        "demand_height_atr": demand_height.clip(0.0, 20.0),
        "supply_height_atr": supply_height.clip(0.0, 20.0),
        "zone_balance": balance,
    }


# ---------------------------------------------------------------------------
# higher timeframe context
# ---------------------------------------------------------------------------

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _htf_frame(df_htf: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Per-bar HTF context stamped with the bar's **close** time.

    `candles.time` is the bar's OPEN time. An H1 bar stamped 09:00 is not
    finished — and its close/ATR/ADX are not knowable — until 10:00. Stamping
    it at 09:00 and joining backward would leak up to 59 minutes of future
    into every M5 row between 09:05 and 09:55. Shifting the stamp to the
    close time is what makes the join honest.
    """
    minutes = _TF_MINUTES.get(timeframe, 60)
    out = pd.DataFrame(index=df_htf.index)
    atr_htf = atr(df_htf)
    close = df_htf["close"]
    ema_fast = _ema(close, 20)
    ema_slow = _ema(close, 50)
    roll_max = df_htf["high"].rolling(50, min_periods=10).max()
    roll_min = df_htf["low"].rolling(50, min_periods=10).min()

    out["trend"] = np.sign(ema_fast - ema_slow)
    out["dist_atr"] = _safe_div(close - ema_fast, atr_htf).clip(-10.0, 10.0)
    out["adx_norm"] = (_adx(df_htf) / 100.0).clip(0.0, 1.0)
    out["range_pos"] = ((close - roll_min) / (roll_max - roll_min).replace(0.0, np.nan)).clip(
        0.0, 1.0
    )
    out["close_time"] = df_htf["time"].astype("int64") + minutes * 60
    return out


def _join_htf(
    base_time: pd.Series, df_htf: pd.DataFrame | None, timeframe: str, prefix: str
) -> dict[str, pd.Series]:
    """Backward as-of join of HTF context onto the M5 clock.

    Missing or too-short HTF data yields neutral zeros rather than an
    exception: a live context window occasionally arrives with one timeframe
    short, and the bot degrading to "no HTF opinion" is strictly better than
    it crashing inside the trade loop.
    """
    names = ("trend", "dist_atr", "adx_norm", "range_pos")
    neutral = {f"{prefix}_{n}": pd.Series(0.0, index=base_time.index) for n in names}
    if df_htf is None or df_htf.empty or len(df_htf) < 55:
        return neutral

    htf = _htf_frame(df_htf, timeframe).dropna(subset=["close_time"])
    left = pd.DataFrame({"t": base_time.astype("int64")}).reset_index()
    right = htf.sort_values("close_time").reset_index(drop=True)
    merged = pd.merge_asof(
        left.sort_values("t"),
        right.rename(columns={"close_time": "t"}),
        on="t",
        direction="backward",
    ).set_index("index")

    out: dict[str, pd.Series] = {}
    for name in names:
        series = merged[name].reindex(base_time.index)
        out[f"{prefix}_{name}"] = series.astype(float)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def compute_feature_frame(
    candles: dict[str, pd.DataFrame],
    *,
    point: float = 0.01,
    atr_period: int = DEFAULT_ATR_PERIOD,
    pivot_lookback: int = DEFAULT_PIVOT_LOOKBACK,
    zone_lookback: int = DEFAULT_ZONE_LOOKBACK,
    spread_points: float | None = None,
) -> pd.DataFrame:
    """Feature matrix, one row per M5 bar, columns exactly `FEATURE_NAMES`.

    `candles` maps timeframe -> OHLCV frame with a `time` column in epoch
    seconds (the shape `MarketContext.candles` already has). Only `M5` is
    required; `M15`/`H1`/`H4` contribute context when present and degrade to
    neutral zeros when absent.

    `spread_points` overrides the per-bar `spread_points` column — the live
    context carries the *current* spread separately from the candle history,
    and using it is what keeps the cost features honest at decision time.

    Rows whose warmup is incomplete are returned as NaN rather than filled;
    the trainer drops them and the strategy checks the last row for NaN. A
    zero-filled warmup row is indistinguishable from a genuine zero reading,
    which is how a model ends up trained on a few thousand rows of fiction.
    """
    df_m5 = candles.get("M5")
    if df_m5 is None or len(df_m5) < MAX_WARMUP_BARS:
        return pd.DataFrame(columns=list(FEATURE_NAMES))

    df = df_m5.reset_index(drop=True).copy()
    if "time" not in df.columns:
        # The sandbox smoke test builds frames without a `time` column. A
        # synthetic monotonic clock keeps the time-of-day features defined
        # (they land on a single arbitrary hour) so `evaluate()` returns
        # cleanly instead of raising during validation.
        df["time"] = np.arange(len(df), dtype="int64") * 300

    close = df["close"]
    atr_s = atr(df, atr_period)
    atr_safe = atr_s.replace(0.0, np.nan)
    out: dict[str, pd.Series] = {}

    # --- volatility -------------------------------------------------------
    atr_fast = atr(df, 5)
    atr_slow = _true_range(df).rolling(50, min_periods=50).mean()
    out["atr_pct"] = _safe_div(atr_s, close)
    out["atr_ratio_fast"] = _safe_div(atr_fast, atr_safe).clip(0.0, 10.0)
    out["atr_ratio_slow"] = _safe_div(atr_s, atr_slow).clip(0.0, 10.0)
    out["vol_of_vol"] = _safe_div(
        atr_s.rolling(20, min_periods=20).std(), atr_s.rolling(20, min_periods=20).mean()
    ).clip(0.0, 5.0)

    # --- momentum in ATR units -------------------------------------------
    for horizon in (1, 3, 6, 12, 24):
        out[f"ret_{horizon}_atr"] = _safe_div(close - close.shift(horizon), atr_safe).clip(
            -20.0, 20.0
        )

    # --- candle shape -----------------------------------------------------
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body = (close - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    out["body_ratio"] = (body / bar_range).clip(0.0, 1.0)
    out["upper_wick_ratio"] = (upper / bar_range).clip(0.0, 1.0)
    out["lower_wick_ratio"] = (lower / bar_range).clip(0.0, 1.0)
    out["range_atr"] = _safe_div(df["high"] - df["low"], atr_safe).clip(0.0, 20.0)
    out["close_pos_in_bar"] = ((close - df["low"]) / bar_range).clip(0.0, 1.0)

    # --- trend / location -------------------------------------------------
    ema_fast = _ema(close, 20)
    ema_slow = _ema(close, 50)
    out["ema_fast_dist_atr"] = _safe_div(close - ema_fast, atr_safe).clip(-20.0, 20.0)
    out["ema_slow_dist_atr"] = _safe_div(close - ema_slow, atr_safe).clip(-20.0, 20.0)
    out["ema_spread_atr"] = _safe_div(ema_fast - ema_slow, atr_safe).clip(-20.0, 20.0)
    out["adx_norm"] = (_adx(df) / 100.0).clip(0.0, 1.0)
    roll_max = df["high"].rolling(50, min_periods=50).max()
    roll_min = df["low"].rolling(50, min_periods=50).min()
    out["range_pos_50"] = ((close - roll_min) / (roll_max - roll_min).replace(0.0, np.nan)).clip(
        0.0, 1.0
    )
    out["dist_high_50_atr"] = _safe_div(roll_max - close, atr_safe).clip(0.0, 20.0)
    out["dist_low_50_atr"] = _safe_div(close - roll_min, atr_safe).clip(0.0, 20.0)

    # --- participation ----------------------------------------------------
    volume = (
        df["tick_volume"].astype(float)
        if "tick_volume" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    vol_mean_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std()
    out["vol_ratio_1_20"] = _safe_div(volume, vol_mean_20).clip(0.0, 10.0)
    out["vol_ratio_5_20"] = _safe_div(volume.rolling(5, min_periods=5).mean(), vol_mean_20).clip(
        0.0, 10.0
    )
    out["vol_z"] = _safe_div(volume - vol_mean_20, vol_std_20).clip(-5.0, 5.0)

    # --- cost -------------------------------------------------------------
    if spread_points is not None:
        spread_series = pd.Series(float(spread_points), index=df.index)
    elif "spread_points" in df.columns:
        spread_series = df["spread_points"].astype(float)
    else:
        spread_series = pd.Series(np.nan, index=df.index)
    spread_price = spread_series * point
    out["spread_atr"] = _safe_div(spread_price, atr_safe).clip(0.0, 10.0)
    out["spread_ratio_100"] = _safe_div(
        spread_series, spread_series.rolling(100, min_periods=20).median()
    ).clip(0.0, 20.0)

    # --- time of day (features, never gates) ------------------------------
    stamps = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
    hour = stamps.dt.hour + stamps.dt.minute / 60.0
    dow = stamps.dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    hour_int = stamps.dt.hour
    for name, (lo, hi) in _SESSION_BOUNDS.items():
        out[f"sess_{name}"] = ((hour_int >= lo) & (hour_int < hi)).astype(float)
    out["sess_asian"] = ((hour_int >= 22) | (hour_int < 7)).astype(float)

    # --- structure & zones ------------------------------------------------
    out.update(_structure_features(df, atr_safe, pivot_lookback))
    out.update(_zone_features(df, atr_safe, pivot_lookback, zone_lookback))

    # --- higher timeframe -------------------------------------------------
    for timeframe, prefix in (("M15", "m15"), ("H1", "h1"), ("H4", "h4")):
        out.update(_join_htf(df["time"], candles.get(timeframe), timeframe, prefix))

    frame = pd.DataFrame(out, index=df.index)
    frame["htf_align"] = (
        frame["m15_trend"].fillna(0.0)
        + frame["h1_trend"].fillna(0.0)
        + frame["h4_trend"].fillna(0.0)
    ) / 3.0

    # Reindex to the declared contract so a column added above but not listed
    # in FEATURE_NAMES can never silently shift every downstream weight.
    frame = frame.reindex(columns=list(FEATURE_NAMES))
    return frame.replace([np.inf, -np.inf], np.nan)


def latest_feature_vector(candles: dict[str, pd.DataFrame], **kwargs: object) -> np.ndarray | None:
    """The newest fully-formed feature row, or `None` when warmup is
    incomplete or any feature is undefined.

    `None` rather than a zero-filled vector on purpose: serving a model a
    row of zeros it never saw in training produces a confident, meaningless
    probability. Declining to trade is the correct behaviour.
    """
    frame = compute_feature_frame(candles, **kwargs)  # type: ignore[arg-type]
    if frame.empty:
        return None
    row = frame.iloc[-1]
    if row.isna().any():
        return None
    return row.to_numpy(dtype=np.float64)
