"""Scale-free, lookahead-free feature set for the SMC deep-learning model —
entry-timeframe-agnostic, shared by an M1-entry and an M5-entry strategy.

Builds on ``smc_dl_features_v2.py`` (the M5-only predecessor). Everything v2
already validated is reused here by import, not re-derived: ``atr``, ``rsi``,
``adx``, ``_pos_in_range``, ``_confirmed_pivots``, ``_base_features``,
``_structure_features``, ``_bar_times``, ``_time_features``, ``_merge_htf``
and the small ``_div`` guard. v2's own docstring documents why each of those
is lookahead-free and scale-free; re-deriving them here would risk
reintroducing exactly the bugs v2 exists to fix, so this module only adds
what genuinely does not exist in v2 yet (see "WHAT'S NEW", below).

────────────────────────────────────────────────────────────────────────
HTF-RUNG GENERICIZATION
────────────────────────────────────────────────────────────────────────
v2 hardcoded ``candles_m15``/``candles_h1``/``candles_h4`` parameters and
``m15_trend``/``h1_trend``/``h4_trend`` columns, tied to "M5 is the entry
timeframe". That is wrong for an M1-entry strategy, whose natural
confirmation ladder is M5 -> M15 -> H1, not M15 -> H1 -> H4.

This module names the three confirmation slots by **relative rung**
instead: ``htf1_*`` is the timeframe immediately above whatever the caller
is using as entry, ``htf2_*`` the one above that, ``htf3_*`` the one above
that. ``compute_features_batch``/``compute_features_live`` take the entry
frame plus an ordered ``(rung1, rung2, rung3)`` tuple — the caller decides
what "above the entry TF" means for its own strategy:

  * an M1-entry strategy passes ``(M5_df, M15_df, H1_df)``
  * an M5-entry strategy passes ``(M15_df, H1_df, H4_df)``

Both produce the exact same ``htf1_trend``/``htf2_trend``/``htf3_trend``
(etc.) columns, computed by the same merge-as-of-strictly-backward join
(``_merge_htf``, imported unchanged from v2) — this module never assumes
which absolute timeframe "entry" or "rung 1" actually is. ``mtf_align``
stays the mean of the three rung trend signs, unchanged in meaning.

────────────────────────────────────────────────────────────────────────
WHAT'S NEW (the actual "broadening" — columns that do not exist in v2)
────────────────────────────────────────────────────────────────────────
1. **Liquidity sweep / stop-hunt** — ``sweep_high``/``sweep_low``: a bar
   whose wick exceeds the *prior* bar's already-confirmed swing pivot
   (``_confirmed_pivots(...).shift(1)``, so the current bar's own high/low
   cannot have contributed to the level being tested) but whose close ends
   back on the near side of it — a failed breakout / stop run that reverses
   same-bar. Distinct from ``bos_up``/``bos_down`` (v2), which fire when the
   close also clears the level, i.e. a genuine break rather than a sweep.
2. **BOS/CHoCH regime state** — v2 only carries last-bar binary
   ``bos_up``/``bos_down``/``choch_up``/``choch_down`` flags, with no sense
   of how established the current structural regime is. ``bos_streak`` is a
   signed running count of consecutive same-direction BOS events since the
   last CHoCH (positive = bullish regime, negative = bearish), reset to 0
   *at* the CHoCH bar itself and clipped to ``[-10, 10]`` the same way v2
   clips its own unbounded counts. ``bars_since_choch`` is bars elapsed
   since the last CHoCH, normalized 0..1 by the entry-TF lookback window —
   the same "no event yet" -> 1.0 (maximally stale) sentinel convention v2
   already uses for ``demand_age``/``supply_age``/``base_freshness``.

Both new groups are built strictly from primitives v2 already proved
lookahead-free (``_confirmed_pivots``, ``_structure_features``'s
``bos_up``/``bos_down``/``choch_up``/``choch_down``); no new lookahead risk
is introduced. See ``_sweep_and_streak_features`` below.

Everything else — volatility, candle shape, momentum, location, trend/EMA,
volume, S&D zone freshness, struct_trend/BOS/CHoCH/QM, time-of-day
sin/cos + session flags (via ``RegimeConfig``, unchanged), and
``spread_atr`` — carries over from v2 verbatim; none of it was shown to be
the problem (v2's issue was no discriminative power, not these columns).
Per this repo's own prior measurement (a raw per-hour filter: PF 2.70
in-sample -> PF 0.12 out-of-sample), this module still does **not** add an
hour-of-day gate or feature beyond v2's existing ``sin_hour``/``cos_hour``.

This module has **no trained model of its own yet** — that is a later
phase (a training script + ``smc_dl_m1_v3``/``smc_dl_m5_v3`` strategy
files). Nothing here claims or reports a backtest result.

Sandbox: this module is on ``strategies/sandbox.ALLOWED_IMPORT_MODULES`` so
generated strategy code can import it. Pure numpy/pandas — no I/O, no
network, no broker access, no ``torch``.

────────────────────────────────────────────────────────────────────────
ONE CODE PATH FOR TRAIN AND SERVE
────────────────────────────────────────────────────────────────────────
``compute_features_live()`` is *defined as* the last row of
``compute_features_batch()`` — same guarantee v2 relies on, unchanged here.

────────────────────────────────────────────────────────────────────────
LOOKBACK BUDGET
────────────────────────────────────────────────────────────────────────
``trade_loop.DEFAULT_CONTEXT_BARS = 200``: a strategy sees at most 200 bars
per declared timeframe, live and in backtest — and that cap applies
*independently* to every timeframe a strategy declares (each gets its own
200-bar window, never a slice resampled from another TF's window). The
longest window here is ``_MAX_LOOKBACK_ENTRY = 120`` bars on the entry
timeframe and 60 bars on each HTF rung, leaving headroom for ATR/EMA
warmup on top — the same numeric budget v2 used for M5, reused as-is here
because the cap is per-timeframe-bars, not per wall-clock-time, so it fits
equally well whether "entry" is M1 or M5.

────────────────────────────────────────────────────────────────────────
SPREAD
────────────────────────────────────────────────────────────────────────
Unchanged from v2: the ``MarketContext`` candle frames carry only
``time, open, high, low, close, tick_volume`` — no per-bar spread and no
order-book/depth-of-market data anywhere in this system. Spread enters as
a single scalar supplied by the caller: ``ctx.spread_points`` live, the
``candles.spread_points`` column in training. Nothing here fabricates book
depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.generated.smc_dl_features_v2 import (
    _bar_times,
    _base_features,
    _confirmed_pivots,
    _div,
    _merge_htf,
    _pos_in_range,
    _structure_features,
    _time_features,
    adx,
    atr,
    rsi,
)

# Longest trailing window used on the entry timeframe. Kept well inside
# `trade_loop.DEFAULT_CONTEXT_BARS = 200` — see module docstring.
_MAX_LOOKBACK_ENTRY = 120
_MAX_LOOKBACK_HTF = 60

# Bars needed before the feature frame has any usable row at all.
MIN_ENTRY_BARS = _MAX_LOOKBACK_ENTRY + 30

# Canonical feature order. Persisted alongside the weights and verified at
# load time — see module docstring.
FEATURE_NAMES: tuple[str, ...] = (
    # --- volatility (scale-free) ---
    "atr_rel",  # atr14 / close   -> "how volatile is this instrument, in %"
    "atr_ratio_fast",  # atr5 / atr14
    "atr_ratio_slow",  # atr14 / atr50
    "atr_pctile_100",  # rank of atr14 in its own trailing 100 -> volatility regime
    "rv_ratio",  # realized vol 5 / realized vol 20
    "range_ratio",  # bar range / atr14
    # --- candle shape (already ratios) ---
    "body_ratio",
    "wick_upper_ratio",
    "wick_lower_ratio",
    "body_atr",  # |close-open| / atr14 -> displacement
    "close_dir",  # sign of the bar
    # --- momentum / returns in ATR units ---
    "ret_1_atr",
    "ret_3_atr",
    "ret_5_atr",
    "ret_10_atr",
    "ret_20_atr",
    "ret_60_atr",
    # --- location within recent range (bounded 0..1) ---
    "pos_in_range_20",
    "pos_in_range_50",
    "pos_in_range_100",
    # --- trend (distance to EMA in ATR, slope in ATR/bar) ---
    "ema20_dist_atr",
    "ema50_dist_atr",
    "ema100_dist_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "ema_stack",  # +1 fast>mid>slow, -1 inverted, 0 mixed
    "adx_14",  # bounded 0..1 (raw ADX / 100)
    "rsi_14",  # bounded 0..1
    # --- volume, scale-free ---
    "vol_ratio_5_20",
    "vol_ratio_20_100",
    "vol_z_20",
    "vol_spike",  # current / rolling max 20
    # --- S&D structure (RBR/DBD/RBD/DBR bases) ---
    "has_demand",  # an unbroken demand base exists in the window at all
    "has_supply",
    "demand_dist_atr",  # distance below price to nearest unbroken demand base
    "supply_dist_atr",  # distance above price to nearest unbroken supply base
    "demand_age",  # bars since that base formed / lookback  (0..1)
    "supply_age",
    "in_demand",  # price currently inside an unbroken demand base
    "in_supply",
    "base_freshness",  # bars since the most recent base of any kind (0..1)
    # --- market structure / QM ---
    "struct_trend",  # +1 HH/HL, -1 LH/LL, 0 mixed (confirmed pivots only)
    "bos_up",  # close broke the last confirmed swing high
    "bos_down",
    "choch_up",  # BOS against the prevailing structure -> change of character
    "choch_down",
    "qm_bull",  # quasimodo left-shoulder/head/right-shoulder, bullish
    "qm_bear",
    "swing_high_dist_atr",
    "swing_low_dist_atr",
    # --- liquidity sweeps & structure regime state (NEW in v3 — see module
    # docstring "WHAT'S NEW") ---
    "sweep_high",  # wick above the prior confirmed swing high, close back below it
    "sweep_low",  # wick below the prior confirmed swing low, close back above it
    "bos_streak",  # signed run of same-direction BOS since the last CHoCH, [-10, 10]
    "bars_since_choch",  # bars since the last CHoCH / lookback, 0..1 (1.0 = none yet)
    # --- multi-timeframe (merged as-of, strictly backward). Rung 1 = the
    # timeframe immediately above whatever the caller uses as entry, rung 2
    # the one above that, rung 3 the one above that — see module docstring
    # "HTF-RUNG GENERICIZATION" ---
    "htf1_trend",
    "htf1_slope_atr",
    "htf1_pos_in_range",
    "htf1_ret_atr",
    "htf2_trend",
    "htf2_slope_atr",
    "htf2_pos_in_range",
    "htf2_ret_atr",
    "htf3_trend",
    "htf3_slope_atr",
    "htf3_pos_in_range",
    "htf3_ret_atr",
    "mtf_align",  # mean of the three HTF rung trend signs, -1..1
    # --- time of day, as FEATURES not gates ---
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "sess_asian",
    "sess_london",
    "sess_ny",
    "sess_overlap",  # london+ny overlap, the highest-liquidity window
    "minutes_into_session",  # 0..1 through the current session
    # --- cost ---
    "spread_atr",  # spread (price) / atr14 -> cost as a fraction of risk
)

N_FEATURES = len(FEATURE_NAMES)

_HTF_RUNG_PREFIXES: tuple[str, str, str] = ("htf1", "htf2", "htf3")


def _sweep_and_streak_features(
    df: pd.DataFrame,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    structure: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Liquidity-sweep flags and BOS/CHoCH regime-state columns — new in v3.

    Built entirely from primitives v2 already validated as lookahead-free:
    the confirmed-pivot series (``_confirmed_pivots``) and the per-bar
    ``bos_up``/``bos_down``/``choch_up``/``choch_down`` flags
    (``_structure_features``). No new lookahead risk: every value here is a
    function of the current bar and strictly earlier ones.

    ``sweep_high``/``sweep_low`` compare against the *prior* bar's already
    -confirmed pivot (``.shift(1)``) rather than the current bar's own
    confirmed-pivot value, so the current bar's own high/low cannot have
    contributed to the level being tested — a clean "was this level already
    on the map before this bar printed".

    ``bos_streak``: a CHoCH bar's own break does not count toward either the
    old or the new streak — it *is* the reset, so the streak reads exactly
    0 on that bar and only starts accumulating again from the next bar's
    BOS events. Vectorized as a running signed sum within CHoCH-delimited
    regime segments (``(choch_up | choch_down).cumsum()`` groups bars into
    regimes; the CHoCH bar's own delta is zeroed before the running sum so
    the group starts at 0). Clipped to ``[-10, 10]`` the same way v2 clips
    its own unbounded counts (e.g. ``swing_high_dist_atr``).

    ``bars_since_choch``: bars since the last CHoCH, normalized 0..1 by
    ``lookback`` and clipped — same convention ``_base_features`` uses for
    ``demand_age``/``supply_age``/``base_freshness``, including the "no
    CHoCH yet" sentinel of 1.0 (maximally stale) rather than NaN or a bare
    0.0, so "no event yet" isn't confused with "a fresh event right now".
    """
    n = len(df)
    close = df["close"]
    prior_pivot_high = pivot_high.shift(1)
    prior_pivot_low = pivot_low.shift(1)

    sweep_high = ((df["high"] > prior_pivot_high) & (close <= prior_pivot_high)).astype(float)
    sweep_low = ((df["low"] < prior_pivot_low) & (close >= prior_pivot_low)).astype(float)

    bos_up = structure["bos_up"]
    bos_down = structure["bos_down"]
    choch_event = (structure["choch_up"] > 0) | (structure["choch_down"] > 0)

    regime_id = choch_event.cumsum()
    bos_delta = (bos_up - bos_down).where(~choch_event, 0.0)
    bos_streak = bos_delta.groupby(regime_id).cumsum().clip(-10, 10)

    positions = np.arange(n, dtype=float)
    choch_positions = np.where(choch_event.to_numpy(), positions, np.nan)
    last_choch_position = pd.Series(choch_positions, index=df.index).ffill().to_numpy()
    bars_since_raw = positions - last_choch_position
    bars_since_choch = np.nan_to_num(np.clip(bars_since_raw / lookback, 0.0, 1.0), nan=1.0)

    return pd.DataFrame(
        {
            "sweep_high": sweep_high,
            "sweep_low": sweep_low,
            "bos_streak": bos_streak,
            "bars_since_choch": bars_since_choch,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_features_batch(
    candles_entry: pd.DataFrame,
    htf_candles: tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None] = (
        None,
        None,
        None,
    ),
    spread_points: float | pd.Series = 0.0,
    point_value: float = 0.01,
) -> pd.DataFrame:
    """Feature frame for every entry-timeframe bar, columns in
    ``FEATURE_NAMES`` order.

    ``htf_candles`` is an ordered ``(rung1, rung2, rung3)`` tuple: rung 1 is
    the timeframe immediately above whatever the caller is using as entry,
    rung 2 the one above that, rung 3 the one above that (see module
    docstring, "HTF-RUNG GENERICIZATION"). An M1-entry strategy passes
    ``(M5_df, M15_df, H1_df)``; an M5-entry strategy passes
    ``(M15_df, H1_df, H4_df)``. Any rung may be ``None`` — that rung's
    columns come back as 0.0 (see ``_merge_htf``).

    ``spread_points`` may be a scalar (live: ``ctx.spread_points``) or a
    Series aligned to ``candles_entry`` (training: the
    ``candles.spread_points`` column). ``point_value`` converts points to
    price — 0.01 for XAUUSD.

    Rows whose trailing windows have not warmed up yet are returned as NaN
    rather than 0.0; the caller drops them. Filling them with zeros would
    silently teach the net that "no data" is a real, frequently-occurring
    market state (see v2's docstring for the defect this avoids).
    """
    if candles_entry is None or len(candles_entry) < MIN_ENTRY_BARS:
        return pd.DataFrame(columns=list(FEATURE_NAMES))

    df = candles_entry.copy()
    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    volume = pd.to_numeric(df.get("tick_volume", pd.Series(0.0, index=df.index)), errors="coerce")

    atr5 = atr(df, 5)
    atr14 = atr(df, 14)
    atr50 = atr(df, 50)
    close = df["close"]

    out = pd.DataFrame(index=df.index)

    # --- volatility ---
    out["atr_rel"] = _div(atr14, close.abs())
    out["atr_ratio_fast"] = _div(atr5, atr14).clip(0, 5)
    out["atr_ratio_slow"] = _div(atr14, atr50).clip(0, 5)
    out["atr_pctile_100"] = atr14.rolling(100, min_periods=30).rank(pct=True)
    returns = close.pct_change()
    out["rv_ratio"] = _div(
        returns.rolling(5, min_periods=5).std(), returns.rolling(20, min_periods=20).std()
    ).clip(0, 5)
    bar_range = df["high"] - df["low"]
    out["range_ratio"] = _div(bar_range, atr14).clip(0, 10)

    # --- candle shape ---
    body = (close - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    total = bar_range.replace(0.0, np.nan)
    out["body_ratio"] = (body / total).clip(0, 1)
    out["wick_upper_ratio"] = (upper / total).clip(0, 1)
    out["wick_lower_ratio"] = (lower / total).clip(0, 1)
    out["body_atr"] = _div(body, atr14).clip(0, 10)
    out["close_dir"] = np.sign(close - df["open"])

    # --- momentum ---
    for horizon in (1, 3, 5, 10, 20, 60):
        out[f"ret_{horizon}_atr"] = _div(close.diff(horizon), atr14).clip(-15, 15)

    # --- location ---
    out["pos_in_range_20"] = _pos_in_range(df, 20)
    out["pos_in_range_50"] = _pos_in_range(df, 50)
    out["pos_in_range_100"] = _pos_in_range(df, 100)

    # --- trend ---
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema100 = close.ewm(span=100, adjust=False, min_periods=100).mean()
    out["ema20_dist_atr"] = _div(close - ema20, atr14).clip(-10, 10)
    out["ema50_dist_atr"] = _div(close - ema50, atr14).clip(-10, 10)
    out["ema100_dist_atr"] = _div(close - ema100, atr14).clip(-10, 10)
    out["ema20_slope_atr"] = _div(ema20.diff(3) / 3.0, atr14).clip(-3, 3)
    out["ema50_slope_atr"] = _div(ema50.diff(5) / 5.0, atr14).clip(-3, 3)
    stacked_up = (ema20 > ema50) & (ema50 > ema100)
    stacked_down = (ema20 < ema50) & (ema50 < ema100)
    out["ema_stack"] = stacked_up.astype(float) - stacked_down.astype(float)
    out["adx_14"] = adx(df, 14)
    out["rsi_14"] = rsi(close, 14)

    # --- volume ---
    vol_ma5 = volume.rolling(5, min_periods=5).mean()
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    vol_ma100 = volume.rolling(100, min_periods=50).mean()
    out["vol_ratio_5_20"] = _div(vol_ma5, vol_ma20).clip(0, 5)
    out["vol_ratio_20_100"] = _div(vol_ma20, vol_ma100).clip(0, 5)
    out["vol_z_20"] = _div(volume - vol_ma20, volume.rolling(20, min_periods=20).std()).clip(-5, 5)
    out["vol_spike"] = _div(volume, volume.rolling(20, min_periods=20).max()).clip(0, 1)

    # --- S&D + structure ---
    structure = _structure_features(df, atr14)
    out = pd.concat([out, _base_features(df, atr14, _MAX_LOOKBACK_ENTRY), structure], axis=1)

    # --- liquidity sweeps & structure regime state (new in v3). Recomputes
    # the confirmed-pivot series that `_structure_features` also computes
    # internally — a cheap duplicate call, deliberately made so this
    # function stays a pure consumer of `_structure_features`'s output
    # rather than reaching into its internals or forking its logic. ---
    pivot_high, pivot_low = _confirmed_pivots(df, left=3, right=3)
    sweep_streak = _sweep_and_streak_features(
        df, pivot_high, pivot_low, structure, _MAX_LOOKBACK_ENTRY
    )
    out = pd.concat([out, sweep_streak], axis=1)

    # --- MTF, merged strictly backward, generic rungs ---
    entry_times = _bar_times(df)
    for prefix, htf in zip(_HTF_RUNG_PREFIXES, htf_candles, strict=True):
        block = _merge_htf(entry_times, df.index, htf, prefix)
        out = pd.concat([out, block], axis=1)
    out["mtf_align"] = (
        np.sign(out["htf1_trend"]) + np.sign(out["htf2_trend"]) + np.sign(out["htf3_trend"])
    ) / 3.0

    # --- time ---
    out = pd.concat([out, _time_features(df)], axis=1)

    # --- cost ---
    spread_price = (
        pd.Series(spread_points, index=df.index, dtype=float) * point_value
        if np.isscalar(spread_points)
        else pd.Series(np.asarray(spread_points, dtype=float), index=df.index) * point_value
    )
    out["spread_atr"] = _div(spread_price, atr14).clip(0, 5)

    out = out.replace([np.inf, -np.inf], np.nan)
    # The warmup prefix can never be valid; drop it explicitly so a caller
    # that fills NaN doesn't resurrect it.
    out.iloc[:MIN_ENTRY_BARS] = np.nan
    return out.reindex(columns=list(FEATURE_NAMES))


def compute_features_live(
    candles_entry: pd.DataFrame,
    htf_candles: tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None] = (
        None,
        None,
        None,
    ),
    spread_points: float = 0.0,
    point_value: float = 0.01,
) -> np.ndarray | None:
    """Feature vector for the most recent entry-TF bar, in ``FEATURE_NAMES``
    order.

    Defined as the last row of ``compute_features_batch`` so live inference
    and offline training cannot drift apart (see module docstring). Returns
    ``None`` when there is not enough history or the row is not fully warmed
    up — the caller must treat that as "no opinion", never as zeros.

    Unlike v2 (which took a ``candles: dict[str, DataFrame]`` keyed by fixed
    "M5"/"M15"/"H1"/"H4" names), this takes the entry frame and the ordered
    HTF-rung tuple directly — the caller (a strategy file) is the one that
    knows which of its own declared timeframes is "entry" and which are
    "rung 1/2/3", so that mapping belongs there, not in this module.
    """
    frame = compute_features_batch(
        candles_entry,
        htf_candles,
        spread_points=spread_points,
        point_value=point_value,
    )
    if frame.empty:
        return None
    row = frame.iloc[-1]
    values = row.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    return values.astype(np.float32)
