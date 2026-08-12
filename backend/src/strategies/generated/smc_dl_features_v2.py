"""Scale-free, lookahead-free feature set for the SMC deep-learning M5 model.

Replaces ``smc_dl_features.py``, which has three defects this module exists
to fix (all three are demonstrated by
``tests/unit/strategies/test_smc_dl_features_v2.py``):

1. **Lookahead.** ``swing_high``/``swing_low`` used
   ``rolling(5, center=True)`` and ``fvg_*`` used ``shift(-1)`` — both read
   bars that had not closed yet at the moment the feature is consumed. A
   model trained on those learns the future and cannot reproduce it live.
2. **Train/serve skew.** Those same forward-looking columns are NaN on the
   last row and get filled with 0.0, so the value the live strategy feeds the
   net for a given bar is *systematically different* from the value the
   trainer saw for that same bar.
3. **Scale dependence.** Raw ``body_size``, ``atr_5``, ``fvg_size`` and
   friends are in price units. Weights fitted while gold traded at 1,900 do
   not transfer to gold at 4,000. Every feature here is a ratio, a z-score,
   a bounded oscillator, or a distance divided by ATR.

Sandbox: this module is on ``strategies/sandbox.ALLOWED_IMPORT_MODULES`` so
generated strategy code can import it. It is pure numpy/pandas — no I/O, no
network, no broker access, no ``torch``.

────────────────────────────────────────────────────────────────────────
ONE CODE PATH FOR TRAIN AND SERVE
────────────────────────────────────────────────────────────────────────
``compute_features_live()`` is *defined as* the last row of
``compute_features_batch()``. It does not re-derive anything. That is the
only structural guarantee that offline training and live inference agree,
and the alternative (two implementations kept in sync by hand) is what
produced defect 2 above.

Both return columns in ``FEATURE_NAMES`` order. The previous version fed the
net ``np.array(list(features_dict.values()))``, i.e. whatever order the dict
happened to have — a silent, total mis-scaling the day a feature is added
anywhere but the end. Here the order is an explicit constant, persisted next
to the weights, and checked at load time.

────────────────────────────────────────────────────────────────────────
LOOKBACK BUDGET
────────────────────────────────────────────────────────────────────────
``trade_loop.DEFAULT_CONTEXT_BARS = 200``: a strategy sees at most 200 bars
per timeframe, live and in backtest. Anything needing more silently never
fires. The longest window here is ``_MAX_LOOKBACK_M5 = 120`` M5 bars and 60
bars on each higher timeframe, leaving headroom for the ATR/EMA warmup on
top.

────────────────────────────────────────────────────────────────────────
SPREAD
────────────────────────────────────────────────────────────────────────
The ``MarketContext`` candle frames carry only ``time, open, high, low,
close, tick_volume`` (see ``engine/application/context.candles_to_dataframe``)
— **there is no per-bar spread and no order-book/depth-of-market data
anywhere in this system** (the ``candles`` table stores exactly one liquidity
proxy, ``tick_volume``, plus a scalar ``spread_points``; there is no DOM
table). Spread therefore enters as a single scalar, supplied by the caller:
``ctx.spread_points`` live, the ``candles.spread_points`` column in training.
Nothing here fabricates book depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Longest trailing window used on the entry timeframe. Kept well inside
# `trade_loop.DEFAULT_CONTEXT_BARS = 200` — see module docstring.
_MAX_LOOKBACK_M5 = 120
_MAX_LOOKBACK_HTF = 60

# Bars needed before the feature frame has any usable row at all.
MIN_M5_BARS = _MAX_LOOKBACK_M5 + 30

_EPS = 1e-9

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
    # --- multi-timeframe (merged as-of, strictly backward) ---
    "m15_trend",
    "m15_slope_atr",
    "m15_pos_in_range",
    "m15_ret_atr",
    "h1_trend",
    "h1_slope_atr",
    "h1_pos_in_range",
    "h1_ret_atr",
    "h4_trend",
    "h4_slope_atr",
    "h4_pos_in_range",
    "h4_ret_atr",
    "mtf_align",  # mean of the three HTF trend signs, -1..1
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

# Session boundaries in UTC hours. Used to build features only — the model
# learns whether an hour matters. A hard per-hour *gate* was measured on this
# data at PF 2.70 in-sample and PF 0.12 out-of-sample, so it must not be one.
_ASIAN = (0, 8)
_LONDON = (7, 16)
_NY = (12, 21)


# ---------------------------------------------------------------------------
# small vectorised primitives
# ---------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR via an EWM — trailing only, never centred."""
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    denom = (
        pd.Series(denominator, index=numerator.index) if np.isscalar(denominator) else denominator
    )
    return (
        numerator
        / denom.replace(0.0, np.nan).abs().clip(lower=_EPS)
        * np.sign(denom.replace(0.0, 1.0)).replace(0.0, 1.0)
    )


def _div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Plain division guarded against a zero/NaN denominator."""
    return numerator / denominator.replace(0.0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (1.0 - 1.0 / (1.0 + rs)).fillna(0.5)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX scaled to 0..1. Trailing EWMs only."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_smooth = true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / tr_smooth
    )
    minus_di = (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / tr_smooth
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return (
        dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().fillna(0.0).clip(0, 1)
    )


def _pos_in_range(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].rolling(window, min_periods=2).max()
    low = df["low"].rolling(window, min_periods=2).min()
    return ((df["close"] - low) / (high - low).replace(0.0, np.nan)).clip(0.0, 1.0)


def _confirmed_pivots(
    df: pd.DataFrame, left: int = 3, right: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Swing highs/lows that are *already confirmed* at the bar they are
    reported on.

    A pivot at bar ``i`` needs ``right`` bars after it to be known. Reporting
    it at ``i`` — which ``rolling(center=True)`` does — is lookahead. Here the
    rolling window is trailing and the result is shifted so that a pivot only
    appears ``right`` bars later, i.e. exactly when a live strategy could have
    known about it.

    Returns ``(pivot_high_price, pivot_low_price)``, forward-filled: at every
    bar, the price of the most recently *confirmed* swing.
    """
    window = left + right + 1
    roll_max = df["high"].rolling(window, min_periods=window).max()
    roll_min = df["low"].rolling(window, min_periods=window).min()
    # The centre of the trailing window sits `right` bars back.
    centre_high = df["high"].shift(right)
    centre_low = df["low"].shift(right)
    is_pivot_high = centre_high >= roll_max
    is_pivot_low = centre_low <= roll_min
    pivot_high = centre_high.where(is_pivot_high)
    pivot_low = centre_low.where(is_pivot_low)
    return pivot_high.ffill(), pivot_low.ffill()


def _zone_candidates(
    df: pd.DataFrame, atr14: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bars at which an RBR/DBD base is *confirmed*, and that base's extent.

    Same leg-in / base / leg-out compression geometry as
    ``engine/domain/zone_detection.py``, re-implemented here in numpy because
    the sandbox forbids generated code from importing ``engine.*`` (that
    module's own docstring documents the same duplication in the other
    direction).

    Shared by ``_base_features`` and ``detect_zones`` on purpose: the zones a
    strategy *trades* and the zone distances the model is *trained on* must
    be the same objects, or ``demand_dist_atr`` describes something the entry
    logic never acts on.

    A base at bar ``i`` is only confirmed by the leg-out at ``i+1``, so it is
    reported from ``i+1`` onward. Reporting it at ``i`` would be lookahead.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_arr = atr14.to_numpy(dtype=float)
    atr_arr = np.where(np.isfinite(atr_arr) & (atr_arr > 0), atr_arr, np.nan)

    body = np.abs(close - open_)
    small_body = body <= 0.5 * atr_arr
    leg_up = (close - open_) >= 0.7 * atr_arr
    leg_down = (open_ - close) >= 0.7 * atr_arr

    n = len(df)
    demand_at = np.zeros(n, dtype=bool)
    supply_at = np.zeros(n, dtype=bool)
    demand_at[1:] = small_body[:-1] & leg_up[1:]
    supply_at[1:] = small_body[:-1] & leg_down[1:]

    base_low = np.concatenate(([np.nan], low[:-1]))
    base_high = np.concatenate(([np.nan], high[:-1]))
    return demand_at, supply_at, base_low, base_high


def detect_zones(
    df: pd.DataFrame, atr14: pd.Series, lookback: int = _MAX_LOOKBACK_M5
) -> list[dict]:
    """Unmitigated supply/demand zones with their retest lifecycle.

    Returned newest-last. Each entry carries:

      ``kind``         "demand" (buy zone) or "supply" (sell zone)
      ``price_low`` / ``price_high``   the base's extent
      ``age_bars``     bars since the base was confirmed
      ``touches``      retest *episodes*, not bars — entering, leaving and
                       returning is two touches; sitting inside for thirty
                       bars is one
      ``in_zone``      price is inside the zone on the latest bar
      ``fresh_touch``  price *entered* the zone on the latest bar

    ``fresh_touch`` is the one that matters for entries. A retest is an
    event, not a state: signalling on every bar price sits inside a zone
    fired on 28.8% of all bars in the equivalent M1 work, versus 5.5% with
    the fresh-touch guard. With live transaction cost at 116% of gross
    profit, paying a round trip per bar of a slow retest is how an edge gets
    spent.

    Broken zones are dropped rather than flipped to breakers: this file's job
    is to supply candidates the model can filter, and a breaker is a
    different setup with a different base rate that would need its own
    evidence before being traded.
    """
    demand_at, supply_at, base_low, base_high = _zone_candidates(df, atr14)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n == 0:
        return []

    zones: list[dict] = []
    first_allowed = max(n - lookback, 0)
    for index in range(first_allowed, n):
        for is_demand, formed in ((True, demand_at[index]), (False, supply_at[index])):
            if not formed:
                continue
            zone_low = base_low[index]
            zone_high = base_high[index]
            if not (np.isfinite(zone_low) and np.isfinite(zone_high)) or zone_high <= zone_low:
                continue

            forward_close = close[index + 1 :]
            if forward_close.size == 0:
                continue
            # Mitigation: a demand zone dies when price *closes* below its
            # low, supply when price closes above its high.
            breached = forward_close < zone_low if is_demand else forward_close > zone_high
            if bool(breached.any()):
                continue

            inside = (low[index + 1 :] <= zone_high) & (high[index + 1 :] >= zone_low)
            if inside.size == 0:
                continue
            entries = np.flatnonzero(inside & ~np.concatenate(([False], inside[:-1])))
            zones.append(
                {
                    "kind": "demand" if is_demand else "supply",
                    "price_low": float(zone_low),
                    "price_high": float(zone_high),
                    "index": int(index),
                    "age_bars": int(n - 1 - index),
                    "touches": int(len(entries)),
                    "in_zone": bool(inside[-1]),
                    "fresh_touch": bool(inside[-1] and (inside.size < 2 or not inside[-2])),
                }
            )
    return zones


def _base_features(df: pd.DataFrame, atr14: pd.Series, lookback: int) -> pd.DataFrame:
    """Nearest unbroken RBR/DBD demand & supply bases, in ATR-relative terms.

    Same leg-in / base / leg-out compression geometry as
    ``engine/domain/zone_detection.py``, re-implemented here in numpy because
    the sandbox forbids generated code from importing ``engine.*`` (that
    module's own docstring documents the same duplication in the other
    direction). Simplified for feature extraction: a base is a bar whose body
    is small relative to ATR, immediately followed by a momentum leg.
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    atr_arr = atr14.to_numpy(dtype=float)
    atr_arr = np.where(np.isfinite(atr_arr) & (atr_arr > 0), atr_arr, np.nan)

    # Same candidates `detect_zones` trades — see `_zone_candidates`.
    demand_at, supply_at, base_low, base_high = _zone_candidates(df, atr14)
    demand_low = np.where(demand_at, base_low, np.nan)
    demand_high = np.where(demand_at, base_high, np.nan)
    supply_low = np.where(supply_at, base_low, np.nan)
    supply_high = np.where(supply_at, base_high, np.nan)

    d_dist = np.full(n, np.nan)
    s_dist = np.full(n, np.nan)
    d_age = np.full(n, np.nan)
    s_age = np.full(n, np.nan)
    in_d = np.zeros(n)
    in_s = np.zeros(n)
    any_age = np.full(n, np.nan)

    # Walk forward keeping a bounded stack of recent, still-unbroken bases.
    demand_stack: list[tuple[int, float, float]] = []
    supply_stack: list[tuple[int, float, float]] = []
    last_base_idx = -1

    for i in range(n):
        if demand_at[i] and np.isfinite(demand_low[i]):
            demand_stack.append((i, float(demand_low[i]), float(demand_high[i])))
            last_base_idx = i
        if supply_at[i] and np.isfinite(supply_high[i]):
            supply_stack.append((i, float(supply_low[i]), float(supply_high[i])))
            last_base_idx = i

        price = close[i]
        # A demand base is broken once price closes below its low; supply once
        # price closes above its high. Drop broken and out-of-window bases.
        demand_stack = [b for b in demand_stack if price > b[1] and i - b[0] <= lookback]
        supply_stack = [b for b in supply_stack if price < b[2] and i - b[0] <= lookback]

        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        if demand_stack:
            idx, blow, bhigh = demand_stack[-1]
            d_dist[i] = (price - bhigh) / a
            d_age[i] = (i - idx) / lookback
            in_d[i] = 1.0 if blow <= price <= bhigh else 0.0
        if supply_stack:
            idx, blow, bhigh = supply_stack[-1]
            s_dist[i] = (blow - price) / a
            s_age[i] = (i - idx) / lookback
            in_s[i] = 1.0 if blow <= price <= bhigh else 0.0
        if last_base_idx >= 0:
            any_age[i] = min((i - last_base_idx) / lookback, 1.0)

    # "No unbroken base in the window" is a real, frequent market state (15%
    # of bars for demand, 19% for supply on XAUUSD M5), not missing data.
    # Encoding it as NaN would drop a third of the dataset; encoding it as a
    # bare 0.0 would teach the net that "no base" means "a base right here".
    # So it gets an explicit presence flag plus a far-away sentinel distance.
    has_demand = np.isfinite(d_dist).astype(float)
    has_supply = np.isfinite(s_dist).astype(float)
    return pd.DataFrame(
        {
            "has_demand": has_demand,
            "has_supply": has_supply,
            "demand_dist_atr": np.clip(np.nan_to_num(d_dist, nan=10.0), -10.0, 10.0),
            "supply_dist_atr": np.clip(np.nan_to_num(s_dist, nan=10.0), -10.0, 10.0),
            "demand_age": np.clip(np.nan_to_num(d_age, nan=1.0), 0.0, 1.0),
            "supply_age": np.clip(np.nan_to_num(s_age, nan=1.0), 0.0, 1.0),
            "in_demand": in_d,
            "in_supply": in_s,
            "base_freshness": np.nan_to_num(any_age, nan=1.0),
        },
        index=df.index,
    )


def _structure_features(df: pd.DataFrame, atr14: pd.Series) -> pd.DataFrame:
    """BOS / CHoCH / Quasimodo off confirmed pivots only.

    ``choch_*`` is the change-of-character the exit model keys on: a break of
    structure *against* the prevailing trend, which is the earliest reading
    that a runner is about to give its profit back.
    """
    pivot_high, pivot_low = _confirmed_pivots(df, left=3, right=3)
    close = df["close"]

    prev_pivot_high = pivot_high.shift(1)
    prev_pivot_low = pivot_low.shift(1)

    higher_high = (pivot_high > prev_pivot_high).astype(float)
    higher_low = (pivot_low > prev_pivot_low).astype(float)
    lower_high = (pivot_high < prev_pivot_high).astype(float)
    lower_low = (pivot_low < prev_pivot_low).astype(float)

    struct_trend = (higher_high + higher_low - lower_high - lower_low) / 2.0
    struct_trend = struct_trend.clip(-1.0, 1.0)
    # Smooth so a single ambiguous pivot pair doesn't flip the regime.
    struct_trend = struct_trend.rolling(5, min_periods=1).mean()

    bos_up = (close > pivot_high).astype(float)
    bos_down = (close < pivot_low).astype(float)
    # CHoCH = break against the previous structural bias.
    prior_trend = struct_trend.shift(1).fillna(0.0)
    choch_up = ((bos_up > 0) & (prior_trend < 0)).astype(float)
    choch_down = ((bos_down > 0) & (prior_trend > 0)).astype(float)

    # Quasimodo: head beyond the left shoulder, then a right shoulder that
    # fails to follow through — approximated on the confirmed pivot series.
    ph1, ph2 = pivot_high.shift(1), pivot_high.shift(2)
    pl1, pl2 = pivot_low.shift(1), pivot_low.shift(2)
    qm_bear = ((pivot_high > ph1) & (ph1 > ph2) & (pivot_low < pl1) & (close < pivot_high)).astype(
        float
    )
    qm_bull = ((pivot_low < pl1) & (pl1 < pl2) & (pivot_high > ph1) & (close > pivot_low)).astype(
        float
    )

    return pd.DataFrame(
        {
            "struct_trend": struct_trend,
            "bos_up": bos_up,
            "bos_down": bos_down,
            "choch_up": choch_up,
            "choch_down": choch_down,
            "qm_bull": qm_bull,
            "qm_bear": qm_bear,
            "swing_high_dist_atr": _div(pivot_high - close, atr14).clip(-10, 10),
            "swing_low_dist_atr": _div(close - pivot_low, atr14).clip(-10, 10),
        },
        index=df.index,
    )


def _htf_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-higher-timeframe trend block, indexed by that timeframe's bar time."""
    out = pd.DataFrame(index=df.index)
    if df.empty or len(df) < 25:
        for suffix in ("trend", "slope_atr", "pos_in_range", "ret_atr"):
            out[f"{prefix}_{suffix}"] = 0.0
        return out
    a = atr(df, 14)
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    out[f"{prefix}_trend"] = _div(df["close"] - ema20, a).clip(-5, 5)
    out[f"{prefix}_slope_atr"] = _div(ema20.diff(3) / 3.0, a).clip(-2, 2)
    out[f"{prefix}_pos_in_range"] = _pos_in_range(df, min(_MAX_LOOKBACK_HTF, 20))
    out[f"{prefix}_ret_atr"] = _div(df["close"].diff(5), a).clip(-8, 8)
    return out


def _bar_times(df: pd.DataFrame) -> pd.DatetimeIndex | None:
    """Bar timestamps, whether they live in the index or a ``time`` column.

    The engine's frames carry ``time`` as a column
    (``context.candles_to_dataframe``); the training loader sets a
    ``DatetimeIndex``. Both must work, and must produce the same features.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "time" in df.columns:
        times = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if times.notna().any():
            return pd.DatetimeIndex(times)
    return None


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    times = _bar_times(df)
    out = pd.DataFrame(index=df.index)
    if times is None:
        for name in (
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
            "sess_asian",
            "sess_london",
            "sess_ny",
            "sess_overlap",
            "minutes_into_session",
        ):
            out[name] = 0.0
        return out

    hour = pd.Series(times.hour, index=df.index).astype(float)
    minute = pd.Series(times.minute, index=df.index).astype(float)
    dow = pd.Series(times.dayofweek, index=df.index).astype(float)
    hour_frac = hour + minute / 60.0

    out["sin_hour"] = np.sin(2 * np.pi * hour_frac / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * hour_frac / 24.0)
    out["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    out["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    asian = ((hour_frac >= _ASIAN[0]) & (hour_frac < _ASIAN[1])).astype(float)
    london = ((hour_frac >= _LONDON[0]) & (hour_frac < _LONDON[1])).astype(float)
    ny = ((hour_frac >= _NY[0]) & (hour_frac < _NY[1])).astype(float)
    out["sess_asian"] = asian
    out["sess_london"] = london
    out["sess_ny"] = ny
    out["sess_overlap"] = london * ny

    session_start = np.where(ny > 0, _NY[0], np.where(london > 0, _LONDON[0], _ASIAN[0]))
    session_len = np.where(
        ny > 0,
        _NY[1] - _NY[0],
        np.where(london > 0, _LONDON[1] - _LONDON[0], _ASIAN[1] - _ASIAN[0]),
    )
    out["minutes_into_session"] = np.clip(
        (hour_frac.to_numpy() - session_start) / np.maximum(session_len, 1.0), 0.0, 1.0
    )
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compute_features_batch(
    candles_m5: pd.DataFrame,
    candles_m15: pd.DataFrame | None = None,
    candles_h1: pd.DataFrame | None = None,
    candles_h4: pd.DataFrame | None = None,
    spread_points: float | pd.Series = 0.0,
    point_value: float = 0.01,
) -> pd.DataFrame:
    """Feature frame for every M5 bar, columns in ``FEATURE_NAMES`` order.

    ``spread_points`` may be a scalar (live: ``ctx.spread_points``) or a
    Series aligned to ``candles_m5`` (training: the ``candles.spread_points``
    column). ``point_value`` converts points to price — 0.01 for XAUUSD.

    Rows whose trailing windows have not warmed up yet are returned as NaN
    rather than 0.0; the caller drops them. Filling them with zeros (what the
    v1 module did) silently teaches the net that "no data" is a real,
    frequently-occurring market state.
    """
    if candles_m5 is None or len(candles_m5) < MIN_M5_BARS:
        return pd.DataFrame(columns=list(FEATURE_NAMES))

    df = candles_m5.copy()
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
    out = pd.concat(
        [out, _base_features(df, atr14, _MAX_LOOKBACK_M5), _structure_features(df, atr14)], axis=1
    )

    # --- MTF, merged strictly backward ---
    m5_times = _bar_times(df)
    for prefix, htf in (("m15", candles_m15), ("h1", candles_h1), ("h4", candles_h4)):
        block = _merge_htf(m5_times, df.index, htf, prefix)
        out = pd.concat([out, block], axis=1)
    out["mtf_align"] = (
        np.sign(out["m15_trend"]) + np.sign(out["h1_trend"]) + np.sign(out["h4_trend"])
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
    out.iloc[:MIN_M5_BARS] = np.nan
    return out.reindex(columns=list(FEATURE_NAMES))


def _merge_htf(
    m5_times: pd.DatetimeIndex | None,
    m5_index: pd.Index,
    htf: pd.DataFrame | None,
    prefix: str,
) -> pd.DataFrame:
    """As-of merge a higher timeframe onto the M5 index, using **closed HTF
    bars only**.

    Matching on the HTF bar's *open* time is the obvious implementation and
    it is a severe lookahead leak. An H4 bar opening at 04:00 is stored in
    the database complete — its high, low and close summarise 04:00-08:00.
    Attaching it to the M5 bar at 04:05 hands the model four hours of future
    price. Measured: with open-time matching this feature set scored OOS avg
    R 0.47 and PF 4.45 against an unconditional base rate of +0.017R, i.e.
    the "model" was mostly reading the answer.

    So the join key is the HTF bar's *close* time (open + one bar duration,
    inferred from the series' own spacing). At M5 bar ``t`` the attached HTF
    bar is the last one that had actually finished by ``t`` — which is also
    what a live strategy can trust, since the forming HTF bar it receives is
    incomplete and cannot be reconstructed from stored candles. Train and
    serve therefore see the same thing, and neither sees the future.
    """
    empty = pd.DataFrame(
        {f"{prefix}_{s}": 0.0 for s in ("trend", "slope_atr", "pos_in_range", "ret_atr")},
        index=m5_index,
    )
    if htf is None or len(htf) < 25 or m5_times is None:
        return empty
    htf_times = _bar_times(htf)
    if htf_times is None:
        return empty

    spacing = pd.Series(htf_times).diff().median()
    if pd.isna(spacing) or spacing <= pd.Timedelta(0):
        return empty
    close_times = htf_times + spacing

    block = _htf_frame(htf.reset_index(drop=True), prefix)
    left = pd.DataFrame({"_t": m5_times})
    right = block.reset_index(drop=True)
    right.insert(0, "_t", close_times)
    right = right.sort_values("_t")
    merged = pd.merge_asof(
        left.sort_values("_t"), right, on="_t", direction="backward", allow_exact_matches=True
    )
    merged.index = m5_index
    return merged.drop(columns=["_t"])


def compute_features_live(
    candles: dict[str, pd.DataFrame],
    spread_points: float = 0.0,
    point_value: float = 0.01,
) -> np.ndarray | None:
    """Feature vector for the most recent M5 bar, in ``FEATURE_NAMES`` order.

    Defined as the last row of ``compute_features_batch`` so live inference
    and offline training cannot drift apart (see module docstring). Returns
    ``None`` when there is not enough history or the row is not fully warmed
    up — the caller must treat that as "no opinion", never as zeros.
    """
    frame = compute_features_batch(
        candles.get("M5"),
        candles.get("M15"),
        candles.get("H1"),
        candles.get("H4"),
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
