"""XAUUSD pure market-structure bot — M1.

Trades BOS/CHoCH structure breaks and trend-weakness reversals directly off
swing highs/lows, with no supply/demand zone dependency — see
`src.strategies.domain.structure_continuation`'s module docstring for the
two pure detectors this wraps (`detect_structure_reversal`,
`detect_trend_continuation`). Every other XAUUSD structure bot in this repo
(`xauusd_snd_qm_structure_*`, `xauusd_snd_adaptive_m1`) only fires when
price touches a marked zone; this one trades the structure itself.

DECISION FLOW
────────────────────────────────────────────────────────────────────────
Each bar: try a reversal setup first (this bot's reason for existing — a
trend whose latest impulse leg is measurably weaker, in both size and
volume, than its prior leg, then broken); fall back to a continuation
setup (a healthy trend resuming after a shallow pullback) when no reversal
fires. Continuation additionally requires `tick_volume` on the trigger bar
to be at or above its own rolling mean — real participation behind the
resumption, which `detect_trend_continuation` has no volume awareness of
on its own. Reversal does *not* get this same check: it is already
volume-gated inside `detect_structure_reversal` (the final leg's mean
volume must be measurably *below* the prior leg's), so requiring the
trigger bar to also clear a recent trailing mean would fight that
detector's own logic rather than add to it.

An `AdaptiveLearner` (`src.strategies.domain.online_learning` — pure numpy,
self-labeling from price, no offline training) gates whether a firing
candidate is actually traded, keyed by a `{kind}_{direction}` bucket, and
picks the take-profit via `best_target`'s MFE-survival optimum. There is no
live history for this setup family yet, so `prior_logit=0.0`: a cold
bucket falls back to the un-adapted base rule (fire on every valid
candidate) rather than guessing — the learner's own documented failure
mode. Every candidate is recorded via `observe()` regardless of whether the
gate passes, or a declined family could never be re-admitted once it
started working again.

Anti-spam: `self._observed` dedupes by `(kind, direction, pivot_index)` — a
setup that keeps satisfying its trigger bar after bar (price sitting past
the same broken level) is the same setup, not new evidence, and must not
be graded or re-signaled repeatedly. Mirrors the "fresh touch, not in_zone"
lesson from the zone-bot family, adapted to a bot with no zone to track
freshness against.

SL: pivot price plus an ATR buffer, floored at a minimum ATR multiple. TP:
`learner.best_target`'s R-multiple, off a grid starting at 1.8R — clears
XAUUSD's `min_rr=1.5` `SpreadGate` floor with the same headroom already
established elsewhere in this fleet (a 1.2R floor was silently rejecting
every order).

Sandbox-safe: only `math`, `numpy`, `pandas`, and the already-allowlisted
`src.strategies.domain.models` / `online_learning` / `structure_continuation`.
"""

import math

import numpy as np
import pandas as pd

from src.strategies.domain.models import (
    Direction,
    IndicatorReading,
    MarketContext,
    Signal,
    StrategySpec,
)
from src.strategies.domain.online_learning import AdaptiveLearner, LearnerConfig
from src.strategies.domain.structure_continuation import (
    detect_structure_reversal,
    detect_trend_continuation,
)

_RESET_GAP_NS = 30 * 24 * 60 * 60 * 1_000_000_000  # 30 days, matches xauusd_snd_adaptive_m1

FEATURE_NAMES: tuple[str, ...] = (
    "adx_norm",
    "strength",  # weakness_score for a reversal setup, pullback_depth_atr for a continuation one
    "volume_ratio",
    "ema_spread_atr",
    "range_atr",
    "hour_sin",
    "hour_cos",
    "is_reversal",
)
N_FEATURES = len(FEATURE_NAMES)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


class XauusdStructureShiftM1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="xauusd_structure_shift_m1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            confirmation_timeframes=("M15",),
            htf_veto=True,
            close_on_opposite_signal=True,
            params={
                # ── structure detection (keys match structure_continuation's
                # own `params` lookups exactly, so this dict is handed
                # straight through to both detectors) ──
                "atr_period": 14,
                "pivot_bars": 2,
                "trend_adx_period": 14,
                "trend_adx_threshold": 25.0,
                "trend_ema_fast_span": 12,
                "trend_ema_slow_span": 34,
                "min_pullback_atr": 0.5,
                "max_pullback_atr": 2.5,
                "min_weakness_ratio": 0.7,
                "momentum_confirm_atr_mult": 0.1,
                # ── volume confirmation ──
                "volume_confirm_lookback": 20,
                # ── SL / TP ──
                "sl_zone_buffer_atr_mult": 0.2,
                "sl_min_atr_mult": 0.5,
                # Absolute backstop, in price units, for when ATR itself is
                # briefly too low for the ATR-relative floor above to mean
                # anything (see the comment at its point of use).
                "sl_min_points": 2.0,
                "tp_grid_r": (1.8, 2.2, 2.8, 3.5, 4.5),
                "point_value": 0.01,
                # ── adaptive learner ──
                "learner_enabled": True,
                "learner_secure_r": 0.2,
                "learner_gate_p": 0.65,
                "learner_min_bucket_samples": 25.0,
                "learner_min_model_samples": 150,
                "learner_min_quantile_samples": 30,
                "learner_half_life_samples": 750.0,
                "learner_global_prior_rate": 0.62,
                "learner_max_pending": 400,
                "learner_horizon_bars": 60,
            },
        )
        self._learner: AdaptiveLearner | None = None
        self._last_bar_ns: int | None = None
        self._observed: dict[tuple[str, str, int], int] = {}

    # ── learner lifecycle ────────────────────────────────────────────────

    def _ensure_learner(self, params: dict) -> AdaptiveLearner:
        if self._learner is None:
            self._learner = AdaptiveLearner(
                N_FEATURES,
                LearnerConfig(
                    secure_r=float(params["learner_secure_r"]),
                    min_bucket_samples=float(params["learner_min_bucket_samples"]),
                    min_model_samples=int(params["learner_min_model_samples"]),
                    min_quantile_samples=int(params["learner_min_quantile_samples"]),
                    half_life_samples=float(params["learner_half_life_samples"]),
                    global_prior_rate=float(params["learner_global_prior_rate"]),
                    max_pending=int(params["learner_max_pending"]),
                ),
            )
        return self._learner

    def reset_state(self) -> None:
        """Forget everything learned. Public because a sweep harness
        building one instance per variant needs a guaranteed-clean slate,
        and because `evaluate` calls it itself whenever the candle stream
        shows it is looking at a different replay."""
        if self._learner is not None:
            self._learner.reset()
        self._last_bar_ns = None
        self._observed = {}

    def _check_continuity(self, first_ns: int, last_ns: int) -> None:
        if self._last_bar_ns is None:
            return
        if last_ns < self._last_bar_ns or first_ns > self._last_bar_ns + _RESET_GAP_NS:
            self.reset_state()

    # ── main entry point ────────────────────────────────────────────────

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        params = self.spec.params
        df = ctx.candles.get(self.spec.entry_timeframe)
        if df is None or len(df) < 60 or "time" not in df.columns:
            return None
        if ctx.own_position is not None:
            # One trade at a time — without this guard, evaluate() would
            # re-fire on every bar a broken pivot stays broken.
            return None

        dt_index = pd.DatetimeIndex(df["time"]).as_unit("ns")
        times_ns = dt_index.asi8
        self._check_continuity(int(times_ns[0]), int(times_ns[-1]))

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = (
            df["tick_volume"].to_numpy(dtype=float)
            if "tick_volume" in df.columns
            else np.ones(len(df), dtype=float)
        )

        atr_period = int(params["atr_period"])
        atr_series = _atr(df, atr_period)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            return None
        atr_val = float(atr_val)

        learner_on = bool(params.get("learner_enabled", True))
        learner = self._ensure_learner(params) if learner_on else None
        # Grade whatever the new bars resolved *before* asking about a new
        # setup, so the gate is always a function of already-determined
        # labels only — never a sample still influencing its own decision.
        if learner is not None:
            learner.advance(times_ns, highs, lows)
        self._last_bar_ns = int(times_ns[-1])

        times = list(dt_index)
        kind = "reversal"
        strength = 0.0
        setup = detect_structure_reversal(
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            times=times,
            atr_series=atr_series,
            params=params,
        )
        if setup is not None:
            strength = float(setup.weakness_score)
        else:
            kind = "continuation"
            setup = detect_trend_continuation(
                highs=highs,
                lows=lows,
                closes=closes,
                times=times,
                atr_series=atr_series,
                params=params,
            )
            if setup is not None:
                strength = float(setup.pullback_depth_atr)
        if setup is None:
            return None

        # Volume confirmation — continuation only. A reversal candidate is
        # *defined* by its final leg's volume being below the prior leg's
        # (see detect_structure_reversal's own gate), so requiring the
        # trigger bar to clear a recent trailing mean would fight that
        # detector's own logic. Continuation has no volume awareness of its
        # own, so this is genuinely additive there: the resumption should
        # have real participation behind it, not be a low-volume drift.
        lookback = int(params.get("volume_confirm_lookback", 20))
        if kind == "continuation" and len(volumes) > lookback:
            trailing_mean = float(volumes[-lookback - 1 : -1].mean())
            if trailing_mean > 0.0 and volumes[-1] < trailing_mean:
                return None

        pivot_key = (kind, setup.direction.value, int(setup.pivot_index))
        if pivot_key in self._observed:
            return None

        # ── feature vector ──
        adx_norm = float(min(max(setup.trend_strength / 100.0, 0.0), 1.0))
        vol_window = volumes[-lookback:] if len(volumes) >= lookback else volumes
        vol_mean = float(vol_window.mean())
        volume_ratio = float(volumes[-1] / vol_mean) if vol_mean > 0.0 else 1.0
        ema_fast = _ema(df["close"], int(params["trend_ema_fast_span"])).iloc[-1]
        ema_slow = _ema(df["close"], int(params["trend_ema_slow_span"])).iloc[-1]
        ema_spread_atr = (
            float((ema_fast - ema_slow) / atr_val)
            if pd.notna(ema_fast) and pd.notna(ema_slow)
            else 0.0
        )
        range_atr = float((highs[-1] - lows[-1]) / atr_val)
        bar_time = pd.Timestamp(df["time"].iloc[-1])
        hour_frac = bar_time.hour + bar_time.minute / 60.0
        hour_sin = math.sin(2.0 * math.pi * hour_frac / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour_frac / 24.0)
        is_reversal = 1.0 if kind == "reversal" else 0.0

        features = np.array(
            [
                adx_norm,
                strength,
                volume_ratio,
                ema_spread_atr,
                range_atr,
                hour_sin,
                hour_cos,
                is_reversal,
            ],
            dtype=float,
        )

        # ── SL (needed before scoring: the learner grades against it) ──
        buffer = atr_val * float(params.get("sl_zone_buffer_atr_mult", 0.2))
        close = float(closes[-1])
        if setup.direction is Direction.BUY:
            sl_points = (close - setup.pivot_price) + buffer
        else:
            sl_points = (setup.pivot_price - close) + buffer
        # An ATR-relative floor alone under-protects during a quiet-ATR
        # moment (confirmed live: this bot printed a 0.81-point stop on
        # XAUUSD despite the 0.5x floor, because ATR itself was briefly
        # ~1.6 — well inside normal spread+slippage noise). An absolute
        # floor in price units is the backstop for exactly that case.
        sl_points = max(
            sl_points,
            atr_val * float(params.get("sl_min_atr_mult", 0.5)),
            float(params.get("sl_min_points", 2.0)),
        )
        if sl_points <= 0.0:
            return None

        bucket = f"{kind}_{setup.direction.value}"
        verdict = learner.score(features, bucket, prior_logit=0.0) if learner is not None else None

        self._observed[pivot_key] = int(times_ns[-1])
        limit = int(params.get("learner_max_pending", 400)) * 4
        if len(self._observed) > limit:
            for stale in sorted(self._observed, key=lambda k: self._observed[k])[: limit // 2]:
                del self._observed[stale]

        if learner is not None:
            bar_ns = int(times_ns[-1] - times_ns[-2]) if len(times_ns) > 1 else 60_000_000_000
            horizon_ns = int(params.get("learner_horizon_bars", 60)) * bar_ns
            learner.observe(
                features=features,
                bucket=bucket,
                entry_ns=int(times_ns[-1]),
                entry_price=close,
                direction=1 if setup.direction is Direction.BUY else -1,
                sl_dist=sl_points,
                atr=atr_val,
                deadline_ns=int(times_ns[-1]) + horizon_ns,
            )

        gate_p = float(params.get("learner_gate_p", 0.65))
        # Cold bucket (`not verdict.ready`) falls back to the un-adapted
        # base rule — fire — rather than guessing; only a *confident*
        # verdict below the gate declines the trade.
        if verdict is not None and verdict.ready and verdict.p_secure < gate_p:
            return None

        grid = tuple(float(x) for x in params.get("tp_grid_r", (1.8, 2.2, 2.8, 3.5, 4.5)))
        # A cold bucket's MFE-survival reservoir is empty, so p_reach() falls
        # back to the same constant probability for every r in the grid —
        # expected R is then strictly increasing in r, and best_target()
        # would always hand back the grid's largest value, every time,
        # regardless of setup quality. That is not a learned target, just an
        # artifact of having nothing to learn from yet, so this bucket stays
        # on the same conservative un-adapted default (grid[0]) as the gate
        # above uses until it has real evidence to optimise against.
        min_quantile_samples = (
            learner.config.min_quantile_samples if learner is not None else None
        )
        if (
            learner is not None
            and verdict is not None
            and min_quantile_samples is not None
            and verdict.bucket_samples >= min_quantile_samples
        ):
            r_target, _expected_r, _p_hit = learner.best_target(bucket, grid=grid)
        else:
            r_target = grid[0]

        spread_price = float(ctx.spread_points) * float(params.get("point_value", 0.01))
        risk_price = sl_points + spread_price
        tp_points = risk_price * r_target

        confidence = float(verdict.p_secure) if verdict is not None else 0.5
        pattern = "CHOCH_REV" if kind == "reversal" else "TREND_CONT"
        reason = (
            f"STRUCTURE_SHIFT/{pattern}/{setup.direction.value} "
            f"pivot={setup.pivot_price:.2f} strength={strength:.2f} "
            f"sl={sl_points:.2f} tp={tp_points:.2f} r={r_target:.2f}"
        )
        indicators: tuple[IndicatorReading, ...] = ()
        if verdict is not None:
            indicators = (
                IndicatorReading(
                    name="learner_p_secure",
                    value=round(verdict.p_secure, 4),
                    threshold=gate_p,
                    comparison=">",
                    passed=verdict.p_secure >= gate_p,
                ),
            )

        return Signal(
            direction=setup.direction,
            sl_points=float(sl_points),
            tp_points=float(tp_points),
            confidence=confidence,
            reason=reason,
            pattern=pattern,
            structure=setup.structure_points,
            indicators=indicators,
        )
