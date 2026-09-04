"""SMC Deep Learning strategy — M5 Order-Block entry on Step Index 200.

ARCHITECTURE (v2 — OB-first, DL-filter)
────────────────────────────────────────
v1 was purely DL-driven: fire when model says bull/bear >= 0.70.
The problem: the model rarely reaches 0.70 on OB touch bars because the
3-class softmax spreads probability mass over the neutral class even when
the structural setup is clean.  OB touches were systematically missed.

v2 flips the pipeline:

  1. detect_zones proposes unmitigated demand/supply OBs using the same
     RBR/DBD base geometry as smc_dl_features_v2.detect_zones (shared
     code, identical zones between the feature set and the trade logic).
  2. Fresh-touch guard: signal only on the bar price *enters* the zone
     for the first time, not on every bar it sits inside.
  3. Opposing-zone structural veto: never buy into unmitigated supply
     within 1.5x the stop distance overhead, and vice-versa.
  4. DL advisory veto: if the model outputs a *strong negative* opinion
     (it actively disagrees with direction at > model_veto_threshold),
     the setup is skipped.  A neutral/uncertain model output does NOT block
     -- structure is the primary edge, not the model.
  5. SL sits just beyond the zone far edge + ATR noise buffer.
     TP is target_r x sl.  Confidence from model tp_prob when available.

Why the DL model is advisory-only
──────────────────────────────────
The model's directional softmax has a p99 of 0.784 on this instrument
(post Aug 9 retrain).  Using it as an entry gate with threshold 0.70 fires
on ~5% of bars and misses every OB touch where the model is merely neutral.
Using it as a *veto* only -- block if the model is confidently wrong, pass
otherwise -- means the structural setup does the heavy lifting and the model
contributes when it has strong negative conviction.

SANDBOX
────────
Passes strategies/sandbox.validate_and_load.  Imports only allowlisted
modules.  Model weights loaded once in __init__; evaluate() has no I/O.
"""

from pathlib import Path

import numpy as np
import torch

from src.strategies.domain.models import (
    Direction,
    MarketContext,
    PriceZone,
    Signal,
    StrategySpec,
    ZoneKind,
)
from src.strategies.generated.smc_dl_features import compute_smc_features
from src.strategies.generated.smc_dl_features_v2 import atr as _atr_v2
from src.strategies.generated.smc_dl_features_v2 import detect_zones
from src.strategies.generated.smc_dl_model import SmcMultiTaskNet


def _resolve_models_dir() -> "Path":
    try:
        return Path(__file__).resolve().parents[3] / "data" / "ml_models"
    except NameError:
        candidate = Path(".").resolve()
        for _ in range(6):
            probe = candidate / "data" / "ml_models"
            if probe.is_dir():
                return probe
            candidate = candidate.parent
        return Path(".").resolve() / "data" / "ml_models"


_MODELS_DIR = _resolve_models_dir()


class SmcDlM5Step200:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="smc_dl_m5_step200",
            version=1,
            symbols=("Step Index 200",),
            entry_timeframe="M5",
            confirmation_timeframes=("M15", "H1", "H4"),
            params={
                # --- zone geometry ---
                # Stop sits this many ATR beyond the zone's far edge.
                "sl_zone_buffer_atr": 0.20,
                # Reject setup if the resulting SL is wider than this.
                "max_sl_atr": 3.0,
                # A zone tested this many times is no longer fresh.
                "max_zone_touches": 3,
                # A base needs this many bars of age before the return to it
                # is a retest rather than the original impulse still unfolding.
                "min_zone_age_bars": 2,
                # Hard structural veto: reject a buy with unmitigated supply
                # within this multiple of the stop distance overhead.
                "opposing_zone_veto_r": 1.5,
                # --- reward:risk ---
                "target_r": 1.5,
                # --- DL advisory veto ---
                # Block the OB trade only when the model scores the OPPOSITE
                # direction at or above this threshold.  Below it the model is
                # neutral/uncertain and does NOT veto.
                "model_veto_threshold": 0.65,
                # --- cost ---
                "point_value": 0.01,
            },
            htf_veto=False,
        )

        model_path = _MODELS_DIR / "smc_dl_m5_step200.pt"
        scaler_path = _MODELS_DIR / "smc_dl_m5_step200_scaler.npz"

        self._model: SmcMultiTaskNet | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None

        if model_path.exists() and scaler_path.exists():
            try:
                scaler_data = np.load(str(scaler_path))
                self._scaler_mean = scaler_data["mean"]
                self._scaler_scale = np.clip(scaler_data["scale"], 1e-8, None)
                input_dim = len(self._scaler_mean)
                self._model = SmcMultiTaskNet(input_dim=input_dim, hidden_dim=192)
                self._model.load_state_dict(
                    torch.load(str(model_path), weights_only=True, map_location="cpu")
                )
                self._model.eval()
            except Exception:
                self._model = None

    # ------------------------------------------------------------------
    def evaluate(self, ctx: MarketContext) -> Signal | None:
        # Max 1 open position at a time.
        if ctx.own_position is not None:
            return None

        df_m5 = ctx.candles.get("M5")
        if df_m5 is None or len(df_m5) < 60:
            return None

        atr_series = _atr_v2(df_m5, 14)
        atr_value = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if not (atr_value > 0 and np.isfinite(atr_value)):
            return None

        params = self.spec.params
        price = float(df_m5["close"].iloc[-1])
        spread_price = float(ctx.spread_points) * params["point_value"]

        # ── 1. Detect unmitigated OB zones ──────────────────────────────
        zones = detect_zones(df_m5, atr_series)

        # ── 2. Find the freshest qualifying zone price just entered ──────
        candidate = self._select_zone(zones, price, params)
        if candidate is None:
            return None

        is_buy = candidate["kind"] == "demand"

        # ── 3. SL anchored to zone far edge + noise buffer ───────────────
        buffer = atr_value * params["sl_zone_buffer_atr"] + spread_price
        if is_buy:
            stop_price = candidate["price_low"] - buffer
            sl_points = price - stop_price
        else:
            stop_price = candidate["price_high"] + buffer
            sl_points = stop_price - price

        if sl_points <= 0:
            return None

        # Chasing guard: if price has already run far from the zone,
        # the stop this produces is the wide one the retest was supposed to avoid.
        if sl_points > atr_value * params["max_sl_atr"]:
            return None

        # ── 4. Hard structural veto (opposing zone overhead/below) ────────
        if self._veto_opposing_zone(zones, is_buy, price, sl_points, params):
            return None

        tp_points = sl_points * params["target_r"]
        direction = Direction.BUY if is_buy else Direction.SELL

        zone_note = (
            candidate["kind"]
            + " ["
            + _f(candidate["price_low"])
            + ","
            + _f(candidate["price_high"])
            + "] touch="
            + str(candidate["touches"])
            + " age="
            + str(candidate["age_bars"])
        )

        # ── 5. DL advisory veto ───────────────────────────────────────────
        confidence, vetoed, model_note = self._model_opinion(ctx, is_buy, params)
        if vetoed:
            return None

        return Signal(
            direction=direction,
            sl_points=sl_points,
            tp_points=tp_points,
            confidence=confidence,
            reason=("OB retest " + zone_note + " sl=" + _f(sl_points) + " " + model_note),
            zone=PriceZone(
                kind=ZoneKind.DEMAND if is_buy else ZoneKind.SUPPLY,
                price_low=candidate["price_low"],
                price_high=candidate["price_high"],
                time_start=df_m5.index[0],
                time_end=df_m5.index[-1],
                pattern="OB",
            ),
        )

    # ------------------------------------------------------------------
    def _select_zone(self, zones, price, params):
        """Freshest qualifying zone whose price has just freshly entered, or None."""
        best = None
        for zone in zones:
            # Only fire on the bar price first enters the zone.
            if not zone["fresh_touch"]:
                continue
            if zone["touches"] > params["max_zone_touches"]:
                continue
            if zone["age_bars"] < params["min_zone_age_bars"]:
                continue
            if zone["price_high"] <= zone["price_low"]:
                continue
            # Price already blown through — not a retest.
            if zone["kind"] == "demand" and price < zone["price_low"]:
                continue
            if zone["kind"] == "supply" and price > zone["price_high"]:
                continue
            # Take the most recently formed zone.
            if best is None or zone["index"] > best["index"]:
                best = zone
        return best

    def _veto_opposing_zone(self, zones, is_buy, price, sl_points, params):
        """Block a buy with unmitigated supply directly overhead (and vice-versa)."""
        reach = sl_points * params["opposing_zone_veto_r"]
        for zone in zones:
            if is_buy and zone["kind"] == "supply":
                distance = zone["price_low"] - price
                if -sl_points <= distance <= reach:
                    return True
            if (not is_buy) and zone["kind"] == "demand":
                distance = price - zone["price_high"]
                if -sl_points <= distance <= reach:
                    return True
        return False

    def _model_opinion(self, ctx, is_buy, params):
        """Return (confidence, vetoed, note).

        confidence  -- tp_prob from the model, or 0.55 when unavailable
        vetoed      -- True only if the model strongly disagrees with direction
        note        -- human-readable string appended to Signal.reason
        """
        if self._model is None:
            return 0.55, False, "model=unavailable"

        try:
            features_dict = compute_smc_features(ctx.candles, ctx.symbol, lookback=20)
        except Exception:
            return 0.55, False, "model=feature-err"

        if not features_dict or len(features_dict) != len(self._scaler_mean):
            return 0.55, False, "model=dim-mismatch"

        x = np.array(list(features_dict.values()), dtype=np.float32)
        x_scaled = (x - self._scaler_mean) / self._scaler_scale
        x_t = torch.FloatTensor(x_scaled).unsqueeze(0)

        with torch.no_grad():
            tp_prob, direction_logits, _risk = self._model(x_t)

        # head_tp already ends in nn.Sigmoid() (smc_dl_model.SmcMultiTaskNet)
        # so tp_prob is already a 0..1 probability — do not sigmoid it again.
        tp_p = float(tp_prob.item())
        dir_probs = torch.softmax(direction_logits, dim=1).squeeze().numpy()
        bear_p = float(dir_probs[0])
        bull_p = float(dir_probs[2])

        veto_threshold = float(params["model_veto_threshold"])

        # Veto only when the model is *confidently against* direction:
        if is_buy and bear_p >= veto_threshold and bear_p > bull_p:
            return tp_p, True, ("model-veto(bear=" + _f(bear_p) + ">=" + _f(veto_threshold) + ")")
        if (not is_buy) and bull_p >= veto_threshold and bull_p > bear_p:
            return tp_p, True, ("model-veto(bull=" + _f(bull_p) + ">=" + _f(veto_threshold) + ")")

        return (
            tp_p,
            False,
            "model=ok(tp=" + _f(tp_p) + ",bull=" + _f(bull_p) + ",bear=" + _f(bear_p) + ")",
        )


def _f(v) -> str:
    return str(round(float(v), 3))
