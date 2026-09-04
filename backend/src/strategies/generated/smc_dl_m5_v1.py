"""SMC Deep Learning strategy — M5 entry on XAUUSD.

Loads a pre-trained PyTorch multi-task model (trained offline by
``backend/scripts/train_smc_dl.py``) and runs inference every M5 bar to
produce a Signal when the model is confident enough.  The model predicts:

- P(TP hit before SL) — "tp_probability"
- Direction (bearish / neutral / bullish) — "direction"
- Risk score — "risk"

Sandbox note: this file imports ``torch`` which must be added to the
sandbox allowlist.  It does NOT perform I/O beyond loading model weights
at construction time (``__init__``).
"""

from pathlib import Path

import numpy as np
import torch

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec
from src.strategies.generated.smc_dl_features import compute_smc_features
from src.strategies.generated.smc_dl_model import SmcMultiTaskNet


# Resolve model artifacts in a way that works both during normal import
# (where __file__ is defined) and inside the sandbox exec() context used by
# validate_and_load (where __file__ is NOT set in the exec namespace).
# When __file__ is available we anchor to it (always correct). When it
# isn't (sandbox path) we walk upward from cwd looking for the first
# ancestor that contains a 'data/ml_models' directory.
def _resolve_models_dir() -> "Path":
    try:
        # Normal import: __file__ = backend/src/strategies/generated/smc_dl_m5_v1.py
        # parents[3]                = backend/
        return Path(__file__).resolve().parents[3] / "data" / "ml_models"
    except NameError:
        # Sandbox exec context: __file__ is not defined — walk up from cwd.
        candidate = Path(".").resolve()
        for _ in range(6):  # never walk more than 6 levels up
            probe = candidate / "data" / "ml_models"
            if probe.is_dir():
                return probe
            candidate = candidate.parent
        # Final fallback: assume cwd is the backend directory.
        return Path(".").resolve() / "data" / "ml_models"


_MODELS_DIR = _resolve_models_dir()


class SmcDlM5V1:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="smc_dl_m5",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            confirmation_timeframes=("M15", "H1", "H4"),
            params={
                "tp_atr_mult": 2.0,
                "sl_atr_mult": 1.0,
                "min_tp_prob": 0.60,
                "min_direction_conf": 0.75,
            },
            htf_veto=False,  # model does its own MTF analysis
        )

        model_path = _MODELS_DIR / "smc_dl_m5.pt"
        scaler_path = _MODELS_DIR / "smc_dl_m5_scaler.npz"

        self._model: SmcMultiTaskNet | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None

        if model_path.exists() and scaler_path.exists():
            scaler_data = np.load(str(scaler_path))
            self._scaler_mean = scaler_data["mean"]
            self._scaler_scale = scaler_data["scale"]

            input_dim = len(self._scaler_mean)
            self._model = SmcMultiTaskNet(input_dim=input_dim, hidden_dim=192)
            self._model.load_state_dict(
                torch.load(str(model_path), weights_only=True, map_location="cpu")
            )
            self._model.eval()

    # ------------------------------------------------------------------
    def evaluate(self, ctx: MarketContext) -> Signal | None:
        if self._model is None:
            return None

        try:
            features_dict = compute_smc_features(ctx.candles, ctx.symbol, lookback=20)
        except Exception:
            return None

        if not features_dict:
            return None

        feature_values = np.array(list(features_dict.values()), dtype=np.float32)
        if len(feature_values) != len(self._scaler_mean):
            return None

        x_scaled = (feature_values - self._scaler_mean) / np.clip(self._scaler_scale, 1e-8, None)
        x_t = torch.FloatTensor(x_scaled).unsqueeze(0)

        with torch.no_grad():
            tp_prob, direction_logits, _risk = self._model(x_t)

        tp_p = tp_prob.item()
        dir_probs = torch.softmax(direction_logits, dim=1).squeeze().numpy()

        # head_dir is a THREE-class softmax: 0=bearish, 1=neutral, 2=bullish.
        bear_prob = float(dir_probs[0])
        neutral_prob = float(dir_probs[1])
        bull_prob = float(dir_probs[2])

        df_m5 = ctx.candles.get("M5")
        if df_m5 is None or len(df_m5) < 15:
            return None

        atr = float((df_m5["high"] - df_m5["low"]).tail(14).mean())
        if atr <= 0:
            return None

        sl_pts = atr * self.spec.params["sl_atr_mult"]
        tp_pts = atr * self.spec.params["tp_atr_mult"]

        # --- gate ----------------------------------------------------------
        # This used to read `if bull_prob > bear_prob and bull_prob > 0.35`,
        # which had two defects that together produced live trade
        # #8731164505: a BUY into unmitigated supply at the top of structure,
        # logged as "DL Buy (tp=0.57, bull=0.39, bear=0.17)".
        #
        #   1. The neutral class was never consulted. On that trade neutral
        #      was 1 - 0.39 - 0.17 = 0.44 — the argmax. The model's actual
        #      answer was "no directional opinion" and the code bought.
        #   2. `min_tp_prob` and `min_direction_conf` were declared in
        #      `spec.params` and never read; the real threshold was the
        #      hardcoded 0.35. That trade breached both declared limits
        #      (tp 0.57 < 0.60, bull 0.39 < 0.75) and was taken anyway, so
        #      tuning those params — including via
        #      `scripts/optimize_dl_thresholds.py` — changed nothing.
        #
        # Both declared thresholds are now the gate, and the directional
        # class must actually win the softmax. With the current weights this
        # makes the bot far more selective; that is the declared
        # configuration finally taking effect, not a new restriction.
        min_tp_prob = float(self.spec.params["min_tp_prob"])
        min_direction_conf = float(self.spec.params["min_direction_conf"])

        if tp_p < min_tp_prob:
            return None
        if neutral_prob >= max(bull_prob, bear_prob):
            return None

        if bull_prob > bear_prob and bull_prob >= min_direction_conf:
            return Signal(
                direction=Direction.BUY,
                sl_points=sl_pts,
                tp_points=tp_pts,
                confidence=tp_p,
                reason=(
                    f"DL Buy (tp={tp_p:.2f}, bull={bull_prob:.2f}, "
                    f"neutral={neutral_prob:.2f}, bear={bear_prob:.2f})"
                ),
            )
        if bear_prob > bull_prob and bear_prob >= min_direction_conf:
            return Signal(
                direction=Direction.SELL,
                sl_points=sl_pts,
                tp_points=tp_pts,
                confidence=tp_p,
                reason=(
                    f"DL Sell (tp={tp_p:.2f}, bull={bull_prob:.2f}, "
                    f"neutral={neutral_prob:.2f}, bear={bear_prob:.2f})"
                ),
            )

        return None
