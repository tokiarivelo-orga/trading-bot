"""SMC deep-learning M5 strategy, v3 — same expected-R gating as
``smc_dl_m5_v2.py``, entry-timeframe-agnostic feature set, and (unlike the
M1 sibling) a model that clears its own out-of-sample bar.

WHAT THIS IS
────────────────────────────────────────────────────────────────────────
The same zone-retest-then-model-gate pipeline as ``smc_dl_m5_v2.py``, built
on ``smc_dl_features_v3`` (the entry-TF-agnostic broadened feature set —
liquidity sweeps, BOS/CHoCH regime state, and named-by-rung HTF columns
instead of ``smc_dl_features_v2``'s hardcoded M15/H1/H4). Confirmation
rungs stay ``M15``, ``H1``, ``H4`` — the same ladder ``smc_dl_m5_v2`` uses,
since M5 is still the entry timeframe here.

HONEST STATUS OF THE MODEL
────────────────────────────────────────────────────────────────────────
Walk-forward on 108,959 M5 XAUUSD bars (2025-02-17 to 2026-09-01), 5
chronological folds: in-sample avg R 0.1719, **out-of-sample avg R
+0.0096**, 4 of 5 folds positive (the worst fold: -0.027). That clears
``_model_has_oos_edge``, so this file loads with ``_model_trusted = True``
and the strict gate (``min_expected_r``/``min_p_secure``) applies. The EV
threshold's in-sample and out-of-sample optima still do not correlate (IS
optima all 0.3, OOS optima [0.0, 0.3, 0.3, 0.3, 0.3], r = 0.00 — explicitly
marked "NOT TUNABLE" by the training script, not retuned here), and the
secure head's Brier score is close between in-sample (0.2242) and
out-of-sample (0.2383). This is a thin, marginal edge — one fold negative,
a mean of one hundredth of an R — not a demonstrated moneymaker, and it is
reported here exactly as measured rather than rounded up.

SANDBOX
────────────────────────────────────────────────────────────────────────
Passes ``strategies/sandbox.validate_and_load``: imports only allowlisted
modules, no dunder attribute access, no ``from __future__ import
annotations``, no ``exec``/``eval``/``open``/``getattr``. Model weights are
read once in ``__init__``; ``evaluate()`` performs no I/O.
"""

from pathlib import Path

import numpy as np
import torch

from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec
from src.strategies.generated.smc_dl_features_v2 import atr, detect_zones
from src.strategies.generated.smc_dl_features_v3 import (
    FEATURE_NAMES,
    MIN_ENTRY_BARS,
    compute_features_live,
)
from src.strategies.generated.smc_dl_labels_v2 import expected_r
from src.strategies.generated.smc_dl_model_v2 import SmcExitAwareNet

_MODELS_DIR = Path("data") / "ml_models"
_MODEL_NAME = "smc_dl_m5_v3"


class SmcDlM5V3:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="smc_dl_m5_v3",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            confirmation_timeframes=("M15", "H1", "H4"),
            params={
                "target_r": 1.75,
                "secure_r": 0.2,
                # --- zone retest geometry -------------------------------
                # Stop sits this many ATR beyond the zone's far edge, plus
                # the spread. The zone edge is the level that invalidates the
                # idea; everything else is noise buffer.
                "sl_zone_buffer_atr": 0.25,
                # Floor beneath the zone-edge stop: a thin zone (common —
                # nothing here requires a minimum zone height) otherwise
                # produces a stop distance small enough that ordinary spread
                # noise stops it out before the thesis has a chance to play
                # out. Matches the sl_min_atr_mult floor every other bot in
                # this fleet already applies to its own zone/pivot-based
                # stop.
                "sl_min_atr_mult": 0.5,
                # Refuse the setup when price has run so far from the zone
                # that the resulting stop is the wide one the retest exists
                # to avoid.
                "max_sl_atr": 2.5,
                # A level tested this many times is no longer that level.
                "max_zone_touches": 2,
                # A base needs to have been left behind before a return to it
                # is a "retest" rather than still being the impulse.
                "min_zone_age_bars": 3,
                # Hard structural veto: reject a buy with unmitigated supply
                # within this fraction of the stop distance overhead (and the
                # mirror for sells).
                "opposing_zone_veto_r": 1.5,
                # --- cost ------------------------------------------------
                # Spread as a fraction of risk. Cost is 116% of gross profit
                # on this account, so this is a hard decline, not a penalty.
                # It is also what keeps `min_rr` legal: `SpreadGate` silently
                # rejects a leg whose TP is under 1.5x(sl + spread), i.e.
                # 1.5 x (1 + cost_r) in R. At the 0.12 cap that is 1.68, so a
                # 1.75R target clears it with margin; loosening this cap
                # without raising `min_rr` starts producing legs the broker
                # drops on the floor without an error.
                "max_cost_r": 0.12,
                # `SpreadGate` rejects a leg whose TP is under this multiple
                # of (stop + spread). Applied per trade against that trade's
                # own cost rather than a worst case, so a cheap setup keeps
                # the full range between the floor and `target_r`.
                "broker_min_rr": 1.5,
                # --- model filter ----------------------------------------
                # Minimum expected R, net of spread. Well above zero because
                # the model's out-of-sample edge, while positive, is thin
                # (see module docstring).
                "min_expected_r": 0.10,
                # Break-even for the engine's 0.2R trailing rule is
                # 1/(1+0.2) = 0.8333. Below it the trade loses money on the
                # secure leg however attractive the far target looks.
                "min_p_secure": 0.8333,
                # Used instead of the two gates above while the model has not
                # demonstrated out-of-sample skill (see `_model_trusted`);
                # this model has, so in practice this file uses the strict
                # gate. Kept for parity with the sibling files and as the
                # fallback if a future retrain loses the edge.
                "advisory_reject_ev": -0.15,
                "point_value": 0.01,
            },
            htf_veto=False,
        )

        self._model = None
        self._mean = None
        self._scale = None
        self._temperature = 1.0
        # Whether the weights earned the right to veto a structural setup —
        # set from the model's own walk-forward summary, never assumed.
        self._model_trusted = False
        self._status = "no model: rule fallback active"
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load weights + scaler + metadata, or stay in fallback mode.

        Every failure path here is silent-but-recorded rather than raising:
        a strategy that throws in ``__init__`` fails sandbox validation and
        takes the whole bot offline, which is a worse outcome than trading
        the fallback rule.
        """
        weights = _MODELS_DIR / (_MODEL_NAME + ".pt")
        scaler = _MODELS_DIR / (_MODEL_NAME + "_scaler.npz")
        meta = _MODELS_DIR / (_MODEL_NAME + "_meta.json")
        if not weights.exists() or not scaler.exists():
            return

        try:
            if meta.exists():
                text = meta.read_text()
                names, temperature, hidden = _read_meta(text)
                self._model_trusted = _model_has_oos_edge(text)
                # A model whose feature list differs from this build's is a
                # silent mis-scaling of every weight, not a minor mismatch.
                if names is not None and tuple(names) != tuple(FEATURE_NAMES):
                    self._status = "model rejected: feature list differs from this build"
                    return
                self._temperature = temperature
            else:
                hidden = 128

            scaler_data = np.load(str(scaler))
            mean = scaler_data["mean"]
            scale = scaler_data["scale"]
            if len(mean) != len(FEATURE_NAMES):
                self._status = "model rejected: scaler width differs from feature count"
                return

            model = SmcExitAwareNet(input_dim=len(FEATURE_NAMES), hidden_dim=hidden)
            model.load_state_dict(torch.load(str(weights), weights_only=True, map_location="cpu"))
            model.eval()
            self._model = model
            self._mean = mean
            self._scale = np.clip(scale, 1e-8, None)
            self._status = (
                "model loaded (out-of-sample edge demonstrated: strict gate)"
                if self._model_trusted
                else "model loaded (no out-of-sample edge: advisory only)"
            )
        except Exception:
            self._model = None
            self._status = "model failed to load: rule fallback active"

    # ------------------------------------------------------------------
    def evaluate(self, ctx: MarketContext):
        """Structure proposes, the model disposes.

        The entry is a **zone retest**, never a momentum signal — the same
        fix motivated by live trade #8731164505, where the v1 M5 bot bought
        at the top of an impulse because its only entry condition was a
        direction probability. Chasing the impulse instead of waiting for
        the retest is not a cosmetic difference: entering at the zone edge
        rather than the impulse extreme is routinely a 2-3x tighter stop for
        the identical thesis, which is 2-3x more size for the same risk
        budget and 2-3x less loss when wrong. Entering at the extreme also
        makes the very next move toward the zone adverse *by construction* —
        the mechanism behind "88.7% of losers were in profit first".

        So the pipeline is:

          1. ``detect_zones`` proposes unmitigated demand/supply zones.
          2. Only a **fresh touch** qualifies — the bar price enters the
             zone, not every bar it sits inside. The equivalent M1 guard cut
             signal rate from 28.8% of bars to 5.5%; with transaction cost at
             116% of gross profit, that reduction is most of the work.
          3. A hard structural veto rejects buying into unmitigated supply
             (and selling into demand) regardless of what the model says.
          4. The stop goes just beyond the zone's far edge, so the tight
             stop and the good reward:risk are a *consequence* of entering at
             the zone rather than a separate parameter.
          5. The model filters and sizes what survives — and unlike the M1
             sibling, this model has demonstrated a (thin) out-of-sample
             edge, so it gates strictly rather than just vetoing.

        Steps 1-4 stand on their own regardless of what step 5 finds.
        """
        df_m5 = ctx.candles.get("M5")
        if df_m5 is None or len(df_m5) < MIN_ENTRY_BARS:
            return None

        atr_series = atr(df_m5, 14)
        atr_value = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if not np.isfinite(atr_value) or atr_value <= 0:
            return None

        params = self.spec.params
        price = float(df_m5["close"].iloc[-1])
        spread_price = float(ctx.spread_points) * params["point_value"]

        zones = detect_zones(df_m5, atr_series)
        candidate = self._select_retest(zones, price, params)
        if candidate is None:
            return None

        is_buy = candidate["kind"] == "demand"
        # The stop sits just beyond the zone's far edge — the level that
        # actually invalidates the idea — plus a buffer for noise and spread.
        buffer_price = atr_value * params["sl_zone_buffer_atr"] + spread_price
        if is_buy:
            stop_price = candidate["price_low"] - buffer_price
            sl_points = price - stop_price
        else:
            stop_price = candidate["price_high"] + buffer_price
            sl_points = stop_price - price
        if sl_points <= 0:
            return None
        sl_points = max(sl_points, atr_value * params["sl_min_atr_mult"])

        # Chasing guard: if price has already run far from the zone, the
        # stop this produces is the wide one the retest was supposed to
        # avoid. Decline rather than take a degraded version of the setup.
        if sl_points > atr_value * params["max_sl_atr"]:
            return None

        if self._veto_opposing_zone(zones, is_buy, price, sl_points, params):
            return None

        cost_r = min(spread_price / sl_points, 1.0)
        if cost_r > params["max_cost_r"]:
            return None

        direction = Direction.BUY if is_buy else Direction.SELL
        zone_note = (
            candidate["kind"]
            + " ["
            + _fmt(candidate["price_low"])
            + ","
            + _fmt(candidate["price_high"])
            + "] touch="
            + str(candidate["touches"])
            + " age="
            + str(candidate["age_bars"])
        )

        verdict = self._model_verdict(ctx, is_buy, cost_r, params)
        if verdict is None:
            # No usable model: trade the structural setup on its own, which
            # is the part with a mechanism behind it.
            tp_r = params["target_r"]
            if params["broker_min_rr"] * (1.0 + cost_r) > tp_r:
                return None
            return Signal(
                direction=direction,
                sl_points=sl_points,
                tp_points=sl_points * tp_r,
                confidence=0.5,
                reason=(
                    "DL m5 v3 retest ("
                    + self._status
                    + "): "
                    + zone_note
                    + " sl="
                    + _fmt(sl_points)
                    + " cost="
                    + _fmt(cost_r)
                    + "R"
                ),
            )

        ev, p_sec, expected_mfe = verdict
        if self._model_trusted:
            # The weights beat their own walk-forward: let them gate.
            if ev < params["min_expected_r"] or p_sec < params["min_p_secure"]:
                return None
        elif ev < params["advisory_reject_ev"]:
            # The weights did not beat their own walk-forward, so they get a
            # veto over clearly-bad setups and nothing more — see module
            # docstring "HONEST STATUS OF THE MODEL". Not the live path for
            # this file's current weights, but kept for a future retrain
            # that loses the edge.
            return None

        # Target where the model expects price to actually reach, floored at
        # the broker's minimum reward:risk and capped at `target_r`.
        broker_floor_r = params["broker_min_rr"] * (1.0 + cost_r)
        if broker_floor_r > params["target_r"]:
            return None
        tp_r = min(max(expected_mfe, broker_floor_r), params["target_r"])
        return Signal(
            direction=direction,
            sl_points=sl_points,
            tp_points=sl_points * tp_r,
            confidence=min(max(p_sec, 0.0), 1.0),
            reason=(
                "DL m5 v3 retest "
                + zone_note
                + " E[R]="
                + _fmt(ev)
                + " p_secure="
                + _fmt(p_sec)
                + " tp="
                + _fmt(tp_r)
                + "R cost="
                + _fmt(cost_r)
                + "R"
            ),
        )

    # ------------------------------------------------------------------
    def _select_retest(self, zones, price, params):
        """The freshest untested zone price has just entered, or None.

        Requires ``fresh_touch``: the bar price *enters* the zone. Sitting
        inside a zone is not a signal, or the bot re-enters every bar for as
        long as the retest lasts.
        """
        best = None
        for zone in zones:
            if not zone["fresh_touch"]:
                continue
            if zone["touches"] > params["max_zone_touches"]:
                continue
            if zone["age_bars"] < params["min_zone_age_bars"]:
                continue
            if zone["price_high"] <= zone["price_low"]:
                continue
            # A zone price is already through is not a retest.
            if zone["kind"] == "demand" and price < zone["price_low"]:
                continue
            if zone["kind"] == "supply" and price > zone["price_high"]:
                continue
            if best is None or zone["index"] > best["index"]:
                best = zone
        return best

    def _veto_opposing_zone(self, zones, is_buy, price, sl_points, params):
        """Hard structural veto, independent of the model.

        Never buy with unmitigated supply sitting just overhead, never sell
        into unmitigated demand just below. This runs before the model and
        cannot be overridden by it: a veto a confident-but-wrong probability
        can switch off is not a veto.
        """
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

    def _model_verdict(self, ctx, is_buy, cost_r, params):
        """``(expected_r, p_secure, expected_mfe_r)`` for this side, or None.

        None means "no usable opinion" — missing weights, a rejected feature
        contract, or a row that is not fully warmed up. The caller then
        trades the structural setup alone rather than guessing.

        ``smc_dl_features_v3.compute_features_live`` takes the entry frame
        and an ordered ``(rung1, rung2, rung3)`` HTF tuple directly rather
        than a fixed-name dict, so this strategy (the only thing that knows
        which of its own declared timeframes is "entry" and which are
        confirmation rungs) assembles that tuple itself.
        """
        if self._model is None:
            return None
        candles_entry = ctx.candles.get("M5")
        htf_candles = (ctx.candles.get("M15"), ctx.candles.get("H1"), ctx.candles.get("H4"))
        features = compute_features_live(
            candles_entry,
            htf_candles,
            spread_points=float(ctx.spread_points),
            point_value=params["point_value"],
        )
        if features is None:
            return None

        scaled = (features - self._mean) / self._scale
        with torch.no_grad():
            secure_logits, target_logits, _revert_logits, mfe = self._model(
                torch.FloatTensor(scaled).unsqueeze(0)
            )
        p_secure = torch.sigmoid(secure_logits / self._temperature).squeeze(0).numpy()
        p_target = torch.sigmoid(target_logits / self._temperature).squeeze(0).numpy()
        mfe_r = mfe.squeeze(0).numpy()

        side_index = 0 if is_buy else 1
        p_sec = float(p_secure[side_index])
        p_tgt = float(p_target[side_index])
        ev = expected_r(p_sec, p_tgt, params["target_r"], params["secure_r"], cost_r)
        return ev, p_sec, float(mfe_r[side_index])


def _fmt(value):
    return str(round(float(value), 3))


def _read_meta(text):
    """Pull feature names, temperature and hidden width out of the meta JSON.

    Hand-parsed because ``json`` is not on the sandbox import allowlist and
    adding a stdlib module to that list to read three fields is a worse
    trade than twenty lines of scanning. The format is written by
    ``scripts/train_smc_dl_v3.py``, so it is not arbitrary input.
    """
    names = None
    temperature = 1.0
    hidden = 128
    marker = '"feature_names"'
    start = text.find(marker)
    if start >= 0:
        open_bracket = text.find("[", start)
        close_bracket = text.find("]", open_bracket)
        if open_bracket >= 0 and close_bracket > open_bracket:
            body = text[open_bracket + 1 : close_bracket]
            names = [part.strip().strip('"') for part in body.split(",") if part.strip()]
    temperature = _read_number(text, '"temperature"', temperature)
    hidden = int(_read_number(text, '"hidden_dim"', hidden))
    return names, temperature, hidden


def _model_has_oos_edge(text):
    """Did this model beat its own walk-forward validation?

    Read from the metadata the trainer wrote, so the strategy cannot claim
    an edge the training run did not measure. The bar is deliberately plain:
    a positive mean out-of-sample average R, and a majority of folds
    positive. The current M5 v3 weights score +0.0096 with 4 of 5 folds
    positive and do clear it — see module docstring.
    """
    oos_avg_r = _read_number(text, '"oos_avg_r_mean"', 0.0)
    folds = _read_number(text, '"folds"', 0.0)
    positive = _read_number(text, '"folds_positive_oos"', 0.0)
    return bool(oos_avg_r > 0.0 and folds > 0 and positive * 2 > folds)


def _read_number(text, marker, default):
    start = text.find(marker)
    if start < 0:
        return default
    colon = text.find(":", start)
    if colon < 0:
        return default
    end = colon + 1
    while end < len(text) and text[end] not in ",}\n":
        end = end + 1
    try:
        return float(text[colon + 1 : end].strip())
    except ValueError:
        return default
