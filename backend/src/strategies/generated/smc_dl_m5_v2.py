"""SMC deep-learning M5 strategy, v2 — expected-R gating with a rule fallback.

WHAT THIS DOES DIFFERENTLY FROM ``smc_dl_m5_v1.py``
────────────────────────────────────────────────────────────────────────
* Gates on **expected R**, not on a direction probability. v1 fired whenever
  ``bull_prob > 0.35``, which on a 3-class softmax is close to "always", and
  its declared ``min_tp_prob``/``min_direction_conf`` params were never read
  by ``evaluate()`` at all. Here the decision is
  ``expected_r(p_secure, p_target) >= min_expected_r``, net of the live
  spread, with the same arithmetic the trainer scored folds with.
* **Verifies the feature contract.** v1 built its input as
  ``np.array(list(features_dict.values()))`` and only checked the *length*
  against the scaler. Any reordering or insertion silently mis-fed every
  weight. Here the model's ``_meta.json`` carries the exact feature name
  list and the strategy refuses to use weights whose list disagrees.
* **Falls back instead of dying.** With no weights, a shape mismatch, or a
  bad row, this trades a conservative MTF-aligned structure rule rather than
  returning ``None`` forever (v1 was silently inert whenever the model was
  missing) or trading blind.

HONEST STATUS OF THE MODEL
────────────────────────────────────────────────────────────────────────
Walk-forward on 104,681 M5 XAUUSD bars, 4 chronological folds: in-sample avg
R 0.124, **out-of-sample avg R -0.005**, 1 of 4 folds positive, and the EV
threshold's in-sample and out-of-sample optima do not correlate (r = 0.00).
The heads are well calibrated (predicted 0.561 vs actual 0.593 in the
dominant bin) but have almost no discriminative power — permutation
importance peaks at 0.0008 Brier.

So the model is **not** a demonstrated entry edge, and this strategy is
deliberately conservative about acting on it: ``min_expected_r`` and
``min_p_secure`` default to levels that trade rarely, and ``min_p_secure``
sits at the arithmetic break-even for the engine's 0.2R trailing rule
(0.833) rather than at a number tuned on in-sample data. The value in this
file is the plumbing — correct features, correct EV arithmetic, verified
contract, safe fallback — not a claim that it prints money.

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
from src.strategies.generated.smc_dl_features_v2 import (
    FEATURE_NAMES,
    atr,
    compute_features_live,
    detect_zones,
)
from src.strategies.generated.smc_dl_labels_v2 import expected_r
from src.strategies.generated.smc_dl_model_v2 import SmcExitAwareNet

_MODELS_DIR = Path("data") / "ml_models"
_MODEL_NAME = "smc_dl_m5_v2"


class SmcDlM5V2:
    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="smc_dl_m5_v2",
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
                # the model's out-of-sample edge is not established.
                "min_expected_r": 0.10,
                # Break-even for the engine's 0.2R trailing rule is
                # 1/(1+0.2) = 0.8333. Below it the trade loses money on the
                # secure leg however attractive the far target looks.
                "min_p_secure": 0.8333,
                # Used instead of the two gates above while the model has not
                # demonstrated out-of-sample skill (see `_model_trusted`):
                # the model may then only veto a setup it scores clearly
                # negative, never require it to clear break-even.
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

        The entry is a **zone retest**, never a momentum signal. That
        ordering is the fix for live trade #8731164505, where the v1 bot
        bought at 4420.23 — above the top of the RBR zone [4394.66, 4419.35]
        that two other bots were trading — because its only entry condition
        was a direction probability.

        Chasing the impulse instead of waiting for the retest is not a
        cosmetic difference. Against the same invalidation level (4380.74,
        the zone's lower edge):

            entry at the impulse top 4416.39  ->  risk 35.65 pts
            entry at the zone top    ~4400    ->  risk ~19.3 pts
            entry at the zone middle ~4393    ->  risk ~12.3 pts

        Identical thesis, a 2-3x wider stop: 2-3x less size for the same risk
        budget and 2-3x the loss when wrong. And entering at the extreme
        makes the next move toward the zone adverse *by construction*, which
        is the mechanism behind "88.7% of losers were in profit first".

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
          5. The model filters and sizes what survives.

        Steps 1-4 stand on their own. Step 5 is the only part that depends on
        weights whose out-of-sample edge is not established, so its absence
        degrades the bot rather than disabling it.
        """
        df_m5 = ctx.candles.get("M5")
        if df_m5 is None or len(df_m5) < 60:
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
                    "DL v2 retest ("
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
            # veto over clearly-bad setups and nothing more.
            #
            # Requiring `p_secure >= 0.8333` from an unproven model is not
            # conservative, it is arbitrary: this model's p_secure reaches
            # 0.764 at the 99th percentile across 104,675 bars, so the strict
            # gate passes 0.015% of bars and the bot simply never trades.
            # Handing an edgeless model absolute veto power over a setup that
            # does have a mechanism is the wrong way round — and the strict
            # gate switches itself on automatically the moment a retrain
            # produces genuine out-of-sample edge.
            return None

        # Target where the model expects price to actually reach, floored at
        # the broker's minimum reward:risk and capped at `target_r`.
        #
        # The floor is computed from *this* trade's cost: `SpreadGate`
        # silently drops a leg whose TP is under `broker_min_rr x (stop +
        # spread)`, which in R is `broker_min_rr x (1 + cost_r)`. A bot that
        # ignores this runs one leg short and never learns why. `target_r` is
        # fixed by the labels the model was trained on, so when the floor
        # exceeds it there is no placeable target and the setup is declined
        # rather than sent in a form the broker will discard.
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
                "DL v2 retest "
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
        inside a zone is not a signal, or the bot re-enters every five
        minutes for as long as the retest lasts.
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
        into unmitigated demand just below. The v1 bot had no structural
        gating whatsoever and ``htf_veto=False``, so nothing prevented it
        entering directly against structure — which is exactly what it did.

        This runs before the model and cannot be overridden by it: the model
        has demonstrated a willingness to buy the high, and a veto that a
        confident-but-wrong probability can switch off is not a veto.
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
        """
        if self._model is None:
            return None
        features = compute_features_live(
            ctx.candles,
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
    ``scripts/train_smc_dl_v2.py``, so it is not arbitrary input.
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
    positive. The current XAUUSD weights score -0.005 with 1 of 4 folds
    positive and do not clear it.
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
