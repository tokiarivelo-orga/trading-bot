"""Give-back exit policy: stop handing back profit that was already earned.

WHY THIS EXISTS
────────────────────────────────────────────────────────────────────────
Measured on 4,113 closed live XAUUSD trades that carry excursion data:

  * **89.5% of losing trades were in profit first** (1,949 of 2,178).
  * Those losers gave back **3,579R** in aggregate.
  * Across every trade, peak-to-close give-back totals **4,286R**.

For comparison, the entire entry-selection edge this project has been able
to demonstrate out-of-sample is worth a small fraction of that. The trades
were not wrong. They were right, and then died — and no entry filter can
recover a single R of it.

The existing ``PositionManager`` rules do not address this. Rule 1 moves to
breakeven at +1R and Rule 2 secures +0.2R once a base is cleared, but both
are one-shot ratchets keyed to structure; a position that runs to +1.5R and
then collapses is stopped at +0.2R, booking a 1.3R give-back as a "win".

WHY IT IS A RULE AND NOT A PREDICTION
────────────────────────────────────────────────────────────────────────
The obvious design is to predict the reversal. That was built and measured:
the ``revert`` head of ``smc_dl_m5_v2`` scores **50.4-50.8% out-of-sample
accuracy across four walk-forward folds** on a ~50/50 base rate. It has no
skill, and gating exits on it would be gating on noise.

So the policy is driven by **realised position state** — peak favourable
excursion, which is a fact, not a forecast. The model is wired in as an
optional *modulator* (``revert_probability``) that can only make the policy
act sooner, never later, and only when it is far past chance. With no model
present the policy is unchanged. That ordering is deliberate: the proven
part must not depend on the unproven part.

WHAT WAS MEASURED, AND WHAT IS STILL UNPROVEN
────────────────────────────────────────────────────────────────────────
Two independent harnesses, both chronologically split, both driving *this*
function (``scripts/eval_giveback_exit.py``):

  synthetic M5 entries, 2,830 trades, split 2026-01
    baseline        OOS avg R -0.0076   PF 0.98   OOS maxDD 33.0R
    arm 0.3/keep 0.5 OOS avg R +0.0102  PF 1.03   OOS maxDD 18.5R

  replay of 6,686 real closed live trades over their own M1 price paths
    baseline        OOS avg R -0.0737   PF 0.85   OOS maxDD 287.6R
    arm 0.3/keep 0.5 OOS avg R -0.0032  PF 0.99   OOS maxDD  81.2R

The drawdown reduction is large and consistent — roughly 45% on synthetic
entries, 72% on real trades — and out-of-sample average R improves in both.

**But the parameters are not tunable.** The IS/OOS correlation across the
grid is r = -0.13 (synthetic) and r = -0.00 (live): the best in-sample
setting is among the worst out-of-sample and vice versa, and every variant
is *worse* than baseline in-sample. The defaults below are therefore chosen
as a value that is reasonable in both halves of both harnesses, **not** as
the optimum of either. Re-fitting them on in-sample data would be the same
mistake that produced a per-hour entry filter at PF 2.70 in-sample and PF
0.12 out-of-sample on this data.

Treat this as: drawdown control with a plausible but small expectancy
benefit, not as a validated profit engine.

COMPOSITION WITH THE EXISTING RULES
────────────────────────────────────────────────────────────────────────
This never loosens a stop. It proposes a candidate that ``PositionManager``
merges through the same ``_improves`` check every other rule uses, so
whichever rule is currently most protective wins. It also never touches
``configs/risk.yaml``, the consecutive-loss pause, or the kill switch.

Pure functions over floats — no I/O, no broker, no market data — so the
offline simulator in ``scripts/eval_giveback_exit.py`` and the live
``PositionManager`` run the *same* code and their results are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Mirrors `position_manager.DEFAULT_SECURE_BUFFER_R_MULT`; if that moves,
# this follows.
DEFAULT_SECURE_BUFFER_R_MULT = 0.2


class ExitAction(StrEnum):
    NONE = "none"
    TIGHTEN_SL = "tighten_sl"
    REDUCE_TP = "reduce_tp"
    CLOSE = "close"


@dataclass(frozen=True, kw_only=True)
class ExitPolicyConfig:
    """Every knob, with the reasoning for each default.

    ``enabled=False`` reproduces pre-existing behaviour exactly — the policy
    contributes no candidate at all — so it can be switched off without
    touching wiring.
    """

    enabled: bool = True
    # Peak unrealised profit (in R) before the give-back trail arms at all.
    # Below this the position has not earned anything worth protecting and
    # the existing breakeven/secure rules are the right tool.
    # 0.3 was the best-behaved arm level out-of-sample in both harnesses and
    # the worst in-sample in both — see "what was measured" above; it is a
    # compromise, not a fitted optimum.
    arm_r: float = 0.3
    # Fraction of the peak that is locked in once armed. 0.5 = "give back at
    # most half of the best you saw".
    keep_fraction: float = 0.5
    # The locked level never sits below this, so arming can never produce a
    # stop worse than the engine's own secure buffer.
    min_lock_r: float = DEFAULT_SECURE_BUFFER_R_MULT
    # A confirmed change of character against an already-profitable position
    # tightens to at least this. Structure-driven, not model-driven.
    choch_lock_r: float = DEFAULT_SECURE_BUFFER_R_MULT
    # Take-profit is pulled in to the model's expected favourable excursion
    # when that is materially closer than the resting TP — a TP the market
    # is not going to reach is a TP that never fills.
    reduce_tp_enabled: bool = True
    min_tp_reduction_r: float = 0.3
    # Fixed profit target in R, as a resting take-profit. `None` = off, which
    # is the default: this measurably rescues the M1 bots (OOS -375.8R ->
    # +105.6R at 0.30R) and does almost nothing for M15, so it belongs in a
    # `per_bot:` block rather than in the fleet-wide defaults. See the block
    # in `decide_exit` for the full sweep.
    fixed_target_r: float | None = None
    # Model modulator. Only consulted above this probability, and only to act
    # sooner. 0.5 would be chance; the measured OOS accuracy of the revert
    # head is 0.504, so this default is set high enough that today's model
    # effectively never triggers it.
    revert_probability_threshold: float = 0.80
    revert_min_peak_r: float = 0.3


@dataclass(frozen=True, kw_only=True)
class ExitPolicySettings:
    """The policy as configured for a *fleet*: one default plus per-strategy
    overrides.

    Per-strategy exists because the measurement says it must. Replaying every
    real closed trade over its own M1 path, chronologically split per bot
    (`scripts/eval_giveback_per_bot.py`), the policy helped 8 bots and hurt 7:

        xauusd_snd_qm_structure_m1        -150.6R -> -31.5R   (+119.0R)
        xauusd_snd_qm_structure_fixed_m1  -106.0R ->  -7.7R    (+98.3R)
        xauusd_snd_qm_structure_adaptive_m1 +37.3R -> -2.2R    (-39.5R)

    The pattern is consistent: it rescues bots that are bleeding and taxes
    bots that are working, because capping give-back also caps winners. A
    single global switch therefore cannot be right for every bot, and the
    aggregate `+188.3R` hides that it is nearly all one bot.

    `resolve()` returns `None` when the policy is off for that strategy,
    which is exactly what `PositionManager` already treats as "rule
    disabled" — so a disabled bot reproduces pre-existing behaviour with no
    special-casing anywhere.
    """

    default: ExitPolicyConfig
    # strategy name (e.g. "xauusd_snd_qm_structure_m1") -> override.
    per_strategy: dict[str, ExitPolicyConfig] = field(default_factory=dict)

    def resolve(self, strategy: str | None) -> ExitPolicyConfig | None:
        config = self.per_strategy.get(strategy or "", self.default)
        return config if config.enabled else None


@dataclass(frozen=True)
class ExitDecision:
    action: ExitAction
    stop_price: float | None = None
    take_profit: float | None = None
    reason: str = ""


def peak_r(*, is_buy: bool, entry_price: float, extreme_favorable: float, risk: float) -> float:
    """Best unrealised progress seen since entry, in R.

    ``extreme_favorable`` is the highest price reached for a buy, the lowest
    for a sell. Returns 0.0 rather than a negative number when the position
    has never been in profit — "peak profit" is floored at flat.
    """
    if risk <= 0:
        return 0.0
    direction = 1.0 if is_buy else -1.0
    return max((extreme_favorable - entry_price) * direction / risk, 0.0)


def decide_exit(
    *,
    is_buy: bool,
    entry_price: float,
    current_sl: float,
    mark: float,
    extreme_favorable: float,
    risk: float,
    take_profit: float | None = None,
    expected_mfe_r: float | None = None,
    revert_probability: float | None = None,
    choch_against: bool = False,
    config: ExitPolicyConfig,
) -> ExitDecision:
    """One exit decision for one open position at one candle close.

    ``mark`` is the current bid (buy) / ask (sell), matching what
    ``PositionManager`` already uses. ``extreme_favorable`` is the running
    high-water mark the caller tracks per ticket.

    Returns ``ExitAction.NONE`` when the policy has nothing to add — which is
    the common case, and the correct behaviour before the position has earned
    anything.
    """
    if not config.enabled or risk <= 0:
        return ExitDecision(ExitAction.NONE)

    direction = 1.0 if is_buy else -1.0
    best_r = peak_r(
        is_buy=is_buy, entry_price=entry_price, extreme_favorable=extreme_favorable, risk=risk
    )
    current_r = (mark - entry_price) * direction / risk

    # --- model modulator: close outright on a confident reversal call -------
    # Deliberately first and deliberately narrow. It requires the position to
    # already be in real profit, so the worst case is banking a winner early
    # rather than converting a winner into a loser.
    if (
        revert_probability is not None
        and revert_probability >= config.revert_probability_threshold
        and best_r >= config.revert_min_peak_r
        and current_r > 0.0
    ):
        return ExitDecision(
            ExitAction.CLOSE,
            reason=(
                f"give-back exit: model reversal p={revert_probability:.2f} "
                f"with {current_r:.2f}R unrealised"
            ),
        )

    # --- fixed profit target -----------------------------------------------
    # Pulls the resting take-profit in to `fixed_target_r` x risk. A *limit*,
    # not a market close: this rule runs at M5 candle closes, so closing at
    # market would book whatever the close happens to be, while a resting TP
    # fills at the level itself — which is what the counterfactual below
    # measured. `SpreadGate`'s min_rr floor applies to entries, not to
    # modifying an already-open position, so a sub-min_rr target is reachable
    # here and nowhere else.
    #
    # Measured on live XAUUSD M1 trades, chronologically split
    # (`scripts/eval_fixed_target_by_tf.py`), 1,549 OOS trades:
    #
    #   baseline  -375.8R  PF 0.60
    #   0.20R      +79.4R  PF 1.56   fires on 89%
    #   0.30R     +105.6R  PF 1.47   fires on 82%
    #   0.40R      +70.2R  PF 1.20
    #   0.50R     +103.5R  PF 1.26
    #   1.00R      -25.0R  PF 0.96
    #
    # Four adjacent grid points positive, not a single-point spike — the
    # shape is why this is trusted where a per-hour filter was not. It is
    # *not* a general improvement: on M15 the same sweep moves OOS from
    # -14.9R only to about zero, so this is a per-bot rule, off by default.
    if config.fixed_target_r is not None and config.fixed_target_r > 0.0:
        target_price = entry_price + direction * config.fixed_target_r * risk
        # Only ever pulls the target closer; never pushes it further out.
        if take_profit is None or (target_price - take_profit) * direction < 0:
            return ExitDecision(
                ExitAction.REDUCE_TP,
                take_profit=target_price,
                reason=(
                    f"fixed target: tp -> {config.fixed_target_r:.2f}R "
                    f"(peak {best_r:.2f}R)"
                ),
            )

    candidate: float | None = None
    reason = ""

    # --- rule A: give-back trail on realised peak ---------------------------
    if best_r >= config.arm_r:
        lock_r = max(best_r * config.keep_fraction, config.min_lock_r)
        candidate = entry_price + direction * lock_r * risk
        reason = f"give-back trail: peak {best_r:.2f}R, locking {lock_r:.2f}R"

    # --- rule B: change of character against a profitable position ----------
    if choch_against and best_r > 0.0:
        choch_level = entry_price + direction * config.choch_lock_r * risk
        if candidate is None or (choch_level - candidate) * direction > 0:
            candidate = choch_level
            reason = f"change of character against position at {current_r:.2f}R"

    if candidate is not None and (candidate - current_sl) * direction > 0:
        # A stop that has already been passed by price means "close now" —
        # sending it to the broker as a modification would be rejected or,
        # worse, filled at a random later price.
        if (mark - candidate) * direction <= 0:
            return ExitDecision(ExitAction.CLOSE, reason=reason + " (level already breached)")
        return ExitDecision(ExitAction.TIGHTEN_SL, stop_price=candidate, reason=reason)

    # --- rule C: pull an unreachable take-profit in --------------------------
    if (
        config.reduce_tp_enabled
        and take_profit is not None
        and expected_mfe_r is not None
        and expected_mfe_r > 0.0
    ):
        current_tp_r = (take_profit - entry_price) * direction / risk
        if current_tp_r - expected_mfe_r >= config.min_tp_reduction_r:
            new_tp = entry_price + direction * expected_mfe_r * risk
            if (new_tp - mark) * direction > 0:
                return ExitDecision(
                    ExitAction.REDUCE_TP,
                    take_profit=new_tp,
                    reason=(
                        f"take-profit pulled in from {current_tp_r:.2f}R to "
                        f"{expected_mfe_r:.2f}R (model expected excursion)"
                    ),
                )

    return ExitDecision(ExitAction.NONE)
