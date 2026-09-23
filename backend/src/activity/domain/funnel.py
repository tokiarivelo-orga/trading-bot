"""Veto funnel: how far each of a bot's signals got before something stopped
it (OBSERVABILITY_PLAN.md Phase 2).

Pure domain code — takes already-loaded `SignalDecision`s and folds them into
per-bot stage counts plus the drop reasons at each stage. No I/O, no
framework, so the aggregation is trivially unit-testable.

Stage order here is the order `TradeEngine._enter_for_bot` actually evaluates
its gates: HTF confirmation first, then the open-position cap / lot sizing,
and only then the broker's spread + RR gate. (The plan's prose lists
"spread" before "sized OK"; the engine sizes first, and the funnel must
reflect the real order or the counts would not be monotonic.)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.activity.domain.models import SignalDecision

STAGE_FIRED = "fired"
STAGE_PASSED_HTF = "passed_htf"
STAGE_SIZED_OK = "sized_ok"
STAGE_PASSED_SPREAD = "passed_spread"
STAGE_FILLED = "filled"

FUNNEL_STAGES: tuple[str, ...] = (
    STAGE_FIRED,
    STAGE_PASSED_HTF,
    STAGE_SIZED_OK,
    STAGE_PASSED_SPREAD,
    STAGE_FILLED,
)

# How far a decision with this terminal outcome got, as an index into
# FUNNEL_STAGES. A decision "reached" every stage up to and including this
# index, and was dropped at the next one.
_OUTCOME_STAGE: dict[str, int] = {
    # Never got past being fired: pending, pre-trade risk gate, circuit
    # breaker, or the HTF confirmation itself.
    "skipped": 0,
    "daily_loss_breaker": 0,
    "risk_rejected": 0,
    "htf_veto": 0,
    # Passed HTF, died before/at sizing. `volatility_guard` is legacy (the
    # guard was removed) — kept so historical rows land in the right stage.
    "volatility_guard": 1,
    "max_positions": 1,
    "risk_sizing": 1,
    # Sized fine, died at the broker's spread / risk-reward gate.
    "spread_veto": 2,
    "rr_gate": 2,
    # Passed every engine gate; the broker itself refused the order.
    "broker_rejected": 3,
    "opened": 4,
}

# An outcome we've never heard of must not silently vanish from the funnel:
# it counts as fired and is reported as a drop at the first stage.
_UNKNOWN_STAGE = 0


@dataclass(frozen=True, kw_only=True)
class FunnelDrop:
    """One reason signals stopped at a given stage, with how often."""

    stage: str
    """The `FUNNEL_STAGES` entry these signals failed to reach."""
    outcome: str
    count: int
    example_reason: str
    """The reason text of one of the dropped signals, so the number is
    actionable without a second query."""


@dataclass(frozen=True, kw_only=True)
class BotFunnel:
    """One bot's signal funnel over the queried period."""

    bot: str
    symbols: tuple[str, ...]
    fired: int
    passed_htf: int
    sized_ok: int
    passed_spread: int
    filled: int
    drops: tuple[FunnelDrop, ...]


def stage_reached(outcome: str) -> int:
    """Index into `FUNNEL_STAGES` of the last stage a decision with this
    terminal `outcome` reached."""
    return _OUTCOME_STAGE.get(outcome, _UNKNOWN_STAGE)


def build_funnels(decisions: list[SignalDecision]) -> list[BotFunnel]:
    """Folds decisions into one `BotFunnel` per bot, ordered by fired count
    descending then bot name, so the busiest bot leads the panel."""
    by_bot: dict[str, list[SignalDecision]] = {}
    for decision in decisions:
        by_bot.setdefault(decision.bot, []).append(decision)
    funnels = [_build_one(bot, rows) for bot, rows in by_bot.items()]
    funnels.sort(key=lambda f: (-f.fired, f.bot))
    return funnels


def _build_one(bot: str, decisions: list[SignalDecision]) -> BotFunnel:
    reached = [stage_reached(d.outcome) for d in decisions]
    counts = [sum(1 for r in reached if r >= stage) for stage in range(len(FUNNEL_STAGES))]
    counts[0] = len(decisions)  # every recorded decision fired by definition

    dropped: Counter[tuple[int, str]] = Counter()
    examples: dict[tuple[int, str], str] = {}
    for decision, r in zip(decisions, reached, strict=True):
        if r >= len(FUNNEL_STAGES) - 1:
            continue  # filled: not a drop
        key = (r + 1, decision.outcome)
        dropped[key] += 1
        examples.setdefault(key, decision.reason)

    drops = tuple(
        FunnelDrop(
            stage=FUNNEL_STAGES[stage_index],
            outcome=outcome,
            count=count,
            example_reason=examples[(stage_index, outcome)],
        )
        for (stage_index, outcome), count in sorted(
            dropped.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])
        )
    )
    return BotFunnel(
        bot=bot,
        symbols=tuple(sorted({d.symbol for d in decisions})),
        fired=counts[0],
        passed_htf=counts[1],
        sized_ok=counts[2],
        passed_spread=counts[3],
        filled=counts[4],
        drops=drops,
    )
