"""The backtest's in-memory decision sink (OBSERVABILITY_PLAN.md Phase 4).

Wiring the engine's own `SignalDecisionSinkPort` into the replay is what makes
a backtest emit the **Phase 2 split outcome vocabulary** instead of the
collapsed one the log-scraper recovered — which is the precondition for
comparing a backtest funnel with the live one at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.activity.domain.models import SIGNAL_OUTCOMES, DecisionCheck
from src.backtest.adapters.decision_sink import InMemorySignalDecisionSink

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


async def fire(sink: InMemorySignalDecisionSink, signal_id: str, *, direction: str = "buy") -> None:
    await sink.record(
        signal_id=signal_id,
        bot="backtest",
        strategy="breakout_v1",
        symbol="XAUUSD",
        timeframe="M5",
        direction=direction,
        price=2400.0,
        created_at=NOW,
        reason="M5 close broke 20-bar high",
        confidence=0.7,
    )


@pytest.mark.asyncio
async def test_a_fresh_signal_starts_with_no_terminal_outcome() -> None:
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a")
    (decision,) = sink.decisions()
    assert decision.outcome == "skipped"
    assert decision.account_id == "backtest"
    assert decision.price == 2400.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "htf_veto",
        "volatility_guard",
        "max_positions",
        "risk_sizing",
        "spread_veto",
        "rr_gate",
        "broker_rejected",
        "daily_loss_breaker",
        "opened",
    ],
)
async def test_each_split_outcome_survives_to_the_report(outcome: str) -> None:
    """The old log-scraper collapsed five of these into `risk_rejected`."""
    assert outcome in SIGNAL_OUTCOMES
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a")
    await sink.record_outcome("a", outcome)
    (signal,) = sink.signals()
    assert signal.outcome == outcome


@pytest.mark.asyncio
async def test_an_opened_signal_is_never_downgraded_by_a_later_rejection() -> None:
    """A multi-target signal whose first target fills and whose second is
    refused did become a trade; reporting it as rejected would undercount
    fills and corrupt the funnel."""
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a")
    await sink.record_outcome("a", "opened")
    await sink.record_outcome(
        "a",
        "broker_rejected",
        checks=(
            DecisionCheck(
                name="broker_retcode",
                value=10016.0,
                threshold=10009.0,
                comparison="==",
                passed=False,
            ),
        ),
    )
    (decision,) = sink.decisions()
    assert decision.outcome == "opened"
    # The refused target's numbers are still on the record.
    assert [c.name for c in decision.checks] == ["broker_retcode"]


@pytest.mark.asyncio
async def test_checks_append_across_gates_rather_than_replacing() -> None:
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a")
    await sink.record_checks(
        "a",
        (
            DecisionCheck(
                name="spread_points", value=30, threshold=50, comparison="<=", passed=True
            ),
        ),
    )
    await sink.record_outcome(
        "a",
        "rr_gate",
        checks=(
            DecisionCheck(
                name="risk_reward", value=0.8, threshold=1.5, comparison=">=", passed=False
            ),
        ),
    )
    (decision,) = sink.decisions()
    assert [c.name for c in decision.checks] == ["spread_points", "risk_reward"]
    assert decision.outcome == "rr_gate"


@pytest.mark.asyncio
async def test_signals_keep_the_order_they_fired_in() -> None:
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a", direction="buy")
    await fire(sink, "b", direction="sell")
    await fire(sink, "c", direction="buy")
    await sink.record_outcome("b", "opened")
    assert [s.direction for s in sink.signals()] == ["buy", "sell", "buy"]
    assert [s.outcome for s in sink.signals()] == ["skipped", "opened", "skipped"]


@pytest.mark.asyncio
async def test_outcomes_for_unknown_signals_are_ignored_not_invented() -> None:
    sink = InMemorySignalDecisionSink()
    await sink.record_outcome("never-fired", "opened")
    await sink.record_checks("never-fired", ())
    assert sink.decisions() == ()


@pytest.mark.asyncio
async def test_a_duplicate_signal_id_does_not_double_count() -> None:
    sink = InMemorySignalDecisionSink()
    await fire(sink, "a")
    await fire(sink, "a")
    assert len(sink.decisions()) == 1
