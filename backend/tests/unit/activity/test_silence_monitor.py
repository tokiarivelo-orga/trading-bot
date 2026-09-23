"""SilenceMonitor's periodic check (OBSERVABILITY_PLAN.md Phase 5) — same
testing shape as `tests/unit/broker/test_health_monitor.py`: drive
`check_once()` directly rather than the background task loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.activity.application.silence_monitor import SilenceMonitor
from src.activity.domain.models import SignalDecision
from src.shared.events.bus import EventBus
from src.shared.events.definitions import BotWentSilent

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


class FakeSignalDecisionRepository:
    def __init__(self, decisions: list[SignalDecision]) -> None:
        self._decisions = decisions

    def list_between(self, *, account_id: str, created_from: int, **_kwargs):
        return [
            d
            for d in self._decisions
            if d.account_id == account_id and int(d.created_at.timestamp()) >= created_from
        ]


def _decision(bot: str, minutes_ago: float) -> SignalDecision:
    return SignalDecision(
        signal_id=f"sig-{bot}-{minutes_ago}",
        account_id="default",
        bot=bot,
        strategy="fake",
        symbol="XAUUSD",
        timeframe="M5",
        direction="buy",
        price=2400.0,
        created_at=NOW - timedelta(minutes=minutes_ago),
        outcome="opened",
        reason="test",
    )


def _collector():
    published: list[BotWentSilent] = []

    async def handler(event: BotWentSilent) -> None:
        published.append(event)

    return published, handler


async def test_regular_bot_never_alerts():
    decisions = [_decision("bot-a", m) for m in (50, 40, 30, 20, 10, 8)]
    repo = FakeSignalDecisionRepository(decisions)
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(BotWentSilent, handler)
    monitor = SilenceMonitor(repo, event_bus, clock=lambda: NOW)

    await monitor.check_once()

    assert published == []


async def test_silent_bot_publishes_bot_went_silent_once():
    decisions = [_decision("bot-b", m) for m in (120, 110, 100, 90, 80, 60)]
    repo = FakeSignalDecisionRepository(decisions)
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(BotWentSilent, handler)
    monitor = SilenceMonitor(repo, event_bus, clock=lambda: NOW)

    await monitor.check_once()
    await monitor.check_once()  # still silent, same underlying data

    assert len(published) == 1  # not re-alerted every poll
    assert published[0].bot == "bot-b"
    assert published[0].elapsed_s == 3600.0


async def test_bot_recovering_allows_a_future_re_alert():
    # Regular 10-min cadence ending exactly at T0 — establishes a 10-min
    # median / 50-min (5x) threshold.
    decisions = [_decision("bot-c", m) for m in (50, 40, 30, 20, 10, 0)]
    repo = FakeSignalDecisionRepository(decisions)
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(BotWentSilent, handler)
    clock_state = {"now": NOW}
    monitor = SilenceMonitor(repo, event_bus, clock=lambda: clock_state["now"])

    # 70 min after the last signal: past the 50-min threshold -> silent.
    clock_state["now"] = NOW + timedelta(minutes=70)
    await monitor.check_once()
    assert len(published) == 1

    # Bot fires again one minute after that check, then only 4 more minutes
    # pass — well under threshold again, so the alerted-bots set clears.
    repo._decisions = [*decisions, _decision("bot-c", -71)]
    clock_state["now"] = NOW + timedelta(minutes=75)
    await monitor.check_once()
    assert len(published) == 1  # no new alert while healthy

    # No further signals, and enough time passes past the same threshold —
    # this must alert again, not stay silenced from the first spell.
    clock_state["now"] = NOW + timedelta(minutes=122)
    await monitor.check_once()
    assert len(published) == 2  # went silent again -> alerts again
    assert published[1].bot == "bot-c"


async def test_bot_below_min_signals_baseline_never_alerts():
    decisions = [_decision("bot-d", m) for m in (500, 400, 300)]
    repo = FakeSignalDecisionRepository(decisions)
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(BotWentSilent, handler)
    monitor = SilenceMonitor(repo, event_bus, clock=lambda: NOW, min_signals=5)

    await monitor.check_once()

    assert published == []


async def test_repository_error_is_caught_and_logged_not_raised():
    class RaisingRepository:
        def list_between(self, **_kwargs):
            raise RuntimeError("db unavailable")

    monitor = SilenceMonitor(RaisingRepository(), EventBus(), clock=lambda: NOW)

    await monitor.check_once()  # must not raise
