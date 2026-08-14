from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.news.adapters.repository import NewsEventRepository
from src.news.application.news_window_service import NewsWindowService
from src.news.domain.models import ImpactLevel, NewsConfig, NewsEvent, TrackedEvent, WindowSpec
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import NewsWindowEntered, NewsWindowExited

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


class FakeCalendar:
    def __init__(self, events: list[NewsEvent]) -> None:
        self.events = events
        self.calls = 0

    async def fetch_upcoming(self, days_ahead: int) -> list[NewsEvent]:
        self.calls += 1
        return self.events


class FakeAccountService:
    """Stands in for `broker.application.account_service.AccountService` —
    only `status()`'s return shape matters here (`{"account": {"balance":
    ...}}` or `{"account": None}`, mirroring `AccountService.status()`'s
    real contract, see `trade_loop.py::_current_balance` for the same
    read)."""

    def __init__(self, balance: float | None) -> None:
        self.balance = balance
        self.calls = 0

    async def status(self) -> dict:
        self.calls += 1
        account = {"balance": self.balance} if self.balance is not None else None
        return {"gateway_up": True, "connected": True, "account": account}


def make_config(tracked: tuple[TrackedEvent, ...]) -> NewsConfig:
    return NewsConfig(
        calendar_source="forexfactory",
        refresh_minutes=60,
        tracked_events=tracked,
        default_before_min=30,
        default_after_min=60,
    )


def make_repository(tmp_path) -> NewsEventRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/news_window_service.db")
    Base.metadata.create_all(engine)
    return NewsEventRepository(sessionmaker(bind=engine, expire_on_commit=False))


def make_service(events, tracked, specs, event_bus=None, repository=None) -> NewsWindowService:
    return NewsWindowService(
        calendar=FakeCalendar(events),
        config=make_config(tracked),
        window_specs=specs,
        event_bus=event_bus or EventBus(),
        repository=repository,
    )


async def test_refresh_populates_cache():
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    service = make_service([event], (), {})
    await service.refresh(NOW)
    assert service.upcoming(now=NOW - timedelta(hours=1)) == [event]


async def test_active_window_for_matches_exact_tracked_event():
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {
        "nfp": WindowSpec(
            skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",), close_all=True
        )
    }
    service = make_service([event], tracked, specs)
    await service.refresh(NOW)

    window = service.active_window_for("XAUUSD", NOW)
    assert window is not None
    assert window.skill == "nfp"
    assert service.active_window_for("BTCUSD", NOW) is None  # not in spec.symbols


async def test_active_window_for_falls_back_to_wildcard():
    event = NewsEvent(name="Some Unlisted Release", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (
        TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),
        TrackedEvent(name="*", impact=ImpactLevel.HIGH, skill="generic_high_impact"),
    )
    specs = {
        "generic_high_impact": WindowSpec(
            skill_name="generic_high_impact", before_min=15, after_min=30, symbols=("XAUUSD",)
        )
    }
    service = make_service([event], tracked, specs)
    await service.refresh(NOW)

    window = service.active_window_for("XAUUSD", NOW)
    assert window is not None
    assert window.skill == "generic_high_impact"


async def test_no_match_means_no_active_window():
    event = NewsEvent(name="Retail Sales", time=NOW, impact=ImpactLevel.LOW)
    tracked = (TrackedEvent(name="*", impact=ImpactLevel.HIGH, skill="generic_high_impact"),)
    service = make_service([event], tracked, {})
    await service.refresh(NOW)

    assert service.active_window_for("XAUUSD", NOW) is None


async def test_active_window_only_within_before_after_bounds():
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}
    service = make_service([event], tracked, specs)
    await service.refresh(NOW)

    assert service.active_window_for("XAUUSD", NOW - timedelta(minutes=31)) is None
    assert service.active_window_for("XAUUSD", NOW + timedelta(minutes=61)) is None
    assert service.active_window_for("XAUUSD", NOW + timedelta(minutes=30)) is not None


async def test_check_transitions_publishes_entered_then_exited():
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {
        "nfp": WindowSpec(
            skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",), close_all=True
        )
    }
    event_bus = EventBus()
    entered: list[NewsWindowEntered] = []
    exited: list[NewsWindowExited] = []

    async def on_entered(e: NewsWindowEntered) -> None:
        entered.append(e)

    async def on_exited(e: NewsWindowExited) -> None:
        exited.append(e)

    event_bus.subscribe(NewsWindowEntered, on_entered)
    event_bus.subscribe(NewsWindowExited, on_exited)
    service = make_service([event], tracked, specs, event_bus=event_bus)
    await service.refresh(NOW)

    await service._check_transitions(NOW)  # inside the window
    assert len(entered) == 1
    assert entered[0].event_name == "Non-Farm Payrolls"
    assert entered[0].symbols == ("XAUUSD",)
    assert entered[0].close_all is True

    await service._check_transitions(NOW + timedelta(minutes=61))  # outside the window
    assert len(exited) == 1
    assert exited[0].event_name == "Non-Farm Payrolls"


# ── persistence (Phase 6 Part B) ─────────────────────────────────────────────


async def test_refresh_persists_fetched_events(tmp_path):
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    service = make_service([event], (), {}, repository=repository)

    await service.refresh(NOW)

    persisted = repository.list_between(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    assert [e.name for e in persisted] == ["Non-Farm Payrolls"]


async def test_refresh_without_repository_still_populates_in_memory_cache():
    """`repository=None` (the pre-Phase-6 default) must not break the
    zero-I/O `active_window_for()` cache path."""
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    service = make_service([event], (), {})  # no repository

    await service.refresh(NOW)

    assert service.upcoming(now=NOW - timedelta(hours=1)) == [event]


async def test_list_events_empty_without_repository():
    service = make_service([], (), {})
    assert await service.list_events(NOW - timedelta(days=1), NOW + timedelta(days=1)) == []


async def test_list_events_reads_persisted_history(tmp_path):
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    service = make_service([event], (), {}, repository=repository)
    await service.refresh(NOW)

    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))

    assert len(records) == 1
    row_id, stored_event, balance_before, balance_after = records[0]
    assert isinstance(row_id, int)
    assert stored_event.name == "Non-Farm Payrolls"
    assert balance_before is None
    assert balance_after is None


# ── balance_before/balance_after stamping (Phase 6 Part C) ──────────────────


async def test_window_entered_and_exited_stamp_balance_before_and_after(tmp_path):
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}
    service = make_service([event], tracked, specs, repository=repository)
    await service.refresh(NOW)

    account_service = FakeAccountService(balance=10_000.0)
    service.set_account_service(account_service)

    await service._check_transitions(NOW)  # inside the window -> entered
    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    _, _, balance_before, balance_after = records[0]
    assert balance_before == 10_000.0
    assert balance_after is None

    account_service.balance = 10_075.0
    await service._check_transitions(NOW + timedelta(minutes=61))  # outside -> exited
    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    _, _, balance_before, balance_after = records[0]
    assert balance_before == 10_000.0  # untouched by the exit stamp
    assert balance_after == 10_075.0


async def test_window_never_exits_leaves_balance_after_null_without_raising(tmp_path):
    """A process restart mid-window (or any other reason a window is never
    observed to exit) must not raise or block — `balance_after` just stays
    null, per the module's "enrichment, not core function" contract."""
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}
    service = make_service([event], tracked, specs, repository=repository)
    await service.refresh(NOW)
    service.set_account_service(FakeAccountService(balance=10_000.0))

    await service._check_transitions(NOW)  # entered, never checked again (simulated restart)

    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    _, _, balance_before, balance_after = records[0]
    assert balance_before == 10_000.0
    assert balance_after is None


async def test_balance_stamping_skipped_without_account_service(tmp_path):
    """`set_account_service` was never called (e.g. a unit test, or before
    container.py wires it post-account-construction) — must not raise."""
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}
    service = make_service([event], tracked, specs, repository=repository)
    await service.refresh(NOW)

    await service._check_transitions(NOW)  # must not raise

    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    _, _, balance_before, balance_after = records[0]
    assert balance_before is None
    assert balance_after is None


async def test_balance_stamping_skipped_when_account_disconnected(tmp_path):
    """`AccountService.status()["account"]` is `None` when disconnected —
    matches `trade_loop.py::_current_balance`'s same null-check."""
    repository = make_repository(tmp_path)
    event = NewsEvent(name="Non-Farm Payrolls", time=NOW, impact=ImpactLevel.HIGH)
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}
    service = make_service([event], tracked, specs, repository=repository)
    await service.refresh(NOW)
    service.set_account_service(FakeAccountService(balance=None))

    await service._check_transitions(NOW)  # must not raise

    records = await service.list_events(NOW - timedelta(hours=1), NOW + timedelta(hours=1))
    _, _, balance_before, balance_after = records[0]
    assert balance_before is None
    assert balance_after is None
