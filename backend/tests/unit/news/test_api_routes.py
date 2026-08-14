from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.news.adapters.repository import NewsEventRepository
from src.news.api.routes import events_router, router
from src.news.application.news_window_service import NewsWindowService
from src.news.domain.models import ImpactLevel, NewsConfig, NewsEvent, TrackedEvent, WindowSpec
from src.shared.db.base import Base
from src.shared.events.bus import EventBus

NOW = datetime.now(UTC)  # the route always resolves "now" live, so tests must match


class FakeCalendar:
    def __init__(self, events: list[NewsEvent]) -> None:
        self.events = events

    async def fetch_upcoming(self, days_ahead: int) -> list[NewsEvent]:
        return self.events


def _api(events: list[NewsEvent], tracked, specs) -> httpx.AsyncClient:
    service = NewsWindowService(
        calendar=FakeCalendar(events),
        config=NewsConfig(
            calendar_source="forexfactory",
            refresh_minutes=60,
            tracked_events=tracked,
            default_before_min=30,
            default_after_min=60,
        ),
        window_specs=specs,
        event_bus=EventBus(),
    )
    service._events = events  # populate cache without hitting the refresh loop
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(news_window_service=service)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


async def test_upcoming_lists_events_with_resolved_skill():
    event = NewsEvent(
        name="Non-Farm Payrolls", time=NOW + timedelta(hours=1), impact=ImpactLevel.HIGH
    )
    tracked = (TrackedEvent(name="Non-Farm Payrolls", impact=ImpactLevel.HIGH, skill="nfp"),)
    specs = {"nfp": WindowSpec(skill_name="nfp", before_min=30, after_min=60, symbols=("XAUUSD",))}

    async with _api([event], tracked, specs) as client:
        response = await client.get("/news/upcoming", params={"days_ahead": 7})

    assert response.status_code == 200
    (body,) = response.json()
    assert body["name"] == "Non-Farm Payrolls"
    assert body["skill"] == "nfp"


async def test_upcoming_skill_is_null_when_unmatched():
    event = NewsEvent(name="Retail Sales", time=NOW + timedelta(hours=1), impact=ImpactLevel.LOW)

    async with _api([event], (), {}) as client:
        response = await client.get("/news/upcoming")

    (body,) = response.json()
    assert body["skill"] is None


async def test_active_windows_empty_when_nothing_active():
    async with _api([], (), {}) as client:
        response = await client.get("/news/active-windows")

    assert response.status_code == 200
    assert response.json() == []


# ── GET /accounts/{account_id}/news/events (Phase 6 Part B/C) ───────────────


def _events_api(repository: NewsEventRepository) -> httpx.AsyncClient:
    service = NewsWindowService(
        calendar=FakeCalendar([]),
        config=NewsConfig(
            calendar_source="forexfactory",
            refresh_minutes=60,
            tracked_events=(),
            default_before_min=30,
            default_after_min=60,
        ),
        window_specs={},
        event_bus=EventBus(),
        repository=repository,
    )
    app = FastAPI()
    app.include_router(events_router)
    app.state.container = SimpleNamespace(
        news_window_service=service, accounts={"default": SimpleNamespace()}
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


def _make_repository(tmp_path) -> NewsEventRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/events.db")
    Base.metadata.create_all(engine)
    return NewsEventRepository(sessionmaker(bind=engine, expire_on_commit=False))


async def test_get_news_events_returns_persisted_rows_with_balance(tmp_path):
    repository = _make_repository(tmp_path)
    event_time = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)
    event = NewsEvent(name="Non-Farm Payrolls", time=event_time, impact=ImpactLevel.HIGH)
    repository.save_many([event])
    repository.set_balance_before("Non-Farm Payrolls", event_time, 10_000.0)
    repository.set_balance_after("Non-Farm Payrolls", event_time, 10_050.0)

    start = int(datetime(2026, 8, 14, 0, 0, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 8, 14, 23, 59, tzinfo=UTC).timestamp())

    async with _events_api(repository) as client:
        response = await client.get(
            "/accounts/default/news/events", params={"start": start, "end": end}
        )

    assert response.status_code == 200
    (body,) = response.json()
    assert body["name"] == "Non-Farm Payrolls"
    assert body["balance_before"] == 10_000.0
    assert body["balance_after"] == 10_050.0
    assert isinstance(body["id"], int)


async def test_get_news_events_filters_by_impact(tmp_path):
    repository = _make_repository(tmp_path)
    repository.save_many(
        [
            NewsEvent(
                name="High", time=datetime(2026, 8, 14, 10, 0, tzinfo=UTC), impact=ImpactLevel.HIGH
            ),
            NewsEvent(
                name="Low", time=datetime(2026, 8, 14, 11, 0, tzinfo=UTC), impact=ImpactLevel.LOW
            ),
        ]
    )
    start = int(datetime(2026, 8, 14, 0, 0, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 8, 14, 23, 59, tzinfo=UTC).timestamp())

    async with _events_api(repository) as client:
        response = await client.get(
            "/accounts/default/news/events",
            params={"start": start, "end": end, "impact": "high"},
        )

    assert response.status_code == 200
    (body,) = response.json()
    assert body["name"] == "High"


async def test_get_news_events_404s_on_unknown_account(tmp_path):
    repository = _make_repository(tmp_path)
    start = int(datetime(2026, 8, 14, 0, 0, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 8, 14, 23, 59, tzinfo=UTC).timestamp())

    async with _events_api(repository) as client:
        response = await client.get(
            "/accounts/unknown-account/news/events", params={"start": start, "end": end}
        )

    assert response.status_code == 404


async def test_get_news_events_empty_when_nothing_persisted(tmp_path):
    repository = _make_repository(tmp_path)
    start = int(datetime(2026, 8, 14, 0, 0, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 8, 14, 23, 59, tzinfo=UTC).timestamp())

    async with _events_api(repository) as client:
        response = await client.get(
            "/accounts/default/news/events", params={"start": start, "end": end}
        )

    assert response.status_code == 200
    assert response.json() == []
