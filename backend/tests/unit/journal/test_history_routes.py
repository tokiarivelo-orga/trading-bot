"""GET /journal/history — filtered, paginated trade history (F7 extension)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.journal.adapters.repository import JournalRepository
from src.journal.api.routes import router
from src.journal.domain.models import TradeRecord
from src.shared.db.base import Base
from src.shared.events.bus import EventBus


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def make_record(id: str, symbol: str = "XAUUSD", **kw) -> TradeRecord:
    defaults = dict(
        id=id,
        symbol=symbol,
        side="buy",
        volume=0.1,
        open_price=2400.35,
        open_time=utc(2026, 7, 10, 14, 0),
        sl=2390.0,
        tp=2420.0,
        spread_points_at_entry=25,
        comment="",
    )
    return TradeRecord(**{**defaults, **kw})


@pytest.fixture
def repository(tmp_path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))


class FakeMarketContext:
    async def capture(self, symbol):
        raise AssertionError("market context should not be hit by history search")


@pytest.fixture
async def api(repository):
    from src.journal.application.trade_journal import TradeJournalService

    trade_journal = TradeJournalService(
        repository=repository, market_context=FakeMarketContext(), event_bus=EventBus()
    )
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={"default": SimpleNamespace(trade_journal=trade_journal)}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client


@pytest.fixture(autouse=True)
def _seed(repository):
    repository.save(
        make_record(
            "1",
            symbol="XAUUSD",
            side="buy",
            strategy_version="breakout_v1:v1",
            open_time=utc(2026, 7, 10, 14, 0),
            close_price=2410.0,
            close_time=utc(2026, 7, 10, 15, 0),
            profit=9.65,
        )
    )
    repository.save(
        make_record(
            "2",
            symbol="EURUSD",
            side="sell",
            open_time=utc(2026, 7, 10, 15, 0),
            close_price=1.0990,
            close_time=utc(2026, 7, 10, 16, 0),
            profit=-3.0,
        )
    )
    repository.save(
        make_record("3", symbol="XAUUSD", open_time=utc(2026, 7, 10, 16, 0))
    )  # still open


async def test_returns_all_trades_with_total(api):
    response = await api.get("/accounts/default/journal/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["total_profit"] == pytest.approx(6.65)
    assert [t["id"] for t in body["items"]] == ["3", "2", "1"]  # open_time desc


async def test_filters_by_symbol(api):
    response = await api.get("/accounts/default/journal/history", params={"symbol": "EURUSD"})
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "2"


async def test_filters_by_outcome_open(api):
    response = await api.get("/accounts/default/journal/history", params={"outcome": "open"})
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "3"


async def test_filters_by_outcome_loss(api):
    response = await api.get("/accounts/default/journal/history", params={"outcome": "loss"})
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == "2"


async def test_pagination_limit_offset(api):
    response = await api.get("/accounts/default/journal/history", params={"limit": 1, "offset": 1})
    body = response.json()
    assert body["total"] == 3
    assert [t["id"] for t in body["items"]] == ["2"]


async def test_markers_filters_by_skill(api, repository):
    repository.save(make_record("4", symbol="XAUUSD", skill="normal/xauusd/breakout_v1"))
    repository.save(make_record("5", symbol="XAUUSD", skill="normal/xauusd/mean_reversion"))

    response = await api.get(
        "/accounts/default/journal/markers",
        params={"symbol": "XAUUSD", "skill": "normal/xauusd/breakout_v1"},
    )

    assert response.status_code == 200
    assert [t["id"] for t in response.json()] == ["4"]


async def test_markers_without_skill_returns_every_bot(api):
    response = await api.get("/accounts/default/journal/markers", params={"symbol": "XAUUSD"})

    assert response.status_code == 200
    assert {t["id"] for t in response.json()} == {"1", "3"}
