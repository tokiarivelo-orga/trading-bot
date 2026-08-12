"""`/market-data` REST endpoints: candle reads, broker symbol browsing (chart/
watchlist only — never touches configs/app.yaml or the engine), and the
candle-gap scan/repair pair behind the chart's "fill gaps" button."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.market_data.api.routes import router
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base

CANDLE_WIRE = {
    "time": 1_752_100_500,
    "open": 2400.0,
    "high": 2401.0,
    "low": 2399.0,
    "close": 2400.5,
    "tick_volume": 1000,
    "spread": 25,
}

SYMBOLS_WIRE = [
    {"name": "XAUUSD", "description": "Gold vs US Dollar", "path": "Metals", "visible": True},
    {
        "name": "EURUSD",
        "description": "Euro vs US Dollar",
        "path": "Forex\\Majors",
        "visible": False,
    },
]


def _api(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    gateway_client = httpx.AsyncClient(transport=transport, base_url="http://gw")
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={"default": SimpleNamespace(market_data=GatewayMarketData(gateway_client))}
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


def _candles_api(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    gateway_client = httpx.AsyncClient(transport=transport, base_url="http://gw")
    app = FastAPI()
    app.include_router(router)
    # Repository is never touched here — the fake gateway handler always
    # succeeds, so the DB fallback path in CandleHistoryService is unused.
    app.state.container = SimpleNamespace(
        accounts={
            "default": SimpleNamespace(
                candle_history=CandleHistoryService(
                    GatewayMarketData(gateway_client), repository=None
                )
            )
        }
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


async def test_candles_omits_before_when_not_requested():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "before" not in request.url.params
        return httpx.Response(200, json=[CANDLE_WIRE])

    async with _candles_api(handler) as client:
        response = await client.get(
            "/accounts/default/market-data/candles", params={"symbol": "XAUUSD", "timeframe": "M5"}
        )
    assert response.status_code == 200
    assert response.json()[0]["time"] == 1_752_100_500


async def test_candles_forwards_before_as_epoch_seconds():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["before"] == "1752100000"
        return httpx.Response(200, json=[CANDLE_WIRE])

    async with _candles_api(handler) as client:
        response = await client.get(
            "/accounts/default/market-data/candles",
            params={"symbol": "XAUUSD", "timeframe": "M5", "before": 1_752_100_000},
        )
    assert response.status_code == 200


async def test_broker_symbols_lists_catalog():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/symbols"
        return httpx.Response(200, json={"items": SYMBOLS_WIRE, "total": len(SYMBOLS_WIRE)})

    async with _api(handler) as client:
        response = await client.get("/accounts/default/market-data/broker-symbols")
    assert response.status_code == 200
    body = response.json()
    assert {s["name"] for s in body["items"]} == {"XAUUSD", "EURUSD"}
    assert body["total"] == 2


async def test_broker_symbols_forwards_search_limit_and_offset():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "gold"
        assert request.url.params["limit"] == "10"
        assert request.url.params["offset"] == "5"
        return httpx.Response(200, json={"items": SYMBOLS_WIRE[:1], "total": 1})

    async with _api(handler) as client:
        response = await client.get(
            "/accounts/default/market-data/broker-symbols",
            params={"search": "gold", "limit": 10, "offset": 5},
        )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


async def test_broker_symbols_maps_gateway_unavailable_to_503():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with _api(handler) as client:
        response = await client.get("/accounts/default/market-data/broker-symbols")
    assert response.status_code == 503


# ── Candle gaps (chart's "fill gaps" button) ────────────────────────────────

M5 = 300
GAP_ORIGIN = 1_767_610_800  # 2026-01-05 11:00 UTC, a Monday


def _gaps_api(handler, repository: CandleRepository) -> httpx.AsyncClient:
    """Same wiring as `_candles_api`, but with a real (SQLite-backed) candle
    repository — the gap endpoints read stored history, so a `None` repository
    would make every scan trivially empty."""
    gateway_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gw")
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={
            "default": SimpleNamespace(
                candle_history=CandleHistoryService(
                    GatewayMarketData(gateway_client), repository=repository
                )
            )
        }
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


@pytest.fixture
def repository(tmp_path) -> CandleRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/gaps.db")
    Base.metadata.create_all(engine)
    return CandleRepository(sessionmaker(bind=engine, expire_on_commit=False))


def bars_at(indexes: list[int]) -> list[Candle]:
    return [
        Candle(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            time=datetime.fromtimestamp(GAP_ORIGIN + i * M5, tz=UTC),
            open=2400.0,
            high=2401.0,
            low=2399.0,
            close=2400.5,
            tick_volume=100,
            spread_points=25,
        )
        for i in indexes
    ]


def wire_bars(indexes: list[int]) -> list[dict]:
    return [{**CANDLE_WIRE, "time": GAP_ORIGIN + i * M5} for i in indexes]


async def test_candle_gaps_reports_a_hole_in_stored_history(repository):
    repository.upsert_many(bars_at(list(range(10)) + list(range(20, 30))))

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - unused
        raise AssertionError("scanning is read-only; it must not call the gateway")

    async with _gaps_api(handler, repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candle-gaps",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "start": GAP_ORIGIN,
                "end": GAP_ORIGIN + 29 * M5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["missing_bars"] == 10
    assert len(body["gaps"]) == 1
    assert body["gaps"][0]["start"] == GAP_ORIGIN + 10 * M5
    assert body["gaps"][0]["end"] == GAP_ORIGIN + 20 * M5
    assert body["gaps"][0]["weekend"] is False


async def test_candle_gaps_is_empty_for_contiguous_history(repository):
    repository.upsert_many(bars_at(list(range(30))))

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - unused
        raise AssertionError("scanning is read-only; it must not call the gateway")

    async with _gaps_api(handler, repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candle-gaps",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "start": GAP_ORIGIN,
                "end": GAP_ORIGIN + 29 * M5,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "start": GAP_ORIGIN,
        "end": GAP_ORIGIN + 29 * M5,
        "gaps": [],
        "missing_bars": 0,
    }


async def test_repair_downloads_the_hole_and_reports_it_closed(repository):
    repository.upsert_many(bars_at(list(range(10)) + list(range(20, 30))))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/candles"
        # Cursored at the hole's end, not the whole scanned window.
        assert request.url.params["before"] == str(GAP_ORIGIN + 20 * M5)
        return httpx.Response(200, json=wire_bars(list(range(20))))

    async with _gaps_api(handler, repository) as client:
        response = await client.post(
            "/accounts/default/market-data/candle-gaps/repair",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "start": datetime.fromtimestamp(GAP_ORIGIN, tz=UTC).isoformat(),
                "end": datetime.fromtimestamp(GAP_ORIGIN + 29 * M5, tz=UTC).isoformat(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["found"]) == 1
    assert len(body["repaired"]) == 1
    assert body["remaining"] == []
    assert body["bars_recovered"] == 10
    assert body["bars_downloaded"] == 20
    assert (
        len(
            repository.get_range(
                "XAUUSD",
                Timeframe.M5,
                datetime.fromtimestamp(GAP_ORIGIN, tz=UTC),
                datetime.fromtimestamp(GAP_ORIGIN + 30 * M5, tz=UTC),
            )
        )
        == 30
    )


async def test_repair_reports_holes_the_broker_cannot_fill(repository):
    repository.upsert_many(bars_at(list(range(10)) + list(range(20, 30))))

    def handler(request: httpx.Request) -> httpx.Response:
        # The broker's own history has the same hole (holiday/halt).
        return httpx.Response(200, json=wire_bars(list(range(10))))

    async with _gaps_api(handler, repository) as client:
        response = await client.post(
            "/accounts/default/market-data/candle-gaps/repair",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "start": datetime.fromtimestamp(GAP_ORIGIN, tz=UTC).isoformat(),
                "end": datetime.fromtimestamp(GAP_ORIGIN + 29 * M5, tz=UTC).isoformat(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["repaired"] == []
    assert len(body["remaining"]) == 1
    assert body["bars_recovered"] == 0


async def test_repair_maps_gateway_unavailable_to_503(repository):
    repository.upsert_many(bars_at(list(range(10)) + list(range(20, 30))))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with _gaps_api(handler, repository) as client:
        response = await client.post(
            "/accounts/default/market-data/candle-gaps/repair",
            json={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "start": datetime.fromtimestamp(GAP_ORIGIN, tz=UTC).isoformat(),
                "end": datetime.fromtimestamp(GAP_ORIGIN + 29 * M5, tz=UTC).isoformat(),
            },
        )

    assert response.status_code == 503
