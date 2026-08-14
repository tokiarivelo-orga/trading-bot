"""`GET /accounts/{account_id}/market-data/candles/export` — bulk candle
export (CSV streamed, JSON typed), required date range, real_volume/atr_14/
day_of_week passthrough."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.market_data.api.export import router
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import Timeframe
from src.shared.db.base import Base

ORIGIN = 1_767_610_800  # 2026-01-05 11:00 UTC, a Monday
M5 = 300


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path) -> CandleRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/export.db")
    Base.metadata.create_all(engine)
    return CandleRepository(sessionmaker(bind=engine, expire_on_commit=False))


def _api(repository: CandleRepository) -> httpx.AsyncClient:
    gateway_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500)), base_url="http://gw"
    )
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


async def test_export_requires_from_and_to_time(repository):
    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={"symbol": "XAUUSD", "timeframe": "M5"},
        )
    assert response.status_code == 422


async def test_export_rejects_non_positive_range(repository):
    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN,
            },
        )
    assert response.status_code == 400


async def test_export_json_is_empty_for_no_candles_in_range(repository):
    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN + 10 * M5,
                "format": "json",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert body["candles"] == []


async def test_export_csv_is_header_only_for_no_candles_in_range(repository):
    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN + 10 * M5,
            },
        )
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.text)))
    assert len(rows) == 1
    assert rows[0][0] == "symbol"


def _seed(repository: CandleRepository, count: int) -> None:
    from src.market_data.domain.models import Candle

    candles = [
        Candle(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            time=datetime.fromtimestamp(ORIGIN + i * M5, tz=UTC),
            open=2400.0 + i,
            high=2401.0 + i,
            low=2399.0 + i,
            close=2400.5 + i,
            tick_volume=1000 + i,
            spread_points=25,
            real_volume=42 + i,
        )
        for i in range(count)
    ]
    repository.upsert_many(candles)
    repository.enrich_missing("XAUUSD", Timeframe.M5, atr_period=2)


async def test_export_json_includes_enriched_columns(repository):
    _seed(repository, 5)

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN + 5 * M5,
                "format": "json",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    for row in body["candles"]:
        assert "real_volume" in row
        assert "atr_14" in row
        assert "day_of_week" in row
    assert body["candles"][0]["real_volume"] == 42
    # Enrichment needs `atr_period` bars of trailing context; the last bars do have it.
    assert body["candles"][-1]["atr_14"] is not None
    assert body["candles"][-1]["day_of_week"] == 0  # Monday


async def test_export_csv_includes_enriched_columns_and_streams_in_pages(repository):
    _seed(repository, 5)

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN + 5 * M5,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    header = rows[0]
    assert header == [
        "symbol",
        "timeframe",
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread_points",
        "real_volume",
        "atr_14",
        "day_of_week",
    ]
    assert len(rows) == 6  # header + 5 candles
    real_volume_index = header.index("real_volume")
    assert rows[1][real_volume_index] == "42"


async def test_export_range_paginates_across_page_boundaries(repository):
    """`export_range`'s default `page_size` is 2000 — force a tiny page size
    directly to prove the cursor actually advances across pages rather than
    silently truncating at the first page."""
    _seed(repository, 7)

    service = CandleHistoryService(market_data=None, repository=repository)
    collected = [
        c
        async for c in service.export_range(
            "XAUUSD",
            Timeframe.M5,
            utc(2026, 1, 5, 11, 0),
            datetime.fromtimestamp(ORIGIN + 7 * M5, tz=UTC),
            page_size=2,
        )
    ]
    assert len(collected) == 7
    assert [c.time for c in collected] == sorted(c.time for c in collected)


async def test_export_json_rejects_range_over_the_row_cap(repository, monkeypatch):
    import src.market_data.api.export as export_module

    monkeypatch.setattr(export_module, "_MAX_JSON_EXPORT_ROWS", 3)
    _seed(repository, 5)

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/market-data/candles/export",
            params={
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "from_time": ORIGIN,
                "to_time": ORIGIN + 5 * M5,
                "format": "json",
            },
        )
    assert response.status_code == 400
