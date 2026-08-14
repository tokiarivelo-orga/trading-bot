"""`GET /accounts/{account_id}/order-book/export` — bulk order-book export
(CSV/JSON), filtered by symbol and/or a capture-time window."""

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

from src.order_book.adapters.repository import OrderBookSnapshotRepository
from src.order_book.api.routes import router
from src.order_book.application.capture import OrderBookCaptureService
from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot
from src.shared.db.base import Base


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path) -> OrderBookSnapshotRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/export.db")
    Base.metadata.create_all(engine)
    return OrderBookSnapshotRepository(sessionmaker(bind=engine, expire_on_commit=False))


def _api(repository: OrderBookSnapshotRepository) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={
            "default": SimpleNamespace(
                order_book_capture=OrderBookCaptureService(gateway=None, repository=repository)
            )
        }
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


async def test_export_json_is_empty_list_when_nothing_captured(repository):
    async with _api(repository) as client:
        response = await client.get("/accounts/default/order-book/export")
    assert response.status_code == 200
    assert response.json() == []


async def test_export_csv_is_header_only_when_nothing_captured(repository):
    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/order-book/export", params={"format": "csv"}
        )
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows == [
        ["signal_id", "symbol", "captured_at", "level_index", "side", "price", "volume"]
    ]


async def test_export_json_returns_typed_snapshots(repository):
    repository.save(
        "sig-1",
        OrderBookSnapshot(
            symbol="XAUUSD",
            time=utc(2026, 8, 13, 12, 0),
            levels=(
                BookLevel(side=BookSide.BID, price=2400.10, volume=5.5),
                BookLevel(side=BookSide.ASK, price=2400.35, volume=3.0),
            ),
        ),
    )

    async with _api(repository) as client:
        response = await client.get("/accounts/default/order-book/export")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["signal_id"] == "sig-1"
    assert body[0]["symbol"] == "XAUUSD"
    assert body[0]["captured_at"] == int(utc(2026, 8, 13, 12, 0).timestamp())
    assert body[0]["levels"] == [
        {"side": "bid", "price": 2400.10, "volume": 5.5},
        {"side": "ask", "price": 2400.35, "volume": 3.0},
    ]


async def test_export_csv_flattens_one_row_per_level(repository):
    repository.save(
        "sig-1",
        OrderBookSnapshot(
            symbol="XAUUSD",
            time=utc(2026, 8, 13, 12, 0),
            levels=(
                BookLevel(side=BookSide.BID, price=2400.10, volume=5.5),
                BookLevel(side=BookSide.ASK, price=2400.35, volume=3.0),
            ),
        ),
    )

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/order-book/export", params={"format": "csv"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == [
        "signal_id",
        "symbol",
        "captured_at",
        "level_index",
        "side",
        "price",
        "volume",
    ]
    assert len(rows) == 3  # header + 2 levels
    assert rows[1][:2] == ["sig-1", "XAUUSD"]
    assert rows[1][4] == "bid"
    assert rows[2][4] == "ask"


async def test_export_filters_by_symbol(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 10, 0)))
    repository.save("sig-2", OrderBookSnapshot(symbol="VIX75", time=utc(2026, 8, 13, 11, 0)))

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/order-book/export", params={"symbol": "XAUUSD"}
        )

    body = response.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "XAUUSD"


async def test_export_filters_by_since_and_until(repository):
    repository.save("sig-1", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 9, 0)))
    repository.save("sig-2", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 10, 0)))
    repository.save("sig-3", OrderBookSnapshot(symbol="XAUUSD", time=utc(2026, 8, 13, 11, 0)))

    async with _api(repository) as client:
        response = await client.get(
            "/accounts/default/order-book/export",
            params={
                "since": int(utc(2026, 8, 13, 10, 0).timestamp()),
                "until": int(utc(2026, 8, 13, 10, 0).timestamp()),
            },
        )

    body = response.json()
    assert [row["signal_id"] for row in body] == ["sig-2"]
