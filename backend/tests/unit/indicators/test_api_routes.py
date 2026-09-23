"""Custom indicator CRUD + compute endpoints — mirrors
`tests/unit/strategies/test_api_routes.py`'s fixture shape."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.indicators.adapters.repository import IndicatorRepository
from src.indicators.api.routes import router
from src.indicators.application.service import IndicatorService
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base

VALID_CODE = """
import pandas as pd


class SimpleMovingAverage:
    def compute(self, candles: pd.DataFrame, params: dict) -> dict:
        period = int(params.get("period", 3))
        sma = candles["close"].rolling(period).mean()
        return {"value": sma.tolist()}
"""

INVALID_CODE = "import os\nx = 1\n"


def _seed_candles(repository: CandleRepository, count: int = 40) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    candles = [
        Candle(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            time=start + timedelta(minutes=5 * i),
            open=2000.0 + i,
            high=2001.0 + i,
            low=1999.0 + i,
            close=2000.5 + i,
            tick_volume=100 + i,
            spread_points=20,
        )
        for i in range(count)
    ]
    repository.upsert_many(candles)


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    candle_repository = CandleRepository(session_factory)
    _seed_candles(candle_repository)
    return IndicatorService(IndicatorRepository(session_factory), candle_repository)


@pytest.fixture
async def api(service):
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(indicators=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client


async def test_list_indicators_empty(api):
    response = await api.get("/indicators")
    assert response.status_code == 200
    assert response.json() == []


async def test_create_and_get_indicator(api):
    create_response = await api.post(
        "/indicators", json={"name": "sma3", "code": VALID_CODE, "default_params": {"period": 3}}
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["name"] == "sma3"
    assert created["code"] == VALID_CODE

    get_response = await api.get(f"/indicators/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["code"] == VALID_CODE

    list_response = await api.get("/indicators")
    (summary,) = list_response.json()
    assert summary["id"] == created["id"]
    assert "code" not in summary


async def test_create_rejects_invalid_code(api):
    response = await api.post("/indicators", json={"name": "evil", "code": INVALID_CODE})
    assert response.status_code == 422


async def test_create_rejects_duplicate_name(api):
    await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})
    response = await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})
    assert response.status_code == 409


async def test_get_indicator_not_found(api):
    response = await api.get("/indicators/does-not-exist")
    assert response.status_code == 404


async def test_edit_indicator(api):
    created = (await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})).json()
    new_code = VALID_CODE.replace('"period", 3', '"period", 5')
    response = await api.post(f"/indicators/{created['id']}/edit", json={"code": new_code})
    assert response.status_code == 200
    assert response.json()["code"] == new_code
    assert response.json()["id"] == created["id"]


async def test_edit_rejects_invalid_code(api):
    created = (await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})).json()
    response = await api.post(f"/indicators/{created['id']}/edit", json={"code": INVALID_CODE})
    assert response.status_code == 422


async def test_edit_not_found(api):
    response = await api.post("/indicators/does-not-exist/edit", json={"code": VALID_CODE})
    assert response.status_code == 404


async def test_duplicate_indicator(api):
    created = (await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})).json()
    response = await api.post(f"/indicators/{created['id']}/duplicate", json={"name": "sma3-copy"})
    assert response.status_code == 200
    assert response.json()["name"] == "sma3-copy"
    assert response.json()["code"] == VALID_CODE


async def test_duplicate_rejects_name_conflict(api):
    created = (await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})).json()
    response = await api.post(f"/indicators/{created['id']}/duplicate", json={"name": "sma3"})
    assert response.status_code == 409


async def test_delete_indicator(api):
    created = (await api.post("/indicators", json={"name": "sma3", "code": VALID_CODE})).json()
    delete_response = await api.delete(f"/indicators/{created['id']}")
    assert delete_response.status_code == 204
    assert (await api.get(f"/indicators/{created['id']}")).status_code == 404


async def test_delete_not_found(api):
    response = await api.delete("/indicators/does-not-exist")
    assert response.status_code == 404


async def test_compute_indicator(api):
    created = (
        await api.post(
            "/indicators",
            json={"name": "sma3", "code": VALID_CODE, "default_params": {"period": 3}},
        )
    ).json()
    response = await api.post(
        f"/indicators/{created['id']}/compute",
        json={"symbol": "XAUUSD", "timeframe": "M5", "period": "2026-06:2026-07"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert len(body["times"]) == len(body["series"]["value"])
    assert body["series"]["value"][-1] is not None


async def test_compute_not_found(api):
    response = await api.post(
        "/indicators/does-not-exist/compute",
        json={"symbol": "XAUUSD", "timeframe": "M5", "period": "2026-06:2026-07"},
    )
    assert response.status_code == 404


async def test_preview_indicator(api):
    response = await api.post(
        "/indicators/preview",
        json={
            "code": VALID_CODE,
            "params": {"period": 3},
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "period": "2026-06:2026-07",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["series"]["value"][-1] is not None


async def test_preview_invalid_code_returns_error_in_body(api):
    response = await api.post(
        "/indicators/preview",
        json={
            "code": INVALID_CODE,
            "params": {},
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "period": "2026-06:2026-07",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is not None
