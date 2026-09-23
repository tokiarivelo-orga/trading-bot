"""Custom indicator CRUD + compute (mirrors
`tests/unit/strategies/test_versioning.py`'s fixture shape, minus the
versioning/activation lifecycle indicators don't have)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.indicators.adapters.repository import IndicatorRepository
from src.indicators.application.service import (
    IndicatorNameConflictError,
    IndicatorService,
    IndicatorValidationError,
)
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


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    candle_repository = CandleRepository(session_factory)
    _seed_candles(candle_repository)
    return IndicatorService(IndicatorRepository(session_factory), candle_repository)


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


def test_create_and_get(service):
    created = service.create(name="sma3", code=VALID_CODE, default_params={"period": 3})
    assert created.name == "sma3"
    fetched = service.get(created.id)
    assert fetched is not None
    assert fetched.code == VALID_CODE


def test_create_rejects_invalid_code(service):
    with pytest.raises(IndicatorValidationError):
        service.create(name="evil", code=INVALID_CODE)


def test_create_rejects_duplicate_name(service):
    service.create(name="sma3", code=VALID_CODE)
    with pytest.raises(IndicatorNameConflictError):
        service.create(name="sma3", code=VALID_CODE)


def test_edit_updates_in_place_no_new_row(service):
    created = service.create(name="sma3", code=VALID_CODE)
    new_code = VALID_CODE.replace('"period", 3', '"period", 5')
    edited = service.edit(created.id, new_code)
    assert edited.id == created.id
    assert edited.code_hash != created.code_hash
    assert len(service.list_all()) == 1


def test_edit_rejects_invalid_code(service):
    created = service.create(name="sma3", code=VALID_CODE)
    with pytest.raises(IndicatorValidationError):
        service.edit(created.id, INVALID_CODE)
    # Original code is untouched after a failed edit.
    assert service.get(created.id).code == VALID_CODE


def test_edit_missing_indicator_raises(service):
    with pytest.raises(ValueError):
        service.edit("does-not-exist", VALID_CODE)


def test_duplicate_clones_code_under_new_name(service):
    created = service.create(name="sma3", code=VALID_CODE, default_params={"period": 3})
    duplicated = service.duplicate(created.id, new_name="sma3-copy")
    assert duplicated.id != created.id
    assert duplicated.code == created.code
    assert duplicated.default_params == created.default_params
    assert {d.name for d in service.list_all()} == {"sma3", "sma3-copy"}


def test_duplicate_rejects_name_conflict(service):
    created = service.create(name="sma3", code=VALID_CODE)
    with pytest.raises(IndicatorNameConflictError):
        service.duplicate(created.id, new_name="sma3")


def test_delete_removes_indicator(service):
    created = service.create(name="sma3", code=VALID_CODE)
    service.delete(created.id)
    assert service.get(created.id) is None


def test_delete_missing_indicator_raises(service):
    with pytest.raises(ValueError):
        service.delete("does-not-exist")


def test_compute_returns_aligned_series(service):
    created = service.create(name="sma3", code=VALID_CODE, default_params={"period": 3})
    result = service.compute(created.id, symbol="XAUUSD", timeframe="M5", period="2026-06:2026-07")
    assert result.error is None
    assert len(result.times) == len(result.series["value"])
    assert result.series["value"][0] is None  # warm-up gap before period=3 fills in
    assert result.series["value"][-1] is not None


def test_compute_missing_indicator_raises(service):
    with pytest.raises(ValueError):
        service.compute("does-not-exist", symbol="XAUUSD", timeframe="M5", period="2026-06:2026-07")


def test_compute_no_candles_returns_error_not_exception(service):
    created = service.create(name="sma3", code=VALID_CODE)
    result = service.compute(
        created.id, symbol="UNKNOWN_SYMBOL", timeframe="M5", period="2026-06:2026-07"
    )
    assert result.error is not None
    assert result.series == {}


def test_preview_does_not_persist(service):
    result = service.preview(
        VALID_CODE,
        symbol="XAUUSD",
        timeframe="M5",
        period="2026-06:2026-07",
        params={"period": 3},
    )
    assert result.error is None
    assert result.series["value"][-1] is not None
    assert service.list_all() == []
