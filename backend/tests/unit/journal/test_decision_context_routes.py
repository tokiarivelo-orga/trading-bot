"""GET /journal/trades/{trade_id}/decision-context — entry chart snapshot +
decision annotations for one trade (Phase 2 of the "why did the bot take this
trade" chart-snippet feature)."""

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
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import CandleSnapshot, TradeRecord
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
        raise AssertionError("market context should not be hit by decision-context reads")


@pytest.fixture
async def api(repository):
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


def _m5_snapshot() -> tuple[CandleSnapshot, ...]:
    return (
        CandleSnapshot(
            time=utc(2026, 7, 10, 13, 55),
            open=2398.0,
            high=2401.0,
            low=2397.5,
            close=2400.35,
            tick_volume=120,
        ),
        CandleSnapshot(
            time=utc(2026, 7, 10, 14, 0),
            open=2400.35,
            high=2402.0,
            low=2399.8,
            close=2401.5,
            tick_volume=95,
        ),
    )


def _h1_snapshot() -> tuple[CandleSnapshot, ...]:
    return (
        CandleSnapshot(
            time=utc(2026, 7, 10, 13, 0),
            open=2390.0,
            high=2405.0,
            low=2388.0,
            close=2400.35,
            tick_volume=4200,
        ),
    )


async def test_returns_snapshot_and_decision_annotations(api, repository):
    repository.save(
        make_record(
            "1",
            m5_entry_snapshot=_m5_snapshot(),
            h1_entry_snapshot=_h1_snapshot(),
            reason="RBR base retest + M15 bullish engulf",
            confidence=0.82,
            zone_kind="demand",
            zone_price_low=2395.0,
            zone_price_high=2398.5,
            zone_time_start=utc(2026, 7, 10, 10, 0),
            zone_time_end=utc(2026, 7, 10, 13, 45),
            zone_pattern="RBR",
            pattern="bullish_engulfing",
            structure=(("HL", 2397.2, utc(2026, 7, 10, 13, 30)),),
            indicators=(("RSI", 62.3, 50.0, ">", True),),
        )
    )

    response = await api.get("/accounts/default/journal/trades/1/decision-context")

    assert response.status_code == 200
    body = response.json()
    assert body["trade_id"] == "1"
    assert body["symbol"] == "XAUUSD"
    assert body["side"] == "buy"
    assert body["open_price"] == 2400.35
    assert body["open_time"] == int(utc(2026, 7, 10, 14, 0).timestamp())
    assert len(body["entry_candles"]) == 2
    assert body["entry_candles"][0] == {
        "time": int(utc(2026, 7, 10, 13, 55).timestamp()),
        "open": 2398.0,
        "high": 2401.0,
        "low": 2397.5,
        "close": 2400.35,
        "tick_volume": 120,
    }
    assert len(body["higher_tf_candles"]) == 1
    assert body["zone"] == {
        "kind": "demand",
        "price_low": 2395.0,
        "price_high": 2398.5,
        "time_start": int(utc(2026, 7, 10, 10, 0).timestamp()),
        "time_end": int(utc(2026, 7, 10, 13, 45).timestamp()),
        "pattern": "RBR",
    }
    assert body["pattern"] == "bullish_engulfing"
    assert body["structure"] == [
        {"label": "HL", "price": 2397.2, "time": int(utc(2026, 7, 10, 13, 30).timestamp())}
    ]
    assert body["indicators"] == [
        {"name": "RSI", "value": 62.3, "threshold": 50.0, "comparison": ">", "passed": True}
    ]
    assert body["reason"] == "RBR base retest + M15 bullish engulf"
    assert body["confidence"] == 0.82


async def test_unknown_trade_id_returns_404(api):
    response = await api.get("/accounts/default/journal/trades/missing/decision-context")

    assert response.status_code == 404


async def test_trade_with_empty_snapshots_returns_200_with_empty_lists(api, repository):
    repository.save(make_record("2"))  # defaults: no snapshots, no zone/pattern/structure

    response = await api.get("/accounts/default/journal/trades/2/decision-context")

    assert response.status_code == 200
    body = response.json()
    assert body["entry_candles"] == []
    assert body["higher_tf_candles"] == []
    assert body["zone"] is None
    assert body["pattern"] is None
    assert body["structure"] == []
    assert body["indicators"] == []


async def test_scoped_to_account_missing_in_other_account(api, repository):
    repository.save(make_record("3"), account_id="ftmo-1")

    response = await api.get("/accounts/default/journal/trades/3/decision-context")

    assert response.status_code == 404
