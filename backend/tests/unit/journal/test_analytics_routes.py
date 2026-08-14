"""GET /journal/analytics/symbols and /journal/analytics/bots."""

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
        raise AssertionError("market context should not be hit by analytics")


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
            skill="normal/xauusd/breakout_v1",
            strategy_version="breakout_v1:v1",
            open_time=utc(2026, 7, 10, 14, 0),
            close_time=utc(2026, 7, 10, 15, 0),
            profit=10.0,
        )
    )
    repository.save(
        make_record(
            "2",
            symbol="XAUUSD",
            skill="normal/xauusd/breakout_v1",
            strategy_version="breakout_v1:v1",
            open_time=utc(2026, 7, 10, 15, 0),
            close_time=utc(2026, 7, 10, 16, 0),
            profit=-4.0,
        )
    )
    repository.save(
        make_record(
            "3",
            symbol="EURUSD",
            skill=None,
            open_time=utc(2026, 7, 10, 16, 0),
            close_time=utc(2026, 7, 10, 17, 0),
            profit=2.0,
        )
    )


async def test_symbol_analytics_returns_one_entry_per_symbol(api):
    response = await api.get("/accounts/default/journal/analytics/symbols")

    assert response.status_code == 200
    body = response.json()
    assert {r["symbol"] for r in body} == {"XAUUSD", "EURUSD"}
    xau = next(r for r in body if r["symbol"] == "XAUUSD")
    assert xau["closed_count"] == 2
    assert xau["win_count"] == 1
    assert xau["loss_count"] == 1
    assert xau["total_profit"] == 6.0
    assert xau["profit_factor"] == 2.5


async def test_bot_analytics_excludes_manual_trades_and_includes_equity_curve(api):
    response = await api.get("/accounts/default/journal/analytics/bots")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    bot = body[0]
    assert bot["skill"] == "normal/xauusd/breakout_v1"
    assert bot["bot_name"] == "breakout_v1"
    assert bot["total_profit"] == 6.0
    assert [p["cumulative_profit"] for p in bot["equity_curve"]] == [10.0, 6.0]


async def test_analytics_endpoints_empty_when_no_trades(api, repository, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
    Base.metadata.create_all(engine)
    empty_repo = JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))

    from src.journal.application.trade_journal import TradeJournalService

    empty_journal = TradeJournalService(
        repository=empty_repo, market_context=FakeMarketContext(), event_bus=EventBus()
    )
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={"default": SimpleNamespace(trade_journal=empty_journal)}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        symbols = await client.get("/accounts/default/journal/analytics/symbols")
        bots = await client.get("/accounts/default/journal/analytics/bots")

    assert symbols.json() == []
    assert bots.json() == []


# ── regime analytics (OBSERVABILITY_PLAN.md Phase 6) ────────────────────────


async def test_bot_analytics_includes_cost_fields(api, repository):
    repository.save(
        make_record(
            "10",
            symbol="XAUUSD",
            skill="normal/xauusd/breakout_v1",
            open_time=utc(2026, 7, 10, 18, 0),
            close_time=utc(2026, 7, 10, 19, 0),
            profit=10.0,
            transaction_cost=2.0,
        )
    )

    response = await api.get("/accounts/default/journal/analytics/bots")

    assert response.status_code == 200
    bot = next(b for b in response.json() if b["skill"] == "normal/xauusd/breakout_v1")
    assert bot["total_transaction_cost"] == 2.0
    assert bot["avg_transaction_cost_per_trade"] == 2.0
    # gross_edge = total_profit (6.0 seeded + 10.0 here = 16.0) + cost (2.0) = 18.0
    assert bot["cost_pct_of_gross_edge"] == pytest.approx(2.0 / 18.0)


async def test_regime_analytics_returns_one_entry_per_bot_dimension_bucket(api, repository):
    repository.save(
        make_record(
            "10",
            symbol="XAUUSD",
            skill="normal/xauusd/breakout_v1",
            open_time=utc(2026, 7, 10, 18, 0),
            close_time=utc(2026, 7, 10, 19, 0),
            profit=5.0,
            regime_volatility="high",
            regime_trend="trending",
            regime_session="london",
        )
    )
    repository.save(
        make_record(
            "11",
            symbol="XAUUSD",
            skill="normal/xauusd/breakout_v1",
            open_time=utc(2026, 7, 10, 19, 0),
            close_time=utc(2026, 7, 10, 20, 0),
            profit=-3.0,
            regime_volatility="low",
            regime_trend="ranging",
            regime_session="asian",
        )
    )

    response = await api.get("/accounts/default/journal/analytics/regimes")

    assert response.status_code == 200
    body = response.json()
    by_key = {(r["dimension"], r["bucket"]): r for r in body}
    assert by_key[("volatility", "high")]["trade_count"] == 1
    assert by_key[("volatility", "high")]["total_profit"] == 5.0
    assert by_key[("volatility", "low")]["trade_count"] == 1
    assert by_key[("trend", "trending")]["bot_name"] == "breakout_v1"
    assert by_key[("session", "london")]["skill"] == "normal/xauusd/breakout_v1"
    # None of the auto-seeded fixture trades carry a regime tag, so they
    # contribute no bucket at all — only the two trades seeded here do.
    assert {r["dimension"] for r in body} == {"volatility", "trend", "session"}


async def test_regime_analytics_empty_when_no_trades_are_tagged(api):
    """The three fixture-seeded trades carry no regime tag at all."""
    response = await api.get("/accounts/default/journal/analytics/regimes")

    assert response.status_code == 200
    assert response.json() == []


async def test_analytics_endpoints_filter_by_open_from_and_open_to(api):
    t_15 = int(utc(2026, 7, 10, 15, 0).timestamp())

    res = await api.get(f"/accounts/default/journal/analytics/symbols?open_from={t_15}")
    assert res.status_code == 200
    xau = next(r for r in res.json() if r["symbol"] == "XAUUSD")
    assert xau["closed_count"] == 1
    assert xau["total_profit"] == -4.0

    res_bots = await api.get(f"/accounts/default/journal/analytics/bots?open_to={t_15}")
    assert res_bots.status_code == 200
    assert len(res_bots.json()) == 1
    bot = res_bots.json()[0]
    assert bot["trade_count"] == 2


# ── daily P&L (Phase 6 Part A) ───────────────────────────────────────────────


async def test_daily_pnl_aggregates_the_seeded_day(api):
    """All three fixture-seeded trades close on 2026-07-10: +10 -4 +2 = 8."""
    response = await api.get("/accounts/default/journal/analytics/daily")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    day = body[0]
    assert day["date"] == "2026-07-10"
    assert day["pnl"] == 8.0
    assert day["trade_count"] == 3
    assert day["win_count"] == 2
    assert day["loss_count"] == 1


async def test_daily_pnl_empty_when_no_trades(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/empty_daily.db")
    Base.metadata.create_all(engine)
    empty_repo = JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))

    from src.journal.application.trade_journal import TradeJournalService

    empty_journal = TradeJournalService(
        repository=empty_repo, market_context=FakeMarketContext(), event_bus=EventBus()
    )
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={"default": SimpleNamespace(trade_journal=empty_journal)}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        response = await client.get("/accounts/default/journal/analytics/daily")

    assert response.json() == []
