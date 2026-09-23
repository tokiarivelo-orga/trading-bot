"""Phase 4 end-to-end (paper mode, no real MT5): a CandleClosed(M5) event
drives the full engine pipe — skill selection, the real breakout_v1 strategy,
HTF confirmation, risk sizing, and order placement through the broker API —
then confirms the trade lands in the journal with strategy/skill attribution.

Mirrors tests/integration/test_phase3_broker_flow.py's fake-gateway approach.
"""

from __future__ import annotations

import time

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.broker.adapters.paper import PaperBroker
from src.broker.api.routes import router as account_router
from src.broker.api.trading_routes import router as trading_router
from src.broker.application.account_service import AccountService
from src.broker.application.order_service import OrderService
from src.broker.application.spread_gate import SpreadGate
from src.broker.domain.symbol_config import SymbolTradingConfig
from src.engine.api.routes import router as engine_router
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager
from src.engine.application.trade_loop import TradeEngine
from src.engine.domain.models import RiskCaps
from src.journal.adapters.market_context import CandleRepositoryMarketContext
from src.journal.adapters.repository import JournalRepository
from src.journal.api.routes import router as journal_router
from src.journal.application.trade_journal import TradeJournalService
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.market_data.api.routes import router as market_data_router
from src.market_data.api.ws import WsBroadcaster
from src.market_data.application.candle_stream import CandleStreamService
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import Timeframe
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed, PositionClosed, PositionOpened
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import NormalSkill
from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec
from src.strategies.generated.breakout_v1 import BreakoutV1
from src.strategies.registry import StrategyRegistry

M5 = 300
XAUUSD_CONFIG = SymbolTradingConfig(
    symbol="XAUUSD",
    max_spread_points=35,
    min_rr=1.5,
    contract_size=100,
    point=0.01,
    digits=2,
    stops_level=0,
    volume_min=0.01,
    volume_max=50,
    volume_step=0.01,
)
RISK_CAPS = RiskCaps(
    risk_per_trade_pct=0.5,
    daily_loss_limit_pct=2.0,
    max_open_positions=2,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=5,
)


def make_fake_gateway() -> FastAPI:
    """M5 candles form a clean 20-bar range followed by a breakout bar; every
    other timeframe (including each bot's own HTF-veto timeframe) only gets a
    handful of bars so `mtf_confirm` skips (insufficient history) instead of
    vetoing on trend."""
    gw = FastAPI()
    account = {
        "login": 123456,
        "server": "Demo-Server",
        "name": "Test User",
        "currency": "USD",
        "balance": 10_000.0,
        "equity": 10_000.0,
        "leverage": 100,
    }

    @gw.get("/health")
    def health():
        return {"status": "ok", "terminal_connected": True, "account": account}

    @gw.get("/candles")
    def candles(symbol: str, timeframe: str, count: int = 300):
        latest_open = int(time.time()) // M5 * M5
        if timeframe == "M5":
            n = 21
            bars = [
                {
                    "time": latest_open - (n - 1 - i) * M5,
                    "open": 2400.0,
                    "high": 2401.0,
                    "low": 2399.0,
                    "close": 2400.0,
                    "tick_volume": 1000,
                    "spread": 25,
                }
                for i in range(n - 1)
            ]
            bars.append(
                {
                    "time": latest_open - M5,
                    "open": 2401.0,
                    "high": 2411.0,
                    "low": 2400.5,
                    "close": 2410.0,
                    "tick_volume": 1500,
                    "spread": 25,
                }
            )
            return bars
        n = 5
        return [
            {
                "time": latest_open - (n - 1 - i) * M5,
                "open": 2400.0 + i,
                "high": 2401.0 + i,
                "low": 2399.0 + i,
                "close": 2400.5 + i,
                "tick_volume": 1000,
                "spread": 25,
            }
            for i in range(n)
        ]

    @gw.get("/symbol_info")
    def symbol_info(symbol: str):
        return {
            "symbol": symbol,
            "bid": 2400.10,
            "ask": 2400.35,
            "spread_points": 25,
            "point": 0.01,
            "digits": 2,
            "stops_level": 10,
            "contract_size": 100.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }

    return gw


class M1ScalpProbe:
    """M1-entry probe bot: signals only when the engine actually handed it M1
    entry bars and its own M5 confirmation bars — the exact data an M1 scalp
    strategy (e.g. rbr_dbd_zones_scalp_*) needs live. Never fires on the M5
    bot's closes, so the existing M5-driven tests still see one position."""

    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="m1_probe",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M1",
            confirmation_timeframes=("M5",),
            params={},
        )

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        if ctx.candles.get("M1") is None or ctx.candles.get("M5") is None:
            return None
        return Signal(direction=Direction.BUY, sl_points=10.0, tp_points=17.0, reason="m1 probe")


class ContainerForTest:
    def __init__(self, tmp_path):
        gateway_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_fake_gateway()), base_url="http://gw"
        )
        engine = create_engine(f"sqlite:///{tmp_path}/test.db")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)

        self.event_bus = EventBus()
        self.symbols = ["XAUUSD"]
        self.gateway_client = gateway_client
        self.market_data = GatewayMarketData(gateway_client)
        candle_repository = CandleRepository(session_factory)
        self.ws_broadcaster = WsBroadcaster("default")
        self.candle_history = CandleHistoryService(self.market_data, candle_repository)
        self.candle_stream = CandleStreamService(
            market_data=self.market_data,
            repository=candle_repository,
            event_bus=self.event_bus,
            broadcaster=self.ws_broadcaster,
            symbols=self.symbols,
            timeframes=[Timeframe.M5],
        )

        broker = PaperBroker(self.market_data)
        spread_gate = SpreadGate({"XAUUSD": XAUUSD_CONFIG})
        self.order_service = OrderService(
            broker=broker,
            market_data=self.market_data,
            spread_gate=spread_gate,
            event_bus=self.event_bus,
        )
        self.account = AccountService(gateway=_FakeAccountGateway(), store=_NullStore())

        journal_repository = JournalRepository(session_factory)
        market_context = CandleRepositoryMarketContext(candle_repository)
        self.trade_journal = TradeJournalService(
            repository=journal_repository,
            market_context=market_context,
            event_bus=self.event_bus,
            review_every_n_trades=10,
        )
        self.event_bus.subscribe(PositionOpened, self.trade_journal.on_position_opened)
        self.event_bus.subscribe(PositionClosed, self.trade_journal.on_position_closed)

        risk_manager = RiskManager(caps=RISK_CAPS, timezone="UTC")
        self.risk_manager = risk_manager
        position_manager = PositionManager(self.order_service, self.market_data)
        self.position_manager = position_manager
        strategy_registry = StrategyRegistry()
        strategy_registry.register("breakout_v1", BreakoutV1())
        strategy_registry.register("m1_probe", M1ScalpProbe())
        skill_selector = SkillSelector(
            skills={
                "XAUUSD": [
                    NormalSkill(
                        name="normal/xauusd/breakout_v1",
                        symbol="XAUUSD",
                        strategy="breakout_v1",
                        sessions=(),
                    ),
                    NormalSkill(
                        name="normal/xauusd/m1_probe",
                        symbol="XAUUSD",
                        strategy="m1_probe",
                        sessions=(),
                    ),
                ]
            },
            timezone="UTC",
        )
        self.trade_engine = TradeEngine(
            market_data=self.market_data,
            order_service=self.order_service,
            account=self.account,
            risk_manager=risk_manager,
            position_manager=position_manager,
            skill_selector=skill_selector,
            strategy_source=strategy_registry,
            entry_timeframe="M5",
            context_bars=30,
        )
        self.event_bus.subscribe(CandleClosed, self.trade_engine.on_candle_closed)
        self.event_bus.subscribe(PositionClosed, self.trade_engine.on_position_closed)

        # Resolves `get_account_runtime`'s `container.accounts[account_id]` lookup —
        # this flat object already carries every AccountRuntime-scoped field the
        # per-account routes need, so it doubles as its own single-entry registry.
        self.accounts = {"default": self}


class _FakeAccountGateway:
    async def login(self, credentials):  # pragma: no cover - unused in this test
        raise NotImplementedError

    async def logout(self) -> None:  # pragma: no cover - unused in this test
        raise NotImplementedError

    async def health(self):
        from src.broker.domain.account import AccountInfo, GatewayHealth

        return GatewayHealth(
            gateway_up=True,
            terminal_connected=True,
            account=AccountInfo(
                login=123456,
                server="Demo-Server",
                name="Test User",
                currency="USD",
                balance=10_000.0,
                equity=10_000.0,
                leverage=100,
            ),
        )


class _NullStore:
    def save(self, credentials) -> None:  # pragma: no cover - unused in this test
        pass

    def load(self):
        return None

    def clear(self) -> None:  # pragma: no cover - unused in this test
        pass


@pytest.fixture
async def api(tmp_path):
    app = FastAPI()
    app.include_router(account_router)
    app.include_router(market_data_router)
    app.include_router(trading_router)
    app.include_router(journal_router)
    app.include_router(engine_router)
    container = ContainerForTest(tmp_path)
    app.state.container = container
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        client.container = container
        yield client


async def test_candle_close_drives_full_entry_through_the_engine(api):
    await api.container.trade_engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    positions = (await api.get("/accounts/default/broker/positions")).json()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "XAUUSD"
    assert positions[0]["side"] == "buy"

    trades = (await api.get("/accounts/default/journal/trades", params={"symbol": "XAUUSD"})).json()
    assert len(trades) == 1
    assert trades[0]["strategy_version"] == "breakout_v1:v1"
    assert trades[0]["skill"] == "normal/xauusd/breakout_v1"

    status = (await api.get("/accounts/default/engine/status")).json()
    assert status["trades_today"] == 1
    assert not status["paused"]


async def test_m1_scalp_bot_enters_on_m1_close_through_paper_broker(api):
    # An M1 candle close must drive the M1-entry bot (and only it) through
    # the same full paper pipe — skill routing, evaluation with M1 context,
    # risk sizing, paper order, journal attribution.
    await api.container.trade_engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M1"))

    positions = (await api.get("/accounts/default/broker/positions")).json()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "XAUUSD"

    trades = (await api.get("/accounts/default/journal/trades", params={"symbol": "XAUUSD"})).json()
    assert len(trades) == 1
    assert trades[0]["strategy_version"] == "m1_probe:v1"
    assert trades[0]["skill"] == "normal/xauusd/m1_probe"


async def test_kill_switch_endpoint_pauses_and_closes_positions(api):
    await api.container.trade_engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))
    assert len((await api.get("/accounts/default/broker/positions")).json()) == 1

    killed = await api.post("/accounts/default/engine/kill")
    assert killed.status_code == 200
    assert killed.json()["paused"] is True

    assert (await api.get("/accounts/default/broker/positions")).json() == []

    resumed = await api.post("/accounts/default/engine/resume")
    assert resumed.json()["paused"] is False


async def test_get_risk_caps_reflects_configured_caps(api):
    caps = (await api.get("/accounts/default/engine/risk-caps")).json()
    assert caps["risk_per_trade_pct"] == RISK_CAPS.risk_per_trade_pct
    assert caps["min_lot_fallback_enabled"] is False
    assert caps["max_risk_per_trade_pct"] is None


async def test_update_min_lot_fallback_takes_effect_live(api):
    updated = await api.put(
        "/accounts/default/engine/risk-caps/min-lot-fallback",
        json={"enabled": True, "max_risk_per_trade_pct": 5.0},
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["min_lot_fallback_enabled"] is True
    assert body["max_risk_per_trade_pct"] == 5.0
    # Every other cap is untouched by the live update.
    assert body["risk_per_trade_pct"] == RISK_CAPS.risk_per_trade_pct
    assert body["max_open_positions"] == RISK_CAPS.max_open_positions

    # The running RiskManager (not just the API's echo) actually changed.
    assert api.container.risk_manager.caps.min_lot_fallback_enabled is True

    again = (await api.get("/accounts/default/engine/risk-caps")).json()
    assert again["min_lot_fallback_enabled"] is True


async def test_update_max_trades_per_day_enabled_takes_effect_live(api):
    assert RISK_CAPS.max_trades_per_day_enabled is False

    updated = await api.put(
        "/accounts/default/engine/risk-caps/max-trades-per-day-enabled",
        json={"enabled": True},
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["max_trades_per_day_enabled"] is True
    # Every other cap is untouched by the live update.
    assert body["risk_per_trade_pct"] == RISK_CAPS.risk_per_trade_pct
    assert body["max_open_positions"] == RISK_CAPS.max_open_positions

    # The running RiskManager (not just the API's echo) actually changed,
    # and it now rejects the very next pretrade check.
    risk_manager = api.container.risk_manager
    assert risk_manager.caps.max_trades_per_day_enabled is True
    decision = risk_manager.check_pretrade(open_positions_count=0)
    assert not decision.approved
    assert "max_trades_per_day_enabled" in decision.reason

    disabled = await api.put(
        "/accounts/default/engine/risk-caps/max-trades-per-day-enabled",
        json={"enabled": False},
    )
    assert disabled.json()["max_trades_per_day_enabled"] is False


async def test_update_core_risk_caps_partial_update_takes_effect_live(api):
    updated = await api.put(
        "/accounts/default/engine/risk-caps/core",
        json={"risk_per_trade_pct": 1.5, "max_open_positions": 3},
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["risk_per_trade_pct"] == 1.5
    assert body["max_open_positions"] == 3
    # Fields not sent are untouched.
    assert body["daily_loss_limit_pct"] == RISK_CAPS.daily_loss_limit_pct
    assert body["consecutive_loss_pause"] == RISK_CAPS.consecutive_loss_pause

    # The running RiskManager (not just the API's echo) actually changed.
    risk_manager = api.container.risk_manager
    assert risk_manager.caps.risk_per_trade_pct == 1.5
    assert risk_manager.caps.max_open_positions == 3

    # Free to loosen too — not restricted to tighten-only like apply_risk_override.
    loosened = await api.put(
        "/accounts/default/engine/risk-caps/core",
        json={"risk_per_trade_pct": 5.0, "max_open_positions": 50},
    )
    assert loosened.json()["risk_per_trade_pct"] == 5.0
    assert loosened.json()["max_open_positions"] == 50


async def test_volatility_guard_endpoints_are_gone(api):
    # The volatility guard was removed entirely — no config to read, no
    # switch to flip.
    assert (await api.get("/accounts/default/engine/volatility-config")).status_code == 404
    updated = await api.put(
        "/accounts/default/engine/volatility-config/enabled", json={"enabled": False}
    )
    assert updated.status_code in (404, 405)
    assert not hasattr(api.container.trade_engine, "set_volatility_guard_enabled")
    assert not hasattr(api.container.position_manager, "set_volatility_guard_enabled")


async def test_engine_keeps_entering_past_max_open_positions_while_the_cap_is_bypassed(api):
    # The pre-trade risk gate, max_open_positions included, is intentionally
    # bypassed (trade_loop.py: "ENTRY BLOCKED (risk gate): ... BYPASSED PER
    # USER REQUEST"). RISK_CAPS allows 2 open positions, yet every repeat of
    # the same breakout candle opens one more.
    for expected in (1, 2, 3):
        await api.container.trade_engine.on_candle_closed(
            CandleClosed(symbol="XAUUSD", timeframe="M5")
        )
        positions = (await api.get("/accounts/default/broker/positions")).json()
        assert len(positions) == expected
    assert RISK_CAPS.max_open_positions == 2
