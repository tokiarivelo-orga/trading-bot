"""Phase 5 end-to-end (paper mode, no real MT5): a CandleClosed(M5) event
drives the full engine pipe — skill selection, the real breakout_v1
strategy, HTF confirmation, risk sizing, and order placement through
`OrderService`/`PaperBroker` — then confirms the two Phase 5 deliverables
that only show up once the whole money path runs together:

1. The `signal_id` correlation id lands on the *persisted* activity-log rows
   for both the `SIGNAL:` line (`engine/application/trade_loop.py`) and the
   `ENTRY OPENED:` line (`broker/application/order_service.py`) — the same
   id on both, proving the ContextVar binding survives the
   `QueueListener` background-thread hop (see `test_log_handler.py`'s
   regression test for that bug in isolation).
2. `GET /metrics` reports the fill as `tradingbot_signal_outcomes_total{...,
   outcome="opened"}`, counts the fired signal, and reflects the open
   position in `tradingbot_open_positions`.

Wiring mirrors `tests/integration/test_phase4_engine_flow.py`'s fake-gateway
approach, trimmed to what this phase's assertions need.
"""

from __future__ import annotations

import logging
import time

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.activity.adapters.log_handler import attach_activity_log_handler
from src.activity.adapters.repository import ActivityLogRepository
from src.broker.adapters.paper import PaperBroker
from src.broker.application.account_service import AccountService
from src.broker.application.order_service import OrderService
from src.broker.application.spread_gate import SpreadGate
from src.broker.domain.account import AccountInfo, GatewayHealth
from src.broker.domain.symbol_config import SymbolTradingConfig
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager
from src.engine.application.trade_loop import TradeEngine
from src.engine.domain.models import RiskCaps
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed, PositionOpened
from src.shared.metrics.registry import REGISTRY, position_opened
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import NormalSkill
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
    """Same shape as test_phase4_engine_flow.py's: a clean 20-bar M5 range
    followed by a breakout bar, so breakout_v1 fires exactly one BUY."""
    gw = FastAPI()

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


class _FakeAccountGateway:
    async def health(self) -> GatewayHealth:
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
    def load(self):
        return None


@pytest.fixture
def wired(tmp_path):
    """Real TradeEngine -> OrderService -> PaperBroker over a fake gateway,
    plus a real DB-backed activity log handler so the persisted `signal_id`
    can be asserted on, not just the in-memory objects."""
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_fake_gateway()), base_url="http://gw"
    )
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    market_data = GatewayMarketData(gateway_client)
    event_bus = EventBus()

    broker = PaperBroker(market_data)
    spread_gate = SpreadGate({"XAUUSD": XAUUSD_CONFIG})
    order_service = OrderService(
        broker=broker, market_data=market_data, spread_gate=spread_gate, event_bus=event_bus
    )
    account = AccountService(gateway=_FakeAccountGateway(), store=_NullStore())

    risk_manager = RiskManager(caps=RISK_CAPS, timezone="UTC")
    position_manager = PositionManager(order_service, market_data)
    strategy_registry = StrategyRegistry()
    strategy_registry.register("breakout_v1", BreakoutV1())
    skill_selector = SkillSelector(
        skills={
            "XAUUSD": [
                NormalSkill(
                    name="normal/xauusd/breakout_v1",
                    symbol="XAUUSD",
                    strategy="breakout_v1",
                    sessions=(),
                )
            ]
        },
        timezone="UTC",
    )
    trade_engine = TradeEngine(
        market_data=market_data,
        order_service=order_service,
        account=account,
        risk_manager=risk_manager,
        position_manager=position_manager,
        skill_selector=skill_selector,
        strategy_source=strategy_registry,
        entry_timeframe="M5",
        context_bars=30,
    )
    event_bus.subscribe(CandleClosed, trade_engine.on_candle_closed)

    # Mirrors container.py's `_on_position_opened_metric` wiring — a real
    # deployment's `tradingbot_open_positions` gauge is driven off this same
    # event-bus subscription, not counted at scrape time.
    async def _on_position_opened_metric(_event: PositionOpened) -> None:
        position_opened(account_id="default")

    event_bus.subscribe(PositionOpened, _on_position_opened_metric)

    activity_repository = ActivityLogRepository(session_factory)
    return trade_engine, order_service, activity_repository


def _drain(activity_repository, **search_kwargs):
    """The DB write runs on `QueueListener`'s background thread — give it a
    beat, same pattern as `test_log_handler.py`."""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        entries, total = activity_repository.search(**search_kwargs)
        if total:
            return entries
        time.sleep(0.05)
    return []


async def test_signal_id_joins_the_signal_and_fill_log_lines(wired):
    trade_engine, _order_service, activity_repository = wired
    listener = attach_activity_log_handler(activity_repository)
    logging.getLogger("src").setLevel(logging.INFO)
    try:
        await trade_engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

        signal_rows = _drain(activity_repository, logger_contains="trade_loop", q="SIGNAL: XAUUSD")
        fill_rows = _drain(activity_repository, logger_contains="order_service", q="ENTRY OPENED")
    finally:
        listener.stop()
        logging.getLogger("src").handlers.clear()
        logging.getLogger("src").setLevel(logging.NOTSET)

    assert len(signal_rows) == 1
    assert len(fill_rows) == 1
    assert signal_rows[0].signal_id is not None
    # Same correlation id joins the signal to the fill it produced — the
    # concrete "signal -> sizing -> order -> fill" chain OBSERVABILITY_PLAN.md
    # Phase 5 asks for.
    assert signal_rows[0].signal_id == fill_rows[0].signal_id


async def test_metrics_reflect_the_fill(wired):
    trade_engine, _order_service, _activity_repository = wired

    await trade_engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    from prometheus_client import generate_latest

    text = generate_latest(REGISTRY).decode()
    assert (
        'tradingbot_signals_total{account_id="default",'
        'bot="normal/xauusd/breakout_v1",symbol="XAUUSD"}' in text
    )
    assert 'tradingbot_signal_outcomes_total{account_id="default",outcome="opened"}' in text
    assert 'tradingbot_open_positions{account_id="default"}' in text
