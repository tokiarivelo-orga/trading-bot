"""Phase 9 end-to-end (paper mode, no real MT5/gateway): wires the real
event bus, `PaperBroker`, `OrderService`, `TradeEngine`, `ReconciliationService`
and `AlertService` the same way `container.py` does, then exercises the
kill switch to confirm the full alert fan-out and circuit-breaker path fire
together — not just each piece in isolation (already covered by the unit
tests in tests/unit/engine, tests/unit/broker, tests/unit/alerting).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.alerting.adapters.composite import CompositeAlertAdapter
from src.alerting.application.alert_service import AlertService
from src.alerting.domain.models import AlertingConfig, AlertMessage
from src.broker.adapters.paper import PaperBroker
from src.broker.application.order_service import OrderService
from src.broker.application.reconciliation import ReconciliationService
from src.broker.application.spread_gate import SpreadGate
from src.broker.domain.symbol_config import SymbolTradingConfig
from src.broker.domain.trading import Side
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager
from src.engine.application.trade_loop import TradeEngine
from src.engine.domain.models import RiskCaps
from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import MarketSnapshot
from src.market_data.domain.models import SymbolInfo
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CircuitBreakerTripped, PositionClosed, PositionOpened
from src.skills.ports.skill_selector import SkillDecision

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


class FakeMarketData:
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(
            symbol=symbol,
            bid=2400.00,
            ask=2400.30,
            spread_points=30,
            point=0.01,
            digits=2,
            stops_level=0,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=50.0,
            volume_step=0.01,
        )

    async def get_candles(self, symbol, timeframe, count, before=None):
        return []


class FakeMarketContext:
    async def capture(self, symbol):
        return MarketSnapshot(m5=(), h1=())


class AlwaysAllowSkillSelector:
    def select(self, symbol, now=None):
        return SkillDecision(allowed=True, skill_name="normal", strategy_name="none")


class FakeAlertPort:
    def __init__(self) -> None:
        self.sent: list[AlertMessage] = []

    async def send(self, message: AlertMessage) -> None:
        self.sent.append(message)


class FakeAccount:
    def __init__(self, balance: float = 10_000.0) -> None:
        self._balance = balance

    async def status(self):
        return {"account": {"balance": self._balance}}


async def test_kill_switch_closes_paper_position_and_fires_alerts(tmp_path):
    market_data = FakeMarketData()
    broker = PaperBroker(market_data)
    event_bus = EventBus()
    spread_gate = SpreadGate({"XAUUSD": XAUUSD_CONFIG})
    order_service = OrderService(
        broker=broker, market_data=market_data, spread_gate=spread_gate, event_bus=event_bus
    )

    db_engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(db_engine)
    journal_repository = JournalRepository(sessionmaker(bind=db_engine, expire_on_commit=False))
    trade_journal = TradeJournalService(
        repository=journal_repository,
        market_context=FakeMarketContext(),
        event_bus=event_bus,
    )
    event_bus.subscribe(PositionOpened, trade_journal.on_position_opened)
    event_bus.subscribe(PositionClosed, trade_journal.on_position_closed)

    alert_port = FakeAlertPort()
    alert_service = AlertService(port=CompositeAlertAdapter([alert_port]), config=AlertingConfig())
    event_bus.subscribe(PositionOpened, alert_service.on_position_opened)
    event_bus.subscribe(PositionClosed, alert_service.on_position_closed)
    event_bus.subscribe(CircuitBreakerTripped, alert_service.on_circuit_breaker_tripped)

    reconciliation = ReconciliationService(
        broker=broker, journal=trade_journal, event_bus=event_bus
    )
    position_manager = PositionManager(
        order_service=order_service, market_data=market_data, reconciliation=reconciliation
    )
    risk_manager = RiskManager(
        caps=RiskCaps(
            risk_per_trade_pct=1.0,
            daily_loss_limit_pct=5.0,
            max_open_positions=5,
            max_trades_per_day_enabled=False,
            consecutive_loss_pause=5,
        ),
        timezone="UTC",
    )
    engine = TradeEngine(
        market_data=market_data,
        order_service=order_service,
        account=FakeAccount(),
        risk_manager=risk_manager,
        position_manager=position_manager,
        skill_selector=AlwaysAllowSkillSelector(),
        strategy_source=type("S", (), {"get": staticmethod(lambda name: None)})(),
        entry_timeframe="M5",
        event_bus=event_bus,
    )

    # Open a paper position directly through the same OrderService the
    # engine uses, so the journal has a real open TradeRecord to reconcile
    # against (mirrors how a strategy-driven entry would land here).
    await order_service.open_position("XAUUSD", Side.BUY, 0.1, sl=2390.0, tp=2420.0)
    open_trades = await trade_journal.get_open_trades("XAUUSD")
    assert len(open_trades) == 1

    await engine.kill_switch()

    # Position closed for real through the paper broker...
    assert (await order_service.get_positions("XAUUSD")) == []
    # ...journaled as closed (not left dangling open)...
    assert await trade_journal.get_open_trades("XAUUSD") == []
    # ...engine paused...
    assert risk_manager.paused
    # ...and both the fill alert and the circuit-breaker alert fired.
    titles = [m.title for m in alert_port.sent]
    assert any("Closed XAUUSD" in t for t in titles)
    assert any("Engine paused" in t for t in titles)
