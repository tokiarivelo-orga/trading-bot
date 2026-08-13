"""Order-book capture wired into the live trade loop (order_book/ Phase 5),
end to end (paper mode, no real MT5): a `CandleClosed(M5)` event drives the
full engine pipe — skill selection, the real `breakout_v1` strategy, HTF
confirmation, risk sizing, and order placement through
`OrderService`/`PaperBroker` — for two symbols on the same candle close, one
whose `OrderBookCapturePort` fake reports real depth and one that reports
none, plus a third pass whose fake raises. Confirms the three Phase 5
deliverables that only show up once the whole money path runs together:

1. Exactly one `order_book_snapshots` row exists for the symbol given depth,
   and its `signal_id` matches the `signal_id` on the `signal_decisions` row
   for that same signal (the join key described in
   `order_book/application/capture.py`/`activity/adapters/signal_decision_
   repository.py`).
2. No row exists for the no-depth symbol — absence is the graceful-
   degradation signal, not an empty-but-present row (same contract
   `OrderBookCaptureService` implements).
3. The trade still opens and gets journaled normally regardless of what the
   capture port does — including when it raises — proving order-book
   capture (or its failure) never blocks the money path. This is
   `TradeEngine._capture_order_book_safely`'s contract, exercised end to end
   rather than in isolation.

Wiring mirrors `tests/integration/test_phase5_observability_flow.py`'s
fake-gateway approach, trimmed to what this phase's assertions need, plus a
real `signal_decisions`/`trade_journal` wiring so the join-key assertions
have real DB rows to read.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.activity.adapters.orm import SignalDecisionRow
from src.activity.adapters.signal_decision_repository import SignalDecisionRepository
from src.activity.application.signal_decision_service import SignalDecisionService
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
from src.engine.domain.volatility import VolatilityConfig
from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import MarketSnapshot
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.order_book.adapters.orm import OrderBookSnapshotRow
from src.order_book.adapters.repository import OrderBookSnapshotRepository
from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot
from src.shared.db.base import Base
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed, PositionOpened
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import NormalSkill
from src.strategies.generated.breakout_v1 import BreakoutV1
from src.strategies.registry import StrategyRegistry

M5 = 300
DEPTH_SYMBOL = "XAUUSD"
NO_DEPTH_SYMBOL = "VIX75"

_SYMBOL_CONFIG_KWARGS = dict(
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
    max_open_positions=5,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=5,
)


def make_fake_gateway() -> FastAPI:
    """Same shape as test_phase5_observability_flow.py's: a clean 20-bar M5
    range followed by a breakout bar, so breakout_v1 fires exactly one BUY —
    ignores `symbol`, so both `DEPTH_SYMBOL` and `NO_DEPTH_SYMBOL` see the
    same breakout shape."""
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
                    "real_volume": 0,
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
                    "real_volume": 0,
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
                "real_volume": 0,
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


class FakeOrderBookCapture:
    """Not a live gateway: `depth_symbols` decides, per symbol, whether a
    canned snapshot is written through the *real* repository (simulating
    depth) or nothing happens at all (simulating "no depth" — same
    absence-is-the-signal contract `OrderBookCaptureService` implements).
    `raise_symbols` simulates the port violating its own never-raises
    contract, to prove the engine's fire-and-forget wrapper still isolates
    that from the money path."""

    def __init__(
        self,
        repository: OrderBookSnapshotRepository,
        depth_symbols: set[str],
        raise_symbols: set[str] = frozenset(),
        account_id: str = "default",
    ) -> None:
        self._repository = repository
        self._depth_symbols = depth_symbols
        self._raise_symbols = raise_symbols
        self._account_id = account_id
        self.calls: list[tuple[str, str]] = []

    async def capture(self, *, signal_id: str, symbol: str) -> None:
        self.calls.append((signal_id, symbol))
        if symbol in self._raise_symbols:
            raise RuntimeError(f"order-book gateway blew up for {symbol}")
        if symbol not in self._depth_symbols:
            return
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            time=datetime.now(UTC),
            levels=(
                BookLevel(side=BookSide.BID, price=2400.00, volume=1.5),
                BookLevel(side=BookSide.ASK, price=2400.30, volume=1.2),
            ),
        )
        await asyncio.to_thread(self._repository.save, signal_id, snapshot, self._account_id)


@pytest.fixture
def wired(tmp_path):
    """Real TradeEngine -> OrderService -> PaperBroker over a fake gateway,
    plus real DB-backed signal-decision and journal repositories so the
    join-key assertions have real rows to read."""
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_fake_gateway()), base_url="http://gw"
    )
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    market_data = GatewayMarketData(gateway_client)
    event_bus = EventBus()

    broker = PaperBroker(market_data)
    spread_gate = SpreadGate(
        {
            DEPTH_SYMBOL: SymbolTradingConfig(symbol=DEPTH_SYMBOL, **_SYMBOL_CONFIG_KWARGS),
            NO_DEPTH_SYMBOL: SymbolTradingConfig(symbol=NO_DEPTH_SYMBOL, **_SYMBOL_CONFIG_KWARGS),
        }
    )
    signal_decision_repository = SignalDecisionRepository(session_factory)
    signal_decisions = SignalDecisionService(signal_decision_repository)
    order_service = OrderService(
        broker=broker,
        market_data=market_data,
        spread_gate=spread_gate,
        event_bus=event_bus,
        signal_decisions=signal_decisions,
    )
    account = AccountService(gateway=_FakeAccountGateway(), store=_NullStore())

    risk_manager = RiskManager(caps=RISK_CAPS, timezone="UTC")
    position_manager = PositionManager(
        order_service, market_data, volatility_config=VolatilityConfig(atr_period=30)
    )
    strategy_registry = StrategyRegistry()
    strategy_registry.register("breakout_v1", BreakoutV1())
    skill_selector = SkillSelector(
        skills={
            DEPTH_SYMBOL: [
                NormalSkill(
                    name="normal/xauusd/breakout_v1",
                    symbol=DEPTH_SYMBOL,
                    strategy="breakout_v1",
                    sessions=(),
                )
            ],
            NO_DEPTH_SYMBOL: [
                NormalSkill(
                    name="normal/vix75/breakout_v1",
                    symbol=NO_DEPTH_SYMBOL,
                    strategy="breakout_v1",
                    sessions=(),
                )
            ],
        },
        timezone="UTC",
    )

    order_book_repository = OrderBookSnapshotRepository(session_factory)
    journal_repository = JournalRepository(session_factory)
    trade_journal = TradeJournalService(
        repository=journal_repository,
        market_context=_NullMarketContext(),
        event_bus=event_bus,
        review_every_n_trades=1000,
    )
    event_bus.subscribe(PositionOpened, trade_journal.on_position_opened)

    return _Wired(
        session_factory=session_factory,
        event_bus=event_bus,
        order_service=order_service,
        risk_manager=risk_manager,
        position_manager=position_manager,
        skill_selector=skill_selector,
        strategy_registry=strategy_registry,
        market_data=market_data,
        account=account,
        order_book_repository=order_book_repository,
        journal_repository=journal_repository,
        signal_decisions=signal_decisions,
    )


class _NullMarketContext:
    async def capture(self, symbol):
        return MarketSnapshot()


class _Wired:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def make_engine(self, order_book_capture) -> TradeEngine:
        engine = TradeEngine(
            market_data=self.market_data,
            order_service=self.order_service,
            account=self.account,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            skill_selector=self.skill_selector,
            strategy_source=self.strategy_registry,
            entry_timeframe="M5",
            volatility_config=VolatilityConfig(atr_period=30),
            context_bars=30,
            signal_decisions=self.signal_decisions,
            order_book_capture=order_book_capture,
        )
        self.event_bus.subscribe(CandleClosed, engine.on_candle_closed)
        return engine


def _signal_decision_rows_for_symbol(session_factory, symbol: str) -> list[SignalDecisionRow]:
    query = select(SignalDecisionRow).where(SignalDecisionRow.symbol == symbol)
    with session_factory() as session:
        return list(session.scalars(query))


def _order_book_rows(session_factory) -> list[OrderBookSnapshotRow]:
    with session_factory() as session:
        return list(session.scalars(select(OrderBookSnapshotRow)))


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Polls `predicate` until it's true or `timeout` elapses.

    The fire-and-forget capture task hops through `asyncio.to_thread` (real
    thread-pool scheduling, not just an event-loop turn), so a fixed number
    of `await asyncio.sleep(0)` calls doesn't reliably wait long enough —
    it's fast and deterministic when this file runs alone, but flakes under
    load from the rest of the suite. Polling on the actual observable
    end-state is the robust way to synchronize with a task this test doesn't
    hold a handle to."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


async def test_depth_symbol_gets_a_snapshot_joined_to_its_signal_decision(wired):
    capture = FakeOrderBookCapture(
        wired.order_book_repository, depth_symbols={DEPTH_SYMBOL}
    )
    engine = wired.make_engine(capture)

    await engine.on_candle_closed(CandleClosed(symbol=DEPTH_SYMBOL, timeframe="M5"))
    # Fire-and-forget capture task — wait for its actual effect (the DB row),
    # not a fixed number of event-loop turns (see `_wait_until`'s docstring).
    await _wait_until(lambda: len(_order_book_rows(wired.session_factory)) >= 1)

    snapshot_rows = _order_book_rows(wired.session_factory)
    assert len(snapshot_rows) == 1
    decision_rows = _signal_decision_rows_for_symbol(wired.session_factory, DEPTH_SYMBOL)
    assert len(decision_rows) == 1
    assert snapshot_rows[0].signal_id == decision_rows[0].signal_id

    # The trade still opened and got journaled normally.
    trade = wired.journal_repository.get("1")
    assert trade is not None
    assert trade.symbol == DEPTH_SYMBOL


async def test_no_depth_symbol_gets_no_snapshot_row_but_still_trades(wired):
    capture = FakeOrderBookCapture(
        wired.order_book_repository, depth_symbols={DEPTH_SYMBOL}
    )
    engine = wired.make_engine(capture)

    await engine.on_candle_closed(CandleClosed(symbol=NO_DEPTH_SYMBOL, timeframe="M5"))
    # `calls` is appended synchronously as the first line of `capture()`, and
    # this fake returns immediately after for a no-depth symbol (no further
    # `await`) — so once `calls` has an entry, the whole attempt is done.
    await _wait_until(lambda: len(capture.calls) >= 1)

    assert _order_book_rows(wired.session_factory) == []
    decision_rows = _signal_decision_rows_for_symbol(wired.session_factory, NO_DEPTH_SYMBOL)
    assert len(decision_rows) == 1

    trades = wired.journal_repository.get_all()
    assert len(trades) == 1
    assert trades[0].symbol == NO_DEPTH_SYMBOL


async def test_raising_capture_port_still_lets_the_trade_open_and_journal(wired):
    """Proves `_capture_order_book_safely` end to end, not just in
    isolation: even when the fake port raises on every call, the trade
    opens, fills, and gets journaled exactly as if capture didn't exist."""
    capture = FakeOrderBookCapture(
        wired.order_book_repository, depth_symbols=set(), raise_symbols={DEPTH_SYMBOL}
    )
    engine = wired.make_engine(capture)

    await engine.on_candle_closed(CandleClosed(symbol=DEPTH_SYMBOL, timeframe="M5"))
    # This fake raises immediately after appending to `calls`, with no
    # intervening `await` — so once `calls` has an entry, the raise (and the
    # engine's swallow of it) has already happened.
    await _wait_until(lambda: len(capture.calls) >= 1)

    assert len(capture.calls) == 1
    assert _order_book_rows(wired.session_factory) == []

    trades = wired.journal_repository.get_all()
    assert len(trades) == 1
    assert trades[0].symbol == DEPTH_SYMBOL
    assert trades[0].is_open is True
