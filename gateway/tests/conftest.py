"""Contract tests run against a stubbed MetaTrader5 module (Linux-friendly).

The stub mimics the official package's shapes: numpy-style subscriptable rows
for rates, attribute objects for account/terminal/symbol info.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gateway import mt5_client
from gateway.main import app


class FakePosition:
    def __init__(
        self, ticket, symbol, type_, volume, price_open, sl, tp, time, profit, comment="", magic=0
    ) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = type_
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.time = time
        self.profit = profit
        self.comment = comment
        self.magic = magic


class FakePendingOrder:
    def __init__(
        self, ticket, symbol, type_, volume_current, price_open, sl, tp, time_setup, comment=""
    ) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = type_
        self.volume_current = volume_current
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.time_setup = time_setup
        self.comment = comment


class FakeMt5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_REMOVE = 8
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_NO_CHANGES = 10025
    BOOK_TYPE_BUY = 1
    BOOK_TYPE_SELL = 2

    def __init__(self) -> None:
        self.reject_login = False
        self.shutdown_called = False
        self.reject_order = False
        self.reject_no_changes = False
        self._next_ticket = 1000
        self._positions: dict[int, FakePosition] = {}
        self._pending_orders: dict[int, FakePendingOrder] = {}
        # Bitmask a symbol reports as supported filling modes — defaults to
        # IOC (matches the old hardcoded behavior other tests rely on).
        # Tests exercising `_filling_type`'s fallback set this directly.
        self.filling_mode = self.SYMBOL_FILLING_IOC
        # A symbol's lot-size constraints — defaults match the old hardcoded
        # forex assumption. Tests exercising `_normalize_volume` (e.g. a
        # synthetic index with a coarser step) set these directly.
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.01
        # Market depth (DOM) — most symbols/brokers don't support it, so the
        # default mirrors that: `market_book_add` fails and no book exists.
        # Tests exercising `order_book()` flip these on directly.
        self.market_book_add_succeeds = False
        self.market_book_rows: list[SimpleNamespace] | None = None
        self.market_book_add_calls: list[str] = []
        self.market_book_release_calls: list[str] = []

    def initialize(self) -> bool:
        return True

    def login(self, login, password=None, server=None) -> bool:
        return not self.reject_login

    def shutdown(self) -> None:
        self.shutdown_called = True

    def last_error(self):
        return (-6, "Authorization failed")

    def terminal_info(self):
        return SimpleNamespace(connected=True)

    def account_info(self):
        return SimpleNamespace(
            login=123456,
            server="Demo-Server",
            name="Test User",
            currency="USD",
            balance=10_000.0,
            equity=10_050.0,
            leverage=100,
        )

    def symbol_select(self, symbol, enable) -> bool:
        return True

    def symbols_get(self):
        return [
            SimpleNamespace(
                name="XAUUSD", description="Gold vs US Dollar", path="Metals", visible=True
            ),
            SimpleNamespace(
                name="XAGUSD", description="Silver vs US Dollar", path="Metals", visible=True
            ),
            SimpleNamespace(
                name="EURUSD", description="Euro vs US Dollar", path="Forex\\Majors", visible=False
            ),
            SimpleNamespace(
                name="BTCUSD", description="Bitcoin vs US Dollar", path="Crypto", visible=True
            ),
        ]

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        base = 1_752_100_500  # aligned to a 5-min boundary
        return [
            {
                "time": base + i * 300,
                "open": 2400.0 + i,
                "high": 2401.0 + i,
                "low": 2399.0 + i,
                "close": 2400.5 + i,
                "tick_volume": 1000 + i,
                "spread": 25,
                "real_volume": 500 + i,
            }
            for i in range(min(count, 3))
        ]

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        # One bar every 5 minutes spanning the requested range, aligned to
        # the same 5-min boundary as copy_rates_from_pos's fixed base time.
        start = int(date_from.timestamp())
        start -= start % 300
        end = int(date_to.timestamp())
        times = range(start, end + 1, 300)
        return [
            {
                "time": t,
                "open": 2300.0,
                "high": 2301.0,
                "low": 2299.0,
                "close": 2300.5,
                "tick_volume": 900,
                "spread": 20,
                "real_volume": 450,
            }
            for t in times
        ]

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(time=1_752_100_812, bid=2400.10, ask=2400.35)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            name=symbol,
            bid=2400.10,
            ask=2400.35,
            spread=25,
            point=0.01,
            digits=2,
            trade_stops_level=10,
            trade_contract_size=100.0,
            filling_mode=self.filling_mode,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
        )

    def market_book_add(self, symbol) -> bool:
        self.market_book_add_calls.append(symbol)
        return self.market_book_add_succeeds

    def market_book_get(self, symbol):
        return self.market_book_rows

    def market_book_release(self, symbol) -> bool:
        self.market_book_release_calls.append(symbol)
        return True

    def order_send(self, request):
        if self.reject_order:
            return SimpleNamespace(retcode=10004, order=0, volume=0.0, price=0.0)
        if request["action"] == self.TRADE_ACTION_SLTP:
            if self.reject_no_changes:
                return SimpleNamespace(
                    retcode=self.TRADE_RETCODE_NO_CHANGES,
                    order=request["position"],
                    volume=0.0,
                    price=0.0,
                )
            position = self._positions.get(request["position"])
            if position is not None:
                position.sl = request.get("sl") or None
                position.tp = request.get("tp") or None
            return SimpleNamespace(
                retcode=self.TRADE_RETCODE_DONE, order=request["position"], volume=0.0, price=0.0
            )
        if request["action"] == self.TRADE_ACTION_MODIFY:
            if self.reject_no_changes:
                return SimpleNamespace(
                    retcode=self.TRADE_RETCODE_NO_CHANGES,
                    order=request["order"],
                    volume=0.0,
                    price=0.0,
                )
            order = self._pending_orders.get(request["order"])
            if order is not None:
                order.price_open = request["price"]
                order.sl = request.get("sl") or None
                order.tp = request.get("tp") or None
            return SimpleNamespace(
                retcode=self.TRADE_RETCODE_DONE, order=request["order"], volume=0.0, price=0.0
            )
        if request["action"] == self.TRADE_ACTION_REMOVE:
            self._pending_orders.pop(request["order"], None)
            return SimpleNamespace(
                retcode=self.TRADE_RETCODE_DONE, order=request["order"], volume=0.0, price=0.0
            )
        if "position" in request:
            ticket = request["position"]
            position = self._positions.get(ticket)
            close_volume = request["volume"]
            if position is not None:
                if close_volume >= position.volume:
                    del self._positions[ticket]
                else:
                    position.volume -= close_volume
            return SimpleNamespace(
                retcode=self.TRADE_RETCODE_DONE,
                order=ticket,
                volume=close_volume,
                price=request["price"],
            )
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = FakePosition(
            ticket=ticket,
            symbol=request["symbol"],
            type_=request["type"],
            volume=request["volume"],
            price_open=request["price"],
            sl=request["sl"] or None,
            tp=request["tp"] or None,
            time=1_752_100_812,
            profit=12.5,
            comment=request.get("comment", ""),
            magic=request.get("magic", 0),
        )
        return SimpleNamespace(
            retcode=self.TRADE_RETCODE_DONE,
            order=ticket,
            volume=request["volume"],
            price=request["price"],
        )

    def positions_get(self, symbol=None, ticket=None):
        rows = list(self._positions.values())
        if ticket is not None:
            rows = [p for p in rows if p.ticket == ticket]
        if symbol is not None:
            rows = [p for p in rows if p.symbol == symbol]
        return rows

    def add_pending_order(self, order: FakePendingOrder) -> None:
        self._pending_orders[order.ticket] = order

    def orders_get(self, symbol=None, ticket=None):
        rows = list(self._pending_orders.values())
        if ticket is not None:
            rows = [o for o in rows if o.ticket == ticket]
        if symbol is not None:
            rows = [o for o in rows if o.symbol == symbol]
        return rows


@pytest.fixture
def fake_mt5(monkeypatch) -> FakeMt5:
    fake = FakeMt5()
    monkeypatch.setattr(mt5_client, "mt5", fake)
    monkeypatch.setattr(mt5_client.client, "_connected", False)
    return fake


@pytest.fixture
def api(fake_mt5) -> TestClient:
    return TestClient(app)
