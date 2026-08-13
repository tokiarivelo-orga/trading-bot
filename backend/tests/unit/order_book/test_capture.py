"""OrderBookCaptureService owns the graceful-degradation contract: never
raises, logs "no depth" once per symbol, and stores nothing for symbols
that report no depth."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from src.order_book.application.capture import OrderBookCaptureService
from src.order_book.domain.models import (
    BookLevel,
    BookSide,
    OrderBookSnapshot,
    OrderBookUnavailable,
)


class FakeGateway:
    def __init__(self) -> None:
        self.snapshots: dict[str, OrderBookSnapshot] = {}
        self.unavailable_symbols: set[str] = set()
        self.calls: list[str] = []

    async def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        self.calls.append(symbol)
        if symbol in self.unavailable_symbols:
            raise OrderBookUnavailable(f"gateway unreachable for {symbol}")
        return self.snapshots.get(symbol, OrderBookSnapshot(symbol=symbol, time=datetime.now(UTC)))


class FakeRepository:
    def __init__(self) -> None:
        self.saved: list[tuple[str, OrderBookSnapshot, str]] = []

    def save(self, signal_id: str, snapshot: OrderBookSnapshot, account_id: str) -> None:
        self.saved.append((signal_id, snapshot, account_id))

    def get_for_signal(self, signal_id: str, account_id: str):
        for sid, snapshot, acct in self.saved:
            if sid == signal_id and acct == account_id:
                return snapshot
        return None


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def service(gateway, repository) -> OrderBookCaptureService:
    return OrderBookCaptureService(gateway=gateway, repository=repository, account_id="acct-1")


async def test_capture_persists_non_empty_snapshot(service, gateway, repository):
    gateway.snapshots["XAUUSD"] = OrderBookSnapshot(
        symbol="XAUUSD",
        time=datetime.now(UTC),
        levels=(BookLevel(side=BookSide.BID, price=2400.10, volume=5.5),),
    )

    await service.capture(signal_id="sig-1", symbol="XAUUSD")

    assert len(repository.saved) == 1
    signal_id, snapshot, account_id = repository.saved[0]
    assert signal_id == "sig-1"
    assert snapshot.symbol == "XAUUSD"
    assert account_id == "acct-1"


async def test_capture_stores_nothing_for_empty_levels(service, repository):
    # gateway defaults to an empty-levels snapshot for any unseeded symbol.
    await service.capture(signal_id="sig-1", symbol="VIX75")
    assert repository.saved == []


async def test_capture_logs_once_per_symbol_then_stays_silent(service, caplog):
    caplog.set_level(logging.INFO)

    await service.capture(signal_id="sig-1", symbol="VIX75")
    await service.capture(signal_id="sig-2", symbol="VIX75")
    await service.capture(signal_id="sig-3", symbol="VIX75")

    no_depth_records = [r for r in caplog.records if "no order-book depth" in r.message]
    assert len(no_depth_records) == 1
    assert "VIX75" in no_depth_records[0].message


async def test_capture_logs_separately_per_distinct_symbol(service, caplog):
    caplog.set_level(logging.INFO)

    await service.capture(signal_id="sig-1", symbol="VIX75")
    await service.capture(signal_id="sig-2", symbol="Boom500")

    no_depth_records = [r for r in caplog.records if "no order-book depth" in r.message]
    assert len(no_depth_records) == 2


async def test_capture_swallows_order_book_unavailable(service, gateway, repository, caplog):
    caplog.set_level(logging.WARNING)
    gateway.unavailable_symbols.add("XAUUSD")

    # Must never raise.
    await service.capture(signal_id="sig-1", symbol="XAUUSD")

    assert repository.saved == []
    assert any("order-book capture failed" in r.message for r in caplog.records)


async def test_capture_never_raises_even_when_gateway_raises_repeatedly(service, gateway):
    gateway.unavailable_symbols.add("XAUUSD")
    for i in range(3):
        await service.capture(signal_id=f"sig-{i}", symbol="XAUUSD")
    # No exception propagated — reaching here is the assertion.


async def test_get_for_signal_returns_none_when_never_captured(service):
    assert await service.get_for_signal("unknown") is None


async def test_get_for_signal_returns_captured_snapshot(service, gateway, repository):
    gateway.snapshots["XAUUSD"] = OrderBookSnapshot(
        symbol="XAUUSD",
        time=datetime.now(UTC),
        levels=(BookLevel(side=BookSide.ASK, price=2400.35, volume=3.0),),
    )
    await service.capture(signal_id="sig-1", symbol="XAUUSD")

    snapshot = await service.get_for_signal("sig-1")

    assert snapshot is not None
    assert snapshot.symbol == "XAUUSD"
