"""GatewayOrderBookClient against the exact wire shape of gateway/schemas.py."""

from datetime import UTC, datetime

import httpx
import pytest

from src.order_book.adapters.gateway_client import GatewayOrderBookClient
from src.order_book.domain.models import BookSide, OrderBookUnavailable


def adapter_with(handler) -> GatewayOrderBookClient:
    transport = httpx.MockTransport(handler)
    return GatewayOrderBookClient(httpx.AsyncClient(transport=transport, base_url="http://gw"))


async def test_get_snapshot_parses_non_empty_response():
    payload = {
        "symbol": "XAUUSD",
        "time": 1_752_100_500,
        "levels": [
            {"type": "bid", "price": 2400.10, "volume": 5.5},
            {"type": "ask", "price": 2400.35, "volume": 3.0},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/order_book"
        assert request.url.params["symbol"] == "XAUUSD"
        return httpx.Response(200, json=payload)

    snapshot = await adapter_with(handler).get_snapshot("XAUUSD")

    assert snapshot.symbol == "XAUUSD"
    assert snapshot.time == datetime.fromtimestamp(1_752_100_500, tz=UTC)
    assert len(snapshot.levels) == 2
    assert snapshot.levels[0].side is BookSide.BID
    assert snapshot.levels[0].price == 2400.10
    assert snapshot.levels[0].volume == 5.5
    assert snapshot.levels[1].side is BookSide.ASK


async def test_get_snapshot_parses_empty_levels_response():
    # Empty levels is a normal, successful result — most CFD/forex and
    # synthetic-index symbols report no depth at all.
    payload = {"symbol": "VIX75", "time": 1_752_100_500, "levels": []}

    snapshot = await adapter_with(lambda r: httpx.Response(200, json=payload)).get_snapshot("VIX75")

    assert snapshot.symbol == "VIX75"
    assert snapshot.levels == ()


async def test_gateway_error_becomes_order_book_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "not logged in — POST /login first"})

    with pytest.raises(OrderBookUnavailable, match="not logged in"):
        await adapter_with(handler).get_snapshot("XAUUSD")


async def test_connection_failure_becomes_order_book_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OrderBookUnavailable, match="unreachable"):
        await adapter_with(handler).get_snapshot("XAUUSD")
