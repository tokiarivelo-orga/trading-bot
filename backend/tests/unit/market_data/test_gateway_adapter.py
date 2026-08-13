"""GatewayMarketData against the exact wire shapes of gateway/schemas.py."""

from datetime import UTC, datetime

import httpx
import pytest

from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.market_data.domain.models import MarketDataUnavailable, Timeframe

CANDLE_WIRE = {
    "time": 1_752_100_500,
    "open": 2400.0,
    "high": 2401.0,
    "low": 2399.0,
    "close": 2400.5,
    "tick_volume": 1000,
    "spread": 25,
    "real_volume": 42,
}


def adapter_with(handler) -> GatewayMarketData:
    transport = httpx.MockTransport(handler)
    return GatewayMarketData(httpx.AsyncClient(transport=transport, base_url="http://gw"))


async def test_get_candles_parses_wire_format():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/candles"
        assert request.url.params["timeframe"] == "M5"
        return httpx.Response(200, json=[CANDLE_WIRE])

    candles = await adapter_with(handler).get_candles("XAUUSD", Timeframe.M5, 10)
    (candle,) = candles
    assert candle.symbol == "XAUUSD"
    assert candle.time == datetime.fromtimestamp(1_752_100_500, tz=UTC)
    assert candle.time.tzinfo is UTC
    assert candle.spread_points == 25
    assert candle.real_volume == 42


async def test_get_candles_forwards_before_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["before"] == "1752100000"
        return httpx.Response(200, json=[CANDLE_WIRE])

    await adapter_with(handler).get_candles(
        "XAUUSD", Timeframe.M5, 10, before=datetime.fromtimestamp(1_752_100_000, tz=UTC)
    )


async def test_get_candles_omits_before_param_when_none():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "before" not in request.url.params
        return httpx.Response(200, json=[CANDLE_WIRE])

    await adapter_with(handler).get_candles("XAUUSD", Timeframe.M5, 10)


async def test_gateway_error_becomes_market_data_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "not logged in — POST /login first"})

    with pytest.raises(MarketDataUnavailable, match="not logged in"):
        await adapter_with(handler).get_candles("XAUUSD", Timeframe.M5, 10)


async def test_connection_failure_becomes_market_data_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(MarketDataUnavailable, match="unreachable"):
        await adapter_with(handler).get_tick("XAUUSD")


async def test_symbol_info_maps_all_fields():
    payload = {
        "symbol": "XAUUSD",
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
    info = await adapter_with(lambda r: httpx.Response(200, json=payload)).get_symbol_info("XAUUSD")
    assert info.spread_points == 25
    assert info.stops_level == 10


async def test_list_symbols_maps_all_fields_and_forwards_query_params():
    payload = {
        "items": [
            {
                "name": "XAUUSD",
                "description": "Gold vs US Dollar",
                "path": "Metals",
                "visible": True,
            },
        ],
        "total": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/symbols"
        assert request.url.params["search"] == "gold"
        assert request.url.params["limit"] == "50"
        assert request.url.params["offset"] == "0"
        return httpx.Response(200, json=payload)

    page = await adapter_with(handler).list_symbols("gold", 50)
    assert page.total == 1
    (symbol,) = page.items
    assert symbol.name == "XAUUSD"
    assert symbol.path == "Metals"
    assert symbol.visible is True


async def test_list_symbols_omits_search_param_when_none():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "search" not in request.url.params
        return httpx.Response(200, json={"items": [], "total": 0})

    await adapter_with(handler).list_symbols(None, 200)


async def test_list_symbols_forwards_offset():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["offset"] == "40"
        return httpx.Response(200, json={"items": [], "total": 0})

    await adapter_with(handler).list_symbols(None, 20, offset=40)
