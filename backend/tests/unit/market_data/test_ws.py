"""Tests for Socket.IO multi-window room and symbol reference counting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.market_data.api import ws
from src.market_data.domain.models import Timeframe


@pytest.fixture(autouse=True)
def clean_ws_state():
    ws._sid_symbols.clear()
    ws._sid_rooms.clear()
    ws._candle_streams.clear()
    ws._live_candles.clear()
    yield
    ws._sid_symbols.clear()
    ws._sid_rooms.clear()
    ws._candle_streams.clear()
    ws._live_candles.clear()


@pytest.mark.asyncio
async def test_multi_window_symbol_refcounting():
    """Unsubscribing from one timeframe must not unwatch the symbol if another
    window on the same connection is still streaming a different timeframe."""
    candle_stream = MagicMock()
    live_candle = MagicMock()
    ws.bind_candle_stream("default", candle_stream)
    ws.bind_live_candle("default", live_candle)

    with (
        patch.object(ws.sio, "enter_room", new_callable=AsyncMock) as enter,
        patch.object(ws.sio, "leave_room", new_callable=AsyncMock) as leave,
    ):
        # Window 1 subscribes to EURUSD M5
        await ws.subscribe(
            "client_1", {"account_id": "default", "symbol": "EURUSD", "timeframe": "M5"}
        )
        candle_stream.watch.assert_called_once_with("EURUSD")
        live_candle.watch.assert_called_once_with("EURUSD", Timeframe.M5)
        enter.assert_called_with("client_1", "default:EURUSD:M5")
        candle_stream.watch.reset_mock()

        # Window 2 subscribes to EURUSD M15 on the same connection
        await ws.subscribe(
            "client_1", {"account_id": "default", "symbol": "EURUSD", "timeframe": "M15"}
        )
        candle_stream.watch.assert_not_called()
        live_candle.watch.assert_called_with("EURUSD", Timeframe.M15)
        enter.assert_called_with("client_1", "default:EURUSD:M15")

        # Window 2 closes (or switches symbol/timeframe), unsubscribing from M15
        await ws.unsubscribe(
            "client_1", {"account_id": "default", "symbol": "EURUSD", "timeframe": "M15"}
        )
        live_candle.unwatch.assert_called_once_with("EURUSD", Timeframe.M15)
        leave.assert_called_once_with("client_1", "default:EURUSD:M15")
        # Critical assertion: EURUSD symbol must NOT be unwatched because M5 is still active
        candle_stream.unwatch.assert_not_called()

        # Window 1 finally unsubscribes from M5
        await ws.unsubscribe(
            "client_1", {"account_id": "default", "symbol": "EURUSD", "timeframe": "M5"}
        )
        live_candle.unwatch.assert_called_with("EURUSD", Timeframe.M5)
        candle_stream.unwatch.assert_called_once_with("EURUSD")
        leave.assert_called_with("client_1", "default:EURUSD:M5")


@pytest.mark.asyncio
async def test_duplicate_room_refcounting():
    """Subscribing to the exact same room twice on the same connection (e.g. during
    React component layout transitions) should reference count room membership."""
    live_candle = MagicMock()
    ws.bind_live_candle("default", live_candle)

    with (
        patch.object(ws.sio, "enter_room", new_callable=AsyncMock),
        patch.object(ws.sio, "leave_room", new_callable=AsyncMock) as leave,
    ):
        await ws.subscribe(
            "client_1", {"account_id": "default", "symbol": "XAUUSD", "timeframe": "M1"}
        )
        await ws.subscribe(
            "client_1", {"account_id": "default", "symbol": "XAUUSD", "timeframe": "M1"}
        )
        live_candle.watch.assert_called_once_with("XAUUSD", Timeframe.M1)

        # Unsubscribing once must not unwatch live candle or leave the Socket.IO room
        await ws.unsubscribe(
            "client_1", {"account_id": "default", "symbol": "XAUUSD", "timeframe": "M1"}
        )
        live_candle.unwatch.assert_not_called()
        leave.assert_not_called()

        # Unsubscribing second time leaves the room
        await ws.unsubscribe(
            "client_1", {"account_id": "default", "symbol": "XAUUSD", "timeframe": "M1"}
        )
        live_candle.unwatch.assert_called_once_with("XAUUSD", Timeframe.M1)
        leave.assert_called_once_with("client_1", "default:XAUUSD:M1")
