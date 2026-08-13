"""OrderBookGatewayPort adapter that calls the MT5 gateway HTTP API.

Wire shape is defined by gateway/src/gateway/schemas.py's `OrderBookOut` —
parse exactly that, translate transport failures into `OrderBookUnavailable`.
An empty `levels` in the response is a normal, successful result (the
broker/symbol reports no market depth), not a failure.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from src.order_book.domain.models import (
    BookLevel,
    BookSide,
    OrderBookSnapshot,
    OrderBookUnavailable,
)

# Matches GatewayMarketData's read timeout (market_data/adapters/mt5_gateway.py):
# once connected, the gateway answers well under a second, so a stuck/
# overloaded terminal should fail fast here rather than block signal capture.
_READ_TIMEOUT_S = 8.0


class GatewayOrderBookClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        payload = await self._get("/order_book", {"symbol": symbol})
        return OrderBookSnapshot(
            symbol=payload["symbol"],
            time=datetime.fromtimestamp(payload["time"], tz=UTC),
            levels=tuple(
                BookLevel(
                    side=BookSide.BID if lvl["type"] == "bid" else BookSide.ASK,
                    price=lvl["price"],
                    volume=lvl["volume"],
                )
                for lvl in payload["levels"]
            ),
        )

    async def _get(self, path: str, params: dict[str, str]) -> dict:
        try:
            response = await self._client.get(path, params=params, timeout=_READ_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise OrderBookUnavailable(f"gateway unreachable: {exc}") from exc
        if response.status_code != 200:
            raise OrderBookUnavailable(f"gateway {path} -> {response.status_code}: {response.text}")
        return response.json()
