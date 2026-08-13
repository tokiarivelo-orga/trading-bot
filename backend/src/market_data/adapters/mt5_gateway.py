"""MarketDataPort adapter that calls the MT5 gateway HTTP API.

Wire shapes are defined by gateway/src/gateway/schemas.py — parse exactly
those, translate transport failures into MarketDataUnavailable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from src.market_data.domain.models import (
    BrokerSymbol,
    Candle,
    MarketDataUnavailable,
    SymbolInfo,
    SymbolPage,
    Tick,
    Timeframe,
)

# The shared gateway client's default timeout (see container.py) is 30s,
# sized for the rare cold `mt5.initialize()` handshake on account/broker
# calls. Market-data reads back the interactive chart, though — once
# connected, MT5 answers `copy_rates_from_pos` in well under a second, so a
# stuck/overloaded terminal should fail fast here and let CandleHistoryService
# fall back to the DB instead of leaving a timeframe switch hanging for 30s.
_READ_TIMEOUT_S = 8.0


class GatewayMarketData:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def get_candles(
        self, symbol: str, timeframe: Timeframe, count: int, before: datetime | None = None
    ) -> list[Candle]:
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe.value, "count": count}
        if before is not None:
            params["before"] = int(before.timestamp())
        payload = await self._get("/candles", params)
        return [
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                time=_utc(c["time"]),
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                tick_volume=c["tick_volume"],
                spread_points=c["spread"],
                real_volume=c.get("real_volume", 0),
            )
            for c in payload
        ]

    async def get_tick(self, symbol: str) -> Tick:
        t = await self._get("/tick", {"symbol": symbol})
        return Tick(symbol=symbol, time=_utc(t["time"]), bid=t["bid"], ask=t["ask"])

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        info = await self._get("/symbol_info", {"symbol": symbol})
        return SymbolInfo(**info)

    async def list_symbols(self, search: str | None, limit: int, offset: int = 0) -> SymbolPage:
        """A page of the broker's tradable-symbol catalog, optionally filtered.
        Not part of `MarketDataPort` — it's a live-broker-only browsing
        feature with no meaning for the backtest replay adapter."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        page = await self._get("/symbols", params)
        return SymbolPage(items=[BrokerSymbol(**row) for row in page["items"]], total=page["total"])

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = await self._client.get(path, params=params, timeout=_READ_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise MarketDataUnavailable(f"gateway unreachable: {exc}") from exc
        if response.status_code != 200:
            raise MarketDataUnavailable(
                f"gateway {path} -> {response.status_code}: {response.text}"
            )
        return response.json()


def _utc(epoch_seconds: int) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC)
