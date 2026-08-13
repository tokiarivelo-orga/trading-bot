"""Port: where a symbol's current market-depth snapshot comes from."""

from __future__ import annotations

from typing import Protocol

from src.order_book.domain.models import OrderBookSnapshot


class OrderBookGatewayPort(Protocol):
    async def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        """Always returns a snapshot — `levels=()` means "no depth reported"
        for this symbol/broker, the common case for CFD/forex and synthetic-
        index symbols. Raising `OrderBookUnavailable` is reserved for a
        genuine transport/connection failure, not "no depth"."""
        ...
