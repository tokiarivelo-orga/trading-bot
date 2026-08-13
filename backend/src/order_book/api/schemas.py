"""Wire schema for the `/order-book` HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OrderBookLevelOut(BaseModel):
    """One price level of a market-depth snapshot."""

    side: str = Field(description="'bid' or 'ask'.")
    price: float = Field(description="Price at this level.")
    volume: float = Field(description="Size available at this level (broker's reported volume).")


class OrderBookSnapshotOut(BaseModel):
    """The order-book (market depth) snapshot captured at the moment one
    trading signal fired. Absence of a row for a signal is expected, not an
    error — it means either the symbol reported no depth (common for OTC
    CFD/synthetic symbols), or capture hadn't run yet for that signal."""

    signal_id: str = Field(description="The signal this snapshot was captured for.")
    symbol: str = Field(description="Broker symbol, e.g. 'XAUUSD'.")
    captured_at: int = Field(description="Epoch seconds UTC when the snapshot was read.")
    levels: list[OrderBookLevelOut] = Field(
        description="Price levels at capture time, broker order not guaranteed."
    )
