"""Port: capture a market-depth snapshot for one signal.

This is what a later phase will depend on from `engine/` to log a snapshot
at the moment a trading signal fires — nothing wires this into the trade
loop yet (Phase 4 only builds the independently-testable vertical slice).
"""

from __future__ import annotations

from typing import Protocol


class OrderBookCapturePort(Protocol):
    async def capture(self, *, signal_id: str, symbol: str) -> None:
        """Best-effort capture — must never raise. See
        `order_book/application/capture.py::OrderBookCaptureService` for the
        "log once, store nothing" contract this implements."""
        ...
