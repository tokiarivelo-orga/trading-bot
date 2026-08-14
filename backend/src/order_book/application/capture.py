"""Best-effort order-book capture for AI-training data export.

`OrderBookCaptureService.capture` owns the graceful-degradation contract:
never raises, never blocks/slows trading, and "no depth for this symbol" is
encoded as "no database row" — not an empty-but-present row, not a bubbled
exception. Nothing in `engine/` calls this yet (a later phase wires it into
the trade loop); this phase only needs the service to exist and be
constructible (see `container.py`).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.order_book.adapters.repository import OrderBookSnapshotRepository
from src.order_book.domain.models import OrderBookSnapshot, OrderBookUnavailable
from src.order_book.ports.gateway import OrderBookGatewayPort

logger = logging.getLogger(__name__)


class OrderBookCaptureService:
    def __init__(
        self,
        gateway: OrderBookGatewayPort,
        repository: OrderBookSnapshotRepository,
        account_id: str = "default",
    ) -> None:
        self._gateway = gateway
        self._repository = repository
        self._account_id = account_id
        # Symbols already confirmed to report no depth — logged once each,
        # not on every signal, so a chatty OTC symbol doesn't spam INFO.
        self._warned_symbols: set[str] = set()

    async def capture(self, *, signal_id: str, symbol: str) -> None:
        try:
            snapshot = await self._gateway.get_snapshot(symbol)
        except OrderBookUnavailable:
            logger.warning("order-book capture failed for %s", symbol)
            return
        if not snapshot.levels:
            if symbol not in self._warned_symbols:
                logger.info("no order-book depth available for %s — will not retry logging", symbol)
                self._warned_symbols.add(symbol)
            return
        await asyncio.to_thread(self._repository.save, signal_id, snapshot, self._account_id)

    async def get_for_signal(self, signal_id: str) -> OrderBookSnapshot | None:
        """The snapshot captured for `signal_id`, or `None` when none was
        ever captured — either the symbol reported no depth, or capture
        hadn't run yet for that signal. Backs `GET .../order-book/signal/
        {signal_id}`."""
        return await asyncio.to_thread(self._repository.get_for_signal, signal_id, self._account_id)

    async def list_for_account(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[str, OrderBookSnapshot]]:
        """Every snapshot captured for this account, newest first, optionally
        filtered by `symbol` and/or a `[since, until]` capture-time window —
        backs `GET .../order-book/export`. Sparse by design (see module
        docstring), so this is a plain filtered read rather than a paginated/
        streamed one, unlike `market_data`'s candle export. Returns `[]`,
        never raises, when nothing matches."""
        return await asyncio.to_thread(
            self._repository.list_for_account, self._account_id, symbol, since, until
        )
