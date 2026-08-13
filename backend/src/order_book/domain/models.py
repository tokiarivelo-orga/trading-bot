"""Order-book (market depth) domain: a snapshot captured at the moment a
trading signal fires, for AI-training data export.

Pure values — no I/O, no framework imports. Most symbols this bot trades
(XAUUSD via a forex/CFD broker, VIX75/Boom via Deriv — synthetic/OTC
instruments) will almost certainly not return real depth from MT5's
`market_book_get`; that is encoded as `levels=()` on a snapshot, never as an
exception — `OrderBookUnavailable` is reserved for genuine transport/
connection failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class BookSide(StrEnum):
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True, kw_only=True)
class BookLevel:
    side: BookSide
    price: float
    volume: float


@dataclass(frozen=True, kw_only=True)
class OrderBookSnapshot:
    symbol: str
    time: datetime  # when the snapshot was read, UTC
    # Empty when the broker/symbol reports no market depth — not an error.
    levels: tuple[BookLevel, ...] = ()


class OrderBookUnavailable(Exception):
    """Gateway unreachable, not logged in, or the terminal rejected the call.

    Never raised for "this symbol has no market depth" — that is a normal,
    successful `OrderBookSnapshot` with `levels=()`."""
