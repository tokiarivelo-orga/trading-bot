"""Raw market data straight from the terminal: candles, ticks, symbol specs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ..mt5_client import Mt5Error, client
from ..schemas import (
    VALID_TIMEFRAMES,
    BrokerSymbolPageOut,
    CandleOut,
    OrderBookOut,
    SymbolInfoOut,
    TickOut,
)

router = APIRouter()

Symbol = Annotated[str, Query(min_length=3, max_length=32)]


def _validate_timeframe(timeframe: str) -> None:
    if timeframe not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"timeframe must be one of {', '.join(VALID_TIMEFRAMES)}",
        )


@router.get("/candles", response_model=list[CandleOut])
def candles(
    symbol: Symbol,
    timeframe: str,
    count: Annotated[int, Query(ge=1, le=5000)] = 300,
    before: Annotated[
        int | None,
        Query(
            description=(
                "Epoch seconds UTC; returns bars ending just before this "
                "instant instead of the most recent ones."
            )
        ),
    ] = None,
) -> list[CandleOut]:
    _validate_timeframe(timeframe)
    try:
        return [CandleOut(**c) for c in client.candles(symbol, timeframe, count, before)]
    except Mt5Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/tick", response_model=TickOut)
def tick(symbol: Symbol) -> TickOut:
    try:
        return TickOut(**client.tick(symbol))
    except Mt5Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/symbol_info", response_model=SymbolInfoOut)
def symbol_info(symbol: Symbol) -> SymbolInfoOut:
    try:
        return SymbolInfoOut(**client.symbol_info(symbol))
    except Mt5Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/order_book", response_model=OrderBookOut)
def order_book(symbol: Symbol) -> OrderBookOut:
    """Level-2 market depth for `symbol`, when the broker reports it.

    Always 200 — never 404/422 — even when the broker/symbol has no market
    depth to offer (the common case for CFD/forex and synthetic-index
    symbols): `levels` is simply empty in that case, so callers never need to
    special-case an error status for "no depth" versus a real failure.
    """
    try:
        book = client.order_book(symbol)
    except Mt5Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if book is None:
        return OrderBookOut(symbol=symbol, time=int(datetime.now(UTC).timestamp()), levels=[])
    return OrderBookOut(**book)


@router.get("/symbols", response_model=BrokerSymbolPageOut)
def symbols(
    search: Annotated[str | None, Query(max_length=64)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> BrokerSymbolPageOut:
    try:
        return BrokerSymbolPageOut(**client.symbols(search, limit, offset))
    except Mt5Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
