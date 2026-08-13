"""Pydantic models shared over the wire between gateway and backend.

The backend's market_data/broker adapters parse exactly these shapes —
change them in lockstep.
"""

from __future__ import annotations

from pydantic import BaseModel

VALID_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN")


class LoginRequest(BaseModel):
    login: int
    password: str
    server: str


class AccountInfoOut(BaseModel):
    login: int
    server: str
    name: str
    currency: str
    balance: float
    equity: float
    leverage: int


class HealthOut(BaseModel):
    status: str
    terminal_connected: bool
    account: AccountInfoOut | None = None


class CandleOut(BaseModel):
    time: int  # bar open time, epoch seconds UTC
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int  # points, as recorded on the bar
    real_volume: int  # actual traded volume during the bar, when the broker reports it


class TickOut(BaseModel):
    time: int  # epoch seconds UTC
    bid: float
    ask: float


class BrokerSymbolOut(BaseModel):
    name: str
    description: str
    path: str  # broker's Market Watch group, e.g. "Forex\\Majors"
    visible: bool  # already in Market Watch (candle/tick calls auto-add it either way)


class BrokerSymbolPageOut(BaseModel):
    items: list[BrokerSymbolOut]
    total: int  # count after search filtering, before limit/offset — for pagination


class OrderBookLevelOut(BaseModel):
    type: str  # "bid" | "ask"
    price: float
    volume: float


class OrderBookOut(BaseModel):
    symbol: str
    time: int  # epoch seconds UTC when the snapshot was read
    # Empty (not a 404/422) when the broker/symbol reports no market depth —
    # true for most CFD/forex and synthetic-index symbols this bot trades.
    levels: list[OrderBookLevelOut]


class SymbolInfoOut(BaseModel):
    symbol: str
    bid: float
    ask: float
    spread_points: int  # live spread in points
    point: float
    digits: int
    stops_level: int  # broker min SL/TP distance in points
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float


VALID_SIDES = ("buy", "sell")


class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" | "sell"
    volume: float
    sl: float | None = None
    tp: float | None = None
    comment: str = ""
    magic: int = 0  # MT5 magic number — identifies which bot placed the order, 0 if none


class OrderResultOut(BaseModel):
    ticket: int
    symbol: str
    side: str
    volume: float
    price: float
    sl: float | None
    tp: float | None
    time: int  # epoch seconds UTC
    spread_points: int
    comment: str = ""
    magic: int = 0
    profit: float | None = None  # populated on close, None on open
    retcode: int | None = None
    """MT5 `OrderSendResult.retcode` for the deal — 10009 (TRADE_RETCODE_DONE)
    on a successful fill. Reported so the backend can journal execution
    quality; None from a gateway/broker that has no code to give."""


class ModifyRequest(BaseModel):
    sl: float | None = None
    tp: float | None = None


class CloseRequest(BaseModel):
    volume: float | None = None  # None = close in full


class PositionOut(BaseModel):
    ticket: int
    symbol: str
    side: str
    volume: float
    open_price: float
    sl: float | None
    tp: float | None
    open_time: int
    profit: float
    comment: str = ""
    magic: int = 0  # which bot opened this position (MT5 magic number), 0 if none


VALID_ORDER_TYPES = ("limit", "stop")


class PendingOrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" | "sell"
    order_type: str  # "limit" | "stop"
    volume: float
    price: float  # trigger price
    sl: float | None = None
    tp: float | None = None
    comment: str = ""


class PendingOrderOut(BaseModel):
    ticket: int
    symbol: str
    side: str
    order_type: str
    volume: float
    price: float
    sl: float | None
    tp: float | None
    placed_time: int  # epoch seconds UTC
    comment: str = ""


class ModifyPendingOrderRequest(BaseModel):
    price: float | None = None
    sl: float | None = None
    tp: float | None = None


class PositionCloseInfoOut(BaseModel):
    """How a position that's no longer open actually closed — from MT5's
    deal history, not the (transient) open-positions list. Used to detect
    and reconcile broker-side SL/TP fills the backend didn't initiate."""

    ticket: int
    symbol: str
    side: str
    close_price: float
    close_time: int  # epoch seconds UTC
    profit: float
