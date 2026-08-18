"""The ONLY file in the whole repository that imports MetaTrader5.

Wraps the official package (Windows/Wine only) in a thin client that returns
plain dicts matching `schemas.py`. No business logic: raw broker facts in,
explicit commands out. Credentials are held in memory only — never persisted
or logged here.

On non-Windows platforms the import fails and every call raises Mt5Error, so
the module stays importable for tests (which stub the `mt5` attribute).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - Linux dev machines
    mt5 = None

# Set by the launcher (Makefile's dev-gateway, or the installer's generated
# systemd/Scheduled-Task services — see configs/accounts.yaml's
# mt5_terminal_path/mt5_terminal_subpath) to this account's own
# terminal64.exe path. Required for any second concurrent account:
# MetaTrader5.initialize() with no path attaches to *any* already-running
# terminal, so two gateway processes sharing one terminal instance would
# silently log each other out. Two forms, checked in this order:
#
#   MT5_TERMINAL_PATH    — absolute path *as seen by the process running
#                           this code*: a native "C:\..." path on real
#                           Windows, or a host-absolute Linux path under
#                           Wine (e.g. "/home/user/.mt5/drive_c/Program
#                           Files/MetaTrader 5/terminal64.exe"). The Wine
#                           case needs converting: MetaTrader5 runs *inside*
#                           the Wine process, which sees its own C: drive,
#                           not the host filesystem — so when the path
#                           contains a "drive_c" segment, everything up to
#                           and including it is stripped and the remainder
#                           becomes the "C:\..." form. Left as-is otherwise
#                           (native Windows already gives the right form).
#   MT5_TERMINAL_SUBPATH — legacy form: path relative to the Wine prefix's
#                           drive_c/ (e.g. "MT5-demo-1/terminal64.exe").
#                           Still supported for existing setups; prefer
#                           MT5_TERMINAL_PATH for new ones.
#
# Left unset for the primary/default account under `make dev` with no
# accounts.yaml entry configured — keeps attaching to whatever terminal is
# already running, unchanged from pre-multi-account behavior.


def _windows_terminal_path(raw: str) -> str:
    """Convert a host-absolute Wine terminal path to the "C:\\..." form
    MetaTrader5 needs from inside the Wine process; pass a native Windows
    path through unchanged (no "drive_c" segment to convert)."""
    normalized = raw.replace("\\", "/")
    marker = "drive_c/"
    idx = normalized.lower().find(marker)
    if idx == -1:
        return raw
    windows_relative = normalized[idx + len(marker) :]
    return "C:\\" + windows_relative.replace("/", "\\")


_TERMINAL_PATH_ENV = os.environ.get("MT5_TERMINAL_PATH") or None
_TERMINAL_SUBPATH = os.environ.get("MT5_TERMINAL_SUBPATH") or None
if _TERMINAL_PATH_ENV:
    _TERMINAL_PATH = _windows_terminal_path(_TERMINAL_PATH_ENV)
elif _TERMINAL_SUBPATH:
    _TERMINAL_PATH = "C:\\" + _TERMINAL_SUBPATH.replace("/", "\\")
else:
    _TERMINAL_PATH = None


class Mt5Error(Exception):
    """Raised when the terminal is unreachable or a call is rejected.

    `retcode` carries MT5's own `OrderSendResult.retcode` when the failure was
    a trade-server rejection, so the backend can record the code structurally
    instead of regex-ing it back out of this message (OBSERVABILITY_PLAN.md
    Phase 3). `None` for non-trade failures (terminal unreachable, symbol
    lookup, no result object at all)."""

    def __init__(self, message: str, retcode: int | None = None) -> None:
        super().__init__(message)
        self.retcode = retcode


_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN": 2592000,  # approximate (30-day average) — only used to size the copy_rates_range span
}

# MetaTrader5's monthly constant is TIMEFRAME_MN1, not TIMEFRAME_MN — every
# other timeframe matches our naming (TIMEFRAME_M1, TIMEFRAME_H4, ...).
_MT5_CONSTANT_NAMES = {"MN": "MN1"}

# The modify request's price/sl/tp already match the order/position — MT5
# rejects it as a no-op rather than silently accepting it, but the desired
# end state already holds, so callers should see this as success.
TRADE_RETCODE_NO_CHANGES = 10025

# MT5 trade-server return codes worth explaining in plain English — `_last_error()`
# reports the last *IPC* error, which is frequently "[1] Success" even when the
# trade itself was rejected (the rejection reason is the retcode alone), so a raw
# number here is otherwise a dead end without looking it up.
_RETCODE_MESSAGES = {
    10004: "requote — price moved, resend the order",
    10006: "request rejected by the dealer",
    10013: "invalid request",
    10014: "invalid volume",
    10015: "invalid price",
    10016: "invalid stops — sl/tp too close to price (see symbol's stops_level)",
    10017: "trading disabled for this account/symbol",
    10018: "market closed",
    10019: "not enough money",
    10020: "price changed — resend the order",
    10021: "no quotes to process the request",
    10024: "too many requests — rate limited",
    10026: "autotrading disabled by the trade server",
    10027: "autotrading disabled in the terminal — enable the 'AutoTrading' "
    "button in MT5, or Tools > Options > Expert Advisors > Allow algorithmic trading",
    10028: "account locked",
    10030: "unsupported order filling mode",
    10031: "no connection to the trade server",
    10033: "pending-orders limit reached",
    10034: "trading volume limit reached",
    10040: "open-positions limit reached",
    10042: "only long positions allowed for this symbol",
    10043: "only short positions allowed for this symbol",
    10044: "only closing existing positions is allowed for this symbol",
    10046: "hedging prohibited — an opposite position already exists",
}


def _retcode_reason(code: int | None) -> str:
    if code is None:
        return ""
    known = _RETCODE_MESSAGES.get(code)
    return f" — {known}" if known else ""


def _last_error() -> str:
    code, message = mt5.last_error()
    return f"[{code}] {message}"


class Mt5Client:
    """Stateful connection to the local MT5 terminal (one account at a time)."""

    def __init__(self) -> None:
        self._connected = False

    # ── session ─────────────────────────────────────────────────────────

    def login(self, login: int, password: str, server: str) -> dict[str, Any]:
        if mt5 is None:
            raise Mt5Error("MetaTrader5 package unavailable — run the gateway on Windows/Wine")
        initialized = mt5.initialize(path=_TERMINAL_PATH) if _TERMINAL_PATH else mt5.initialize()
        if not initialized:
            raise Mt5Error(f"terminal initialize failed: {_last_error()}")
        if not mt5.login(login, password=password, server=server):
            raise Mt5Error(f"login rejected: {_last_error()}")
        self._connected = True
        logger.info("logged in to %s as %s", server, login)
        return self.account_info()

    def logout(self) -> None:
        if mt5 is not None:
            mt5.shutdown()
        self._connected = False
        logger.info("terminal connection shut down")

    def health(self) -> dict[str, Any]:
        if mt5 is None or not self._connected:
            return {"status": "ok", "terminal_connected": False, "account": None}
        terminal = mt5.terminal_info()
        if terminal is None or not terminal.connected:
            return {"status": "ok", "terminal_connected": False, "account": None}
        return {"status": "ok", "terminal_connected": True, "account": self.account_info()}

    def account_info(self) -> dict[str, Any]:
        self._require_connection()
        info = mt5.account_info()
        if info is None:
            raise Mt5Error(f"account_info failed: {_last_error()}")
        return {
            "login": info.login,
            "server": info.server,
            "name": info.name,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "leverage": info.leverage,
        }

    # ── market data ─────────────────────────────────────────────────────

    def candles(
        self, symbol: str, timeframe: str, count: int, before: int | None = None
    ) -> list[dict[str, Any]]:
        self._require_connection()
        self._select(symbol)
        if before is None:
            rates = mt5.copy_rates_from_pos(symbol, self._timeframe(timeframe), 0, count)
            if rates is None:
                raise Mt5Error(f"copy_rates_from_pos({symbol},{timeframe}) failed: {_last_error()}")
        else:
            # No MT5 call fetches "count bars ending just before a timestamp"
            # directly, so pull a generously wide date range up to (but not
            # including) `before` and keep the `count` bars closest to it.
            # 3x the nominal calendar span comfortably covers weekend/holiday
            # gaps without a per-symbol trading-calendar lookup.
            date_to = datetime.fromtimestamp(before, tz=UTC) - timedelta(seconds=1)
            span = timedelta(seconds=_TIMEFRAME_SECONDS[timeframe] * count * 3)
            rates = mt5.copy_rates_range(
                symbol, self._timeframe(timeframe), date_to - span, date_to
            )
            if rates is None:
                raise Mt5Error(f"copy_rates_range({symbol},{timeframe}) failed: {_last_error()}")
            rates = rates[-count:]
        return [
            {
                "time": int(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "tick_volume": int(r["tick_volume"]),
                "spread": int(r["spread"]),
                "real_volume": int(r["real_volume"]),
            }
            for r in rates
        ]

    def tick(self, symbol: str) -> dict[str, Any]:
        self._require_connection()
        self._select(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Mt5Error(f"symbol_info_tick({symbol}) failed: {_last_error()}")
        return {"time": int(tick.time), "bid": float(tick.bid), "ask": float(tick.ask)}

    def symbols(self, search: str | None, limit: int, offset: int) -> dict[str, Any]:
        """A page of the broker's symbol catalog, optionally filtered by a
        case-insensitive substring match on name or description. `mt5.symbols_get()`
        has no text-search or pagination of its own, so this fetches the whole
        catalog once and filters/pages it here rather than round-tripping per
        symbol. `total` is the filtered count (before paging), so callers can
        tell whether more pages remain."""
        self._require_connection()
        rows = mt5.symbols_get()
        if rows is None:
            raise Mt5Error(f"symbols_get failed: {_last_error()}")
        if search:
            needle = search.lower()
            rows = [r for r in rows if needle in r.name.lower() or needle in r.description.lower()]
        page = rows[offset : offset + limit]
        return {
            "items": [
                {"name": r.name, "description": r.description, "path": r.path, "visible": r.visible}
                for r in page
            ],
            "total": len(rows),
        }

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        self._require_connection()
        self._select(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise Mt5Error(f"symbol_info({symbol}) failed: {_last_error()}")
        return {
            "symbol": info.name,
            "bid": float(info.bid),
            "ask": float(info.ask),
            "spread_points": int(info.spread),
            "point": float(info.point),
            "digits": int(info.digits),
            "stops_level": int(info.trade_stops_level),
            "contract_size": float(info.trade_contract_size),
            "volume_min": float(info.volume_min),
            "volume_max": float(info.volume_max),
            "volume_step": float(info.volume_step),
        }

    def order_book(self, symbol: str) -> dict[str, Any] | None:
        """Level-2 market depth (DOM), when the broker/symbol reports one.

        Most CFD/forex and synthetic-index symbols traded here (XAUUSD via a
        forex broker, VIX75/Boom via Deriv) do NOT publish real depth — MT5
        signals this by `market_book_add` returning `False`, or a `None`/empty
        `market_book_get` result, not by raising. Both are treated as "no
        depth for this symbol" (`None` return, not an error) — only a genuine
        connection failure raises `Mt5Error`, matching this class's other
        methods. `market_book_release` always runs (via `finally`) once
        `market_book_add` has succeeded, so a subscription is never leaked on
        an exception from `market_book_get`.
        """
        self._require_connection()
        self._select(symbol)
        if not mt5.market_book_add(symbol):
            return None  # broker/symbol doesn't support market depth
        try:
            book = mt5.market_book_get(symbol)
        finally:
            mt5.market_book_release(symbol)
        if not book:
            return None
        return {
            "symbol": symbol,
            "time": int(datetime.now(UTC).timestamp()),
            "levels": [
                {
                    "type": "bid" if b.type == mt5.BOOK_TYPE_BUY else "ask",
                    "price": float(b.price),
                    "volume": float(getattr(b, "volume_real", 0) or b.volume),
                }
                for b in book
            ],
        }

    # ── trading ─────────────────────────────────────────────────────────

    def order_send(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl: float | None,
        tp: float | None,
        comment: str,
        magic: int = 0,
    ) -> dict[str, Any]:
        self._require_connection()
        self._select(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Mt5Error(f"symbol_info_tick({symbol}) failed: {_last_error()}")
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == "buy" else tick.bid
        volume = self._normalize_volume(symbol, volume)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": 20,
            "comment": comment,
            "magic": magic,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"order_send({symbol},{side}) rejected: retcode={code}{_retcode_reason(code)} "
                f"{_last_error()}"
            ,
                retcode=code,
            )
        return {
            "ticket": int(result.order),
            "symbol": symbol,
            "side": side,
            "volume": float(result.volume),
            "price": float(result.price),
            "sl": sl,
            "tp": tp,
            "time": int(tick.time),
            "spread_points": int(mt5.symbol_info(symbol).spread),
            "comment": comment,
            "magic": magic,
            "profit": None,
            "retcode": int(result.retcode),
        }

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._require_connection()
        rows = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if rows is None:
            return []
        return [self._position_dict(p) for p in rows]

    def position_modify(self, ticket: int, sl: float | None, tp: float | None) -> None:
        self._require_connection()
        position = self._get_position(ticket)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode == TRADE_RETCODE_NO_CHANGES:
            logger.info("position_modify(%s): sl/tp already match request, no-op", ticket)
            return
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"position_modify({ticket}) rejected: retcode={code}{_retcode_reason(code)} "
                f"{_last_error()}"
            ,
                retcode=code,
            )

    def position_close(self, ticket: int, volume: float | None = None) -> dict[str, Any]:
        self._require_connection()
        position = self._get_position(ticket)
        close_volume = volume if volume is not None else float(position.volume)
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            raise Mt5Error(f"symbol_info_tick({position.symbol}) failed: {_last_error()}")
        is_buy = position.type == mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(position.symbol),
        }
        # Realized profit isn't returned by order_send; approximate it from the
        # position's floating profit at the moment of the close request.
        profit = float(position.profit) * (close_volume / float(position.volume))
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"position_close({ticket}) rejected: retcode={code}{_retcode_reason(code)} "
                f"{_last_error()}"
            ,
                retcode=code,
            )
        return {
            "ticket": ticket,
            "symbol": position.symbol,
            "side": "buy" if is_buy else "sell",
            "volume": close_volume,
            "price": float(result.price),
            "sl": float(position.sl) if position.sl else None,
            "tp": float(position.tp) if position.tp else None,
            "time": int(tick.time),
            "spread_points": int(mt5.symbol_info(position.symbol).spread),
            "comment": position.comment,
            "profit": profit,
            "retcode": int(result.retcode),
        }

    def position_close_info(self, ticket: int) -> dict[str, Any] | None:
        """How `ticket` actually closed, from MT5's deal history — used to
        detect and reconcile broker-side SL/TP fills the backend never
        initiated (Phase 9). `None` if MT5 has no deal history for it
        (unknown ticket, or history purged)."""
        self._require_connection()
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        exit_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if not exit_deals:
            return None
        last_exit = max(exit_deals, key=lambda d: d.time)
        entry_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN]
        if entry_deals:
            side = "buy" if entry_deals[0].type == mt5.DEAL_TYPE_BUY else "sell"
        else:
            # No entry deal in this history window (rare) — infer from the
            # exit fill instead: a BUY exit closed a short, a SELL exit
            # closed a long.
            side = "sell" if last_exit.type == mt5.DEAL_TYPE_BUY else "buy"
        return {
            "ticket": ticket,
            "symbol": last_exit.symbol,
            "side": side,
            "close_price": float(last_exit.price),
            "close_time": int(last_exit.time),
            "profit": float(sum(d.profit for d in deals)),
        }

    def place_pending_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float | None,
        tp: float | None,
        comment: str,
    ) -> dict[str, Any]:
        self._require_connection()
        self._select(symbol)
        mt5_type = {
            ("buy", "limit"): mt5.ORDER_TYPE_BUY_LIMIT,
            ("sell", "limit"): mt5.ORDER_TYPE_SELL_LIMIT,
            ("buy", "stop"): mt5.ORDER_TYPE_BUY_STOP,
            ("sell", "stop"): mt5.ORDER_TYPE_SELL_STOP,
        }[(side, order_type)]
        volume = self._normalize_volume(symbol, volume)
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"place_pending_order({symbol},{side},{order_type}) rejected: "
                f"retcode={code}{_retcode_reason(code)} {_last_error()}"
            ,
                retcode=code,
            )
        return {
            "ticket": int(result.order),
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "volume": volume,
            "price": price,
            "sl": sl,
            "tp": tp,
            "placed_time": int(datetime.now(UTC).timestamp()),
            "comment": comment,
        }

    def modify_pending_order(
        self, ticket: int, price: float | None, sl: float | None, tp: float | None
    ) -> None:
        self._require_connection()
        order = self._get_pending_order(ticket)
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": ticket,
            "symbol": order.symbol,
            "price": price if price is not None else order.price_open,
            "sl": sl if sl is not None else order.sl,
            "tp": tp if tp is not None else order.tp,
            "type": order.type,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_type(order.symbol),
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode == TRADE_RETCODE_NO_CHANGES:
            logger.info(
                "modify_pending_order(%s): price/sl/tp already match request, no-op", ticket
            )
            return
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"modify_pending_order({ticket}) rejected: retcode={code}{_retcode_reason(code)} "
                f"{_last_error()}"
            ,
                retcode=code,
            )

    def cancel_pending_order(self, ticket: int) -> None:
        self._require_connection()
        request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result is not None else None
            raise Mt5Error(
                f"cancel_pending_order({ticket}) rejected: retcode={code}{_retcode_reason(code)} "
                f"{_last_error()}"
            ,
                retcode=code,
            )

    def pending_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self._require_connection()
        rows = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if rows is None:
            return []
        return [self._pending_order_dict(r) for r in rows]

    _PENDING_TYPE_NAMES = {
        "ORDER_TYPE_BUY_LIMIT": ("buy", "limit"),
        "ORDER_TYPE_SELL_LIMIT": ("sell", "limit"),
        "ORDER_TYPE_BUY_STOP": ("buy", "stop"),
        "ORDER_TYPE_SELL_STOP": ("sell", "stop"),
    }

    @classmethod
    def _pending_order_dict(cls, o: Any) -> dict[str, Any]:
        side, order_type = next(
            (side, otype)
            for name, (side, otype) in cls._PENDING_TYPE_NAMES.items()
            if o.type == getattr(mt5, name)
        )
        return {
            "ticket": int(o.ticket),
            "symbol": o.symbol,
            "side": side,
            "order_type": order_type,
            "volume": float(o.volume_current),
            "price": float(o.price_open),
            "sl": float(o.sl) if o.sl else None,
            "tp": float(o.tp) if o.tp else None,
            "placed_time": int(o.time_setup),
            "comment": o.comment,
        }

    # ENUM_SYMBOL_TRADE_EXECUTION filling-mode bitmask (MQL5 docs) — the
    # MetaTrader5 Python package never wraps these as SYMBOL_FILLING_* module
    # attributes (only the unrelated ORDER_FILLING_* enum), so referencing
    # `mt5.SYMBOL_FILLING_FOK` raises AttributeError on every real terminal.
    _SYMBOL_FILLING_FOK = 1
    _SYMBOL_FILLING_IOC = 2

    def _filling_type(self, symbol: str) -> int:
        """The order-filling mode a symbol actually accepts. Hardcoding IOC
        rejects with retcode=10030 (unsupported filling mode) on brokers/
        symbols — commonly synthetic indices — that only support FOK or
        neither, in which case RETURN (no explicit filling type, exchange-
        style execution) is the only option left. `symbol_info().filling_mode`
        is a bitmask of what's supported (`SYMBOL_FILLING_*`, distinct from
        the `ORDER_FILLING_*` enum a request's `type_filling` expects)."""
        info = mt5.symbol_info(symbol)
        mode = info.filling_mode if info is not None else 0
        if mode & self._SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if mode & self._SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """Clamp/snap a requested lot size to what the symbol actually
        accepts. Synthetic-index symbols (e.g. Deriv's Boom/Crash) commonly
        have a `volume_min`/`volume_step` far from the 0.01 forex default —
        sending a smaller or off-step volume gets retcode=10014 (invalid
        volume) rather than an auto-adjustment, so callers that don't know a
        symbol's step get bumped up to its minimum instead of rejected."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return volume
        step = info.volume_step or volume
        snapped = round(round(volume / step) * step, 8)
        return min(max(snapped, info.volume_min), info.volume_max)

    def _get_pending_order(self, ticket: int) -> Any:
        rows = mt5.orders_get(ticket=ticket)
        if not rows:
            raise Mt5Error(f"no pending order with ticket {ticket}")
        return rows[0]

    def _get_position(self, ticket: int) -> Any:
        rows = mt5.positions_get(ticket=ticket)
        if not rows:
            raise Mt5Error(f"no open position with ticket {ticket}")
        return rows[0]

    @staticmethod
    def _position_dict(p: Any) -> dict[str, Any]:
        return {
            "ticket": int(p.ticket),
            "symbol": p.symbol,
            "side": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
            "volume": float(p.volume),
            "open_price": float(p.price_open),
            "sl": float(p.sl) if p.sl else None,
            "tp": float(p.tp) if p.tp else None,
            "open_time": int(p.time),
            "profit": float(p.profit),
            "comment": p.comment,
            "magic": int(p.magic),
        }

    # ── internals ───────────────────────────────────────────────────────

    def _require_connection(self) -> None:
        if mt5 is None:
            raise Mt5Error("MetaTrader5 package unavailable — run the gateway on Windows/Wine")
        if not self._connected:
            raise Mt5Error("not logged in — POST /login first")

    def _select(self, symbol: str) -> None:
        # Symbols must be in Market Watch before data calls return anything.
        if not mt5.symbol_select(symbol, True):
            raise Mt5Error(f"symbol_select({symbol}) failed: {_last_error()}")

    @staticmethod
    def _timeframe(timeframe: str) -> int:
        name = _MT5_CONSTANT_NAMES.get(timeframe, timeframe)
        try:
            return getattr(mt5, f"TIMEFRAME_{name}")
        except AttributeError:
            raise Mt5Error(f"unsupported timeframe {timeframe!r}") from None


client = Mt5Client()
