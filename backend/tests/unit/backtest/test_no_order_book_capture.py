"""Guards `run_backtest.py`'s `TradeEngine(...)` construction against ever
being wired with a live-gateway-backed `order_book_capture` (order_book/
Phase 5). Backtests replay historical candles and must never make a network
call to a live gateway — `TradeEngine`'s `order_book_capture` parameter
defaults to `None`, and the `if self._order_book_capture is not None` guard
in `engine/application/trade_loop.py` skips the capture task entirely when
it's absent. This test is a source-level regression guard: a future edit
that accidentally threads an `order_book_capture=` keyword into that
construction call would silently make every backtest attempt to hit a live
gateway, and this is the trip-wire for that."""

from __future__ import annotations

import inspect

from src.backtest.application import run_backtest as run_backtest_module


def test_run_backtest_never_passes_order_book_capture_to_trade_engine():
    source = inspect.getsource(run_backtest_module)
    assert "order_book_capture" not in source, (
        "run_backtest.py must never construct TradeEngine with "
        "order_book_capture= — backtests must never hit a live gateway"
    )
