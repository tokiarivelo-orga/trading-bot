"""MFE / MAE: how far a trade ran in its favor and against it (Phase 3).

Pure arithmetic — no I/O, no framework imports. Both numbers are in **price
units** and are **non-negative magnitudes** measured from the entry price:

  - MFE (maximum favorable excursion): the best unrealized price the trade
    ever saw. A large MFE on a *losing* trade means the take-profit was
    parked beyond where the market actually turned.
  - MAE (maximum adverse excursion): the worst unrealized price the trade
    ever saw. A large MAE on a *winning* trade means the stop-loss survived
    only narrowly — tighten it and the winner would have been a loser.

Where this is computed, and why
-------------------------------
Accumulation lives in `journal/application/trade_journal.py`, driven by the
`CandleClosed` event, and is finalized in `on_position_closed`:

  - the journal already owns the `TradeRecord` these values belong to and
    already subscribes to the event bus, so no other module has to learn
    about excursion or reach into the journal's storage;
  - the engine's `PositionManager` also sees every candle for an open
    position, but it is deliberately journal-free (it manages broker
    positions, including ones the journal never recorded), and giving it a
    write path into the journal would couple two modules that today only
    meet on the bus.

Both the per-candle step and the finalize step go through the functions
below, so the "opened and closed inside a single candle" case (no candle
ever closes during the position's life) still produces real numbers: the
close price alone is enough to extend the excursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

BUY = "buy"


@dataclass(frozen=True, kw_only=True)
class Excursion:
    """A trade's excursion so far. Starts at zero on both axes at entry —
    at the instant of the fill the market has not yet moved either way."""

    mfe: float = 0.0
    mae: float = 0.0
    mfe_time: datetime | None = None
    mae_time: datetime | None = None


def extend_excursion(
    current: Excursion,
    *,
    side: str,
    open_price: float,
    high: float,
    low: float,
    time: datetime,
) -> Excursion:
    """Widens `current` with a price range the trade lived through.

    `high`/`low` are a candle's extremes while the position was open; pass the
    same value for both to extend with a single price (that is how the close
    price is folded in on finalize). Never narrows either number — excursion
    is a running maximum, so a later quiet candle cannot undo an earlier
    spike."""
    if side == BUY:
        favorable = high - open_price
        adverse = open_price - low
    else:
        favorable = open_price - low
        adverse = high - open_price
        
    new_mfe = max(current.mfe, favorable, 0.0)
    mfe_time = time if new_mfe > current.mfe or current.mfe_time is None else current.mfe_time
    
    new_mae = max(current.mae, adverse, 0.0)
    mae_time = time if new_mae > current.mae or current.mae_time is None else current.mae_time
    
    return Excursion(
        mfe=new_mfe,
        mae=new_mae,
        mfe_time=mfe_time,
        mae_time=mae_time,
    )


def finalize_excursion(
    current: Excursion, *, side: str, open_price: float, close_price: float, time: datetime
) -> Excursion:
    """Excursion at the moment the trade closed: whatever was accumulated
    from closed candles, extended by the exit price itself. A position that
    opened and closed within one candle never saw a candle close, so this
    call is the only thing that gives it non-zero numbers."""
    return extend_excursion(
        current, side=side, open_price=open_price, high=close_price, low=close_price, time=time
    )
