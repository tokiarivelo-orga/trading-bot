"""Order-book (market depth) endpoints — read-only, per signal."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path

from src.order_book.api.schemas import OrderBookLevelOut, OrderBookSnapshotOut
from src.order_book.application.capture import OrderBookCaptureService
from src.order_book.domain.models import OrderBookSnapshot
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/order-book", tags=["order-book"])


def _service(account: AccountRuntimeDep) -> OrderBookCaptureService:
    return account.order_book_capture


def _snapshot_out(signal_id: str, snapshot: OrderBookSnapshot) -> OrderBookSnapshotOut:
    return OrderBookSnapshotOut(
        signal_id=signal_id,
        symbol=snapshot.symbol,
        captured_at=int(snapshot.time.timestamp()),
        levels=[
            OrderBookLevelOut(side=level.side.value, price=level.price, volume=level.volume)
            for level in snapshot.levels
        ],
    )


@router.get(
    "/signal/{signal_id}",
    response_model=OrderBookSnapshotOut,
    summary="Get the market-depth snapshot captured for one signal",
    description=(
        "Returns the order-book (market depth) snapshot captured at the moment `signal_id` "
        "fired, for AI-training data export. This is a **best-effort, gracefully-degrading** "
        "capture: most symbols this bot trades (XAUUSD via a forex/CFD broker, VIX75/Boom via "
        "Deriv — synthetic/OTC instruments) do not publish real depth, so most signals have no "
        "row here at all — that is expected, not an error."
    ),
    responses={
        404: {
            "description": (
                "No order-book snapshot was ever captured for this signal (either the symbol "
                "reported no depth, or capture hadn't run yet)."
            )
        }
    },
)
async def get_snapshot_for_signal(
    account: AccountRuntimeDep,
    signal_id: str = Path(description="The signal id this snapshot was captured for."),
) -> OrderBookSnapshotOut:
    snapshot = await _service(account).get_for_signal(signal_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"No order-book snapshot for signal '{signal_id}'"
        )
    return _snapshot_out(signal_id, snapshot)
