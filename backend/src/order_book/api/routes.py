"""Order-book (market depth) endpoints — read-only, per signal or bulk export."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from src.order_book.api.schemas import OrderBookLevelOut, OrderBookSnapshotOut
from src.order_book.application.capture import OrderBookCaptureService
from src.order_book.domain.models import OrderBookSnapshot
from src.shared.api.dependencies import AccountRuntimeDep

router = APIRouter(prefix="/accounts/{account_id}/order-book", tags=["order-book"])

_CSV_HEADER = ("signal_id", "symbol", "captured_at", "level_index", "side", "price", "volume")


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


def _rows_to_csv(rows: list[tuple[str, OrderBookSnapshot]]) -> str:
    """One row per price level (a snapshot with 10 levels becomes 10 CSV
    rows sharing the same `signal_id`/`captured_at`) rather than a single
    JSON-encoded `levels` column — this keeps every value directly
    filterable/aggregatable in a spreadsheet or `pandas.read_csv` without a
    second JSON-decode pass, which matters more than row count here since
    real snapshots rarely carry more than a handful of levels."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    for signal_id, snapshot in rows:
        captured_at = int(snapshot.time.timestamp())
        if not snapshot.levels:
            writer.writerow([signal_id, snapshot.symbol, captured_at, "", "", "", ""])
            continue
        for index, level in enumerate(snapshot.levels):
            writer.writerow(
                [
                    signal_id,
                    snapshot.symbol,
                    captured_at,
                    index,
                    level.side.value,
                    level.price,
                    level.volume,
                ]
            )
    return buffer.getvalue()


@router.get(
    "/export",
    response_model=list[OrderBookSnapshotOut],
    summary="Bulk-export captured order-book snapshots (CSV or JSON)",
    description=(
        "Downloads every order-book (market depth) snapshot captured for this account, "
        "newest first, optionally filtered by `symbol` and/or a `[since, until]` "
        "capture-time window (epoch seconds UTC, both inclusive) — the bulk counterpart "
        "to `GET .../order-book/signal/{signal_id}`, for a downstream AI training "
        "pipeline that wants every captured snapshot rather than one at a time. This "
        "data is sparse by design (most symbols/brokers never report real depth, so most "
        "signals have no row at all) and therefore small enough that, unlike the candle "
        "export, `since`/`until` are optional and the whole result is returned in one "
        "response rather than streamed.\n\n"
        "`format=json` returns the nested shape (`OrderBookSnapshotOut.levels` as a list "
        "per snapshot). `format=csv` flattens it to one row per price level — a 10-level "
        "snapshot becomes 10 CSV rows sharing the same `signal_id`/`captured_at` — so "
        "every field stays directly usable in a spreadsheet or `pandas.read_csv` without "
        "a second JSON-decode pass; a snapshot with a fully depleted book (no levels at "
        "all — vanishingly rare, since capture already skips empty books, see module "
        "docstring) would emit one row with blank level columns.\n\n"
        "An account/symbol/window with nothing captured is not an error: both formats "
        "return an empty-but-valid response (`[]` for JSON, header-only for CSV) with a 200."
    ),
    responses={
        200: {
            "description": (
                "CSV attachment (`format=csv`) or a JSON list of `OrderBookSnapshotOut` "
                "(`format=json`). Either can be empty when nothing matches the filters."
            ),
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/OrderBookSnapshotOut"},
                    }
                },
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": (
                            "Header row `signal_id,symbol,captured_at,level_index,side,"
                            "price,volume` followed by one row per price level, newest "
                            "snapshot first."
                        ),
                    }
                },
            },
        }
    },
)
async def export_order_book(
    account: AccountRuntimeDep,
    symbol: str | None = Query(
        default=None, description="Restrict to one symbol, e.g. 'XAUUSD'. Omit for every symbol."
    ),
    since: Annotated[
        int | None,
        Query(description="Only snapshots captured at/after this instant, epoch seconds UTC."),
    ] = None,
    until: Annotated[
        int | None,
        Query(description="Only snapshots captured at/before this instant, epoch seconds UTC."),
    ] = None,
    format: Annotated[  # noqa: A002 - "format" is the clearest name for this query param
        Literal["csv", "json"], Query(description="'csv' flattens levels; 'json' nests them.")
    ] = "json",
) -> Response | list[OrderBookSnapshotOut]:
    since_dt = datetime.fromtimestamp(since, tz=UTC) if since is not None else None
    until_dt = datetime.fromtimestamp(until, tz=UTC) if until is not None else None
    rows = await _service(account).list_for_account(symbol, since_dt, until_dt)

    if format == "csv":
        return Response(
            content=_rows_to_csv(rows),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="order_book_export.csv"'},
        )
    return [_snapshot_out(signal_id, snapshot) for signal_id, snapshot in rows]
