"""Export closed live trades as a training dataset for the SMC deep-learning
model: each row joins one closed `TradeRecord` to the v2 feature vector
(`strategies/generated/smc_dl_features_v2.py`) computed at its entry bar,
plus the v1 triple-barrier label and every enrichment field this dataset
export can attach (regime tags, transaction cost, the entry candle's
volume/ATR/day-of-week, and — when the trade carries a `signal_id` — the
captured order-book snapshot).

Reads only already-stored data through the same hexagonal ports the rest of
the app uses (`CandleRepository`, `SymbolSpecRepository`,
`OrderBookSnapshotRepository` via `OrderBookCaptureService`) — no raw DB
connections, no live-gateway calls, so this endpoint works even when the MT5
gateway is unreachable, which is normal for reprocessing historical data.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response

from src.container import AccountRuntime
from src.journal.api.schemas import DatasetOrderBookLevelOut, DatasetRowOut
from src.journal.domain.models import TradeRecord
from src.market_data.domain.models import Candle, Timeframe
from src.order_book.domain.models import OrderBookSnapshot
from src.shared.api.dependencies import AccountRuntimeDep
from src.strategies.generated import smc_dl_features_v2
from src.strategies.generated.smc_dl_features_v2 import (
    FEATURE_NAMES,
    MIN_M5_BARS,
    compute_features_batch,
)
from src.strategies.generated.smc_dl_labels import generate_labels

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["journal-export"])

# Higher timeframes the v2 feature module merges in (see its `_merge_htf`).
_HTF_TIMEFRAMES = (Timeframe.M15, Timeframe.H1, Timeframe.H4)

# How far before the earliest trade's entry to pull candle history so every
# feature's trailing window (deepest: 60 H4 bars = 10 days, per
# `_MAX_LOOKBACK_HTF` in smc_dl_features_v2.py) and every rolling
# indicator's warmup (atr_50, ema_100, etc.) is actually filled in by the
# time the earliest trade is reached — comfortably more than needed, since
# weekend/holiday closures eat into calendar days without adding bars.
_HISTORY_LOOKBACK_BUFFER = timedelta(days=25)
# `CandleRepository.get_range`'s `end` is exclusive; pad past the last
# trade's close so its own entry bar (and the label's forward barrier walk)
# has somewhere to land.
_HISTORY_LOOKAHEAD_BUFFER = timedelta(days=2)

_FEATURE_NAME_SET = frozenset(FEATURE_NAMES)


def _as_utc(value: datetime) -> pd.Timestamp:
    """`TradeRecord` timestamps are UTC by convention (CLAUDE.md), but
    aren't type-enforced tz-aware — treat a naive value as already UTC
    rather than letting `datetime.timestamp()` silently interpret it in the
    host's local timezone (which is what `CandleRepository.get_range` calls
    under the hood)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


_REASON_RE = {
    "tp": re.compile(r"tp=([0-9.]+)"),
    "bull": re.compile(r"bull=([0-9.]+)"),
    "bear": re.compile(r"bear=([0-9.]+)"),
}


def parse_reason_for_probs(reason: str) -> dict[str, float | str]:
    """Parses a strategy's logged reason string, e.g.
    'DL Buy (tp=0.85, bull=0.58, bear=0.28)', into the probabilities and
    decision it recorded. Trades whose `reason` doesn't carry this shape
    (most non-DL bots) come back as all-zero/'SKIP', not an error."""
    probs: dict[str, float | str] = {
        "tp_prob": 0.0,
        "bull_prob": 0.0,
        "bear_prob": 0.0,
        "model_decision": "SKIP",
    }
    if "Buy" in reason:
        probs["model_decision"] = "BUY"
    elif "Sell" in reason:
        probs["model_decision"] = "SELL"

    for key, pattern in _REASON_RE.items():
        match = pattern.search(reason)
        if match:
            probs[f"{key}_prob"] = float(match.group(1))
    return probs


def _candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    """`list[Candle]` (oldest first, as `CandleRepository.get_range` returns
    them) to the OHLCV(+enrichment) frame `compute_features_batch` and
    `generate_labels` expect: a UTC `DatetimeIndex` and lowercase columns."""
    if not candles:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread_points",
                "real_volume",
                "atr_14",
                "day_of_week",
            ]
        )
    return pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "tick_volume": [c.tick_volume for c in candles],
            "spread_points": [c.spread_points for c in candles],
            "real_volume": [c.real_volume for c in candles],
            "atr_14": [c.atr_14 for c in candles],
            "day_of_week": [c.day_of_week for c in candles],
        },
        index=pd.DatetimeIndex([c.time for c in candles], name="time"),
    )


async def _load_candle_frames(
    account: AccountRuntime, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> dict[Timeframe, pd.DataFrame]:
    """Every timeframe `compute_features_batch` needs, read straight from
    the local DB via `CandleRepository.get_range` — never the live gateway
    (see module docstring). Fetched concurrently since these are independent
    reads of the same shared SQLite DB via `asyncio.to_thread`."""
    timeframes = (Timeframe.M5, *_HTF_TIMEFRAMES)
    results = await asyncio.gather(
        *(
            asyncio.to_thread(
                account.candle_repository.get_range,
                symbol,
                tf,
                start.to_pydatetime(),
                end.to_pydatetime(),
                account.id,
            )
            for tf in timeframes
        )
    )
    return {tf: _candles_to_frame(candles) for tf, candles in zip(timeframes, results, strict=True)}


async def _load_order_books(
    account: AccountRuntime, records: list[TradeRecord]
) -> dict[str, OrderBookSnapshot]:
    """Order-book snapshots for every distinct `signal_id` among `records`,
    fetched concurrently via `OrderBookCaptureService.get_for_signal` — the
    same graceful-degradation read path `GET .../order-book/signal/
    {signal_id}` uses. Absent for most signals/symbols by design (most of
    this bot's traded symbols report no market depth); that is `None`, never
    an error and never fabricated."""
    signal_ids = sorted({r.signal_id for r in records if r.signal_id})
    if not signal_ids:
        return {}
    snapshots = await asyncio.gather(
        *(account.order_book_capture.get_for_signal(sid) for sid in signal_ids)
    )
    return {sid: snap for sid, snap in zip(signal_ids, snapshots, strict=True) if snap is not None}


def _order_book_levels_out(
    snapshot: OrderBookSnapshot | None,
) -> list[DatasetOrderBookLevelOut] | None:
    if snapshot is None or not snapshot.levels:
        return None
    return [
        DatasetOrderBookLevelOut(side=level.side.value, price=level.price, volume=level.volume)
        for level in snapshot.levels
    ]


_RESPONSES = {
    404: {
        "description": (
            "Either `symbol` has no closed trades in the journal, or the local DB has no M5 "
            "candles at all in the range spanned by those trades (or too few for the v2 "
            "feature set's warmup — see `MIN_M5_BARS`) — run `POST "
            ".../market-data/backfill` for `symbol` first."
        )
    }
}


@router.get(
    "/dataset",
    response_model=list[DatasetRowOut],
    summary="Export closed trades as an SMC v2 training dataset",
    description=(
        "Joins every closed trade on `symbol` to the SMC v2 feature vector "
        "(`strategies/generated/smc_dl_features_v2.py` — see its `FEATURE_NAMES` for the "
        "current count/list) computed at its entry bar from locally stored candle history, "
        "plus the v1 triple-barrier label and "
        "regime/cost/order-book enrichment already on the trade record. Use this to build or "
        "refresh the DL model's training data — not for live inference (see "
        "`compute_features_live` for that). `format=json` returns the typed array below "
        "(`DatasetRowOut`); `format=csv`/`format=parquet` return the same rows flattened into "
        "one column per feature, as a file download (see this operation's alternate 200 "
        "content types) rather than the JSON schema shown here. Order-book fields are null "
        "whenever no depth was ever captured for a trade's signal — the normal case for most "
        "symbols, not an error. Reads only already-stored data (no live-gateway calls), so it "
        "works even with the MT5 gateway disconnected."
    ),
    responses={
        200: {
            "content": {
                "text/csv": {"schema": {"type": "string"}},
                "application/octet-stream": {
                    "description": "Parquet file (same rows, `format=parquet`).",
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        },
        **_RESPONSES,
    },
)
async def export_dataset(
    account: AccountRuntimeDep,
    symbol: str = Query(..., description="Trading symbol, e.g. 'XAUUSD'."),
    format: Literal["csv", "json", "parquet"] = Query(
        "csv",
        description="Output format: 'csv' or 'parquet' download the dataset as a file; "
        "'json' returns a typed array of rows.",
    ),
    strategy_version: str | None = Query(
        default=None,
        description="Restrict to trades from this exact strategy version, e.g. "
        "'breakout_v1:v1'. Omit to include every strategy/manual trade on `symbol`.",
    ),
) -> list[DatasetRowOut] | Response:
    service = account.trade_journal

    limit = 10000
    records, _, _ = await service.search_trades(
        symbol=symbol,
        strategy_version=strategy_version,
        limit=limit,
    )
    closed_records = [r for r in records if not r.is_open]
    if not closed_records:
        raise HTTPException(status_code=404, detail=f"No closed trades found for {symbol}")

    open_times = [_as_utc(r.open_time) for r in closed_records]
    close_times = [_as_utc(r.close_time) for r in closed_records if r.close_time is not None]
    range_start = min(open_times) - _HISTORY_LOOKBACK_BUFFER
    range_end = max(close_times or open_times) + _HISTORY_LOOKAHEAD_BUFFER

    frames = await _load_candle_frames(account, symbol, range_start, range_end)
    df_m5 = frames[Timeframe.M5]
    if df_m5.empty:
        raise HTTPException(
            status_code=404, detail=f"No M5 candles found for {symbol} in the trade date range"
        )

    # Symbol contract size, for the R-multiple formula below — synced from
    # the gateway at backfill time (see `market_data.application.history.
    # sync_symbol_spec`), never fetched live here.
    spec = await asyncio.to_thread(account.symbol_spec_repository.get, symbol, account.id)

    spread_series = df_m5["spread_points"].astype(float)
    features_df = compute_features_batch(
        df_m5,
        frames[Timeframe.M15],
        frames[Timeframe.H1],
        frames[Timeframe.H4],
        spread_points=spread_series,
        point_value=spec.point if spec is not None else 0.01,
    )
    if features_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Fewer than {MIN_M5_BARS} M5 candles ({len(df_m5)} found) for {symbol} in the "
                "trade date range — not enough history for the v2 feature set's warmup"
            ),
        )

    # v1's triple-barrier label (`hit_tp_before_sl`/mfe/mae/bars_to_resolution) needs an ATR
    # series; v2's public `atr()` replaces the old module's private `_compute_atr` (same
    # Wilder-style rolling ATR, already computed for the feature set above).
    atr_series = smc_dl_features_v2.atr(df_m5, 14)
    labels_df = generate_labels(df_m5, atr_series)

    order_books = await _load_order_books(account, closed_records)

    rows: list[dict] = []
    skipped_no_bar = 0
    for r in closed_records:
        entry_time = _as_utc(r.open_time)

        mask = features_df.index <= entry_time
        if not mask.any():
            skipped_no_bar += 1
            continue
        entry_idx = features_df.index[mask][-1]

        feats = features_df.loc[entry_idx].to_dict()

        if entry_idx not in labels_df.index:
            skipped_no_bar += 1
            continue
        labs = labels_df.loc[entry_idx].to_dict()
        candle_row = df_m5.loc[entry_idx]

        probs = parse_reason_for_probs(r.reason)

        actual_direction = int(labs.get("direction", 0))
        model_correct = (
            1
            if (
                (probs["model_decision"] == "BUY" and actual_direction == 1)
                or (probs["model_decision"] == "SELL" and actual_direction == -1)
            )
            else 0
        )
        hit_tp = int(labs.get("hit_tp_before_sl", 0))
        conf_calib = float(probs["tp_prob"]) - hit_tp

        # R-multiple = profit / initial risk, where initial risk is the SL
        # distance converted to account currency — the same formula
        # `backtest/adapters/bookkeeper.py::on_position_closed` uses for
        # `BacktestTrade.r_multiple`, reused here rather than inventing a
        # second definition of "R" for live trades. Null (not a fabricated
        # 0/guess) when the trade has no `sl` or this symbol's contract size
        # was never synced.
        profit_r: float | None = None
        if r.sl is not None and spec is not None and r.profit is not None:
            initial_risk = abs(r.open_price - r.sl) * r.volume * spec.contract_size
            if initial_risk > 0:
                profit_r = r.profit / initial_risk

        snapshot = order_books.get(r.signal_id) if r.signal_id else None

        row: dict = {
            "trade_id": r.id,
            "symbol": r.symbol,
            "strategy_version": r.strategy_version,
            "timestamp_entry": r.open_time.isoformat(),
            "timestamp_exit": r.close_time.isoformat() if r.close_time else None,
            "tp_prob": probs["tp_prob"],
            "bull_prob": probs["bull_prob"],
            "bear_prob": probs["bear_prob"],
            "model_decision": probs["model_decision"],
            "actual_direction": actual_direction,
            "hit_tp_before_sl": hit_tp,
            "profit_r": profit_r,
            "mfe_atr": float(labs.get("mfe", 0.0)),
            "mae_atr": float(labs.get("mae", 0.0)),
            "bars_to_exit": int(labs.get("bars_to_resolution", 0)),
            "entry_price": r.open_price,
            "exit_price": r.close_price,
            "sl": r.sl,
            "tp": r.tp,
            "spread": r.spread_points_at_entry,
            "slippage": r.slippage,
            "profit_raw": r.profit,
            "volume": r.volume,
            "model_correct": model_correct,
            "confidence_calibration": conf_calib,
            # Enrichment: regime/cost/signal, already on the trade record.
            "regime_volatility": r.regime_volatility,
            "regime_volatility_percentile": r.regime_volatility_percentile,
            "regime_trend": r.regime_trend,
            "regime_adx": r.regime_adx,
            "regime_session": r.regime_session,
            "transaction_cost": r.transaction_cost,
            "signal_id": r.signal_id,
            # Enrichment: the M5 candle the entry features were built from.
            "real_volume": (
                int(candle_row["real_volume"]) if pd.notna(candle_row["real_volume"]) else None
            ),
            "atr_14": (float(candle_row["atr_14"]) if pd.notna(candle_row["atr_14"]) else None),
            "day_of_week": (
                int(candle_row["day_of_week"]) if pd.notna(candle_row["day_of_week"]) else None
            ),
        }
        row["_order_book_levels"] = _order_book_levels_out(snapshot)

        # The SMC v2 features (FEATURE_NAMES), merged in as their own
        # top-level columns (flat, for the CSV/parquet path).
        for name in FEATURE_NAMES:
            value = feats.get(name)
            row[name] = float(value) if value is not None and np.isfinite(value) else None
        rows.append(row)

    if skipped_no_bar:
        logger.info(
            "export_dataset: skipped %s/%s trades for %s with no matching entry bar",
            skipped_no_bar,
            len(closed_records),
            symbol,
        )

    if format == "json":
        return [
            DatasetRowOut(
                **{
                    k: v
                    for k, v in row.items()
                    if k not in _FEATURE_NAME_SET and k != "_order_book_levels"
                },
                order_book_levels=row["_order_book_levels"],
                features={name: row[name] for name in FEATURE_NAMES if row[name] is not None},
            )
            for row in rows
        ]

    # CSV/parquet: flatten order-book levels out (not tabular-friendly) and
    # drop the internal-only key; every other column, including the SMC v2
    # feature columns, is written as-is.
    for row in rows:
        row["order_book_levels"] = (
            None if row["_order_book_levels"] is None else len(row["_order_book_levels"])
        )
        del row["_order_book_levels"]

    out_df = pd.DataFrame(rows)
    if format == "csv":
        csv_data = out_df.to_csv(index=False)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=dataset_{symbol}.csv"},
        )
    buf = io.BytesIO()
    out_df.to_parquet(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=dataset_{symbol}.parquet"},
    )
