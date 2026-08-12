"""Export live trades as a dataset for DL training."""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Response

from src.shared.api.dependencies import AccountRuntimeDep
from src.journal.domain.models import TradeRecord
from src.strategies.generated.smc_dl_features import compute_features_batch, _compute_atr
from src.strategies.generated.smc_dl_labels import generate_labels

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["journal-export"])

def parse_reason_for_probs(reason: str) -> dict[str, float]:
    """Parse reason string like 'DL Buy (tp=0.85, bull=0.58, bear=0.28)'"""
    import re
    probs = {"tp_prob": 0.0, "bull_prob": 0.0, "bear_prob": 0.0, "model_decision": "SKIP"}
    
    if "Buy" in reason:
        probs["model_decision"] = "BUY"
    elif "Sell" in reason:
        probs["model_decision"] = "SELL"
        
    tp_match = re.search(r"tp=([0-9.]+)", reason)
    bull_match = re.search(r"bull=([0-9.]+)", reason)
    bear_match = re.search(r"bear=([0-9.]+)", reason)
    
    if tp_match: probs["tp_prob"] = float(tp_match.group(1))
    if bull_match: probs["bull_prob"] = float(bull_match.group(1))
    if bear_match: probs["bear_prob"] = float(bear_match.group(1))
    
    return probs

def load_candles_for_symbol(db_path: str, symbol: str) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    dfs = {}
    for tf in ["M5", "M15", "H1", "H4"]:
        query = "SELECT time, open, high, low, close, tick_volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY time"
        df = pd.read_sql_query(query, conn, params=(symbol, tf))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("timestamp")
        dfs[tf] = df
    conn.close()
    return dfs

@router.get("/dataset")
async def export_dataset(
    account: AccountRuntimeDep,
    symbol: str = Query(..., description="Trading symbol, e.g. 'XAUUSD'"),
    format: Literal["csv", "json", "parquet"] = Query("csv"),
    strategy_version: str | None = None
):
    """Export live trades as a dataset with recomputed SMC features and labels."""
    service = account.trade_journal
    
    # Get closed trades
    limit = 10000
    records, _, _ = await service.search_trades(
        symbol=symbol, 
        strategy_version=strategy_version,
        limit=limit,
        outcome=None # we want all trades that are closed... wait, outcome isn't easily just closed. Let's filter in memory
    )
    closed_records = [r for r in records if not r.is_open]
    
    if not closed_records:
        raise HTTPException(status_code=404, detail="No closed trades found")
        
    # Recompute features requires candles.
    # In API context, it's easier to hit the local sqlite DB directly for speed, 
    # since we need a large batch of candles to compute features & labels.
    import os
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "trading.db")
    candles = load_candles_for_symbol(db_path, symbol)
    df_m5 = candles.get("M5")
    
    if df_m5 is None or df_m5.empty:
        raise HTTPException(status_code=404, detail=f"No M5 candles found for {symbol} to compute features")
        
    # Precompute all features and labels for the entire candle series
    features_df = compute_features_batch(df_m5, candles.get("M15", pd.DataFrame()), candles.get("H1", pd.DataFrame()), candles.get("H4", pd.DataFrame()))
    atr_series = _compute_atr(df_m5, 14)
    labels_df = generate_labels(df_m5, atr_series)
    
    rows = []
    for r in closed_records:
        # Find nearest M5 bar before or at open_time
        entry_time = pd.Timestamp(r.open_time).tz_convert("UTC") if r.open_time.tzinfo else pd.Timestamp(r.open_time, tz="UTC")
        
        # We need the bar whose timestamp is <= entry_time
        mask = features_df.index <= entry_time
        if not mask.any():
            continue
            
        entry_idx = features_df.index[mask][-1]
        
        # Get features at entry
        feats = features_df.loc[entry_idx].to_dict()
        
        # Get labels computed from that entry bar onwards
        if entry_idx in labels_df.index:
            labs = labels_df.loc[entry_idx].to_dict()
        else:
            continue
            
        probs = parse_reason_for_probs(r.reason)
        
        # Calculate derived metrics
        actual_direction = labs.get("direction", 0)
        model_correct = 1 if ((probs["model_decision"] == "BUY" and actual_direction == 1) or 
                              (probs["model_decision"] == "SELL" and actual_direction == -1)) else 0
        hit_tp = labs.get("hit_tp_before_sl", 0)
        conf_calib = probs["tp_prob"] - hit_tp
        
        profit_r = r.profit / abs(r.transaction_cost or 1.0) if r.transaction_cost else r.profit # Rough approximation if no R available
        
        row = {
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
            "mfe_atr": labs.get("mfe", 0),
            "mae_atr": labs.get("mae", 0),
            "bars_to_exit": labs.get("bars_to_resolution", 0),
            
            "entry_price": r.open_price,
            "exit_price": r.close_price,
            "sl": r.sl,
            "tp": r.tp,
            "spread": r.spread_points_at_entry,
            "slippage": r.slippage,
            "profit_raw": r.profit,
            "volume": r.volume,
            "account_balance_at_entry": 0, # Cannot know easily from TradeRecord
            
            "model_correct": model_correct,
            "confidence_calibration": conf_calib,
        }
        
        # Add all SMC features
        row.update(feats)
        rows.append(row)
        
    out_df = pd.DataFrame(rows)
    
    if format == "csv":
        csv_data = out_df.to_csv(index=False)
        return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=dataset_{symbol}.csv"})
    elif format == "parquet":
        buf = io.BytesIO()
        out_df.to_parquet(buf, index=False)
        return Response(content=buf.getvalue(), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=dataset_{symbol}.parquet"})
    else:
        return out_df.to_dict(orient="records")
