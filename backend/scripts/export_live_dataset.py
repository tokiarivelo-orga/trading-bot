"""Export live dataset script for DL training integration.

Usage:
  uv run python scripts/export_live_dataset.py --symbol=XAUUSD
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.generated.smc_dl_features import compute_features_batch, _compute_atr
from src.strategies.generated.smc_dl_labels import generate_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_reason_for_probs(reason: str) -> dict[str, float]:
    import re
    probs = {"tp_prob": 0.0, "bull_prob": 0.0, "bear_prob": 0.0, "model_decision": "SKIP"}
    if "DL Buy" in reason: probs["model_decision"] = "BUY"
    elif "DL Sell" in reason: probs["model_decision"] = "SELL"
    tp = re.search(r"tp=([0-9.]+)", reason)
    bull = re.search(r"bull=([0-9.]+)", reason)
    bear = re.search(r"bear=([0-9.]+)", reason)
    if tp: probs["tp_prob"] = float(tp.group(1))
    if bull: probs["bull_prob"] = float(bull.group(1))
    if bear: probs["bear_prob"] = float(bear.group(1))
    return probs

def main():
    parser = argparse.ArgumentParser(description="Export Live Dataset")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--format", type=str, choices=["csv", "parquet"], default="csv")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    if "SYMBOL" in os.environ:
        args.symbol = os.environ["SYMBOL"]
    
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "trading.db")
    
    logger.info(f"Loading trades for {args.symbol}...")
    conn = sqlite3.connect(db_path)
    
    # Query closed trades for symbol
    trades_df = pd.read_sql_query(
        "SELECT * FROM trades WHERE symbol=? AND close_time IS NOT NULL", 
        conn, params=(args.symbol,)
    )
    
    if trades_df.empty:
        logger.error(f"No closed trades found for {args.symbol}.")
        return
        
    logger.info(f"Found {len(trades_df)} closed trades.")
    
    # Load candles
    logger.info(f"Loading candles for {args.symbol}...")
    dfs = {}
    for tf in ["M5", "M15", "H1", "H4"]:
        df = pd.read_sql_query("SELECT time, open, high, low, close, tick_volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY time", conn, params=(args.symbol, tf))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("timestamp")
        dfs[tf] = df
    conn.close()
    
    df_m5 = dfs.get("M5")
    if df_m5 is None or df_m5.empty:
        logger.error("No M5 candles found.")
        return
        
    logger.info("Computing features and labels on all history... this may take a moment.")
    features_df = compute_features_batch(df_m5, dfs.get("M15", pd.DataFrame()), dfs.get("H1", pd.DataFrame()), dfs.get("H4", pd.DataFrame()))
    atr_series = _compute_atr(df_m5, 14)
    labels_df = generate_labels(df_m5, atr_series)
    
    rows = []
    
    for _, r in trades_df.iterrows():
        entry_time = pd.Timestamp(r["open_time"], unit="s", tz="UTC")
        
        mask = features_df.index <= entry_time
        if not mask.any(): continue
        
        entry_idx = features_df.index[mask][-1]
        
        if entry_idx not in labels_df.index: continue
        
        feats = features_df.loc[entry_idx].to_dict()
        labs = labels_df.loc[entry_idx].to_dict()
        
        reason = r.get("reason", "")
        probs = parse_reason_for_probs(reason)
        
        actual_direction = 1 if r["close_price"] > r["open_price"] else -1
        model_correct = 1 if ((probs["model_decision"] == "BUY" and actual_direction == 1) or 
                              (probs["model_decision"] == "SELL" and actual_direction == -1)) else 0
        hit_tp = labs.get("hit_tp_before_sl", 0)
        
        profit = r["profit"] if pd.notna(r["profit"]) else 0.0
        sl = r.get("sl")
        entry_price = r["open_price"]
        
        if pd.isna(sl) or sl is None or sl == 0:
            profit_r = None
        else:
            denom = abs(entry_price - float(sl))
            if denom > 0:
                profit_r = profit / denom
                profit_r = max(-20.0, min(20.0, profit_r))
            else:
                profit_r = 0.0
        
        row = {
            "trade_id": r["id"],
            "symbol": r["symbol"],
            "strategy_version": r.get("strategy_version", ""),
            "timestamp_entry": entry_time.isoformat(),
            "timestamp_exit": pd.Timestamp(r["close_time"], unit="s", tz="UTC").isoformat(),
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
            "entry_price": r["open_price"],
            "exit_price": r["close_price"],
            "sl": r["sl"],
            "tp": r["tp"],
            "spread": r.get("spread_points_at_entry", 0),
            "slippage": r.get("slippage", 0.0),
            "profit_raw": r["profit"],
            "volume": r["volume"],
            "model_correct": model_correct,
            "confidence_calibration": probs["tp_prob"] - hit_tp
        }
        row.update(feats)
        rows.append(row)
        
    out_df = pd.DataFrame(rows)
    
    if out_df.empty:
        logger.error("No trades matched with candle history.")
        return
        
    out_file = args.output or os.path.join(os.path.dirname(__file__), "..", "data", f"dataset_{args.symbol}.{args.format}")
    
    if args.format == "csv":
        out_df.to_csv(out_file, index=False)
    else:
        out_df.to_parquet(out_file, index=False)
        
    logger.info(f"Dataset exported to {out_file} with {len(out_df)} records.")
    
    # Print statistics
    logger.info("================ STATISTICS ================")
    logger.info(f"Total Rows: {len(out_df)}")
    if len(out_df) > 0:
        logger.info(f"Win Rate (Hit TP): {out_df['hit_tp_before_sl'].mean():.2%}")
        dl_mask = out_df['model_decision'] != 'SKIP'
        if dl_mask.any():
            logger.info(f"Model Correct Direction Rate: {out_df.loc[dl_mask, 'model_correct'].mean():.2%}")
        else:
            logger.info(f"Model Correct Direction Rate: 0.00%")
        logger.info(f"Avg Profit (R): {out_df['profit_r'].mean():.2f}")
        logger.info(f"Label Balance (Direction): \n{out_df['actual_direction'].value_counts(normalize=True).to_string()}")
    logger.info("============================================")

if __name__ == "__main__":
    main()
