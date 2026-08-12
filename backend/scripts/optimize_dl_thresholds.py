import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.generated.smc_dl_features import _compute_atr, compute_features_batch
from src.strategies.generated.smc_dl_labels import generate_labels
from src.strategies.generated.smc_dl_model import SmcMultiTaskNet

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading.db")

def _load_candles(db_path: str, symbol: str) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    dfs: dict[str, pd.DataFrame] = {}
    for tf in ("M5", "M15", "H1", "H4"):
        query = (
            "SELECT time, open, high, low, close, tick_volume "
            "FROM candles "
            f"WHERE symbol=? AND timeframe=? "
            "ORDER BY time"
        )
        df = pd.read_sql_query(query, conn, params=(symbol, tf))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("timestamp")
        dfs[tf] = df
    conn.close()
    return dfs

def optimize(symbol: str):
    model_name = "smc_dl_m5" if symbol == "XAUUSD" else "smc_dl_m5_step200"
    print(f"\n--- Optimizing for {symbol} ({model_name}) ---")
    
    candles = _load_candles(DB_PATH, symbol)
    df_m5 = candles.get("M5")
    
    features_df = compute_features_batch(
        candles.get("M5", pd.DataFrame()),
        candles.get("M15", pd.DataFrame()),
        candles.get("H1", pd.DataFrame()),
        candles.get("H4", pd.DataFrame()),
    )
    
    atr_series = _compute_atr(df_m5, 14)
    # Use the same parameters as the strategy
    labels_df = generate_labels(df_m5, atr_series, tp_atr_mult=2.0, sl_atr_mult=1.0)
    
    common_idx = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common_idx]
    y = labels_df.loc[common_idx]
    
    valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    split_date = pd.Timestamp("2026-06-01", tz="UTC")
    X_val = X[X.index >= split_date]
    y_val = y[y.index >= split_date]
    
    print(f"Validation samples: {len(X_val)}")
    
    # Load scaler and model
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "ml_models")
    scaler_path = os.path.join(out_dir, f"{model_name}_scaler.npz")
    model_path = os.path.join(out_dir, f"{model_name}_enriched.pt")
    
    scaler_data = np.load(scaler_path)
    mean = scaler_data["mean"]
    scale = scaler_data["scale"]
    
    X_val_s = (X_val.values - mean) / np.clip(scale, 1e-8, None)
    
    input_dim = len(mean)
    model = SmcMultiTaskNet(input_dim=input_dim, hidden_dim=192)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    
    X_t = torch.FloatTensor(X_val_s)
    with torch.no_grad():
        # process in batches to avoid OOM just in case
        tp_probs = []
        dir_probs = []
        batch_size = 10000
        for i in range(0, len(X_t), batch_size):
            tp_prob, direction_logits, _ = model(X_t[i:i+batch_size])
            tp_probs.append(tp_prob)
            dir_probs.append(torch.softmax(direction_logits, dim=1))
        
        tp_probs = torch.cat(tp_probs).numpy().squeeze()
        dir_probs = torch.cat(dir_probs).numpy()
    
    bear_prob = dir_probs[:, 0]
    bull_prob = dir_probs[:, 2]
    
    label_direction = y_val["direction"].values
    
    results = []
    
    for min_tp in np.arange(0.35, 0.76, 0.05):
        for min_dir in np.arange(0.35, 0.76, 0.05):
            buy_mask = (tp_probs > min_tp) & (bull_prob > bear_prob) & (bull_prob > min_dir)
            sell_mask = (tp_probs > min_tp) & (bear_prob > bull_prob) & (bear_prob > min_dir)
            
            total_trades = buy_mask.sum() + sell_mask.sum()
            if total_trades < 50:
                continue
            
            wins = 0
            # Wins for buy: label_direction == 1
            wins += (buy_mask & (label_direction == 1)).sum()
            # Wins for sell: label_direction == -1
            wins += (sell_mask & (label_direction == -1)).sum()
            
            losses = total_trades - wins
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # RR is 2:1
            expected_profit_factor = (wins * 2.0) / (losses * 1.0) if losses > 0 else float('inf')
            if losses == 0 and wins > 0:
                expected_profit_factor = wins * 2.0
            
            results.append({
                "min_tp_prob": min_tp,
                "min_direction_conf": min_dir,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": expected_profit_factor
            })
            
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No combinations yielded >= 50 trades.")
        return
        
    df_res = df_res.sort_values(by="profit_factor", ascending=False).head(5)
    print("Top 5 Threshold Combinations:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    optimize("XAUUSD")
    optimize("Step Index 200")
