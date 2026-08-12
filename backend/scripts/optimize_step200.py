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

def optimize_step_index():
    symbol = "Step Index 200"
    model_name = "smc_dl_m5_step200"
    print(f"\n--- Deep Optimization for {symbol} ---")
    
    candles = _load_candles(DB_PATH, symbol)
    df_m5 = candles.get("M5")
    
    features_df = compute_features_batch(
        candles.get("M5", pd.DataFrame()),
        candles.get("M15", pd.DataFrame()),
        candles.get("H1", pd.DataFrame()),
        candles.get("H4", pd.DataFrame()),
    )
    
    atr_series = _compute_atr(df_m5, 14)
    
    # Load model
    model_path = os.path.join(os.path.dirname(DB_PATH), "ml_models", f"{model_name}.pt")
    scaler_path = os.path.join(os.path.dirname(DB_PATH), "ml_models", f"{model_name}_scaler.npz")
    
    scaler_data = np.load(scaler_path)
    scaler_mean = scaler_data["mean"]
    scaler_scale = scaler_data["scale"]
    
    model = SmcMultiTaskNet(input_dim=len(scaler_mean), hidden_dim=192)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Pre-generate labels for different RR ratios
    rr_ratios = [1.5, 2.0, 2.5, 3.0]
    labels_dict = {}
    for rr in rr_ratios:
        labels_dict[rr] = generate_labels(df_m5, atr_series, tp_atr_mult=rr, sl_atr_mult=1.0)
    
    common_idx = features_df.index
    for rr in rr_ratios:
        common_idx = common_idx.intersection(labels_dict[rr].index)
        
    X = features_df.loc[common_idx]
    
    valid_mask = X.notna().all(axis=1)
    for rr in rr_ratios:
        valid_mask &= labels_dict[rr].loc[common_idx].notna().all(axis=1)
        
    X = X.loc[valid_mask]
    for rr in rr_ratios:
        labels_dict[rr] = labels_dict[rr].loc[common_idx].loc[valid_mask]

    split_date = pd.Timestamp("2026-06-01").tz_localize(X.index.tz)
    val_mask = X.index >= split_date
    X_val = X[val_mask]
    
    for rr in rr_ratios:
        labels_dict[rr] = labels_dict[rr][val_mask]

    X_scaled = (X_val.values - scaler_mean) / scaler_scale
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        tp_logits, dir_logits, risk_logits = model(X_tensor)
        tp_probs = torch.sigmoid(tp_logits).numpy().squeeze()
        dir_probs = torch.softmax(dir_logits, dim=1).numpy()

    bear_probs = dir_probs[:, 0]
    bull_probs = dir_probs[:, 2]
    
    results = []
    
    # Sweep RR ratios and thresholds
    for rr in rr_ratios:
        y_val = labels_dict[rr]
        actual_dirs = y_val["direction"].values
        hit_tp = y_val["hit_tp_before_sl"].values
        
        for min_tp in np.arange(0.50, 0.85, 0.05):
            for min_conf in np.arange(0.60, 0.90, 0.05):
                
                # Simulate strategy logic
                buy_signals = (tp_probs > min_tp) & (bull_probs > min_conf) & (bull_probs > bear_probs)
                sell_signals = (tp_probs > min_tp) & (bear_probs > min_conf) & (bear_probs > bull_probs)
                
                total_trades = np.sum(buy_signals) + np.sum(sell_signals)
                if total_trades < 50:
                    continue
                    
                # A trade is a "win" if it hit TP and was in the right direction
                wins = np.sum(buy_signals & (actual_dirs == 1) & (hit_tp == 1)) + \
                       np.sum(sell_signals & (actual_dirs == -1) & (hit_tp == 1))
                       
                win_rate = wins / total_trades
                
                # Estimate profit factor (RR * win_rate) / (1 - win_rate)
                # This assumes win = +RR R, loss = -1 R
                losses = total_trades - wins
                pf = (wins * rr) / max(losses, 1)
                
                results.append({
                    "rr_ratio": rr,
                    "min_tp_prob": round(min_tp, 2),
                    "min_direction_conf": round(min_conf, 2),
                    "total_trades": int(total_trades),
                    "win_rate": win_rate,
                    "profit_factor": pf
                })
                
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No configurations found with > 50 trades.")
        return
        
    df_res = df_res.sort_values("profit_factor", ascending=False)
    print("Top 10 Ultra-Optimized Configurations for Step Index 200:")
    print(df_res.head(10).to_string(index=False))

if __name__ == "__main__":
    optimize_step_index()
