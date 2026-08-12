"""Train the SMC Deep Learning multi-task model.

Runs from ``backend/``:

    uv run python scripts/train_smc_dl.py

Or imported by the optimisation loop (``run_smc_dl_loop.py``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategies.generated.smc_dl_features import (  # noqa: E402
    _compute_atr,
    compute_features_batch,
)
from src.strategies.generated.smc_dl_labels import generate_labels  # noqa: E402
from src.strategies.generated.smc_dl_model import SmcMultiTaskNet  # noqa: E402

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "trading.db"
)


def _load_candles(db_path: str, symbol: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV candles from SQLite for all required timeframes."""
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


def train(
    *,
    hidden_dim: int = 192,
    lr: float = 1e-3,
    dropout: float = 0.2,
    epochs: int = 15,
    batch_size: int = 256,
    db_path: str | None = None,
) -> dict[str, float]:
    """Train the model and save weights. Returns final metrics dict."""
    db = db_path or os.path.normpath(DB_PATH)
    if not os.path.exists(db):
        raise FileNotFoundError(f"Database not found at {db}")

    symbol = os.environ.get("SYMBOL", "XAUUSD")
    model_name = "smc_dl_m5" if symbol == "XAUUSD" else "smc_dl_m5_step200"

    print(f"Loading candle data for {symbol}...")
    candles = _load_candles(db, symbol)
    df_m5 = candles.get("M5")
    if df_m5 is None or df_m5.empty:
        raise RuntimeError(f"No M5 data found for {symbol}")

    print(f"  M5 bars: {len(df_m5)}")

    print("Computing features...")
    features_df = compute_features_batch(
        candles.get("M5", pd.DataFrame()),
        candles.get("M15", pd.DataFrame()),
        candles.get("H1", pd.DataFrame()),
        candles.get("H4", pd.DataFrame()),
    )
    if features_df.empty:
        raise RuntimeError("Feature computation returned empty DataFrame")

    print("Generating labels...")
    atr_series = _compute_atr(df_m5, 14)
    if symbol == "XAUUSD":
        labels_df = generate_labels(df_m5, atr_series, tp_atr_mult=2.0)
    else:
        labels_df = generate_labels(df_m5, atr_series, tp_atr_mult=1.5)

    # Align features and labels on index
    common_idx = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common_idx]
    y = labels_df.loc[common_idx]

    # Drop NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if X.empty:
        raise RuntimeError("No valid data after cleaning")

    X['sample_weight'] = 1.0
    X['is_live'] = 0.0

    csv_path = os.path.join(os.path.dirname(db), f"dataset_{symbol}.csv")
    if os.path.exists(csv_path):
        live_df = pd.read_csv(csv_path)
        if not live_df.empty:
            cols = X.columns.drop(['sample_weight', 'is_live'])
            live_X = live_df[cols].copy()
            live_X['sample_weight'] = 2.0
            live_X['is_live'] = 1.0
            if 'timestamp_entry' in live_df.columns:
                ts = pd.to_datetime(live_df['timestamp_entry'])
                if X.index.tz is not None:
                    ts = ts.dt.tz_convert(X.index.tz)
                live_X.index = ts
            
            live_y = pd.DataFrame({
                'hit_tp_before_sl': live_df['hit_tp_before_sl'],
                'direction': live_df['actual_direction'],
                'mae': live_df['mae_atr']
            }, index=live_X.index)
            
            X = pd.concat([X, live_X])
            y = pd.concat([y, live_y])

    # Temporal split: train < split_date, val >= split_date
    start_date_str = os.environ.get("START_DATE", "2026-06-01")
    # Ensure it's a valid date string for Timestamp
    if len(start_date_str) == 7:
        start_date_str += "-01"
        
    split_date = pd.Timestamp(start_date_str)
    if X.index.tz is not None:
        split_date = split_date.tz_localize(X.index.tz)

    X_train, X_val = X[X.index < split_date], X[X.index >= split_date]
    y_train, y_val = y[y.index < split_date], y[y.index >= split_date]

    print(f"  Train: {len(X_train)} bars | Val: {len(X_val)} bars")
    if len(X_train) == 0:
        raise RuntimeError(f"No training data before {start_date_str}")

    w_train = X_train['sample_weight'].values.astype(np.float32)
    w_val = X_val['sample_weight'].values.astype(np.float32) if len(X_val) > 0 else np.array([])
    l_val = X_val['is_live'].values.astype(bool) if len(X_val) > 0 else np.array([])

    X_train = X_train.drop(columns=['sample_weight', 'is_live'])
    X_val = X_val.drop(columns=['sample_weight', 'is_live'])

    # Scale features
    mean = X_train.values.mean(axis=0)
    scale = X_train.values.std(axis=0)
    scale[scale < 1e-8] = 1.0  # avoid div-by-zero
    X_train_s = (X_train.values - mean) / scale
    X_val_s = (X_val.values - mean) / scale if len(X_val) > 0 else np.empty((0, X_train.shape[1]))

    # Label tensors
    def _dir_to_class(d: float) -> int:
        if d == -1:
            return 0
        if d == 1:
            return 2
        return 1

    y_train_dir = np.array([_dir_to_class(d) for d in y_train["direction"]])
    y_train_tp = np.clip(y_train["hit_tp_before_sl"].fillna(0.0).values.astype(np.float32), 0.0, 1.0)
    y_train_risk = np.clip(y_train["mae"].fillna(0.0).values / 5.0, 0, 1).astype(np.float32)

    X_t = torch.FloatTensor(X_train_s.astype(np.float32))
    y_tp_t = torch.FloatTensor(y_train_tp).unsqueeze(1)
    y_dir_t = torch.LongTensor(y_train_dir)
    y_risk_t = torch.FloatTensor(y_train_risk).unsqueeze(1)
    w_t = torch.FloatTensor(w_train).unsqueeze(1)

    train_ds = TensorDataset(X_t, y_tp_t, y_dir_t, y_risk_t, w_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation tensors
    has_val = len(X_val) > 0
    if has_val:
        y_val_dir = np.array([_dir_to_class(d) for d in y_val["direction"]])
        X_v = torch.FloatTensor(X_val_s.astype(np.float32))
        y_tp_v = torch.FloatTensor(np.clip(y_val["hit_tp_before_sl"].fillna(0.0).values.astype(np.float32), 0.0, 1.0)).unsqueeze(1)
        y_dir_v = torch.LongTensor(y_val_dir)
        y_risk_v = torch.FloatTensor(
            np.clip(y_val["mae"].fillna(0.0).values / 5.0, 0, 1).astype(np.float32)
        ).unsqueeze(1)

    # Model
    input_dim = X_t.shape[1]
    model = SmcMultiTaskNet(input_dim=input_dim, hidden_dim=hidden_dim)

    # Patch dropout if needed
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    print(f"Training: {epochs} epochs, hidden={hidden_dim}, lr={lr}, dropout={dropout}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, btp, bdir, brisk, bw in train_loader:
            optimizer.zero_grad()
            out_tp, out_dir, out_risk = model(bx)
            l_tp = bce(out_tp, btp) * bw
            l_dir = ce(out_dir, bdir) * bw.squeeze()
            l_risk = mse(out_risk, brisk) * bw
            loss = l_tp.mean() + l_dir.mean() + l_risk.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        val_str = ""
        if has_val:
            model.eval()
            with torch.no_grad():
                vtp, vdir, vrisk = model(X_v)
                l_tp = bce(vtp, y_tp_v)
                l_dir = ce(vdir, y_dir_v)
                l_risk = mse(vrisk, y_risk_v)
                val_loss = (l_tp.mean() + l_dir.mean() + l_risk.mean()).item()
                
                # Accuracy tracking on live dataset subset
                if l_val.any():
                    live_mask = l_val
                    vdir_live = vdir[live_mask]
                    y_dir_live = y_dir_v[live_mask]
                    if len(vdir_live) > 0:
                        preds = vdir_live.argmax(dim=1)
                        acc = (preds == y_dir_live).float().mean().item()
                        val_str = f" | Val Loss: {val_loss:.4f} | Live Acc: {acc:.2%}"
                    else:
                        val_str = f" | Val Loss: {val_loss:.4f}"
                else:
                    val_str = f" | Val Loss: {val_loss:.4f}"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train:.4f}{val_str}")

    # Save model + scaler
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "ml_models")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{model_name}.pt")
    scaler_path = os.path.join(out_dir, f"{model_name}_scaler.npz")

    torch.save(model.state_dict(), model_path)
    # Save enriched model too
    enriched_model_path = os.path.join(out_dir, f"{model_name}_enriched.pt")
    torch.save(model.state_dict(), enriched_model_path)
    np.savez(scaler_path, mean=mean, scale=scale)
    print(f"Model saved to {model_path} and {enriched_model_path}")

    return {
        "train_loss": avg_train,
        "val_loss": float(val_loss) if has_val else 0.0,
        "input_dim": input_dim,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
