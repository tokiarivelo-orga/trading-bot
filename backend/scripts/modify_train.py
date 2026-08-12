import sys
import os

with open('/home/tokiarivelo/Downloads/Documents/trafing-bot/backend/scripts/train_smc_dl.py', 'r') as f:
    content = f.read()

# Replace the data preparation logic
old_data_prep = """    # Drop NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    if X.empty:
        raise RuntimeError("No valid data after cleaning")

    # Temporal split: train < split_date, val >= split_date"""

new_data_prep = """    # Drop NaN rows
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

    # Temporal split: train < split_date, val >= split_date"""

content = content.replace(old_data_prep, new_data_prep)

# Replace the split and weights extraction
old_split = """    X_train, X_val = X[X.index < split_date], X[X.index >= split_date]
    y_train, y_val = y[y.index < split_date], y[y.index >= split_date]

    print(f"  Train: {len(X_train)} bars | Val: {len(X_val)} bars")
    if len(X_train) == 0:
        raise RuntimeError(f"No training data before {start_date_str}")

    # Scale features"""

new_split = """    X_train, X_val = X[X.index < split_date], X[X.index >= split_date]
    y_train, y_val = y[y.index < split_date], y[y.index >= split_date]

    print(f"  Train: {len(X_train)} bars | Val: {len(X_val)} bars")
    if len(X_train) == 0:
        raise RuntimeError(f"No training data before {start_date_str}")

    w_train = X_train['sample_weight'].values.astype(np.float32)
    w_val = X_val['sample_weight'].values.astype(np.float32) if len(X_val) > 0 else np.array([])
    l_val = X_val['is_live'].values.astype(bool) if len(X_val) > 0 else np.array([])

    X_train = X_train.drop(columns=['sample_weight', 'is_live'])
    X_val = X_val.drop(columns=['sample_weight', 'is_live'])

    # Scale features"""
content = content.replace(old_split, new_split)

# Replace the dataset creation to include weights
old_ds = """    X_t = torch.FloatTensor(X_train_s.astype(np.float32))
    y_tp_t = torch.FloatTensor(y_train_tp).unsqueeze(1)
    y_dir_t = torch.LongTensor(y_train_dir)
    y_risk_t = torch.FloatTensor(y_train_risk).unsqueeze(1)

    train_ds = TensorDataset(X_t, y_tp_t, y_dir_t, y_risk_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)"""

new_ds = """    X_t = torch.FloatTensor(X_train_s.astype(np.float32))
    y_tp_t = torch.FloatTensor(y_train_tp).unsqueeze(1)
    y_dir_t = torch.LongTensor(y_train_dir)
    y_risk_t = torch.FloatTensor(y_train_risk).unsqueeze(1)
    w_t = torch.FloatTensor(w_train).unsqueeze(1)

    train_ds = TensorDataset(X_t, y_tp_t, y_dir_t, y_risk_t, w_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)"""
content = content.replace(old_ds, new_ds)

# Replace loss functions
old_loss = """    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()"""

new_loss = """    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')"""
content = content.replace(old_loss, new_loss)

# Replace training loop
old_train_loop = """        for bx, btp, bdir, brisk in train_loader:
            optimizer.zero_grad()
            out_tp, out_dir, out_risk = model(bx)
            loss = bce(out_tp, btp) + ce(out_dir, bdir) + mse(out_risk, brisk)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        val_str = ""
        if has_val:
            model.eval()
            with torch.no_grad():
                vtp, vdir, vrisk = model(X_v)
                val_loss = bce(vtp, y_tp_v) + ce(vdir, y_dir_v) + mse(vrisk, y_risk_v)
            val_str = f" | Val Loss: {val_loss.item():.4f}"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train:.4f}{val_str}")"""

new_train_loop = """        for bx, btp, bdir, brisk, bw in train_loader:
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
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train:.4f}{val_str}")"""
content = content.replace(old_train_loop, new_train_loop)

# Replace save model
old_save = """    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=mean, scale=scale)
    print(f"Model saved to {model_path}")"""

new_save = """    torch.save(model.state_dict(), model_path)
    # Save enriched model too
    enriched_model_path = os.path.join(out_dir, f"{model_name}_enriched.pt")
    torch.save(model.state_dict(), enriched_model_path)
    np.savez(scaler_path, mean=mean, scale=scale)
    print(f"Model saved to {model_path} and {enriched_model_path}")"""
content = content.replace(old_save, new_save)

with open('/home/tokiarivelo/Downloads/Documents/trafing-bot/backend/scripts/train_smc_dl.py', 'w') as f:
    f.write(content)
