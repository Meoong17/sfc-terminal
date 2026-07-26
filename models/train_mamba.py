#!/usr/bin/env python3
"""
train_mamba.py — Train Mamba Encoder on historical SFC data
=============================================================
Extracts ~1838 data.json snapshots from git history,
builds sliding-window sequences (seq_len=30 → target),
trains Mamba model, saves best weights.

Usage:
    cd /home/ubuntu/sfc
    python3 train_mamba.py
"""

import json, os, sys, time, subprocess, math
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mamba_encoder import build_feature_vector, MambaEncoder

# ── CONFIG ──
SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN = 30          # Number of time steps per sample
INPUT_DIM = 39        # Auto-detected from feature vector
BATCH_SIZE = 32
EPOCHS = 100          # More epochs with early stopping
LEARNING_RATE = 0.002  # Slightly higher initial LR
VAL_SPLIT = 0.15       # 15% for validation
TEST_SPLIT = 0.10      # 10% for testing
MODEL_DIR = os.path.join(SFC_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "mamba_weights.pth")
ENSEMBLE_DIR = os.path.join(MODEL_DIR, "ensemble")
os.makedirs(ENSEMBLE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SFC_DIR, "mamba_train.log")
N_ENSEMBLE = 3         # Number of ensemble models
AUGMENT_NOISE = 0.01   # Gaussian noise std for data augmentation
HORIZONS = [1, 3, 6]   # Multi-horizon prediction targets

device = torch.device('cpu')
print(f"[Train] Using device: {device}")


# ================================================================
# 1. EXTRACT HISTORICAL DATA FROM GIT
# ================================================================
def extract_historical_snapshots():
    """Extract all clean data.json snapshots from git history."""
    print("[Train] Extracting historical data from git...")
    
    # Get all commits that modified data.json (oldest first)
    result = subprocess.check_output(
        ['git', 'log', '--oneline', '--all', '--diff-filter=M', '--reverse', '--', 'data.json'],
        text=True, cwd=SFC_DIR
    ).strip().split('\n')
    
    snapshots = []
    errors = 0
    for i, line in enumerate(result):
        sha = line.split()[0]
        try:
            content = subprocess.check_output(
                ['git', 'show', f'{sha}:data.json'], text=True, timeout=5, cwd=SFC_DIR
            )
            # Skip corrupted (non-JSON) entries
            if not content.startswith('{'):
                errors += 1
                continue
            data = json.loads(content)
            snapshots.append(data)
        except (json.JSONDecodeError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            errors += 1
            continue
        
        if (i + 1) % 300 == 0:
            print(f"[Train]  Extracted {i+1}/{len(result)}...")
    
    print(f"[Train] Extracted {len(snapshots)} clean snapshots ({errors} skipped)")
    return snapshots


# ================================================================
# 1b. LOAD HISTORICAL BTC FEATURES (from fetch_historical_btc.py)
# ================================================================
def load_historical_btc_data():
    """
    Load pre-computed historical BTC feature vectors and compute
    proxy stress targets from price action.
    
    Historical data covers 2021-01-01 to present (~2000 daily snapshots).
    Since these don't have SFC-derived stress targets, we compute a
    proxy from RSI + daily returns + volatility.
    
    Returns:
        features: (n, n_features) numpy array
        targets: (n,) numpy array — proxy stress 0-1
        dates: list of ISO date strings
    """
    feat_path = os.path.join(SFC_DIR, ".historical_features.npy")
    dates_path = os.path.join(SFC_DIR, ".historical_dates.npy")
    
    if not os.path.exists(feat_path) or not os.path.exists(dates_path):
        print("[Train] ⚠ Historical BTC data not found. Run fetch_historical_btc.py first.")
        return None, None, None
    
    print("[Train] Loading historical BTC data...")
    features = np.load(feat_path)
    dates = np.load(dates_path, allow_pickle=True)
    
    print(f"[Train] Historical features: {features.shape}, dates: {len(dates)}")
    print(f"[Train] Date range: {dates[0]} to {dates[-1]}")
    
    # Compute proxy stress targets from price features
    n = len(features)
    targets = np.zeros(n, dtype=np.float32)
    
    # Derive proxy stress from available features:
    # feature[0] = btc_price/100k, feature[1] = btc_24h_change/20, 
    # feature[5] = RSI/100, feature[4] = DVOL proxy
    
    for i in range(n):
        stress = 0.0
        
        # RSI-based stress: RSI < 30 = stress, RSI > 70 = low stress
        rsi = features[i][5] * 100.0  # de-normalize
        if rsi > 0:
            if rsi < 30:
                stress += 0.4 * (1.0 - rsi / 30.0)  # 0-0.4 stress from RSI
            elif rsi > 70:
                stress += 0.0  # Low stress in overbought
            else:
                stress += 0.2 * (1.0 - abs(rsi - 50) / 20.0)  # moderate
        
        # Daily return stress: large negative = stress
        daily_return = features[i][1] * 20.0  # de-normalize (%)
        if daily_return < -5:
            stress += 0.3  # Sharp drop
        elif daily_return < -3:
            stress += 0.2
        elif daily_return < -1:
            stress += 0.1
        
        # Volatility stress: high vol = moderate stress
        dvol = features[i][4]  # already 0-1 normalized
        if dvol > 0.6:
            stress += 0.2 * (dvol - 0.6) / 0.4
        
        # FNG-based stress: extreme fear = stress
        fng = features[i][8] * 100.0
        if fng > 0 and fng < 20:
            stress += 0.1 * (20.0 - fng) / 20.0
        
        targets[i] = min(stress, 1.0)
    
    print(f"[Train] Proxy stress targets: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
    
    return features, targets, dates


def merge_datasets(git_features, git_targets, hist_features, hist_targets):
    """
    Merge git-snapshot data with historical BTC data chronologically.
    Historical data (2021 onwards) comes BEFORE git snapshot data (June 2026).
    """
    if hist_features is None:
        print("[Train] No historical data — using only git snapshots")
        return git_features, git_targets
    
    # Combine: historical data first (older), then git snapshots (newer)
    combined_features = np.vstack([hist_features, git_features])
    combined_targets = np.concatenate([hist_targets, git_targets])
    
    print(f"\n[Train] Dataset merge:")
    print(f"  Git snapshots:  {len(git_features)}")
    print(f"  Historical BTC: {len(hist_features)}")
    print(f"  Combined:       {len(combined_features)}")
    print(f"  Feature dim:    {combined_features.shape[1]}")
    
    return combined_features, combined_targets


# ================================================================
# 2. BUILD FEATURES AND TARGETS
# ================================================================
def build_dataset(snapshots):
    """
    Build feature array and target array from snapshots.

    Returns:
        features: (n, n_features) numpy array
        targets: (n,) numpy array — realized future BTC price-outcome
            proxy (0-1), NOT sfc_effective. See note below for why.

    WHY THIS CHANGED FROM "target = sfc_effective / 100":
    sfc_effective is itself one of the 39 features in build_feature_vector()
    (see mamba_encoder.py), and is highly autocorrelated at the snapshot
    intervals this pipeline runs at (every few minutes) — empirically, a
    naive "predict(t+1) = value(t)" persistence baseline reaches R²≈0.94 at
    horizon=1 on simulated data with this kind of autocorrelation, which is
    higher than what most genuine forecasting models achieve. Training
    Mamba to predict sfc_effective(t+h) when sfc_effective(t) is already in
    its own input sequence rewards the model for learning to copy a recent
    input value forward, not for learning genuine macro-liquidity dynamics.

    The realized BTC price target below is independent of every feature in
    build_feature_vector() (model scores, sfc_effective/sfc_base are all
    excluded from determining this), so the model can no longer get credit
    just for reproducing something already visible in its own input window.

    Each snapshot's target is the price-outcome PRICE_LOOKAHEAD_MINUTES
    ahead of it (own price_outcome window, NOT the create_sequences()
    multi-step horizon below — these compose: create_sequences() then
    additionally looks further ahead in snapshot-index space for its
    [1,3,6]-step horizons, so the effective lookahead for horizon h
    becomes PRICE_LOOKAHEAD_MINUTES + h*avg_snapshot_interval).
    """
    PRICE_LOOKAHEAD_MINUTES = 360
    PRICE_LOOKAHEAD_TOLERANCE_MINUTES = 60
    PRICE_STRESS_DROP_PCT = -3.0
    PRICE_CALM_FLOOR_PCT = 1.0

    def _parse_ts(snap):
        ts_str = snap.get('ts')
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    parsed_times = [_parse_ts(s) for s in snapshots]
    btc_prices = [s.get('btc') for s in snapshots]

    features = []
    targets = []
    skipped_no_future = 0
    skipped_no_price = 0

    for i, snap in enumerate(snapshots):
        obs_time = parsed_times[i]
        obs_price = btc_prices[i]
        if obs_time is None or obs_price is None:
            skipped_no_price += 1
            continue

        target_time = obs_time + timedelta(minutes=PRICE_LOOKAHEAD_MINUTES)
        best_diff = None
        future_price = None
        for j in range(i + 1, len(snapshots)):
            t_j = parsed_times[j]
            if t_j is None:
                continue
            diff_minutes = abs((t_j - target_time).total_seconds()) / 60.0
            if diff_minutes <= PRICE_LOOKAHEAD_TOLERANCE_MINUTES:
                if best_diff is None or diff_minutes < best_diff:
                    best_diff = diff_minutes
                    future_price = btc_prices[j]
            if (t_j - target_time).total_seconds() / 60.0 > PRICE_LOOKAHEAD_TOLERANCE_MINUTES:
                break

        if future_price is None:
            skipped_no_future += 1
            continue

        pct_change = (future_price - obs_price) / obs_price * 100.0
        if pct_change <= PRICE_STRESS_DROP_PCT:
            target_01 = 1.0
        elif pct_change >= -PRICE_CALM_FLOOR_PCT:
            target_01 = 0.0
        else:
            span = PRICE_STRESS_DROP_PCT - (-PRICE_CALM_FLOOR_PCT)
            target_01 = (pct_change - (-PRICE_CALM_FLOOR_PCT)) / span
            target_01 = max(0.0, min(1.0, target_01))

        vec = build_feature_vector(snap)
        features.append(vec)
        targets.append(target_01)

        if (i + 1) % 500 == 0:
            print(f"[Train]  Processed {i+1}/{len(snapshots)} snapshots...")

    print(f"[Train] Skipped {skipped_no_price} snapshot(s) missing ts/btc, "
          f"{skipped_no_future} with no future reference point in range")

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print(f"[Train] Feature array: {features.shape}")
    if len(targets) > 0:
        print(f"[Train] Target range: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")
    else:
        print("[Train] WARNING: no usable (feature, target) pairs were built — "
              "check that snapshots have parseable 'ts' and numeric 'btc' fields.")

    return features, targets


def create_sequences(features, targets, seq_len=30, horizons=None):
    """
    Create sliding-window sequences with multi-horizon targets.
    Each sample: sequence of seq_len feature vectors → target at multiple future steps.
    
    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples, n_horizons) — targets at 1, 3, 6 steps ahead
    """
    if horizons is None:
        horizons = [1, 3, 6]
    
    n = len(features)
    max_horizon = max(horizons)
    X, y_list = [], []
    
    for i in range(n - seq_len - max_horizon):
        X.append(features[i:i + seq_len])
        # Multi-horizon targets
        y_row = []
        for h in horizons:
            target_idx = i + seq_len + h - 1  # -1 because h=1 means next step
            if target_idx < n:
                y_row.append(targets[target_idx])
            else:
                y_row.append(targets[-1])  # pad with last value
        y_list.append(y_row)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"[Train] Sequences: X={X.shape}, y={y.shape} (horizons={horizons})")
    return X, y


# ================================================================
# 3. PYTORCH DATASET
# ================================================================
class SFCDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.01):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # (n, n_horizons)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.noise_std > 0:
            # Add gaussian noise for regularization
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x, self.y[idx]


# ================================================================
# 4. TRAINING LOOP
# ================================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    """Train Mamba model with multi-horizon loss and progressive scheduler."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    # Horizon weights: predict nearer future with higher weight
    horizon_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 15
    patience_counter = 0
    
    print(f"\n[Train] Starting training for {epochs} epochs...")
    print(f"[Train] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"[Train] Horizon weights: {horizon_weights.tolist()}")
    
    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            
            # Multi-horizon loss: weighted sum across horizons
            pred = output['combined']  # (batch, 1)
            # For multi-horizon, we need to expand pred to match batch_y shape
            loss = criterion(pred.expand(-1, batch_y.shape[1]), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_X)
        
        train_loss /= len(train_loader.dataset)
        
        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        horizon_errors = [0.0] * batch_y.shape[1] if batch_y.shape[1] > 1 else [0.0]
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                pred = output['combined']
                loss = criterion(pred.expand(-1, batch_y.shape[1]), batch_y)
                val_loss += loss.item() * len(batch_X)
                
                # Per-horizon MAE
                for h in range(batch_y.shape[1]):
                    horizon_errors[h] += torch.abs(pred.squeeze() - batch_y[:, h]).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_mae = sum(horizon_errors) / (len(val_loader.dataset) * len(horizon_errors))
        
        # LR scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            horizon_str = ' | '.join([f'H{h}={e/len(val_loader.dataset)*100:.2f}%' 
                                       for h, e in zip([1,3,6], horizon_errors)])
            print(f"[Train] Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                  f"{horizon_str} | lr={current_lr:.2e} | {elapsed:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f"[Train]  → New best model! val_loss={val_loss:.6f} (epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    print(f"\n[Train] Training complete!")
    print(f"[Train] Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.6f}")
    return best_val_loss


# ================================================================
# 5. EVALUATION
# ================================================================
def evaluate_model(model, test_loader, label="Test"):
    """Evaluate model on a dataset (supports multi-horizon targets)."""
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    horizon_errors_list = []
    
    criterion = nn.MSELoss()
    n_horizons = None
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            pred = output['combined']  # (batch, 1)
            
            if n_horizons is None:
                n_horizons = batch_y.shape[1]
            
            # Loss on first horizon (same-step prediction)
            loss = criterion(pred, batch_y[:, 0:1])
            total_loss += loss.item() * len(batch_X)
            
            predictions.extend(pred.squeeze().cpu().numpy())
            actuals.extend(batch_y[:, 0].cpu().numpy())  # first horizon
            
            # Per-horizon errors
            for h in range(n_horizons):
                h_err = torch.abs(pred.squeeze() - batch_y[:, h])
                if len(horizon_errors_list) <= h:
                    horizon_errors_list.append([])
                horizon_errors_list[h].extend(h_err.cpu().numpy())
    
    mse = total_loss / len(test_loader.dataset)
    rmse = math.sqrt(mse)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # R² score
    ss_res = ((actuals - predictions) ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Per-horizon MAE
    horizon_summary = ""
    if n_horizons and n_horizons > 1:
        horizon_parts = []
        for h in range(n_horizons):
            h_mae = np.mean(horizon_errors_list[h])
            horizon_parts.append(f"H{h+1}={h_mae*100:.2f}%")
        horizon_summary = " | " + " ".join(horizon_parts)
    
    print(f"\n[Train] {label} Evaluation:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f} ({rmse*100:.2f}%)")
    print(f"  R²:   {r2:.4f}{horizon_summary}")
    print(f"  Pred range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Actual range: [{actuals.min():.4f}, {actuals.max():.4f}]")
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'preds': predictions, 'actuals': actuals}


# ================================================================
# 6. MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MAMBA TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Extract git snapshots
    snapshots = extract_historical_snapshots()
    if len(snapshots) < SEQ_LEN + 10:
        print(f"[Train] ERROR: Need at least {SEQ_LEN + 10} snapshots, got {len(snapshots)}")
        sys.exit(1)
    
    # 2. Build features from git snapshots
    git_features, git_targets = build_dataset(snapshots)
    
    # 2b. Load historical BTC data and merge
    hist_features, hist_targets, hist_dates = load_historical_btc_data()
    features, targets = merge_datasets(git_features, git_targets, hist_features, hist_targets)
    
    # 3. Create sequences
    X, y = create_sequences(features, targets, SEQ_LEN, HORIZONS)
    
    # 4. Train/val/test split (chronological — test is newest)
    n_total = len(X)
    n_test = int(n_total * TEST_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_test - n_val
    
    # Chronological split: oldest → train, middle → val, newest → test
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[-n_test:], y[-n_test:]
    
    print(f"\n[Train] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"[Train] Target distribution:")
    print(f"  Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    print(f"  Val:   mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    print(f"  Test:  mean={y_test.mean():.4f}, std={y_test.std():.4f}")
    
    # 5. Create datasets (with augmentation for training)
    train_dataset = SFCDataset(X_train, y_train, augment=True, noise_std=AUGMENT_NOISE)
    val_dataset = SFCDataset(X_val, y_val, augment=False)
    test_dataset = SFCDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Create model (fresh, not loaded)
    actual_dim = X.shape[2]  # Auto-detect from data
    print(f"\n[Train] Creating MambaEncoder(input_dim={actual_dim})")
    model = MambaEncoder(
        input_dim=actual_dim,
        d_model=128,
        d_state=16,
        d_conv=4,
        n_layers=2
    )
    print(f"[Train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Evaluate BEFORE training (random weights baseline)
    print(f"\n{'='*50}")
    print("BEFORE TRAINING — Random Weights Baseline")
    print(f"{'='*50}")
    baseline = evaluate_model(model, test_loader, "Test (baseline)")
    
    # 8. Train
    best_val_loss = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # 9. Load best model and evaluate
    print(f"\n{'='*50}")
    print("AFTER TRAINING — Best Model")
    print(f"{'='*50}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    model.eval()
    results = evaluate_model(model, test_loader, "Test (trained)")
    
    # 10. Improvement
    print(f"\n{'='*50}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*50}")
    print(f"  Baseline RMSE: {baseline['rmse']*100:.2f}%")
    print(f"  Trained  RMSE: {results['rmse']*100:.2f}%")
    print(f"  RMSE Improvement: {(baseline['rmse'] - results['rmse']) * 100:+.2f}%")
    print(f"  Baseline R²: {baseline['r2']:.4f}")
    print(f"  Trained  R²: {results['r2']:.4f}")
    
    if results['r2'] > baseline['r2']:
        print(f"\n✅ MODEL TELAH DILATIH DAN MENINGKAT!")
    else:
        print(f"\n⚠️  Model belum cukup improve — mungkin perlu lebih banyak data atau lebih banyak epoch.")
    
    print(f"\n[Train] Model saved to: {MODEL_PATH}")
    print("[Train] Training complete!")
