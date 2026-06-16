#!/usr/bin/env python3
"""
train_mamba.py — Train Mamba Encoder on historical SFC data
=============================================================
Extracts ~1838 data.json snapshots from git history,
builds sliding-window sequences (seq_len=30 → target),
trains Mamba model, saves best weights.

Usage:
    cd /home/ubuntu/sfc
    PYTHONPATH="/home/ubuntu/sfc2/venv/lib/python3.12/site-packages:$PYTHONPATH" python3 train_mamba.py
"""

import json, os, sys, time, subprocess, math
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
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 30          # Number of time steps per sample
INPUT_DIM = 39        # Auto-detected from feature vector
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_SPLIT = 0.15       # 15% for validation
TEST_SPLIT = 0.10      # 10% for testing
MODEL_DIR = os.path.join(SFC_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "mamba_weights.pth")
LOG_FILE = os.path.join(SFC_DIR, "mamba_train.log")

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
# 2. BUILD FEATURES AND TARGETS
# ================================================================
def build_dataset(snapshots):
    """
    Build feature array and target array from snapshots.
    
    Returns:
        features: (n, n_features) numpy array
        targets: (n,) numpy array — sfc_effective / 100 (0-1)
    """
    features = []
    targets = []
    skipped = 0
    
    for i, snap in enumerate(snapshots):
        # Build feature vector
        vec = build_feature_vector(snap)
        
        # Target: sfc_effective / 100 (normalized 0-1)
        target_val = snap.get('sfc_effective')
        if target_val is None:
            # Fallback: use sfc_base or ensemble
            target_val = snap.get('sfc_base', 50.0)
        
        target_01 = float(target_val) / 100.0
        target_01 = max(0.0, min(1.0, target_01))  # Clamp
        
        features.append(vec)
        targets.append(target_01)
        
        if (i + 1) % 500 == 0:
            print(f"[Train]  Processed {i+1}/{len(snapshots)} snapshots...")
    
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"[Train] Feature array: {features.shape}")
    print(f"[Train] Target range: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")
    
    return features, targets


def create_sequences(features, targets, seq_len=30):
    """
    Create sliding-window sequences.
    Each sample: sequence of seq_len feature vectors → next target value.
    
    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    n = len(features)
    X, y = [], []
    
    for i in range(n - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])  # predict NEXT step
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"[Train] Sequences: X={X.shape}, y={y.shape}")
    return X, y


# ================================================================
# 3. PYTORCH DATASET
# ================================================================
class SFCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (n, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================================================================
# 4. TRAINING LOOP
# ================================================================
def train_model(model, train_loader, val_loader, epochs, lr):
    """Train Mamba model with Adam optimizer and MSELoss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 10
    patience_counter = 0
    
    print(f"\n[Train] Starting training for {epochs} epochs...")
    print(f"[Train] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            pred = output['combined']
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_X)
        
        train_loss /= len(train_loader.dataset)
        
        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                pred = output['combined']
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * len(batch_X)
                val_mae += torch.abs(pred - batch_y).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        # LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        elapsed = time.time() - start_time
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Train] Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                  f"val_mae={val_mae:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s")
        
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
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    predictions = []
    actuals = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            pred = output['combined']
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * len(batch_X)
            total_mae += torch.abs(pred - batch_y).sum().item()
            predictions.extend(pred.squeeze().cpu().numpy())
            actuals.extend(batch_y.squeeze().cpu().numpy())
    
    mse = total_loss / len(test_loader.dataset)
    mae = total_mae / len(test_loader.dataset)
    rmse = math.sqrt(mse)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # R² score
    ss_res = ((actuals - predictions) ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"\n[Train] {label} Evaluation:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f} ({rmse*100:.2f}%)")
    print(f"  MAE:  {mae:.4f} ({mae*100:.2f}%)")
    print(f"  R²:   {r2:.4f}")
    print(f"  Pred range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Actual range: [{actuals.min():.4f}, {actuals.max():.4f}]")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': predictions, 'actuals': actuals}


# ================================================================
# 6. MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MAMBA TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Extract snapshots
    snapshots = extract_historical_snapshots()
    if len(snapshots) < SEQ_LEN + 10:
        print(f"[Train] ERROR: Need at least {SEQ_LEN + 10} snapshots, got {len(snapshots)}")
        sys.exit(1)
    
    # 2. Build features
    features, targets = build_dataset(snapshots)
    
    # 3. Create sequences
    X, y = create_sequences(features, targets, SEQ_LEN)
    
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
    
    # 5. Create datasets
    train_dataset = SFCDataset(X_train, y_train)
    val_dataset = SFCDataset(X_val, y_val)
    test_dataset = SFCDataset(X_test, y_test)
    
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
