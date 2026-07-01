#!/usr/bin/env python3
"""
QLSTM v3 — Hybrid Quantum LSTM for SFC Volatility
==================================================
Quantum circuit sebagai context vector (1x per forward pass).
Classical LSTM untuk temporal processing.

Arsitektur:
  Input (seq_len, features)
    +-- Classical LSTM -> temporal features
    +-- Quantum circuit (mean pooling) -> quantum context
         +-- Concatenate -> FC -> SFC prediction (0-100)

TARGET FIX (see prepare_data() below for full explanation):
  Training target changed from a circular linear-combination-of-inputs
  formula to real BTC price-outcome labels (from ml_ensemble.py's
  resolve_pending_labels()). The quantum circuit + LSTM architecture
  itself is unchanged — this was a labeling bug, not an architecture
  problem. If you have an older qlstm_model.pt trained on the previous
  circular target, delete it and retrain; the old weights are not
  meaningfully comparable to what this version produces.
"""

import json, os, sys, math, warnings, time
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pennylane as qml
from pennylane.qnn import TorchLayer

# ──────────────────────────────────────────────
# Quantum Context Encoder
# ──────────────────────────────────────────────
# Single quantum circuit that encodes the mean of input features
# into a quantum state, then measures to produce a context vector.

N_QUBITS = 4
N_QLAYERS = 2

def build_quantum_encoder():
    """Build a quantum encoder that takes N_QUBITS features → N_QUBITS measurements."""
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    
    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    
    weight_shapes = {"weights": (N_QLAYERS, N_QUBITS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes, init_method=lambda t: torch.randn_like(t) * 0.1)
    
    return qlayer


class QLSTMVolatilityPredictor(nn.Module):
    """
    Hybrid Quantum LSTM.
    Quantum encoder (1× per forward pass) + Classical LSTM.
    """
    
    def __init__(self, input_dim, hidden_dim=16, seq_len=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Quantum context encoder (processes mean of input sequence)
        self.quantum = build_quantum_encoder()
        self.quantum_proj = nn.Linear(N_QUBITS, hidden_dim // 2)
        
        # Classical LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Output head: quantum context + LSTM output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: (batch,) SFC predictions (0-100)
        """
        batch = x.shape[0]
        
        # ── Classical LSTM ──
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # ── Quantum context (1 call per forward pass) ──
        # Pool features across time dimension, then across batch if needed
        x_pooled = x.mean(dim=1)  # (batch, input_dim)
        # Only use first N_QUBITS features (or pad)
        q_input = x_pooled[:, :N_QUBITS]
        if q_input.shape[1] < N_QUBITS:
            pad = torch.zeros(batch, N_QUBITS - q_input.shape[1], device=x.device)
            q_input = torch.cat([q_input, pad], dim=1)
        
        # Process first sample through quantum (supports batch=1 efficiently)
        # For larger batches, process iteratively
        q_outs = []
        for i in range(min(batch, 4)):  # quantum context from first 4 samples
            q_out = self.quantum(q_input[i])
            q_outs.append(q_out)
        # Average quantum contexts
        q_context = torch.stack(q_outs).mean(dim=0)  # (N_QUBITS,)
        
        # Project quantum context
        q_feat = self.quantum_proj(q_context.unsqueeze(0))  # (1, hidden_dim//2)
        q_feat = q_feat.expand(batch, -1)  # (batch, hidden_dim//2)
        
        # ── Combine and predict ──
        combined = torch.cat([lstm_features, q_feat], dim=1)
        sfc_pred = self.fc(combined).squeeze(-1)
        sfc_pred = torch.sigmoid(sfc_pred) * 100.0
        
        return sfc_pred


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SFCSequenceDataset(Dataset):
    def __init__(self, features, targets, seq_len=8):
        self.seq_len = seq_len
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sequences = []
        self.labels = []
        for i in range(seq_len, len(features)):
            self.sequences.append(self.features[i-seq_len:i])
            self.labels.append(self.targets[i])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_data(data_path, seq_len=8):
    """
    Build (features, targets) training pairs from data_collection.json.

    IMPORTANT — target source changed from a circular formula to real
    price-outcome labels:

    Previously, sfc_targets was computed as:
        sfc_targets = (features[:, :6] @ weights[:6]) * 100.0
    i.e. a fixed linear combination of M1-M6 — which are ALSO part of the
    input feature sequence the LSTM/quantum encoder consumes. The model
    was therefore being trained to reproduce a formula built from its own
    inputs, not to predict anything about future market behavior. This is
    the same circularity pattern found and fixed in train_mamba.py and
    ensemble_meta.py (see those files' comments for the full explanation
    and an empirical demonstration: a naive linear regression on the old
    target reached R²=1.000 exactly, because the target IS a linear
    function of 6 of the input features).

    This version instead reads "labels" from data_collection.json, which
    ml_ensemble.py's resolve_pending_labels() now populates from REALIZED
    BTC PRICE OUTCOME (price 360 minutes after the observation, relative
    to price at observation time) — completely independent of any method
    score. Observations whose label is still None ("pending" — not enough
    time has passed yet for the outcome to be known) are excluded from
    training, not filled with a guessed value.

    Returns:
        (input_dim, train_loader, val_loader) — same signature as before,
        or (None, None, None) if there isn't enough labeled data yet.
    """
    with open(data_path) as f:
        data = json.load(f)

    features_raw = data.get("features", [])
    labels_raw = data.get("labels", [])

    if len(features_raw) != len(labels_raw):
        print(f"[QLSTM] WARNING: features ({len(features_raw)}) and labels "
              f"({len(labels_raw)}) length mismatch — data_collection.json "
              f"may be from an older format. Re-run collect.py a few cycles "
              f"to rebuild it with the current ml_ensemble.py.")
        return None, None, None

    # Keep only observations with a RESOLVED label (not None/pending).
    # Labels are 0.0 (confirmed calm) or 1.0 (confirmed stress) — see
    # ml_ensemble.py's resolve_pending_labels() for the exact thresholds.
    labeled_pairs = [
        (f, l) for f, l in zip(features_raw, labels_raw) if l is not None
    ]

    if len(labeled_pairs) < seq_len + 20:
        print(f"[QLSTM] Only {len(labeled_pairs)} labeled observations available "
              f"(need >= {seq_len + 20}). Labels resolve ~6 hours after each "
              f"observation (see LABEL_LOOKAHEAD_MINUTES in ml_ensemble.py) — "
              f"let collect.py run longer before training.")
        return None, None, None

    features_list = [f for f, _ in labeled_pairs]
    targets_list = [l * 100.0 for _, l in labeled_pairs]  # 0/1 -> 0-100 SFC scale, matches model output range

    max_len = max(len(f) for f in features_list)
    features_padded = []
    for f in features_list:
        f_arr = np.array(f, dtype=np.float32)
        if len(f_arr) < max_len:
            f_arr = np.concatenate([f_arr, np.full(max_len - len(f_arr), 0.5, dtype=np.float32)])
        features_padded.append(f_arr)

    features = np.array(features_padded)
    sfc_targets = np.array(targets_list, dtype=np.float32)

    print(f"[QLSTM] Training on {len(features)} labeled observations "
          f"(price-outcome based, not circular formula)")
    print(f"[QLSTM] Target distribution: {(sfc_targets > 50).sum()} stress, "
          f"{(sfc_targets <= 50).sum()} calm "
          f"({100*(sfc_targets>50).mean():.1f}% stress rate)")

    # Normalize
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    features_norm = (features - mean) / std

    dataset = SFCSequenceDataset(features_norm, sfc_targets, seq_len)
    n = len(dataset)
    if n < 10:
        print(f"[QLSTM] Only {n} sequences after windowing (seq_len={seq_len}) — need more data.")
        return None, None, None

    n_train = int(n * 0.8)

    # Chronological split (not random_split) — this is a time series;
    # randomly shuffling train/val would let the model validate on
    # observations that occurred BEFORE some of its training data,
    # leaking future information into what should be an out-of-sample
    # check. random_split with a fixed seed was the previous approach;
    # changed here for the same reason train_mamba.py's original
    # chronological split was already correct and preserved during its
    # own circularity fix — only the target changed, this consistency
    # issue was specific to this file.
    train_ds = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_ds = torch.utils.data.Subset(dataset, list(range(n_train, n)))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    return features.shape[1], train_loader, val_loader


def train(model, train_loader, val_loader, epochs=60, lr=0.003, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    best_state = None
    patience = 12
    pc = 0
    history = {"train": [], "val": []}
    
    total = sum(p.numel() for p in model.parameters())
    qp = sum(p.numel() for n, p in model.named_parameters() if "quantum" in n)
    
    print(f"\n[QLSTM] Params: {total:,} total ({qp:,} quantum)")
    print(f"[QLSTM] Training {epochs} epochs (lightning.qubit)\n")
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * bx.shape[0]
        tl /= len(train_loader.dataset)
        
        model.eval()
        vl = 0.0
        preds, targs = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                p = model(bx)
                vl += criterion(p, by).item() * bx.shape[0]
                preds.extend(p.cpu().numpy())
                targs.extend(by.cpu().numpy())
        vl /= len(val_loader.dataset)
        scheduler.step()
        
        history["train"].append(tl)
        history["val"].append(vl)
        
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
        
        if (ep + 1) % 10 == 0 or ep == 0:
            mae = float(np.mean(np.abs(np.array(preds) - np.array(targs))))
            print(f"  Ep {ep+1:3d}/{epochs} | Train: {tl:.4f} | Val: {vl:.4f} | MAE: {mae:.2f}")
        
        if pc >= patience:
            print(f"  [Early stop at ep {ep+1}]")
            break
    
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    
    # Final eval
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            p = model(bx.to(device))
            preds.extend(p.cpu().numpy())
            targs.extend(by.cpu().numpy())
    preds, targs = np.array(preds), np.array(targs)
    mae = float(np.mean(np.abs(preds - targs)))
    rmse = float(np.sqrt(np.mean((preds - targs)**2)))
    
    print(f"\n[QLSTM] ✅ Done in {elapsed:.0f}s")
    print(f"  Best Val Loss: {best_val:.4f}")
    print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} SFC pts")
    print(f"  Target range: [{targs.min():.1f}, {targs.max():.1f}]")
    
    return history, best_val


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  QLSTM v3 — Hybrid Quantum LSTM")
    print("  Quantum context + Classical LSTM")
    print(f"  Qubits: {N_QUBITS} | Layers: {N_QLAYERS} | Device: lightning.qubit")
    print("=" * 65)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), "sfc", "data_collection.json")
    
    if not os.path.exists(data_path):
        print(f"[QLSTM] Data not found")
        sys.exit(1)
    
    input_dim, train_loader, val_loader = prepare_data(data_path, seq_len=8)
    if input_dim is None:
        sys.exit(1)
    
    print(f"\n[QLSTM] Input: {input_dim} features | Seq: 8 days")
    print(f"[QLSTM] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    
    model = QLSTMVolatilityPredictor(input_dim, hidden_dim=16, seq_len=8)
    
    history, best_loss = train(model, train_loader, val_loader, epochs=60, lr=0.003)
    
    # Analysis
    print(f"\n{'─' * 55}")
    print("  QUANTUM CIRCUIT ANALYSIS")
    print(f"{'─' * 55}")
    print(f"  Device:    lightning.qubit (C++ backend, ~10× faster)")
    print(f"  Qubits:    {N_QUBITS}")
    print(f"  Layers:    {N_QLAYERS} StronglyEntangling")
    print(f"  Encoding:  AngleEmbedding (X rotations)")
    print(f"  Readout:   PauliZ expectation (4 values → context vector)")
    print(f"  Calls/fwd: 1 (mean pooling across time)")
    
    print(f"\n{'─' * 55}")
    print("  SFC INTEGRATION OPTIONS")
    print(f"{'─' * 55}")
    print(f"  1. M32 — Sinyal terpisah: QLSTM_pred sebagai metode ke-7")
    print(f"  2. Dynamic weighting: QLSTM menentukan bobot M1-M6 harian")
    print(f"  3. Ensemble boost: (Ensemble + QLSTM) / 2")
    print(f"\n  Retrain: bisa dijadwalkan tiap minggu via cron job")
    
    # Save model
    model_path = os.path.join(script_dir, "qlstm_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": 16,
        "seq_len": 8,
        "val_loss": best_loss,
    }, model_path)
    print(f"\n  Model: {model_path}")
    
    with open(os.path.join(script_dir, "qlstm_history.json"), "w") as f:
        json.dump({
            "train_loss": [float(v) for v in history["train"]],
            "val_loss": [float(v) for v in history["val"]],
        }, f, indent=2)
    
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
