#!/bin/bash
# ============================================================
# sfc-weekly-train.sh — Heavy model training (Mamba + QLSTM)
# Runs weekly on Sunday 3AM via cron. Uses .venv (torch).
# ============================================================
cd /home/ubuntu/sfc || exit 1
LOG="/home/ubuntu/sfc/logs/train_weekly_$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG")"

echo "╔═══════════════════════════════════════════════" | tee -a "$LOG"
echo "║ SFC WEEKLY HEAVY TRAIN — $(date)" | tee -a "$LOG"
echo "╚═══════════════════════════════════════════════" | tee -a "$LOG"

# ── 1. Mamba Encoder (needs .venv + einops) ──
echo "" | tee -a "$LOG"
echo "[1/2] Mamba model training..." | tee -a "$LOG"
/home/ubuntu/sfc/.venv/bin/python /home/ubuntu/sfc/models/train_mamba.py 2>&1 | tee -a "$LOG"
MAMBA_EXIT=${PIPESTATUS[0]}
if [ $MAMBA_EXIT -eq 0 ]; then
    echo "[MAMBA] ✅ Training complete" | tee -a "$LOG"
else
    echo "[MAMBA] ❌ Training failed (exit=$MAMBA_EXIT)" | tee -a "$LOG"
fi

# ── 2. QLSTM Retrain (needs .venv + torch + pennylane) ──
echo "" | tee -a "$LOG"
echo "[2/2] QLSTM model retrain..." | tee -a "$LOG"
/home/ubuntu/sfc/.venv/bin/python -c "
import sys, os
sys.path.insert(0, '/home/ubuntu/sfc')
from qlstm_model import train
from qlstm_enhanced import build_training_data
print('[QLSTM] Building training data from git history...')
X, y = build_training_data(max_samples=2000)
print(f'[QLSTM] Data: X={X.shape}, y={y.shape}')
# Training happens here — qlstm_model.train() handles loader creation
import torch
from torch.utils.data import TensorDataset, DataLoader
trainset = TensorDataset(torch.tensor(X[:-200], dtype=torch.float32),
                         torch.tensor(y[:-200], dtype=torch.float32))
valset = TensorDataset(torch.tensor(X[-200:], dtype=torch.float32),
                       torch.tensor(y[-200:], dtype=torch.float32))
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
val_loader = DataLoader(valset, batch_size=32)
print('[QLSTM] Starting training (60 epochs)...')
model, history = train(None, train_loader, val_loader, epochs=60, lr=0.003, device='cpu')
if model is not None:
    torch.save(model.state_dict(), '/home/ubuntu/sfc/qlstm_model.pt')
    print('[QLSTM] ✅ Model saved to qlstm_model.pt')
else:
    print('[QLSTM] ❌ Training failed')
" 2>&1 | tee -a "$LOG"

# ── Summary ──
echo "" | tee -a "$LOG"
echo "✅ Weekly heavy training complete — $(date)" | tee -a "$LOG"
