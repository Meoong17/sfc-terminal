#!/bin/bash
# ============================================================
# sfc-weekly-train.sh — Heavy model training (Mamba encoder only)
# Runs weekly on Sunday 3AM via cron. Uses .venv (torch + einops).
#
# Langkah QLSTM DIPENSIUNKAN 2026-09-18 (audit G1/A3):
#   - memanggil `build_training_data()` dari models/qlstm_enhanced.py yang sudah tidak ada (ImportError),
#   - checkpoint qlstm_model.pt saat ini berasal dari target sirkular (val_loss 8.1e-08) sehingga
#     melatih ulang dengan skrip itu tidak memperbaiki apa pun.
# Skrip ini sekarang hanya melatih Mamba dan WAJIB keluar non-zero bila gagal,
# supaya kegagalan terlihat di status cron (dulu selalu dilaporkan "ok").
# ============================================================
set -uo pipefail
cd /home/ubuntu/sfc || exit 1
PY=/home/ubuntu/sfc/.venv/bin/python
LOG="/home/ubuntu/sfc/logs/train_weekly_$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG")"

echo "╔═══════════════════════════════════════════════" | tee -a "$LOG"
echo "║ SFC WEEKLY HEAVY TRAIN (Mamba hanya) — $(date)" | tee -a "$LOG"
echo "╚═══════════════════════════════════════════════" | tee -a "$LOG"

# ── Pre-flight: dependensi wajib ──
if ! "$PY" -c "import torch, einops" >>"$LOG" 2>&1; then
    echo "[PRE-FLIGHT] ❌ torch/einops tidak dapat di-import dari $PY" | tee -a "$LOG"
    exit 1
fi
if [ ! -f /home/ubuntu/sfc/models/train_mamba.py ]; then
    echo "[PRE-FLIGHT] ❌ models/train_mamba.py tidak ditemukan" | tee -a "$LOG"
    exit 1
fi

# ── 1. Mamba Encoder ──
echo "" | tee -a "$LOG"
echo "[1/1] Mamba model training..." | tee -a "$LOG"
"$PY" /home/ubuntu/sfc/models/train_mamba.py 2>&1 | tee -a "$LOG"
MAMBA_EXIT=${PIPESTATUS[0]}

if [ "$MAMBA_EXIT" -eq 0 ]; then
    echo "[MAMBA] ✅ Training complete" | tee -a "$LOG"
else
    echo "[MAMBA] ❌ Training failed (exit=$MAMBA_EXIT)" | tee -a "$LOG"
fi

# ── Summary ──
echo "" | tee -a "$LOG"
if [ "$MAMBA_EXIT" -eq 0 ]; then
    echo "✅ Weekly heavy training complete — $(date)" | tee -a "$LOG"
    exit 0
else
    echo "❌ Weekly heavy training FAILED (mamba=$MAMBA_EXIT) — $(date)" | tee -a "$LOG"
    exit 1
fi
