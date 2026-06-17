#!/usr/bin/env bash
# mamba-weekly-train.sh — Weekly Mamba retraining with historical BTC data
# Runs: fetch_historical_btc.py → train_mamba.py
# Saves best model to models/mamba_weights.pth

set -euo pipefail

REPO_DIR="$HOME/sfc"
LOG_FILE="$REPO_DIR/mamba_weekly_train.log"
cd "$REPO_DIR" || { echo "FATAL: Cannot cd to $REPO_DIR"; exit 1; }

log() { echo "[Mamba] $(date -u '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

log "=== Weekly Mamba Training Started ==="

# 1. Fetch latest historical BTC data
log "Step 1/2: Fetching historical BTC data..."
if python3 fetch_historical_btc.py >> "$LOG_FILE" 2>&1; then
    log "✅ Historical BTC data fetched"
else
    log "⚠️  BTC fetch failed — continuing with cached data"
fi

# 2. Train Mamba
log "Step 2/2: Training Mamba SSM..."
PYTHONPATH="${REPO_DIR}/sfc2/venv/lib/python3.12/site-packages:${PYTHONPATH:-}" \
  timeout 600 python3 train_mamba.py >> "$LOG_FILE" 2>&1

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -eq 0 ]; then
    log "✅ Mamba training completed successfully"
    log "   Model saved to: models/mamba_weights.pth"
else
    log "❌ Mamba training failed (exit=$TRAIN_EXIT)"
fi

log "=== Weekly Mamba Training Finished ==="
echo ""
