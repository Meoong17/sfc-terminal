#!/usr/bin/env bash
# weekly-model-train.sh — Weekly training for all SFC ML models
# Runs: fetch_historical → train_mamba → train_xgboost → train_hmm
set -euo pipefail

REPO_DIR="$HOME/sfc"
LOG_FILE="$REPO_DIR/weekly_train.log"
cd "$REPO_DIR" || { echo "FATAL: Cannot cd to $REPO_DIR"; exit 1; }
export PYTHONPATH="${REPO_DIR}/sfc2/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

log() { echo "[Train] $(date -u '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

log "=== Weekly Model Training Started ==="

# 1. Fetch historical BTC data
log "Step 1/4: Fetching historical BTC data..."
if python3 fetch_historical_btc.py >> "$LOG_FILE" 2>&1; then
    log "✅ Historical BTC data fetched"
else
    log "⚠️ BTC fetch failed — continuing with cached data"
fi

# 2. Train Mamba SSM
log "Step 2/4: Training Mamba SSM..."
if timeout 600 python3 train_mamba.py >> "$LOG_FILE" 2>&1; then
    log "✅ Mamba training done"
else
    log "❌ Mamba training failed"
fi

# 3. Train XGBoost Meta-Ensemble
log "Step 3/4: Training XGBoost Meta-Ensemble..."
if timeout 600 python3 ensemble_meta.py >> "$LOG_FILE" 2>&1; then
    log "✅ XGBoost training done"
else
    log "⚠️ XGBoost training failed (non-fatal)"
fi

# 4. Train HMM Regime Detector
log "Step 4/4: Training HMM Regime Detector..."
if timeout 300 python3 hmm_regime.py >> "$LOG_FILE" 2>&1; then
    log "✅ HMM training done"
else
    log "⚠️ HMM training failed (non-fatal)"
fi

log "=== Weekly Model Training Finished ==="
echo ""
