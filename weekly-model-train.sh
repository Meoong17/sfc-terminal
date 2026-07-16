#!/usr/bin/env bash
# weekly-model-train.sh — Weekly training for all SFC ML models
# Runs: fetch_historical → train_mamba → train_xgboost → train_hmm → train_drl
set -euo pipefail

REPO_DIR="$HOME/sfc"
LOG_FILE="$REPO_DIR/weekly_train.log"
cd "$REPO_DIR" || { echo "FATAL: Cannot cd to $REPO_DIR"; exit 1; }
export PYTHONPATH="${REPO_DIR}/sfc2/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

log() { echo "[Train] $(date -u '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

log "=== Weekly Model Training Started ==="

# 1. Fetch historical BTC data
log "Step 1/5: Fetching historical BTC data..."
if python3 data_sources/fetch_historical_btc.py >> "$LOG_FILE" 2>&1; then
    log "✅ Historical BTC data fetched"
else
    log "⚠️ BTC fetch failed — continuing with cached data"
fi

# 2. Train Mamba SSM
log "Step 2/5: Training Mamba SSM..."
if timeout 600 python3 models/train_mamba.py >> "$LOG_FILE" 2>&1; then
    log "✅ Mamba training done"
else
    log "❌ Mamba training failed"
fi

# 3. Train XGBoost Meta-Ensemble
log "Step 3/5: Training XGBoost Meta-Ensemble..."
if timeout 600 python3 models/ensemble_meta.py >> "$LOG_FILE" 2>&1; then
    log "✅ XGBoost training done"
else
    log "⚠️ XGBoost training failed (non-fatal)"
fi

# 4. Train HMM Regime Detector
log "Step 4/5: Training HMM Regime Detector..."
if timeout 300 python3 models/hmm_regime.py >> "$LOG_FILE" 2>&1; then
    log "✅ HMM training done"
else
    log "⚠️ HMM training failed (non-fatal)"
fi

# 5. Train M68 DRL Agent
# Non-fatal like HMM/XGBoost — collect.py already falls back to the
# existing rule-based signal if models/drl_agent.pkl doesn't exist or
# fails to load, so a failed training run here doesn't break the pipeline,
# it just means M68 stays on the rule-based fallback until the next
# successful weekly run.
log "Step 5/5: Training M68 DRL Agent..."
if timeout 300 python3 trading/train_drl_agent_script.py >> "$LOG_FILE" 2>&1; then
    log "✅ DRL agent training done"
else
    log "⚠️ DRL agent training failed (non-fatal)"
fi

log "=== Weekly Model Training Finished ==="
echo ""
