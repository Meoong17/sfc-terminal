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
if python3 data_sources/fetch_historical_btc.py >> "$LOG_FILE" 2>&1; then
    log "✅ Historical BTC data fetched"
else
    log "⚠️  BTC fetch failed — continuing with cached data"
fi

# 2. Train Mamba
# NOTE: the || true here is intentional and NOT a mistake. `set -e` causes
# the shell to exit immediately when any command returns a non-zero exit
# code, which means if train_mamba.py fails (exit≠0), the script would
# exit right there — before `TRAIN_EXIT=$?` on the next line can read
# the exit code, and before the `if [ $TRAIN_EXIT -eq 0 ]` block below
# can log a meaningful error message. Confirmed by test: the "❌ Mamba
# training failed" message was never appearing in the log on failure,
# making it impossible to diagnose training issues from the log alone.
# `|| true` tells bash "treat this pipeline as successful regardless of
# exit code", letting set -e pass through so we can capture and log it.
log "Step 2/2: Training Mamba SSM..."
# Temporarily disable -e for this one command so we can capture its exit
# code and log a clear error message before deciding what to do.
# `set -e` would otherwise exit the script immediately on non-zero return,
# before the `if [ $TRAIN_EXIT -eq 0 ]` block below can run — confirmed:
# "❌ Mamba training failed" was never appearing in the log on failure.
# We restore `set -e` immediately after so the rest of the script is
# still protected by it.
set +e
PYTHONPATH="${REPO_DIR}/sfc2/venv/lib/python3.12/site-packages:${PYTHONPATH:-}" \
  timeout 600 python3 models/train_mamba.py >> "$LOG_FILE" 2>&1
TRAIN_EXIT=$?
set -e
if [ $TRAIN_EXIT -eq 0 ]; then
    log "✅ Mamba training completed successfully"
    log "   Model saved to: models/mamba_weights.pth"
else
    log "❌ Mamba training failed (exit=$TRAIN_EXIT)"
    log "   Check $LOG_FILE for full Python traceback"
fi

log "=== Weekly Mamba Training Finished ==="
echo ""
