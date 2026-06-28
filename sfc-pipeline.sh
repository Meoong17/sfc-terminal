#!/usr/bin/env bash
# SFC Terminal — Real-time data pipeline
set -uo pipefail

log() { echo "[SFC] $(date -u '+%H:%M:%S') $*"; }

LOCKFILE="/tmp/sfc-pipeline.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    log "⚠ Another pipeline instance is already running — exiting."
    exit 0
fi
trap 'rm -f "$LOCKFILE"' EXIT

REPO_DIR="$HOME/sfc"
cd "$REPO_DIR" || { log "FATAL: Cannot cd to $REPO_DIR"; exit 1; }

COLLECT_RESULT="skipped"
GIT_RESULT="skipped"

PYTHON="/usr/bin/python3"
SFC2_VENV="/home/ubuntu/sfc2/venv/lib/python3.12/site-packages"
[ -d "$SFC2_VENV" ] && export PYTHONPATH="${SFC2_VENV}:${PYTHONPATH:-}"

MAX_RETRIES=2; RETRY_DELAY=10; COLLECT_TIMEOUT=150

collect_with_retry() {
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    log "Collecting data (attempt $attempt/$MAX_RETRIES)..."
    timeout $COLLECT_TIMEOUT $PYTHON collect.py > data.json 2>>sfc-pipeline.log
    local exit_code=$?
    if [ $exit_code -eq 0 ] && [ -s data.json ]; then
      log "Collect succeeded (exit=$exit_code)"
      COLLECT_RESULT="ok"
      return 0
    fi
    log "Collect failed (exit=$exit_code), retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
    attempt=$((attempt + 1))
  done
  log "⚠ All collect attempts failed."
  COLLECT_RESULT="failed"
  return 1
}

collect_with_retry

log "Running paper trader..."
$PYTHON paper_trader.py 2>>sfc-pipeline.log || log "⚠ Paper trader skipped"

# ── Commit & push — pendekatan nuclear reset ──
# Simpan data files, reset keras ke remote, timpa data files, commit
log "Committing..."
TMP_DATA=$(mktemp -d)
cp data.json paper_trades.json paper_history.json "$TMP_DATA/" 2>/dev/null || true

git fetch origin main 2>&1 || { log "❌ Fetch failed"; GIT_RESULT="fetch-failed"; }
git reset --hard origin/main 2>&1 || { log "❌ Reset failed"; GIT_RESULT="reset-failed"; }

# Restore data files yang baru dikoleksi
cp "$TMP_DATA/data.json" . 2>/dev/null || true
cp "$TMP_DATA/paper_trades.json" . 2>/dev/null || true
cp "$TMP_DATA/paper_history.json" . 2>/dev/null || true
rm -rf "$TMP_DATA"

git add data.json paper_trades.json paper_history.json

if git diff --staged --quiet; then
    log "No changes — skipping push"
    GIT_RESULT="no-change"
else
    git commit -m "auto: SFC data $(date -u '+%Y-%m-%d %H:%M:%S')"
    log "Pushing..."
    if git push origin main 2>&1; then
        log "✅ Pushed — Pages deploying..."
        GIT_RESULT="ok"
    else
        log "❌ Push failed!"
        GIT_RESULT="push-failed"
    fi
fi

# Restore index.html dari source aman — git reset --hard nimpah file apapun
cp /home/ubuntu/index.html index.html 2>/dev/null || true
log "index.html restored from /home/ubuntu/index.html"

log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
