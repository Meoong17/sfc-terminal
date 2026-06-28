#!/usr/bin/env bash
# SFC Terminal — Real-time data pipeline
# Runs: collect.py → commit data.json → push to GitHub → Pages auto-deploy
set -uo pipefail
# NOTE: no 'set -e' — collect.py may occasionally hit API timeouts;
# we handle that gracefully with retry logic instead of aborting.

# ── Timestamped logging ──
log() { echo "[SFC] $(date -u '+%H:%M:%S') $*"; }

# ── File locking — prevent overlapping runs ──
LOCKFILE="/tmp/sfc-pipeline.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    log "⚠ Another pipeline instance is already running — exiting."
    exit 0
fi
trap 'rm -f "$LOCKFILE"' EXIT

REPO_DIR="$HOME/sfc"
cd "$REPO_DIR" || { log "FATAL: Cannot cd to $REPO_DIR"; exit 1; }

# ── Track overall result ──
COLLECT_RESULT="skipped"
GIT_RESULT="skipped"

# ── Make index.html + app.js invisible to git — never committed/updated by pipeline ──
git update-index --skip-worktree index.html app.js 2>/dev/null || true

# ── Collect data with timeout & retry ──
PYTHON="/usr/bin/python3"
# Add sfc2 venv path so numpy loads before collect.py line 9
SFC2_VENV="/home/ubuntu/sfc2/venv/lib/python3.12/site-packages"
if [ -d "$SFC2_VENV" ]; then
  export PYTHONPATH="${SFC2_VENV}:${PYTHONPATH:-}"
fi
MAX_RETRIES=2
RETRY_DELAY=10
COLLECT_TIMEOUT=150  # seconds

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
  log "⚠ All collect attempts failed. Using cached data if available."
  COLLECT_RESULT="failed"
  return 1
}

collect_with_retry

# ── Paper Trading (after data.json is fully written) ──
log "Running paper trader..."
$PYTHON paper_trader.py 2>>sfc-pipeline.log || log "⚠ Paper trader skipped (no data)"


# ── Data ready — page fetches data.json live ──
log "Data collection complete (index.html unchanged)"

# ── Commit & push (skip-worktree melindungi index.html & app.js) ──
log "Committing..."
git add data.json paper_trades.json paper_history.json
git add -u 2>/dev/null || true

if git diff --staged --quiet; then
    log "No changes — skipping push"
    GIT_RESULT="no-change"
else
    # Strategy: fetch remote, soft-reset ke remote HEAD, lalu commit di atasnya
    # Soft reset tdk sentuh working tree — aman untuk skip-worktree files
    log "Fetching latest remote..."
    git fetch origin main 2>&1 || {
        log "❌ Fetch failed — trying direct push"
        git push origin main 2>&1 && GIT_RESULT="ok" || GIT_RESULT="push-failed"
        # Final restore
        cp /home/ubuntu/index.html index.html 2>/dev/null || true
        log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
        exit 0
    }

    # Soft reset ke remote — pindahin branch pointer tanpa sentuh file
    git reset --soft origin/main 2>/dev/null || true

    # Commit di atas remote HEAD
    if git diff --staged --quiet; then
        log "No new changes after sync — skipping push"
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
fi

# ── Remove skip-worktree + restore index.html yg bener ──
git update-index --no-skip-worktree index.html app.js 2>/dev/null || true
cp /home/ubuntu/index.html index.html 2>/dev/null || true
log "index.html restored (post-pipeline cleanup)"

log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
