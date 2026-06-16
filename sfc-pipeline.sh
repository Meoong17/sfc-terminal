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


# ── Inject data into HTML ──
# Restore real dashboard from git (find last commit with id="app" = real dashboard, not login page)
log "Restoring clean index.html from git..."
GOOD_HASH=$(git log --all --format='%H' -- index.html 2>/dev/null | while read h; do
  if git show "$h:index.html" 2>/dev/null | grep -q 'id="app"'; then
    echo "$h" && break
  fi
done)
if [ -n "$GOOD_HASH" ]; then
  git show "$GOOD_HASH:index.html" > index.html 2>/dev/null
  log "Restored from ${GOOD_HASH:0:7} ($(wc -c < index.html) bytes, #app present)"
else
  log "⚠ No good index.html found in git history — using current file"
fi
log "Injecting data into index.html..."
$PYTHON inject_data.py data.json index.html index.html 2>>sfc-pipeline.log || \
  log "⚠ Inject failed (non-fatal)"

# ── Commit & push ──
log "Committing..."
git add data.json index.html paper_trades.json paper_history.json
git add -u 2>/dev/null || true  # add any tracked file changes
if git diff --staged --quiet; then
    log "No changes — skipping push"
    GIT_RESULT="no-change"
else
    git commit -m "auto: SFC data $(date -u '+%Y-%m-%d %H:%M:%S')"

    # Pull remote changes before pushing (handles GH Actions concurrent pushes)
    log "Syncing with remote..."
    if git pull --rebase --autostash -X theirs origin main 2>&1; then
        log "Pushing..."
        if git push origin main 2>&1; then
            log "✅ Pushed — Pages deploying..."
            GIT_RESULT="ok"
        else
            log "❌ Push failed!"
            GIT_RESULT="push-failed"
        fi
    else
        log "❌ Rebase failed, trying merge strategy..."
        git rebase --abort 2>/dev/null || true
        if git pull -X theirs origin main 2>&1 && git push origin main 2>&1; then
            log "✅ Pushed (merge path) — Pages deploying..."
            GIT_RESULT="ok"
        else
            log "❌ Pull+push both failed!"
            GIT_RESULT="sync-failed"
        fi
    fi
fi

log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
