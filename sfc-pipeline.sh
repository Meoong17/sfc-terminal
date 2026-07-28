#!/usr/bin/env bash
# SFC Terminal — Real-time data pipeline (repo copy)
# Cron runs ~/.hermes/scripts/sfc-pipeline.sh — sync both when editing
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

PYTHON="/home/ubuntu/sfc/.venv/bin/python3"

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

# REMOVED (2026-07): paper_trader.py execution — Paper Trading has been
# removed; SFC Terminal is now an analysis-only tool. trading/paper_trader.py
# is left in place (harmless, unused) rather than deleted, in case any other
# script references it.

# ── Make git ignore index.html ──
git update-index --skip-worktree index.html 2>/dev/null || true

# ── Restore index.html from authoritative source ──
log "Restoring index.html from /home/ubuntu/index.html..."
if [ -f /home/ubuntu/index.html ] && [ "$(wc -c < /home/ubuntu/index.html)" -gt 50000 ]; then
  cp /home/ubuntu/index.html index.html
  log "Restored from /home/ubuntu/index.html ($(wc -c < index.html) bytes)"
elif [ -f index.html.bak ] && [ "$(wc -c < index.html.bak)" -gt 50000 ]; then
  cp index.html.bak index.html
  log "Restored from .bak fallback ($(wc -c < index.html) bytes)"
else
  log "⚠ No valid restore source found — keeping current index.html"
fi

log "Injecting data into index.html..."
$PYTHON inject_data.py data.json index.html 2>>sfc-pipeline.log || \
  log "⚠ Inject failed (non-fatal)"

# ── Commit & push (data files ONLY) ──
log "Committing..."
git add data.json paper_trades.json paper_history.json
git add -u 2>/dev/null || true
git reset HEAD index.html 2>/dev/null || true

if git diff --staged --quiet; then
    log "No changes — skipping push"
    GIT_RESULT="no-change"
else
    git commit -m "auto: SFC data $(date -u '+%Y-%m-%d %H:%M:%S')"
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

git update-index --skip-worktree index.html 2>/dev/null || true

log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
