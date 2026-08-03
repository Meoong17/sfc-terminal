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
TMP_FILE="$REPO_DIR/data.json.tmp"

MAX_RETRIES=2; RETRY_DELAY=10; COLLECT_TIMEOUT=150

collect_with_retry() {
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    log "Collecting data (attempt $attempt/$MAX_RETRIES)..."
    rm -f "$TMP_FILE"
    timeout $COLLECT_TIMEOUT $PYTHON collect.py > "$TMP_FILE" 2>>sfc-pipeline.log
    local exit_code=$?
    # Atomic publish: only promote tmp -> data.json when output is complete & valid JSON.
    # Readers always see a whole file (old or new) — never empty/half-written.
    if [ $exit_code -eq 0 ] && [ -s "$TMP_FILE" ] \
       && $PYTHON -c 'import sys,json; json.load(open(sys.argv[1]))' "$TMP_FILE" 2>>sfc-pipeline.log; then
      mv -f "$TMP_FILE" data.json
      log "Collect succeeded (exit=$exit_code)"
      COLLECT_RESULT="ok"
      return 0
    fi
    log "Collect failed (exit=$exit_code), retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
    attempt=$((attempt + 1))
  done
  rm -f "$TMP_FILE"
  log "⚠ All collect attempts failed — keeping previous data.json"
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

# ── Commit & push (data files ONLY, throttled) ──
# Auto data commits are THROTTLED to cut .git growth (~78MB/mo at 7-min cadence) and
# network churn. Live dashboard freshness comes via SSE, so the static data snapshot
# only needs refreshing every THROTTLE_MIN. Non-data changes (code/dashboard) and any
# unpushed local commits always push immediately.
log "Committing (throttled)..."
git add data.json paper_trades.json paper_history.json
# Audit 2026-08-03: plain `git add -u` staged EVERY modified tracked file,
# so unrelated code edits got swept into "auto: SFC data" commits with a
# generic message (happened to collect.py/circuit_breaker.py/etf_flow.py).
# Restrict auto-staging to the data files only — code changes stay uncommitted
# and attributable (manual commits are still pushed via the AHEAD check below).
git add -u data.json paper_trades.json paper_history.json 2>/dev/null || true
git reset HEAD index.html 2>/dev/null || true

THROTTLE_MIN=60
NOW=$(date +%s)
LAST_PUSH=$(cat /tmp/sfc_last_data_push 2>/dev/null || echo 0)
AHEAD=$(git status -sb | grep -c "ahead" || true)
# staged non-data files?
NON_DATA=$(git diff --cached --name-only 2>/dev/null | grep -v -E '^(data\.json|paper_trades\.json|paper_history\.json)$' | wc -l)

if [ "$AHEAD" -eq 0 ] && [ "$NON_DATA" -eq 0 ] && [ $((NOW - LAST_PUSH)) -lt $((THROTTLE_MIN * 60)) ]; then
    log "Throttled data commit ($(((NOW - LAST_PUSH) / 60))m < ${THROTTLE_MIN}m, data-only) — skipping commit/push"
    git reset HEAD . 2>/dev/null || true
    GIT_RESULT="throttled"
elif git diff --staged --quiet && [ "$AHEAD" -eq 0 ]; then
    log "No changes — skipping push"
    GIT_RESULT="no-change"
else
    if ! git diff --staged --quiet -- data.json paper_trades.json paper_history.json; then
        # Commit ONLY the data pathspec (not the whole index). This prevents
        # sweeping code files that another process pre-staged with `git add`
        # into an "auto: SFC data" commit (same class of bug as 2e104fd8, which
        # only guarded against UNSTAGED code files — insufficient when a file
        # was already staged before this pipeline ran).
        git commit -m "auto: SFC data $(date -u '+%Y-%m-%d %H:%M:%S')" -- data.json paper_trades.json paper_history.json
    fi
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
    if [ "$GIT_RESULT" = "ok" ]; then
        date +%s > /tmp/sfc_last_data_push
    fi
fi

git update-index --skip-worktree index.html 2>/dev/null || true

log "Pipeline done: collect=$COLLECT_RESULT | git=$GIT_RESULT"
