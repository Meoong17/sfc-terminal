#!/usr/bin/env bash
# Daily refresh of the FLQD v2 measurement engine (display-only layer).
# Output: analysis/flqd_engine_results.json -> read by collect.py as data.json.flqd
set -u
REPO="$HOME/sfc"
cd "$REPO" || exit 1
LOG="$REPO/logs/flqd_engine.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "=== $(date -u +%FT%TZ) start"
  "$REPO/.venv/bin/python" "$REPO/analysis/flqd_engine.py" 2>&1 | tail -40
  rc=${PIPESTATUS[0]}
  echo "=== $(date -u +%FT%TZ) exit=$rc"
} >> "$LOG" 2>&1
