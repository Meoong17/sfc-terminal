#!/usr/bin/env bash
# refresh_engine_inputs.sh — refresh the automated inputs the engine needs.
#
# Why this exists: the structure engine froze at 2026-10-06 because two of its
# inputs died silently — BitMEX /funding started returning 403 (2026-09-16) and
# the CME basis CSV was a manual TradingView export (last fresh 2026-09-01).
# Both now have automated replacements; this script keeps them fresh, and it is
# called from run_flqd_engine.sh BEFORE the engines run.
#
# Failure policy: a failed fetcher must NOT abort the run. The engine has
# explicit fallbacks and data_integrity reports the staleness — but silence is
# not allowed either, so every failure is echoed with a [FAIL] marker. The
# script exits non-zero if ALL fetchers failed (that is a real outage).
set -uo pipefail

REPO="$HOME/sfc"
cd "$REPO" || { echo "[FAIL] cannot cd to $REPO"; exit 1; }
PY="$REPO/.venv/bin/python"
LOG="$REPO/logs/engine_inputs.log"
mkdir -p "$(dirname "$LOG")"

fails=0
total=0

run_fetcher() {
  local name="$1"; shift
  total=$((total + 1))
  echo "=== $(date -u +%FT%TZ) $name" >> "$LOG"
  if timeout 180 "$PY" "$@" >> "$LOG" 2>&1; then
    echo "[OK]   $name"
  else
    local rc=$?
    echo "[FAIL] $name (exit $rc) — see $LOG"
    fails=$((fails + 1))
  fi
}

run_fetcher "funding (Binance perp, 2019+)"  analysis/fetch_binance_funding.py
run_fetcher "basis (Binance quarterly vs spot)" analysis/fetch_binance_basis.py
run_fetcher "orderflow daily tail (closes the monthly lag)" analysis/fetch_orderflow.py --daily-tail

if [ "$fails" -eq "$total" ]; then
  echo "[FAIL] all engine-input fetchers failed — engine will read stale inputs"
  exit 1
fi
echo "engine inputs refreshed ($((total - fails))/$total ok)"
