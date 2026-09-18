#!/usr/bin/env bash
# SFC Terminal — weekly backup of accumulating state that lives OUTSIDE git.
#
# Why: data_collection_daily.json is the daily observation dataset feeding the
# adv_regime (k-means + Markov) detector. It grows by exactly one row per day
# and only publishes once it reaches 250 rows (currently ~46 since 2026-08-04),
# so losing the file resets that gate to zero and pushes publication out by
# ~8 months. It is .gitignore'd (runtime artifact), hence this snapshot job:
# the live file stays untracked, a plain-JSON copy lands in git every week.
#
# Cron runs the COPY at ~/.hermes/scripts/backup_daily_state.sh — sync both.
set -uo pipefail

REPO_DIR="/home/ubuntu/sfc"
cd "$REPO_DIR" || exit 1

LOG="$REPO_DIR/logs/daily_state_backup.log"
PYTHON="/home/ubuntu/sfc/.venv/bin/python3"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# src:dest — dest must NOT reuse the ignored basename (the .gitignore entry
# `data_collection_daily.json` is a bare pattern and matches any directory).
FILES=(
  "data_collection_daily.json:adv_regime_dataset.json"
)

SNAP_DIR="$REPO_DIR/backups/daily_state"
mkdir -p "$SNAP_DIR" "$(dirname "$LOG")"

snapped=0
for entry in "${FILES[@]}"; do
  src="${entry%%:*}"
  dest="${entry##*:}"
  if [ ! -f "$src" ]; then
    log "SKIP $src (missing)"
    continue
  fi
  cp -f "$src" "$SNAP_DIR/$dest.tmp" && mv -f "$SNAP_DIR/$dest.tmp" "$SNAP_DIR/$dest" || {
    log "ERROR copy failed for $src"
    exit 1
  }
  snapped=$((snapped + 1))
  log "snapshot $src ($(stat -c%s "$src") B) -> backups/daily_state/$dest"
done

if [ "$snapped" -eq 0 ]; then
  log "ERROR nothing snapshotted"
  exit 1
fi

ROWS=$("$PYTHON" -c "
import json
d = json.load(open('data_collection_daily.json'))
print(len(d.get('dates', [])))" 2>/dev/null || echo "?")
log "adv_regime rows=$ROWS (publication gate = 250)"

git add -- "$SNAP_DIR" 2>>"$LOG" || { log "ERROR git add failed"; exit 1; }
if git diff --cached --quiet -- "$SNAP_DIR"; then
  log "no change since last snapshot — nothing to commit"
  exit 0
fi

if git commit -q \
    -m "chore(backup): snapshot gitignored daily state ($(date +%F), adv_regime rows=$ROWS)" \
    -m "Weekly plain-JSON copy of state kept out of git by .gitignore. Source data_collection_daily.json is the adv_regime (k-means+Markov) daily observation set: +1 row/day, publishes at 250 rows, so a lost file resets that gate to zero (would delay publication to ~2027). Snapshot is named adv_regime_dataset.json because the .gitignore pattern matches the original basename in any directory. Live file remains untracked." \
    -- "$SNAP_DIR"; then
  log "committed $(git rev-parse --short HEAD)"
  if git push -q origin main 2>>"$LOG"; then
    log "pushed to origin/main"
  else
    log "WARN push failed — commit is local; sfc-pipeline.sh pushes whenever AHEAD>0"
  fi
else
  log "ERROR commit failed"
  exit 1
fi
