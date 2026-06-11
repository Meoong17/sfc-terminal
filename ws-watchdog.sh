#!/usr/bin/env bash
# ws-watchdog.sh — Ensures binance_ws.py is always running
# Checks every invocation; restarts if dead.
set -u

SCRIPT_DIR="/home/ubuntu/sfc"
PID_FILE="/tmp/binance_ws.pid"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        # Already running
        exit 0
    fi
    # Stale PID
    rm -f "$PID_FILE"
fi

# Start new instance
cd "$SCRIPT_DIR"
nohup python3 binance_ws.py > /dev/null 2>&1 &
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo "[WS] Started (PID=$NEW_PID)" >&2
