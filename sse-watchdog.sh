#!/usr/bin/env bash
# sse-watchdog.sh — Ensures sse_server.py (SFC live dashboard SSE) is always running.
# Mirrors ws-watchdog.sh: checks every invocation; restarts if dead.
# Runs on .venv/bin/python (sse_server imports fastapi/uvicorn from the venv),
# binds 127.0.0.1:8765, reached via the Cloudflare named tunnel.
set -u

SFC_DIR="/home/ubuntu/sfc"
PY="$SFC_DIR/.venv/bin/python3"
LOG="$SFC_DIR/sse_server.log"
PID_FILE="/tmp/sse_server.pid"

# Already running? (check PID file first, then pgrep as a fallback)
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# Fallback: even without a PID file, avoid double-start if a stale process lives on.
if pgrep -f "sse_server\.py" > /dev/null 2>&1; then
    # It's alive but our PID file is gone/stale — just re-record the PID.
    NEW_PID=$(pgrep -f "sse_server\.py" | head -1)
    echo "$NEW_PID" > "$PID_FILE"
    exit 0
fi

# Start a new instance (loopback only; the Cloudflare tunnel reaches it via localhost).
cd "$SFC_DIR"
nohup "$PY" sse_server.py >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "[SSE] Restarted sse_server.py (PID=$NEW_PID) on $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
