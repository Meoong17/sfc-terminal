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
# FIX (audit 2026-11-27): starting the process directly from this script lost
# SFC_ORIGIN_TOKEN (it lives in /etc/sfc-origin.env, root-only, injected into the
# systemd unit). sse_server.py treats a missing token as "open" — so one watchdog
# restart silently disabled origin authentication. Restart the unit instead, which
# carries the environment (User=ubuntu has passwordless sudo), and refuse to start
# an unauthenticated server if the unit is unavailable.
if systemctl cat sfc-sse.service >/dev/null 2>&1; then
    if sudo -n systemctl restart sfc-sse.service; then
        NEW_PID=$(systemctl show -p MainPID --value sfc-sse.service 2>/dev/null)
        echo "${NEW_PID:-unknown}" > "$PID_FILE"
        echo "[SSE] Restarted via systemd sfc-sse.service (PID=${NEW_PID:-?}) on $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
        exit 0
    fi
    echo "[SSE] ERROR: systemctl restart sfc-sse.service failed — refusing to start an unauthenticated instance" >> "$LOG"
    exit 1
fi
if [ -z "${SFC_ORIGIN_TOKEN:-}" ]; then
    echo "[SSE] ERROR: SFC_ORIGIN_TOKEN is not set — refusing to start an OPEN origin server" >> "$LOG"
    exit 1
fi
nohup "$PY" sse_server.py >> "$LOG" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "[SSE] Restarted sse_server.py (PID=$NEW_PID) on $(date -u '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
