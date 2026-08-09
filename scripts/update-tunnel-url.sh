#!/bin/bash
# update-tunnel-url.sh — Auto-detect tunnel URL from cloudflared log & deploy to worker
# Run: every 5 minutes (cron) + on boot

LOG_FILE="/home/ubuntu/sfc/tunnel.log"
WORKER_FILE="/home/ubuntu/sfc/worker/index.js"
WRANGLER_DIR="/home/ubuntu/sfc"
URL_STORE="/tmp/tunnel-url-current.txt"

# Extract latest tunnel URL from log
NEW_URL=$(grep -oP 'https://[a-z-]+\.trycloudflare\.com' "$LOG_FILE" | tail -1)

if [ -z "$NEW_URL" ]; then
  echo "[$(date)] ERROR: No tunnel URL found in log" >> /home/ubuntu/sfc/tunnel-update.log
  exit 1
fi

# Check if URL actually changed
if [ -f "$URL_STORE" ]; then
  OLD_URL=$(cat "$URL_STORE")
  if [ "$NEW_URL" = "$OLD_URL" ]; then
    echo "[$(date)] No change — tunnel URL still $NEW_URL" >> /home/ubuntu/sfc/tunnel-update.log
    exit 0
  fi
fi

echo "[$(date)] Tunnel URL changed: $NEW_URL" >> /home/ubuntu/sfc/tunnel-update.log

# Update worker file
sed -i "s|const TUNNEL = 'https://.*\.trycloudflare\.com'|const TUNNEL = '$NEW_URL'|" "$WORKER_FILE"

# Save current URL
echo "$NEW_URL" > "$URL_STORE"

# Deploy to Cloudflare
cd "$WRANGLER_DIR" && npx wrangler deploy --env production 2>&1 >> /home/ubuntu/sfc/tunnel-update.log

echo "[$(date)] Deploy complete (URL: $NEW_URL)" >> /home/ubuntu/sfc/tunnel-update.log
