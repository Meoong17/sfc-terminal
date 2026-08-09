#!/usr/bin/env bash
# deploy.sh — Manual deploy: copy clean design + inject data + deploy to GitHub Pages
# Usage: bash deploy.sh
set -euo pipefail

REPO_DIR="$HOME/sfc"
MASTER="/home/ubuntu/index.html"
DATA="$REPO_DIR/data.json"

# ── Guard ──
if [ ! -f "$MASTER" ] || [ "$(wc -c < "$MASTER")" -le 50000 ]; then
  echo "❌ /home/ubuntu/index.html kosong/rusak/terhapus — hentikan."
  exit 1
fi

cd "$REPO_DIR"

echo "→ Copy clean design to repo..."
cp "$MASTER" index.html

echo "→ Inject data..."
python3 inject_data.py "$DATA" index.html

echo "→ Git ops..."
git update-index --no-skip-worktree index.html 2>/dev/null || true
git add index.html
git commit -m "manual: deploy $(date -u '+%Y-%m-%d %H:%M:%S')"

echo "→ Push..."
git pull --rebase --autostash -X theirs origin main
git push origin main

git update-index --skip-worktree index.html 2>/dev/null || true
echo "✅ Deploy selesai."
