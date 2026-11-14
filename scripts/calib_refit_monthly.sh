#!/usr/bin/env bash
# calib_refit_monthly.sh — Refit peta kalibrasi confidence SFC (bulanan)
#
# Konteks: `.calibration_state.json` sebelumnya tak pernah di-refit sejak 2026-08-12
# sementara ada 19.019 commit data.json — peta jadi basi DAN non-monoton. Script ini
# menjalankan refit + gate OOS setiap bulan.
#
# Gate OOS ada di dalam analysis/calibration_isotonic.py:
#   - split temporal 60/40 pada snapshot berlabel harga
#   - peta dipakai HANYA bila ECE **dan** Brier keduanya membaik di hold-out
#   - monoton diverifikasi numerik (monotone_verified) — peta non-monoton ditolak
#   - bila gagal gate, state TIDAK diubah (fallback: peta legacy -> raw confidence)
#
# Keluaran: ringkasan keputusan di stdout (tersimpan sebagai hasil cron),
#           detail penuh di /home/ubuntu/sfc/calib_refit.log.
set -uo pipefail

cd /home/ubuntu/sfc || { echo "GAGAL: /home/ubuntu/sfc tidak ada"; exit 1; }
VENV=/home/ubuntu/sfc/.venv/bin/python
LOG=/home/ubuntu/sfc/calib_refit.log
TMP=$(mktemp)

echo "=== $(date -Is) refit kalibrasi mulai ===" >> "$LOG"
"$VENV" analysis/calibration_isotonic.py > "$TMP" 2>&1
RC=$?
cat "$TMP" >> "$LOG"

# ── Gate kedua: walk-forward (expanding window) ────────────────────────────────
# The isotonic gate is a SINGLE temporal split: a map can pass it and still be
# useless. 2026-11 case: Platt passed (ECE 0.026 vs 0.170, Brier 0.249 vs 0.286)
# while mapping every raw confidence into 0.395-0.443 — pure base rate. Only a
# walk-forward comparison against climatology exposes that, so it is mandatory.
echo "=== $(date -Is) walk-forward gate ===" >> "$LOG"
"$VENV" analysis/calibration_walkforward.py >> "$LOG" 2>&1
RC_WF=$?
"$VENV" - <<'PYEOF' >> "$LOG" 2>&1
import json
S = ".calibration_state.json"
try:
    st = json.load(open(S))
    wf = json.load(open("analysis/calibration_walkforward_results.json"))
except Exception as e:
    print(f"[gate] skipped ({e})")
    raise SystemExit(0)
cm = st.get("calibration_map") or {}
if wf.get("accepted"):
    print("[gate] walk-forward accepted — calibration state kept as is")
elif cm.get("accepted"):
    cm["accepted"] = False
    cm["rejected_by"] = "walk_forward"
    cm["rejected_reason"] = wf.get("verdict")
    st["calibration_map"] = cm
    json.dump(st, open(S, "w"), indent=2)
    print("[gate] isotonic ACCEPT downgraded by walk-forward: " + str(wf.get("verdict")))
else:
    print("[gate] no accepted map to downgrade; verdict: " + str(wf.get("verdict")))
PYEOF
[ "$RC_WF" -eq 0 ] || RC=$RC_WF

echo "=== $(date -Is) selesai (exit $RC) ===" >> "$LOG"

# Batasi ukuran log (simpan 2000 baris terakhir)
tail -n 2000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"

# Ringkas ke stdout (hasil cron). Bila gagal, tetap laporkan alasannya.
if [ "$RC" -ne 0 ]; then
  echo "REFIT GAGAL (exit $RC) — 3 baris terakhir:"
  tail -n 3 "$TMP"
else
  grep -E "snapshot berlabel|ECE |Brier |metode terpilih|monoton terverifikasi|KEPUTUSAN" "$TMP" || tail -n 5 "$TMP"
fi
rm -f "$TMP"
exit "$RC"
