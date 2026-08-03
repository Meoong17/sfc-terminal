#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused walk-forward: does the CONSOLIDATED regime severity (single driver)
predict forward BTC returns, point-in-time, no look-ahead?

Scope honesty (matches walk-forward-validation skill pitfall 14): the regime
subsystems (hmm/adv) only exist in the ~2-month live window (data.json history
starts 2026-06-09). adv_regime needs m1-m5 which have no earlier history, so a
long FRED reconstruction of the consolidation driver is impossible. This is a
DIRECTIONAL check on the available live window, not a portable-cutoff result.

Method:
  1. Extract every data.json snapshot from git history (dedup by calendar day).
  2. For each, rebuild the consolidated severity with the FIXED consolidation
     (adv now in 4-regime space; behavior included as display-only modifier).
  3. Forward returns from actual BTC price at +7d and +30d.
  4. Bucket by consolidated label (BULLISH/ELEVATED/STRESSED); bootstrap CI on
     the BULLISH-vs-STRESSED mean gap (two-tailed).
"""
import json, subprocess, sys, math
import numpy as np

SFC_DIR = "/home/ubuntu/sfc"
sys.path.insert(0, SFC_DIR)
from data_sources.regime_consolidation import consolidate_regime

# ── 1. Extract daily snapshots ──
def daily_snapshots():
    # Get commits touching data.json with their commit timestamps.
    result = subprocess.check_output(
        ["git", "log", "--format=%H %cI", "--all", "--diff-filter=M", "--", "data.json"],
        text=True, timeout=60, cwd=SFC_DIR).strip().split("\n")
    result = [r for r in result if r.strip()]
    # Map day -> (sha, commit_iso). Keep the LAST commit of each calendar day.
    day_commit = {}
    for line in result:
        parts = line.split()
        if len(parts) < 2: continue
        sha, ciso = parts[0], parts[1]
        day = ciso[:10]
        day_commit[day] = sha  # git log is reverse? default newest-first; keep latest per day
    days = {}
    for day, sha in day_commit.items():
        try:
            content = subprocess.check_output(["git", "show", f"{sha}:data.json"],
                                              text=True, timeout=10, cwd=SFC_DIR)
            data = json.loads(content)
            ts = data.get("ts")
            if not ts: continue
            days[day] = {"ts": ts, "sha": sha, "data": data}
        except Exception:
            continue
    return days

days = daily_snapshots()
print(f"Extracted {len(days)} daily snapshots ({min(days)} .. {max(days)})")

# ── 2. Build series: (date, consolidated_severity, label, btc) ──
series = []
for day in sorted(days):
    d = days[day]["data"]
    btc = d.get("btc")
    if not btc: continue
    lab, det = consolidate_regime(
        regime=d.get("regime"),
        regime_prob=d.get("regime_prob"),
        hmm_regime=d.get("hmm_regime"),
        hmm_crisis_prob=d.get("hmm_crisis_prob"),
        adv_regime=d.get("adv_regime"),
        adv_crisis_prob=d.get("adv_crisis_prob"),
        behavior_state=d.get("behavior_state"),
    )
    sev = det.get("severity")
    if sev is None: continue
    series.append({"date": day, "sev": float(sev), "label": lab, "btc": float(btc)})

series.sort(key=lambda x: x["date"])
n = len(series)
print(f"Series points with btc: {n}")

# ── 3. Forward returns ──
def fwd_return(idx, horizon_days):
    t0 = series[idx]["date"]
    target_date = series[idx]["date"]  # placeholder
    # find future price: last snapshot whose date >= t0 + horizon
    # (approx by ordinal within the series)
    import datetime
    d0 = datetime.date.fromisoformat(t0)
    dt = d0 + datetime.timedelta(days=horizon_days)
    tgt = dt.isoformat()
    fut = [s for s in series if s["date"] >= tgt]
    if not fut:
        return None
    p_fut = fut[0]["btc"]
    p0 = series[idx]["btc"]
    return (p_fut / p0 - 1.0) * 100.0

for H in (7, 30):
    buckets = {"BULLISH": [], "ELEVATED": [], "STRESSED": []}
    for i, s in enumerate(series):
        r = fwd_return(i, H)
        if r is not None and s["label"] in buckets:
            buckets[s["label"]].append(r)

    def boot_diff_ci(a, b, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        a = np.array(a); b = np.array(b)
        if len(a) == 0 or len(b) == 0:
            return None, None, None
        na, nb = len(a), len(b)
        diffs = np.empty(n_boot)
        idxa = rng.integers(0, na, (n_boot, na))
        idxb = rng.integers(0, nb, (n_boot, nb))
        diffs = a[idxa].mean(axis=1) - b[idxb].mean(axis=1)
        lo, hi = np.percentile(diffs, [5, 95])
        return float(diffs.mean()), float(lo), float(hi)

    out = {}
    for k, v in buckets.items():
        out[k] = (len(v), round(np.mean(v), 3) if v else None)
    print(f"\n=== Horizon {H}d ===")
    for k, (cnt, mn) in out.items():
        print(f"  {k:9s} n={cnt:3d} mean_fwd={mn}")
    if out["STRESSED"][0] > 0 and out["BULLISH"][0] > 0:
        est, lo, hi = boot_diff_ci(buckets["BULLISH"], buckets["STRESSED"])
        sig = (hi < 0 or lo > 0)
        print(f"  BULLISH−STRESSED gap: {est:+.2f}pp  90%CI=[{lo:+.2f},{hi:+.2f}]  significant={sig}")
        print(f"  (negative gap = lower severity → higher forward return = correct polarity)")

print("\nDONE. This is a DIRECTIONAL check on ~2mo live window; NOT a portable cutoff.")
