#!/usr/bin/env python3
"""Empirically tune calibration parameters to ACTUAL production snapshot cadence.

The stock confidence_calibration.py assumes PRICE_BASE_INTERVAL_MINUTES=5 and
PRICE_LOOKAHEAD_STEPS=6 (=> ~30 min). But real production snapshots are
median 72 min apart, so 6 snapshots = ~7h and the adaptive threshold scales
to sqrt(432/5)~9x => ~2.6-5%. BTC rarely moves that far in 6-12h, so 97.9%
of snapshots classify FLAT and the 0.70 price-outcome weight is dormant.

This script scans lookahead-horizons and base-thresholds against the REAL
spacing to find settings that yield meaningful labeled sample counts while
keeping the threshold a plausible daily move. It reports labeled % per
(lookahead_snapshots, base_threshold_abs) so we can pick empirically.
"""
import json, math, subprocess
from datetime import datetime

SFC_DIR="/home/ubuntu/sfc"

def extract(max_count=800):
    res=subprocess.check_output(["git","log","--oneline","--all","--diff-filter=M","--reverse","--","data.json"],text=True,timeout=40,cwd=SFC_DIR).strip().split("\n")
    res=[r for r in res if r.strip()]
    step=max(1,len(res)//max_count) if len(res)>max_count else 1
    res=res[::step]
    snaps=[]
    for line in res:
        sha=line.split()[0]
        try:
            c=subprocess.check_output(["git","show",f"{sha}:data.json"],text=True,timeout=10,cwd=SFC_DIR)
            if c.strip().startswith("{"): snaps.append(json.loads(c))
        except Exception: continue
    snaps.sort(key=lambda s:s.get("ts",""))
    return snaps

snaps=extract()

def classify(idx, look, base_int, base_thr):
    tgt=min(idx+look,len(snaps)-1)
    cp,tp=snaps[idx].get("btc",0),snaps[tgt].get("btc",0)
    if tgt<=idx or not cp or not tp: return "NONE"
    pct=(tp-cp)/cp
    try:
        a=datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
        b=datetime.fromisoformat(snaps[tgt].get("ts","").replace("Z","+00:00"))
        dm=max(base_int,(b-a).total_seconds()/60.0)
    except Exception: dm=base_int*look
    thr=max(0.002,min(0.05,base_thr*math.sqrt(dm/base_int)))
    if pct<=-thr: return "STRESS"
    if pct>=thr: return "CALM"
    return "FLAT"

print("scanning (lookahead_snapshots x base_threshold) -> labeled%  [n labeled]")
print("real median spacing ~72min; base_int set to 72")
for look in [1,2,3,4,6,8,12]:
    for base_thr in [0.005,0.01,0.015,0.02]:
        counts={"STRESS":0,"CALM":0,"FLAT":0,"NONE":0}
        for idx in range(len(snaps)):
            counts[classify(idx,look,72,base_thr)]+=1
        lab=counts["STRESS"]+counts["CALM"]
        total=len(snaps)
        print(f"  look={look:>2} thr={base_thr:.3f}  labeled={lab:>4} ({lab/total*100:>5.1f}%)  "
              f"S={counts['STRESS']} C={counts['CALM']} FLAT={counts['FLAT']}")

print("\n-- also: with lookahead fixed by TIME (pick snapshot ~6h ahead), base_int=5 --")
def classify_time(idx, target_min, base_int, base_thr):
    # find snapshot closest to target_min ahead
    try:
        t0=datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
    except Exception: return "NONE"
    best=None;bestd=None
    for j in range(idx+1,len(snaps)):
        try:
            tj=datetime.fromisoformat(snaps[j].get("ts","").replace("Z","+00:00"))
        except Exception: continue
        dm=(tj-t0).total_seconds()/60.0
        if dm>target_min:
            if bestd is None or dm<bestd:
                bestd=dm;best=j
            break
    if best is None: return "NONE"
    cp,tp=snaps[idx].get("btc",0),snaps[best].get("btc",0)
    if not cp or not tp: return "NONE"
    pct=(tp-cp)/cp
    dm=bestd if bestd else target_min
    thr=max(0.002,min(0.05,base_thr*math.sqrt(dm/base_int)))
    if pct<=-thr: return "STRESS"
    if pct>=thr: return "CALM"
    return "FLAT"

for target_min in [180,360,720]:
    for base_thr in [0.01,0.015]:
        counts={"STRESS":0,"CALM":0,"FLAT":0,"NONE":0}
        for idx in range(len(snaps)):
            counts[classify_time(idx,target_min,5,base_thr)]+=1
        lab=counts["STRESS"]+counts["CALM"]; total=len(snaps)
        print(f"  target={target_min:>4}min thr={base_thr:.3f}  labeled={lab:>4} ({lab/total*100:>5.1f}%)  "
              f"S={counts['STRESS']} C={counts['CALM']} FLAT={counts['FLAT']}")
