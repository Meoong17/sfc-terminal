#!/usr/bin/env python3
"""Quantify price-outcome ground-truth sparsity across ALL git snapshots.

Root-cause hypothesis: production snapshots are ~1-2h apart (NOT 5 min).
Lookahead=6 => ~6-12h horizon, adaptive threshold scales sqrt(dt/5) up to
~2.5-5%. BTC rarely moves that much in 6-12h => almost everything is
"FLAT(<thr)" => price_outcome=None => the 0.70 price-outcome weight is
effectively dormant and the calibration falls back to model-only.
"""
import json, math, subprocess
from datetime import datetime
from collections import Counter

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
print(f"total snapshots: {len(snaps)}")

# 1. Actual snapshot spacing distribution
deltas=[]
for i in range(1,len(snaps)):
    try:
        a=datetime.fromisoformat(snaps[i-1].get("ts","").replace("Z","+00:00"))
        b=datetime.fromisoformat(snaps[i].get("ts","").replace("Z","+00:00"))
        deltas.append((b-a).total_seconds()/60.0)
    except Exception: pass
if deltas:
    import statistics
    print(f"\nsnapshot spacing (min): median={statistics.median(deltas):.1f} mean={statistics.mean(deltas):.1f} min={min(deltas):.1f} max={max(deltas):.1f}")
    # bucket
    bkt=Counter()
    for d in deltas:
        bkt["<10min"]+=1 if d<10 else 0
        bkt["10-60min"]+=1 if 10<=d<60 else 0
        bkt["1-2h"]+=1 if 60<=d<120 else 0
        bkt["2-6h"]+=1 if 120<=d<360 else 0
        bkt[">6h"]+=1 if d>=360 else 0
    print("spacing buckets:",dict(bkt))

# 2. Price-outcome classification across ALL
outcome=Counter()
thr_used=[]
def classify(idx):
    tgt=min(idx+6,len(snaps)-1)
    cp,tp=snaps[idx].get("btc",0),snaps[tgt].get("btc",0)
    if tgt<=idx or not cp or not tp: return "NO-FUTURE/NO-BTC",None
    pct=(tp-cp)/cp
    try:
        a=datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
        b=datetime.fromisoformat(snaps[tgt].get("ts","").replace("Z","+00:00"))
        dm=max(5,(b-a).total_seconds()/60.0)
    except Exception: dm=30
    thr=max(0.002,min(0.05,0.003*math.sqrt(dm/5)))
    if pct<=-thr: return "STRESS",thr
    if pct>=thr: return "CALM",thr
    return "FLAT(<thr)",thr

for idx in range(len(snaps)):
    lab,thr=classify(idx)
    outcome[lab]+=1
    if thr is not None: thr_used.append(thr)

total=len(snaps)
print("\n--- price-outcome classification (ALL snapshots) ---")
for k,v in sorted(outcome.items(),key=lambda x:-x[1]):
    print(f"  {k:<16} {v:>5}  {v/total*100:>5.1f}%")
labeled=outcome["STRESS"]+outcome["CALM"]
print(f"\nlabeled (STRESS+CALM): {labeled}/{total} = {labeled/total*100:.1f}%")
print(f"unlabeled (None/FLAT/NO-FUTURE): {total-labeled} = {(total-labeled)/total*100:.1f}%")

if thr_used:
    import statistics
    print(f"\nadaptive threshold used: median={statistics.median(thr_used)*100:.2f}% min={min(thr_used)*100:.2f}% max={max(thr_used)*100:.2f}%")

# 3. How many bins would have >=3 price samples for blending?
print("\n--- per-bin price sample availability ---")
n=10; bins=[(i/n,(i+1)/n) for i in range(n)]
binp={f"{lo:.1f}-{hi:.1f}":0 for lo,hi in bins}
for idx,snap in enumerate(snaps):
    conf=snap.get("composite_confidence") or 0.5
    lab,_=classify(idx)
    if lab in ("STRESS","CALM"):
        for lo,hi in bins:
            if lo<=conf<hi: binp[f"{lo:.1f}-{hi:.1f}"]+=1; break
for k,v in sorted(binp.items()):
    print(f"  {k}: {v} price-labeled snapshots" + ("  <-- enough for blend(>=3)" if v>=3 else ""))
