#!/usr/bin/env python3
"""Diagnose why price-outcome labels are so sparse in git-history snapshots."""
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
print(f"total snapshots: {len(snaps)}")
# check timestamps spacing
print("\n--- timestamp spacing (first 8 transitions) ---")
for i in range(1,min(9,len(snaps))):
    try:
        a=datetime.fromisoformat(snaps[i-1].get("ts","").replace("Z","+00:00"))
        b=datetime.fromisoformat(snaps[i].get("ts","").replace("Z","+00:00"))
        print(f"  {snaps[i-1].get('ts','?')[:19]} -> {snaps[i].get('ts','?')[:19]}  delta={(b-a).total_seconds()/60:.1f} min")
    except Exception as e:
        print("  parse err", e)

# how many have btc?
has_btc=sum(1 for s in snaps if s.get("btc"))
no_ts=sum(1 for s in snaps if not s.get("ts"))
print(f"\nsnapshots with btc: {has_btc}/{len(snaps)}")
print(f"snapshots without ts: {no_ts}")

# For last N snapshots, compute price-outcome feasibility
print("\n--- price-outcome feasibility on last 60 snapshots ---")
n=min(60,len(snaps))
for idx in range(len(snaps)-n, len(snaps)):
    tgt=min(idx+6, len(snaps)-1)
    cp,tp=snaps[idx].get("btc",0),snaps[tgt].get("btc",0)
    if tgt<=idx or not cp or not tp:
        label="NO-FUTURE/NO-BTC"
    else:
        pct=(tp-cp)/cp
        try:
            a=datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
            b=datetime.fromisoformat(snaps[tgt].get("ts","").replace("Z","+00:00"))
            dm=max(5,(b-a).total_seconds()/60.0)
        except Exception: dm=30
        thr=max(0.002,min(0.05,0.003*math.sqrt(dm/5)))
        label=("STRESS" if pct<=-thr else "CALM" if pct>=thr else "FLAT(<thr)")
    ts=snaps[idx].get("ts","")[:19]
    print(f"  {ts}  btc={cp}  tgt+6={tp}  pct={((tp-cp)/cp*100 if cp and tp else 0):+.2f}%  -> {label}")
