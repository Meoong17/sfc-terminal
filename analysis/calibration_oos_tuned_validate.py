#!/usr/bin/env python3
"""OOS validation of the TUNED calibration (base_int=72, look=3, thr=0.005)
vs raw confidence, on held-out snapshots. Confirms the fix adds value."""
import json, math, subprocess
from datetime import datetime

SFC_DIR="/home/ubuntu/sfc"
BASE_INT=72; LOOK=3; BASE_THR=0.005

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

def price_outcome(snaps, idx):
    tgt=min(idx+LOOK,len(snaps)-1)
    if tgt<=idx: return None
    cp,tp=snaps[idx].get("btc",0),snaps[tgt].get("btc",0)
    if not cp or not tp: return None
    pct=(tp-cp)/cp
    try:
        a=datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
        b=datetime.fromisoformat(snaps[tgt].get("ts","").replace("Z","+00:00"))
        dm=max(BASE_INT,(b-a).total_seconds()/60.0)
    except Exception: dm=BASE_INT*LOOK
    thr=max(0.002,min(0.05,BASE_THR*math.sqrt(dm/BASE_INT)))
    if pct<=-thr: return True
    if pct>=thr: return False
    return None

def build_map(snaps):
    n=10; bins=[(i/n,(i+1)/n) for i in range(n)]
    bd={f"{lo:.1f}-{hi:.1f}":{"count":0,"ps":0,"pc":0,"ms":0} for lo,hi in bins}
    for idx,snap in enumerate(snaps):
        conf=snap.get("composite_confidence") or 0.5
        sfc=snap.get("sfc_effective") or 0
        sm=sfc>25.0
        po=price_outcome(snaps,idx)
        for lo,hi in bins:
            if lo<=conf<hi:
                k=f"{lo:.1f}-{hi:.1f}"; bd[k]["count"]+=1
                if sm: bd[k]["ms"]+=1
                if po is True: bd[k]["ps"]+=1
                elif po is False: bd[k]["pc"]+=1
                break
    curve=[]
    for label,d in sorted(bd.items()):
        lo,hi=float(label.split("-")[0]),float(label.split("-")[1]); mid=(lo+hi)/2
        count=d["count"]; mr=d["ms"]/count if count else mid
        pt=d["ps"]+d["pc"]; pr=d["ps"]/pt if pt>0 else None
        actual=(0.7*pr+0.3*mr) if (pr is not None and pt>=3) else mr
        curve.append({"bin":label,"count":count,"raw":round(mid,3),"actual":round(actual,3)})
    return curve

def apply_map(raw, curve):
    pts=sorted(curve,key=lambda c:c["raw"])
    if raw<=pts[0]["raw"]: return pts[0]["actual"]
    if raw>=pts[-1]["raw"]: return pts[-1]["actual"]
    for i in range(len(pts)-1):
        x1,y1=pts[i]["raw"],pts[i]["actual"]; x2,y2=pts[i+1]["raw"],pts[i+1]["actual"]
        if x1<=raw<=x2:
            if abs(x2-x1)<1e-10: return y1
            t=(raw-x1)/(x2-x1)
            return max(0.0,min(1.0,y1+t*(y2-y1)))
    return raw

def ece(pairs):
    preds=[p for p,l in pairs]; labels=[l for p,l in pairs]
    bins={i:{"n":0,"s":0,"psum":0} for i in range(10)}
    for p,l in zip(preds,labels):
        b=min(9,int(p*10)); bins[b]["n"]+=1; bins[b]["s"]+=l; bins[b]["psum"]+=p
    total=len(preds); e=0.0
    for b in bins.values():
        if b["n"]>0:
            actual=b["s"]/b["n"]; pred=b["psum"]/b["n"]
            e+=(b["n"]/total)*abs(actual-pred)
    return round(e,4) if total else None

def brier(preds,labels):
    return round(sum((p-l)**2 for p,l in zip(preds,labels))/len(preds),4) if preds else None

def monotonic(curve):
    pts=sorted([c for c in curve if c["count"]>0],key=lambda c:c["raw"])
    viol=[]
    for i in range(len(pts)-1):
        if pts[i]["actual"]>pts[i+1]["actual"]: viol.append((pts[i]["bin"],pts[i+1]["bin"]))
    return viol

snaps=extract()
print(f"total snapshots: {len(snaps)}")
split=int(len(snaps)*0.6)
train,test=snaps[:split],snaps[split:]
print(f"train {len(train)} ({train[0].get('ts','?')[:10]}..{train[-1].get('ts','?')[:10]}) | test {len(test)} ({test[0].get('ts','?')[:10]}..{test[-1].get('ts','?')[:10]})")

curve=build_map(train)
print("\nTRAIN curve (tuned params):")
print(f"{'bin':<10}{'cnt':>5}{'raw':>6}{'actual':>8}")
for c in curve:
    if c["count"]>0: print(f"{c['bin']:<10}{c['count']:>5}{c['raw']:>6.2f}{c['actual']:>8.3f}")

v=monotonic(curve)
print(f"\nmonotonicity violations (populated): {len(v)} {v}")

# test labels
test_raw=[]
for idx,snap in enumerate(test):
    conf=snap.get("composite_confidence") or 0.5
    po=price_outcome(test,idx)
    if po is None: continue
    test_raw.append((conf,1.0 if po else 0.0))

print(f"\ntest snapshots with price-outcome label: {len(test_raw)}")
if len(test_raw)>=10:
    cal_pairs=[(apply_map(p,curve),l) for p,l in test_raw]
    er=ece(test_raw); ec=ece(cal_pairs)
    br=brier([p for p,l in test_raw],[l for p,l in test_raw])
    bc=brier([p for p,l in cal_pairs],[l for p,l in cal_pairs])
    print(f"ECE raw={er} calibrated={ec}")
    print(f"Brier raw={br} calibrated={bc}")
    print("\nVERDICT:")
    print(f"  ECE improved? {ec<er}")
    print(f"  Brier improved? {bc<br}")
    if ec<er and bc<br: print("  -> TUNED calibration ADDS value OOS. FIX VALIDATED.")
    else: print("  -> Tuned calibration still not clearly better OOS.")
else:
    print("  insufficient labeled test samples for a robust verdict")
