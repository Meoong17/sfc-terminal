#!/usr/bin/env python3
"""Out-of-sample empirical validation of SFC confidence calibration.

Builds the calibration curve on a TRAIN temporal slice of git-history
snapshots, then applies it to a HELD-OUT TEST slice (the most recent
snapshots). Reports:
  - per-bin counts (train vs test)
  - raw-confidence ECE on test (does raw conf predict realized stress?)
  - calibrated-confidence ECE on test (does the map actually improve it?)
  - monotonicity check on the mapping
  - Brier skill of calibrated vs raw (lower = better)
If calibrated ECE/Brier is NOT better than raw on held-out data, the
map is not adding value and should not be trusted.
"""
import json, math, os, subprocess, sys
from datetime import datetime

SFC_DIR = "/home/ubuntu/sfc"

def extract_snapshots(max_count=800):
    res = subprocess.check_output(
        ["git", "log", "--oneline", "--all", "--diff-filter=M", "--reverse", "--", "data.json"],
        text=True, timeout=40, cwd=SFC_DIR).strip().split("\n")
    res = [r for r in res if r.strip()]
    step = max(1, len(res)//max_count) if len(res) > max_count else 1
    res = res[::step]
    snaps = []
    for line in res:
        sha = line.split()[0]
        try:
            c = subprocess.check_output(["git","show",f"{sha}:data.json"], text=True, timeout=10, cwd=SFC_DIR)
            if c.strip().startswith("{"):
                snaps.append(json.loads(c))
        except Exception:
            continue
    snaps.sort(key=lambda s: s.get("ts",""))
    return snaps

def price_outcome(snaps, idx, lookahead=6, base_interval=5, base_thr=0.003,
                  min_thr=0.002, max_thr=0.05):
    target = min(idx+lookahead, len(snaps)-1)
    if target <= idx: return None
    cp, tp = snaps[idx].get("btc",0), snaps[target].get("btc",0)
    if not cp or not tp: return None
    pct = (tp-cp)/cp
    dm = base_interval*lookahead
    try:
        a = datetime.fromisoformat(snaps[idx].get("ts","").replace("Z","+00:00"))
        b = datetime.fromisoformat(snaps[target].get("ts","").replace("Z","+00:00"))
        dm = max(base_interval,(b-a).total_seconds()/60.0)
    except Exception: pass
    thr = max(min_thr, min(max_thr, base_thr*math.sqrt(dm/base_interval)))
    if pct <= -thr: return True
    if pct >= thr: return False
    return None

def build_map(snaps):
    n=10
    bins=[(i/n,(i+1)/n) for i in range(n)]
    bd={f"{lo:.1f}-{hi:.1f}":{"count":0,"ps":0,"pc":0,"ms":0} for lo,hi in bins}
    for idx,snap in enumerate(snaps):
        conf=snap.get("composite_confidence") or 0.5
        sfc=snap.get("sfc_effective") or 0
        stress_model = sfc>25.0
        po=price_outcome(snaps,idx)
        for lo,hi in bins:
            if lo<=conf<hi:
                bd[f"{lo:.1f}-{hi:.1f}"]["count"]+=1
                if stress_model: bd[f"{lo:.1f}-{hi:.1f}"]["ms"]+=1
                if po is True: bd[f"{lo:.1f}-{hi:.1f}"]["ps"]+=1
                elif po is False: bd[f"{lo:.1f}-{hi:.1f}"]["pc"]+=1
                break
    curve=[]
    for label,d in sorted(bd.items()):
        lo,hi=float(label.split("-")[0]),float(label.split("-")[1])
        mid=(lo+hi)/2; count=d["count"]
        mr=d["ms"]/count if count else mid
        pt=d["ps"]+d["pc"]
        pr=d["ps"]/pt if pt>0 else None
        if pr is not None and pt>=3:
            actual=0.7*pr+0.3*mr
        else:
            actual=mr
        curve.append({"bin":label,"count":count,"raw":round(mid,3),"actual":round(actual,3)})
    return curve

def apply_map(raw, curve):
    # find bracket
    pts=sorted([c for c in curve], key=lambda c:c["raw"])
    if raw<=pts[0]["raw"]: return pts[0]["actual"]
    if raw>=pts[-1]["raw"]: return pts[-1]["actual"]
    for i in range(len(pts)-1):
        x1,y1=pts[i]["raw"],pts[i]["actual"]
        x2,y2=pts[i+1]["raw"],pts[i+1]["actual"]
        if x1<=raw<=x2:
            if abs(x2-x1)<1e-10: return y1
            t=(raw-x1)/(x2-x1)
            return max(0.0,min(1.0,y1+t*(y2-y1)))
    return raw

def monotonic(curve):
    pts=[c for c in curve if c["count"]>0]
    pts.sort(key=lambda c:c["raw"])
    viol=[]
    for i in range(len(pts)-1):
        if pts[i]["actual"]>pts[i+1]["actual"]:
            viol.append((pts[i]["bin"],pts[i]["actual"],pts[i+1]["bin"],pts[i+1]["actual"]))
    return viol

def brier(preds, labels):
    if not preds: return None
    return round(sum((p-l)**2 for p,l in zip(preds,labels))/len(preds),4)

def ece(conf_pairs):
    # ECE: bin by predicted confidence into 10, compare to realized label rate
    if not conf_pairs: return None
    preds=[p for p,l in conf_pairs]; labels=[l for p,l in conf_pairs]
    bins={i:{"n":0,"s":0,"psum":0} for i in range(10)}
    for p,l in zip(preds,labels):
        b=min(9,int(p*10)); bins[b]["n"]+=1; bins[b]["s"]+=l; bins[b]["psum"]+=p
    total=len(preds); e=0.0
    for b in bins.values():
        if b["n"]>0:
            actual=b["s"]/b["n"]; pred=b["psum"]/b["n"]
            e+=(b["n"]/total)*abs(actual-pred)
    return round(e,4)

def main():
    snaps=extract_snapshots()
    print(f"total snapshots: {len(snaps)}")
    if not snaps:
        print("NO SNAPSHOTS"); return
    # temporal split: oldest 60% train, newest 40% test
    split=int(len(snaps)*0.6)
    train,test=snaps[:split],snaps[split:]
    print(f"train: {len(train)} | test (held-out): {len(test)}")
    print(f"train ts: {train[0].get('ts','?')[:10]} .. {train[-1].get('ts','?')[:10]}")
    print(f"test  ts: {test[0].get('ts','?')[:10]} .. {test[-1].get('ts','?')[:10]}")

    curve_train=build_map(train)
    print("\n--- TRAIN calibration curve ---")
    print(f"{'bin':<10}{'cnt':>5}{'raw':>6}{'actual':>8}")
    for c in curve_train:
        if c["count"]>0:
            print(f"{c['bin']:<10}{c['count']:>5}{c['raw']:>6.2f}{c['actual']:>8.3f}")

    print("\n--- monotonicity (train, populated bins) ---")
    viol=monotonic(curve_train)
    print("NON-MONOTONIC VIOLATIONS:", len(viol))
    for v in viol:
        print(f"  {v[0]}={v[1]} then {v[2]}={v[3]}")

    # build labels for test using price-outcome ground truth (only where available)
    test_pairs=[]  # (pred, label)
    test_raw=[]
    for idx,snap in enumerate(test):
        conf=snap.get("composite_confidence") or 0.5
        po=price_outcome(test,idx)
        if po is None: continue
        label=1.0 if po else 0.0
        test_pairs.append((conf,label))
        test_raw.append((conf,label))

    print(f"\ntest snapshots with price-outcome label: {len(test_pairs)}")

    if test_pairs:
        # RAW performance on test
        ece_raw=ece(test_raw)
        # CALIBRATED performance on test (apply train map)
        cal_pairs=[]
        for p,l in test_raw:
            cp=apply_map(p,curve_train)
            cal_pairs.append((cp,l))
        ece_cal=ece(cal_pairs)
        # Brier
        br_raw=brier([p for p,l in test_raw],[l for p,l in test_raw])
        br_cal=brier([p for p,l in cal_pairs],[l for p,l in cal_pairs])

        print("\n=== OUT-OF-SAMPLE RESULTS (test, held-out) ===")
        print(f"ECE raw conf       : {ece_raw}")
        print(f"ECE calibrated     : {ece_cal}")
        print(f"Brier raw conf     : {br_raw}")
        print(f"Brier calibrated   : {br_cal}")
        improved = (ece_cal is not None and ece_raw is not None and ece_cal<ece_raw)
        bimproved= (br_cal is not None and br_raw is not None and br_cal<br_raw)
        print("\nVERDICT:")
        print(f"  ECE improved by calibration? {improved}")
        print(f"  Brier improved by calibration? {bimproved}")
        if improved and bimproved:
            print("  -> Calibration map ADDS value on held-out data.")
        else:
            print("  -> Calibration map does NOT reliably beat raw confidence out-of-sample. DO NOT TRUST/APPLY it.")
    else:
        print("No price-outcome labels available on test set — cannot validate empirically.")

if __name__=="__main__":
    main()
