#!/usr/bin/env python3
"""
funding_regime_analysis.py — Analyze the BitMEX XBTUSD funding regime 2016-2026
    as a crypto-native behaviour/positioning signal for SFC.
================================================================================
Now that funding has 10 years of history (bitmex_funding_daily.json, 2016+),
apply the SAME validity framework used to reject macro-liquidity:
  1. distribution / funding-regime definitions
  2. crisis-elevation (funding during known BTC crises)
  3. state-discrimination (does funding separate behaviour states?)
  4. era-stability (2016-19 / 2019-22 / 2022-26)
  5. lead/transition (does funding change BEFORE BTC regime flips?)
Research only — no SFC scoring impact.
"""
import json, os, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from historical_backtest_m1m6 import fetch_fred_series

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND = os.path.join(REPO, "data", "bitmex_funding_daily.json")

def load_funding():
    d = json.load(open(FUND))
    out = {}
    for date, rec in d.items():
        m = rec.get("funding_mean")
        if m is not None:
            out[date] = m
    return out

def load_btc():
    raw = fetch_fred_series("CBBTCUSD", start_date="2014-01-01")
    dates = sorted(raw); closes = np.array([raw[d] for d in dates])
    return dates, closes

def behavior_regimes(dates, closes):
    n = len(closes)
    vol = np.full(n, np.nan)
    for i in range(29, n):
        vol[i] = np.std(np.diff(np.log(closes[max(0, i - 29):i + 1]))) * np.sqrt(365)
    p90 = np.nanpercentile(vol, 90)
    out = {}
    for i in range(n):
        d = dates[i]
        trend = 1 if closes[i] >= np.mean(closes[max(0, i - 200):i + 1]) else 0
        stress = 1 if (not np.isnan(vol[i]) and vol[i] >= p90) else 0
        out[d] = {"trend": trend, "stress": stress, "vol": vol[i]}
    return out

CRISES = {  # label: (start, end)
    "COVID_2020": ("2020-03-01", "2020-04-30"),
    "Luna_2022": ("2022-05-01", "2022-06-30"),
    "FTX_2022": ("2022-11-01", "2022-11-30"),
    "Carry_2024": ("2024-08-01", "2024-08-15"),
}

def erax(d):
    y = int(d[:4])
    return 0 if y <= 2019 else (1 if y <= 2022 else 2)

def main():
    fund = load_funding()
    dates, closes = load_btc()
    reg = behavior_regimes(dates, closes)
    # align: funding date present in btc regime
    common = [d for d in sorted(fund) if d in reg and not np.isnan(reg[d]["vol"])]
    f = np.array([fund[d] for d in common])
    trend = np.array([reg[d]["trend"] for d in common])
    stress = np.array([reg[d]["stress"] for d in common])
    vol = np.array([reg[d]["vol"] for d in common])
    era = np.array([erax(d) for d in common])
    # forward 30d return for lead test
    ci = {d: i for i, d in enumerate(dates)}
    fwd30 = np.full(len(common), np.nan)
    for j, d in enumerate(common):
        i = ci[d]
        if i + 30 < len(dates):
            fwd30[j] = closes[i + 30] / closes[i] - 1

    print(f"Funding days={len(common)} range={common[0]}..{common[-1]}")
    print(f"  funding_mean={f.mean():+.5f} median={np.median(f):+.5f} "
          f"sd={f.std():+.5f} min={f.min():+.5f} max={f.max():+.5f}")

    # 1. funding-regime incidence
    z = (f - f.mean()) / f.std()
    hi = (z > 1).mean() * 100; lo = (z < -1).mean() * 100
    print(f"\n[1] Funding regime incidence (z>+1 long-crowding={hi:.1f}%, z<-1 short-crowding={lo:.1f}%)")

    # 2. crisis-elevation (funding mean in crisis vs 90d-prior control)
    print("\n[2] Crisis funding vs baseline (mean over window):")
    base = f.mean()
    for label, (s, e) in CRISES.items():
        m = np.array([fund[d] for d in common if s <= d <= e])
        ctrl = np.array([fund[d] for d in common if d >= "2016-05-01" and s[:4] in d[:4] and not (s <= d <= e)][-90:]) if False else None
        print(f"  {label:12s} n={len(m):4d} mean={m.mean():+.5f} (baseline={base:+.5f})  {'SIGNAL' if len(m) and abs(m.mean()-base) > 0.8*f.std() else ''}")

    # 3. state-discrimination: funding tercile vs behaviour
    print("\n[3] Funding tercile -> behaviour (mean per tercile):")
    q = np.quantile(f, [0.33, 0.67])
    lows = f < q[0]; mids = (f >= q[0]) & (f < q[1]); highs = f >= q[1]
    for name, lab, arr in [("trend", "bull%", trend), ("stress", "stress%", stress),
                           ("vol", "rvol", vol), ("fwd30", "30d ret", fwd30)]:
        ok = ~np.isnan(arr)
        row = []
        for sel in (lows & ok, mids & ok, highs & ok):
            a = arr[sel]
            row.append(f"{a.mean():+.4f}" if name == "fwd30" else f"{a.mean():.3f}")
        print(f"  {name:6s}: low={row[0]:>8} mid={row[1]:>8} high={row[2]:>8}")

    # 4. era-stability: correlation funding vs trend & stress per era
    print("\n[4] Era-stability (funding vs behaviour per era):")
    for e in range(3):
        m = era == e
        if m.sum() < 30:
            print(f"  era{e}: n={m.sum()} too small"); continue
        r_t = np.corrcoef(f[m], trend[m])[0, 1]
        r_s = np.corrcoef(f[m], stress[m])[0, 1] if stress[m].std() > 0 else float("nan")
        print(f"  era{e} ({np.datetime64(common[np.argmax(m)],'Y')}): n={m.sum():4d} "
              f"corr(funding,bull)={r_t:+.3f} corr(funding,stress)={r_s:+.3f}")

    # 5. lead/transition: pre-transition drift + cross-corr of Δfunding vs regime flip
    print("\n[5] Lead/transition:")
    for key in ["trend", "stress"]:
        dts = sorted(reg)
        flips = [(d, reg[d][key], reg[d2][key]) for d, d2 in zip(dts, dts[1:]) if reg[d][key] != reg[d2][key]]
        pre = []
        for (td, frm, to) in flips:
            seg = [fund[d] for d in common if td > d and (datetime.strptime(td, "%Y-%m-%d") - datetime.strptime(d, "%Y-%m-%d")).days <= 30]
            if seg:
                pre.append(np.mean(seg))
        if pre:
            print(f"  {key}: n_flip={len(flips)} pre30d funding mean={np.mean(pre):+.5f} vs baseline={f.mean():+.5f}")
    # cross-corr Δfunding[t-l] vs trend-flip[t]
    dts = sorted(fund)
    idx = {d: i for i, d in enumerate(dts)}
    flip = np.zeros(len(dts))
    rd = sorted(reg)
    for d, d2 in zip(rd, rd[1:]):
        if reg[d]["trend"] != reg[d2]["trend"] and d2 in idx:
            flip[idx[d2]] = 1
    F = np.array([fund[d] for d in dts]); dF = np.diff(F)
    best = []
    for l in range(0, 31):
        a = dF[:len(dF) - l]; b = flip[l:]
        m = ~np.isnan(a)
        if m.sum() > 30:
            best.append((l, np.corrcoef(a[m], b[m])[0, 1]))
    bl = max(best, key=lambda t: abs(t[1]))
    print(f"  strongest corr(dF[t-l], trend_flip[t]) = {bl[1]:+.3f} at lag {bl[0]} (all |r|<0.1 = no lead)")

from datetime import datetime
if __name__ == "__main__":
    main()
