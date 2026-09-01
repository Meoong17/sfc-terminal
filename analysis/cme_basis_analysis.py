#!/usr/bin/env python3
"""
cme_basis_analysis.py — institutional carry signal: CME BTC1! continuous futures
    basis (futures close vs spot close). Is it a valid, era-stable regime read
    and does it ADD to price/vol (or is it redundant, like funding)?
===========================================================
Questions:
  1. Basis level/character over time (institutional futures premium/carry).
  2. State-discrimination + era-stability of basis as a bull/bear & stress read.
  3. Incremental value over price/vol baseline (CV AUC) — the decision test.
Prior (honest): basis is a carry signal kin to funding, which was era-stable but
redundant (Δ=0 AUC). Expect the same unless CME adds unique info.
"""
import json, os, sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from historical_backtest_m1m6 import fetch_fred_series

def load_basis():
    df = pd.read_csv(os.path.join(ROOT, "data", "tradingview_btc1_daily.csv"))
    df["dt"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    cme = dict(zip(df["dt"], df["close"].values))
    raw = fetch_fred_series("CBBTCUSD", start_date="2017-01-01")
    spot = {d: v for d, v in raw.items() if d in cme}
    days = sorted(spot)
    basis = np.array([cme[d] / spot[d] - 1.0 for d in days])
    dates = np.array(days, dtype="datetime64[D]")
    px = np.array([spot[d] for d in days], dtype=float)
    return dates, basis, px

def cv_auc(X, y):
    if len(y) < 40 or y.sum() < 5 or (~y).sum() < 5:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y.astype(int),
                          cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

def main():
    dates, basis, px = load_basis()
    n = len(px)
    ret20 = np.full(n, np.nan); vol = np.full(n, np.nan)
    for i in range(20, n): ret20[i] = px[i]/px[i-20]-1.0
    for i in range(30, n): vol[i] = np.std(px[i-29:i+1]/px[i-30:i]-1.0)
    trend = np.zeros(n, bool)
    for i in range(200, n): trend[i] = px[i] >= np.median(px[i-200:i])
    stress = np.zeros(n, bool)
    for i in range(60, n): stress[i] = px[i]/np.max(px[i-60:i+1])-1.0 < -0.25

    valid = ~np.isnan(ret20) & ~np.isnan(vol)
    i = np.arange(n)[valid]
    print(f"CME basis: n={n}, {np.datetime64(dates.min(),'D')}..{np.datetime64(dates.max(),'D')}")
    print(f"  basis mean={basis.mean():+.4f} median={np.median(basis):+.4f} "
          f"min={basis.min():+.4f} max={basis.max():+.4f} (fraction of spot)")
    print(f"  basis>0 (futures contango): {(basis>0).mean()*100:.0f}% days")

    # state-discrimination + era-stability
    q = np.quantile(basis, [0.33, 0.67])
    print("\n[2] Basis tercile -> behaviour:")
    for name, arr in [("bull%", trend), ("stress%", stress)]:
        row = [f"{arr[basis < q[0]].mean()*100:.1f}%", f"{arr[(basis>=q[0])&(basis<q[1])].mean()*100:.1f}%", f"{arr[basis >= q[1]].mean()*100:.1f}%"]
        print(f"  {name}: low={row[0]} mid={row[1]} high={row[2]}")
    print("\n[3] Era-stability corr(basis, regime):")
    eras = {"2017-20": (dates >= np.datetime64("2017-12-01")) & (dates < np.datetime64("2020-01-01")),
            "2020-23": (dates >= np.datetime64("2020-01-01")) & (dates < np.datetime64("2023-01-01")),
            "2023-26": dates >= np.datetime64("2023-01-01")}
    for en, m in eras.items():
        if m.sum() < 40: continue
        r_t = np.corrcoef(basis[m], trend[m])[0, 1]
        r_s = np.corrcoef(basis[m], stress[m])[0, 1] if stress[m].std() > 0 else float("nan")
        print(f"  {en} n={m.sum():4d} corr(basis,bull)={r_t:+.3f} corr(basis,stress)={r_s:+.3f}")

    # incremental (decision test)
    print("\n[4] Incremental value over price/vol baseline (CV AUC):")
    Xb = np.column_stack([ret20[i], vol[i]])
    Xf = np.column_stack([ret20[i], vol[i], basis[i]])
    for name, y in [("BULL/BEAR", trend[i]), ("STRESS", stress[i])]:
        a_b = cv_auc(Xb, y); a_f = cv_auc(Xf, y); a_u = cv_auc(basis[i].reshape(-1,1), y)
        print(f"  {name}: baseline={a_b:.3f} +basis={a_f:.3f} (Δ={a_f-a_b:+.3f})  basis-uni={a_u:.3f}")

if __name__ == "__main__":
    main()
