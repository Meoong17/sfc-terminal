#!/usr/bin/env python3
"""
funding_incremental_test.py — does BitMEX funding ADD incremental regime-separation
    over the price/vol baseline, or is it redundant (like macro was)?
PLUS cross-exchange check (BitMEX vs Binance funding, 2020+).
===========================================================
Two questions:
  1. Cross-exchange: is BitMEX funding representative (corr vs Binance funding)?
  2. Incremental value: baseline features (ret20, vol) vs baseline+funding for
     classifying bull/bear and stress. If funding adds ~0 AUC -> redundant with
     price/vol -> CONTEXT only (display). If it adds meaningful AUC -> earns a layer.
Same validation discipline as regime_conditioning_test.py.
"""
import json, os, sys
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from historical_backtest_m1m6 import fetch_fred_series

def load_btc_regimes():
    raw = fetch_fred_series('CBBTCUSD', start_date='2014-01-01')
    dates = sorted(raw)
    px = np.array([raw[d] for d in dates], dtype=float)
    n = len(px)
    ret20 = np.full(n, np.nan); vol = np.full(n, np.nan)
    for i in range(20, n):
        ret20[i] = px[i] / px[i-20] - 1.0
    for i in range(30, n):
        vol[i] = np.std(px[i-29:i+1] / px[i-30:i] - 1.0)
    # bull/bear via 200d trailing median of price (point-in-time)
    trend = np.zeros(n, dtype=bool)
    for i in range(200, n):
        trend[i] = px[i] >= np.median(px[i-200:i])
    # stress via 60d max-drawdown (label free of vol leakage)
    stress = np.zeros(n, dtype=bool)
    for i in range(60, n):
        stress[i] = px[i] / np.max(px[i-60:i+1]) - 1.0 < -0.25
    return dates, px, ret20, vol, trend, stress

def cv_auc(X, y):
    if len(y) < 40 or y.sum() < 5 or (~y).sum() < 5:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y.astype(int),
                          cv=5, method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)

def spearman(a, b):
    ar, br = a.argsort(), b.argsort()
    return np.corrcoef(ar, br)[0, 1]

def main():
    fund = json.load(open(os.path.join(ROOT, "data", "bitmex_funding_daily.json")))
    fund = {k: v["funding_mean"] for k, v in fund.items()}
    binf = json.load(open(os.path.join(ROOT, "data", "binance_vision_daily.json")))
    binf = {k: v.get("funding_mean") for k, v in binf.items() if v.get("funding_mean") is not None}
    dates, px, ret20, vol, trend, stress = load_btc_regimes()
    dmap = {d: i for i, d in enumerate(dates)}

    # 1. cross-exchange
    fs = sorted(k for k in fund if k in binf and k in dmap)
    fbm = np.array([fund[k] for k in fs]); fbn = np.array([binf[k] for k in fs])
    print(f"[1] Cross-exchange funding (BitMEX vs Binance, 2020+): n={len(fs)}")
    print(f"    spearman={spearman(fbm, fbn):+.3f}  pearson={np.corrcoef(fbm,fbn)[0,1]:+.3f}")
    print(f"    BitMEX sd={fbm.std():.5f} Binance sd={fbn.std():.5f}")

    # 2. incremental regime-separation vs baseline
    print("\n[2] Incremental value of funding over price/vol baseline (CV AUC):")
    common = [d for d in dates if d in fund and not np.isnan(vol[dmap[d]]) and not np.isnan(ret20[dmap[d]])]
    i = np.array([dmap[d] for d in common])
    Xb = np.column_stack([ret20[i], vol[i]])
    Xf = np.column_stack([ret20[i], vol[i], np.array([fund[d] for d in common])])
    fZ = np.array([fund[d] for d in common])
    for name, y in [("BULL/BEAR", trend[i]), ("STRESS", stress[i])]:
        a_b = cv_auc(Xb, y); a_f = cv_auc(Xf, y); a_u = cv_auc(fZ.reshape(-1,1), y)
        print(f"  {name}: baseline={a_b:.3f}  baseline+funding={a_f:.3f} (Δ={a_f-a_b:+.3f})  funding-uni={a_u:.3f}")

    # 3. era-stability of the INCREMENT (does funding help across eras?)
    print("\n[3] Increment per era (funding delta AUC):")
    darr = np.array(common, dtype="datetime64[D]")
    eras = {"2016-19": (darr >= np.datetime64("2016-05-01")) & (darr < np.datetime64("2019-01-01")),
            "2019-22": (darr >= np.datetime64("2019-01-01")) & (darr < np.datetime64("2022-01-01")),
            "2022-26": darr >= np.datetime64("2022-01-01")}
    for ename, m in eras.items():
        if m.sum() < 40:
            print(f"  {ename}: n={m.sum()} skip"); continue
        for name, y in [("BULL", trend[i][m]), ("STRESS", stress[i][m])]:
            if y.sum() < 3 or (~y).sum() < 3:
                print(f"  {ename} {name}: n={m.sum()} skip (class imbalance)"); continue
            a_b = cv_auc(Xb[m], y); a_f = cv_auc(Xf[m], y)
            print(f"  {ename} {name}: base={a_b:.3f} +fund={a_f:.3f} (Δ={a_f-a_b:+.3f})")

if __name__ == "__main__":
    main()
