#!/usr/bin/env python3
"""
vix_regime_validation.py — validate VIX / risk-appetite as the conditioning input
                       for SFC regime detection (the one evidence-backed upgrade)
====================================================================================
Across all tests, GLF / term-premium / ΔM2 / flow add ~0 AUC to regime detection
over price-behavior baseline. The one macro channel with robust power (per repo's
inflation-transmission study) is VIX / risk-appetite (partial-F p=0.02, negative,
contemporaneous + 1-month-lag predictive). This validates VIX under the SFC
regime-detection objective before any integration:

  A. REGIME CONDITIONING: reuse regime_conditioning_test.py framework, add VIX and a
     direct term-premium proxy (US30Y-US2Y) + REAL_Y10, and compare incremental
     regime-detection AUC vs GLF / M2-impulse / flow. Does VIX add meaningfully (>~0.03)?
  B. FORWARD PREDICTIVE (reconfirm with current data): expanding walk-forward OOS
     of VIX -> forward BTC return (1-month), vs AR baseline. The repo found VIX
     robust; re-verify on 2017-2026.

Data: VIX, US30Y, US2Y, REAL_Y10 from data/cleaned/macro_daily_clean.json (2017+).
USAGE: cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/vix_regime_validation.py
"""
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from historical_backtest_m1m6 import fetch_fred_series
from causal_liquidity_btc import build_monthly_glf
from behavior_reading_validity_test import monthly_to_daily
from liquidity_impulse_test import m2_impulse
# reuse regime-conditioning machinery
from regime_conditioning_test import (
    price_features, build_matrix, mann_whitney_auc, cohens_d, cv_auc,
)

OUTPUT = os.path.join(SFC_ROOT, ".vix_regime_validation.json")

VIX_JSON = os.path.join(SFC_ROOT, "data/cleaned/macro_daily_clean.json")


def load_macro_daily():
    with open(VIX_JSON) as f:
        data = json.load(f)
    out = {"vix": {}, "term_prem_spread": {}, "real_y10": {}}
    for r in data:
        d = r["date"]
        if r.get("VIX") is not None:
            out["vix"][d] = float(r["VIX"])
        if r.get("US30Y") is not None and r.get("US2Y") is not None:
            out["term_prem_spread"][d] = float(r["US30Y"]) - float(r["US2Y"])
        if r.get("REAL_Y10") is not None:
            out["real_y10"][d] = float(r["REAL_Y10"])
    return out


def walk_forward_vix_btc(btc_daily, vix, p=2, min_train=36, lag=1):
    """Expanding walk-forward: predict monthly BTC return from VIX known at t-lag
    plus AR(p). Reports OOS R2 vs AR and DM."""
    from scipy import stats as sps
    # monthly BTC returns
    closes = {}
    for d in sorted(btc_daily):
        closes[d[:7]] = btc_daily[d]
    months = sorted(closes)
    ret = {}
    for i in range(1, len(months)):
        p0, p1 = closes[months[i-1]], closes[months[i]]
        if p0 and p0 != 0:
            ret[months[i]] = (p1 - p0) / p0 * 100.0
    # monthly VIX (last of month)
    vix_m = {}
    for d in sorted(vix):
        vix_m[d[:7]] = vix[d]
    common = sorted(set(ret) & set(vix_m))
    if len(common) < min_train + 10:
        return {"error": f"n={len(common)}"}
    r = np.array([ret[m] for m in common], float)
    v = np.array([vix_m[m] for m in common], float)
    n = len(common)

    def _feat(t, mode):
        cols = [1.0]
        for k in range(1, p + 1):
            cols.append(r[t - k])
        if mode >= 1:
            cols.append(v[t - lag])
        return cols

    resid = {0: [], 1: []}
    for t in range(min_train, n):
        X, y = {}, {}
        for mode in (0, 1):
            X[mode] = np.array([_feat(j, mode) for j in range(min_train, t)])
            y[mode] = r[min_train:t]
        if X[0].shape[0] < 20:
            continue
        b = {m: np.linalg.lstsq(X[m], y[m], rcond=None)[0] for m in (0, 1)}
        for m in (0, 1):
            resid[m].append(r[t] - float(_feat(t, m) @ b[m]))
    if len(resid[1]) < 10:
        return {"error": f"oos n={len(resid[1])}"}
    e0, e1 = np.array(resid[0]), np.array(resid[1])
    mse0, mse1 = np.mean(e0**2), np.mean(e1**2)
    oos_r2 = 1 - mse1 / mse0
    d = e0**2 - e1**2
    dbar = d.mean()
    gamma0 = np.mean((d - dbar)**2)
    gamma1 = np.mean((d[:-1]-dbar)*(d[1:]-dbar))
    var = (gamma0 + 2*gamma1)/len(d) if len(d) > 1 else gamma0/len(d)
    stat = dbar/np.sqrt(var) if var > 0 else 0.0
    dm_p = float(2*(1 - sps.norm.cdf(abs(stat)))) if var > 0 else 1.0
    return {"n": n, "oos_n": len(e0), "oos_r2_vix_vs_ar": round(float(oos_r2), 4),
            "dm_p": round(dm_p, 4), "vix_improves": bool(oos_r2 > 0 and dm_p < 0.05),
            "mse_ar": round(float(mse0), 4), "mse_ar_vix": round(float(mse1), 4)}


def main():
    print("=" * 78)
    print("VIX / RISK-APPETITE REGIME VALIDATION (evidence-backed upgrade)")
    print("=" * 78)

    macro = load_macro_daily()
    print(f"\n[1] Macro daily: VIX={len(macro['vix'])}  "
          f"term_prem_spread={len(macro['term_prem_spread'])}  real_y10={len(macro['real_y10'])}")

    # build full factor set: existing + VIX + term_prem_spread + real_y10
    from behavior_reading_validity_test import load_factors
    f = load_factors()
    btc = fetch_fred_series("CBBTCUSD")
    glf = build_monthly_glf(full=True)
    m2 = fetch_fred_series("M2SL", start_date="2009-01-01")
    imp = m2_impulse(m2)
    li = {m: v["LI"] for m, v in imp.items() if "LI" in v}
    f["glf"] = monthly_to_daily(glf, sorted(btc))
    f["m2_impulse"] = monthly_to_daily(li, sorted(btc))
    f["vix"] = macro["vix"]
    f["term_prem_spread"] = macro["term_prem_spread"]
    f["real_y10"] = macro["real_y10"]
    print("    factors:", {k: len(v) for k, v in f.items()})

    px = price_features(btc)
    print("    price features:", len(px), "days")

    result = {"generated_at": datetime.now().isoformat()}

    for rname, labels in [("TREND (bull/bear)", {d: r["trend"] for d, r in px.items()}),
                          ("STRESS (high-drawdown)", {d: r["stress"] for d, r in px.items()})]:
        M = build_matrix(px, f, labels)
        print(f"\n{'='*78}\nREGIME: {rname}   (n={len(M)})\n{'='*78}")
        if len(M) < 60:
            print("    insufficient n")
            continue
        result[rname] = {"n": len(M)}
        rows = [M[d] for d in sorted(M)]
        y = np.array([r["y"] for r in rows])
        base_cols = ["ret20", "vol"]
        base_auc = cv_auc(rows, base_cols)
        result[rname]["baseline_auc"] = base_auc
        print("  [A] Incremental conditioning (5-fold CV AUC): baseline vs +conditioning")
        print(f"      {'var':<16}{'MW-AUC':<9}{'base+var':<10}{'delta'}")
        for name in ["vix", "term_prem_spread", "real_y10", "glf", "m2_impulse", "term_prem", "order_flow", "etf_flow"]:
            if name not in f or name not in M[list(M)[0]]:
                continue
            vals = np.array([r[name] for r in rows])
            g0 = vals[y == 0]; g1 = vals[y == 1]
            if len(g0) < 5 or len(g1) < 5:
                continue
            mw = mann_whitney_auc(g1, g0)
            auc = cv_auc(rows, base_cols + [name])
            delta = (auc - base_auc) if (auc is not None and base_auc is not None) else None
            result[rname][name] = {"mw_auc": round(mw, 3) if mw else None,
                                   "base_plus_var_auc": round(auc, 3) if auc else None,
                                   "delta": round(delta, 3) if delta is not None else None}
            print(f"      {name:<16}{(round(mw,3) if mw else '--'):<9}"
                  f"{(round(auc,3) if auc else '--'):<10}"
                  f"{('+' if delta is not None and delta>0 else '') + str(round(delta,3)) if delta is not None else '--'}")

    print("\n[B] Forward predictive (walk-forward OOS): VIX -> next-month BTC return")
    vix_d = macro["vix"]
    for lag in (1,):
        wf = walk_forward_vix_btc(btc, vix_d, p=2, lag=lag)
        result["vix_walk_forward"] = wf
        if "error" not in wf:
            print(f"    lag={lag}: OOS R2={wf['oos_r2_vix_vs_ar']}  DM p={wf['dm_p']}  "
                  f"{'IMPROVES' if wf['vix_improves'] else 'no edge'}  (n={wf['n']}, oos_n={wf['oos_n']})")

    with open(OUTPUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
