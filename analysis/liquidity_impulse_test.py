#!/usr/bin/env python3
"""
liquidity_impulse_test.py — 1.docx priority #1: does the Liquidity Impulse
                             (ΔM2) or Acceleration (ΔΔM2) predict BTC OUT-OF-SAMPLE?
====================================================================================
1.docx claims ΔM2 is the strongest component (Model D': β=+3.44, p=0.002). But that
is IN-SAMPLE. The repo's prior finding (Pitfall 32) is that every monthly macro model
has expanding-OOS R2<=0 (no skill vs naive), and m2_yoy was REMOVED from sfc_effective
in 2026-08. This test settles whether the IMPULSE/ACCELERATION formulation — which is
what 1.docx actually proposes — has any OOS predictive edge.

Design (honest, lagged predictors — no lookahead):
  L_t  = M2_t - M2_{t-1}                       (liquidity impulse, level change)
  LI_t = z-score of L_t over trailing 12m      (normalized impulse)
  LA_t = LI_t - LI_{t-1}                       (liquidity acceleration)

  Predict BTC forward return using LI/LA KNOWN at t-1 (lagged 1 month, respecting
  the ~1-month M2 release lag).

  A. Walk-forward OOS (expanding window), nested:
       H0  AR(p)
       H1  AR + LI_{t-1}
       H2  AR + LI_{t-1} + LA_{t-1}
     Reported as OOS R2 vs AR and Diebold-Mariano.
  B. Purged-CV/embargo OOS AUC: P(BTC_{t+1}>0 | LI, LA).
  C. Spearman IC + sign test: LI/LA vs forward returns.
  D. YoY-M2 (the GLF formulation) for comparison.

USAGE:
    cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/liquidity_impulse_test.py
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

from statsmodels.api import OLS, add_constant
from scipy import stats as sps

from historical_backtest_m1m6 import fetch_fred_series

OUTPUT = os.path.join(SFC_ROOT, ".liquidity_impulse_test.json")
M2 = "M2SL"
BTCCB = "CBBTCUSD"


# --------------------------------------------------------------------------- #
def monthly_close(btc_daily):
    """{month: last close of that month}."""
    closes = {}
    for d in sorted(btc_daily):
        closes[d[:7]] = btc_daily[d]
    return closes


def monthly_ret_from_closes(closes):
    months = sorted(closes)
    ret = {}
    for i in range(1, len(months)):
        p0, p1 = closes[months[i - 1]], closes[months[i]]
        if p0 and p0 != 0:
            ret[months[i]] = (p1 - p0) / p0 * 100.0
    return ret


def m2_impulse(m2):
    """{month 'YYYY-MM': (level, L=ΔM2, LI=z(ΔM2,trail12), LA=ΔLI)}."""
    months = sorted(m2)
    out = {}
    prev = None
    for d in months:
        m = d[:7]                      # normalize to 'YYYY-MM'
        if prev is None:
            prev = m2[d]
            continue
        L = m2[d] - prev
        prev = m2[d]
        out[m] = {"level": m2[d], "L": L}
    # LI = z of L over trailing 12 (needs 12 prior L values)
    mlist = sorted(out)
    for i, m in enumerate(mlist):
        if i < 12:
            continue
        window = [out[mlist[j]]["L"] for j in range(max(0, i - 12), i)]
        mu, sd = float(np.mean(window)), float(np.std(window))
        if sd <= 1e-9:
            continue
        out[m]["LI"] = (out[m]["L"] - mu) / sd
    # LA = ΔLI
    prev_li = None
    for m in mlist:
        if "LI" in out[m]:
            out[m]["LA"] = out[m]["LI"] - prev_li if prev_li is not None else 0.0
            prev_li = out[m]["LI"]
    return out


def m2_yoy(m2):
    months = sorted(m2)
    out = {}
    for i in range(12, len(months)):
        v0, v1 = m2[months[i - 12]], m2[months[i]]
        if v0 and v0 != 0:
            out[months[i][:7]] = (v1 - v0) / v0 * 100.0
    return out


def build_df(btc_ret, imp, yoy=None):
    """Aligned rows: predictor uses value known at t-1 (lag 1). 
    y_t = BTC return in month t; LI_{t-1}, LA_{t-1}, M2yoy_{t-1}."""
    months = sorted(btc_ret)
    rows = []
    for t, m in enumerate(months):
        if t == 0:
            continue
        pm = months[t - 1]
        if pm not in imp:
            continue
        li = imp[pm].get("LI")
        la = imp[pm].get("LA")
        yy = yoy.get(pm) if yoy else None
        if li is None or la is None:
            continue
        rows.append({"month": m, "ret": btc_ret[m], "LI": li, "LA": la,
                     "M2yoy": yy, "prev_ret": btc_ret[pm]})
    return rows


# --------------------------------------------------------------------------- #
# A. Walk-forward OOS (expanding), lagged predictors
# --------------------------------------------------------------------------- #
def walk_forward_oos(rows, p=2, min_train=36):
    """Predict ret_t from {LI, LA}_{t-1} + AR(p) of past returns.
    H0 AR | H1 +LI | H2 +LI+LA."""
    n = len(rows)
    if n < min_train + 10:
        return {"error": f"n={n}"}
    r = np.array([x["ret"] for x in rows])
    li = np.array([x["LI"] for x in rows])
    la = np.array([x["LA"] for x in rows])

    def _feat(t, mode):
        cols = [1.0]
        for k in range(1, p + 1):
            cols.append(r[t - k])
        if mode >= 1:
            cols.append(li[t])
        if mode >= 2:
            cols.append(la[t])
        return cols

    resid = {0: [], 1: [], 2: []}
    for t in range(min_train, n):
        X, y = {}, {}
        for mode in (0, 1, 2):
            X[mode] = np.array([_feat(j, mode) for j in range(min_train, t)])
            y[mode] = r[min_train:t]
        if X[0].shape[0] < 20:
            continue
        b = {m: np.linalg.lstsq(X[m], y[m], rcond=None)[0] for m in (0, 1, 2)}
        for m in (0, 1, 2):
            resid[m].append(r[t] - float(_feat(t, m) @ b[m]))

    if len(resid[2]) < 10:
        return {"error": f"oos n={len(resid[2])}"}
    mse = {m: float(np.mean(np.array(resid[m]) ** 2)) for m in (0, 1, 2)}
    out = {"n": n, "p": p, "oos_n": len(resid[2]),
           "mse_ar": round(mse[0], 4), "mse_ar_li": round(mse[1], 4),
           "mse_ar_li_la": round(mse[2], 4),
           "oos_r2_li": round(1 - mse[1] / mse[0], 4),
           "oos_r2_li_la": round(1 - mse[2] / mse[0], 4),
           "oos_r2_la_incremental": round(1 - mse[2] / mse[1], 4)}
    out["dm_li"] = _dm(np.array(resid[0]), np.array(resid[1]))
    out["dm_li_la"] = _dm(np.array(resid[0]), np.array(resid[2]))
    return out


def _dm(e0, e1):
    d = e0 ** 2 - e1 ** 2
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    gamma1 = np.mean((d[:-1] - dbar) * (d[1:] - dbar))
    var = (gamma0 + 2 * gamma1) / len(d)
    if var <= 0:
        var = gamma0 / len(d)
    stat = dbar / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * (1 - sps.norm.cdf(abs(stat)))) if var > 0 else 1.0
    return {"stat": round(float(stat), 3), "p": round(p, 4), "improves": bool(dbar > 0)}


# --------------------------------------------------------------------------- #
# B. Purged-CV / embargo OOS AUC for direction
# --------------------------------------------------------------------------- #
def purged_cv_auc(rows, feat_cols, h=1, n_folds=5):
    """Logistic P(return_t > 0 | features known at t-1), purged + embargo."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    r = np.array([x["ret"] for x in rows])
    n = len(rows)
    y = (r > 0).astype(float)
    X = np.array([[x[c] for c in feat_cols] for x in rows])
    folds = np.array_split(np.arange(n), n_folds)
    scores, truth = [], []
    for fold in folds:
        foldset = set(fold.tolist())
        tr = []
        for j in range(n):
            if j in foldset:
                continue
            # purge: training sample whose label window overlaps test (h=1 monthly
            # labels don't overlap across non-adjacent months; embargo 1 month)
            if any(abs(j - k) <= h for k in fold):
                continue
            tr.append(j)
        if len(tr) < 20 or len(fold) < 4:
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        scores.extend(clf.predict_proba(X[fold])[:, 1])
        truth.extend(y[fold])
    if len(scores) < 20 or len(set(truth)) < 2:
        return None
    auc = float(roc_auc_score(truth, scores))
    se = float(np.sqrt(auc * (1 - auc) / max(len(truth) - 1, 1))) if 0 < auc < 1 else None
    return {"auc": round(auc, 3),
            "ci90_lo": round(max(0.0, auc - 1.645 * se), 3) if se else None,
            "n": len(scores)}


# --------------------------------------------------------------------------- #
# C. IC / sign test
# --------------------------------------------------------------------------- #
def ic_sign(rows):
    r = np.array([x["ret"] for x in rows])
    res = {}
    for col in ("LI", "LA", "M2yoy"):
        vals = np.array([x[col] for x in rows])
        mask = ~np.isnan(vals.astype(float))
        if mask.sum() < 20:
            res[col] = {"error": "insufficient"}
            continue
        v, p = sps.spearmanr(vals[mask], r[mask])
        res[col] = {"spearman": round(float(v), 3), "p": round(float(p), 4),
                    "n": int(mask.sum())}
    return res


def main():
    print("=" * 74)
    print("LIQUIDITY IMPULSE TEST (1.docx priority #1): ΔM2 / ΔΔM2 OOS")
    print("=" * 74)

    print("\n[1] Fetching M2 (M2SL) + BTC...")
    m2 = fetch_fred_series(M2, start_date="2009-01-01")
    btc = fetch_fred_series(BTCCB, start_date="2010-01-01")
    print(f"    M2={len(m2)}  BTC={len(btc)}")

    closes = monthly_close(btc)
    btc_ret = monthly_ret_from_closes(closes)
    imp = m2_impulse(m2)
    yoy = m2_yoy(m2)
    print(f"    BTC monthly ret={len(btc_ret)}  impulse months={len(imp)}")

    rows = build_df(btc_ret, imp, yoy)
    print(f"    aligned rows (lagged 1) = {len(rows)}  ({rows[0]['month']}..{rows[-1]['month']})")

    result = {"generated_at": datetime.now().isoformat(),
              "period": (rows[0]["month"], rows[-1]["month"]) if rows else None}

    print("\n[A] Walk-forward OOS (lagged, expanding)...")
    for p in (1, 2, 3):
        wf = walk_forward_oos(rows, p=p)
        result.setdefault("walk_forward", {})[p] = wf
        if "error" not in wf:
            print(f"    p={p}: OOS_R2 LI={wf['oos_r2_li']} | LI+LA={wf['oos_r2_li_la']} "
                  f"| DM_LI_p={wf['dm_li']['p']} DM_LILA_p={wf['dm_li_la']['p']}")

    print("\n[B] Purged-CV OOS AUC — P(BTC_{t+1}>0 | features lagged 1)...")
    for name, cols in [("LI", ["LI"]), ("LI+LA", ["LI", "LA"]),
                       ("M2yoy", ["M2yoy"]), ("LI+LA+M2yoy", ["LI", "LA", "M2yoy"])]:
        auc = purged_cv_auc(rows, cols)
        result.setdefault("purged_cv", {})[name] = auc
        if auc:
            print(f"    {name:<14} AUC={auc['auc']} (CI90_lo={auc['ci90_lo']}) n={auc['n']}")
        else:
            print(f"    {name:<14} insufficient")

    print("\n[C] Spearman IC (LI/LA/M2yoy vs same-period BTC return)...")
    ic = ic_sign(rows)
    result["spearman_ic"] = ic
    for k, v in ic.items():
        if "error" in v:
            print(f"    {k:<8} {v['error']}")
        else:
            print(f"    {k:<8} spearman={v['spearman']} p={v['p']} n={v['n']}")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
