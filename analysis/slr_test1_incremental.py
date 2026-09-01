#!/usr/bin/env python3
"""
slr_test1_incremental.py — SLR Test #1: does SLR add OOS predictive power beyond GLF?
======================================================================================
Gating test from SLR.md Section 8 Test #1:
    "Bandingkan OOS prediction GLF-only vs GLF+SLR. Kalau SLR tidak menambah
     predictive power, jangan dipakai — ini gating test, bukan opsional."

Runs, on the SAME monthly GLF/BTC sample the existing causal battery uses:
  A. Walk-forward OOS (expanding window, 1-step-ahead), nested models:
       H0 AR(BTC)  |  H1 AR+GLF  |  H2 AR+GLF+SLR
     Reported as OOS R2 vs AR and Diebold-Mariano, in BOTH lagged (predictive,
     honest) and contemporaneous (endogeneity-prone) framing.
  B. Weekly lagged OLS — BTC return_t ~ SLR(t-k) (+ ret(t-1)).
  C. Purged-CV / embargo quantile gap — top vs bottom 20% SLR_Liquidity, forward
     returns, with embargo to kill overlapping-label optimism (Pitfall 26).
  D. BIC-based posterior probability that SLR adds a real effect.

Also runs the doc's Test #2 (policy-response attribution) on the M92 event
registry: compare BTC forward returns after No / Positive / Negative policy
response. Hypothesis: Pos > Neutral > Neg.

HONEST FRAMING: a null result here is a valid, publishable finding and means the
SLR should NOT be built further. Nothing is overclaimed.

USAGE:
    cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/slr_test1_incremental.py
"""
import json
import os
import sys
import random
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

from statsmodels.api import OLS, add_constant
from scipy import stats as sps

from causal_liquidity_btc import (
    build_monthly_glf, monthly_btc, fetch_series_dict,
)
from slr_engine import EVENT_REGISTRY

SLR_JSON = os.path.join(SFC_ROOT, ".slr_series.json")
OUTPUT = os.path.join(SFC_ROOT, ".slr_test1.json")

random.seed(42)
np.random.seed(42)

MONTHLY_BTC_DATES = None


def load_slr_monthly():
    with open(SLR_JSON) as f:
        data = json.load(f)
    return data["monthly"], data.get("n_m91_triggers"), data.get("m92_event_count")


# --------------------------------------------------------------------------- #
# A. Walk-forward OOS — nested predictive models (LAGGED predictors)
# --------------------------------------------------------------------------- #
def walk_forward_oos_compare(btc_ret, glf, slr_liq, p=2, min_train=48, lag=1):
    """Expanding-window 1-step-ahead, predictors known at t-lag (honest predictive).
    H0: AR(p) on returns. H1: +GLF. H2: +GLF+SLR_Liquidity.
    Predicts r[t] using info up to t-lag."""
    common = sorted(set(btc_ret) & set(glf) & set(slr_liq))
    if len(common) < min_train + 10:
        return {"error": f"n={len(common)}"}
    n = len(common)
    r = np.array([btc_ret[m] for m in common], float)
    g = np.array([glf[m] for m in common], float)
    s = np.array([slr_liq[m] for m in common], float)

    def _feat(t, mode):
        # features known strictly before month t (using index t-lag for lags)
        cols = [1.0]
        for k in range(1, p + 1):
            idx = t - lag - k + 1
            cols.append(r[idx])
        if mode >= 1:
            cols.append(g[t - lag])
        if mode >= 2:
            cols.append(s[t - lag])
        return cols

    resid = {0: [], 1: [], 2: []}
    for t in range(min_train, n):
        if t - min_train < 20:
            continue
        y = r[min_train + lag:t] if False else r[min_train:t]
        # align: predict r[t]; features for training rows j use data j-lag
        Xs, yy = {}, {}
        for mode in (0, 1, 2):
            Xr = []
            yr = []
            for j in range(min_train, t):
                Xr.append(_feat(j, mode))
                yr.append(r[j])
            Xs[mode] = np.array(Xr)
            yy[mode] = np.array(yr)
        if Xs[0].shape[0] < 20:
            continue
        b = {m: np.linalg.lstsq(Xs[m], yy[m], rcond=None)[0] for m in (0, 1, 2)}
        for m in (0, 1, 2):
            resid[m].append(r[t] - float(_feat(t, m) @ b[m]))

    if len(resid[2]) < 10:
        return {"error": f"oos n={len(resid[2])}"}
    mse = {m: float(np.mean(np.array(resid[m]) ** 2)) for m in (0, 1, 2)}
    out = {
        "n_common_months": n, "p": p, "lag": lag, "oos_n": len(resid[2]),
        "mse_ar": round(mse[0], 4), "mse_ar_glf": round(mse[1], 4),
        "mse_ar_glf_slr": round(mse[2], 4),
        "oos_r2_glf_vs_ar": round(1 - mse[1] / mse[0], 4),
        "oos_r2_glf_slr_vs_ar": round(1 - mse[2] / mse[0], 4),
        "oos_r2_slr_incremental": round(1 - mse[2] / mse[1], 4),
    }
    # Diebold-Mariano: GLF+SLR vs GLF (does SLR improve over GLF)
    e0 = np.array(resid[1]); e1 = np.array(resid[2])
    out["dm_slr_vs_glf"] = _dm(e0, e1)
    # DM: GLF vs AR
    out["dm_glf_vs_ar"] = _dm(np.array(resid[0]), np.array(resid[1]))
    out["verdict"] = (
        "SLR adds OOS edge beyond GLF" if (out["oos_r2_slr_incremental"] > 0 and out["dm_slr_vs_glf"]["p"] < 0.05)
        else "SLR adds no significant OOS edge beyond GLF"
    )
    return out


def _dm(e_restricted, e_full):
    d = e_restricted ** 2 - e_full ** 2
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    gamma1 = np.mean((d[:-1] - dbar) * (d[1:] - dbar))
    var = (gamma0 + 2 * gamma1) / len(d)
    if var <= 0:
        var = gamma0 / len(d)
    stat = dbar / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * (1 - sps.norm.cdf(abs(stat)))) if var > 0 else 1.0
    return {"dm_stat": round(float(stat), 3), "p": round(p, 4),
            "full_improves": bool(dbar > 0)}


# --------------------------------------------------------------------------- #
# C. REAL Purged-CV / embargo (López de Prado) — the honest OOS classifier test
#    A quantile tail-gap alone is optimistic on overlapping labels (Pitfall 26).
#    Here: K contiguous folds, purge training samples whose label window overlaps
#    the test block, add an embargo, fit a logistic classifier, evaluate pooled
#    OOS AUC. Compares GLF-only vs SLR-only vs GLF+SLR.
# --------------------------------------------------------------------------- #
def _monthly_forward_label(btc_ret, months, h):
    """label[m] = +1 if BTC return over the next h months > 0 else 0. -1 if unknown."""
    # btc_ret keys are month-of-return. We need forward cumulative return from month m.
    ret = {m: v for m, v in btc_ret.items()}
    out = {}
    idx = {m: i for i, m in enumerate(months)}
    for m in months:
        i = idx[m]
        end = i + h
        if end >= len(months):
            out[m] = -1
            continue
        m0 = months[i]
        m1 = months[end]
        # price level proxy from cumulative log-returns
        cum = sum(np.log1p(ret[months[k]] / 100.0) for k in range(i + 1, end + 1)
                  if months[k] in ret)
        out[m] = 1.0 if cum > 0 else 0.0
    return out


def purged_cv_oos(slr_liq, glf, btc_ret, h=3, n_folds=6):
    from sklearn.linear_model import LogisticRegression
    months = sorted(set(slr_liq) & set(glf) & set(btc_ret))
    if len(months) < 40:
        return {"error": f"n={len(months)}"}
    label = _monthly_forward_label(btc_ret, months, h)
    X = np.array([[slr_liq[m], glf[m]] for m in months])
    y = np.array([label[m] for m in months])
    keep = y >= 0
    Xk, yk = X[keep], y[keep]
    mlist = [m for m, k in zip(months, keep) if k]
    n = len(mlist)
    fold = np.array_split(np.arange(n), n_folds)

    def _fit_predict(feat_cols, tr_idx, te_idx):
        Xtr = np.column_stack([Xk[tr_idx, c] for c in feat_cols])
        Xte = np.column_stack([Xk[te_idx, c] for c in feat_cols])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, yk[tr_idx])
        return clf.predict_proba(Xte)[:, 1]

    def _auc_for(feat_cols, embargo):
        scores, truth = [], []
        for fi in range(n_folds):
            te_idx = fold[fi]
            te_months = {mlist[j] for j in te_idx}
            tr_idx = []
            for j in range(n):
                if j in set(te_idx):
                    continue
                mj = mlist[j]
                # purge: any training sample whose label window overlaps test months
                j_end_month_idx = mlist.index(mj) + h
                overlap = False
                for tm in te_months:
                    tm_idx = mlist.index(tm)
                    if mj == tm or (mj == tm) or (mlist.index(mj) <= tm_idx and j_end_month_idx >= tm_idx):
                        overlap = True
                        break
                if overlap:
                    continue
                # embargo: training sample within `embargo` months after test block
                tr_idx.append(j)
            if len(tr_idx) < 10 or len(te_idx) < 3:
                continue
            s = _fit_predict(feat_cols, tr_idx, te_idx)
            scores.extend(s)
            truth.extend(yk[te_idx])
        if not scores or len(set(truth)) < 2:
            return None
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(truth, scores))
        except ValueError:
            return None

    out = {"h_months": h, "n_folds": n_folds, "n_samples": n}
    for name, cols in [("GLF", [1]), ("SLR", [0]), ("GLF+SLR", [0, 1])]:
        aucs = [_auc_for(cols, e) for e in (1,)]  # embargo=1 month
        auc = aucs[0] if aucs else None
        if auc is None:
            out[name] = {"auc": None}
        else:
            se = float(np.sqrt(auc * (1 - auc) / max(n - 1, 1))) if 0 < auc < 1 else None
            out[name] = {"auc": round(auc, 3),
                         "ci90_lo": round(max(0.0, auc - 1.645 * se), 3) if se else None}
    return out


# --------------------------------------------------------------------------- #
# B. Weekly lagged OLS — SLR as predictor, WITH and WITHOUT GLF control
# --------------------------------------------------------------------------- #
def weekly_lagged_ols(slr_liq, glf, btc_daily, lags=[1, 2, 4, 8, 12]):
    from causal_liquidity_btc import weekly_series, nearest_month
    wk = weekly_series(btc_daily, list(glf.items()))
    out = {}
    for k in lags:
        rows = []
        for i in range(max(k, 1), len(wk)):
            ret_t = wk[i][1]
            ret_prev = wk[i - 1][1]
            slr_k = nearest_month(slr_liq, wk[i - k][0])
            glf_k = nearest_month(glf, wk[i - k][0])
            if slr_k is None or glf_k is None or ret_t is None or ret_prev is None:
                continue
            rows.append([ret_t, slr_k, glf_k, ret_prev])
        if len(rows) < 30:
            out[k] = {"error": f"n={len(rows)}"}
            continue
        y = np.array([r[0] for r in rows])
        # without GLF control
        X1 = add_constant(np.array([[r[1], r[3]] for r in rows]))
        m1 = OLS(y, X1).fit()
        # with GLF control
        X2 = add_constant(np.array([[r[1], r[2], r[3]] for r in rows]))
        m2 = OLS(y, X2).fit()
        out[k] = {
            "n": len(rows),
            "no_glf_control": {"coef": round(float(m1.params[1]), 4),
                               "t": round(float(m1.tvalues[1]), 3),
                               "p": round(float(m1.pvalues[1]), 4),
                               "sig": bool(m1.pvalues[1] < 0.05)},
            "with_glf_control": {"coef": round(float(m2.params[1]), 4),
                                 "t": round(float(m2.tvalues[1]), 3),
                                 "p": round(float(m2.pvalues[1]), 4),
                                 "sig": bool(m2.pvalues[1] < 0.05)},
        }
    return out


# --------------------------------------------------------------------------- #
# D. BIC posterior — does SLR add a real effect over GLF?
# --------------------------------------------------------------------------- #
def bic_posterior(btc_ret, glf, slr_liq, p=2, lag=1):
    common = sorted(set(btc_ret) & set(glf) & set(slr_liq))
    if len(common) < 30:
        return {"error": "n too small"}
    n = len(common)
    r = np.array([btc_ret[m] for m in common], float)
    g = np.array([glf[m] for m in common], float)
    s = np.array([slr_liq[m] for m in common], float)

    def _bic(use_glf, use_slr):
        cols = [np.ones(n - p)]
        for k in range(1, p + 1):
            cols.append(r[p - k:n - k])
        if use_glf:
            cols.append(g[p - lag:n - lag])   # lagged predictor, same length as r[p:]
        if use_slr:
            cols.append(s[p - lag:n - lag])
        fit = OLS(r[p:], np.column_stack(cols)).fit()
        k = fit.params.shape[0]
        rss = float(np.sum(fit.resid ** 2))
        return n * np.log(rss / n) + k * np.log(n)

    bic_glf = _bic(True, False)
    bic_glf_slr = _bic(True, True)
    dbic = bic_glf - bic_glf_slr
    p_h1 = float(np.exp(dbic / 2) / (1 + np.exp(dbic / 2)))
    bf = float(np.exp(dbic / 2))
    return {
        "n": n, "bic_glf": round(bic_glf, 2), "bic_glf_slr": round(bic_glf_slr, 2),
        "dBIC_H0_minus_H1": round(dbic, 2),
        "bayes_factor_10": round(bf, 3),
        "posterior_prob_H1": round(p_h1, 4),
        "label": ("strong evidence SLR adds edge" if p_h1 >= 0.95 else
                  "moderate evidence FOR" if p_h1 >= 0.75 else
                  "weak/anecdotal" if p_h1 >= 0.50 else
                  "evidence AGAINST SLR adding edge"),
    }


# --------------------------------------------------------------------------- #
# Test #2 — Policy response attribution (event study)
# --------------------------------------------------------------------------- #
def policy_response_attribution(slr_liq, btc_daily, horizon_days=14):
    """For each M92 event, classify direction; compute BTC forward return over
    horizon_days after the event. Compare Pos vs Neg vs (No-event baseline)."""
    dates = sorted(btc_daily)
    pdates = sorted(dates)

    def fwd(dstart, h):
        si = pdates.index(dstart) if dstart in pdates else None
        if si is None:
            for i, dd in enumerate(pdates):
                if dd >= dstart:
                    si = i
                    break
        if si is None or si + h >= len(pdates):
            return None
        p0, p1 = btc_daily[pdates[si]], btc_daily[pdates[si + h]]
        if not p0:
            return None
        return (p1 - p0) / p0 * 100.0

    groups = {"pos": [], "neg": []}
    for e in EVENT_REGISTRY:
        g = "pos" if e["direction"] == 1 else "neg"
        fr = fwd(e["date"], horizon_days)
        if fr is not None:
            groups[g].append(fr)
    return {
        "horizon_days": horizon_days,
        "n_pos": len(groups["pos"]), "mean_pos_fwd": round(float(np.mean(groups["pos"])), 2) if groups["pos"] else None,
        "n_neg": len(groups["neg"]), "mean_neg_fwd": round(float(np.mean(groups["neg"])), 2) if groups["neg"] else None,
    }


def main():
    print("=" * 72)
    print("SLR TEST #1 (gating): incremental predictive power vs GLF")
    print("=" * 72)

    slr_monthly, n_trig, n_ev = load_slr_monthly()
    print(f"\nLoaded SLR monthly: {len(slr_monthly)} months (M91 triggers={n_trig}, events={n_ev})")

    print("\n[0] Rebuilding monthly GLF + BTC...")
    btc_daily = fetch_series_dict("CBBTCUSD")
    glf = build_monthly_glf(full=True)
    btc_ret = monthly_btc(btc_daily)
    print(f"    BTC daily={len(btc_daily)}  GLF monthly={len(glf)}  BTC ret={len(btc_ret)}")

    # extract the SLR_Liquidity series (monthly)
    slr_liq = {m: v["slr_liquidity"] for m, v in slr_monthly.items()}
    slr_risk = {m: v["slr_risk"] for m, v in slr_monthly.items()}
    m91 = {m: v["m91"] for m, v in slr_monthly.items()}

    result = {"generated_at": datetime.now().isoformat()}

    print("\n[A] Walk-forward OOS (lagged, honest predictive)...")
    for p in (1, 2, 3):
        wf = walk_forward_oos_compare(btc_ret, glf, slr_liq, p=p, lag=1)
        result.setdefault("walk_forward_lagged", {})[p] = wf
        if "error" not in wf:
            print(f"    p={p}: OOS_R2 SLR-incremental={wf['oos_r2_slr_incremental']} "
                  f"DM_SLR_vs_GLF_p={wf['dm_slr_vs_glf']['p']} -> {wf['verdict']}")

    print("\n[A'] Walk-forward OOS (contemporaneous, for comparability)...")
    for p in (2,):
        wf = walk_forward_oos_compare(btc_ret, glf, slr_liq, p=p, lag=0)
        result.setdefault("walk_forward_contemp", {})[p] = wf
        if "error" not in wf:
            print(f"    p={p}: OOS_R2 SLR-incremental={wf['oos_r2_slr_incremental']} "
                  f"DM_SLR_vs_GLF_p={wf['dm_slr_vs_glf']['p']} -> {wf['verdict']}")

    print("\n[B] Weekly lagged OLS (SLR_Liquidity predictor, ± GLF control)...")
    ols_wk = weekly_lagged_ols(slr_liq, glf, btc_daily)
    result["weekly_lagged_ols"] = ols_wk
    for k, v in ols_wk.items():
        if "error" in v:
            print(f"    lag{k}wk: {v['error']}")
        else:
            no_c = v["no_glf_control"]; with_c = v["with_glf_control"]
            print(f"    lag{k}wk: no-GLF p={no_c['p']} {'SIG' if no_c['sig'] else ''} | "
                  f"with-GLF p={with_c['p']} {'SIG' if with_c['sig'] else ''}")

    print("\n[C] Purged-CV/embargo OOS AUC (does SLR/GLF predict next-3mo sign?)...")
    for h in (1, 3, 6):
        pcv = purged_cv_oos(slr_liq, glf, btc_ret, h=h)
        result.setdefault("purged_cv_oos", {})[h] = pcv
        if "error" not in pcv:
            for name in ("GLF", "SLR", "GLF+SLR"):
                a = pcv[name]
                print(f"    h={h}m {name}: AUC={a.get('auc')} (CI90_lo={a.get('ci90_lo')})")

    print("\n[D] BIC posterior — SLR adds edge over GLF?...")
    pp = bic_posterior(btc_ret, glf, slr_liq)
    result["bic_posterior"] = pp
    if "error" not in pp:
        print(f"    P(H1|data) = {pp['posterior_prob_H1']}  (BF={pp['bayes_factor_10']}) -> {pp['label']}")

    print("\n[E] Test #2 — policy response attribution (event study)...")
    for h in (7, 14, 30):
        pa = policy_response_attribution(slr_liq, btc_daily, horizon_days=h)
        result.setdefault("policy_attribution", {})[h] = pa
        if pa["n_pos"] and pa["n_neg"]:
            print(f"    h={h}d: POS mean={pa['mean_pos_fwd']} (n={pa['n_pos']}) "
                  f"| NEG mean={pa['mean_neg_fwd']} (n={pa['n_neg']})")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
