#!/usr/bin/env python3
"""
regime_conditioning_test.py — does conditioning on liquidity/macro state IMPROVE
                              BTC regime detection? (SFC objective = regime detection)
====================================================================================
SFC = BTC behavior/regime detection system conditioned on liquidity & macro state.
This tests the conditioning claim directly:

  Define the TRUE behavior regime ex-post from price behavior:
    REGIME_A  trend     : bull vs bear (BTC vs 200DMA)
    REGIME_B  stress    : stress vs calm (realized-vol top/bottom tercile)

  Then for each conditioning variable V in {GLF, term_prem/M91, ΔM2 impulse,
  order_flow, etf_flow}:

  1. UNIVARIATE SEPARATION (does V carry state info?)
       AUC of V predicting the regime label, + Cohen's d between group means.
  2. INCREMENTAL CONDITIONING (does adding V to price-behavior features help?)
       5-fold CV logistic AUC:  baseline(price features)  vs  baseline + V
       If AUC rises meaningfully, conditioning on V improves regime detection.

This is a MEASUREMENT/discrimination test (contemporaneous, known labels), NOT a
forward-return forecast. In-sample separation is the correct bar for "does the
conditioning carry regime information."

USAGE:
    cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/regime_conditioning_test.py
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
from behavior_reading_validity_test import load_factors, monthly_to_daily
from liquidity_impulse_test import m2_impulse

OUTPUT = os.path.join(SFC_ROOT, ".regime_conditioning_test.json")


# --------------------------------------------------------------------------- #
def price_features(btc_daily):
    """Daily price-behavior features and true regime labels. STRESS is defined from
    MAX DRAWDOWN (60d), NOT realized-vol, so the baseline feature set can include
    realized-vol without leaking the label. Returns dict of aligned rows."""
    pdates = sorted(btc_daily)
    closes = np.array([btc_daily[d] for d in pdates])
    rets = np.zeros(len(pdates))
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    W = 30
    rows = []
    for i in range(W, len(pdates)):
        r = rets[i - W:i]
        vol = float(np.std(r) * np.sqrt(365))
        ret20 = float(np.prod(1 + rets[i - 20:i]) - 1) * 100.0
        # max drawdown over 60d window ending i
        lo = max(0, i - 60)
        win = closes[lo:i]
        peak = np.maximum.accumulate(win)
        maxdd = float((win - peak).min() / peak.max()) * 100.0 if peak.max() else 0.0
        worst20 = float(rets[i - 20:i].min()) * 100.0
        if i >= 200:
            ma = float(closes[i - 200:i].mean())
            trend = 1 if closes[i] >= ma else 0
        else:
            trend = None
        rows.append({"date": pdates[i], "ret20": ret20, "vol": vol, "maxdd": maxdd,
                     "worst20": worst20, "trend": trend})
    # stress label = deepest-drawdown tercile (negative maxdd)
    dd = np.array([r["maxdd"] for r in rows])
    dd_hi = np.percentile(dd, 66.7)   # most negative = stress
    dd_lo = np.percentile(dd, 33.3)
    for r in rows:
        r["stress"] = 1 if r["maxdd"] <= dd_hi else (0 if r["maxdd"] >= dd_lo else None)
    return {r["date"]: r for r in rows if r["trend"] is not None}


def build_matrix(px, factors, labels):
    """rows: {date, y=label, features{name:val}} for dates with all present."""
    out = {}
    for d in px:
        if d not in labels:
            continue
        if labels[d] is None:
            continue
        feat = {"ret20": px[d]["ret20"], "vol": px[d]["vol"]}
        ok = True
        for name, ser in factors.items():
            if d not in ser:
                ok = False
                break
            feat[name] = ser[d]
        if not ok:
            continue
        out[d] = {"y": labels[d], **feat}
    return out


def cohens_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2)) if na + nb > 2 else 1e-9
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def mann_whitney_auc(pos, neg):
    """Rank-based AUC = P(feature in pos group > feature in neg group). Robust, no
    model fit. <0.5 means the variable is inverted relative to the label encoding."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import mannwhitneyu
    try:
        u, _ = mannwhitneyu(pos, neg)
        return float(u / (len(pos) * len(neg)))
    except Exception:
        return None


def cv_auc(rows, cols, n_folds=5, seed=42):
    """5-fold CV logistic AUC for predicting y from the given feature columns."""
    X = np.array([[r[c] for c in cols] for r in rows])
    y = np.array([r["y"] for r in rows])
    if len(set(y)) < 2:
        return None
    idx = np.arange(len(rows))
    folds = np.array_split(idx, n_folds)
    scores, truth = [], []
    for fold in folds:
        te = fold
        tr = np.array([j for j in idx if j not in set(te)])
        if len(tr) < 10 or len(te) < 3 or len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        scores.extend(clf.predict_proba(X[te])[:, 1])
        truth.extend(y[te])
    if len(scores) < 10 or len(set(truth)) < 2:
        return None
    try:
        return float(roc_auc_score(truth, scores))
    except ValueError:
        return None


def main():
    print("=" * 78)
    print("REGIME-CONDITIONING TEST (does liquidity condition improve regime detection?)")
    print("=" * 78)

    print("\n[1] Loading conditioning variables + BTC...")
    f = load_factors()
    btc = fetch_fred_series("CBBTCUSD")
    glf = build_monthly_glf(full=True)
    m2 = fetch_fred_series("M2SL", start_date="2009-01-01")
    imp = m2_impulse(m2)
    li = {m: v["LI"] for m, v in imp.items() if "LI" in v}
    f["glf"] = monthly_to_daily(glf, sorted(btc))
    f["m2_impulse"] = monthly_to_daily(li, sorted(btc))
    print("    factors:", {k: len(v) for k, v in f.items()})

    px = price_features(btc)
    print("    price features:", len(px), "days")

    # two regime definitions
    labels_trend = {d: r["trend"] for d, r in px.items()}
    labels_stress = {d: r["stress"] for d, r in px.items()}

    result = {"generated_at": datetime.now().isoformat()}

    for rname, labels in [("TREND (bull/bear)", labels_trend), ("STRESS (high-vol)", labels_stress)]:
        M = build_matrix(px, f, labels)
        print(f"\n{'='*78}\nREGIME: {rname}   (n={len(M)})\n{'='*78}")
        if len(M) < 60:
            print("    insufficient n")
            continue
        result[rname] = {"n": len(M)}

        rows = [M[d] for d in sorted(M)]
        y = np.array([r["y"] for r in rows])

        # 1. univariate separation per conditioning variable (robust Mann-Whitney AUC)
        print("  [1] Univariate separation (MW-AUC / Cohen's d of V vs regime label):")
        base_cols = ["ret20", "vol"]
        result[rname]["univariate"] = {}
        result[rname]["baseline_auc"] = cv_auc(rows, base_cols)
        print(f"      {'var':<14}{'MW-AUC':<8}{'Cohen d':<10}{'n_pos':<8}{'n_neg'}")
        for name in f:
            vals = np.array([r[name] for r in rows])
            g0 = vals[y == 0]
            g1 = vals[y == 1]
            if len(g0) < 5 or len(g1) < 5:
                continue
            auc = mann_whitney_auc(g1, g0)
            d = cohens_d(g1, g0)
            result[rname]["univariate"][name] = {"auc": round(auc, 3) if auc else None,
                                                 "cohen_d": round(d, 3),
                                                 "n_pos": int(len(g1)), "n_neg": int(len(g0))}
            print(f"      {name:<14}{(round(auc,3) if auc else '--')}{(d if d is None else round(d,3)):<10}{len(g1):<8}{len(g0)}")
        print(f"      {'BASELINE(ret20+vol)':<14}{(result[rname]['baseline_auc'] if result[rname]['baseline_auc'] is not None else '--')}")

        # 2. incremental conditioning: baseline vs baseline + V
        print("  [2] Incremental conditioning (5-fold CV AUC): baseline vs +conditioning")
        result[rname]["incremental"] = {}
        for name in f:
            if name not in M[list(M)[0]]:
                continue
            cols = base_cols + [name]
            auc_cond = cv_auc(rows, cols)
            base_auc = result[rname]["baseline_auc"]
            delta = (auc_cond - base_auc) if (auc_cond is not None and base_auc is not None) else None
            result[rname]["incremental"][name] = {"auc": auc_cond, "delta_vs_baseline": round(delta, 3) if delta is not None else None}
            print(f"      +{name:<12} AUC={auc_cond}  delta={delta if delta is None else round(delta,3)}")

    with open(OUTPUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
