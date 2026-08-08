#!/usr/bin/env python3
"""
audit_factor_predictive.py — audit which SFC factor INPUTS actually predict forward
BTC returns out-of-sample. For each input that drives a weighted factor, test with
BOTH era-split IC and purged-CV OOS AUC (López de Prado, embargo=h) on the canonical
Binance 9-year series.

The goal: find "large-weight but no-impact" features (e.g. St weight 1.34 but ~0
effective contribution) and confirm empirically whether each input carries real
out-of-sample predictive information, or is noise to drop.

Pure analysis. No production change.
"""
import os, sys, math, json
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

HORIZONS = [7, 30, 90]
ERA_CUT = "2022-01-01"  # era1 before, era2 after


def spearman_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 30:
        return None
    xv, yv = x[m], y[m]
    rx = np.argsort(np.argsort(xv)).astype(float)
    ry = np.argsort(np.argsort(yv)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def era_ic(days, x, fwd, h):
    """Point IC per era for a given horizon (aligns fwd to days)."""
    m = ~(np.isnan(x) | np.isnan(fwd))
    out = {}
    for ename, cond in (("era1", days < ERA_CUT), ("era2", days >= ERA_CUT)):
        mm = m & cond
        if mm.sum() < 30:
            out[ename] = None
        else:
            out[ename] = spearman_ic(x[mm], fwd[mm])
    return out


def purged_auc(x, fwd, h, k=5):
    """Purged-CV OOS AUC of the input (via logistic on the single feature) for
    binary up/down over horizon h. Returns pooled AUC + per-fold list."""
    m = ~(np.isnan(x) | np.isnan(fwd))
    xx, yy = x[m], (fwd[m] > 0).astype(int)
    if len(xx) < 100 or len(np.unique(yy)) < 2:
        return None, []
    n = len(xx)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    fold_size = n // k
    per = []
    oos_p, oos_y = [], []
    for f in range(k):
        i0, i1 = f * fold_size, (n if f == k - 1 else (f + 1) * fold_size)
        test = list(range(i0, i1))
        train = [j for j in range(n) if not (i0 <= j < i1)]
        # purge label-overlap + embargo
        train = [j for j in train if not (j < i0 and j + h >= i0) and not (i1 <= j <= i1 + h)]
        if len(train) < 50 or len(np.unique(yy[test])) < 2:
            continue
        clf = LogisticRegression()
        clf.fit(xx[train].reshape(-1, 1), yy[train])
        p = clf.predict_proba(xx[test].reshape(-1, 1))[:, 1]
        per.append(roc_auc_score(yy[test], p))
        oos_p.append(p); oos_y.append(yy[test])
    if not per:
        return None, []
    pooled = roc_auc_score(np.concatenate(oos_y), np.concatenate(oos_p))
    return float(pooled), per


def verdict(pooled, per, era_res, h):
    if pooled is None or len(per) < 3:
        return "no-data"
    mean = np.mean(per); se = np.std(per) / np.sqrt(len(per))
    sig = (mean - 1.96 * se) > 0.5
    e1, e2 = era_res.get("era1"), era_res.get("era2")
    if e1 is None or e2 is None:
        era_cons = "n/a"
    else:
        era_cons = "cons" if (e1 > 0) == (e2 > 0) else "FLIP"
    if sig:
        return f"EDGE({pooled:.3f},era-{era_cons})"
    if pooled < 0.45:
        return f"REVERSED({pooled:.3f},era-{era_cons})"
    return f"none({pooled:.3f},era-{era_cons})"


def run():
    feat = compute_features(load_daily())
    days = np.array(feat["days"])
    canon_close = feat["close"]
    n = len(days)

    # Forward returns per horizon from canonical close
    fwd = {}
    for h in HORIZONS:
        fwd[h] = np.full(n, np.nan)
        fwd[h][:n - h] = canon_close[h:] / canon_close[:n - h] - 1.0

    # ---- Build input series aligned to canonical days ----
    inputs = OrderedDict()
    inputs["btc_momentum_7d (Lt)"] = feat["mom_7"]
    inputs["btc_momentum_30d (Lt)"] = feat["mom_30"]
    inputs["realized_vol_30d (Ft)"] = feat["rvol_30"]

    # FNG (Rt) + liquidity (Lt) from trend/imbs caches (2014-2026)
    trend = {p["date"]: p.get("fng") for p in json.load(open(".walk_forward_trend_continuation.json"))}
    imbs = {p["date"]: p for p in json.load(open(".walk_forward_imbs_l8.json"))}
    dstrs = [d[:10] for d in days]
    fng = np.array([trend.get(ds) for ds in dstrs], dtype=float)
    liq = np.array([imbs.get(ds, {}).get("liquidity_stress") for ds in dstrs], dtype=float)
    btc24 = np.array([imbs.get(ds, {}).get("btc_24h") for ds in dstrs], dtype=float)
    inputs["FNG (Rt, sentiment)"] = fng
    inputs["GLO liquidity_stress (Lt)"] = liq
    inputs["btc_24h raw (Lt)"] = btc24

    print(f"AUDIT — factor inputs vs forward BTC return (canonical, {n} days, "
          f"{days[0][:10]} -> {days[-1][:10]})\n")
    print(f"{'INPUT':32s} {'h':>3s} {'pooledAUC':>9s} {'per-fold':>16s} {'era1IC':>7s} {'era2IC':>7s}  VERDICT")
    for name, x in inputs.items():
        for h in HORIZONS:
            res = era_ic(days, x, fwd[h], h)
            pooled, per = purged_auc(x, fwd[h], h)
            v = verdict(pooled, per, res, h)
            pf = ",".join(f"{a:.2f}" for a in (per[:5] if per else [])
                          ) or "-"
            e1 = f"{res['era1']:+.2f}" if res["era1"] is not None else "-"
            e2 = f"{res['era2']:+.2f}" if res["era2"] is not None else "-"
            pu = f"{pooled:.3f}" if pooled is not None else "-"
            print(f"{name:32s} {h:>3d} {pu:>9s} {pf:>16s} {e1:>7s} {e2:>7s}  {v}")
        print()


if __name__ == "__main__":
    run()
