#!/usr/bin/env python3
"""
purged_cv_momentum.py — formal purged-CV (López de Prado) validation of momentum
as an out-of-sample predictor of BTC up/down over horizon h.

Why purged-CV: forward-return labels overlap between consecutive days. A naive
train/test split leaks label information across the boundary. Purging removes any
training sample whose label window overlaps the test set, plus an embargo, so the
reported OOS AUC is leakage-free (the standard PROJECT_STATUS now mandates).

Design:
  - Label: y_t = 1 if forward return over h days > 0 (binary up/down).
  - Feature: trailing momentum (mom_30 / mom_90).
  - Model: LogisticRegression (single feature) — its score is monotonic in momentum.
  - K contiguous folds; per fold: purge train samples whose [j, j+h] overlaps the
    test block [+ embargo], fit on train, predict OOS AUC on test.
  - Report per-fold AUC + pooled OOS AUC vs 0.5 (chance).

Pure analysis. No production change.
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

FEATURES = ["mom_30", "mom_90"]
HORIZONS = [7, 30, 90]
K = 5


def purged_folds(n, h, k=K):
    """Contiguous test blocks; purge train samples whose label window [j,j+h]
    overlaps the test block [i0,i1) plus embargo h after the block."""
    fold_size = n // k
    folds = []
    for f in range(k):
        i0 = f * fold_size
        i1 = n if f == k - 1 else (f + 1) * fold_size
        test = list(range(i0, i1))
        train = []
        for j in range(n):
            if i0 <= j < i1:
                continue  # test
            # purge if label window [j, j+h] overlaps test [i0,i1) or embargo [i1,i1+h]
            if j < i0 and j + h >= i0:
                continue
            if i1 <= j <= i1 + h:
                continue
            train.append(j)
        if len(train) < 50 or len(test) < 30:
            continue
        folds.append((np.array(train), np.array(test)))
    return folds


def run():
    feat = compute_features(load_daily())
    days = np.array(feat["days"])
    n = len(days)
    print(f"PURGED-CV (López de Prado) — momentum OOS up/down prediction, {K} folds "
          f"+ embargo=h. ({n} days, {days[0]} -> {days[-1]})")
    print("Label: y=1 if fwd return>0. Feature: trailing momentum. Metric: OOS AUC.\n")

    for fname in FEATURES:
        x = feat[fname]
        for h in HORIZONS:
            if h >= n:
                continue
            fwd = feat[f"ret_{h}"]
            m = ~(np.isnan(x) | np.isnan(fwd))
            xx = x[m]; yy = (fwd[m] > 0).astype(int)
            base_rate = yy.mean()
            folds = purged_folds(len(xx), h)
            if not folds:
                print(f"{fname:>8} {h:>3}d : not enough folds"); continue
            per_fold = []
            oos_all_p = []; oos_all_y = []
            for tr, te in folds:
                clf = LogisticRegression()
                clf.fit(xx[tr].reshape(-1, 1), yy[tr])
                p = clf.predict_proba(xx[te].reshape(-1, 1))[:, 1]
                if len(np.unique(yy[te])) < 2:
                    continue
                a = roc_auc_score(yy[te], p)
                per_fold.append(a)
                oos_all_p.append(p); oos_all_y.append(yy[te])
            if not per_fold:
                continue
            pooled = roc_auc_score(np.concatenate(oos_all_y), np.concatenate(oos_all_p))
            mean_fold = np.mean(per_fold)
            se = np.std(per_fold) / np.sqrt(len(per_fold))
            n_up = len(per_fold)
            sig = "***" if (mean_fold - 1.96 * se) > 0.5 else ""
            print(f"{fname:>8} {h:>3}d : pooled AUC={pooled:.4f} | per-fold "
                  f"{np.round(per_fold,3)} | mean={mean_fold:.4f}±{se:.4f} "
                  f"base_rate={base_rate:.2f} {sig}")

    print("\nInterpretasi: pooled AUC > 0.5 & mean-fold - 1.96SE > 0.5 = momentum punya"
          "\ninformasi up/down yang nyata OOS, bebas leakage label.")


if __name__ == "__main__":
    run()
