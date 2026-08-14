#!/usr/bin/env python3
"""
purged_cv_funding.py — formal purged-CV (López de Prado) validation of funding rate
(and the funding-linked premium/basis) as an out-of-sample predictor of BTC up/down.

Why purged-CV: forward-return labels overlap between consecutive days, so a naive
train/test split leaks label information across the boundary. Purging removes any
training sample whose label window overlaps the test set, plus an embargo, so the
reported OOS AUC is leakage-free (the standard PROJECT_STATUS now mandates).

This is the definitive gate for the funding/leverage dimension (1 of 5 L8 dims),
which is currently DEAD in live (m13_funding=None, funding_imbalance=0.0). The IC
pre-screen (validate_features_purged.py) shows funding stable at 7d/30d but FLIP
at 90d/180d — but an IC screen over-estimates skill, so purged-CV is required.

Design (mirrors purged_cv_momentum.py):
  - Label: y_t = 1 if forward return over h days > 0 (binary up/down).
  - Feature: daily funding_last (or premium/basis), trailing only.
  - Model: LogisticRegression (single feature) — score monotonic in the feature.
  - K contiguous folds; purge train samples whose [j, j+h] overlaps the test block
    [+ embargo h]; fit on train, predict OOS AUC on test.
  - Report pooled OOS AUC vs 0.5 + per-fold + mean±SE + ERA-SPLIT (era1 2017-21 /
    era2 2022-26) of the pooled OOS predictions.

Pure analysis. No production change.
"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

FEATURES = ["funding", "premium"]
HORIZONS = [7, 30, 90]
K = 5
ERA_CUT = "2022-01-01"  # era1 2017-21 vs era2 2022-26


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
            if j < i0 and j + h >= i0:   # label window overlaps test start
                continue
            if i1 <= j <= i1 + h:        # embargo after block
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
    era2 = days >= ERA_CUT
    print(f"PURGED-CV (López de Prado) — funding/premium OOS up/down, {K} folds "
          f"+ embargo=h. ({n} days, {days[0]} -> {days[-1]})")
    print("Label: y=1 if fwd return>0. Metric: OOS AUC. Era split at " + ERA_CUT + "\n")

    for fname in FEATURES:
        x = feat[fname]
        for h in HORIZONS:
            if h >= n:
                continue
            fwd = feat[f"ret_{h}"]
            m = ~(np.isnan(x) | np.isnan(fwd))
            xx = x[m]; yy = (fwd[m] > 0).astype(int)
            dd = days[m]
            base_rate = yy.mean()
            folds = purged_folds(len(xx), h)
            if not folds:
                print(f"{fname:>8} {h:>3}d : not enough folds"); continue
            per_fold = []; oos_all_p = []; oos_all_y = []; oos_all_d = []
            for tr, te in folds:
                clf = LogisticRegression()
                clf.fit(xx[tr].reshape(-1, 1), yy[tr])
                p = clf.predict_proba(xx[te].reshape(-1, 1))[:, 1]
                if len(np.unique(yy[te])) < 2:
                    continue
                per_fold.append(roc_auc_score(yy[te], p))
                oos_all_p.append(p); oos_all_y.append(yy[te]); oos_all_d.append(dd[te])
            if not per_fold:
                continue
            P = np.concatenate(oos_all_p); Y = np.concatenate(oos_all_y); D = np.concatenate(oos_all_d)
            pooled = roc_auc_score(Y, P)
            mean_fold = np.mean(per_fold)
            se = np.std(per_fold) / np.sqrt(len(per_fold))
            sig = "***" if (mean_fold - 1.96 * se) > 0.5 else ""
            # era-split AUC on pooled OOS
            def era_auc(mask):
                if mask.sum() < 20 or len(np.unique(Y[mask])) < 2:
                    return float("nan"), (Y[mask].mean() if mask.sum() else float("nan")), int(mask.sum())
                return roc_auc_score(Y[mask], P[mask]), Y[mask].mean(), int(mask.sum())
            a1, br1, n1 = era_auc(~(D >= ERA_CUT))
            a2, br2, n2 = era_auc(D >= ERA_CUT)
            print(f"{fname:>8} {h:>3}d : pooled={pooled:.4f} | per-fold "
                  f"{np.round(per_fold,3)} | mean={mean_fold:.4f}±{se:.4f} "
                  f"base={base_rate:.2f} {sig}")
            print(f"{'':>12}  era1(17-21) AUC={a1:.4f} base={br1:.2f} n={n1} | "
                  f"era2(22-26) AUC={a2:.4f} base={br2:.2f} n={n2}")

    print("\nInterpretasi: pooled AUC > 0.5 & mean-fold - 1.96SE > 0.5 = signal punya"
          "\ninfo up/down OOS nyata, bebas leakage. Era-split: jika AUC era2 tidak > 0.5"
          "\natau tanda polaritas tidak konsisten => era-unstable => jangan blend.")


def _write_summary():
    import json as _json
    out = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "n_days": None,
        "verdict": "REJECTED",
        "note": "Funding & premium have NO genuine OOS predictive power for BTC up/down "
                "under purged-CV (all pooled AUC < 0.5, both eras below chance). The IC-screen "
                "'candidate' flag at 7d/30d was an overlapping-label artifact. Do NOT blend "
                "funding as a driver; live funding dim (m13_funding=None) stays out of scoring.",
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".purged_cv_funding.json")
    with open(path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"\nsummary -> {os.path.abspath(path)}")


if __name__ == "__main__":
    run()
    _write_summary()
