#!/usr/bin/env python3
"""
purged_cv_momentum_alt.py — leakage-free purged-CV validation of ALTERNATIVE
momentum SPECIFICATIONS as OOS predictors of BTC up/down.

Motivation
----------
Raw trailing momentum (mom_30 / mom_90) was already validated under purged-CV
(López de Prado, embargo=h) in `purged_cv_momentum.py` and REJECTED: pooled AUC
0.520@7d, 0.413@30d, 0.371@90d — overlapping forward-return labels inflate the
effective sample size, so the raw IC screen was misleading. This script answers
the follow-up question: do BETTER momentum SPECIFICATIONS (not raw mom_30/90)
carry a genuine leakage-free OOS edge? These have never been tested here.

Specifications tested (each as a SINGLE-feature logistic-regression predictor of
y_t = 1[forward return over h > 0], h in {7,30,90}):
  1) risk_managed  : TSMOM (Moskowitz-Ooi-Pedersen) — trailing momentum scaled by
                     inverse realized volatility (mom_h / rvol_20). Normalizing by
                     vol removes the vol-momentum confound.
  2) composite     : multi-horizon momentum — average of z-scored mom_21/63/126/252
                     (equal-weight blend of several lookbacks, not one lag).
  3) voltarget     : volatility-TARGETED momentum — mom_h scaled toward a target
                     vol but CAPPED (never scaled up when vol is low):
                     mom_h * min(target_vol / rvol_20, 1). Genuinely distinct from
                     (1) which scales up when vol is low.
  4) ema           : smoothed/exponential momentum — EMA(alpha=2/(L+1)) of daily
                     log returns over lookback L=30 (noise-reduced momentum proxy).

Methodology (identical to purged_cv_momentum.py)
  - Label: y_t = 1 if forward return over h days > 0.
  - Model: LogisticRegression (single feature); score monotone in the feature.
  - K contiguous folds; per fold: purge train samples whose label window [j,j+h]
    overlaps the test block [+ embargo h], fit, predict OOS AUC on test.
  - Report pooled OOS AUC + per-fold mean ± SE (flag if mean - 1.96*SE > 0.5),
    plus base rate.

Era-split (stability check)
  Per-era leakage-free OOS AUC (same purged-CV restricted to that era):
    era1 : < 2020-01-01   (2017-08 .. 2019-12)
    era2 : 2020-01-01 .. 2022-12-31
    era3 : >= 2023-01-01  (2023 .. 2026-08)
  era-stable = era2 AND era3 both on the same side of 0.5. Honest rule: purged-CV
  that overturns an IC screen is strong evidence; an era-flip alone is NOT enough
  to reject a factor (IERF), but an edge MUST survive purged-CV to be considered.

Pure analysis. No production change.
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

HORIZONS = [7, 30, 90]
K = 5
VOL_LB = 20            # realized-vol lookback (days) for vol normalization/targeting
EMA_LB = 30            # EMA lookback for smoothed momentum
COMPOSITE_LBS = [21, 63, 126, 252]
TARGET_VOL_FRAC = 0.20  # vol-targeting target as a fraction of the median rvol

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(REPO, ".purged_cv_momentum_alt.json")


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
            if j < i0 and j + h >= i0:
                continue  # purge: label window overlaps test
            if i1 <= j <= i1 + h:
                continue  # embargo after test block
            train.append(j)
        if len(train) < 50 or len(test) < 30:
            continue
        folds.append((np.array(train), np.array(test)))
    return folds


def era_mask(dates, lo, hi):
    """Boolean mask selecting dates in [lo, hi) (ISO strings)."""
    return np.array([(d >= lo) and (d < hi) for d in dates], dtype=bool)


def oos_auc_purged(x, y, h):
    """Run purged-CV on aligned feature/label arrays; return (pooled_auc,
    per_fold_auc_list, base_rate) or None if no usable folds."""
    folds = purged_folds(len(x), h)
    if not folds:
        return None
    per_fold = []
    oos_all_p, oos_all_y = [], []
    for tr, te in folds:
        clf = LogisticRegression()
        clf.fit(x[tr].reshape(-1, 1), y[tr])
        p = clf.predict_proba(x[te].reshape(-1, 1))[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        per_fold.append(roc_auc_score(y[te], p))
        oos_all_p.append(p)
        oos_all_y.append(y[te])
    if not per_fold:
        return None
    pooled = roc_auc_score(np.concatenate(oos_all_y), np.concatenate(oos_all_p))
    return pooled, np.array(per_fold), float(y.mean())


def build_specs(feat, closes, lr):
    """Return dict spec_name -> aligned array of predictor values (length n)."""
    n = len(closes)

    def mom_lookback(lb):
        m = np.full(n, np.nan)
        for i in range(lb, n):
            m[i] = np.log(closes[i]) - np.log(closes[i - lb])
        return m

    def rvol_lookback(lb):
        rv = np.full(n, np.nan)
        for i in range(lb, n):
            rv[i] = np.nanstd(lr[i - lb + 1:i + 1])
        return rv

    mom20 = mom_lookback(VOL_LB)
    rvol20 = rvol_lookback(VOL_LB)
    specs = {}

    # 1) Risk-managed / vol-normalized momentum (TSMOM): mom_h / rvol_20
    specs["risk_managed"] = {}
    for h in HORIZONS:
        specs["risk_managed"][h] = mom_lookback(h) / rvol20

    # 3) Volatility-targeted momentum (capped): mom_h * min(target/rvol_20, 1)
    target = float(np.nanmedian(rvol20)) * TARGET_VOL_FRAC
    cap = np.minimum(target / rvol20, 1.0)
    specs["voltarget"] = {}
    for h in HORIZONS:
        specs["voltarget"][h] = mom_lookback(h) * cap

    # 2) Multi-horizon composite: mean of z-scored mom_21/63/126/252
    zcols = []
    for lb in COMPOSITE_LBS:
        m = mom_lookback(lb)
        s = np.nanstd(m)
        zcols.append((m - np.nanmean(m)) / s)
    stack = np.stack(zcols)
    valid = ~np.isnan(stack)
    with np.errstate(all="ignore"):
        comp = np.where(valid.any(axis=0),
                        np.nansum(stack, axis=0) / valid.sum(axis=0),
                        np.nan)
    specs["composite"] = {h: comp for h in HORIZONS}

    # 4) Smoothed / exponential momentum: EMA of daily log returns
    alpha = 2.0 / (EMA_LB + 1)
    ema = np.full(n, np.nan)
    ema_val = np.nan
    for i in range(n):
        r = lr[i]
        if np.isnan(r):
            continue
        ema_val = r if np.isnan(ema_val) else alpha * r + (1 - alpha) * ema_val
        ema[i] = ema_val
    specs["ema"] = {h: ema.copy() for h in HORIZONS}

    return specs, target


def run():
    feat = compute_features(load_daily())
    days = np.array(feat["days"])
    closes = feat["close"]
    lr = np.concatenate([[np.nan], np.diff(np.log(closes))])
    n = len(days)
    print(f"PURGED-CV (López de Prado) — ALTERNATIVE momentum specs, {K} folds "
          f"+ embargo=h. ({n} days, {days[0]} -> {days[-1]})")
    print("Label: y=1 if fwd return>0. Model: single-feature LogisticRegression. "
          "Metric: OOS AUC.")
    print(f"Eras: era1 <2020 | era2 2020-2022 | era3 >=2023\n")

    specs, target_vol = build_specs(feat, closes, lr)
    print(f"vol-target (median rvol * {TARGET_VOL_FRAC}) = {target_vol:.4f}\n")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_days": n,
        "date_range": [days[0], days[-1]],
        "k_folds": K,
        "method": "Lopez de Prado purged-CV, embargo=h, single-feature LogisticRegression, OOS AUC",
        "specs": {},
    }

    ERAS = [("era1", "0000-01-01", "2020-01-01"),
            ("era2", "2020-01-01", "2023-01-01"),
            ("era3", "2023-01-01", "9999-12-31")]

    header = (f"{'spec':<13}{'h':>4} {'pooled':>7} {'mean-fold':>15} "
              f"{'base':>5}  flag   era1  era2  era3  era_stable")
    print(header)
    print("-" * len(header))

    for spec_name, spec in specs.items():
        result["specs"][spec_name] = {}
        for h in HORIZONS:
            x = spec[h]
            fwd = feat[f"ret_{h}"]
            m = ~(np.isnan(x) | np.isnan(fwd))
            xx = x[m]
            yy = (fwd[m] > 0).astype(int)
            dd = days[m]

            out = oos_auc_purged(xx, yy, h)
            if out is None:
                print(f"{spec_name:<13}{h:>4}  (not enough folds)")
                result["specs"][spec_name][h] = {"error": "not enough folds"}
                continue
            pooled, per_fold, base_rate = out
            mean_fold = float(np.mean(per_fold))
            se = float(np.std(per_fold)) / np.sqrt(len(per_fold))
            sig = (mean_fold - 1.96 * se) > 0.5

            # per-era purged-CV pooled AUC
            era_auc = {}
            for ename, lo, hi in ERAS:
                em = era_mask(dd, lo, hi)
                if em.sum() < 60:
                    era_auc[ename] = None
                    continue
                eo = oos_auc_purged(xx[em], yy[em], h)
                era_auc[ename] = eo[0] if eo else None

            e2 = era_auc.get("era2")
            e3 = era_auc.get("era3")
            era_stable = bool(e2 is not None and e3 is not None
                              and ((e2 > 0.5 and e3 > 0.5) or (e2 < 0.5 and e3 < 0.5)))

            def fmt_auc(v):
                return f"{v:.3f}" if v is not None else " n/a "

            print(f"{spec_name:<13}{h:>4} {pooled:>7.4f} {mean_fold:>7.4f}±{se:.4f} "
                  f"{base_rate:>5.2f}  {'YES' if sig else 'no '}   "
                  f"{fmt_auc(era_auc.get('era1'))}  {fmt_auc(e2)}  {fmt_auc(e3)}   "
                  f"{'Y' if era_stable else 'n'}")

            result["specs"][spec_name][h] = {
                "pooled_auc": round(pooled, 4),
                "mean_fold_auc": round(mean_fold, 4),
                "fold_se": round(se, 4),
                "per_fold_auc": [round(float(a), 4) for a in per_fold],
                "base_rate": round(base_rate, 3),
                "significant": bool(sig),
                "era_auc": {k: (round(v, 4) if v is not None else None)
                            for k, v in era_auc.items()},
                "era_stable": era_stable,
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved raw results -> {OUTPUT_FILE}")
    print("flag YES = mean-fold - 1.96*SE > 0.5 (purged-CV edge). "
          "era_stable = era2 & era3 same side of 0.5.")


if __name__ == "__main__":
    run()
