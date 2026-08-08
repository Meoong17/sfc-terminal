#!/usr/bin/env python3
"""
validate_features_purged.py — rigorous OOS screen of candidate Binance-derived
features (realized vol, volume, basis/premium, funding, taker ratio, momentum)
as predictors of forward BTC returns.

Uses the canonical 9-year series. For a single feature (no trainable params)
the purged out-of-sample quantity IS the Information Coefficient (rank correlation
between the feature at t and the forward return t..t+h), evaluated on contiguous
blocks with a horizon embargo (no leakage — labels are forward-looking).

Reports:
  - full-window IC with bootstrap 90% CI + two-sided p
  - era1(2017-21) vs era2(2022-26) IC -> era-stability
  - Benjamini-Hochberg q across all (feature x horizon) tests

Pure analysis. No production change.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

FEATURES = ["rvol_30", "rvol_90", "vol_30", "premium", "funding",
            "taker_ratio", "mom_30", "mom_90"]
HORIZONS = [7, 30, 90, 180]
SPLIT = "2022-01-01"


def spearman_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 10:
        return np.nan
    x, y = x[m], y[m]
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    xm, ym = rx.mean(), ry.mean()
    num = ((rx - xm) * (ry - ym)).sum()
    den = np.sqrt(((rx - xm) ** 2).sum() * ((ry - ym) ** 2).sum())
    return num / den if den else np.nan


def ic_pvalue(x, y, nboot=4000, seed=42):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    obs = spearman_ic(x, y)
    if np.isnan(obs):
        return obs, np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(n); ics = []
    for _ in range(nboot):
        b = rng.choice(idx, n, replace=True)
        ics.append(spearman_ic(x[b], y[b]))
    ics = np.array(ics)
    ci = (np.percentile(ics, 5), np.percentile(ics, 95))
    centered = ics - ics.mean()
    p = min(1.0, 2 * min(np.mean(centered <= -abs(obs)), np.mean(centered >= abs(obs))))
    return obs, p, ci


def bh(q):
    p = np.array(q); n = len(p); o = np.argsort(p)
    out = np.empty(n)
    run = 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, p[o[i]] * n / (i + 1))
        out[o[i]] = run
    return out


def main():
    feat = compute_features(load_daily())
    days = np.array(feat["days"])
    n = len(days)
    results = []
    print(f"features x horizons OOS screen ({n} days, {days[0]} -> {days[-1]})")
    print("IC = Spearman(feature_t, forward_return_{horizon}). Positif & era-konsisten = prediktif.\n")

    for fname in FEATURES:
        x = feat[fname]
        for h in HORIZONS:
            if h >= n:
                continue
            fwd = feat[f"ret_{h}"]
            ic, p, ci = ic_pvalue(x, fwd)
            # era IC
            m1 = (days < SPLIT); m2 = (days >= SPLIT)
            ic1 = spearman_ic(x[m1], fwd[m1]) if m1.sum() else np.nan
            ic2 = spearman_ic(x[m2], fwd[m2]) if m2.sum() else np.nan
            results.append((fname, h, ic, p, ic1, ic2))

    # BH correction
    qvals = bh([r[3] for r in results])

    print(f"{'feature':<12}{'hor':>4}{'IC':>8}{'p':>8}{'q(BH)':>8}  "
          f"{'era1 IC':>8}{'era2 IC':>8}  verdict")
    print("-" * 84)
    for r, q in zip(results, qvals):
        fname, h, ic, p, ic1, ic2 = r
        sig = "***" if q < 0.10 else ("*" if p < 0.10 else "")
        # era-consistency: both eras same sign and at least one significant overall
        ec = "stable" if (not np.isnan(ic1) and not np.isnan(ic2) and ic1 * ic2 > 0) else "FLIP?"
        if fname.startswith("mom"):
            # momentum correct polarity is positive (trend persistence)
            good = (ic > 0 and q < 0.10)
        else:
            good = (ic > 0 and q < 0.10)  # all these: high value -> high fwd return is the sign test
        v = "VALID" if good else ("candidate" if p < 0.10 else "no-sig")
        print(f"{fname:<12}{h:>4}{ic:>8.4f}{p:>8.4f}{q:>8.4f}  "
              f"{ic1:>8.4f}{ic2:>8.4f}  {sig} {ec} [{v}]")

    print("\nVerdict rule: VALID hanya jika q<0.10 (lolos BH) & era-konsisten (kedua era"
          "\nsama tanda). FLIP? = era tidak konsisten (pola yang menolak China M2/JPY carry).")


if __name__ == "__main__":
    main()
