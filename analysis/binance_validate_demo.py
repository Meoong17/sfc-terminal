#!/usr/bin/env python3
"""
Demo validasi memakai cache multi-timeframe Binance (canonical 9-yr series).

Sebelumnya: sinyal funding tidak bisa divalidasi karena funding SFC cuma ~17 hari.
Sekarang: funding futures BTCUSDT 2020-01 -> 2026-07 (~6.5 tahun) memungkinkan
walk-forward panjang dgn bootstrap CI + era-split.

Juga demo: realized-vol & premium(basis) sebagai faktor, dan forward-return label
canonical yang span 9 tahun (vs snapshot SFC 2 bulan).

Skrip ini ANALISIS murni — tidak menyentuh produksi.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

HORIZONS = [7, 30, 90, 365]


def tail_gap_bootstrap(feat, feature, horizon, nboot=20000, seed=42):
    x = feat[feature]; fwd = feat[f"ret_{horizon}"]
    m = ~(np.isnan(x) | np.isnan(fwd))
    x, fwd = x[m], fwd[m]
    n = len(x)
    if n < 40:
        return np.nan, (np.nan, np.nan), np.nan, n
    qh, ql = np.quantile(x, [0.75, 0.25])
    obs = fwd[x >= qh].mean() - fwd[x <= ql].mean()
    rng = np.random.default_rng(seed)
    idx = np.arange(n); gaps = []
    for _ in range(nboot):
        b = rng.choice(idx, n, replace=True)
        xb, fb = x[b], fwd[b]
        q1, q0 = np.quantile(xb, [0.75, 0.25])
        h = xb >= q1; l = xb <= q0
        if h.sum() and l.sum():
            gaps.append(fb[h].mean() - fb[l].mean())
    gaps = np.array(gaps)
    return obs, (np.percentile(gaps, 5), np.percentile(gaps, 95)), np.nanmean(fwd), n


def era_gap(feat, feature, horizon, split_dates):
    # forward-ret sign consistency across two eras (date-based split)
    dates = np.array(feat["days"])
    fwd = feat[f"ret_{horizon}"]; x = feat[feature]
    res = {}
    for label, (d0, d1) in split_dates.items():
        m = (dates >= d0) & (dates < d1) & ~np.isnan(x) & ~np.isnan(fwd)
        if m.sum() < 20:
            res[label] = np.nan; continue
        xx, ff = x[m], fwd[m]
        qh, ql = np.quantile(xx, [0.75, 0.25])
        res[label] = ff[xx >= qh].mean() - ff[xx <= ql].mean()
    return res


def main():
    daily = load_daily()
    feat = compute_features(daily)
    n = len(feat["days"])
    print(f"canonical daily series: {n} days "
          f"({feat['days'][0]} -> {feat['days'][-1]})")
    print("forward-return labels kini span 9 tahun (vs snapshot SFC 2 bulan).\n")

    for feature, label, polarity in [
        ("funding", "FUNDING (min|fr|*10, 1)", "neg"),
        ("premium", "PREMIUM/basis", "neg"),
        ("rvol_90", "REALIZED VOL 90d", "neg"),
        ("mom_30", "MOMENTUM 30d", "pos"),
    ]:
        print(f"=== {label} (prediksi arah: {polarity}) ===")
        for h in HORIZONS:
            obs, ci, base, nn = tail_gap_bootstrap(feat, feature, h)
            if np.isnan(obs):
                continue
            sig = "***" if (ci[0] < 0 and ci[1] < 0) else ("+++" if (ci[0] > 0 and ci[1] > 0) else "")
            print(f"  {h:>3}d gap(top-bot)={obs:+.4f} [CI90 {ci[0]:+.3f},{ci[1]:+.3f}] {sig}  (n={nn})")
        er = era_gap(feat, feature, 30, {"era1": ("2020-01-01", "2023-01-01"),
                                          "era2": ("2023-01-01", "2027-01-01")})
        if "funding" in feature or "premium" in feature:
            print(f"  30d gap era1(2020-22)={er.get('era1', float('nan')):+.4f}  "
                  f"era2(2023-26)={er.get('era2', float('nan')):+.4f}")
        print()

    print("Catatan: demo struktur — signal 'funding' dan 'premium' kini punya 6.5 tahun "
          "untuk validasi (sebelumnya 17 hari). Era-split = cek stabilitas lintas rezim.")


if __name__ == "__main__":
    main()
