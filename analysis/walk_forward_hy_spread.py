#!/usr/bin/env python3
"""
walk_forward_hy_spread.py — Walk-forward test of the high-yield credit spread
(BAMLH0A0HYM2) as a predictive signal for BTC forward returns.

WHY
    Thesis under test (Phase-1 claim): "credit stress -> liquidity squeeze ->
    Bitcoin pressured early (risk-off)." This is the first, testable half of a
    two-phase narrative. We test whether the high-yield credit spread (the
    closest public-market proxy for credit stress available in the system —
    M30 Rajan FSI uses it) predicts BTC forward returns out-of-sample.

    Note this is PUBLIC high-yield, NOT private credit. Private credit has no
    free, long-history, live series in the system. A null/weak result here
    applies to the public-HY proxy, not a blanket statement about all credit
    stress. Per walk-forward-validation skill Pitfall 14: report as a research
    recommendation, not a deployable cutoff.

SIGNAL
    hy_spread = BAMLH0A0HYM2 (ICE BofA US High Yield OAS, basis points).
    Point-in-time expanding-window z-score (no look-ahead): HIGH z = credit
    stress elevated -> hypothesis predicts LOWER forward BTC return (risk-off).

METHOD (identical to walk_forward_carry_jpy.py)
    1. Daily HY spread, point-in-time expanding z (min warmup), no look-ahead.
    2. Forward BTC return (CBBTCUSD) over 30/90/180 days.
    3. Top-vs-bottom tail gap, direct bootstrap difference, 90% CI, two-tailed.
    4. Era split (2015-2020 vs 2021-2026).
    5. Confound: date & price range per bucket.
"""
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SFC_DIR)
sys.path.insert(0, os.path.join(SFC_DIR, "analysis"))

OUTPUT_FILE = os.path.join(SFC_DIR, ".walk_forward_hy_spread.json")
SUMMARY_FILE = os.path.join(SFC_DIR, ".walk_forward_hy_spread_summary.json")

N_BOOTSTRAP = 2000
CI = 0.90
WARMUP_DAYS = 250
TAIL = 0.25
HORIZONS_DAYS = [30, 90, 180]


def fetch_fred_map(series_id):
    from historical_backtest_m1m6 import fetch_fred_series
    return fetch_fred_series(series_id)


def build_series():
    hy = fetch_fred_map("BAMLH0A0HYM2")
    btc = fetch_fred_map("CBBTCUSD")
    if not hy or not btc:
        raise RuntimeError("Missing FRED data (BAMLH0A0HYM2/CBBTCUSD) — check FRED_API_KEY/network.")

    btc_dates = sorted(btc.keys())
    hy_dates = sorted(hy.keys())

    # Align: for each BTC date, use most recent prior HY value
    rows = []  # (date, hy_spread)
    hy_i = -1
    prev_date = None
    for d in btc_dates:
        # advance hy_i to the latest HY date <= d
        while hy_i + 1 < len(hy_dates) and hy_dates[hy_i + 1] <= d:
            hy_i += 1
        if hy_i < 0:
            continue
        rows.append((d, hy[hy_dates[hy_i]]))
    if len(rows) < WARMUP_DAYS + 50:
        raise RuntimeError(f"Too few aligned points: {len(rows)}")

    # Point-in-time expanding z-score
    series = []
    sps = [c for _, c in rows]
    for i, (d, c) in enumerate(rows):
        if i < WARMUP_DAYS:
            continue
        hist = sps[:i]
        if len(hist) < 2:
            continue
        mu = statistics.mean(hist)
        sd = statistics.stdev(hist)
        if sd == 0:
            continue
        z = (c - mu) / sd
        z = max(-3.0, min(3.0, z))
        price = btc.get(d)
        if price is None:
            continue
        row = {"date": d, "hy": round(c, 2), "z": round(z, 3), "price": price}
        di = btc_dates.index(d)
        for h in HORIZONS_DAYS:
            fi = di + h
            if fi < len(btc_dates):
                fp = btc[btc_dates[fi]]
                row[f"fwd_{h}d"] = (fp - price) / price * 100
            else:
                row[f"fwd_{h}d"] = None
        series.append(row)
    return series


def bootstrap_diff_ci(a, b, seed=42):
    if len(a) < 3 or len(b) < 3:
        return None, None, None
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sa = [a[rng.randrange(na)] for _ in range(na)]
        sb = [b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(sum(sb) / nb - sum(sa) / na)
    diffs.sort()
    lo = int((1 - CI) / 2 * N_BOOTSTRAP)
    hi = int((1 + CI) / 2 * N_BOOTSTRAP) - 1
    return (sum(b) / nb - sum(a) / na), diffs[lo], diffs[hi]


def analyze(series):
    print("\n" + "=" * 72)
    print("WALK-FORWARD TEST — High-Yield credit spread (BAMLH0A0HYM2) vs BTC")
    print("=" * 72)
    print(f"Points: {len(series)}  | range {series[0]['date']} .. {series[-1]['date']}")
    print("Hypothesis: HIGH hy-spread z (credit stress) -> LOWER BTC fwd return (risk-off).")
    print("Gap computed as (high-spread bucket) - (low-spread bucket); expected NEGATIVE.")

    for h in HORIZONS_DAYS:
        key = f"fwd_{h}d"
        pts = [p for p in series if p.get(key) is not None]
        if len(pts) < 10:
            print(f"\n--- {h}d: insufficient ({len(pts)}) ---")
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tn = max(3, int(n * TAIL))
        low = zs[:tn]    # low spread (credit benign)
        high = zs[-tn:]  # high spread (credit stress)
        lr = [p[key] for p in low]
        hr = [p[key] for p in high]
        lm, hm = statistics.mean(lr), statistics.mean(hr)
        est, lo, hi = bootstrap_diff_ci(lr, hr, seed=42)
        sig = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0)
        gap = hm - lm  # high - low; negative supports hypothesis
        print(f"\n--- {h}d forward return ---")
        print(f"  Low  spread (benign): n={len(low):3d} mean {lm:+7.2f}%  z<= {low[-1]['z']:.2f}")
        print(f"  High spread (stress): n={len(high):3d} mean {hm:+7.2f}%  z>= {high[0]['z']:.2f}")
        if est is not None:
            print(f"  Gap (stress-benign): {gap:+6.2f}pp  bootstrap {'SIGNIFICANT' if sig else 'n.s.'}  "
                  f"[{est:+.2f} ({lo:+.2f}, {hi:+.2f})]")
        else:
            print(f"  Gap: {gap:+6.2f}pp  (insufficient)")

    print("\n" + "=" * 72)
    print("ERA SPLIT — 90d forward return")
    print("=" * 72)
    key = "fwd_90d"
    for en, lo_d, hi_d in [("2015-2020", "2015-01-01", "2020-12-31"), ("2021-2026", "2021-01-01", "2099-12-31")]:
        pts = [p for p in series if p.get(key) is not None and lo_d <= p["date"] <= hi_d]
        if len(pts) < 10:
            print(f"  {en}: n={len(pts)} insufficient")
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tn = max(3, int(n * TAIL))
        lr = [p[key] for p in zs[:tn]]
        hr = [p[key] for p in zs[-tn:]]
        gap = statistics.mean(hr) - statistics.mean(lr)
        est, lo, hi = bootstrap_diff_ci(lr, hr, seed=7)
        sig = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0)
        if est is not None:
            print(f"  {en}: n={len(pts):3d}  gap(stress-benign)={gap:+6.2f}pp  {'SIG' if sig else 'n.s.'}  "
                  f"[{est:+.2f} ({lo:+.2f}, {hi:+.2f})]")
        else:
            print(f"  {en}: n={len(pts):3d} gap={gap:+.2f}pp insufficient")

    print("\n" + "=" * 72)
    print("CONFOUND — date & BTC price range by spread bucket (90d)")
    print("=" * 72)
    zs = sorted([p for p in series if p.get("fwd_90d") is not None], key=lambda p: p["z"])
    if zs:
        n = len(zs)
        tn = max(3, int(n * TAIL))
        for label, grp in [("LOW (benign)", zs[:tn]), ("MID", zs[tn:-tn]), ("HIGH (stress)", zs[-tn:])]:
            if not grp:
                continue
            ds = [p["date"] for p in grp]
            ps = [p["price"] for p in grp]
            print(f"  {label:<15} n={len(grp):3d}  {min(ds)}..{max(ds)}  btc ${min(ps):,.0f}..${max(ps):,.0f}")


def write_summary(series):
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "n_points": len(series),
               "method": "point-in-time expanding z of BAMLH0A0HYM2, no look-ahead"}
    for h in HORIZONS_DAYS:
        key = f"fwd_{h}d"
        pts = [p for p in series if p.get(key) is not None]
        if len(pts) < 10:
            summary[f"gap_{h}d"] = None
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tn = max(3, int(n * TAIL))
        lr = [p[key] for p in zs[:tn]]
        hr = [p[key] for p in zs[-tn:]]
        est, lo, hi = bootstrap_diff_ci(lr, hr, seed=42)
        summary[f"gap_{h}d"] = round(est, 2) if est is not None else None
        summary[f"gap_{h}d_ci_lo"] = round(lo, 2) if lo is not None else None
        summary[f"gap_{h}d_ci_hi"] = round(hi, 2) if hi is not None else None
        summary[f"gap_{h}d_significant"] = bool(lo is not None and hi is not None and (lo > 0 or hi < 0))
        summary[f"n_low_{h}d"] = len(lr)
        summary[f"n_high_{h}d"] = len(hr)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[HYSpr] Summary -> {SUMMARY_FILE}")


def main():
    random.seed(42)
    series = build_series()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f, indent=2)
    print(f"[HYSpr] Full series -> {OUTPUT_FILE}")
    analyze(series)
    write_summary(series)


if __name__ == "__main__":
    main()
