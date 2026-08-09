#!/usr/bin/env python3
"""
walk_forward_china_m2.py — Walk-forward validation of the China M2 component's
z-score calibration in global_liquidity_engine.py.

WHY THIS EXISTS
    global_liquidity_engine.py's China M2 component was re-derived 2026-08 from
    chinadata.live history (2015-2026): YoY mean = 9.53%, std = 1.73%. Those are
    FULL-SAMPLE constants — calibrated in-sample on the whole history. Before
    treating them as the live cutoff, they must be validated out-of-sample.

WHAT IT TESTS
    The predictive question, separated from the calibration question:
      "Does the China M2 YoY z-score — computed POINT-IN-TIME (rolling stats,
       no look-ahead) — genuinely predict subsequent BTC forward returns?"
    If low China z-score (contraction) predicts worse BTC forward returns than
    high z-score, the component's direction is real and its calibration is
    directionally sound. We use point-in-time rolling mean/std so we never peek
    at future data — mirroring how the GLF z-score is consumed downstream.

    This validates the China component ALONE (a 1-factor signal). It does NOT
    claim the live GLF China weight (0.04) is optimal, nor that this replays the
    full GLF composite. Per walk-forward-validation skill Pitfall 14: we validate
    only the reconstructable subset and report direction as research, not a
    deployable cutoff.

DATA
    - China M2: chinadata.live CSV (PBC-sourced), monthly 2015-01 .. 2026-05.
      Fetch reuses the same endpoint/parse as _fetch_china_m2() in the engine.
    - BTC price: FRED CBBTCUSD (daily), via historical_backtest_m1m6.fetch_fred_series.
      Requires FRED_API_KEY.

METHOD
    1. Build monthly China M2 YoY series.
    2. Point-in-time z-score at month T using only data up to T
       (expanding-window mean/std, min 24 months warmup).
    3. Look FORWARD to realized BTC return over 30/90/180 days after T
       (month-end price -> future month-end price). This is a fact, not a projection.
    4. Compare forward returns across z-score buckets + a threshold-free
       top-vs-bottom-tail gap, with 90% bootstrap CIs (direct bootstrap of the
       difference). Two-tailed significance.
    5. Era-split (2015-2020 vs 2021-2026) to check the edge is not one era.
    6. Confound check: date range + BTC price range per bucket.

HONEST LIMITS
    - Only 11 years of monthly China M2 data -> ~120 usable points. Stress events
      are rare; CIs are wide. A weak/null result here is NOT proof China M2 is
      useless — it proves the 0.04-weight component alone lacks a clean standalone
      predictive edge at monthly cadence, which is expected for a small weight.
"""
import csv
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SFC_DIR)
sys.path.insert(0, os.path.join(SFC_DIR, "analysis"))

CHINA_M2_URL = "https://chinadata.live/api/v2/data/china-m2-money-supply?format=csv"
OUTPUT_FILE = os.path.join(SFC_DIR, ".walk_forward_china_m2.json")
SUMMARY_FILE = os.path.join(SFC_DIR, ".walk_forward_china_m2_summary.json")

N_BOOTSTRAP = 2000
CI = 0.90
WARMUP_MONTHS = 24   # min history before first point-in-time z-score
TAIL = 0.25          # top 25% vs bottom 25% of z-score distribution
HORIZONS_DAYS = [30, 90, 180]


def fetch_china_m2():
    """Return [(date_str 'YYYY-MM', value), ...] ascending (oldest->newest)."""
    import requests
    r = requests.get(CHINA_M2_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"chinadata.live HTTP {r.status_code}")
    rows = []
    for line in r.text.strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            rows.append((parts[0].strip(), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError("chinadata.live returned no rows")
    # ascending by construction (oldest first)
    return rows


def fetch_btc_daily():
    """Return {date_str 'YYYY-MM-DD': price} via FRED CBBTCUSD."""
    try:
        from historical_backtest_m1m6 import fetch_fred_series
        btc = fetch_fred_series("CBBTCUSD")
    except Exception as e:
        print(f"[ChinaM2] fetch_fred_series failed: {e}", file=sys.stderr)
        return None
    return btc or None


def month_end_price(btc_daily, date_str):
    """Return BTC price at end of month date_str ('YYYY-MM')."""
    y, m = date_str.split("-")
    # last calendar day of month
    from calendar import monthrange
    last_day = monthrange(int(y), int(m))[1]
    target = f"{y}-{m}-{last_day:02d}"
    # nearest available price at-or-before month end
    best = None
    best_date = None
    for d in range(last_day, 0, -1):
        cand = f"{y}-{m}-{d:02d}"
        if cand in btc_daily:
            return btc_daily[cand]
    return best


def month_plus_days(date_str, days):
    """Return 'YYYY-MM' of the month containing date_str + days."""
    d = datetime.strptime(date_str, "%Y-%m")
    from datetime import timedelta
    nd = d + timedelta(days=days)
    return f"{nd.year}-{nd.month:02d}"


def build_series():
    china = fetch_china_m2()
    btc = fetch_btc_daily()
    if not btc:
        raise RuntimeError("No BTC price data — check FRED_API_KEY/network.")

    dates = [d for d, _ in china]
    vals = [v for _, v in china]
    yoy = []
    for i in range(12, len(vals)):
        yoy.append((dates[i], (vals[i] - vals[i - 12]) / vals[i - 12] * 100))
    print(f"[ChinaM2] YoY series: {len(yoy)} monthly obs, {yoy[0][0]} .. {yoy[-1][0]}")

    # Point-in-time expanding-window z-score (no look-ahead)
    series = []
    yoy_vals = [v for _, v in yoy]
    for i, (d, v) in enumerate(yoy):
        if i < WARMUP_MONTHS:
            continue
        hist = yoy_vals[:i]  # values BEFORE current (strictly prior)
        if len(hist) < 2:
            continue
        mu = statistics.mean(hist)
        sd = statistics.stdev(hist)
        if sd == 0:
            continue
        z = (v - mu) / sd
        z = max(-3.0, min(3.0, z))
        # forward BTC return
        end_price = month_end_price(btc, d)
        if end_price is None:
            continue
        row = {"date": d, "z": z, "yoy": v, "price": end_price}
        for h in HORIZONS_DAYS:
            fut_m = month_plus_days(d, h)
            fut_price = month_end_price(btc, fut_m)
            if fut_price is not None:
                row[f"fwd_{h}d"] = (fut_price - end_price) / end_price * 100
            else:
                row[f"fwd_{h}d"] = None
        series.append(row)
    return series


def bootstrap_diff_ci(a, b, seed=42):
    """Direct bootstrap of mean(b) - mean(a). Returns (est, lo, hi) or (None,None,None)."""
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
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION — China M2 z-score (point-in-time) vs BTC")
    print("=" * 70)
    print(f"Points (post-warmup, with BTC): {len(series)}")
    print(f"Range: {series[0]['date']} .. {series[-1]['date']}")

    for h in HORIZONS_DAYS:
        key = f"fwd_{h}d"
        pts = [p for p in series if p.get(key) is not None]
        if len(pts) < 10:
            print(f"\n--- {h}d: insufficient data ({len(pts)}) ---")
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tail_n = max(3, int(n * TAIL))
        low = zs[:tail_n]     # bottom z (contraction)
        high = zs[-tail_n:]   # top z (expansion)
        low_ret = [p[key] for p in low]
        high_ret = [p[key] for p in high]
        low_mean = statistics.mean(low_ret)
        high_mean = statistics.mean(high_ret)
        est, lo, hi = bootstrap_diff_ci(low_ret, high_ret, seed=42)
        significant = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0)
        print(f"\n--- {h}d forward return ---")
        print(f"  Bottom {TAIL*100:.0f}% z (contraction) : n={len(low):3d} mean {low_mean:+7.2f}%  z<= {low[-1]['z']:.2f}")
        print(f"  Top    {TAIL*100:.0f}% z (expansion)    : n={len(high):3d} mean {high_mean:+7.2f}%  z>= {high[0]['z']:.2f}")
        gap = high_mean - low_mean  # positive = expansion -> higher fwd return (correct direction)
        if est is not None:
            sig_str = "SIGNIFICANT" if significant else "not significant"
            print(f"  Gap (expansion - contraction): {gap:+6.2f}pp  direct-bootstrap {sig_str}  "
                  f"[90% CI {est:+.2f} ({lo:+.2f}, {hi:+.2f})]")
        else:
            print(f"  Gap (expansion - contraction): {gap:+6.2f}pp  (insufficient for bootstrap)")

    # Era split
    print("\n" + "=" * 70)
    print("ERA SPLIT (2015-2020 vs 2021-2026) — 90d forward return")
    print("=" * 70)
    key = "fwd_90d"
    for era_name, lo_d, hi_d in [("2015-2020", "2015-01", "2020-12"), ("2021-2026", "2021-01", "2099-01")]:
        pts = [p for p in series if p.get(key) is not None and lo_d <= p["date"] <= hi_d]
        if len(pts) < 10:
            print(f"  {era_name}: {len(pts)} pts — insufficient")
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tail_n = max(3, int(n * TAIL))
        low = [p[key] for p in zs[:tail_n]]
        high = [p[key] for p in zs[-tail_n:]]
        gap = statistics.mean(high) - statistics.mean(low)
        est, lo, hi = bootstrap_diff_ci(low, high, seed=7)
        sig = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0)
        print(f"  {era_name}: n={len(pts):3d}  gap(exp-contr)={gap:+6.2f}pp  "
              f"{'SIG' if sig else 'n.s.'}  [{est:+.2f} ({lo:+.2f}, {hi:+.2f})]" if est is not None else
              f"  {era_name}: n={len(pts):3d}  gap={gap:+.2f}pp  insufficient")

    # Confound: date/price range per bucket
    print("\n" + "=" * 70)
    print("CONFOUND CHECK — date & BTC price range by z bucket (all horizons pooled at 90d)")
    print("=" * 70)
    zs = sorted([p for p in series if p.get("fwd_90d") is not None], key=lambda p: p["z"])
    if zs:
        n = len(zs)
        tn = max(3, int(n * TAIL))
        for label, grp in [("LOW (contraction)", zs[:tn]), ("MID", zs[tn:-tn]), ("HIGH (expansion)", zs[-tn:])]:
            if not grp:
                continue
            ds = [p["date"] for p in grp]
            ps = [p["price"] for p in grp]
            print(f"  {label:<18} n={len(grp):3d}  dates {min(ds)}..{max(ds)}  "
                  f"btc ${min(ps):,.0f}..${max(ps):,.0f}")


def write_summary(series):
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "n_points": len(series), "method": "point-in-time expanding z, no look-ahead"}
    for h in HORIZONS_DAYS:
        key = f"fwd_{h}d"
        pts = [p for p in series if p.get(key) is not None]
        if len(pts) < 10:
            summary[f"gap_{h}d"] = None
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tn = max(3, int(n * TAIL))
        low = [p[key] for p in zs[:tn]]
        high = [p[key] for p in zs[-tn:]]
        est, lo, hi = bootstrap_diff_ci(low, high, seed=42)
        summary[f"gap_{h}d"] = round(est, 2) if est is not None else None
        summary[f"gap_{h}d_ci_lo"] = round(lo, 2) if lo is not None else None
        summary[f"gap_{h}d_ci_hi"] = round(hi, 2) if hi is not None else None
        summary[f"gap_{h}d_significant"] = bool(lo is not None and hi is not None and (lo > 0 or hi < 0))
        summary[f"n_bottom_{h}d"] = len(low)
        summary[f"n_top_{h}d"] = len(high)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ChinaM2] Summary cache -> {SUMMARY_FILE}")


def main():
    random.seed(42)
    series = build_series()
    if not series:
        print("[ChinaM2] No series produced — aborting.", file=sys.stderr)
        sys.exit(1)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f, indent=2)
    print(f"[ChinaM2] Full series -> {OUTPUT_FILE}")
    analyze(series)
    write_summary(series)


if __name__ == "__main__":
    main()
