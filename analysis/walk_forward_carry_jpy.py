#!/usr/bin/env python3
"""
walk_forward_carry_jpy.py — Walk-forward test of the JAPANESE CARRY TRADE
spread as a predictive signal for BTC, as an incremental-contribution check.

WHY
    Question: does the JPY carry trade affect GLF liquidity and BTC?
    Answer so far (code audit): GLF weights are FIXED constants; carry enters
    only via BOJ balance sheet (3%) and JPY inside DXY (0.136 exponent, 13%),
    so a 10% yen move shifts GLF ~0.5pt. There is NO explicit carry model.
    This script tests the real question empirically: does the US-JP interest
    rate differential (the carry trade's funding condition) predict BTC
    forward returns out-of-sample, and is the edge era-stable?

SIGNAL
    carry_spread = US 10Y yield (DGS10, daily) - JP long yield
                   (IRLTLT01JPM156N, monthly). Reconstructed POINT-IN-TIME:
                   each day uses the most recent PRIOR JP yield value — never
                   future data. Higher spread = more carry available (yen weak,
                   risk-on). Point-in-time z-score (expanding window, no
                   look-ahead) as the test statistic.

METHOD (per walk-forward-validation skill)
    1. Build daily carry spread series point-in-time.
    2. Point-in-time expanding-window z-score (min warmup) — no look-ahead.
    3. Forward BTC return (CBBTCUSD) over 30/90/180 days.
    4. Top-vs-bottom tail gap (direct bootstrap of difference, 90% CI,
       two-tailed).
    5. Era split (2015-2020 vs 2021-2026) to check era stability.
    6. Confound: date & price range per bucket.

HONEST LIMIT
    Carry spread is a FUNDING-condition proxy, not a direct carry-unwind
    (leverage-deleveraging) measure. A weak/null result here does not prove
    carry trade is irrelevant — it proves this particular rate-differential
    proxy alone lacks a clean standalone predictive edge. Report as a research
    recommendation, not a deployable cutoff (skill Pitfall 14).
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

OUTPUT_FILE = os.path.join(SFC_DIR, ".walk_forward_carry_jpy.json")
SUMMARY_FILE = os.path.join(SFC_DIR, ".walk_forward_carry_jpy_summary.json")

N_BOOTSTRAP = 2000
CI = 0.90
WARMUP_DAYS = 250      # ~1 year of daily data before first point-in-time z
TAIL = 0.25
HORIZONS_DAYS = [30, 90, 180]


def fetch_fred_map(series_id):
    """Fetch full FRED series -> {date_str: float}. Reuses historical_backtest."""
    from historical_backtest_m1m6 import fetch_fred_series
    return fetch_fred_series(series_id)


def nearest_prior(series_map, target_date, max_lookback_days=45):
    """Most recent value at-or-before target_date within lookback."""
    from datetime import timedelta
    d = datetime.strptime(target_date, "%Y-%m-%d")
    for i in range(max_lookback_days + 1):
        cand = (d - timedelta(days=i)).strftime("%Y-%m-%d")
        if cand in series_map:
            return series_map[cand]
    return None


def build_series():
    us10 = fetch_fred_map("DGS10")
    jp10 = fetch_fred_map("IRLTLT01JPM156N")
    btc = fetch_fred_map("CBBTCUSD")
    if not us10 or not jp10 or not btc:
        raise RuntimeError("Missing FRED data (DGS10/IRLTLT01JPM156N/CBBTCUSD) — check FRED_API_KEY/network.")

    btc_dates = sorted(btc.keys())
    # Build daily carry spread point-in-time
    rows = []  # (date, carry)
    for d in btc_dates:
        us = us10.get(d)
        if us is None:
            us = nearest_prior(us10, d, 10)
        if us is None:
            continue
        jp = nearest_prior(jp10, d, 60)
        if jp is None:
            continue
        rows.append((d, us - jp))
    if len(rows) < WARMUP_DAYS + 50:
        raise RuntimeError(f"Too few carry points: {len(rows)}")

    # Point-in-time expanding z-score
    series = []
    carries = [c for _, c in rows]
    for i, (d, c) in enumerate(rows):
        if i < WARMUP_DAYS:
            continue
        hist = carries[:i]
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
        row = {"date": d, "carry": round(c, 3), "z": round(z, 3), "price": price}
        # forward returns by index in btc_dates
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
    print("WALK-FORWARD TEST — JPY carry spread (US10Y - JP10Y) vs BTC forward return")
    print("=" * 72)
    print(f"Points: {len(series)}  | range {series[0]['date']} .. {series[-1]['date']}")
    print("Note: higher carry z = high US-JP spread (risk-on / weak yen).")

    for h in HORIZONS_DAYS:
        key = f"fwd_{h}d"
        pts = [p for p in series if p.get(key) is not None]
        if len(pts) < 10:
            print(f"\n--- {h}d: insufficient ({len(pts)}) ---")
            continue
        zs = sorted(pts, key=lambda p: p["z"])
        n = len(zs)
        tn = max(3, int(n * TAIL))
        low = zs[:tn]    # low spread (risk-off)
        high = zs[-tn:]  # high spread (risk-on)
        lr = [p[key] for p in low]
        hr = [p[key] for p in high]
        lm, hm = statistics.mean(lr), statistics.mean(hr)
        est, lo, hi = bootstrap_diff_ci(lr, hr, seed=42)
        sig = (lo is not None) and (hi is not None) and (lo > 0 or hi < 0)
        gap = hm - lm  # positive = high carry -> higher return (risk-on premium)
        print(f"\n--- {h}d forward return ---")
        print(f"  Low  carry (risk-off): n={len(low):3d} mean {lm:+7.2f}%  z<= {low[-1]['z']:.2f}")
        print(f"  High carry (risk-on) : n={len(high):3d} mean {hm:+7.2f}%  z>= {high[0]['z']:.2f}")
        if est is not None:
            print(f"  Gap (high-low): {gap:+6.2f}pp  bootstrap {'SIGNIFICANT' if sig else 'n.s.'}  "
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
            print(f"  {en}: n={len(pts):3d}  gap(high-low)={gap:+6.2f}pp  {'SIG' if sig else 'n.s.'}  "
                  f"[{est:+.2f} ({lo:+.2f}, {hi:+.2f})]")
        else:
            print(f"  {en}: n={len(pts):3d} gap={gap:+.2f}pp insufficient")

    print("\n" + "=" * 72)
    print("CONFOUND — date & BTC price range by carry bucket (90d)")
    print("=" * 72)
    zs = sorted([p for p in series if p.get("fwd_90d") is not None], key=lambda p: p["z"])
    if zs:
        n = len(zs)
        tn = max(3, int(n * TAIL))
        for label, grp in [("LOW (risk-off)", zs[:tn]), ("MID", zs[tn:-tn]), ("HIGH (risk-on)", zs[-tn:])]:
            if not grp:
                continue
            ds = [p["date"] for p in grp]
            ps = [p["price"] for p in grp]
            print(f"  {label:<16} n={len(grp):3d}  {min(ds)}..{max(ds)}  btc ${min(ps):,.0f}..${max(ps):,.0f}")


def write_summary(series):
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "n_points": len(series),
               "method": "point-in-time expanding z of US10Y-JP10Y spread, no look-ahead"}
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
    print(f"\n[CarryJPY] Summary -> {SUMMARY_FILE}")


def main():
    random.seed(42)
    series = build_series()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f, indent=2)
    print(f"[CarryJPY] Full series -> {OUTPUT_FILE}")
    analyze(series)
    write_summary(series)


if __name__ == "__main__":
    main()
