#!/usr/bin/env python3
"""
walk_forward_trend_continuation.py — P3 institutional output.
=============================================================
Answers the framework's most valuable question:
    "Given today's signal, what is the probability the BTC trend
     CONTINUES (forward return > 0) over 1-3 months and 6 months?"

METHOD (honest, follows walk-forward skill):
    - Reconstruct the core SFC signal (price + DXY + M2 + FNG) point-in-time
      using the SAME reduced-factor replay as walk_forward_validation.py /
      walk_forward_imbs_l1l2.py. This is a reduced-set replay, NOT the live
      full 90+ method score — caveat is mandatory.
    - For each day, compute the ACTUAL forward return over 30 / 90 / 180 days.
    - Bucket by signal severity (CALM / ELEVATED / STRESS using the live
      threshold bands).
    - For each bucket, estimate P(forward return > 0) with a bootstrap CI
      (exact binomial-ish proportion, resampled). This is the empirical
      "trend continuation probability" conditioned on the signal.
    - ALSO report an unconditional baseline for reference.

The live pipeline reads the cached summary to DISPLAY a calibrated
continuation probability for the current signal bucket. Because the replay
is a reduced set, the displayed number is a RESEARCH estimate, not the
live-score probability (same precedent as STRESS=55 / L8).

USAGE:
    cd ~/sfc
    .venv/bin/python analysis/walk_forward_trend_continuation.py
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from historical_backtest_m1m6 import (
        fetch_fred_series, fetch_fng_historical_dict, _nearest_prior_value,
        score_factors_from_market, calculate_sfc_ensemble,
    )
except ImportError as e:
    print(f"[P3] Could not import historical_backtest_m1m6.py: {e}", file=sys.stderr)
    sys.exit(1)

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(SFC_ROOT, ".walk_forward_trend_continuation.json")
SUMMARY_CACHE_FILE = os.path.join(SFC_ROOT, ".trend_continuation_summary.json")

HORIZONS = [30, 90, 180]       # 1 mo, 3 mo, 6 mo
N_BOOTSTRAP = 2000
# Live threshold bands (same as zone): CALM <25, ELEVATED 25-45, STRESS >45.
BUCKET_EDGES = [(0, 25, "CALM"), (25, 45, "ELEVATED"), (45, 101, "STRESS")]


def _bucket_label(sfc_pct):
    for lo, hi, lbl in BUCKET_EDGES:
        if lo <= sfc_pct < hi:
            return lbl
    return "STRESS"


def compute_series():
    btc = fetch_fred_series("CBBTCUSD")
    dxy = fetch_fred_series("DTWEXBGS")
    m2 = fetch_fred_series("M2SL")
    fng = fetch_fng_historical_dict()
    if not btc:
        print("[P3] No BTC price data.", file=sys.stderr)
        return []
    dates = sorted(btc.keys())
    rows, prev = [], None
    for ds in dates:
        price = btc[ds]
        chg = (price - prev) / prev * 100 if prev else 0.0
        prev = price
        dxy_v = _nearest_prior_value(dxy, ds, max_lookback_days=10)
        m2v = _nearest_prior_value(m2, ds, max_lookback_days=45)
        m2_yoy = None
        if m2v:
            d = datetime.strptime(ds, "%Y-%m-%d")
            yago = (d - timedelta(days=365)).strftime("%Y-%m-%d")
            prior = _nearest_prior_value(m2, yago, max_lookback_days=45)
            if prior:
                m2_yoy = (m2v - prior) / prior * 100
        f = score_factors_from_market(btc=price, btc_24h=chg, dom=None, dvol=None,
                                      fng=fng.get(ds), pc_oi=None, m2_yoy=m2_yoy, dxy=dxy_v)
        sfc = calculate_sfc_ensemble(f)[0]
        rows.append({"date": ds, "price": price, "sfc_pct": sfc, "fng": fng.get(ds)})
    # forward returns
    n = len(rows)
    for i, r in enumerate(rows):
        for h in HORIZONS:
            j = i + h
            if j < n:
                r[f"fwd_{h}d"] = (rows[j]["price"] - r["price"]) / r["price"] * 100
            else:
                r[f"fwd_{h}d"] = None
    return rows


def bootstrap_prob(series_vals, n_bootstrap=N_BOOTSTRAP, ci=0.90):
    """P(value > 0) with bootstrap CI (resample proportion)."""
    vals = [v for v in series_vals if v is not None]
    if len(vals) < 5:
        return None, None, None, len(vals)
    p = sum(1 for v in vals if v > 0) / len(vals)
    probs = []
    for _ in range(n_bootstrap):
        s = [vals[random.randrange(len(vals))] for _ in range(len(vals))]
        probs.append(sum(1 for x in s if x > 0) / len(s))
    probs.sort()
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 + ci) / 2 * n_bootstrap) - 1
    return round(p, 3), round(probs[lo], 3), round(probs[hi], 3), len(vals)


def run_validation():
    print("=" * 66)
    print("P3 — TREND CONTINUATION PROBABILITY (walk-forward)")
    print("P(forward return > 0) conditioned on SFC signal bucket")
    print("=" * 66)
    print("Fetching historical series (a minute)...")
    rows = compute_series()
    if not rows:
        return
    with open(OUTPUT_FILE, "w") as f:
        json.dump(rows, f)
    print(f"Computed {len(rows)} daily observations")

    # Unconditional baseline
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "caveat": "Reduced-set replay (price/DXY/M2/FNG). RESEARCH estimate, "
                         "not the live full-90-method score probability."}
    for h in HORIZONS:
        p, lo, hi, n = bootstrap_prob([r.get(f"fwd_{h}d") for r in rows])
        summary[f"baseline_p_cont_{h}d"] = p
        summary[f"baseline_p_cont_{h}d_ci"] = [lo, hi] if lo is not None else None
        print(f"\n[{h}d] Unconditional P(trend continues) = {p}  [CI {lo},{hi}]  n={n}")

    # Per-bucket
    for h in HORIZONS:
        buckets = {lbl: [] for _, _, lbl in BUCKET_EDGES}
        for r in rows:
            fwd = r.get(f"fwd_{h}d")
            if fwd is None:
                continue
            buckets[_bucket_label(r["sfc_pct"])].append(fwd)
        print(f"\n[{h}d] P(trend continues) by signal bucket:")
        for _, _, lbl in BUCKET_EDGES:
            p, lo, hi, n = bootstrap_prob(buckets[lbl])
            print(f"    {lbl:<9} n={n:<5} P(cont)={p} [CI {lo},{hi}]")
            summary[f"{lbl.lower()}_p_cont_{h}d"] = p
            summary[f"{lbl.lower()}_p_cont_{h}d_ci"] = [lo, hi] if lo is not None else None
            summary[f"{lbl.lower()}_n_{h}d"] = n

    with open(SUMMARY_CACHE_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary cache -> {SUMMARY_CACHE_FILE}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    random.seed(42)
    run_validation()
