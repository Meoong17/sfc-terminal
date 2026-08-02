#!/usr/bin/env python3
"""
walk_forward_imbs_l8.py — IMBS Layer 8 (Tail Risk) component validation.

WHAT THIS VALIDATES (honest scope — per user decision, Option 1):
    L8 Tail Risk = geometric_mean(liquidity, behavior, expectation,
                                   leverage, correlation) * amplifier.
    Only TWO of the five dimensions have long free history via FRED:
        - liquidity_stress   : from GLF (Fed/ECB/BOJ balance sheets, TGA,
                               RRP, DXY) -> high GLF = liquid = LOW stress.
        - expectation_shock  : L6 proxy (T10YIE - CPI YoY, real rate, curve,
                               unemployment) -> high = fragile expectations.
    behavior_stress, leverage, correlation depend on MPI/ETF/whale/cascade
    /funding data that only exist ~2021+, so they CANNOT be reconstructed
    for a long walk-forward.

    Therefore this script validates whether the RECONSTRUCTABLE L8 subset
    (liquidity + expectation) has genuine out-of-sample predictive power
    against forward BTC returns. It does NOT re-calibrate the live L8
    cutoff — the live signal uses 4-5 active dimensions and has a different
    distribution, so a cutoff calibrated here is NOT portable (same caveat
    as the IMBS STRESS=55 research recommendation).

METHODOLOGY (identical to walk_forward_imbs_l1l2.py / skill):
    - Point-in-time reconstruction (no look-ahead, _nearest_prior_value).
    - Forward returns over N days from actual BTC price.
    - Fixed-threshold bucket analysis + bootstrap CI on the CALM-vs-STRESS
      gap (mirroring the live severity bands).
    - Threshold-free quantile (top vs bottom 20%) comparison, which is
      robust to the compressed distribution of a reduced factor set.
    - Time-period confound check (year distribution per bucket).

USAGE (run on VPS, needs FRED_API_KEY + network):
    cd ~/sfc
    .venv/bin/python analysis/walk_forward_imbs_l8.py
    (--quantile-only to re-render analysis from cached series)
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from historical_backtest_m1m6 import (
        fetch_fred_series, _nearest_prior_value,
    )
except ImportError as e:
    print(f"[IMBS-L8] Could not import historical_backtest_m1m6.py: {e}",
          file=sys.stderr)
    sys.exit(1)

# Reuse the point-in-time GLF liquidity reconstruction from the validated
# L1-L2 walk-forward (same component math, no look-ahead).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from walk_forward_imbs_l1l2 import _compute_liquidity_index
except ImportError:
    _compute_liquidity_index = None

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(SFC_ROOT, ".walk_forward_imbs_l8.json")
SUMMARY_CACHE_FILE = os.path.join(SFC_ROOT, ".walk_forward_imbs_l8_summary.json")

FORWARD_HORIZONS_DAYS = [7, 30]
N_BOOTSTRAP = 2000
QUANTILE_TAIL = 0.20

# Severity bands mirroring tail_risk_engine._severity (LOW/MODERATE/ELEVATED/
# HIGH/CRITICAL). We collapse to 3 buckets for sufficient n: LOW-MODERATE
# (<40), ELEVATED (40-60), HIGH-CRITICAL (>=60).
BUCKET_EDGES = [(0, 40, "LOW-MOD"), (40, 60, "ELEVATED"), (60, 101, "HIGH")]

# LIQUIDITY_SERIES (Fed/ECB/BOJ/TGA/RRP) as in the L1-L2 walk-forward.
LIQUIDITY_SERIES = {
    "walcl": ("WALCL", 2002, "Fed BS YoY"),
    "ecb": ("ECBASSETSW", 2001, "ECB BS YoY"),
    "boj": ("JPNASSETS", 2001, "BOJ BS YoY"),
    "tga": ("WTREGEN", 2001, "TGA 4w chg + level"),
    "rrp": ("RRPONTSYD", 2013, "RRP level + trend"),
}


# ── Point-in-time expectation_shock (L6 subset) ──────────────────────────
def _yoy_pt(series, date_str, lookback_days=365, max_lookback=45):
    """Point-in-time YoY % change of a level series (e.g. CPI)."""
    level = _nearest_prior_value(series, date_str, max_lookback_days=max_lookback)
    if level is None:
        return None
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year_ago = (d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    prior = _nearest_prior_value(series, year_ago, max_lookback_days=max_lookback)
    if prior is None or prior == 0:
        return None
    return (level - prior) / prior * 100


def _slope_to_stress(yoy_gap):
    """Mirror expectations_engine._slope_to_stress (verbatim)."""
    g = yoy_gap if yoy_gap is not None else 0.0
    if g <= -2.5:
        return 85.0
    if g <= -1.0:
        return 65.0
    if g <= 0.0:
        return 50.0
    if g <= 1.0:
        return 35.0
    if g <= 2.5:
        return 45.0
    if g <= 4.0:
        return 60.0
    return 75.0


def _expectation_shock(date_str, series):
    """Compute the L6 expectation stress (0-100) point-in-time from the
    reconstructable FRED series. Mirrors compute_expectations() but with
    point-in-time lookups (no look-ahead, no module cache)."""
    cpi_yoy = _yoy_pt(series["cpi"], date_str, lookback_days=365)
    t10yie = _nearest_prior_value(series["t10yie"], date_str, max_lookback_days=10)
    dgs10 = _nearest_prior_value(series["dgs10"], date_str, max_lookback_days=10)
    curve = _nearest_prior_value(series["t10y2y"], date_str, max_lookback_days=10)
    unrate = _nearest_prior_value(series["unrate"], date_str, max_lookback_days=45)

    infl_gap = (t10yie - cpi_yoy) if (t10yie is not None and cpi_yoy is not None) else None
    real_rate = (dgs10 - t10yie) if (dgs10 is not None and t10yie is not None) else None

    comps = []
    if infl_gap is not None:
        comps.append(_slope_to_stress(infl_gap))
    if real_rate is not None:
        comps.append(25.0 if real_rate < 1.0 else 50.0 if real_rate < 2.0 else 65.0 if real_rate < 3.0 else 80.0)
    if curve is not None:
        comps.append(80.0 if curve < 0 else 45.0 if curve < 0.5 else 30.0)
    if unrate is not None:
        comps.append(75.0 if unrate >= 6.5 else 55.0 if unrate >= 5.5 else 40.0 if unrate >= 4.0 else 30.0)
    if not comps:
        return None
    return sum(comps) / len(comps)


# ── L8-subset composite (mirror compute_tail_risk, 2 active dims) ────────
def _geometric_mean(dims):
    """Geometric-mean combination over the ACTIVE (non-None) normalized
    [0,100] dims. Missing dims are excluded (not forced to neutral 50, since
    here only liquidity+expectation are ever supplied)."""
    active = [d for d in dims if d is not None]
    if not active:
        return None
    prod = 1.0
    for d in active:
        prod *= max(0.0, min(100.0, d)) / 100.0
    return (prod ** (1.0 / len(active))) * 100.0


# ── Time series build ─────────────────────────────────────────────────────
def compute_l8_time_series():
    btc_price = fetch_fred_series("CBBTCUSD")
    dxy_series = fetch_fred_series("DTWEXBGS")

    series = {}
    for name, (sid, _, label) in LIQUIDITY_SERIES.items():
        series[name] = fetch_fred_series(sid)
        print(f"[IMBS-L8] {label} ({sid}): {len(series[name])} obs", file=sys.stderr)

    for key, sid in (("cpi", "CPIAUCSL"), ("t10yie", "T10YIE"),
                     ("dgs10", "DGS10"), ("t10y2y", "T10Y2Y"),
                     ("unrate", "UNRATE")):
        series[key] = fetch_fred_series(sid)
        print(f"[IMBS-L8] {key} ({sid}): {len(series[key])} obs", file=sys.stderr)

    if not btc_price:
        print("[IMBS-L8] No BTC price data — check FRED_API_KEY/network.", file=sys.stderr)
        return []

    sorted_dates = sorted(btc_price.keys())
    results = []
    prev_price = None

    for date_str in sorted_dates:
        price = btc_price[date_str]
        if prev_price is None:
            prev_price = price
            continue
        results.append({
            "date": date_str,
            "price": price,
            "btc_24h": (price - prev_price) / prev_price * 100,
        })
        prev_price = price

    # Second pass: compute signals point-in-time (needs forward-independent
    # lookups only).
    for row in results:
        date_str = row["date"]
        glf, _ = _compute_liquidity_index(date_str, series, dxy_series) if _compute_liquidity_index else (None, {})
        liq_stress = (100.0 - glf) if glf is not None else None
        exp_shock = _expectation_shock(date_str, series)

        row["liquidity_stress"] = round(liq_stress, 2) if liq_stress is not None else None
        row["expectation_shock"] = round(exp_shock, 2) if exp_shock is not None else None

        composite = _geometric_mean([liq_stress, exp_shock])
        row["l8_subset"] = round(composite, 2) if composite is not None else None

    return results


def add_forward_returns(series):
    n = len(series)
    for i, point in enumerate(series):
        for h in FORWARD_HORIZONS_DAYS:
            fi = i + h
            if fi < n:
                fut = series[fi]["price"]
                point[f"fwd_return_{h}d"] = (fut - point["price"]) / point["price"] * 100
            else:
                point[f"fwd_return_{h}d"] = None
    return series


# ── Analysis ──────────────────────────────────────────────────────────────
def bucket_label(sig):
    for lo, hi, label in BUCKET_EDGES:
        if lo <= sig < hi:
            return label
    return BUCKET_EDGES[-1][2]


def bootstrap_diff_ci(group_a, group_b, n_bootstrap=N_BOOTSTRAP, ci=0.90):
    if len(group_a) < 2 or len(group_b) < 2:
        return None, None, None
    na, nb = len(group_a), len(group_b)
    diffs = []
    for _ in range(n_bootstrap):
        sa = [group_a[random.randrange(na)] for _ in range(na)]
        sb = [group_b[random.randrange(nb)] for _ in range(nb)]
        diffs.append(sum(sb) / nb - sum(sa) / na)
    diffs.sort()
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 + ci) / 2 * n_bootstrap) - 1
    return sum(group_b) / nb - sum(group_a) / na, diffs[lo], diffs[hi]


def _render_signal(series, horizons):
    for h in horizons:
        fk = f"fwd_return_{h}d"
        buckets = {lbl: [] for _, _, lbl in BUCKET_EDGES}
        for p in series:
            fwd = p.get(fk)
            if fwd is None or p.get("l8_subset") is None:
                continue
            buckets[bucket_label(p["l8_subset"])].append(fwd)
        print(f"\n  [{h}d forward]")
        for _, _, lbl in BUCKET_EDGES:
            vals = buckets[lbl]
            if len(vals) < 2:
                print(f"    {lbl:<10} n={len(vals):<5} (insufficient)")
                continue
            print(f"    {lbl:<10} n={len(vals):<5} mean fwd: {sum(vals)/len(vals):+.2f}%")

        low, high = buckets["LOW-MOD"], buckets["HIGH"]
        if len(low) >= 2 and len(high) >= 2:
            est, lo_, hi_ = bootstrap_diff_ci(low, high)
            sig = (hi_ < 0 or lo_ > 0) if (hi_ is not None and lo_ is not None) else False
            print(f"    LOW-MOD vs HIGH gap ({h}d): {est:+.2f}pp "
                  f"[90% CI {lo_:+.2f}, {hi_:+.2f}] "
                  f"{'— SIGNIFICANT' if sig else '— NOT significant'}")


def _quantile_analysis(series, horizons):
    """Threshold-free top-20% vs bottom-20% by actual distribution."""
    print("\n" + "=" * 64)
    print("QUANTILE (bottom 20% vs top 20% of actual l8_subset distribution)")
    print("=" * 64)
    for h in horizons:
        fk = f"fwd_return_{h}d"
        pts = [(p["l8_subset"], p.get(fk)) for p in series
               if p.get("l8_subset") is not None and p.get(fk) is not None]
        if len(pts) < 20:
            print(f"  [{h}d] insufficient ({len(pts)})")
            continue
        pts.sort(key=lambda x: x[0])
        n = len(pts)
        tail_n = int(n * QUANTILE_TAIL)
        bottom = [v for _, v in pts[:tail_n]]   # lowest l8 (least stress)
        top = [v for _, v in pts[-tail_n:]]     # highest l8 (most stress)
        mb = sum(bottom) / len(bottom)
        mt = sum(top) / len(top)
        est, lo_, hi_ = bootstrap_diff_ci(bottom, top)
        sig = (hi_ < 0 or lo_ > 0) if (hi_ is not None and lo_ is not None) else False
        print(f"  [{h}d] bottom(low stress) n={len(bottom)} mean={mb:+.2f}% | "
              f"top(high stress) n={len(top)} mean={mt:+.2f}%")
        print(f"        bottom−top gap: {est:+.2f}pp [90% CI {lo_:+.2f},{hi_:+.2f}] "
              f"{'— SIGNIFICANT' if sig else '— NOT significant'}")


def _confound_check(series):
    """Year distribution per l8_subset bucket (fixed-threshold)."""
    print("\n" + "=" * 64)
    print("CONFOUND CHECK — year distribution per bucket")
    print("=" * 64)
    buckets = {lbl: [] for _, _, lbl in BUCKET_EDGES}
    for p in series:
        if p.get("l8_subset") is None:
            continue
        buckets[bucket_label(p["l8_subset"])].append(p["date"][:4])
    for _, _, lbl in BUCKET_EDGES:
        yrs = buckets[lbl]
        from collections import Counter
        c = Counter(yrs)
        top = ", ".join(f"{y}:{c[y]}" for y in sorted(c, key=lambda y: -c[y])[:6])
        print(f"  {lbl:<10} n={len(yrs):<5} top years: {top}")


def run_validation():
    print("=" * 64)
    print("IMBS LAYER 8 (TAIL RISK) — reconstructable-subset WALK-FORWARD")
    print("(liquidity_stress GLF + expectation_shock L6; behavior/leverage/")
    print(" correlation not reconstructable pre-2021)")
    print("=" * 64)

    print("\nFetching historical series (this can take a minute)...")
    series = compute_l8_time_series()
    if not series:
        return
    series = add_forward_returns(series)
    print(f"Computed {len(series)} daily observations")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f)
    print(f"Saved full series to {OUTPUT_FILE}")

    n_comp = sum(1 for p in series if p.get("l8_subset") is not None)
    print(f"\nCoverage: {n_comp}/{len(series)} days have full L8-subset signal.")

    print("\n[Fixed-threshold buckets by l8_subset]")
    _render_signal(series, FORWARD_HORIZONS_DAYS)

    _quantile_analysis(series, FORWARD_HORIZONS_DAYS)
    _confound_check(series)

    # Summary cache
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version_check": "L8 subset = GLF liquidity stress + L6 expectation shock (2/5 dims)",
        "n_periods": len(series),
        "n_with_l8": n_comp,
        "caveat": "Reduced 2-dim L8 subset (behavior/leverage/correlation unavailable pre-2021). "
                  "Cutoff NOT portable to live 4-5 dim signal.",
    }
    for h in FORWARD_HORIZONS_DAYS:
        fk = f"fwd_return_{h}d"
        low, high = [], []
        for p in series:
            if p.get(fk) is None or p.get("l8_subset") is None:
                continue
            lbl = bucket_label(p["l8_subset"])
            if lbl == "LOW-MOD":
                low.append(p[fk])
            elif lbl == "HIGH":
                high.append(p[fk])
        est, lo_, hi_ = bootstrap_diff_ci(low, high)
        summary[f"bucket_gap_{h}d"] = round(est, 2) if est is not None else None
        summary[f"bucket_gap_{h}d_ci"] = [round(lo_, 2), round(hi_, 2)] if lo_ is not None else None
        summary[f"bucket_gap_{h}d_significant"] = (hi_ < 0 or lo_ > 0) if (hi_ is not None and lo_ is not None) else None
        summary[f"n_low_{h}d"] = len(low)
        summary[f"n_high_{h}d"] = len(high)

    with open(SUMMARY_CACHE_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary cache -> {SUMMARY_CACHE_FILE}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--quantile-only" in sys.argv:
        if os.path.exists(OUTPUT_FILE):
            print("Loading cached series ...")
            with open(OUTPUT_FILE) as f:
                series = json.load(f)
            series = add_forward_returns(series)
            print(f"Loaded {len(series)} observations")
            _render_signal(series, FORWARD_HORIZONS_DAYS)
            _quantile_analysis(series, FORWARD_HORIZONS_DAYS)
            _confound_check(series)
        else:
            print("No cached series found — run without --quantile-only first.")
        sys.exit(0)

    random.seed(42)
    run_validation()
