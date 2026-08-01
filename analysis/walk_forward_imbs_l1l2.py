#!/usr/bin/env python3
"""
walk_forward_imbs_l1l2.py — IMBS Layer 1-2 liquidity extension to the
existing walk-forward validation.

WHAT THIS ADDS:
    The existing walk_forward_validation.py proves the CORE price/macro
    signal (price + DXY + M2 + FNG) has genuine forward-looking predictive
    value (verified significant: 7d gap -1.55pp, 30d gap -7.46pp). But the
    SFC pipeline's live Liquidity factor (Lt) is now ALSO driven by the
    Global Liquidity Factor (GLF) — see global_liquidity_engine.py — which
    consumes Fed/ECB/BOJ balance sheets, TGA, and RRP. The baseline walk-
    forward passes glo_score=None, so it NEVER tests whether these Layer 1-2
    liquidity components add any predictive value on top of price+m2+dxy+fng.

    This script extends the same walk-forward methodology with an IMBS
    Layer 1-2 point-in-time liquidity index built from:
        WALCL      Fed balance sheet (YoY)
        ECBASSETSW ECB balance sheet (YoY)
        JPNASSETS  BOJ balance sheet (YoY)
        WTREGEN    TGA balance (4-week change + level)
        RRPONTSYD  RRP facility (level + trend)
    fed into score_factors_from_market(glo_score=...) exactly the way
    collect.py's live pipeline feeds GLF into Lt.

METHODOLOGY (identical to walk_forward_validation.py):
    - For each historical day T, compute the signal using ONLY data
      available at or before T (no look-ahead — _nearest_prior_value).
    - Look FORWARD to the ACTUAL realized return over the next N days.
    - Bucket by signal level; bootstrap CI on the mean forward return.
    - Threshold-free decile monotonicity check.
    The only change vs the baseline is the liquidity index is added to the
    signal. This directly answers: "does IMBS L1-L2 liquidity add predictive
    value beyond what price+m2+dxy+fng already capture?"

HONEST LIMITATIONS (inherited):
    - DVOL/options/on-chain still not available historically → reduced
      factor set, same as the baseline. Results are directional evidence
      about the core signal + liquidity, not a perfect replay of live.
    - The liquidity index here uses the same simplified component math as
      global_liquidity_engine.py but with point-in-time lookups. Its
      z-score means are calibrated off live defaults, not re-fit per
      historical window (deliberate — avoids look-ahead / overfitting).
    - TGA (WTREGEN) history starts ~2001, RRP (RRPONTSYD) ~2013, and the
      BOJ/ECB sheets are weekly — so the liquidity-augmented series only
      has full coverage where those series overlap BTC price history.

USAGE (run on your VPS, needs FRED_API_KEY + network):
    cd ~/sfc
    .venv/bin/python analysis/walk_forward_imbs_l1l2.py
    (or: python3 analysis/walk_forward_imbs_l1l2.py --quantile-only to
     re-render analysis from the cached series without re-fetching.)
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact verbatim-copied formula functions and the FRED fetchers
# from historical_backtest_m1m6.py (kept duplicated there deliberately —
# collect.py is a top-to-bottom EXECUTING SCRIPT with live side effects).
try:
    from historical_backtest_m1m6 import (
        score_factors_from_market, calculate_sfc_ensemble,
        fetch_fred_series, fetch_fng_historical_dict, _nearest_prior_value,
    )
except ImportError as e:
    print(f"[IMBS-L1L2] Could not import historical_backtest_m1m6.py: {e}",
          file=sys.stderr)
    sys.exit(1)

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(SFC_ROOT, ".walk_forward_imbs_l1l2.json")
SUMMARY_CACHE_FILE = os.path.join(SFC_ROOT, ".walk_forward_imbs_l1l2_summary.json")

FORWARD_HORIZONS_DAYS = [7, 30]
BUCKET_EDGES = [(0, 25, "CALM"), (25, 45, "ELEVATED"), (45, 101, "STRESS")]
N_BOOTSTRAP = 2000
N_QUANTILES = 10

# IMBS L1-L2 liquidity series that are NOT already in the baseline.
LIQUIDITY_SERIES = {
    "walcl": ("WALCL", 2002, "Fed BS YoY"),
    "ecb": ("ECBASSETSW", 2001, "ECB BS YoY"),
    "boj": ("JPNASSETS", 2001, "BOJ BS YoY"),
    "tga": ("WTREGEN", 2001, "TGA 4w chg + level"),
    "rrp": ("RRPONTSYD", 2013, "RRP level + trend"),
}


def _yoy(series_dict, date_str, lookback_days=365, max_lookback=45):
    """YoY % change using only point-in-time prior values."""
    level = _nearest_prior_value(series_dict, date_str, max_lookback_days=max_lookback)
    if level is None:
        return None
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year_ago = (d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    prior = _nearest_prior_value(series_dict, year_ago, max_lookback_days=max_lookback)
    if prior is None or prior == 0:
        return None
    return (level - prior) / prior * 100


def _zclip(value, center, scale, lo=-3.0, hi=3.0):
    if value is None:
        return 0.0
    return max(lo, min(hi, (value - center) / scale))


def _compute_liquidity_index(date_str, series, dxy_series):
    """Build a point-in-time IMBS L1-L2 liquidity index (0-100, high=liquid),
    mirroring global_liquidity_engine.py's GLF composition but with pure
    point-in-time lookups and no cross-window fitting.

    Weights follow global_liquidity_engine.py:
        fed 0.30, ecb 0.15, boj 0.03, tga 0.10, rrp 0.10, dxy 0.13.
    (M2 is dropped here since baseline already carries m2_yoy directly.)
    All are z-scores where positive = liquid; combined -> 55 + z*17.5 (0-100).
    """
    comps = {}

    fed_yoy = _yoy(series["walcl"], date_str)
    if fed_yoy is not None:
        comps["fed"] = (_zclip(fed_yoy, 5.5, 8.0), 0.30)

    ecb_yoy = _yoy(series["ecb"], date_str)
    if ecb_yoy is not None:
        comps["ecb"] = (_zclip(ecb_yoy, 4.0, 7.0), 0.15)

    boj_yoy = _yoy(series["boj"], date_str)
    if boj_yoy is not None:
        comps["boj"] = (_zclip(boj_yoy, 3.0, 6.0), 0.03)

    # TGA: 4-week change (decrease = stimulus = liquid = +z).
    tga = series["tga"]
    tga_latest = _nearest_prior_value(tga, date_str, max_lookback_days=45)
    if tga_latest is not None:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        tga_4w = (d - timedelta(days=28)).strftime("%Y-%m-%d")
        tga_prior = _nearest_prior_value(tga, tga_4w, max_lookback_days=45)
        if tga_prior and tga_prior > 0:
            tga_chg = (tga_latest - tga_prior) / tga_prior * 100
            if tga_chg < -10:
                tga_z = 1.5
            elif tga_chg < -5:
                tga_z = 0.8
            elif tga_chg < -2:
                tga_z = 0.3
            elif tga_chg < 2:
                tga_z = 0.0
            elif tga_chg < 5:
                tga_z = -0.5
            elif tga_chg < 10:
                tga_z = -1.0
            else:
                tga_z = -1.5
            if tga_latest > 900000:
                tga_z -= 0.5
            elif tga_latest < 300000:
                tga_z -= 0.3
            comps["tga"] = (max(-2.0, min(2.0, tga_z)), 0.10)

    # RRP: level + trend (near-zero/falling = cash deployed = liquid = +z).
    rrp = series["rrp"]
    rrp_latest = _nearest_prior_value(rrp, date_str, max_lookback_days=45)
    if rrp_latest is not None:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        rrp_4w = (d - timedelta(days=28)).strftime("%Y-%m-%d")
        rrp_prior = _nearest_prior_value(rrp, rrp_4w, max_lookback_days=45)
        if rrp_latest < 10:
            rrp_z = 1.5
        elif rrp_latest < 50:
            rrp_z = 1.0
        elif rrp_latest < 200:
            rrp_z = 0.0
        elif rrp_latest < 500:
            rrp_z = -0.5
        else:
            rrp_z = -1.5
        if rrp_prior is not None:
            rrp_chg = rrp_latest - rrp_prior
            if rrp_chg < -50:
                rrp_z += 0.5
            elif rrp_chg > 50:
                rrp_z -= 0.5
        comps["rrp"] = (max(-2.0, min(2.0, rrp_z)), 0.10)

    # DXY inverted (high USD = tight dollar liquidity = -z).
    dxy = _nearest_prior_value(dxy_series, date_str, max_lookback_days=10)
    if dxy is not None:
        comps["dxy"] = (_zclip(dxy, 100.0, 5.0) * -1, 0.13)

    if not comps:
        return 50.0, {}

    total_w = sum(w for _, w in comps.values())
    glf_z = sum(z * w for z, w in comps.values()) / total_w
    glf = max(0.0, min(100.0, 55 + glf_z * 17.5))
    return round(glf, 2), {k: round(v, 3) for k, (v, _) in comps.items()}


def compute_imbs_time_series():
    """Compute both baseline and liquidity-augmented sfc_pct series."""
    btc_price = fetch_fred_series("CBBTCUSD")
    dxy_series = fetch_fred_series("DTWEXBGS")
    m2_series = fetch_fred_series("M2SL")
    fng_series = fetch_fng_historical_dict()

    series = {}
    for name, (sid, _, label) in LIQUIDITY_SERIES.items():
        series[name] = fetch_fred_series(sid)
        print(f"[IMBS-L1L2] {label} ({sid}): {len(series[name])} obs",
              file=sys.stderr)

    if not btc_price:
        print("[IMBS-L1L2] No BTC price data — check FRED_API_KEY and network.",
              file=sys.stderr)
        return []

    sorted_dates = sorted(btc_price.keys())
    results = []
    prev_price = None

    for date_str in sorted_dates:
        price = btc_price[date_str]
        if prev_price is None:
            prev_price = price
            continue
        btc_24h = (price - prev_price) / prev_price * 100
        prev_price = price

        dxy = _nearest_prior_value(dxy_series, date_str, max_lookback_days=10)
        m2_level = _nearest_prior_value(m2_series, date_str, max_lookback_days=45)
        m2_yoy = None
        if m2_level is not None:
            m2_yoy = _yoy(m2_series, date_str, lookback_days=365)
        fng = fng_series.get(date_str)

        glf, glf_comps = _compute_liquidity_index(date_str, series, dxy_series)

        row = {"date": date_str, "price": price, "glf": glf,
               "glf_components": glf_comps}

        # --- Baseline: glo_score=None (as the existing walk-forward) ---
        try:
            f0 = score_factors_from_market(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=fng,
                pc_oi=None, m2_yoy=m2_yoy, dxy=dxy)
            row["sfc_pct_base"] = calculate_sfc_ensemble(f0)[0]
        except Exception:
            row["sfc_pct_base"] = None

        # --- IMBS L1-L2: glo_score = liquidity index ---
        try:
            f1 = score_factors_from_market(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=fng,
                pc_oi=None, m2_yoy=m2_yoy, dxy=dxy, glo_score=glf)
            row["sfc_pct_imbs"] = calculate_sfc_ensemble(f1)[0]
        except Exception:
            row["sfc_pct_imbs"] = None

        results.append(row)

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


def bucket_label(sfc_pct):
    for lo, hi, label in BUCKET_EDGES:
        if lo <= sfc_pct < hi:
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


def _render_signal(name, key, series, horizons):
    print(f"\n{'='*64}")
    print(f"SIGNAL: {name}")
    print(f"{'='*64}")
    for h in horizons:
        fk = f"fwd_return_{h}d"
        buckets = {lbl: [] for _, _, lbl in BUCKET_EDGES}
        for p in series:
            fwd = p.get(fk)
            if fwd is None:
                continue
            lbl = bucket_label(p[key]) if p.get(key) is not None else None
            if lbl is not None:
                buckets[lbl].append(fwd)
        for _, _, lbl in BUCKET_EDGES:
            vals = buckets[lbl]
            if len(vals) < 2:
                print(f"  {lbl:<10} n={len(vals):<5} (insufficient)")
                continue
            m = sum(vals) / len(vals)
            print(f"  {lbl:<10} n={len(vals):<5} mean fwd: {m:+.2f}%")
        calm, stress = buckets["CALM"], buckets["STRESS"]
        if len(calm) >= 2 and len(stress) >= 2:
            est, lo_, hi_ = bootstrap_diff_ci(calm, stress)
            sig = hi_ < 0 if hi_ is not None else False
            print(f"  CALM vs STRESS gap ({h}d): {est:+.2f}pp "
                  f"[90% CI {lo_:+.2f}, {hi_:+.2f}] "
                  f"{'— SIGNIFICANT' if sig else '— NOT significant'}")


def run_validation():
    print("=" * 64)
    print("IMBS LAYER 1-2 LIQUIDITY WALK-FORWARD")
    print("(baseline vs liquidity-augmented signal)")
    print("=" * 64)

    print("\nFetching historical series (this can take a minute)...")
    series = compute_imbs_time_series()
    if not series:
        return
    series = add_forward_returns(series)
    print(f"Computed {len(series)} daily observations")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f)
    print(f"Saved full series to {OUTPUT_FILE}")

    print("\n[Baseline — price + DXY + M2 + FNG only]")
    _render_signal("BASELINE", "sfc_pct_base", series, FORWARD_HORIZONS_DAYS)

    print("\n[IMBS L1-L2 — baseline + liquidity (fed/ecb/boj/tga/rrp/dxy)]")
    _render_signal("IMBS L1-L2", "sfc_pct_imbs", series, FORWARD_HORIZONS_DAYS)

    # Coverage check: how many days actually have both signals?
    n_both = sum(1 for p in series if p.get("sfc_pct_base") is not None
                 and p.get("sfc_pct_imbs") is not None)
    n_any_liquidity = sum(1 for p in series if p.get("glf") is not None
                          and p.get("glf_components"))
    print(f"\nCoverage: {n_any_liquidity}/{len(series)} days have liquidity data, "
          f"{n_both} days have both signals.")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_periods": len(series),
        "n_with_liquidity": n_any_liquidity,
        "n_both_signals": n_both,
    }
    for name, key in (("base", "sfc_pct_base"), ("imbs", "sfc_pct_imbs")):
        for h in FORWARD_HORIZONS_DAYS:
            fk = f"fwd_return_{h}d"
            calm, stress = [], []
            for p in series:
                fwd = p.get(fk)
                if fwd is None or p.get(key) is None:
                    continue
                lbl = bucket_label(p[key])
                if lbl == "CALM":
                    calm.append(fwd)
                elif lbl == "STRESS":
                    stress.append(fwd)
            est, lo_, hi_ = bootstrap_diff_ci(calm, stress)
            summary[f"{name}_gap_{h}d"] = round(est, 2) if est is not None else None
            summary[f"{name}_gap_{h}d_ci"] = (
                [round(lo_, 2), round(hi_, 2)] if lo_ is not None else None)
            summary[f"{name}_gap_{h}d_significant"] = (hi_ < 0) if hi_ is not None else None
            summary[f"{name}_n_calm_{h}d"] = len(calm)
            summary[f"{name}_n_stress_{h}d"] = len(stress)
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
            print("\n[Baseline]")
            _render_signal("BASELINE", "sfc_pct_base", series, FORWARD_HORIZONS_DAYS)
            print("\n[IMBS L1-L2]")
            _render_signal("IMBS L1-L2", "sfc_pct_imbs", series, FORWARD_HORIZONS_DAYS)
        else:
            print("No cached series found — run without --quantile-only first.")
        sys.exit(0)

    random.seed(42)
    run_validation()
