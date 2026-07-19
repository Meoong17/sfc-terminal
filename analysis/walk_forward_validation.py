#!/usr/bin/env python3
"""
walk_forward_validation.py — Genuine walk-forward predictive validation
for the SFC core ensemble, extending historical_backtest_m1m6.py.

WHY THIS IS DIFFERENT FROM historical_backtest_m1m6.py:
    That script asks "does the core ensemble math RESPOND to known
    historical crashes" (it does — verified 14-16x score increase during
    calm-vs-crisis synthetic tests, and real elevation during the 2018/
    COVID/2022 windows). This script asks the more important, still-open
    question: "does the sfc_pct SIGNAL have genuine forward-looking
    predictive value" — i.e., when the signal reads HIGH at time T, is
    the ACTUAL SUBSEQUENT price return (which the model could not have
    seen) meaningfully worse than when the signal reads LOW? That is
    the walk-forward validation this project's bt_label has been
    honestly flagged as still lacking ("ESTIMATED, not walk-forward
    validated") throughout this whole audit process.

METHODOLOGY (standard walk-forward validation, not walk-forward
OPTIMIZATION — this project's M1-M6 weights are fixed formulas, not
fitted parameters, so there is nothing to "re-fit" each window; what
IS being validated is whether the fixed formula's output correlates
with genuine OUT-OF-SAMPLE forward returns):
    1. For every historical day T with a computed sfc_pct (using only
       data available up to and including T — no look-ahead, since the
       underlying calculate_sfc_ensemble() formula only ever consumes
       point-in-time inputs)
    2. Look FORWARD (not recompute anything) to the ACTUAL realized
       price return over the next N days — this is a fact that already
       happened, not a projection
    3. Bucket each day by its sfc_pct level (using the SAME thresholds
       paper_trader.py already uses for BUY/SELL/HOLD: <25 calm, 25-45
       elevated, >=45 stress)
    4. Compare average forward returns across buckets, with BOOTSTRAP
       resampling for confidence intervals — necessary because genuine
       stress-level days are RARE in the available history (this
       project's own live data_collection.json audit found a real
       window with ZERO stress-labeled observations), so a single point
       estimate could easily be noise rather than signal.

HONEST LIMITATIONS (inherited from historical_backtest_m1m6.py, not
new to this script):
    - Uses the SIMPLIFIED factor set (price + DXY + M2 + FNG) since DVOL/
      options/on-chain data isn't available for most of 2014-present —
      see that script's own docstring for the full explanation. This
      means the sfc_pct computed HERE is systematically less complete
      than what collect.py produces live today; treat results as
      directional evidence about the CORE price/macro-driven signal,
      not a perfect replay of the current full system.
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the exact same verbatim-copied formula functions as
# historical_backtest_m1m6.py (kept duplicated deliberately rather than
# imported, for the same reason that script gives: collect.py is a
# top-to-bottom EXECUTING SCRIPT with live side effects, unsafe to
# import from directly).
try:
    from historical_backtest_m1m6 import (
        score_factors_from_market, calculate_sfc_ensemble,
        fetch_fred_series, fetch_fng_historical_dict, _nearest_prior_value,
    )
except ImportError:
    print("[WalkForward] Could not import historical_backtest_m1m6.py — "
          "make sure it's in the same directory (analysis/).", file=sys.stderr)
    sys.exit(1)

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".walk_forward_validation.json")

FORWARD_HORIZONS_DAYS = [7, 30]  # matches this project's own signal-generation
                                   # horizon discussion (paper_trader.py's
                                   # 6-hour prediction horizon is much shorter
                                   # than these; 7/30 days chosen here to have
                                   # enough historical resolution given
                                   # FRED CBBTCUSD is a DAILY series, not
                                   # intraday — can't validate at the same
                                   # granularity collect.py actually operates
                                   # at without intraday historical data)

# Same regime buckets as paper_trader.py's own BUY/SELL/HOLD thresholds —
# reusing these (rather than inventing new bucket boundaries) so this
# validation directly answers "does the threshold this project ALREADY
# uses for trading decisions have genuine forward-looking meaning."
BUCKET_EDGES = [(0, 25, "CALM"), (25, 45, "ELEVATED"), (45, 101, "STRESS")]

N_BOOTSTRAP = 2000  # resamples for confidence interval estimation


def compute_sfc_time_series():
    """Reproduces historical_backtest_m1m6.py's core computation loop,
    but returns the full (date, price, sfc_pct) series rather than just
    printing a crisis-window summary — this script needs the complete
    series to compute forward returns."""
    btc_price = fetch_fred_series("CBBTCUSD")
    dxy_series = fetch_fred_series("DTWEXBGS")
    m2_series = fetch_fred_series("M2SL")
    fng_series = fetch_fng_historical_dict()

    if not btc_price:
        print("[WalkForward] No BTC price data — check FRED_API_KEY and network access.", file=sys.stderr)
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
            d = datetime.strptime(date_str, "%Y-%m-%d")
            year_ago_str = (d - timedelta(days=365)).strftime("%Y-%m-%d")
            m2_year_ago = _nearest_prior_value(m2_series, year_ago_str, max_lookback_days=45)
            if m2_year_ago:
                m2_yoy = (m2_level - m2_year_ago) / m2_year_ago * 100
        fng = fng_series.get(date_str)

        try:
            factors = score_factors_from_market(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=fng,
                pc_oi=None, m2_yoy=m2_yoy, dxy=dxy,
            )
            sfc_pct = calculate_sfc_ensemble(factors)[0]
        except Exception:
            continue

        results.append({"date": date_str, "price": price, "sfc_pct": sfc_pct})

    return results


def add_forward_returns(series, horizons=FORWARD_HORIZONS_DAYS):
    """For each point, look up the price N days later (by index, since
    this is a daily series with no gaps expected from FRED) and compute
    the forward return. Points too close to the end of the series (no
    future data available yet) get None — excluded from analysis, not
    treated as zero."""
    n = len(series)
    for i, point in enumerate(series):
        for h in horizons:
            future_idx = i + h
            if future_idx < n:
                future_price = series[future_idx]["price"]
                point[f"fwd_return_{h}d"] = (future_price - point["price"]) / point["price"] * 100
            else:
                point[f"fwd_return_{h}d"] = None
    return series


def bucket_label(sfc_pct):
    for lo, hi, label in BUCKET_EDGES:
        if lo <= sfc_pct < hi:
            return label
    return BUCKET_EDGES[-1][2]


def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.90):
    """Simple percentile bootstrap for a confidence interval on the mean
    — appropriate here since we're NOT assuming normality (forward
    returns during rare stress events are exactly the kind of fat-tailed
    data where a naive t-test confidence interval could be misleading)."""
    if len(values) < 2:
        return None, None, None
    means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap) - 1
    point_estimate = sum(values) / n
    return point_estimate, means[lo_idx], means[hi_idx]


def run_validation():
    print("=" * 60)
    print("WALK-FORWARD VALIDATION — SFC Core Ensemble")
    print("=" * 60)

    print("\nComputing historical sfc_pct time series (this can take a while)...")
    series = compute_sfc_time_series()
    if not series:
        return
    print(f"Computed {len(series)} daily observations")

    series = add_forward_returns(series)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(series, f)
    print(f"Saved full time series to {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("FORWARD RETURN BY SIGNAL BUCKET (with 90% bootstrap CI)")
    print("=" * 60)

    for horizon in FORWARD_HORIZONS_DAYS:
        print(f"\n--- {horizon}-day forward return ---")
        buckets = {label: [] for _, _, label in BUCKET_EDGES}
        for point in series:
            fwd = point.get(f"fwd_return_{horizon}d")
            if fwd is None:
                continue
            label = bucket_label(point["sfc_pct"])
            buckets[label].append(fwd)

        for _, _, label in BUCKET_EDGES:
            vals = buckets[label]
            if len(vals) < 2:
                print(f"  {label:<10} n={len(vals):<5} (insufficient data)")
                continue
            point_est, ci_lo, ci_hi = bootstrap_mean_ci(vals)
            print(f"  {label:<10} n={len(vals):<5} mean fwd return: {point_est:+.2f}%  "
                  f"[90% CI: {ci_lo:+.2f}% to {ci_hi:+.2f}%]")

        calm_vals = buckets["CALM"]
        stress_vals = buckets["STRESS"]
        if len(calm_vals) >= 2 and len(stress_vals) >= 2:
            calm_mean = sum(calm_vals) / len(calm_vals)
            stress_mean = sum(stress_vals) / len(stress_vals)
            print(f"\n  CALM vs STRESS gap: {stress_mean - calm_mean:+.2f}pp "
                  f"({'stress days had WORSE forward returns, as hypothesized' if stress_mean < calm_mean else 'stress days did NOT show worse forward returns — worth investigating why'})")


if __name__ == "__main__":
    import sys as _sys
    if "--skip-self-tests" not in _sys.argv:
        print("=== Self-test: bootstrap_mean_ci() and bucket_label() ===\n")

    # bucket_label boundary tests
    assert bucket_label(10) == "CALM"
    assert bucket_label(24.99) == "CALM"
    assert bucket_label(25) == "ELEVATED"
    assert bucket_label(44.99) == "ELEVATED"
    assert bucket_label(45) == "STRESS"
    assert bucket_label(90) == "STRESS"
    print("✅ PASS: bucket_label() boundaries correct\n")

    # bootstrap_mean_ci sanity test — known distribution
    random.seed(42)
    test_vals = [random.gauss(-2.0, 5.0) for _ in range(200)]  # true mean approx -2.0
    point_est, ci_lo, ci_hi = bootstrap_mean_ci(test_vals, n_bootstrap=1000)
    print(f"Point estimate: {point_est:.2f} (should be close to true mean -2.0)")
    print(f"90% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    assert -3.5 < point_est < -0.5, f"FAIL: point estimate {point_est} not close to true mean"
    assert ci_lo < point_est < ci_hi, "FAIL: point estimate should fall within its own CI"
    print("✅ PASS: bootstrap CI correctly brackets a known distribution's mean\n")

    # add_forward_returns test
    fake_series = [{"date": f"2024-01-{i+1:02d}", "price": 100 + i} for i in range(10)]
    fake_series = add_forward_returns(fake_series, horizons=[3])
    assert fake_series[0]["fwd_return_3d"] is not None
    assert fake_series[-1]["fwd_return_3d"] is None  # not enough future data
    assert fake_series[-2]["fwd_return_3d"] is None
    assert fake_series[-3]["fwd_return_3d"] is None
    assert fake_series[-4]["fwd_return_3d"] is not None
    expected_return = (103 - 100) / 100 * 100  # day 0 price=100, day 3 price=103
    assert abs(fake_series[0]["fwd_return_3d"] - expected_return) < 0.01
    print("✅ PASS: add_forward_returns() correctly computes forward return and excludes tail with no future data\n")

    print("ALL SELF-TESTS PASSED")

    run_validation()
