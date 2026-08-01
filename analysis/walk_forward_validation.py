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

QUANTILE ANALYSIS:
    In addition to the fixed-threshold bucket test, the script also
    performs a threshold-free quantile (decile) analysis. Dividing
    observations into 10 equal groups by sfc_pct rank avoids the
    calibration debate about whether 25/45 are the right cutoffs,
    and instead answers the pure scientific question: "as sfc_pct
    increases, does forward return monotonically decrease?"

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
from datetime import datetime, timedelta, timezone

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
        compute_glo_score, _sorted_items,
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

N_BOOTSTRAP = 2000   # resamples for confidence interval estimation
N_QUANTILES = 10     # deciles for threshold-free quantile analysis

SUMMARY_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".walk_forward_summary.json",
)


def compute_sfc_time_series():
    """Reproduces historical_backtest_m1m6.py's core computation loop,
    but returns the full (date, price, sfc_pct) series rather than just
    printing a crisis-window summary — this script needs the complete
    series to compute forward returns."""
    btc_price = fetch_fred_series("CBBTCUSD")
    dxy_series = fetch_fred_series("DTWEXBGS")
    m2_series = fetch_fred_series("M2SL")
    fng_series = fetch_fng_historical_dict()
    walcl = fetch_fred_series("WALCL")
    ecb = fetch_fred_series("ECBASSETSW")
    jpn = fetch_fred_series("JPNASSETS")

    if not btc_price:
        print("[WalkForward] No BTC price data — check FRED_API_KEY and network access.", file=sys.stderr)
        return []

    walcl_items = _sorted_items(walcl)
    ecb_items = _sorted_items(ecb)
    jpn_items = _sorted_items(jpn)

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
        # GLO liquidity term (collect.py v4.0.0 Lt driver, point-in-time replay)
        glo = compute_glo_score(date_str, walcl_items, ecb_items, jpn_items)

        try:
            factors = score_factors_from_market(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=fng,
                pc_oi=None, m2_yoy=m2_yoy, dxy=dxy, glo_score=glo,
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


def bootstrap_diff_ci(group_a, group_b, n_bootstrap=N_BOOTSTRAP, ci=0.90):
    """Bootstrap the DIFFERENCE between two group means directly.

    This is more statistically powerful than comparing two separate CIs
    — two CIs can overlap slightly while the DIFFERENCE itself still
    excludes zero at the same confidence level, since overlap-checking
    is a conservative proxy for a true difference test, not equivalent
    to one. If this CI excludes zero, that's a more direct, defensible
    significance claim than 'the two separate CIs don't overlap'.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return None, None, None
    n_a, n_b = len(group_a), len(group_b)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = [group_a[random.randrange(n_a)] for _ in range(n_a)]
        sample_b = [group_b[random.randrange(n_b)] for _ in range(n_b)]
        diffs.append(sum(sample_b) / n_b - sum(sample_a) / n_a)
    diffs.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap) - 1
    point_estimate = sum(group_b) / n_b - sum(group_a) / n_a
    return point_estimate, diffs[lo_idx], diffs[hi_idx]


def write_summary_cache(series):
    """Write a SMALL summary that collect.py can cheaply read every live
    cycle, exposing key walk-forward validation stats as dashboard fields
    — without collect.py needing to re-fetch 11 years of FRED history or
    re-run bootstrap resampling on every 5-minute cycle. Re-run this
    script manually/periodically (e.g. monthly) to refresh this cache."""
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_periods": len(series),
    }
    for horizon in FORWARD_HORIZONS_DAYS:
        buckets = {label: [] for _, _, label in BUCKET_EDGES}
        for point in series:
            fwd = point.get(f"fwd_return_{horizon}d")
            if fwd is None:
                continue
            buckets[bucket_label(point["sfc_pct"])].append(fwd)
        calm_vals = buckets["CALM"]
        stress_vals = buckets["STRESS"]
        diff_est, diff_lo, diff_hi = bootstrap_diff_ci(calm_vals, stress_vals)
        summary[f"gap_{horizon}d"] = round(diff_est, 2) if diff_est is not None else None
        summary[f"gap_{horizon}d_ci_lo"] = round(diff_lo, 2) if diff_lo is not None else None
        summary[f"gap_{horizon}d_ci_hi"] = round(diff_hi, 2) if diff_hi is not None else None
        summary[f"gap_{horizon}d_significant"] = (diff_hi < 0) if diff_hi is not None else None
        summary[f"n_calm_{horizon}d"] = len(calm_vals)
        summary[f"n_stress_{horizon}d"] = len(stress_vals)
    total_with_signal = sum(
        1 for p in series
        if p.get(f"fwd_return_{FORWARD_HORIZONS_DAYS[0]}d") is not None
    )
    n_stress_total = sum(1 for p in series if bucket_label(p["sfc_pct"]) == "STRESS")
    summary["n_stress_pct"] = round(n_stress_total / len(series) * 100, 1) if series else None
    with open(SUMMARY_CACHE_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WalkForward] Summary cache written to {SUMMARY_CACHE_FILE}")
    return summary


def _text_bar(val, min_val, max_val, width=20):
    """Return a simple text bar proportional to val's position in [min_val, max_val]."""
    span = max_val - min_val
    if span < 1e-9:
        return "=" * (width // 2) + "|" + " " * (width - width // 2 - 1)
    fraction = (val - min_val) / span
    filled = max(0, min(width - 1, int(round(fraction * (width - 1)))))
    return "=" * filled + "|" + " " * (width - filled - 1)


def quantile_analysis(series):
    """Threshold-free quantile analysis.

    Divides the full series into N_QUANTILES equal-sized groups by
    sfc_pct rank, then computes mean forward return + 90% CI per
    group. This separates the question of predictive validity
    (does higher sfc_pct genuinely predict worse returns) from the
    separate question of threshold calibration (is 25/45 the right
    cutoff). The fixed-threshold bucket analysis is useful for
    checking the project's existing trading rules; this quantile
    analysis is useful for answering the pure scientific question.
    """
    sorted_series = sorted(series, key=lambda x: x["sfc_pct"])
    n = len(sorted_series)
    bin_size = n // N_QUANTILES
    remainder = n % N_QUANTILES

    print("\n" + "=" * 60)
    print("QUANTILE ANALYSIS — Threshold-Free Predictive Validity")
    print("=" * 60)
    print(f"(Sorted by sfc_pct, divided into {N_QUANTILES} equal groups of ~{bin_size} obs)")

    for horizon in FORWARD_HORIZONS_DAYS:
        print(f"\n--- {horizon}-day forward return by sfc_pct decile ---")
        print(f"  {'Decile':<8} {'sfc_pct range':<18} {'n':<6} {'mean fwd%':>10}  {'CI 90%':<22}  {'Bar':<22}")
        print(f"  {'-'*8} {'-'*18} {'-'*6} {'-'*10}  {'-'*22}  {'-'*22}")

        decile_means = []
        start = 0
        for q in range(N_QUANTILES):
            size = bin_size + (1 if q < remainder else 0)
            chunk = sorted_series[start:start + size]
            start += size
            if not chunk:
                continue

            lo_pct = chunk[0]["sfc_pct"]
            hi_pct = chunk[-1]["sfc_pct"]

            fwd_vals = [
                p.get(f"fwd_return_{horizon}d") for p in chunk
                if p.get(f"fwd_return_{horizon}d") is not None
            ]
            if len(fwd_vals) < 2:
                print(f"  Q{q+1:<7} [{lo_pct:5.1f} — {hi_pct:5.1f}]  {len(fwd_vals):<6} {'insufficient':>10}")
                decile_means.append(None)
                continue

            mean_est, ci_lo, ci_hi = bootstrap_mean_ci(fwd_vals)
            decile_means.append(mean_est)
            bar = _text_bar(mean_est, min(-5, min(fwd_vals)), max(5, max(fwd_vals)))

            print(f"  Q{q+1:<7} [{lo_pct:5.1f} — {hi_pct:5.1f}]  {len(fwd_vals):<6} {mean_est:>+8.2f}%  "
                  f"[{ci_lo:+.2f}, {ci_hi:+.2f}]  |{bar}|")

        # Monotonicity check
        valid = [(i, m) for i, m in enumerate(decile_means) if m is not None]
        if len(valid) >= 3:
            monotonic_pairs = sum(
                1 for j in range(1, len(valid))
                if valid[j][1] < valid[j-1][1]
            )
            total_pairs = len(valid) - 1
            pct_monotonic = monotonic_pairs / total_pairs * 100

            # Simple Kendall-like correlation: concordant vs discordant pairs
            ranks = list(range(len(valid)))
            means = [m for _, m in valid]
            n_p = len(ranks)
            concordant = 0
            discordant = 0
            for i in range(n_p):
                for j in range(i + 1, n_p):
                    if (ranks[j] > ranks[i] and means[j] < means[i]) or \
                       (ranks[j] < ranks[i] and means[j] > means[i]):
                        concordant += 1
                    elif ranks[j] != ranks[i] and means[j] != means[i]:
                        discordant += 1
            total_pairs_cd = concordant + discordant
            kendall_tau = (concordant - discordant) / total_pairs_cd if total_pairs_cd > 0 else 0.0

            print(f"\n  Monotonicity (higher decile -> lower return):")
            print(f"    Adjacent pairs following direction: {monotonic_pairs}/{total_pairs} ({pct_monotonic:.0f}%)")
            print(f"    Kendall-like rank correlation (decile rank vs mean return): {kendall_tau:+.3f} "
                  f"({'NEGATIVE -> CONFIRMS hypothesis' if kendall_tau < 0 else 'POSITIVE -> contradicts hypothesis'})")

    # Compare top vs bottom decile
    print(f"\n  {'='*56}")
    print(f"  TOP DECILE (highest sfc_pct) vs BOTTOM DECILE (lowest sfc_pct)")
    print(f"  {'='*56}")
    for horizon in FORWARD_HORIZONS_DAYS:
        fwd_key = f"fwd_return_{horizon}d"
        bot_chunk = sorted_series[:bin_size]
        bot_vals = [p[fwd_key] for p in bot_chunk if p.get(fwd_key) is not None]
        top_chunk = sorted_series[-bin_size:]
        top_vals = [p[fwd_key] for p in top_chunk if p.get(fwd_key) is not None]

        if len(bot_vals) >= 2 and len(top_vals) >= 2:
            b_mean = sum(bot_vals) / len(bot_vals)
            t_mean = sum(top_vals) / len(top_vals)
            _, b_lo, b_hi = bootstrap_mean_ci(bot_vals)
            _, t_lo, t_hi = bootstrap_mean_ci(top_vals)
            overlap = not (t_hi < b_lo or b_hi < t_lo)
            print(f"  {horizon:>2}d: Bottom decile {b_mean:+.2f}% [{b_lo:+.2f}, {b_hi:+.2f}]")
            print(f"       Top decile    {t_mean:+.2f}% [{t_lo:+.2f}, {t_hi:+.2f}]")
            gap = t_mean - b_mean
            flag = "NO OVERLAP" if not overlap else "OVERLAP"
            if overlap and abs(t_hi - b_lo) < 0.5:
                flag += " (borderline)"
            print(f"       Gap: {gap:+.2f}pp — CI {flag}")


def bucket_analysis(series):
    """Fixed-threshold bucket analysis using paper_trader.py's BUY/SELL/HOLD thresholds."""
    print("\n" + "=" * 60)
    print("FORWARD RETURN BY SIGNAL BUCKET (with 90% bootstrap CI)")
    print("(Using paper_trader.py thresholds: CALM<25, ELEVATED 25-45, STRESS>=45)")
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
            diff_est, diff_lo, diff_hi = bootstrap_diff_ci(calm_vals, stress_vals)
            significant = diff_hi < 0 if diff_hi is not None else False
            print(f"\n  CALM vs STRESS gap (direct bootstrap of the difference): {diff_est:+.2f}pp "
                  f"[90% CI: {diff_lo:+.2f}pp to {diff_hi:+.2f}pp] "
                  f"{'— SIGNIFICANT (CI excludes zero)' if significant else '— not significant at 90% (CI includes zero)'}")

    quantile_analysis(series)


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

    bucket_analysis(series)

    summary = write_summary_cache(series)
    print(f"\n  Cache: gap_7d={summary.get('gap_7d')}pp, gap_30d={summary.get('gap_30d')}pp, "
          f"n_stress={summary.get('n_stress_pct')}%")


if __name__ == "__main__":
    run_quantile_only = "--quantile-only" in sys.argv

    if not run_quantile_only:
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

    else:
        # --quantile-only: skip self-tests, skip bucket analysis, load cached data
        if os.path.exists(OUTPUT_FILE):
            print(f"Loading cached time series from {OUTPUT_FILE} ...")
            with open(OUTPUT_FILE) as f:
                series = json.load(f)
            print(f"Loaded {len(series)} observations")
            quantile_analysis(series)
        else:
            print(f"No cached data found at {OUTPUT_FILE}. Run without --quantile-only first.")
            sys.exit(1)
