#!/usr/bin/env python3
"""
fed_regime_flip_test.py — Tests Karau (2023)'s finding ("Monetary policy
and Bitcoin", Journal of International Money and Finance) against SFC's
own historical data: did the Fed-BTC relationship direction genuinely
flip around 2020, the way that peer-reviewed paper found?

WHAT KARAU (2023) FOUND (verified genuine via independent search — see
this project's own literature-verification discussion):
    Historically (pre-2020), US monetary TIGHTENING was associated with
    BITCOIN PRICES RISING (a counter-intuitive finding). Post-2020, this
    reversed: tightening began coinciding with Bitcoin prices FALLING —
    more consistent with BTC behaving like a conventional risk asset.

WHY THIS MATTERS FOR SFC:
    GLF's fed_z component (data_sources/global_liquidity_engine.py) uses
    a FIXED-SIGN relationship: Fed contraction always maps to negative
    Lt (stress), Fed expansion always maps to positive Lt (calm) —
    regardless of era. If Karau's regime-dependent finding holds in our
    own data too, this specific piece of GLF's design may be
    systematically MISCALIBRATED for one of the two eras (whichever one
    doesn't match the CURRENT fixed-sign assumption). This is separate
    from (and does not affect) the Lt de-duplication already applied in
    collect.py v4.0.0 — that fix addressed REDUNDANT counting of the
    same Fed/ECB/BOJ/M2 signal; this test addresses whether the SIGN of
    that signal's relationship to BTC is stable over time.

METHODOLOGY:
    Split BTC/Fed history into two periods (pre-2020-03 vs post), and
    within EACH period separately, compute:
      1. Correlation between Fed YoY growth and BTC's 30-day FORWARD
         return (not contemporaneous — this avoids conflating "Fed
         growing while BTC also happens to be rising" with genuine
         lead-lag structure)
      2. Bucket-based comparison (same bootstrap-CI spirit as
         walk_forward_validation.py): does "Fed contracting" precede
         better or worse BTC forward returns in each era?

    If the correlation/bucket-gap SIGN differs between the two periods,
    that's direct, own-data confirmation of Karau's regime-dependent
    finding — and a concrete signal that GLF's fixed-sign fed_z
    treatment could benefit from being regime-aware.

HONEST CAVEATS:
    - This is a SIMPLER test than Karau's own SVAR-with-external-
      instruments approach (which specifically isolates policy SHOCKS,
      not just raw Fed balance sheet level changes) — a correlation/
      bucket-gap comparison is a legitimate, but less rigorous, cross-
      check. A flip found here is suggestive, not as definitive as a
      full SVAR replication would be.
    - The pre-2020 window in FRED WALCL/BTC data is comparatively short
      (2014-2020, ~6 years) — fewer independent "Fed cycles" to observe
      than the post-2020 window, so any pre-2020 finding should be
      treated with proportionally more caution (wider uncertainty).

USAGE:
    python3 analysis/fed_regime_flip_test.py
    (needs FRED_API_KEY + network access — run on the VPS)
"""
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from historical_backtest_m1m6 import fetch_fred_series, _nearest_prior_value
    from walk_forward_validation import bootstrap_diff_ci
except ImportError as e:
    print(f"[FedRegimeFlip] Could not import required modules: {e}", file=sys.stderr)
    sys.exit(1)

SPLIT_DATE = "2020-03-01"  # COVID/QE-infinity onset — the same rough
                            # dividing line Karau (2023) uses for its
                            # pre/post comparison
FORWARD_DAYS = 30


def _pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def compute_fed_yoy_series(fed_level):
    result = {}
    for date_str, level in fed_level.items():
        d = datetime.strptime(date_str, "%Y-%m-%d")
        year_ago_str = (d - timedelta(days=365)).strftime("%Y-%m-%d")
        year_ago_level = _nearest_prior_value(fed_level, year_ago_str, max_lookback_days=45)
        if year_ago_level and year_ago_level != 0:
            result[date_str] = (level - year_ago_level) / year_ago_level * 100
    return result


def run_test():
    print("=" * 60)
    print("FED-BTC REGIME FLIP TEST (Karau 2023 replication-lite)")
    print("=" * 60)

    print("\nFetching FRED Fed balance sheet (WALCL) + BTC price...")
    fed_level = fetch_fred_series("WALCL")
    btc_price = fetch_fred_series("CBBTCUSD")

    if not fed_level or not btc_price:
        print("[FedRegimeFlip] Missing FRED data — check FRED_API_KEY and network access.")
        return

    fed_yoy = compute_fed_yoy_series(fed_level)
    sorted_dates = sorted(btc_price.keys())

    pre_pairs, post_pairs = [], []  # (fed_yoy, fwd_return) tuples per era

    for i, date_str in enumerate(sorted_dates):
        f_yoy = _nearest_prior_value(fed_yoy, date_str, max_lookback_days=10)
        if f_yoy is None:
            continue

        future_idx = i + FORWARD_DAYS
        if future_idx >= len(sorted_dates):
            continue
        future_date = sorted_dates[future_idx]
        fwd_return = (btc_price[future_date] - btc_price[date_str]) / btc_price[date_str] * 100

        pair = (f_yoy, fwd_return)
        if date_str < SPLIT_DATE:
            pre_pairs.append(pair)
        else:
            post_pairs.append(pair)

    print(f"\nPre-{SPLIT_DATE}: {len(pre_pairs)} observations")
    print(f"Post-{SPLIT_DATE}: {len(post_pairs)} observations")

    for label, pairs in [("PRE-2020", pre_pairs), ("POST-2020", post_pairs)]:
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print("=" * 60)
        if len(pairs) < 10:
            print("Insufficient observations for this era.")
            continue

        fed_vals = [p[0] for p in pairs]
        fwd_vals = [p[1] for p in pairs]
        corr = _pearson_corr(fed_vals, fwd_vals)
        print(f"Correlation(Fed YoY growth, {FORWARD_DAYS}d forward BTC return): {corr:+.3f}")
        if corr is not None:
            if corr > 0.1:
                print("  -> POSITIVE: Fed EXPANSION precedes BETTER BTC returns "
                      "(matches GLF's fixed-sign assumption)")
            elif corr < -0.1:
                print("  -> NEGATIVE: Fed EXPANSION precedes WORSE BTC returns "
                      "(OPPOSITE of GLF's fixed-sign assumption for this era)")
            else:
                print("  -> Weak/no clear relationship in this era")

        # Bucket comparison: Fed contracting (yoy < median) vs expanding (>= median)
        median_fed = sorted(fed_vals)[len(fed_vals) // 2]
        contracting_fwd = [fwd for fed, fwd in pairs if fed < median_fed]
        expanding_fwd = [fwd for fed, fwd in pairs if fed >= median_fed]
        if len(contracting_fwd) >= 2 and len(expanding_fwd) >= 2:
            diff, lo, hi = bootstrap_diff_ci(contracting_fwd, expanding_fwd)
            sig = (hi < 0 or lo > 0) if (hi is not None and lo is not None) else False
            print(f"Bucket gap (expanding - contracting): {diff:+.2f}pp "
                  f"[90% CI: {lo:+.2f}, {hi:+.2f}] {'SIGNIFICANT' if sig else 'not significant'}")

    if len(pre_pairs) >= 10 and len(post_pairs) >= 10:
        pre_corr = _pearson_corr([p[0] for p in pre_pairs], [p[1] for p in pre_pairs])
        post_corr = _pearson_corr([p[0] for p in post_pairs], [p[1] for p in post_pairs])
        print(f"\n{'=' * 60}")
        print("VERDICT")
        print("=" * 60)
        if pre_corr is not None and post_corr is not None:
            if (pre_corr > 0.1 and post_corr < -0.1) or (pre_corr < -0.1 and post_corr > 0.1):
                print("SIGN FLIP CONFIRMED in this project's own data — consistent with "
                      "Karau (2023)'s finding. GLF's fixed-sign fed_z treatment may be "
                      "worth revisiting for regime-awareness.")
            else:
                print("No clear sign flip detected in this project's own data using this "
                      "simpler correlation method. This does NOT invalidate Karau (2023) — "
                      "their SVAR-with-external-instruments approach isolates genuine policy "
                      "SHOCKS specifically, which this simpler test cannot fully replicate — "
                      "but it means this particular cross-check doesn't independently confirm "
                      "the flip using this project's own data and methodology.")


if __name__ == "__main__":
    print("=== Self-test: _pearson_corr and compute_fed_yoy_series ===\n")

    print("--- Test 1: _pearson_corr detects clear positive correlation ---")
    xs = [1, 2, 3, 4, 5]
    ys = [10, 20, 30, 40, 50]
    corr = _pearson_corr(xs, ys)
    assert abs(corr - 1.0) < 1e-9, f"FAIL: {corr}"
    print(f"✅ PASS: correlation={corr}\n")

    print("--- Test 2: _pearson_corr detects clear negative correlation ---")
    xs = [1, 2, 3, 4, 5]
    ys = [50, 40, 30, 20, 10]
    corr = _pearson_corr(xs, ys)
    assert abs(corr - (-1.0)) < 1e-9, f"FAIL: {corr}"
    print(f"✅ PASS: correlation={corr}\n")

    print("--- Test 3: compute_fed_yoy_series computes correct YoY growth ---")
    fake_fed_level = {}
    base_date = datetime(2019, 1, 1)
    for i in range(400):
        d = base_date + timedelta(days=i)
        fake_fed_level[d.strftime("%Y-%m-%d")] = 4000 + i  # linear growth for simplicity
    yoy = compute_fed_yoy_series(fake_fed_level)
    test_date = "2020-01-01"  # 365 days after base_date=2019-01-01
    if test_date in yoy:
        expected_level_now = 4000 + 365
        expected_level_year_ago = 4000
        expected_yoy = (expected_level_now - expected_level_year_ago) / expected_level_year_ago * 100
        assert abs(yoy[test_date] - expected_yoy) < 1.0, f"FAIL: {yoy[test_date]} vs {expected_yoy}"
        print(f"✅ PASS: YoY growth computed correctly ({yoy[test_date]:.2f}%, expected ~{expected_yoy:.2f}%)\n")
    else:
        print("⚠ Test date not in range — adjust test window\n")

    print("--- Test 4: Simulated sign-flip scenario is correctly detected ---")
    import random
    random.seed(42)
    # Simulate: pre-2020 POSITIVE relationship, post-2020 NEGATIVE relationship
    pre_pairs_sim = [(x, x * 2 + random.gauss(0, 0.5)) for x in [random.gauss(0, 5) for _ in range(50)]]
    post_pairs_sim = [(x, -x * 2 + random.gauss(0, 0.5)) for x in [random.gauss(0, 5) for _ in range(50)]]
    pre_corr_sim = _pearson_corr([p[0] for p in pre_pairs_sim], [p[1] for p in pre_pairs_sim])
    post_corr_sim = _pearson_corr([p[0] for p in post_pairs_sim], [p[1] for p in post_pairs_sim])
    print(f"Simulated pre-2020 corr: {pre_corr_sim:.3f}, post-2020 corr: {post_corr_sim:.3f}")
    assert pre_corr_sim > 0.5 and post_corr_sim < -0.5, "FAIL: simulated flip not detected"
    print("✅ PASS: sign-flip scenario correctly distinguishable via correlation\n")

    print("ALL SELF-TESTS PASSED")
    print("\n" + "=" * 60)
    print("Self-tests only verify the CALCULATION LOGIC is correct.")
    print("Proceeding to the REAL test (fetches FRED historical data —")
    print("needs FRED_API_KEY + network access)...")
    print("=" * 60 + "\n")

    run_test()
