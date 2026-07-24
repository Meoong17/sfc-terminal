#!/usr/bin/env python3
"""
lt_redundancy_experiment.py — Walk-forward comparison: current (redundant)
Lt factor construction vs a de-duplicated version.

WHAT THIS TESTS:
    factors["Lt"] in live collect.py currently receives additive
    contributions from AT LEAST 3 overlapping sources for the same
    underlying Fed/ECB/BOJ/M2 data:
      1. Direct: sigmoid(m2_yoy) inside score_factors_from_market()
      2. M33 GLO: its own independent Fed/ECB/BOJ z-score composite
         (calculate_m33_global_liquidity() — confirmed via its own
         docstring to fetch WALCL, ECBASSETSW, JPNASSETS from FRED,
         the EXACT SAME series IDs GLF uses)
      3. GLF: a broader Fed/ECB/BOJ/China/M2/TGA/RRP/DXY composite
         that ALSO uses WALCL/ECBASSETSW/JPNASSETS/M2SL internally

    This script builds two parallel historical Lt-contribution series
    using real FRED data (2014-present, same source as
    historical_backtest_m1m6.py) and runs the SAME walk-forward
    bucket + bootstrap-CI methodology from walk_forward_validation.py
    on both, to see whether de-duplicating helps, hurts, or doesn't
    meaningfully change the already-validated signal.

    VERSION A (current, redundant): Lt = sigmoid(m2_yoy) [direct]
        + GLO-style z-score composite of Fed/ECB/BOJ [mirrors M33]
        + GLF-style z-score composite of Fed/ECB/BOJ/M2 [mirrors GLF,
          simplified — excludes China/TGA/RRP/DXY since those need
          data sources or rule-based logic not worth replicating for
          this specific, narrowly-scoped comparison; see caveats below]

    VERSION B (de-duplicated): Lt = GLF-style z-score composite of
        Fed/ECB/BOJ/M2 ONLY [same single composite as version A's GLF
        term — nothing removed from ITS calculation, just not added a
        second/third time via the direct M2 sigmoid or the separate
        GLO term]

HONEST CAVEATS (same spirit as historical_backtest_m1m6.py's own):
    - This is a SIMPLIFIED reconstruction of Lt, not a perfect replay
      of live collect.py's full Lt calculation (which also includes
      TGA/RRP/China/DXY/ETF/M90-GSLS adjustments this script does not
      attempt to reproduce historically — some of those need data
      sources without clean long-history free access, e.g. China's
      central bank balance sheet). Both Version A and Version B
      EXCLUDE these other pieces equally, so the comparison between
      A and B is still fair (apples-to-apples), even though NEITHER
      is a complete replica of live collect.py's actual current Lt.
    - St, Rt, Ft, Sc factors are held IDENTICAL between Version A and
      B (only Lt's construction differs) — isolating the comparison
      to exactly the redundancy in question.
    - Z-score constants (mean/std pairs) are copied verbatim from
      global_liquidity_engine.py's current hardcoded values — this
      script does NOT independently verify those constants are
      well-calibrated (that's a separate, already-flagged concern in
      analysis/liquidity_zscore_calibration.py).

USAGE:
    python3 analysis/lt_redundancy_experiment.py
    (needs FRED_API_KEY + network access — run on the VPS, not sandboxed)
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from historical_backtest_m1m6 import (
        fetch_fred_series, fetch_fng_historical_dict, _nearest_prior_value,
        _sigmoid_factor, calculate_sfc_ensemble,
    )
    from walk_forward_validation import (
        add_forward_returns, bucket_label, bootstrap_mean_ci, bootstrap_diff_ci,
        BUCKET_EDGES, FORWARD_HORIZONS_DAYS,
    )
except ImportError as e:
    print(f"[LtExperiment] Could not import required modules — make sure "
          f"historical_backtest_m1m6.py and walk_forward_validation.py are "
          f"in the same directory (analysis/): {e}", file=sys.stderr)
    sys.exit(1)

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".lt_redundancy_experiment.json")

# Z-score constants copied verbatim from data_sources/global_liquidity_engine.py
# (see that file's _z_score() calls) — NOT independently re-verified here.
Z_FED = (5.5, 8.0)
Z_ECB = (4.0, 7.0)
Z_JPN = (3.0, 6.0)
Z_M2 = (6.0, 4.0)


def _z(value, mean, std):
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_lt_versions(fed_yoy, ecb_yoy, jpn_yoy, m2_yoy):
    """Returns (lt_version_a, lt_version_b) for one point in time, given
    the four YoY growth rates already computed from FRED levels."""
    fed_z = _z(fed_yoy, *Z_FED) if fed_yoy is not None else 0.0
    ecb_z = _z(ecb_yoy, *Z_ECB) if ecb_yoy is not None else 0.0
    jpn_z = _z(jpn_yoy, *Z_JPN) if jpn_yoy is not None else 0.0
    m2_z = _z(m2_yoy, *Z_M2) if m2_yoy is not None else 0.0

    # GLF-style composite (simplified: fed/ecb/jpn/m2 only, weights
    # renormalized from GLF's own fed=0.30/ecb=0.15/jpn=0.03/m2=0.15
    # subset, excluding china/tga/rrp/dxy which this script does not
    # attempt to reproduce historically)
    _glf_weights = {"fed": 0.30, "ecb": 0.15, "jpn": 0.03, "m2": 0.15}
    _w_sum = sum(_glf_weights.values())
    glf_composite_z = (fed_z * _glf_weights["fed"] + ecb_z * _glf_weights["ecb"] +
                        jpn_z * _glf_weights["jpn"] + m2_z * _glf_weights["m2"]) / _w_sum
    # Convert z-score composite to a factor-scale adjustment (bounded -1..+1,
    # mirroring get_glf_for_factors()'s general shape without needing that
    # function's exact live-only internals). Positive z (expansion, above
    # historical average) -> positive Lt (calming); negative z (contraction)
    # -> negative Lt (stress) — no sign flip needed since z is already signed
    # correctly for this direction.
    glf_adj = max(-1.0, min(1.0, glf_composite_z / 3.0))

    # GLO-style composite (M33: fed/ecb/jpn only, EQUAL weight per that
    # module's own simpler averaging — mirrors calculate_m33_global_liquidity())
    glo_composite_z = (fed_z + ecb_z + jpn_z) / 3.0
    glo_adj = max(-1.0, min(1.0, glo_composite_z / 3.0))

    # Direct M2 sigmoid (mirrors score_factors_from_market's own m2_yoy term)
    direct_m2 = _sigmoid_factor(m2_yoy, center=5.0, k=0.8) if m2_yoy is not None else 0.0

    lt_version_a = direct_m2 + glo_adj + glf_adj  # current: all three stacked
    lt_version_b = glf_adj  # de-duplicated: single composite only

    return lt_version_a, lt_version_b


def compute_yoy_series(level_series):
    """Given {date: level}, return {date: yoy_pct_change} using the value
    ~365 days prior (nearest available, matching historical_backtest_m1m6.py's
    own _nearest_prior_value pattern)."""
    result = {}
    for date_str, level in level_series.items():
        d = datetime.strptime(date_str, "%Y-%m-%d")
        year_ago_str = (d - timedelta(days=365)).strftime("%Y-%m-%d")
        year_ago_level = _nearest_prior_value(level_series, year_ago_str, max_lookback_days=45)
        if year_ago_level and year_ago_level != 0:
            result[date_str] = (level - year_ago_level) / year_ago_level * 100
    return result


def run_experiment():
    print("=" * 60)
    print("Lt REDUNDANCY EXPERIMENT — Version A (current) vs B (de-duplicated)")
    print("=" * 60)

    print("\nFetching FRED series (Fed, ECB, BOJ, M2, DXY, BTC price)...")
    btc_price = fetch_fred_series("CBBTCUSD")
    fed_level = fetch_fred_series("WALCL")
    ecb_level = fetch_fred_series("ECBASSETSW")
    jpn_level = fetch_fred_series("JPNASSETS")
    m2_level = fetch_fred_series("M2SL")

    if not btc_price or not fed_level:
        print("[LtExperiment] Missing FRED data — check FRED_API_KEY and network access.")
        return

    fed_yoy = compute_yoy_series(fed_level)
    ecb_yoy = compute_yoy_series(ecb_level)
    jpn_yoy = compute_yoy_series(jpn_level)
    m2_yoy = compute_yoy_series(m2_level)

    sorted_dates = sorted(btc_price.keys())
    series_a, series_b = [], []
    prev_price = None

    for date_str in sorted_dates:
        price = btc_price[date_str]
        if prev_price is None:
            prev_price = price
            continue
        btc_24h = (price - prev_price) / prev_price * 100
        prev_price = price

        f_yoy = _nearest_prior_value(fed_yoy, date_str, max_lookback_days=10)
        e_yoy = _nearest_prior_value(ecb_yoy, date_str, max_lookback_days=10)
        j_yoy = _nearest_prior_value(jpn_yoy, date_str, max_lookback_days=10)
        m_yoy = _nearest_prior_value(m2_yoy, date_str, max_lookback_days=45)

        if f_yoy is None or m_yoy is None:
            continue  # need at minimum Fed + M2 for a meaningful comparison point

        lt_a, lt_b = compute_lt_versions(f_yoy, e_yoy, j_yoy, m_yoy)

        # Build the OTHER factors identically for both versions — only Lt differs.
        # (St, Ft, Sc stay at 0 as in historical_backtest_m1m6.py's own
        # simplified reconstruction; Rt uses FNG same as that script.)
        base_factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}
        for factors, lt_val, series in [
            (dict(base_factors), lt_a, series_a),
            (dict(base_factors), lt_b, series_b),
        ]:
            factors["Lt"] = max(-3.0, min(3.0, lt_val))
            try:
                sfc_pct = calculate_sfc_ensemble(factors)[0]
            except Exception:
                continue
            series.append({"date": date_str, "price": price, "sfc_pct": sfc_pct})

    print(f"Computed {len(series_a)} points for Version A, {len(series_b)} for Version B")

    series_a = add_forward_returns(series_a)
    series_b = add_forward_returns(series_b)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"version_a": series_a, "version_b": series_b}, f)
    print(f"Saved to {OUTPUT_FILE}")

    for label, series in [("VERSION A (current, redundant)", series_a),
                          ("VERSION B (de-duplicated)", series_b)]:
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)
        for horizon in FORWARD_HORIZONS_DAYS:
            buckets = {lbl: [] for _, _, lbl in BUCKET_EDGES}
            for point in series:
                fwd = point.get(f"fwd_return_{horizon}d")
                if fwd is None:
                    continue
                buckets[bucket_label(point["sfc_pct"])].append(fwd)

            calm_vals = buckets["CALM"]
            stress_vals = buckets["STRESS"]
            if len(calm_vals) >= 2 and len(stress_vals) >= 2:
                diff_est, diff_lo, diff_hi = bootstrap_diff_ci(calm_vals, stress_vals)
                significant = diff_hi < 0 if diff_hi is not None else False
                print(f"  {horizon}d gap: {diff_est:+.2f}pp [90% CI: {diff_lo:+.2f}, {diff_hi:+.2f}] "
                      f"{'SIGNIFICANT' if significant else 'not significant'} "
                      f"(n_calm={len(calm_vals)}, n_stress={len(stress_vals)})")
            else:
                print(f"  {horizon}d gap: insufficient data (n_calm={len(calm_vals)}, n_stress={len(stress_vals)})")


if __name__ == "__main__":
    print("=== Self-test: compute_lt_versions() ===\n")

    print("--- Test 1: Strong expansion (Fed/ECB/BOJ/M2 all growing fast) ---")
    lt_a, lt_b = compute_lt_versions(fed_yoy=15.0, ecb_yoy=12.0, jpn_yoy=10.0, m2_yoy=10.0)
    print(f"Version A: {lt_a:.3f}, Version B: {lt_b:.3f}")
    assert lt_a > 0 and lt_b > 0, "FAIL: strong expansion should push Lt positive (calming) in both versions"
    print("✅ PASS: both versions agree on direction for a clear case\n")

    print("--- Test 2: Strong contraction (Fed/ECB/BOJ/M2 all shrinking) ---")
    lt_a, lt_b = compute_lt_versions(fed_yoy=-15.0, ecb_yoy=-12.0, jpn_yoy=-10.0, m2_yoy=-2.0)
    print(f"Version A: {lt_a:.3f}, Version B: {lt_b:.3f}")
    assert lt_a < 0 and lt_b < 0, "FAIL: strong contraction should push Lt negative (stress) in both versions"
    print("✅ PASS\n")

    print("--- Test 3: Version A should show LARGER magnitude swings than B ---")
    print("(since A stacks 3 redundant contributions, B uses only 1 composite)")
    _, lt_b_expand = compute_lt_versions(fed_yoy=15.0, ecb_yoy=12.0, jpn_yoy=10.0, m2_yoy=10.0)
    lt_a_expand, _ = compute_lt_versions(fed_yoy=15.0, ecb_yoy=12.0, jpn_yoy=10.0, m2_yoy=10.0)
    print(f"Version A magnitude: {abs(lt_a_expand):.3f}, Version B magnitude: {abs(lt_b_expand):.3f}")
    assert abs(lt_a_expand) >= abs(lt_b_expand), "FAIL: redundant version A should show >= magnitude vs single-composite B"
    print("✅ PASS: confirms A over-weights the same underlying signal relative to B\n")

    print("ALL SELF-TESTS PASSED")

    # === Full experiment with real FRED data ===
    run_experiment()
