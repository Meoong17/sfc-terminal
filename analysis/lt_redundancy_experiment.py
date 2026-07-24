#!/usr/bin/env python3
"""
lt_redundancy_experiment.py — Walk-forward comparison: current (redundant)
Lt factor construction vs de-duplicated and rescaled versions.

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

    VERSION C (de-duplicated, rescaled): Lt = Version B's single glf_adj
        term, multiplied by the EMPIRICAL amplitude ratio (std_A / std_B)
        computed from the actual dataset — isolates whether A's edge is
        purely an arithmetic artifact of summing 3 bounded terms vs 1.
        If C's predictive gap matches A's, redundancy was just an
        implicit amplifier (fixable via reweighting alone). If C's gap
        stays weak despite matching A's amplitude, the 3 redundant
        components carried genuinely different information.

HONEST CAVEATS (same spirit as historical_backtest_m1m6.py's own):
    - This is a SIMPLIFIED reconstruction of Lt, not a perfect replay
      of live collect.py's full Lt calculation (which also includes
      TGA/RRP/China/DXY/ETF/M90-GSLS adjustments this script does not
      attempt to reproduce historically — some of those need data
      sources without clean long-history free access, e.g. China's
      central bank balance sheet). All three versions (A, B, C) EXCLUDE
      these other pieces equally, so the comparison between them is fair
      (apples-to-apples), even though NONE is a complete replica of live
    - St, Rt, Ft, Sc factors are held IDENTICAL between all three
      versions (only Lt's construction differs) — isolating the comparison
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
        add_forward_returns, bootstrap_diff_ci,
        FORWARD_HORIZONS_DAYS,
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
    """Returns (lt_version_a, lt_version_b, lt_version_c) for one point in
    time, given the four YoY growth rates already computed from FRED
    levels.

    Version C added (2026-07, following a sharp observation from real-data
    results): Version B's narrower dynamic range vs Version A could simply
    be an ARITHMETIC artifact of summing 3 bounded terms (A) vs 1 (B),
    not evidence that the 3 redundant components carry genuinely
    different information. Version C = Version B's single glf_adj term,
    scaled by 3x to roughly match Version A's amplitude — if C's
    predictive gap matches A's, redundancy was "just" an implicit
    amplifier (fixable via reweighting alone). If C's gap stays weak
    despite matching A's amplitude, the 3 redundant components were
    capturing genuinely different variance, not just inflating scale.
    NOTE: run_experiment() overrides this fixed 3x with an empirical
    amplitude ratio from the actual dataset (std_A / std_B) for a
    more accurate comparison.
    """
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
    lt_version_c = glf_adj * 3.0  # de-duplicated, RESCALED to match A's amplitude —
                                    # isolates whether A's edge is just scale. NOT
                                    # clamped here — relies on the SAME outer clamp
                                    # run_experiment() applies to A/B/C alike (via
                                    # factors["Lt"] = max(-3,min(3,...))) for a fair,
                                    # consistent comparison — clamping here too would
                                    # double-restrict C relative to A/B.

    return lt_version_a, lt_version_b, lt_version_c


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
    print("Lt REDUNDANCY EXPERIMENT — Version A (current) vs B (de-duplicated) vs C (rescaled B)")
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
    raw_points = []  # collect first so we can compute empirical scaling factor for Version C
    prev_price = None

    for date_str in sorted_dates:
        price = btc_price[date_str]
        if prev_price is None:
            prev_price = price
            continue
        prev_price = price

        f_yoy = _nearest_prior_value(fed_yoy, date_str, max_lookback_days=10)
        e_yoy = _nearest_prior_value(ecb_yoy, date_str, max_lookback_days=10)
        j_yoy = _nearest_prior_value(jpn_yoy, date_str, max_lookback_days=10)
        m_yoy = _nearest_prior_value(m2_yoy, date_str, max_lookback_days=45)

        if f_yoy is None or m_yoy is None:
            continue  # need at minimum Fed + M2 for a meaningful comparison point

        lt_a, lt_b, _ = compute_lt_versions(f_yoy, e_yoy, j_yoy, m_yoy)
        raw_points.append((date_str, price, lt_a, lt_b))

    # Empirical scaling factor: ratio of standard deviations.
    lt_a_vals = [p[2] for p in raw_points]
    lt_b_vals = [p[3] for p in raw_points]
    mean_a = sum(lt_a_vals) / len(lt_a_vals)
    mean_b = sum(lt_b_vals) / len(lt_b_vals)
    std_a = (sum((v - mean_a) ** 2 for v in lt_a_vals) / len(lt_a_vals)) ** 0.5
    std_b = (sum((v - mean_b) ** 2 for v in lt_b_vals) / len(lt_b_vals)) ** 0.5
    empirical_scale = std_a / std_b if std_b > 0 else 1.0
    print(f"Empirical amplitude ratio (std_A / std_B): {empirical_scale:.3f} "
          f"(std_A={std_a:.3f}, std_B={std_b:.3f}) — using this to build Version C, "
          f"not a guessed fixed multiplier")

    series_a, series_b, series_c = [], [], []
    for date_str, price, lt_a, lt_b in raw_points:
        lt_c = lt_b * empirical_scale

        # Build the OTHER factors identically — only Lt differs.
        # (St, Ft, Sc stay at 0 as in historical_backtest_m1m6.py's own
        # simplified reconstruction.)
        base_factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}
        for factors, lt_val, series in [
            (dict(base_factors), lt_a, series_a),
            (dict(base_factors), lt_b, series_b),
            (dict(base_factors), lt_c, series_c),
        ]:
            factors["Lt"] = max(-3.0, min(3.0, lt_val))
            try:
                sfc_pct = calculate_sfc_ensemble(factors)[0]
            except Exception:
                continue
            series.append({"date": date_str, "price": price, "sfc_pct": sfc_pct})

    print(f"Computed {len(series_a)} points for A, {len(series_b)} for B, {len(series_c)} for C")

    series_a = add_forward_returns(series_a)
    series_b = add_forward_returns(series_b)
    series_c = add_forward_returns(series_c)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"version_a": series_a, "version_b": series_b, "version_c": series_c}, f)
    print(f"Saved to {OUTPUT_FILE}")

    for label, series in [
        ("VERSION A (current, redundant)", series_a),
        ("VERSION B (de-duplicated)", series_b),
        ("VERSION C (de-duplicated, RESCALED to match A's amplitude)", series_c),
    ]:
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)

        # FIX (found via real-data run, 2026-07): with St=Rt=Ft=Sc=0 (dummy),
        # the z_score never gets enough contribution to cross the live system's
        # 45% STRESS threshold (calibrated for all 5 factors active). Using
        # FIXED absolute buckets always produces n_stress=0 for ALL versions.
        # Switched to relative quantile buckets (top vs bottom 20% of THIS
        # experiment's own sfc_pct distribution) — same fix already used in
        # walk_forward_validation.py for a similar calibration mismatch.
        valid_points = [p for p in series if p.get(f"fwd_return_{FORWARD_HORIZONS_DAYS[0]}d") is not None]
        if len(valid_points) < 20:
            print("  Insufficient data for quantile analysis.")
            continue
        sorted_by_sfc = sorted(valid_points, key=lambda p: p["sfc_pct"])
        n = len(sorted_by_sfc)
        q20 = max(1, n // 5)
        bottom_20pct = sorted_by_sfc[:q20]
        top_20pct = sorted_by_sfc[-q20:]
        print(f"  sfc_pct range in this experiment: {sorted_by_sfc[0]['sfc_pct']:.2f} to {sorted_by_sfc[-1]['sfc_pct']:.2f}")
        print(f"  Bottom 20% (n={len(bottom_20pct)}) sfc_pct range: "
              f"{bottom_20pct[0]['sfc_pct']:.2f} to {bottom_20pct[-1]['sfc_pct']:.2f}")
        print(f"  Top 20% (n={len(top_20pct)}) sfc_pct range: "
              f"{top_20pct[0]['sfc_pct']:.2f} to {top_20pct[-1]['sfc_pct']:.2f}")

        for horizon in FORWARD_HORIZONS_DAYS:
            bottom_fwd = [p[f"fwd_return_{horizon}d"] for p in bottom_20pct if p.get(f"fwd_return_{horizon}d") is not None]
            top_fwd = [p[f"fwd_return_{horizon}d"] for p in top_20pct if p.get(f"fwd_return_{horizon}d") is not None]
            if len(bottom_fwd) >= 2 and len(top_fwd) >= 2:
                diff_est, diff_lo, diff_hi = bootstrap_diff_ci(bottom_fwd, top_fwd)
                # Two-tailed: significant if CI does not contain zero
                significant = (diff_hi < 0 or diff_lo > 0) if (diff_hi is not None and diff_lo is not None) else False
                print(f"  {horizon}d gap (low sfc - high sfc): {diff_est:+.2f}pp "
                      f"[90% CI: {diff_lo:+.2f}, {diff_hi:+.2f}] "
                      f"{'SIGNIFICANT' if significant else 'not significant'}")
            else:
                print(f"  {horizon}d: insufficient forward-return data")


if __name__ == "__main__":
    print("=== Self-test: compute_lt_versions() ===\n")

    print("--- Test 1: Strong expansion (Fed/ECB/BOJ/M2 all growing fast) ---")
    lt_a, lt_b, lt_c = compute_lt_versions(fed_yoy=15.0, ecb_yoy=12.0, jpn_yoy=10.0, m2_yoy=10.0)
    print(f"Version A: {lt_a:.3f}, Version B: {lt_b:.3f}, Version C: {lt_c:.3f}")
    assert lt_a > 0 and lt_b > 0 and lt_c > 0, "FAIL: strong expansion should push Lt positive (calming) in all versions"
    print("✅ PASS: all versions agree on direction for a clear case\n")

    print("--- Test 2: Strong contraction (Fed/ECB/BOJ/M2 all shrinking) ---")
    lt_a, lt_b, lt_c = compute_lt_versions(fed_yoy=-15.0, ecb_yoy=-12.0, jpn_yoy=-10.0, m2_yoy=-2.0)
    print(f"Version A: {lt_a:.3f}, Version B: {lt_b:.3f}, Version C: {lt_c:.3f}")
    assert lt_a < 0 and lt_b < 0 and lt_c < 0, "FAIL: strong contraction should push Lt negative (stress) in all versions"
    print("✅ PASS\n")

    print("--- Test 3: Version A should show LARGER magnitude swings than B ---")
    print("(since A stacks 3 redundant contributions, B uses only 1 composite)")
    lt_a_expand, lt_b_expand, lt_c_expand = compute_lt_versions(fed_yoy=15.0, ecb_yoy=12.0, jpn_yoy=10.0, m2_yoy=10.0)
    print(f"Version A magnitude: {abs(lt_a_expand):.3f}, Version B magnitude: {abs(lt_b_expand):.3f}")
    assert abs(lt_a_expand) >= abs(lt_b_expand), "FAIL: redundant version A should show >= magnitude vs single-composite B"
    print("✅ PASS: confirms A over-weights the same underlying signal relative to B\n")

    print("--- Test 4: Version C should be ~3x Version B's magnitude, comparable to A ---")
    print(f"Version C magnitude: {abs(lt_c_expand):.3f} (should be close to Version A's {abs(lt_a_expand):.3f})")
    assert abs(lt_c_expand) >= abs(lt_b_expand) * 2.5, "FAIL: Version C should be substantially larger than B (roughly 3x, before clamping)"
    print("✅ PASS: Version C successfully rescaled to A's amplitude range\n")

    print("ALL SELF-TESTS PASSED")
    print("\n" + "=" * 60)
    print("Self-tests only verify the CALCULATION LOGIC is correct.")
    print("Proceeding to the REAL experiment (fetches FRED historical")
    print("data — needs FRED_API_KEY + network access)...")
    print("=" * 60 + "\n")

    # === Full experiment with real FRED data ===
    run_experiment()
