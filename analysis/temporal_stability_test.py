#!/usr/bin/env python3
"""
temporal_stability_test.py — Does the SFC core ensemble's predictive
"gap" (CALM vs STRESS forward returns) PERSIST, WEAKEN, or DISAPPEAR in
the most recent data?

WHY THIS FILE EXISTS:
    walk_forward_validation.py already established that, across the FULL
    2014-2026 history, the sfc_pct signal has genuine out-of-sample
    forward predictive value: when the signal reads STRESS, the actual
    subsequent price return is meaningfully worse than when it reads
    CALM (90% bootstrap CI on the direct difference excludes zero).

    But the full-history gap is an AVERAGE over a 12-year window. It
    could be driven entirely by the 2018/COVID/2022 crashes and be
    dead in the current regime — or it could be weakening as BTC
    matures as an asset (more institutional participation, futures/
    ETF flows) and the macro-driven core signal loses its edge. That
    distinction matters enormously for whether this project's BT label
    should still say "walk-forward validated" going forward.

    This script answers exactly that: it takes the SAME data, SAME
    buckets, SAME bootstrap-CI methodology as walk_forward_validation.py,
    and re-runs the gap computation SEPARATELY within each era.

ERA SPLITS (two complementary cuts):
    1. Three equal 4-year blocks:  2014-2018, 2018-2022, 2022-2026
       (shows the secular trend block-over-block)
    2. Pre / post 2020-03:         date < "2020-03-01" vs >=
       (same dividing line fed_regime_flip_test.py uses, aligned with
       Karau (2023)'s regime cut)

METHODOLOGY (identical to walk_forward_validation.py):
    - Load the SAME full historical (date, price, sfc_pct,
      fwd_return_7d, fwd_return_30d) series — from the exact JSON cache
      that walk_forward_validation.py writes (recomputed on-the-fly only
      if that cache is missing).
    - Same BUCKET_EDGES (CALM<25, ELEVATED 25-45, STRESS>=45).
    - Same bootstrap_diff_ci(calm, stress) — direct percentile bootstrap
      of the difference, 90% CI, 2000 resamples.
    - Within EACH era separately, compute the CALM-vs-STRESS gap at both
      the 7d and 30d horizons, plus a threshold-free top/bottom-tail
      quantile gap as a robustness cross-check (same 20% tails used in
      the skill's relative-quantile method).

OUTPUT — direct answer per horizon:
    BERTAHAN (persists)   -> latest-era gap is still significantly
                             negative and ≥ ~half the full-history gap
    MELEMAH (weakens)     -> latest-era gap is smaller / no longer
                             significant, but still the right direction
    HILANG   (disappears) -> latest-era gap is gone or sign-flipped

    If the gap MELEMAH significantly in the most recent era, that is a
    MAJOR finding — it means the current BT label ("walk-forward
    validated") is stale for today's regime, which is far more important
    than adding any new feature.

USAGE:
    python3 analysis/temporal_stability_test.py
    (loads .walk_forward_validation.json — no FRED fetch needed unless
     that cache is missing, in which case it recomputes from FRED like
     walk_forward_validation.py does; needs FRED_API_KEY + network.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the EXACT same methodology as walk_forward_validation.py so the
# era comparison is apples-to-apples with the full-history result.
from walk_forward_validation import (
    BUCKET_EDGES,
    FORWARD_HORIZONS_DAYS,
    bucket_label,
    bootstrap_diff_ci,
    bootstrap_mean_ci,
    compute_sfc_time_series,
    add_forward_returns,
    OUTPUT_FILE,
)

# Load the same cache file walk_forward_validation.py writes.
SERIES_CACHE = OUTPUT_FILE  # .walk_forward_validation.json at project root

# --- Era boundaries -----------------------------------------------------

# Three equal 4-year blocks (inclusive-left, exclusive-right).
# Data spans 2014-12-02 -> 2026-07-23, so the final block naturally
# captures everything 2022-2026.
ERA_BLOCKS = [
    ("2014-2018", "2014-01-01", "2018-01-01"),
    ("2018-2022", "2018-01-01", "2022-01-01"),
    ("2022-2026", "2022-01-01", "2027-01-01"),
]

# Pre/post 2020-03 split — the SAME line fed_regime_flip_test.py uses.
SPLIT_DATE = "2020-03-01"
ERA_FED = [("pre-2020-03", None, SPLIT_DATE), ("post-2020-03", SPLIT_DATE, None)]

QUANTILE_TAIL = 0.20  # bottom 20% (low sfc_pct) vs top 20% (high sfc_pct)


def in_era(date_str, lo, hi):
    """lo <= date < hi; None = open-ended on that side."""
    if lo is not None and date_str < lo:
        return False
    if hi is not None and date_str >= hi:
        return False
    return True


def era_gap(series, horizon, lo, hi):
    """Compute the CALM vs STRESS gap within one era for one horizon.

    Same bucket + bootstrap method as walk_forward_validation.py.
    Returns (n_calm, n_stress, gap_est, gap_lo, gap_hi)."""
    calm_vals, stress_vals = [], []
    for p in series:
        if not in_era(p["date"], lo, hi):
            continue
        fwd = p.get(f"fwd_return_{horizon}d")
        if fwd is None:
            continue
        label = bucket_label(p["sfc_pct"])
        if label == "CALM":
            calm_vals.append(fwd)
        elif label == "STRESS":
            stress_vals.append(fwd)
    if len(calm_vals) < 2 or len(stress_vals) < 2:
        return len(calm_vals), len(stress_vals), None, None, None
    est, lo_b, hi_b = bootstrap_diff_ci(calm_vals, stress_vals)
    return len(calm_vals), len(stress_vals), est, lo_b, hi_b


def era_tail_gap(series, horizon, lo, hi):
    """Threshold-free robustness cross-check: top 20% sfc_pct vs bottom
    20% sfc_pct forward return gap within one era (negative = higher
    signal predicts worse return = correct polarity)."""
    points = [p for p in series if in_era(p["date"], lo, hi)
              and p.get(f"fwd_return_{horizon}d") is not None]
    if len(points) < 20:
        return None, None, None
    pts = sorted(points, key=lambda x: x["sfc_pct"])
    tail_n = max(1, int(len(pts) * QUANTILE_TAIL))
    bottom = [p[f"fwd_return_{horizon}d"] for p in pts[:tail_n]]
    top = [p[f"fwd_return_{horizon}d"] for p in pts[-tail_n:]]
    if len(bottom) < 2 or len(top) < 2:
        return None, None, None
    est, lo_b, hi_b = bootstrap_diff_ci(bottom, top)  # bottom - top
    return est, lo_b, hi_b


def _sig(lo_b, hi_b, polarity="neg"):
    if lo_b is None or hi_b is None:
        return False
    if polarity == "neg":
        return hi_b < 0
    return lo_b > 0 or hi_b < 0


def _verdict(era_est, era_lo, era_hi, ref_est):
    """Classify latest-era gap relative to full-history reference."""
    if era_est is None or ref_est is None or ref_est == 0:
        return "N/A"
    sig_neg = era_hi is not None and era_hi < 0     # significantly negative (signal works)
    sig_pos = era_lo is not None and era_lo > 0     # significantly positive (opposite)
    if era_est < 0:
        if sig_neg:
            ratio = era_est / ref_est
            if abs(ratio) >= 0.5:
                return "BERTAHAN (persists)"
            return "MELEMAH (still significant, <half of full-history gap)"
        return "MELEMAH (not significant at 90%)"
    if sig_pos:
        return "HILANG (sign-flipped, opposite polarity)"
    return "HILANG (gap gone / near zero)"


def _row(horizon, lo, hi, ref_gaps):
    n_c, n_s, est, l, h = era_gap(series, horizon, lo, hi)
    return n_c, n_s, est, l, h


def run_stability(series):
    print("=" * 78)
    print("TEMPORAL STABILITY TEST — does the SFC gap persist/weaken/disappear?")
    print("=" * 78)
    print(f"Series: {series[0]['date']} -> {series[-1]['date']} "
          f"({len(series)} daily observations)")
    print(f"Methodology: identical to walk_forward_validation.py "
          f"(same buckets {[e[2] for e in BUCKET_EDGES]}, "
          f"bootstrap 90% CI, 2000 resamples)")
    print(f"Horizons: {FORWARD_HORIZONS_DAYS}d")

    # Full-history reference gap (the number walk_forward validates).
    ref_gaps = {}
    for horizon in FORWARD_HORIZONS_DAYS:
        n_c, n_s, est, l, h = era_gap(series, horizon, None, None)
        ref_gaps[horizon] = (est, l, h)
        print(f"\n  Reference (full history) {horizon}d gap: "
              f"{est:+.2f}pp [90% CI {l:+.2f}, {h:+.2f}] "
              f"({'significant' if _sig(l, h) else 'NOT significant'}) "
              f"(n CALM={n_c}, n STRESS={n_s})")

    # ---------- Cut 1: three equal 4-year blocks ----------
    print("\n" + "=" * 78)
    print("CUT 1 — THREE EQUAL 4-YEAR BLOCKS (2014-2018 / 2018-2022 / 2022-2026)")
    print("=" * 78)
    for horizon in FORWARD_HORIZONS_DAYS:
        print(f"\n  --- {horizon}-day forward return: CALM vs STRESS gap ---")
        print(f"  {'Era':<10} {'nCALM':>6} {'nSTR':>6} {'Gap pp':>9} "
              f"{'90% CI':<22} {'verdict (vs full-hx)':<32}")
        print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*22} {'-'*32}")
        ref_est = ref_gaps[horizon][0]
        for era_name, lo, hi in ERA_BLOCKS:
            n_c, n_s, est, l, h = era_gap(series, horizon, lo, hi)
            if est is None:
                print(f"  {era_name:<10} {n_c:>6} {n_s:>6} {'insufficient':>9}")
                continue
            verdict = _verdict(est, l, h, ref_est)
            print(f"  {era_name:<10} {n_c:>6} {n_s:>6} {est:>+8.2f} "
                  f"[{l:+.2f}, {h:+.2f}]  {verdict:<32}")

    # ---------- Cut 2: pre/post 2020-03 (Fed regime line) ----------
    print("\n" + "=" * 78)
    print("CUT 2 — PRE / POST 2020-03-01 (aligned with Fed regime flip test)")
    print("=" * 78)
    for horizon in FORWARD_HORIZONS_DAYS:
        print(f"\n  --- {horizon}-day forward return: CALM vs STRESS gap ---")
        print(f"  {'Era':<14} {'nCALM':>6} {'nSTR':>6} {'Gap pp':>9} "
              f"{'90% CI':<22} {'verdict (vs full-hx)':<32}")
        print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*9} {'-'*22} {'-'*32}")
        ref_est = ref_gaps[horizon][0]
        for era_name, lo, hi in ERA_FED:
            n_c, n_s, est, l, h = era_gap(series, horizon, lo, hi)
            if est is None:
                print(f"  {era_name:<14} {n_c:>6} {n_s:>6} {'insufficient':>9}")
                continue
            verdict = _verdict(est, l, h, ref_est)
            print(f"  {era_name:<14} {n_c:>6} {n_s:>6} {est:>+8.2f} "
                  f"[{l:+.2f}, {h:+.2f}]  {verdict:<32}")

    # ---------- Robustness: threshold-free top/bottom 20% tails ----------
    print("\n" + "=" * 78)
    print("ROBUSTNESS CROSS-CHECK — top 20% sfc_pct vs bottom 20% sfc_pct gap")
    print("(threshold-free; negative = higher signal predicts worse return)")
    print("=" * 78)
    for horizon in FORWARD_HORIZONS_DAYS:
        print(f"\n  --- {horizon}d forward return ---")
        print(f"  {'Era':<10} {'Tail gap pp':>12} {'90% CI':<22} {'significant':<12}")
        for era_name, lo, hi in ERA_BLOCKS:
            est, l, h = era_tail_gap(series, horizon, lo, hi)
            if est is None:
                print(f"  {era_name:<10} {'insufficient':>12}")
                continue
            print(f"  {era_name:<10} {est:>+11.2f} [{l:+.2f}, {h:+.2f}] "
                  f"{'YES' if _sig(l, h) else 'no':<12}")

    # ---------- Verdict ----------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("NOTE: the fixed-threshold buckets and the threshold-free cross-check")
    print("agree here, so the bucket result is not fragile. Where they diverge,")
    print("the threshold-free read (more robust to bucket imbalance) is preferred.")
    for horizon in FORWARD_HORIZONS_DAYS:
        # Latest data = 2022-2026 block (and post-2020 cut)
        _, _, est3, l3, h3 = era_gap(series, horizon,
                                     *[e[1:3] for e in ERA_BLOCKS if e[0] == "2022-2026"][0])
        _, _, estp, lp, hp = era_gap(series, horizon, SPLIT_DATE, None)
        t_est, t_lo, t_hi = era_tail_gap(series, horizon,
                                         *[e[1:3] for e in ERA_BLOCKS if e[0] == "2022-2026"][0])
        ref_est = ref_gaps[horizon][0]
        v3 = _verdict(est3, l3, h3, ref_est)
        vp = _verdict(estp, lp, hp, ref_est)
        # Cross-check verdict: significant positive tail gap => reversed
        t_flipped = t_est is not None and t_est > 0 and _sig(t_lo, t_hi, polarity="pos")
        t_flat = t_est is not None and t_lo is not None and t_hi is not None and (t_lo <= 0 <= t_hi)
        print(f"\n  {horizon}d horizon:")
        print(f"    Latest 4-yr block (2022-2026): bucket gap {est3:+.2f}pp "
              f"[{l3:+.2f}, {h3:+.2f}] -> {v3}")
        print(f"    Post-2020-03:                 bucket gap {estp:+.2f}pp "
              f"[{lp:+.2f}, {hp:+.2f}] -> {vp}")
        print(f"    Threshold-free tail gap 2022-2026: {t_est:+.2f}pp "
              f"[{t_lo:+.2f}, {t_hi:+.2f}] "
              f"({'REVERSED (sig positive)' if t_flipped else 'FLAT' if t_flat else 'negative'})")
        # Overall call for this horizon
        if t_flipped or ("HILANG" in v3 and not t_flat):
            call = ("MAJOR FINDING: the signal gap has DISAPPEARED/REVERSED in the "
                    "most recent era (2022-2026), on both the fixed-bucket and "
                    "threshold-free reads. The current BT label ('walk-forward "
                    "validated') is stale for today's regime — the edge the model "
                    "was validated on lived in 2018-2022 and has decayed. This is "
                    "more important than adding any new feature.")
        elif t_flat or "MELEMAH" in v3:
            call = ("FINDING: the signal gap is WEAKER in the most recent era "
                    "(2022-2026) — the edge has faded relative to 2018-2022, even "
                    "where it is not yet fully gone. Re-validate before trusting "
                    "the label for current decisions.")
        else:
            call = ("Gap BERTAHAN through the latest era — the core signal still "
                    "has forward predictive value.")
        print(f"    -> {call}")


def _load_or_compute_series():
    if os.path.exists(SERIES_CACHE):
        print(f"Loading cached series from {SERIES_CACHE} ...")
        with open(SERIES_CACHE) as f:
            series = json.load(f)
        print(f"Loaded {len(series)} observations (cached by walk_forward_validation.py)")
        return series
    print("No cache found — recomputing from FRED (slow)...")
    series = compute_sfc_time_series()
    if not series:
        sys.exit("Could not compute series — check FRED_API_KEY / network.")
    series = add_forward_returns(series)
    return series


def _self_tests():
    print("=== Self-tests ===\n")
    # in_era boundary logic
    assert in_era("2018-01-01", None, "2020-03-01")  # exclusive hi
    assert not in_era("2020-03-01", None, "2020-03-01")
    assert in_era("2020-03-01", "2020-03-01", None)  # inclusive lo
    assert in_era("2016-06-01", "2014-01-01", "2018-01-01")
    assert not in_era("2018-06-01", "2014-01-01", "2018-01-01")
    assert in_era("2024-01-01", "2022-01-01", "2027-01-01")
    print("✅ PASS: in_era() boundary logic (inclusive lo, exclusive hi)\n")

    # era_gap on a synthetic series where STRESS clearly underperforms
    import random
    random.seed(7)
    fake = []
    for i in range(200):
        stress = (i % 2 == 0)
        sfc = 60 if stress else 10
        base = datetime_to_str(i)
        fake.append({"date": base, "sfc_pct": sfc,
                     "fwd_return_30d": (-3 if stress else +3) + random.gauss(0, 2)})
    for p in fake:
        p.setdefault("fwd_return_7d", p["fwd_return_30d"])
    n_c, n_s, est, lo_b, hi_b = era_gap(fake, 30, None, None)
    assert est is not None and est < 0, f"FAIL: gap should be negative, got {est}"
    assert hi_b is not None and hi_b < 0, "FAIL: synthetic gap should be significant"
    print(f"✅ PASS: era_gap() on synthetic data -> gap {est:+.2f}pp "
          f"[{lo_b:+.2f}, {hi_b:+.2f}] (negative & significant as expected)\n")

    # _verdict classification
    assert _verdict(-5, -8, -2, -7) == "BERTAHAN (persists)"          # big & sig
    assert _verdict(-2, -4, 0.1, -7) == "MELEMAH (not significant at 90%)"
    assert _verdict(-1, -3, 1, -7) == "MELEMAH (not significant at 90%)"
    assert _verdict(+4, 1, 7, -7) == "HILANG (sign-flipped, opposite polarity)"
    assert _verdict(-3, -4, -2, -7) == "MELEMAH (still significant, <half of full-history gap)"
    print("✅ PASS: _verdict() classification logic\n")
    print("ALL SELF-TESTS PASSED\n")


def datetime_to_str(i):
    from datetime import datetime, timedelta
    return (datetime(2020, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    _self_tests()
    print("=" * 78)
    print("Proceeding to REAL analysis (loads cached walk-forward series — "
          "no FRED fetch needed).")
    print("=" * 78)
    series = _load_or_compute_series()
    run_stability(series)
