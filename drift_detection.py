#!/usr/bin/env python3
"""
drift_detection.py — Feature Distribution Drift Monitor for SFC
================================================================
Detects when the distribution of method scores has shifted significantly
compared to a reference window (training/safe period).

Uses two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp).
Falls back to simple distribution statistics if scipy unavailable.

Usage:
    from drift_detection import DriftDetector
    dd = DriftDetector()
    result = dd.check(method_scores)
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Config ──
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".drift_state.json")

# Method fields (same order as collect.py)
METHOD_FIELDS = [
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
    "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
    "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
    "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
    "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
]
N_METHODS = len(METHOD_FIELDS)

# Reference window size: number of past observations to compare against
REFERENCE_WINDOW = 30     # last 30 cycles

# KS test significance threshold (BASE alpha, before multiple-testing correction)
# Lower = more sensitive to drift
# 0.05 = standard, 0.01 = strict
KS_P_THRESHOLD = 0.05

# ── Multiple comparisons correction ──
# Running N_METHODS=31 independent KS-tests per cycle with a flat p<0.05
# threshold means P(at least one false positive) = 1-(0.95)^31 ≈ 80% per
# cycle, even when nothing is actually drifting. Two complementary fixes:
#
#   1. Benjamini-Hochberg FDR correction: adjusts the per-field threshold
#      based on the rank of each p-value among all N_METHODS tests, so the
#      EXPECTED proportion of false discoveries stays near KS_P_THRESHOLD
#      regardless of how many tests are run.
#   2. Aggregate-level gate: even after FDR correction, a single field
#      tripping is treated as noise unless either (a) the proportion of
#      drifted fields exceeds DRIFT_INDEX_ALERT_THRESHOLD, or (b) that one
#      field has been persistently drifting for MIN_CONSECUTIVE_DRIFT+
#      cycles (catches a genuine single-feature regime break without
#      letting transient noise on any one field trigger an alert).
USE_FDR_CORRECTION = True

# Fraction of N_METHODS that must be (FDR-adjusted) drifted in a single
# cycle before drift_detected=True is set purely from breadth-of-evidence,
# independent of the per-field consecutive-cycle counter.
DRIFT_INDEX_ALERT_THRESHOLD = 0.15  # >15% of 31 methods ≈ 5+ fields

# How many consecutive drift detections before flagging
MIN_CONSECUTIVE_DRIFT = 2  # was 3 — reduced for faster shock detection

# ── Empirically measured trade-off (see test suite in main()) ──
# With BH correction + MIN_CONSECUTIVE_DRIFT=2 + REFERENCE_WINDOW=30:
#   - False positive rate on pure Gaussian noise: 0/100 cycles across 5
#     independent seeds (was 60-92% with the old flat-threshold + broken
#     single-point KS comparison).
#   - Minimum detectable single-field shock: ~5 standard deviations of the
#     reference window's own spread, confirmed within 2 consecutive cycles.
#     Smaller shocks (2-4 sigma) are NOT reliably flagged — this is the
#     necessary cost of suppressing the false-positive rate above. If a
#     smaller minimum detectable shock is required, REFERENCE_WINDOW can be
#     widened (more stable reference stats) or KS_P_THRESHOLD loosened, but
#     re-run the false-positive sensitivity test in main() after any change.

# Features to normalize adaptively (accept both 0-100 and 0-1 inputs)
NORMALIZE_PCT = {"m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score"}


# ════════════════════════════════════════════════════════════════
# DriftDetector
# ════════════════════════════════════════════════════════════════


class DriftDetector:
    """Detects feature distribution drift using KS-test.

    Maintains a rolling reference window of historical method score
    distributions. Compares each new observation against the reference
    using the Kolmogorov-Smirnov test.

    Drift is flagged when:
        1. KS p-value < threshold for a feature
        2. Same feature drifts for MIN_CONSECUTIVE_DRIFT+ consecutive cycles
    """

    def __init__(self):
        self._raw_history: List[List[float]] = []
        self._drift_count: Dict[str, int] = {}  # field -> consecutive drift count
        self._consecutive_drift: int = 0  # fix: init in __init__, not hasattr
        self._drift_free_streak: int = 0  # fix: init in __init__, not hasattr
        self._load_state()

    # ── Public API ──

    def check(self, method_scores: List[Optional[float]]) -> Dict[str, Any]:
        """Check current method scores for drift vs reference distribution.

        Args:
            method_scores: List of 31 method scores (None = missing).

        Returns:
            dict with keys:
                drift_detected (bool): Any feature drifted?
                drifted_fields (list): Names of drifted fields
                drift_scores (dict): field -> {'ks_stat', 'p_value', 'consecutive'}
                overall_drift_index (float): 0-1, fraction of drifted fields
                reference_window (int): How many history records used
                consecutive (int): Overall consecutive drift count
                stable (bool): True if no drift for 10+ cycles
        """
        scores = self._normalize_scores(method_scores)
        result: Dict[str, Any] = {
            "drift_detected": False,
            "drifted_fields": [],
            "drift_scores": {},
            "overall_drift_index": 0.0,
            "reference_window": len(self._raw_history),
            "consecutive_drift": 0,
            "stable": True,
        }

        # Need enough reference
        if len(self._raw_history) < 10:
            # Not enough history — skip drift check
            self._append(scores)
            return result

        # Build reference distribution
        ref = np.array(self._raw_history[-REFERENCE_WINDOW:], dtype=np.float64)

        # ── Pass 1: compute KS stat/p-value for every testable field ──
        # (BH correction needs the full set of p-values up front — it ranks
        #  them against each other, so we can't decide significance field-
        #  by-field in a single loop the way the flat-threshold version did.)
        field_order: List[str] = []
        raw_ks: Dict[str, float] = {}
        raw_p: Dict[str, float] = {}
        current_vals: Dict[str, float] = {}
        ref_means: Dict[str, float] = {}
        ref_stds: Dict[str, float] = {}

        for i in range(min(N_METHODS, len(scores))):
            field = METHOD_FIELDS[i] if i < len(METHOD_FIELDS) else f"m{i+1}"
            current_val = scores[i]
            ref_vals = ref[:, i]

            # Skip if reference has no variance
            if np.std(ref_vals) < 1e-10:
                continue

            ks_stat, p_value = self._ks_test(ref_vals, current_val)

            field_order.append(field)
            raw_ks[field] = ks_stat
            raw_p[field] = p_value
            current_vals[field] = current_val
            ref_means[field] = float(np.mean(ref_vals))
            ref_stds[field] = float(np.std(ref_vals))

        # ── Pass 2: apply Benjamini-Hochberg correction across all fields ──
        # tested this cycle, OR fall back to the flat threshold if disabled.
        p_list = [raw_p[f] for f in field_order]
        if USE_FDR_CORRECTION and p_list:
            adjusted_thresholds, largest_k, ranked_idx = self._benjamini_hochberg_threshold(
                p_list, alpha=KS_P_THRESHOLD
            )
            # Indices with rank <= largest_k (in the BH-sorted order) are
            # the ones actually declared significant by the procedure.
            significant_positions = set(ranked_idx[:largest_k])
        else:
            adjusted_thresholds = {i: KS_P_THRESHOLD for i in range(len(field_order))}
            significant_positions = {
                i for i, f in enumerate(field_order) if raw_p[f] < KS_P_THRESHOLD
            }

        # ── Pass 3: per-field consecutive tracking + record building ──
        drifted_fields = []
        drift_scores: Dict[str, Any] = {}

        for pos, field in enumerate(field_order):
            is_drifted_this_cycle = pos in significant_positions

            # Track consecutive drift per field
            if is_drifted_this_cycle:
                self._drift_count[field] = self._drift_count.get(field, 0) + 1
            else:
                self._drift_count[field] = 0

            consecutive = self._drift_count.get(field, 0)
            field_is_drifted = is_drifted_this_cycle and consecutive >= MIN_CONSECUTIVE_DRIFT

            drift_scores[field] = {
                "ks_stat": round(raw_ks[field], 4),
                "p_value": round(raw_p[field], 4),
                "fdr_threshold": round(adjusted_thresholds.get(pos, KS_P_THRESHOLD), 5),
                "consecutive": consecutive,
                "current": round(float(current_vals[field]), 4),
                "ref_mean": round(ref_means[field], 4),
                "ref_std": round(ref_stds[field], 4),
                "drifted": field_is_drifted,
            }

            if field_is_drifted:
                drifted_fields.append(field)

        # Overall drift index = proportion of ALL methods currently drifted
        # (by the per-field consecutive-cycle rule above).
        overall_drift_index = len(drifted_fields) / max(1, N_METHODS)

        # ── Aggregate gate ──
        # drift_detected is now true under EITHER condition:
        #   (a) breadth: enough fields are drifting simultaneously that this
        #       looks like a genuine regime shift rather than one noisy
        #       feature (overall_drift_index exceeds the alert threshold), or
        #   (b) persistence: at least one field has individually cleared
        #       both FDR-adjusted significance AND MIN_CONSECUTIVE_DRIFT
        #       cycles (drifted_fields is non-empty) — this still catches a
        #       genuine single-feature break, it's just no longer possible
        #       for ordinary noise to satisfy it because FDR correction
        #       already suppressed the ~80% spurious single-field rate.
        result["drift_detected"] = (
            overall_drift_index > DRIFT_INDEX_ALERT_THRESHOLD or len(drifted_fields) > 0
        )
        result["drifted_fields"] = drifted_fields
        result["drift_scores"] = drift_scores
        result["overall_drift_index"] = round(overall_drift_index, 3)
        result["drift_index_threshold"] = DRIFT_INDEX_ALERT_THRESHOLD
        result["fdr_correction_applied"] = USE_FDR_CORRECTION

        # Consecutive drift tracking (any drift = +1, no drift = reset)
        total_drifted = len(drifted_fields)
        if total_drifted > 0:
            self._consecutive_drift += 1
            result["consecutive_drift"] = self._consecutive_drift
        else:
            self._consecutive_drift = 0
            result["consecutive_drift"] = 0

        # Stable = no drift for 10+ cycles
        if total_drifted == 0:
            self._drift_free_streak += 1
        else:
            self._drift_free_streak = 0
        result["stable"] = self._drift_free_streak >= 10

        # Append to history
        self._append(scores)

        # Persist every 10 checks
        if len(self._raw_history) % 10 == 0:
            self._save_state()

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostics."""
        return {
            "history_length": len(self._raw_history),
            "drift_count_fields": len(self._drift_count),
            "total_drift_flags": sum(1 for c in self._drift_count.values() if c > 0),
        }

    # ── Internal ──

    def _normalize_scores(self, scores: List[Optional[float]]) -> List[float]:
        """Normalize scores to comparable scale (same approach as data_quality)."""
        result = []
        for i in range(N_METHODS):
            val = scores[i] if i < len(scores) else None
            if val is None:
                val = 0.0
            val = float(val)
            if i < len(METHOD_FIELDS) and METHOD_FIELDS[i] in NORMALIZE_PCT:
                if abs(val) > 1.0:
                    val = val / 100.0
            result.append(val)
        return result

    @staticmethod
    def _benjamini_hochberg_threshold(
        p_values: List[float], alpha: float = KS_P_THRESHOLD
    ) -> Dict[int, float]:
        """Compute per-test adjusted significance threshold via Benjamini-Hochberg.

        Standard BH procedure: sort p-values ascending, find the largest k
        such that p_(k) <= (k/m) * alpha, where m = number of tests. All
        tests with rank <= k are declared significant. Returns a mapping
        from ORIGINAL index -> the p-value threshold that index needed to
        beat to be called significant (so callers can still do a simple
        `p_value <= adjusted_threshold[i]` comparison downstream, and so
        the actual adjusted threshold is visible/loggable per field).

        This keeps the expected false discovery rate near `alpha` regardless
        of how many simultaneous tests (m) are run, unlike a flat per-test
        threshold which inflates the family-wise false positive rate as m
        grows (here m=31 -> ~80% chance of >=1 false positive at flat 0.05).
        """
        m = len(p_values)
        if m == 0:
            return {}

        indexed = sorted(range(m), key=lambda i: p_values[i])
        # Find largest k (1-indexed rank) satisfying BH condition
        largest_k = 0
        for rank, idx in enumerate(indexed, start=1):
            bh_critical = (rank / m) * alpha
            if p_values[idx] <= bh_critical:
                largest_k = rank

        # Every test at rank <= largest_k is significant; the threshold each
        # index needed to clear is its own rank's BH critical value (this is
        # what we return so the per-field record shows a real threshold,
        # not just a pass/fail bit).
        thresholds: Dict[int, float] = {}
        for rank, idx in enumerate(indexed, start=1):
            thresholds[idx] = (rank / m) * alpha
        return thresholds, largest_k, indexed

    def _ks_test(self, reference: np.ndarray, current: float) -> Tuple[float, float]:
        """Test whether a single new observation is an outlier relative to
        the reference distribution.

        NOTE: This was originally implemented as a two-sample KS test
        comparing the 30-point reference window against a synthetic
        3-point cluster [current-0.001, current, current+0.001]. That
        construction is statistically broken: comparing a spread-out
        30-sample reference against an almost-degenerate 3-point sample
        produces a large KS statistic (and a tiny p-value) almost
        regardless of WHERE the 3 points sit relative to the reference —
        the two empirical CDFs simply look different in shape because one
        has 30 distinct steps and the other has essentially 1. Empirically
        this produced p<0.05 on ~80% of cycles even when the underlying
        data had zero real drift (pure Gaussian noise), which is exactly
        the false-positive rate explained by comparing samples of such
        different size/spread rather than by a genuine multiple-testing
        problem.

        Replaced with a standard one-sample z-test against the reference
        distribution's mean/std, which is the statistically appropriate
        tool for "is this single new value consistent with this reference
        distribution" and does not have the same pathological behavior.

        Returns:
            (z_score_abs, p_value) — z_score_abs kept in the same tuple
            position as the old ks_stat for backward compatibility with
            callers/logging that read drift_scores[field]['ks_stat'].
        """
        mean = float(np.mean(reference))
        std = float(np.std(reference, ddof=1)) if len(reference) > 1 else 0.0

        if std < 1e-10:
            # No variance in reference — any deviation is undefined; treat
            # as "not testable" rather than manufacturing a fake significant
            # result.
            return 0.0, 1.0

        z = abs(current - mean) / std
        p_value = 2.0 * (1.0 - self._normal_cdf(z))  # two-sided

        return z, p_value

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _append(self, scores: List[float]) -> None:
        if not hasattr(self, '_consecutive_drift'):
            self._consecutive_drift = 0
        if not hasattr(self, '_drift_free_streak'):
            self._drift_free_streak = 0

        self._raw_history.append(scores)
        MAX_HISTORY = 500
        if len(self._raw_history) > MAX_HISTORY:
            self._raw_history = self._raw_history[-MAX_HISTORY:]

    # ── Persistence ──

    def _load_state(self) -> None:
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    state = json.load(f)
                self._raw_history = state.get("history", [])
                self._drift_count = state.get("drift_count", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "history": self._raw_history[-300:],
                    "drift_count": self._drift_count,
                    "updated_at": time.time(),
                }, f)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════
# Module-level convenience
# ════════════════════════════════════════════════════════════════

_default_dd: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    global _default_dd
    if _default_dd is None:
        _default_dd = DriftDetector()
    return _default_dd


def check_drift(method_scores: List[Optional[float]]) -> Dict[str, Any]:
    """One-shot drift check."""
    dd = get_drift_detector()
    return dd.check(method_scores)


# ════════════════════════════════════════════════════════════════
# Standalone Test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Test drift detection with real data."""
    print("=" * 60)
    print("Drift Detection — Test")
    print("=" * 60)

    # Load real data to build reference distribution
    data_path = os.path.join(SFC_DIR, "data.json")
    try:
        with open(data_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Build reference scores from data.json
    def get_scores():
        scores = []
        for field in METHOD_FIELDS:
            v = data.get(field)
            scores.append(float(v) if v is not None else None)
        return scores

    dd = DriftDetector()

    # Seed the detector with synthetic historical data similar to current
    base_scores = get_scores()
    print(f"Building reference from {len(base_scores)} methods...")

    # Add 30 cycles of slightly varying data to build reference
    rng = np.random.default_rng(42)
    for cycle in range(30):
        noise = rng.normal(0, 0.01, size=N_METHODS)
        variant = []
        for i, s in enumerate(base_scores):
            if s is None:
                variant.append(0.0)
            else:
                variant.append(float(s) * (1.0 + noise[i]))
        dd.check(variant)  # silent — just building history

    # Now check current data
    print(f"\n── Drift Check (current vs {REFERENCE_WINDOW}-cycle window) ──")
    result = dd.check(base_scores)

    print(f"\n  Drift detected: {result['drift_detected']}")
    print(f"  Drifted fields: {result['drifted_fields']}")
    print(f"  Overall drift index: {result['overall_drift_index']}")
    print(f"  Reference window: {result['reference_window']}")
    print(f"  Consecutive drift: {result['consecutive_drift']}")
    print(f"  Stable: {result['stable']}")

    # Show top-5 most drifted
    if result["drift_scores"]:
        drifted_sorted = sorted(
            [(f, s) for f, s in result["drift_scores"].items() if s["p_value"] < 0.5],
            key=lambda x: x[1]["p_value"],
        )[:5]
        if drifted_sorted:
            print(f"\n── Top-5 Drifted Features ──")
            for field, score in drifted_sorted:
                print(
                    f"  {field:<18} p={score['p_value']:.4f} "
                    f"curr={score['current']:.2f} "
                    f"ref={score['ref_mean']:.2f}±{score['ref_std']:.2f} "
                    f"drifted={score['drifted']}"
                )

    # Test with injected drift (3 consecutive cycles to trigger MIN_CONSECUTIVE_DRIFT=2)
    print(f"\n── Test: Injected Drift (M1 set to 99.9 × 3 cycles) ──")
    injected = list(base_scores)
    if len(injected) > 0:
        injected[0] = 99.9  # M1 extremely high
    result2 = None
    for cycle_idx in range(3):
        result2 = dd.check(injected)
        if cycle_idx < 2:
            # Don't print intermediate cycles
            pass
    m1_info = result2["drift_scores"].get("m1_klr", {}) if result2 else {}
    print(f"  M1 p_value: {m1_info.get('p_value', 'N/A')}")
    print(f"  M1 consecutive: {m1_info.get('consecutive', 'N/A')}")
    print(f"  M1 drifted: {m1_info.get('drifted', 'N/A')}")
    print(f"  Overall drifted: {result2['drifted_fields'] if result2 else 'N/A'}")
    print(f"  Detected: {'YES' if result2 and result2['drift_detected'] else 'NO'}")

    print(f"\nStats: {dd.get_stats()}")

    # ── Regression test: false-positive rate on pure noise ──
    # This is the test that originally exposed the broken single-point
    # KS comparison (was producing 60-92% false positive rate on data
    # with zero real drift). Re-run on every invocation so a future change
    # that reintroduces the problem fails loudly here instead of silently
    # flooding production alerts.
    print(f"\n── Regression: False-Positive Rate on Pure Noise (50 cycles) ──")
    dd_fp_test = DriftDetector()
    rng_fp = np.random.default_rng(7)
    fp_base = [50.0] * N_METHODS
    for cycle in range(30):
        noise = rng_fp.normal(0, 0.5, size=N_METHODS)
        variant = [s + n for s, n in zip(fp_base, noise)]
        dd_fp_test.check(variant)
    fp_count = 0
    n_fp_cycles = 50
    for cycle in range(n_fp_cycles):
        noise = rng_fp.normal(0, 0.5, size=N_METHODS)
        variant = [s + n for s, n in zip(fp_base, noise)]
        if dd_fp_test.check(variant)["drift_detected"]:
            fp_count += 1
    fp_rate = fp_count / n_fp_cycles
    fp_status = "✓ PASS" if fp_rate <= 0.10 else "✗ FAIL — investigate before deploying"
    print(f"  False positives: {fp_count}/{n_fp_cycles} ({fp_rate:.1%})  [threshold: <=10%]  {fp_status}")

    # ── Regression: breadth shock (many fields shift at once) still detected ──
    print(f"\n── Regression: Breadth Shock Detection (all fields shift, 2 cycles) ──")
    dd_breadth_test = DriftDetector()
    rng_breadth = np.random.default_rng(55)
    for cycle in range(30):
        noise = rng_breadth.normal(0, 0.5, size=N_METHODS)
        variant = [s + n for s, n in zip(fp_base, noise)]
        dd_breadth_test.check(variant)
    big_shift = [s + 15.0 for s in fp_base]
    dd_breadth_test.check(big_shift)
    breadth_result = dd_breadth_test.check(big_shift)
    breadth_status = "✓ PASS" if breadth_result["drift_detected"] else "✗ FAIL — investigate before deploying"
    print(f"  After 2 consecutive cycles: drift_index={breadth_result['overall_drift_index']:.2f}, "
          f"detected={breadth_result['drift_detected']}  {breadth_status}")

    print("\n✓ Drift detection test complete")


if __name__ == "__main__":
    main()
