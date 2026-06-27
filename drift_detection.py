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

# KS test significance threshold
# Lower = more sensitive to drift
# 0.05 = standard, 0.01 = strict
KS_P_THRESHOLD = 0.05

# How many consecutive drift detections before flagging
MIN_CONSECUTIVE_DRIFT = 3

# Features to normalize (same as data_quality.py)
NORMALIZE_PCT = {"m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m6_regime_score"}


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

        # Test each feature
        drifted_fields = []
        drift_scores = {}

        for i in range(min(N_METHODS, len(scores))):
            field = METHOD_FIELDS[i] if i < len(METHOD_FIELDS) else f"m{i+1}"
            current_val = scores[i]
            ref_vals = ref[:, i]

            # Skip if reference has no variance
            if np.std(ref_vals) < 1e-10:
                continue

            # KS test
            ks_stat, p_value = self._ks_test(ref_vals, current_val)

            is_drifted = p_value < KS_P_THRESHOLD

            # Track consecutive drift per field
            if is_drifted:
                self._drift_count[field] = self._drift_count.get(field, 0) + 1
            else:
                self._drift_count[field] = 0

            consecutive = self._drift_count.get(field, 0)
            drift_scores[field] = {
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "consecutive": consecutive,
                "current": round(float(current_val), 4),
                "ref_mean": round(float(np.mean(ref_vals)), 4),
                "ref_std": round(float(np.std(ref_vals)), 4),
                "drifted": is_drifted and consecutive >= MIN_CONSECUTIVE_DRIFT,
            }

            if is_drifted and consecutive >= MIN_CONSECUTIVE_DRIFT:
                drifted_fields.append(field)

        # Overall
        result["drift_detected"] = len(drifted_fields) > 0
        result["drifted_fields"] = drifted_fields
        result["drift_scores"] = drift_scores
        result["overall_drift_index"] = round(
            len(drifted_fields) / max(1, N_METHODS), 3
        )

        # Consecutive drift tracking (any drift = +1, no drift = reset)
        total_drifted = len(drifted_fields)
        if total_drifted > 0:
            result["consecutive_drift"] = self._consecutive_drift + 1 if hasattr(self, '_consecutive_drift') else 1
        else:
            self._consecutive_drift = 0
            result["consecutive_drift"] = 0

        # Stable = no drift for 10+ cycles
        result["stable"] = self._drift_free_streak >= 10 if hasattr(self, '_drift_free_streak') else True

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
                val = val / 100.0
            elif i < len(METHOD_FIELDS) and METHOD_FIELDS[i] == "m5_qreg":
                val = val / 10.0
            result.append(val)
        return result

    def _ks_test(self, reference: np.ndarray, current: float) -> Tuple[float, float]:
        """Two-sample KS test: reference distribution vs single current value.

        Uses scipy when available, falls back to approximate test.

        Returns:
            (ks_statistic, p_value)
        """
        try:
            from scipy.stats import ks_2samp

            # Create "current" as a distribution of n=3 similar values
            # (single point KS is degenerate)
            current_dist = np.array([current - 0.001, current, current + 0.001])
            stat, p = ks_2samp(reference, current_dist)
            return stat, p

        except ImportError:
            # Fallback: z-score based approximate drift
            mean = float(np.mean(reference))
            std = float(np.std(reference))
            if std < 1e-10:
                return 1.0, 1.0

            z = abs(current - mean) / std

            # Approximate p-value from z-score (normal distribution)
            p_approx = 2.0 * (1.0 - self._normal_cdf(z))

            # KS stat ~ z-score normalized
            ks_stat = min(1.0, z / 10.0)

            return ks_stat, p_approx

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

    # Test with injected drift
    print(f"\n── Test: Injected Drift (M1 set to 99.9) ──")
    injected = list(base_scores)
    if len(injected) > 0:
        injected[0] = 99.9  # M1 extremely high
    result2 = dd.check(injected)
    m1_info = result2["drift_scores"].get("m1_klr", {})
    print(f"  M1 p_value: {m1_info.get('p_value', 'N/A')}")
    print(f"  M1 drifted: {m1_info.get('drifted', 'N/A')}")
    print(f"  Overall drifted: {result2['drifted_fields']}")
    print(f"  Detected: {'YES' if result2['drift_detected'] else 'NO'}")

    print(f"\nStats: {dd.get_stats()}")
    print("\n✓ Drift detection test complete")


if __name__ == "__main__":
    main()
