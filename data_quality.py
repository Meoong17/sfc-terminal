#!/usr/bin/env python3
"""
data_quality.py — Outlier Detection + Kalman Imputation for SFC
================================================================
Two-stage data quality pipeline:

Stage 1: IsolationForest outlier detection on method scores
Stage 2: Multi-dimensional Kalman filter imputation for missing/outlier values

Usage:
    from data_quality import DataQualityPipeline
    dq = DataQualityPipeline()
    cleaned_scores, flags = dq.process(method_scores)
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Config ──
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".data_quality_state.json")

# Method field order (M1-M31) — must match collect.py's output
METHOD_FIELDS = [
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
    "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
    "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
    "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
    "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
]
N_METHODS = len(METHOD_FIELDS)  # 31

# Outlier threshold: contamination rate for IsolationForest
OUTLIER_CONTAMINATION = 0.05  # 5% expected outliers

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 0.05   # Q: how fast the state can change
KALMAN_MEASUREMENT_NOISE = 0.2  # R: how noisy the measurements are


# ════════════════════════════════════════════════════════════════
# 1D Kalman Filter (per-method tracking)
# ════════════════════════════════════════════════════════════════


class KalmanFilter1D:
    """Single-variable Kalman filter for smoothing & imputation.

    Tracks one method score over time. When a measurement is None or
    flagged as outlier, the filter predicts forward (imputes) instead of
    updating with a bad measurement.
    """

    __slots__ = ("x", "P", "Q", "R", "initialized")

    def __init__(self, q: float = KALMAN_PROCESS_NOISE, r: float = KALMAN_MEASUREMENT_NOISE):
        self.Q = q       # process noise
        self.R = r       # measurement noise
        self.x = 0.0     # state estimate
        self.P = 1.0     # error covariance
        self.initialized = False

    def update(self, measurement: Optional[float], force: bool = False) -> float:
        """Update filter with a measurement, or predict forward if None.

        Args:
            measurement: Observed value, or None to impute (predict only).
            force: If True, accept the measurement even if it seems outlier-ish.

        Returns:
            Filtered (or imputed) value.
        """
        if not self.initialized:
            if measurement is not None:
                self.x = float(measurement)
                self.P = 1.0
                self.initialized = True
            return self.x

        # Predict step
        x_pred = self.x
        P_pred = self.P + self.Q

        if measurement is not None:
            # Update (correct) step
            K = P_pred / (P_pred + self.R)   # Kalman gain
            innovation = float(measurement) - x_pred
            self.x = x_pred + K * innovation
            self.P = (1.0 - K) * P_pred
        else:
            # No measurement — just predict forward (imputation)
            self.x = x_pred
            self.P = P_pred

        return self.x

    def reset(self, value: float = 0.0) -> None:
        self.x = value
        self.P = 1.0
        self.initialized = False


# ════════════════════════════════════════════════════════════════
# DataQualityPipeline
# ════════════════════════════════════════════════════════════════


class DataQualityPipeline:
    """Two-stage data quality: outlier detection → Kalman imputation.

    Stage 1: IsolationForest flags outlier method scores.
    Stage 2: Kalman filters impute the flagged values.

    Lazy-loads IsolationForest only when first used (scikit-learn dep).
    """

    def __init__(self):
        self._filters: List[KalmanFilter1D] = [KalmanFilter1D() for _ in range(N_METHODS)]
        self._history: np.ndarray = np.empty((0, N_METHODS), dtype=np.float64)
        self._isolation_forest = None
        self._forest_ready = False
        self._load_state()

    # ── Public API ──

    def process(
        self,
        method_scores: List[Optional[float]],
        force: bool = False,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Run full data quality pipeline on method scores.

        Args:
            method_scores: List of 31 method scores (None = missing).
            force: If True, skip outlier detection (forced update).

        Returns:
            (cleaned_scores, flags)
            cleaned_scores: list of 31 float values (imputed if needed)
            flags: dict with 'outliers', 'imputed', 'outlier_pct', etc.
        """
        scores = list(method_scores)
        flags: Dict[str, Any] = {
            "outliers": [],
            "imputed": [],
            "missing": [],
            "outlier_pct": 0.0,
            "active": False,
        }

        # ── Stage 1: Outlier Detection ──
        outlier_mask = self._detect_outliers(scores)

        for i in range(N_METHODS):
            if i >= len(scores) or scores[i] is None:
                flags["missing"].append(METHOD_FIELDS[i])
                scores[i] = None  # normalize
            elif outlier_mask[i]:
                flags["outliers"].append(METHOD_FIELDS[i])

        # ── Stage 2: Kalman Filter (impute outliers + missing) ──
        cleaned: List[float] = []
        for i in range(N_METHODS):
            raw = scores[i] if i < len(scores) else None
            is_outlier = outlier_mask[i] if i < len(outlier_mask) else False

            if is_outlier and not force:
                # Outlier → predict forward (impute), don't update
                imputed_val = self._filters[i].update(None)
                cleaned.append(imputed_val)
                flags["imputed"].append(METHOD_FIELDS[i])
            else:
                # Normal → update filter
                val = self._filters[i].update(raw, force=force)
                cleaned.append(val)

        # Update history
        self._append_history(np.array(cleaned, dtype=np.float64))

        flags["outlier_pct"] = round(len(flags["outliers"]) / N_METHODS, 3)
        flags["imputed_pct"] = round(len(flags["imputed"]) / N_METHODS, 3)
        flags["active"] = len(flags["outliers"]) > 0 or len(flags["imputed"]) > 0

        # Persist every 10 cycles
        if len(self._history) % 10 == 0:
            self._save_state()

        return cleaned, flags

    def process_from_dict(self, data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Extract method scores from a data.json-like dict and process."""
        scores = []
        for field in METHOD_FIELDS:
            val = data.get(field)
            if val is not None:
                try:
                    scores.append(float(val))
                except (TypeError, ValueError):
                    scores.append(None)
            else:
                scores.append(None)
        return self.process(scores)

    def get_filter_states(self) -> List[Dict[str, float]]:
        """Return current Kalman filter states for debugging."""
        states = []
        for i, kf in enumerate(self._filters):
            states.append({
                "method": METHOD_FIELDS[i] if i < len(METHOD_FIELDS) else f"m{i+1}",
                "x": round(kf.x, 4),
                "P": round(kf.P, 6),
                "initialized": kf.initialized,
            })
        return states

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline diagnostics."""
        return {
            "history_length": len(self._history),
            "forest_ready": self._forest_ready,
            "filter_initialized": sum(1 for kf in self._filters if kf.initialized),
            "last_outlier_pct": 0.0,
        }

    # ── Stage 1: Outlier Detection (IsolationForest) ──

    # ── M1-M6 are percentages (0-100), M7-M31 are decimals (0-1)
    # Normalize before outlier detection to avoid false positives
    NORMALIZE_PCT = {"m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m6_regime_score"}

    def _normalize_for_detection(self, vals: np.ndarray) -> np.ndarray:
        """Normalize method scores to comparable scale for outlier detection."""
        out = vals.copy()
        for i in range(len(out)):
            if i < len(METHOD_FIELDS) and METHOD_FIELDS[i] in self.NORMALIZE_PCT:
                # These are 0-100 → normalize to 0-1
                out[i] = out[i] / 100.0
            elif i < len(METHOD_FIELDS) and METHOD_FIELDS[i] == "m5_qreg":
                # m5_qreg is 0-10
                out[i] = out[i] / 10.0
        return out

    def _detect_outliers(self, scores: List[Optional[float]]) -> List[bool]:
        """Run IsolationForest on method scores.

        Returns boolean mask: True = outlier.
        Falls back to statistical z-score if sklearn unavailable.
        """
        n = N_METHODS
        if len(scores) < n:
            scores = scores + [None] * (n - len(scores))

        # Extract valid values
        vals = np.array([float(s) if s is not None else np.nan for s in scores[:n]])
        mask = np.zeros(n, dtype=bool)

        # If too many NaN, skip
        if np.isnan(vals).sum() > n // 2:
            return mask.tolist()

        # Impute NaN with column mean for IsolationForest
        valid_mean = np.nanmean(vals)
        vals_filled = np.nan_to_num(vals, nan=valid_mean)

        # Normalize to comparable scale before outlier detection
        vals_norm = self._normalize_for_detection(vals_filled)

        try:
            # Lazy import sklearn
            from sklearn.ensemble import IsolationForest

            if self._isolation_forest is None:
                self._isolation_forest = IsolationForest(
                    n_estimators=50,
                    contamination=OUTLIER_CONTAMINATION,
                    random_state=42,
                )
                # Fit on history if available (fix: need >1 sample for IF)
                if len(self._history) >= 5:
                    hist_norm = np.array([self._normalize_for_detection(row)
                                          for row in self._history[-50:]])
                    train_data = np.vstack([hist_norm, vals_norm.reshape(1, -1)])
                    self._isolation_forest.fit(train_data)
                else:
                    # Fallback: use z-score until enough history
                    self._forest_ready = False
                    self._isolation_forest = None
            else:
                # Incremental: refit occasionally
                if len(self._history) % 20 == 0 or not self._forest_ready:
                    # Use history for better fitting
                    if len(self._history) >= 5:
                        # Normalize history too
                        hist_norm = np.array([self._normalize_for_detection(row)
                                              for row in self._history[-50:]])
                        train_data = np.vstack([hist_norm, vals_norm.reshape(1, -1)])
                        self._isolation_forest.fit(train_data)
                        self._forest_ready = True
                    else:
                        # Not enough history yet — skip IF, rely on z-score fallback
                        self._isolation_forest = None
                        self._forest_ready = False

            if self._isolation_forest is None:
                # No IF available — use z-score fallback
                raise ValueError("IF not fitted, use z-score fallback")

            preds = self._isolation_forest.predict(vals_norm.reshape(1, -1))
            # -1 = outlier, 1 = inlier
            mask = preds[0] == -1

        except Exception:
            # Fallback: z-score based detection (works without sklearn)
            if len(self._history) >= 5:
                recent = self._history[-10:]
                # Normalize history too
                recent_norm = np.array([self._normalize_for_detection(row)
                                        for row in recent])
                means = np.mean(recent_norm, axis=0)
                stds = np.std(recent_norm, axis=0)
                z_scores = np.abs((vals_norm - means) / np.maximum(stds, 0.01))
                mask = z_scores > 3.0  # 3-sigma rule
            else:
                # Not enough history — use IQR on normalized values
                q75, q25 = np.percentile(vals_norm, [75, 25])
                iqr = q75 - q25
                if iqr > 0.01:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    mask = (vals_norm < lower) | (vals_norm > upper)
                else:
                    mask = np.zeros(n, dtype=bool)

        return mask.tolist()

    # ── History Management ──

    def _append_history(self, clean_vals: np.ndarray) -> None:
        self._history = np.vstack([self._history, clean_vals.reshape(1, -1)])
        MAX_HISTORY = 200
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

    # ── Persistence ──

    def _load_state(self) -> None:
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    state = json.load(f)
                hist = state.get("history", [])
                if hist:
                    self._history = np.array(hist, dtype=np.float64)
                filters_state = state.get("filters", [])
                for i, fs in enumerate(filters_state):
                    if i < len(self._filters):
                        self._filters[i].x = fs.get("x", 0.0)
                        self._filters[i].P = fs.get("P", 1.0)
                        self._filters[i].initialized = fs.get("init", False)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        try:
            state = {
                "history": self._history.tolist() if len(self._history) > 0 else [],
                "filters": [
                    {"x": kf.x, "P": kf.P, "init": kf.initialized}
                    for kf in self._filters
                ],
                "forest_ready": self._forest_ready,
                "updated_at": time.time(),
            }
            with open(STATE_PATH, "w") as f:
                json.dump(state, f)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════
# Module-level convenience
# ════════════════════════════════════════════════════════════════

_default_dq: Optional[DataQualityPipeline] = None


def get_data_quality() -> DataQualityPipeline:
    global _default_dq
    if _default_dq is None:
        _default_dq = DataQualityPipeline()
    return _default_dq


def clean_method_scores(
    method_scores: List[Optional[float]],
) -> Tuple[List[float], Dict[str, Any]]:
    """One-shot: run data quality pipeline on method scores."""
    dq = get_data_quality()
    return dq.process(method_scores)


# ════════════════════════════════════════════════════════════════
# Standalone Test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Test data quality pipeline with real data from data.json."""
    print("=" * 60)
    print("Data Quality Pipeline — Test")
    print("=" * 60)

    # Load real data
    data_path = os.path.join(SFC_DIR, "data.json")
    try:
        with open(data_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Build method scores from data.json
    scores = []
    for field in METHOD_FIELDS:
        v = data.get(field)
        scores.append(float(v) if v is not None else None)

    available = sum(1 for s in scores if s is not None)
    print(f"Methods: {available}/{N_METHODS} available\n")

    # Run pipeline
    dq = DataQualityPipeline()
    cleaned, flags = dq.process(scores)

    print("── Results ──")
    print(f"  Outliers flagged: {flags['outliers']}")
    print(f"  Imputed values:  {flags['imputed']}")
    print(f"  Missing:         {flags['missing']}")
    print(f"  Outlier %:       {flags['outlier_pct']*100:.1f}%")
    print(f"  Imputed %:       {flags['imputed_pct']*100:.1f}%")

    print("\n── Method Scores: Before vs After ──")
    print(f"  {'Method':<18} {'Raw':>8} {'Cleaned':>8} {'Δ':>8}")
    print("  " + "-" * 44)
    max_print = min(31, len(scores), len(cleaned))
    for i in range(max_print):
        raw = scores[i]
        clean = cleaned[i]
        field = METHOD_FIELDS[i] if i < len(METHOD_FIELDS) else f"m{i+1}"
        if raw is not None:
            delta = clean - raw
            marker = " ← outlier" if field in flags["outliers"] else ""
            print(f"  {field:<18} {raw:>8.2f} {clean:>8.2f} {delta:>+8.2f}{marker}")
        else:
            print(f"  {field:<18} {'N/A':>8} {clean:>8.2f} {'imputed':>8}")

    # Test with injected NaN
    print("\n── Test: Injected NaN ──")
    bad_scores = list(scores)
    if len(bad_scores) > 5:
        bad_scores[5] = None  # m6_regime_score
        bad_scores[10] = None  # m11_var
    cleaned2, flags2 = dq.process(bad_scores)
    print(f"  Imputed: {flags2['imputed']}")
    print(f"  M6 before: {bad_scores[5]},  after: {cleaned2[5]:.2f}")
    print(f"  M11 before: {bad_scores[10]}, after: {cleaned2[10]:.2f}")

    print(f"\nFilter states: {dq.get_stats()}")
    print("\n✓ Data quality pipeline test complete")


if __name__ == "__main__":
    main()
