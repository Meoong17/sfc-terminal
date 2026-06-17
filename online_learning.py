#!/usr/bin/env python3
"""EWMA online learning correction for stress prediction.

Provides incremental Exponentially Weighted Moving Average (EWMA) baseline
tracking and a simple 1D Kalman filter for smoothing noisy predictions.
"""

import json
import os
import sys
from typing import Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
STATE_PATH = os.path.join(MODELS_DIR, "ewma_state.json")


# ──────────────────────────────────────────────────────────────────────
# OnlineEWMA
# ──────────────────────────────────────────────────────────────────────

class OnlineEWMA:
    """Incremental EWMA (Exponentially Weighted Moving Average) tracker.

    Maintains a running baseline and a bounded history of corrected values.
    """

    def __init__(self, alpha: float = 0.15, window: int = 30) -> None:
        if not 0 < alpha <= 1:
            alpha = 0.15
        if window < 1:
            window = 30
        self.alpha: float = alpha
        self.window: int = window
        self.baseline: float = 0.0
        self.history: list[float] = []

    # ------------------------------------------------------------------

    def update(self, value: float) -> float:
        """Feed a new raw value and return the EWMA-corrected value.

        If history is empty, the first value seeds the baseline.
        Subsequent values apply the EWMA: baseline = alpha*value + (1-alpha)*baseline.
        """
        if not self.history:
            self.baseline = float(value)
            self.history.append(self.baseline)
            return self.baseline

        self.baseline = self.alpha * float(value) + (1.0 - self.alpha) * self.baseline
        self.history.append(self.baseline)

        # Keep history bounded
        if len(self.history) > self.window * 2:
            self.history = self.history[-self.window:]

        return self.baseline

    # ------------------------------------------------------------------

    def get_baseline(self) -> float:
        """Return the current EWMA baseline."""
        return self.baseline

    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None) -> bool:
        """Restore EWMA state from *path* (default: ``models/ewma_state.json``).

        Returns ``True`` on success, ``False`` if the file is missing or
        corrupt (graceful fallback — leaves current state untouched).
        """
        p = path or STATE_PATH
        try:
            with open(p, "r") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

        try:
            self.alpha = float(data.get("alpha", self.alpha))
            self.window = int(data.get("window", self.window))
            self.baseline = float(data.get("baseline", self.baseline))
            raw_history = data.get("history", [])
            self.history = [float(v) for v in raw_history]
        except (TypeError, ValueError):
            return False

        return True

    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> bool:
        """Persist current EWMA state to *path*.

        Returns ``True`` on success, ``False`` on failure (graceful
        fallback — never raises).
        """
        p = path or STATE_PATH
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(
                    {
                        "baseline": self.baseline,
                        "history": self.history,
                        "alpha": self.alpha,
                        "window": self.window,
                    },
                    fh,
                    indent=2,
                )
            return True
        except (OSError, TypeError):
            return False


# ──────────────────────────────────────────────────────────────────────
# AdaptiveKalman
# ──────────────────────────────────────────────────────────────────────

class AdaptiveKalman:
    """Simple 1D Kalman filter for smoothing noisy stress predictions.

    Tracks a single state (estimate) and its uncertainty (error covariance).
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        if process_noise <= 0:
            process_noise = 0.01
        if measurement_noise <= 0:
            measurement_noise = 0.1

        self.Q: float = process_noise       # process noise covariance
        self.R: float = measurement_noise   # measurement noise covariance
        self.x: float = 0.0                 # state estimate
        self.P: float = 1.0                 # error covariance
        self.initialized: bool = False

    # ------------------------------------------------------------------

    def update(self, measurement: float) -> float:
        """Feed a measurement and return the filtered (posterior) estimate.

        The first call seeds the filter; subsequent calls perform the
        standard predict-correct cycle.
        """
        m = float(measurement)

        if not self.initialized:
            self.x = m
            self.P = 1.0
            self.initialized = True
            return self.x

        # --- Predict ---
        x_pred = self.x
        P_pred = self.P + self.Q

        # --- Update (correct) ---
        K = P_pred / (P_pred + self.R)            # Kalman gain
        self.x = x_pred + K * (m - x_pred)        # posterior estimate
        self.P = (1.0 - K) * P_pred               # posterior covariance

        return self.x


# ──────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ──────────────────────────────────────────────────────────────────────

def load_ewma() -> OnlineEWMA:
    """Load persisted EWMA state or return a fresh OnlineEWMA instance."""
    ewma = OnlineEWMA()
    ewma.load()  # graceful — no-op on failure
    return ewma


def save_ewma(ewma: OnlineEWMA) -> bool:
    """Persist an OnlineEWMA instance to the default state path."""
    return ewma.save()


def correct_stress(
    raw_stress: float,
    confidence: float = 1.0,
    transition_risk: float = 0.0,
    ewma: Optional[OnlineEWMA] = None,
) -> float:
    """Apply EWMA correction and confidence / transition-risk adjustment.

    Parameters
    ----------
    raw_stress : float
        Raw stress prediction (typically 0-1).
    confidence : float
        Prediction confidence (0-1).  Lower confidence pulls the result
        back toward the EWMA baseline.
    transition_risk : float
        Transition risk multiplier (0-1).  Higher risk dampens the
        correction, keeping the value closer to the raw input.
    ewma : OnlineEWMA, optional
        EWMA tracker.  If ``None``, a module-level singleton is used.

    Returns
    -------
    float
        Corrected stress value, clipped to [0, 1].
    """
    if ewma is None:
        ewma = _default_ewma

    corrected = ewma.update(raw_stress)

    # Blend between raw stress and EWMA baseline based on confidence
    # Low confidence → trust the baseline more.
    blended = confidence * raw_stress + (1.0 - confidence) * corrected

    # Transition risk dampens the correction toward raw
    if transition_risk > 0:
        blended = (1.0 - transition_risk) * blended + transition_risk * raw_stress

    return max(0.0, min(1.0, blended))


# Module-level singleton used by ``correct_stress`` when no ewma is passed.
_default_ewma = load_ewma()


# ──────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Standalone demo of both filters with synthetic data."""
    print("=" * 60)
    print("OnlineEWMA Demo")
    print("=" * 60)

    ewma = OnlineEWMA(alpha=0.15, window=30)
    test_values = [0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.6, 0.85]
    print(f"{'Raw':>6} | {'Corrected':>9} | {'Baseline':>8}")
    print("-" * 30)
    for v in test_values:
        c = ewma.update(v)
        print(f"{v:6.3f} | {c:9.4f} | {ewma.get_baseline():8.4f}")

    # Persist and reload
    assert ewma.save(), "Save failed"
    ewma2 = OnlineEWMA()
    assert ewma2.load(), "Load failed"
    assert abs(ewma2.get_baseline() - ewma.get_baseline()) < 1e-9, "State mismatch after load"

    print("\n✓ Save / load round-trip OK")

    # ── Kalman ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AdaptiveKalman Demo")
    print("=" * 60)

    kf = AdaptiveKalman()
    noisy = [0.3, 0.35, 0.32, 0.38, 0.31, 0.36, 0.33, 0.37, 0.34, 0.39]
    print(f"{'Meas':>6} | {'Filtered':>8}")
    print("-" * 18)
    for m in noisy:
        f = kf.update(m)
        print(f"{m:6.3f} | {f:8.4f}")

    # ── correct_stress ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("correct_stress Demo")
    print("=" * 60)

    ewma3 = OnlineEWMA(alpha=0.15, window=30)
    scenarios = [
        (0.7, 0.9, 0.0),
        (0.7, 0.5, 0.0),
        (0.7, 0.9, 0.8),
        (0.2, 0.3, 0.0),
        (0.9, 0.8, 0.5),
    ]
    print(f"{'Raw':>6} {'Conf':>5} {'Risk':>5} | {'Corrected':>9}")
    print("-" * 32)
    for raw, conf, risk in scenarios:
        out = correct_stress(raw, conf, risk, ewma3)
        print(f"{raw:6.2f} {conf:5.2f} {risk:5.2f} | {out:9.4f}")

    # Verify cleanup of old state file
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
        print(f"\n✓ Cleaned up {STATE_PATH}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
