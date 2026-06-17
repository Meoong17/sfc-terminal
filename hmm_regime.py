#!/usr/bin/env python3
"""
hmm_regime.py — Hidden Markov Model Regime Detection for SFC

Trains a Gaussian HMM on 5 market features to detect 4 regimes:
  BULL, BEAR, SIDEWAYS, CRISIS

Features (5): [daily_return, dvol/100, sfc_effective/100, rsi_14/100, fng/100]

Usage:
    # Standalone training
    python3 hmm_regime.py

    # As a library
    from hmm_regime import HMMRegimeDetector
    detector = HMMRegimeDetector()
    detector.fit(features_matrix)
    result = detector.predict(current_features)
"""

import json
import os
import sys
import subprocess
import pickle
import logging
import traceback
import numpy as np
from hmmlearn import hmm

# ── CONFIG ──
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SFC_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hmm_regime.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── REGIME DEFINITIONS ──
# Ordered by mean return (descending): BULL=0, BEAR=1, SIDEWAYS=2, CRISIS=3
REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "CRISIS"}
REGIME_MAP = {"BULL": 0, "BEAR": 1, "SIDEWAYS": 2, "CRISIS": 3}

# 5 features for HMM
FEATURE_COLS = [
    "daily_return",    # btc_24h (daily % change)
    "dvol",            # dvol / 100
    "sfc_effective",   # sfc_effective / 100
    "rsi_14",          # rsi_14 / 100
    "fng",             # fng / 100
]
N_FEATURES = len(FEATURE_COLS)  # 5

# ── LOGGING ──
logging.basicConfig(
    level=logging.INFO,
    format="[HMMRegime] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("hmm_regime")


# ════════════════════════════════════════════════════════════════
# HMMRegimeDetector CLASS
# ════════════════════════════════════════════════════════════════


class HMMRegimeDetector:
    """Hidden Markov Model regime detection for market regimes.

    Trains a Gaussian HMM on 5 market features to detect 4 latent regimes,
    then maps them to BULL/BEAR/SIDEWAYS/CRISIS ordered by mean daily return.
    """

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self._is_fitted = False
        self._state_order: np.ndarray | None = None  # maps HMM state idx -> regime label
        self._feature_means: np.ndarray | None = None

    # ── fit ──

    def fit(self, features: np.ndarray) -> "HMMRegimeDetector":
        """Train the HMM on a historical feature matrix.

        Args:
            features: (n_samples, n_features) numpy array.
                      Columns: [daily_return, dvol/100, sfc_effective/100,
                                rsi_14/100, fng/100]

        Returns:
            self (fitted)
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")
        if features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {features.shape[1]}"
            )

        n_samples = features.shape[0]
        if n_samples < self.n_regimes * 10:
            log.warning(
                f"Very few samples ({n_samples}) for {self.n_regimes} regimes"
            )

        # Handle NaN / Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Train Gaussian HMM with full covariance
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
            tol=1e-4,
            verbose=False,
        )
        self.model.fit(features)

        # Label regimes by mean daily_return (first feature, column 0)
        means = self.model.means_  # (n_regimes, n_features)
        daily_return_means = means[:, 0]
        sorted_indices = np.argsort(daily_return_means)[::-1]  # descending

        # Map: HMM state index -> regime label (0=BULL ... 3=CRISIS)
        self._state_order = np.zeros(self.n_regimes, dtype=np.int32)
        for label, state_idx in enumerate(sorted_indices):
            self._state_order[state_idx] = label

        self._feature_means = means
        self._is_fitted = True

        log.info(
            f"HMM trained: {n_samples} samples, {self.n_regimes} regimes"
        )
        for label in range(self.n_regimes):
            state_idx = int(np.where(self._state_order == label)[0][0])
            m = means[state_idx]
            log.info(
                f"  Regime {label} ({REGIME_NAMES[label]}): "
                f"ret={m[0]:+.4f}  dvol={m[1]:.4f}  "
                f"sfc={m[2]:.4f}  rsi={m[3]:.4f}  fng={m[4]:.4f}"
            )

        return self

    # ── predict ──

    def predict(self, features: np.ndarray) -> dict:
        """Predict current regime from a feature vector.

        Args:
            features: (n_features,) or (1, n_features) array.

        Returns:
            dict with keys:
              'regime' (str)           — regime name or 'NORMAL' on error
              'crisis_probability' (float) — 0-1 probability of CRISIS regime
              'regime_label' (int)     — numeric label (0-3)
              'state_probs' (list)     — posterior probs per regime
        """
        try:
            if not self._is_fitted or self.model is None:
                return {"regime": "NORMAL", "crisis_probability": 0.0}

            features = np.asarray(features, dtype=np.float64)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.shape[1] != N_FEATURES:
                return {"regime": "NORMAL", "crisis_probability": 0.0}

            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Most likely hidden state
            state = int(self.model.predict(features)[0])
            regime_label = int(self._state_order[state])
            regime_name = REGIME_NAMES.get(regime_label, "NORMAL")

            # Posterior probabilities over HMM states
            hmm_probs = self.model.predict_proba(features)[0]

            # Reorder to regime-label order (0=BULL, 1=BEAR, 2=SIDEWAYS, 3=CRISIS)
            regime_probs = np.zeros(self.n_regimes, dtype=np.float64)
            for hmm_idx in range(self.n_regimes):
                regime_label_idx = int(self._state_order[hmm_idx])
                regime_probs[regime_label_idx] = float(hmm_probs[hmm_idx])

            # Crisis probability = prob mass of CRISIS regime (label 3)
            crisis_prob = float(regime_probs[3])

            result = {
                "regime": regime_name,
                "regime_label": regime_label,
                "crisis_probability": round(crisis_prob, 4),
                "state_probs": [float(p) for p in regime_probs],
            }
            return result

        except Exception as exc:
            log.warning(f"Prediction failed: {exc}")
            return {"regime": "NORMAL", "crisis_probability": 0.0}

    # ── fit_from_git ──

    def fit_from_git(self) -> "HMMRegimeDetector | None":
        """Extract historical snapshots from git and train the HMM.

        Parses all data.json snapshots from git history and builds a feature
        matrix from [btc_24h, dvol/100, sfc_effective/100, rsi_14/100, fng/100].

        Returns:
            self (fitted) or None on failure.
        """
        try:
            snapshots = self._extract_snapshots()
            if len(snapshots) < 50:
                log.warning(
                    f"Too few snapshots ({len(snapshots)}), need at least 50"
                )
                return None

            X = self._build_feature_matrix(snapshots)
            if len(X) < 50:
                log.warning(
                    f"Too few samples ({len(X)}), need at least 50"
                )
                return None

            log.info(f"Feature matrix shape: {X.shape}")
            for i, col in enumerate(FEATURE_COLS):
                log.info(
                    f"  {col}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}], "
                    f"mean={X[:, i].mean():.4f}"
                )

            self.fit(X)
            return self

        except Exception as exc:
            log.error(f"fit_from_git failed: {exc}")
            traceback.print_exc()
            return None

    def _extract_snapshots(self) -> list:
        """Extract all clean data.json snapshots from git history.

        Returns:
            List of parsed data.json dicts, oldest first.
        """
        log.info("Extracting historical snapshots from git...")
        try:
            result = subprocess.check_output(
                [
                    "git", "log", "--oneline", "--all", "--diff-filter=M",
                    "--reverse", "--", "data.json",
                ],
                text=True,
                timeout=30,
                cwd=SFC_DIR,
            ).strip().split("\n")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error(f"Git log failed: {e}")
            return []

        result = [r for r in result if r.strip()]

        snapshots = []
        errors = 0
        for i, line in enumerate(result):
            sha = line.split()[0]
            try:
                content = subprocess.check_output(
                    ["git", "show", f"{sha}:data.json"],
                    text=True,
                    timeout=10,
                    cwd=SFC_DIR,
                )
                if not content or not content.strip().startswith("{"):
                    errors += 1
                    continue
                data = json.loads(content)
                snapshots.append(data)
            except (json.JSONDecodeError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                errors += 1
                continue

            if (i + 1) % 300 == 0:
                log.info(f"  Extracted {i+1}/{len(result)}...")

        log.info(
            f"Extracted {len(snapshots)} clean snapshots ({errors} skipped)"
        )
        return snapshots

    def _build_feature_matrix(self, snapshots: list) -> np.ndarray:
        """Build 5-feature matrix from snapshot dicts.

        Extracts: [btc_24h, dvol/100, sfc_effective/100,
                   rsi_14/100, fng/100]

        Args:
            snapshots: List of parsed data.json dicts.

        Returns:
            (n, 5) numpy float64 array.
        """
        X_list = []
        skipped = 0

        for snap in snapshots:
            try:
                daily_return = float(snap.get("btc_24h", 0.0) or 0.0)
                dvol = float(snap.get("dvol", 0.0) or 0.0) / 100.0
                sfc = float(snap.get("sfc_effective", 50.0) or 50.0) / 100.0
                rsi = float(snap.get("rsi_14", 50.0) or 50.0) / 100.0
                fng = float(snap.get("fng", 50.0) or 50.0) / 100.0

                X_list.append([daily_return, dvol, sfc, rsi, fng])

            except (ValueError, TypeError):
                skipped += 1
                continue

        if skipped:
            log.warning(f"Skipped {skipped} malformed snapshots")

        if not X_list:
            log.warning("No valid snapshots with required features")
            return np.empty((0, N_FEATURES), dtype=np.float64)

        X = np.array(X_list, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ── save / load ──

    def save(self, path: str | None = None) -> None:
        """Save trained model to pickle file.

        Args:
            path: Path to save.  Defaults to MODEL_PATH.
        """
        if not self._is_fitted:
            raise ValueError("No fitted model to save.  Train or load first.")

        save_path = path or MODEL_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        state = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "_state_order": self._state_order,
            "_feature_means": self._feature_means,
            "_is_fitted": self._is_fitted,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        log.info(f"Model saved to {save_path}")

    def load(self, path: str | None = None) -> "HMMRegimeDetector":
        """Load trained model from pickle file.

        Args:
            path: Path to load from.  Defaults to MODEL_PATH.

        Returns:
            self (loaded)
        """
        load_path = path or MODEL_PATH
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.n_regimes = state["n_regimes"]
        self._state_order = state["_state_order"]
        self._feature_means = state["_feature_means"]
        self._is_fitted = state["_is_fitted"]

        log.info(f"Model loaded from {load_path}")
        return self

    # ── transition matrix access ──

    @property
    def transition_matrix(self) -> np.ndarray:
        """Raw transition matrix (n_regimes × n_regimes) in HMM state order."""
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted.  Train or load first.")
        return self.model.transmat_

    @property
    def transition_matrix_labeled(self) -> dict:
        """Transition matrix labelled by regime names.

        Returns:
            {from_regime_name: {to_regime_name: prob, ...}, ...}
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted.  Train or load first.")

        trans = self.model.transmat_
        labeled: dict = {}
        for i in range(self.n_regimes):
            from_label = int(self._state_order[i])
            from_name = REGIME_NAMES[from_label]
            targets: dict[str, float] = {}
            for j in range(self.n_regimes):
                to_label = int(self._state_order[j])
                to_name = REGIME_NAMES[to_label]
                targets[to_name] = round(float(trans[i, j]), 4)
            labeled[from_name] = targets
        return labeled

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_features(self) -> int:
        return N_FEATURES


# ════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════


def load_detector(path: str | None = None) -> HMMRegimeDetector | None:
    """Load a saved HMMRegimeDetector from disk.

    Args:
        path: Path to model file.  Defaults to MODEL_PATH.

    Returns:
        HMMRegimeDetector instance or None on failure.
    """
    load_path = path or MODEL_PATH
    if not os.path.exists(load_path):
        log.warning(f"No trained model found at {load_path}")
        return None
    try:
        detector = HMMRegimeDetector()
        detector.load(load_path)
        return detector
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        return None


def predict_regime(features: np.ndarray) -> dict:
    """Convenience: load saved model and predict regime.

    Args:
        features: (n_features,) or (1, n_features) feature vector.

    Returns:
        dict with 'regime' and 'crisis_probability', or fallback on error.
    """
    detector = load_detector()
    if detector is None:
        return {"regime": "NORMAL", "crisis_probability": 0.0}
    return detector.predict(features)


# ════════════════════════════════════════════════════════════════
# MAIN — Standalone Training
# ════════════════════════════════════════════════════════════════


def main():
    """Standalone entrypoint: train from git history and save model."""
    print("=" * 60)
    print("HMM Regime Detection — Training")
    print("=" * 60)

    detector = HMMRegimeDetector(n_regimes=4)
    result = detector.fit_from_git()

    if result is None:
        print("\n❌ Training failed.")
        sys.exit(1)

    detector.save()

    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Regimes: {detector.n_regimes}")
    print(f"   Fitted:  {detector.is_fitted}")

    # Print transition matrix
    print("\nTransition matrix (regime labels):")
    for from_name, targets in detector.transition_matrix_labeled.items():
        print(f"  {from_name}: {targets}")

    # Predict on latest data.json
    latest_path = os.path.join(SFC_DIR, "data.json")
    if os.path.exists(latest_path):
        try:
            with open(latest_path) as f:
                latest = json.load(f)

            daily_return = float(latest.get("btc_24h", 0.0) or 0.0)
            dvol = float(latest.get("dvol", 0.0) or 0.0) / 100.0
            sfc = float(latest.get("sfc_effective", 50.0) or 50.0) / 100.0
            rsi = float(latest.get("rsi_14", 50.0) or 50.0) / 100.0
            fng = float(latest.get("fng", 50.0) or 50.0) / 100.0

            latest_features = np.array(
                [[daily_return, dvol, sfc, rsi, fng]], dtype=np.float64
            )
            prediction = detector.predict(latest_features)

            print(f"\nLatest prediction ({latest.get('ts', 'N/A')}):")
            print(f"  Regime:            {prediction['regime']} "
                  f"(label={prediction.get('regime_label')})")
            print(f"  Crisis probability: {prediction['crisis_probability']:.4f}")
            if "state_probs" in prediction:
                for i, p in enumerate(prediction["state_probs"]):
                    rname = REGIME_NAMES.get(i, f"S{i}")
                    print(f"  P({rname}): {p:.4f}")

        except Exception as exc:
            print(f"\n  ⚠ Could not evaluate on latest data: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
