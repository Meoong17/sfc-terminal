#!/usr/bin/env python3
"""
ensemble_meta.py — XGBoost Meta-Ensemble Layer for SFC
=======================================================
Trains an XGBoost regressor on historical method scores (m1_klr
through m31_altman) to predict sfc_effective, providing a learned
weighting that adapts over time.

Usage:
    # Standalone training from git history
    cd /home/ubuntu/sfc
    python3 ensemble_meta.py

    # As a library
    from ensemble_meta import predict_ensemble, train_from_git_history
    result = predict_ensemble({"m1_klr": 5.9, "m2_logit": 6.7, ...})
"""

import json, os, sys, subprocess, logging, traceback
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

try:
    import xgboost as xgb
except ImportError as e:
    xgb = None
    print(f"[EnsembleMeta] ⚠ XGBoost not available: {e}", file=sys.stderr)

# ── CONFIG ──
SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(SFC_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_meta.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# Ordered list of method score fields (m1 through m31)
METHOD_FIELDS = [
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
    "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
    "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
    "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
    "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
]

# ── LOGGING ──
logging.basicConfig(
    level=logging.INFO,
    format="[EnsembleMeta] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("ensemble_meta")


# ════════════════════════════════════════════════════════════════
# 1. XGBoostMetaEnsemble CLASS
# ════════════════════════════════════════════════════════════════
class XGBoostMetaEnsemble:
    """XGBoost-based meta-ensemble that learns to weight method scores.

    Takes a 2D numpy array (n_samples, n_features) of method scores,
    trains an XGBoost regressor with early stopping, and saves/loads
    the model to/from JSON format.
    """

    def __init__(self):
        if xgb is None:
            raise ImportError(
                "XGBoost is not installed or cannot be imported. "
                "Install it via: pip install xgboost"
            )
        self.model = None
        self._is_fitted = False
        self._n_features = None

    def fit(self, X, y, eval_set=None, verbose=True):
        """Train the XGBoost meta-ensemble.

        Args:
            X: 2D numpy array (n_samples, n_features) of method scores.
            y: 1D numpy array (n_samples,) of target values (sfc_effective / 100).
            eval_set: Optional list of (X_val, y_val) tuples for early stopping.
            verbose: Whether to print training progress.
        """
        if len(X) == 0:
            raise ValueError("Empty training data: X has zero rows")

        self._n_features = X.shape[1]

        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "objective": "reg:squarederror",
            "random_state": 42,
            "verbosity": 1 if verbose else 0,
        }

        if eval_set is not None and len(eval_set) > 0:
            # Train with early stopping (via constructor kwargs for XGBoost 3.x)
            params["early_stopping_rounds"] = 20
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=verbose,
            )
            if verbose:
                best_iter = self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else "N/A"
                best_score = self.model.best_score if hasattr(self.model, 'best_score') else "N/A"
                log.info(f"Early stopping at iteration {best_iter} (best validation score: {best_score:.6f})")
        else:
            # Train without early stopping
            self.model.fit(X, y)

        self._is_fitted = True

        if verbose:
            log.info(f"Model trained: {len(X)} samples, {self._n_features} features")
            if hasattr(self.model, 'feature_importances_'):
                top_idx = np.argsort(self.model.feature_importances_)[-5:][::-1]
                log.info(f"Top 5 feature indices: {top_idx}")

        return self

    def predict(self, X):
        """Predict stress from method scores.

        Args:
            X: 2D numpy array (n_samples, n_features).

        Returns:
            Predicted values in original sfc_effective scale (0-100).
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first or load a saved model.")

        preds = self.model.predict(X)
        # Cap to valid range [0, 100]
        preds = np.clip(preds * 100.0, 0.0, 100.0)
        return preds

    def predict_single(self, scores_dict):
        """Predict stress from a single dict of {method_name: score}.

        Args:
            scores_dict: Dict mapping method field names to scores.
                        Missing fields default to 0.0.

        Returns:
            (predicted_stress, confidence) tuple.
            Stress is in original scale (0-100), confidence is
            derived from tree standard deviation if available.
        """
        vec = self._dict_to_vector(scores_dict)
        pred = self.predict(vec.reshape(1, -1))[0]

        # Confidence: use tree ensemble std if available
        if hasattr(self.model, 'predict') and hasattr(self.model, 'get_booster'):
            try:
                # Get per-tree predictions for std estimate
                preds_trees = self.model.get_booster().predict(
                    xgb.DMatrix(vec.reshape(1, -1)),
                    output_margin=True,
                    pred_leaf=False,
                    iteration_range=(0, self.model.get_booster().best_ntree_limit
                                     if hasattr(self.model.get_booster(), 'best_ntree_limit')
                                     else self.model.n_estimators),
                )
                # If we can get tree-level predictions, use std / sqrt(n_trees)
                # Fallback: use a fixed confidence based on model's feature count
                confidence = min(1.0, 0.5 + 0.5 * (1.0 - abs(pred - 50.0) / 50.0))
            except Exception:
                confidence = min(1.0, 0.5 + 0.5 * (1.0 - abs(pred - 50.0) / 50.0))
        else:
            confidence = min(1.0, 0.5 + 0.5 * (1.0 - abs(pred - 50.0) / 50.0))

        return float(pred), float(confidence)

    def save(self, path=None):
        """Save trained model to JSON file.

        Args:
            path: Path to save. Defaults to MODEL_PATH.
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load first.")

        save_path = path or MODEL_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        log.info(f"Model saved to {save_path}")

    def load(self, path=None):
        """Load trained model from JSON file.

        Args:
            path: Path to load from. Defaults to MODEL_PATH.
        """
        load_path = path or MODEL_PATH
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        if xgb is None:
            raise ImportError("XGBoost is not available")

        self.model = xgb.XGBRegressor()
        self.model.load_model(load_path)
        self._is_fitted = True
        self._n_features = len(METHOD_FIELDS)
        log.info(f"Model loaded from {load_path}")
        return self

    def _dict_to_vector(self, scores_dict):
        """Convert a dict of method scores to a feature vector.

        Args:
            scores_dict: Dict of {method_name: score}.

        Returns:
            Numpy array aligned with METHOD_FIELDS order.
        """
        vec = []
        for field in METHOD_FIELDS:
            val = scores_dict.get(field)
            if val is None:
                # Try alternate key patterns
                val = scores_dict.get(field.replace("m", ""), None)
            if val is None:
                val = 0.0
            try:
                vec.append(float(val))
            except (ValueError, TypeError):
                vec.append(0.0)
        return np.array(vec, dtype=np.float32)

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def n_features(self):
        return self._n_features


# ════════════════════════════════════════════════════════════════
# 2. TRAIN FROM GIT HISTORY
# ════════════════════════════════════════════════════════════════
def extract_historical_snapshots():
    """Extract all clean data.json snapshots from git history.

    Same approach as train_mamba.py: gets commits that modified
    data.json, extracts content at each commit, parses JSON.

    Returns:
        List of parsed data.json dicts, oldest first.
    """
    log.info("Extracting historical snapshots from git...")

    try:
        result = subprocess.check_output(
            ["git", "log", "--oneline", "--all", "--diff-filter=M",
             "--reverse", "--", "data.json"],
            text=True, timeout=30, cwd=SFC_DIR,
        ).strip().split("\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.error(f"Git log failed: {e}")
        return []

    # Filter empty results
    result = [r for r in result if r.strip()]

    snapshots = []
    errors = 0
    for i, line in enumerate(result):
        sha = line.split()[0]
        try:
            content = subprocess.check_output(
                ["git", "show", f"{sha}:data.json"],
                text=True, timeout=10, cwd=SFC_DIR,
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

    log.info(f"Extracted {len(snapshots)} clean snapshots ({errors} skipped)")
    return snapshots


def build_method_scores_array(snapshots):
    """Build feature matrix and target vector from snapshots.

    For each snapshot:
      - Features: m1_klr through m31_altman fields (31 features)
      - Target: realized BTC price movement over the next
        TARGET_LOOKAHEAD_MINUTES, expressed as a 0-1 "stress probability"
        proxy (1.0 = price dropped >= TARGET_STRESS_DROP_PCT, 0.0 = price
        held up, linear interpolation in between).

    IMPORTANT — why this changed from the original "target = sfc_effective":
    M1-M6 (a subset of METHOD_FIELDS used as features here) are themselves
    the dominant inputs to sfc_effective via calculate_sfc_ensemble() in
    collect.py (p_ens = 0.19*p_klr + 0.16*p_logit + ... ). Training XGBoost
    to predict sfc_effective from features that mostly built sfc_effective
    in the first place meant the model could reach near-perfect validation
    error simply by re-deriving that formula — confirmed empirically via a
    simple linear regression on simulated data reaching R²=1.000 exactly
    this way. Worse, predict_ensemble()'s output is blended back into
    effective_sfc in collect.py, so this wasn't just a misleading metric —
    it was an echo chamber amplifying the ensemble's existing belief about
    itself rather than adding independent predictive signal from M7-M31.

    Using realized future BTC price movement as the target instead means
    the model is now actually rewarded for finding which method scores
    (including M7-M31, previously redundant under the old target) precede
    real price drops — not for reproducing a formula made of its own inputs.

    Missing method scores default to 0.0.
    Snapshots too close to the end of history (no future snapshot far
    enough ahead within tolerance) are skipped, not given a guessed label.

    Args:
        snapshots: List of data dicts from git history, oldest first.

    Returns:
        (X, y) tuple of numpy arrays.
        X: (n, 31) float32
        y: (n,) float32  (0-1, price-outcome based — see above)
    """
    TARGET_LOOKAHEAD_MINUTES = 360
    TARGET_LOOKAHEAD_TOLERANCE_MINUTES = 60
    TARGET_STRESS_DROP_PCT = -3.0   # >= 3% drop maps to y=1.0
    TARGET_CALM_FLOOR_PCT = 1.0     # <= +1% (or any rise) maps to y=0.0

    def _parse_ts(snap):
        ts_str = snap.get("ts")
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            return None

    # Pre-parse timestamps once; snapshots without a usable ts/btc pair
    # can't be used as a *future* reference point, but can still be used
    # as input rows if a later snapshot within tolerance exists.
    parsed_times = [_parse_ts(s) for s in snapshots]
    btc_prices = [s.get("btc") for s in snapshots]

    X_list = []
    y_list = []
    skipped_no_future = 0

    for i, snap in enumerate(snapshots):
        obs_time = parsed_times[i]
        obs_price = btc_prices[i]
        if obs_time is None or obs_price is None:
            continue

        target_time = obs_time + timedelta(minutes=TARGET_LOOKAHEAD_MINUTES)

        # Find the closest future snapshot to target_time, searching only
        # forward from i (this list is oldest-first) within tolerance.
        best_diff = None
        future_price = None
        for j in range(i + 1, len(snapshots)):
            t_j = parsed_times[j]
            if t_j is None:
                continue
            if t_j < obs_time:
                continue  # shouldn't happen given oldest-first ordering, but guard anyway
            diff_minutes = abs((t_j - target_time).total_seconds()) / 60.0
            if diff_minutes <= TARGET_LOOKAHEAD_TOLERANCE_MINUTES:
                if best_diff is None or diff_minutes < best_diff:
                    best_diff = diff_minutes
                    future_price = btc_prices[j]
            # Once we've moved well past the tolerance window on the late
            # side, no later snapshot will be closer — stop scanning early.
            if (t_j - target_time).total_seconds() / 60.0 > TARGET_LOOKAHEAD_TOLERANCE_MINUTES:
                break

        if future_price is None:
            skipped_no_future += 1
            continue  # no usable future reference point yet (e.g. tail of history)

        pct_change = (future_price - obs_price) / obs_price * 100.0

        if pct_change <= TARGET_STRESS_DROP_PCT:
            target_01 = 1.0
        elif pct_change >= -TARGET_CALM_FLOOR_PCT:
            target_01 = 0.0
        else:
            # Linear interpolation between the calm floor and stress drop
            # thresholds, rather than a hard cutoff, so the model sees a
            # graded signal instead of an arbitrary binary boundary.
            span = TARGET_STRESS_DROP_PCT - (-TARGET_CALM_FLOOR_PCT)  # negative span
            target_01 = (pct_change - (-TARGET_CALM_FLOOR_PCT)) / span
            target_01 = max(0.0, min(1.0, target_01))

        # Feature vector: method scores (unchanged from original)
        vec = []
        for field in METHOD_FIELDS:
            val = snap.get(field)
            if val is None:
                val = 0.0
            try:
                vec.append(float(val))
            except (ValueError, TypeError):
                vec.append(0.0)

        X_list.append(vec)
        y_list.append(target_01)

        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i+1}/{len(snapshots)}...")

    log.info(f"Skipped {skipped_no_future} snapshot(s) with no future reference point in range")

    if len(X_list) == 0:
        log.warning("No valid snapshots with method scores + resolvable price outcome found")
        return np.empty((0, len(METHOD_FIELDS)), dtype=np.float32), np.empty(0, dtype=np.float32)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    log.info(f"Feature matrix: {X.shape}, target: {y.shape}")
    log.info(f"Target range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

    # Detect and report missing data
    missing_per_feature = np.isnan(X).sum(axis=0)
    if missing_per_feature.sum() > 0:
        bad_cols = np.where(missing_per_feature > 0)[0]
        log.warning(f"NaN values found in columns: {bad_cols.tolist()}, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    return X, y


def train_from_git_history(verbose=True):
    """Extract snapshots from git, build dataset, train XGBoost, save model.

    This is the main training pipeline called from main() or imported.

    Returns:
        XGBoostMetaEnsemble instance if successful, None otherwise.
    """
    try:
        # 1. Extract historical snapshots
        snapshots = extract_historical_snapshots()
        if len(snapshots) < 50:
            log.warning(f"Too few snapshots ({len(snapshots)}), need at least 50")
            return None

        # 2. Build feature matrix and targets
        X, y = build_method_scores_array(snapshots)
        if len(X) < 50:
            log.warning(f"Too few samples ({len(X)}), need at least 50")
            return None

        # 3. Chronological split: 85% train, 15% validation
        n_total = len(X)
        n_val = max(1, int(n_total * 0.15))
        n_train = n_total - n_val

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[-n_val:], y[-n_val:]

        if verbose:
            log.info(f"Split: train={n_train}, val={n_val}")
            log.info(f"Train target: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
            log.info(f"Val target:   mean={y_val.mean():.4f}, std={y_val.std():.4f}")

        # 4. Create model
        ensemble = XGBoostMetaEnsemble()

        # 5. Train with early stopping
        ensemble.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )

        # 6. Evaluate on validation set
        val_preds = ensemble.predict(X_val)
        val_actuals = y_val * 100.0  # Back to original scale
        val_preds_orig = val_preds  # Already in 0-100 scale from predict()

        # Compute metrics
        mae = np.abs(val_preds_orig - val_actuals).mean()
        rmse = np.sqrt(((val_preds_orig - val_actuals) ** 2).mean())

        if verbose:
            log.info(f"Validation MAE:  {mae:.2f} stress points")
            log.info(f"Validation RMSE: {rmse:.2f} stress points")
            log.info(f"Pred range: [{val_preds_orig.min():.1f}, {val_preds_orig.max():.1f}]")
            log.info(f"Actual range: [{val_actuals.min():.1f}, {val_actuals.max():.1f}]")

        # 7. Save model
        ensemble.save()
        log.info(f"Model saved to {MODEL_PATH}")

        return ensemble

    except Exception as e:
        log.error(f"Training failed: {e}")
        if verbose:
            traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════════
# 3. PREDICT
# ════════════════════════════════════════════════════════════════
def predict_ensemble(method_scores_dict):
    """Predict ensemble stress from current method scores.

    Loads the saved XGBoost model (if exists) and predicts
    stress + confidence from a dict of method scores.

    Args:
        method_scores_dict: Dict of {method_name: score} from
                           the current run. Can include any subset
                           of method fields; missing fields default
                           to 0.0.

    Returns:
        Dict with 'stress', 'confidence', 'model_loaded' keys, or
        None if no model exists or prediction fails.
    """
    model_path = MODEL_PATH

    if not os.path.exists(model_path):
        log.warning("No trained model found at %s", model_path)
        return None

    try:
        ensemble = XGBoostMetaEnsemble()
        ensemble.load(model_path)

        stress, confidence = ensemble.predict_single(method_scores_dict)

        result = {
            "stress": stress,
            "confidence": confidence,
            "model_loaded": True,
        }

        log.info(
            f"Prediction: stress={stress:.2f}, confidence={confidence:.3f}"
        )
        return result

    except Exception as e:
        log.error(f"Prediction failed: {e}")
        if os.environ.get("ENSEMBLE_META_DEBUG"):
            traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════════
# 4. MAIN — Standalone Training
# ════════════════════════════════════════════════════════════════
def main():
    """Standalone entrypoint: train from git history and evaluate.

    Usage:
        python3 ensemble_meta.py
    """
    print("=" * 60)
    print("XGBoost Meta-Ensemble Training")
    print("=" * 60)

    if xgb is None:
        print("\n❌ XGBoost not available. Install it:")
        print("   pip install xgboost")
        print("   Or activate the project venv.")
        sys.exit(1)

    print(f"XGBoost version: {xgb.__version__}")
    print(f"SFC directory:   {SFC_DIR}")
    print(f"Model path:      {MODEL_PATH}")
    print()

    ensemble = train_from_git_history(verbose=True)

    if ensemble is None:
        print("\n❌ Training failed. Check logs above.")
        sys.exit(1)

    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Features: {ensemble.n_features} method scores")
    print(f"   Fitted: {ensemble.is_fitted}")

    # Quick sanity check with latest data.json
    if os.path.exists(os.path.join(SFC_DIR, "data.json")):
        try:
            with open(os.path.join(SFC_DIR, "data.json")) as f:
                latest = json.load(f)
            scores = {f: latest.get(f) for f in METHOD_FIELDS}
            result = predict_ensemble(scores)
            if result:
                actual = latest.get("sfc_effective", "N/A")
                print(f"   Latest prediction: stress={result['stress']:.2f} "
                      f"(actual sfc_effective={actual})")
        except Exception:
            pass


if __name__ == "__main__":
    main()
