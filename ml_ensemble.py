#!/usr/bin/env python3
"""
SFC ML Ensemble — Random Forest + Online Learning Feedback Loop
===============================================================
Strategi 3 & 4 dari PATH_TO_90_PERCENT.md:
- Neural Network / Random Forest ensemble untuk belajar optimal weights
- Online learning: retrain dari error setiap hari
- Backtesting: walk-forward validation

Data disimpan di data_collection.json untuk persistensi antar run.
"""

import json, os, sys, math, time
from datetime import datetime, timezone
import pickle

COLLECTION_FILE = os.path.join(os.path.dirname(__file__), "data_collection.json")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "ml_ensemble_model.pkl")
SCALER_FILE = os.path.join(os.path.dirname(__file__), "ml_ensemble_scaler.pkl")

# ──────────────────────────────────────────────────────
# DATA COLLECTION — store feature vectors + labels
# ──────────────────────────────────────────────────────

def load_collection():
    """Load historical training data."""
    if not os.path.exists(COLLECTION_FILE):
        return {"features": [], "labels": [], "predictions": [], "dates": []}
    try:
        with open(COLLECTION_FILE, "r") as f:
            return json.load(f)
    except:
        return {"features": [], "labels": [], "predictions": [], "dates": []}

def save_collection(data):
    """Save training data."""
    with open(COLLECTION_FILE, "w") as f:
        json.dump(data, f)

def add_observation(feature_vector, prediction=None, actual_label=None, date_str=None):
    """
    Add one observation to the collection.
    
    feature_vector: list of method scores [m1, m2, ..., m31, ...] in order
    prediction: what the model predicted (0 or 1)
    actual_label: what actually happened (0=no stress, 1=stress event)
    """
    data = load_collection()
    
    # Validate feature vector
    if not isinstance(feature_vector, (list, tuple)):
        print("[ML] Invalid feature vector, skipping collection", file=sys.stderr)
        return data
    
    data["features"].append([float(v) if v is not None else 0.5 for v in feature_vector])
    data["labels"].append(float(actual_label) if actual_label is not None else None)
    data["predictions"].append(float(prediction) if prediction is not None else None)
    data["dates"].append(date_str or datetime.now(timezone.utc).isoformat())
    
    # Keep at most 2000 observations
    if len(data["features"]) > 2000:
        data["features"] = data["features"][-2000:]
        data["labels"] = data["labels"][-2000:]
        data["predictions"] = data["predictions"][-2000:]
        data["dates"] = data["dates"][-2000:]
    
    save_collection(data)
    return data

def compute_actual_stress(dvol=None, sfc_pct=None, news_stress=None, btc_24h=None):
    """
    Determine if TODAY was a stress event (label = 1).
    Used for feedback loop — called on next day with known outcomes.
    
    Returns 0 or 1.
    """
    stress_signals = 0
    total_signals = 0
    
    if dvol is not None:
        total_signals += 1
        if dvol > 80: stress_signals += 1
    
    if sfc_pct is not None:
        total_signals += 1
        if sfc_pct > 50: stress_signals += 1
    
    if btc_24h is not None:
        total_signals += 1
        if btc_24h < -5: stress_signals += 1
    
    if news_stress is not None:
        total_signals += 1
        if news_stress > 30: stress_signals += 1
    
    if total_signals == 0:
        return 0
    
    # 2+ stress signals out of available = stress event
    return 1 if stress_signals >= max(2, total_signals // 2) else 0


# ──────────────────────────────────────────────────────
# RANDOM FOREST ENSEMBLE
# ──────────────────────────────────────────────────────

def _try_import_sklearn():
    """Try to import sklearn, return False if not available."""
    try:
        import sklearn
        return True
    except ImportError:
        return False

def _ensure_sklearn_deps():
    """Install sklearn if not available."""
    if _try_import_sklearn():
        return True
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "scikit-learn", "numpy"],
            capture_output=True, timeout=60
        )
        return _try_import_sklearn()
    except:
        return False

def train_model(force=False):
    """
    Train or retrain the Random Forest ensemble.
    
    Uses all labeled data (labels != None) from the collection.
    Walk-forward: uses 80% oldest for training, 20% newest for validation.
    """
    data = load_collection()
    
    # Need labeled data
    labeled = [(f, l) for f, l in zip(data["features"], data["labels"])
               if l is not None]
    
    if len(labeled) < 30:
        print(f"[ML] Not enough labeled data ({len(labeled)}/30 needed)", file=sys.stderr)
        return None
    
    if not _ensure_sklearn_deps():
        print("[ML] sklearn not available, can't train model", file=sys.stderr)
        return None
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    X = np.array([f for f, _ in labeled], dtype=float)
    y = np.array([l for _, l in labeled], dtype=int)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Train/val split (time-series safe: oldest 80% train)
    split = int(len(X) * 0.80)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else None
    
    # Train ensemble (from PATH_TO_90_PERCENT.md)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Validate
    train_acc = model.score(X_train_scaled, y_train)
    if X_val_scaled is not None and len(X_val) > 0:
        val_acc = model.score(X_val_scaled, y_val)
    else:
        val_acc = None
    
    print(f"[ML] Model trained: {len(X)} samples, "
          f"train_acc={train_acc:.3f}, "
          f"{'val_acc=' + f'{val_acc:.3f}' if val_acc else 'no val set'}",
          file=sys.stderr)
    
    # Feature importance
    importance = model.feature_importances_
    top_features = sorted(enumerate(importance), key=lambda x: -x[1])[:5]
    print(f"[ML] Top features: {[(f'f{i}', round(v, 3)) for i, v in top_features]}",
          file=sys.stderr)
    
    # Save model + scaler
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[ML] Model saved to {MODEL_FILE}", file=sys.stderr)
    except Exception as e:
        print(f"[ML] Could not save model: {e}", file=sys.stderr)
    
    return model, scaler, train_acc, val_acc


def load_model():
    """Load a pre-trained model + scaler."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None


def predict_with_ml(feature_vector, method_count):
    """
    Predict stress probability using ML ensemble.
    
    Falls back to weighted average if model not available.
    Returns (ml_prediction, ml_confidence, vote_message).
    """
    model, scaler = load_model()
    
    if model is None or scaler is None:
        # Weighted average fallback (from PATH_TO_90_PERCENT.md, Strategi 4)
        # Traditional weights as baseline
        return _weighted_average_fallback(feature_vector, method_count)
    
    import numpy as np
    
    # Prepare features
    X = np.array([feature_vector], dtype=float)
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)
    
    try:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # proba[1] = probability of stress (class 1)
        stress_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = abs(stress_prob - 0.5) * 2  # 0-1 scale
        
        if pred == 1:
            return stress_prob, confidence, f"ML={stress_prob:.3f} (CONFIDENCE)"
        else:
            return stress_prob, confidence, f"ML={stress_prob:.3f} (CALM)"
    except Exception as e:
        print(f"[ML] Prediction error: {e}", file=sys.stderr)
        return _weighted_average_fallback(feature_vector, method_count)


def _weighted_average_fallback(feature_vector, method_count):
    """Weighted average baseline (from PATH_TO_90_PERCENT.md)."""
    # Weights: higher for established methods, lower for new ones
    raw_weights = []
    # M1-M6 (original): higher weights
    for i in range(min(6, method_count)):
        raw_weights.append(0.12)
    # M7-M19: medium weights
    for i in range(6, min(19, method_count)):
        raw_weights.append(0.03)
    # M20-M31 (new): lower weights initially
    for i in range(19, method_count):
        raw_weights.append(0.02)
    
    # Trim or pad to match feature_vector length
    while len(raw_weights) < method_count:
        raw_weights.append(0.02)
    raw_weights = raw_weights[:method_count]
    
    total_w = sum(raw_weights)
    if total_w == 0:
        return 0.5, 0.0, "Equal weights (fallback)"
    
    weights = [w / total_w for w in raw_weights]
    
    # Apply weights
    weighted = sum(v * w for v, w in zip(feature_vector, weights) if v is not None)
    
    # Count active methods
    active = sum(1 for v in feature_vector if v is not None)
    confidence = min(active / method_count, 1.0) * 0.8
    
    return weighted, confidence, f"WeightedAvg={weighted:.3f}"


def retrain_on_errors():
    """
    Strategi 3: Online Learning — retrain on labeled errors.
    
    Finds all observations where prediction != actual,
    then retrains the model incorporating that knowledge.
    Called once per day after actual stress is known.
    """
    data = load_collection()
    
    # Find mismatches
    mismatches = sum(1 for p, l in zip(data["predictions"], data["labels"])
                     if p is not None and l is not None and 
                     (round(p) != l))
    
    total_labeled = sum(1 for l in data["labels"] if l is not None)
    
    if mismatches == 0 or total_labeled < 30:
        print(f"[ML] No errors to learn from ({mismatches} mismatches, {total_labeled} labeled)",
              file=sys.stderr)
        return None
    
    accuracy = 1 - (mismatches / total_labeled)
    print(f"[ML] Current accuracy: {accuracy:.3f} ({mismatches}/{total_labeled} errors)",
          file=sys.stderr)
    
    # Retrain
    result = train_model(force=True)
    
    if result:
        print(f"[ML] Retrained on {total_labeled} labeled samples "
              f"(incorporating {mismatches} corrections)",
              file=sys.stderr)
    
    return result


def evaluate_accuracy():
    """
    Evaluate current model accuracy on labeled data.
    Returns dict with various metrics.
    """
    data = load_collection()
    
    labeled = [(p, l) for p, l in zip(data["predictions"], data["labels"])
               if p is not None and l is not None]
    
    if not labeled:
        return {"accuracy": None, "total": 0, "correct": 0, "message": "No labeled data yet"}
    
    correct = sum(1 for p, l in labeled if round(p) == l)
    total = len(labeled)
    accuracy = correct / total
    
    # Count by class
    stress_events = sum(1 for _, l in labeled if l == 1)
    calm_events = total - stress_events
    stress_correct = sum(1 for p, l in labeled if l == 1 and round(p) == l)
    calm_correct = sum(1 for p, l in labeled if l == 0 and round(p) == l)
    
    return {
        "accuracy": round(accuracy, 4),
        "total": total,
        "correct": correct,
        "by_class": {
            "stress": {"total": stress_events, "correct": stress_correct,
                       "acc": round(stress_correct / stress_events, 3) if stress_events else None},
            "calm": {"total": calm_events, "correct": calm_correct,
                     "acc": round(calm_correct / calm_events, 3) if calm_events else None}
        },
        "message": f"Accuracy: {accuracy:.1%} ({correct}/{total}) "
                   f"— STRATEGI 3: Online Learning Active" if total >= 10
                   else f"Too few samples ({total})"
    }


# ──────────────────────────────────────────────────────
# MAIN — when run standalone
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if "--train" in sys.argv:
        print("Training ML ensemble...", file=sys.stderr)
        result = train_model(force=True)
        if result:
            print(f"Done. Model ready for inference.", file=sys.stderr)
    
    elif "--evaluate" in sys.argv:
        metrics = evaluate_accuracy()
        print(json.dumps(metrics, indent=2))
    
    elif "--retrain" in sys.argv:
        print("Retraining on errors (online learning)...", file=sys.stderr)
        result = retrain_on_errors()
        if result:
            print(f"Retrained successfully.", file=sys.stderr)
        else:
            print(f"No retrain needed or not enough data.", file=sys.stderr)
    
    else:
        # Test mode: use synthetic feature vector
        print("ML Ensemble — Test Mode", file=sys.stderr)
        print("Usage: python3 ml_ensemble.py [--train|--evaluate|--retrain]", file=sys.stderr)
        
        # Demo with random-like feature vector (31 methods)
        test_features = [0.5] * 31
        for i in range(31):
            test_features[i] = abs(math.sin(i * 1.7)) * 0.8 + 0.1
        
        ml_score, confidence, msg = predict_with_ml(test_features, 31)
        print(f"Test predict: score={ml_score:.4f}, confidence={confidence:.4f}, {msg}")
