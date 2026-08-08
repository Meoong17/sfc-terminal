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
from datetime import datetime, timezone, timedelta
import pickle

COLLECTION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_collection.json")
MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_ensemble_model.pkl")
SCALER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_ensemble_scaler.pkl")

# ──────────────────────────────────────────────────────
# DATA COLLECTION — store feature vectors + labels
# ──────────────────────────────────────────────────────

def load_collection():
    """Load historical training data."""
    if not os.path.exists(COLLECTION_FILE):
        return {"features": [], "labels": [], "predictions": [], "dates": [], "price_log": []}
    try:
        with open(COLLECTION_FILE, "r") as f:
            data = json.load(f)
            data.setdefault("price_log", [])
            return data
    except:
        return {"features": [], "labels": [], "predictions": [], "dates": [], "price_log": []}

def save_collection(data):
    """Save training data."""
    with open(COLLECTION_FILE, "w") as f:
        json.dump(data, f)

# ──────────────────────────────────────────────────────
# DAILY REGIME COLLECTION — 1 row per calendar day
# ──────────────────────────────────────────────────────
# The 5-min data_collection.json (2000-row cap) only retains ~10 days, far too
# short for regime detection or walk-forward validation of the adv regime
# detector. This separate series stores ONE summary row per calendar day, so
# regime history accumulates for years in a small file. Same normalized 0-1
# feature space (cols 0-4 = m1..m5). Not truncated (regime detection needs
# months/years of history).
DAILY_COLLECTION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "data_collection_daily.json")


def load_daily_collection():
    if not os.path.exists(DAILY_COLLECTION_FILE):
        return {"features": [], "labels": [], "predictions": [], "dates": []}
    try:
        with open(DAILY_COLLECTION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"features": [], "labels": [], "predictions": [], "dates": []}


def save_daily_collection(data):
    with open(DAILY_COLLECTION_FILE, "w") as f:
        json.dump(data, f)


def add_daily_observation(feature_vector, date_str=None):
    """Record ONE daily summary observation (deduped by calendar date).

    Called once per day (or repeatedly — it skips if today is already recorded).
    Stores cols m1..m5 in the same normalized 0-1 space as add_observation, so
    the regime detector can be fit on this series and predict the current
    (normalized) feature vector consistently.
    """
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = load_daily_collection()
    if data["dates"] and data["dates"][-1] == date_str:
        return data  # today already recorded
    vec = [float(v) if v is not None else 0.5 for v in feature_vector][:5]
    data["features"].append(vec)
    data["labels"].append(None)
    data["predictions"].append(None)
    data["dates"].append(date_str)
    save_daily_collection(data)
    return data

def record_price_snapshot(btc_price, date_str=None):
    """
    Append a lightweight {ts, btc} snapshot to the collection's own
    price_log, independent of git history. resolve_pending_labels() reads
    from this instead of needing to re-scan git log on every call —
    cheaper, and works even if this script is ever run somewhere without
    git history available at all (e.g. a fresh clone).

    Keeps up to ~30 days of 5-minute-interval snapshots (8640 points) —
    comfortably more than LABEL_LOOKAHEAD_MINUTES (360 min) requires, with
    headroom if the lookahead window is widened later.
    """
    if btc_price is None:
        return
    data = load_collection()
    data["price_log"].append({
        "ts": date_str or datetime.now(timezone.utc).isoformat(),
        "btc": float(btc_price),
    })
    if len(data["price_log"]) > 8640:
        data["price_log"] = data["price_log"][-8640:]
    save_collection(data)

def add_observation(feature_vector, prediction=None, date_str=None):
    """
    Add one observation to the collection. Its label starts as None
    ("pending") and is filled in later by resolve_pending_labels(), once
    enough time has passed to know what BTC price actually did — never
    at the moment the observation is recorded.

    feature_vector: list of method scores [m1, m2, ..., m31, ...] in order
    prediction: what the model predicted (0 or 1), for tracking accuracy later

    NOTE: this function intentionally no longer accepts an actual_label
    parameter. The previous version let the caller pass in a label
    computed from sfc_pct (an output built from this same feature vector),
    which meant the model was effectively learning to reproduce its own
    formula rather than predict real market outcomes. If you need to
    backfill a label, use resolve_pending_labels() with real price history
    instead of writing to data["labels"] directly.
    """
    data = load_collection()
    
    # Validate feature vector
    if not isinstance(feature_vector, (list, tuple)):
        print("[ML] Invalid feature vector, skipping collection", file=sys.stderr)
        return data
    
    data["features"].append([float(v) if v is not None else 0.5 for v in feature_vector])
    data["labels"].append(None)  # always starts pending — see resolve_pending_labels()
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
    DEPRECATED — kept only for backward compatibility with any external
    caller that may still import this name directly.

    This used to be called as the label at the SAME time the feature
    vector was recorded (see add_observation below), using sfc_pct as one
    of its inputs. Because sfc_pct is itself a function of M1-M6, which
    are also entries in the feature vector being labeled, the model could
    reach near-perfect "accuracy" simply by reproducing the formula that
    built its own label — confirmed empirically: a naive threshold rule
    on sfc_pct alone (no training at all) already matched this label
    function ~77% of the time in simulation. See resolve_pending_labels()
    for the actual fix: labels are now assigned later, from realized BTC
    price movement, not from the model's own contemporaneous output.

    Do not wire this back into the labeling path.
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

    return 1 if stress_signals >= max(2, total_signals // 2) else 0


# ──────────────────────────────────────────────────────
# PRICE-OUTCOME LABELING — ground truth independent of model output
# ──────────────────────────────────────────────────────
#
# Replaces compute_actual_stress() as the source of training labels.
# A "stress event" is now defined purely from what BTC price actually did
# in the following window — never from sfc_pct, dvol, or any other value
# the ensemble itself produced. This breaks the circularity where the
# model was effectively being trained to reproduce its own formula.

LABEL_LOOKAHEAD_MINUTES = 360      # how far ahead to check price outcome
LABEL_LOOKAHEAD_TOLERANCE_MINUTES = 60
LABEL_STRESS_DROP_PCT = -3.0       # BTC drop >= 3% in the window = stress event
LABEL_CALM_RISE_PCT = 1.5          # BTC didn't fall this much = calm, even if flat/up


def _parse_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def resolve_pending_labels(price_history=None):
    """
    Walk through observations whose label is still None ("pending") and
    assign a label from realized BTC price movement, if enough time has
    now passed.

    Args:
        price_history: optional list of {"ts": iso_str, "btc": float}
            snapshots. If omitted, reads from this collection's own
            price_log (populated by record_price_snapshot(), which should
            be called once per collect.py cycle alongside add_observation).
            Does not need to be pre-sorted — sorted internally by ts.

    Returns:
        number of labels resolved in this call.
    """
    data = load_collection()

    if price_history is None:
        price_history = data.get("price_log", [])

    # Sort defensively — caller-supplied history isn't guaranteed ordered,
    # and out-of-order entries would silently break the nearest-match scan.
    price_history = sorted(
        (s for s in price_history if _parse_ts(s.get("ts", "")) is not None),
        key=lambda s: _parse_ts(s["ts"])
    )

    resolved = 0

    for i, label in enumerate(data["labels"]):
        if label is not None:
            continue  # already resolved
        date_str = data["dates"][i] if i < len(data["dates"]) else None
        obs_time = _parse_ts(date_str) if date_str else None
        if obs_time is None:
            continue

        target_time = obs_time + timedelta(minutes=LABEL_LOOKAHEAD_MINUTES)

        # Find the price at observation time and at the target lookahead time
        price_at_obs = None
        price_at_target = None
        best_obs_diff = None
        best_target_diff = None
        for snap in price_history:
            snap_ts = _parse_ts(snap.get("ts", ""))
            snap_btc = snap.get("btc")
            if snap_ts is None or snap_btc is None:
                continue
            obs_diff = abs((snap_ts - obs_time).total_seconds())
            if obs_diff <= 120 and (best_obs_diff is None or obs_diff < best_obs_diff):
                price_at_obs = snap_btc
                best_obs_diff = obs_diff
            target_diff = abs((snap_ts - target_time).total_seconds())
            if (target_diff <= LABEL_LOOKAHEAD_TOLERANCE_MINUTES * 60
                    and (best_target_diff is None or target_diff < best_target_diff)):
                price_at_target = snap_btc
                best_target_diff = target_diff

        if price_at_obs is None or price_at_target is None:
            continue  # not enough history yet to resolve this one — try again later

        pct_change = (price_at_target - price_at_obs) / price_at_obs * 100.0

        if pct_change <= LABEL_STRESS_DROP_PCT:
            data["labels"][i] = 1.0  # confirmed stress: price actually dropped
            resolved += 1
        elif pct_change >= -LABEL_CALM_RISE_PCT:
            # Price held up (flat or rose) — confirmed calm. Using a band
            # around zero (not just ">0") avoids mislabeling small noise
            # as a confident "calm" call.
            data["labels"][i] = 0.0
            resolved += 1
        # else: ambiguous mild decline between the two thresholds — leave
        # pending in case a clearer signal emerges from a later, larger
        # lookahead pass; do not force a label onto an ambiguous outcome.

    if resolved > 0:
        save_collection(data)
        print(f"[ML] Resolved {resolved} pending labels from realized BTC price movement",
              file=sys.stderr)

    return resolved


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
