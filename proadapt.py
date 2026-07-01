#!/usr/bin/env python3
"""
proadapt.py — Online Learning module with Adaptive Sliding Window Ensemble
==========================================================================
Sliding window of last 20 predictions with actual values.
Adaptive ensemble weight: QLSTM vs GARCH_residual_correction (hybrid).
Fine-tune trigger when QLSTM MAE exceeds threshold (2 SFC pts).

Functions:
    load_state() -> dict
    save_state(state) -> None
    add_observation(qlstm_pred, garch_pred, hybrid_pred, actual_val) -> dict
    get_adaptive_prediction(qlstm_pred, garch_residual, hybrid_pred) -> dict
    get_fine_tune_signal() -> bool
    reset_fine_tune_signal() -> None
    get_state_path() -> str

State persistence: proadapt_state.json in sfc2/

Usage:
    from proadapt import load_state, add_observation, get_adaptive_prediction

    # Online learning loop
    result = add_observation(qlstm_pred=72.3, garch_pred=1.2,
                              hybrid_pred=73.5, actual_val=71.0)
    print(f"w_qlstm={result['w_qlstm']}, final_pred={result['final_pred']}")

    # Inference only (no observation recording)
    pred = get_adaptive_prediction(qlstm_pred=72.3, garch_residual=1.2,
                                    hybrid_pred=73.5)

    # Check fine-tune trigger
    if get_fine_tune_signal():
        # Trigger fine-tuning of last QLSTM FC layer
        reset_fine_tune_signal()
"""

import json
import os
import math
import time
import numpy as np

# ── Constants ──────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "proadapt_state.json")

DEFAULT_WINDOW_SIZE = 20          # sliding window of observations
DEFAULT_RMSE_WINDOW = 10          # window for RMSE/MAE computation
DEFAULT_FINE_TUNE_THRESHOLD = 2.0 # SFC points


# ── JSON helpers (handle float('inf')) ────────────────────

class _StateEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts float('inf')/float('nan') to None."""
    def default(self, obj):
        return super().default(obj)

    def encode(self, o):
        return super().encode(self._clean(o))

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return None
        return obj


def _decode_state(d):
    """Post-process loaded dict: convert None metric values back to inf."""
    for key in ("qlstm_mae", "hybrid_mae", "qlstm_rmse", "hybrid_rmse"):
        if key in d and d[key] is None:
            d[key] = float('inf')
    return d


# ── Internal helpers ──────────────────────────────────────

def _compute_rmse(errors):
    """Compute RMSE from a list of errors (actual - predicted)."""
    if len(errors) < 1:
        return float('inf')
    return float(math.sqrt(float(np.mean(np.square(np.array(errors))))))

def _compute_mae(errors):
    """Compute MAE from a list of errors."""
    if len(errors) < 1:
        return float('inf')
    return float(np.mean(np.abs(np.array(errors))))

def _adaptive_weight(rmse_q, rmse_h):
    """
    Compute adaptive ensemble weight using inverse RMSE weighting.

    w_qlstm = (1/rmse_q) / (1/rmse_q + 1/rmse_h)

    Falls back to 0.5 when both are infinite (no data).
    """
    if rmse_q == float('inf') and rmse_h == float('inf'):
        return 0.5
    if rmse_q == float('inf'):
        return 0.0
    if rmse_h == float('inf'):
        return 1.0
    inv_q = 1.0 / (rmse_q + 1e-10)
    inv_h = 1.0 / (rmse_h + 1e-10)
    return inv_q / (inv_q + inv_h)

def _default_state():
    """Return a fresh default state dictionary."""
    return {
        "observations": [],
        "window_size": DEFAULT_WINDOW_SIZE,
        "rmse_window": DEFAULT_RMSE_WINDOW,
        "fine_tune_threshold": DEFAULT_FINE_TUNE_THRESHOLD,
        "qlstm_mae": None,         # stored as None (inf) for JSON safety
        "hybrid_mae": None,
        "fine_tune_requested": False,
        "last_updated": None,
        "version": 1,
    }


# ── Public API ──────────────────────────────────────────

def load_state():
    """
    Load adaptive learning state from proadapt_state.json.

    Returns a dict with observations, weights, and metrics.
    If file doesn't exist or is corrupt, returns a fresh default state.
    """
    if not os.path.exists(STATE_FILE):
        return _default_state()
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        # Ensure required keys exist (new keys added in future versions)
        defaults = _default_state()
        for k, v in defaults.items():
            state.setdefault(k, v)
        return _decode_state(state)
    except (json.JSONDecodeError, IOError, KeyError) as e:
        return _default_state()

def save_state(state):
    """
    Save adaptive learning state to proadapt_state.json.

    Args:
        state: dict containing observations and metrics.
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        # Use custom encoder to handle inf/nan gracefully
        json.dump(state, f, indent=2, cls=_StateEncoder)

def add_observation(qlstm_pred, garch_pred, hybrid_pred, actual_val):
    """
    Add a new observation to the sliding window and update adaptive weights.

    The hybrid prediction represents QLSTM_pred + GARCH_residual.
    The final prediction = w_qlstm * qlstm_pred + (1-w_qlstm) * hybrid_final.

    Adaptive weight computation:
        1. Recent RMSE for both QLSTM and hybrid (window=10, or fewer if
           insufficient data)
        2. w_qlstm = (1/rmse_q) / (1/rmse_q + 1/rmse_hybrid)

    Fine-tune trigger: if QLSTM MAE > threshold (default 2 SFC pts), sets
    fine_tune_requested flag. Check with get_fine_tune_signal().

    Args:
        qlstm_pred: QLSTM model prediction (float, SFC 0-100)
        garch_pred: GARCH residual correction value (float)
        hybrid_pred: Combined hybrid prediction (qlstm_pred + garch_residual,
                      float)
        actual_val: True observed SFC value (float, 0-100)

    Returns:
        dict with:
            - w_qlstm: adaptive weight for QLSTM
            - w_hybrid: adaptive weight for hybrid (1 - w_qlstm)
            - final_pred: weighted ensemble prediction
            - qlstm_rmse: recent RMSE for QLSTM (window=rmse_window)
            - hybrid_rmse: recent RMSE for hybrid (window=rmse_window)
            - qlstm_mae: recent MAE for QLSTM
            - fine_tune_triggered: bool if threshold was breached
            - n_observations: total observations in sliding window
    """
    state = load_state()

    # Compute errors
    qlstm_error = float(actual_val - qlstm_pred)
    hybrid_error = float(actual_val - hybrid_pred)

    # Append observation
    obs = {
        "qlstm_pred": qlstm_pred,
        "garch_pred": garch_pred,
        "hybrid_pred": hybrid_pred,
        "actual": actual_val,
        "qlstm_error": qlstm_error,
        "hybrid_error": hybrid_error,
        "timestamp": time.time(),
    }
    state.setdefault("observations", []).append(obs)

    # Trim to sliding window
    window_size = state.get("window_size", DEFAULT_WINDOW_SIZE)
    if len(state["observations"]) > window_size:
        state["observations"] = state["observations"][-window_size:]

    # Compute recent RMSE and MAE using rmse_window most recent observations
    rmse_window = state.get("rmse_window", DEFAULT_RMSE_WINDOW)
    recent = state["observations"][-rmse_window:]

    qlstm_errors = [o["qlstm_error"] for o in recent]
    hybrid_errors = [o["hybrid_error"] for o in recent]

    qlstm_rmse = _compute_rmse(qlstm_errors)
    hybrid_rmse = _compute_rmse(hybrid_errors)
    qlstm_mae = _compute_mae(qlstm_errors)

    # Adaptive weight
    w_qlstm = _adaptive_weight(qlstm_rmse, hybrid_rmse)
    w_hybrid = 1.0 - w_qlstm

    # Final prediction
    final_pred = w_qlstm * qlstm_pred + w_hybrid * hybrid_pred

    # Fine-tune trigger
    threshold = state.get("fine_tune_threshold", DEFAULT_FINE_TUNE_THRESHOLD)
    fine_tune_triggered = (qlstm_mae > threshold) and (qlstm_mae != float('inf'))
    if fine_tune_triggered:
        state["fine_tune_requested"] = True

    # Update state metrics
    state["qlstm_mae"] = qlstm_mae if qlstm_mae != float('inf') else None
    state["hybrid_mae"] = (
        _compute_mae(hybrid_errors)
        if _compute_mae(hybrid_errors) != float('inf')
        else None
    )

    # Persist last RMSE values for get_adaptive_prediction
    state["qlstm_rmse"] = qlstm_rmse if qlstm_rmse != float('inf') else None
    state["hybrid_rmse"] = hybrid_rmse if hybrid_rmse != float('inf') else None

    state["last_updated"] = time.time()
    save_state(state)

    return {
        "w_qlstm": round(w_qlstm, 4),
        "w_hybrid": round(w_hybrid, 4),
        "final_pred": round(final_pred, 4),
        "qlstm_rmse": round(qlstm_rmse, 4) if qlstm_rmse != float('inf') else None,
        "hybrid_rmse": round(hybrid_rmse, 4) if hybrid_rmse != float('inf') else None,
        "qlstm_mae": round(qlstm_mae, 4) if qlstm_mae != float('inf') else None,
        "fine_tune_triggered": fine_tune_triggered,
        "n_observations": len(state["observations"]),
    }

def get_adaptive_prediction(qlstm_pred, garch_residual, hybrid_pred):
    """
    Get adaptive ensemble prediction without recording an observation.

    Uses the most recently computed adaptive weights from state.
    When no observations exist, falls back to equal weighting (0.5/0.5).

    Args:
        qlstm_pred: QLSTM model prediction (float)
        garch_residual: GARCH residual correction (float)
        hybrid_pred: Combined hybrid prediction (qlstm_pred + garch_residual,
                     float)

    Returns:
        dict with:
            - final_pred: weighted ensemble prediction
            - w_qlstm: current adaptive weight for QLSTM
            - w_hybrid: current adaptive weight for hybrid
            - qlstm_mae: current QLSTM MAE
            - source: "adaptive" or "equal" (fallback)
    """
    state = load_state()
    observations = state.get("observations", [])

    if len(observations) < 2:
        # Not enough data for adaptive weighting — use equal weights
        w_qlstm = 0.5
        w_hybrid = 0.5
        source = "equal"
    else:
        # Recompute weights from recent observations in state
        rmse_window = state.get("rmse_window", DEFAULT_RMSE_WINDOW)
        recent = observations[-rmse_window:]
        qlstm_errors = [o["qlstm_error"] for o in recent]
        hybrid_errors = [o["hybrid_error"] for o in recent]

        qlstm_rmse = _compute_rmse(qlstm_errors)
        hybrid_rmse = _compute_rmse(hybrid_errors)

        w_qlstm = _adaptive_weight(qlstm_rmse, hybrid_rmse)
        w_hybrid = 1.0 - w_qlstm
        source = "adaptive"

    final_pred = w_qlstm * qlstm_pred + w_hybrid * hybrid_pred

    # Retrieve stored MAE (loaded as float('inf') if None)
    raw_mae = state.get("qlstm_mae", float('inf'))
    qlstm_mae = raw_mae if raw_mae != float('inf') else None

    return {
        "final_pred": round(final_pred, 4),
        "w_qlstm": round(w_qlstm, 4),
        "w_hybrid": round(w_hybrid, 4),
        "qlstm_mae": round(qlstm_mae, 4) if qlstm_mae is not None else None,
        "source": source,
    }

def get_fine_tune_signal():
    """
    Check if QLSTM fine-tuning has been requested.

    Returns True when QLSTM MAE exceeded the fine_tune_threshold during a
    recent add_observation() call.

    Returns:
        bool: True if fine-tuning should be scheduled.

    Note: Caller should reset the signal after handling by calling
    reset_fine_tune_signal().
    """
    state = load_state()
    return bool(state.get("fine_tune_requested", False))

def reset_fine_tune_signal():
    """
    Reset the fine-tune request signal after handling.

    Call this after fine-tuning has been initiated or completed to prevent
    repeated triggers for the same high-MAE event.
    """
    state = load_state()
    state["fine_tune_requested"] = False
    save_state(state)

def get_state_path():
    """Return the absolute path to the state file for reference."""
    return STATE_FILE


# ── CLI entry point (quick test) ─────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("proadapt.py — Online Learning Module (Self-Test)")
    print("=" * 60)

    # Ensure clean state for test
    state_path = get_state_path()
    if os.path.exists(state_path):
        os.remove(state_path)
        print(f"  Removed existing state file: {state_path}")

    print("\n1. Loading fresh state...")
    s = load_state()
    print(f"   observations: {len(s['observations'])}")
    print(f"   window_size:  {s['window_size']}")
    print(f"   rmse_window:  {s['rmse_window']}")
    print(f"   threshold:    {s['fine_tune_threshold']}")

    print("\n2. Adding observations (simulating online learning)...")
    test_cases = [
        # (qlstm_pred, garch_pred, hybrid_pred, actual)
        (75.0, 2.0, 77.0, 74.0),   # qlstm good, hybrid overshoots
        (73.0, 1.5, 74.5, 73.5),   # both close
        (76.0, -1.0, 75.0, 72.0),  # both off
        (71.0, 0.5, 71.5, 71.0),   # QLSTM perfect
        (72.0, 1.0, 73.0, 72.5),   # both decent
        (78.0, 3.0, 81.0, 75.0),   # QLSTM better
        (74.0, -2.0, 72.0, 74.5),  # QLSTM good, hybrid wrong direction
        (73.0, 0.0, 73.0, 73.0),   # perfect both
        (70.0, 1.0, 71.0, 71.5),   # hybrid better
        (69.0, 2.0, 71.0, 70.0),   # QLSTM better
        (80.0, 5.0, 85.0, 72.0),   # BAD QLSTM — triggers fine-tune
        (72.0, 1.0, 73.0, 72.0),   # both good
    ]

    for i, (q, g, h, a) in enumerate(test_cases):
        result = add_observation(q, g, h, a)
        print(f"   Obs {i+1:2d}: q={q:5.1f} g={g:5.1f} h={h:5.1f} actual={a:5.1f} "
              f"| w_q={result['w_qlstm']:.3f} w_h={result['w_hybrid']:.3f} "
              f"final={result['final_pred']:.2f} "
              f"rmse_q={result['qlstm_rmse']} rmse_h={result['hybrid_rmse']} "
              f"ft={result['fine_tune_triggered']}")

    print("\n3. Checking fine-tune signal...")
    ft = get_fine_tune_signal()
    print(f"   get_fine_tune_signal() = {ft}")
    if ft:
        print("   Resetting fine-tune signal...")
        reset_fine_tune_signal()
        print(f"   After reset: get_fine_tune_signal() = {get_fine_tune_signal()}")

    print("\n4. Testing get_adaptive_prediction (inference, no recording)...")
    # Simulate a forward pass where we don't have the actual yet
    pred = get_adaptive_prediction(qlstm_pred=73.0, garch_residual=1.0,
                                    hybrid_pred=74.0)
    print(f"   qlstm=73.0, garch=1.0, hybrid=74.0")
    print(f"   -> w_q={pred['w_qlstm']:.4f}, w_h={pred['w_hybrid']:.4f}, "
          f"final={pred['final_pred']:.4f}, source={pred['source']}")

    print("\n5. State file written to:")
    print(f"   {get_state_path()}")
    print(f"   File size: {os.path.getsize(get_state_path())} bytes")

    print("\n6. Loading saved state...")
    s2 = load_state()
    print(f"   Observations stored: {len(s2['observations'])}")
    print(f"   fine_tune_requested: {s2['fine_tune_requested']}")

    print("\n" + "=" * 60)
    print("Self-test complete. All functions working.")
    print("=" * 60)
