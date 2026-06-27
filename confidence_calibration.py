#!/usr/bin/env python3
"""
confidence_calibration.py — Confidence Score Recalibration for SFC
==================================================================
Fixes Expected Calibration Error (ECE) by mapping raw confidence to
calibrated confidence using bin-based recalibration.

Method:
  1. Load historical snapshots from git history
  2. Bin raw confidence into 10 buckets
  3. Compute actual stress signal rate per bucket using TWO ground truths:
     a) Model-internal: sfc_effective > threshold (legacy)
     b) Price-outcome: BTC price dropped >X% between consecutive snapshots
  4. Build a calibration mapping curve (uses price-outcome by default)
  5. Apply mapping at inference time

Currently raw (pre-calibration) ECE ≈ 0.422. Target post-calibration ECE < 0.05.

Usage:
    from confidence_calibration import recalibrate
    calibrated_conf = recalibrate(raw_confidence=0.38, regime="CRISIS")
"""

import json
import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".calibration_state.json")

# ── Price-Outcome Ground Truth Config ──
PRICE_DROP_THRESHOLD = -0.02  # 2% drop = "stress was correct" for price-outcome
PRICE_RISE_THRESHOLD = 0.02   # 2% rise = "calm was correct"
# Weight assigned to price-outcome calibration (vs model-internal)
# 1.0 = only price-outcome, 0.0 = only model-internal
PRICE_OUTCOME_WEIGHT = 0.7


# ════════════════════════════════════════════════════════════════
# Calibration Builder
# ════════════════════════════════════════════════════════════════


def build_calibration_map(
    snapshots: Optional[List[Dict[str, Any]]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Build calibration mapping from historical data using dual ground truth.

    Args:
        snapshots: List of sorted (chronological) data.json dicts.
                   If None, loads from git history.
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Dict with calibration map, ECE, and per-bin data.
    """
    if snapshots is None:
        snapshots = _extract_snapshots()
        # Sort chronologically by timestamp for price-outcome comparison
        snapshots.sort(key=lambda s: s.get("ts", ""))

    if not snapshots:
        return {"error": "No snapshots available"}

    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    bin_data = {
        f"{lo:.1f}-{hi:.1f}": {
            "count": 0,
            "model_stress": 0,
            "model_calm": 0,
            "price_stress": 0,    # stress confirmed by actual price drop
            "price_calm": 0,      # calm confirmed by actual price rise/flat
            "price_neutral": 0,   # price moved opposite to prediction
            "conf_sum": 0.0,
        }
        for lo, hi in bins
    }

    for idx, snap in enumerate(snapshots):
        conf = snap.get("composite_confidence") or 0.5
        sfc = snap.get("sfc_effective") or 0
        is_stress_model = sfc > 25.0  # model-internal threshold

        # Price-outcome ground truth: compare current BTC to next snapshot
        is_stress_price = _compute_price_outcome(snap, snapshots, idx)

        for lo, hi in bins:
            if lo <= conf < hi:
                label = f"{lo:.1f}-{hi:.1f}"
                bin_data[label]["count"] += 1
                bin_data[label]["conf_sum"] += conf
                if is_stress_model:
                    bin_data[label]["model_stress"] += 1
                else:
                    bin_data[label]["model_calm"] += 1

                if is_stress_price is True:
                    bin_data[label]["price_stress"] += 1
                elif is_stress_price is False:
                    bin_data[label]["price_calm"] += 1
                else:
                    # is_stress_price is None — no next snapshot to compare
                    pass
                break

    # Build calibration mapping using weighted ground truth
    calibration_curve = []
    for label, data in sorted(bin_data.items()):
        lo = float(label.split("-")[0])
        hi = float(label.split("-")[1])
        mid = (lo + hi) / 2.0
        count = data["count"]

        # Model-internal actual rate
        model_rate = data["model_stress"] / count if count > 0 else mid

        # Price-outcome actual rate
        price_total = data["price_stress"] + data["price_calm"]
        price_rate = data["price_stress"] / price_total if price_total > 0 else None

        # Weighted blended rate
        if price_rate is not None and price_total >= 3:
            # Enough price-outcome data — blend
            w = PRICE_OUTCOME_WEIGHT
            actual_rate = w * price_rate + (1 - w) * model_rate
        else:
            # Fall back to model-internal when price data is sparse
            actual_rate = model_rate

        avg_conf = data["conf_sum"] / count if count > 0 else mid

        calibration_curve.append({
            "bin": label,
            "count": count,
            "raw_confidence_mid": round(mid, 3),
            "avg_raw_confidence": round(avg_conf, 3),
            "model_stress_rate": round(model_rate, 3),
            "price_stress_rate": round(price_rate, 3) if price_rate is not None else None,
            "blended_stress_rate": round(actual_rate, 3),
            "calibrated_confidence": round(min(1.0, max(0.0, actual_rate)), 3),
        })

    # ECE calculation (against blended ground truth)
    total = sum(d["count"] for d in bin_data.values())
    ece = 0.0
    if total > 0:
        for d in calibration_curve:
            if d["count"] > 0:
                gap = abs(d["blended_stress_rate"] - d["raw_confidence_mid"])
                ece += (d["count"] / total) * gap

    # Build mapping function: raw_conf -> calibrated_conf
    mapping_points = {}
    for d in calibration_curve:
        raw = d["raw_confidence_mid"]
        cal = d["calibrated_confidence"]
        mapping_points[raw] = cal

    state = {
        "calibration_curve": calibration_curve,
        "mapping_points": mapping_points,
        "ece": round(ece, 4),
        "ece_model_only": _compute_ece_model_only(calibration_curve),
        "total_snapshots": len(snapshots),
        "n_bins": n_bins,
        "price_outcome_weight": PRICE_OUTCOME_WEIGHT,
        "price_drop_threshold": PRICE_DROP_THRESHOLD,
        "interpretation": (
            "Well calibrated" if ece < 0.05 else
            "Moderately miscalibrated" if ece < 0.10 else
            f"Poorly calibrated (ECE={ece:.3f})"
        ),
    }

    # Persist state
    _save_state(state)
    return state


def _compute_price_outcome(
    snap: Dict,
    snapshots: List[Dict],
    idx: int,
) -> Optional[bool]:
    """Determine if stress was correct based on actual BTC price movement.

    Compares current snapshot's BTC price to the next snapshot's BTC price.
    Returns:
        True  = price dropped significantly — stress was correct
        False = price rose/flat — stress was wrong
        None  = no next snapshot available
    """
    if idx >= len(snapshots) - 1:
        return None  # no next snapshot to compare

    curr_price = snap.get("btc", 0)
    next_price = snapshots[idx + 1].get("btc", 0)
    if not curr_price or not next_price:
        return None

    pct_change = (next_price - curr_price) / curr_price
    sfc = snap.get("sfc_effective") or 0
    is_stress_model = sfc > 25.0

    if pct_change <= PRICE_DROP_THRESHOLD:
        # Price dropped significantly
        return True  # stress confirmed
    elif pct_change >= PRICE_RISE_THRESHOLD:
        # Price rose significantly
        return False  # stress was wrong
    else:
        # Price was flat — neutral (not clearly stress or calm)
        return None


def _compute_ece_model_only(curve: List[Dict]) -> float:
    """Compute ECE using only model-internal ground truth for comparison."""
    total = sum(d["count"] for d in curve)
    if total == 0:
        return 0.0
    ece = 0.0
    for d in curve:
        if d["count"] > 0:
            gap = abs(d["model_stress_rate"] - d["raw_confidence_mid"])
            ece += (d["count"] / total) * gap
    return round(ece, 4)


# ════════════════════════════════════════════════════════════════
# Recalibration (inference)
# ════════════════════════════════════════════════════════════════


def recalibrate(raw_confidence: float, state: Optional[Dict] = None) -> float:
    """Map raw confidence to calibrated confidence.

    Uses linear interpolation between calibration points.
    Falls back to raw confidence if no calibration data.
    """
    if state is None:
        state = _load_state()
        if not state:
            return raw_confidence

    mapping = state.get("mapping_points", {})
    if not mapping:
        return raw_confidence

    # Convert string keys back to floats (JSON serialization)
    mapping = {float(k): v for k, v in mapping.items()}

    # Sort calibration points
    points = sorted(mapping.items(), key=lambda x: x[0])

    # Edge cases
    if raw_confidence <= points[0][0]:
        return points[0][1]
    if raw_confidence >= points[-1][0]:
        return points[-1][1]

    # Linear interpolation between nearest points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= raw_confidence <= x2:
            # Linear interpolation
            t = (raw_confidence - x1) / (x2 - x1) if (x2 - x1) > 1e-10 else 0.0
            cal = y1 + t * (y2 - y1)
            return round(max(0.0, min(1.0, cal)), 3)

    return raw_confidence


def get_calibration_info() -> Dict[str, Any]:
    """Return current calibration state info."""
    state = _load_state()
    if not state:
        return {"status": "Not calibrated — run build_calibration_map() first"}
    return {
        "ece": state.get("ece"),
        "ece_model_only": state.get("ece_model_only"),
        "interpretation": state.get("interpretation"),
        "curve_points": len(state.get("calibration_curve", [])),
        "total_snapshots": state.get("total_snapshots"),
        "price_outcome_weight": state.get("price_outcome_weight"),
    }


# ════════════════════════════════════════════════════════════════
# Internal
# ════════════════════════════════════════════════════════════════


def _extract_snapshots(max_count: int = 500) -> List[Dict[str, Any]]:
    """Extract data.json snapshots from git history."""
    try:
        result = subprocess.check_output(
            ["git", "log", "--oneline", "--all", "--diff-filter=M",
             "--reverse", "--", "data.json"],
            text=True, timeout=30, cwd=SFC_DIR,
        ).strip().split("\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    result = [r for r in result if r.strip()]
    if max_count and len(result) > max_count:
        step = len(result) // max_count
        result = result[::step]

    snapshots = []
    for line in result:
        sha = line.split()[0]
        try:
            content = subprocess.check_output(
                ["git", "show", f"{sha}:data.json"],
                text=True, timeout=10, cwd=SFC_DIR,
            )
            if content.strip().startswith("{"):
                snapshots.append(json.loads(content))
        except (json.JSONDecodeError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            continue

    return snapshots


def _load_state() -> Optional[Dict]:
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_state(state: Dict) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


# ════════════════════════════════════════════════════════════════
# Standalone Test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60)
    print("Confidence Recalibration — Build + Test (v2)")
    print("=" * 60)

    print("\n📦 Building calibration map from git history...")
    state = build_calibration_map()
    if "error" in state:
        print(f"❌ {state['error']}")
        return

    print(f"\n📊 Calibration Results ({state['total_snapshots']} snapshots)")
    print(f"   ECE (blended):  {state['ece']} — {state['interpretation']}")
    print(f"   ECE (model-only legacy): {state['ece_model_only']}")
    print(f"   Price-outcome weight: {state.get('price_outcome_weight', 'N/A')}")
    print(f"\n   Calibration Curve:")
    print(f"   {'Bin':<12} {'Count':>6} {'Raw':>6} {'Model':>7} {'Price':>7} {'Blend':>7}")
    print(f"   " + "-" * 49)
    for d in state["calibration_curve"]:
        if d["count"] > 0:
            price_str = f"{d['price_stress_rate']:.3f}" if d['price_stress_rate'] is not None else "  N/A"
            print(f"   {d['bin']:<12} {d['count']:>6} {d['raw_confidence_mid']:>6.2f} "
                  f"{d['model_stress_rate']:>7.3f} {price_str:>7} {d['blended_stress_rate']:>7.3f}")

    print(f"\n🔄 Test Recalibration:")
    test_confs = [0.1, 0.2, 0.3, 0.38, 0.5, 0.6, 0.8, 0.9]
    print(f"   {'Raw Conf':>10} → {'Calibrated':>12}")
    print(f"   " + "-" * 24)
    for raw in test_confs:
        cal = recalibrate(raw, state)
        print(f"   {raw:>10.2f} → {cal:>12.3f}")

    print(f"\n✅ Calibration complete — state saved to {STATE_PATH}")


if __name__ == "__main__":
    main()
