#!/usr/bin/env python3
"""
confidence_calibration.py — Confidence Score Recalibration for SFC
==================================================================
Fixes Expected Calibration Error (ECE) by mapping raw confidence to
calibrated confidence using bin-based recalibration.

Method: 
  1. Load historical snapshots from git history
  2. Bin raw confidence into 10 buckets
  3. Compute actual stress signal rate per bucket
  4. Build a calibration mapping curve
  5. Apply mapping at inference time

Currently ECE = 0.422 (very poor). Target: ECE < 0.05.

Usage:
    from confidence_calibration import recalibrate
    calibrated_conf = recalibrate(raw_confidence=0.38, regime="CRISIS")
"""

import json
import math
import os
import subprocess
from typing import Any, Dict, List, Optional

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".calibration_state.json")


# ════════════════════════════════════════════════════════════════
# Calibration Builder
# ════════════════════════════════════════════════════════════════


def build_calibration_map(
    snapshots: Optional[List[Dict[str, Any]]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Build calibration mapping from historical data.

    Args:
        snapshots: List of data.json dicts. If None, loads from git history.
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Dict with calibration map, ECE, and per-bin data.
    """
    if snapshots is None:
        snapshots = _extract_snapshots()

    if not snapshots:
        return {"error": "No snapshots available"}

    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    bin_data = {f"{lo:.1f}-{hi:.1f}": {"count": 0, "stress": 0, "calm": 0, "conf_sum": 0.0}
                for lo, hi in bins}

    for snap in snapshots:
        conf = snap.get("composite_confidence") or 0.5
        sfc = snap.get("sfc_effective") or 0
        is_stress = sfc > 25.0  # use base threshold for calibration

        for lo, hi in bins:
            if lo <= conf < hi:
                label = f"{lo:.1f}-{hi:.1f}"
                bin_data[label]["count"] += 1
                bin_data[label]["conf_sum"] += conf
                if is_stress:
                    bin_data[label]["stress"] += 1
                else:
                    bin_data[label]["calm"] += 1
                break

    # Build calibration mapping
    calibration_curve = []
    for label, data in sorted(bin_data.items()):
        lo = float(label.split("-")[0])
        hi = float(label.split("-")[1])
        mid = (lo + hi) / 2.0
        count = data["count"]
        if count > 0:
            actual_rate = data["stress"] / count
            avg_conf = data["conf_sum"] / count
        else:
            actual_rate = mid
            avg_conf = mid

        calibration_curve.append({
            "bin": label,
            "count": count,
            "raw_confidence_mid": round(mid, 3),
            "avg_raw_confidence": round(avg_conf, 3),
            "actual_stress_rate": round(actual_rate, 3),
            "calibrated_confidence": round(min(1.0, max(0.0, actual_rate)), 3),
        })

    # ECE calculation
    total = sum(d["count"] for d in bin_data.values())
    ece = 0.0
    if total > 0:
        for d in calibration_curve:
            if d["count"] > 0:
                gap = abs(d["actual_stress_rate"] - d["raw_confidence_mid"])
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
        "total_snapshots": len(snapshots),
        "n_bins": n_bins,
        "interpretation": (
            "Well calibrated" if ece < 0.05 else
            "Moderately miscalibrated" if ece < 0.10 else
            f"Poorly calibrated (ECE={ece:.3f})"
        ),
    }

    # Persist state
    _save_state(state)
    return state


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
            return round(y1 + t * (y2 - y1), 3)

    return raw_confidence


def get_calibration_info() -> Dict[str, Any]:
    """Return current calibration state info."""
    state = _load_state()
    if not state:
        return {"status": "Not calibrated — run build_calibration_map() first"}
    return {
        "ece": state.get("ece"),
        "interpretation": state.get("interpretation"),
        "curve_points": len(state.get("calibration_curve", [])),
        "total_snapshots": state.get("total_snapshots"),
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
    print("Confidence Recalibration — Build + Test")
    print("=" * 60)

    print("\n📦 Building calibration map from git history...")
    state = build_calibration_map()
    if "error" in state:
        print(f"❌ {state['error']}")
        return

    print(f"\n📊 Calibration Results ({state['total_snapshots']} snapshots)")
    print(f"   ECE: {state['ece']} — {state['interpretation']}")
    print(f"\n   Calibration Curve:")
    print(f"   {'Bin':<12} {'Count':>6} {'Raw':>6} {'Actual':>8} {'Calibrated':>12}")
    print(f"   " + "-" * 46)
    for d in state["calibration_curve"]:
        if d["count"] > 0:
            print(f"   {d['bin']:<12} {d['count']:>6} {d['raw_confidence_mid']:>6.2f} "
                  f"{d['actual_stress_rate']:>8.3f} {d['calibrated_confidence']:>12.3f}")

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
