#!/usr/bin/env python3
"""
m2_analysis.py — Step 1: Error Per Regime
==========================================
Extract historical data.json snapshots from git history,
group by regime (BULL/BEAR/SIDEWAYS/CRISIS), compute per-regime
statistics and error patterns.

Usage:  python3 m2_analysis.py
"""

import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SFC_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Extract all data.json snapshots from git ──


def extract_snapshots(max_count: int = 500) -> List[Dict[str, Any]]:
    """Extract data.json snapshots from git history."""
    try:
        result = subprocess.check_output(
            ["git", "log", "--oneline", "--all", "--diff-filter=M",
             "--reverse", "--", "data.json"],
            text=True, timeout=30, cwd=SFC_DIR,
        ).strip().split("\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Git log failed: {e}", file=sys.stderr)
        return []

    result = [r for r in result if r.strip()]
    if max_count and len(result) > max_count:
        # Take evenly spaced samples
        step = len(result) // max_count
        result = result[::step]

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

    print(f"  Extracted {len(snapshots)} snapshots ({errors} errors)")
    return snapshots


# ── Per-Regime Analysis ──


def analyze_per_regime(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy and statistics per regime."""
    regime_groups: Dict[str, List[Dict]] = defaultdict(list)
    for snap in snapshots:
        regime = str(snap.get("regime", "NORMAL")).upper()
        regime_groups[regime].append(snap)

    results = {}
    for regime, group in sorted(regime_groups.items()):
        n = len(group)
        sfc_vals = [s.get("sfc_effective", 50) or 50 for s in group]
        conf_vals = [s.get("composite_confidence", 0.5) or 0.5 for s in group]
        agreement_vals = [s.get("method_agreement", 0.5) or 0.5 for s in group]
        btc_vals = [s.get("btc_24h", 0) or 0 for s in group]

        mean_sfc = sum(sfc_vals) / n
        std_sfc = math.sqrt(sum((s - mean_sfc)**2 for s in sfc_vals) / n)
        mean_conf = sum(conf_vals) / n
        mean_agree = sum(agreement_vals) / n
        mean_btc = sum(btc_vals) / n

        # Signal distribution
        signal_types = defaultdict(int)
        zones = defaultdict(int)
        for s in group:
            signal_types[str(s.get("signal_type", "?"))] += 1
            zones[str(s.get("zone", "?"))] += 1

        # Transition detection rate (% of time in this regime showing STRESS signal)
        stress_count = sum(1 for s in group
                          if str(s.get("signal_type", "")).startswith("STRESS"))
        stress_rate = stress_count / n if n > 0 else 0

        # Calm misclassification (CALM signal in CRISIS/BEAR regime)
        calm_in_crisis = sum(1 for s in group
                            if str(s.get("signal_type", "")) == "CALM"
                            and regime in ("CRISIS", "BEAR"))
        calm_misrate = calm_in_crisis / n if n > 0 else 0

        results[regime] = {
            "count": n,
            "pct": round(n / len(snapshots) * 100, 1),
            "mean_sfc": round(mean_sfc, 2),
            "std_sfc": round(std_sfc, 2),
            "mean_confidence": round(mean_conf, 3),
            "mean_agreement": round(mean_agree, 3),
            "mean_btc_24h_pct": round(mean_btc, 2),
            "signal_types": dict(signal_types),
            "zones": dict(zones),
            "stress_signal_rate": round(stress_rate, 3),
            "calm_misclassification_rate": round(calm_misrate, 3),
            "signal_alignment": "ALIGNED" if (
                (regime in ("CRISIS", "BEAR") and stress_rate > 0.5) or
                (regime in ("BULL", "SIDEWAYS") and stress_rate < 0.5)
            ) else "MISALIGNED",
        }

    return results


# ── Step 2: Residual Analysis ──


def analyze_residuals(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find patterns in prediction errors.

    Defines error types:
      - False STRESS: SFC > 25 (STRESS signal) but BTC_24h > +2% (market up)
      - Missed STRESS: SFC < 25 (CALM signal) but BTC_24h < -3% (market down)
    """
    false_stress = []    # model said STRESS, was wrong
    missed_stress = []   # model said CALM, was wrong
    correct_stress = []  # model said STRESS, was right (BTC down)
    correct_calm = []    # model said CALM, was right (BTC flat/up)

    stress_threshold = 25.0
    btc_down_threshold = -2.0   # BTC dropped >2%
    btc_up_threshold = 1.5      # BTC rose >1.5%

    for snap in snapshots:
        sfc = snap.get("sfc_effective") or 0
        btc_24h = snap.get("btc_24h") or 0
        conf = snap.get("composite_confidence") or 0.5
        agree = snap.get("method_agreement") or 0.5
        regime = str(snap.get("regime", "NORMAL"))
        dvol = snap.get("dvol")
        news_sent = snap.get("news_sentiment")

        features = {
            "sfc": sfc, "btc_24h": btc_24h, "conf": conf,
            "agree": agree, "regime": regime, "dvol": dvol,
            "news_sentiment": news_sent,
        }

        if sfc > stress_threshold:
            if btc_24h < btc_down_threshold:
                correct_stress.append(features)
            elif btc_24h > btc_up_threshold:
                false_stress.append(features)
        else:
            if btc_24h < btc_down_threshold:
                missed_stress.append(features)
            else:
                correct_calm.append(features)

    def avg_features(items, field):
        vals = [i.get(field) for i in items if i.get(field) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def avg_regime(items):
        regimes = [i["regime"] for i in items]
        return max(set(regimes), key=regimes.count) if regimes else "N/A"

    return {
        "total": len(snapshots),
        "correct_stress": len(correct_stress),
        "false_stress": len(false_stress),
        "missed_stress": len(missed_stress),
        "correct_calm": len(correct_calm),
        "false_stress_pct": round(len(false_stress) / max(1, len(snapshots)) * 100, 1),
        "missed_stress_pct": round(len(missed_stress) / max(1, len(snapshots)) * 100, 1),
        "false_stress_profile": {
            "avg_sfc": avg_features(false_stress, "sfc"),
            "avg_conf": avg_features(false_stress, "conf"),
            "avg_agree": avg_features(false_stress, "agree"),
            "avg_btc": avg_features(false_stress, "btc_24h"),
            "dominant_regime": avg_regime(false_stress),
            "avg_dvol": avg_features(false_stress, "dvol"),
            "avg_news": avg_features(false_stress, "news_sentiment"),
        },
        "missed_stress_profile": {
            "avg_sfc": avg_features(missed_stress, "sfc"),
            "avg_conf": avg_features(missed_stress, "conf"),
            "avg_agree": avg_features(missed_stress, "agree"),
            "avg_btc": avg_features(missed_stress, "btc_24h"),
            "dominant_regime": avg_regime(missed_stress),
            "avg_dvol": avg_features(missed_stress, "dvol"),
            "avg_news": avg_features(missed_stress, "news_sentiment"),
        },
    }


# ── Step 3: Error Autocorrelation ──


def analyze_autocorrelation(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if SFC errors are autocorrelated (model slow to adapt)."""
    if len(snapshots) < 5:
        return {"error": "Not enough snapshots"}

    sfc_vals = [s.get("sfc_effective", 50) or 50 for s in snapshots]
    btc_vals = [s.get("btc_24h", 0) or 0 for s in snapshots]

    # SFC changes (1st diff)
    sfc_diff = [sfc_vals[i] - sfc_vals[i-1] for i in range(1, len(sfc_vals))]

    # Lag-1 autocorrelation of SFC
    if len(sfc_diff) > 3:
        lag1_sfc = sfc_diff[:-1]
        lag0_sfc = sfc_diff[1:]
        n = len(lag0_sfc)
        if n > 2:
            mean0 = sum(lag0_sfc) / n
            mean1 = sum(lag1_sfc) / n
            num = sum((lag0_sfc[i] - mean0) * (lag1_sfc[i] - mean1) for i in range(n))
            den = math.sqrt(sum((x - mean0)**2 for x in lag0_sfc) *
                            sum((x - mean1)**2 for x in lag1_sfc)) + 1e-10
            sfc_ac = round(num / den, 4)
        else:
            sfc_ac = 0
    else:
        sfc_ac = 0

    # SFC vs BTC correlation
    n2 = min(len(sfc_vals), len(btc_vals))
    if n2 > 2:
        mean_s = sum(sfc_vals[:n2]) / n2
        mean_b = sum(btc_vals[:n2]) / n2
        num2 = sum((sfc_vals[i] - mean_s) * (btc_vals[i] - mean_b) for i in range(n2))
        den2 = math.sqrt(sum((x - mean_s)**2 for x in sfc_vals[:n2]) *
                         sum((x - mean_b)**2 for x in btc_vals[:n2])) + 1e-10
        sfc_btc_corr = round(num2 / den2, 4)
    else:
        sfc_btc_corr = 0

    # Run length: how many consecutive same-direction SFC moves?
    if sfc_diff:
        streaks = []
        current = 1
        for i in range(1, len(sfc_diff)):
            if (sfc_diff[i] > 0) == (sfc_diff[i-1] > 0):
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)
        max_streak = max(streaks) if streaks else 1
        avg_streak = round(sum(streaks) / len(streaks), 2) if streaks else 1
    else:
        max_streak = 1
        avg_streak = 1

    return {
        "sfc_lag1_autocorrelation": sfc_ac,
        "sfc_btc_correlation": sfc_btc_corr,
        "consecutive_same_direction": {
            "max": max_streak,
            "avg": avg_streak,
        },
        "interpretation": (
            "SFC changes are autocorrelated (model slow)" if abs(sfc_ac) > 0.3 else
            "SFC changes are not significantly autocorrelated"
        ),
    }


# ── Step 4: Confidence Calibration ──


def analyze_calibration(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if confidence scores are well-calibrated.

    Bins confidence into 5 groups and checks actual stress signal rate per bin.
    """
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_data = {f"{lo:.1f}-{hi:.1f}": {"count": 0, "stress": 0, "calm": 0}
                for lo, hi in bins}

    stress_threshold = 25.0

    for snap in snapshots:
        conf = snap.get("composite_confidence") or 0.5
        sfc = snap.get("sfc_effective") or 0
        is_stress = sfc > stress_threshold

        for lo, hi in bins:
            if lo <= conf < hi:
                label = f"{lo:.1f}-{hi:.1f}"
                bin_data[label]["count"] += 1
                if is_stress:
                    bin_data[label]["stress"] += 1
                else:
                    bin_data[label]["calm"] += 1
                break

    # Compute calibration
    calibration_results = {}
    for label, data in bin_data.items():
        if data["count"] > 0:
            actual_stress_rate = data["stress"] / data["count"]
            # Midpoint of bin = predicted confidence
            lo, hi = [float(x) for x in label.split("-")]
            predicted_conf = (lo + hi) / 2
            calibration_results[label] = {
                "count": data["count"],
                "actual_stress_rate": round(actual_stress_rate, 3),
                "predicted_confidence": predicted_conf,
                "gap": round(actual_stress_rate - predicted_conf, 3),
            }
        else:
            calibration_results[label] = {
                "count": 0, "actual_stress_rate": None,
                "predicted_confidence": (lo + hi) / 2, "gap": None,
            }

    # Overall calibration error (ECE: Expected Calibration Error)
    total = sum(d["count"] for d in bin_data.values())
    ece = sum(
        d["count"] / total * abs(d.get("actual_stress_rate", 0) or 0 - d["predicted_confidence"])
        for d in calibration_results.values()
    ) if total > 0 else 0

    return {
        "calibration_by_bin": calibration_results,
        "expected_calibration_error": round(ece, 4),
        "interpretation": (
            "Well calibrated" if ece < 0.05 else
            "Moderately miscalibrated" if ece < 0.10 else
            "Poorly calibrated — needs recalibration"
        ),
    }


# ── Step 5: Ensemble Disagreement Analysis ──


def analyze_disagreement(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if ensemble disagreement predicts accuracy.

    Bins method_agreement into 4 groups and checks signal accuracy in each.
    """
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    bin_data = {f"{lo:.2f}-{hi:.2f}": {"count": 0, "stress_correct": 0}
                for lo, hi in bins}

    stress_threshold = 25.0
    btc_down = -2.0

    for snap in snapshots:
        agree = snap.get("method_agreement") or 0.5
        sfc = snap.get("sfc_effective") or 0
        btc_24h = snap.get("btc_24h") or 0

        # Is the signal correct?
        if sfc > stress_threshold and btc_24h < btc_down:
            correct = 1  # Stress signal, BTC down — correct
        elif sfc <= stress_threshold and btc_24h >= btc_down:
            correct = 1  # Calm signal, BTC not down — correct
        else:
            correct = 0  # Wrong

        for lo, hi in bins:
            if lo <= agree < hi:
                label = f"{lo:.2f}-{hi:.2f}"
                bin_data[label]["count"] += 1
                bin_data[label]["stress_correct"] += correct
                break

    disagreement_results = {}
    for label, data in bin_data.items():
        if data["count"] > 0:
            acc = data["stress_correct"] / data["count"]
            disagreement_results[label] = {
                "count": data["count"],
                "accuracy": round(acc, 3),
            }
        else:
            disagreement_results[label] = {"count": 0, "accuracy": None}

    # Optimal abstain threshold: find agreement level where accuracy drops
    best_threshold = 0.0
    best_accuracy = 0.0
    for lo, hi in bins:
        label = f"{lo:.2f}-{hi:.2f}"
        d = disagreement_results[label]
        if d["accuracy"] and d["accuracy"] > best_accuracy:
            best_accuracy = d["accuracy"]
            best_threshold = lo

    return {
        "accuracy_by_agreement": disagreement_results,
        "optimal_abstain_threshold": best_threshold,
        "recommendation": (
            f"ABSTAIN when method_agreement < {best_threshold:.2f}"
            if best_threshold > 0 else
            "No clear abstain threshold found"
        ),
    }


# ── Per-Regime Transition Analysis ──


def analyze_transitions(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze accuracy around regime transition points."""
    if len(snapshots) < 3:
        return {"error": "Not enough snapshots"}

    regimes = [str(s.get("regime", "NORMAL")) for s in snapshots]
    sfc_vals = [s.get("sfc_effective", 50) or 50 for s in snapshots]

    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append({
                "index": i,
                "from": regimes[i-1],
                "to": regimes[i],
                "sfc_before": sfc_vals[i-1],
                "sfc_after": sfc_vals[i],
                "sfc_change": round(sfc_vals[i] - sfc_vals[i-1], 2),
            })

    # What's the SFC trend in the 3 days before a transition?
    pre_transition_sfc = []
    for t in transitions:
        idx = t["index"]
        window = sfc_vals[max(0, idx-3):idx]
        if window:
            pre_transition_sfc.append({
                "transition": f"{t['from']}→{t['to']}",
                "mean_sfc_before": round(sum(window) / len(window), 2),
                "sfc_at_transition": t["sfc_before"],
            })

    return {
        "total_transitions": len(transitions),
        "transitions": transitions[-20:],  # last 20
        "pre_transition_analysis": pre_transition_sfc[-10:] if pre_transition_sfc else [],
    }


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 65)
    print("M2 Analysis — Step 1: Error Per Regime")
    print("=" * 65)

    print("\n📦 Extracting snapshots from git history...")
    snapshots = extract_snapshots(max_count=500)
    if not snapshots:
        print("❌ No snapshots extracted!")
        sys.exit(1)

    print(f"\n📊 Per-Regime Statistics ({len(snapshots)} snapshots)")
    print("=" * 65)

    results = analyze_per_regime(snapshots)
    for regime, r in sorted(results.items()):
        alignment_mark = "✅" if r["signal_alignment"] == "ALIGNED" else "⚠️"
        print(f"\n  {alignment_mark} {regime} ({r['count']} snap, {r['pct']}%)")
        print(f"     SFC:          {r['mean_sfc']:.1f} ± {r['std_sfc']:.1f}")
        print(f"     Confidence:   {r['mean_confidence']:.3f}")
        print(f"     Agreement:    {r['mean_agreement']:.3f}")
        print(f"     Avg BTC 24h: {r['mean_btc_24h_pct']:+.2f}%")
        print(f"     Stress sig:   {r['stress_signal_rate']*100:.0f}%")
        print(f"     Calm in crisis: {r['calm_misclassification_rate']*100:.1f}%")
        print(f"     Zones:        {r['zones']}")
        print(f"     Signal types: {r['signal_types']}")

    print(f"\n🔄 Regime Transition Analysis")
    print("=" * 65)
    trans = analyze_transitions(snapshots)
    print(f"  Total transitions: {trans['total_transitions']}")

    if trans.get("pre_transition_analysis"):
        print("\n  Last transitions (SFC before switch):")
        for t in trans["transitions"][-5:]:
            print(f"    {t['from']}→{t['to']}: SFC {t['sfc_before']:.1f} → {t['sfc_after']:.1f}")

    # ── Key Insights ──
    print(f"\n💡 Key Insights")
    print("=" * 65)

    # Find which regime has lowest confidence
    lowest_conf = min(results.items(), key=lambda x: x[1]["mean_confidence"])
    print(f"  🔸 Lowest confidence: {lowest_conf[0]} ({lowest_conf[1]['mean_confidence']:.3f})")

    # Find which regime has highest SFC volatility
    highest_vol = max(results.items(), key=lambda x: x[1]["std_sfc"])
    print(f"  🔸 Highest SFC variance: {highest_vol[0]} (σ={highest_vol[1]['std_sfc']:.1f})")

    # Calm misclassification rate
    total_calm_mis = sum(
        r["calm_misclassification_rate"] * r["count"]
        for r in results.values()
    )
    total_count = sum(r["count"] for r in results.values())
    overall_calm_mis = total_calm_mis / total_count if total_count > 0 else 0
    print(f"  🔸 Overall calm-in-crisis rate: {overall_calm_mis*100:.1f}%")

    # Regime with worst alignment
    misaligned = [(r, v) for r, v in results.items()
                  if v["signal_alignment"] == "MISALIGNED"]
    if misaligned:
        for r, v in misaligned:
            print(f"  ⚠️ Misaligned signal in {r}: stress_rate={v['stress_signal_rate']*100:.0f}%")

    print(f"\n✅ Step 1 complete\n")

    # ════════════════════════════════════════════════════════════════
    # STEP 2: Residual Analysis
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("Step 2: Residual Analysis — Pola Error Sistematis")
    print("=" * 65)
    residuals = analyze_residuals(snapshots)
    print(f"\n  Correct stress: {residuals['correct_stress']}")
    print(f"  False stress:   {residuals['false_stress']} ({residuals['false_stress_pct']}%)")
    print(f"  Missed stress:  {residuals['missed_stress']} ({residuals['missed_stress_pct']}%)")
    print(f"  Correct calm:   {residuals['correct_calm']}")

    if residuals['false_stress'] > 0:
        fp = residuals['false_stress_profile']
        print(f"\n  🔸 False Stress Profile:")
        print(f"     Avg SFC:      {fp['avg_sfc']}")
        print(f"     Avg Conf:     {fp['avg_conf']}")
        print(f"     Avg Agree:    {fp['avg_agree']}")
        print(f"     Avg BTC:      {fp['avg_btc']:+.2f}%")
        print(f"     Regime:       {fp['dominant_regime']}")
        print(f"     Avg DVol:     {fp['avg_dvol']}")
        print(f"     Avg News:     {fp['avg_news']}")

    if residuals['missed_stress'] > 0:
        mp = residuals['missed_stress_profile']
        print(f"\n  🔸 Missed Stress Profile:")
        print(f"     Avg SFC:      {mp['avg_sfc']}")
        print(f"     Avg Conf:     {mp['avg_conf']}")
        print(f"     Avg Agree:    {mp['avg_agree']}")
        print(f"     Avg BTC:      {mp['avg_btc']:+.2f}%")
        print(f"     Regime:       {mp['dominant_regime']}")
        print(f"     Avg DVol:     {mp['avg_dvol']}")
        print(f"     Avg News:     {mp['avg_news']}")

    print(f"\n✅ Step 2 complete\n")

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Error Autocorrelation
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("Step 3: Error Autocorrelation")
    print("=" * 65)
    autocorr = analyze_autocorrelation(snapshots)
    print(f"\n  SFC Lag-1 autocorrelation: {autocorr['sfc_lag1_autocorrelation']}")
    print(f"  SFC-BTC correlation:       {autocorr['sfc_btc_correlation']}")
    print(f"  Consecutive SFC direction: avg={autocorr['consecutive_same_direction']['avg']} max={autocorr['consecutive_same_direction']['max']}")
    print(f"  → {autocorr['interpretation']}")
    print(f"\n✅ Step 3 complete\n")

    # ════════════════════════════════════════════════════════════════
    # STEP 4: Confidence Calibration
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("Step 4: Confidence Calibration")
    print("=" * 65)
    calib = analyze_calibration(snapshots)
    print(f"\n  ECE: {calib['expected_calibration_error']} — {calib['interpretation']}")
    print(f"\n  Calibration by bin:")
    for label, d in sorted(calib['calibration_by_bin'].items()):
        if d['count'] > 0:
            gap_mark = "⚠️" if d['gap'] and abs(d['gap']) > 0.1 else "✅"
            print(f"    {label}: pred={d['predicted_confidence']:.1f} "
                  f"actual={d['actual_stress_rate']:.3f} "
                  f"gap={d['gap']:+.3f} {gap_mark}")
        else:
            print(f"    {label}: no data")
    print(f"\n✅ Step 4 complete\n")

    # ════════════════════════════════════════════════════════════════
    # STEP 5: Ensemble Disagreement
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("Step 5: Ensemble Disagreement Analysis")
    print("=" * 65)
    dis = analyze_disagreement(snapshots)
    print(f"\n  Accuracy by agreement bin:")
    for label, d in sorted(dis['accuracy_by_agreement'].items()):
        if d['count'] > 0:
            print(f"    {label}: acc={d['accuracy']:.3f} (n={d['count']})")
    print(f"\n  → {dis['recommendation']}")
    print(f"\n✅ Step 5 complete\n")

    # ════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("FINAL SUMMARY — Actionable Insights")
    print("=" * 65)
    print(f"""
  📊 Per-Regime:
     • CRISIS: {results.get('CRISIS', {}).get('calm_misclassification_rate', 0)*100:.1f}% calm-in-crisis
     • Fix: raise stress threshold sensitivity in CRISIS regime

  🎯 Residuals:
     • False stress: {residuals['false_stress_pct']}% of all predictions
     • Missed stress: {residuals['missed_stress_pct']}%

  🔄 Autocorrelation:
     • SFC lag-1: {autocorr['sfc_lag1_autocorrelation']}
     • {autocorr['interpretation']}

  📐 Calibration:
     • ECE: {calib['expected_calibration_error']}
     • {calib['interpretation']}

  🤝 Disagreement:
     • {dis['recommendation']}
""")


if __name__ == "__main__":
    main()
