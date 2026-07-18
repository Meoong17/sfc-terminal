#!/usr/bin/env python3
"""
composite_weight_optimizer.py — Correlation-aware weight optimization for
GLF/SLI/MPI/Q10's INTERNAL sub-components (experimental, display-only).

WHY THIS EXISTS:
    GLF, SLI, MPI, and Q10's 4 groups currently use MANUALLY-ASSIGNED
    weights for their internal sub-components (e.g. GLF: Fed=30%, ECB=15%,
    etc.) — reasonable starting points, but not empirically derived from
    how these components actually co-move in practice.

    Standard approaches for this (OECD Handbook on Constructing Composite
    Indicators) include PCA/Factor Analysis — but naive PCA has a
    documented, well-known flaw: it assigns the HIGHEST weight to
    components with the HIGHEST correlation with others in the set. That
    is the OPPOSITE of what this project wants — we've repeatedly found
    and REMOVED redundant, highly-correlated components this session
    (netflow=inflow-outflow, M81/M82 ETF flow r=-1.000, m2_logit/m5_qreg
    r=0.996) specifically because double-counting correlated signals
    inflates their effective weight beyond what's intended.

    This module instead extends method_independence_analysis.py's own
    logic (already used for M1-M31, this session) DOWN into GLF/SLI/MPI/
    Q10's internal weights: components more correlated with OTHERS in the
    same composite get their weight REDUCED (not increased), and genuinely
    independent components get relatively MORE weight, since they
    contribute unique information the others don't.

ARCHITECTURE (Option A — safe, non-invasive, consistent with this
project's established rollout pattern for new signals):
    This module computes ALTERNATIVE, RECOMMENDED weights alongside the
    EXISTING manual ones — it does NOT automatically replace GLF/SLI/MPI/
    Q10's live weights. Compare the two before deciding whether to adopt
    the recalculated weights.

DATA REQUIREMENT:
    Needs a ROLLING HISTORY of each component's z_score/score (not just
    one snapshot) to compute meaningful correlations — this history did
    not previously exist (GLF/SLI/MPI/Q10 only ever stored the CURRENT
    cycle's breakdown, not accumulated over time). This module builds and
    maintains that rolling cache itself; correlation-based reweighting
    only activates once MIN_POINTS_FOR_CORRELATION cycles have
    accumulated — before that, it reports the existing manual weights
    unchanged (fails safe, same pattern as reflexivity_divergence.py).
"""
import json
import os
import time

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".composite_weight_history.json")
MAX_POINTS = 180  # keep a reasonably long rolling window once accumulated
MIN_POINTS_FOR_CORRELATION = 30  # arbitrary starting threshold, NOT
                                  # validated against real regime-diversity
                                  # data yet — same honest caveat as every
                                  # other new threshold introduced this
                                  # session without live backtesting


def _load_history():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_history(history):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(history, f)
    except OSError:
        pass


def update_composite_history(composite_name, component_values):
    """
    Append this cycle's component values for one composite (e.g. "GLF")
    to its rolling history.

    Args:
        composite_name: e.g. "GLF", "SLI", "MPI", "Q10_whale_pressure"
        component_values: dict of {component_name: numeric_value} for
            THIS cycle (e.g. {"fed": -0.632, "ecb": -1.141, ...} for GLF's
            z_scores, or {"dominance": 0.3, "exchange_flow": 0.64, ...}
            for SLI's scores)
    """
    history = _load_history()
    if composite_name not in history:
        history[composite_name] = []
    history[composite_name].append({"ts": time.time(), **component_values})
    history[composite_name] = history[composite_name][-MAX_POINTS:]
    _save_history(history)
    return history[composite_name]


def _pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def compute_independence_weights(composite_name, original_weights):
    """
    Compute correlation-adjusted ("independence-weighted") alternative
    weights for one composite's sub-components.

    Args:
        composite_name: matches what was used in update_composite_history()
        original_weights: dict {component_name: current_manual_weight},
            e.g. GLF's {"fed": 0.30, "ecb": 0.15, "boj": 0.03, ...}

    Returns:
        (recommended_weights, detail)
        recommended_weights: dict, same keys as original_weights, summing
            to 1.0 — the CORRELATION-ADJUSTED alternative (not yet
            applied live; for comparison only)
        detail: status, points available, per-component independence
            scores and average |correlation| with the rest of the set
    """
    history = _load_history().get(composite_name, [])
    component_names = list(original_weights.keys())

    if len(history) < MIN_POINTS_FOR_CORRELATION:
        return dict(original_weights), {
            "status": "insufficient_history",
            "points_available": len(history),
            "points_needed": MIN_POINTS_FOR_CORRELATION,
            "note": "Reporting existing manual weights unchanged until enough history accumulates.",
        }

    # Build per-component value series from history (skip cycles missing a component)
    series = {}
    for name in component_names:
        vals = [pt[name] for pt in history if name in pt]
        series[name] = vals

    # Pairwise |correlation|, then average |corr| per component vs. all others
    avg_abs_corr = {}
    for name in component_names:
        if len(series[name]) < MIN_POINTS_FOR_CORRELATION:
            avg_abs_corr[name] = 0.0  # not enough data for THIS specific component; treat as independent (conservative)
            continue
        corrs = []
        for other in component_names:
            if other == name:
                continue
            # Align on the shorter common length (simple approach: use
            # last N points where N = min length of the two series)
            n = min(len(series[name]), len(series[other]))
            if n < MIN_POINTS_FOR_CORRELATION:
                continue
            r = _pearson_corr(series[name][-n:], series[other][-n:])
            if r is not None:
                corrs.append(abs(r))
        avg_abs_corr[name] = sum(corrs) / len(corrs) if corrs else 0.0

    # Independence score: 1 - avg|corr| (bounded to [0.05, 1.0] — never
    # let a component's weight go to exactly zero purely from this
    # adjustment; that's a bigger decision than a weight nudge and
    # should be a deliberate removal, like netflow/M81-M82, not an
    # automatic side effect of this formula)
    independence_score = {name: max(0.05, 1.0 - avg_abs_corr[name]) for name in component_names}

    # Redistribute: new_weight_i = original_weight_i * independence_score_i, renormalized
    raw_adjusted = {name: original_weights[name] * independence_score[name] for name in component_names}
    total = sum(raw_adjusted.values())
    recommended = {name: round(v / total, 4) for name, v in raw_adjusted.items()}

    return recommended, {
        "status": "ok",
        "points_used": len(history),
        "avg_abs_correlation": {k: round(v, 3) for k, v in avg_abs_corr.items()},
        "independence_score": {k: round(v, 3) for k, v in independence_score.items()},
        "original_weights": dict(original_weights),
    }


if __name__ == "__main__":
    print("=== Self-test: compute_independence_weights() ===\n")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)

    print("--- Test 1: insufficient history reports original weights unchanged ---")
    original = {"a": 0.5, "b": 0.5}
    for i in range(5):
        update_composite_history("TEST1", {"a": i, "b": i * 2})
    rec, detail = compute_independence_weights("TEST1", original)
    assert detail["status"] == "insufficient_history"
    assert rec == original
    print(f"Status: {detail['status']}, weights unchanged: {rec}")
    print("✅ PASS\n")

    print("--- Test 2: two components PERFECTLY CORRELATED (redundant) + one independent ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    import random
    random.seed(42)
    original2 = {"redundant_a": 0.4, "redundant_b": 0.4, "independent_c": 0.2}
    for i in range(MIN_POINTS_FOR_CORRELATION + 5):
        base = random.gauss(0, 1)
        update_composite_history("TEST2", {
            "redundant_a": base,               # perfectly correlated with b
            "redundant_b": base * 2 + 0.001,   # same underlying signal, different scale
            "independent_c": random.gauss(0, 1),  # unrelated random noise
        })
    rec2, detail2 = compute_independence_weights("TEST2", original2)
    print(f"Original weights:    {original2}")
    print(f"Recommended weights: {rec2}")
    print(f"Avg |corr|: {detail2['avg_abs_correlation']}")
    assert rec2["independent_c"] > original2["independent_c"], "FAIL: independent component should gain relative weight"
    assert rec2["redundant_a"] < original2["redundant_a"], "FAIL: redundant component should lose relative weight"
    print(f"✅ PASS: redundant pair's weight REDUCED ({original2['redundant_a']}->{rec2['redundant_a']}), "
          f"independent component's weight INCREASED ({original2['independent_c']}->{rec2['independent_c']})\n")

    print("--- Test 3: weights always sum to 1.0 ---")
    assert abs(sum(rec2.values()) - 1.0) < 0.001
    print(f"Sum: {sum(rec2.values()):.4f}")
    print("✅ PASS\n")

    print("--- Test 4: all components mutually independent -> weights stay close to original ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    original4 = {"x": 0.33, "y": 0.33, "z": 0.34}
    for i in range(MIN_POINTS_FOR_CORRELATION + 5):
        update_composite_history("TEST4", {
            "x": random.gauss(0, 1), "y": random.gauss(0, 1), "z": random.gauss(0, 1),
        })
    rec4, detail4 = compute_independence_weights("TEST4", original4)
    print(f"Original: {original4}")
    print(f"Recommended (all independent, should be close to original): {rec4}")
    for name in original4:
        assert abs(rec4[name] - original4[name]) < 0.05, f"FAIL: {name} shifted too much when all independent"
    print("✅ PASS: weights stay close to original when nothing is correlated\n")

    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    print("ALL SELF-TESTS PASSED")
