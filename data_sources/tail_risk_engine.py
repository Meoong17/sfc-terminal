#!/usr/bin/env python3
"""
SFC Tail Risk Engine (L8) — Extreme-Risk Composite
==================================================
Layer 8 of the IMBS / Macro-Intelligence blueprint. Tail risk is NOT just
volatility — it is the NON-LINEAR CO-OCCURRENCE of several independent
stress dimensions. This module computes a single tail-risk score by
combining Layer 2-4 stress inputs.

FORMULA (blueprint IMBS-design.md, with blockers RESOLVED):
    TailRisk = LiquidityStress  × BehaviorStress × ExpectationShock
             × Leverage         × CorrelationFactor

WHY THIS DESIGN (resolves the three IMBS blockers flagged in IMBS-design.md):
    1. INPUT SOURCE is explicit. This module takes each dimension as a
       SEPARATE named parameter (liquidity_stress, behavior_stress,
       expectation_shock, leverage, correlation). The CALLER (collect.py)
       decides which live layer-2/3/4 signal feeds each slot — no
       ambiguity about "does Tail Risk come from Layer 5 or Layer 2-4?".
    2. LEVERAGE is an explicit parameter, fed by the caller from a real
       leverage source (e.g. funding-rate / OI-derived MPI leverage or
       cascade exposure). It is NOT silently dropped from the formula.
    3. NORMALIZATION is uniform. Every input is clamped to [0,100] before
       combination, so no unit mismatch (liquidity index vs probability vs
       a possibly-negative gap). See _norm().

COMBINATION RULE (multiplicative, not additive):
    The blueprint says tail risk spikes only when MANY dimensions are high
    simultaneously. A pure product would collapse to ~0 whenever even one
    input is near-neutral, so we use a GEOMETRIC-MEAN-style combination:
        geometric = (prod(inputs/100))^(1/N) * 100
    which rewards ALL-HIGH co-occurrence but does not zero out on a single
    low dimension. This is the defensible reading of "combination
    non-linear: jika semuanya tinggi bersamaan, probabilitas kejadian
    ekstrem meningkat tajam."

DISPLAY-ONLY:
    Exposed as its own field. NOT blended into sfc_effective/signal/
    composite_confidence until walk-forward re-validation shows a stable
    edge (cautious-rollout pattern of M86/M90/reflexivity).

Usage:
    from data_sources.tail_risk_engine import compute_tail_risk
    score, details = compute_tail_risk(
        liquidity_stress=0.42,   # 0-1 or 0-100
        behavior_stress=0.30,
        expectation_shock=53.8,
        leverage=0.50,
        correlation=0.20,
    )
All inputs are auto-normalized; supply 0-1 OR 0-100, doesn't matter.
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".tail_risk_cache.json")
CACHE_TTL = 900  # 15 min — tail risk inputs move at intraday frequency


def _norm(v):
    """Normalize an input to [0,100]. Accepts 0-1 or 0-100 ranges,
    clamped. None -> 50 (neutral, so a missing component does not zero
    the product or inflate it)."""
    if v is None:
        return 50.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 50.0
    if f > 1.0:  # assume already 0-100
        return max(0.0, min(100.0, f))
    return max(0.0, min(100.0, f * 100.0))


def _severity(x):
    """Map 0-100 score to severity label."""
    if x >= 80:
        return "CRITICAL"
    if x >= 60:
        return "HIGH"
    if x >= 40:
        return "ELEVATED"
    if x >= 25:
        return "MODERATE"
    return "LOW"


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(state):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def compute_tail_risk(liquidity_stress=None, behavior_stress=None,
                      expectation_shock=None, leverage=None,
                      correlation=None):
    """Compute the L8 Tail Risk composite.

    Args (all optional, auto-normalized to 0-100):
        liquidity_stress  : Layer-2/3 liquidity stress (0-1 or 0-100).
        behavior_stress   : Layer-3/4 behavior/positioning stress.
        expectation_shock : L6 expectation stress (0-100, from
                            expectations_engine.gap_score).
        leverage          : explicit leverage stress (funding/OI/cascade).
        correlation       : cross-asset correlation stress (0-1 or 0-100).

    Returns:
        (tail_risk_score 0-100, details dict)
    """
    cached = _load_cache()
    now = time.time()
    # Cache is keyed by the (normalized) input tuple so different live
    # inputs never collide into a stale result for a different input.
    _norm_inputs = (
        _norm(liquidity_stress), _norm(behavior_stress),
        _norm(expectation_shock), _norm(leverage), _norm(correlation),
    )
    _cache_key = ",".join(f"{x:.2f}" for x in _norm_inputs)
    if (cached.get("key") == _cache_key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("score", 50.0), cached.get("details", {"status": "cached"})

    # ── Normalize all inputs to [0,100] ──
    dims = {
        "liquidity_stress": _norm_inputs[0],
        "behavior_stress": _norm_inputs[1],
        "expectation_shock": _norm_inputs[2],
        "leverage": _norm_inputs[3],
        "correlation": _norm_inputs[4],
    }
    # Track which dimensions the caller actually supplied vs defaulted.
    supplied = {
        "liquidity_stress": liquidity_stress is not None,
        "behavior_stress": behavior_stress is not None,
        "expectation_shock": expectation_shock is not None,
        "leverage": leverage is not None,
        "correlation": correlation is not None,
    }
    active_dims = [k for k in dims if supplied[k]]
    if not active_dims:
        score, status = 50.0, "partial"
        details = {
            "status": status, "available": False,
            "score": score,
            "reason": "No input dimensions supplied — cannot compute tail risk.",
        }
        _save_cache({"score": score, "details": details, "ts": now, "key": _cache_key})
        return score, details

    # ── Geometric-mean combination (see docstring) ──
    prod = 1.0
    for k in dims:
        prod *= dims[k] / 100.0
    n = len(dims)
    geometric = (prod ** (1.0 / n)) * 100.0

    # ── Leverage / correlation "amplifier" tilt ──
    # When leverage AND correlation are BOTH elevated, tail risk
    # accelerates beyond the geometric mean (forced selling cascades).
    # Apply a modest multiplicative tilt, capped so it can't blow past 100.
    lev, corr = dims["leverage"], dims["correlation"]
    amp = 1.0 + 0.25 * (lev / 100.0) * (corr / 100.0) * (lev >= 60 and corr >= 60)
    score = min(100.0, geometric * amp)

    details = {
        "status": "ok",
        "available": True,
        "score": round(score, 1),
        "severity": _severity(score),
        "dimensions": {k: round(v, 1) for k, v in dims.items()},
        "active_dimensions": active_dims,
        "missing_dimensions": [k for k in dims if not supplied[k]],
        "combination": "geometric-mean (multiplicative, all-high co-occurrence)",
        "method": "TailRisk = geometric_mean(Liquidity, Behavior, Expectation, Leverage, Correlation) * amplifier",
        "caveat": "Display-only proxy. Thresholds provisional (not walk-forward validated). "
                  "NOT blended into sfc_effective/signal.",
        "ts": now,
    }
    _save_cache({"score": score, "details": details, "ts": now, "key": _cache_key})
    return round(score, 1), details


if __name__ == "__main__":
    # Self-test with representative scenarios.
    import sys
    cases = [
        ("all-neutral", 30, 30, 50, 30, 30),
        ("mild stress", 50, 45, 55, 45, 40),
        ("high co-occurrence (crisis)", 85, 80, 75, 80, 75),
        ("liquidity only high", 90, 30, 30, 30, 30),
    ]
    for name, liq, beh, exp, lev, corr in cases:
        s, d = compute_tail_risk(liquidity_stress=liq, behavior_stress=beh,
                                 expectation_shock=exp, leverage=lev, correlation=corr)
        print(f"{name:28s} score={s:5.1f} severity={d['severity']}")
