#!/usr/bin/env python3
"""
SFC Dynamic Feature Weighting
==============================
Adjusts 5-factor weights (Lt, St, Rt, Ft, Sc) dynamically based on
HMM regime detection. Different market regimes = different feature importance.

Regime weight matrices (from empirical research + document suggestions):

               Lt     St     Rt     Ft     Sc
  BULL        0.30   0.10   0.10   0.15   0.35
  BEAR        0.25   0.20   0.20   0.25   0.10
  SIDEWAYS    0.20   0.25   0.20   0.15   0.20
  CRISIS      0.15   0.10   0.10   0.40   0.25
  NORMAL      0.25   0.20   0.20   0.20   0.15

Lt = Liquidity/Trend (M2, Fed, TGA, RRP, GLO, DXY, Stablecoin)
St = Structural (Dominance, Put/Call, Concentration)
Rt = Sentiment (Fear & Greed, Whale, News)
Ft = Systemic (DVOL, GARCH, Var, Jump, Skew, Funding)
Sc = External (DXY-BTC correlation gate)

Usage:
    from dynamic_feature_weighting import (
        get_regime_weights,
        apply_dynamic_weights,
        get_feature_group_weights,
    )
    factor_weights = get_regime_weights("BULL")
    adjusted_factors = apply_dynamic_weights(factors, "CRISIS")
"""

import json, os

# ── Regime → Factor Weight Matrices ──
# Rows must sum to 1.0
# Based on empirical research:
#   Crisis: Ft (volatility/risk) dominates
#   Bull: Sc (external/liquidity) + Lt dominate
#   Bear: Ft + Rt (fear) dominate
#   Sideways: St (structural) matters more
#   Normal: balanced

_REGIME_WEIGHTS = {
    "BULL": {
        "name": "Bull Market",
        "weights": {"Lt": 0.30, "St": 0.10, "Rt": 0.10, "Ft": 0.15, "Sc": 0.35},
        "description": "Liquidity-driven rally. Sc (external/liquidity) most important.",
    },
    "BEAR": {
        "name": "Bear Market",
        "weights": {"Lt": 0.25, "St": 0.20, "Rt": 0.20, "Ft": 0.25, "Sc": 0.10},
        "description": "Risk-off. Ft (volatility) + Rt (sentiment) dominate.",
    },
    "SIDEWAYS": {
        "name": "Sideways / Accumulation",
        "weights": {"Lt": 0.20, "St": 0.25, "Rt": 0.20, "Ft": 0.15, "Sc": 0.20},
        "description": "Range-bound. St (structural) reveals direction next.",
    },
    "CRISIS": {
        "name": "Crisis / Capitulation",
        "weights": {"Lt": 0.15, "St": 0.10, "Rt": 0.10, "Ft": 0.40, "Sc": 0.25},
        "description": "Volatility explosion. Ft (systemic risk) dominates.",
    },
    "NORMAL": {
        "name": "Normal Oscillation",
        "weights": {"Lt": 0.25, "St": 0.20, "Rt": 0.20, "Ft": 0.20, "Sc": 0.15},
        "description": "Balanced default weights.",
    },
    "STRESS": {
        "name": "Elevated Stress",
        "weights": {"Lt": 0.25, "St": 0.15, "Rt": 0.15, "Ft": 0.30, "Sc": 0.15},
        "description": "Volatility rising. Ft weight increases.",
    },
    "CAPITULATION": {
        "name": "Capitulation",
        "weights": {"Lt": 0.15, "St": 0.10, "Rt": 0.10, "Ft": 0.40, "Sc": 0.25},
        "description": "Same as CRISIS — maximum Ft weight.",
    },
}

# ── Method Group Weightings (for document suggestion #1 & #5) ──
# These weights change by regime for the method-level scoring
_METHOD_GROUP_WEIGHTS = {
    "BULL":       {"liquidity": 0.45, "stablecoin": 0.30, "onchain": 0.10, "derivatives": 0.10, "technical": 0.05},
    "BEAR":       {"liquidity": 0.20, "stablecoin": 0.15, "onchain": 0.20, "derivatives": 0.35, "technical": 0.10},
    "SIDEWAYS":   {"liquidity": 0.25, "stablecoin": 0.20, "onchain": 0.25, "derivatives": 0.15, "technical": 0.15},
    "CRISIS":     {"liquidity": 0.20, "stablecoin": 0.10, "onchain": 0.10, "derivatives": 0.45, "technical": 0.15},
    "NORMAL":     {"liquidity": 0.35, "stablecoin": 0.20, "onchain": 0.20, "derivatives": 0.15, "technical": 0.10},
}


def get_regime_weights(regime_name="NORMAL"):
    """
    Get 5-factor weights for a given regime.

    Args:
        regime_name: BULL, BEAR, SIDEWAYS, CRISIS, NORMAL, STRESS, CAPITULATION

    Returns:
        dict with weights for Lt, St, Rt, Ft, Sc
        Falls back to NORMAL weights if regime unknown.
    """
    regime = regime_name.upper() if regime_name else "NORMAL"

    if regime in _REGIME_WEIGHTS:
        return dict(_REGIME_WEIGHTS[regime]["weights"])

    # Fuzzy match: strip prefixes
    for key in _REGIME_WEIGHTS:
        if key in regime:
            return dict(_REGIME_WEIGHTS[key]["weights"])

    return dict(_REGIME_WEIGHTS["NORMAL"]["weights"])


def apply_dynamic_weights(factors, regime_name="NORMAL", factor_order=None):
    """
    Apply regime-adaptive weights to the 5 factors.

    Args:
        factors: dict of {Lt, St, Rt, Ft, Sc} with values in [-3, +3]
        regime_name: Market regime label
        factor_order: Optional factor key order for output

    Returns:
        (weighted_factors_dict, z_score, applied_weights)
        weighted_factors: same keys as input with values adjusted
        z_score: weighted sum (for SFC ensemble input)
        applied_weights: dict of weights that were used
    """
    weights = get_regime_weights(regime_name)

    # Normalize factors to [0, 1] range for weighting
    norm = {}
    for k in factors:
        if k in weights:
            # Map [-3, +3] → [0, 1] where 1 = max stress
            norm[k] = max(0.0, min(1.0, (-factors[k] + 3.0) / 6.0))

    # Weighted z-score (for ensemble)
    z_score = sum(norm.get(k, 0.5) * weights.get(k, 0.20) for k in factors if k in weights)

    # Clamp z_score to [0, 1]
    z_score = max(0.0, min(1.0, z_score))

    # Also return original-normalized factors (unweighted)
    return norm, z_score, weights


def get_feature_group_weights(regime_name="NORMAL"):
    """
    Get method group weights for feature families.
    Used for document's suggestion #1: group methods into families.

    Args:
        regime_name: Market regime

    Returns:
        dict of {group_name: weight}
    """
    regime = regime_name.upper() if regime_name else "NORMAL"
    if regime in _METHOD_GROUP_WEIGHTS:
        return dict(_METHOD_GROUP_WEIGHTS[regime])
    return dict(_METHOD_GROUP_WEIGHTS["NORMAL"])


def get_regime_info(regime_name="NORMAL"):
    """
    Get full regime info including weights and description.

    Args:
        regime_name: Market regime

    Returns:
        dict with name, weights, description
    """
    regime = regime_name.upper() if regime_name else "NORMAL"
    if regime in _REGIME_WEIGHTS:
        return dict(_REGIME_WEIGHTS[regime])
    return dict(_REGIME_WEIGHTS["NORMAL"])


def get_sfc_effective_with_dynamic_weights(
    factors,
    sfc_effective_pct,
    regime_name="NORMAL",
):
    """
    Adjust the final SFC effective score based on regime-weighted factors.

    In crisis: amplify SFC (volatility dominates)
    In bull: dampen SFC slightly (liquidity supports)
    In side/accumulation: keep as is

    Args:
        factors: raw factor dict
        sfc_effective_pct: current SFC score (0-100)
        regime_name: detected regime

    Returns:
        (adjusted_sfc_pct, adjustment_pct)
    """
    weights = get_regime_weights(regime_name)

    reg = regime_name.upper() if regime_name else "NORMAL"

    # Dynamic adjustment
    if reg in ("CRISIS", "CAPITULATION"):
        # In crisis, Ft weight is higher → amplify signal
        ft_weight = weights.get("Ft", 0.20)
        if ft_weight > 0.30:
            # Amplify SFC by Ft weight excess
            excess = ft_weight - 0.20
            adj = min(sfc_effective_pct * excess * 0.3, 10.0)
            return min(100.0, sfc_effective_pct + adj), round(adj, 1)
        return sfc_effective_pct, 0.0

    elif reg == "BULL":
        # In strong bull, dampen SFC slightly (liquidity cushion)
        sc_weight = weights.get("Sc", 0.15)
        if sc_weight > 0.25:
            dampen = min(sfc_effective_pct * 0.08, 5.0)
            return max(0.0, sfc_effective_pct - dampen), round(-dampen, 1)
        return sfc_effective_pct, 0.0

    elif reg == "BEAR":
        # In bear, amplify SFC by Rt + Ft weights
        bear_amp = (weights.get("Rt", 0.20) + weights.get("Ft", 0.25) - 0.35) * sfc_effective_pct * 0.2
        bear_amp = max(0, min(bear_amp, 8.0))
        if bear_amp > 0.5:
            return min(100.0, sfc_effective_pct + bear_amp), round(bear_amp, 1)
        return sfc_effective_pct, 0.0

    return sfc_effective_pct, 0.0


if __name__ == "__main__":
    # Test
    import json
    print("=== Regime Weights ===")
    for regime in ["BULL", "BEAR", "SIDEWAYS", "CRISIS", "NORMAL"]:
        w = get_regime_weights(regime)
        info = get_regime_info(regime)
        print(f"{regime}: {info['name']:30s} weights={w}")

    print("\n=== Method Group Weights ===")
    for regime in ["BULL", "BEAR", "SIDEWAYS", "CRISIS", "NORMAL"]:
        gw = get_feature_group_weights(regime)
        print(f"{regime}: {gw}")

    print("\n=== Test apply_dynamic_weights ===")
    factors = {"Lt": 2.0, "St": -0.5, "Rt": -1.0, "Ft": -2.5, "Sc": 1.5}
    for regime in ["NORMAL", "CRISIS", "BULL"]:
        norm, z, w = apply_dynamic_weights(factors, regime)
        print(f"{regime}: z_score={z:.3f}, weights={w}")

    print("\n=== Test SFC adjustment ===")
    for regime in ["NORMAL", "CRISIS", "BULL", "BEAR"]:
        adj_sfc, adj = get_sfc_effective_with_dynamic_weights(factors, 35.0, regime)
        print(f"{regime}: {35.0} -> {adj_sfc:.1f} (adj={adj:+.1f}pp)")
