#!/usr/bin/env python3
"""
SFC Dynamic Feature Selector
=============================
Selects relevant feature subsets based on market regime.

Instead of feeding ALL features to QLSTM, dynamically choose only
the features most relevant for the current regime:

  BULL     → liquidity + stablecoin features (liquidity drives rallies)
  BEAR     → macro + derivatives features (risk-off dominates)
  SIDEWAYS → on-chain + volume features (accumulation/distribution)
  CRISIS   → volatility + liquidity features (survival mode)
  NORMAL   → all feature groups (balanced)

This reduces noise, improves QLSTM convergence, and matches
how professional trading desks operate — different regimes
need different information sets.

Usage:
    from dynamic_feature_selector import DynamicFeatureSelector
    
    selector = DynamicFeatureSelector()
    selected_groups = selector.select(regime="BULL")
    # Returns: ["global_liquidity", "stablecoin", ...]
    
    feature_map = selector.get_feature_list(regime="BULL")
    # Returns: ["glf_score", "sli_score", "fed_yoy", ...]
"""

import json

# ── Feature Group Definitions ──
# Each group has:
#   - module: where to get the data
#   - features: list of field names in the output dict
#   - importance_by_regime: dict of regime → weight (0-1)

_FEATURE_GROUPS = {
    "global_liquidity": {
        "name": "Global Liquidity",
        "module": "global_liquidity_engine",
        "features": [
            "glf_score", "glf_stress", "glf_regime",
            "fed_yoy", "ecb_yoy", "m2_yoy", "dxy",
            "tga_score", "rrp_score",
        ],
        "importance": {
            "BULL": 0.95,
            "BEAR": 0.70,
            "SIDEWAYS": 0.60,
            "CRISIS": 0.90,
            "NORMAL": 0.80,
        },
        "description": "Central bank balance sheets, M2, TGA, RRP, DXY",
    },
    "stablecoin": {
        "name": "Stablecoin Liquidity",
        "module": "stablecoin_intelligence",
        "features": [
            "sli_score", "sli_stress", "sli_label",
            "usdt_supply_growth", "usdc_supply_growth",
            "stablecoin_dominance", "exchange_netflow",
        ],
        "importance": {
            "BULL": 0.90,
            "BEAR": 0.50,
            "SIDEWAYS": 0.60,
            "CRISIS": 0.70,
            "NORMAL": 0.70,
        },
        "description": "Stablecoin supply, velocity, exchange flows, mint/burn",
    },
    "onchain": {
        "name": "On-Chain",
        "module": "onchain_fetch",
        "features": [
            "whale_pressure", "onchain_value", "buying_power",
            "sopr_proxy", "sopr_signal",
            "exchange_reserve", "mvrv_proxy",
        ],
        "importance": {
            "BULL": 0.50,
            "BEAR": 0.80,
            "SIDEWAYS": 0.90,
            "CRISIS": 0.40,
            "NORMAL": 0.65,
        },
        "description": "SOPR, whale activity, exchange reserves, MVRV",
    },
    "derivatives": {
        "name": "Derivatives",
        "module": "market_positioning_index",
        "features": [
            "mpi_score", "mpi_stress", "mpi_label",
            "funding_rate", "pc_oi", "basis",
            "liquidation_volume", "open_interest",
        ],
        "importance": {
            "BULL": 0.40,
            "BEAR": 0.90,
            "SIDEWAYS": 0.60,
            "CRISIS": 0.85,
            "NORMAL": 0.55,
        },
        "description": "Funding rate, put/call OI, basis, liquidation cascade",
    },
    "volatility": {
        "name": "Volatility / Risk",
        "module": "feature_engineering + collect",
        "features": [
            "dvol", "atr", "bb_width", "realized_vol",
            "garch_vol", "var_95", "jump_risk",
        ],
        "importance": {
            "BULL": 0.30,
            "BEAR": 0.75,
            "SIDEWAYS": 0.40,
            "CRISIS": 0.95,
            "NORMAL": 0.50,
        },
        "description": "DVOL, ATR, Bollinger width, GARCH, VaR",
    },
    "technical": {
        "name": "Technical",
        "module": "feature_engineering",
        "features": [
            "rsi_14", "rsi_7", "stoch_k", "stoch_d",
            "macd_line", "macd_histogram",
            "ema_crossover", "ema200_slope",
            "vwap", "obv", "cmf",
        ],
        "importance": {
            "BULL": 0.25,
            "BEAR": 0.30,
            "SIDEWAYS": 0.50,
            "CRISIS": 0.20,
            "NORMAL": 0.40,
        },
        "description": "RSI, MACD, EMAs, VWAP, OBV, CMF (16% weight)",
    },
    "macro": {
        "name": "Macro / External",
        "module": "methods_institutional + collect",
        "features": [
            "m2_yoy", "dxy", "dxy_btc_corr",
            "m7_fisher", "m8_yield", "m9_liquidity",
            "fiscal_composite",
            "m75_liquidity_composite",
        ],
        "importance": {
            "BULL": 0.40,
            "BEAR": 0.85,
            "SIDEWAYS": 0.55,
            "CRISIS": 0.60,
            "NORMAL": 0.55,
        },
        "description": "Fisher rates, yield curve, M2 multiplier, fiscal stance",
    },
}

# ── Active group selection thresholds ──
# A group is included if its importance >= threshold for that regime
_SELECTION_THRESHOLD = {
    "BULL": 0.35,
    "BEAR": 0.50,
    "SIDEWAYS": 0.45,
    "CRISIS": 0.40,
    "NORMAL": 0.45,
}


class DynamicFeatureSelector:
    """
    Select feature groups based on market regime.

    In BULL: focus on liquidity + stablecoin (drivers of rallies)
    In BEAR: focus on macro + derivatives + on-chain (risk-off signals)
    In CRISIS: focus on volatility + liquidity (survival)
    In SIDEWAYS: focus on on-chain + technical (accumulation signals)
    In NORMAL: balanced mix
    """

    def __init__(self):
        self.groups = _FEATURE_GROUPS
        self.thresholds = _SELECTION_THRESHOLD

    def select(self, regime="NORMAL"):
        """
        Select active feature groups for a given regime.

        Args:
            regime: BULL, BEAR, SIDEWAYS, CRISIS, or NORMAL

        Returns:
            list of group keys that pass the importance threshold
        """
        regime = regime.upper() if regime else "NORMAL"
        if regime not in self.thresholds:
            regime = "NORMAL"

        threshold = self.thresholds[regime]
        selected = []

        for key, group in self.groups.items():
            importance = group["importance"].get(regime, 0.5)
            if importance >= threshold:
                selected.append(key)

        return selected

    def get_feature_list(self, regime="NORMAL"):
        """
        Get flat list of all feature names for a regime.

        Args:
            regime: Market regime

        Returns:
            list of feature names (strings)
        """
        selected = self.select(regime)
        features = []
        for key in selected:
            group = self.groups.get(key, {})
            features.extend(group.get("features", []))
        return features

    def get_group_weights(self, regime="NORMAL"):
        """
        Get normalized weights for each selected group.

        Returns dict of {group_key: weight} where weights sum to 1.0.
        """
        regime = regime.upper() if regime else "NORMAL"
        if regime not in self.thresholds:
            regime = "NORMAL"

        selected = self.select(regime)
        raw_weights = {}
        for key in selected:
            group = self.groups.get(key, {})
            raw_weights[key] = group["importance"].get(regime, 0.5)

        total = sum(raw_weights.values())
        if total == 0:
            return {k: 1.0 / len(raw_weights) for k in raw_weights} if raw_weights else {}

        return {k: round(v / total, 3) for k, v in raw_weights.items()}

    def summarize(self, regime="NORMAL"):
        """
        Get a human-readable summary of feature selection for a regime.

        Returns:
            dict with regime, selected groups, feature count, weights
        """
        selected = self.select(regime)
        weights = self.get_group_weights(regime)

        feature_count = 0
        groups_detail = []
        for key in selected:
            group = self.groups.get(key, {})
            n_feat = len(group.get("features", []))
            feature_count += n_feat
            groups_detail.append({
                "key": key,
                "name": group.get("name", key),
                "features": n_feat,
                "weight": weights.get(key, 0),
                "description": group.get("description", ""),
            })

        return {
            "regime": regime,
            "selected_groups": selected,
            "n_groups": len(selected),
            "total_features": feature_count,
            "groups": groups_detail,
        }

    def get_regime_profile(self, regime="NORMAL"):
        """
        Get the complete regime profile: which groups are active, their
        weights, and which features to pass to QLSTM.

        This is the main API for integration with collect.py.
        """
        regime = regime.upper() if regime else "NORMAL"
        summary = self.summarize(regime)
        feature_list = self.get_feature_list(regime)
        group_weights = self.get_group_weights(regime)

        return {
            "regime": regime,
            "active_groups": summary["selected_groups"],
            "n_groups": summary["n_groups"],
            "n_features": summary["total_features"],
            "feature_list": feature_list,
            "group_weights": group_weights,
            "groups_detail": summary["groups"],
        }


# ════════════════════════════════════════════
# Convenience function
# ════════════════════════════════════════════


def select_features_for_regime(regime="NORMAL"):
    """
    One-shot: get feature selection profile for a regime.
    """
    selector = DynamicFeatureSelector()
    return selector.get_regime_profile(regime)


if __name__ == "__main__":
    import json
    selector = DynamicFeatureSelector()

    print("=== Dynamic Feature Selection by Regime ===\n")
    for regime in ["BULL", "BEAR", "SIDEWAYS", "CRISIS", "NORMAL"]:
        profile = selector.get_regime_profile(regime)
        print(f"{regime:12s} | {profile['n_groups']} groups, {profile['n_features']} features | "
              f"Active: {', '.join(profile['active_groups'])}")

    print()
    print("=== BULL Profile (detail) ===")
    print(json.dumps(select_features_for_regime("BULL"), indent=2))
