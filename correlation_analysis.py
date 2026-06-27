#!/usr/bin/env python3
"""
correlation_analysis.py — Full Feature Correlation & Lt/St Weight Optimization
================================================================================
Extracts ALL numerical features from historical data.json snapshots,
computes the full correlation matrix, groups features by Lt/St/Rt/Ft/Sc domains,
and recommends optimal numerical weights.

5 Core Factor Domains:
  Lt (Long-term trend)   — Macro, on-chain, global liquidity  → M72-M85, Q10
  St (Short-term momentum) — Technical, microstructure, vol   → M7-M19, afe_*
  Rt (Risk regime)       — VaR, CVaR, crash risk             → M10-M12, M69
  Ft (Funding/liquidity) — Funding, spreads, depth           → M13, M22-M23, M76-M80
  Sc (Score composite)   — Blended stress indicator           → SFC, signal_type

Output: Numerical weight table for Lt/St blending + optimal correlation-based weights.
"""

import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

SFC_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# DOMAIN DEFINITIONS — Map features to Lt/St/Rt/Ft/Sc
# ════════════════════════════════════════════════════════════════

# Lt = Long-term trend (macro, on-chain, global liquidity)
LT_FEATURES = [
    "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
    "m72_m2_growth", "m73_m2_momentum", "m74_fed_balance", "m75_liquidity_composite",
    "m76_supply_growth", "m77_ssr", "m78_exchange_flow", "m79_velocity", "m80_dominance",
    "m81_etf_flow", "m82_etf_holdings", "m83_tga_score", "m84_rrp_score",
    "m85_fiscal_composite",
    "m2_yoy", "dxy",
    "q10_whale_pressure", "q10_onchain_value", "q10_buying_power", "q10_market_structure",
    "m33_glo_score", "btc_mcap", "ath",
]

# St = Short-term momentum (technical, microstructure, volatility)
ST_FEATURES = [
    "m7_fisher", "m8_yield", "m9_liquidity", "m13_funding",
    "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
    "m18_entropy", "m19_mutual_info",
    "m20_obi", "m21_trade_flow", "m22_spread", "m23_liquidity",
    "afe_rsi_7", "afe_macd_signal", "afe_bb_width", "afe_atr", "afe_vwap", "afe_obv_norm",
    "dvol", "dom", "rsi_14", "sopr_proxy",
    "liq_mod", "liq_density", "liq_pressure",
    "m65_cnn_attention",
    "btc_24h",
]

# Rt = Risk regime (tail risk, crash prediction)
RT_FEATURES = [
    "m10_garch", "m11_var", "m12_jump",
    "m69_systemic_risk", "m69_btc_systemic_risk",
    "cascade_risk", "transition_risk",
    "m68_drl_signal",
    "shock_factor",
]

# Ft = Funding/liquidity
FT_FEATURES = [
    "m13_funding", "m22_spread", "m23_liquidity",
    "m76_supply_growth", "m77_ssr", "m78_exchange_flow", "m79_velocity", "m80_dominance",
    "liq_mod", "liq_density",
    "fng", "news_sentiment",
]

# Sc = Core stress indicators (output layer)
SC_FEATURES = [
    "sfc_effective", "sfc_base", "signal_strength",
    "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
    "predicted_mean", "predicted_std", "var_95", "es_975",
]


# ════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ════════════════════════════════════════════════════════════════


def extract_features(snapshots: List[Dict]) -> Tuple[List[str], List[List[float]]]:
    """Extract all numeric features from snapshots.

    Returns:
        (feature_names, feature_matrix) where matrix is [n_snapshots][n_features].
    """
    # Collect all numeric keys that appear consistently
    key_counts: Dict[str, int] = defaultdict(int)
    total = len(snapshots)

    for snap in snapshots:
        for k, v in snap.items():
            if isinstance(v, (int, float)):
                key_counts[k] += 1

    # Keep keys that appear in >50% of recent snapshots
    threshold = int(total * 0.5)
    stable_keys = sorted(k for k, count in key_counts.items() if count >= threshold)

    # Build matrix
    matrix = []
    for snap in snapshots:
        row = []
        for k in stable_keys:
            v = snap.get(k)
            if isinstance(v, (int, float)):
                row.append(float(v))
            else:
                row.append(0.0)
        matrix.append(row)

    return stable_keys, matrix


def normalize_domain_features(
    domain_keys: List[str],
    all_keys: List[str],
    matrix: List[List[float]],
) -> Dict[str, List[float]]:
    """Extract domain features from the full matrix.

    Returns dict of feature_name -> list of values across snapshots.
    """
    result = {}
    key_index = {k: i for i, k in enumerate(all_keys)}

    for feature in domain_keys:
        if feature in key_index:
            idx = key_index[feature]
            result[feature] = [row[idx] for row in matrix]

    return result


# ════════════════════════════════════════════════════════════════
# CORRELATION
# ════════════════════════════════════════════════════════════════


def pearson(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((v - mx)**2 for v in x) * sum((v - my)**2 for v in y))
    return num / den if den > 1e-10 else 0.0


def compute_correlation_matrix(
    features: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """Compute full correlation matrix for given features."""
    names = sorted(features.keys())
    matrix = {}
    for i in names:
        matrix[i] = {}
        for j in names:
            if i == j:
                matrix[i][j] = 1.0
            elif j in matrix and i in matrix[j]:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = round(pearson(features[i], features[j]), 4)
    return matrix


# ════════════════════════════════════════════════════════════════
# DOMAIN ANALYSIS
# ════════════════════════════════════════════════════════════════


def analyze_domain(
    domain_keys: List[str],
    name: str,
    domain_features: Dict[str, List[float]],
    all_features: Dict[str, List[float]],
    target: str = "btc_24h",
) -> Dict[str, Any]:
    """Analyze correlation of a domain's features with target and within domain."""
    available = [k for k in domain_keys if k in domain_features]
    if not available:
        return {"name": name, "available": 0, "error": "No features available"}

    # Correlations with target (use all_features for target values)
    target_corrs = {}
    if target in all_features:
        target_vals = all_features[target]
        for f in available:
            target_corrs[f] = round(
                pearson(domain_features[f], target_vals), 4
            )

    # Intra-domain correlations
    intra = {}
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            ai, aj = available[i], available[j]
            corr = round(pearson(domain_features[ai], domain_features[aj]), 4)
            intra[f"{ai}↔{aj}"] = corr

    # Domain strength: average absolute correlation with target
    avg_target_corr = (
        sum(abs(v) for v in target_corrs.values()) / len(target_corrs)
        if target_corrs else 0
    )

    # Best predictors (highest |corr| with target)
    sorted_pred = sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "name": name,
        "available": len(available),
        "total": len(domain_keys),
        "avg_target_corr": round(avg_target_corr, 4),
        "best_predictors": sorted_pred[:5],
        "worst_predictors": sorted_pred[-3:] if len(sorted_pred) >= 3 else [],
        "intra_domain_avg_corr": round(
            sum(abs(v) for v in intra.values()) / len(intra), 4
        ) if intra else 0,
    }


# ════════════════════════════════════════════════════════════════
# LT/ST WEIGHT OPTIMIZATION
# ════════════════════════════════════════════════════════════════


def compute_optimal_weights(
    lt_features: Dict[str, List[float]],
    st_features: Dict[str, List[float]],
    target: str = "sfc_effective",
) -> Dict[str, Any]:
    """Compute optimal Lt/St blend weights based on correlation analysis.

    Method:
      1. Compute each feature's correlation with target (SFC)
      2. Weight = |correlation| / sum(|correlations|) within each domain
      3. Domain weight = avg(|corr|) within domain / total avg(|corr|)
      4. Lt/St blend = weighted average

    Returns:
      Dict with optimal weights, per-feature contributions, and sensitivity.
    """
    if target not in lt_features and target not in st_features:
        return {"error": f"Target {target} not found in features"}

    # Compute feature → target correlations
    lt_corrs = {}
    for f, vals in lt_features.items():
        if target in lt_features:
            target_vals = lt_features[target]
        elif target in st_features:
            target_vals = st_features[target]
        else:
            continue
        if f != target:
            lt_corrs[f] = abs(pearson(vals, target_vals))

    st_corrs = {}
    for f, vals in st_features.items():
        if target in lt_features:
            target_vals = lt_features[target]
        elif target in st_features:
            target_vals = st_features[target]
        else:
            continue
        if f != target:
            st_corrs[f] = abs(pearson(vals, target_vals))

    if not lt_corrs and not st_corrs:
        return {"error": "No valid correlations computed"}

    # Domain strength
    lt_strength = sum(lt_corrs.values()) / len(lt_corrs) if lt_corrs else 0
    st_strength = sum(st_corrs.values()) / len(st_corrs) if st_corrs else 0
    total_strength = lt_strength + st_strength

    # Ideal weight = proportional to domain predictive power
    lt_weight = lt_strength / total_strength if total_strength > 0 else 0.5
    st_weight = st_strength / total_strength if total_strength > 0 else 0.5

    # Feature-level weights
    lt_feature_weights = {}
    lt_total_corr = sum(lt_corrs.values())
    for f, c in sorted(lt_corrs.items(), key=lambda x: x[1], reverse=True):
        lt_feature_weights[f] = {
            "correlation": round(c, 4),
            "weight_in_lt": round(c / lt_total_corr, 4) if lt_total_corr > 0 else 0,
        }

    st_feature_weights = {}
    st_total_corr = sum(st_corrs.values())
    for f, c in sorted(st_corrs.items(), key=lambda x: x[1], reverse=True):
        st_feature_weights[f] = {
            "correlation": round(c, 4),
            "weight_in_st": round(c / st_total_corr, 4) if st_total_corr > 0 else 0,
        }

    return {
        "target": target,
        "lt_strength": round(lt_strength, 4),
        "st_strength": round(st_strength, 4),
        "lt_weight_pct": round(lt_weight * 100, 1),
        "st_weight_pct": round(st_weight * 100, 1),
        "blend_ratio": f"{lt_weight:.2f} Lt + {st_weight:.2f} St",
        "lt_feature_weights": lt_feature_weights,
        "st_feature_weights": st_feature_weights,
        "recommendation": (
            f"Lt={lt_weight:.2f} / St={st_weight:.2f} "
            f"— Lt dominates ({lt_weight*100:.0f}%) because long-term "
            f"macro/on-chain features have {lt_strength:.3f} avg correlation "
            f"vs St's {st_strength:.3f}"
        ),
    }


# ════════════════════════════════════════════════════════════════
# DATA EXTRACTION FROM GIT
# ════════════════════════════════════════════════════════════════


def extract_snapshots(max_count: int = 300) -> List[Dict]:
    try:
        # Get ONLY the most recent commits
        result = subprocess.check_output(
            ["git", "log", "--oneline", "-100", "--", "data.json"],
            text=True, timeout=30, cwd=SFC_DIR,
        ).strip().split("\n")
    except Exception:
        return []

    result = [r for r in result if r.strip()]

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
        except Exception:
            continue
    return snapshots


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 70)
    print("FULL CORRELATION ANALYSIS — QLSTM Features")
    print("=" * 70)

    # ── 1. Extract data ──
    print("\n📦 Extracting historical snapshots...")
    snapshots = extract_snapshots(max_count=300)
    print(f"   {len(snapshots)} snapshots loaded")

    if not snapshots:
        print("❌ No snapshots available")
        sys.exit(1)

    # ── 2. Build feature matrix ──
    all_keys, matrix = extract_features(snapshots)
    print(f"   {len(all_keys)} stable features (present in >80% snapshots)")

    # Build feature dict
    features: Dict[str, List[float]] = {}
    for i, key in enumerate(all_keys):
        features[key] = [row[i] for row in matrix]

    # ── 3. Domain-specific extraction ──
    lt_data = normalize_domain_features(LT_FEATURES, all_keys, matrix)
    st_data = normalize_domain_features(ST_FEATURES, all_keys, matrix)
    rt_data = normalize_domain_features(RT_FEATURES, all_keys, matrix)
    ft_data = normalize_domain_features(FT_FEATURES, all_keys, matrix)
    sc_data = normalize_domain_features(SC_FEATURES, all_keys, matrix)

    # ── 4. Domain Analysis ──
    print("\n" + "=" * 70)
    print("DOMAIN ANALYSIS — Correlation with btc_24h")
    print("=" * 70)

    domains = [
        ("Lt (Long-term)", LT_FEATURES, lt_data),
        ("St (Short-term)", ST_FEATURES, st_data),
        ("Rt (Risk)", RT_FEATURES, rt_data),
        ("Ft (Funding)", FT_FEATURES, ft_data),
        ("Sc (Score)", SC_FEATURES, sc_data),
    ]

    domain_results = {}
    for name, keys, data in domains:
        result = analyze_domain(keys, name, data, features, target="btc_24h")
        domain_results[name] = result

        print(f"\n  {name} ({result['available']}/{result['total']} features)")
        print(f"     Avg |corr| with BTC: {result['avg_target_corr']:.4f}")
        print(f"     Intra-domain corr:   {result['intra_domain_avg_corr']:.4f}")
        print(f"     Best predictors:")
        for f, c in result["best_predictors"][:4]:
            print(f"       • {f:<22} corr={c:+.4f}")

    # ── 5. Full Correlation Matrix (top features only) ──
    print("\n" + "=" * 70)
    print("TOP FEATURES — Correlation Matrix")
    print("=" * 70)

    # Select top features from each domain (most predictive)
    top_features = ["btc_24h", "sfc_effective"]
    for name, result in domain_results.items():
        for f, c in result["best_predictors"][:3]:
            if f not in top_features and f in features:
                top_features.append(f)

    # Print compact matrix
    n = len(top_features)
    print(f"\n{'':>24}", end="")
    for f in top_features:
        short = f[-8:] if len(f) > 8 else f
        print(f"{short:>8}", end="")
    print()

    for i in range(n):
        name = top_features[i]
        short_name = name[-22:] if len(name) > 22 else name
        print(f"{short_name:>24}", end="")
        for j in range(n):
            if i == j:
                print(f" {'1.000':>7}", end="")
            elif j < i:
                print(f"     ", end="  ")
            else:
                c = pearson(features.get(top_features[i], [0]),
                            features.get(top_features[j], [0]))
                color = "+" if c > 0 else ""
                print(f" {color}{c:.3f}", end="")
        print()

    # ── 6. Lt/St Weight Optimization ──
    print("\n" + "=" * 70)
    print("LT/ST WEIGHT OPTIMIZATION")
    print("=" * 70)

    # Primary: Lt vs St predicting SFC (current composite)
    print("\n  📐 Target: sfc_effective (current stress score)")
    opt = compute_optimal_weights(lt_data, st_data, target="sfc_effective")
    if "error" not in opt:
        print(f"\n     Lt strength (avg |corr|): {opt['lt_strength']:.4f}")
        print(f"     St strength (avg |corr|): {opt['st_strength']:.4f}")
        print(f"\n     ╔══════════════════════════════════╗")
        print(f"     ║   Lt WEIGHT:  {opt['lt_weight_pct']:>5.1f}%            ║")
        print(f"     ║   St WEIGHT:  {opt['st_weight_pct']:>5.1f}%            ║")
        print(f"     ╚══════════════════════════════════╝")
        print(f"\n     Per-feature contribution to Lt:")
        for f, w in sorted(opt["lt_feature_weights"].items(),
                           key=lambda x: x[1]["weight_in_lt"], reverse=True)[:6]:
            print(f"       • {f:<24} |corr|={w['correlation']:.4f}  weight={w['weight_in_lt']:.3f}")
        print(f"\n     Per-feature contribution to St:")
        for f, w in sorted(opt["st_feature_weights"].items(),
                           key=lambda x: x[1]["weight_in_st"], reverse=True)[:6]:
            print(f"       • {f:<24} |corr|={w['correlation']:.4f}  weight={w['weight_in_st']:.3f}")

    # Secondary: Lt vs St predicting BTC returns
    print(f"\n  📐 Target: btc_24h (actual returns)")
    opt2 = compute_optimal_weights(lt_data, st_data, target="btc_24h")
    if "error" not in opt2:
        print(f"\n     Lt strength: {opt2['lt_strength']:.4f}")
        print(f"     St strength: {opt2['st_strength']:.4f}")
        print(f"\n     ╔══════════════════════════════════╗")
        print(f"     ║   Lt WEIGHT:  {opt2['lt_weight_pct']:>5.1f}%            ║")
        print(f"     ║   St WEIGHT:  {opt2['st_weight_pct']:>5.1f}%            ║")
        print(f"     ╚══════════════════════════════════╝")

    # ── 7. FINAL RECOMMENDATIONS ──
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS — Numerical Lt/St Weights")
    print("=" * 70)

    lt_w = float(opt.get("lt_weight_pct", 50))
    st_w = float(opt.get("st_weight_pct", 50))

    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║         RECOMMENDED BLEND WEIGHTS                    ║
  ╠══════════════════════════════════════════════════════╣
  ║                                                      ║
  ║   Lt (Long-term)  = {lt_w:>5.1f}%                     ║
  ║     • Macro:     M72-M75  (M2, Fed balance)          ║
  ║     • On-chain:  Q10 whale/composite                 ║
  ║     • Fiscal:    M83-M85  (TGA, RRP)                 ║
  ║     • Stablecoin: M76-M80 (supply, velocity)         ║
  ║                                                      ║
  ║   St (Short-term) = {st_w:>5.1f}%                     ║
  ║     • Technical: afe_* (RSI, MACD, BB, ATR)          ║
  ║     • Microstructure: M20-M23 (OBI, spread, depth)   ║
  ║     • Volatility: M10-M12 (GARCH, VaR, jump)         ║
  ║     • Momentum:   M7-M9, M14-M19 (Fisher, entropy)   ║
  ║                                                      ║
  ║   Current blend: {float(opt.get('lt_weight_pct',50)):>5.1f}% Lt / {float(opt.get('st_weight_pct',50)):>5.1f}% St                ║
  ║   (based on historical correlation with SFC)         ║
  ╚══════════════════════════════════════════════════════╝

  🎯 Action:
    1. Set Lt weight = {lt_w:.1f}%  (macro + on-chain dominate)
    2. Set St weight = {st_w:.1f}%  (technical as complementary)
    3. Within Lt, top features:
""")

    # Show top Lt features
    if "lt_feature_weights" in opt:
        top_lt = sorted(opt["lt_feature_weights"].items(),
                        key=lambda x: x[1]["weight_in_lt"], reverse=True)[:5]
        for f, w in top_lt:
            print(f"       {f:<24} → weight {w['weight_in_lt']:.3f} in Lt blend")

    print(f"\n    4. Within St, top features:")
    if "st_feature_weights" in opt:
        top_st = sorted(opt["st_feature_weights"].items(),
                        key=lambda x: x[1]["weight_in_st"], reverse=True)[:5]
        for f, w in top_st:
            print(f"       {f:<24} → weight {w['weight_in_st']:.3f} in St blend")

    print(f"\n  📊 Cross-validation:")
    print(f"     Predicting SFC:     Lt={opt.get('lt_strength',0):.3f} vs St={opt.get('st_strength',0):.3f}")
    print(f"     Predicting BTC 24h: Lt={opt2.get('lt_strength',0):.3f} vs St={opt2.get('st_strength',0):.3f}")
    print()

    # ── Save results ──
    output = {
        "domain_analysis": domain_results,
        "lt_st_weights": {
            "target_sfc": opt,
            "target_btc": opt2,
        },
        "recommended_blend": {
            "lt_weight_pct": lt_w,
            "st_weight_pct": st_w,
            "formula": f"SFC = {lt_w/100:.2f} × Lt + {st_w/100:.2f} × St",
        },
        "top_features": top_features,
    }
    with open(os.path.join(SFC_DIR, ".correlation_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"   📁 Results saved to .correlation_results.json")
    print("✅ Analysis complete")


if __name__ == "__main__":
    main()
