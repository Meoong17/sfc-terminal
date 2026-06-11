#!/usr/bin/env python3
"""
SFC v2.1 — NO LLM VERSION (Rule-based)
Untuk kompatibilitas dengan collect.py
"""

import math, sys

def run_sfc_no_llm(btc=None, btc_24h=None, dom=None, dvol=None, fng=None, pc_oi=None, m2_yoy=None, dxy=None):
    """Run SFC calculation without LLM"""
    
    # Score factors
    factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}
    
    if m2_yoy:
        if m2_yoy < 2: factors["Lt"] -= 1.5
        elif m2_yoy < 4: factors["Lt"] -= 0.5
        elif m2_yoy > 7: factors["Lt"] += 1.0
        elif m2_yoy > 5: factors["Lt"] += 0.5
    if btc_24h:
        if btc_24h < -10: factors["Lt"] -= 1.0
        elif btc_24h < -5: factors["Lt"] -= 0.5
        elif btc_24h > 10: factors["Lt"] += 1.0
        elif btc_24h > 5: factors["Lt"] += 0.5
    
    if dom:
        if dom > 65: factors["St"] -= 1.0
        elif dom < 45: factors["St"] += 0.5
    if pc_oi:
        if pc_oi > 1.2: factors["St"] -= 1.5
        elif pc_oi > 1.0: factors["St"] -= 0.5
        elif pc_oi < 0.6: factors["St"] += 0.5
    
    if fng:
        if fng <= 15: factors["Rt"] = -2.0
        elif fng <= 30: factors["Rt"] = -1.0
        elif fng <= 45: factors["Rt"] = -0.5
        elif fng <= 55: factors["Rt"] = 0.0
        elif fng <= 75: factors["Rt"] = 1.0
        else: factors["Rt"] = 2.0
    
    if dvol:
        if dvol >= 120: factors["Ft"] = -2.5
        elif dvol >= 100: factors["Ft"] = -1.5
        elif dvol >= 80: factors["Ft"] = -0.5
        elif dvol >= 60: factors["Ft"] = 0.0
        elif dvol < 40: factors["Ft"] = 1.0
        else: factors["Ft"] = 0.5
    
    if dxy:
        if dxy > 110: factors["Sc"] = -1.5
        elif dxy > 105: factors["Sc"] = -0.5
        elif dxy < 95: factors["Sc"] = 1.0
    if dom and dom > 70:
        factors["Sc"] -= 0.5
    
    for k in factors:
        factors[k] = max(-3.0, min(3.0, factors[k]))
    
    # Calculate ensemble
    norm = {k: v/6 for k, v in factors.items()}
    z_score = sum(norm.values())
    
    ns_r = {"Lt":0.35, "St":0.50, "Rt":0.40, "Ft":0.25, "Sc":0.80}
    w = {k:1/v for k,v in ns_r.items()}
    sig = sum((1.0 if norm[k]<-2 else 0.7 if norm[k]<-1 else 0.3 if norm[k]<0 else 0) * w[k] for k in factors)
    p_klr = sig / sum(w.values())
    
    zc = [-1.0, -2.0, -3.0, -4.0, -8.0]
    pc = [0.08, 0.20, 0.55, 0.75, 0.95]
    yc = [math.log(p/(1-p)) for p in pc]
    n_z = len(zc)
    zm = sum(zc)/n_z
    ym = sum(yc)/n_z
    b1 = sum((zc[i]-zm)*(yc[i]-ym) for i in range(n_z)) / sum((z-zm)**2 for z in zc)
    b0 = ym - b1*zm
    z_l = b0 + b1*z_score
    p_logit = 1/(1+math.exp(-z_l))
    
    prior = 0.04
    odds = prior/(1-prior)
    bayes_mult = [2.5, 2.0, 2.0, 3.0, 1.5]
    for i, k in enumerate(factors):
        if norm[k] < -0.5:
            odds *= bayes_mult[i]
    p_bayes = odds/(1+odds)
    
    w_ad = {"Lt":0.25, "St":0.20, "Rt":0.20, "Ft":0.30, "Sc":0.05}
    ewc = sum(w_ad[k] * abs(norm[k]) for k in factors)
    p_ewc = ewc/3.0
    
    qr_anchors = [(-8.0, 0.95), (-5.0, 0.75), (-3.0, 0.50), (-1.5, 0.25), (-0.5, 0.10), (0.5, 0.04), (2.0, 0.01)]
    def quantile_stress(z):
        anchors = sorted(qr_anchors, key=lambda x: x[0])
        if z <= anchors[0][0]: return anchors[0][1]
        if z >= anchors[-1][0]: return anchors[-1][1]
        for i in range(len(anchors)-1):
            z0, p0 = anchors[i]
            z1, p1 = anchors[i+1]
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                return p0 + t * (p1 - p0)
        return 0.04
    p_quantile = quantile_stress(z_score)
    
    vals = list(norm.values())
    n = len(vals)
    extreme_count = sum(1 for v in vals if v < -1.0)
    severe_count = sum(1 for v in vals if v < -2.0)
    p_extremity = (extreme_count * 0.15 + severe_count * 0.20)
    mean_v = sum(vals) / n
    variance = sum((v - mean_v)**2 for v in vals) / n
    coherence_bonus = 0.10 * (1.0 - variance) if mean_v < -0.5 and variance < 1.0 else 0.0
    ft_val = norm.get("Ft", 0)
    lt_val = norm.get("Lt", 0)
    tail_contribution = (0.15 if ft_val < -1.5 else 0.0) + (0.10 if lt_val < -1.5 else 0.0)
    p_baseline = max(0.0, min((-mean_v) * 0.12, 0.50))
    p_regime = min(p_baseline + p_extremity + coherence_bonus + tail_contribution, 0.99)
    p_regime = max(p_regime, 0.01)
    
    p_ens = 0.19*p_klr + 0.16*p_logit + 0.12*p_bayes + 0.16*p_ewc + 0.24*p_quantile + 0.14*p_regime
    
    method_probs = [p_klr, p_logit, p_bayes, p_ewc, p_quantile, p_regime]
    mean_p = sum(method_probs) / len(method_probs)
    variance_p = sum((p - mean_p)**2 for p in method_probs) / len(method_probs)
    std_p = math.sqrt(variance_p)
    method_agreement = max(0.0, min(1.0, 1.0 - (std_p / 0.20)))
    
    zone = "CRITICAL" if p_ens > 0.75 else "HIGH" if p_ens > 0.5 else "ELEVATED" if p_ens > 0.25 else "NORMAL"
    
    print(f"SFC v2.1 = {p_ens*100:.1f}% | Zone: {zone} | Agreement: {method_agreement*100:.0f}%")
    print(f"   Methods: KLR={p_klr*100:.1f}% Logit={p_logit*100:.1f}% Bayes={p_bayes*100:.1f}% ECB={p_ewc*100:.1f}% QReg={p_quantile*100:.1f}% Regime={p_regime*100:.1f}%")
    print(f"   Factors: {factors}")
    print(f"   M5_QReg: {p_quantile*100:.4f}")
    print(f"   M6_Regime: {p_regime*100:.4f}")
    print(f"   Agreement: {method_agreement:.4f}")
    
    return p_ens*100, zone, factors, p_quantile*100, p_regime*100, method_agreement

if __name__ == "__main__":
    # For standalone testing
    run_sfc_no_llm(btc=60000, btc_24h=-2, dom=58, dvol=65, fng=42, pc_oi=0.85, m2_yoy=5.2, dxy=102.5)
