#!/usr/bin/env python3
"""
probabilistic_output.py — Probabilistic Head for SFC Terminal
=============================================================
Converts point-estimate SFC scores into full probability distributions.

Approach (no retraining needed):
  1. Estimate uncertainty from method disagreement (cross-sectional sigma)
  2. Blend with confidence score and regime-aware volatility
  3. Output: mu, sigma, VaR 95%, ES 97.5%, confidence intervals, stress probability

Usage:
    from probabilistic_output import ProbabilisticHead
    head = ProbabilisticHead()
    result = head.compute(sfc_score=37.5, method_scores=[...], composite_confidence=0.38, regime="NORMAL")
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Config ──
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".probabilistic_state.json")

# Stress thresholds (SFC score in %)
STRESS_THRESHOLD = 25.0   # above this = stress zone
CRITICAL_THRESHOLD = 50.0  # above this = critical

# Regime volatility multipliers (higher = more uncertainty)
REGIME_VOL_MULTIPLIER = {
    "BULL": 0.8,
    "BEAR": 1.4,
    "SIDEWAYS": 1.0,
    "CRISIS": 2.0,
    "NORMAL": 1.0,
}

# ── Normal CDF (approximation via error function) ──


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Approximate standard normal quantile (inverse CDF) using rational approximation.

    Uses the Acklam algorithm for high accuracy (~1.15e-9).
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Rational approximation coefficients
    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return z
    elif p <= p_high:
        # Rational approximation for central region
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return z
    else:
        # Rational approximation for upper region
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return z


# ════════════════════════════════════════════════════════════════
# ProbabilisticHead
# ════════════════════════════════════════════════════════════════


class ProbabilisticHead:
    """Estimates a full probability distribution for SFC predictions.

    Uses cross-sectional method disagreement + confidence score
    to estimate uncertainty without requiring labeled stress events.
    """

    def __init__(self):
        self._history: List[float] = []  # recent SFC scores
        self._errors: List[float] = []   # estimated errors (|score - method_mean|)
        self._loaded = False
        self._load_state()

    # ── Public API ──

    def compute(
        self,
        sfc_score: float,
        method_scores: Optional[List[float]] = None,
        composite_confidence: float = 0.5,
        regime: str = "NORMAL",
        zone: str = "NORMAL",
    ) -> Dict[str, Any]:
        """Compute probabilistic output from SFC point estimate.

        Args:
            sfc_score: SFC effective score (0-100)
            method_scores: List of individual method scores (M1-M31+)
            composite_confidence: Confidence score (0-1)
            regime: Market regime string
            zone: Stress zone string

        Returns:
            Dict with keys: predicted_mean, predicted_std, var_95, es_975,
                            prob_stress, prob_critical, prob_crash_10pct,
                            confidence_interval_90, quantiles,
                            uncertainty_breakdown
        """
        sfc = max(0.0, min(100.0, float(sfc_score)))
        conf = max(0.0, min(1.0, float(composite_confidence)))

        # 1. Estimate sigma from method disagreement
        method_sigma = self._estimate_method_sigma(method_scores, sfc)

        # 2. Confidence-based sigma (low conf = wide distribution)
        conf_sigma = 15.0 * (1.0 - conf)  # 0% conf → 15pp sigma, 100% conf → 0pp

        # 3. Regime-aware multiplier
        regime_mult = REGIME_VOL_MULTIPLIER.get(regime.upper(), 1.0)

        # 4. Historical noise floor (from recent SFC score volatility)
        hist_sigma = self._estimate_historical_sigma(sfc)

        # 5. Blend all sigmas (root-sum-square)
        raw_sigma = math.sqrt(
            method_sigma ** 2 +
            conf_sigma ** 2 +
            hist_sigma ** 2
        )
        final_sigma = min(25.0, max(1.0, raw_sigma * regime_mult))

        # Update history
        self._update_history(sfc, method_sigma)

        # 6. Compute distribution metrics
        mu = sfc  # point estimate = mean

        # VaR 95%: 5th percentile (worst-case loss in SFC units)
        var_95 = mu - 1.645 * final_sigma

        # ES 97.5%: Expected shortfall (average of worst 2.5%)
        es_975 = mu - 2.0 * final_sigma

        # Probabilities
        prob_stress = 1.0 - _normal_cdf((STRESS_THRESHOLD - mu) / max(final_sigma, 0.1))
        prob_critical = 1.0 - _normal_cdf((CRITICAL_THRESHOLD - mu) / max(final_sigma, 0.1))

        # Probability of >10% crash (SFC > 60)
        prob_crash_10pct = 1.0 - _normal_cdf((60.0 - mu) / max(final_sigma, 0.1))

        # Probability of calm (SFC < 15)
        prob_calm = _normal_cdf((15.0 - mu) / max(final_sigma, 0.1))

        # 90% confidence interval
        ci_lower = mu - 1.645 * final_sigma
        ci_upper = mu + 1.645 * final_sigma

        # Quantiles
        quantiles = {
            "q_01": round(mu - 2.326 * final_sigma, 2),
            "q_05": round(mu - 1.645 * final_sigma, 2),
            "q_10": round(mu - 1.282 * final_sigma, 2),
            "q_25": round(mu - 0.674 * final_sigma, 2),
            "q_50": round(mu, 2),
            "q_75": round(mu + 0.674 * final_sigma, 2),
            "q_90": round(mu + 1.282 * final_sigma, 2),
            "q_95": round(mu + 1.645 * final_sigma, 2),
            "q_99": round(mu + 2.326 * final_sigma, 2),
        }

        # Risk metrics
        sharpe_ratio = (50.0 - mu) / max(final_sigma, 0.1)  # distance from neutral 50
        sortino_ratio = sharpe_ratio * 1.5  # approximate

        return {
            "predicted_mean": round(mu, 2),
            "predicted_std": round(final_sigma, 2),
            "var_95": round(max(0.0, var_95), 2),
            "es_975": round(max(0.0, es_975), 2),
            "ci_90_lower": round(max(0.0, ci_lower), 2),
            "ci_90_upper": round(min(100.0, ci_upper), 2),
            "prob_stress": round(prob_stress, 4),
            "prob_critical": round(prob_critical, 4),
            "prob_crash_10pct": round(prob_crash_10pct, 4),
            "prob_calm": round(prob_calm, 4),
            "quantiles": quantiles,
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "uncertainty_breakdown": {
                "method_disagreement_sigma": round(method_sigma, 2),
                "confidence_sigma": round(conf_sigma, 2),
                "historical_noise_sigma": round(hist_sigma, 2),
                "regime_multiplier": round(regime_mult, 2),
                "final_sigma": round(final_sigma, 2),
            },
            "raw_sfc": round(sfc, 2),
            "conf_used": round(conf, 2),
            "regime": regime.upper(),
        }

    # ── Internal: Sigma Estimation ──

    def _estimate_method_sigma(
        self,
        method_scores: Optional[List[float]],
        sfc: float,
    ) -> float:
        """Estimate uncertainty from dispersion among individual method scores."""
        if not method_scores or len(method_scores) < 3:
            return 10.0  # fallback: 10pp sigma

        # Filter out None/0 values (disabled methods)
        valid = [float(s) for s in method_scores if s is not None and isinstance(s, (int, float))]
        valid = [s for s in valid if 0 < s < 100]

        if len(valid) < 3:
            return 10.0

        # Coefficient of variation as sigma proxy
        mean_method = sum(valid) / len(valid)
        if mean_method < 0.01:
            return 10.0

        variance = sum((s - mean_method) ** 2 for s in valid) / len(valid)
        std_dev = math.sqrt(variance)

        # Std dev in SFC percentage points
        # Higher dispersion = higher uncertainty
        sigma = min(20.0, std_dev)

        # Also consider: distance between M1-M6 avg and full avg
        if len(valid) >= 6:
            m1m6_avg = sum(valid[:6]) / 6
            full_avg = sum(valid) / len(valid)
            gap_penalty = abs(m1m6_avg - full_avg) * 0.5
            sigma = max(sigma, min(15.0, gap_penalty))

        return max(3.0, min(20.0, sigma))  # clamp [3, 20]

    def _estimate_historical_sigma(self, current_sfc: float) -> float:
        """Estimate sigma from recent SFC score volatility.

        Uses the last N scores to compute rolling std.
        """
        if len(self._history) < 3:
            return 3.0  # low starting sigma

        # Recent window
        recent = self._history[-10:] if len(self._history) >= 10 else self._history
        if len(recent) < 3:
            return 3.0

        mean_h = sum(recent) / len(recent)
        var_h = sum((s - mean_h) ** 2 for s in recent) / len(recent)
        sigma = math.sqrt(var_h)

        return min(15.0, max(1.0, sigma))

    def _update_history(self, sfc: float, method_sigma: float) -> None:
        """Update rolling history of SFC scores."""
        self._history.append(sfc)
        self._errors.append(method_sigma)

        # Keep bounded
        MAX_HISTORY = 100
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]
        if len(self._errors) > MAX_HISTORY:
            self._errors = self._errors[-MAX_HISTORY:]

        # Persist every 5 updates
        if len(self._history) % 5 == 0:
            self._save_state()

    # ── Persistence ──

    def _load_state(self) -> None:
        """Restore history from disk."""
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    state = json.load(f)
                self._history = state.get("history", [])
                self._errors = state.get("errors", [])
                self._loaded = True
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        """Persist history to disk."""
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "history": self._history[-100:],
                    "errors": self._errors[-100:],
                    "updated_at": time.time(),
                }, f)
        except OSError:
            pass

    def get_state_info(self) -> Dict[str, Any]:
        """Return state info for debugging."""
        return {
            "history_len": len(self._history),
            "errors_len": len(self._errors),
            "loaded": self._loaded,
            "latest_sfc": self._history[-1] if self._history else None,
        }


# ════════════════════════════════════════════════════════════════
# Module-level convenience
# ════════════════════════════════════════════════════════════════

_default_head: Optional[ProbabilisticHead] = None


def get_probabilistic_head() -> ProbabilisticHead:
    """Get or create module-level singleton."""
    global _default_head
    if _default_head is None:
        _default_head = ProbabilisticHead()
    return _default_head


def compute_probabilistic(
    sfc_score: float,
    method_scores: Optional[List[float]] = None,
    composite_confidence: float = 0.5,
    regime: str = "NORMAL",
    zone: str = "NORMAL",
) -> Dict[str, Any]:
    """One-shot convenience: compute probabilistic output."""
    head = get_probabilistic_head()
    return head.compute(
        sfc_score=sfc_score,
        method_scores=method_scores,
        composite_confidence=composite_confidence,
        regime=regime,
        zone=zone,
    )


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Run probabilistic head with current data.json values."""
    print("=" * 60)
    print("Probabilistic Head — Test")
    print("=" * 60)

    # Load current SFC values
    data_path = os.path.join(SFC_DIR, "data.json")
    try:
        with open(data_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"! Cannot load data.json: {e}")
        data = {}

    sfc = float(data.get("sfc_effective", 37.5))
    conf = float(data.get("composite_confidence", 0.38))
    regime = str(data.get("regime", "NORMAL"))
    zone = str(data.get("zone", "NORMAL"))

    # Collect method scores
    method_fields = [
        "m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
        "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
        "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
        "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
        "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
        "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
    ]
    method_scores = [data.get(f) for f in method_fields]

    print(f"SFC: {sfc:.1f}%  Conf: {conf:.2f}  Regime: {regime}  Zone: {zone}")
    print(f"Methods: {sum(1 for s in method_scores if s is not None)}/{len(method_scores)} active\n")

    head = ProbabilisticHead()
    result = head.compute(
        sfc_score=sfc,
        method_scores=method_scores,
        composite_confidence=conf,
        regime=regime,
        zone=zone,
    )

    print("── Distribution ──")
    print(f"  μ (mean):       {result['predicted_mean']:.1f}")
    print(f"  σ (std):        {result['predicted_std']:.1f}")
    print(f"  VaR 95%:        {result['var_95']:.1f}")
    print(f"  ES 97.5%:       {result['es_975']:.1f}")
    print(f"  90% CI:         [{result['ci_90_lower']:.1f}, {result['ci_90_upper']:.1f}]")

    print("\n── Probabilities ──")
    print(f"  P(Stress):      {result['prob_stress']*100:.1f}%")
    print(f"  P(Critical):    {result['prob_critical']*100:.1f}%")
    print(f"  P(Crash>10%):   {result['prob_crash_10pct']*100:.1f}%")
    print(f"  P(Calm):        {result['prob_calm']*100:.1f}%")

    print("\n── Uncertainty Sources ──")
    for k, v in result['uncertainty_breakdown'].items():
        print(f"  {k}: {v}")

    print("\n── Quantiles ──")
    for q, v in result['quantiles'].items():
        print(f"  {q}: {v:.1f}")

    print(f"\n── Risk Ratios ──")
    print(f"  Sharpe-like:    {result['sharpe_ratio']:.3f}")
    print(f"  Sortino-like:   {result['sortino_ratio']:.3f}")

    print(f"\nState: {head.get_state_info()}")
    print("\n✓ Probabilistic head test complete")


if __name__ == "__main__":
    main()
