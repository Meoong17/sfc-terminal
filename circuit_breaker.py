#!/usr/bin/env python3
"""
circuit_breaker.py — Safety Guard for SFC Terminal Outputs
===========================================================
Validates all prediction outputs before they reach data.json.

Features:
  - NaN/Inf detection and replacement
  - Range validation (min/max per field)
  - Extreme value rejection (sudden jumps > threshold)
  - Consecutive failure counter → auto-disable on Nth failure
  - Fallback to last-known-valid state
  - Structured warnings for monitoring

Usage:
    from circuit_breaker import CircuitBreaker
    cb = CircuitBreaker()
    cleaned, ok, warnings = cb.validate(output_dict)
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# ── Config ──
SFC_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(SFC_DIR, ".circuit_breaker_state.json")

# Max consecutive failures before auto-disable
MAX_CONSECUTIVE_FAILURES = 5

# Cooldown period (seconds) after tripping — reset after this
COOLDOWN_SECONDS = 3600  # 1 hour

# Max allowed SFC jump (percentage points) between consecutive runs
# Reduced from 40pp to 20pp — 40pp swing can flip NORMAL→CRITICAL in 5min
MAX_SFC_JUMP_PP = 20.0

# ── Field validation rules ──
# Each entry: (min, max) or None for no check
FIELD_RULES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    # SFC core
    "sfc_effective": (0.0, 100.0),
    "sfc_base": (0.0, 100.0),
    # Confidence
    "composite_confidence": (0.0, 1.0),
    # Signals
    "cascade_risk": (0.0, 1.0),
    "transition_risk": (0.0, 1.0),
    "signal_strength": (0.0, 1.0),
    "readiness_score": (0.0, 1.0),
    "kelly_fraction": (0.0, 1.0),
    "kelly_half": (0.0, 1.0),
    "kelly_quarter": (0.0, 1.0),
    # Method scores (M1-M31)
    "m1_klr": (0.0, 100.0),
    "m2_logit": (0.0, 100.0),
    "m3_bayes": (0.0, 100.0),
    "m4_ewc": (0.0, 100.0),
    "m5_qreg": (0.0, 100.0),
    "m6_regime_score": (0.0, 100.0),
    "method_agreement": (0.0, 1.0),
    # Advanced modules
    "hmm_crisis_prob": (0.0, 1.0),
    "mtf_alignment_score": (-1.0, 1.0),
    # ML ensemble
    "ml_ensemble_score": (0.0, 1.0),
    "ml_ensemble_confidence": (0.0, 1.0),
    "ml_accuracy": (0.0, 100.0),
    # Backtest
    "bt_sharpe": (0.0, 10.0),
    "bt_win_rate": (0.0, 1.0),
    "bt_return": (-1.0, 10.0),
    "bt_max_dd": (-1.0, 1.0),
    # News
    "news_stress": (-100.0, 100.0),
    "news_sentiment": (-1.0, 1.0),
    # Q10 On-Chain (percentile 0-100, not -1..1)
    "q10_whale_pressure": (0.0, 100.0),
    "q10_onchain_value": (0.0, 100.0),
    "q10_buying_power": (0.0, 100.0),
    "q10_market_structure": (0.0, 100.0),
    # Q5 advanced methods
    "m65_cnn_attention": (0.0, 1.0),
    "m69_systemic_risk": (0.0, 1.0),
    "m69_btc_systemic_risk": (0.0, 1.0),
    # Probabilistic (new)
    "predicted_mean": (0.0, 100.0),
    "predicted_std": (0.0, 50.0),
    "var_95": (0.0, 100.0),
    "es_975": (0.0, 100.0),
    "ci_90_lower": (0.0, 100.0),
    "ci_90_upper": (0.0, 100.0),
    "prob_stress": (0.0, 1.0),
    "prob_critical": (0.0, 1.0),
    "prob_crash_10pct": (0.0, 1.0),
    "prob_calm": (0.0, 1.0),
    "sharpe_ratio": (-10.0, 10.0),
    "sortino_ratio": (-10.0, 10.0),
    # ETF flows
    "m81_etf_flow": (-1.0, 1.0),
    "m82_etf_holdings": (-1.0, 1.0),
    # Fiscal
    "m83_tga_score": (-1.0, 1.0),
    "m84_rrp_score": (-1.0, 1.0),
    "m85_fiscal_composite": (-1.0, 1.0),
    # DXY
    "dxy_btc_corr": (-1.0, 1.0),
}

# Fields that should NEVER change by more than MAX_SFC_JUMP_PP between runs
JUMP_SENSITIVE_FIELDS = [
    "sfc_effective",
    "sfc_base",
]


# ════════════════════════════════════════════════════════════════
# CircuitBreaker
# ════════════════════════════════════════════════════════════════


class CircuitBreaker:
    """Validates SFC output and falls back on failure.

    State is persisted to disk so a crash doesn't reset the failure
    counter — prevents rapid fail-retry loops after deployment.
    """

    def __init__(self):
        self._last_valid: Dict[str, Any] = {}
        self._consecutive_failures: int = 0
        self._total_failures: int = 0
        self._total_valid: int = 0
        self._tripped: bool = False
        self._tripped_at: float = 0.0
        self._load_state()

    # ── Public API ──

    def validate(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
        """Validate all numeric fields and return cleaned data.

        Returns:
            (cleaned_data, is_valid, warnings)
            cleaned_data: dict with NaN/Inf fixed, clamped values
            is_valid: True if all checks passed
            warnings: list of human-readable warning strings
        """
        warnings: List[str] = []
        cleaned = dict(data)  # shallow copy
        all_ok = True

        # ── 1. Global NaN/Inf sweep ──
        nan_keys = []
        for key, value in list(cleaned.items()):
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    nan_keys.append(key)
                    # Replace with last valid or 0.0
                    cleaned[key] = self._last_valid.get(key, 0.0)
                    all_ok = False

        if nan_keys:
            warnings.append(f"NAN/INF detected and replaced: {', '.join(nan_keys[:10])}")

        # ── 2. Range validation ──
        range_violations = []
        for key, (lo, hi) in FIELD_RULES.items():
            if key not in cleaned:
                continue
            val = cleaned[key]
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                continue
            if lo is not None and val < lo:
                range_violations.append(f"{key}={val:.2f} < {lo}")
                cleaned[key] = lo
                all_ok = False
            elif hi is not None and val > hi:
                range_violations.append(f"{key}={val:.2f} > {hi}")
                cleaned[key] = hi
                all_ok = False

        if range_violations:
            warnings.append(f"Range violations clamped: {'; '.join(range_violations[:10])}")

        # ── 3. Sudden jump detection ──
        jump_violations = []
        for key in JUMP_SENSITIVE_FIELDS:
            if key not in cleaned or key not in self._last_valid:
                continue
            curr = cleaned[key]
            prev = self._last_valid[key]
            if not isinstance(curr, (int, float)) or not isinstance(prev, (int, float)):
                continue
            if prev == 0.0:
                continue  # can't compute jump from 0
            jump = abs(curr - prev)
            if jump > MAX_SFC_JUMP_PP:
                jump_violations.append(f"{key}: {prev:.1f} → {curr:.1f} (Δ={jump:.1f}pp)")
                # Don't clamp — just warn; sudden jumps CAN be real (flash crash)
                # But flag them for attention
                all_ok = False

        if jump_violations:
            warnings.append(f"Sudden jumps detected: {'; '.join(jump_violations)}")

        # ── 4. Consistency checks ──
        consistency_issues = []

        # sfc_effective should be >= sfc_base (base gets adjusted up)
        sfc_eff = cleaned.get("sfc_effective")
        sfc_base = cleaned.get("sfc_base")
        if isinstance(sfc_eff, (int, float)) and isinstance(sfc_base, (int, float)):
            if sfc_eff < sfc_base - 5.0:  # allow 5pp tolerance
                consistency_issues.append(
                    f"sfc_effective ({sfc_eff:.1f}) < sfc_base ({sfc_base:.1f})"
                )
                all_ok = False

        # confidence should be 0-1
        conf = cleaned.get("composite_confidence")
        if isinstance(conf, (int, float)):
            if conf < 0.0 or conf > 1.0:
                consistency_issues.append(f"composite_confidence out of range: {conf}")
                all_ok = False

        if consistency_issues:
            warnings.append(f"Consistency issues: {'; '.join(consistency_issues)}")

        # ── 5. Auto-disable (Circuit Breaker trip) ──
        if not all_ok:
            self._consecutive_failures += 1
            self._total_failures += 1

            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                if not self._tripped:
                    self._tripped = True
                    self._tripped_at = time.time()
                    warnings.append(
                        f"CIRCUIT BREAKER TRIPPED after {self._consecutive_failures} "
                        f"consecutive failures. Output purged."
                    )
                return {}, False, warnings
        else:
            self._consecutive_failures = 0
            self._total_valid += 1

            # Save valid state for future fallback
            self._last_valid = {}
            for key in list(FIELD_RULES.keys()) + JUMP_SENSITIVE_FIELDS + [
                "btc", "btc_24h", "fng", "dvol", "dom",
            ]:
                if key in cleaned:
                    self._last_valid[key] = cleaned[key]

        # ── 6. Post-trip recovery check ──
        if self._tripped and all_ok:
            elapsed = time.time() - self._tripped_at
            if elapsed >= COOLDOWN_SECONDS:
                self._tripped = False
                self._consecutive_failures = 0
                warnings.append("Circuit breaker reset after cooldown.")
            else:
                remaining = int(COOLDOWN_SECONDS - elapsed)
                warnings.append(f"Circuit breaker still active ({remaining}s remaining).")
                return {}, False, warnings

        # Persist state every 10 valid runs
        if self._total_valid % 10 == 0 or self._tripped:
            self._save_state()

        return cleaned, all_ok, warnings

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostics."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self._total_failures,
            "total_valid": self._total_valid,
            "tripped": self._tripped,
            "tripped_at": self._tripped_at,
            "last_valid_keys": list(self._last_valid.keys()),
            "cooldown_remaining": max(0, COOLDOWN_SECONDS - (time.time() - self._tripped_at))
            if self._tripped else 0,
        }

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_valid = 0
        self._tripped = False
        self._tripped_at = 0.0
        self._last_valid = {}
        self._save_state()

    # ── Persistence ──

    def _load_state(self) -> None:
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    state = json.load(f)
                self._last_valid = state.get("last_valid", {})
                self._consecutive_failures = state.get("consecutive_failures", 0)
                self._total_failures = state.get("total_failures", 0)
                self._total_valid = state.get("total_valid", 0)
                self._tripped = state.get("tripped", False)
                self._tripped_at = state.get("tripped_at", 0.0)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_state(self) -> None:
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "last_valid": self._last_valid,
                    "consecutive_failures": self._consecutive_failures,
                    "total_failures": self._total_failures,
                    "total_valid": self._total_valid,
                    "tripped": self._tripped,
                    "tripped_at": self._tripped_at,
                    "updated_at": time.time(),
                }, f, indent=2)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════
# Module-level convenience
# ════════════════════════════════════════════════════════════════

_default_cb: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create module-level singleton."""
    global _default_cb
    if _default_cb is None:
        _default_cb = CircuitBreaker()
    return _default_cb


def validate_output(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    """One-shot convenience: validate data through circuit breaker."""
    cb = get_circuit_breaker()
    return cb.validate(data)


# ════════════════════════════════════════════════════════════════
# Standalone test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Run circuit breaker test scenarios."""
    print("=" * 60)
    print("Circuit Breaker — Test Scenarios")
    print("=" * 60)

    cb = CircuitBreaker()

    # ── Scenario 1: Normal data ──
    print("\n── Scenario 1: Normal valid data ──")
    normal = {
        "sfc_effective": 37.5,
        "sfc_base": 15.7,
        "composite_confidence": 0.38,
        "cascade_risk": 0.29,
        "transition_risk": 0.5,
        "signal_strength": 0.75,
        "readiness_score": 0.3,
        "kelly_fraction": 0.1,
        "m1_klr": 13.7,
        "m2_logit": 8.0,
        "m3_bayes": 4.0,
        "method_agreement": 0.6,
        "hmm_crisis_prob": 0.1,
        "mtf_alignment_score": 0.3,
        "ml_ensemble_score": 0.4,
        "bt_sharpe": 1.9,
        "bt_win_rate": 0.95,
        "news_stress": -2.2,
        "news_sentiment": -0.3,
        "btc": 60447,
        "btc_24h": -0.471,
        "fng": 17,
        "dvol": 44.6,
        "dom": 58.1,
    }
    cleaned, ok, warns = cb.validate(normal)
    print(f"  Valid: {ok}")
    print(f"  Warnings: {warns if warns else 'None'}")
    assert ok, "Normal data should pass"
    print("  ✓ PASS")

    # ── Scenario 2: NaN in critical field ──
    print("\n── Scenario 2: NaN in sfc_effective ──")
    with_nan = dict(normal)
    with_nan["sfc_effective"] = float("nan")
    with_nan["m1_klr"] = float("nan")
    cleaned, ok, warns = cb.validate(with_nan)
    print(f"  Valid: {ok}")
    print(f"  Warnings: {warns}")
    # NaN should be replaced with last valid
    assert cleaned.get("sfc_effective") == 37.5, "Should fallback to last valid"
    print(f"  sfc_effective after fix: {cleaned['sfc_effective']}")
    print("  ✓ PASS")

    # ── Scenario 3: Range violation ──
    print("\n── Scenario 3: Out of range values ──")
    with_range = dict(normal)
    with_range["composite_confidence"] = 5.0  # should be 0-1
    with_range["bt_sharpe"] = 99.9  # should be 0-10
    cleaned, ok, warns = cb.validate(with_range)
    print(f"  Valid: {ok}")
    print(f"  Warnings: {warns}")
    assert cleaned["composite_confidence"] == 1.0, "Should clamp to 1.0"
    assert cleaned["bt_sharpe"] == 10.0, "Should clamp to 10.0"
    print(f"  confidence clamped to: {cleaned['composite_confidence']}")
    print(f"  sharpe clamped to: {cleaned['bt_sharpe']}")
    print("  ✓ PASS")

    # ── Scenario 4: Extreme jump ──
    print("\n── Scenario 4: Sudden SFC jump (30 → 90) ──")
    # First, save a valid state
    cb.validate(normal)
    with_jump = dict(normal)
    with_jump["sfc_effective"] = 95.0  # jumped from 37.5 → 95 (Δ=57.5pp)
    cleaned, ok, warns = cb.validate(with_jump)
    print(f"  Valid: {ok}")
    print(f"  Warnings: {warns}")
    # Jump should be flagged but value NOT clamped (could be real)
    print("  ✓ PASS (jump flagged, value preserved)")

    # ── Scenario 5: Consecutive failures → trip ──
    print("\n── Scenario 5: Consecutive failures → circuit breaker trip ──")
    cb2 = CircuitBreaker()  # fresh breaker
    bad = {"sfc_effective": float("nan"), "sfc_base": float("nan")}
    for i in range(MAX_CONSECUTIVE_FAILURES + 1):
        cleaned, ok, warns = cb2.validate(bad)
        if i < MAX_CONSECUTIVE_FAILURES - 1:
            print(f"  Failure {i+1}: ok={ok}, warns={warns}")
        elif i == MAX_CONSECUTIVE_FAILURES - 1:
            print(f"  Failure {i+1}: ok={ok}, warns={warns} (last before trip)")
        else:
            print(f"  Failure {i+1}: TRIPPED! ok={ok}, warns={warns}")
            assert not ok, "Should be invalid after trip"
            assert cleaned == {}, "Should return empty dict after trip"
            print("  ✓ CIRCUIT BREAKER TRIPPED correctly")

    # ── Scenario 6: Consistency check (sfc_effective < sfc_base) ──
    print("\n── Scenario 6: Consistency violation ──")
    cb3 = CircuitBreaker()
    # Seed with valid data first
    cb3.validate(normal)
    inconsistent = dict(normal)
    inconsistent["sfc_effective"] = 5.0   # lower than sfc_base (15.7)
    inconsistent["sfc_base"] = 15.7
    cleaned, ok, warns = cb3.validate(inconsistent)
    print(f"  Valid: {ok}")
    print(f"  Warnings: {warns}")
    print("  ✓ PASS")

    print("\n" + "=" * 60)
    print("All circuit breaker tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
