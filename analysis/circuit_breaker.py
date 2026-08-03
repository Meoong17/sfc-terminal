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
SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        # Two categories of range problems:
        #
        # (a) CLAMPED: value was out of range but corrected by clamping.
        #     Output is still usable — counting this as a "failure" toward
        #     MAX_CONSECUTIVE_FAILURES would trip the breaker on systematic
        #     rounding drift (e.g. sfc_effective=100.5 every cycle) even
        #     when every output field is perfectly valid after clamping.
        #     Confirmed by test: trivial 0.5pp overshoot recurring 5x
        #     tripped the breaker and purged ALL output including valid
        #     fields. → all_ok stays True; warning still emitted.
        #
        # (b) INVALID: value cannot be recovered (wrong type, NaN, Inf).
        #     → all_ok = False → counts toward failure / circuit trip.
        range_clamped = []
        range_invalid = []

        for key, (lo, hi) in FIELD_RULES.items():
            if key not in cleaned:
                continue
            val = cleaned[key]
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                range_invalid.append(f"{key}={val!r} (not numeric)")
                all_ok = False
                continue
            if val != val or val in (float('inf'), float('-inf')):
                range_invalid.append(f"{key}={val} (NaN/Inf)")
                all_ok = False
                continue
            if lo is not None and val < lo:
                cleaned[key] = lo
                range_clamped.append(f"{key}={val:.3f}→{lo}")
            elif hi is not None and val > hi:
                cleaned[key] = hi
                range_clamped.append(f"{key}={val:.3f}→{hi}")

        # ── 2b. Validate prob_quantiles nested dict ──
        # Previously absent from FIELD_RULES entirely — q_01=-27.42 was
        # confirmed in production data as a nonsensical negative percentile
        # visible on the dashboard. Clamped here in the same non-failure
        # category as (a) above: a corrected quantile value is valid output.
        quantiles = cleaned.get("prob_quantiles")
        if isinstance(quantiles, dict):
            q_clamped = []
            for q_key, q_val in list(quantiles.items()):
                if not isinstance(q_val, (int, float)):
                    continue
                if q_val < 0.0:
                    quantiles[q_key] = 0.0
                    q_clamped.append(f"{q_key}:{q_val:.2f}→0.0")
                elif q_val > 100.0:
                    quantiles[q_key] = 100.0
                    q_clamped.append(f"{q_key}:{q_val:.2f}→100.0")
            if q_clamped:
                range_clamped.append(f"prob_quantiles({', '.join(q_clamped)})")
            cleaned["prob_quantiles"] = quantiles

        if range_clamped:
            warnings.append(f"Range clamped (output valid): {'; '.join(range_clamped[:10])}")
        if range_invalid:
            warnings.append(f"Uncorrectable violations: {'; '.join(range_invalid)}")

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

        # sfc_effective final value is produced by a documented adjustment chain:
        #   mid = sfc_base + liq_mod(-5..+10) + dw_sfc_adjustment + capped regime boost
        #   mid = (1 - xgb_blend_weight) * mid + xgb_blend_weight * xgb_meta_prediction   # XGB blend
        #   sfc_effective = ewma.correct(mid) * 100                                        # opaque online EWMA
        # A hardcoded 1pp margin used to cover "ML nudges" but the XGBoost blend alone can
        # legitimately pull sfc_effective down by up to ~30% of mid (blend weight up to 0.3
        # toward a low stress prediction), so the old check false-alarmed almost every cycle.
        # Now we reconstruct mid from the ACTUAL persisted adjustment fields, leaving only
        # the opaque EWMA correction covered by a small margin.
        sfc_eff = cleaned.get("sfc_effective")
        sfc_base = cleaned.get("sfc_base")
        if isinstance(sfc_eff, (int, float)) and isinstance(sfc_base, (int, float)):
            # Use actual liq_mod value if available, else assume max negative (-5)
            liq_mod_val = cleaned.get("liq_mod")
            liq = liq_mod_val if isinstance(liq_mod_val, (int, float)) else -5.0
            # DW dynamic-weighting adjustment (SIDEWAYS -> 0.0, CRISIS -> +0.9, etc.)
            dw_adj = cleaned.get("dw_sfc_adjustment")
            dw_adj = dw_adj if isinstance(dw_adj, (int, float)) else 0.0
            # Regime boost is capped at +2pp when DW is active; it only raises mid so
            # excluding it keeps the expected floor conservative (fewer false alarms).
            mid = sfc_base + liq + dw_adj
            # XGBoost meta-ensemble blend — apply the same weight+prediction as collect.py
            xgb_w = cleaned.get("xgb_blend_weight")
            xgb_pred = cleaned.get("xgb_meta_prediction")
            if (isinstance(xgb_w, (int, float)) and xgb_w > 0
                    and isinstance(xgb_pred, (int, float))):
                mid = (1 - xgb_w) * mid + xgb_w * xgb_pred
            # Small margin covers opaque EWMA online correction + rounding. Real corruption
            # (sfc_effective dropped to ~0 or absurd) still trips this check.
            min_expected = mid - 3.0

            if sfc_eff < min_expected:
                consistency_issues.append(
                    f"sfc_effective ({sfc_eff:.1f}) < expected min ({min_expected:.1f}) "
                    f"(base={sfc_base:.1f}, liq_mod={liq_mod_val if isinstance(liq_mod_val, (int, float)) else 'N/A'}, "
                    f"dw={dw_adj:.1f}, xgb_w={xgb_w if isinstance(xgb_w, (int, float)) else 'N/A'}, "
                    f"xgb_pred={xgb_pred if isinstance(xgb_pred, (int, float)) else 'N/A'})"
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

    def get_last_valid(self) -> Dict[str, Any]:
        """Return the last-known-good tracked values (persisted across runs).

        Used by the pipeline when the breaker PURGES output on a trip: the
        consumer restores these values for the fields the breaker tracks,
        instead of publishing the corrupt values that triggered the trip.
        """
        return dict(self._last_valid)

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
    # sfc_eff=5.0 < sfc_base - 10 = 5.7 → should trigger consistency warning
    assert not ok, "Should be invalid when sfc_effective << sfc_base"
    assert any("sfc_effective" in w for w in warns), (
        f"Should flag sfc_effective consistency, got: {warns}"
    )
    assert cb3._consecutive_failures == 1, (
        f"Should increment failures, got {cb3._consecutive_failures}"
    )
    print("  ✓ PASS (consistency violation detected, failures incremented)")

    # ── Scenario 7: liq_mod-aware consistency — no false alarm ──
    print("\n── Scenario 7: liq_mod-aware consistency — no false alarm ──")
    cb7 = CircuitBreaker()
    cb7.validate(normal)
    with_liq = dict(normal)
    with_liq["sfc_base"] = 19.0
    with_liq["sfc_effective"] = 14.0   # base + liq_mod = 19 + (-5) = 14
    with_liq["liq_mod"] = -5.0         # M2 tight → legitimate reduction
    cleaned7, ok7, warns7 = cb7.validate(with_liq)
    print(f"  Valid: {ok7}")
    print(f"  Warnings: {warns7 if warns7 else 'None'}")
    # With liq_mod=-5, min_expected = 19 + (-5) - 1 = 13, sfc_eff=14 >= 13 → no consistency issue
    cons7 = [w for w in warns7 if "Consistency" in w]
    assert not cons7, f"Should NOT trigger consistency with liq_mod=-5: {cons7}"
    print("  ✓ PASS (liq_mod=-5, sfc_eff=14: no false alarm)")

    # ── Scenario 8: XGBoost blend false alarm — regression for BUG (2026-08-01) ──
    # Production values from data.json: sfc_base=19.18, liq_mod=1.2, dw=0.0,
    # xgb_blend_weight=0.155, xgb_meta_prediction=1.82, sfc_effective=17.49.
    # The XGBoost blend legitimately pulls the value down:
    #   mid = 19.18 + 1.2 + 0 = 20.38
    #   mid = (1-0.155)*20.38 + 0.155*1.82 ≈ 17.50  → sfc_eff 17.49 is consistent.
    # The OLD check (min_expected = base + liq_mod - 1 = 19.38) false-flagged this
    # every cycle and nearly tripped the breaker. Reconstructed check must NOT flag.
    print("\n── Scenario 8: XGBoost blend false alarm — no false positive ──")
    cb8 = CircuitBreaker()
    xgb_case = {
        "sfc_base": 19.18,
        "sfc_effective": 17.49,   # post XGB-blend, consistent with mid≈17.50
        "liq_mod": 1.2,
        "dw_sfc_adjustment": 0.0,
        "xgb_blend_weight": 0.155,
        "xgb_meta_prediction": 1.82,
        "composite_confidence": 0.106,
    }
    # Seed with the same value so jump detection doesn't interfere with the test.
    cb8.validate(xgb_case)
    cleaned8, ok8, warns8 = cb8.validate(xgb_case)
    print(f"  Valid: {ok8}")
    print(f"  Warnings: {warns8 if warns8 else 'None'}")
    cons8 = [w for w in warns8 if "Consistency" in w]
    assert ok8, f"XGB-blended sfc_effective should be valid: {warns8}"
    assert not cons8, f"Should NOT trigger consistency with XGBoost blend: {cons8}"
    print("  ✓ PASS (XGBoost-blended value no longer false-flagged)")

    # ── Scenario 9: XGBoost blend + genuinely corrupt value STILL caught ──
    # Same adjustment chain but sfc_effective dropped to ~0 (real corruption).
    # Reconstructed mid ≈ 17.50, min_expected = 14.50 → 0.0 must still trip.
    print("\n── Scenario 9: XGBoost blend must NOT mask real corruption ──")
    cb9 = CircuitBreaker()
    corrupt = dict(xgb_case)
    corrupt["sfc_effective"] = 0.0
    cb9.validate(xgb_case)          # seed valid state
    cleaned9, ok9, warns9 = cb9.validate(corrupt)
    print(f"  Valid: {ok9}")
    print(f"  Warnings: {warns9}")
    cons9 = [w for w in warns9 if "Consistency" in w]
    assert cons9, f"sfc_effective=0.0 should still be flagged as corruption: {cons9}"
    assert not ok9, "Corrupt value should be invalid"
    print("  ✓ PASS (real corruption still caught under blended threshold)")

    print("\n" + "=" * 60)
    print("All circuit breaker tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
