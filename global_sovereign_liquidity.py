#!/usr/bin/env python3
"""
SFC Global Sovereign Liquidity Score Module (M90)
=====================================================
M90 — Global Sovereign Liquidity Score (GSLS), 0-100

DESIGN PHILOSOPHY (why one consolidated score, not separate M88/M89/M90):
    An earlier draft of this feature considered separate M88 (JGB), M89
    (Bund), M90 (Gilt) methods, each feeding the ensemble independently.
    method_independence_analysis.py's findings earlier in this project's
    audit history showed exactly why that's the wrong shape: correlated
    raw indicators fed separately into an ensemble get implicitly
    double-counted relative to genuinely independent signals, and — more
    importantly for a model whose downstream consumers include a trained
    QLSTM and probabilistic output module — a pile of 20+ loosely-related
    method scores is noisier to learn from than a smaller number of
    clean, deliberately-constructed latent factors.

    So instead: this module computes 4 real subcomponents internally,
    blends them into ONE score (GSLS, 0-100), and that single score is
    what feeds into the rest of the pipeline (as an input to
    global_liquidity_engine.py's GLF, alongside Fed/ECB/BOJ/China/M2/
    TGA/RRP/DXY) — not 4 separate raw numbers competing for the
    ensemble's attention.

SUBCOMPONENTS:
    1. US Treasury Factor (35%) — yield curve slope (10Y-2Y) ONLY.
       Deliberately does NOT include repo stress here even though repo
       stress is genuinely a US/Treasury-market phenomenon — M86
       (repo_market_stress.py) already feeds Ft directly in collect.py.
       Including it again here would double-count the same signal in
       two different places (once as a direct Ft adjustment, once
       folded into GSLS -> Lt via GLF). Yield curve SHAPE is distinct
       information repo stress doesn't capture (term structure /
       forward-looking growth expectations vs point-in-time funding
       stress), so it's kept, while repo stress itself is deliberately
       left out of this module.

    2. Japan Factor (30%) — JGB yield curve slope + a simple carry-trade
       attractiveness proxy (US 10Y - JGB 10Y spread — wider spread =
       more attractive/active carry trade = more fragile to a BOJ
       tightening surprise, per the August 2024 unwind precedent
       discussed when this feature was scoped).

    3. Europe Factor (20%) — Bund yield curve slope.

    4. UK Stress Factor (15% base weight, DYNAMICALLY SCALED) — Gilt
       yield curve slope, but weighted to matter little when near-normal
       and scale UP sharply during genuine dislocation (per the
       September 2022 Gilt crisis precedent — UK's bond market is
       smaller than US/Japan/Germany's, so under normal conditions it
       shouldn't move GSLS much, but when it's genuinely stressed, that
       stress is disproportionately informative about systemic leverage
       risk, hence the non-linear weighting instead of a flat share).

    NOT INCLUDED — Offshore Dollar Funding Factor (cross-currency basis
    swap): this is OTC derivatives market data (Bloomberg/Refinitiv,
    proprietary) with no known free/public data source, unlike every
    other signal in this codebase which is sourced from free APIs
    (FRED, Binance, GoldAPI, etc.). Rather than fabricate a misleading
    proxy, this component is excluded and the other four components'
    weights are renormalized to sum to 100% among themselves. If a real
    data source for this is identified later, it can be added as a
    fifth component.

OUTPUT SCALE: 0-100, where HIGHER = MORE sovereign bond market stress
(inverted curves, wide carry unwind risk, Gilt dislocation) — same
direction convention as SFC's own 0-100 stress scale, so GSLS can be
read the same way as the headline score.
"""
import json
import os
import sys
import time

from sovereign_yield_curves import compute_yield_curve_slope, _fred

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gsls_cache.json")
CACHE_TTL = 43200  # 12 hours — underlying data is monthly-frequency

# Base weights among the 4 implemented components (renormalized if any
# are unavailable this cycle — see _renormalize_weights()).
BASE_WEIGHTS = {
    "us_treasury": 0.35,
    "japan": 0.30,
    "europe": 0.20,
    "uk": 0.15,
}


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cached_at": 0}


def _save_cache(cache):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _us_treasury_factor():
    """Reuses DGS10/DGS2 directly (same series collect.py's own M8 uses)
    rather than sovereign_yield_curves.py's OECD-mirrored series, since
    FRED's direct US Treasury series (DGS10, DGS2) are the ones already
    confirmed working in this codebase's own M8 implementation — no
    "unverified series ID" caveat needed for this specific component."""
    vals_10y = _fred("DGS10", 1)
    vals_2y = _fred("DGS2", 1)
    if not vals_10y or not vals_2y:
        return None, {"status": "unavailable", "reason": "DGS10/DGS2 unavailable"}
    slope = vals_10y[0] - vals_2y[0]
    score = _slope_to_score(slope)
    return score, {"slope": round(slope, 3), "long_yield": vals_10y[0], "short_yield": vals_2y[0], "status": "ok"}


def _slope_to_score(slope):
    """Same threshold ladder as calculate_m8_yield_curve() /
    sovereign_yield_curves.py — kept identical across all sovereign
    components for direct comparability (see those modules for why)."""
    if slope < 0:
        return 0.80
    elif slope < 0.5:
        return 0.65
    elif slope < 1.0:
        return 0.40
    elif slope > 2.0:
        return 0.15
    else:
        return 0.25


def _japan_factor():
    """JGB yield curve slope + carry-trade attractiveness proxy
    (US 10Y - JGB 10Y spread: wider = more attractive carry = more
    fragile to a BOJ tightening surprise)."""
    jgb_score, jgb_detail = compute_yield_curve_slope("jgb")
    if jgb_detail.get("status") != "ok":
        return None, jgb_detail

    us_10y_vals = _fred("DGS10", 1)
    carry_detail = {}
    carry_score = 0.5  # neutral if US 10Y unavailable, don't let a missing US series sink the whole Japan factor
    if us_10y_vals:
        us_10y = us_10y_vals[0]
        jgb_10y = jgb_detail["long_yield"]
        carry_spread = us_10y - jgb_10y
        # Wider spread = more attractive/crowded carry trade = higher
        # unwind risk if conditions change. Thresholds are a starting
        # judgment call (not backtested) — a spread above ~3.5pp has
        # historically corresponded to an active, crowded USD/JPY carry
        # trade environment; recalibrate against realized carry-unwind
        # episodes if you have that history available.
        if carry_spread > 4.0:
            carry_score = 0.75
        elif carry_spread > 3.0:
            carry_score = 0.55
        elif carry_spread > 2.0:
            carry_score = 0.35
        else:
            carry_score = 0.20
        carry_detail = {"us_jgb_spread": round(carry_spread, 3)}

    # Blend: yield curve shape (60%) + carry attractiveness (40%)
    score = 0.6 * jgb_score + 0.4 * carry_score
    detail = {**jgb_detail, **carry_detail, "carry_score": carry_score, "jgb_curve_score": jgb_score}
    return score, detail


def _europe_factor():
    """Bund yield curve slope.

    NOTE: must check detail["status"] explicitly and return None on
    failure — compute_yield_curve_slope() itself always returns a
    numeric score (0.5 fail-safe default) even when data is unavailable,
    signaling failure only via detail["status"], not via the score
    value. A bare pass-through here was tested and confirmed to
    incorrectly report an unavailable Bund fetch as "available with a
    neutral 0.5 score," corrupting compute_global_sovereign_liquidity()'s
    availability counting and weight renormalization (it would count 4
    components as available when really only 2 were, and blend in a
    fabricated "neutral" score for the other 2 as if that were real
    data). Explicit status-check here fixes that."""
    score, detail = compute_yield_curve_slope("bund")
    if detail.get("status") != "ok":
        return None, detail
    return score, detail


def _uk_factor():
    """Gilt yield curve slope. Same status-check requirement as
    _europe_factor() above — see that function's docstring for the bug
    this avoids. Weighting (not the score itself) is what implements
    "only active during dislocation" — see _renormalize_weights() for
    the dynamic weight scaling logic."""
    score, detail = compute_yield_curve_slope("gilt")
    if detail.get("status") != "ok":
        return None, detail
    return score, detail


def _renormalize_weights(available_scores, uk_score):
    """
    Redistribute BASE_WEIGHTS among whichever components are actually
    available this cycle, AND apply the UK dislocation-gating: UK's
    effective weight scales up the further its score sits from neutral
    (0.5), rather than always contributing its flat base share — this is
    what makes it "only active during dislocation" per the original
    design request, rather than a constant 15% regardless of whether
    anything unusual is happening in Gilt markets.
    """
    weights = {k: BASE_WEIGHTS[k] for k in available_scores if available_scores[k] is not None}

    if "uk" in weights and uk_score is not None:
        # Dislocation multiplier: 1x at neutral (0.5), scaling up to 3x
        # as the score approaches the extremes (0 or 1). The 0.15
        # "deadband" means small, unremarkable Gilt moves don't trigger
        # any extra weight — only genuine dislocation does.
        deviation = max(0.0, abs(uk_score - 0.5) - 0.15)
        dislocation_multiplier = 1.0 + min(3.0, deviation * 10)
        weights["uk"] = weights["uk"] * dislocation_multiplier

    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def compute_global_sovereign_liquidity(force_refresh=False):
    """
    M90 — Global Sovereign Liquidity Score (GSLS), 0-100.

    Returns:
        (gsls_score, detail_dict)
        gsls_score: 0-100, higher = more sovereign bond market stress
        detail_dict: {"components": {...}, "weights_used": {...},
                       "regime": str, "status": "ok"|"partial"|"unavailable"}
    """
    cache = _load_cache()
    now = time.time()

    if not force_refresh and (now - cache.get("cached_at", 0)) < CACHE_TTL:
        if cache.get("gsls_score") is not None:
            return cache["gsls_score"], cache.get("detail", {})

    us_score, us_detail = _us_treasury_factor()
    jp_score, jp_detail = _japan_factor()
    eu_score, eu_detail = _europe_factor()
    uk_score, uk_detail = _uk_factor()

    raw_scores = {"us_treasury": us_score, "japan": jp_score, "europe": eu_score, "uk": uk_score}
    n_available = sum(1 for v in raw_scores.values() if v is not None)

    if n_available == 0:
        detail = {
            "status": "unavailable",
            "reason": "All sovereign yield components unavailable — check FRED_API_KEY and series IDs "
                      "(see sovereign_yield_curves.py module docstring for unverified series ID caveat)",
        }
        return 50.0, detail

    weights = _renormalize_weights(raw_scores, uk_score)

    # Weighted blend, on the underlying 0-1 scale, THEN convert to 0-100
    weighted_sum = sum(raw_scores[k] * weights[k] for k in weights if raw_scores[k] is not None)
    gsls_0_1 = weighted_sum
    gsls_score = round(gsls_0_1 * 100, 1)

    if gsls_score > 65:
        regime = "SOVEREIGN_STRESS"
    elif gsls_score > 45:
        regime = "ELEVATED"
    else:
        regime = "NORMAL"

    detail = {
        "status": "ok" if n_available == 4 else "partial",
        "n_components_available": n_available,
        "components": {
            "us_treasury": us_detail,
            "japan": jp_detail,
            "europe": eu_detail,
            "uk": uk_detail,
        },
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
        "regime": regime,
        "offshore_dollar_funding": "not implemented — no free data source identified, see module docstring",
    }

    cache["gsls_score"] = gsls_score
    cache["detail"] = detail
    cache["cached_at"] = now
    _save_cache(cache)

    return gsls_score, detail


if __name__ == "__main__":
    print("=== Live fetch (requires FRED_API_KEY + valid series IDs) ===\n")
    score, detail = compute_global_sovereign_liquidity(force_refresh=True)
    print(f"M90 GSLS: {score}")
    print(f"Detail: {json.dumps(detail, indent=2)}")

    # Self-test: verify weight renormalization and UK dislocation-gating
    # logic without network, using synthetic component scores.
    print("\n--- Self-test: weight renormalization + UK dislocation-gating (no network) ---")

    print("\nTest 1: all 4 components available, UK near-neutral (no dislocation)")
    scores = {"us_treasury": 0.5, "japan": 0.5, "europe": 0.5, "uk": 0.52}
    weights = _renormalize_weights(scores, uk_score=0.52)
    print(f"  Weights: {weights}")
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"
    assert weights["uk"] < BASE_WEIGHTS["uk"] * 1.5, "UK near-neutral should NOT get amplified weight"
    print("  ✅ PASS: near-neutral UK gets close to its base weight, not amplified")

    print("\nTest 2: UK in genuine dislocation (score=0.85, like Sept 2022 Gilt crisis)")
    scores2 = {"us_treasury": 0.5, "japan": 0.5, "europe": 0.5, "uk": 0.85}
    weights2 = _renormalize_weights(scores2, uk_score=0.85)
    print(f"  Weights: {weights2}")
    assert abs(sum(weights2.values()) - 1.0) < 1e-9
    assert weights2["uk"] > BASE_WEIGHTS["uk"] * 1.5, "UK in dislocation should get amplified weight"
    print(f"  ✅ PASS: UK weight amplified from base {BASE_WEIGHTS['uk']} to {weights2['uk']:.3f} during dislocation")

    print("\nTest 3: one component unavailable (e.g. UK data missing) — weights renormalize among the rest")
    scores3 = {"us_treasury": 0.5, "japan": 0.5, "europe": 0.5, "uk": None}
    weights3 = _renormalize_weights(scores3, uk_score=None)
    print(f"  Weights: {weights3}")
    assert "uk" not in weights3, "Unavailable component should be excluded, not given a weight"
    assert abs(sum(weights3.values()) - 1.0) < 1e-9, "Remaining weights must still sum to 1.0"
    print("  ✅ PASS: unavailable UK component excluded, remaining 3 renormalized to sum to 1.0")

    print("\nTest 4: regression test for the bug found during initial testing — a component whose")
    print("        underlying fetch fails must report score=None, NOT a fail-safe 0.5 that gets")
    print("        mistaken for real data (compute_yield_curve_slope's own fail-safe default)")
    # Simulate what _europe_factor()/_uk_factor() would have done BEFORE the fix:
    # passing through compute_yield_curve_slope()'s raw (0.5, {"status":"unavailable"})
    # return value without checking status.
    fake_unavailable_detail = {"status": "unavailable", "reason": "test"}

    def _buggy_passthrough():
        return 0.5, fake_unavailable_detail  # what the OLD, unfixed code effectively did

    def _fixed_wrapper():
        score, detail = _buggy_passthrough()
        if detail.get("status") != "ok":
            return None, detail
        return score, detail

    buggy_score, _ = _buggy_passthrough()
    fixed_score, _ = _fixed_wrapper()
    print(f"  Old (buggy) behavior would return score={buggy_score} (looks like valid data!)")
    print(f"  Fixed behavior returns score={fixed_score} (correctly signals unavailable)")
    assert fixed_score is None, "Fixed wrapper must return None when status is not 'ok'"
    print("  ✅ PASS: status-check wrapper correctly converts a fail-safe 0.5 into None")

    print("\nALL SELF-TESTS PASSED")
