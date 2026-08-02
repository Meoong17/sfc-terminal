#!/usr/bin/env python3
"""
SFC Transmission Divergence (P1) — liquidity → BTC transmission status.
=======================================================================
Addresses the two most critical scenarios in the institutional framework
(analysis/gap_analysis_dokumen_vs_model.md §2, scenarios #2 and #3):

    #2  Liquidity macro RISES but BTC data WEAKENS → transmission gap /
        rotation to other assets.
    #3  Liquidity macro WEAKENS but BTC still RISES → divergence to watch.

The pipeline already computes GLF liquidity stress and the SFC composite
structural stress, but never compares their DIRECTION. This module turns
that implicit contradiction into an explicit, actionable status.

CONCEPT:
    - Liquidity direction from GLF: high `liquidity_stress` (0-1) = tight /
      illiquid (bad for BTC); low = loose/liquid (good for BTC).
    - BTC structural health from sfc_effective / a strength proxy: high
      structural stress = weak BTC.

    By comparing whether liquidity and BTC-structure AGREE (same direction)
    or DIVERGE, we classify the transmission state:

        STRENGTHENING      : liquid + BTC strong      (transmission working)
        TRANSMITTING       : neutral-but-aligned       (benign)
        TRANSMISSION_GAP   : liquid + BTC weak        (scenario #2 — money
                              not reaching BTC → gap / rotation)
        DIVERGENCE         : illiquid + BTC strong    (scenario #3 — BTC up
                              despite tightening → warning)
        DISTRESS_CASCADE   : illiquid + BTC weak      (scenario #4 — both weak)

This is DISPLAY-ONLY — it does NOT change sfc_effective / signal / kelly.

Usage:
    from data_sources.transmission_divergence import classify_transmission
    out = classify_transmission(
        liquidity_stress=0.30,   # 0-1 (GLF stress) or 0-100
        structural_stress=27.9,  # sfc_effective (0-100)
        btc_change_24h=+1.2,     # optional, sign check
    )
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".transmission_divergence_cache.json")
CACHE_TTL = 300  # 5 min


def _norm(v, is_pct=False):
    """Normalize to [0,1]. Accepts 0-1 or 0-100; None -> None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if is_pct or f > 1.0:  # 0-100 input
        return max(0.0, min(1.0, f / 100.0))
    return max(0.0, min(1.0, f))


def classify_transmission(liquidity_stress=None, structural_stress=None,
                          btc_change_24h=None):
    """Classify the liquidity→BTC transmission state.

    Args:
        liquidity_stress  : GLF stress (0-1 or 0-100). HIGH = illiquid/tight.
        structural_stress : sfc_effective (0-100). HIGH = weak BTC structure.
        btc_change_24h    : optional, for the real-time BTC sign.

    Returns (status, details dict).
    """
    cached = _load_cache()
    now = time.time()
    _key = "|".join(f"{x if x is None else round(float(x),3)}"
                    for x in (liquidity_stress, structural_stress, btc_change_24h))
    if (cached.get("key") == _key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("status", "UNKNOWN"), cached.get("details", {"status": "cached"})

    liq = _norm(liquidity_stress)             # high = illiquid
    struct = _norm(structural_stress, is_pct=True)  # high = weak

    if liq is None or struct is None:
        status, status_detail = "UNAVAILABLE", {
            "status": "unavailable", "available": False,
            "reason": "Missing liquidity_stress or structural_stress."}
        _save_cache({"status": status, "details": status_detail, "ts": now, "key": _key})
        return status, status_detail

    # Liquidity direction: stress low = liquid (loose); stress high = illiquid.
    liquid = liq < 0.40
    illiquid = liq >= 0.60
    mid_liq = not (liquid or illiquid)

    # BTC structure: low structural stress = strong; high = weak.
    btc_strong = struct < 0.40
    btc_weak = struct >= 0.60
    mid_btc = not (btc_strong or btc_weak)

    # Four-quadrant classification.
    if liquid and btc_strong:
        status = "STRENGTHENING"
        message = ("Loose liquidity + strong BTC structure — transmission working; "
                   "probable sustained uptrend.")
        tone = "good"
    elif liquid and btc_weak:
        status = "TRANSMISSION_GAP"
        message = ("Loose liquidity but weak BTC structure — money NOT reaching BTC. "
                   "Likely transmission gap or rotation to other assets (watch alts).")
        tone = "warn"
    elif illiquid and btc_strong:
        status = "DIVERGENCE"
        message = ("Tightening liquidity but BTC still strong — a divergence to "
                   "monitor; BTC rising on its own momentum, vulnerable to a catch-up.")
        tone = "warn"
    elif illiquid and btc_weak:
        status = "DISTRESS_CASCADE"
        message = ("Tight liquidity + weak BTC structure — both deteriorating; "
                   "higher probability of a trend change. Defensive.")
        tone = "bad"
    else:
        status = "TRANSMITTING"
        message = ("Liquidity and BTC structure broadly aligned (transitional zone) "
                   "— transmission functioning but not extreme either way.")
        tone = "neutral"

    # Confidence: strongest when both dimensions are in clear territory.
    edges = int(liquid or illiquid) + int(btc_strong or btc_weak)
    confidence = round(0.5 + 0.25 * edges, 2)
    if btc_change_24h is not None:
        sign = 1 if btc_change_24h >= 0 else -1
        # Real-time sign corroborates (or questions) the structural read.
        confidence = round(min(0.95, confidence + 0.10 * (sign > 0 and not btc_weak)
                               - 0.10 * (sign < 0 and not btc_strong)), 2)

    status_detail = {
        "status": "ok",
        "available": True,
        "status_label": status,
        "liquidity_stress": round(liq * 100, 1),
        "liquidity_state": "LIQUID" if liquid else "ILLIQUID" if illiquid else "NEUTRAL",
        "structural_stress": round(struct * 100, 1),
        "btc_state": "STRONG" if btc_strong else "WEAK" if btc_weak else "NEUTRAL",
        "message": message,
        "tone": tone,
        "confidence": confidence,
        "quadrant": {
            "liquidity": "LOW(loose)" if liquid else "HIGH(tight)" if illiquid else "MID",
            "btc_structure": "LOW(strong)" if btc_strong else "HIGH(weak)" if btc_weak else "MID",
        },
        "rule": "Quadrant of liquidity-direction vs BTC-structure-direction; "
                "display-only, not blended into signal.",
        "ts": now,
    }
    _save_cache({"status": status, "details": status_detail, "ts": now, "key": _key})
    return status, status_detail


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


if __name__ == "__main__":
    # Self-test covering all four quadrants + unavailable.
    cases = [
        ("STRENGTHENING",    dict(liquidity_stress=0.25, structural_stress=20.0, btc_change_24h=+1.5)),
        ("TRANSMISSION_GAP", dict(liquidity_stress=0.25, structural_stress=70.0, btc_change_24h=-0.5)),  # scenario #2
        ("DIVERGENCE",       dict(liquidity_stress=0.70, structural_stress=20.0, btc_change_24h=+2.0)),  # scenario #3
        ("DISTRESS_CASCADE", dict(liquidity_stress=0.70, structural_stress=70.0, btc_change_24h=-3.0)),  # scenario #4
        ("UNAVAILABLE",      dict(liquidity_stress=None, structural_stress=50.0)),
    ]
    for name, kw in cases:
        st, det = classify_transmission(**kw)
        print(f"{name:18s} -> {st:16s} tone={det.get('tone')} "
              f"conf={det.get('confidence')} | {det.get('message','')[:60]}")
