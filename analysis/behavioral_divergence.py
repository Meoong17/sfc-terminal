#!/usr/bin/env python3
"""
behavioral_divergence.py — Behavioral Divergence Detector (experimental,
display-only).

WHAT THIS DETECTS:
    A mismatch between BTC's price action and the DIRECTION of
    institutional/whale flow signals SFC already computes elsewhere
    (M81 ETF flow, Q10 whale pressure, SLI stablecoin liquidity):

    - Price DOWN + flow signals BULLISH  -> "HIDDEN_ACCUMULATION"
      (smart money appears to be buying while price looks weak —
      the classic "accumulation during weakness" pattern)
    - Price UP + flow signals BEARISH    -> "HIDDEN_DISTRIBUTION"
      (smart money appears to be selling while price looks strong —
      a potential distribution-into-strength warning)
    - Otherwise -> "NO_DIVERGENCE" (price and flow are pointing the
      same direction, or the signal isn't strong enough to call)

WHY THIS IS SAFE TO ADD (Option A — no new redundancy):
    This does NOT introduce any new raw data collection — it purely
    RE-COMBINES three signals (M81 ETF flow, Q10 whale pressure, SLI)
    that already separately feed into GLF/Q10/SLI and, through them,
    into factors/sfc_pct. Deliberately kept as a SEPARATE, display-only
    field (not folded into factors) for exactly the reason discussed
    when this was designed: re-adding these same signals a second time
    into the core ensemble would double-count them, the same mistake
    already found and fixed for netflow/M81-M82 elsewhere in this
    project. This module is a different LENS on existing signals
    (are they contradicting price?), not a new signal itself.

HONEST CAVEATS:
    - The 0.15 threshold for calling a "regime" is a deliberate starting
      guess, NOT validated against historical outcomes — same caveat as
      every other new threshold introduced in this project without a
      dedicated backtest. Treat the exact cutoff as provisional.
    - When M81 (ETF flow) is in its fallback/unavailable state (score
      exactly 0.5, m81_detail is null), it is EXCLUDED from the
      weighted average rather than contributing a fake "neutral 0"
      value — the remaining available components' weights are
      implicitly renormalized (simple mean of whatever IS available).
"""
import sys

DIVERGENCE_THRESHOLD = 0.15  # arbitrary starting point — see module
                              # docstring's honest caveat about this
                              # not being backtested/validated yet


def compute_behavioral_divergence(m81_etf_flow=None, m81_available=False,
                                   q10_whale_pressure=None, sli_score=None,
                                   btc_24h=None):
    """
    Args:
        m81_etf_flow: 0-1 scale ETF flow score from data.json's
            m81_etf_flow field (0.5 = neutral/fallback)
        m81_available: whether M81 has REAL data this cycle (not the
            0.5 fallback) — pass (m81_detail is not None) from collect.py
        q10_whale_pressure: 0-100 scale from data.json's q10_whale_pressure
        sli_score: 0-100 scale from data.json's sli_score
        btc_24h: raw % change, from data.json's btc_24h

    Returns:
        (divergence_score, detail)
        divergence_score: 0-100, higher = stronger divergence detected
        detail: dict with regime classification, component breakdown,
            and status — fails safe (score=0, status="insufficient_data")
            if too few components or btc_24h is missing.
    """
    components = {}

    if m81_available and m81_etf_flow is not None:
        components["etf_flow"] = (m81_etf_flow - 0.5) * 2  # -> [-1, +1]

    if q10_whale_pressure is not None:
        components["whale_pressure"] = (q10_whale_pressure - 50) / 50  # -> [-1, +1]

    if sli_score is not None:
        components["stablecoin"] = (sli_score - 50) / 50  # -> [-1, +1]

    if not components or btc_24h is None:
        return 0.0, {
            "status": "insufficient_data",
            "components_available": list(components.keys()),
        }

    flow_direction_score = sum(components.values()) / len(components)
    price_direction = 1 if btc_24h > 0 else -1 if btc_24h < 0 else 0

    # Positive divergence_raw = price and flow point in OPPOSITE
    # directions (genuine divergence). Negative = same direction
    # (aligned, not a divergence).
    divergence_raw = -price_direction * flow_direction_score
    divergence_raw = max(-1.0, min(1.0, divergence_raw))

    if divergence_raw > DIVERGENCE_THRESHOLD and price_direction < 0:
        regime = "HIDDEN_ACCUMULATION"
    elif divergence_raw > DIVERGENCE_THRESHOLD and price_direction > 0:
        regime = "HIDDEN_DISTRIBUTION"
    elif price_direction == 0:
        regime = "NO_SIGNAL"
    else:
        regime = "NO_DIVERGENCE"

    score = round(max(0.0, divergence_raw) * 100, 1)

    return score, {
        "status": "ok",
        "regime": regime,
        "flow_direction_score": round(flow_direction_score, 3),
        "price_direction": price_direction,
        "btc_24h": btc_24h,
        "components_used": list(components.keys()),
        "component_values": {k: round(v, 3) for k, v in components.items()},
    }


if __name__ == "__main__":
    print("=== Self-test: compute_behavioral_divergence() ===\n")

    print("--- Test 1: HIDDEN_ACCUMULATION (price down, flow bullish) ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=0.8, m81_available=True,
        q10_whale_pressure=75, sli_score=70,
        btc_24h=-3.5,
    )
    print(f"Score: {score}, Regime: {detail['regime']}")
    assert detail["regime"] == "HIDDEN_ACCUMULATION", f"FAIL: expected HIDDEN_ACCUMULATION, got {detail['regime']}"
    assert score > 0
    print("✅ PASS\n")

    print("--- Test 2: HIDDEN_DISTRIBUTION (price up, flow bearish) ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=0.2, m81_available=True,
        q10_whale_pressure=25, sli_score=30,
        btc_24h=2.8,
    )
    print(f"Score: {score}, Regime: {detail['regime']}")
    assert detail["regime"] == "HIDDEN_DISTRIBUTION", f"FAIL: expected HIDDEN_DISTRIBUTION, got {detail['regime']}"
    assert score > 0
    print("✅ PASS\n")

    print("--- Test 3: NO_DIVERGENCE (price down, flow also bearish — aligned) ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=0.2, m81_available=True,
        q10_whale_pressure=25, sli_score=30,
        btc_24h=-2.0,
    )
    print(f"Score: {score}, Regime: {detail['regime']}")
    assert detail["regime"] == "NO_DIVERGENCE", f"FAIL: expected NO_DIVERGENCE, got {detail['regime']}"
    assert score == 0.0, f"FAIL: aligned signals should score 0, got {score}"
    print("✅ PASS: sinyal selaras benar menghasilkan skor 0 (bukan divergence)\n")

    print("--- Test 4: M81 unavailable (fallback) — gracefully excluded ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=0.5, m81_available=False,  # fallback state
        q10_whale_pressure=80, sli_score=75,
        btc_24h=-4.0,
    )
    print(f"Score: {score}, Regime: {detail['regime']}, components used: {detail['components_used']}")
    assert "etf_flow" not in detail["components_used"], "FAIL: M81 fallback should be excluded, not counted as neutral"
    assert detail["regime"] == "HIDDEN_ACCUMULATION"
    print("✅ PASS: M81 fallback dikecualikan dengan benar, tetap hitung dari whale_pressure+stablecoin\n")

    print("--- Test 5: insufficient data (semua komponen kosong) ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=None, m81_available=False,
        q10_whale_pressure=None, sli_score=None,
        btc_24h=-2.0,
    )
    print(f"Score: {score}, Status: {detail['status']}")
    assert detail["status"] == "insufficient_data"
    assert score == 0.0
    print("✅ PASS: gagal aman ke status insufficient_data tanpa crash\n")

    print("--- Test 6: btc_24h hilang ---")
    score, detail = compute_behavioral_divergence(
        m81_etf_flow=0.8, m81_available=True,
        q10_whale_pressure=75, sli_score=70,
        btc_24h=None,
    )
    assert detail["status"] == "insufficient_data"
    print("✅ PASS: btc_24h hilang juga gagal aman\n")

    print("ALL SELF-TESTS PASSED")
