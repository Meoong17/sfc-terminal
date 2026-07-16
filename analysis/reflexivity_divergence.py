#!/usr/bin/env python3
"""
reflexivity_divergence.py — Reflexivity Divergence Score (experimental)
=========================================================================

Implements a SIMPLIFIED, discrete-derivative version of the Soros-style
reflexivity feedback loop described in the reference document:

    dP/dt = alpha*(P-F) + beta*L*sentiment   (price driven by P-F gap + leverage*sentiment)
    dL/dt = gamma*(P-F)                       (leverage rises as P-F gap widens)
    dF/dt = -delta*L                          (fundamental weakens as leverage builds)

WHY NOT FIT THE FULL ODE SYSTEM:
    Properly fitting alpha/beta/gamma/delta requires historical data
    spanning diverse market regimes (boom, bust, calm) — this project's
    own audit found a real accumulated data_collection.json window
    covering only 3.5 days with ZERO stress-labeled observations (100%
    "calm"), nowhere near enough diversity to fit ODE parameters
    meaningfully. Instead, this computes DISCRETE RATES OF CHANGE (finite
    differences) over a rolling window, and looks for the QUALITATIVE
    "reflexivity building" pattern (price rising + leverage rising +
    fundamental gap widening simultaneously) rather than claiming a
    calibrated dynamical model. This is explicitly a v1/experimental
    signal — see module status note in compute_reflexivity_divergence()'s
    returned detail dict.

DATA MAPPING (P, L, F — all from data already computed elsewhere in this
codebase, no new data source needed):
    P (price)       -> data.json's "btc" (raw BTC/USD price)
    F (fundamental) -> derived from Q10's mvrv_ratio.value (MVRV ratio;
                       MVRV > 1 means market price sits above realized/
                       cost-basis value — this project's best available
                       proxy for BTC "fundamental value", acknowledged as
                       inherently looser than a stock's earnings/cash-flow
                       based fundamental, since BTC has no equivalent)
    L (leverage)    -> data.json's Q10 open_interest.value (raw USD open
                       interest) — a direct, standard leverage proxy

ARCHITECTURE (consistent with the "Option A" pattern agreed for the
Behavioral Divergence Detector): this module computes an EXPERIMENTAL,
SEPARATE signal — collect.py wiring (when added) will expose it as its
own field, NOT fold it into factors/sfc_pct, until/unless it's validated
against live data over time.
"""
import json
import os
import sys
import time

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".reflexivity_rolling.json")
MAX_POINTS = 90  # ~90 cycles of history retained (exact real-world time span
                  # depends on how often this gets called — see collect.py
                  # wiring notes when integrated)
MIN_POINTS_FOR_SIGNAL = 14  # need at least this many points before computing
                             # a rate-of-change (arbitrary starting choice —
                             # NOT validated against real regime-transition
                             # data yet, same caveat as every other new
                             # threshold introduced in this project without
                             # live backtesting)


def _load_rolling_history():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_rolling_history(history):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(history, f)
    except OSError:
        pass


def update_reflexivity_history(price, mvrv, leverage):
    """
    Append the current (price, mvrv, leverage) snapshot to the rolling
    history and trim to MAX_POINTS. Call this once per collect.py cycle,
    BEFORE calling compute_reflexivity_divergence() so the just-appended
    point is included in the calculation.
    """
    history = _load_rolling_history()
    history.append({
        "ts": time.time(),
        "price": price,
        "mvrv": mvrv,
        "leverage": leverage,
    })
    history = history[-MAX_POINTS:]
    _save_rolling_history(history)
    return history


def compute_reflexivity_divergence(price=None, mvrv=None, leverage=None, force_history_update=True):
    """
    M-experimental — Reflexivity Divergence Score, 0-100.

    Args:
        price, mvrv, leverage: current values (from data.json's btc,
            q10_details.mvrv_ratio.value, q10_details.open_interest.value
            respectively) — pass None for any that are currently
            unavailable, the function fails safe rather than crashing.
        force_history_update: if True (default), appends the current
            snapshot to history before computing — set False if you've
            already called update_reflexivity_history() separately this
            cycle (avoids double-appending the same cycle's data point).

    Returns:
        (score, detail)
        score: 0-100, higher = stronger "reflexivity building" signal
            (price + leverage rising together while the fundamental gap
            widens — a boom-loop pattern per the reference document)
        detail: dict with the raw rate-of-change components, status, and
            an explicit "experimental" flag/note.
    """
    if force_history_update and price is not None and mvrv is not None and leverage is not None:
        history = update_reflexivity_history(price, mvrv, leverage)
    else:
        history = _load_rolling_history()

    if len(history) < MIN_POINTS_FOR_SIGNAL:
        return 50.0, {
            "status": "insufficient_history",
            "points_available": len(history),
            "points_needed": MIN_POINTS_FOR_SIGNAL,
            "experimental": True,
        }

    now = history[-1]
    past = history[-MIN_POINTS_FOR_SIGNAL]

    if not all([now.get("price"), now.get("mvrv"), now.get("leverage"),
                past.get("price"), past.get("mvrv"), past.get("leverage")]):
        return 50.0, {"status": "missing_fields_in_history", "experimental": True}

    price_roc = (now["price"] - past["price"]) / past["price"]
    leverage_roc = (now["leverage"] - past["leverage"]) / past["leverage"] if past["leverage"] else 0.0

    mvrv_gap_now = now["mvrv"] - 1.0
    mvrv_gap_past = past["mvrv"] - 1.0
    mvrv_gap_widening = mvrv_gap_now - mvrv_gap_past

    # "Reflexivity building" (boom-loop) pattern: price AND leverage both
    # rising together (coherent, not just one or the other by chance).
    # Scale factor (5.0) and amplification (1.3) below are DELIBERATE
    # starting guesses, NOT fit to real data — same honest caveat applied
    # to every other new threshold in this project (JGB/Bund/Gilt series
    # IDs, M1-M6 rescaling, etc.) until empirically validated.
    building_signal = 0.0
    if price_roc > 0 and leverage_roc > 0:
        magnitude = (price_roc + leverage_roc) / 2.0
        building_signal = min(1.0, magnitude * 5.0)
        if mvrv_gap_widening > 0:
            # fundamental gap ALSO widening while price+leverage rise —
            # amplify, since this is the specific pattern the reference
            # document describes (leverage up despite/because fundamental
            # not keeping pace)
            building_signal = min(1.0, building_signal * 1.3)

    score = round(building_signal * 100, 1)

    if score > 65:
        regime = "REFLEXIVITY_BUILDING"
    elif score > 35:
        regime = "MILD_DIVERGENCE"
    else:
        regime = "NO_SIGNAL"

    return score, {
        "status": "ok",
        "experimental": True,
        "price_roc": round(price_roc, 4),
        "leverage_roc": round(leverage_roc, 4),
        "mvrv_gap_now": round(mvrv_gap_now, 4),
        "mvrv_gap_widening": round(mvrv_gap_widening, 4),
        "regime": regime,
        "window_points": MIN_POINTS_FOR_SIGNAL,
    }


if __name__ == "__main__":
    print("=== Self-test: compute_reflexivity_divergence() ===\n")

    # Clean slate for the test
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)

    print("--- Test 1: insufficient history (first few calls) ---")
    for i in range(5):
        score, detail = compute_reflexivity_divergence(
            price=60000 + i * 10, mvrv=1.1, leverage=20_000_000_000
        )
    print(f"Score after 5 points: {score}, status: {detail['status']}")
    assert detail["status"] == "insufficient_history"
    print("✅ PASS: correctly reports insufficient history before MIN_POINTS_FOR_SIGNAL\n")

    print("--- Test 2: CALM scenario (price flat, leverage flat, mvrv flat) ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    for i in range(MIN_POINTS_FOR_SIGNAL + 2):
        score, detail = compute_reflexivity_divergence(
            price=60000, mvrv=1.1, leverage=20_000_000_000
        )
    print(f"Score (calm, no movement): {score}, regime: {detail.get('regime')}")
    assert score < 35, f"FAIL: calm scenario should score low, got {score}"
    print("✅ PASS: flat/calm scenario correctly scores low\n")

    print("--- Test 3: BOOM-BUILDING scenario (price up + leverage up + MVRV gap widening) ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    for i in range(MIN_POINTS_FOR_SIGNAL + 2):
        price = 60000 * (1 + 0.01 * i)       # price rising ~1%/cycle
        leverage = 20_000_000_000 * (1 + 0.015 * i)  # leverage rising ~1.5%/cycle
        mvrv = 1.0 + 0.02 * i                 # MVRV gap widening steadily
        score, detail = compute_reflexivity_divergence(price=price, mvrv=mvrv, leverage=leverage)
    print(f"Score (boom-building): {score}, regime: {detail.get('regime')}")
    print(f"Detail: price_roc={detail['price_roc']}, leverage_roc={detail['leverage_roc']}, "
          f"mvrv_gap_widening={detail['mvrv_gap_widening']}")
    assert score > 65, f"FAIL: boom-building scenario should score high, got {score}"
    print("✅ PASS: boom-building pattern correctly scores high\n")

    print("--- Test 4: price up but leverage FLAT (not coherent boom pattern) ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    for i in range(MIN_POINTS_FOR_SIGNAL + 2):
        price = 60000 * (1 + 0.02 * i)   # price rising fast
        leverage = 20_000_000_000         # leverage NOT rising at all
        mvrv = 1.1
        score, detail = compute_reflexivity_divergence(price=price, mvrv=mvrv, leverage=leverage)
    print(f"Score (price up, leverage flat): {score}, regime: {detail.get('regime')}")
    assert score < 35, f"FAIL: non-coherent pattern (price up alone) should NOT score high, got {score}"
    print("✅ PASS: price rising alone (without leverage confirming) correctly does NOT trigger signal\n")

    print("--- Test 5: missing data fails safe ---")
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    score, detail = compute_reflexivity_divergence(price=None, mvrv=None, leverage=None, force_history_update=False)
    print(f"Score with no data at all: {score}, status: {detail['status']}")
    assert score == 50.0
    print("✅ PASS: missing data fails safe to neutral 50.0, no crash\n")

    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
    print("ALL SELF-TESTS PASSED")
