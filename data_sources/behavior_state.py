#!/usr/bin/env python3
"""
SFC Behavior-State Overlay (L5) — Market Participant Behavior State
===================================================================
Layer 5 of the IMBS / Macro-Intelligence blueprint. SFC/liquidity tell us
WHERE money is, but not HOW investors are behaving. This overlay maps
available market-participant signals onto the blueprint's five behavior
states:

    ACCUMULATION  — smart money buying while price weak / sentiment fearful
    EXPANSION     — healthy uptrend, improving sentiment, rising participation
    EUPHORIA      — extreme greed, crowded longs, aggressive funding
    DISTRIBUTION  — price high/strong but institutions selling (distribution
                    into strength)
    PANIC         — extreme fear, forced selling / cascade

WHAT THIS IS (honest scope):
    This is a RULE-BASED OVERLAY that RE-COMBINES signals the pipeline
    already computes (MPI positioning, FNG, behavioral divergence, ETF
    flow, whale pressure, HMM regime, cascade risk). It does NOT collect
    any new raw data and does NOT produce a new standalone signal. It is
    a different LENS — "what state is market behavior in?" — over existing
    features. Deliberately display-only: NOT blended into sfc_effective /
    signal / composite_confidence.

WHY SAFE (same rationale as behavioral_divergence.py):
    It only re-uses signals that already feed GLF/Q10/SLI/factors, without
    adding them a second time into the ensemble (which would double-count).
    It is exposed as its own field for observation before any decision to
    fold it into the core.

HONEST CAVEATS:
    - Thresholds (FNG 15/25/70/85, MPI bands, etc.) are deliberate starting
      guesses, NOT validated against historical outcomes — same provisional
      status as every other threshold in this project without a dedicated
      backtest. Treat cutoffs as provisional.
    - Rule priority matters; the state returned is the first matching rule.

Usage:
    from data_sources.behavior_state import compute_behavior_state
    state, details = compute_behavior_state(
        mpi_score=55.0, fng=50, cascade_risk=0.2,
        behavioral_divergence="NO_DIVERGENCE",
        etf_flow=0.6, whale_pressure=0.55, hmm_regime="SIDEWAYS",
    )
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".behavior_state_cache.json")
CACHE_TTL = 300  # 5 min — behavior inputs move intraday


def _f(v, dflt):
    """Coerce to float with default on None/invalid."""
    if v is None:
        return dflt
    try:
        return float(v)
    except (TypeError, ValueError):
        return dflt


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


def compute_behavior_state(mpi_score=None, fng=None, cascade_risk=None,
                           behavioral_divergence=None, etf_flow=None,
                           whale_pressure=None, hmm_regime=None):
    """Classify the current behavior state from available signals.

    Args (all optional — only supplied signals are used):
        mpi_score             : MPI positioning (0-100; low=bearish/neutral,
                                high=bullish/crowded long).
        fng                   : Fear & Greed (0-100).
        cascade_risk          : 0-1 liquidation cascade probability.
        behavioral_divergence : 'HIDDEN_ACCUMULATION' | 'HIDDEN_DISTRIBUTION'
                                | 'NO_DIVERGENCE' (from behavioral_divergence.py).
        etf_flow              : institutional ETF flow (0-1; high=bullish inflow).
        whale_pressure        : whale buying power (0-1; high=bullish).
        hmm_regime            : HMM regime label (BULL/BEAR/SIDEWAYS/CRISIS).

    Returns:
        (state str, details dict)
    """
    cached = _load_cache()
    now = time.time()
    _inp = (mpi_score, fng, cascade_risk, behavioral_divergence,
            etf_flow, whale_pressure, hmm_regime)
    _key = repr(_inp)
    if cached.get("key") == _key and cached.get("ts") and now - cached.get("ts", 0) < CACHE_TTL:
        return cached.get("state", "UNKNOWN"), cached.get("details", {"status": "cached"})

    mpi = _f(mpi_score, None)
    fngv = _f(fng, None)
    cas = _f(cascade_risk, None)
    div = str(behavioral_divergence or "").upper()
    etf = _f(etf_flow, None)
    whale = _f(whale_pressure, None)
    hmm = str(hmm_regime or "").upper()

    supplied = [k for k, v in {
        "mpi": mpi, "fng": fngv, "cascade": cas,
        "divergence": div, "etf": etf, "whale": whale, "hmm": hmm,
    }.items() if v is not None and v != ""]
    if not supplied:
        _save_cache({"state": "UNKNOWN", "details": {
            "status": "partial", "available": False,
            "reason": "No behavior signals supplied.",
        }, "ts": now, "key": _key})
        return "UNKNOWN", {"status": "partial", "available": False,
                           "reason": "No behavior signals supplied."}

    # Score-like evidence (higher = more bullish/euphoric tilt, lower = more bearish)
    bull_ev, bear_ev = [], []

    if mpi is not None:
        if mpi >= 75:
            bull_ev.append("mpi_crowded_long")
        elif mpi >= 60:
            bull_ev.append("mpi_bullish")
        elif mpi <= 25:
            bear_ev.append("mpi_bearish")
        elif mpi <= 40:
            bear_ev.append("mpi_weak")

    if fngv is not None:
        if fngv >= 85:
            bull_ev.append("fng_extreme_greed")
        elif fngv >= 70:
            bull_ev.append("fng_greed")
        elif fngv <= 15:
            bear_ev.append("fng_extreme_fear")
        elif fngv <= 25:
            bear_ev.append("fng_fear")

    if cas is not None:
        if cas >= 0.5:
            bear_ev.append("cascade_high")
        elif cas >= 0.25:
            bear_ev.append("cascade_elevated")

    if etf is not None:
        if etf >= 0.65:
            bull_ev.append("etf_inflow")
        elif etf <= 0.35:
            bear_ev.append("etf_outflow")

    if whale is not None:
        if whale >= 0.65:
            bull_ev.append("whale_buying")
        elif whale <= 0.35:
            bear_ev.append("whale_selling")

    if hmm in ("BEAR", "CRISIS"):
        bear_ev.append(f"hmm_{hmm}")
    elif hmm == "BULL":
        bull_ev.append("hmm_bull")

    # ── Rule priority (first match wins) ──
    # 1. PANIC: cascade crisis dominates
    if "cascade_high" in bear_ev or hmm == "CRISIS":
        state = "PANIC"
    # 2. EUPHORIA: extreme greed + crowded longs + inflows (but no distribution flag)
    elif (("fng_extreme_greed" in bull_ev or "mpi_crowded_long" in bull_ev)
          and div != "HIDDEN_DISTRIBUTION"):
        state = "EUPHORIA"
    # 3. DISTRIBUTION: institutions selling into strength (price high but flow bearish)
    elif (div == "HIDDEN_DISTRIBUTION"
          or ("etf_outflow" in bear_ev and "mpi_bullish" in bull_ev)
          or ("whale_selling" in bear_ev and "fng_greed" in bull_ev)):
        state = "DISTRIBUTION"
    # 4. ACCUMULATION: smart money buying into weakness
    elif (div == "HIDDEN_ACCUMULATION"
          or ("etf_inflow" in bull_ev and ("fng_fear" in bear_ev or "fng_extreme_fear" in bear_ev))
          or ("whale_buying" in bull_ev and ("fng_fear" in bear_ev or "fng_extreme_fear" in bear_ev))):
        state = "ACCUMULATION"
    # 5. PANIC (secondary): broad fear
    elif len(bear_ev) >= 3 and not bull_ev:
        state = "PANIC"
    # 6. EXPANSION: healthy bull (bullish evidence, minimal bear)
    elif len(bull_ev) >= 1 and len(bear_ev) <= 1:
        state = "EXPANSION"
    # 7. fallback
    else:
        state = "ACCUMULATION" if len(bull_ev) > len(bear_ev) else \
                "DISTRIBUTION" if len(bear_ev) > len(bull_ev) else \
                "EXPANSION"

    details = {
        "status": "ok",
        "available": True,
        "state": state,
        "bullish_evidence": bull_ev,
        "bearish_evidence": bear_ev,
        "inputs_supplied": supplied,
        "mpi": round(mpi, 1) if mpi is not None else None,
        "fng": fngv,
        "cascade_risk": round(cas, 3) if cas is not None else None,
        "behavioral_divergence": div or None,
        "etf_flow": round(etf, 3) if etf is not None else None,
        "whale_pressure": round(whale, 3) if whale is not None else None,
        "hmm_regime": hmm or None,
        "caveat": "Rule-based display-only overlay. Thresholds provisional (not walk-forward validated). "
                  "NOT blended into sfc_effective/signal.",
        "ts": now,
    }
    _save_cache({"state": state, "details": details, "ts": now, "key": _key})
    return state, details


if __name__ == "__main__":
    cases = [
        ("panic", dict(cascade_risk=0.6, fng=10)),
        ("euphoria", dict(mpi_score=80, fng=90, etf_flow=0.8, behavioral_divergence="NO_DIVERGENCE")),
        ("distribution", dict(mpi_score=70, fng=75, etf_flow=0.2, behavioral_divergence="NO_DIVERGENCE")),
        ("accumulation", dict(fng=20, etf_flow=0.7, behavioral_divergence="NO_DIVERGENCE")),
        ("expansion", dict(fng=55, mpi_score=55, hmm_regime="BULL")),
        ("no-input", dict()),
    ]
    for name, kw in cases:
        s, d = compute_behavior_state(**kw)
        print(f"{name:14s} -> {s}")
