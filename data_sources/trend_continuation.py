#!/usr/bin/env python3
"""
SFC Trend Continuation Probability (P3) — institutional output.
==============================================================
Reads the walk-forward-calibrated summary
(analysis/walk_forward_trend_continuation.py -> .trend_continuation_summary.json)
and reports the empirical probability that the BTC trend continues
(forward return > 0) over 30 / 90 / 180 days, CONDITIONED on today's
signal severity bucket.

This is a DISPLAY-ONLY research estimate. The walk-forward used a reduced
replay (price / DXY / M2 / FNG), so the number is a directional, calibrated
estimate — NOT the live full-90-method score's exact probability. Same
precedent as the IMBS STRESS=55 / L8 research cutoffs (not blended into
sfc_effective / signal / kelly).

Usage:
    from data_sources.trend_continuation import compute_trend_continuation
    out = compute_trend_continuation(sfc_effective=27.8, sfc_zone='ELEVATED')
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_CACHE_FILE = os.path.join(SFC_DIR, ".trend_continuation_summary.json")
_CACHE_FILE = os.path.join(SFC_DIR, ".trend_continuation_cache.json")
CACHE_TTL = 900  # 15 min

HORIZONS = [30, 90, 180]
# MUST match walk_forward_trend_continuation.py BUCKET_EDGES.
BUCKET_EDGES = [(0, 25, "CALM"), (25, 45, "ELEVATED"), (45, 101, "STRESS")]


def _bucket_label(sfc):
    for lo, hi, lbl in BUCKET_EDGES:
        if lo <= sfc < hi:
            return lbl
    return "STRESS"


def _load_summary():
    try:
        with open(SUMMARY_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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


def compute_trend_continuation(sfc_effective=None, sfc_zone=None):
    """Return continuation probabilities for today's signal bucket.

    Args:
        sfc_effective : live sfc_effective (0-100).
        sfc_zone      : optional zone label (NORMAL/CALM/ELEVATED/HIGH/CRITICAL).
                        If provided, used to disambiguate; otherwise derived
                        from sfc_effective via flat bucket edges.

    Returns (prob dict, details dict).
    """
    cached = _load_cache()
    now = time.time()
    _key = f"{sfc_effective}|{sfc_zone}"
    if (cached.get("key") == _key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("probs", {}), cached.get("details", {"status": "cached"})

    summary = _load_summary()
    if not summary:
        details = {"status": "unavailable", "available": False,
                   "reason": "No walk-forward summary cache. Run "
                             "analysis/walk_forward_trend_continuation.py first."}
        _save_cache({"probs": {}, "details": details, "ts": now, "key": _key})
        return {}, details

    # Map live signal to flat bucket. sfc_zone may be a live zone label
    # (NORMAL/ELEVATED/HIGH/CRITICAL); treat NORMAL as CALM, HIGH/CRITICAL as
    # STRESS, ELEVATED as ELEVATED. If no zone, derive from sfc_effective.
    if sfc_zone:
        z = str(sfc_zone).upper()
        if z in ("NORMAL", "CALM"):
            bucket = "CALM"
        elif z in ("HIGH", "CRITICAL", "STRESS", "BEAR", "CRISIS"):
            bucket = "STRESS"
        else:
            bucket = "ELEVATED"
    else:
        bucket = _bucket_label(sfc_effective) if sfc_effective is not None else None

    if bucket is None:
        details = {"status": "unavailable", "available": False,
                   "reason": "No sfc_effective or sfc_zone provided."}
        _save_cache({"probs": {}, "details": details, "ts": now, "key": _key})
        return {}, details

    probs = {}
    for h in HORIZONS:
        p = summary.get(f"{bucket.lower()}_p_cont_{h}d")
        ci = summary.get(f"{bucket.lower()}_p_cont_{h}d_ci")
        n = summary.get(f"{bucket.lower()}_n_{h}d")
        base = summary.get(f"baseline_p_cont_{h}d")
        probs[h] = {
            "probability": p,
            "ci": ci,
            "n": n,
            "baseline": base,
            "relative": (round(p - base, 3) if (p is not None and base is not None) else None),
        }

    details = {
        "status": "ok",
        "available": True,
        "bucket": bucket,
        "sfc_effective": sfc_effective,
        "sfc_zone": sfc_zone,
        "horizons": HORIZONS,
        "method": "Walk-forward calibrated P(forward return>0) per signal bucket "
                  "(reduced replay price/DXY/M2/FNG)",
        "caveat": "RESEARCH estimate, not the live full-90-method probability. "
                  "Not blended into signal.",
        "ts": now,
    }
    _save_cache({"probs": probs, "details": details, "ts": now, "key": _key})
    return probs, details


if __name__ == "__main__":
    import sys
    # Live-like: sfc_effective 27.8 -> ELEVATED bucket.
    for name, kw in (("LIVE(ELEVATED)", dict(sfc_effective=27.8, sfc_zone="ELEVATED")),
                     ("STRESS", dict(sfc_effective=55.0, sfc_zone="HIGH")),
                     ("CALM", dict(sfc_effective=18.0, sfc_zone="NORMAL")),
                     ("NO_DATA", dict())):
        probs, det = compute_trend_continuation(**kw)
        print(f"{name:16s} bucket={det.get('bucket')} available={det.get('available')}")
        for h in HORIZONS:
            v = probs.get(h)
            if v:
                print(f"    {h:4d}d P(cont)={v['probability']} "
                      f"baseline={v['baseline']} rel={v['relative']}")
