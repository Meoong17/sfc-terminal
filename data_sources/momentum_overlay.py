#!/usr/bin/env python3
"""
SFC Regime-Gated Momentum Overlay (P4) — institutional output.
===============================================================
Turns raw momentum into a DEFENSIBLE, regime-conditional read by GATING it
on the signal-regime bucket and the walk-forward continuation evidence.

WHY THIS IS NEEDED (honest framing):
    Unconditional BTC momentum has NO reliable OOS edge — purged-CV
    (analysis/purged_cv_momentum.py, docs/feature_validation.md, 2026-08-08)
    shows mom_30 pooled AUC 0.520@7d / 0.413@30d / 0.371@90d (below chance),
    because overlapping forward-return labels inflate the effective sample
    size. So we do NOT blend raw momentum into scoring.

    BUT the walk-forward continuation replay (.trend_continuation_era.json)
    shows momentum is regime-CONDITIONAL and era-stable in a specific way:
        - CALM / ELEVATED (era3): p_cont_90d = 0.56-0.68  (>baseline)
          → trend/momentum is VALID, follow it.
        - STRESS (era3):          p_cont_90d = 0.383, p_cont_180d = 0.459
          (well below baseline/0.5) → continuation BREAKS and REVERSES
          → fade / act contrarian to momentum.
    This overlay makes that regime-dependence explicit and DISPLAY-ONLY
    (not blended into sfc_effective / signal / kelly_fraction).

RULE (transparent, not an opaque model):
    momentum_direction = sign of (trend_strength.momentum_domain - 0.5)
        > 0.55 bullish | < 0.45 bearish | else neutral.
    regime_action      = margin-weighted vote across horizons of the
        LATEST-ERA (era3) continuation evidence:
            margin_h = era3_p_cont_h - baseline_h
            positive margin  -> supports FOLLOW (continuation holds)
            negative margin  -> supports FADE   (continuation fails/reverses)
        Each horizon's vote is weighted by |margin| and by era-stability
        (era_stable True => weight x1.0, absent/False => x0.5). The bucket
        (CALM/ELEVATED/STRESS) is the conditioning variable; the action is
        read from the era3 margins themselves, which already reflect regime
        behaviour:
            ELEVATED (era3) -> strong FOLLOW;  STRESS (era3) -> strong FADE;
            CALM (era3, eroded 30/90d) -> FADE.
        A neutral momentum read yields a NEUTRAL overlay regardless of regime.

    gated_score (0-100) = 50 + gate_sign * momentum_strength_sign * confidence * 50
        gate_sign = +1 FOLLOW (preserve momentum), -1 FADE (contrarian).
        > 50 = bullish gated momentum, < 50 = bearish, |offset| = confidence.

DISPLAY-ONLY / SAFE: introduces NO new raw data, NO new scoring input. It
RE-COMBINES two existing display signals (trend strength momentum domain +
trend continuation buckets) under a regime gate. Same cautious-rollout
pattern as trend_strength / trend_continuation.

Usage:
    from data_sources.momentum_overlay import compute_momentum_overlay
    score, detail = compute_momentum_overlay(
        momentum_strength=0.76, bucket='ELEVATED', cont_probs={...},
    )
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".momentum_overlay_cache.json")
CACHE_TTL = 900  # 15 min — matches trend_continuation

HORIZONS = [30, 90, 180]
MOM_BULL = 0.55   # momentum domain > this => bullish trend read
MOM_BEAR = 0.45   # momentum domain < this => bearish trend read


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


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


def _era_p(cont_probs, h, era3_only=True):
    """Latest-era (era3) continuation probability + baseline + era_stable."""
    v = cont_probs.get(h) or {}
    if era3_only:
        p = v.get("era3_probability")
    else:
        p = v.get("probability")
    return {
        "p": p,
        "baseline": v.get("baseline"),
        "era_stable": v.get("era_stable"),
        "relative": v.get("relative"),
    }


def _probs_sig(cont_probs, era3_only=True):
    """Stable signature of the continuation probs so the cache invalidates when
    the walk-forward summary changes (era3 values move), not just on momentum/
    bucket. era3 data is slow-moving, so misses are rare; this guards against
    serving stale gating after a summary refresh."""
    parts = []
    for h in (30, 90, 180):
        v = cont_probs.get(h) or {}
        p = v.get("era3_probability" if era3_only else "probability")
        b = v.get("baseline")
        s = v.get("era_stable")
        parts.append(f"{h}:{p}:{b}:{s}")
    return "|".join(parts)


def compute_momentum_overlay(momentum_strength=None, bucket=None,
                             cont_probs=None, era3_only=True):
    """Compute the regime-gated momentum overlay.

    Args:
        momentum_strength : 0-1 momentum-domain value from trend_strength
                            (high = bullish trend read).
        bucket            : regime bucket 'CALM' / 'ELEVATED' / 'STRESS'
                            (from trend_continuation details).
        cont_probs        : dict of continuation probs from
                            compute_trend_continuation() (keyed by horizon).
        era3_only         : use latest-era (era3) probabilities when present
                            (default True) — honest "today" read, avoids the
                            era1-inflated full-sample values.

    Returns (gated_score 0-100, detail dict).
    """
    cached = _load_cache()
    now = time.time()
    _key = f"{momentum_strength}|{bucket}|{era3_only}|{_probs_sig(cont_probs, era3_only)}"
    if (cached.get("key") == _key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("score", 50.0), cached.get("detail", {"status": "cached"})

    if momentum_strength is None or bucket is None:
        detail = {"status": "unavailable", "available": False,
                  "reason": "Need momentum_strength and bucket."}
        _save_cache({"score": 50.0, "detail": detail, "ts": now, "key": _key})
        return 50.0, detail

    bucket = str(bucket).upper()
    cont_probs = cont_probs or {}

    # 1) momentum direction from the trend-strength momentum domain
    if momentum_strength >= MOM_BULL:
        momentum_dir = +1.0
        momentum_bias = "BULLISH_MOMENTUM"
    elif momentum_strength <= MOM_BEAR:
        momentum_dir = -1.0
        momentum_bias = "BEARISH_MOMENTUM"
    else:
        momentum_dir = 0.0
        momentum_bias = "NEUTRAL"

    # A neutral momentum read has nothing to gate — report neutral regardless of
    # regime. (Kept for honesty; a no-directional-momentum day is not a signal.)
    if momentum_dir == 0.0:
        detail = {
            "status": "ok", "available": True,
            "gated_score": 50.0, "gated_bias": "NEUTRAL",
            "momentum_strength": round(momentum_strength, 3),
            "momentum_bias": "NEUTRAL", "bucket": bucket,
            "regime_action": "NEUTRAL",
            "action_reason": "No directional momentum (momentum domain near "
                             "0.5) — nothing to gate.",
            "confidence": 0.0, "era3_only": era3_only,
            "horizon_evidence": [],
            "caveat": ("DISPLAY-ONLY research overlay — regime-gated momentum. "
                       "NOT blended into signal / scoring."),
            "ts": now,
        }
        _save_cache({"score": 50.0, "detail": detail, "ts": now, "key": _key})
        return 50.0, detail

    # 2) regime evidence per horizon (era3 probabilities when present)
    horizon_ev = []
    for h in HORIZONS:
        e = _era_p(cont_probs, h, era3_only=era3_only)
        if e["p"] is None or e["baseline"] is None:
            horizon_ev.append({"horizon": h, "p_cont": None,
                               "baseline": None, "support": None,
                               "era_stable": e["era_stable"]})
            continue
        support = "FOLLOW" if e["p"] > e["baseline"] else \
                  ("FADE" if e["p"] < e["baseline"] else "NEUTRAL")
        horizon_ev.append({
            "horizon": h, "p_cont": round(e["p"], 3),
            "baseline": round(e["baseline"], 3),
            "margin": round(e["p"] - e["baseline"], 3),
            "support": support, "era_stable": e["era_stable"],
        })

    # 3) regime action + confidence — margin-weighted vote across horizons.
    #    Each horizon's margin = era3 p_cont - baseline (latest era). A positive
    #    margin supports FOLLOW (continuation holds), negative supports FADE
    #    (continuation fails/reverses). era_stable True raises a horizon's vote
    #    weight; absent/False means its era3 value is less trustworthy, so it is
    #    down-weighted. This lets STRESS (era_stable flags often absent) still
    #    contribute its strong era3 reversal signal instead of collapsing to
    #    neutral, while respecting era-stability where it is available.
    follow_w = 0.0
    fade_w = 0.0
    for e in horizon_ev:
        if e["p_cont"] is None or e["baseline"] is None:
            continue
        stability = 1.0 if e.get("era_stable") else 0.5  # absent/False = uncertain
        w = abs(e["margin"]) * stability
        if e["margin"] > 0:
            follow_w += w
        elif e["margin"] < 0:
            fade_w += w

    total_w = follow_w + fade_w
    if total_w <= 0:
        action = "NEUTRAL"
        confidence = 0.0
        action_reason = "No era3 continuation evidence (margins all zero or missing)."
        gate_sign = 0.0
    elif follow_w > fade_w:
        action = "FOLLOW"
        confidence = follow_w / total_w
        action_reason = (f"{bucket} regime: era3 continuation holds (p_cont > "
                         "baseline on weighted horizon vote). Follow momentum.")
        gate_sign = +1.0
    elif fade_w > follow_w:
        action = "FADE"
        confidence = fade_w / total_w
        action_reason = (f"{bucket} regime: era3 continuation fails/reverses "
                         "(p_cont < baseline on weighted horizon vote). "
                         "Fade momentum (contrarian).")
        gate_sign = -1.0  # invert momentum
    else:
        action = "NEUTRAL"
        confidence = 0.0
        action_reason = "Era3 continuation evidence evenly split — no clean gate."
        gate_sign = 0.0

    # 4) gated score. `strength` is SIGNED (-1..+1: bearish..bullish momentum),
    #    so one gate_sign multiplier is all that is needed:
    #      follow -> preserve sign (momentum valid), fade -> invert (contrarian).
    strength = _clamp((momentum_strength - 0.5) * 2.0, -1.0, 1.0)
    gated_score = round(_clamp(50.0 + gate_sign * strength * confidence * 50.0,
                               lo=0.0, hi=100.0), 1)

    if gated_score > 55:
        gated_bias = "BULLISH"
    elif gated_score < 45:
        gated_bias = "BEARISH"
    else:
        gated_bias = "NEUTRAL"

    detail = {
        "status": "ok", "available": True,
        "gated_score": gated_score, "gated_bias": gated_bias,
        "momentum_strength": round(momentum_strength, 3),
        "momentum_bias": momentum_bias,
        "bucket": bucket, "regime_action": action,
        "action_reason": action_reason,
        "confidence": round(confidence, 3),
        "era3_only": era3_only,
        "horizon_evidence": horizon_ev,
        "caveat": ("DISPLAY-ONLY research overlay — regime-gated momentum "
                   "(re-combines trend_strength.momentum + trend_continuation "
                   "buckets). NOT blended into signal / scoring. Unconditional "
                   "momentum was rejected by purged-CV; this gates it by regime."),
        "ts": now,
    }
    _save_cache({"score": gated_score, "detail": detail, "ts": now, "key": _key})
    return gated_score, detail


if __name__ == "__main__":
    # Synthetic continuation probs matching the live .trend_continuation_cache.json
    # era3 shape (era3 p_cont: CALM 0.518@30/0.56@90/0.669@180; ELEVATED 0.586/0.683/0.749;
    # STRESS 0.515/0.383/0.459). Baseline ~0.573@30/0.596@90/0.659@180.
    BASE = {30: 0.573, 90: 0.596, 180: 0.659}
    ERA3 = {
        "CALM":     {30: 0.518, 90: 0.560, 180: 0.669},
        "ELEVATED": {30: 0.586, 90: 0.683, 180: 0.749},
        "STRESS":   {30: 0.515, 90: 0.383, 180: 0.459},
    }
    def mk_probs(bucket):
        if bucket is None:
            return {}
        p = {}
        for h in HORIZONS:
            p[h] = {"probability": None, "baseline": BASE[h],
                    "era3_probability": ERA3[bucket][h], "era_stable": True,
                    "relative": None}
        return p

    print("=== Regime-Gated Momentum Overlay self-test ===\n")
    for name, ms, bucket in (("BULL/STRESS(fade)", 0.76, "STRESS"),
                             ("BULL/ELEVATED(follow)", 0.76, "ELEVATED"),
                             ("BEAR/ELEVATED(follow)", 0.30, "ELEVATED"),
                             ("BULL/CALM(era3 fade)", 0.70, "CALM"),
                             ("NEUTRAL momentum", 0.50, "ELEVATED"),
                             ("NO_DATA", None, None)):
        score, det = compute_momentum_overlay(
            momentum_strength=ms, bucket=bucket, cont_probs=mk_probs(bucket),
        )
        print(f"{name:24s} score={score:5.1f} bias={str(det.get('gated_bias')):8s} "
              f"action={str(det.get('regime_action')):8s} conf={det.get('confidence')}")
    print("\nDetail contoh (BULL/STRESS):")
    _, d = compute_momentum_overlay(0.76, "STRESS", mk_probs("STRESS"))
    for e in d["horizon_evidence"]:
        print(f"  h={e['horizon']:4d} p_cont={e['p_cont']} base={e['baseline']} "
              f"support={e['support']} era_stable={e['era_stable']}")
    print("ALL OK")
