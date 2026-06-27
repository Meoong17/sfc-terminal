#!/usr/bin/env python3
"""
SFC Liquidity Momentum (LM)
=============================
Measures the RATE OF CHANGE in global liquidity — not the absolute level.

  LM_t = GLF_t - GLF_{t-30}

Bitcoin often reacts to CHANGES in liquidity, not the absolute level.
A central bank pausing QT is more impactful than the QT level itself.

Usage:
    from liquidity_momentum import compute_liquidity_momentum
    lm_score, lm_details = compute_liquidity_momentum(current_glf=48.3)

Stores a rolling history of GLF scores in .liq_momentum_cache.json.
"""

import json, os, math, time
from datetime import datetime, timezone

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(SFC_DIR, '.liq_momentum_cache.json')
MAX_HISTORY = 60  # keep 60 daily entries


def _load_history():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"history": [], "cached_at": 0}


def _save_history(cache):
    cache["cached_at"] = time.time()
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def compute_liquidity_momentum(current_glf=None, current_glf_stress=None):
    """
    Compute Liquidity Momentum (LM).

    Stores a daily history of GLF scores. On each call, checks if today
    already has an entry. If not, appends the current GLF.

    Args:
        current_glf: Current GLF score (0-100) from global_liquidity_engine
        current_glf_stress: Current GLF stress (0-1) for SFC pipeline

    Returns:
        (lm_score, lm_stress_adjustment, details)
        lm_score: -100 to +100 (positive = improving liquidity)
        lm_stress_adjustment: delta stress adjustment for SFC (0-0.15)
    """
    if current_glf is None:
        return 0.0, {"lm": 0, "status": "no_data", "n_points": 0}

    cache = _load_history()
    history = cache.get("history", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check if today already has an entry
    found = False
    for entry in history:
        if entry.get("date") == today:
            entry["glf"] = current_glf
            entry["ts"] = time.time()
            found = True
            break

    if not found:
        history.append({
            "date": today,
            "glf": current_glf,
            "ts": time.time(),
        })

    # Keep max history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    # Compute momentum
    if len(history) >= 30:
        glf_now = history[-1]["glf"]
        glf_30d = history[-30]["glf"]
        lm_change = glf_now - glf_30d  # positive = liquidity improving
        lm_pct = ((glf_now - glf_30d) / max(glf_30d, 1)) * 100
    elif len(history) >= 7:
        glf_now = history[-1]["glf"]
        glf_7d = history[-7]["glf"]
        lm_change = glf_now - glf_7d
        lm_pct = ((glf_now - glf_7d) / max(glf_7d, 1)) * 100
    else:
        lm_change = 0
        lm_pct = 0
        glf_now = current_glf  # use current value even without history

    # Score: LM change in [-20, +20] range → stress adjustment
    # Positive LM = liquidity improving = reducing stress
    # Negative LM = liquidity deteriorating = increasing stress
    if lm_change > 3:
        lm_stress_adj = -0.05  # improving → reduce stress
    elif lm_change > 1:
        lm_stress_adj = -0.02
    elif lm_change > -1:
        lm_stress_adj = 0.0  # neutral
    elif lm_change > -3:
        lm_stress_adj = 0.03  # deteriorating → increase stress
    elif lm_change > -10:
        lm_stress_adj = 0.08
    else:
        lm_stress_adj = 0.15  # sharp deterioration

    # Trend label
    if lm_change > 5:
        label = "ACCELERATING_IMPROVEMENT"
    elif lm_change > 2:
        label = "IMPROVING"
    elif lm_change > -2:
        label = "STABLE"
    elif lm_change > -5:
        label = "DETERIORATING"
    else:
        label = "SHARP_DETERIORATION"

    details = {
        "lm_change": round(lm_change, 2),
        "lm_pct": round(lm_pct, 2),
        "lm_stress_adj": round(lm_stress_adj, 3),
        "label": label,
        "n_points": len(history),
        "glf_now": round(glf_now, 1),
        "glf_30d_ago": round(history[-30]["glf"], 1) if len(history) >= 30 else None,
        "date_range": f"{history[0]['date']} → {history[-1]['date']}" if len(history) >= 2 else today,
        "status": "ok",
    }

    # Save updated history
    cache["history"] = history
    _save_history(cache)

    return round(lm_change, 2), round(lm_stress_adj, 3), details


if __name__ == "__main__":
    import json
    lm, adj, det = compute_liquidity_momentum(current_glf=48.3)
    print(json.dumps({
        "lm": lm,
        "stress_adjustment": adj,
        "n_points": det.get("n_points"),
        "label": det.get("label"),
        "details": det,
    }, indent=2))
