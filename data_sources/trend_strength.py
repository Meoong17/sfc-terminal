#!/usr/bin/env python3
"""
SFC Trend Strength Score (P2) — institutional output.
=====================================================
Addresses the framework's recommendation for a "Trend Strength Score"
(analysis/gap_analysis_dokumen_vs_model.md §3). It quantifies how strong
the CURRENT BTC trend is — separate from the stress/regime read.

THREE INDEPENDENT DOMAINS (defensible, no fabricated inputs):
    1. MOMENTUM    — RSI level vs neutral (45.6 = mild; <30 oversold;
                     >70 overbought) + MACD sign + OBV (volume trend).
    2. ALIGNMENT   — multi-timeframe alignment (mtf_alignment_score,
                     normalized -1..+1) + DFS regime trend persistence.
    3. STRUCTURE   — HMM regime trend persistence (BULL/CRISIS contribution)
                     + crisis probability (low = healthy trend).

Composite is a weighted blend normalized to 0-100, where HIGH = strong
(healthy, persistent, aligned) uptrend and LOW = weak / absent / broken.

DISPLAY-ONLY: exposed as its own field, NOT blended into sfc_effective /
signal / kelly_fraction (cautious-rollout pattern).

Usage:
    from data_sources.trend_strength import compute_trend_strength
    score, details = compute_trend_strength(
        rsi=45.6, mtf_alignment=-0.5, hmm_regime='SIDEWAYS',
        hmm_crisis_prob=0.0, dfs_regime='SIDEWAYS',
        macd_signal=0.02, bb_width=0.002, obv_norm=0.08,
    )
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".trend_strength_cache.json")
CACHE_TTL = 300  # 5 min — trend inputs move intraday


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _rsi_to_strength(rsi):
    """RSI contributes to trend strength: healthy range 50-65 is strongest,
    extreme oversold (<30) is a (counter-trend) recovery signal, extreme
    overbought (>70) is stretched. None -> neutral 0.5."""
    if rsi is None:
        return 0.5
    if rsi < 30:
        return 0.30  # deeply oversold — weak trend, possible reversal
    if rsi <= 50:
        return 0.35 + 0.30 * ((rsi - 30) / 20.0)   # 30→50: 0.35→0.65
    if rsi <= 65:
        return 0.65 + 0.30 * ((rsi - 50) / 15.0)   # 50→65: 0.65→0.95 (sweet spot)
    if rsi <= 80:
        return 0.95 - 0.35 * ((rsi - 65) / 15.0)   # 65→80: 0.95→0.60 (stretched)
    return 0.50  # extremely overbought


def _macd_to_strength(macd, bb_width):
    """MACD sign + Bollinger width. Positive MACD = bullish momentum;
    narrow BB = low vol (trend quieter), wide BB = high vol (less reliable
    trend read). None inputs treated neutral."""
    if macd is None:
        return 0.5
    base = 0.5 + 2.0 * _clamp(macd, -0.1, 0.1) / 0.1 * 0.5  # +-0.5 around 0.5
    base = _clamp(base)
    # Wide BB reduces confidence in the trend signal (noise).
    if bb_width is not None:
        base = base * (1.0 - 0.25 * _clamp(bb_width / 0.01, 0, 1))
    return _clamp(base)


def _alignment_to_strength(mtf_alignment, dfs_regime):
    """Multi-timeframe alignment (normalized to ~-1..+1) + DFS trend.
    Positive alignment = bullish multi-TF agreement = strong trend."""
    if mtf_alignment is None:
        align = 0.5
    else:
        align = _clamp(0.5 + mtf_alignment * 0.5)  # -1..+1 -> 0..1
    # DFS regime reinforces: BULL adds, BEAR/CRISIS reduces, SIDEWAYS neutral.
    dfs = str(dfs_regime or '').upper()
    if dfs == 'BULL':
        align = _clamp(align + 0.10)
    elif dfs in ('BEAR', 'CRISIS'):
        align = _clamp(align - 0.15)
    return align


def _structure_to_strength(hmm_regime, hmm_crisis_prob):
    """HMM regime trend persistence + crisis probability."""
    if hmm_regime is None:
        return 0.5
    r = str(hmm_regime).upper()
    if r == 'BULL':
        base = 0.85
    elif r == 'BEAR':
        base = 0.25
    elif r == 'SIDEWAYS':
        base = 0.50
    elif r == 'CRISIS':
        base = 0.15
    else:
        base = 0.50
    # High crisis probability detracts from trend strength.
    if hmm_crisis_prob is not None:
        base -= _clamp(hmm_crisis_prob) * 0.5
    return _clamp(base)


def compute_trend_strength(rsi=None, mtf_alignment=None, hmm_regime=None,
                           hmm_crisis_prob=None, dfs_regime=None,
                           macd_signal=None, bb_width=None, obv_norm=None,
                           momentum_weight=0.40, alignment_weight=0.35,
                           structure_weight=0.25):
    """Compute the Trend Strength Score (0-100, high = strong trend).

    Args: optional inputs; a missing domain is excluded and its weight is
    redistributed over the available ones. Returns (score 0-100, details).
    """
    cached = _load_cache()
    now = time.time()
    _key = "|".join(f"{x if x is None else round(float(x),4)}"
                    if not isinstance(x, str) else str(x)
                    for x in (rsi, mtf_alignment, hmm_regime, hmm_crisis_prob,
                              dfs_regime, macd_signal, bb_width, obv_norm))
    if (cached.get("key") == _key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("score", 50.0), cached.get("details", {"status": "cached"})

    domains = []
    m = _rsi_to_strength(rsi)
    if rsi is not None:
        domains.append(["momentum", m, momentum_weight, {"rsi": rsi}])
    a = _alignment_to_strength(mtf_alignment, dfs_regime)
    if mtf_alignment is not None or dfs_regime is not None:
        domains.append(["alignment", a, alignment_weight,
                        {"mtf_alignment": mtf_alignment, "dfs_regime": dfs_regime}])
    s = _structure_to_strength(hmm_regime, hmm_crisis_prob)
    if hmm_regime is not None:
        domains.append(["structure", s, structure_weight,
                        {"hmm_regime": hmm_regime, "hmm_crisis_prob": hmm_crisis_prob}])

    # Optional momentum enhancers (MACD/OBV) fold into the momentum domain.
    if macd_signal is not None:
        macd = _macd_to_strength(macd_signal, bb_width)
        for dom in domains:
            if dom[0] == "momentum":
                dom[1] = 0.6 * dom[1] + 0.4 * macd
    if obv_norm is not None:
        # OBV normalized (-1..1): positive = accumulation volume.
        for dom in domains:
            if dom[0] == "momentum":
                dom[1] = _clamp(dom[1] + _clamp(obv_norm, -1, 1) * 0.1)

    if not domains:
        score, status = 50.0, "unavailable"
        details = {"status": status, "available": False, "label": "UNKNOWN",
                   "reason": "No trend inputs supplied."}
        _save_cache({"score": score, "details": details, "ts": now, "key": _key})
        return score, details

    total_w = sum(d[2] for d in domains)
    raw = sum(d[1] * d[2] for d in domains) / total_w
    score = round(_clamp(raw) * 100, 1)

    label = ("STRONG" if score >= 65 else "MODERATE" if score >= 45 else
             "WEAK" if score >= 25 else "BROKEN")

    details = {
        "status": "ok",
        "available": True,
        "score": score,
        "label": label,
        "domains": [{"name": d[0], "value": round(d[1], 3), "weight": d[2]}
                    for d in domains],
        "domain_values": {d[0]: round(d[1], 3) for d in domains},
        "weights_used": {d[0]: d[2] for d in domains},
        "rule": "Weighted blend (momentum/alignment/structure) normalized 0-100; "
                "missing domains redistributed. Display-only, not blended into signal.",
        "ts": now,
    }
    _save_cache({"score": score, "details": details, "ts": now, "key": _key})
    return score, details


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
    import sys
    # Live-like, strong bull, weak/broken, and unavailable.
    live = dict(rsi=45.61, mtf_alignment=-0.5, hmm_regime="SIDEWAYS",
                hmm_crisis_prob=0.0, dfs_regime="SIDEWAYS",
                macd_signal=0.0249, bb_width=0.0019, obv_norm=0.0848)
    bull = dict(rsi=58.0, mtf_alignment=+0.6, hmm_regime="BULL",
                hmm_crisis_prob=0.01, dfs_regime="BULL",
                macd_signal=0.05, obv_norm=0.3)
    broken = dict(rsi=25.0, mtf_alignment=-0.8, hmm_regime="BEAR",
                  hmm_crisis_prob=0.4, dfs_regime="CRISIS",
                  macd_signal=-0.05, obv_norm=-0.4)
    for name, kw in (("LIVE", live), ("BULL", bull), ("BROKEN", broken),
                     ("NO_DATA", {})):
        sc, det = compute_trend_strength(**kw)
        print(f"{name:8s} score={sc:5.1f} label={det.get('label')} "
              f"domains={det.get('domain_values')}")
