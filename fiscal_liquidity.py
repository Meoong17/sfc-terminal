#!/usr/bin/env python3
"""
SFC Fiscal Liquidity Module (M83-M84)
========================================
M83 — TGA Balance (U.S. Treasury General Account at Fed)
      TGA drawdown = fiscal stimulus = liquidity injection = bullish
      TGA accumulation = liquidity withdrawal = bearish

M84 — RRP Facility Usage (Overnight Reverse Repo at Fed)
      RRP decline = cash entering markets = bullish
      RRP increase = cash parking at Fed = bearish

Data sources:
  - FRED: WTREGEN (TGA, weekly, $M)
  - FRED: RRPONTSYD (ON RRP, daily, $B)
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone

# ── Cache ──
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fiscal_cache.json')
CACHE_TTL = 43200  # 12 hours

FRED_KEY = os.getenv("FRED_API_KEY", "")
_FRED_CACHE = {}

def _fred(series, limit=2):
    """Fetch FRED data with module-level cache."""
    global _FRED_CACHE
    cache_key = f"{series}:{limit}"
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]
    if not FRED_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
            f"&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}",
            timeout=15
        )
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        result = vals if vals else None
        _FRED_CACHE[cache_key] = result
        return result
    except:
        return None


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cached_at": 0}

def _save_cache(cache):
    cache["cached_at"] = time.time()
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def compute_fiscal_liquidity_metrics():
    """Compute M83 and M84 scores.

    Returns:
        (m83_score, m84_score, composite_score, details)
        All scores: 0-1 where high = high stress (tight fiscal liquidity)
        composite: weighted blend of M83 + M84
    """
    now = time.time()
    cache = _load_cache()

    # ── M83: TGA Balance ──
    # Fetch weekly TGA data (last 8 weeks to see trend)
    tga_vals = _fred("WTREGEN", 8)
    m83_score = 0.5
    m83_detail = {"status": "unavailable"}

    if tga_vals and len(tga_vals) >= 2:
        tga_latest = tga_vals[0]
        tga_4w_ago = tga_vals[3] if len(tga_vals) >= 4 else tga_vals[-1]
        tga_chg_pct = ((tga_latest - tga_4w_ago) / tga_4w_ago * 100) if tga_4w_ago > 0 else 0
        tga_chg_abs_b = (tga_latest - tga_4w_ago) / 1000  # convert $M to $B

        # TGA decreasing = fiscal stimulus = bullish (low stress)
        # TGA increasing = liquidity withdrawal = bearish (high stress)
        if tga_chg_pct < -10:
            m83_score = 0.15  # significant drawdown → stimulus
        elif tga_chg_pct < -5:
            m83_score = 0.25
        elif tga_chg_pct < -2:
            m83_score = 0.35
        elif tga_chg_pct < 2:
            m83_score = 0.50  # stable
        elif tga_chg_pct < 5:
            m83_score = 0.60
        elif tga_chg_pct < 10:
            m83_score = 0.75
        else:
            m83_score = 0.85  # significant accumulation → drain

        # Also consider absolute level: very high TGA means potential future drain
        # Very low TGA means Treasury needs to rebuild (future headwind)
        if tga_latest > 900000:  # >$900B — high, potential future drain
            m83_score = min(0.90, m83_score + 0.10)
        elif tga_latest < 300000:  # <$300B — low, Treasury needs to rebuild
            m83_score = min(0.80, m83_score + 0.08)

        m83_score = max(0.05, min(0.95, m83_score))

        m83_detail = {
            "tga_latest_b": round(tga_latest / 1000, 1),  # convert $M to $B
            "tga_4w_chg_pct": round(tga_chg_pct, 2),
            "tga_4w_chg_b": round(tga_chg_abs_b, 1),
            "tga_trend": "DRAWING_DOWN" if tga_chg_pct < -2 else "ACCUMULATING" if tga_chg_pct > 2 else "STABLE",
            "status": "ok",
        }
    # ── M84: RRP Facility Usage ──
    # RRPONTSYD is in $B, daily
    rrp_vals = _fred("RRPONTSYD", 8)
    m84_score = 0.5
    m84_detail = {"status": "unavailable"}

    if rrp_vals and len(rrp_vals) >= 2:
        rrp_latest = rrp_vals[0]
        rrp_4w_ago = rrp_vals[3] if len(rrp_vals) >= 4 else rrp_vals[-1]
        rrp_chg = rrp_latest - rrp_4w_ago

        # RRP near zero = cash deployed in markets = bullish (low stress)
        # RRP high (>$100B) = cash parked at Fed = bearish (high stress)
        # RRP increasing = bearish trend, RRP decreasing = bullish trend

        if rrp_latest < 10:
            rrp_level_score = 0.15  # essentially zero — all cash deployed
        elif rrp_latest < 50:
            rrp_level_score = 0.30
        elif rrp_latest < 200:
            rrp_level_score = 0.50
        elif rrp_latest < 500:
            rrp_level_score = 0.65
        elif rrp_latest < 1000:
            rrp_level_score = 0.80
        else:
            rrp_level_score = 0.90  # massive cash parked

        # Trend adjustment
        if rrp_chg < -50:
            trend_adj = -0.15  # rapidly declining — very bullish
        elif rrp_chg < -10:
            trend_adj = -0.08
        elif rrp_chg < 0:
            trend_adj = -0.03
        elif rrp_chg < 10:
            trend_adj = 0.03
        elif rrp_chg < 50:
            trend_adj = 0.08
        else:
            trend_adj = 0.15  # rapidly increasing — very bearish

        m84_score = max(0.05, min(0.95, rrp_level_score + trend_adj))

        m84_detail = {
            "rrp_latest_b": round(rrp_latest, 1),
            "rrp_4w_chg_b": round(rrp_chg, 1),
            "rrp_level_label": "NEAR_ZERO" if rrp_latest < 10 else "LOW" if rrp_latest < 100 else "ELEVATED" if rrp_latest < 500 else "HIGH",
            "rrp_trend": "DECLINING" if rrp_chg < -10 else "RISING" if rrp_chg > 10 else "STABLE",
            "status": "ok",
        }

    # ── Composite: Fiscal Liquidity (weighted) ──
    scores = []
    weights = []
    if m83_detail["status"] == "ok":
        scores.append(m83_score)
        weights.append(0.55)  # TGA is more directly impactful
    if m84_detail["status"] == "ok":
        scores.append(m84_score)
        weights.append(0.45)

    if not scores:
        return 0.5, 0.5, 0.5, {"composite": None, "status": "no_data"}

    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w

    # Regime label
    if composite < 0.25:
        regime = "FISCAL_STIMULUS"
    elif composite < 0.45:
        regime = "ACCOMMODATIVE"
    elif composite < 0.55:
        regime = "NEUTRAL"
    elif composite < 0.70:
        regime = "TIGHTENING"
    else:
        regime = "FISCAL_DRAG"

    details = {
        "m83": m83_detail,
        "m84": m84_detail,
        "composite": round(composite, 3),
        "regime": regime,
        "status": "ok",
    }

    return round(m83_score, 3), round(m84_score, 3), round(composite, 3), details


if __name__ == "__main__":
    m83, m84, comp, details = compute_fiscal_liquidity_metrics()
    print(json.dumps({
        "m83_tga_score": m83,
        "m84_rrp_score": m84,
        "fiscal_composite": comp,
        "details": details,
    }, indent=2))
