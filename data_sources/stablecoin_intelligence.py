#!/usr/bin/env python3
"""
SFC Stablecoin Intelligence Engine
===================================
Enhanced stablecoin analysis beyond M76-M80. Produces a single
Stablecoin Liquidity Index (SLI) from:

  M76 — Stablecoin Supply Growth (7d & 30d % change)
  M77 — SSR (Stablecoin Supply Ratio)
  M78 — Exchange Flow (netflow to/from exchanges)
  M79 — Stablecoin Velocity
  M80 — Stablecoin Dominance
  NEW — Mint/Burn Ratio (USDT/USDC net creation vs destruction)
  NEW — USDT vs USDC Growth Split (which stablecoin is growing faster)
  NEW — Per-stablecoin supply trend

Output: SLI score 0-100 and SFC stress score 0-1

Takes existing stablecoin_liquidity.py results + onchain data + fresh
stablecoin market cap data from CoinGecko.

Usage:
    from stablecoin_intelligence import compute_stablecoin_liquidity_index
    sli_score, sli_sfc_stress, details = compute_stablecoin_liquidity_index(...)
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, '.stablecoin_intel_cache.json')
CACHE_TTL = 21600  # 6 hours

# ── Stablecoin IDs for per-coin tracking ──
STABLECOIN_IDS = ["tether", "usd-coin", "dai", "first-digital-usd"]
CG_BASE = "https://api.coingecko.com/api/v3"
# Audit 2026-08-03: demo key was hardcoded in source (committed secret).
# Now read from env; empty => CoinGecko 401 => _fetch_cg() degrades to neutral.
# Add to your .env:  COINGECKO_API_KEY=CG-xxxx (see .env)
CG_API_PARAM = "x_cg_demo_api_key=" + os.getenv("COINGECKO_API_KEY", "")


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cached_at": 0}

def _save_cache(cache):
    cache["cached_at"] = time.time()
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _fetch_cg(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


def _fetch_per_coin_data():
    """
    Fetch individual stablecoin market cap history for USDT, USDC, DAI, FDUSD.
    Returns dict: {coin_id: [latest_mcap, mcap_7d_ago, mcap_30d_ago]}
    """
    result = {}

    def _fetch_one(coin_id):
        data = _fetch_cg(f"{CG_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days=30&{CG_API_PARAM}")
        if data and "market_caps" in data and data["market_caps"]:
            mcaps = [m[1] for m in data["market_caps"]]
            if len(mcaps) >= 2:
                latest = mcaps[-1]
                week_ago = mcaps[-7] if len(mcaps) >= 7 else mcaps[0]
                month_ago = mcaps[0]
                return coin_id, (latest, week_ago, month_ago)
        return coin_id, None

    with ThreadPoolExecutor(max_workers=len(STABLECOIN_IDS)) as ex:
        futures = {ex.submit(_fetch_one, sid): sid for sid in STABLECOIN_IDS}
        for f in as_completed(futures):
            sid, data = f.result()
            if data:
                result[sid] = data

    return result


def compute_stablecoin_liquidity_index(
    existing_sc_results=None,
    existing_sc_details=None,
    btc_price=None,
    btc_mcap=None,
    btc_dominance_pct=None,
    onchain_details=None,
    force_refresh=False,
):
    """
    Compute Stablecoin Liquidity Index (SLI).

    Combines:
      - M76-M80 from existing stablecoin_liquidity module
      - Per-coin breakdown (USDT growth vs USDC growth)
      - Mint/burn ratio approximation
      - Composite SLI

    Args:
        existing_sc_results: Dict from stablecoin_liquidity.compute_all_stablecoin_metrics
        existing_sc_details: Detail dict from same function
        btc_price: Current BTC price
        btc_mcap: Current BTC market cap
        btc_dominance_pct: BTC dominance %
        onchain_details: Q10 onchain details dict
        force_refresh: Force re-fetch from CoinGecko

    Returns:
        (sli_score_0_100, sli_sfc_stress_0_1, details)
        sli_score_0_100: 0=very bearish, 100=very bullish (for stablecoin liquidity)
        sli_sfc_stress_0_1: high = stress (follows SFC convention)
    """
    now = time.time()
    cache = _load_cache()

    # Try cache
    if not force_refresh and (now - cache.get("cached_at", 0)) < CACHE_TTL:
        if cache.get("sli_score") is not None:
            return cache["sli_score"], cache["sfc_stress"], cache.get("details", {})

    # ── Component scores (each 0-1, high = bearish/liquidity stress) ──
    component_scores = {}  # name -> (value_0_1, weight)

    # 1. Supply Growth (from existing module or per-coin data)
    m76_score = None
    if existing_sc_results and "m76_supply_growth" in existing_sc_results:
        growth_score = existing_sc_results["m76_supply_growth"]
        # Invert: existing M76 scores high = bullish, we want high = bearish
        m76_score = 1.0 - growth_score
    if m76_score is not None:
        component_scores["supply_growth"] = (m76_score, 0.20)

    # 2. SSR (inverted: low SSR = buying power = bullish = low stress)
    m77_score = None
    if existing_sc_results and "m77_ssr" in existing_sc_results:
        ssr_score = existing_sc_results["m77_ssr"]
        # Invert: M77 high = bullish, we want high = bearish
        m77_score = 1.0 - ssr_score
    if m77_score is not None:
        component_scores["ssr"] = (m77_score, 0.15)

    # 3. Exchange Flow
    m78_score = None
    if existing_sc_results and "m78_exchange_flow" in existing_sc_results:
        flow_score = existing_sc_results["m78_exchange_flow"]
        m78_score = 1.0 - flow_score
    if m78_score is not None:
        component_scores["exchange_flow"] = (m78_score, 0.15)

    # 4. Velocity (from existing)
    m79_score = None
    if existing_sc_results and "m79_velocity" in existing_sc_results:
        vel_score = existing_sc_results["m79_velocity"]
        m79_score = 1.0 - vel_score
    if m79_score is not None:
        component_scores["velocity"] = (m79_score, 0.10)

    # 5. Dominance (from existing)
    m80_score = None
    if existing_sc_results and "m80_dominance" in existing_sc_results:
        dom_score = existing_sc_results["m80_dominance"]
        m80_score = 1.0 - dom_score
    if m80_score is not None:
        component_scores["dominance"] = (m80_score, 0.10)

    # 6. Per-coin growth split
    per_coin = _fetch_per_coin_data()
    usdt_growth = usdc_growth = None
    if "tether" in per_coin:
        latest_usdt, _, m30_usdt = per_coin["tether"]
        # Previously this read from existing_sc_details["m76_detail"]["growth_30d_pct"]
        # which is the total stablecoin MARKET growth (all coins combined), not
        # USDT-specific growth — so growth_divergence was comparing apples to
        # oranges: total-market-growth vs USDC-specific-growth. Now both sides
        # use per_coin data consistently with the same calculation method.
        if m30_usdt and m30_usdt > 0 and latest_usdt:
            usdt_growth = (latest_usdt - m30_usdt) / m30_usdt * 100
    if "usd-coin" in per_coin:
        latest_usdc, _, m30_usdc = per_coin["usd-coin"]
        if m30_usdc and m30_usdc > 0 and latest_usdc:
            usdc_growth = (latest_usdc - m30_usdc) / m30_usdc * 100

    # Growth divergence: USDT growing much faster than USDC = risk signal
    # (capital fleeing regulated into offshore)
    growth_divergence = 0.5  # default neutral when data unavailable
    if usdt_growth is not None and usdc_growth is not None:
        diff = usdt_growth - usdc_growth
        if diff > 5:
            growth_divergence = 0.70  # USDT dominates = bearish
        elif diff > 2:
            growth_divergence = 0.55
        elif diff < -2:
            growth_divergence = 0.30  # USDC dominates = regulatory confidence
        else:
            growth_divergence = 0.45
    component_scores["growth_divergence"] = (growth_divergence, 0.10)

    # 7. Exchange reserve / supply ratio
    # Previously this was split into two separate components ("mint_burn" and
    # "reserve_ratio") both deriving their score from the same input variable
    # (exchange_reserve / total_mcap_now). Giving both a 0.10 weight effectively
    # double-counted a single signal with 0.20 total weight. Merged here into
    # one "reserve_ratio" component (weight 0.20) with a more granular 5-tier
    # scale that preserves the full resolution of the original two components
    # without rewarding the same data point twice.
    reserve_ratio_score = 0.5
    total_mcap_now = existing_sc_details.get("m76_detail", {}).get("latest_mcap") if existing_sc_details else None
    if total_mcap_now and onchain_details:
        reserve = onchain_details.get("stablecoin_reserve", {})
        if isinstance(reserve, dict):
            r_val = reserve.get("value")
            if r_val and r_val > 0 and total_mcap_now > 0:
                rr = r_val / total_mcap_now
                # Low ratio = coins in cold storage / DeFi = capital deployed = bullish
                # High ratio = coins sitting on exchanges = potential sell pressure = bearish
                if rr < 0.03:
                    reserve_ratio_score = 0.20   # very low exchange concentration
                elif rr < 0.06:
                    reserve_ratio_score = 0.35
                elif rr < 0.10:
                    reserve_ratio_score = 0.45
                elif rr < 0.15:
                    reserve_ratio_score = 0.60
                else:
                    reserve_ratio_score = 0.75   # high concentration on exchanges

    # Additionally factor in total supply momentum (a genuinely separate signal
    # from the reserve ratio): rapid supply growth can indicate speculative
    # minting; rapid contraction indicates redemptions. This was previously mixed
    # into the "mint_burn" component alongside the reserve ratio calculation.
    if existing_sc_details and "m76_detail" in existing_sc_details:
        g30 = existing_sc_details["m76_detail"].get("growth_30d_pct")
        if g30 is not None:
            if g30 > 10:      # Very fast growth → frothy, nudge score upward
                reserve_ratio_score = min(0.80, reserve_ratio_score + 0.10)
            elif g30 < -3:    # Shrinking supply → liquidity withdrawal signal
                reserve_ratio_score = min(0.80, reserve_ratio_score + 0.10)

    # Use the full 0.20 combined weight (was 0.10 + 0.10 for two redundant components)
    component_scores["reserve_ratio"] = (reserve_ratio_score, 0.20)

    # ── Compute weighted SLI score ──
    if not component_scores:
        return 50.0, 0.50, {"error": "no data available", "status": "fallback"}

    total_w = sum(w for _, w in component_scores.values())
    sli_z = sum((v - 0.5) * w for v, w in component_scores.values()) / total_w

    # Map to SLI 0-100
    # sli_z = 0 (neutral) → SLI=55
    # sli_z = +0.4 (bullish) → SLI=80
    # sli_z = -0.4 (bearish) → SLI=20
    sli_raw = 55 + sli_z * 62.5
    sli_score = max(0, min(100, sli_raw))

    # Map to SFC stress (0-1, high = stress)
    # SLI high = healthy stablecoin liquidity = low stress
    sli_sfc_stress = 1.0 - (sli_score / 100.0)
    sli_sfc_stress = max(0.05, min(0.95, sli_sfc_stress))

    # Label
    if sli_score > 75:
        label = "ABUNDANT"
    elif sli_score > 55:
        label = "HEALTHY"
    elif sli_score > 40:
        label = "NEUTRAL"
    elif sli_score > 25:
        label = "STRESSED"
    else:
        label = "CRITICAL"

    # Component detail
    comp_detail = {}
    for name, (val, weight) in sorted(component_scores.items()):
        comp_detail[name] = {
            "score": round(val, 3),
            "weight": weight,
            "contribution": round((val - 0.5) * weight / total_w, 4),
        }

    details = {
        "sli_score": round(sli_score, 1),
        "sli_sfc_stress": round(sli_sfc_stress, 3),
        "label": label,
        "components": comp_detail,
        "n_components": len(component_scores),
        "per_coin": {
            sid: {"latest_mcap": round(v[0], 0), "mcap_30d_ago": round(v[2], 0) if v[2] else None}
            for sid, v in per_coin.items()
        } if per_coin else {},
        "usdt_growth_pct": round(usdt_growth, 2) if usdt_growth is not None else None,
        "usdc_growth_pct": round(usdc_growth, 2) if usdc_growth is not None else None,
        "growth_divergence_pct": round((usdt_growth or 0) - (usdc_growth or 0), 2) if usdt_growth is not None and usdc_growth is not None else None,
        "status": "ok",
    }

    # Save cache
    cache["sli_score"] = sli_score
    cache["sfc_stress"] = sli_sfc_stress
    cache["details"] = details
    _save_cache(cache)

    return round(sli_score, 1), round(sli_sfc_stress, 3), details


if __name__ == "__main__":
    # Test with mock data
    mock_results = {
        "m76_supply_growth": 0.65,
        "m77_ssr": 0.70,
        "m78_exchange_flow": 0.55,
        "m79_velocity": 0.45,
        "m80_dominance": 0.60,
    }
    mock_details = {
        "m76_detail": {"growth_30d_pct": 3.5, "latest_mcap": 180_000_000_000},
    }
    sli, stress, details = compute_stablecoin_liquidity_index(
        existing_sc_results=mock_results,
        existing_sc_details=mock_details,
        force_refresh=False,
    )
    print(json.dumps({
        "sli_score": sli,
        "sfc_stress": stress,
        "label": details.get("label"),
        "n_components": details.get("n_components"),
        "details": details,
    }, indent=2))
