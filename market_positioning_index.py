#!/usr/bin/env python3
"""
SFC Market Positioning Index (MPI)
====================================
Combines derivatives data into ONE composite index:

  Funding Rate     → leverage cost / euphoria
  Open Interest    → total notional exposure
  Liquidation Vol  → cascade risk
  Put/Call Ratio   → options skew / hedging demand
  Basis Spread     → futures premium (basis = futures - spot)

Data sources:
  Deribit (free REST API): funding rate, options OI, put/call
  OKX / CoinGlass (liquidation_client.py): liquidation volume

MPI score: 0-100 where:
  0 = extremely bearish positioning (crowded short, high OI, mass liqs)
  50 = neutral
  100 = extremely bullish positioning (crowded long, aggressive funding)

SFC stress score: inverted — high MPI = low stress.
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone

SFC_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(SFC_DIR, '.mpi_cache.json')
CACHE_TTL = 900  # 15 min (derivatives data changes fast)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
BINANCE_BASE = "https://fapi.binance.com"


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


def _derive_index_price():
    """Fetch Deribit BTC index price (reference for basis calculation)."""
    try:
        r = requests.get(f"{DERIBIT_BASE}/get_index_price?index_name=btc_usdc", timeout=10)
        data = r.json()
        return data.get("result", {}).get("index_price")
    except:
        return None


def _funding_rate():
    """Fetch recent funding rates from Deribit."""
    try:
        r = requests.get(
            f"{DERIBIT_BASE}/get_funding_rate_history?currency=BTC"
            f"&start_timestamp=0&end_timestamp={int(time.time() * 1000)}",
            timeout=10
        )
        data = r.json().get("result", [])
        if len(data) < 3:
            return None, None, None
        rates = [d["interest_8h"] for d in data[:8]]
        fr_now = rates[0]
        fr_avg = sum(rates) / len(rates)
        accel = (rates[0] - rates[1]) - (rates[1] - rates[2]) if len(rates) >= 3 else 0
        return fr_now, fr_avg, accel
    except:
        return None, None, None


def _options_data():
    """Fetch options OI and put/call IV from Deribit."""
    try:
        r = requests.get(f"{DERIBIT_BASE}/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        if not opts:
            return None, None, None, None

        puts_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-P"))
        calls_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-C"))
        puts_iv = [o["mark_iv"] for o in opts if o.get("instrument_name", "").endswith("-P") and o.get("mark_iv")]
        calls_iv = [o["mark_iv"] for o in opts if o.get("instrument_name", "").endswith("-C") and o.get("mark_iv")]

        pc_oi = round(puts_oi / calls_oi, 4) if calls_oi else None
        put_iv = sum(puts_iv) / len(puts_iv) if puts_iv else None
        call_iv = sum(calls_iv) / len(calls_iv) if calls_iv else None
        skew = (put_iv - call_iv) / max(call_iv, 0.01) if put_iv and call_iv else None

        return pc_oi, skew, puts_oi + calls_oi, pc_oi
    except:
        return None, None, None, None


def _binance_futures_basis():
    """Fetch futures basis (difference between futures and spot)."""
    try:
        # Futures perpetual price
        r = requests.get(f"{BINANCE_BASE}/fapi/v1/ticker/price?symbol=BTCUSDT", timeout=10)
        fut_price = float(r.json()["price"]) if r.status_code == 200 else None
        # Spot price (from Binance spot ticker)
        r2 = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=10)
        spot_price = float(r2.json()["price"]) if r2.status_code == 200 else None

        if fut_price and spot_price and spot_price > 0:
            basis_annualized = ((fut_price / spot_price) - 1) * 365 * 100  # in %
            basis_raw = (fut_price - spot_price) / spot_price * 100  # in %
            return basis_raw, basis_annualized
        return None, None
    except:
        return None, None


def compute_market_positioning_index(
    liq_long_vol=None, liq_short_vol=None, liq_total_24h=None,
    funding_rate=None, pc_oi=None,
):
    """
    Compute Market Positioning Index (MPI).

    Args:
        liq_long_vol:  24h long liquidation volume (USD)
        liq_short_vol: 24h short liquidation volume (USD)
        liq_total_24h: Total 24h liquidation volume (USD)
        funding_rate:  Current perpetual funding rate (from collect.py)
        pc_oi:         Put/Call OI ratio (from collect.py)

    Returns:
        (mpi_score_0_100, mpi_stress_0_1, details)
    """
    now = time.time()
    cache = _load_cache()

    # Check cache
    if now - cache.get("cached_at", 0) < CACHE_TTL and cache.get("mpi_score") is not None:
        return cache["mpi_score"], cache["sfc_stress"], cache.get("details", {})

    # Fetch fresh data
    fr_now, fr_avg, fr_accel = _funding_rate()
    pc_oi_opt, skew, total_oi, _ = _options_data()
    basis_raw, basis_ann = _binance_futures_basis()

    # Use passed values if provided (from collect.py pipeline)
    if funding_rate is not None:
        fr_now = funding_rate
    if pc_oi is not None:
        pc_oi_opt = pc_oi

    # ── Component scores (all 0-1, high = bearish positioning) ──
    comps = {}

    # 1. Funding Rate: high positive = crowded long = bearish (overheated)
    if fr_now is not None:
        if fr_now > 0.15:
            fr_score = 0.80  # extremely crowded long
        elif fr_now > 0.08:
            fr_score = 0.65
        elif fr_now > 0.03:
            fr_score = 0.55
        elif fr_now > 0.01:
            fr_score = 0.45  # normal
        elif fr_now > -0.01:
            fr_score = 0.40  # neutral
        elif fr_now > -0.05:
            fr_score = 0.60  # mild bearish funding = bearish sentiment
        else:
            fr_score = 0.75  # extreme negative funding = panic
        comps["funding_rate"] = (fr_score, 0.25)
    else:
        comps["funding_rate"] = (0.50, 0.25)

    # 2. Put/Call OI: high = hedging/fear = bearish
    if pc_oi_opt is not None:
        if pc_oi_opt > 1.5:
            pc_score = 0.80  # extreme put dominance
        elif pc_oi_opt > 1.0:
            pc_score = 0.60
        elif pc_oi_opt > 0.7:
            pc_score = 0.45  # normal range
        elif pc_oi_opt > 0.4:
            pc_score = 0.30  # call dominance = bullish
        else:
            pc_score = 0.20
        comps["put_call"] = (pc_score, 0.20)
    else:
        comps["put_call"] = (0.50, 0.20)

    # 3. Liquidation imbalance: one-sided liq = cascade risk
    if liq_long_vol is not None and liq_short_vol is not None:
        total = liq_long_vol + liq_short_vol
        if total > 0:
            long_ratio = liq_long_vol / total
            # High long liq ratio = long squeeze = short-term bullish
            # High short liq ratio = short squeeze = short-term bearish
            if long_ratio > 0.80:
                liq_score = 0.70  # long squeeze happening = stress
            elif long_ratio < 0.20:
                liq_score = 0.30  # short squeeze = bullish positioning
            elif long_ratio > 0.60:
                liq_score = 0.55
            else:
                liq_score = 0.40  # balanced
            # Amplify by total volume
            if liq_total_24h and liq_total_24h > 2_000_000_000:
                liq_score = min(0.95, liq_score + 0.10)
            comps["liquidation"] = (liq_score, 0.20)
        else:
            comps["liquidation"] = (0.50, 0.20)
    else:
        comps["liquidation"] = (0.50, 0.20)

    # 4. Basis: high premium = crowded long = overheated
    if basis_raw is not None:
        if basis_ann and basis_ann > 30:
            basis_score = 0.80  # very high premium = frothy
        elif basis_ann and basis_ann > 15:
            basis_score = 0.65
        elif basis_raw > 0.10:
            basis_score = 0.55  # moderate premium
        elif basis_raw > 0.03:
            basis_score = 0.45  # normal
        elif basis_raw > -0.05:
            basis_score = 0.40  # slight discount
        else:
            basis_score = 0.65  # backwardation = bearish
        comps["basis"] = (basis_score, 0.20)
    else:
        comps["basis"] = (0.50, 0.20)

    # 5. Total OI (volume): very high OI = max leverage = fragile
    if total_oi is not None and total_oi > 0:
        # Scale: $10B+ OI = extreme
        oi_b = total_oi / 1e9  # in billions of USD
        if oi_b > 20:
            oi_score = 0.75
        elif oi_b > 12:
            oi_score = 0.60
        elif oi_b > 6:
            oi_score = 0.50
        elif oi_b > 2:
            oi_score = 0.40
        else:
            oi_score = 0.30
        comps["open_interest"] = (oi_score, 0.15)
    else:
        comps["open_interest"] = (0.50, 0.15)

    # ── Weighted composite ──
    total_w = sum(w for _, w in comps.values())
    mpi_raw = sum(s * w for s, w in comps.values()) / total_w

    # Map to MPI 0-100
    mpi_score = (1.0 - mpi_raw) * 100  # Invert: high raw = bearish → low MPI
    mpi_score = max(0, min(100, mpi_score))

    # SFC stress mapping (high raw = high stress)
    if mpi_raw < 0.3:
        mpi_stress = 0.20  # bullish positioning = low stress
    elif mpi_raw < 0.45:
        mpi_stress = 0.35
    elif mpi_raw < 0.55:
        mpi_stress = 0.50  # neutral
    elif mpi_raw < 0.70:
        mpi_stress = 0.70  # bearish positioning = elevated stress
    else:
        mpi_stress = 0.85  # extreme bearish = high stress

    # Label
    if mpi_score > 70:
        label = "BULLISH_POSITIONING"
    elif mpi_score > 55:
        label = "NEUTRAL"
    elif mpi_score > 40:
        label = "CAUTIOUS"
    else:
        label = "BEARISH_POSITIONING"

    comp_detail = {}
    for name, (val, weight) in comps.items():
        comp_detail[name] = {
            "score": round(val, 3),
            "weight": weight,
            "contribution": round((val - 0.5) * weight / total_w, 4),
        }

    details = {
        "mpi_score": round(mpi_score, 1),
        "mpi_stress": round(mpi_stress, 3),
        "mpi_raw": round(mpi_raw, 3),
        "label": label,
        "components": comp_detail,
        "n_components": len(comps),
        "raw_data": {
            "funding_rate": fr_now,
            "pc_oi": pc_oi_opt,
            "basis_ann": round(basis_ann, 2) if basis_ann is not None else None,
            "total_oi_b": round(total_oi / 1e9, 2) if total_oi else None,
        },
        "status": "ok",
    }

    cache["mpi_score"] = mpi_score
    cache["sfc_stress"] = mpi_stress
    cache["details"] = details
    _save_cache(cache)

    return round(mpi_score, 1), round(mpi_stress, 3), details


if __name__ == "__main__":
    mpi, stress, det = compute_market_positioning_index()
    print(json.dumps({
        "mpi": mpi,
        "stress": stress,
        "label": det.get("label"),
        "components": det.get("components"),
        "raw_data": det.get("raw_data"),
    }, indent=2))
