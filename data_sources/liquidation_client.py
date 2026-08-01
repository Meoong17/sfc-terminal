"""
SFC Terminal — Liquidation Data Client
Aggregates real liquidation data from free + paid sources:
  1. OKX (free, no key) — real BTC liquidation orders
  2. CoinGlass (paid, API key) — heatmap & aggregated data

Caches aggressively (5 min) to respect rate limits.
Falls back to current proxy estimates (dvol-based) when unavailable.
"""

import json, os, time, math, requests
from pathlib import Path

# ── Cache ───────────────────────────────────────────────────
CACHE_FILE  = Path(__file__).parent.parent / '.liq_cache.json'
CACHE_TTL   = 300  # 5 min — respect rate limits

# ── CoinGlass ────────────────────────────────────────────────
COINGLASS_BASE    = "https://open-api-v4.coinglass.com/api"
COINGLASS_HEADERS = {}

_cg_key = os.getenv("COINGLASS_API_KEY", "")
if _cg_key:
    COINGLASS_HEADERS["CG-API-KEY"] = _cg_key

# Once CoinGlass reports its plan is insufficient (HTTP 401 / code 401 for the
# aggregated-heatmap endpoint, which requires the paid Professional+ plan), stop
# retrying it for the rest of this process. Without this, every cache miss calls
# the paid endpoint (wasting a request + logging noise) only to fall back to OKX
# anyway. Free-tier account stays on the OKX source, which already returns real
# liquidation data.
_cg_plan_insufficient = False

# ── OKX (free, no key) ──────────────────────────────────────
OKX_BASE = "https://www.okx.com"

# ── In-memory cache ──────────────────────────────────────────
_liq_cache = {"ts": 0, "data": {}}

def _load_cache():
    """Load cache from disk if not in memory."""
    global _liq_cache
    now = time.time()
    if _liq_cache["ts"] > now - CACHE_TTL:
        return _liq_cache["data"]
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("ts", 0) > now - CACHE_TTL:
                _liq_cache = cached
                return cached.get("data", {})
        except: pass
    return {}

def _save_cache(data):
    """Save cache to memory + disk."""
    global _liq_cache
    _liq_cache = {"ts": time.time(), "data": data}
    try:
        CACHE_FILE.write_text(json.dumps(_liq_cache))
    except: pass

# ── OKX Free Liquidation Orders ─────────────────────────────
def fetch_okx_liquidations(symbol="BTC-USDT-SWAP", limit=100):
    """
    Fetch real liquidation orders from OKX (free, no API key).
    Returns: {
        "long_vol_usd": float,   # Total LONG-side liquidated (longs forced to sell)
        "short_vol_usd": float,  # Total SHORT-side liquidated (shorts forced to buy)
        "order_count": int,
        "dominant": "long"|"short"|"balanced",
        "liq_intensity": float,  # 0-1 normalised
        "source": "okx"
    }

    NOTE (fix, 2026-07): prior to this fix, long_vol_usd/short_vol_usd and
    "dominant" were inverted from their names — long_vol_usd actually held
    SHORT liquidations and vice versa, "dominant='long'" actually meant
    "short liquidations dominate". collect.py's liq_pressure classification
    had a compensating (equally confusing) flip that happened to cancel
    this out for the FINAL "SHORT_SQUEEZE"/"LONG_SQUEEZE" label, but the
    raw long_vol_usd/short_vol_usd numbers shown directly on the dashboard
    (as "Long $XXXM / Short $XXXM") were genuinely displaying swapped
    values. Both this function AND collect.py's classification logic were
    fixed together — see collect.py's liq_pressure block for the matching
    other half of this fix.
    """
    url = f"{OKX_BASE}/api/v5/public/liquidation-orders"
    params = {
        "instType": "SWAP",
        "instFamily": "BTC-USDT",
        "state": "filled",
        "limit": min(limit, 100)
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        orders = data.get("data", [])
        if not orders:
            return None

        # Parse each liquidation event
        long_vol = 0.0   # longs liquidated (side=sell = long liq)
        short_vol = 0.0  # shorts liquidated (side=buy = short liq)
        count = 0

        for event in orders:
            details = event.get("details", [])
            for d in details:
                side = d.get("side", "")  # "sell" = long liq, "buy" = short liq
                sz = float(d.get("sz", 0))
                px = float(d.get("bkPx", 0))
                vol_usd = sz * px
                if side == "sell":   # longs being liquidated (selling pressure)
                    long_vol += vol_usd
                elif side == "buy":  # shorts being liquidated (buying pressure)
                    short_vol += vol_usd
                count += 1

        total = long_vol + short_vol
        if total == 0:
            return None

        # Determine dominant side (now correctly: long_vol = long-side liquidated)
        ratio = long_vol / total if total > 0 else 0.5
        if ratio > 0.65:
            dominant = "long"       # long liquidations dominating (longs forced to sell)
        elif ratio < 0.35:
            dominant = "short"      # short liquidations dominating (shorts forced to buy)
        else:
            dominant = "balanced"

        # Intensity: normalise by historical max (~$500M is extreme for 1h)
        intensity = min(total / 500_000_000, 1.0)

        return {
            "long_vol_usd": round(long_vol, 2),
            "short_vol_usd": round(short_vol, 2),
            "total_vol_usd": round(total, 2),
            "order_count": count,
            "dominant": dominant,
            "long_ratio": round(ratio, 3),
            "liq_intensity": round(intensity, 3),
            "source": "okx"
        }

    except requests.RequestException:
        return None
    except (ValueError, KeyError, TypeError) as e:
        print(f"[liq] OKX parse error: {e}", file=__import__('sys').stderr)
        return None


# ── CoinGlass (paid, Professional+ plan required) ────────────
def fetch_coinglass_heatmap(symbol="BTC", range_days="3d"):
    """
    Fetch CoinGlass aggregated liquidation heatmap (paid plan).
    Returns None if plan insufficient or API error.
    """
    if not _cg_key:
        return None

    # Skip the paid endpoint entirely once we know the current plan can't serve it.
    global _cg_plan_insufficient
    if _cg_plan_insufficient:
        return None

    url = f"{COINGLASS_BASE}/futures/liquidation/aggregated-heatmap/model1"
    params = {"symbol": symbol, "range": range_days}

    try:
        r = requests.get(url, headers=COINGLASS_HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("code") == "401":
            # Plan insufficient — don't retry this session
            _cg_plan_insufficient = True
            print("[liq] CoinGlass: plan upgrade needed for liquidation data (skipping for this session)", file=__import__('sys').stderr)
            return None
        if data.get("code") != "0":
            return None
        return {"data": data.get("data", {}), "source": "coinglass"}
    except requests.RequestException:
        return None


# ── Public API ──────────────────────────────────────────────
def get_liquidation_data(symbol_btc="BTC", symbol_swap="BTC-USDT"):
    """
    Get best available liquidation data.
    Priority: CoinGlass > OKX > fallback (None = caller uses proxy)
    Respects 5-min cache.
    """
    # Check cache
    cached = _load_cache()
    if cached:
        return cached

    result = None

    # 1. Try CoinGlass (paid, best quality)
    cg = fetch_coinglass_heatmap(symbol_btc)
    if cg:
        result = cg

    # 2. Try OKX (free)
    if not result:
        okx = fetch_okx_liquidations(symbol_swap)
        if okx:
            result = okx

    # Cache whatever we got (even None)
    _save_cache(result)
    return result
