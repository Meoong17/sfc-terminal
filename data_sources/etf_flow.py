#!/usr/bin/env python3
"""
SFC ETF Flow Module (M81-M82)
===============================
M81 — ETF Net Flow (daily total inflow/outflow, USD)
M82 — Cumulative BTC Holdings trend

Data sources (tried in order):
  1. Local cache file (.etf_cache.json) — populated by cron job or manual
  2. CoinGlass API (capi.coinglass.com) — requires session cookies
  3. Estimation from price × AUM deltas (last resort)

Cache auto-updated every 6h by cron job 'ETF Cache Update' which scrapes
https://farside.co.uk/btc/ via browser (bypasses Cloudflare).

To manually seed: python3 -c "from etf_flow import seed_cache_from_browser; ..."
To force refresh: delete .etf_cache.json and wait for cron (or run update_etf_cache.py)

Cache file format (.etf_cache.json):
  {
    "flows": [
      {"date": "2026-06-17", "total_btc": -783.0, "total_usd": -51400000,
       "etfs": {"GBTC": -236.12, "IBIT": 0, "FBTC": 213.27, ...}}
    ],
    "cumulative_btc": 676776.58,
    "cumulative_usd": 53954600000,
    "last_update": "2026-06-17T19:35:00"
  }
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Cache path ──
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.etf_cache.json')
CACHE_TTL = 43200  # 12 hours (ETF data updates daily after market close)

# ── ETF tickers (US Spot BTC ETFs) ──
ETF_TICKERS = ["GBTC", "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL",
               "BRRR", "EZBC", "BTCW", "BTC", "MSBT"]

# ── Cache helpers ──
def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"flows": [], "cumulative_btc": 0, "cumulative_usd": 0,
                "last_update": None, "cached_at": 0}

def _save_cache(cache):
    cache["cached_at"] = time.time()
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ── Seed data (populated from CoinGlass Jun 2026) ──
_SEED_FLOWS = [
    # (date, total_btc, total_usd_est)
    # USD estimated as BTC * ~$65K price proxy
]

# ── Primary: CoinGlass API (needs browser cookies) ──
def _fetch_from_coinglass():
    """Try CoinGlass ETF flow API. Returns list of flow dicts or None."""
    try:
        r = requests.get(
            "https://capi.coinglass.com/api/etf/flow",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.coinglass.com/etf/bitcoin",
                "Accept": "application/json",
            },
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("success") and data.get("data"):
                return data["data"]
    except Exception:
        pass
    return None

# ── Secondary: Farside scrape (Cloudflare protected, rarely works) ──
def _fetch_from_farside():
    """Try Farside UK page. Returns list of flow dicts or None."""
    try:
        r = requests.get(
            "https://farside.co.uk/btc/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        if r.status_code == 200:
            # Try to parse the HTML table
            import re
            # Look for the total row in the main table
            # The table on farside shows per-ETF flows
            # Format: rows with date followed by values
            html = r.text
            # Simple regex to find rows with flow data
            # This is fragile but Farside is a fallback
            return None  # Cloudflare blocks most requests
    except Exception:
        pass
    return None

# ── Public API ──

def compute_etf_metrics(btc_price=None):
    """Compute M81 and M82 scores.

    Returns:
        (m81_score, m82_score, details)
        m81_score: 0-1 where high = high stress (large outflows)
        m82_score: 0-1 where high = high stress (declining holdings)
        details: dict with raw data
    """
    cache = _load_cache()
    now = time.time()

    # Try to fetch fresh data if cache is stale
    if now - cache.get("cached_at", 0) > CACHE_TTL:
        fresh = None
        fresh = _fetch_from_coinglass()
        if fresh is None:
            fresh = _fetch_from_farside()

        if fresh and isinstance(fresh, list) and len(fresh) > 0:
            # Parse CoinGlass API response
            # Format depends on API — try to normalize
            cache["flows"] = _normalize_coinglass_data(fresh)
            cache["cached_at"] = now
            _save_cache(cache)

    flows = cache.get("flows", [])

    if not flows:
        # No data available
        return 0.5, 0.5, {
            "m81_net_flow_usd": None,
            "m82_cumulative_btc": None,
            "status": "no_data",
            "note": "ETF data unavailable — need cache seed or API access"
        }

    # ── M81: ETF Net Flow Score ──
    # Use last 5 trading days of net flows (in BTC)
    #
    # FIX (found via live cache inspection, 2026-07): two compounding bugs
    # were here previously:
    #   1. `flows[:5]` assumed the cache was newest-first, but it's
    #      actually stored chronologically ascending (oldest first) —
    #      confirmed live: first entry was 2024-01-11, last was
    #      2026-07-20. Taking flows[:5] silently grabbed 2024 data, not
    #      recent data, every single cycle.
    #   2. `total_btc` is never populated by the cache-writer (always
    #      None) — confirmed in EVERY entry checked, including the most
    #      recent. The per-ETF breakdown dict (`etfs`) IS populated
    #      though, so total_btc is reconstructed as sum(etfs.values())
    #      rather than silently discarding every entry as "no usable data."
    def _entry_total_btc(f):
        if f.get("total_btc") is not None:
            return f["total_btc"]
        etfs = f.get("etfs")
        if isinstance(etfs, dict) and etfs:
            return sum(v for v in etfs.values() if v is not None)
        return None

    sorted_flows = sorted(flows, key=lambda f: f.get("date", ""), reverse=True)
    recent_flows = []
    for f in sorted_flows:
        tb = _entry_total_btc(f)
        if tb is not None:
            recent_flows.append({**f, "total_btc": tb})
        if len(recent_flows) >= 5:
            break

    if not recent_flows:
        return 0.5, 0.5, {"status": "no_recent_flows"}

    # ── Staleness guard (Audit 2026-08-03) ──
    # The cache can hold old/hardcoded rows (e.g. a manual backfill of a past
    # month) that are still "fresh" by the cached_at TTL. If the NEWEST flow
    # date in the cache is old, do NOT let that stale data move Rt/Lt — return
    # neutral instead. This prevents a stale/hardcoded cache from feeding a
    # fabricated ETF flow signal into the factor adjustments.
    ETF_MAX_STALE_DAYS = 10
    try:
        _newest_dt = datetime.strptime(sorted_flows[0]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        _age_days = (datetime.now(timezone.utc) - _newest_dt).days
    except Exception:
        _age_days = None
    if _age_days is not None and _age_days > ETF_MAX_STALE_DAYS:
        return 0.5, 0.5, {
            "status": "stale",
            "latest_flow_date": sorted_flows[0]["date"],
            "age_days": _age_days,
            "note": f"ETF cache not updated in {_age_days}d — returning neutral so stale data does not move the score",
        }

    # Average daily BTC flow over last 5 days
    avg_flow_btc = sum(f["total_btc"] for f in recent_flows) / len(recent_flows)
    latest_flow_btc = recent_flows[0]["total_btc"] if recent_flows else 0

    # Score: large sustained outflows = stress (high score)
    # BTC flow is in BTC units. Typical flows range -5000 to +5000 BTC/day
    # Normalize: -2000 BTC/day sustained = 0.8 stress
    #            0 BTC = 0.5 neutral
    #            +2000 BTC/day = 0.2 low stress
    if avg_flow_btc < -1000:
        m81_score = 0.85  # heavy sustained outflows
    elif avg_flow_btc < -500:
        m81_score = 0.70
    elif avg_flow_btc < -100:
        m81_score = 0.60
    elif avg_flow_btc < 100:
        m81_score = 0.50  # neutral
    elif avg_flow_btc < 500:
        m81_score = 0.35
    elif avg_flow_btc < 1000:
        m81_score = 0.25
    else:
        m81_score = 0.15  # heavy inflows → bullish

    # ── M82: Cumulative Holdings Trend ──
    # Compare cumulative BTC holdings change over last 20 trading days
    # (about 1 month)
    cum = cache.get("cumulative_btc")
    if cum is None or cum == 0:
        # Reconstruct from flow entries if cache summary is missing
        cum = 0
        for f in flows:
            tb = _entry_total_btc(f)
            if tb is not None:
                cum += tb

    # If we have enough historical data, compute the trend
    if len(flows) >= 20:
        # FIX: same root cause as M81 above — total_btc is never
        # populated by the cache-writer, so this loop previously never
        # added anything to cumulative_over_time (the `is not None`
        # check always failed), meaning M82 was ALWAYS stuck at the
        # neutral 0.50 fallback further below. Reuse the same
        # reconstruct-from-etfs-breakdown helper defined above.
        cumulative_over_time = []
        running = 0
        for f in reversed(flows):
            tb = _entry_total_btc(f)
            if tb is not None:
                running += tb
                cumulative_over_time.append(running)
        cumulative_over_time.reverse()

        cum_change = None
        if len(cumulative_over_time) >= 20:
            cum_now = cumulative_over_time[0]
            cum_20d = cumulative_over_time[min(19, len(cumulative_over_time)-1)]
            cum_change = cum_now - cum_20d

            # Cumulative BTC holdings increasing = bullish (low stress)
            if cum_change and cum_change > 50000:
                m82_score = 0.15
            elif cum_change > 20000:
                m82_score = 0.25
            elif cum_change > 5000:
                m82_score = 0.35
            elif cum_change > -5000:
                m82_score = 0.50  # stable
            elif cum_change > -20000:
                m82_score = 0.65
            elif cum_change > -50000:
                m82_score = 0.80
            else:
                m82_score = 0.90
        else:
            m82_score = 0.50
    else:
        # Not enough history — use latest cumulative value
        if cum > 500000:
            m82_score = 0.20  # strong institutional adoption
        elif cum > 200000:
            m82_score = 0.35
        elif cum > 50000:
            m82_score = 0.50
        else:
            m82_score = 0.50

    details = {
        "m81_avg_flow_5d_btc": round(avg_flow_btc, 1),
        "m81_latest_flow_btc": round(latest_flow_btc, 1),
        "m81_flow_trend": "OUTFLOW" if avg_flow_btc < -100 else "INFLOW" if avg_flow_btc > 100 else "NEUTRAL",
        "m82_cumulative_btc": round(cum, 1) if cum is not None else 0.0,
        "m82_cumulative_usd": round(cache.get("cumulative_usd") or 0, 0),
        "m82_trend_20d_btc": round(cum_change, 1) if 'cum_change' in dir() else None,
        "flows_count": len(flows),
        "recent_days": len(recent_flows),
        "status": "ok",
    }

    return round(m81_score, 3), round(m82_score, 3), details


def _normalize_coinglass_data(raw):
    """Normalize CoinGlass API response to our format."""
    result = []
    # CoinGlass typically returns array of {date, flowData}
    # Each flow entry has ticker-level BTC flows
    try:
        for entry in raw:
            date = entry.get("date", entry.get("time", "")).split("T")[0]
            total_btc = entry.get("total", entry.get("totalFlow", 0))
            if isinstance(total_btc, str):
                total_btc = float(total_btc.replace(",", "").replace("K", "000").replace("M", "000000"))
            etfs = {}
            for t in ETF_TICKERS:
                val = entry.get(t.lower(), entry.get(t, 0))
                if isinstance(val, str):
                    val = float(val.replace(",", "").replace("+", "").replace("K", "000").replace("M", "000000"))
                etfs[t] = val
            result.append({
                "date": date,
                "total_btc": float(total_btc) if total_btc else 0,
                "total_usd": None,  # CoinGlass gives BTC, not USD
                "etfs": etfs,
            })
    except Exception:
        pass
    return result


def seed_cache_from_browser(flows_data, cumulative_btc, cumulative_usd):
    """Manually seed the cache with ETF data (e.g., from browser scrape).

    Args:
        flows_data: list of dicts with date, total_btc, total_usd, etfs
        cumulative_btc: total BTC held across all ETFs
        cumulative_usd: total AUM in USD
    """
    cache = {
        "flows": flows_data,
        "cumulative_btc": cumulative_btc,
        "cumulative_usd": cumulative_usd,
        "last_update": datetime.now(timezone.utc).isoformat(),
        "cached_at": time.time(),
        "seeded": True,
    }
    _save_cache(cache)
    print(f"[ETF] Cache seeded: {len(flows_data)} flow days, "
          f"{cumulative_btc:,.0f} BTC cumulative", file=sys.stderr)


# ── Run directly: test module ──
if __name__ == "__main__":
    m81, m82, details = compute_etf_metrics()
    print(json.dumps({"m81_score": m81, "m82_score": m82, "details": details}, indent=2))
