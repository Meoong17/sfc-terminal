#!/usr/bin/env python3
"""
SFC Stablecoin Liquidity Module (M76-M80) — Layer 2 Crypto Liquidity
====================================================================
M76 — Stablecoin Supply Growth (7d & 30d % change in total supply)
M77 — SSR (Stablecoin Supply Ratio = BTC mcap / Stablecoin mcap)
M78 — Exchange Flow (stablecoin netflow to/from exchanges)
M79 — Stablecoin Velocity (tx volume / supply — proxy via exchange flow)
M80 — Stablecoin Dominance (stablecoin mcap / total crypto mcap)

Data sources:
  - CoinGecko free API: USDT, USDC, DAI total market caps (historical daily)
  - ErcinDedeoglu on-chain: exchange reserve, netflow, inflows/outflows
"""

import json, os, sys, math, time, requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Cache ──────────────────────────────────────────────────────────────
_STABLECOIN_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.stablecoin_cache.json')
CACHE_TTL = 43200  # 12 hours for stablecoin market cap data

def _load_cache():
    try:
        with open(_STABLECOIN_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cached_at": 0, "raw": {}, "supply_history": {}}

def _save_cache(cache):
    cache["cached_at"] = time.time()
    with open(_STABLECOIN_CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── Stablecoin IDs for CoinGecko ──────────────────────────────────────
STABLECOIN_IDS = ["tether", "usd-coin", "dai", "first-digital-usd"]
# CoinGecko ID → DefiLlama symbol (for fallback)
DEFILLAMA_SYMBOLS = {"tether": "USDT", "usd-coin": "USDC", "dai": "DAI", "first-digital-usd": "FDUSD"}
CG_BASE = "https://api.coingecko.com/api/v3"
DL_BASE = "https://stablecoins.llama.fi"
# Audit 2026-08-03: demo key was hardcoded in source (committed secret).
# Now read from env; empty => CoinGecko 401 => _fetch_cg() degrades to neutral.
# Audit 2026-08-10: use header auth (x-cg-demo-api-key / x-cg-pro-api-key),
# NOT the demo-only query param, which is rejected for pro keys. If the key is
# missing we hit the free public tier (works but throttles on parallel bursts).
_CG_KEY = os.getenv("COINGECKO_API_KEY", "")

def _cg_headers():
    h = {"Accept": "application/json"}
    if _CG_KEY:
        # CoinGecko keys auth via header; demo vs pro key differ by header name.
        # Send whichever applies; the free tier works with no key at all.
        h["x-cg-demo-api-key"] = _CG_KEY
        h["x-cg-pro-api-key"] = _CG_KEY
    return h

def _fetch_cg_single(url, retries=2):
    """GET a CoinGecko URL with retry + exponential backoff on 429/5xx.
    retries=2 keeps fallback latency bounded when CG is fully throttled;
    DefiLlama covers the all-down case."""
    import time as _time
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_cg_headers(), timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                backoff = 2 ** attempt + (attempt * 1.5)
                print(f"[SC] CG {r.status_code} on {url[:55]}... retry in {backoff:.1f}s", file=sys.stderr)
                _time.sleep(backoff)
                continue
            # 401/403/404 → auth/not-found, no point retrying
            return None
        except Exception as e:
            print(f"[SC] CG fetch failed: {url[:60]}... — {e}", file=sys.stderr)
            return None
    return None

def _fetch_market_chart(coin_id, days=30):
    """Fetch stablecoin market-cap history. days=30 (not 365) — enough for
    M76 7d/30d growth and far smaller payload (less throttling)."""
    url = f"{CG_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    data = _fetch_cg_single(url)
    if data and "market_caps" in data and data["market_caps"]:
        return data["market_caps"]  # [[ts, mcap], ...]
    return None

# ── DefiLlama fallback (free, no rate limit) ─────────────────────────
def _fetch_defillama_supply():
    """Fetch 4-coin stablecoin circulating supply snapshot from DefiLlama.

    Returns {coin_id: {"latest": mcap, "mcap_30d_ago": mcap, "mcap_7d_ago": mcap}}
    or None on failure. DefiLlama provides cur / prevDay / prevWeek / prevMonth
    per asset, which is exactly what M76 growth needs — no full 365d series,
    but always returns all coins (no partial-sum poisoning).
    """
    try:
        r = requests.get(f"{DL_BASE}/stablecoins", headers={"Accept": "application/json"}, timeout=25)
        if r.status_code != 200:
            return None
        data = r.json()
        by_sym = {}
        for a in data.get("peggedAssets", []):
            by_sym[a.get("symbol")] = a
        out = {}
        for cid, sym in DEFILLAMA_SYMBOLS.items():
            a = by_sym.get(sym)
            if not a:
                continue
            cur = (a.get("circulating") or {}).get("peggedUSD")
            pw = (a.get("circulatingPrevWeek") or {}).get("peggedUSD")
            pm = (a.get("circulatingPrevMonth") or {}).get("peggedUSD")
            if cur and pm:
                out[cid] = {"latest": cur, "mcap_30d_ago": pm, "mcap_7d_ago": pw if pw else None}
        return out
    except Exception as e:
        print(f"[SC] DefiLlama fallback failed: {e}", file=sys.stderr)
        return None

def _fetch_supply_from_defillama():
    """Build supply_history (day-keyed dict) from DefiLlama snapshot.

    DefiLlama lacks a per-coin daily series endpoint, so we synthesize a
    minimal history with latest / 7d-ago / 30d-ago timestamps (UTC). This is
    enough for M76 growth, M77 SSR, M80 dominance — the same fields M76-M80
    actually consume.
    """
    snap = _fetch_defillama_supply()
    if not snap:
        return {}
    today = time.time()
    points = {0: today, 7: today - 7 * 86400, 30: today - 30 * 86400}
    history = {}
    for off, ts in points.items():
        day = int(ts / 86400) * 86400
        date_str = datetime.fromtimestamp(day, tz=timezone.utc).strftime("%Y-%m-%d")
        total = 0.0
        ok = True
        for cid in STABLECOIN_IDS:
            s = snap.get(cid)
            if off == 0:
                v = s["latest"] if s else None
            elif off == 7:
                v = s["mcap_7d_ago"] if s else None
            else:
                v = s["mcap_30d_ago"] if s else None
            if v is None:
                ok = False
                break
            total += v
        if ok:
            history[date_str] = round(total, 2)
    return history

def fetch_stablecoin_supply_history(force_refresh=False):
    """Fetch total stablecoin market cap history (daily, 365d).
    
    Returns dict of { date_str: total_mcap_usd } sorted by date.
    """
    cache = _load_cache()
    now = time.time()
    
    if not force_refresh and (now - cache.get("cached_at", 0)) < CACHE_TTL:
        if cache.get("supply_history"):
            print(f"[SC] Using cached stablecoin data ({now - cache['cached_at']:.0f}s old)", file=sys.stderr)
            return cache["supply_history"]
    
    print("[SC] Fetching stablecoin market cap data from CoinGecko...", file=sys.stderr)
    
    # Fetch stablecoin market charts in parallel (max 2 workers — lower than
    # 4 to avoid CoinGecko 429 on parallel bursts). Retry/backoff handled per
    # request in _fetch_cg_single.
    all_series = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_fetch_market_chart, sid): sid for sid in STABLECOIN_IDS}
        for f in as_completed(futures):
            sid = futures[f]
            result = f.result()
            if result:
                all_series[sid] = result
                print(f"[SC] {sid}: {len(result)} data points", file=sys.stderr)

    # If CoinGecko failed to return ALL stablecoins, fall back to DefiLlama.
    # CG returning a subset (e.g. only 2 of 4 coins on 429) would poison the
    # partial-sum guard into building a reduced-supply history (260B -> 187B),
    # so we prefer a complete DefiLlama snapshot (always 4 coins) instead.
    if len(all_series) < len(STABLECOIN_IDS):
        missing = [sid for sid in STABLECOIN_IDS if sid not in all_series]
        print(f"[SC] CoinGecko partial ({len(all_series)}/{len(STABLECOIN_IDS)}) — "
              f"missing {missing}. Falling back to DefiLlama...", file=sys.stderr)
        dl_hist = _fetch_supply_from_defillama()
        if dl_hist:
            cache["supply_history"] = dl_hist
            cache["raw"]["n_stablecoins"] = len(STABLECOIN_IDS)
            cache["raw"]["source"] = "defillama"
            _save_cache(cache)
            print(f"[SC] DefiLlama fallback: {len(dl_hist)} synthetic day-points, "
                  f"{len(STABLECOIN_IDS)} stablecoins", file=sys.stderr)
            return dl_hist
        print("[SC] DefiLlama fallback also failed — using partial CoinGecko data", file=sys.stderr)
    
    if not all_series:
        print("[SC] WARNING: No stablecoin data fetched, using fallback estimates", file=sys.stderr)
        cache["cached_at"] = now
        _save_cache(cache)
        return cache.get("supply_history", {})
    
    # Merge: sum mcaps across stablecoin IDs for each day, but ONLY count a
    # day when EVERY stablecoin has a data point. A partial day (one coin
    # missing from its market_chart that day) would otherwise sum a subset and
    # be misread as a supply collapse (e.g. 260B -> 4.9B when only DAI+FDUSD
    # reported that day), which poisons M76 growth, SSR and SLI.
    merged = {}      # day_key -> total_mcap
    day_counts = {}  # day_key -> # of distinct coins with a point that day
    n_coins = len(all_series)
    for sid, series in all_series.items():
        for ts, mcap in series:
            day_key = int(ts / 86400) * 86400  # daily bucket
            merged[day_key] = merged.get(day_key, 0) + mcap
            day_counts[day_key] = day_counts.get(day_key, 0) + 1

    # Convert to sorted date-keyed dict, dropping incomplete days
    history = {}
    for ts in sorted(merged.keys()):
        if day_counts[ts] < n_coins:
            # incomplete day (not all stablecoins present) — skip so it can't
            # be read as a real supply change
            continue
        date_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        history[date_str] = round(merged[ts], 2)
    
    # Save to cache
    cache["supply_history"] = history
    cache["raw"]["n_stablecoins"] = len(all_series)
    _save_cache(cache)
    
    print(f"[SC] Built supply history: {len(history)} days, {len(all_series)} stablecoins", file=sys.stderr)
    return history

def get_latest_stablecoin_mcap(supply_history):
    """Get latest and 30d-ago stablecoin market cap from history."""
    if not supply_history:
        return None, None, None, None
    
    dates = sorted(supply_history.keys())
    latest_date = dates[-1]
    latest_mcap = supply_history[latest_date]
    
    # 30 days ago
    target = None
    for d in reversed(dates):
        try:
            if (datetime.strptime(latest_date, "%Y-%m-%d") - 
                datetime.strptime(d, "%Y-%m-%d")).days >= 30:
                target = d
                break
        except ValueError:
            continue
    
    if target is None and len(dates) > 1:
        target = dates[-min(len(dates), 30)]  # fallback
    
    mcap_30d_ago = supply_history.get(target, latest_mcap)
    return latest_mcap, mcap_30d_ago, latest_date, target


# ── M76: Stablecoin Supply Growth ─────────────────────────────────────
def calculate_m76_supply_growth(supply_history):
    """Score 0.0-1.0: higher supply growth = more liquidity entering crypto.
    
    Uses 30-day % change. >5% monthly growth = strong bullish signal.
    """
    latest_mcap, mcap_30d_ago, _, _ = get_latest_stablecoin_mcap(supply_history)
    
    if not latest_mcap or not mcap_30d_ago or mcap_30d_ago == 0:
        return None, {"growth_30d_pct": None, "growth_7d_pct": None, "latest_mcap": None}
    
    growth_30d = (latest_mcap - mcap_30d_ago) / mcap_30d_ago * 100
    
    # Also compute 7d growth if possible. Use date-based lookup (nearest point
    # >=7 days before latest) rather than dates[-7], so it works for BOTH full
    # CoinGecko daily history AND the sparse 3-point DefiLlama fallback history.
    dates = sorted(supply_history.keys())
    growth_7d = None
    latest_date = dates[-1] if dates else None
    mcap_7d_ago = None
    if latest_date:
        try:
            latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            for d in reversed(dates):
                if (latest_dt - datetime.strptime(d, "%Y-%m-%d")).days >= 7:
                    mcap_7d_ago = supply_history.get(d)
                    break
        except ValueError:
            mcap_7d_ago = None
    if mcap_7d_ago and mcap_7d_ago > 0:
        growth_7d = (latest_mcap - mcap_7d_ago) / mcap_7d_ago * 100
    
    # Score: sigmoid on growth rate
    # -2% monthly = score 0.1 (contracting)
    # 0% = score 0.3 (neutral/flat)
    # +2% = score 0.5 (moderate)
    # +5% = score 0.7 (strong)
    # +10% = score 0.9 (very strong)
    score = 1.0 / (1.0 + math.exp(-0.4 * (growth_30d - 2)))
    # Clip and scale
    score = max(0.0, min(1.0, score))
    
    detail = {
        "growth_30d_pct": round(growth_30d, 2),
        "growth_7d_pct": round(growth_7d, 2) if growth_7d is not None else None,
        "latest_mcap": round(latest_mcap, 0),
        "mcap_30d_ago": round(mcap_30d_ago, 0),
        "status": "ok",
    }
    
    return round(score, 3), detail


# ── M77: SSR (Stablecoin Supply Ratio) ────────────────────────────────
def calculate_m77_ssr(supply_history, btc_mcap):
    """Score 0.0-1.0: SSR = BTC mcap / Stablecoin mcap.
    
    Low SSR (stablecoins large vs BTC) = buying power accumulated.
    High SSR (stablecoins small vs BTC) = limited dry powder.
    
    SSR < 5 = strong bullish (lots of dry powder)
    SSR 5-10 = moderate
    SSR > 15 = limited buying power
    """
    latest_mcap, _, _, _ = get_latest_stablecoin_mcap(supply_history)
    
    if not latest_mcap or not btc_mcap or latest_mcap == 0:
        return None, {"ssr": None, "btc_mcap": btc_mcap, "stablecoin_mcap": latest_mcap}
    
    ssr = btc_mcap / latest_mcap
    
    # Score: lower SSR = higher score (more buying power)
    # SSR 3 = score ~0.9 (lots of dry powder)
    # SSR 6 = score ~0.5 (neutral)
    # SSR 9 = score ~0.3 (limited)
    # SSR 15 = score ~0.1 (very limited)
    score = 1.0 / (1.0 + math.exp(0.35 * (ssr - 6)))
    score = max(0.0, min(1.0, score))
    
    # Label
    if ssr < 4:
        label = "HIGH_BUYING_POWER"
    elif ssr < 7:
        label = "MODERATE"
    elif ssr < 12:
        label = "LIMITED"
    else:
        label = "DEPLETED"
    
    detail = {
        "ssr": round(ssr, 2),
        "stablecoin_mcap": round(latest_mcap, 0),
        "btc_mcap": round(btc_mcap, 0),
        "label": label,
        "status": "ok",
    }
    
    return round(score, 3), detail


# ── M78: Exchange Flow ────────────────────────────────────────────────
def calculate_m78_exchange_flow(onchain_details):
    """Score 0.0-1.0: stablecoin netflow to/from exchanges.
    
    Positive netflow (into exchanges) = buying power ready = bullish.
    Negative netflow (leaving exchanges) = accumulation / cold storage.
    
    Uses stablecoin_exchange_netflow from ErcinDedeoglu on-chain data.
    """
    netflow_data = onchain_details.get("stablecoin_exchange_netflow", {})
    netflow_value = netflow_data.get("value") if isinstance(netflow_data, dict) else None
    
    if netflow_value is None:
        # Fallback to raw netflow score
        netflow_score = onchain_details.get("stablecoin_exchange_netflow", {}).get("score", 50)
        if isinstance(netflow_score, (int, float)):
            return round(netflow_score / 100.0, 3), {
                "netflow": None,
                "netflow_score": netflow_score,
                "status": "fallback_score",
            }
        return None, {"netflow": None, "status": "unavailable"}
    
    # Score based on netflow magnitude relative to reserve size
    reserve_data = onchain_details.get("stablecoin_reserve", {})
    reserve_value = reserve_data.get("value") if isinstance(reserve_data, dict) else None
    
    netflow_norm = 0
    if reserve_value and reserve_value > 0:
        netflow_norm = netflow_value / reserve_value * 100  # % of reserve
    else:
        # Absolute scoring if no reserve reference
        netflow_norm = netflow_value / 1e9  # normalized to billions
    
    # Positive netflow = stablecoins entering exchanges = bullish
    score = 1.0 / (1.0 + math.exp(-0.8 * (netflow_norm - 0.5)))
    score = max(0.0, min(1.0, score))
    
    # Label
    if netflow_value > 0:
        if netflow_norm > 2:
            label = "SURGE_INFLOW"
        elif netflow_norm > 0.5:
            label = "STRONG_INFLOW"
        else:
            label = "MILD_INFLOW"
    else:
        if abs(netflow_norm) > 2:
            label = "SURGE_OUTFLOW"
        elif abs(netflow_norm) > 0.5:
            label = "STRONG_OUTFLOW"
        else:
            label = "MILD_OUTFLOW"
    
    detail = {
        "netflow": round(netflow_value, 2),
        "netflow_pct_of_reserve": round(netflow_norm, 2),
        "reserve": round(reserve_value, 0) if reserve_value else None,
        "label": label,
        "status": "ok",
    }
    
    return round(score, 3), detail


# ── M79: Stablecoin Velocity ──────────────────────────────────────────
def calculate_m79_velocity(onchain_details):
    """Score 0.0-1.0: Velocity = (Inflow + Outflow) / Reserve.
    
    Proxy: total exchange volume / exchange reserve.
    Higher velocity = money actively moving = early cycle signal.
    Very high velocity can also signal speculative excess.
    """
    inflow_data = onchain_details.get("stablecoin_exchange_inflow", {})
    outflow_data = onchain_details.get("stablecoin_exchange_outflow", {})
    reserve_data = onchain_details.get("stablecoin_reserve", {})
    
    inflow = inflow_data.get("value") if isinstance(inflow_data, dict) else None
    outflow = outflow_data.get("value") if isinstance(outflow_data, dict) else None
    reserve = reserve_data.get("value") if isinstance(reserve_data, dict) else None
    
    if inflow is None or outflow is None or not reserve or reserve == 0:
        return None, {"velocity": None, "status": "unavailable"}
    
    velocity = (inflow + outflow) / reserve  # daily turnover ratio
    
    # Score: moderate velocity = healthy, very high = speculative
    # velocity 0.0-0.02 = dormant (low score)
    # velocity 0.02-0.08 = active (high score)
    # velocity > 0.15 = speculative (moderate score)
    if velocity < 0.01:
        score = velocity / 0.01 * 0.5
    elif velocity < 0.08:
        score = 0.5 + (velocity - 0.01) / 0.07 * 0.4  # 0.5 -> 0.9
    elif velocity < 0.15:
        score = 0.9 - (velocity - 0.08) / 0.07 * 0.2  # 0.9 -> 0.7
    else:
        score = 0.7 - min(0.3, (velocity - 0.15) * 0.5)  # declining from 0.7
    
    score = max(0.0, min(1.0, score))
    
    if velocity < 0.01:
        label = "DORMANT"
    elif velocity < 0.04:
        label = "LOW"
    elif velocity < 0.10:
        label = "ACTIVE"
    else:
        label = "SPECULATIVE"
    
    detail = {
        "velocity": round(velocity, 4),
        "daily_inflow": round(inflow, 0),
        "daily_outflow": round(outflow, 0),
        "reserve": round(reserve, 0),
        "label": label,
        "status": "ok",
    }
    
    return round(score, 3), detail


# ── M80: Stablecoin Dominance ─────────────────────────────────────────
def calculate_m80_dominance(supply_history, btc_mcap, btc_dominance_pct):
    """Score 0.0-1.0: Stablecoin Dominance = Stablecoin mcap / Total Crypto mcap%.
    
    Uses BTC dominance to estimate total crypto mcap: total = btc_mcap / (dom/100)
    
    Interpretation:
      - High & rising = capital preservation / risk-off
      - High & falling = rotation into risk assets (early bull)
      - Low & stable = risk-on (late cycle)
      - Very low = euphoria (late stage)
    """
    latest_mcap, mcap_30d_ago, _, _ = get_latest_stablecoin_mcap(supply_history)
    
    if not latest_mcap or not btc_mcap or btc_mcap == 0:
        return None, {"dominance_pct": None, "status": "unavailable"}
    
    # Estimate total crypto market cap
    if btc_dominance_pct and btc_dominance_pct > 0:
        total_crypto_mcap = btc_mcap / (btc_dominance_pct / 100)
    else:
        total_crypto_mcap = btc_mcap * 1.8  # rough fallback
    
    dom_pct = (latest_mcap / total_crypto_mcap) * 100
    
    # Also compute 30d change
    dom_30d_ago_pct = None
    if mcap_30d_ago and mcap_30d_ago > 0 and btc_dominance_pct:
        # approximate: use same btc_dominance for 30d ago as proxy
        total_30d = btc_mcap / (btc_dominance_pct / 100)
        dom_30d_ago_pct = (mcap_30d_ago / total_30d) * 100
    
    dom_change_30d = None
    if dom_30d_ago_pct:
        dom_change_30d = dom_pct - dom_30d_ago_pct
    
    # Score: moderate dominance = bearish (money parked), falling = bullish
    # dom < 2% = euphoric (low score - late cycle risk)
    # dom 2-5% = normal range
    # dom 5-8% = elevated (money parked)
    # dom > 8% = extreme risk-off
    # If dom is falling over 30d = bullish rotation
    
    if dom_pct < 2:
        base_score = 0.2  # euphoric / low stablecoin dominance
    elif dom_pct < 5:
        base_score = 0.5  # normal range
    elif dom_pct < 10:
        base_score = 0.7  # elevated (money parked)
    else:
        base_score = 0.8  # extreme risk-off
    
    # Trend adjustment: falling dominance = rotation into risk = more bullish
    if dom_change_30d is not None:
        if dom_change_30d < -1:
            trend_adj = 0.3  # strong rotation into risk
            trend_label = "FALLING_FAST"
        elif dom_change_30d < -0.3:
            trend_adj = 0.15
            trend_label = "FALLING"
        elif dom_change_30d > 1:
            trend_adj = -0.2  # capital preservation mode
            trend_label = "RISING"
        elif dom_change_30d > 0.3:
            trend_adj = -0.1
            trend_label = "RISING_SLOWLY"
        else:
            trend_adj = 0
            trend_label = "STABLE"
    else:
        trend_adj = 0
        trend_label = "UNKNOWN"
    
    score = max(0.0, min(1.0, base_score + trend_adj))
    
    detail = {
        "dominance_pct": round(dom_pct, 2),
        "stablecoin_mcap": round(latest_mcap, 0),
        "total_crypto_mcap": round(total_crypto_mcap, 0),
        "dom_change_30d": round(dom_change_30d, 2) if dom_change_30d is not None else None,
        "trend": trend_label,
        "status": "ok",
    }
    
    return round(score, 3), detail


# ── Main entry point ──────────────────────────────────────────────────
def _fmt_num(val, fmt):
    """Format a number for debug output; '?' if not a real number.
    Guard against `.get(key, default)` returning None when the key is present
    but None (format on NoneType would otherwise raise)."""
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else "?"


def compute_all_stablecoin_metrics(btc_price=None, btc_mcap=None, 
                                    btc_dominance_pct=None, onchain_details=None,
                                    force_refresh=False):
    """Compute all stablecoin liquidity metrics (M76-M80).
    
    Args:
        btc_price: Current BTC price in USD
        btc_mcap: Current BTC market cap
        btc_dominance_pct: BTC dominance % (e.g., 58.3)
        onchain_details: q10_details dict from onchain_fetch (contains
                        stablecoin_exchange_netflow, stablecoin_reserve, etc.)
        force_refresh: Force re-fetch from CoinGecko
    
    Returns:
        (results_dict, details_dict, active_count, avg_score)
    """
    results = {}
    details = {}
    active = 0
    
    # Fetch stablecoin supply history
    supply_history = fetch_stablecoin_supply_history(force_refresh=force_refresh)
    
    # Derive BTC mcap from price if not provided
    if btc_mcap is None and btc_price is not None:
        btc_mcap = btc_price * 19700000  # ~19.7M circulating supply
    
    if onchain_details is None:
        onchain_details = {}
    
    # M76: Supply Growth
    try:
        s, d = calculate_m76_supply_growth(supply_history)
        if s is not None:
            results["m76_supply_growth"] = s
            details["m76_detail"] = d
            active += 1
            print(f"  ✓ M76 (SupplyGrowth): {s:.3f} — 30d={_fmt_num(d.get('growth_30d_pct'), '+.2f')}%", file=sys.stderr)
    except Exception as e:
        print(f"[SC] M76 error: {e}", file=sys.stderr)
    
    # M77: SSR
    try:
        s, d = calculate_m77_ssr(supply_history, btc_mcap)
        if s is not None:
            results["m77_ssr"] = s
            details["m77_detail"] = d
            active += 1
            print(f"  ✓ M77 (SSR): {s:.3f} — SSR={_fmt_num(d.get('ssr'), '.2f')} | {d.get('label','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[SC] M77 error: {e}", file=sys.stderr)
    
    # M78: Exchange Flow
    try:
        s, d = calculate_m78_exchange_flow(onchain_details)
        if s is not None:
            results["m78_exchange_flow"] = s
            details["m78_detail"] = d
            active += 1
            print(f"  ✓ M78 (ExchangeFlow): {s:.3f} — netflow={_fmt_num(d.get('netflow'), '+.2e')} | {d.get('label','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[SC] M78 error: {e}", file=sys.stderr)
    
    # M79: Velocity
    try:
        s, d = calculate_m79_velocity(onchain_details)
        if s is not None:
            results["m79_velocity"] = s
            details["m79_detail"] = d
            active += 1
            print(f"  ✓ M79 (Velocity): {s:.3f} — vel={_fmt_num(d.get('velocity'), '.4f')} | {d.get('label','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[SC] M79 error: {e}", file=sys.stderr)
    
    # M80: Dominance
    try:
        s, d = calculate_m80_dominance(supply_history, btc_mcap, btc_dominance_pct)
        if s is not None:
            results["m80_dominance"] = s
            details["m80_detail"] = d
            active += 1
            print(f"  ✓ M80 (Dominance): {s:.3f} — dom={_fmt_num(d.get('dominance_pct'), '.2f')}% | {d.get('trend','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[SC] M80 error: {e}", file=sys.stderr)
    
    print(f"[SC] M76-M80: {active}/5 active", file=sys.stderr)
    
    return results, details, active


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Stablecoin Liquidity Module — CLI Test")
    print("=" * 60)
    
    # Try to load onchain details from cache
    try:
        from onchain_fetch import fetch_all_onchain
        oc = fetch_all_onchain(force_refresh=False)
        onchain_details = oc.get("details", {})
        print(f"[CLI] Loaded onchain data: {oc.get('active_metrics', 0)} metrics")
    except Exception as e:
        print(f"[CLI] On-chain data unavailable: {e}")
        onchain_details = {}
    
    results, details, active = compute_all_stablecoin_metrics(
        btc_price=64000,
        btc_dominance_pct=58.3,
        onchain_details=onchain_details,
        force_refresh=False,
    )
    
    print(f"\nResults: {active}/5 active")
    print(json.dumps({"results": results, "details": details}, indent=2))
