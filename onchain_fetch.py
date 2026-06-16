#!/usr/bin/env python3
"""
On-chain data fetcher for SFC Terminal.
Downloads institutional-grade on-chain metrics from
ErcinDedeoglu/crypto-market-data (GitHub) — free data worth $500-2000+/mo.

Maps to SFC factors:
  - Whale Pressure    → Rt  (risk)
  - On-chain Value    → Lt  (long-term)
  - Buying Power      → Ft  (funding/liquidity)
  - Market Structure  → St  (short-term / derivatives risk)

Cached locally to avoid redundant downloads (24h TTL).
"""

import json, os, time, math, sys
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    requests = None

ONCHAIN_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.onchain_cache.json')
RAW_BASE = "https://raw.githubusercontent.com/ErcinDedeoglu/crypto-market-data/main/data/daily"

CACHE_TTL = 86400  # 24 hours

# ── Metric definitions ──────────────────────────────────────────────────
# Each entry: file name, SFC factor, signal direction, percentile expected range

METRICS = {
    # ── Existing 9 metrics ──
    "exchange_netflow": {
        "file": "btc_exchange_netflow.json",
        "factor": "Rt",
        "desc": "BTC Exchange Netflow (negative = supply leaving exchanges)",
        "direction": "neg",
        "expected_min": -3000,
        "expected_max": 3000,
    },
    "whale_ratio": {
        "file": "btc_exchange_whale_ratio.json",
        "factor": "Rt",
        "desc": "Whale Ratio (top10 inflows / total inflows)",
        "direction": "pos",
        "expected_min": 0.3,
        "expected_max": 0.8,
    },
    "exchange_supply_ratio": {
        "file": "btc_exchange_supply_ratio.json",
        "factor": "Rt",
        "desc": "BTC Exchange Supply Ratio (% of supply on exchanges)",
        "direction": "neg",
        "expected_min": 0.12,
        "expected_max": 0.15,
    },
    "funding_rates": {
        "file": "btc_funding_rates.json",
        "factor": "Rt",
        "desc": "Funding Rates (>0.05% = overheated, <-0.05% = capitulation)",
        "direction": "neutral",
        "expected_min": -0.0008,
        "expected_max": 0.0008,
    },
    "mvrv_ratio": {
        "file": "btc_mvrv_ratio.json",
        "factor": "Lt",
        "desc": "MVRV Ratio (market value / realized value)",
        "direction": "neg",
        "expected_min": 0.8,
        "expected_max": 4.0,
    },
    "puell_multiple": {
        "file": "btc_puell_multiple.json",
        "factor": "Lt",
        "desc": "Puell Multiple (miner revenue / 365d MA)",
        "direction": "neg",
        "expected_min": 0.2,
        "expected_max": 6.0,
    },
    "miners_position": {
        "file": "btc_miners_position_index.json",
        "factor": "Lt",
        "desc": "Miners Position Index (<0.5 = miner confidence)",
        "direction": "neg",
        "expected_min": -2.0,
        "expected_max": 3.0,
    },
    "stablecoin_reserve": {
        "file": "stablecoin_exchange_reserve.json",
        "factor": "Ft",
        "desc": "Stablecoin Exchange Reserve (buying power ready)",
        "direction": "pos",
        "expected_min": 6e10,
        "expected_max": 7e10,
    },
    "taker_buy_sell": {
        "file": "btc_taker_buy_sell_ratio.json",
        "factor": "Ft",
        "desc": "Taker Buy/Sell Ratio (>1 = aggressive buying)",
        "direction": "pos",
        "expected_min": 0.7,
        "expected_max": 1.3,
    },

    # ── NEW: Supply / Demand ──
    "exchange_inflow_total": {
        "file": "btc_exchange_inflow_total.json",
        "factor": "Rt",
        "desc": "BTC Exchange Inflow Total (sell pressure)",
        "direction": "pos",
        "expected_min": 0,
        "expected_max": 50000,
    },
    "exchange_outflow_total": {
        "file": "btc_exchange_outflow_total.json",
        "factor": "Rt",
        "desc": "BTC Exchange Outflow Total (accumulation)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 50000,
    },
    "coinbase_premium_gap": {
        "file": "btc_coinbase_premium_gap.json",
        "factor": "Ft",
        "desc": "Coinbase Premium Gap (US institutional demand)",
        "direction": "pos",
        "expected_min": -100,
        "expected_max": 100,
    },
    "korea_premium_index": {
        "file": "btc_korea_premium_index.json",
        "factor": "Sc",
        "desc": "Korea Premium Index (retail euphoria/panic)",
        "direction": "neutral",
        "expected_min": -5,
        "expected_max": 10,
    },

    # ── NEW: Miner ──
    "miner_netflow_total": {
        "file": "btc_miner_netflow_total.json",
        "factor": "Lt",
        "desc": "Miner Netflow Total (miner distribution pressure)",
        "direction": "neg",
        "expected_min": -500,
        "expected_max": 500,
    },

    # ── NEW: Stablecoin Flows ──
    "stablecoin_exchange_inflow": {
        "file": "stablecoin_exchange_inflow_total.json",
        "factor": "Ft",
        "desc": "Stablecoin Exchange Inflow Total (ready to buy)",
        "direction": "pos",
        "expected_min": 0,
        "expected_max": 5e10,
    },
    "stablecoin_exchange_outflow": {
        "file": "stablecoin_exchange_outflow_total.json",
        "factor": "Ft",
        "desc": "Stablecoin Exchange Outflow Total (leaving exchanges)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 5e10,
    },
    "stablecoin_exchange_netflow": {
        "file": "stablecoin_exchange_netflow.json",
        "factor": "Ft",
        "desc": "Stablecoin Exchange Netflow",
        "direction": "pos",
        "expected_min": -5e10,
        "expected_max": 5e10,
    },
    "exchange_stablecoins_ratio": {
        "file": "btc_exchange_stablecoins_ratio.json",
        "factor": "Ft",
        "desc": "BTC Exchange Stablecoins Ratio (BTC/Stablecoin supply)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 5,
    },

    # ── NEW: Market Structure (Derivatives) ──
    "open_interest": {
        "file": "btc_open_interest.json",
        "factor": "St",
        "desc": "BTC Futures Open Interest (leverage buildup)",
        "direction": "neutral",
        "expected_min": 5e9,
        "expected_max": 3e10,
    },
    "long_liquidations_usd": {
        "file": "btc_long_liquidations_usd.json",
        "factor": "St",
        "desc": "Long Liquidations in USD (cascade risk)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 5e8,
    },
    "short_liquidations_usd": {
        "file": "btc_short_liquidations_usd.json",
        "factor": "St",
        "desc": "Short Liquidations in USD (squeeze risk)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 5e8,
    },
    "fund_flow_ratio": {
        "file": "btc_fund_flow_ratio.json",
        "factor": "St",
        "desc": "Fund Flow Ratio (BTC flowing to/from exchanges)",
        "direction": "neg",
        "expected_min": 0,
        "expected_max": 2,
    },
}


def _load_cache():
    """Load cached on-chain data."""
    if os.path.exists(ONCHAIN_CACHE_FILE):
        try:
            with open(ONCHAIN_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"cached_at": 0, "raw": {}, "scores": {}}


def _save_cache(cache):
    """Save on-chain cache."""
    cache["cached_at"] = time.time()
    with open(ONCHAIN_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _fetch_single(url):
    """Fetch a single JSON file from raw GitHub."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[OnChain] Fetch failed: {url} — {e}", file=sys.stderr)
        return None


def _compute_percentile(value, data_vals, direction="pos"):
    """Score 0-100 based on where value falls in historical distribution."""
    if not data_vals or len(data_vals) < 10:
        return 50  # fallback neutral

    sorted_vals = sorted(data_vals)
    n = len(sorted_vals)

    # Count values below current
    below = sum(1 for v in sorted_vals if v < value)
    percentile = below / n  # 0.0 - 1.0

    if direction == "pos":
        score = percentile * 100
    elif direction == "neg":
        score = (1 - percentile) * 100
    else:  # neutral — distance from median
        median_pct = 0.5
        dist = abs(percentile - median_pct) * 2  # 0.0 at median, 1.0 at edge
        score = (1 - dist) * 100

    return max(0, min(100, score))


def compute_onchain_scores(raw_data=None):
    """Compute 0-100 scores from raw on-chain data.

    Returns {
        "whale_pressure": 0-100,     # → factor Rt
        "onchain_value": 0-100,      # → factor Lt
        "buying_power": 0-100,       # → factor Ft
        "market_structure": 0-100,   # → factor St (NEW)
        "details": {...}
    }
    """
    if raw_data is None:
        cache = _load_cache()
        raw_data = cache.get("raw", {})

    # Group metrics by factor
    factor_groups = {
        "whale_pressure": [
            "exchange_netflow", "whale_ratio", "exchange_supply_ratio",
            "funding_rates", "exchange_inflow_total", "exchange_outflow_total",
        ],
        "onchain_value": [
            "mvrv_ratio", "puell_multiple", "miners_position",
            "miner_netflow_total",
        ],
        "buying_power": [
            "stablecoin_reserve", "taker_buy_sell",
            "stablecoin_exchange_inflow", "stablecoin_exchange_outflow",
            "stablecoin_exchange_netflow", "exchange_stablecoins_ratio",
            "coinbase_premium_gap",
        ],
        "market_structure": [
            "open_interest", "long_liquidations_usd",
            "short_liquidations_usd", "fund_flow_ratio",
            "korea_premium_index",
        ],
    }

    # Weights for each factor group (must match metric order above)
    weights_map = {
        "whale_pressure": [0.20, 0.20, 0.15, 0.10, 0.20, 0.15],
        "onchain_value": [0.30, 0.25, 0.25, 0.20],
        "buying_power": [0.15, 0.10, 0.15, 0.15, 0.15, 0.15, 0.15],
        "market_structure": [0.25, 0.25, 0.20, 0.15, 0.15],
    }

    result = {
        "whale_pressure": 50.0,
        "onchain_value": 50.0,
        "buying_power": 50.0,
        "market_structure": 50.0,
        "details": {},
        "active_metrics": 0,
    }

    # Compute individual metric scores
    metric_scores = {}
    for name, meta in METRICS.items():
        entry = raw_data.get(name)
        if not entry or "data" not in entry:
            metric_scores[name] = 50.0
            result["details"][name] = {"score": 50.0, "value": None, "status": "unavailable"}
            continue

        data = entry["data"]
        if not data:
            metric_scores[name] = 50.0
            result["details"][name] = {"score": 50.0, "value": None, "status": "no_data"}
            continue

        # Get latest value
        latest = sorted(data, key=lambda x: x["timestamp"], reverse=True)[0]
        value = latest["value"]

        # Use last 365 values as historical distribution for percentile scoring
        hist_vals = [x["value"] for x in data[-365:] if x.get("value") is not None]

        if len(hist_vals) >= 10:
            score = _compute_percentile(value, hist_vals, meta["direction"])
        else:
            score = 50.0

        metric_scores[name] = score
        result["details"][name] = {
            "score": round(score, 1),
            "value": value,
            "date": datetime.fromtimestamp(latest["timestamp"]/1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "status": "ok",
        }

    # Compute composite factor scores (weighted averages)
    for group_name, metric_names in factor_groups.items():
        scores = [metric_scores.get(n, 50.0) for n in metric_names]
        w = weights_map.get(group_name)
        if w and len(scores) == len(w):
            weighted = sum(s * w[i] for i, s in enumerate(scores)) / sum(w)
        else:
            weighted = sum(scores) / len(scores) if scores else 50.0

        result[group_name] = round(weighted, 1)
        result["active_metrics"] += sum(1 for s in scores if s != 50.0)

    # Store composite scores in details for frontend access
    result["details"]["whale_pressure_score"] = result["whale_pressure"]
    result["details"]["onchain_value_score"] = result["onchain_value"]
    result["details"]["buying_power_score"] = result["buying_power"]
    result["details"]["market_structure_score"] = result["market_structure"]

    return result


def fetch_all_onchain(force_refresh=False):
    """Fetch all on-chain metrics from GitHub, cache locally.

    Returns {
        "whale_pressure": 0-100,
        "onchain_value": 0-100,
        "buying_power": 0-100,
        "market_structure": 0-100,  # NEW
        "details": {...}
    }
    """
    if requests is None:
        print("[OnChain] requests library not available", file=sys.stderr)
        return compute_onchain_scores({})

    cache = _load_cache()
    now = time.time()

    # Use cache if fresh
    if not force_refresh and (now - cache.get("cached_at", 0)) < CACHE_TTL:
        cached_raw = cache.get("raw", {})
        if cached_raw and len(cached_raw) >= 3:
            print(f"[OnChain] Using cached data ({now - cache.get('cached_at', 0):.0f}s old)", file=sys.stderr)
            return compute_onchain_scores(cached_raw)

    # Fetch fresh data
    print("[OnChain] Fetching fresh on-chain data from GitHub...", file=sys.stderr)
    raw = {}
    success_count = 0

    for name, meta in METRICS.items():
        url = f"{RAW_BASE}/{meta['file']}"
        data = _fetch_single(url)
        if data:
            raw[name] = data
            success_count += 1
        else:
            # Fall back to cache for this metric
            if name in cache.get("raw", {}):
                raw[name] = cache["raw"][name]
                print(f"[OnChain] Using cached {name}", file=sys.stderr)

    print(f"[OnChain] Fetched {success_count}/{len(METRICS)} metrics", file=sys.stderr)

    # Cache and compute
    cache["raw"] = raw
    _save_cache(cache)

    return compute_onchain_scores(raw)


if __name__ == "__main__":
    """CLI: test fetch and print scores."""
    result = fetch_all_onchain(force_refresh=True)
    print(json.dumps(result, indent=2))
