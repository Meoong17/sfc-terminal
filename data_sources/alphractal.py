#!/usr/bin/env python3
"""
SFC Alphractal data source — on-chain & derivatives metrics.

Fetches institutional-grade Bitcoin on-chain / derivatives metrics from the
Alphractal API (https://docs.alphractal.com/api-doc). Auth is the `X-Api-Key`
header; read from env var ALPHRACTAL_API_KEY (see .env / .env.example).

IMPORTANT — plan/tier limits (verified 2026-08-15 with the ak-... key):
  * Only the endpoints listed in WORKING_METRICS below are fetched. Many premium
    on-chain endpoints (MVRV z-score, realized price, SOPR, exchange netflow,
    Puell multiple, active supply, smart-money flow, vANV, ADCI, spot/risk, ...)
    return **403 Forbidden** on this plan. We DO NOT attempt them — a 403 just
    wastes a rate-limit slot.
  * Rate limit is bursty: rapid calls → 429 "Rate limit exceeded", resets after
    ~20-25s. fetch_all() paces requests and backs off on 429.
  * Data is historical DAILY series (startDate/endDate are optional; default =
    full history). Some series are long (PriceUSD back to 2009-10) and are
    paginated/fetched in one call.

OUTPUT (alphractal_daily.json): a date-keyed dict { "YYYY-MM-DD": { metric: val } }
matching the convention of data/binance_vision_daily.json.

SFC status: DATA COLLECTION ONLY. This module does NOT blend into any SFC score.
Per SFC rules, no metric may enter scoring before it is walk-forward validated
(see analysis/walk_forward_validation.py). Use this for research/tape first.

Usage (as library):
    from data_sources.alphractal import fetch_all, load_daily
    data = fetch_all()          # returns {date: {metric: val}, ...} and caches
    daily = load_daily()        # read cached JSON

Usage (CLI):
    python3 -m data_sources.alphractal            # fetch all, save cache
    python3 -m data_sources.alphractal --json     # fetch, print as JSON
"""

import os, sys, json, time, urllib.request, urllib.error
from datetime import datetime, timezone

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_BASE = "https://api.alphractal.com"
_CACHE_FILE = os.path.join(SFC_DIR, "data", "alphractal_daily.json")
CACHE_TTL = 21600  # 6 hours
ASSET = "btc"

# Per-endpoint request pacing (seconds between calls) — keeps us under the burst
# rate limit while still clearing the full set in a reasonable time.
_REQUEST_GAP = 1.5
_RETRY_BACKOFF = 25.0  # sleep after a 429 before retrying
_MAX_RETRIES = 3

# ── Verified-working metrics on this plan (HTTP 200) ─────────────────────
# key            endpoint suffix                         field name in payload
# Derived from a live probe on 2026-08-15. Each entry was confirmed 200 and
# its payload field names were inspected. Anything else (403) is deliberately
# NOT included — see module docstring.
WORKING_METRICS = [
    # -- market / price --
    ("PriceUSD",              "market/PriceUSD",                "PriceUSD"),
    ("SplyCur",               "supply/SplyCur",                 "SplyCur"),
    ("supply_in_profit_pct",  "supply/supply_in_profit_pct",    "supply_in_profit_pct"),
    # -- network activity --
    ("TxCnt",                 "transactions/TxCnt",             "TxCnt"),
    ("AdrActCnt",             "addresses/AdrActCnt",            "AdrActCnt"),
    # -- mining --
    ("HashRate",              "mining/HashRate",                "HashRate"),
    ("DiffLast",              "mining/DiffLast",                "DiffLast"),
    # -- derivatives (shorter history) --
    ("Funding_Rate",          "derivatives/Funding_Rate",       "value"),
    ("Open_Interest",         "derivatives/Open_Interest",      "open_Interest"),
    ("Long_short_ratio",      "derivatives/Long_short_ratio",   "long_short_ratio"),
    ("Taker_long_short",      "derivatives/Taker_long_short_vol_ratio",
                                                               "taker_long_short_vol_ratio"),
    ("Liquidations",          "derivatives/Liquidations",       "total_liquidations_usd"),
]

# Endpoints verified 403 on this plan (kept for documentation / future plan
# upgrade). NOT fetched — they just burn rate-limit slots.
BLOCKED_ON_PLAN = [
    "market/Mvrv_zscore", "market/Nupl", "market/vanv_signal", "market/CapMrktCurUSD",
    "lifespan/Realized_price", "lifespan/Sopr", "lifespan/Reserve_risk",
    "lifespan/Active_supply", "holder_supply/SplyAct30d", "holder_supply/lth_momentum",
    "mining/Puellmultiple", "transactions/VelCur1yr", "addresses_supply/smart_money_flow",
    "exchange_flow/Exchange_netflow", "exchange_flow/Exchange_netflow_usd",
    "derivatives/Whale_vs_retail_delta", "cycle/adci", "spot/risk",
]


def _api_key():
    key = os.getenv("ALPHRACTAL_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ALPHRACTAL_API_KEY not set. Add it to /home/ubuntu/sfc/.env "
            "(see .env.example)."
        )
    return key


def _request_json(url, retries=_MAX_RETRIES):
    """GET url with X-Api-Key, return parsed JSON, backing off on 429."""
    key = _api_key()
    req = urllib.request.Request(url, headers={
        "X-Api-Key": key, "User-Agent": "SFC/1.0 (+SFC Terminal)",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(_RETRY_BACKOFF)
                continue
            # 403/404/5xx: not a rate-limit issue — surface it.
            raise RuntimeError(f"Alphractal {e.code} on {url}: {e.read().decode()[:200]}")
    raise RuntimeError(f"Alphractal rate-limited after {retries} retries: {url}")


def fetch_metric(metric_key):
    """Fetch one metric's full daily series as list of {time, value-ish} dicts."""
    suffix = dict((m[0], m[1]) for m in WORKING_METRICS)[metric_key]
    url = f"{API_BASE}/{ASSET}/{suffix}"
    rows = _request_json(url)
    return rows


def fetch_all(verbose=False):
    """
    Fetch every working metric and merge into a date-keyed dict:
        { "YYYY-MM-DD": { "PriceUSD": 123.4, "TxCnt": 5, ... } }
    Saves to data/alphractal_daily.json and returns it.
    """
    merged = {}
    for key, suffix, field in WORKING_METRICS:
        try:
            rows = fetch_metric(key)
        except RuntimeError as e:
            if verbose:
                print(f"  !! {key}: {e}", file=sys.stderr)
            continue
        for row in rows:
            t = row.get("time")
            if not t:
                continue
            day = t[:10]  # "YYYY-MM-DD" from ISO timestamp
            merged.setdefault(day, {})[key] = row.get(field)
        if verbose:
            print(f"  fetched {key}: {len(rows)} rows", file=sys.stderr)
        time.sleep(_REQUEST_GAP)

    # sort by date for stable output
    merged = dict(sorted(merged.items()))
    _save(merged)
    return merged


def _save(data):
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(data, f)


def load_daily(max_age_s=CACHE_TTL):
    """Read cached alphractal_daily.json if fresh, else refetch."""
    if os.path.exists(_CACHE_FILE):
        age = time.time() - os.path.getmtime(_CACHE_FILE)
        if age < max_age_s:
            with open(_CACHE_FILE) as f:
                return json.load(f)
    return fetch_all()


def last_update():
    if os.path.exists(_CACHE_FILE):
        ts = datetime.fromtimestamp(os.path.getmtime(_CACHE_FILE), tz=timezone.utc)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Alphractal BTC daily series into data/alphractal_daily.json")
    ap.add_argument("--json", action="store_true", help="print fetched data as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    data = fetch_all(verbose=args.verbose)
    n_days = len(data)
    print(f"alphractal_daily.json: {n_days} days, {len(WORKING_METRICS)} metrics "
          f"(last updated {last_update()})")
    if args.json:
        print(json.dumps(data))


if __name__ == "__main__":
    main()
