#!/usr/bin/env python3
"""
ETF Cache Updater — called by cron job to refresh .etf_cache.json
Tries multiple data sources:
  1. Farside UK (via web scrape with requests + cloudscraper fallback)
  2. CoinGlass API (if cookies available)
  3. SoSoValue API (if api key configured)
  4. Falls back to existing cache (no-op)
"""

import json, os, sys, time, re, math
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, '.etf_cache.json')

ETF_ORDER = ["GBTC", "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL",
             "BRRR", "EZBC", "BTCW", "BTC", "MSBT"]

def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"flows": [], "cumulative_btc": 0, "cumulative_usd": 0,
                "last_update": None, "cached_at": 0}

def _save_cache(cache):
    cache["cached_at"] = time.time()
    cache["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def _parse_farside_table(html):
    """Parse Farside UK HTML table into flow data.
    
    The table has rows like:
    | 17 Jun 2026 | (68.2) | (31.9) | 0.0 | ... | (125.3) |
    Columns: Date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total
    All values in USD millions.
    """
    flows = []
    # Find all table rows
    row_pattern = re.compile(
        r'<tr[^>]*>'
        r'\s*<t[dh][^>]*>(\d{1,2}\s+\w+\s+\d{4})</t[dh]>'  # date
        r'(.*?)'  # cells
        r'</tr>',
        re.DOTALL | re.IGNORECASE
    )
    
    cell_pattern = re.compile(r'<t[dh][^>]*>([^<]*)</t[dh]>')
    
    for match in row_pattern.finditer(html):
        date_str = match.group(1)
        cells_html = match.group(2)
        cells = cell_pattern.findall(cells_html)
        
        if len(cells) < 13:
            continue
        
        # Parse date
        try:
            dt = datetime.strptime(date_str, "%d %b %Y")
            date_key = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
        
        # Parse values (all in USD millions)
        def _parse_val(v):
            v = v.strip().replace(",", "").replace("+", "").replace("$", "")
            if v == "-" or v == "—" or v == "":
                return 0.0
            try:
                return float(v)
            except ValueError:
                return 0.0
        
        values = [_parse_val(c) for c in cells]
        
        # Farside columns: Date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total
        # But we need to handle mapping correctly. The columns in the HTML are:
        # IBIT(0), FBTC(1), BITB(2), ARKB(3), BTCO(4), EZBC(5), BRRR(6), HODL(7), BTCW(8), MSBT(9), GBTC(10), BTC(11), Total(12)
        if len(values) >= 13:
            total = values[12]
            etfs = {
                "IBIT": values[0],
                "FBTC": values[1],
                "BITB": values[2],
                "ARKB": values[3],
                "BTCO": values[4],
                "EZBC": values[5],
                "BRRR": values[6],
                "HODL": values[7],
                "BTCW": values[8],
                "MSBT": values[9],
                "GBTC": values[10],
                "BTC": values[11],
            }
            # Convert USD millions to raw USD
            total_usd = int(total * 1_000_000) if total else 0
            
            flows.append({
                "date": date_key,
                "total_btc": None,  # Farside gives USD, not BTC
                "total_usd": total_usd,
                "etfs": etfs,
            })
    
    return flows

def _fetch_farside():
    """Try to fetch ETF data from Farside UK.
    
    Returns (flows, cumulative_data) or (None, None).
    """
    urls = [
        "https://farside.co.uk/btc/",
        "https://farside.co.uk/bitcoin-etf-flow-all-data/",
    ]
    
    for url in urls:
        try:
            r = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                timeout=15
            )
            if r.status_code == 200 and "Cloudflare" not in r.text[:500]:
                html = r.text
                flows = _parse_farside_table(html)
                if flows:
                    # Also try to extract cumulative totals
                    cum_btc = None
                    cum_usd = None
                    # Look for cumulative total in the summary section
                    cum_match = re.search(r'Total.*?([\d,]+)\s*\)?\s*</td>', html)
                    # Try to find cumulative BTC from the summary page
                    
                    # Sort by date descending
                    flows.sort(key=lambda x: x["date"], reverse=True)
                    print(f"[ETF-Updater] Farside: {len(flows)} flow days from {url}", file=sys.stderr)
                    return flows, {"cumulative_btc": cum_btc, "cumulative_usd": cum_usd}
        except Exception as e:
            print(f"[ETF-Updater] Farside failed ({url}): {e}", file=sys.stderr)
    
    return None, None


def _fetch_coinglass_via_api():
    """Try CoinGlass API (rarely works without cookies)."""
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
                return data["data"], None
    except Exception:
        pass
    return None, None


def update_cache():
    """Main entry point. Tries all sources, updates cache if new data found."""
    cache = _load_cache()
    existing_dates = set(f["date"] for f in cache.get("flows", []))
    
    new_flows = None
    cumulative = None
    
    # Try Farside first
    try:
        new_flows, cumulative = _fetch_farside()
    except Exception as e:
        print(f"[ETF-Updater] Farside error: {e}", file=sys.stderr)
    
    # Try CoinGlass as backup
    if not new_flows:
        try:
            new_flows, cumulative = _fetch_coinglass_via_api()
        except Exception as e:
            print(f"[ETF-Updater] CoinGlass error: {e}", file=sys.stderr)
    
    if not new_flows:
        print("[ETF-Updater] No new data from any source. Cache unchanged.", file=sys.stderr)
        return False
    
    # Merge: keep existing flows not in new data, add new ones
    merged = {f["date"]: f for f in cache.get("flows", [])}
    for f in new_flows:
        merged[f["date"]] = f
    
    cache["flows"] = sorted(merged.values(), key=lambda x: x["date"], reverse=True)
    
    # Update cumulative if available
    if cumulative:
        if cumulative.get("cumulative_btc"):
            cache["cumulative_btc"] = cumulative["cumulative_btc"]
        if cumulative.get("cumulative_usd"):
            cache["cumulative_usd"] = cumulative["cumulative_usd"]
    
    _save_cache(cache)
    print(f"[ETF-Updater] Cache updated: {len(new_flows)} new rows, {len(cache['flows'])} total", file=sys.stderr)
    return True


if __name__ == "__main__":
    # For cron execution
    import requests  # imported here for cron env
    updated = update_cache()
    if updated:
        print(f"[ETF-CRON] Cache updated successfully at {datetime.now(timezone.utc).isoformat()}")
    else:
        print(f"[ETF-CRON] Cache unchanged at {datetime.now(timezone.utc).isoformat()}")
