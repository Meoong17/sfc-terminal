#!/usr/bin/env python3
"""Scrape Farside Bitcoin ETF data and merge into cache file."""

import re
import json
import time
import cloudscraper
from datetime import datetime, timezone

CACHE_PATH = '/home/ubuntu/sfc/.etf_cache.json'
URL = 'https://farside.co.uk/btc/'

# ETF columns in order: IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total
ETF_ORDER = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

MONTHS = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def parse_value(cell_html):
    """Parse a single ETF value from a table cell HTML."""
    # Negative values: <span class="redFont">(440.3)</span>
    m = re.search(r'<span class="redFont">\(([\d,]+\.?\d*)\)</span>', cell_html)
    if m:
        return -float(m.group(1).replace(',', ''))
    # Regular positive values
    m = re.search(r'<span class="tabletext">([\d,]+\.?\d*)</span>', cell_html)
    if m:
        return float(m.group(1).replace(',', ''))
    # Dash or empty — treat as 0
    return 0.0


def parse_date(date_str):
    """Convert '29 Jul 2026' to '2026-07-29'."""
    parts = date_str.split()
    day = parts[0].zfill(2)
    month = MONTHS[parts[1]]
    year = parts[2]
    return f"{year}-{month}-{day}"


def scrape_flows(html_content):
    """Parse ETF flow table rows from HTML."""
    table_match = re.search(r'<table class="etf">.*?</table>', html_content, re.DOTALL | re.IGNORECASE)
    if not table_match:
        print("ERROR: Could not find ETF table in HTML")
        return []

    tbl = table_match.group()

    # Match data rows (not Fee/Total/Average/Max/Min rows)
    row_pattern = re.compile(
        r'<tr[^>]*>.*?<td><span class="tabletext">(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})</span></td>'
        r'(.*?)</tr>',
        re.DOTALL
    )

    flows = []
    for date_str, cells_html in row_pattern.findall(tbl):
        td_cells = re.findall(r'<td[^>]*>.*?</td>', cells_html, re.DOTALL)
        if len(td_cells) < 13:
            print(f"  WARNING: {date_str} has only {len(td_cells)} cells, skipping")
            continue

        etf_values = {}
        for i, etf_name in enumerate(ETF_ORDER):
            etf_values[etf_name] = parse_value(td_cells[i])

        total_val = parse_value(td_cells[12])  # 13th cell (index 12) = Total

        flow_entry = {
            "date": parse_date(date_str),
            "total_btc": None,
            "total_usd": int(round(total_val * 1_000_000)),
            "etfs": etf_values
        }
        flows.append(flow_entry)

    return flows


def parse_cumulative_data(html_content):
    """Extract cumulative totals from the page."""
    result = {}

    # Cumulative USD from the chart dataPoints array (last value in millions)
    chart_js = re.search(r'dataPoints\s*=\s*\[(.*?)\];', html_content, re.DOTALL)
    if chart_js:
        values_str = chart_js.group(1)
        values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
        if values:
            last_val_m = values[-1]
            result['cumulative_usd'] = int(round(last_val_m * 1_000_000))
            print(f"  Cumulative USD from chart: {last_val_m}M = ${result['cumulative_usd']}")

    # Try to find cumulative BTC from the page
    # Look for patterns like "676.78K BTC" or similar
    btc_match = re.search(r'([\d,]+\.?\d*)\s*K\s*BTC', html_content, re.IGNORECASE)
    if btc_match:
        try:
            result['cumulative_btc'] = float(btc_match.group(1).replace(',', '')) * 1000
            print(f"  Cumulative BTC from text: {result['cumulative_btc']}")
        except ValueError:
            pass

    return result


def load_cache():
    """Load existing cache file."""
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Cache not found or invalid: {e}, starting fresh")
        return {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "last_update": None, "cached_at": 0}


def merge_flows(old_flows, new_flows):
    """Merge new flows into old, replacing duplicates by date."""
    date_map = {f['date']: f for f in old_flows}
    for f in new_flows:
        date_map[f['date']] = f
    # Return sorted by date ascending
    return sorted(date_map.values(), key=lambda x: x['date'])


def main():
    print("=== Farside ETF Cache Updater ===")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    
    # 1. Scrape page
    print("\n1. Fetching page...")
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code}")
        return
    html_content = resp.text
    print(f"   Page loaded: {len(html_content)} bytes")
    
    # 2. Parse flows
    print("\n2. Parsing ETF flow data...")
    new_flows = scrape_flows(html_content)
    print(f"   Found {len(new_flows)} data rows:")
    for f in new_flows:
        print(f"     {f['date']}: total_usd={f['total_usd']}")
    
    if not new_flows:
        print("ERROR: No flow data parsed, aborting!")
        return
    
    # 3. Parse cumulative data
    print("\n3. Parsing cumulative data...")
    cum = parse_cumulative_data(html_content)
    
    # 4. Load existing cache
    print("\n4. Loading existing cache...")
    cache = load_cache()
    print(f"   Existing flows: {len(cache['flows'])} rows")
    
    # 5. Merge flows
    print("\n5. Merging data...")
    cache['flows'] = merge_flows(cache['flows'], new_flows)
    print(f"   Resulting flows: {len(cache['flows'])} rows")
    
    # 6. Update cumulative values
    if 'cumulative_usd' in cum:
        cache['cumulative_usd'] = cum['cumulative_usd']
    if 'cumulative_btc' in cum:
        cache['cumulative_btc'] = cum['cumulative_btc']
    
    # 7. Update timestamps
    now = datetime.now(timezone.utc)
    cache['last_update'] = now.strftime('%Y-%m-%dT%H:%M:%S')
    cache['cached_at'] = time.time()
    
    # 8. Write cache
    print(f"\n6. Writing cache to {CACHE_PATH}...")
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)
    print("   Done!")
    
    # 9. Verify
    print("\n7. Verification...")
    with open(CACHE_PATH, 'r') as f:
        verified = json.load(f)
    print(f"   Valid JSON: YES")
    print(f"   Flows: {len(verified['flows'])} entries")
    print(f"   Latest dates: {[f['date'] for f in verified['flows'][-5:]]}")
    print(f"   Cumulative USD: {verified.get('cumulative_usd')}")
    print(f"   Cumulative BTC: {verified.get('cumulative_btc')}")
    print(f"   Last update: {verified.get('last_update')}")
    print(f"   Cached at: {verified.get('cached_at')}")
    
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
