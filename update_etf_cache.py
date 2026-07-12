#!/usr/bin/env python3
"""
Update .etf_cache.json by scraping Farside's all-data page.
Merges new data with existing, preserves all historical flows.
"""
import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

CACHE_PATH = "/home/ubuntu/sfc/.etf_cache.json"
ALL_DATA_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
MAIN_URL = "https://farside.co.uk/btc/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

ETF_NAMES = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

MONTHS = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def parse_value(val):
    """Parse a table cell value. (440.3) -> -440.3, '-' -> 0.0, '0.0' -> 0.0"""
    val = val.strip()
    if val == '-' or val == '':
        return 0.0
    negative = False
    if val.startswith('(') and val.endswith(')'):
        negative = True
        val = val[1:-1]
    val = val.replace(',', '')
    try:
        v = float(val)
        return -v if negative else v
    except ValueError:
        return 0.0


def parse_date(date_str):
    """Parse '11 Jan 2024' -> '2024-01-11'"""
    parts = date_str.strip().split()
    if len(parts) != 3:
        return None
    day, month_str, year = parts
    month = MONTHS.get(month_str)
    if month is None:
        return None
    return f"{year}-{month}-{day.zfill(2)}"


def scrape_farside():
    """Scrape all ETF flow data from the all-data page."""
    resp = requests.get(ALL_DATA_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')

    table = soup.find('table', class_='etf')
    if not table:
        raise ValueError("Could not find ETF table on all-data page")

    rows = table.find_all('tr')
    flows = []

    for row in rows:
        cells = row.find_all(['th', 'td'])
        texts = [c.get_text(strip=True) for c in cells]

        if len(texts) != 14:
            continue

        date_str = texts[0]

        # Skip non-data rows
        if date_str in ('Date', 'Fee', 'Total', 'Average', 'Maximum', 'Minimum', ''):
            continue

        parsed_date = parse_date(date_str)
        if parsed_date is None:
            continue

        # Parse ETF values
        etfs = {}
        for i, name in enumerate(ETF_NAMES):
            etfs[name] = parse_value(texts[i + 1])

        total_val = parse_value(texts[13])

        flows.append({
            'date': parsed_date,
            'total_btc': None,
            'total_usd': int(round(total_val * 1_000_000)),
            'etfs': etfs
        })

    # Sort by date
    flows.sort(key=lambda f: f['date'])

    # Calculate cumulative USD from the scraped data
    cumulative_usd = sum(f['total_usd'] for f in flows)

    return flows, cumulative_usd


def read_existing_cache():
    """Read existing cache file, return data dict or empty structure."""
    if not os.path.exists(CACHE_PATH):
        return {'flows': [], 'cumulative_btc': None, 'cumulative_usd': None, 'last_update': None}

    try:
        with open(CACHE_PATH, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {'flows': [], 'cumulative_btc': None, 'cumulative_usd': None, 'last_update': None}


def merge_flows(existing_flows, new_flows):
    """Merge new flows into existing flows, keyed by date. Newer data wins for same date."""
    # Build dict keyed by date
    flow_map = {}

    for f in existing_flows:
        flow_map[f['date']] = f

    for f in new_flows:
        flow_map[f['date']] = f

    # Return sorted by date
    merged = sorted(flow_map.values(), key=lambda x: x['date'])
    return merged


def main():
    print(f"[{datetime.now().isoformat()}] Starting ETF cache update...")

    # Read existing cache
    existing = read_existing_cache()
    existing_flows = existing.get('flows', [])
    print(f"Existing cache: {len(existing_flows)} flows")

    # Scrape new data
    print("Scraping Farside all-data page...")
    new_flows, scraped_cumulative_usd = scrape_farside()
    print(f"Scraped: {len(new_flows)} flows, cumulative USD=${scraped_cumulative_usd:,.0f}")

    # Merge
    merged_flows = merge_flows(existing_flows, new_flows)
    print(f"Merged: {len(merged_flows)} flows")

    # Calculate cumulative from merged data
    cumulative_usd = sum(f['total_usd'] for f in merged_flows)

    # Build output
    now = time.time()
    now_dt = datetime.fromtimestamp(now)
    now_iso = now_dt.strftime('%Y-%m-%dT%H:%M:%S')

    output = {
        'flows': merged_flows,
        'cumulative_btc': None,  # Farside doesn't provide BTC cumulative
        'cumulative_usd': cumulative_usd,
        'last_update': now_iso,
        'cached_at': now
    }

    # Write
    with open(CACHE_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Written to {CACHE_PATH}")
    print(f"  Flows: {len(merged_flows)}")
    print(f"  Cumulative USD: ${cumulative_usd:,.0f}")
    print(f"  Last update: {now_iso}")
    print(f"  Cached at: {now}")

    # Verify
    with open(CACHE_PATH, 'r') as f:
        verify = json.load(f)
    print(f"\nVerification: {len(verify['flows'])} flows, valid JSON ✓")
    print(f"  Date range: {verify['flows'][0]['date']} -> {verify['flows'][-1]['date']}")
    print(f"  cumulative_usd: {verify['cumulative_usd']}")
    print(f"  cached_at: {verify['cached_at']}")

    return True


if __name__ == '__main__':
    main()
