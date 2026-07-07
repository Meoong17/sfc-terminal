#!/usr/bin/env python3
"""Scrape ETF flow data from farside.co.uk and update the cache file."""

import json
import re
import time
from datetime import datetime, timezone
from collections import OrderedDict

import cloudscraper
from bs4 import BeautifulSoup

ETF_KEYS = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

CACHE_FILE = '/home/ubuntu/sfc/.etf_cache.json'


def parse_value(val_str):
    """Parse a value string like '(440.3)' -> -440.3, '57.4' -> 57.4, '0.0' -> 0.0"""
    val_str = val_str.strip()
    if val_str.startswith('(') and val_str.endswith(')'):
        return -float(val_str[1:-1].replace(',', ''))
    return float(val_str.replace(',', ''))


def parse_date(date_str):
    """Parse '18 Jun 2026' -> '2026-06-18'"""
    parts = date_str.split()
    if len(parts) == 3:
        day, month_str, year = parts
        if len(day) == 1:
            day = '0' + day
        month = MONTH_MAP.get(month_str, '01')
        return f'{year}-{month}-{day}'
    return date_str


def main():
    print("Fetching page...")
    scraper = cloudscraper.create_scraper()
    resp = scraper.get('https://farside.co.uk/btc/', timeout=30)
    resp.raise_for_status()
    print(f"Page fetched: {len(resp.text)} bytes, status={resp.status_code}")

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', class_='etf')
    if not table:
        print("ERROR: Table with class 'etf' not found!")
        return

    rows = table.find_all('tr')
    print(f"Found {len(rows)} rows")

    # Parse new data rows (skip header/fee/total/average/max/min rows)
    new_flows = []
    for row in rows:
        tds = row.find_all('td')
        if not tds:
            continue
        cols = [td.get_text(strip=True) for td in tds]
        date_raw = cols[0]

        # Skip non-date rows (Total, Average, Maximum, Minimum, Fee)
        if date_raw in ('Total', 'Average', 'Maximum', 'Minimum', 'Fee'):
            continue

        date = parse_date(date_raw)
        etfs = {}
        for i, key in enumerate(ETF_KEYS):
            if i + 1 < len(cols):
                etfs[key] = parse_value(cols[i + 1])

        total_usd = None
        if len(cols) > len(ETF_KEYS) + 1:
            total_val = parse_value(cols[len(ETF_KEYS) + 1])
            total_usd = int(total_val * 1_000_000)

        flow = {
            'date': date,
            'total_btc': None,
            'total_usd': total_usd,
            'etfs': etfs,
        }
        new_flows.append(flow)
        print(f"  Parsed: {date} total={total_usd} etfs={len(etfs)}")

    # Get cumulative from chart data dataPoints
    cumulative_usd = None
    cumulative_btc = None
    m = re.search(r'const dataPoints = \[([^\]]+)\]', resp.text)
    if m:
        values = re.findall(r'[\d.]+', m.group(1))
        if values:
            last_val = float(values[-1])
            cumulative_usd = int(last_val * 1_000_000)
            print(f"Cumulative USD from chart: ${cumulative_usd:,}")

    # Read existing cache
    existing_data = {'flows': [], 'cumulative_btc': None, 'cumulative_usd': None, 'cached_at': time.time()}
    try:
        with open(CACHE_FILE, 'r') as f:
            existing_data = json.load(f)
            print(f"Existing cache: {len(existing_data.get('flows', []))} flows")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"No existing cache or parse error: {e}")

    # Merge: combine new_flows with existing flows, dedup by date (newer wins)
    flow_map = OrderedDict()
    # First add existing flows
    for f in existing_data.get('flows', []):
        flow_map[f['date']] = f
    # Then add/update with new flows
    for f in new_flows:
        flow_map[f['date']] = f

    merged_flows = list(flow_map.values())
    # Sort by date
    merged_flows.sort(key=lambda x: x['date'])
    print(f"Merged: {len(merged_flows)} total flows")

    # Keep existing cumulative values if we couldn't find new ones
    if cumulative_usd is None:
        cumulative_usd = existing_data.get('cumulative_usd')
    if cumulative_btc is None:
        cumulative_btc = existing_data.get('cumulative_btc')

    # Build last_update timestamp
    now = datetime.now(timezone.utc)
    last_update = now.strftime('%Y-%m-%dT%H:%M:%S')

    output = {
        'flows': merged_flows,
        'cumulative_btc': cumulative_btc,
        'cumulative_usd': cumulative_usd,
        'last_update': last_update,
        'cached_at': time.time(),
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nCache written to {CACHE_FILE}")
    print(f"  Flows: {len(merged_flows)}")
    print(f"  Cumulative USD: {cumulative_usd}")
    print(f"  Cumulative BTC: {cumulative_btc}")
    print(f"  Last update: {last_update}")


if __name__ == '__main__':
    main()
