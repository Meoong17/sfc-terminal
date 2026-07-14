#!/usr/bin/env python3
"""Scrape Farside BTC ETF data and update .etf_cache.json"""

import cloudscraper
import json
import re
import time
import os
from bs4 import BeautifulSoup
from datetime import datetime, timezone

CACHE_PATH = '/home/ubuntu/sfc/.etf_cache.json'
URL = 'https://farside.co.uk/btc/'

ETF_NAMES = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def parse_value(val_str):
    """Parse a cell value - handles parentheses for negative, dashes for no data."""
    val_str = val_str.replace(',', '').replace('$', '').strip()
    if val_str in ('', 'N/A', '-', 'n/a'):
        return 0.0
    if val_str.startswith('(') and val_str.endswith(')'):
        return -float(val_str[1:-1])
    return float(val_str)


def scrape_data():
    """Scrape ETF table and cumulative data from Farside."""
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL, timeout=30)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find the ETF table
    table = soup.find('table', class_='etf')
    if not table:
        raise RuntimeError("Could not find ETF table on page")

    rows = table.find_all('tr')

    new_flows = []
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if not cells:
            continue

        date_text = cells[0]

        # Skip non-data rows
        if date_text in ('', 'Fee', 'Average', 'Maximum', 'Minimum', 'Total'):
            continue

        # Check if first cell is a date
        date_match = re.match(r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', date_text)
        if not date_match:
            continue

        day, month_str, year = date_match.groups()
        month = MONTH_MAP[month_str]
        date_formatted = f"{year}-{month}-{int(day):02d}"

        # Parse ETF values
        etf_values = {}
        has_any_data = False
        for idx, etf_name in enumerate(ETF_NAMES):
            val_str = cells[idx + 1] if len(cells) > idx + 1 else '0'
            val = parse_value(val_str)
            etf_values[etf_name] = val
            if val_str not in ('-', '0', '0.0', '', 'N/A'):
                has_any_data = True

        # Parse total
        total_str = cells[13] if len(cells) > 13 else '0'
        total_val = parse_value(total_str)
        total_usd = int(total_val * 1_000_000)

        # If the row has all dashes (no data yet), keep total_usd = 0
        # but store the entry anyway with 0s for all ETFs
        if not has_any_data and total_val == 0:
            for etf_name in ETF_NAMES:
                etf_values[etf_name] = 0.0

        new_flows.append({
            "date": date_formatted,
            "total_btc": None,
            "total_usd": total_usd,
            "etfs": etf_values
        })

    # Extract cumulative USD from chart dataPoints
    cum_usd = None
    cum_btc = None

    for s in soup.find_all('script'):
        content = s.string or ''
        if 'dataPoints' in content:
            match = re.search(r'const\s+dataPoints\s*=\s*(\[.*?\]);', content, re.DOTALL)
            if match:
                vals = re.findall(r'([+-]?\d+\.?\d*)', match.group(1))
                if vals:
                    last = float(vals[-1])
                    cum_usd = int(last * 1_000_000)

    # Extract today's date from footer
    last_update = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    today_match = re.search(r"Today['\u2019]*s*\s*date\s*is\s*(\d{1,2})\s+(\w+)\s+(\d{4})", resp.text)
    if today_match:
        day = int(today_match.group(1))
        month_str = today_match.group(2)
        year = today_match.group(3)
        month_num = MONTH_MAP.get(month_str[:3], '01')
        # Use footer date but our current time for the timestamp
        # Actually just use current time - that's fine

    return new_flows, cum_usd, cum_btc, last_update


def merge_flows(existing_flows, new_flows):
    """Merge new flows into existing flows, deduplicating by date (new wins)."""
    flow_map = {}
    for flow in existing_flows:
        flow_map[flow['date']] = flow
    for flow in new_flows:
        flow_map[flow['date']] = flow
    
    # Sort by date
    merged = sorted(flow_map.values(), key=lambda x: x['date'])
    return merged


def main():
    # Read existing cache
    existing_data = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            existing_data = json.load(f)

    existing_flows = existing_data.get('flows', [])

    # Scrape new data
    print("Scraping Farside BTC ETF data...")
    new_flows, cum_usd, cum_btc, last_update = scrape_data()
    print(f"  Scraped {len(new_flows)} flow rows")

    # Merge flows
    merged_flows = merge_flows(existing_flows, new_flows)
    print(f"  Merged: {len(existing_flows)} existing + {len(new_flows)} new = {len(merged_flows)} total")

    # Update cumulative values (only if we got new ones from the page)
    if cum_usd is not None:
        print(f"  Cumulative USD: ${cum_usd:,}")
    else:
        cum_usd = existing_data.get('cumulative_usd')

    if cum_btc is None:
        cum_btc = existing_data.get('cumulative_btc')

    # Build new cache data
    cache_data = {
        "flows": merged_flows,
        "cumulative_btc": cum_btc,
        "cumulative_usd": cum_usd,
        "last_update": last_update,
        "cached_at": time.time()
    }

    # Write
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\nWritten to {CACHE_PATH}")
    print(f"  Last update: {last_update}")
    print(f"  Total flows: {len(merged_flows)}")

    # Verify by reading back
    with open(CACHE_PATH, 'r') as f:
        verified = json.load(f)
    print(f"  Verified: {len(verified['flows'])} flows, cumulative_usd={verified['cumulative_usd']}")


if __name__ == '__main__':
    main()
