#!/usr/bin/env python3
"""Scrape Farside Bitcoin ETF data and update the cache file."""

import cloudscraper
import json
import re
import time
from datetime import datetime, timedelta
from collections import OrderedDict

ETF_COLUMNS = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

def parse_value(s):
    """Parse a cell value like '209.4', '(44.5)', '0.0', or '0'."""
    s = s.strip()
    if not s or s == '&nbsp;':
        return 0.0
    # Remove commas
    s = s.replace(',', '')
    # Handle parentheses for negative values
    if s.startswith('(') and s.endswith(')'):
        return -float(s[1:-1])
    return float(s)

def parse_date(s):
    """Parse '06 Jul 2026' -> '2026-07-06'"""
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    day, month_str, year = parts
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    month = month_map.get(month_str[:3])
    if not month:
        return None
    return f"{year}-{month}-{int(day):02d}"

def main():
    scraper = cloudscraper.create_scraper()
    resp = scraper.get('https://farside.co.uk/btc/', timeout=30)
    html = resp.text

    # ---- Parse the main ETF table ----
    table_match = re.search(r'<table class="etf">.*?</table>', html, re.DOTALL)
    if not table_match:
        print("ERROR: Could not find ETF table")
        return

    table_html = table_match.group()

    # Find all data rows (tr elements in tbody)
    # We need rows that have a date in the first td
    # Skip rows with "Total", "Average", "Maximum", "Minimum", "Fee"
    skip_labels = {'Total', 'Average', 'Maximum', 'Minimum', 'Fee'}

    # Extract rows within tbody
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL)
    if not tbody_match:
        print("ERROR: Could not find tbody")
        return

    tbody_html = tbody_match.group(1)

    # Find each <tr>...</tr> in tbody
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tbody_html, re.DOTALL)

    new_flows = []

    for row_html in rows:
        # Get all td cells
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
        if not cells:
            continue

        # First cell should be the date/label
        label_cell = cells[0]
        label_text = re.sub(r'<[^>]+>', '', label_cell).strip()

        # Skip non-date rows
        if label_text in skip_labels:
            continue

        # Parse date
        date_str = parse_date(label_text)
        if not date_str:
            print(f"WARNING: Could not parse date from: {label_text}")
            continue

        # The table has 13 data columns (IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total)
        # cells[0] is date, cells[1..13] are the values
        if len(cells) < 14:
            print(f"WARNING: Row for {date_str} has only {len(cells)} cells, skipping")
            continue

        etfs = {}
        for i, col_name in enumerate(ETF_COLUMNS):
            cell_content = cells[i + 1]
            # Extract text, handling redFont negative values
            text = re.sub(r'<[^>]+>', '', cell_content).strip()
            etfs[col_name] = parse_value(text)

        # Total is the last cell (index 13)
        total_cell = cells[13]
        total_text = re.sub(r'<[^>]+>', '', total_cell).strip()
        total_value = parse_value(total_text)

        flow_entry = {
            "date": date_str,
            "total_btc": None,
            "total_usd": int(round(total_value * 1_000_000)),
            "etfs": etfs
        }
        new_flows.append(flow_entry)

    print(f"Parsed {len(new_flows)} date rows from Farside")

    if not new_flows:
        print("ERROR: No data rows parsed!")
        return

    # ---- Parse cumulative totals ----
    # From the chart data: last data point in dataPoints array
    cumulative_usd = None
    cumulative_btc = None

    chart_match = re.search(r'const dataPoints\s*=\s*\[([^\]]+)\]', html)
    if chart_match:
        points_str = chart_match.group(1)
        points = [float(x.strip()) for x in points_str.split(',') if x.strip()]
        if points:
            # dataPoints are in US$m (millions)
            cumulative_usd = int(round(points[-1] * 1_000_000))
            print(f"Cumulative USD from chart data: ${cumulative_usd:,}")

    # Also check the Total row in the table for confirmation
    # The Total row shows cumulative values per ETF in US$m
    total_row_match = re.search(
        r'<tr[^>]*>\s*<td><span class="tabletext">Total</span></td>'
        r'(.*?)</tr>',
        table_html, re.DOTALL
    )
    if total_row_match:
        total_cells = re.findall(r'<td[^>]*>(.*?)</td>', total_row_match.group(1), re.DOTALL)
        if len(total_cells) >= 13:
            last_val_text = re.sub(r'<[^>]+>', '', total_cells[-1]).strip()
            last_val = parse_value(last_val_text)
            total_from_table = int(round(last_val * 1_000_000))
            print(f"Cumulative USD from Total row: ${total_from_table:,}")
            if cumulative_usd is None:
                cumulative_usd = total_from_table

    # ---- Load existing cache ----
    try:
        with open('/home/ubuntu/sfc/.etf_cache.json', 'r') as f:
            existing = json.load(f)
        existing_flows = existing.get('flows', [])
        print(f"Loaded {len(existing_flows)} existing flow entries")
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
        existing_flows = []
        print("No existing cache found, starting fresh")

    # ---- Merge: keep existing flows, add/update with new flows ----
    flow_map = OrderedDict()
    for f in existing_flows:
        flow_map[f['date']] = f

    for f in new_flows:
        flow_map[f['date']] = f

    merged_flows = list(flow_map.values())
    # Sort by date ascending
    merged_flows.sort(key=lambda x: x['date'])

    print(f"Merged to {len(merged_flows)} total flow entries")

    # ---- Build output ----
    # Keep existing cumulative if we couldn't find new ones
    if cumulative_usd is None:
        cumulative_usd = existing.get('cumulative_usd')
    if cumulative_btc is None:
        cumulative_btc = existing.get('cumulative_btc')

    output = {
        "flows": merged_flows,
        "cumulative_btc": cumulative_btc,
        "cumulative_usd": cumulative_usd,
        "last_update": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
        "cached_at": time.time()
    }

    # ---- Write ----
    with open('/home/ubuntu/sfc/.etf_cache.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Written to cache: {len(merged_flows)} flows, cumulative_usd={cumulative_usd}")

if __name__ == '__main__':
    main()
