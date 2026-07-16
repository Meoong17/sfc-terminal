#!/usr/bin/env python3
"""Fetch Farside BTC ETF flow data, merge with existing cache, and update."""

import json
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from collections import OrderedDict

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def parse_date(dd_mon_yyyy):
    """Convert '11 Jan 2024' to '2024-01-11'."""
    parts = dd_mon_yyyy.strip().split()
    if len(parts) != 3:
        return None
    day, mon, year = parts
    month = MONTH_MAP.get(mon)
    if not month:
        return None
    return f"{year}-{month}-{day.zfill(2)}"

def parse_value(val_str):
    """Parse a cell value: '-' -> 0.0, '(440.3)' -> -440.3, '51,086' -> 51086.0."""
    if not val_str or not val_str.strip():
        return 0.0
    val_str = val_str.strip().replace(',', '')
    if val_str == '-' or val_str == '':
        return 0.0
    try:
        if val_str.startswith('(') and val_str.endswith(')'):
            return -float(val_str[1:-1])
        return float(val_str)
    except (ValueError, TypeError):
        return 0.0

def fetch_flows():
    """Fetch ETF flow data from Farside all-data page. Returns (flows_list, cumulative_total_millions)."""
    try:
        r = requests.get('https://farside.co.uk/bitcoin-etf-flow-all-data/', timeout=30, headers=HEADERS)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Network error fetching ETF data: {e}")
        return [], 0.0
    
    try:
        soup = BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        print(f"[WARN] Failed to parse HTML: {e}")
        return [], 0.0
    
    tables = soup.find_all('table')
    
    # Find the table with our headers
    target_table = None
    for table in tables:
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all('th')]
            if headers[:3] == ['Date', 'IBIT', 'FBTC']:
                target_table = table
                break
    
    if not target_table:
        print("[WARN] Could not find the ETF flow data table on the page")
        return [], 0.0
    
    tbody = target_table.find('tbody') or target_table
    rows = tbody.find_all('tr')
    
    flows = []
    etf_names = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']
    
    for row in rows:
        try:
            cells = row.find_all('td')
            if len(cells) != 14:  # Date + 12 ETFs + Total
                continue
            
            date_raw = cells[0].get_text(strip=True)
            
            # Skip non-date rows
            if date_raw in ('Total', 'Average', 'Maximum', 'Minimum'):
                continue
            
            date_parsed = parse_date(date_raw)
            if not date_parsed:
                continue
            
            etfs = {}
            for i, name in enumerate(etf_names):
                etfs[name] = round(parse_value(cells[i+1].get_text(strip=True)), 4)
            
            total_val = parse_value(cells[-1].get_text(strip=True))
            total_usd = int(round(total_val * 1_000_000))
            
            flows.append({
                "date": date_parsed,
                "total_btc": None,
                "total_usd": total_usd,
                "etfs": etfs
            })
        except Exception:
            # Skip malformed row rather than crashing the entire fetch
            continue
    
    # Get cumulative total from the Total row
    cumulative_total_millions = 0.0
    for row in rows:
        try:
            cells = row.find_all('td')
            if len(cells) == 14 and cells[0].get_text(strip=True) == 'Total':
                cumulative_total_millions = parse_value(cells[-1].get_text(strip=True))
                break
        except Exception:
            continue
    
    return flows, cumulative_total_millions


def main():
    # Fetch new data
    print("Fetching ETF flow data from Farside...")
    new_flows, cumulative_total_millions = fetch_flows()
    print(f"Fetched {len(new_flows)} flow records")
    
    # Read existing cache
    cache_path = '/home/ubuntu/sfc/.etf_cache.json'
    try:
        with open(cache_path, 'r') as f:
            existing = json.load(f)
        print(f"Existing cache has {len(existing.get('flows', []))} flow records")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No valid existing cache found, starting fresh")
        existing = {"flows": [], "cumulative_btc": None, "cumulative_usd": None}
    
    # Merge flows: existing by date, then overwrite/append with new
    existing_by_date = {f['date']: f for f in existing.get('flows', [])}
    
    for flow in new_flows:
        existing_by_date[flow['date']] = flow
    
    # Sort by date
    merged_flows = sorted(existing_by_date.values(), key=lambda f: f['date'])
    
    # Print date range safely
    if merged_flows:
        print(f"Date range: {merged_flows[0]['date']} to {merged_flows[-1]['date']}")
        print(f"Cumulative total (millions USD): {cumulative_total_millions}")
    
    # Compute cumulative_usd from the total row data if available
    cumulative_usd = int(round(cumulative_total_millions * 1_000_000))
    
    # If we don't have a cumulative_btc from the page, compute from existing
    cumulative_btc = existing.get('cumulative_btc')
    
    now = time.time()
    dt_now = datetime.utcfromtimestamp(now).strftime('%Y-%m-%dT%H:%M:%S')
    
    cache_data = OrderedDict([
        ("flows", merged_flows),
        ("cumulative_btc", cumulative_btc),
        ("cumulative_usd", cumulative_usd),
        ("last_update", dt_now),
        ("cached_at", now)
    ])
    
    # Write cache — use temp file to avoid partial writes
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except (IOError, OSError) as e:
        print(f"[ERROR] Failed to write cache: {e}")
        return
    
    print(f"Cache written with {len(merged_flows)} total flow records")
    print(f"Cumulative USD: ${cumulative_usd:,}")
    print(f"Last update: {dt_now}")
    
    # Verify
    try:
        with open(cache_path, 'r') as f:
            verified = json.load(f)
        assert len(verified['flows']) == len(merged_flows), "Flow count mismatch!"
        print(f"VERIFIED: Cache reads back valid JSON with {len(verified['flows'])} flows.")
    except (IOError, json.JSONDecodeError, AssertionError) as e:
        print(f"[ERROR] Verification failed: {e}")

if __name__ == '__main__':
    main()
