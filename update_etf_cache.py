#!/usr/bin/env python3
"""
Farside ETF data scraper & cache updater.
Fetches Bitcoin ETF flow data from https://farside.co.uk/btc/,
parses the main table, and merges into .etf_cache.json
"""

import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.etf_cache.json')
URL = "https://farside.co.uk/btc/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Column mapping: index -> ticker
# Cell 0 = Date, Cells 1-12 = ETFs, Cell 13 = Total
ETF_COLUMNS = {
    1: "IBIT",   # Blackrock
    2: "FBTC",   # Fidelity
    3: "BITB",   # Bitwise
    4: "ARKB",   # Ark
    5: "BTCO",   # Invesco
    6: "EZBC",   # Franklin
    7: "BRRR",   # Valkyrie
    8: "HODL",   # VanEck
    9: "BTCW",   # WTree
    10: "MSBT",  # MS
    11: "GBTC",  # Grayscale
    12: "BTC",   # Grayscale Mini
}

ALL_TICKERS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]


def parse_value(text):
    """Parse a cell value. Parentheses = negative. Commas removed."""
    text = text.strip()
    if not text or text == '-':
        return 0.0
    negative = False
    if text.startswith('(') and text.endswith(')'):
        negative = True
        text = text[1:-1]
    # Remove commas
    text = text.replace(',', '')
    try:
        val = float(text)
    except ValueError:
        return 0.0
    if negative:
        val = -val
    return val


def parse_date(text):
    """Parse '29 Jun 2026' -> '2026-06-29'"""
    text = text.strip()
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    parts = text.split()
    if len(parts) != 3:
        return None
    day, mon_str, year = parts
    mon = months.get(mon_str[:3])
    if not mon:
        return None
    return f"{year}-{mon}-{int(day):02d}"


def scrape_farside():
    """Scrape Farside page and return flows list and cumulative values."""
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'lxml')

    table = soup.find('table', class_='etf')
    if not table:
        raise Exception("Could not find ETF table on page")

    tbody = table.find('tbody')
    if not tbody:
        raise Exception("Could not find tbody in ETF table")

    rows = tbody.find_all('tr')
    
    flows = []
    
    for tr in rows:
        cells = tr.find_all('td')
        if len(cells) < 14:
            continue
        
        date_text = cells[0].get_text(strip=True)
        # Skip non-date rows (Total, Average, Maximum, Minimum, Fee)
        if not re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', date_text):
            continue
        
        date = parse_date(date_text)
        if not date:
            continue
        
        # Parse ETF values
        etfs = {}
        for ci in range(1, 13):
            ticker = ETF_COLUMNS[ci]
            val = parse_value(cells[ci].get_text(strip=True))
            etfs[ticker] = val
        
        # Parse Total (cell 13)
        total_val = parse_value(cells[13].get_text(strip=True))
        total_usd = int(round(total_val * 1_000_000))
        
        flow_entry = {
            "date": date,
            "total_btc": None,
            "total_usd": total_usd,
            "etfs": etfs
        }
        flows.append(flow_entry)
    
    # Get cumulative totals from the Total row in the table
    cumulative_usd = None
    cumulative_btc = None
    
    for tr in rows:
        cells = tr.find_all('td')
        if len(cells) < 14:
            continue
        first_cell = cells[0].get_text(strip=True)
        if first_cell == 'Total':
            total_val = parse_value(cells[13].get_text(strip=True))
            cumulative_usd = int(round(total_val * 1_000_000))
            break
    
    # Try to find cumulative BTC from page text
    body_text = soup.get_text()
    btc_match = re.search(r'([\+\-]?\s*\d+[\.\,]?\d*)\s*K?\s*BTC', body_text, re.IGNORECASE)
    if btc_match:
        try:
            val_str = btc_match.group(1).strip().replace(',', '')
            val = float(val_str)
            # Check if there's a 'K' after the number
            context_end = btc_match.end()
            if context_end < len(body_text) and body_text[context_end:context_end+1] == 'K':
                val *= 1000
            cumulative_btc = val
        except ValueError:
            pass

    return flows, cumulative_usd, cumulative_btc


def merge_flows(existing_flows, new_flows):
    """Merge new flows into existing flows. New data overwrites old on same date."""
    # Build lookup by date
    flow_by_date = {f["date"]: f for f in existing_flows}
    
    for f in new_flows:
        flow_by_date[f["date"]] = f  # overwrite with new data
    
    # Sort by date
    merged = sorted(flow_by_date.values(), key=lambda x: x["date"])
    return merged


def main():
    print("Scraping Farside page...")
    try:
        new_flows, cumulative_usd, cumulative_btc = scrape_farside()
        print(f"  Scraped {len(new_flows)} daily flow entries")
        if new_flows:
            print(f"  Date range: {new_flows[0]['date']} to {new_flows[-1]['date']}")
        print(f"  Cumulative USD: {cumulative_usd}")
        print(f"  Cumulative BTC: {cumulative_btc}")
    except Exception as e:
        print(f"  ERROR scraping: {e}")
        return

    # Read existing cache
    existing_data = {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "cached_at": 0}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r') as f:
                existing_data = json.load(f)
            print(f"  Read existing cache: {len(existing_data.get('flows', []))} entries")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read existing cache: {e}")
            existing_data = {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "cached_at": 0}

    # Merge flows
    merged_flows = merge_flows(existing_data.get("flows", []), new_flows)
    print(f"  After merge: {len(merged_flows)} total entries")

    # Build new cache
    new_cache = {
        "flows": merged_flows,
        "cumulative_btc": cumulative_btc if cumulative_btc is not None else existing_data.get("cumulative_btc"),
        "cumulative_usd": cumulative_usd if cumulative_usd is not None else existing_data.get("cumulative_usd"),
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cached_at": time.time()
    }

    # Write
    with open(CACHE_PATH, 'w') as f:
        json.dump(new_cache, f, indent=2)
    print(f"  Written to {CACHE_PATH}")

    # Verify
    with open(CACHE_PATH, 'r') as f:
        verified = json.load(f)
    print(f"  Verified: {len(verified['flows'])} flows, "
          f"cumulative_usd={verified['cumulative_usd']}, "
          f"cumulative_btc={verified['cumulative_btc']}, "
          f"cached_at={verified['cached_at']}")
    print("Done!")


if __name__ == "__main__":
    main()
