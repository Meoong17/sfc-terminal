#!/usr/bin/env python3
"""
BTC ETF Data Scraper for Farside.co.uk
Fetches full ETF flow data, parses it, merges with existing cache, and writes updated file.
"""
import json
import re
import time
import urllib.request
from datetime import datetime

# URLs
MAIN_URL = "https://farside.co.uk/btc/"
ALL_DATA_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
CACHE_FILE = "/home/ubuntu/sfc/.etf_cache.json"

# ETF column order in the table
ETF_COLUMNS = {
    1: "IBIT", 2: "FBTC", 3: "BITB", 4: "ARKB", 5: "BTCO",
    6: "EZBC", 7: "BRRR", 8: "HODL", 9: "BTCW", 10: "MSBT",
    11: "GBTC", 12: "BTC"
}

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}


def parse_date(dd_mon_yyyy):
    """Convert '25 Jun 2026' -> '2026-06-25'"""
    dd_mon_yyyy = dd_mon_yyyy.strip()
    # Some rows might already be in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', dd_mon_yyyy):
        return dd_mon_yyyy
    m = re.match(r'(\d{1,2})\s+(\w{3})\s+(\d{4})', dd_mon_yyyy)
    if m:
        day, mon, year = m.group(1), m.group(2), m.group(3)
        mon_num = MONTH_MAP.get(mon)
        if mon_num:
            return f"{year}-{mon_num}-{int(day):02d}"
    return None


def parse_value(val_str):
    """Parse ETF value from string. Returns float."""
    val_str = val_str.strip().replace(",", "").replace(" ", "")
    if not val_str or val_str in ("-", "", "N/A", "—", "–"):
        return 0.0
    if val_str.startswith("(") and val_str.endswith(")"):
        return -float(val_str[1:-1])
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def strip_html(text):
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text).strip()


def fetch(url):
    """Fetch a URL and return the HTML content."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8', errors='replace')


def parse_table(html):
    """Parse the Farside ETF table and return list of flow dicts."""
    # Find the etf table
    table_match = re.search(r'<table\s+class="etf">.*?</table>', html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        print("ERROR: No etf table found")
        return []
    
    table_html = table_match.group(0)
    
    # Extract all <tr> blocks inside tbody
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL | re.IGNORECASE)
    if not tbody_match:
        print("ERROR: No tbody found in table")
        return []
    
    tbody = tbody_match.group(1)
    
    # Parse rows
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    
    date_pattern = re.compile(r'^\d{1,2}\s+\w{3}\s+\d{4}$')
    date_pattern_iso = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    
    flows = []
    
    for tr_match in row_pattern.finditer(tbody):
        tr_content = tr_match.group(1)
        cells = []
        for td_match in cell_pattern.finditer(tr_content):
            cell_html = td_match.group(1)
            cell_text = strip_html(cell_html)
            cells.append(cell_text)
        
        if not cells:
            continue
        
        first = cells[0].strip()
        
        # Skip non-date rows (Total, Average, Maximum, Minimum, Fee, etc.)
        if not date_pattern.match(first) and not date_pattern_iso.match(first):
            continue
        
        date = parse_date(first)
        if not date:
            continue
        
        etfs = {}
        for idx, etf_name in ETF_COLUMNS.items():
            if idx < len(cells):
                val = parse_value(cells[idx])
            else:
                val = 0.0
            etfs[etf_name] = round(val, 2)
        
        # Total column (index 13)
        total_val = 0.0
        if len(cells) > 13 and cells[13].strip():
            total_val = parse_value(cells[13])
        else:
            total_val = sum(etfs.values())
        
        total_usd = int(round(total_val * 1_000_000))
        
        flows.append({
            "date": date,
            "total_btc": None,
            "total_usd": total_usd,
            "etfs": etfs
        })
    
    return flows


def parse_cumulative_from_total_row(html):
    """Parse the Total row from the all-data table to get cumulative USD."""
    table_match = re.search(r'<table\s+class="etf">.*?</table>', html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return None
    
    table_html = table_match.group(0)
    
    # Find tbody first to avoid matching header rows
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL | re.IGNORECASE)
    if not tbody_match:
        return None
    
    tbody = tbody_match.group(1)
    
    # Find the Total row inside tbody
    total_match = re.search(r'<span class="tabletext">Total</span>.*?</tr>', tbody, re.DOTALL)
    if not total_match:
        return None
    
    total_html = total_match.group(0)
    
    # Extract cells
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    cells = [strip_html(td) for td in cell_pattern.findall(total_html)]
    
    print(f"Total row cells ({len(cells)}): {cells}")
    
    # cells[0] is the first ETF value (IBIT=60,100), 
    # the "Total" label cell is not captured because the regex starts inside it
    # Last cell (index -1) is the cumulative total in millions
    if len(cells) >= 13:
        total_val = cells[-1].strip().replace(",", "")
        if total_val:
            try:
                val = float(total_val)
                return int(val * 1_000_000)  # Convert millions to raw USD
            except ValueError:
                pass
    
    return None


def read_cache():
    """Read existing cache file."""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Cache read error: {e}")
        return {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "last_update": None, "cached_at": 0}


def merge_flows(existing_flows, new_flows):
    """Merge new flows into existing flows by date. 
    New data overwrites old data for same date. Returns sorted list."""
    by_date = {}
    for f in existing_flows:
        by_date[f["date"]] = f
    for f in new_flows:
        by_date[f["date"]] = f
    # Sort by date
    merged = sorted(by_date.values(), key=lambda x: x["date"])
    return merged


def main():
    now = time.time()
    ts_now = datetime.utcfromtimestamp(now).strftime("%Y-%m-%dT%H:%M:%S")
    
    print(f"[{ts_now}] Fetching BTC ETF data from Farside...")
    
    # Read existing cache
    cache = read_cache()
    existing_flows = cache.get("flows", [])
    existing_dates = {f["date"] for f in existing_flows}
    print(f"Existing cache: {len(existing_flows)} flows, last update: {cache.get('last_update')}")
    
    # Fetch the all-data page (has full history)
    print(f"Fetching {ALL_DATA_URL}...")
    html = fetch(ALL_DATA_URL)
    print(f"Downloaded {len(html)} bytes")
    
    # Parse table
    new_flows = parse_table(html)
    print(f"Parsed {len(new_flows)} flow entries from table")
    
    if not new_flows:
        print("ERROR: No flows parsed from table!")
        return
    
    # Show what we got
    print(f"Date range: {new_flows[0]['date']} to {new_flows[-1]['date']}")
    new_dates = {f["date"] for f in new_flows}
    unique_new = new_dates - existing_dates
    print(f"New dates: {len(unique_new)} - {sorted(unique_new)[-5:] if unique_new else 'none'}")
    
    # Parse cumulative totals from Total row
    cumulative_usd = parse_cumulative_from_total_row(html)
    if cumulative_usd:
        print(f"Cumulative USD from Total row: ${cumulative_usd:,}")
    
    # Try to get cumulative BTC from main page
    cumulative_btc = cache.get("cumulative_btc")
    # Parse main page for any cumulative values
    try:
        main_html = fetch(MAIN_URL)
        for m in re.finditer(r'[\d,]+\.?\d*\s*K\s*BTC', main_html):
            val_str = re.sub(r'[^\d.]', '', m.group().split('K')[0])
            try:
                val = float(val_str) * 1000
                cumulative_btc = val
                print(f"Cumulative BTC from main page: {val}")
            except ValueError:
                pass
    except Exception as e:
        print(f"Could not fetch main page for BTC cumulative: {e}")
    
    # Merge flows
    merged_flows = merge_flows(existing_flows, new_flows)
    print(f"Merged flows: {len(merged_flows)} total")
    
    # Build output
    output = {
        "flows": merged_flows,
        "cumulative_btc": cumulative_btc if cumulative_btc else cache.get("cumulative_btc"),
        "cumulative_usd": cumulative_usd if cumulative_usd is not None else cache.get("cumulative_usd"),
        "last_update": ts_now,
        "cached_at": now
    }
    
    # Write cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten {len(merged_flows)} flows to {CACHE_FILE}")
    
    # Print summary
    last_5 = merged_flows[-5:]
    print(f"\nLast 5 entries:")
    for f in last_5:
        total = f["total_usd"]
        print(f"  {f['date']}: ${total:+,}")
    
    print(f"\nCumulative BTC: {output['cumulative_btc']}")
    print(f"Cumulative USD: ${output['cumulative_usd']:,}" if output['cumulative_usd'] else "Cumulative USD: None")
    print(f"Cached at: {ts_now}")


if __name__ == "__main__":
    main()
