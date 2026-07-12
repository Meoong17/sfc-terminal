#!/usr/bin/env python3
"""
Scrape Bitcoin ETF flow data from Farside, merge with existing cache, write updated file.
"""
import json
import re
import time
import urllib.request
import ssl
from datetime import datetime

CACHE_PATH = "/home/ubuntu/sfc/.etf_cache.json"

# ETF column order (skip "Date" column)
ETF_COLUMNS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]
TOTAL_COLUMN = "Total"

def fetch_page(url):
    """Download the Farside BTC ETF page."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    return resp.read().decode('utf-8')

def parse_value(text):
    """Parse a Farside value. Parentheses (440.3) -> -440.3. Commas removed."""
    text = text.strip()
    if not text:
        return 0.0
    # Check for parentheses (negative value)
    m = re.match(r'^\(([\d,.-]+)\)$', text)
    if m:
        return -float(m.group(1).replace(',', ''))
    return float(text.replace(',', ''))

def parse_date(text):
    """Parse '23 Jun 2026' -> '2026-06-23'"""
    text = text.strip()
    # Map month abbreviations
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    parts = text.split()
    if len(parts) != 3:
        return None
    day, month_str, year = parts
    month = months.get(month_str)
    if not month:
        return None
    return f"{year}-{month}-{day.zfill(2)}"

def extract_innermost_text(cell_html):
    """Extract the innermost text from a cell, stripping all HTML tags."""
    # Remove all HTML tags, get the text content
    text = re.sub(r'<[^>]+>', '', cell_html)
    return text.strip()

def parse_rows(html):
    """
    Parse all data rows from the HTML table.
    Each row: <td><span class="tabletext">DATE</span></td>
    followed by 13 <td><div...><span...>VALUE</span></div></td> cells
    Last cell = Total column.
    
    Returns list of dicts and cumulative total from Total row.
    """
    flows = []
    cumulative_total = None
    
    # Find the tbody section
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
    if not tbody_match:
        print("ERROR: Could not find <tbody>")
        return flows, cumulative_total
    
    tbody = tbody_match.group(1)
    
    # Split into individual <tr> blocks
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tbody, re.DOTALL)
    
    for row in rows:
        # Extract all td cells
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if not cells:
            continue
        
        # First cell contains the label (date or "Total", "Average", etc.)
        first_cell = cells[0].strip()
        label = extract_innermost_text(first_cell)
        if not label:
            continue
        
        # Skip non-data rows
        if label in ('Total', 'Average', 'Maximum', 'Minimum', 'Fee', ''):
            if label == 'Total' and len(cells) >= 14:
                # Extract cumulative total from the last cell (Total column)
                total_cell = cells[-1]
                total_text = extract_innermost_text(total_cell)
                cumulative_total = parse_value(total_text)
            continue
        
        # Try to parse as a date
        date_str = parse_date(label)
        if date_str is None:
            continue
        
        # We need at least 13 data cells + the date cell = 14 cells total
        if len(cells) < 14:
            print(f"WARNING: Row for {label} has only {len(cells)} cells, skipping")
            continue
        
        # Cells[0] is date. Cells[1..12] are ETFs. Cells[13] is Total.
        # Some rows might have different structure. We'll get cells[1:13] for ETFs
        # and cells[-1] for Total.
        etf_values = {}
        for i, col_name in enumerate(ETF_COLUMNS):
            if i + 1 < len(cells):
                cell = cells[i + 1]
                val_text = extract_innermost_text(cell)
                etf_values[col_name] = parse_value(val_text)
            else:
                etf_values[col_name] = 0.0
        
        # Parse Total from last cell
        total_cell = cells[-1]
        total_text = extract_innermost_text(total_cell)
        total_raw = parse_value(total_text)
        
        flow = {
            "date": date_str,
            "total_btc": None,
            "total_usd": int(total_raw * 1_000_000),
            "etfs": etf_values
        }
        flows.append(flow)
    
    return flows, cumulative_total

def load_cache(path):
    """Load existing cache file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "last_update": None, "cached_at": None}

def merge_flows(existing_flows, new_flows):
    """Merge new flows with existing flows, keeping existing ones and overwriting with new data by date."""
    date_map = {f['date']: f for f in existing_flows}
    for f in new_flows:
        date_map[f['date']] = f
    # Sort by date
    merged = sorted(date_map.values(), key=lambda x: x['date'])
    return merged

def main():
    print("Fetching Farside BTC ETF page...")
    html = fetch_page("https://farside.co.uk/btc/")
    print(f"Downloaded {len(html)} bytes")
    
    print("Parsing table rows...")
    new_flows, cumulative_total = parse_rows(html)
    print(f"Parsed {len(new_flows)} new/updated flow rows")
    
    if cumulative_total is not None:
        print(f"Cumulative total from table: {cumulative_total} (US$m)")
        cumulative_usd = int(cumulative_total * 1_000_000)
    else:
        print("WARNING: Could not parse cumulative total from table")
        cumulative_usd = None
    
    print("Loading existing cache...")
    cache = load_cache(CACHE_PATH)
    existing_count = len(cache.get("flows", []))
    print(f"Existing cache has {existing_count} flows")
    
    print("Merging flows...")
    cache["flows"] = merge_flows(cache.get("flows", []), new_flows)
    print(f"Merged to {len(cache['flows'])} flows")
    
    # Update cumulative values
    if cumulative_usd is not None:
        cache["cumulative_usd"] = cumulative_usd
    if cache.get("cumulative_btc") is None:
        cache["cumulative_btc"] = None  # keep as None
    
    # Update timestamps
    now = time.time()
    cache["cached_at"] = now
    
    # last_update: use current time formatted
    dt = datetime.fromtimestamp(now)
    cache["last_update"] = dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    print("Writing cache file...")
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Wrote {len(cache['flows'])} flows to {CACHE_PATH}")
    print(f"cumulative_usd: {cache.get('cumulative_usd')}")
    print(f"cumulative_btc: {cache.get('cumulative_btc')}")
    print(f"last_update: {cache.get('last_update')}")
    print(f"cached_at: {cache.get('cached_at')}")
    
    # Verify
    with open(CACHE_PATH, 'r') as f:
        verified = json.load(f)
    assert len(verified['flows']) == len(cache['flows']), "Verification failed: flow count mismatch"
    print("✓ Verification: valid JSON, file integrity confirmed")

if __name__ == "__main__":
    main()
