#!/usr/bin/env python3
"""
Update .etf_cache.json with fresh ETF flow data from farside.co.uk/btc/
"""
import json
import re
import time
import cloudscraper
from bs4 import BeautifulSoup
from datetime import datetime

CACHE_PATH = "/home/ubuntu/sfc/.etf_cache.json"
URL = "https://farside.co.uk/btc/"

ETF_KEYS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]
MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}

def parse_date(date_str):
    """Convert '08 Jul 2026' to '2026-07-08'"""
    parts = date_str.strip().split()
    if len(parts) != 3:
        return None
    day, month, year = parts
    month_num = MONTH_MAP.get(month)
    if not month_num:
        return None
    return f"{year}-{month_num}-{day.zfill(2)}"

def parse_value(val_str):
    """Parse a value like '(59.1)' -> -59.1, '86.8' -> 86.8, '-' -> None"""
    val_str = val_str.strip().replace(",", "")
    if val_str in ("-", "—", "", "–"):
        return None
    if val_str.startswith("(") and val_str.endswith(")"):
        return -float(val_str[1:-1])
    return float(val_str)

def fetch_page():
    """Fetch the page using cloudscraper"""
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL, timeout=30)
    resp.raise_for_status()
    return resp.text

def parse_table(html):
    """Parse the ETF flow table from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        print("ERROR: No tables found")
        return []
    
    table = tables[0]
    rows = table.find_all('tr')
    
    flows = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 14:
            continue
        
        date_text = cells[0].get_text(strip=True)
        # Skip non-date rows
        if not date_text or date_text in ("", "Fee", "Total", "Average", "Maximum", "Minimum"):
            continue
        
        # Check if it looks like a date
        date_match = re.match(r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', date_text)
        if not date_match:
            continue
        
        date_iso = parse_date(date_text)
        if not date_iso:
            continue
        
        # Parse ETF values - cell indices: 1=IBIT, 2=FBTC, ..., 12=BTC, 13=Total
        etfs = {}
        all_dashes = True
        total_val = None
        
        for i, key in enumerate(ETF_KEYS):
            raw = cells[i + 1].get_text(strip=True)
            val = parse_value(raw)
            if val is not None:
                etfs[key] = val
                all_dashes = False
            else:
                etfs[key] = 0.0  # Default to 0.0 for unavailable data
        
        # Parse total
        total_raw = cells[13].get_text(strip=True)
        total_val = parse_value(total_raw)
        
        if total_val is not None:
            total_usd = int(total_val * 1_000_000)
        else:
            total_usd = 0
        
        flows.append({
            "date": date_iso,
            "total_btc": None,
            "total_usd": total_usd,
            "etfs": etfs
        })
        
        print(f"  Parsed: {date_iso} | Total={total_val} USD={total_usd}")
    
    return flows

def extract_cumulative_data(html):
    """Extract cumulative totals from seriesData in JavaScript"""
    m = re.search(r'const seriesData = (\{.*?\});', html, re.DOTALL)
    if not m:
        print("WARNING: seriesData not found")
        return None, None
    
    try:
        series_data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        print(f"ERROR parsing seriesData: {e}")
        return None, None
    
    # Sum last values of all ETFs
    total_usd_m = 0
    for key, values in series_data.items():
        if values and len(values) > 0:
            total_usd_m += values[-1]
    
    cumulative_usd = int(total_usd_m * 1_000_000)
    print(f"  Cumulative USD: {cumulative_usd} (${cumulative_usd:,.0f})")
    
    return None, cumulative_usd  # cumulative_btc = None (not available on page)

def update_cache(existing, new_flows, cumulative_btc, cumulative_usd):
    """Merge new flows into existing cache, keeping existing data"""
    # Build dict of existing flows by date
    existing_by_date = {f["date"]: f for f in existing.get("flows", [])}
    
    # Merge new flows
    updated_count = 0
    added_count = 0
    for flow in new_flows:
        date = flow["date"]
        if date in existing_by_date:
            # Update only if values differ (data may have been corrected)
            existing_by_date[date] = flow
            updated_count += 1
        else:
            existing_by_date[date] = flow
            added_count += 1
    
    print(f"  Updated: {updated_count} dates, Added: {added_count} new dates")
    
    # Sort by date
    merged_flows = sorted(existing_by_date.values(), key=lambda x: x["date"])
    
    # Update cumulative values
    if cumulative_usd is not None:
        existing["cumulative_usd"] = cumulative_usd
    if cumulative_btc is not None:
        existing["cumulative_btc"] = cumulative_btc
    
    existing["flows"] = merged_flows
    existing["last_update"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    existing["cached_at"] = time.time()
    
    return existing

def main():
    print("=== ETF Cache Updater ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Load existing cache
    try:
        with open(CACHE_PATH, 'r') as f:
            existing = json.load(f)
        print(f"Loaded existing cache: {len(existing.get('flows', []))} entries")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing cache found, starting fresh")
        existing = {"flows": [], "cumulative_btc": None, "cumulative_usd": None}
    
    # Fetch page
    print("Fetching page...")
    html = fetch_page()
    print(f"  Got {len(html)} bytes")
    
    # Check for Cloudflare block
    if "challenge" in html.lower()[:2000] and "cf_chl" in html[:2000]:
        print("ERROR: Blocked by Cloudflare!")
        return
    
    # Parse table
    print("Parsing table...")
    new_flows = parse_table(html)
    print(f"  Parsed {len(new_flows)} flow entries")
    
    if not new_flows:
        print("ERROR: No flow data parsed!")
        return
    
    # Extract cumulative data
    print("Extracting cumulative totals...")
    cumulative_btc, cumulative_usd = extract_cumulative_data(html)
    
    # Merge and write
    print("Merging with existing cache...")
    result = update_cache(existing, new_flows, cumulative_btc, cumulative_usd)
    
    print("Writing cache file...")
    with open(CACHE_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nDone! Cache now has {len(result['flows'])} flow entries")
    print(f"  cumulative_usd: {result.get('cumulative_usd')}")
    print(f"  cumulative_btc: {result.get('cumulative_btc')}")
    print(f"  last_update: {result.get('last_update')}")
    print(f"  cached_at: {result.get('cached_at')}")

if __name__ == "__main__":
    main()
