#!/usr/bin/env python3
"""Update .etf_cache.json with latest data from farside.co.uk/btc/"""
import json, time, re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

CACHE_PATH = Path("/home/ubuntu/sfc/.etf_cache.json")
ETF_COLUMNS = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']

def parse_value(val):
    """Parse a table cell value. Parentheses = negative."""
    val = val.strip()
    if val == '-' or val == '':
        return 0.0
    # Remove commas from numbers like "60,770"
    val = val.replace(',', '')
    if val.startswith('(') and val.endswith(')'):
        return -float(val[1:-1])
    return float(val)

def parse_date(datestr):
    """Parse '06 Jul 2026' -> '2026-07-06'"""
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    parts = datestr.split()
    if len(parts) != 3:
        return None
    day, month_str, year = parts
    month = months.get(month_str)
    if not month:
        return None
    return f"{year}-{month}-{day.zfill(2)}"

def scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        page.goto('https://farside.co.uk/btc/', wait_until='networkidle', timeout=60000)
        time.sleep(5)

        # Get the main table
        table = page.query_selector_all('table')[0]
        rows = table.query_selector_all('tr')

        new_flows = []
        for row in rows:
            cells = row.query_selector_all('td, th')
            texts = [cell.inner_text().strip() for cell in cells]
            if not texts:
                continue
            
            first_cell = texts[0]
            
            # Skip non-date rows
            if first_cell in ('Fee', 'Total', 'Average', 'Maximum', 'Minimum', '', 'Date'):
                continue
            
            # Check if first cell looks like a date (e.g., "06 Jul 2026")
            if not re.match(r'\d{1,2} \w{3} \d{4}', first_cell):
                continue
            
            date_str = parse_date(first_cell)
            if not date_str:
                continue
            
            # Parse ETF values (columns 1-13, the last is Total)
            etfs = {}
            for idx, col_name in enumerate(ETF_COLUMNS):
                if idx + 1 < len(texts):
                    etfs[col_name] = parse_value(texts[idx + 1])
            
            # Parse Total value (last column)
            total_val = parse_value(texts[-1]) if len(texts) > 1 else 0.0
            
            flow_entry = {
                "date": date_str,
                "total_btc": None,
                "total_usd": int(round(total_val * 1_000_000)),
                "etfs": etfs
            }
            new_flows.append(flow_entry)
            print(f"  Parsed: {date_str} total={total_val}M USD={flow_entry['total_usd']}")

        # Get cumulative data from "Total" row
        cumulative_usd = None
        for row in rows:
            cells = row.query_selector_all('td, th')
            texts = [cell.inner_text().strip() for cell in cells]
            if texts and texts[0] == 'Total':
                total_cumul = parse_value(texts[-1])
                cumulative_usd = int(round(total_cumul * 1_000_000))
                print(f"  Cumulative Total: {total_cumul}M = ${cumulative_usd}")
                break

        # Look for cumulative BTC on page
        cumulative_btc = None
        body_text = page.inner_text('body')
        # Look for patterns like "+676.78K BTC" or similar
        btc_patterns = re.findall(r'([+-]?\d+\.?\d*)\s*K\s*BTC', body_text)
        if btc_patterns:
            cumulative_btc = float(btc_patterns[-1]) * 1000
            print(f"  Cumulative BTC (from K pattern): {cumulative_btc}")

        # Also try looking for "BTC" near numbers
        if cumulative_btc is None:
            # Try broader patterns
            for line in body_text.split('\n'):
                m = re.search(r'([+-]?\d+\.?\d*)\s*K?\s*BTC', line)
                if m and not any(x in line for x in ['fee', 'Fee', 'IBIT', 'FBTC']):
                    val = m.group(1)
                    if 'K' in line:
                        cumulative_btc = float(val) * 1000
                    else:
                        cumulative_btc = float(val)
                    print(f"  Cumulative BTC (from line): {line.strip()} -> {cumulative_btc}")
                    break

        browser.close()
        return new_flows, cumulative_usd, cumulative_btc

def main():
    print("=== Scraping Farside ETF data ===")
    new_flows, cumulative_usd, cumulative_btc = scrape()
    
    if not new_flows:
        print("ERROR: No data scraped!")
        sys.exit(1)
    
    print(f"\nScraped {len(new_flows)} flow entries")
    print(f"Cumulative USD: {cumulative_usd}")
    print(f"Cumulative BTC: {cumulative_btc}")
    
    # Load existing cache
    existing = {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "last_update": None, "cached_at": 0}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, 'r') as f:
                existing = json.load(f)
            print(f"Loaded cache: {len(existing.get('flows', []))} existing entries")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse cache: {e}")
    
    # Merge flows: keep existing + upsert new
    existing_flows = {f['date']: f for f in existing.get('flows', [])}
    
    for flow in new_flows:
        existing_flows[flow['date']] = flow
    
    merged_flows = sorted(existing_flows.values(), key=lambda x: x['date'])
    print(f"Merged flows: {len(merged_flows)} total entries")
    
    # Update metadata
    now = time.time()
    from datetime import datetime
    last_update_dt = datetime.utcfromtimestamp(now)
    last_update_str = last_update_dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    result = {
        "flows": merged_flows,
        "cumulative_btc": cumulative_btc if cumulative_btc is not None else existing.get('cumulative_btc'),
        "cumulative_usd": cumulative_usd if cumulative_usd is not None else existing.get('cumulative_usd'),
        "last_update": last_update_str,
        "cached_at": now
    }
    
    # Write cache
    with open(CACHE_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nWritten cache: {len(merged_flows)} flows, {CACHE_PATH}")
    print(f"Last update: {last_update_str}")
    print(f"cumulative_usd: {result['cumulative_usd']}")
    print(f"cumulative_btc: {result['cumulative_btc']}")
    
    # Verify
    with open(CACHE_PATH, 'r') as f:
        verified = json.load(f)
    print(f"\nVerification: {len(verified['flows'])} flows, valid JSON OK")

if __name__ == '__main__':
    main()
