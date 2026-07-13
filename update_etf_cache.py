#!/usr/bin/env python3
"""
ETF cache updater script.
Fetches latest data from farside.co.uk/btc/ and merges with existing cache.
"""
import json, time, re, sys, os
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ETF_COLUMNS = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']
CACHE_FILE = '/home/ubuntu/sfc/.etf_cache.json'

def parse_value(val):
    """Parse value: '(239.3)' -> -239.3, '0.0' -> 0.0, '23.6' -> 23.6"""
    val = val.strip()
    if not val or val == '-':
        return 0.0
    if val.startswith('(') and val.endswith(')'):
        return -float(val[1:-1].replace(',', ''))
    return float(val.replace(',', ''))

def parse_date(date_str):
    """Parse '24 Jun 2026' to '2026-06-24'"""
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    parts = date_str.strip().split()
    day = parts[0].zfill(2)
    month = months.get(parts[1], '01')
    year = parts[2] if len(parts) > 2 else str(datetime.now().year)
    return f"{year}-{month}-{day}"

def read_existing_cache():
    if not os.path.exists(CACHE_FILE):
        return {'flows': [], 'cumulative_btc': None, 'cumulative_usd': None}
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {'flows': [], 'cumulative_btc': None, 'cumulative_usd': None}

def scrape_farside():
    """Scrape ETF data from farside.co.uk/btc/"""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
        )
        page = context.new_page()
        
        page.goto('https://farside.co.uk/btc/', wait_until='networkidle', timeout=60000)
        page.wait_for_timeout(3000)
        
        html = page.content()
        if 'Just a moment' in html or 'challenge' in html.lower()[:200]:
            print("ERROR: Blocked by Cloudflare", file=sys.stderr)
            browser.close()
            return None
        
        # Scroll to ensure all data loaded
        for _ in range(15):
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(300)
        page.wait_for_timeout(2000)
        
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find ETF table
        etf_table = None
        for table in soup.find_all('table'):
            header_texts = [h.get_text(strip=True) for h in table.find_all('th')]
            if 'IBIT' in header_texts or 'FBTC' in header_texts:
                etf_table = table
                break
        
        if not etf_table:
            print("ERROR: Could not find ETF table", file=sys.stderr)
            browser.close()
            return None
        
        # Parse table rows
        new_flows = []
        rows = etf_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if not cells:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            first = texts[0].strip()
            
            # Skip non-date rows
            if first in ('Fee', 'Total', 'Average', 'Maximum', 'Minimum', ''):
                continue
            if not re.match(r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\s+\d{4})?$', first):
                continue
            
            date = parse_date(first)
            total_val = parse_value(texts[-1] if len(texts) > 13 else '0')
            
            etfs = {}
            for idx, col in enumerate(ETF_COLUMNS):
                val_raw = texts[idx + 1] if idx + 1 < len(texts) else '0'
                etfs[col] = parse_value(val_raw)
            
            new_flows.append({
                'date': date,
                'total_btc': None,
                'total_usd': int(total_val * 1_000_000),
                'etfs': etfs
            })
        
        # Get cumulative totals from totalData JavaScript array
        cumulative_usd = None
        match = re.search(r'totalData\s*=\s*\[([^\]]+)\]', html)
        if match:
            values = [float(v.strip()) for v in match.group(1).split(',') if v.strip()]
            if values:
                cumulative_usd = int(values[-1] * 1_000_000)
        
        browser.close()
        
        return {
            'new_flows': new_flows,
            'cumulative_usd': cumulative_usd,
            'cumulative_btc': None,  # Not available on Farside page
        }

def merge_flows(existing_flows, new_flows):
    """Merge new flows with existing flows, deduplicating by date."""
    # Build dict of existing flows by date
    flow_by_date = {}
    for flow in existing_flows:
        flow_by_date[flow['date']] = flow
    
    # Add/update with new flows
    for flow in new_flows:
        flow_by_date[flow['date']] = flow
    
    # Return sorted by date
    return sorted(flow_by_date.values(), key=lambda f: f['date'])

def main():
    now = datetime.now(timezone.utc)
    
    # Read existing cache
    existing = read_existing_cache()
    print(f"Existing cache: {len(existing.get('flows', []))} flows, "
          f"cumulative_usd={existing.get('cumulative_usd')}", file=sys.stderr)
    
    # Scrape new data
    result = scrape_farside()
    if result is None:
        print("Scrape failed, keeping existing cache", file=sys.stderr)
        return
    
    new_flows = result['new_flows']
    print(f"Scraped {len(new_flows)} new date rows", file=sys.stderr)
    for f in new_flows:
        print(f"  {f['date']}: total_usd={f['total_usd']}", file=sys.stderr)
    
    # Merge flows
    merged_flows = merge_flows(existing.get('flows', []), new_flows)
    print(f"Merged: {len(merged_flows)} total flows", file=sys.stderr)
    
    # Determine cumulative values
    cumulative_usd = result['cumulative_usd'] or existing.get('cumulative_usd')
    cumulative_btc = existing.get('cumulative_btc')  # Keep existing BTC cumulative
    
    # Build output
    output = {
        'flows': merged_flows,
        'cumulative_btc': cumulative_btc,
        'cumulative_usd': cumulative_usd,
        'last_update': now.strftime('%Y-%m-%dT%H:%M:%S'),
        'cached_at': time.time(),
    }
    
    # Write to file
    with open(CACHE_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Written {CACHE_FILE}", file=sys.stderr)
    print(f"  flows: {len(merged_flows)}", file=sys.stderr)
    print(f"  cumulative_usd: {cumulative_usd}", file=sys.stderr)
    print(f"  last_update: {output['last_update']}", file=sys.stderr)
    print(f"  cached_at: {output['cached_at']}", file=sys.stderr)

if __name__ == '__main__':
    main()
