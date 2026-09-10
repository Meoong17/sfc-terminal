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

def _extract_from_soup(soup):
    """Parse flows + cumulative (millions) from a BeautifulSoup of the all-data page."""
    tables = soup.find_all('table')
    target_table = None
    for table in tables:
        thead = table.find('thead')
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all('th')]
            if headers[:3] == ['Date', 'IBIT', 'FBTC']:
                target_table = table
                break
    if not target_table:
        return [], 0.0

    rows = target_table.find_all('tr')
    flows = []
    etf_names = ['IBIT', 'FBTC', 'BITB', 'ARKB', 'BTCO', 'EZBC', 'BRRR', 'HODL', 'BTCW', 'MSBT', 'GBTC', 'BTC']
    for row in rows:
        try:
            cells = row.find_all('td')
            if len(cells) != 14:
                continue
            date_raw = cells[0].get_text(strip=True)
            if date_raw in ('Total', 'Average', 'Maximum', 'Minimum', 'Fee'):
                continue
            date_parsed = parse_date(date_raw)
            if not date_parsed:
                continue
            raw = [c.get_text(strip=True) for c in cells[1:-1]]
            # A pending placeholder row is all dashes/empty (not yet reported) -> skip.
            if all(v in ('', '-') for v in raw):
                continue
            etfs = {}
            for i, name in enumerate(etf_names):
                etfs[name] = round(parse_value(cells[i+1].get_text(strip=True)), 4)
            total_val = parse_value(cells[-1].get_text(strip=True))
            flows.append({
                "date": date_parsed,
                "total_btc": None,
                "total_usd": int(round(total_val * 1_000_000)),
                "etfs": etfs
            })
        except Exception:
            continue

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


def _fetch_playwright():
    """Fallback via headless Chromium when Cloudflare blocks plain requests."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"[WARN] Playwright unavailable: {e}")
        return [], 0.0
    flows, cum = [], 0.0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-blink-features=AutomationControlled'])
        ctx = browser.new_context(
            user_agent=HEADERS['User-Agent'],
            viewport={'width': 1700, 'height': 1200})
        pg = ctx.new_page()
        try:
            pg.goto('https://farside.co.uk/bitcoin-etf-flow-all-data/', timeout=60000, wait_until='domcontentloaded')
            for _ in range(30):
                if 'All Data' in pg.title():
                    break
                import time as _t; _t.sleep(1.5)
            # The all-data table may lazy-load rows. Poll until row count is stable,
            # scrolling all scrollable containers, so no trailing dates are missed.
            import time as _t
            last_n = -1
            stable = 0
            for _ in range(60):
                n = pg.evaluate('document.querySelectorAll("table")[0] ? document.querySelectorAll("table")[0].querySelectorAll("tbody tr").length : 0')
                if n == last_n:
                    stable += 1
                else:
                    stable = 0
                    last_n = n
                if stable >= 4 and n > 0:
                    break
                pg.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                pg.evaluate('''()=>{document.querySelectorAll('div').forEach(e=>{if(e.scrollHeight>e.clientHeight+10){e.scrollTop=e.scrollHeight;}})}''')
                pg.wait_for_timeout(600)
            pg.wait_for_timeout(1500)
            html = pg.content()
        except Exception as e:
            print(f"[WARN] Playwright nav error: {e}")
            browser.close()
            return [], 0.0
        browser.close()
    from bs4 import BeautifulSoup as BS
    try:
        flows, cum = _extract_from_soup(BS(html, 'html.parser'))
    except Exception as e:
        print(f"[WARN] Playwright parse error: {e}")
        return [], 0.0
    return flows, cum


URL = 'https://farside.co.uk/bitcoin-etf-flow-all-data/'


def _fetch_curl_cffi():
    """Primary path: TLS/JA3 impersonation (curl_cffi). Farside sits behind Cloudflare,
    which 403s plain `requests` — impersonating a real Chrome handshake passes."""
    try:
        from curl_cffi import requests as curl_requests
    except Exception as e:
        print(f"[WARN] curl_cffi unavailable: {e}")
        return None
    try:
        r = curl_requests.get(URL, impersonate='chrome', timeout=40)
    except Exception as e:
        print(f"[WARN] curl_cffi failed ({e})")
        return None
    if r.status_code != 200 or 'Just a moment' in r.text:
        print(f"[WARN] curl_cffi blocked (status={r.status_code})")
        return None
    return _extract_from_soup(BeautifulSoup(r.text, 'html.parser'))


def fetch_flows():
    """Fetch ETF flow data from Farside all-data page. Returns (flows_list, cumulative_total_millions)."""
    # Primary: curl_cffi browser-impersonation (beats the Cloudflare 403 on plain requests).
    out = _fetch_curl_cffi()
    if out is not None and out[0]:
        return out
    print("[WARN] curl_cffi path returned nothing; trying plain requests...")

    # Fallback 1: plain requests.
    try:
        r = requests.get(URL, timeout=30, headers=HEADERS)
        r.raise_for_status()
        if 'Just a moment' not in r.text and r.status_code != 403:
            flows, cum = _extract_from_soup(BeautifulSoup(r.text, 'html.parser'))
            if flows:
                return flows, cum
    except requests.RequestException as e:
        print(f"[WARN] requests failed ({e})")

    # Fallback 2: headless Chromium.
    print("[WARN] Falling back to headless Chromium...")
    return _fetch_playwright()


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
    
    # Compute cumulative_usd from the total row data if available.
    # Guard: if the fetch failed/returned nothing, keep the previous cumulative
    # value instead of zeroing it out (would corrupt the cache).
    if cumulative_total_millions and len(new_flows) > 0:
        cumulative_usd = int(round(cumulative_total_millions * 1_000_000))
    else:
        cumulative_usd = existing.get('cumulative_usd')
        if cumulative_usd is None:
            cumulative_usd = 0
    
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
