#!/usr/bin/env python3
"""
Scrape Bitcoin ETF flow data from Farside.co.uk and update .etf_cache.json
"""
import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime, timezone

CACHE_PATH = "/home/ubuntu/sfc/.etf_cache.json"
URL = "https://farside.co.uk/btc/"

ETF_ORDER = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}

def parse_farside_date(date_str):
    """Convert '29 Jun 2026' -> '2026-06-29'"""
    parts = date_str.strip().split()
    if len(parts) != 3:
        return None
    day, month_name, year = parts
    month = MONTH_MAP.get(month_name)
    if not month:
        return None
    return f"{year}-{month}-{day.zfill(2)}"

def parse_farside_value(val_str):
    """Parse value like '(300.4)' -> -300.4, '86.8' -> 86.8, '0.0' -> 0.0"""
    val_str = val_str.strip().replace(",", "")
    if val_str.startswith("(") and val_str.endswith(")"):
        return -float(val_str[1:-1])
    else:
        return float(val_str)

def main():
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL, timeout=30)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    # --- Find ETF table ---
    tables = soup.find_all('table')
    etf_table = None
    for table in tables:
        text = table.get_text()
        if 'IBIT' in text and 'FBTC' in text and 'GBTC' in text:
            etf_table = table
            break

    if not etf_table:
        raise Exception("Could not find ETF table on page")

    rows = etf_table.find_all('tr')

    # --- Parse data rows (skip Fee, Total, Average, Max, Min rows) ---
    new_flows = []
    skip_keywords = {"fee", "total", "average", "maximum", "minimum"}

    for row in rows:
        cells = row.find_all(['td', 'th'])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        first_cell = texts[0].strip().lower()
        # Skip non-date rows
        if first_cell in skip_keywords:
            continue
        if first_cell == '':
            continue

        # Check if first cell is a date
        date_str = texts[0].strip()
        date_iso = parse_farside_date(date_str)
        if not date_iso:
            continue

        # Parse ETF values (positions 1 through 12 for 12 ETFs)
        etf_values = {}
        for i, etf_name in enumerate(ETF_ORDER):
            if i + 1 < len(texts):
                val_str = texts[i + 1]
                try:
                    etf_values[etf_name] = parse_farside_value(val_str)
                except (ValueError, IndexError):
                    etf_values[etf_name] = 0.0
            else:
                etf_values[etf_name] = 0.0

        # Parse Total (last column)
        total_val = 0.0
        if len(texts) > len(ETF_ORDER) + 1:
            try:
                total_val = parse_farside_value(texts[len(ETF_ORDER) + 1])
            except ValueError:
                total_val = 0.0

        total_usd = int(round(total_val * 1_000_000))

        new_flows.append({
            "date": date_iso,
            "total_btc": None,
            "total_usd": total_usd,
            "etfs": etf_values
        })

    # Sort by date ascending
    new_flows.sort(key=lambda x: x["date"])

    print(f"Parsed {len(new_flows)} date rows from Farside")

    # --- Extract cumulative from chart data ---
    cumulative_usd = None
    cumulative_btc = None

    # Find the cumulative chart data points
    match = re.search(r'const dataPoints\s*=\s*\[([^\]]+)\]', html)
    if match:
        points_str = match.group(1)
        points = [float(p.strip()) for p in points_str.split(",") if p.strip()]
        if points:
            # Last data point is the cumulative total in US$m
            last_point = points[-1]
            cumulative_usd = int(round(last_point * 1_000_000))
            print(f"Cumulative USD (from chart): ${cumulative_usd:,}")

    # --- Read existing cache ---
    existing_flows = []
    existing_cumulative_btc = None
    existing_cumulative_usd = None

    try:
        with open(CACHE_PATH, 'r') as f:
            existing = json.load(f)
            existing_flows = existing.get("flows", [])
            existing_cumulative_btc = existing.get("cumulative_btc")
            existing_cumulative_usd = existing.get("cumulative_usd")
        print(f"Existing cache has {len(existing_flows)} flow entries")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing cache found, starting fresh")

    # --- Merge: new data overwrites old data for same date ---
    existing_by_date = {f["date"]: f for f in existing_flows}

    for new_f in new_flows:
        existing_by_date[new_f["date"]] = new_f

    merged_flows = sorted(existing_by_date.values(), key=lambda x: x["date"])
    print(f"Merged: {len(merged_flows)} total flow entries")

    # --- Preserve existing cumulative if not found on page ---
    if cumulative_usd is None:
        cumulative_usd = existing_cumulative_usd
    if cumulative_btc is None:
        cumulative_btc = existing_cumulative_btc

    # --- Get current time ---
    now = datetime.now(timezone.utc)
    last_update = now.strftime("%Y-%m-%dT%H:%M:%S")
    cached_at = time.time()

    # --- Build output ---
    output = {
        "flows": merged_flows,
        "cumulative_btc": cumulative_btc,
        "cumulative_usd": cumulative_usd,
        "last_update": last_update,
        "cached_at": cached_at
    }

    # --- Write ---
    with open(CACHE_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Written {len(merged_flows)} flows to {CACHE_PATH}")
    print(f"Last update: {last_update}")
    print(f"Cumulative USD: {cumulative_usd}")
    print(f"Cumulative BTC: {cumulative_btc}")

if __name__ == "__main__":
    main()
