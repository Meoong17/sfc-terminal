#!/usr/bin/env python3
"""Update .etf_cache.json with fresh Farside data."""
import json, time
from datetime import datetime

# Raw data from Farside table (date rows only)
# Format: [date_str, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total]
raw_rows = [
    ["08 Jun 2026", "(232.9)", "59.4", "14.1", "63.1", "0.0", "0.0", "0.0", "0.0", "0.0", "4.9", "0.0", "0.0", "(91.4)"],
    ["09 Jun 2026", "(61.6)", "(20.2)", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "4.4", "(77.4)"],
    ["10 Jun 2026", "(148.5)", "4.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "1.0", "0.0", "(87.9)", "17.5", "(213.9)"],
    ["11 Jun 2026", "30.3", "(5.5)", "(13.1)", "(27.2)", "0.0", "0.0", "0.0", "(14.8)", "0.0", "2.2", "0.0", "5.6", "(22.5)"],
    ["12 Jun 2026", "57.7", "18.0", "5.2", "3.2", "0.0", "0.0", "0.0", "1.8", "0.0", "0.0", "0.0", "0.0", "85.9"],
    ["15 Jun 2026", "66.4", "(8.7)", "0.0", "(6.6)", "0.0", "(5.8)", "0.0", "(6.1)", "0.0", "9.4", "(124.0)", "10.6", "(64.8)"],
    ["16 Jun 2026", "16.4", "4.3", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "1.9", "(16.8)", "4.4", "10.2"],
    ["17 Jun 2026", "(30.8)", "14.0", "0.0", "(43.5)", "(6.4)", "0.0", "0.0", "(4.1)", "0.0", "4.1", "(15.5)", "0.0", "(82.2)"],
    ["18 Jun 2026", "(96.7)", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "(4.4)", "0.0", "10.4", "0.0", "0.0", "(90.7)"],
    ["22 Jun 2026", "(172.0)", "57.4", "0.0", "64.0", "0.0", "3.7", "0.0", "0.0", "3.4", "8.1", "(81.0)", "48.1", "(68.3)"],
    ["23 Jun 2026", "(182.0)", "23.0", "0.0", "31.0", "0.0", "0.0", "0.0", "5.3", "0.0", "8.9", "0.0", "0.0", "(113.8)"],
    ["24 Jun 2026", "(239.3)", "(120.8)", "(27.5)", "(50.7)", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "(54.3)", "23.6", "(469.0)"],
]

etf_keys = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]

def parse_val(s):
    """Parse Farside value: '(232.9)' -> -232.9, '59.4' -> 59.4"""
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        return -float(s[1:-1])
    return float(s)

def parse_date(s):
    """Convert '08 Jun 2026' to '2026-06-08'"""
    months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
              "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    parts = s.split()
    day = parts[0].zfill(2)
    month = months[parts[1]]
    year = parts[2]
    return f"{year}-{month}-{day}"

# Load existing cache
try:
    with open("/home/ubuntu/sfc/.etf_cache.json", "r") as f:
        cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cache = {"flows": [], "cumulative_btc": None, "cumulative_usd": None, "last_update": None, "cached_at": None}

# Build index of existing flows by date
existing_by_date = {f["date"]: f for f in cache.get("flows", [])}

# Create/update flows from scraped data
for row in raw_rows:
    date_str = row[0]
    iso_date = parse_date(date_str)
    
    etfs = {}
    for i, key in enumerate(etf_keys):
        etfs[key] = parse_val(row[i + 1])
    
    total_val = parse_val(row[13])
    total_usd = int(total_val * 1_000_000)
    
    flow = {
        "date": iso_date,
        "total_btc": None,
        "total_usd": total_usd,
        "etfs": etfs
    }
    existing_by_date[iso_date] = flow

# Cumulative from Total row: 52,794 US$m
cumulative_usd = 52794 * 1_000_000

# Build final flows list sorted by date
all_dates = sorted(existing_by_date.keys())
flows = [existing_by_date[d] for d in all_dates]

now = datetime.now()
cache["flows"] = flows
cache["cumulative_btc"] = None  # Farside doesn't show BTC cumulative
cache["cumulative_usd"] = cumulative_usd
cache["last_update"] = now.strftime("%Y-%m-%dT%H:%M:%S")
cache["cached_at"] = time.time()

with open("/home/ubuntu/sfc/.etf_cache.json", "w") as f:
    json.dump(cache, f, indent=2)

print(f"Updated cache: {len(flows)} flow entries")
print(f"Cumulative USD: ${cumulative_usd:,}")
print(f"Last update: {cache['last_update']}")
print(f"Cached at: {cache['cached_at']}")
