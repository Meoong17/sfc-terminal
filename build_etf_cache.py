#!/usr/bin/env python3
import json
import time
from datetime import datetime

# Raw data from Farside table (rows 3-15, skipping header rows)
# Columns: Date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total
raw_rows = [
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
    ["25 Jun 2026", "(265.7)", "(274.5)", "(7.1)", "(82.1)", "(53.0)", "(6.8)", "0.0", "(11.7)", "0.0", "9.2", "0.0", "0.0", "(691.7)"],
    ["26 Jun 2026", "(444.5)", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "(444.5)"],
]

ETF_KEYS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
}

def parse_value(val):
    """Parse Farside value: (440.3) -> -440.3, 53.95 -> 53.95"""
    val = val.strip()
    if val.startswith("(") and val.endswith(")"):
        return -float(val[1:-1].replace(",", ""))
    return float(val.replace(",", ""))

def parse_date(date_str):
    """Convert '09 Jun 2026' to '2026-06-09'"""
    parts = date_str.split()
    day = parts[0].zfill(2)
    month = MONTH_MAP[parts[1]]
    year = parts[2]
    return f"{year}-{month}-{day}"

# Build new flows
new_flows = []
for row in raw_rows:
    date_str = row[0]
    date_iso = parse_date(date_str)
    
    etf_values = {}
    total_val = None
    for i, key in enumerate(ETF_KEYS):
        val = parse_value(row[i + 1])
        etf_values[key] = val
    
    total_val = parse_value(row[13])  # Total column
    
    flow_entry = {
        "date": date_iso,
        "total_btc": None,
        "total_usd": int(total_val * 1_000_000),
        "etfs": etf_values
    }
    new_flows.append(flow_entry)

# Cumulative totals from Total row: 51,658 (in $m)
cumulative_usd = 51658 * 1_000_000  # $51.658B

# No BTC cumulative data on page
cumulative_btc = None

# Now check if existing cache exists and merge
try:
    with open("/home/ubuntu/sfc/.etf_cache.json", "r") as f:
        existing = json.load(f)
    existing_flows = existing.get("flows", [])
    existing_dates = {f["date"] for f in existing_flows}
    
    # Add new flows that don't already exist
    for nf in new_flows:
        if nf["date"] not in existing_dates:
            existing_flows.append(nf)
    
    # Sort by date
    existing_flows.sort(key=lambda x: x["date"])
    
    # Preserve existing cumulative values if the new ones are null
    if cumulative_btc is None and existing.get("cumulative_btc") is not None:
        cumulative_btc = existing["cumulative_btc"]
    if existing.get("cumulative_usd") is not None:
        # Update with latest if available
        pass
    
    flows = existing_flows
except FileNotFoundError:
    flows = new_flows

# Build final output
now = datetime.now()
output = {
    "flows": flows,
    "cumulative_btc": cumulative_btc,
    "cumulative_usd": cumulative_usd,
    "last_update": now.strftime("%Y-%m-%dT%H:%M:%S"),
    "cached_at": time.time()
}

with open("/home/ubuntu/sfc/.etf_cache.json", "w") as f:
    json.dump(output, f, indent=2)

print("Written successfully!")
print(f"Flows count: {len(flows)}")
print(f"cumulative_usd: {cumulative_usd}")
print(f"cumulative_btc: {cumulative_btc}")
print(f"last_update: {output['last_update']}")
print(f"cached_at: {output['cached_at']}")
