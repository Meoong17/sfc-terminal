#!/usr/bin/env python3
"""Parse Farside ETF data and update the cache file."""
import json
import time
import re
from datetime import datetime

# Read existing cache
try:
    with open('/home/ubuntu/sfc/.etf_cache.json', 'r') as f:
        cache = json.load(f)
    print(f"Existing cache: {len(cache['flows'])} flow entries")
    existing_dates = {f['date'] for f in cache['flows']}
    print(f"Existing dates: {sorted(existing_dates)}")
except Exception as e:
    print(f"Error reading cache: {e}")
    cache = {"flows": [], "cumulative_btc": 0, "cumulative_usd": 0, "last_update": "", "cached_at": 0}
    existing_dates = set()

# Parse Farside all-data page
# Data from the scraped page (rows 657-679 of all-data page)
raw_data = [
    # date, IBIT, FBTC, BITB, ARKB, BTCO, EZBC, BRRR, HODL, BTCW, MSBT, GBTC, BTC, Total
    ("2026-06-01", -440.3, -37.3, 0.0, -12.3, 0.0, 0.0, 0.0, 0.0, 0.0, 6.1, 0.0, 0.0, -483.8),
    ("2026-06-02", -388.6, -45.1, 0.0, -16.7, 0.0, 0.0, 0.0, 0.0, 0.0, 14.8, -83.5, 0.0, -519.1),
    ("2026-06-03", -342.3, -54.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -396.6),
    ("2026-06-04", 47.7, -5.5, -15.6, -20.7, -12.6, 0.0, 0.0, 0.0, 0.0, 9.9, 0.0, 0.0, 3.2),
    ("2026-06-05", -213.7, -59.7, 0.0, 0.0, 0.0, 0.0, 0.0, 4.2, 0.0, 4.3, -60.8, 0.0, -325.7),
    ("2026-06-08", -232.9, 59.4, 14.1, 63.1, 0.0, 0.0, 0.0, 0.0, 0.0, 4.9, 0.0, 0.0, -91.4),
    ("2026-06-09", -61.6, -20.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.4, -77.4),
    ("2026-06-10", -148.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -87.9, 17.5, -213.9),
    ("2026-06-11", 30.3, -5.5, -13.1, -27.2, 0.0, 0.0, 0.0, -14.8, 0.0, 2.2, 0.0, 5.6, -22.5),
    ("2026-06-12", 57.7, 18.0, 5.2, 3.2, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.0, 85.9),
    ("2026-06-15", 66.4, -8.7, 0.0, -6.6, 0.0, -5.8, 0.0, -6.1, 0.0, 9.4, -124.0, 10.6, -64.8),
    ("2026-06-16", 16.4, 4.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.9, -16.8, 4.4, 10.2),
    ("2026-06-17", -30.8, 14.0, 0.0, -43.5, -6.4, 0.0, 0.0, -4.1, 0.0, 4.1, -15.5, 0.0, -82.2),
    ("2026-06-18", -96.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.4, 0.0, 10.4, 0.0, 0.0, -90.7),
    ("2026-06-22", -172.0, 57.4, 0.0, 64.0, 0.0, 3.7, 0.0, 0.0, 3.4, 8.1, -81.0, 48.1, -68.3),
    ("2026-06-23", -182.0, 23.0, 0.0, 31.0, 0.0, 0.0, 0.0, 5.3, 0.0, 8.9, 0.0, 0.0, -113.8),
    ("2026-06-24", -239.3, -120.8, -27.5, -50.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -54.3, 23.6, -469.0),
    ("2026-06-25", -265.7, -274.5, -7.1, -82.1, -53.0, -6.8, 0.0, -11.7, 0.0, 9.2, 0.0, 0.0, -691.7),
    ("2026-06-26", -444.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -444.5),
    ("2026-06-29", -300.4, -3.9, 0.0, 50.0, 0.0, 0.0, 0.0, 3.8, 0.0, 7.3, 35.1, -22.9, -231.0),
    ("2026-06-30", -212.4, -10.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -222.6),
    ("2026-07-01", -219.4, -51.0, 0.0, -39.9, 5.4, 3.5, 0.0, 2.1, 0.0, 29.8, -62.8, 36.3, -296.0),
    ("2026-07-02", -40.4, 166.0, 0.0, 91.8, 0.0, 0.0, 1.7, 4.4, 0.0, 0.0, 0.0, 0.0, 223.5),
]

# Build lookup of existing flows by date
existing_flow_map = {f['date']: f for f in cache['flows']}

# Merge data
updated_flows = []
new_dates_added = 0
existing_updated = 0

for row in raw_data:
    date = row[0]
    ibit, fbtc, bitb, arkb, btco, ezbc, brrr, hodl, btcw, msbt, gbtc, btc, total = row[1:]
    
    flow_entry = {
        "date": date,
        "total_btc": None,  # Farside provides USD, not BTC
        "total_usd": int(total * 1_000_000),
        "etfs": {
            "IBIT": ibit,
            "FBTC": fbtc,
            "BITB": bitb,
            "ARKB": arkb,
            "BTCO": btco,
            "EZBC": ezbc,
            "BRRR": brrr,
            "HODL": hodl,
            "BTCW": btcw,
            "MSBT": msbt,
            "GBTC": gbtc,
            "BTC": btc
        }
    }
    
    if date in existing_flow_map:
        old = existing_flow_map[date]
        # Check if data changed
        if old['total_usd'] != flow_entry['total_usd']:
            print(f"UPDATED {date}: total_usd {old['total_usd']} -> {flow_entry['total_usd']}")
            existing_updated += 1
        updated_flows.append(flow_entry)
        del existing_flow_map[date]
    else:
        print(f"NEW {date}: total_usd={flow_entry['total_usd']}")
        new_dates_added += 1
        updated_flows.append(flow_entry)

# Add any remaining cached flows (not in Farside data but might exist)
for date, flow in sorted(existing_flow_map.items()):
    print(f"KEEP (from cache, not in Farside): {date}")
    updated_flows.append(flow)

# Sort by date
updated_flows.sort(key=lambda x: x['date'])

# Keep existing cumulative values (no explicit cumulative totals found on page)
# Update timestamps
now = time.time()
now_dt = datetime.fromtimestamp(now)
last_update_str = now_dt.strftime("%Y-%m-%dT%H:%M:%S")

new_cache = {
    "flows": updated_flows,
    "cumulative_btc": cache.get("cumulative_btc", 0),
    "cumulative_usd": cache.get("cumulative_usd", 0),
    "last_update": last_update_str,
    "cached_at": now
}

# Validate
for f in new_cache['flows']:
    # Verify total_usd matches sum of etfs * 1M
    etf_sum_m = sum(f['etfs'].values())
    expected_usd = int(etf_sum_m * 1_000_000)
    if abs(f['total_usd'] - expected_usd) > 1000:  # allow small rounding diff
        print(f"WARNING {f['date']}: total_usd={f['total_usd']} but etf sum={etf_sum_m} -> expected={expected_usd}")

with open('/home/ubuntu/sfc/.etf_cache.json', 'w') as f:
    json.dump(new_cache, f, indent=2)

print(f"\nDone! Wrote {len(updated_flows)} flow entries")
print(f"New dates added: {new_dates_added}")
print(f"Existing updated: {existing_updated}")
print(f"Last update: {last_update_str}")
print(f"Cached at: {now}")
