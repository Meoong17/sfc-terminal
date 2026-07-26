#!/usr/bin/env python3
"""
Complete ETF cache updater.
Scrapes https://farside.co.uk/btc/, merges with existing cache, writes updated file.
"""

import json
import os
import re
import sys
import time
from datetime import datetime

from playwright.sync_api import sync_playwright

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".etf_cache.json")

ETF_COLUMNS = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "MSBT", "GBTC", "BTC"]

def parse_num(val):
    """Parse number string, handling parentheses for negatives and commas."""
    val = val.strip().replace(",", "").replace("\u00a0", "").replace(" ", "")
    if not val or val in ("—", "-", ""):
        return 0.0
    if val.startswith("(") and val.endswith(")"):
        return -float(val[1:-1])
    return float(val)

def parse_date(d):
    """Convert '07 Jul 2026' -> '2026-07-07'."""
    try:
        dt = datetime.strptime(d.strip(), "%d %b %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return d.strip()

def scrape_farside():
    """Scrape the Farside page and return flows list + cumulative data."""
    result = {
        "flows": [],
        "cumulative_btc": None,
        "cumulative_usd": None,
        "last_update": None,
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = context.new_page()
        
        try:
            page.goto("https://farside.co.uk/btc/", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(5000)
            
            # Scroll to ensure all table rows are loaded
            for _ in range(15):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(300)
            
            # Get HTML for chart data extraction
            html_content = page.content()
            
            # === PARSE MAIN TABLE ===
            tables = page.query_selector_all("table")
            data_rows = []
            cumulative_usd = None
            
            for table in tables:
                rows = table.query_selector_all("tr")
                for row in rows:
                    cells = row.query_selector_all("td, th")
                    cell_texts = [cell.inner_text().strip() for cell in cells]
                    
                    if len(cell_texts) < 14:
                        continue
                    
                    first_cell = cell_texts[0]
                    
                    # Total row -> cumulative_usd (in US$m)
                    if first_cell == "Total":
                        total_cumulative = parse_num(cell_texts[13])
                        cumulative_usd = int(total_cumulative * 1_000_000)
                        continue
                    
                    # Skip non-data rows
                    if first_cell in ("Fee", "Average", "Maximum", "Minimum", "", "Date"):
                        continue
                    
                    # Only parse actual date rows
                    date_match = re.match(r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', first_cell)
                    if not date_match:
                        continue
                    
                    date_str = parse_date(first_cell)
                    etf_values = {}
                    
                    for ci, col in enumerate(ETF_COLUMNS):
                        if ci + 1 < len(cell_texts):
                            etf_values[col] = parse_num(cell_texts[ci + 1])
                    
                    total_val = parse_num(cell_texts[13])
                    
                    data_rows.append({
                        "date": date_str,
                        "total_btc": None,
                        "total_usd": int(total_val * 1_000_000),
                        "etfs": etf_values
                    })
            
            result["flows"] = data_rows
            result["cumulative_usd"] = cumulative_usd
            
            # === EXTRACT CUMULATIVE FROM CHART DATA as fallback ===
            if cumulative_usd is None:
                chart_match = re.search(r'data:\s*dataPoints\s*;\s*\n\s*const\s+labels', html_content)
                # Try to find the dataPoints array
                dp_match = re.search(r'dataPoints\s*=\s*\[([^\]]+)\]', html_content)
                if dp_match:
                    vals = [v.strip() for v in dp_match.group(1).split(",") if v.strip()]
                    if vals:
                        last_val = float(vals[-1])
                        result["cumulative_usd"] = int(last_val * 1_000_000)
            
            # Set last_update
            now = datetime.now()
            result["last_update"] = now.strftime("%Y-%m-%dT%H:%M:%S")
            
            print(f"Scraped {len(data_rows)} rows, cum_usd={result['cumulative_usd']}", file=sys.stderr)
            
        except Exception as e:
            print(f"Scrape error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            browser.close()
    
    return result

def load_existing_cache():
    """Load existing cache file, return dict or None."""
    if not os.path.exists(CACHE_PATH):
        print("No existing cache found", file=sys.stderr)
        return None
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print(f"Loaded existing cache: {len(data.get('flows', []))} rows", file=sys.stderr)
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading cache: {e}", file=sys.stderr)
        return None

def merge_flows(existing_flows, new_flows):
    """Merge new flows into existing flows, keyed by date."""
    flow_map = {}
    
    # Add existing flows
    for flow in existing_flows:
        flow_map[flow["date"]] = flow
    
    # Add/update with new flows
    for flow in new_flows:
        flow_map[flow["date"]] = flow
    
    # Return sorted by date (ascending)
    return sorted(flow_map.values(), key=lambda x: x["date"])

def main():
    print("=== ETF Cache Updater ===", file=sys.stderr)
    
    # 1. Scrape fresh data
    fresh = scrape_farside()
    
    if not fresh["flows"]:
        print("ERROR: No data scraped! Aborting.", file=sys.stderr)
        sys.exit(1)
    
    # 2. Load existing cache
    existing = load_existing_cache()
    
    # 3. Merge
    if existing and "flows" in existing:
        merged_flows = merge_flows(existing["flows"], fresh["flows"])
    else:
        merged_flows = fresh["flows"]
    
    # 4. Build final cache object
    cache = {
        "flows": merged_flows,
        "cumulative_btc": fresh.get("cumulative_btc") or (existing.get("cumulative_btc") if existing else None),
        "cumulative_usd": fresh.get("cumulative_usd") or (existing.get("cumulative_usd") if existing else None),
        "last_update": fresh["last_update"],
        "cached_at": time.time(),
    }
    
    # 5. Write cache
    print(f"Writing {len(merged_flows)} flows to {CACHE_PATH}", file=sys.stderr)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    
    # 6. Verify
    with open(CACHE_PATH) as f:
        verify = json.load(f)
    
    print(f"Verified: {len(verify['flows'])} flows, cached_at={verify['cached_at']}", file=sys.stderr)
    
    # Report summary
    latest_flow = merged_flows[-1] if merged_flows else None
    print(f"\nLatest data: {latest_flow['date'] if latest_flow else 'N/A'}", file=sys.stderr)
    print(f"Cumulative USD: ${cache['cumulative_usd']:,}" if cache['cumulative_usd'] else "Cumulative USD: N/A", file=sys.stderr)
    print(f"Cumulative BTC: {cache['cumulative_btc']}" if cache['cumulative_btc'] else "Cumulative BTC: N/A", file=sys.stderr)
    
    # Output result as JSON for verification
    print(json.dumps({
        "status": "success",
        "total_flows": len(merged_flows),
        "new_flows": len(fresh["flows"]),
        "latest_date": latest_flow["date"] if latest_flow else None,
        "cumulative_usd": cache["cumulative_usd"],
        "cumulative_btc": cache["cumulative_btc"],
        "cached_at": cache["cached_at"],
    }, indent=2))

if __name__ == "__main__":
    main()
