#!/usr/bin/env python3
"""Update .etf_cache.json with fresh Farside data.

Usage:
    python3 update_etf_cache.py [--btc-price PRICE]

If --btc-price is omitted, the script tries to read the current BTC price
from this repo's own data.json (the same file collect.py writes), falling
back to a manual prompt if that's unavailable. A price is required to
derive total_btc per day (see note below).
"""
import json, time, os, sys, argparse
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

def resolve_btc_price(cli_price, repo_dir):
    """
    Determine the BTC price to use for converting total_usd -> total_btc.

    total_btc was previously hardcoded to None for every entry, which
    silently broke compute_etf_metrics() downstream: it filters flows on
    `total_btc is not None`, so every single entry was excluded and the
    function always fell back to its neutral (0.5, 0.5) default — even
    when the cache had a full set of valid per-ETF USD flows. This wasn't
    visible from the log alone (the cache "updated successfully"), only
    from tracing why M81/M82 stayed neutral despite a populated cache.

    Order of resolution:
      1. --btc-price CLI argument, if given (most precise — use the price
         on the date this data is actually current for)
      2. This repo's own data.json "btc" field, if present and recent
      3. Hard failure with an explanit error rather than guessing,
         since an arbitrary fallback price would just reintroduce a
         silent-but-different version of the same bug.
    """
    if cli_price is not None:
        return cli_price

    data_json_path = os.path.join(repo_dir, "data.json")
    try:
        with open(data_json_path, "r") as f:
            d = json.load(f)
        price = d.get("btc")
        if price:
            print(f"Using BTC price from data.json: ${price:,.2f}")
            return float(price)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass

    print(
        "ERROR: No BTC price available to convert total_usd -> total_btc.\n"
        "Pass one explicitly: python3 update_etf_cache.py --btc-price 60000\n"
        "(Refusing to guess a fallback price — that would just reintroduce\n"
        "a different version of the total_btc=None bug this script used to have.)",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--btc-price", type=float, default=None,
                         help="BTC price to use for total_usd -> total_btc conversion")
    parser.add_argument("--cache-path", type=str, default=None,
                         help="Override the cache file path (default: alongside this script)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = args.cache_path or os.path.join(repo_dir, ".etf_cache.json")

    btc_price = resolve_btc_price(args.btc_price, repo_dir)

    # Load existing cache
    try:
        with open(cache_path, "r") as f:
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
        # Previously hardcoded to None — see resolve_btc_price() docstring
        # for why that silently broke every downstream consumer of this
        # cache. Now derived from total_usd at the resolved BTC price.
        total_btc = round(total_usd / btc_price, 4) if btc_price else None

        flow = {
            "date": iso_date,
            "total_btc": total_btc,
            "total_usd": total_usd,
            "etfs": etfs
        }
        existing_by_date[iso_date] = flow

    # Cumulative from Total row: 52,794 US$m
    cumulative_usd = 52794 * 1_000_000
    cumulative_btc = round(cumulative_usd / btc_price, 4) if btc_price else None

    # Build final flows list sorted by date
    all_dates = sorted(existing_by_date.keys())
    flows = [existing_by_date[d] for d in all_dates]

    now = datetime.now()
    cache["flows"] = flows
    cache["cumulative_btc"] = cumulative_btc
    cache["cumulative_usd"] = cumulative_usd
    cache["last_update"] = now.strftime("%Y-%m-%dT%H:%M:%S")
    # NOTE: cached_at still reflects "when this script ran", not "how
    # current the underlying Farside data actually is" — this script's
    # raw_rows are still a manually maintained snapshot, not a live
    # scrape (the original docstring's "auto-updated every 6h by cron"
    # claim was not backed by any actual scheduler in this repo at audit
    # time). cached_at staleness checks downstream can only tell you
    # "this script was run recently", not "the Farside data is fresh" —
    # keep raw_rows updated manually until a real scraper/cron exists.
    cache["cached_at"] = time.time()

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Updated cache: {len(flows)} flow entries")
    print(f"Cumulative USD: ${cumulative_usd:,}")
    print(f"Cumulative BTC: {cumulative_btc:,.2f}" if cumulative_btc else "Cumulative BTC: N/A")
    print(f"Last update: {cache['last_update']}")
    print(f"Cached at: {cache['cached_at']}")


if __name__ == "__main__":
    main()
