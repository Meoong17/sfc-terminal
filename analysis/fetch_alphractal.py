#!/usr/bin/env python3
"""
fetch_alphractal.py — pull Alphractal BTC daily on-chain/derivatives series.

Thin CLI wrapper over data_sources/alphractal.py. Output: data/alphractal_daily.json
(date-keyed, same convention as data/binance_vision_daily.json).

Plan/tier note: only the endpoints verified 200 on the current ak-... key are
fetched (see data_sources/alphractal.py WORKING_METRICS). Premium metrics (MVRV,
realized price, SOPR, exchange netflow, Puell, ...) return 403 on this plan and
are intentionally skipped. This is DATA COLLECTION only — nothing is blended into
an SFC score without walk-forward validation.

Usage:
    python3 analysis/fetch_alphractal.py              # fetch + save cache
    python3 analysis/fetch_alphractal.py --json       # print as JSON too
    python3 analysis/fetch_alphractal.py -v           # per-metric progress
"""

import os, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Alphractal BTC daily series")
    ap.add_argument("--json", action="store_true", help="also print fetched data as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from data_sources.alphractal import fetch_all, WORKING_METRICS, last_update

    data = fetch_all(verbose=args.verbose)
    n_days = len(data)
    out = os.path.join(REPO, "data", "alphractal_daily.json")
    print(f"wrote {out}")
    print(f"  {n_days} days  |  {len(WORKING_METRICS)} metrics  |  last update {last_update()}")
    if n_days:
        first = next(iter(data))
        last = next(reversed(data))
        print(f"  range: {first} .. {last}")
    if args.json:
        print(json.dumps(data))


if __name__ == "__main__":
    main()
