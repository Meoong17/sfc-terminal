#!/usr/bin/env python3
"""Binance USDT-perp funding history -> data/binance_funding_daily.json

The BitMEX endpoint that fed the L1 "funding conditions" input has been returning
403 since 2026-09-16 (data/bitmex_funding_daily.json is frozen there), which is
part of why the structure engine could not advance past 2026-10-06. Binance's
public fundingRate endpoint is free, unauthenticated and covers 2019-09 onward.

Output shape (date-keyed, same convention as the other SFC series):
  {"2026-11-13": {"rate_mean": 0.00004, "rate_std": 0.00002, "n": 3}}

DISPLAY / ENGINE INPUT ONLY — provenance is recorded so a funding read that
starts in 2019 is never presented as the 2016+ series.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "binance_funding_daily.json"
BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"
START_MS = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_all():
    rows, start, page = [], START_MS, 0
    while True:
        r = requests.get(BASE, params={"symbol": SYMBOL, "startTime": start, "limit": 1000}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["fundingTime"]
        if len(batch) < 1000 or last <= start:
            break
        start = last + 1
        page += 1
        if page % 5 == 0:
            print(f"  fetched {len(rows)} funding prints...", file=sys.stderr)
    return rows


def main():
    rows = fetch_all()
    by_day = {}
    for row in rows:
        ts = datetime.fromtimestamp(row["fundingTime"] / 1000, tz=timezone.utc)
        day = ts.date().isoformat()
        by_day.setdefault(day, []).append(float(row["fundingRate"]))
    out = {}
    for day, rates in by_day.items():
        n = len(rates)
        mean = sum(rates) / n
        var = sum((x - mean) ** 2 for x in rates) / n
        out[day] = {"rate_mean": round(mean, 8), "rate_std": round(var ** 0.5, 8), "n": n}
    out = dict(sorted(out.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=0)
    days = sorted(out)
    print(f"[funding] {len(days)} days {days[0]} -> {days[-1]}", file=sys.stderr)
    print(json.dumps({"days": len(days), "first": days[0], "last": days[-1]}, indent=2))


if __name__ == "__main__":
    main()
