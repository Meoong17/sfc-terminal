#!/usr/bin/env python3
"""Extend the canonical Binance Vision daily series into the CURRENT month.

Binance Vision publishes COMPLETED months as monthly zips (the existing
fetch_binance_vision.py handles those, so daily.json runs to the end of the
previous month). The current month is only available as DAILY 1d files
(data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-YYYY-MM-DD.zip), published with
~2-3 day lag. This pulls those and merges them into daily.json so the series
is as current as the exchange has published.

NOTE on symbol: Binance spot trades BTCUSDT (there is no plain "BTCUSD" spot
pair on Binance Vision), so we use BTCUSDT — the same symbol the canonical
fetch uses.
"""
import os, json, io, zipfile, sys, datetime as dt

REPO = "/home/ubuntu/sfc"
DAILY_OUT = os.path.join(REPO, "data", "binance_vision_daily.json")
MONTHLY_OUT = os.path.join(REPO, "data", "binance_vision_monthly.json")
BASE = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"


def norm_ms(raw):
    # newer files use microseconds (1.78e15), older use ms (1.7e12)
    return raw // 1000 if raw > 10**14 else raw


def day_of(ms):
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def fetch_daily(date_str):
    """Download + parse one daily 1d kline zip. Returns dict or None."""
    url = f"{BASE}/spot/daily/klines/{SYMBOL}/1d/{SYMBOL}-1d-{date_str}.zip"
    try:
        raw = __import__("urllib.request", fromlist=["urlopen"]).urlopen(url, timeout=30).read()
    except Exception:
        return None
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            for line in z.read(name).decode().strip().splitlines():
                r = line.split(",")
                if len(r) < 11:
                    continue
                try:
                    o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                    vol, qv = float(r[5]), float(r[7])
                    tb, tq = float(r[9]), float(r[10])
                except (ValueError, IndexError):
                    continue
                out[day_of(norm_ms(int(r[0])))] = {
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": vol, "quote_vol": qv, "taker_base": tb, "taker_quote": tq,
                }
    return out


def main():
    daily = json.load(open(DAILY_OUT))
    today = dt.datetime.now(dt.timezone.utc)
    y, m = today.year, today.month
    # try days 1..today (backoff from today downward; Binance lags 2-3d)
    added, failed = [], []
    for d in range(today.day, 0, -1):
        date_str = f"{y:04d}-{m:02d}-{d:02d}"
        if date_str in daily:
            break  # already have this and everything before it
    start_d = max(1, today.day - 10)
    for d in range(start_d, today.day + 1):
        date_str = f"{y:04d}-{m:02d}-{d:02d}"
        if date_str in daily:
            continue
        rec = fetch_daily(date_str)
        if rec is None:
            failed.append(date_str)
            continue
        daily.update(rec)
        added.append(date_str)
    daily = {k: daily[k] for k in sorted(daily)}

    with open(DAILY_OUT, "w") as f:
        json.dump(daily, f)

    # refresh monthly aggregation for current month
    monthly = json.load(open(MONTHLY_OUT))
    ym = f"{y:04d}-{m:02d}"
    cur = {"open": None, "close": None, "volume": 0.0, "quote_vol": 0.0, "n": 0}
    for day, r in daily.items():
        if not day.startswith(ym):
            continue
        if cur["open"] is None:
            cur["open"] = r.get("open")
        cur["close"] = r.get("close")
        cur["volume"] += r.get("volume", 0.0)
        cur["quote_vol"] += r.get("quote_vol", 0.0)
        cur["n"] += 1
    monthly[ym] = cur
    with open(MONTHLY_OUT, "w") as f:
        json.dump(monthly, f, indent=0)

    ks = sorted(daily)
    print(f"added {len(added)} current-month day(s): {added}")
    print(f"failed (not yet published): {failed}")
    print(f"daily range now: {ks[0]} -> {ks[-1]} | n={len(ks)}")
    print(f"last close: {daily[ks[-1]].get('close')}")


if __name__ == "__main__":
    main()
