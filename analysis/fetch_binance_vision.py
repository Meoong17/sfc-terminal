#!/usr/bin/env python3
"""
Binance Vision multi-timeframe fetcher — canonical historical market data for SFC.

Downloads (public S3 bucket, no key):
  spot klines       1d, 1w   : data/spot/monthly/klines/BTCUSDT/{1d,1w}/  (2017-08+)
  futures funding   8h       : data/futures/um/monthly/fundingRate/BTCUSDT/ (2020-01+)
  futures klines    1d       : markPriceKlines, indexPriceKlines,
                               premiumIndexKlines (2020-01+)

Verifies SHA256 checksums (parallel download). Caches raw zips + consolidated JSON:
  data/binance_vision_cache/<dataset>/<file>.zip
  data/binance_vision_daily.json     date -> {open,high,low,close,volume,quote_vol,
                                              taker_base,taker_quote,funding_last,
                                              funding_mean,index,mark,premium}
  data/binance_vision_weekly.json    iso_week -> {open,high,low,close,volume,quote_vol}
  data/binance_vision_monthly.json   yyyy-mm  -> aggregated from daily

Rationale: canonical single price/funding series so all walk-forward validations use
one clean 9-year series (not 2-month SFC snapshots). Monthly candles aggregated from
daily (no 1M path). Full-resolution trades skipped (kline volume already gives
base/quote/taker volume — sufficient for Almgren-Chriss slippage).
"""
import os, io, sys, json, zipfile, hashlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import datetime as _dt

BASE = "https://data.binance.vision/data"
REPO = "/home/ubuntu/sfc"
CACHE_DIR = os.path.join(REPO, "data", "binance_vision_cache")
DAILY_OUT = os.path.join(REPO, "data", "binance_vision_daily.json")
WEEKLY_OUT = os.path.join(REPO, "data", "binance_vision_weekly.json")
MONTHLY_OUT = os.path.join(REPO, "data", "binance_vision_monthly.json")

SYMBOL = "BTCUSDT"
N_WORKERS = 8

# (dataset, subdir, start_ym, kind) — kind: 'kline' or 'fund' or 'kline' (mark/index/premium use same parse)
DATASETS = [
    ("spot_klines_1d",    "spot/monthly/klines/BTCUSDT/1d",  "2017-08", "kline"),
    ("spot_klines_1w",    "spot/monthly/klines/BTCUSDT/1w",  "2017-08", "kline"),
    ("fut_funding",       "futures/um/monthly/fundingRate/BTCUSDT", "2020-01", "fund"),
    ("fut_mark_1d",       "futures/um/monthly/markPriceKlines/BTCUSDT/1d", "2020-01", "kline"),
    ("fut_index_1d",      "futures/um/monthly/indexPriceKlines/BTCUSDT/1d", "2020-01", "kline"),
    ("fut_premium_1d",    "futures/um/monthly/premiumIndexKlines/BTCUSDT/1d", "2020-01", "kline"),
]


def months(start, end):
    out = []
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1; y += 1
    return out


def last_complete_month():
    today = _dt.datetime.now(_dt.timezone.utc)
    y, m = today.year, today.month
    m -= 1
    if m == 0:
        m = 12; y -= 1
    return f"{y:04d}-{m:02d}"


def _norm_ms(raw):
    # newer files use microseconds (1.78e15), older use ms (1.7e12)
    if raw > 10**14:
        return raw // 1000
    return raw


def _to_day(ms):
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _to_week(ms):
    d = _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).date()
    return d.isocalendar()  # (year, week, weekday)


def url_for(dsname, ym):
    _, subdir, _, kind = _DS_BY_NAME[dsname]
    base = f"{BASE}/{subdir}"
    if kind == "fund":
        return f"{base}/{SYMBOL}-fundingRate-{ym}.zip"
    # All kline datasets (spot 1d/1w AND futures mark/index/premium 1d) name their
    # files by INTERVAL: BTCUSDT-1d-YYYY-MM.zip / BTCUSDT-1w-YYYY-MM.zip.
    prefix = "1w" if "_1w" in dsname else "1d"
    return f"{base}/{SYMBOL}-{prefix}-{ym}.zip"


_DS_BY_NAME = {d[0]: d for d in DATASETS}


def checksum_url(url):
    return url + ".CHECKSUM"


def fetch_with_check(dsname, ym):
    """Download zip + CHECKSUM, verify SHA256. Returns path or None on failure/404."""
    ddir = os.path.join(CACHE_DIR, dsname)
    os.makedirs(ddir, exist_ok=True)
    fname = os.path.basename(url_for(dsname, ym))
    dest = os.path.join(ddir, fname)
    # skip if already cached (assume verified previously)
    if os.path.exists(dest):
        return dest
    url = url_for(dsname, ym)
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None
    # checksum
    try:
        ck = urllib.request.urlopen(checksum_url(url), timeout=20).read().decode().strip().split()[0]
        if hashlib.sha256(data).hexdigest() != ck:
            print(f"  CHECKSUM MISMATCH {fname}", file=sys.stderr)
            return None
    except Exception:
        pass  # checksum optional if unavailable
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def parse_zip(path, kind):
    rows = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            for line in z.read(name).decode().strip().splitlines():
                rows.append(line.split(","))
    if kind == "fund":
        byday = defaultdict(list)
        for r in rows:
            if len(r) < 3:
                continue
            try:
                ms = _norm_ms(int(r[0])); fr = float(r[2])
            except (ValueError, IndexError):
                continue
            byday[_to_day(ms)].append(fr)
        return {"fund": {d: {"last": v[-1], "mean": sum(v)/len(v)} for d, v in byday.items()}}
    # kline: 12 cols open_time,o,h,l,c,vol,ct,qvol,n,tb,tq,ignore
    out = {}
    for r in rows:
        if len(r) < 11:
            continue
        try:
            ms = _norm_ms(int(r[0]))
            o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            vol, qv = float(r[5]), float(r[7])
            tb, tq = float(r[9]), float(r[10])
        except (ValueError, IndexError):
            continue
        key = _to_day(ms)
        out[key] = {"open": o, "high": h, "low": l, "close": c,
                    "volume": vol, "quote_vol": qv, "taker_base": tb, "taker_quote": tq}
    return {"kline": out}


def main():
    start_year, start_month = 2017, 8
    end = os.environ.get("SFC_END_MONTH") or last_complete_month()
    tasks = []
    for ds in DATASETS:
        dsname, _, start, _ = ds
        for ym in months(start, end):
            tasks.append((dsname, ym))
    print(f"downloading {len(tasks)} monthly zips across {len(DATASETS)} datasets "
          f"(threads={N_WORKERS})...")

    results = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_with_check, ds, ym): (ds, ym) for ds, ym in tasks}
        done = 0
        for fut in as_completed(futs):
            ds, ym = futs[fut]
            path = fut.result()
            results[(ds, ym)] = path
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} done")
    n404 = sum(1 for p in results.values() if p is None)
    print(f"download complete: {len(results)-n404} ok, {n404} not-found(404)")

    # consolidate
    daily = {}
    weekly = {}
    # futures klines store a single derived value (premium/index/mark close)
    FUT_FIELD = {"fut_premium_1d": "premium", "fut_index_1d": "index", "fut_mark_1d": "mark"}
    for (dsname, ym), path in results.items():
        if not path:
            continue
        if "fund" in dsname:
            recs = parse_zip(path, "fund")["fund"]
            for day, v in recs.items():
                d = daily.setdefault(day, {})
                d["funding_last"] = v["last"]
                d["funding_mean"] = v["mean"]
            continue
        recs = parse_zip(path, "kline")["kline"]
        if dsname in FUT_FIELD:
            # only store the derived single value; never touch spot OHLC
            field = FUT_FIELD[dsname]
            for day, v in recs.items():
                daily.setdefault(day, {})[field] = v["close"]
        elif dsname == "spot_klines_1w":
            for day, v in recs.items():
                for k, val in v.items():
                    weekly.setdefault(day, {})[k] = val
        else:  # spot_klines_1d
            for day, v in recs.items():
                for k, val in v.items():
                    daily.setdefault(day, {})[k] = val

    # sort daily
    daily_sorted = {d: daily[d] for d in sorted(daily)}
    weekly_sorted = {d: weekly[d] for d in sorted(weekly)}

    os.makedirs(os.path.dirname(DAILY_OUT), exist_ok=True)
    with open(DAILY_OUT, "w") as f:
        json.dump(daily_sorted, f)
    with open(WEEKLY_OUT, "w") as f:
        json.dump(weekly_sorted, f)

    # monthly aggregated from daily close
    monthly = {}
    for day, r in daily_sorted.items():
        ym = day[:7]
        m = monthly.setdefault(ym, {"open": None, "close": None, "volume": 0.0,
                                    "quote_vol": 0.0, "n": 0})
        if m["open"] is None:
            m["open"] = r.get("open")
        m["close"] = r.get("close")
        m["volume"] += r.get("volume", 0.0)
        m["quote_vol"] += r.get("quote_vol", 0.0)
        m["n"] += 1
    with open(MONTHLY_OUT, "w") as f:
        json.dump(monthly, f, indent=0)

    # sanity vs live SFC
    dmin, dmax = min(daily), max(daily)
    print(f"\ndaily  : {len(daily)} days  {dmin} -> {dmax}")
    print(f"weekly : {len(weekly)} weeks {min(weekly)} -> {max(weekly)}")
    print(f"monthly: {len(monthly)} months {min(monthly)} -> {max(monthly)}")
    # funding coverage
    nfund = sum(1 for d in daily.values() if "funding_last" in d)
    print(f"funding days: {nfund}")
    # realized vol sample
    try:
        sfc = json.load(open(os.path.join(REPO, "data.json")))
        print(f"\nsanity: last cached close={daily[max(daily)].get('close')} "
              f"| live SFC btc={sfc.get('btc')} | premium@last={daily[max(daily)].get('premium')}")
    except Exception as e:
        print("sanity skip:", e)


if __name__ == "__main__":
    main()
