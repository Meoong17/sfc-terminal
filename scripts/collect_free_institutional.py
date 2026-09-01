#!/usr/bin/env python3
"""
collect_free_institutional.py — accumulate daily history for institutional-grade
    FREE behaviour/participation signals that were previously unavailable (DATA_TOO_SHORT):
      1. Binance futures OPEN INTEREST (historical backfill + daily append)
      2. Deribit OPTIONS (OI, put/call, mark-IV)
      3. CoinMetrics community ON-CHAIN (active addresses)

Writes (point-in-time safe, append-only):
    data/binance_oi_daily.json        {date: {sum_oi, sum_oi_value}}
    data/deribit_options_daily.json   {date: {oi_btc, oi_call, oi_put, mark_iv, put_call_oi_ratio, n_calls, n_puts}}
    data/coinmetrics_btc_daily.json   {date: {AdrActCnt}}

Run daily (or more often); merges so history accumulates. Idempotent.
"""
import json, os, sys, time
from pathlib import Path
import requests

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OIF = DATA / "binance_oi_daily.json"
OPTF = DATA / "deribit_options_daily.json"
CMF = DATA / "coinmetrics_btc_daily.json"
OKXF = DATA / "okx_oi_daily.json"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) SFC/1.0"}
T = 30
ONCHAIN_METRICS = "AdrActCnt,CapMVRVCur,TxCnt,SplyCur,AdrBalCnt,HashRate"

def load(path):
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}

def save(path, obj):
    tmp = path.with_suffix(".tmp")
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)

def fetch(url, params=None, headers=None):
    r = requests.get(url, params=params, headers=headers or UA, timeout=T)
    r.raise_for_status()
    return r.json()

def collect_oi(out):
    """Backfill/refresh Binance futures OI history + today's snapshot (append daily)."""
    try:
        rows = fetch("https://fapi.binance.com/futures/data/openInterestHist",
                     {"symbol": "BTCUSDT", "period": "1d", "limit": 500})
        n = 0
        for r in rows:
            ts = r["timestamp"]
            d = (str(ts)[:10] if isinstance(ts, str)
                 else time.strftime("%Y-%m-%d", time.gmtime(ts / 1000.0)))
            out[d] = {"sum_oi": float(r["sumOpenInterest"]),
                      "sum_oi_value": float(r["sumOpenInterestValue"])}
            n += 1
        # today's live snapshot (guarantees today present even if hist lags)
        cur = fetch("https://fapi.binance.com/fapi/v1/openInterest", {"symbol": "BTCUSDT"})
        today = time.strftime("%Y-%m-%d")
        out.setdefault(today, {})
        out[today]["sum_oi"] = float(cur["openInterest"])
        out[today]["sum_oi_value"] = out[today].get("sum_oi_value", 0)
        print(f"[OI] rows={n} total_days={len(out)} last={sorted(out)[-1]}")
        return True
    except Exception as e:
        print(f"[OI] ERR {type(e).__name__}: {str(e)[:120]}")
        return False

def collect_options(out):
    """Deribit BTC options aggregate (OI, put/call, mark IV)."""
    try:
        d = fetch("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                  {"currency": "BTC", "kind": "option"})
        recs = d.get("result", [])
        oi_call = oi_put = 0.0
        iv_sum = 0.0; iv_n = 0
        n_c = n_p = 0
        for r in recs:
            oi = float(r.get("open_interest", 0) or 0)
            if r.get("instrument_name", "").endswith("-C"):
                oi_call += oi; n_c += 1
            else:
                oi_put += oi; n_p += 1
            iv = r.get("mark_iv")
            if iv is not None:
                iv_sum += float(iv); iv_n += 1
        today = time.strftime("%Y-%m-%d")
        out[today] = {
            "oi_btc": oi_call + oi_put,
            "oi_call": oi_call, "oi_put": oi_put,
            "put_call_oi_ratio": round((oi_put / oi_call), 4) if oi_call > 0 else None,
            "mark_iv": round(iv_sum / iv_n, 4) if iv_n else None,
            "n_calls": n_c, "n_puts": n_p,
        }
        print(f"[OPT] today OI={oi_call+oi_put:.0f} pc={out[today]['put_call_oi_ratio']} "
              f"iv={out[today]['mark_iv']} n={len(recs)}")
        return True
    except Exception as e:
        print(f"[OPT] ERR {type(e).__name__}: {str(e)[:120]}")
        return False

def collect_onchain(out):
    """CoinMetrics community on-chain behaviour (active addr, MVRV, tx, supply,
    addr-with-balance, hashrate) — free tier; backfill ~400d + append."""
    try:
        d = fetch("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                  {"assets": "btc", "metrics": ONCHAIN_METRICS,
                   "frequency": "1d", "page_size": 10000,
                   "start_time": time.strftime("%Y-%m-%d", time.localtime(time.time() - 400 * 86400))})
        n = 0
        for row in d.get("data", []):
            rec = {k: (float(v) if v not in (None, "") else None)
                   for k, v in row.items() if k not in ("time", "asset")}
            out[row["time"][:10]] = rec
            n += 1
        print(f"[ONC] metrics={ONCHAIN_METRICS} rows={n} total_days={len(out)} last={sorted(out)[-1]}")
        return True
    except Exception as e:
        print(f"[ONC] ERR {type(e).__name__}: {str(e)[:120]}")
        return False

def collect_oi_okx(out):
    """OKX BTC-SWAP open interest snapshot (second exchange, append daily)."""
    try:
        d = fetch("https://www.okx.com/api/v5/public/open-interest",
                  {"instType": "SWAP", "instId": "BTC-USDT-SWAP"})
        row = d.get("data", [])[0]
        today = time.strftime("%Y-%m-%d")
        out[today] = {"oi_btc": float(row["oiCcy"]), "oi_usd": float(row["oiUsd"])}
        print(f"[OKX] today OI={out[today]['oi_btc']:.0f} BTC ({out[today]['oi_usd']/1e9:.2f}B usd)")
        return True
    except Exception as e:
        print(f"[OKX] ERR {type(e).__name__}: {str(e)[:120]}")
        return False

def main():
    oi = load(OIF)
    if collect_oi(oi):
        save(OIF, oi)
    op = load(OPTF)
    if collect_options(op):
        save(OPTF, op)
    cm = load(CMF)
    if collect_onchain(cm):
        save(CMF, cm)
    ok = load(OKXF)
    if collect_oi_okx(ok):
        save(OKXF, ok)
    print("done")

if __name__ == "__main__":
    main()
