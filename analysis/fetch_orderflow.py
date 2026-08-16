#!/usr/bin/env python3
"""
fetch_orderflow.py — historical spot ORDER-FLOW features from Binance Vision aggTrades.

Binance Vision public S3 (no key) exposes spot monthly aggTrades since 2017-08.
Each aggTrade row: aggTradeId, price, quantity, firstTradeId, lastTradeId,
timestamp_ms, isBuyerMaker, isBestMatch  (no header).

Direction semantics (Binance): isBuyerMaker == True  => buyer is the MAKER, so the
TAKER is the SELLER => this is a sell-side aggressive (taker-sell) trade.
                  isBuyerMaker == False => taker is the BUYER => taker-buy trade.

We aggregate per calendar day (UTC):
  taker_buy_qty / taker_sell_qty        : base volume by aggressor side
  taker_buy_quote / taker_sell_quote    : USD notional by aggressor side
  taker_imbalance_qty / _quote          : (buy-sell)/(buy+sell), [-1, +1]
  taker_buy_ratio                       : buy/(buy+sell)
  n_trades / n_buy / n_sell
  whale_*                               : trades with notional >= USD threshold
                                          (1M = active; 10M = aggressive whale)
  med_notional / mean_notional / p99_notional
  total_qty / total_quote               : cross-check against kline volume

Follows conventions of analysis/fetch_binance_vision.py (same S3 base, per-dataset
zip cache, SHA256 verify, ms/us timestamp normalisation, threaded download).

Usage:
    python3 analysis/fetch_orderflow.py --start 2018-01 --end 2020-12
    (default full range 2017-08 .. last complete month)
"""
import os, sys, json, io, zipfile, hashlib, argparse, random, datetime as _dt
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

BASE = "https://data.binance.vision/data"
REPO = "/home/ubuntu/sfc"
CACHE_DIR = os.path.join(REPO, "data", "binance_vision_cache", "spot_aggTrades")
OUT = os.path.join(REPO, "data", "binance_orderflow_daily.json")
SYMBOL = "BTCUSDT"
N_WORKERS = 6

WHALE_USD_LO = 1_000_000.0    # >= $1M notional  -> active whale (price-era dependent)
WHALE_USD_HI = 10_000_000.0   # >= $10M notional -> aggressive whale
WHALE_QTY_LO = 10.0           # >= 10 BTC base   -> era-agnostic whale (comparable across 2017..)
WHALE_QTY_HI = 100.0          # >= 100 BTC base  -> aggressive whale

RES = 200_000                 # reservoir size per day for median/p99 notional (memory-light)


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
    if raw > 10**14:  # microseconds
        return raw // 1000
    return raw


def _to_day(ms):
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def url_for(ym):
    return f"{BASE}/spot/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{ym}.zip"


def fetch_with_check(ym):
    """Download zip + CHECKSUM, verify SHA256. Returns path or None on 404/error.

    STREAMED download (bounded memory): writes to dest+".part" in 1MB blocks while
    hashing incrementally, then renames to dest. The old version did
    `urlopen(...).read()` which loads the WHOLE zip into RAM — fatal on the
    multi-GB months (2022-11 = 2.46GB) on this 3.6GB box (OOM / exit 137).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = os.path.basename(url_for(ym))
    dest = os.path.join(CACHE_DIR, fname)
    if os.path.exists(dest):       # already cached (verified previously)
        return dest
    tmp = dest + ".part"
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url_for(ym), timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                blk = resp.read(1 << 20)
                if not blk:
                    break
                f.write(blk)
                h.update(blk)
    except urllib.error.HTTPError as e:
        try: os.remove(tmp)
        except OSError: pass
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        try: os.remove(tmp)
        except OSError: pass
        return None
    try:
        ck = urllib.request.urlopen(url_for(ym) + ".CHECKSUM", timeout=30).read()\
            .decode().strip().split()[0]
        if h.hexdigest() != ck:
            print(f"  CHECKSUM MISMATCH {fname}", file=sys.stderr)
            try: os.remove(tmp)
            except OSError: pass
            return None
    except Exception:
        pass  # checksum optional
    os.rename(tmp, dest)
    return dest


def iter_lines(f, chunk=1 << 20):
    """Yield decoded lines from a binary file object without loading it wholly."""
    buf = b""
    while True:
        data = f.read(chunk)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode(errors="ignore")
    if buf:
        yield buf.decode(errors="ignore")


class Reservoir:
    """Fixed-size reservoir for streaming median/percentile of a huge population."""
    __slots__ = ("cap", "samples", "seen")

    def __init__(self, cap=RES):
        self.cap = cap
        self.samples = []
        self.seen = 0

    def add(self, v):
        self.seen += 1
        if len(self.samples) < self.cap:
            self.samples.append(v)
        else:
            j = random.randrange(self.seen)
            if j < self.cap:
                self.samples[j] = v

    def pct(self, p):
        if not self.samples:
            return None
        s = sorted(self.samples)
        return s[min(len(s) - 1, int(len(s) * p))]


def parse_month(path):
    """Return dict day -> orderflow aggregates for one monthly aggTrades zip.

    Uses pandas chunked read (C-parsing, bounded memory) instead of line-by-line
    Python: essential on the 3.6GB host for the 150M+ trade months of 2020.
    Percentile/median of notional come from a stride-downsampled array (bounded
    memory), not from retaining every trade.
    """
    import pandas as pd
    import numpy as np
    # running per-day accumulator (key = epoch day int)
    D = {}
    # bounded reservoir of notional samples for med/p99 — NEVER accumulate the
    # full downsample (a 150M+ trade month would hold ~37M floats and blow the
    # 3.6GB box to OOM). Keep a rolling buffer and halve it when it grows past
    # a cap; this preserves the percentile estimate without unbounded memory.
    nsamp = np.empty(0, dtype=np.float64)
    NSAMP_CAP = 6_000_000

    def _mk():
        return {"buy_q": 0.0, "sell_q": 0.0, "buy_u": 0.0, "sell_u": 0.0,
                "n_buy": 0, "n_sell": 0,
                "wlo_c": 0, "wlo_u": 0.0, "whi_c": 0, "whi_u": 0.0,
                "wqlo_c": 0, "wqlo_u": 0.0, "wqhi_c": 0, "wqhi_u": 0.0,
                "last_ts": -1, "last_price": None}

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                for chunk in pd.read_csv(
                        f, header=None, usecols=[1, 2, 5, 6],
                        names=["price", "qty", "ts", "isbm"],
                        dtype={"price": np.float64, "qty": np.float64, "ts": np.uint64},
                        chunksize=1_000_000):
                    ts = chunk["ts"].to_numpy()
                    ts = np.where(ts > 10 ** 14, ts // 1000, ts)  # us -> ms
                    day = ts // 86400000                          # UTC epoch-day
                    price = chunk["price"].to_numpy()
                    qty = chunk["qty"].to_numpy()
                    notional = price * qty
                    buy = ~chunk["isbm"].to_numpy()
                    # vectorised daily aggregates (C groupby)
                    gd = pd.DataFrame({
                        "day": day, "price": price, "qty": qty,
                        "notional": notional, "is_buy": buy, "ts": ts,
                    })
                    b = gd[gd["is_buy"]].groupby("day")
                    s = gd[~gd["is_buy"]].groupby("day")
                    bq = b["qty"].sum().to_dict(); bu = b["notional"].sum().to_dict()
                    sq = s["qty"].sum().to_dict(); su = s["notional"].sum().to_dict()
                    nb = b.size().to_dict(); ns = s.size().to_dict()
                    for k in set(bq) | set(sq):
                        acc = D.setdefault(k, _mk())
                        acc["buy_q"] += bq.get(k, 0.0); acc["sell_q"] += sq.get(k, 0.0)
                        acc["buy_u"] += bu.get(k, 0.0); acc["sell_u"] += su.get(k, 0.0)
                        acc["n_buy"] += nb.get(k, 0); acc["n_sell"] += ns.get(k, 0)
                    for lbl, mask, uidx, cidx in [
                            ("wlo", gd["notional"] >= WHALE_USD_LO, "notional", "wlo"),
                            ("whi", gd["notional"] >= WHALE_USD_HI, "notional", "whi"),
                            ("wqlo", gd["qty"] >= WHALE_QTY_LO, "notional", "wqlo"),
                            ("wqhi", gd["qty"] >= WHALE_QTY_HI, "notional", "wqhi")]:
                        sub = gd[mask]
                        if len(sub):
                            cnt = sub.groupby("day").size().to_dict()
                            vol = sub.groupby("day")[uidx].sum().to_dict()
                            for k in cnt:
                                acc = D.setdefault(k, _mk())
                                acc[cidx + "_c"] += cnt[k]
                                acc[cidx + "_u"] += vol.get(k, 0.0)
                    # last trade price PER DAY (max ts in this chunk), compared
                    # across chunks so the true end-of-day price wins
                    if len(gd):
                        idx = gd.groupby("day")["ts"].idxmax()
                        for dk in idx.index:
                            i = idx[dk]
                            tsv = int(gd.loc[i, "ts"])
                            acc = D.setdefault(int(dk), _mk())
                            if tsv > acc["last_ts"]:
                                acc["last_ts"] = tsv
                                acc["last_price"] = float(gd.loc[i, "price"])
                    # bounded notional reservoir (downsample + cap, see top)
                    nsamp = np.concatenate([nsamp, notional[::4]])
                    if nsamp.size > NSAMP_CAP:
                        nsamp = nsamp[::2]

    nss = np.sort(nsamp) if nsamp.size else np.array([])

    def _pct(p):
        if nss.size == 0:
            return None
        return float(nss[min(nss.size - 1, int(nss.size * p))])

    out = {}
    for k, acc in D.items():
        date = (_dt.datetime(1970, 1, 1) + _dt.timedelta(days=int(k))).strftime("%Y-%m-%d")
        buy_q, sell_q = acc["buy_q"], acc["sell_q"]
        buy_u, sell_u = acc["buy_u"], acc["sell_u"]
        tot_q = buy_q + sell_q; tot_u = buy_u + sell_u
        out[date] = {
            "price_close": acc["last_price"],
            "taker_buy_qty": round(buy_q, 6), "taker_sell_qty": round(sell_q, 6),
            "taker_buy_quote": round(buy_u, 2), "taker_sell_quote": round(sell_u, 2),
            "total_qty": round(tot_q, 6), "total_quote": round(tot_u, 2),
            "taker_imbalance_qty": round((buy_q - sell_q) / tot_q, 6) if tot_q else None,
            "taker_imbalance_quote": round((buy_u - sell_u) / tot_u, 6) if tot_u else None,
            "taker_buy_ratio": round(buy_q / tot_q, 6) if tot_q else None,
            "n_trades": acc["n_buy"] + acc["n_sell"], "n_buy": acc["n_buy"], "n_sell": acc["n_sell"],
            "whale_lo_count": acc["wlo_c"], "whale_lo_quote": round(acc["wlo_u"], 2),
            "whale_hi_count": acc["whi_c"], "whale_hi_quote": round(acc["whi_u"], 2),
            "whale_qty_lo_count": acc["wqlo_c"], "whale_qty_lo_quote": round(acc["wqlo_u"], 2),
            "whale_qty_hi_count": acc["wqhi_c"], "whale_qty_hi_quote": round(acc["wqhi_u"], 2),
            "med_notional": round(_pct(0.5), 2) if nss.size else None,
            "p99_notional": round(_pct(0.99), 2) if nss.size else None,
        }
    return out


def _disk_free_gb():
    st = os.statvfs("/home/ubuntu")
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default=last_complete_month())
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--stream", action="store_true",
                    help="process months sequentially (download->parse->delete zip) to "
                         "keep raw-disk usage to ~one month; merge into existing output")
    args = ap.parse_args()

    yms = months(args.start, args.end)

    # merge base: reuse existing aggregated output so a re-run / tail extension
    # does not require re-fetching months already aggregated.
    daily = {}
    if os.path.exists(OUT):
        try:
            daily = json.load(open(OUT))
            print(f"merge base: {len(daily)} existing days in {OUT}", flush=True)
        except Exception as e:
            print(f"  base load failed ({e}); starting fresh", file=sys.stderr)

    if args.stream:
        print(f"order-flow aggTrades STREAM: {len(yms)} months ({args.start} .. {args.end}), "
              f"disk free {_disk_free_gb():.1f}GB", flush=True)
        for ym in yms:
            p = fetch_with_check(ym)
            if not p:
                print(f"  {ym}: 404/error, skip", flush=True)
                continue
            try:
                recs = parse_month(p)
            except Exception as e:
                print(f"  parse FAIL {ym}: {e} (zip kept for retry)", file=sys.stderr)
                continue
            for day, r in recs.items():
                daily[day] = r
            daily_sorted = {d: daily[d] for d in sorted(daily)}
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w") as f:
                json.dump(daily_sorted, f)
            try:
                os.remove(p)   # free raw-zip disk; data preserved in aggregated JSON
            except OSError:
                pass
            print(f"  {ym}: {len(recs)} days | total {len(daily_sorted)} | "
                  f"disk free {_disk_free_gb():.1f}GB", flush=True)
    else:
        print(f"order-flow aggTrades: {len(yms)} months ({args.start} .. {args.end}), "
              f"threads={args.workers}", flush=True)
        paths = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_with_check, ym): ym for ym in yms}
            for fut in as_completed(futs):
                ym = futs[fut]
                paths[ym] = fut.result()
        n404 = sum(1 for p in paths.values() if p is None)
        print(f"download complete: {len(paths) - n404} ok, {n404} 404", flush=True)
        for ym, p in paths.items():
            if not p:
                continue
            try:
                recs = parse_month(p)
            except Exception as e:
                print(f"  parse FAIL {ym}: {e}", file=sys.stderr)
                continue
            for day, r in recs.items():
                daily[day] = r
            print(f"  {ym}: {len(recs)} days", flush=True)
            daily_sorted = {d: daily[d] for d in sorted(daily)}
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w") as f:
                json.dump(daily_sorted, f)

    daily_sorted = {d: daily[d] for d in sorted(daily)}
    print(f"\nwrote {len(daily_sorted)} days -> {OUT}")
    print(f"range: {min(daily_sorted)} .. {max(daily_sorted)}")


if __name__ == "__main__":
    main()
