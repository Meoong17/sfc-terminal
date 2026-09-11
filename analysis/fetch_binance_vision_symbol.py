#!/usr/bin/env python3
"""
fetch_binance_vision_symbol.py — Ambil klines 1d Binance Vision untuk simbol APA PUN.

Dipakai untuk uji universalitas lintas-aset (ETH/SOL) dengan definisi estimator
yang IDENTIK dengan uji BTC, supaya perbandingannya bukan perbandingan apel-jeruk.
Tidak menyentuh data kanonik BTC (data/binance_vision_daily.json).

Sumber sama dengan fetcher kanonik: data.binance.vision public S3, gratis, tanpa key.
Pemakaian:  python analysis/fetch_binance_vision_symbol.py ETHUSDT [START_YM]
Output:     data/binance_vision_<SYMBOL>_daily.json  ({date: {open,high,low,close,volume}})
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://data.binance.vision/data/spot/monthly/klines"
N_WORKERS = 4


def months(start_ym, end_ym):
    y, m = int(start_ym[:4]), int(start_ym[5:7])
    ey, em = int(end_ym[:4]), int(end_ym[5:7])
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def last_complete_month():
    import datetime as dt
    today = dt.datetime.now(dt.timezone.utc).date()
    first = today.replace(day=1)
    prev = first - dt.timedelta(days=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def fetch_month(symbol, ym, retries=3):
    url = f"{BASE}/{symbol}/1d/{symbol}-1d-{ym}.zip"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                    # bulan belum ada (pra-listing)
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))    # 429/5xx: backoff sebelum ulang
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    rows = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("utf-8", "replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        # Beberapa bulan terbaru menyertakan baris header -> lewati kalau bukan numerik
        try:
            float(parts[0])
        except ValueError:
            continue
        if len(parts) < 6:
            continue
        ts = int(parts[0])
        # Normalisasi satuan waktu. File Binance Vision 2025+ memakai MIKRODETIK
        # (1.7e15), file lama memakai milidetik (1.7e12) — menebak salah satunya
        # menghasilkan tanggal tahun 58551 dan bulan itu hilang tanpa error,
        # yang pernah membuat cakupan ETH/SOL berhenti diam-diam di 2024-12.
        if ts > 10 ** 14:
            ts //= 10 ** 6
        elif ts > 10 ** 11:
            ts //= 10 ** 3
        import datetime as dt
        d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        if not ("2010-01-01" <= d <= "2099-12-31"):
            continue
        rows[d] = {"open": float(parts[1]), "high": float(parts[2]),
                   "low": float(parts[3]), "close": float(parts[4]),
                   "volume": float(parts[5])}
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    symbol = sys.argv[1].upper()
    start = sys.argv[2] if len(sys.argv) > 2 else "2017-08"
    end = os.environ.get("SFC_END_MONTH") or last_complete_month()
    mons = list(months(start, end))
    print(f"[{symbol}] {len(mons)} bulan ({mons[0]}..{mons[-1]})", file=sys.stderr)

    out = {}
    ok = missing = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_month, symbol, ym): ym for ym in mons}
        for fut in as_completed(futs):
            ym = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                       # noqa: BLE001
                print(f"[{symbol}] {ym} GAGAL: {e}", file=sys.stderr)
                continue
            if r is None:
                missing += 1
            else:
                out.update(r)
                ok += 1
    # Laporkan bulan yang GAGAL (bukan 404) secara eksplisit — kegagalan senyap
    # pernah membuat cakupan berhenti di 2024-12 tanpa peringatan apa pun.
    got_months = {d[:7] for d in out}
    failed = [m for m in mons if m not in got_months]
    dest = os.path.join(SFC_DIR, "data", f"binance_vision_{symbol}_daily.json")
    with open(dest, "w") as f:
        json.dump(out, f)
    days = sorted(out)
    print(f"[{symbol}] {ok} bulan ok / {missing} 404 -> {len(out)} hari "
          f"({days[0]}..{days[-1]}) -> {dest}", file=sys.stderr)
    if failed:
        print(f"[{symbol}] PERINGATAN: {len(failed)} bulan TIDAK terunduh "
              f"(bukan 404): {failed[:20]}", file=sys.stderr)


if __name__ == "__main__":
    main()
