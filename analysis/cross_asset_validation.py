#!/usr/bin/env python3
"""cross_asset_validation.py — uji historis gerbang cross-asset (bukan klaim).

Menjawab pertanyaan yang tersisa setelah `analysis/cross_asset_gates.py` dipasang:
apakah gerbang-gerbang itu punya nilai, atau hanya terlihat masuk akal?

Yang diuji (hanya yang riwayatnya benar-benar ada):
  * Model 1a policy trap  : Core PCE > 2,5% YoY & Real PCE <= 0,5% MoM
  * Model 1b flight-to-Q  : ΔUS10Y(30d) < 0 & ΔHY OAS(30d) > 0
  * Model 1c lock         : 1a & 1b
  * Gate 1 ETF exhaustion : IBIT net >= 0 pada >=3 dari 5 hari bursa (2024+)
  * Gate 3 spot absorption: harga 7D lower low & CVD 7D higher low (2017+)
  * Gate 4 structure      : harga di atas 200W SMA (2017+)
  * Gate 2 OI flush       : dilaporkan TIDAK CUKUP DATA (OI lokal mulai 2026-08)

Metodologi:
  * Return forward 30 dan 90 hari; label bertumpang tindih → moving-block bootstrap
    (blok 30 hari) untuk CI, bukan iid (lihat catatan di walk_forward_validation.py).
  * Era split: era1 2017-19, era2 2020-21, era3 2022-26. Tanda harus konsisten;
    era dengan n < 40 hari-kondisi dilaporkan "tidak informatif", bukan "gagal".
  * Tidak ada perubahan kode/scoring: skrip ini murni pengukuran, menulis
    `.cross_asset_validation.json`.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
import urllib.request
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "fred_cache")
OUT = os.path.join(ROOT, ".cross_asset_validation.json")
N_BOOT = 2000
BLOCK = 30
SEED = 20261127
ERAS = [("era1", "2017-01-01", "2020-01-01"),
        ("era2", "2020-01-01", "2022-01-01"),
        ("era3", "2022-01-01", "2099-01-01")]


def fred_key():
    k = os.getenv("FRED_API_KEY")
    if k:
        return k
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("FRED_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def fred(series_id: str) -> list[tuple[str, float]]:
    """Ambil seri FRED dengan cache disk (data/fred_cache/<id>.json, TTL 7 hari)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{series_id}.json")
    if os.path.exists(path):
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
        if age <= 7:
            return [(d, v) for d, v in json.load(open(path))]
    key = fred_key()
    if not key:
        return []
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={key}&file_type=json&observation_start=1990-01-01")
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            data = json.load(r)
    except Exception as e:
        print(f"[XAV] FRED {series_id}: {e}", file=sys.stderr)
        return []
    rows = []
    for o in data.get("observations", []):
        v = o.get("value")
        if v in (None, ".", ""):
            continue
        try:
            rows.append((o["date"], float(v)))
        except ValueError:
            continue
    json.dump(rows, open(path, "w"))
    return rows


def as_map(rows):
    return dict(rows)


def prior(m, day: str, max_back: int = 45):
    d = datetime.strptime(day, "%Y-%m-%d")
    for i in range(max_back + 1):
        k = (d - timedelta(days=i)).strftime("%Y-%m-%d")
        if k in m:
            return m[k]
    return None


def shift(m, day: str, days: int):
    """Nilai seri pada `day` - `days` (untuk delta)."""
    return prior(m, (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d"), 10)


def block_boot(a, b, block=BLOCK, n=N_BOOT, seed=SEED):
    if len(a) < block * 2 or len(b) < block * 2:
        return None, None, None
    rng = random.Random(seed)

    def draw(x):
        out = []
        while len(out) < len(x):
            s = rng.randrange(0, max(1, len(x) - block))
            out.extend(x[s:s + block])
        return out[:len(x)]

    diffs = sorted(st.mean(draw(a)) - st.mean(draw(b)) for _ in range(n))
    return st.mean(a) - st.mean(b), diffs[int(0.05 * n)], diffs[int(0.95 * n)]


def load_btc():
    rows = []
    with open(os.path.join(ROOT, "data_v2", "btc_daily.csv")) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) > 4:
                try:
                    rows.append((p[0], float(p[4])))
                except ValueError:
                    continue
    rows.sort()
    return rows


def load_cvd():
    d = json.load(open(os.path.join(ROOT, "data", "binance_orderflow_daily.json")))
    days = sorted(k for k in d if not k.startswith("_"))
    out, run = {}, 0.0
    for k in days:
        r = d[k]
        run += (r.get("taker_buy_quote") or 0) - (r.get("taker_sell_quote") or 0)
        out[k] = run
    return out


def load_ibit():
    d = json.load(open(os.path.join(ROOT, ".etf_cache.json")))
    out = {}
    for f in d.get("flows") or []:
        etfs = f.get("etfs") or {}
        if "IBIT" in etfs and etfs.get("IBIT") is not None:
            out[f["date"]] = float(etfs["IBIT"])  # juta USD
    return out


def evaluate(name, dates, flags, fwd_by_date, note="", restrict_from=None):
    """flags: {date: bool}; fwd_by_date: {date: {30: r, 90: r}}

    `restrict_from` wajib dipakai bila flag hanya punya riwayat pendek (mis. IBIT
    2024+). Tanpa itu, hari ON (era terbaru) dibandingkan dengan hari OFF dari era
    lama — pembandingnya bukan kondisi yang sama, dan selisihnya mengukur era,
    bukan sinyal.
    """
    if restrict_from:
        dates = [d for d in dates if d >= restrict_from]
    res = {"name": name, "note": note,
           "window": [dates[0], dates[-1]] if dates else None}
    for h in (30, 90):
        on = [fwd_by_date[d][h] for d in dates if flags.get(d) and fwd_by_date.get(d, {}).get(h) is not None]
        off = [fwd_by_date[d][h] for d in dates if not flags.get(d) and fwd_by_date.get(d, {}).get(h) is not None]
        if len(on) < 30 or len(off) < 30:
            res[f"h{h}"] = {"n_on": len(on), "n_off": len(off), "verdict": "sampel kurang"}
            continue
        gap, lo, hi = block_boot(on, off)
        res[f"h{h}"] = {
            "n_on": len(on), "n_off": len(off),
            "mean_on": round(st.mean(on), 2), "mean_off": round(st.mean(off), 2),
            "gap": round(gap, 2) if gap is not None else None,
            "ci": [round(lo, 2), round(hi, 2)] if lo is not None else None,
            "significant_block": bool(lo is not None and lo * hi > 0),
        }
    # era consistency (30d)
    era = {}
    for ename, lo_d, hi_d in ERAS:
        on = [fwd_by_date[d][30] for d in dates
              if lo_d <= d < hi_d and flags.get(d) and fwd_by_date.get(d, {}).get(30) is not None]
        off = [fwd_by_date[d][30] for d in dates
               if lo_d <= d < hi_d and not flags.get(d) and fwd_by_date.get(d, {}).get(30) is not None]
        if len(on) < 20 or len(off) < 20:
            era[ename] = {"n_on": len(on), "n_off": len(off), "note": "tidak informatif"}
            continue
        era[ename] = {"n_on": len(on), "n_off": len(off), "gap": round(st.mean(on) - st.mean(off), 2)}
    res["era_30d"] = era
    gaps = [v["gap"] for v in era.values() if "gap" in v]
    res["era_sign_consistent"] = bool(gaps) and (all(g < 0 for g in gaps) or all(g > 0 for g in gaps))
    res["verdict"] = _verdict(res)
    return res


def _verdict(res):
    h30 = res.get("h30", {})
    if "gap" not in h30:
        return "tidak cukup data"
    sig = h30.get("significant_block")
    gap = h30["gap"]
    consistent = res.get("era_sign_consistent")
    if sig and consistent:
        return f"EDGE ({gap:+.2f}pp, signifikan & era-konsisten)"
    if sig and not consistent:
        return f"edge tidak stabil antar era ({gap:+.2f}pp signifikan pooled)"
    if not sig and consistent:
        return f"arah konsisten tapi tidak signifikan ({gap:+.2f}pp)"
    return f"TIDAK ADA EDGE ({gap:+.2f}pp, tidak signifikan)"


def main():
    btc = load_btc()
    dates = [d for d, _ in btc]
    px = dict(btc)
    fwd = {}
    for i, d in enumerate(dates):
        row = {}
        for h in (30, 90):
            if i + h < len(dates):
                row[h] = (px[dates[i + h]] - px[d]) / px[d] * 100
            else:
                row[h] = None
        fwd[d] = row
    closes = [p for _, p in btc]
    sma_series = {}
    for i, d in enumerate(dates):
        if i >= 1400:
            sma_series[d] = st.mean(closes[i - 1399:i + 1])
        elif i >= 200:
            sma_series[d] = st.mean(closes[:i + 1])

    pce = as_map(fred("PCEPILFE"))
    rpce = as_map(fred("PCEC96"))
    y10 = as_map(fred("DGS10"))
    oas = as_map(fred("BAMLH0A0HYM2"))
    ibit = load_ibit()
    cvd = load_cvd()

    # ── flags ────────────────────────────────────────────────────────────
    trap, f2q, lock = {}, {}, {}
    for d in dates:
        cy = prior(pce, d, 60)
        cy_prev = prior(pce, (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"), 60)
        rm = None
        r_today = prior(rpce, d, 60)
        r_prev = prior(rpce, (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=32)).strftime("%Y-%m-%d"), 60)
        if r_today and r_prev:
            rm = (r_today - r_prev) / abs(r_prev) * 100
        yo = (cy - cy_prev) / abs(cy_prev) * 100 if (cy and cy_prev) else None
        trap[d] = bool(yo is not None and rm is not None and yo > 2.5 and rm <= 0.5)
        y_now, y_prev = prior(y10, d, 7), shift(y10, d, 30)
        o_now, o_prev = prior(oas, d, 7), shift(oas, d, 30)
        f2q[d] = bool(y_now and y_prev and o_now and o_prev and (y_now - y_prev) < 0 and (o_now - o_prev) > 0)
        lock[d] = bool(trap[d] and f2q[d])

    # Gate 1: IBIT >= 0 pada >=3 dari 5 hari bursa terakhir yang tersedia
    g1 = {}
    ib_days = sorted(ibit)
    for d in dates:
        recent = [x for x in ib_days if x <= d][-5:]
        g1[d] = bool(len(recent) == 5 and sum(1 for x in recent if ibit[x] >= 0) >= 3)

    # Gate 3: lower low harga + higher low CVD (7D jendela)
    g3 = {}
    for i, d in enumerate(dates):
        if i < 14 or d not in cvd:
            g3[d] = False
            continue
        prev7 = dates[i - 13:i - 6]
        last7 = dates[i - 6:i + 1]
        if any(x not in cvd for x in prev7 + last7):
            g3[d] = False
            continue
        g3[d] = (min(px[x] for x in last7) < min(px[x] for x in prev7)
                 and min(cvd[x] for x in last7) > min(cvd[x] for x in prev7))

    # Gate 4: di atas 200W SMA
    g4 = {d: bool(d in sma_series and px[d] > sma_series[d]) for d in dates}

    results = [
        evaluate("Model 1a — policy trap (PCE>2.5% & Real PCE<=0.5%)", dates, trap, fwd,
                 "FRED PCEPILFE/PCEC96, BTC 2017+"),
        evaluate("Model 1b — flight-to-quality (ΔY10<0 & ΔOAS>0)", dates, f2q, fwd,
                 "FRED DGS10/BAMLH0A0HYM2, 30d delta"),
        evaluate("Model 1c — CONTRACTION_LOCK (1a & 1b)", dates, lock, fwd),
        evaluate("Gate 1 — ETF exhaustion (IBIT >=0 x3/5)", dates, g1, fwd,
                 "IBIT per-ticker dari .etf_cache.json; dibatasi ke era ETF saja",
                 restrict_from=(sorted(ibit)[0] if ibit else None)),
        evaluate("Gate 3 — spot absorption (lower low px + higher low CVD)", dates, g3, fwd,
                 "orderflow Binance 2017+",
                 restrict_from=(sorted(cvd)[0] if cvd else None)),
        evaluate("Gate 4 — structure reclaim (harga > 200W SMA)", dates, g4, fwd),
    ]

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_days": len(dates),
        "range": [dates[0], dates[-1]],
        "results": results,
        "not_tested": {
            "Gate 2 — OI flush": ("data OI lokal hanya mulai 2026-08 (~118 hari); "
                                  "tidak cukup untuk uji 30/90 hari — TIDAK diuji"),
            "dash-for-cash (gold/silver)": ("tidak ada seri logam harian yang bisa diakses "
                                            "(stooq anti-scraping, Yahoo 429, FRED discontinued)"),
            "ACBI": "tidak ada sumber CDS/bond 144A",
        },
        "meta": {
            "scoring_impact": "NONE — pengukuran saja",
            "caveat": ("Gate adalah kondisi state (bukan event); return forward dari hari "
                       "berstatus gate ON. Overlap label ditangani block bootstrap 30 hari."),
        },
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"n={len(dates)} hari ({dates[0]}..{dates[-1]})\n")
    for r in results:
        h30, h90 = r.get("h30", {}), r.get("h90", {})
        print(f"{r['name']}")
        if "gap" in h30:
            print(f"   30d: ON {h30['mean_on']:+.2f}% (n={h30['n_on']}) vs OFF {h30['mean_off']:+.2f}% "
                  f"(n={h30['n_off']}) gap {h30['gap']:+.2f}pp CI {h30['ci']}")
        if "gap" in h90:
            print(f"   90d: ON {h90['mean_on']:+.2f}% vs OFF {h90['mean_off']:+.2f}% "
                  f"gap {h90['gap']:+.2f}pp CI {h90['ci']}")
        print(f"   era 30d: { {k: v.get('gap', v.get('note')) for k, v in r['era_30d'].items()} }")
        print(f"   → {r['verdict']}\n")
    print("Tidak diuji:")
    for k, v in out["not_tested"].items():
        print(f"  - {k}: {v}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
