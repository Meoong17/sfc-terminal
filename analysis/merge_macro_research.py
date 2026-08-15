#!/usr/bin/env python3
"""
SFC Research Dataset — Merge + Audit
=====================================
Gabung seri macro (data/raw/macro/*.json) dengan harga BTC kanonik
(data/binance_vision_daily.json, Binance Vision) menjadi SATU dataset
harian untuk penelitian regime transition / liquidity timing / stress-gap.

Prinsip (sesuai proposal Bisa.docx + standar SFC):
  - Raw TIDAK disentuh. Output ke data/cleaned + data/merged.
  - MACRO daily/yield series: forward-fill (last-observation-carried-forward)
    ke grid harian — tidak mengarang data, hanya mengisi hari non-trading
    dengan nilai terakhir yang diketahui.
  - Seri mingguan/bulanan (WALCL, M2, dll): di-ffill juga, TAPI ditandai
    freq-nya agar penelitian tahu resolusi aslinya.
  - GAP dicatat (Gold di-retire di FRED).
  - Audit: coverage per kolom, missing count, freshness, alignment.

Output:
  data/cleaned/macro_daily_clean.json   = macro di grid harian (ffill)
  data/merged/sfc_research_daily.json   = gabungan BTC + macro
  data/merged/_audit.json               = audit coverage/missing/freshness
"""
import json, os, sys
from datetime import datetime, date

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(SFC_DIR, "data", "raw", "macro")
CLEAN_DIR = os.path.join(SFC_DIR, "data", "cleaned")
MERGED_DIR = os.path.join(SFC_DIR, "data", "merged")
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

# Frekuensi asli tiap seri (untuk menandai resolusi, bukan ff-accuracy)
FREQ = {
    "DGS2": "daily", "DGS10": "daily", "DGS30": "daily", "T10YIE": "daily",
    "T10Y2Y": "daily", "DFF": "daily", "VIXCLS": "daily", "SP500": "daily",
    "DCOILBRENTEU": "daily", "DCOILWTICO": "daily",
    "RRPONTSYD": "daily",
    "WALCL": "weekly", "WTREGEN": "weekly", "ECBASSETSW": "weekly",
    "M2SL": "monthly", "JPNASSETS": "monthly",
}

# Seri yang disertakan di dataset gabungan + label
INCLUDE = {
    "DGS2": "US2Y", "DGS10": "US10Y", "DGS30": "US30Y",
    "T10YIE": "BE10", "T10Y2Y": "SPREAD_10_2", "DFF": "FEDFUNDS",
    "VIXCLS": "VIX", "SP500": "SPX",
    "DCOILBRENTEU": "BRENT", "DCOILWTICO": "WTI",
    "RRPONTSYD": "RRP",
    "WALCL": "FED_BS", "WTREGEN": "TGA", "ECBASSETSW": "ECB_BS",
    "M2SL": "M2_US", "JPNASSETS": "BOJ_BS",
}


def load_series():
    """Load semua raw macro series sebagai {sid: {date: value}}."""
    out = {}
    manifest = {}
    mpath = os.path.join(RAW_DIR, "_manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    for sid in INCLUDE:
        path = os.path.join(RAW_DIR, f"{sid}.json")
        if os.path.exists(path):
            out[sid] = json.load(open(path))
        else:
            print(f"[gap] {sid} tidak ada di raw (file hilang)")
    return out, manifest


def ff_to_grid(series, grid_dates):
    """Forward-fill seri {date:value} ke grid harian (LOCF)."""
    filled = {}
    last = None
    for d in grid_dates:
        if d in series:
            last = series[d]
        if last is not None:
            filled[d] = last
    return filled


def main():
    series, manifest = load_series()
    if not series:
        print("ERROR: tidak ada seri raw. Jalankan fetch_macro_research.py dulu.")
        sys.exit(1)

    # BTC (kanonik) — dari Binance Vision
    btc_path = os.path.join(SFC_DIR, "data", "binance_vision_daily.json")
    if not os.path.exists(btc_path):
        print("ERROR: data/binance_vision_daily.json tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
    btc = json.load(open(btc_path))
    btc_dates = sorted(btc)  # 'YYYY-MM-DD'
    btc_start = btc_dates[0]
    btc_end = btc_dates[-1]

    # Grid harian: batasi ke jendela yang PALING sempit yang relevan.
    # BTC tersedia dari 2017-08-17. SP500 dari 2016-08. Kita pakai grid
    # dari btc_start sampai max(binance last, macro last) — alignment pada
    # tanggal yang BTC miliki, macro di-ffill ke hari itu.
    grid_dates = btc_dates  # gunakan hari-hari BTC sebagai grid kanonik

    # Bangun record harian
    daily = []
    for d in grid_dates:
        rec = {"date": d}
        b = btc[d]
        rec["BTC_open"] = b.get("open")
        rec["BTC_high"] = b.get("high")
        rec["BTC_low"] = b.get("low")
        rec["BTC_close"] = b.get("close")
        rec["BTC_volume"] = b.get("volume")
        rec["BTC_quote_vol"] = b.get("quote_vol")
        rec["BTC_taker_base"] = b.get("taker_base")
        rec["BTC_taker_quote"] = b.get("taker_quote")
        # Macro: ff ke tanggal ini
        for sid, alias in INCLUDE.items():
            if sid in series:
                s = series[sid]
                rec[alias] = s.get(d)  # value persis pd hari itu (jika ada)
        daily.append(rec)

    # Hitung LOCF macro untuk setiap kolom macro (alignment hari non-trading)
    macro_cols = list(INCLUDE.values())
    # Untuk audit & ffill: buat grid macro df per kolom
    # ffill ke grid_dates (hari BTC)
    filled_cols = {}
    for sid, alias in INCLUDE.items():
        if sid in series:
            filled_cols[alias] = ff_to_grid(series[sid], grid_dates)

    # Terapkan ffill ke daily records
    for rec in daily:
        d = rec["date"]
        for alias in macro_cols:
            if alias in filled_cols and rec.get(alias) is None:
                rec[alias] = filled_cols[alias].get(d)

    # ── Audit ──
    audit = {
        "btc_start": btc_start, "btc_end": btc_end,
        "btc_days": len(grid_dates),
        "n_series": len(INCLUDE),
        "gold_gap": "GOLD retired di FRED (HTTP 400) — perlu sumber lain",
        "columns": {},
    }
    for alias in ["BTC_close", "BTC_volume"] + macro_cols:
        last_d = None
        for r in reversed(daily):
            if r.get(alias) is not None:
                last_d = r["date"]
                break
        n = sum(1 for r in daily if r.get(alias) is not None)
        audit["columns"][alias] = {
            "n": n,
            "coverage_pct": round(n / len(daily) * 100, 1),
            "last_date": last_d,
            "freq": FREQ.get(alias, "daily"),
        }

    # Save cleaned macro (ffill grid)
    cleaned_macro = []
    for i, d in enumerate(grid_dates):
        row = {"date": d}
        for alias in macro_cols:
            if alias in filled_cols:
                row[alias] = filled_cols[alias].get(d)
        cleaned_macro.append(row)
    with open(os.path.join(CLEAN_DIR, "macro_daily_clean.json"), "w") as f:
        json.dump(cleaned_macro, f, indent=1)

    # Save merged
    with open(os.path.join(MERGED_DIR, "sfc_research_daily.json"), "w") as f:
        json.dump(daily, f, indent=1)

    # Save audit
    with open(os.path.join(MERGED_DIR, "_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    # ── Print ringkasan ──
    print(f"=== SFC Research Dataset ===  {btc_start} → {btc_end}  ({len(daily)} hari)")
    print(f"{'kolom':14s} {'freq':8s} {'n':>7s} {'cov%':>6s}  last")
    print("-" * 55)
    for alias in ["BTC_close", "BTC_volume"] + macro_cols:
        a = audit["columns"][alias]
        print(f"{alias:14s} {a['freq']:8s} {a['n']:>7d} {a['coverage_pct']:>5.1f}%  {a['last_date']}")
    print(f"\nSaved: {MERGED_DIR}/sfc_research_daily.json")
    print(f"Clean: {CLEAN_DIR}/macro_daily_clean.json")
    print(f"Audit: {MERGED_DIR}/_audit.json")


if __name__ == "__main__":
    main()
