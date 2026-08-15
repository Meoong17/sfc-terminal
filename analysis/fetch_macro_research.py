#!/usr/bin/env python3
"""
SFC Research Dataset — Macro Fetch (FRED, no API key leak)
===========================================================
Tarik seri macro harian/mingguan/bulanan dari FRED ke data/raw/macro/.
Ini ADALAH layer akuisisi data untuk penelitian (regime transition,
liquidity timing, stress-gap) — TIDAK menyentuh scoring sfc_effective.

Sumber: FRED (api.stlouisfed.org). Key dibaca dari .env (FRED_API_KEY).
Gold FRED series sudah di-retire (HTTP 400) — dicatat sebagai gap.

Output:
  data/raw/macro/<series_id>.json   = {date: value} per seri
  data/raw/macro/_manifest.json     = ringkasan tiap seri (range, n, last)
"""
import json, os, sys, time, requests
from datetime import datetime

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(SFC_DIR, "data", "raw", "macro")
os.makedirs(RAW_DIR, exist_ok=True)

# Load FRED key dari .env (tanpa memuat file .env ke stdout)
FRED_KEY = ""
env_path = os.path.join(SFC_DIR, ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        if line.startswith("FRED_API_KEY="):
            FRED_KEY = line.strip().split("=", 1)[1]
if not FRED_KEY:
    FRED_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_KEY:
    print("ERROR: FRED_API_KEY tidak ditemukan di .env", file=sys.stderr)
    sys.exit(1)

# ── Definisi seri untuk SFC Research Dataset ──
# 10 seri inti dari proposal Bisa.docx + seri yang GLF sudah pakai (biar
# penelitian punya satu dataset gabungan yang konsisten).
SERIES = {
    # Rates / dollar
    "DGS2":   ("US 2Y constant-maturity yield", "daily", "pct"),
    "DGS10":  ("US 10Y constant-maturity yield", "daily", "pct"),
    "DGS30":  ("US 30Y constant-maturity yield", "daily", "pct"),
    "T10YIE": ("10Y breakeven inflation", "daily", "pct"),
    "T10Y2Y": ("10Y-2Y spread", "daily", "pct"),
    "DFF":    ("Effective Federal Funds Rate", "daily", "pct"),
    # Risk appetite / equity
    "VIXCLS": ("VIX", "daily", "index"),
    "SP500":  ("S&P 500", "daily", "index"),
    # Commodities
    "DCOILBRENTEU": ("Brent crude spot", "daily", "usd"),
    "DCOILWTICO":   ("WTI crude spot", "daily", "usd"),
    # Liquidity (yang sudah dipakai GLF — disimpan juga utk research)
    "WALCL":     ("Fed total assets", "weekly", "usd_mil"),
    "WTREGEN":   ("Treasury General Account", "weekly", "usd_mil"),
    "RRPONTSYD": ("Reverse Repo overnight", "daily", "usd_bil"),
    "M2SL":      ("US M2 money supply", "monthly", "usd_bil"),
    "ECBASSETSW":("ECB total assets", "weekly", "usd_mil"),
    "JPNASSETS": ("BOJ total assets", "monthly", "usd_mil"),
    # CORE GAP: Gold (GOLDAMGBD228NLBM retired di FRED → gap, diisi nanti)
    # ── Seri tambahan dari proposal (rates lanjutan, equity, credit, copper) ──
    "NASDAQCOM": ("Nasdaq Composite", "daily", "index"),
    "DFII10":    ("10Y real yield (TIPS)", "daily", "pct"),
    "DGS5":      ("US 5Y constant-maturity yield", "daily", "pct"),
    "BAMLH0A0HYM2": ("US High-Yield OAS", "daily", "pct"),
    "BAMLC0A0CM":   ("US Investment-Grade OAS", "daily", "pct"),
    "PCOPPUSDM": ("Copper (global price, US$)", "monthly", "usd_mt"),
    # Catatan: 5s30s dihitung = DGS30 - DGS5 di merge; RUT (Russell 2000) & 
    # DCOPPERTUSD tidak ada di FRED (HTTP 400) → gap dicatat.
}


def fetch_series(sid):
    """Full-history FRED fetch. Returns {date: value} or None."""
    r = requests.get(
        f"https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": sid, "api_key": FRED_KEY,
                "file_type": "json", "sort_order": "asc"},
        timeout=30,
    )
    if r.status_code != 200:
        return None, r.status_code
    out = {}
    for o in r.json().get("observations", []):
        v = o["value"]
        if v != "." and v != "":
            try:
                out[o["date"]] = float(v)
            except ValueError:
                continue
    return (out, 200) if out else (None, 200)


def main():
    manifest = {}
    for sid, (label, freq, unit) in SERIES.items():
        data, status = fetch_series(sid)
        if data is None:
            manifest[sid] = {"label": label, "freq": freq, "unit": unit,
                             "status": f"FAILED_HTTP{status}", "n": 0}
            print(f"[!] {sid:16s} FAILED HTTP {status}")
            continue
        dates = sorted(data)
        path = os.path.join(RAW_DIR, f"{sid}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
        manifest[sid] = {
            "label": label, "freq": freq, "unit": unit,
            "status": "ok", "n": len(data),
            "start": dates[0], "end": dates[-1],
            "last_value": data[dates[-1]],
        }
        print(f"[ok] {sid:16s} n={len(data):6d} {dates[0]} → {dates[-1]} "
              f"last={data[dates[-1]]}")
        time.sleep(0.2)  # politeness

    with open(os.path.join(RAW_DIR, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    n_ok = sum(1 for m in manifest.values() if m["status"] == "ok")
    print(f"\nDONE: {n_ok}/{len(manifest)} seri OK. "
          f"Raw disimpan di {RAW_DIR}/")


if __name__ == "__main__":
    main()
