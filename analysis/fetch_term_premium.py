#!/usr/bin/env python3
"""
fetch_term_premium.py — pull REAL term-premium estimates (not proxies):
  * Kim-Wright (FRED: THREEFYTP10 / THREEFYTP5) — daily, 1990+
  * ACM 10y (NY Fed spreadsheet, if reachable)
Saves data/term_premium_daily.json  {date: {"KW10":..,"KW5":..,"ACM10":..}}
"""
import json, os, sys, datetime, io
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data/term_premium_daily.json")


def fred_key():
    for line in open(os.path.join(REPO, ".env")):
        if line.startswith("FRED_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fred_series(sid, key, start="1990-01-01"):
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={sid}&api_key={key}&file_type=json&observation_start={start}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    return {o["date"]: float(o["value"]) for o in obs if o["value"] != "."}


def fetch_acm():
    """NY Fed ACM term premium spreadsheet."""
    url = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print(f"  ACM: HTTP {r.status_code}")
            return {}
        import pandas as pd
        xls = pd.ExcelFile(io.BytesIO(r.content))
        # prefer the daily sheet
        sh = next((s for s in xls.sheet_names if "daily" in s.lower()), xls.sheet_names[0])
        d = xls.parse(sh)
        datecol = d.columns[0]
        acmcol = "ACMTP10" if "ACMTP10" in d.columns else next(
            (c for c in d.columns if str(c).upper().startswith("ACM")), None)
        d[datecol] = pd.to_datetime(d[datecol], errors="coerce")
        out = {}
        for _, row in d.iterrows():
            try:
                out[row[datecol].strftime("%Y-%m-%d")] = float(row[acmcol])
            except Exception:
                continue
        print(f"  ACM: sheet '{sh}' col={acmcol} -> {len(out)} rows "
              f"{min(out) if out else '-'}..{max(out) if out else '-'}")
        return out
    except Exception as e:
        print(f"  ACM: EXC {e}")
        return {}


def main():
    key = fred_key()
    if not key:
        sys.exit("no FRED_API_KEY")
    print("Fetching Kim-Wright from FRED ...")
    kw10 = fred_series("THREEFYTP10", key)
    kw5 = fred_series("THREEFYTP5", key)
    print(f"  KW10: {len(kw10)} obs  {min(kw10)}..{max(kw10)}")
    print(f"  KW5 : {len(kw5)} obs")
    print("Fetching ACM from NY Fed ...")
    acm = fetch_acm()

    all_dates = set(kw10) | set(kw5) | set(acm)
    out = {}
    for d in sorted(all_dates):
        rec = {}
        if d in kw10: rec["KW10"] = kw10[d]
        if d in kw5: rec["KW5"] = kw5[d]
        if d in acm: rec["ACM10"] = acm[d]
        out[d] = rec
    json.dump(out, open(OUT, "w"))
    print(f"\nSaved {OUT}: {len(out)} dates  {min(out)}..{max(out)}")
    has_acm = sum(1 for v in out.values() if "ACM10" in v)
    print(f"  with ACM: {has_acm}  with KW10: {sum(1 for v in out.values() if 'KW10' in v)}")


if __name__ == "__main__":
    main()
