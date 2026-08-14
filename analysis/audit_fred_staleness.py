#!/usr/bin/env python3
"""
audit_fred_staleness.py — FRED series freshness audit (China M2 pattern)
=======================================================================
The China-M2 bug (MYAGM2CNM189N frozen at 2019-08) showed that a FRED series can
RESOLVE (HTTP 200, valid title) yet have observations frozen years in the past,
and a `is not None` guard silently computes from stale data. This script audits
every FRED series the SFC pipeline depends on and reports how FRESH the newest
observation is relative to today.

Reusable: add/remove series as the pipeline evolves.

USAGE:
    cd ~/sfc && export FRED_API_KEY=$(grep -oP '(?<=FRED_API_KEY=).*' .env | tr -d '"')
    .venv/bin/python analysis/audit_fred_staleness.py
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

KEY = os.getenv("FRED_API_KEY", "")
if not KEY:
    print("FRED_API_KEY not set — abort.")
    sys.exit(1)

# series -> (description, expected_frequency_days) where expected_frequency_days
# is the max reasonable age before we flag it (weekly ~14d, monthly ~45d, daily ~3d).
SERIES = {
    # GLF liquidity components
    "WALCL": ("Fed total assets", 14),
    "ECBASSETSW": ("ECB total assets", 14),
    "JPNASSETS": ("BOJ total assets (monthly)", 45),
    "M2SL": ("US M2 (monthly)", 45),
    "MYAGM2CNM189N": ("China M2 — INACTIVE (replaced by chinadata.live)", 45),
    "WTREGEN": ("TGA balance", 14),
    "RRPONTSYD": ("Reverse repo facility", 14),
    "DTWEXBGS": ("DXY broad index", 14),
    # Expectations (L6)
    "CPIAUCSL": ("CPI (all urban, monthly)", 45),
    "T10YIE": ("10Y breakeven inflation", 14),
    "DGS10": ("10Y Treasury", 14),
    "T10Y2Y": ("10Y-2Y curve", 14),
    "UNRATE": ("Unemployment (monthly)", 45),
    # Other model inputs
    "FEDFUNDS": ("Fed funds rate (monthly)", 45),
    "DGS2": ("2Y Treasury", 14),
    "GDPC1": ("Real GDP (quarterly)", 140),
    "GFDEGDQ188S": ("Federal debt/GDP (quarterly)", 140),
    "BAMLH0A0HYM2": ("High-yield spread", 14),
    "TOTLL": ("Bank credit (monthly, mediation test)", 45),
    "CBBTCUSD": ("BTC price (Coindesk)", 3),
}


def latest_date(sid):
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
           f"&api_key={KEY}&file_type=json&sort_order=desc&limit=2")
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            data = json.loads(r.read())
        obs = [o for o in data.get("observations", []) if o["value"] != "."]
        if not obs:
            return None, "no-data"
        return obs[0]["date"], None
    except Exception as e:
        return None, str(e)[:60]


def main():
    print("=" * 78)
    print("FRED STALENESS AUDIT — newest observation vs today")
    print("=" * 78)
    now = datetime.now(timezone.utc)
    rows = []
    for sid, (desc, max_age) in SERIES.items():
        d, err = latest_date(sid)
        if err:
            rows.append((sid, desc, None, None, err))
            print(f"  {sid:<16} {desc:<38} ERROR: {err}")
            continue
        ld = datetime.fromisoformat(d + "T00:00:00+00:00")
        age = (now - ld).days
        flag = "STALE" if age > max_age else ("WARN" if age > max_age * 0.7 else "OK")
        print(f"  {sid:<16} {desc:<38} latest={d}  age={age:>4}d  [{flag}]")
        rows.append((sid, desc, d, age, flag))
    print("=" * 78)
    stale = [r for r in rows if r[4] == "STALE"]
    warn = [r for r in rows if r[4] == "WARN"]
    print(f"TOTAL={len(rows)}  OK={len(rows)-len(stale)-len(warn)}  WARN={len(warn)}  STALE={len(stale)}")
    if stale:
        print("\nSTALE (needs action):")
        for sid, desc, d, age, _ in stale:
            print(f"  - {sid} ({desc}) latest {d}, {age}d old")
    # save
    out = {"generated_at": now.isoformat(), "rows": [
        {"series": sid, "desc": d, "latest": dd, "age_days": a, "flag": f}
        for sid, d, dd, a, f in rows]}
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".fred_staleness_audit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> .fred_staleness_audit.json")


if __name__ == "__main__":
    main()
