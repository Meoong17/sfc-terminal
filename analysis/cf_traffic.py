#!/usr/bin/env python3
"""
cf_traffic.py — SFC Terminal (sfcterminal.xyz) traffic report via Cloudflare GraphQL Analytics.

Pulls zone-level analytics for the SFC dashboard domain and prints:
  - daily requests / cached / unique visitors / threats / bytes (last 15 days)
  - hourly requests for today

Usage:
    python3 analysis/cf_traffic.py

Requires a Cloudflare API token with Zone Analytics Read. It reads the token from
the SFC repo .env (CLOUDFLARE_API_TOKEN). If you run it elsewhere, set the token
via a CLOUDFLARE_API_TOKEN env var or edit the ZONE id below.

Note:
  - sfcterminal.xyz is served by Cloudflare (Worker SPA) and proxies /snapshot &
    /data.json to the origin tunnel. These zone analytics are the authoritative
    visitor counts; the box's nginx access.log only sees the tunnel/polling noise.
  - The current token is bound to an Access policy, so the adaptive dataset
    (status-code / top-path / top-country breakdown) is NOT available unless a
    fresh token WITHOUT an Access policy + Account Analytics Read is used.
"""
import json
import os
import urllib.request
from datetime import date

ZONE = "24d18163bbc430aa6dd42cd16b7b66c6"  # sfcterminal.xyz


def load_token():
    tok = os.environ.get("CLOUDFLARE_API_TOKEN")
    if tok:
        return tok
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    for line in open(env_path):
        line = line.strip()
        if line.startswith("CLOUDFLARE_API_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("CLOUDFLARE_API_TOKEN not found in .env or env")


def gql(token, query):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def main():
    token = load_token()

    q = """query { viewer { zones(filter: {zoneTag: "__ZONE__"}) {
      httpRequests1dGroups(limit: 15, filter: {date_geq: "2026-08-04", date_leq: "2026-08-18"}) {
        dimensions { date }
        sum { requests cachedRequests threats bytes }
        uniq { uniques }
      }
    } } }""".replace("__ZONE__", ZONE)
    d = gql(token, q)
    if not d.get("data"):
        raise SystemExit("ERR: " + json.dumps(d.get("errors"))[:600])
    rows = d["data"]["viewer"]["zones"][0]["httpRequests1dGroups"]
    rows.sort(key=lambda g: g["dimensions"]["date"])
    t = sum(g["sum"]["requests"] for g in rows)
    tu = sum(g["uniq"]["uniques"] for g in rows)
    tt = sum(g["sum"]["threats"] for g in rows)
    tb = sum(g["sum"]["bytes"] for g in rows)
    print("TRAFIK HARIAN sfcterminal.xyz (15 hari)")
    print(f"{'tanggal':<12}{'req':>7}{'cached':>7}{'uniq':>7}{'threat':>8}{'MB':>7}")
    for g in rows:
        s = g["sum"]
        print(f"{g['dimensions']['date']:<12}{s['requests']:>7}{s['cachedRequests']:>7}"
              f"{g['uniq']['uniques']:>7}{s['threats']:>8}{s['bytes']/1e6:>7.1f}")
    print("-" * 48)
    print(f"{'TOTAL':<12}{t:>7}{'':>7}{tu:>7}{tt:>8}{tb/1e6:>7.1f}")
    print(f"rata2 visitor unik/hari: {tu/len(rows):.0f} | "
          f"median req/hari: {sorted(x['sum']['requests'] for x in rows)[len(rows)//2]}")

    today = date.today().isoformat()
    q2 = """query { viewer { zones(filter: {zoneTag: "__ZONE__"}) {
      httpRequests1hGroups(limit: 24, filter: {date_geq: "__TODAY__", date_leq: "__TODAY__"}) {
        dimensions { datetimeHour }
        sum { requests }
        uniq { uniques }
      }
    } } }""".replace("__ZONE__", ZONE).replace("__TODAY__", today)
    d2 = gql(token, q2)
    if d2.get("data"):
        h = d2["data"]["viewer"]["zones"][0]["httpRequests1hGroups"]
        h.sort(key=lambda g: g["dimensions"]["datetimeHour"])
        print(f"\nPER JAM HARI INI ({today})")
        for g in h:
            print(f"  {g['dimensions']['datetimeHour'][11:16]}  "
                  f"req={g['sum']['requests']:<4} uniq={g['uniq']['uniques']}")


if __name__ == "__main__":
    main()
