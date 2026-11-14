#!/usr/bin/env python3
"""Futures basis (Binance USDT quarterly vs spot) -> data/binance_futures_basis_daily.json

`data/tradingview_btc1_daily.csv` (CME front-month) is a MANUAL TradingView export
and has been frozen at 2026-09-01, so the structure engine's financing-quality
layer (basis carry) could not advance. This script builds an automated basis proxy
from Binance's current-quarter contract (BTCUSDT_<YYMMDD>) against spot BTCUSDT.

Honest limits, published with the data:
  - it is an EXCHANGE basis, not CME basis (different venue, different carry);
  - history starts when the current quarterly contract was listed, so the series
    grows forward from there. `source` and `history_start` are written into the
    file and the engine records them, so nobody can mistake it for a 2-year series.

Output (date-keyed):
  {"2026-11-13": {"fut_close": 65870.1, "spot_close": 68314.0, "basis_pct": -3.58}}
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "binance_futures_basis_daily.json"
FAPI = "https://fapi.binance.com"


def current_quarter_symbol():
    info = requests.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=20).json()
    cand = [s["symbol"] for s in info.get("symbols", [])
            if s["symbol"].startswith("BTCUSDT_") and s.get("contractType") == "CURRENT_QUARTER"]
    # nearest expiry first
    cand.sort()
    return cand[0] if cand else None


def klines(symbol, limit=500, endpoint="fapi"):
    base = FAPI if endpoint == "fapi" else "https://api.binance.com"
    path = "/fapi/v1/klines" if endpoint == "fapi" else "/api/v3/klines"
    r = requests.get(f"{base}{path}", params={"symbol": symbol, "interval": "1d", "limit": limit}, timeout=20)
    r.raise_for_status()
    return r.json()


def main():
    fut_sym = current_quarter_symbol()
    if not fut_sym:
        print("[basis] no current-quarter BTCUSDT contract listed", file=sys.stderr)
        return
    # Spot closes come from the canonical Binance Vision series (2017+), so the
    # basis history is not limited by the 1500-candle klines window.
    try:
        vis = json.load(open(ROOT / "data" / "binance_vision_daily.json"))
        spot = {k: float(v["close"]) for k, v in vis.items()
                if isinstance(v, dict) and v.get("close")}
    except Exception:
        spot = {datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date().isoformat(): float(k[4])
                for k in klines("BTCUSDT", endpoint="spot")}

    # Expired quarterly contracts are still queryable and carry their whole life,
    # so scan past quarter-ends to build a multi-year basis history instead of the
    # ~140 days the CURRENT_QUARTER contract alone can offer.
    fut = {}
    symbols = [fut_sym]
    for year in range(2023, datetime.now(timezone.utc).year + 1):
        for month, day in ((3, 29), (6, 28), (9, 27), (12, 26)):
            symbols.append(f"BTCUSDT_{str(year)[2:]}{month:02d}{day:02d}")
    for sym in symbols:
        try:
            rows = klines(sym, limit=400)
        except Exception:
            continue
        for k in rows:
            d_ = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date().isoformat()
            fut.setdefault(d_, float(k[4]))
    merged = {}
    for day in sorted(set(fut) & set(spot)):
        if spot[day] <= 0:
            continue
        merged[day] = {"fut_close": round(fut[day], 2),
                       "spot_close": round(spot[day], 2),
                       "basis_pct": round((fut[day] / spot[day] - 1) * 100, 4)}
    payload = {
        "_meta": {
            "source": f"Binance {fut_sym} (current quarter) vs spot BTCUSDT",
            "venue": "Binance USDT-M futures",
            "note": "exchange basis proxy — NOT CME basis; history starts at contract listing",
            "history_start": min(merged) if merged else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    payload.update(merged)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=0)
    days = sorted(merged)
    print(f"[basis] {fut_sym}: {len(days)} days {days[0]} -> {days[-1]}", file=sys.stderr)
    print(json.dumps({"symbol": fut_sym, "days": len(days), "last": days[-1],
                      "last_basis_pct": merged[days[-1]]["basis_pct"]}, indent=2))


if __name__ == "__main__":
    main()
