#!/usr/bin/env python3
"""
validate_orderflow.py — sanity & descriptive validation of the order-flow daily series.

Reads data/binance_orderflow_daily.json (from fetch_orderflow.py) and cross-checks
against the canonical data/binance_vision_daily.json:

  1. Coverage: day range, expected days, gaps.
  2. Cross-check: aggTrades total_qty vs kline volume (should match ~exactly);
     price_close vs kline close.
  3. Descriptive: taker_imbalance_*, taker_buy_ratio, whale counts, notional,
     trade count by month and by era (2018-20 is era1 tail / pre-era2).
  4. Era-split descriptive (2018-2020 window): does order-flow intensity/scale
     evolve in a sane way (volume/trade growth, whale density)?
"""
import json, os
import numpy as np

REPO = "/home/ubuntu/sfc"
OF = os.path.join(REPO, "data", "binance_orderflow_daily.json")
KD = os.path.join(REPO, "data", "binance_vision_daily.json")


def main():
    of = json.load(open(OF))
    kd = json.load(open(KD))
    days = sorted(of)
    n = len(days)
    print(f"order-flow days: {n}  ({days[0]} .. {days[-1]})")

    # 1. coverage / gaps
    from datetime import datetime, timedelta
    d0, d1 = datetime.strptime(days[0], "%Y-%m-%d"), datetime.strptime(days[-1], "%Y-%m-%d")
    span = (d1 - d0).days + 1
    missing = span - n
    print(f"expected span days: {span}, missing: {missing}")
    gap_run = 0; maxgap = 0; prev = None
    for d in days:
        cur = datetime.strptime(d, "%Y-%m-%d")
        if prev is not None:
            g = (cur - prev).days - 1
            maxgap = max(maxgap, g)
            if g > 0: gap_run += 1
        prev = cur
    print(f"date gaps (>1d): {gap_run}, longest gap: {maxgap}d")

    # 2. cross-check vs canonical kline
    qty = np.array([of[d]["total_qty"] for d in days])
    kvol = np.array([(kd[d].get("volume") if d in kd else np.nan) for d in days])
    p_of = np.array([(of[d]["price_close"] if of[d]["price_close"] is not None else np.nan) for d in days], dtype=float)
    p_k = np.array([(kd[d].get("close") if d in kd else np.nan) for d in days])
    ok = ~np.isnan(kvol) & (kvol > 0)
    rel = np.abs(qty[ok] - kvol[ok]) / kvol[ok]
    print(f"\ncross-check total_qty vs kline volume: n={ok.sum()}, "
          f"max rel diff={np.nanmax(rel):.6f}, mean={np.nanmean(rel):.2e}")
    pok = ~np.isnan(p_k) & (p_k > 0)
    prel = np.abs(p_of[pok] - p_k[pok]) / p_k[pok]
    print(f"cross-check price_close vs kline close: n={pok.sum()}, "
          f"max rel diff={np.nanmax(prel):.6f}")

    # 3. descriptive stats
    imb = np.array([(of[d]["taker_imbalance_qty"] if of[d]["taker_imbalance_qty"] is not None else np.nan) for d in days])
    br = np.array([(of[d]["taker_buy_ratio"] if of[d]["taker_buy_ratio"] is not None else np.nan) for d in days])
    nt = np.array([of[d]["n_trades"] for d in days])
    tq = np.array([of[d]["total_quote"] for d in days])
    w10 = np.array([of[d]["whale_qty_lo_count"] for d in days])
    w100 = np.array([of[d]["whale_qty_hi_count"] for d in days])
    print("\n--- full-window descriptive (2018-2020) ---")
    for name, a in [("taker_imbalance_qty", imb), ("taker_buy_ratio", br),
                    ("n_trades", nt), ("total_quote($M)", tq/1e6),
                    ("whale>=10BTC", w10), ("whale>=100BTC", w100)]:
        print(f"  {name:18s} mean={np.nanmean(a):10.3f}  median={np.nanmedian(a):10.3f}  "
              f"min={np.nanmin(a):10.3f}  max={np.nanmax(a):10.3f}")
    # imbalance sign days
    print(f"  taker_imbalance >0 (buy-heavy) days: {np.nansum(imb>0)}/{n} "
          f"({np.nansum(imb>0)/n:.1%})")

    # 4. monthly table + year split
    print("\n--- by year ---")
    for yr in ["2018", "2019", "2020"]:
        m = [i for i, d in enumerate(days) if d.startswith(yr)]
        if not m: continue
        sub = np.array(m)
        print(f"  {yr}: n={len(sub)} | avg total_quote=${tq[sub].mean()/1e6:,.0f}M/d | "
              f"avg n_trades={nt[sub].mean():,.0f} | avg whale10/d={w10[sub].mean():.1f} | "
              f"imbalance mean={imb[sub].mean():+.4f}")

    # monthly notional series (growth sanity)
    print("\n--- monthly average daily quote (2018 vs 2019 vs 2020 first/last) ---")
    for ym in ["2018-01", "2018-12", "2019-06", "2019-12", "2020-03", "2020-12"]:
        m = [i for i, d in enumerate(days) if d.startswith(ym)]
        if m:
            print(f"  {ym}: ${tq[m].mean()/1e6:,.0f}M/d, {nt[m].mean():,.0f} trades/d")


if __name__ == "__main__":
    main()
