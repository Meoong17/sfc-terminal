#!/usr/bin/env python3
"""
live_context.py — place CURRENT SFC live conditions against the 9-year historical
distribution (from canonical Binance cache). Outputs z-score + percentile so the
dashboard/decisions get "how extreme is today?" context.

Pure analysis. Reads live data.json + canonical cache.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def percentile_hist(cur, hist):
    hist = hist[~np.isnan(hist)]
    if len(hist) == 0 or np.isnan(cur):
        return None
    return float(np.mean(hist <= cur) * 100.0)


def zscore(cur, hist):
    hist = hist[~np.isnan(hist)]
    if len(hist) == 0 or np.isnan(cur):
        return None
    sd = hist.std()
    if sd == 0:
        return 0.0
    return float((cur - hist.mean()) / sd)


def main():
    # live SFC snapshot
    try:
        live = json.load(open(os.path.join(REPO, "data.json")))
    except Exception as e:
        print("cannot read data.json:", e); return
    btc_live = live.get("btc")
    cc = live.get("confidence_components") or {}
    funding_live = cc.get("funding_imbalance")
    dvol_live = live.get("dvol")

    daily = load_daily()
    feat = compute_features(daily)
    days = np.array(feat["days"])

    print(f"live SFC snapshot ts={live.get('ts','?')[:16]}  btc={btc_live}  "
          f"dvol={dvol_live}  funding_imbalance={funding_live}")
    print(f"canonical history: {days[0]} -> {days[-1]} ({len(days)} days). "
          f"NOTE: cache ends {days[-1]} (Binance monthly lag) — 'live' values "
          f"below use latest available unless taken from SFC.\n")

    rows = []
    # 1. Price level percentile (live btc vs all historical closes)
    hist_close = feat["close"]
    if btc_live:
        rows.append(("BTC price ($)", btc_live, hist_close,
                     "price now vs 9yr close range"))
    # 2. Realized vol 30d (latest) vs distribution
    rows.append(("Realized vol 30d", feat["rvol_30"][-1], feat["rvol_30"],
                 "annualized vol (latest cache day)"))
    # 3. Funding (live SFC, else latest binance) vs 9yr funding
    fund_hist = feat["funding"]
    cur_fund = funding_live if funding_live is not None else fund_hist[-1]
    rows.append(("Funding (imbalance)", cur_fund, fund_hist,
                 "live SFC funding_imbalance (else last binance)"))
    # 4. Volume 30d avg vs distribution
    rows.append(("Avg daily vol 30d ($B)", feat["vol_30"][-1] / 1e9,
                 feat["vol_30"] / 1e9, "latest 30d mean quote vol (USD B)"))
    # 5. Momentum 30d / 90d (latest) vs distribution
    rows.append(("Momentum 30d", feat["mom_30"][-1], feat["mom_30"],
                 "trailing 30d log-return"))
    rows.append(("Momentum 90d", feat["mom_90"][-1], feat["mom_90"],
                 "trailing 90d log-return"))
    # 6. Premium / basis (latest) vs distribution
    rows.append(("Premium/basis", feat["premium"][-1], feat["premium"],
                 "futures premium index (funding-linked)"))

    print(f"{'Metric':<22}{'cur':>12}{'mean':>10}{'std':>10}{'z':>8}{'pctile':>9}  note")
    print("-" * 90)
    for name, cur, hist, note in rows:
        z = zscore(cur, hist)
        p = percentile_hist(cur, hist)
        cz = f"{z:+.2f}" if z is not None else "NA"
        cp = f"{p:.1f}%" if p is not None else "NA"
        # extreme flags
        flag = ""
        if p is not None:
            if p >= 95: flag = "  <<< EXTREME HIGH"
            elif p <= 5: flag = "  <<< EXTREME LOW"
        print(f"{name:<22}{cur:>12,.4g}{np.nanmean(hist):>10,.4g}"
              f"{np.nanstd(hist):>10,.4g}{cz:>8}{cp:>9}{flag}  {note}")

    print("\nInterpretasi: pctile = persentase hari historis yang nilainya <= saat ini.")
    print("EXTREME (pctile>=95 atau <=5) = kondisi di ekor distribusi 9 tahun.")


if __name__ == "__main__":
    main()
