#!/usr/bin/env python3
"""
binance_features.py — derived features from the Binance Vision multi-timeframe cache.

Consumes data/binance_vision_{daily,weekly,monthly}.json (from fetch_binance_vision.py)
and computes canonical, point-in-time-safe features for validation/backtest:

  daily/weekly/monthly:
    ret_N           : forward N-period log return (label) — canonical for OOS eval
    mom_N           : trailing N-period return (feature)
    rvol_N          : realized volatility = std(daily log returns) over N days, annualized
    vol_N           : average daily quote volume over N days (USD)
    taker_ratio     : taker_buy_quote / quote_volume (buy-side pressure)
    premium         : funding-linked basis (from premiumIndex klines)
    funding_mean    : mean funding rate that day

Point-in-time safe: ret_* look FORWARD (labels), everything else is trailing and
uses only data up to t. Realized vol uses rolling window of PAST returns.

Usage (import):
    from data_sources.binance_features import load_daily, add_forward_returns
    df = load_daily(); df = add_forward_returns(df, [7,30,90,180,365])
"""
import os, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
DAILY = os.path.join(DATA, "binance_vision_daily.json")
WEEKLY = os.path.join(DATA, "binance_vision_weekly.json")
MONTHLY = os.path.join(DATA, "binance_vision_monthly.json")

FIELDS = ["open", "high", "low", "close", "volume", "quote_vol",
          "taker_base", "taker_quote", "funding_last", "funding_mean",
          "index", "mark", "premium"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _to_series(d):
    days = sorted(d)
    arr = {k: np.array([(d[day].get(k) if day in d else np.nan) for day in days],
                        dtype=float) for k in FIELDS}
    arr["day"] = days
    return arr, days


def load_daily():
    return _load(DAILY)


def add_forward_returns(rec, keys, horizons=(7, 30, 90, 180, 365)):
    """Add forward log-return labels. `keys` = sorted date->{close} dict."""
    dates = sorted(keys)
    closes = np.array([keys[d]["close"] for d in dates])
    n = len(dates)
    out = {}
    for h in horizons:
        fwd = np.full(n, np.nan)
        for i in range(n - h):
            fwd[i] = np.log(closes[i + h]) - np.log(closes[i])
        out[f"ret_{h}"] = fwd
    out["close"] = closes
    out["dates"] = dates
    return out


def compute_features(daily, horizons=(7, 30, 90, 180, 365), rvol_windows=(7, 30, 90)):
    """Return dict of aligned numpy arrays + date index, features only (no lookahead
    except ret_* which are labels)."""
    arr, days = _to_series(daily)
    closes = arr["close"]
    qvol = arr["quote_vol"]
    taker_q = arr["taker_quote"]
    n = len(days)
    lr = np.diff(np.log(closes))
    lr = np.concatenate([[np.nan], lr])
    out = {"days": days, "close": closes}

    # trailing momentum & realized vol (use only past data)
    for h in horizons:
        mom = np.full(n, np.nan)
        for i in range(h, n):
            mom[i] = np.log(closes[i]) - np.log(closes[i - h])
        out[f"mom_{h}"] = mom
        fwd = np.full(n, np.nan)
        for i in range(n - h):
            fwd[i] = np.log(closes[i + h]) - np.log(closes[i])
        out[f"ret_{h}"] = fwd
    for w in rvol_windows:
        rv = np.full(n, np.nan)
        for i in range(w, n):
            rv[i] = np.nanstd(lr[i - w + 1:i + 1]) * np.sqrt(365)
        out[f"rvol_{w}"] = rv
        vol = np.full(n, np.nan)
        for i in range(w, n):
            vol[i] = np.nanmean(qvol[i - w + 1:i + 1])
        out[f"vol_{w}"] = vol
    # taker ratio
    tr = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(qvol > 0, taker_q / qvol, np.nan)
    out["taker_ratio"] = tr
    out["premium"] = arr["premium"]
    out["funding"] = arr["funding_last"]
    return out


if __name__ == "__main__":
    daily = load_daily()
    feat = compute_features(daily)
    n = len(feat["days"])
    print(f"daily features: {n} days "
          f"({feat['days'][0]} -> {feat['days'][-1]})")
    for h in [7, 30, 90, 365]:
        print(f"  ret_{h}: {np.nanmean(feat[f'ret_{h}']):+.4f}  "
              f"(n={np.sum(~np.isnan(feat[f'ret_{h}']))})")
    for w in [7, 30, 90]:
        print(f"  rvol_{w}: mean={np.nanmean(feat[f'rvol_{w}']):.3f}  "
              f"vol_{w}: {np.nanmean(feat[f'vol_{w}'])/1e9:.2f}B")
    print(f"  taker_ratio mean: {np.nanmean(feat['taker_ratio']):.4f}")
    print(f"  premium (basis) mean: {np.nanmean(feat['premium']):.6f}")

    # weekly/monthly sanity
    wk = _load(WEEKLY); mo = _load(MONTHLY)
    print(f"\nweekly : {len(wk)} rows ({min(wk)} -> {max(wk)})")
    print(f"monthly: {len(mo)} rows ({min(mo)} -> {max(mo)})")
