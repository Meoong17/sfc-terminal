#!/usr/bin/env python3
"""
NLE (Native Liquidity Expansion) prototype — display-only research, NOT a scoring driver.

Reimplements the NLE proposal (from /home/ubuntu/C/A.pdf) as a testable daily
series built entirely from FREE sources already available or fetchable:

    NLE = ΔS_30d / S_total          (pillar 1: stablecoin dry powder)
        + ln(E_outflow / E_inflow)  (pillar 2: BTC CEX supply shock)
        + A_active / SMA90(A_active)(pillar 3: network velocity / adoption)

Methodology fixes vs the source document:
  1. Each pillar is point-in-time Z-scored over a trailing window before summing
     (the doc adds three unit-incomparable raw terms with no normalization).
  2. Equal weights for the prototype (the doc gives no weights).
  3. Honest era-split predictive test: does NLE (or any pillar) predict forward
     BTC returns 7d/30d out-of-sample? Quantile top/bottom-20% gap + bootstrap
     P(sign) + Spearman IC, split into 3 contiguous eras. Verdict from the
     LATEST era, per project validation rule.

Data sources (all free):
  - Coin Metrics community API: CapMrktCurUSD (USDT/USDC/DAI) -> pillar 1;
    AdrActCnt (btc) -> pillar 3.  No key. Paginated.
  - data/binance_vision_daily.json (canonical BTC price) -> forward returns.
  - .onchain_cache.json (ErcinDedeoglu/crypto-market-data via onchain_fetch.py)
    -> btc_exchange_netflow / inflow_total / outflow_total for pillar 2.

Coverage: 2022-12-03 -> now (~3.7y, n~1352). Cannot test era1(2014-18)/early era2.

Output: data/nle_daily.json (display-only series) + analysis/.nle_prototype_summary.json
"""

import json, os, math, random, sys
from datetime import datetime, timezone, timedelta
import urllib.request

import numpy as np

SFC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_CACHE = os.path.join(SFC, "data", "nle_coinmetrics.json")
PRICE = os.path.join(SFC, "data", "binance_vision_daily.json")
ONCHAIN = os.path.join(SFC, ".onchain_cache.json")
NLE_OUT = os.path.join(SFC, "data", "nle_daily.json")
SUM_OUT = os.path.join(SFC, "analysis", ".nle_prototype_summary.json")

START = "2022-10-01"   # 30d lookback before first netflow point (2022-12-03)
SEED = 42


def _get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise


def cm_series(asset, metric):
    """Fetch full daily series for one asset/metric (paginated). Returns dict date->float."""
    cache = {}
    if os.path.exists(CM_CACHE):
        cache = json.load(open(CM_CACHE))
    key = f"{asset}|{metric}"
    if key in cache:
        return cache[key]
    out = {}
    url = (f"{CM_BASE}?assets={asset}&metrics={metric}&frequency=1d"
           f"&start_time={START}&page_size=10000")
    while url:
        j = _get(url)
        for row in j.get("data", []):
            d = row["time"][:10]
            v = row.get(metric)
            if v is not None:
                out[d] = float(v)
        url = j.get("next_page_url")
    cache[key] = out
    json.dump(cache, open(CM_CACHE, "w"))
    return out


def build_panel():
    # ---- BTC price (close) ----
    price = {d: v["close"] for d, v in json.load(open(PRICE)).items()}

    # ---- stablecoin mcap (pillar 1) ----
    usdt = cm_series("usdt", "CapMrktCurUSD")
    usdc = cm_series("usdc", "CapMrktCurUSD")
    dai = cm_series("dai", "CapMrktCurUSD")
    stables = {d: usdt.get(d, 0) + usdc.get(d, 0) + dai.get(d, 0)
               for d in set(usdt) | set(usdc) | set(dai)}

    # ---- active addresses (pillar 3) ----
    act = cm_series("btc", "AdrActCnt")

    # ---- exchange flows (pillar 2) from onchain cache ----
    oc = json.load(open(ONCHAIN))["raw"]

    def oc_series(k):
        raw = oc[k]
        arr = raw.get("data", []) if isinstance(raw, dict) else raw
        return {datetime.utcfromtimestamp(p["timestamp"] / 1000).strftime("%Y-%m-%d"): p["value"]
                for p in arr}

    inflow = oc_series("exchange_inflow_total")
    outflow = oc_series("exchange_outflow_total")
    netflow = oc_series("exchange_netflow")

    dates = sorted(set(price) & set(netflow))
    return dates, price, stables, act, inflow, outflow, netflow


def add_pillars(dates, price, stables, act, inflow, outflow, netflow):
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    p1 = np.full(n, np.nan)   # (S_t - S_t-30)/S_t
    p2 = np.full(n, np.nan)   # ln(out/in)
    p3 = np.full(n, np.nan)   # A / SMA90(A)
    ret7 = np.full(n, np.nan)
    ret30 = np.full(n, np.nan)
    close = np.array([price[d] for d in dates])

    for i, d in enumerate(dates):
        # pillar 1
        if d in stables:
            s_now = stables[d]
            j30 = i - 30
            if j30 >= 0:
                d30 = dates[j30]
                if d30 in stables and s_now > 0:
                    p1[i] = (s_now - stables[d30]) / s_now
        # pillar 2
        if d in inflow and d in outflow and inflow[d] > 0 and outflow[d] > 0:
            p2[i] = math.log(outflow[d] / inflow[d])
        # pillar 3
        if d in act:
            a = act[d]
            lo = max(0, i - 89)
            wins = [act[dates[k]] for k in range(lo, i + 1) if dates[k] in act]
            if len(wins) >= 60:
                mu = float(np.mean(wins))
                if mu > 0:
                    p3[i] = a / mu
        # forward returns
        if i + 7 < n:
            ret7[i] = close[i + 7] / close[i] - 1
        if i + 30 < n:
            ret30[i] = close[i + 30] / close[i] - 1

    return p1, p2, p3, ret7, ret30


def ptz_score(x, window=365):
    """Point-in-time z-score over trailing window (no lookahead)."""
    z = np.full(len(x), np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        lo = max(0, i - window + 1)
        w = x[lo:i]  # exclude current
        w = w[~np.isnan(w)]
        if len(w) >= 60:
            mu, sd = float(np.mean(w)), float(np.std(w))
            if sd > 0:
                z[i] = (x[i] - mu) / sd
    return z


def forward_signal_metrics(sig, ret, h):
    """Quantile top/bottom-20% gap of forward return + bootstrap P(sign) + Spearman IC."""
    mask = ~(np.isnan(sig) | np.isnan(ret))
    s, r = sig[mask], ret[mask]
    if len(s) < 40:
        return None
    order = np.argsort(s)
    s_s, r_s = s[order], r[order]
    k = max(1, int(len(s) * 0.20))
    bottom, top = r_s[:k], r_s[-k:]
    gap = float(np.mean(top) - np.mean(bottom))   # NLE high -> higher return -> positive
    rng = np.random.default_rng(SEED)
    boots = np.empty(2000)
    idx_b = rng.integers(0, len(bottom), size=(2000, len(bottom)))
    idx_t = rng.integers(0, len(top), size=(2000, len(top)))
    boots = np.mean(top[idx_t], axis=1) - np.mean(bottom[idx_b], axis=1)
    p_sign = float(np.mean(boots > 0))            # P(gap>0)
    ic = float(np.corrcoef(s, r)[0, 1]) if np.std(s) > 0 and np.std(r) > 0 else float("nan")
    return {"n": int(len(s)), "gap_pp": round(gap * 100, 2),
            "P_gap_pos": round(p_sign, 3), "ic": round(ic, 3)}


def main():
    random.seed(SEED)
    dates, price, stables, act, inflow, outflow, netflow = build_panel()
    p1, p2, p3, ret7, ret30 = add_pillars(dates, price, stables, act, inflow, outflow, netflow)
    n = len(dates)

    z1, z2, z3 = ptz_score(p1), ptz_score(p2), ptz_score(p3)
    nle = z1 + z2 + z3

    # ---- save display-only series ----
    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "source": "NLE prototype (display-only). Free data: Coin Metrics + binance_vision + ErcinDedeoglu onchain.",
           "coverage": [dates[0], dates[-1]], "n": n,
           "definition": "NLE = z(ΔS30/S) + z(ln(out/in)) + z(A/SMA90(A)); point-in-time z over 365d",
           "data": []}
    for i, d in enumerate(dates):
        out["data"].append({
            "date": d,
            "stablecoin_z": None if math.isnan(z1[i]) else round(float(z1[i]), 4),
            "netflow_z": None if math.isnan(z2[i]) else round(float(z2[i]), 4),
            "velocity_z": None if math.isnan(z3[i]) else round(float(z3[i]), 4),
            "nle": None if math.isnan(nle[i]) else round(float(nle[i]), 4),
        })
    json.dump(out, open(NLE_OUT, "w"), indent=1)

    print(f"NLE series: n={n}  {dates[0]} .. {dates[-1]}  -> {NLE_OUT}")

    # ---- era split (3 contiguous blocks) ----
    q1 = n // 3
    eras = {"era1": (0, q1), "era2": (q1, 2 * q1), "era3": (2 * q1, n)}
    era_labels = {}
    for k, (a, b) in eras.items():
        era_labels[k] = f"{dates[a]}..{dates[b - 1]}"

    signals = {"NLE": nle, "stablecoin_z": z1, "netflow_z": z2, "velocity_z": z3}

    rows = []
    for name, sig in signals.items():
        for h, ret, hl in [(7, ret7, "7d"), (30, ret30, "30d")]:
            full = forward_signal_metrics(sig, ret, hl)
            if full is None:
                continue
            era_rows = {}
            for k, (a, b) in eras.items():
                r = forward_signal_metrics(sig[a:b], ret[a:b], hl)
                if r:
                    era_rows[k] = r
            rows.append({"signal": name, "horizon": hl, "full": full,
                         "eras": era_rows, "era_labels": era_labels})

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "result": rows}
    json.dump(summary, open(SUM_OUT, "w"), indent=1)

    # ---- human-readable printout ----
    print("\nCoverage per era: " + ", ".join(f"{k}: {v}" for k, v in era_labels.items()))
    for r in rows:
        print(f"\n== {r['signal']} @ {r['horizon']} ==")
        f = r["full"]
        print(f"  FULL: n={f['n']} gap={f['gap_pp']}pp  P(gap>0)={f['P_gap_pos']}  IC={f['ic']}")
        for k in ["era1", "era2", "era3"]:
            e = r["eras"].get(k)
            if e:
                print(f"  {k}: n={e['n']} gap={e['gap_pp']}pp  P(gap>0)={e['P_gap_pos']}  IC={e['ic']}")

    print(f"\nSummary -> {SUM_OUT}")


if __name__ == "__main__":
    main()
