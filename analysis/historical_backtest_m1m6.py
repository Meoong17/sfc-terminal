#!/usr/bin/env python3
"""
historical_backtest_m1m6.py — Validate M1-M6 core ensemble against real
historical BTC crises (2018 bear market, March 2020 COVID crash, 2022
Luna/FTX crash), using FRED's CBBTCUSD series for long-history BTC price.

WHY THIS EXISTS:
    Live data_collection.json only accumulates a few days/weeks of real
    observations at a time — an earlier audit found a real window with
    ZERO stress-labeled observations (100% calm), meaning M1-M6 had never
    been tested against an actual crisis in that data. This script fills
    that gap using FREE, LONG-HISTORY data (FRED CBBTCUSD goes back to
    2014-12-01) to replay collect.py's own calculate_sfc_ensemble() logic
    against real historical crashes.

HONEST LIMITATION — this is a SIMPLIFIED replay, not a perfect match to
collect.py's live pipeline:
    Several of score_factors_from_market()'s inputs simply did not exist,
    or aren't available via free historical APIs, for most of this
    window:
      - dom (BTC dominance): not fetched here — pass None (factor skipped)
      - dvol (Deribit DVOL options-implied vol index): Deribit's DVOL
        index only launched ~2021 — pass None throughout
      - pc_oi (put/call ratio): BTC options markets only became liquid
        much more recently than 2014 — pass None throughout
      - onchain_whale/value/buy/market_structure (Q10 signals): no free
        historical API for these identified — pass None throughout
      - FNG (Fear & Greed Index): alternative.me's index only starts
        ~Feb 2018 — None before that date, real data after
    This means every score computed here reflects a REDUCED factor set
    (price + DXY + M2 + FNG only, missing DVOL/options/on-chain) compared
    to what collect.py computes live today. Given score_factors_from_market()
    fails safe or per-input (see its own `if X is not None:` guards),
    this doesn't crash — it just means the historical replay is
    necessarily less complete than the live system. Treat results as
    "does the CORE price/macro-driven math respond sensibly to real
    crashes" rather than "exact historical sfc_pct reproduction."

    DXY specifically: collect.py's live DXY comes from a real-time-only
    exchange-rate API (open.er-api.com) with no historical endpoint.
    This script substitutes FRED's own DTWEXBGS (Trade Weighted US
    Dollar Index: Broad) as the best available long-history proxy for
    "dollar strength" — same underlying concept, different exact
    calculation methodology than the live pipeline's EUR/JPY/GBP/CAD
    basket. Flagged here explicitly rather than silently substituted.

USAGE (run on your VPS, needs FRED_API_KEY + network access — NOT
runnable in a sandboxed environment without live API access):
    cd ~/sfc
    python3 analysis/historical_backtest_m1m6.py
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import math


# ── Verbatim copies of collect.py's _sigmoid_factor(), score_factors_from_market(),
# and calculate_sfc_ensemble() ──
# collect.py is a top-to-bottom EXECUTING SCRIPT (fetches live APIs,
# writes data.json, etc. at module level) — NOT a side-effect-free
# library module. `from collect import score_factors_from_market` would
# actually RUN the entire live pipeline (real API calls, real data.json
# writes) as an import side effect, which is exactly wrong for a
# standalone historical backtest. These three functions are copied here
# verbatim instead — keep this synchronized with collect.py's versions
# if either changes; a diverged copy would silently backtest against
# stale math.

def _sigmoid_factor(val, center, k=0.15):
    '''Smooth logistic: maps val to [-3, +3] range.
    center = neutral point, k = steepness.
    sigmoid(x) = 6 / (1 + exp(-k*(x-center))) - 3
    '''
    return 6 / (1 + math.exp(-k * (val - center))) - 3


def score_factors_from_market(btc, btc_24h, dom, dvol, fng, pc_oi, m2_yoy, dxy, glo_score=None,
                                onchain_whale=None, onchain_value=None, onchain_buy=None,
                                onchain_market_structure=None, dxy_btc_corr=None):
    """Score 5 factors from market data using smooth sigmoid/logistic functions. Range -3 to +3

    NOTE (synced to collect.py v4.0.0, 2026-07-25): the direct m2_yoy sigmoid
    was REMOVED from Lt in v4.0.0 — Lt is now driven only by glo_score (GLF,
    the consolidated global-liquidity factor) and btc_24h momentum. This copy
    previously retained a stale `if m2_yoy is not None: factors["Lt"] +=
    _sigmoid_factor(m2_yoy, center=5.0, k=0.8)` line that collect.py no longer
    has; it is removed here to keep this historical replay faithful to the
    current model. (The m2_yoy parameter is still accepted for signature
    compatibility but no longer feeds Lt.)
    """
    factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}

    if glo_score is not None:
        factors["Lt"] += _sigmoid_factor(glo_score, center=50.0, k=0.08)
    if btc_24h is not None:
        factors["Lt"] += _sigmoid_factor(btc_24h, center=0.0, k=0.15)

    if dom is not None:
        factors["St"] += -_sigmoid_factor(dom, center=55.0, k=0.2)
    if pc_oi is not None:
        factors["St"] += -_sigmoid_factor(pc_oi, center=0.8, k=2.0)

    if fng is not None:
        factors["Rt"] = _sigmoid_factor(fng, center=50.0, k=0.08)

    if dvol is not None:
        factors["Ft"] = -_sigmoid_factor(dvol, center=65.0, k=0.06)

    if dxy is not None:
        if dxy_btc_corr is not None and dxy_btc_corr > 0.3:
            factors["Sc"] = _sigmoid_factor(dxy, center=100.0, k=0.2)
        elif dxy_btc_corr is not None and dxy_btc_corr > -0.3:
            factors["Sc"] = -_sigmoid_factor(dxy, center=100.0, k=0.2) * 0.5
        else:
            factors["Sc"] = -_sigmoid_factor(dxy, center=100.0, k=0.2)
    if dom is not None and dom > 65 and (dxy_btc_corr is None or dxy_btc_corr < 0.3):
        factors["Sc"] -= 0.5

    if onchain_whale is not None:
        factors["Rt"] += (onchain_whale - 50) / 50 * 2.0
    if onchain_value is not None:
        factors["Lt"] += (onchain_value - 50) / 50 * 2.0
    if onchain_buy is not None:
        factors["Ft"] += (onchain_buy - 50) / 50 * 1.5
    if onchain_market_structure is not None:
        factors["St"] += (onchain_market_structure - 50) / 50 * 1.5

    for k in factors:
        factors[k] = max(-3.0, min(3.0, factors[k]))
    return factors


def calculate_sfc_ensemble(factors):
    """6-method ensemble — copied verbatim from collect.py's FIXED
    (v2026.07.11 scale-mismatch fix) version. See that function's own
    docstring in collect.py for the full bug history."""
    _FACTOR_WT = {"Lt": 0.66, "St": 1.34, "Rt": 1.0, "Ft": 1.0, "Sc": 1.0}
    norm = {k: v/6 for k, v in factors.items()}
    z_score = sum(factors[k] * _FACTOR_WT[k] for k in factors)

    ns_r = {"Lt":0.35, "St":0.50, "Rt":0.40, "Ft":0.25, "Sc":0.80}
    w = {k:1/v for k,v in ns_r.items()}
    sig = sum((1.0 if norm[k]<-0.333 else 0.7 if norm[k]<-0.167 else 0.3 if norm[k]<0 else 0) * w[k] for k in factors)
    p_klr = max(0.0, min(1.0, sig / sum(w.values())))

    zc = [-1.0, -2.0, -3.0, -4.0, -8.0]
    pc = [0.08, 0.20, 0.55, 0.75, 0.95]
    yc = [math.log(p/(1-p)) for p in pc]
    n_z = len(zc)
    zm = sum(zc)/n_z
    ym = sum(yc)/n_z
    b1 = sum((zc[i]-zm)*(yc[i]-ym) for i in range(n_z)) / sum((z-zm)**2 for z in zc)
    b0 = ym - b1*zm
    z_l = b0 + b1*z_score
    p_logit = 1/(1+math.exp(-z_l))

    prior = 0.08
    odds = prior/(1-prior)
    bayes_mult = [2.5, 2.0, 2.0, 3.0, 1.5]
    for i, k in enumerate(factors):
        if norm[k] < -0.3:
            odds *= bayes_mult[i]
    p_bayes = odds/(1+odds)

    w_ad = {"Lt":0.25, "St":0.20, "Rt":0.20, "Ft":0.30, "Sc":0.05}
    ewc = sum(w_ad[k] * abs(factors[k]) for k in factors)
    p_ewc = ewc/3.0

    qr_anchors = [(-8.0, 0.95), (-5.0, 0.75), (-3.0, 0.50), (-1.5, 0.25), (-0.5, 0.10), (0.5, 0.04), (2.0, 0.01)]
    def quantile_stress(z):
        anchors = sorted(qr_anchors, key=lambda x: x[0])
        if z <= anchors[0][0]: return anchors[0][1]
        if z >= anchors[-1][0]: return anchors[-1][1]
        for i in range(len(anchors)-1):
            z0, p0 = anchors[i]
            z1, p1 = anchors[i+1]
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                return p0 + t * (p1 - p0)
        return 0.04
    p_quantile = quantile_stress(z_score)

    vals = list(norm.values())
    n = len(vals)
    extreme_count = sum(1 for v in vals if v < -0.167)
    severe_count = sum(1 for v in vals if v < -0.333)
    p_extremity = (extreme_count * 0.15 + severe_count * 0.20)
    mean_v = sum(vals) / n
    variance = sum((v - mean_v)**2 for v in vals) / n
    coherence_bonus = 0.10 * (1.0 - variance) if mean_v < -0.083 and variance < 0.12 else 0.0
    ft_val = norm.get("Ft", 0)
    lt_val = norm.get("Lt", 0)
    tail_contribution = (0.15 if ft_val < -0.25 else 0.0) + (0.10 if lt_val < -0.25 else 0.0)
    p_baseline = max(0.0, min((-mean_v) * 0.72, 0.50))
    p_regime = min(p_baseline + p_extremity + coherence_bonus + tail_contribution, 0.99)
    p_regime = max(p_regime, 0.01)

    p_ens = 0.19*p_klr + 0.16*p_logit + 0.12*p_bayes + 0.16*p_ewc + 0.23*p_quantile + 0.14*p_regime

    zone = "CRITICAL" if p_ens > 0.75 else "HIGH" if p_ens > 0.5 else "ELEVATED" if p_ens > 0.25 else "NORMAL"

    return p_ens * 100, zone, factors, norm, p_klr, p_logit, p_bayes, p_ewc, p_quantile * 100, p_regime * 100, 0.0

FRED_KEY = os.getenv("FRED_API_KEY", "")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".m1m6_historical_backtest.json")

# Known real crisis windows to specifically check results against —
# these are well-documented BTC crashes, used here purely as sanity-check
# reference points (did M1-M6 show ELEVATED stress during these,
# compared to calmer surrounding periods), not as ground truth labels.
KNOWN_CRISIS_WINDOWS = {
    "2018 Bear Market Bottom": ("2018-11-01", "2018-12-31"),
    "COVID Crash (Mar 2020)": ("2020-03-08", "2020-03-20"),
    "Luna/UST Collapse (May 2022)": ("2022-05-07", "2022-05-16"),
    "FTX Collapse (Nov 2022)": ("2022-11-06", "2022-11-12"),
}


def fetch_fred_series(series_id, start_date="2014-01-01"):
    """Fetch a full FRED series as {date_str: value}. Same _fred() spirit
    as elsewhere in this codebase, but requesting the FULL history rather
    than the latest-N-observations pattern used for live cycles."""
    if not FRED_KEY:
        print(f"[Backtest] FRED_API_KEY not set — cannot fetch {series_id}", file=sys.stderr)
        return {}
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
               f"&api_key={FRED_KEY}&file_type=json&observation_start={start_date}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        result = {}
        for obs in data.get("observations", []):
            if obs["value"] != ".":
                result[obs["date"]] = float(obs["value"])
        print(f"[Backtest] {series_id}: {len(result)} observations fetched", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[Backtest] {series_id} fetch failed: {e}", file=sys.stderr)
        return {}


def fetch_fng_historical_dict():
    """Reuses the same alternative.me endpoint as
    data_sources/fetch_historical_btc.py's fetch_fng_historical(), but
    returns {date_str: value} keyed to match the other series here."""
    try:
        url = "https://api.alternative.me/fng/?limit=0"  # limit=0 = full history
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        result = {}
        for entry in data.get("data", []):
            # datetime.utcfromtimestamp() is deprecated since Python 3.12
            # (its own DeprecationWarning recommends datetime.fromtimestamp()
            # with an explicit UTC timezone instead — easy to misread that
            # warning as "datetime.UTC itself is the problem" when it's
            # actually the fix). Using the modern, non-deprecated form here.
            date_str = datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d")
            result[date_str] = int(entry["value"])
        print(f"[Backtest] FNG: {len(result)} observations fetched", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[Backtest] FNG fetch failed: {e}", file=sys.stderr)
        return {}


def _nearest_prior_value(series_dict, target_date, max_lookback_days=10):
    """FRED macro series (M2, DXY) update monthly/weekly, not daily — for
    any given daily BTC price date, find the most recent PRIOR value
    (standard practice: you wouldn't know a monthly M2 print until it's
    released, so using the latest AVAILABLE value as of that date is the
    correct point-in-time approach, not interpolating or looking ahead)."""
    d = datetime.strptime(target_date, "%Y-%m-%d")
    for i in range(max_lookback_days):
        check_date = (d - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in series_dict:
            return series_dict[check_date]
    return None


# ── GLO score (global central-bank liquidity) replay ─────────────────────
# collect.py v4.0.0 removed the direct m2_yoy term from Lt and REPLACED it
# with glo_score (global liquidity, from Fed/ECB/BOJ balance-sheet growth).
# The historical replay previously passed glo_score=None (its default), so
# Lt collapsed to pure BTC momentum — the backtest silently tested a
# DIFFERENT (degraded) model than the live v4.0.0 pipeline. This block
# replicates calculate_m33_global_liquidity()'s exact math point-in-time so
# the replay exercises the same global-liquidity term the live model uses.

_GLO_SERIES = ("WALCL", "ECBASSETSW", "JPNASSETS")  # Fed / ECB / BOJ assets


def _yoy_at(sorted_items, target_date, n_back=12):
    """Point-in-time replication of m33's yoy_chg: value at the most recent
    observation <= target_date, minus the value `n_back` FRED observations
    earlier (live _fred(series,13) + yoy_chg(arr[0] vs arr[12])). For a
    weekly series like WALCL, 13 obs spans ~13 weeks (~3 months), NOT a
    year — we replicate the live semantics exactly, quirk and all, rather
    than "correcting" it, so the replay matches what the pipeline computes.
    `sorted_items` is a pre-sorted list of (date_str, value) tuples."""
    import bisect
    idx = bisect.bisect_right(sorted_items, target_date, key=lambda x: x[0]) - 1
    if idx < 0:
        return None
    latest_val = sorted_items[idx][1]
    prior_idx = idx - n_back
    if prior_idx < 0:
        return None
    prior_val = sorted_items[prior_idx][1]
    if prior_val == 0:
        return 0.0
    return (latest_val - prior_val) / prior_val * 100.0


def _sorted_items(series_dict):
    """Pre-sort a FRED {date: value} dict into a list of (date, value)
    tuples, so per-date _yoy_at() calls are O(log n) instead of re-sorting
    the whole series every time."""
    return sorted(series_dict.items())


def compute_glo_score(date_str, walcl_items, ecb_items, jpn_items):
    """Replicate calculate_m33_global_liquidity()'s GLO 0-100 score for a
    single historical date. Higher = liquid expansion (low stress).
    Series args are pre-sorted (date, value) lists from _sorted_items()."""
    fed_yoy = _yoy_at(walcl_items, date_str)
    ecb_yoy = _yoy_at(ecb_items, date_str)
    jpn_yoy = _yoy_at(jpn_items, date_str)
    fed_z = (fed_yoy - 5.5) / 3.0 if fed_yoy is not None else 0
    ecb_z = (ecb_yoy - 4.0) / 3.0 if ecb_yoy is not None else 0
    jpn_z = (jpn_yoy - 3.0) / 3.0 if jpn_yoy is not None else 0
    weights, z_vals = [], []
    if fed_yoy is not None: weights.append(0.50); z_vals.append(fed_z)
    if ecb_yoy is not None: weights.append(0.30); z_vals.append(ecb_z)
    if jpn_yoy is not None: weights.append(0.20); z_vals.append(jpn_z)
    if not z_vals:
        return None
    glo_z = sum(w * z for w, z in zip(weights, z_vals)) / sum(weights)
    return max(0.0, min(100.0, 50.0 + glo_z * 20.0))


def run_backtest():
    print("=" * 60)
    print("M1-M6 HISTORICAL BACKTEST (via FRED CBBTCUSD)")
    print("=" * 60)

    print("\nFetching historical series (this can take a minute)...")
    btc_price = fetch_fred_series("CBBTCUSD")
    dxy_series = fetch_fred_series("DTWEXBGS")
    m2_series = fetch_fred_series("M2SL")
    fng_series = fetch_fng_historical_dict()
    walcl = fetch_fred_series("WALCL")
    ecb = fetch_fred_series("ECBASSETSW")
    jpn = fetch_fred_series("JPNASSETS")

    if not btc_price:
        print("❌ No BTC price data — check FRED_API_KEY and network access.")
        return

    # score_factors_from_market() and calculate_sfc_ensemble() are the
    # verbatim copies defined at module level above — no import needed.

    walcl_items = _sorted_items(walcl)
    ecb_items = _sorted_items(ecb)
    jpn_items = _sorted_items(jpn)

    sorted_dates = sorted(btc_price.keys())
    results = []
    prev_price = None

    for date_str in sorted_dates:
        price = btc_price[date_str]
        if prev_price is None:
            prev_price = price
            continue
        btc_24h = (price - prev_price) / prev_price * 100
        prev_price = price

        dxy = _nearest_prior_value(dxy_series, date_str, max_lookback_days=10)
        m2_level = _nearest_prior_value(m2_series, date_str, max_lookback_days=45)

        # M2 YoY growth needs a value from ~12 months prior for comparison
        m2_yoy = None
        if m2_level is not None:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            year_ago_str = (d - timedelta(days=365)).strftime("%Y-%m-%d")
            m2_year_ago = _nearest_prior_value(m2_series, year_ago_str, max_lookback_days=45)
            if m2_year_ago:
                m2_yoy = (m2_level - m2_year_ago) / m2_year_ago * 100

        fng = fng_series.get(date_str)  # None before ~Feb 2018, real value after
        # GLO liquidity term — matches collect.py v4.0.0's Lt construction
        # (m2_yoy is ignored by the function body; glo_score is now the
        # liquidity driver, replicated point-in-time from central-bank
        # balance sheets).
        glo = compute_glo_score(date_str, walcl_items, ecb_items, jpn_items)

        try:
            factors = score_factors_from_market(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=fng,
                pc_oi=None, m2_yoy=m2_yoy, dxy=dxy, glo_score=glo,
            )
            p_ens_components = calculate_sfc_ensemble(factors)
            # calculate_sfc_ensemble returns (sfc_pct, zone, factors_raw,
            # norm_factors, m1, m2, m3, m4, m5, m6, method_agreement)
            # NOTE scale inconsistency in the function's own return tuple:
            # m1_klr/m2_logit/m3_bayes/m4_ewc come back on a 0-1 scale,
            # while m5_qreg/m6_regime are ALREADY *100 — matching exactly
            # how collect.py's own output-writing does `round(m1*100, 1)`
            # for the first four but `round(m5, 1)` (no extra *100) for
            # the last two (verified against collect.py's actual output
            # dict construction, not assumed).
            sfc_pct = p_ens_components[0]
            m1, m2v, m3, m4, m5, m6 = p_ens_components[4:10]
            m1, m2v, m3, m4 = m1 * 100, m2v * 100, m3 * 100, m4 * 100
        except Exception as e:
            continue

        results.append({
            "date": date_str, "btc_price": price, "btc_24h": round(btc_24h, 2),
            "glo_score": round(glo, 2) if glo is not None else None,
            "sfc_pct": round(sfc_pct, 2) if sfc_pct is not None else None,
            "m1_klr": round(m1, 4) if m1 is not None else None,
            "m2_logit": round(m2v, 4) if m2v is not None else None,
            "m3_bayes": round(m3, 4) if m3 is not None else None,
            "m4_ewc": round(m4, 4) if m4 is not None else None,
            "m5_qreg": round(m5, 4) if m5 is not None else None,
            "m6_regime": round(m6, 4) if m6 is not None else None,
        })

    print(f"\nComputed {len(results)} historical daily observations "
          f"({sorted_dates[0]} to {sorted_dates[-1]})")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"Saved full results to {OUTPUT_FILE}")

    # Check against known crisis windows
    print("\n" + "=" * 60)
    print("CRISIS WINDOW CHECK")
    print("=" * 60)
    results_by_date = {r["date"]: r for r in results}
    for crisis_name, (start, end) in KNOWN_CRISIS_WINDOWS.items():
        window_scores = [r["sfc_pct"] for r in results if start <= r["date"] <= end and r["sfc_pct"] is not None]
        if window_scores:
            avg_score = sum(window_scores) / len(window_scores)
            max_score = max(window_scores)
            print(f"\n{crisis_name} ({start} to {end}):")
            print(f"  Avg sfc_pct: {avg_score:.1f}  |  Max sfc_pct: {max_score:.1f}  |  n={len(window_scores)} days")
        else:
            print(f"\n{crisis_name}: NO DATA IN THIS WINDOW (check date range / FRED coverage)")


if __name__ == "__main__":
    run_backtest()
