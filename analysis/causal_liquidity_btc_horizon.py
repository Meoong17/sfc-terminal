#!/usr/bin/env python3
"""
causal_liquidity_btc_horizon.py — longer lags + forward horizons (GLF -> BTC)
=============================================================================
Follows up on causal_liquidity_btc.py which found the sign is right but weak
(min p: ΔGLF lag-4 = 0.116, bear-regime = 0.127). That hint suggests the
liquidity->BTC lead is SLOWER than the 1-week/1-month lags tested, and/or
that liquidity predicts BTC's RETURN OVER A FORWARD HORIZON rather than the
very next period.

This script tests the two natural interpretations:
  1. LONGER LAGS — liquidity changes 2..6 months ago predicting BTC this month
     (Granger maxlag up to 6, plus monthly lagged OLS).
  2. FORWARD HORIZONS — GLF level/change today predicting BTC's return over
     the NEXT 30 / 60 / 90 days (not just the immediate next period). This is
     exactly the "liquidity impulse propagates over a quarter" thesis.

It reuses the FULL 8-component GLF reconstruction from causal_liquidity_btc.py
so results are comparable. All point-in-time, no look-ahead, display-only.

USAGE:
    cd ~/sfc && export FRED_API_KEY=...
    .venv/bin/python analysis/causal_liquidity_btc_horizon.py
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.api import OLS, add_constant
except ImportError as e:
    print(f"[Causal-H] statsmodels unavailable: {e}", file=sys.stderr)
    sys.exit(1)

from causal_liquidity_btc import (
    build_monthly_glf, monthly_btc, fetch_series_dict,
    nearest_month, FULL_GLF,
)

OUTPUT = os.path.join(SFC_ROOT, ".causal_liquidity_btc_horizon.json")

GRANGER_LAGS = [1, 2, 3, 4, 5, 6]        # months
MONTHLY_OLS_LAGS = [1, 2, 3, 4, 5, 6]    # months
HORIZONS = [30, 60, 90]                   # forward days
ALPHA = 0.05


def ols_fit(rows):
    """Fit OLS ret_t ~ pred + ret_prev; return dict or None."""
    if len(rows) < 30:
        return None
    y = np.array([r[0] for r in rows])
    X = add_constant(np.array([[r[1], r[2]] for r in rows]))
    m = OLS(y, X).fit()
    return {
        "n": len(rows), "coef": round(float(m.params[1]), 4),
        "t": round(float(m.tvalues[1]), 3), "p": round(float(m.pvalues[1]), 4),
        "r2": round(float(m.rsquared), 4), "sig": bool(m.pvalues[1] < ALPHA),
        "sign": "pos" if m.params[1] > 0 else "neg",
    }


def _gather(glf, btc_ret, k, use_change):
    common = sorted(set(glf) & set(btc_ret))
    rows = []
    for i, m in enumerate(common):
        if i < k:
            continue
        ret_t = btc_ret[m]
        glf_k = glf.get(common[i - k])
        glf_k_prev = glf.get(common[i - k - 1]) if (i - k - 1) >= 0 else None
        ret_prev = btc_ret.get(common[i - 1])
        if use_change:
            if glf_k is None or glf_k_prev is None:
                pred = None
            else:
                pred = glf_k - glf_k_prev
        else:
            pred = glf_k
        if pred is None or ret_t is None or ret_prev is None:
            continue
        rows.append([ret_t, pred, ret_prev, common[i]])
    return rows


def monthly_lagged_ols(glf, btc_ret, max_lag=6, use_change=True):
    common = sorted(set(glf) & set(btc_ret))
    out = {}
    for k in range(1, max_lag + 1):
        for variant, use_ch in (("level", False), ("change", True)):
            rows = _gather(glf, btc_ret, k, use_ch)
            r = ols_fit(rows)
            out[f"{k}m_{variant}"] = r or {"error": f"n={len(rows)}"}
    return out


def robustness_6m(glf, btc_ret):
    """Deep robustness of the 6-month GLF-level result (p=0.029)."""
    rows_all = _gather(glf, btc_ret, 6, False)   # level
    res = {"n_full": len(rows_all)}
    base = ols_fit(rows_all)
    res["full_sample"] = base

    # 1. Split by regime (BTC above/below 200DMA proxy of that month)
    closes = sorted({d[:7]: None for d in []})
    # monthly BTC closes for regime
    # reuse: build from raw daily is heavy; instead split by GLF tercile (proxy
    # of high vs low liquidity) AND by calendar sub-period.
    gvals = np.array([r[1] for r in rows_all])
    p = np.percentile(gvals, [50])
    hi = [r for r in rows_all if r[1] >= p[0]]
    lo = [r for r in rows_all if r[1] < p[0]]
    res["split_high_glf"] = ols_fit(hi)
    res["split_low_glf"] = ols_fit(lo)

    # 2. Sub-periods (first half vs second half of the sample)
    rows_sorted = sorted(rows_all, key=lambda r: r[3])
    half = len(rows_sorted) // 2
    res["subperiod_first_half"] = ols_fit(rows_sorted[:half])
    res["subperiod_second_half"] = ols_fit(rows_sorted[half:])

    # 3. Drop-extreme: exclude |GLF z| > 1.5 (outliers)
    kept = [r for r in rows_all if abs(r[1]) <= 1.5]
    res["drop_extreme_glf"] = ols_fit(kept)

    # 4. Sign consistency across rolling 3-year windows
    signs = []
    win = 36  # months
    for i in range(len(rows_sorted) - win):
        w = rows_sorted[i:i + win]
        r = ols_fit(w)
        if r:
            signs.append(1 if r["coef"] > 0 else -1)
    res["rolling_sign_pos_pct"] = round(sum(1 for s in signs if s > 0) / len(signs), 3) if signs else None
    res["n_rolling_windows"] = len(signs)

    # 5. FDR correction across all tested specs (Benjamini-Hochberg)
    all_p = []
    for k in range(1, MONTHLY_OLS_LAGS[-1] + 1):
        for v in ("level", "change"):
            r = ols_fit(_gather(glf, btc_ret, k, v == "change"))
            if r:
                all_p.append(r["p"])
    all_p.sort()
    m = len(all_p)
    fdr_significant = []
    for i, pv in enumerate(all_p):
        # BH critical value (i+1)/m * alpha
        if pv <= ((i + 1) / m) * ALPHA:
            fdr_significant.append(pv)
    res["fdr"] = {
        "n_tests": m,
        "sorted_p": [round(x, 4) for x in all_p],
        "alpha": ALPHA,
        "survive_fdr": bool(fdr_significant),
        "surviving_p": [round(x, 4) for x in fdr_significant],
    }
    return res


def forward_horizon(btc_daily, glf, horizons):
    dates = sorted(btc_daily)
    closes = btc_daily
    out = {}
    for h in horizons:
        pairs = []
        for i, d in enumerate(dates):
            if i + h >= len(dates):
                break
            p0, p1 = closes[d], closes[dates[i + h]]
            if not p0 or p0 == 0:
                continue
            ret = (p1 - p0) / p0 * 100
            g = nearest_month(glf, iso_key(d))
            if g is None:
                continue
            pairs.append((g, ret))
        if len(pairs) < 100:
            out[h] = {"error": f"n={len(pairs)}"}
            continue
        gs = np.array([p[0] for p in pairs])
        rs = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(gs, rs)[0, 1]) if (np.std(gs) > 0 and np.std(rs) > 0) else None
        terc = np.percentile(gs, [33.3, 66.7])
        low_mean = float(rs[gs <= terc[0]].mean()) if (gs <= terc[0]).sum() else None
        high_mean = float(rs[gs >= terc[1]].mean()) if (gs >= terc[1]).sum() else None
        spread = (high_mean - low_mean) if (high_mean is not None and low_mean is not None) else None
        out[h] = {
            "n": len(pairs), "corr_glf_ret": round(corr, 4) if corr is not None else None,
            "low_tercile_mean_ret": round(low_mean, 3) if low_mean is not None else None,
            "high_tercile_mean_ret": round(high_mean, 3) if high_mean is not None else None,
            "high_minus_low": round(spread, 3) if spread is not None else None,
            "direction": ("positive" if (spread or 0) > 0 else "negative") if spread is not None else None,
        }
    return out


def iso_key(date_str):
    dt = datetime.fromisoformat(date_str)
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def main():
    print("=" * 76)
    print("CAUSAL TEST (horizon) — longer lags + forward horizons + robustness")
    print("=" * 76)

    print("\n[1/5] Fetching BTC daily closes...")
    btc_daily = fetch_series_dict("CBBTCUSD")
    print(f"      {len(btc_daily)} closes")

    print("[2/5] Reconstructing FULL monthly GLF (8 components)...")
    glf = build_monthly_glf(full=True)
    print(f"      {len(glf)} monthly points")

    if not btc_daily or not glf:
        print("\n⚠ Missing data.")
        return

    btc_ret = monthly_btc(btc_daily)
    common = sorted(set(glf) & set(btc_ret))
    print(f"      {len(common)} aligned months")

    print("\n[3/5] Monthly Granger — GLF -> BTC, lags 1..6 months...")
    df = np.column_stack([[btc_ret[m] for m in common], [glf[m] for m in common]])
    gres = grangercausalitytests(np.column_stack([df[:, 0], df[:, 1]]),
                                 maxlag=GRANGER_LAGS[-1], verbose=False)
    granger = {}
    for lag, r in gres.items():
        p = float(r[0]["ssr_ftest"][1])
        granger[int(lag)] = round(p, 4)
        print(f"      lag {lag}m  p={p:.4f}")
    granger["min_p"] = min(granger.values())
    granger["causal"] = bool(granger["min_p"] < ALPHA)

    print("\n[4/5] Monthly lagged OLS (level & change)...")
    m_ols = monthly_lagged_ols(glf, btc_ret, max_lag=MONTHLY_OLS_LAGS[-1])
    print(f"{'spec':<14}{'n':<8}{'coef':<10}{'t':<8}{'p':<8}{'R²':<8}{'sig'}")
    print("-" * 56)
    for k in range(1, MONTHLY_OLS_LAGS[-1] + 1):
        for v in ("level", "change"):
            r = m_ols[f"{k}m_{v}"]
            if "error" in r:
                print(f"{k}m_{v:<10}{r['error']}")
            else:
                print(f"{k}m_{v:<10}{r['n']:<8}{r['coef']:<10}{r['t']:<8}"
                      f"{r['p']:<8}{r['r2']:<8}{r['sig']}")

    print("\n[5/5] ROBUSTNESS of the 6m-level result...")
    rob = robustness_6m(glf, btc_ret)
    for key, label in [("full_sample", "Full sample"), ("split_high_glf", "High-GLF half"),
                       ("split_low_glf", "Low-GLF half"), ("subperiod_first_half", "First half"),
                       ("subperiod_second_half", "Second half"), ("drop_extreme_glf", "Drop |z|>1.5")]:
        r = rob[key]
        if r and "error" not in r:
            print(f"  {label:<20} coef={r['coef']:<10} p={r['p']:<8} sig={r['sig']} n={r['n']}")
        else:
            print(f"  {label:<20} {r.get('error') if r else 'n/a'}")
    print(f"  Rolling sign positive: {rob.get('rolling_sign_pos_pct')} ({rob.get('n_rolling_windows')} windows)")
    print(f"  FDR: n_tests={rob['fdr']['n_tests']} survive={rob['fdr']['survive_fdr']} "
          f"surviving_p={rob['fdr']['surviving_p']}")

    print("\n[6/6] Forward-horizon: GLF today -> BTC return over next 30/60/90d...")
    fwd = forward_horizon(btc_daily, glf, HORIZONS)
    print(f"{'Horizon':<10}{'n':<8}{'corr':<10}{'lowTerc':<10}{'highTerc':<10}{'high-low':<10}{'dir'}")
    print("-" * 66)
    for h in HORIZONS:
        r = fwd[h]
        if "error" in r:
            print(f"{h}d   {r['error']}")
        else:
            print(f"{h}d  {r['n']:<8}{r['corr_glf_ret']:<10}{r['low_tercile_mean_ret']:<10}"
                  f"{r['high_tercile_mean_ret']:<10}{r['high_minus_low']:<10}{r['direction']}")

    result = {
        "generated_at": datetime.now().isoformat(),
        "method": "Longer-lag Granger (1-6m) + monthly lagged OLS + 6m robustness + forward horizon",
        "note": "Full 8-component GLF. Display-only research.",
        "granger_monthly": granger,
        "monthly_ols": m_ols,
        "robustness_6m": rob,
        "forward_horizon": fwd,
        "n_btc_daily": len(btc_daily),
        "n_glf_monthly": len(glf),
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
