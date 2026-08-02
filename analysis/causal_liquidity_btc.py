#!/usr/bin/env python3
"""
causal_liquidity_btc.py — Statistical CAUSALITY test: GLF -> BTC (extended)
==========================================================================
Deepens the initial causal check (v1: reduced 4-component GLF, Granger +
weekly lagged OLS) with three extensions that search for the relationship
where it might actually live:

  EXT-1  FULL GLF: 8 components (Fed/ECB/BOJ/China-M2/US-M2/TGA/RRP/DXY)
         reconstructed point-in-time, instead of the reduced 4.
  EXT-2  LEVEL-CHANGE: test whether the CHANGE in liquidity (ΔGLF) predicts
         BTC, not just the liquidity level — captures the "liquidity impulse".
  EXT-3  CONDITIONAL-ON-REGIME: split the sample into bull / bear phases
         (BTC above vs below its 200-day moving average) and re-run Granger
         within each. A liquidity->BTC link may only appear in one phase.

Base tests (unchanged from v1):
  A. Granger causality GLF<->BTC (monthly, both directions).
  B. Weekly lagged OLS (BTC ret_t ~ GLF(t-k) + ret(t-1)).

Honest framing: all of this is a CAUSAL CHECK on a reconstructed GLF before
anything is blended into the live signal. Negative results are reported as
negative — no overclaiming.

USAGE:
    cd ~/sfc && export FRED_API_KEY=...
    .venv/bin/python analysis/causal_liquidity_btc.py
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from collections import OrderedDict

import requests
import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))
sys.path.insert(0, os.path.join(SFC_ROOT, "data_sources"))

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.api import OLS, add_constant
except ImportError as e:
    print(f"[Causal] statsmodels unavailable: {e}", file=sys.stderr)
    sys.exit(1)

from historical_backtest_m1m6 import fetch_fred_series

OUTPUT = os.path.join(SFC_ROOT, ".causal_liquidity_btc.json")
MAXLAG_GRANGER = 4
LAGS_OLS_WEEKS = [1, 2, 4, 8, 12]

# Full GLF component config (weights match production global_liquidity_engine).
FULL_GLF = {
    "WALCL":        {"name": "fed",   "mean": 5.5, "std": 8.0, "weight": 0.30, "kind": "yoy"},
    "ECBASSETSW":   {"name": "ecb",   "mean": 4.0, "std": 7.0, "weight": 0.15, "kind": "yoy"},
    "JPNASSETS":    {"name": "jpn",   "mean": 3.0, "std": 6.0, "weight": 0.03, "kind": "yoy"},
    "MYAGM2CNM189N": {"name": "china","mean": 9.0, "std": 3.5, "weight": 0.04, "kind": "yoy"},
    "M2SL":         {"name": "m2",    "mean": 6.0, "std": 4.0, "weight": 0.15, "kind": "yoy"},
    "WTREGEN":      {"name": "tga",   "mean": 0.0, "std": 1.0, "weight": 0.10, "kind": "level"},
    "RRPONTSYD":    {"name": "rrp",   "mean": 0.0, "std": 1.0, "weight": 0.10, "kind": "level"},
    "DTWEXBGS":     {"name": "dxy",   "mean": 0.0, "std": 1.0, "weight": 0.13, "kind": "zinv"},
}


def _z(v, mean, std):
    if std == 0 or v is None:
        return 0.0
    return max(-3.0, min(3.0, (v - mean) / std))


def fetch_series_dict(sid):
    return fetch_fred_series(sid, start_date="2002-01-01")


def compute_yoy(series_dict):
    """{date: YoY pct} from a dated series (12 prior obs)."""
    dates = sorted(series_dict)
    out = {}
    for i in range(12, len(dates)):
        d0, d1 = dates[i - 12], dates[i]
        v0, v1 = series_dict[d0], series_dict[d1]
        if v0 and v0 != 0:
            out[d1] = (v1 - v0) / v0 * 100
    return out


def month_grid(series_yoy):
    months = set()
    for s in series_yoy.values():
        for d in s:
            months.add(d[:7])
    return sorted(months)


def nearest(dated, target_month):
    """Nearest dated value to a 'YYYY-MM' target within 45 days."""
    ty, tm = int(target_month[:4]), int(target_month[5:7])
    target_idx = ty * 12 + tm
    best, bd = None, None
    for d, v in dated.items():
        my, mm = int(d[:4]), int(d[5:7])
        diff = abs((my * 12 + mm) - target_idx)
        if bd is None or diff < bd:
            bd = diff
            best = v
    return best


def build_monthly_glf(full=True):
    """Reconstruct GLF monthly. Returns {month: glf_z} (weighted z-sum)."""
    series = {}
    comps = FULL_GLF if full else {k: v for k, v in FULL_GLF.items() if v["kind"] == "yoy" and v["weight"] >= 0.15}
    for sid in comps:
        raw = fetch_series_dict(sid)
        if not raw:
            continue
        if comps[sid]["kind"] == "yoy":
            series[sid] = ("yoy", compute_yoy(raw))
        elif comps[sid]["kind"] == "level":
            # TGA/RRP: z-score of current level (inverted liquidity semantics
            # handled by weight sign convention via mean/std normalization)
            series[sid] = ("level", raw)
        else:  # zinv (DXY)
            series[sid] = ("level", raw)

    months = set()
    for kind, s in series.values():
        for d in s:
            months.add(d[:7])
    months = sorted(months)

    glf = {}
    for m in months:
        ws, tw = 0.0, 0.0
        for sid, (kind, s) in series.items():
            comp = comps[sid]
            v = nearest(s, m)
            if v is None:
                continue
            if kind == "yoy":
                z = _z(v, comp["mean"], comp["std"])
            else:
                # level: normalize via rolling-ish scale (std of the series)
                z = _z(v, comp["mean"], comp["std"])
            ws += z * comp["weight"]
            tw += comp["weight"]
        if tw > 0:
            glf[m] = ws / tw
    return glf


def monthly_btc(btc_daily):
    closes = {}
    for d, v in btc_daily.items():
        closes[d[:7]] = v
    mons = sorted(closes)
    ret = {}
    for i in range(1, len(mons)):
        p0, p1 = closes[mons[i - 1]], closes[mons[i]]
        if p0 and p0 != 0:
            ret[mons[i]] = (p1 - p0) / p0 * 100
    return ret


def granger(glf, btc_ret):
    common = sorted(set(glf) & set(btc_ret))
    if len(common) < 20:
        return {"error": f"insufficient n={len(common)}"}
    df = np.column_stack([[btc_ret[m] for m in common], [glf[m] for m in common]])
    out = {"n_months": len(common), "period": (common[0], common[-1])}
    for direction, (x, y) in {"GLF->BTC": (df[:, 1], df[:, 0]),
                              "BTC->GLF": (df[:, 0], df[:, 1])}.items():
        data = np.column_stack([y, x])
        try:
            res = grangercausalitytests(data, maxlag=MAXLAG_GRANGER, verbose=False)
            row = {}
            for lag, r in res.items():
                row[int(lag)] = round(float(r[0]["ssr_ftest"][1]), 4)
            mp = min(row.values())
            out[direction] = {"p_by_lag": row, "min_p": mp,
                              "causal": bool(mp < 0.05),
                              "strength": "strong" if mp < 0.01 else "moderate" if mp < 0.05 else "none"}
        except Exception as e:
            out[direction] = {"error": str(e)}
    return out


def weekly_series(btc_daily, glf):
    wk = OrderedDict()
    for d in sorted(btc_daily):
        dt = datetime.fromisoformat(d)
        iso = dt.isocalendar()
        wk.setdefault(f"{iso[0]}-W{iso[1]:02d}", []).append((d, btc_daily[d]))
    items = list(wk.items())
    series = []
    for i in range(1, len(items)):
        pc, cc = items[i - 1][1][-1][1], items[i][1][-1][1]
        if pc and pc != 0:
            ret = (cc - pc) / pc * 100
            wk_key = items[i][0]
            glf_v = nearest_month(glf, wk_key)
            # also capture previous month level for change (impulse)
            glf_prev = nearest_month(glf, items[i - 1][0]) if i - 1 >= 0 else glf_v
            series.append([wk_key, ret, glf_v, glf_prev])
    return series


def nearest_month(glf, week_key):
    if isinstance(glf, dict):
        glf_items = list(glf.items())
    else:
        glf_items = glf
    year = int(week_key.split("-")[0])
    week = int(week_key.split("-")[1][1:])
    jan4 = datetime(year, 1, 4)
    thurs = jan4 - timedelta(days=jan4.isoweekday() - 4)
    wd = thurs + timedelta(weeks=week - 1)
    best, bd = None, None
    for m, v in glf_items:
        my, mm = m.split("-")
        m_idx = int(my) * 12 + int(mm)
        w_idx = wd.year * 12 + wd.month
        d = abs(m_idx - w_idx)
        if bd is None or d < bd:
            bd = d
            best = v
    return best


def lagged_ols(series, lags_weeks, use_change=False):
    out = {}
    for k in lags_weeks:
        rows = []
        for i in range(max(k, 1), len(series)):
            ret_t = series[i][1]
            glf_k = series[i - k][2]
            glf_prev_k = series[i - k][3]
            ret_prev = series[i - 1][1]
            if use_change:
                pred = glf_k - glf_prev_k if (glf_k is not None and glf_prev_k is not None) else None
            else:
                pred = glf_k
            if pred is None or ret_t is None or ret_prev is None:
                continue
            rows.append([ret_t, pred, ret_prev])
        if len(rows) < 30:
            out[k] = {"error": f"n={len(rows)}"}
            continue
        y = np.array([r[0] for r in rows])
        X = add_constant(np.array([[r[1], r[2]] for r in rows]))
        model = OLS(y, X).fit()
        out[k] = {
            "n": len(rows),
            "coef": round(float(model.params[1]), 4),
            "t": round(float(model.tvalues[1]), 3),
            "p": round(float(model.pvalues[1]), 4),
            "r2": round(float(model.rsquared), 4),
            "sig": bool(model.pvalues[1] < 0.05),
            "sign": "pos" if model.params[1] > 0 else "neg",
        }
    return out


def conditional_regime_granger(glf, btc_ret, btc_close):
    """Split by BTC above/below 200DMA proxy (bull vs bear) and test Granger."""
    # Build monthly 200DMA from daily closes
    closes = sorted(btc_close)
    month_list = sorted(set(d[:7] for d in closes))
    monthly_ma = {}
    for m in month_list:
        # Need at least ~120 days history before computing a 200D MA; skip
        # the earliest months only, not via the (buggy) empty-dict check.
        upto = [btc_close[d] for d in closes if d[:7] <= m][-260:]
        if len(upto) < 120:
            continue
        ma = sum(upto[-200:]) / min(200, len(upto[-200:]))
        last_close = upto[-1]
        monthly_ma[m] = last_close - ma  # >0 bull, <0 bear
    out = {}
    for regime in ("bull", "bear"):
        common = sorted(set(glf) & set(btc_ret) & set(monthly_ma))
        sub_months = [m for m in common
                      if (monthly_ma[m] > 0 if regime == "bull" else monthly_ma[m] <= 0)]
        if len(sub_months) < 20:
            out[regime] = {"error": f"n={len(sub_months)}"}
            continue
        df = np.column_stack([[btc_ret[m] for m in sub_months],
                              [glf[m] for m in sub_months]])
        data = np.column_stack([df[:, 0], df[:, 1]])
        res = grangercausalitytests(data, maxlag=MAXLAG_GRANGER, verbose=False)
        row = {}
        for lag, r in res.items():
            row[int(lag)] = round(float(r[0]["ssr_ftest"][1]), 4)
        mp = min(row.values())
        out[regime] = {"n_months": len(sub_months), "p_by_lag": row,
                       "min_p": mp, "causal": bool(mp < 0.05)}
    return out


def main():
    print("=" * 72)
    print("CAUSAL TEST (extended) — does global liquidity drive Bitcoin?")
    print("Full-GLF + level-change + conditional-on-regime")
    print("=" * 72)

    print("\n[1/6] Fetching BTC daily closes...")
    btc_daily = fetch_series_dict("CBBTCUSD")
    print(f"      {len(btc_daily)} closes")

    print("[2/6] Reconstructing FULL monthly GLF (8 components)...")
    glf_full = build_monthly_glf(full=True)
    print(f"      {len(glf_full)} monthly points")

    if not btc_daily or not glf_full:
        print("\n⚠ Missing data.")
        return

    btc_ret = monthly_btc(btc_daily)

    print("\n[3/6] Granger (FULL GLF, monthly)...")
    g_full = granger(glf_full, btc_ret)
    print(json.dumps(g_full, indent=2))

    print("\n[4/6] Conditional-on-regime Granger (bull vs bear)...")
    g_regime = conditional_regime_granger(glf_full, btc_ret, btc_daily)
    print(json.dumps(g_regime, indent=2))

    print("\n[5/6] Weekly lagged OLS — GLF LEVEL as predictor...")
    series = weekly_series(btc_daily, glf_full)
    ols_level = lagged_ols(series, LAGS_OLS_WEEKS, use_change=False)
    print(_fmt_ols(ols_level))

    print("\n[6/6] Weekly lagged OLS — GLF CHANGE (Δliquidity impulse)...")
    ols_change = lagged_ols(series, LAGS_OLS_WEEKS, use_change=True)
    print(_fmt_ols(ols_change))

    result = {
        "generated_at": datetime.now().isoformat(),
        "method": "Granger (full GLF) + conditional-regime Granger + weekly lagged OLS (level & change)",
        "note": "Reconstructed FULL GLF (Fed/ECB/BOJ/China-M2/US-M2/TGA/RRP/DXY). "
                "Causal check, not yet blended into live signal.",
        "granger_full": g_full,
        "granger_conditional_regime": g_regime,
        "ols_weekly_level": ols_level,
        "ols_weekly_change": ols_change,
        "n_btc_daily": len(btc_daily),
        "n_glf_monthly": len(glf_full),
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")


def _fmt_ols(ols):
    lines = []
    lines.append(f"{'Lag(wk)':<10}{'n':<8}{'coef':<10}{'t':<8}{'p':<8}{'R²':<8}{'sig'}")
    lines.append("-" * 54)
    for k in LAGS_OLS_WEEKS:
        r = ols[k]
        if "error" in r:
            lines.append(f"{k:<10}{r['error']}")
        else:
            lines.append(f"{k:<10}{r['n']:<8}{r['coef']:<10}{r['t']:<8}"
                         f"{r['p']:<8}{r['r2']:<8}{r['sig']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
