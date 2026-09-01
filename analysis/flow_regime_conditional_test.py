#!/usr/bin/env python3
"""
flow_regime_conditional_test.py — Ya.docx Test #4: does flow have predictive edge
                              CONDITIONAL on regime, even where it failed unconditional?
======================================================================================
Ya.docx proposes: "ETF flow not predictive in expansion, but very predictive in
liquidity contraction." The repo found ETF flow (lagged) n.s. UNCONDITIONALLY, and
the SLR liquidity→flow→price chain failed. This is the one hypothesis not yet tested:
does a flow gain predictive power inside a specific regime?

Flows (longest history first):
  ORDER_FLOW  taker imbalance ratio, 2017-2026 (3.2k days) from binance_orderflow_daily.json
  ETF_FLOW    net BTC flow, 2024-2026 (677 days) from .etf_cache.json

Regimes (point-in-time, monthly):
  liquidity expansion  = current-month GLF z > 0
  liquidity contraction = current-month GLF z <= 0
  (secondary: bull/bear via BTC vs 200DMA)

Per flow x regime x horizon [1,3,7,14,30]:
  * top-vs-bottom-25% tail gap of flow -> mean forward BTC return, bootstrap CI
  * Spearman IC(flow, forward return)
  * purged-CV OOS AUC (logistic P(return>0), embargo=h) for the primary flow
Core question: does the flow's edge appear in one regime but not the other?

USAGE:
    cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/flow_regime_conditional_test.py
"""
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

from historical_backtest_m1m6 import fetch_fred_series
from causal_liquidity_btc import build_monthly_glf

OUTPUT = os.path.join(SFC_ROOT, ".flow_regime_test.json")
HORIZONS = [1, 3, 7, 14, 30]
TAIL = 0.25


# --------------------------------------------------------------------------- #
def load_order_flow():
    with open(os.path.join(SFC_ROOT, "data/binance_orderflow_daily.json")) as f:
        d = json.load(f)
    out = {}
    for k, r in d.items():
        tq = r.get("total_qty")
        if tq and tq > 0:
            out[k] = (r["taker_buy_qty"] - r["taker_sell_qty"]) / tq
    return out


def load_etf_flow():
    with open(os.path.join(SFC_ROOT, ".etf_cache.json")) as f:
        d = json.load(f)
    # per-day net BTC: reconstruct from per-ETF dict (total_btc often None)
    out = {}
    for e in d.get("flows", []):
        tb = e.get("total_btc")
        if tb is None:
            et = e.get("etfs")
            if isinstance(et, dict) and et:
                tb = sum(v for v in et.values() if v is not None)
        if tb is not None:
            out[e["date"]] = float(tb)
    return out


def rolling_z(series, window=90, min_n=60):
    """{date: point-in-time rolling z-score}."""
    dates = sorted(series)
    out = {}
    vals = [series[d] for d in dates]
    for i, d in enumerate(dates):
        if i < min_n:
            continue
        w = vals[max(0, i - window):i]
        w = [v for v in w if v is not None]
        if len(w) < min_n:
            continue
        mu, sd = float(np.mean(w)), float(np.std(w))
        if sd <= 1e-9:
            continue
        out[d] = (series[d] - mu) / sd
    return out


def regime_labels(glf, btc_daily):
    """{date: 'expansion'|'contraction'} via point-in-time: GLF below its own
    trailing median (expanding window) = contraction, above = expansion.
    Fixed-calibration GLF sign is useless (all-positive in-sample), so use a
    trailing-median split which is genuinely point-in-time and balances the arms.
    Secondary bull/bear via BTC 200DMA."""
    months = sorted(glf)
    liq_month = {}
    for i, m in enumerate(months):
        if i < 24:
            continue
        hist = [glf[months[j]] for j in range(i)]
        med = float(np.median(hist))
        liq_month[m] = "contraction" if glf[m] < med else "expansion"
    liq = {}
    for d in btc_daily:
        if d[:7] in liq_month:
            liq[d] = liq_month[d[:7]]
    trend = {}
    pdates = sorted(btc_daily)
    closes = np.array([btc_daily[d] for d in pdates])
    ma200 = {}
    for i, d in enumerate(pdates):
        if i >= 200:
            ma200[d] = float(closes[i - 200:i].mean())
    for d in pdates:
        if d in ma200:
            trend[d] = "bull" if btc_daily[d] >= ma200[d] else "bear"
    return liq, trend


def forward_ret(btc_daily, start, h):
    pdates = sorted(btc_daily)
    if start not in btc_daily:
        return None
    si = pdates.index(start)
    if si + h >= len(pdates):
        return None
    p0, p1 = btc_daily[start], btc_daily[pdates[si + h]]
    return (p1 - p0) / p0 * 100.0 if p0 else None


def tail_gap_and_ic(flow_z, btc_daily, regime_dates, horizons):
    """Within a regime: top vs bottom TAIL of flow_z -> forward return."""
    res = {}
    for h in horizons:
        rows = []
        for d in regime_dates:
            if d not in flow_z:
                continue
            fr = forward_ret(btc_daily, d, h)
            if fr is None:
                continue
            rows.append((flow_z[d], fr))
        if len(rows) < 60:
            res[h] = {"n": len(rows), "error": "n<60"}
            continue
        rows.sort(key=lambda x: x[0])
        n = len(rows)
        tn = int(n * TAIL)
        bottom = np.array([r[1] for r in rows[:tn]])
        top = np.array([r[1] for r in rows[-tn:]])
        gap = bottom.mean() - top.mean()
        rng = np.random.default_rng(42)
        nboot = 5000
        bd = np.empty(nboot)
        for i in range(nboot):
            bd[i] = rng.choice(bottom, size=len(bottom), replace=True).mean() - \
                    rng.choice(top, size=len(top), replace=True).mean()
        lo, hi = np.percentile(bd, [5, 95])
        from scipy.stats import spearmanr
        ic, icp = spearmanr([r[0] for r in rows], [r[1] for r in rows])
        res[h] = {
            "n": n, "tail_n": tn,
            "bottom_mean_fwd": round(float(bottom.mean()), 3),
            "top_mean_fwd": round(float(top.mean()), 3),
            "gap_bottom_minus_top": round(float(gap), 3),
            "ci90": [round(float(lo), 3), round(float(hi), 3)],
            "significant": bool(lo > 0 or hi < 0),
            "spearman_ic": round(float(ic), 3), "spearman_p": round(float(icp), 4),
        }
    return res


def purged_cv_auc(flow_z, btc_daily, regime_dates, h, n_folds=5):
    """Logistic P(return_h > 0 | flow_z), purged + embargo, OOS AUC within regime."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rows = []
    for d in regime_dates:
        if d not in flow_z:
            continue
        fr = forward_ret(btc_daily, d, h)
        if fr is None:
            continue
        rows.append((d, flow_z[d], 1.0 if fr > 0 else 0.0))
    if len(rows) < 120:
        return None
    rows.sort()
    dates = [r[0] for r in rows]
    X = np.array([[r[1]] for r in rows])
    y = np.array([r[2] for r in rows])
    n = len(rows)
    folds = np.array_split(np.arange(n), n_folds)
    scores, truth = [], []
    date_idx = {dt: i for i, dt in enumerate(dates)}
    for fold in folds:
        te_dates = {dates[j] for j in fold}
        tr = []
        for j in range(n):
            if j in set(fold):
                continue
            jd = dates[j]
            jend = date_idx[jd] + h
            overlap = any(date_idx[jd] <= date_idx[td] <= jend for td in te_dates)
            if overlap:
                continue
            tr.append(j)
        if len(tr) < 30 or len(fold) < 5:
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        scores.extend(clf.predict_proba(X[fold])[:, 1])
        truth.extend(y[fold])
    if len(scores) < 20 or len(set(truth)) < 2:
        return None
    try:
        auc = float(roc_auc_score(truth, scores))
        se = float(np.sqrt(auc * (1 - auc) / max(len(truth) - 1, 1))) if 0 < auc < 1 else None
        return {"auc": round(auc, 3),
                "ci90_lo": round(max(0.0, auc - 1.645 * se), 3) if se else None,
                "n": len(scores)}
    except ValueError:
        return None


def main():
    print("=" * 76)
    print("FLOW REGIME-CONDITIONAL TEST (Ya.docx Test #4)")
    print("=" * 76)

    print("\n[0] Loading flows + BTC + GLF regime...")
    of = load_order_flow()
    etf = load_etf_flow()
    ofz = rolling_z(of)
    etfz = rolling_z(etf)
    print(f"    order_flow days={len(of)}  z={len(ofz)}  range {min(of)}..{max(of)}")
    print(f"    etf_flow  days={len(etf)}  z={len(etfz)}  range {min(etf)}..{max(etf)}")

    btc_daily = fetch_fred_series("CBBTCUSD")
    glf = build_monthly_glf(full=True)
    liq, trend = regime_labels(glf, btc_daily)
    exp_dates = [d for d in liq if liq[d] == "expansion"]
    con_dates = [d for d in liq if liq[d] == "contraction"]
    print(f"    liquidity regime: expansion={len(exp_dates)}d contraction={len(con_dates)}d")

    result = {"generated_at": datetime.now().isoformat(),
              "horizons": HORIZONS, "tail": TAIL}

    for fname, fz, flabel in [("order_flow", ofz, "Taker imbalance (order flow)"),
                              ("etf_flow", etfz, "ETF net flow")]:
        print(f"\n{'='*76}\nFLOW: {flabel}\n{'='*76}")
        result[fname] = {"regime": {}}
        for rname, rdates in [("expansion", exp_dates), ("contraction", con_dates)]:
            tg = tail_gap_and_ic(fz, btc_daily, rdates, HORIZONS)
            result[fname]["regime"][rname] = tg
            print(f"\n  [{rname.upper()}]  tail-gap bottom-minus-top / spearman IC")
            print(f"    {'h':<4}{'n':<6}{'bottom%':<9}{'top%':<9}{'gap':<8}{'CI90':<16}{'IC':<7}{'sig'}")
            for h in HORIZONS:
                r = tg[h]
                if "error" in r:
                    print(f"    {h}d  {r['error']}")
                else:
                    print(f"    {h}d  {r['n']:<6}{r['bottom_mean_fwd']:<9}{r['top_mean_fwd']:<9}"
                          f"{r['gap_bottom_minus_top']:<8}{str(r['ci90']):<16}{r['spearman_ic']:<7}"
                          f"{'SIG' if r['significant'] else ''}")
        # purged-CV AUC per regime for a couple horizons
        result[fname]["purged_cv_auc"] = {}
        for rname, rdates in [("expansion", exp_dates), ("contraction", con_dates)]:
            for h in (3, 7, 30):
                auc = purged_cv_auc(fz, btc_daily, rdates, h)
                result[fname]["purged_cv_auc"].setdefault(rname, {})[h] = auc
            print(f"\n  [{rname.upper()}] purged-CV OOS AUC (P(fwd>0 | flow)):")
            for h in (3, 7, 30):
                a = result[fname]["purged_cv_auc"][rname][h]
                if a:
                    print(f"    h={h}d: AUC={a['auc']} (CI90_lo={a['ci90_lo']}) n={a['n']}")
                else:
                    print(f"    h={h}d: insufficient data")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
