#!/usr/bin/env python3
"""
behavior_reading_validity_test.py — Validate REJECTED factors as MEASUREMENT
                              instruments of BTC market BEHAVIOR (not predictors)
====================================================================================
SFC's objective is to READ/MEASURE BTC market behavior from liquidity/macro/
positioning/flow/regime state — NOT to forecast price. So the correct validation
bar for a factor is not "does it predict forward return OOS" but:

  Does the reading actually track BTC BEHAVIOR STATE it claims to measure?

Three measurement-validity tests (all CONTEMPORANEOUS — no forecasting):
  A. CRISIS-ELEVATION (Pitfall 9): in known BTC-behavior crisis windows, does a
     stress-type reading elevate above its 180-day prior control (sign-aware)?
     A stress reading that doesn't rise in real crises fails to measure stress.
  B. STATE-DISCRIMINATION: split the reading into terciles; do high/low states
     coincide with DIFFERENT concurrent BTC behavior (realized vol, downside
     semideviation, max drawdown)? A valid reader partitions behavior states.
  C. CONVERGENT VALIDITY: contemporaneous Spearman between the reading and
     concurrent behavior dimensions (sign per construct).

Behavior dimensions (the "behavior of the market" a reader should track):
  realized_vol (30d), downside_semidev (30d), max_drawdown (30d), worst_day (30d),
  trend state (200DMA).

Factors re-examined as READERS (previously rejected as predictors):
  TERM_PREM  = SLR M91 score (sovereign duration stress / term-premium proxy)
  GLF        = global liquidity state (benchmark, already validated)
  M2_IMPULSE = z(ΔM2) liquidity impulse
  ORDER_FLOW = taker imbalance ratio (price-derived; interpret with care)
  ETF_FLOW   = net BTC ETF flow (price-derived)

Sign hypotheses:
  stress-type (TERM_PREM, and LOW GLF / LOW liquidity)  -> HIGH vol, HIGH downside
  ORDER_FLOW / ETF_FLOW (flow)                          -> price-derived, reported but
                                                           not a clean external state read

USAGE:
    cd ~/sfc && export FRED_API_KEY=... && .venv/bin/python analysis/behavior_reading_validity_test.py
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

from historical_backtest_m1m6 import fetch_fred_series
from causal_liquidity_btc import build_monthly_glf
from scipy.stats import spearmanr

OUTPUT = os.path.join(SFC_ROOT, ".behavior_reading_test.json")
SLR_JSON = os.path.join(SFC_ROOT, ".slr_series.json")

# Known BTC-behavior stress windows (month_start, month_end)
CRISIS = [
    ("2020-03", "2020-04", "COVID crash"),
    ("2022-05", "2022-07", "Luna / 3AC"),
    ("2022-11", "2022-12", "FTX collapse"),
    ("2024-08", "2024-08", "Carry-trade unwind"),
]


# --------------------------------------------------------------------------- #
def load_factors():
    """Return dict of daily factor series."""
    f = {}
    # term premium (SLR M91 score 0-100)
    slr = json.load(open(SLR_JSON))["daily"]
    f["term_prem"] = {d: v["m91"] for d, v in slr.items()}
    # order flow (taker imbalance)
    of = json.load(open(os.path.join(SFC_ROOT, "data/binance_orderflow_daily.json")))
    ofz = {}
    for k, r in of.items():
        tq = r.get("total_qty")
        if tq and tq > 0:
            ofz[k] = (r["taker_buy_qty"] - r["taker_sell_qty"]) / tq
    f["order_flow"] = rolling_z(ofz, 90)
    # ETF flow
    etf = json.load(open(os.path.join(SFC_ROOT, ".etf_cache.json")))
    ef = {}
    for e in etf.get("flows", []):
        tb = e.get("total_btc")
        if tb is None:
            et = e.get("etfs")
            if isinstance(et, dict) and et:
                tb = sum(v for v in et.values() if v is not None)
        if tb is not None:
            ef[e["date"]] = float(tb)
    f["etf_flow"] = rolling_z(ef, 60)
    return f


def rolling_z(series, window=90, min_n=60):
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


def monthly_to_daily(monthly, btc_dates):
    """Forward-fill last known monthly value onto each date."""
    months = sorted(monthly)
    out = {}
    mi = 0
    for d in btc_dates:
        while mi + 1 < len(months) and months[mi + 1] <= d[:7]:
            mi += 1
        if months[mi] <= d[:7]:
            out[d] = monthly[months[mi]]
    return out


# --------------------------------------------------------------------------- #
def behavior_metrics(btc_daily):
    """Daily trailing behavior metrics. Returns dict of {date: value} per metric,
    plus trend state and a price series aligned to the factor dates."""
    pdates = sorted(btc_daily)
    closes = np.array([btc_daily[d] for d in pdates])
    rets = np.zeros(len(pdates))
    rets[1:] = closes[1:] / closes[:-1] - 1.0
    W = 30
    metrics = {"realized_vol": {}, "downside_semidev": {}, "maxdd": {}, "worst_day": {}}
    for i in range(W, len(pdates)):
        r = rets[i - W:i]
        vol = float(np.std(r) * np.sqrt(365))
        neg = r[r < 0]
        dsd = float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(365)) if len(neg) else 0.0
        peak = np.maximum.accumulate(closes[i - W:i])
        mdd = float((closes[i - W:i] - peak).min() / peak.max()) if peak.max() else 0.0
        metrics["realized_vol"][pdates[i]] = vol
        metrics["downside_semidev"][pdates[i]] = dsd
        metrics["maxdd"][pdates[i]] = mdd * 100.0
        metrics["worst_day"][pdates[i]] = float(r.min()) * 100.0
    return metrics, pdates


def align(factor, metrics, pdates):
    """Rows where factor and behavior metrics all present."""
    rows = []
    for d in factor:
        if d in metrics["realized_vol"]:
            rows.append({"date": d, "factor": factor[d],
                         **{k: metrics[k][d] for k in metrics}})
    return rows


# --------------------------------------------------------------------------- #
# A. Crisis-elevation (sign-aware)
# --------------------------------------------------------------------------- #
def crisis_elevation(factor, pdates, factor_name, sign):
    """For each crisis window: mean(factor) in-window vs 180d prior control.
    sign: +1 for stress-type (elevate in crisis), -1 for liquidity (drop in crisis).
    Valid if in-window moves the expected direction vs control by >0."""
    out = {}
    for (ms, me, label) in CRISIS:
        in_dates = [d for d in factor if ms <= d[:7] <= me]
        if not in_dates:
            out[label] = {"error": "no dates in window"}
            continue
        # prior 180 days control
        w0 = datetime.strptime(ms, "%Y-%m")
        control = []
        for d in factor:
            dt = datetime.strptime(d, "%Y-%m-%d")
            if w0 - timedelta(days=180) <= dt < w0:
                control.append(factor[d])
        if len(control) < 20:
            out[label] = {"error": f"control n={len(control)}"}
            continue
        win_mean = float(np.mean([factor[d] for d in in_dates]))
        ctl_mean = float(np.mean(control))
        delta = win_mean - ctl_mean          # raw move
        correct = (delta * sign) > 0         # moved expected direction
        out[label] = {"win_mean": round(win_mean, 2), "control_mean": round(ctl_mean, 2),
                      "delta": round(delta, 2), "correct_direction": correct}
    return out


# --------------------------------------------------------------------------- #
# B. State-discrimination (contemporaneous behavior across factor terciles)
# --------------------------------------------------------------------------- #
def state_discrimination(rows, behavior_cols):
    """Split factor into terciles; bootstrap diff between top and bottom tercile on
    each concurrent behavior metric. A valid reader shows separation (sign-aware
    interpretation done by caller)."""
    if len(rows) < 90:
        return {"error": f"n={len(rows)}"}
    rows = sorted(rows, key=lambda r: r["factor"])
    n = len(rows)
    tn = n // 3
    bottom = rows[:tn]
    top = rows[-tn:]
    rng = np.random.default_rng(42)
    out = {"n": n, "tercile_n": tn}
    for col in behavior_cols:
        b = np.array([r[col] for r in bottom])
        t = np.array([r[col] for r in top])
        diff = b.mean() - t.mean()   # bottom_tercile - top_tercile
        nboot = 5000
        bd = np.empty(nboot)
        for i in range(nboot):
            bd[i] = rng.choice(b, len(b), replace=True).mean() - rng.choice(t, len(t), replace=True).mean()
        lo, hi = np.percentile(bd, [5, 95])
        out[col] = {"bottom_mean": round(float(b.mean()), 3), "top_mean": round(float(t.mean()), 3),
                    "bottom_minus_top": round(float(diff), 3),
                    "ci90": [round(float(lo), 3), round(float(hi), 3)],
                    "significant": bool(lo > 0 or hi < 0)}
    return out


# --------------------------------------------------------------------------- #
# C. Convergent validity (contemporaneous Spearman with behavior)
# --------------------------------------------------------------------------- #
def convergent(rows, behavior_cols):
    out = {}
    for col in behavior_cols:
        v, p = spearmanr([r["factor"] for r in rows], [r[col] for r in rows])
        out[col] = {"spearman": round(float(v), 3), "p": round(float(p), 4)}
    return out


# --------------------------------------------------------------------------- #
SIGN = {  # +1 stress-type elevates in crisis, -1 liquidity-type drops in crisis
    "term_prem": 1, "glf": -1, "m2_impulse": -1, "order_flow": None, "etf_flow": None,
}

def main():
    print("=" * 76)
    print("BEHAVIOR-READING VALIDITY TEST (SFC objective = read behavior, not forecast)")
    print("=" * 76)

    print("\n[1] Loading factors + BTC...")
    f = load_factors()
    btc = fetch_fred_series("CBBTCUSD")
    # GLF + M2 impulse (monthly -> daily)
    glf = build_monthly_glf(full=True)
    from liquidity_impulse_test import m2_impulse, m2_yoy
    m2 = fetch_fred_series("M2SL", start_date="2009-01-01")
    imp = m2_impulse(m2)
    li = {m: v["LI"] for m, v in imp.items() if "LI" in v}
    f["glf"] = monthly_to_daily(glf, sorted(btc))
    f["m2_impulse"] = monthly_to_daily(li, sorted(btc))

    print("    factor sizes:", {k: len(v) for k, v in f.items()})

    metrics, pdates = behavior_metrics(btc)
    print("    behavior metrics over", len(metrics["realized_vol"]), "days")

    BEHAV = ["realized_vol", "downside_semidev", "maxdd", "worst_day"]
    result = {"generated_at": datetime.now().isoformat(),
              "behavior_dims": BEHAV, "crisis_windows": CRISIS}

    for fname, fser in f.items():
        rows = align(fser, metrics, pdates)
        print(f"\n{'='*76}\nFACTOR: {fname}  (n={len(rows)})\n{'='*76}")
        result[fname] = {"n": len(rows)}

        # A. crisis-elevation (only for macro/state readings with a clear sign)
        if SIGN[fname] is not None:
            ce = crisis_elevation(fser, pdates, fname, SIGN[fname])
            result[fname]["crisis_elevation"] = ce
            print("  [A] Crisis-elevation (sign-aware, vs 180d control):")
            for lab, v in ce.items():
                if "error" in v:
                    print(f"      {lab:<22} {v['error']}")
                else:
                    print(f"      {lab:<22} delta={v['delta']:+.2f}  "
                          f"correct_dir={v['correct_direction']}  "
                          f"(win {v['win_mean']} vs ctl {v['control_mean']})")
        else:
            print("  [A] skipped (price-derived, no clean sign)")

        # B. state-discrimination
        sd = state_discrimination(rows, BEHAV)
        result[fname]["state_discrimination"] = sd
        print("  [B] State-discrimination (bottom-top tercile, concurrent behavior):")
        if "error" in sd:
            print(f"      {sd['error']}")
        else:
            for col in BEHAV:
                s = sd[col]
                print(f"      {col:<18} b-t={s['bottom_minus_top']:+.3f} "
                      f"CI90={s['ci90']} {'SIG' if s['significant'] else ''}")

        # C. convergent validity
        cv = convergent(rows, BEHAV)
        result[fname]["convergent"] = cv
        print("  [C] Contemporaneous Spearman:")
        for col in BEHAV:
            c = cv[col]
            print(f"      {col:<18} rho={c['spearman']} p={c['p']}")

    with open(OUTPUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
