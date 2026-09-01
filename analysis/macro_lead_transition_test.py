#!/usr/bin/env python3
"""
macro_lead_transition_test.py — decisive open test from 2.docx: do external
    macro/liquidity conditions CHANGE BEFORE BTC behaviour switches regime?
    (lead/transition information — NOT price forecasting)
================================================================================
For each external condition (GLF, VIX, REAL_Y10, term-premium spread, HY OAS,
sovereign-stress M91) we test whether it drifts systematically in the WINDOW
BEFORE a BTC behaviour-regime transition (bull<->bear, and calm<->stress).

If a variable moves before regime flips (pre-transition drift, in the direction
of the upcoming regime), it has LEAD/transition info -> deserves a module.
If it only reacts after/at the flip (or not at all), it is context only.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from historical_backtest_m1m6 import fetch_fred_series

def load_btc():
    raw = fetch_fred_series('CBBTCUSD', start_date='2014-01-01')
    dates = sorted(raw)
    closes = np.array([raw[d] for d in dates])
    return dates, closes

def build_regimes(dates, closes):
    """Returns dict date-> {'trend':1/0 (bull/bear 200DMA), 'stress':1/0 (vol>p90)}."""
    n = len(closes)
    vol = np.full(n, np.nan)
    for i in range(29, n):
        r = np.diff(np.log(closes[max(0, i - 29):i + 1]))
        vol[i] = np.std(r) * np.sqrt(365)
    p90 = np.nanpercentile(vol, 90)
    out = {}
    for i in range(n):
        d = dates[i]
        trend = 1 if closes[i] >= np.mean(closes[max(0, i - 200):i + 1]) else 0
        stress = 1 if (not np.isnan(vol[i]) and vol[i] >= p90) else 0
        out[d] = {'trend': trend, 'stress': stress}
    return out, {'vol_p90': p90}

def transitions(regimes, key):
    dts = sorted(regimes)
    out = []
    prev = regimes[dts[0]][key]
    for d in dts[1:]:
        cur = regimes[d][key]
        if cur != prev:
            out.append((d, prev, cur))   # date, from_state, to_state
        prev = cur
    return out

def load_macro():
    m = {}
    for rec in json.load(open('data/cleaned/macro_daily_clean.json')):
        d = rec['date']
        m[d] = rec
    return m

def load_m91():
    s = json.load(open(os.path.join(os.path.dirname(__file__), '..', '.slr_series.json')))
    m91 = {}
    for d in s.get('daily', {}):
        m91[d] = s['daily'][d].get('m91')
    return m91

def load_glf():
    from causal_liquidity_btc import build_monthly_glf
    g = build_monthly_glf(full=True)   # { 'YYYY-MM': glf }
    out = {}
    months = sorted(g)
    last = None
    # forward-fill month value to daily
    for rec in json.load(open('data/cleaned/macro_daily_clean.json')):
        d = rec['date']
        mm = d[:7]
        if mm in g:
            last = g[mm]
        if last is not None:
            out[d] = last
    return out

def event_study(trans, macro_series, pre_wins, label):
    """For each transition, mean macro over pre-windows; compare to sample baseline."""
    base_vals = np.array([v for v in macro_series.values() if v is not None and not np.isnan(v)])
    if len(base_vals) == 0:
        return None
    base_mean, base_std = base_vals.mean(), base_vals.std()
    dts = sorted(macro_series)
    idx = {d: i for i, d in enumerate(dts)}
    print(f"\n  [{label}] pre-transition drift vs sample baseline "
          f"(baseline mean={base_mean:.3f} sd={base_std:.3f}, n_trans={len(trans)}):")
    for w in pre_wins:
        vals = []
        for (td, frm, to) in trans:
            ti = idx.get(td)
            if ti is None:
                continue
            lo = idx[dts[max(0, ti - w)]]
            seg = [macro_series[dts[j]] for j in range(lo, ti)
                   if macro_series[dts[j]] is not None and not np.isnan(macro_series[dts[j]])]
            if seg:
                vals.append(np.mean(seg))
        if vals:
            mv = np.mean(vals)
            z = (mv - base_mean) / base_std if base_std > 0 else 0
            print(f"    window[-{w}:-1] mean={mv:.3f}  z={z:+.2f}  (n_trans={len(vals)})")
    # direction-of-change over pre-window (mean macro at T vs T-30)
    chg = []
    for (td, frm, to) in trans:
        ti = idx.get(td)
        if ti is None:
            continue
        j0 = idx[dts[max(0, ti - 30)]]
        a = macro_series[dts[j0]]; b = macro_series[dts[ti]]
        if None not in (a, b) and not np.isnan(a) and not np.isnan(b):
            chg.append(b - a)
    if chg:
        c = np.mean(chg)
        print(f"    mean Δmacro over [-30,0] = {c:+.3f}  (sign of pre-transition move)")

def lead_lag(trans, macro_series, label, max_lag=30):
    """Cross-correlation: does Δmacro at lag -l correlate with future regime flip?"""
    dts = sorted(macro_series)
    idx = {d: i for i, d in enumerate(dts)}
    flip = np.zeros(len(dts))
    for (td, frm, to) in trans:
        if td in idx:
            flip[idx[td]] = 1.0
    X = np.array([macro_series[d] if macro_series[d] is not None and not np.isnan(macro_series[d]) else np.nan for d in dts])
    dX = np.diff(X)
    n = len(dX)
    print(f"  [{label}] cross-corr corr(dX[t-l], flip[t]) for l in 0..{max_lag}:")
    sig = []
    for l in range(0, max_lag + 1):
        # dX at t-l (leads) vs flip at t
        a = dX[:n - l]; b = flip[l:n]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() > 30:
            r = np.corrcoef(a[m], b[m])[0, 1]
            sig.append((l, r))
    if sig:
        # print strongest |r| and lags where |r|>0.15
        for l, r in sig:
            mark = ' <-- lead' if abs(r) > 0.15 and l > 0 else ''
            print(f"    lag{l}: r={r:+.3f}{mark}")
        bl = max(sig, key=lambda t: abs(t[1]))
        print(f"    strongest |r|={abs(bl[1]):.3f} at lag {bl[0]}")

def main():
    dates, closes = load_btc()
    regimes, _ = build_regimes(dates, closes)
    tr_trend = [t for t in transitions(regimes, 'trend')]
    tr_stress = [t for t in transitions(regimes, 'stress')]
    print(f"BTC days={len(dates)}  trend transitions={len(tr_trend)}  stress transitions={len(tr_stress)}")

    macro = load_macro()
    m91 = load_m91()
    glf = load_glf()
    series = {
        'VIX': {d: r.get('VIX') for d, r in macro.items()},
        'REAL_Y10': {d: r.get('REAL_Y10') for d, r in macro.items()},
        'TERM_PREM(10-2)': {d: r.get('SPREAD_10_2') for d, r in macro.items()},
        'US30-US2': {d: (r.get('US30Y') - r.get('US2Y')) if r.get('US30Y') is not None and r.get('US2Y') is not None else None for d, r in macro.items()},
        'GLF': glf,
        'SOV_STRESS(M91)': m91,
    }

    for key, target in [('trend', tr_trend), ('stress', tr_stress)]:
        print(f"\n===== regime: {key.upper()} (n_trans={len(target)}) =====")
        for name, s in series.items():
            event_study(target, s, [7, 14, 30], f"{name} | {key}")
            lead_lag(target, s, f"{name} | {key}")

if __name__ == '__main__':
    main()
