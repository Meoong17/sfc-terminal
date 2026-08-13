#!/usr/bin/env python3
"""Robustness battery for the era-2012-2017 VAR Granger finding (p=0.020 @ lag 6).

The main battery found one nominally-significant cell (VAR-Granger GLF->BTC
p=0.0201 @ BIC lag=6) in 71 months. Per the project's robustness rule (significant
lag in a multi-lag search is USUALLY an artifact), we must check before reporting:
  (a) multiple-comparison across ALL lags 1..6 (Benjamini-Hochberg FDR)
  (b) sub-period split (first half vs second half)
  (c) drop-outlier sensitivity (|z|>1.5 on the BTC returns)
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))
from causal_liquidity_btc import build_monthly_glf, monthly_btc
from causal_glf_btc_full import var_granger
import pandas as pd

ERA_START, ERA_END = "2012-01", "2017-12"

def load_kaggle_daily():
    df = pd.read_csv("/tmp/btc_2012_2017_daily.csv", parse_dates=['dt'])
    return {r['dt'].strftime('%Y-%m-%d'): float(r['close']) for _, r in df.iterrows()}

def restrict(glf, btr):
    return ({m:v for m,v in glf.items() if ERA_START<=m<=ERA_END},
            {m:v for m,v in btr.items() if ERA_START<=m<=ERA_END})

glf_full = build_monthly_glf(full=True)
btc_ret  = monthly_btc(load_kaggle_daily())
glf, btr = restrict(glf_full, btc_ret)
common = sorted(set(glf) & set(btr))
print(f"common months = {len(common)} ({common[0]}..{common[-1]})")

# --- (a) FDR across all lags 1..6 in var_granger ---
def exclusion_p_values(glf, btr, max_lag=6):
    c = sorted(set(glf) & set(btr))
    n = len(c)
    y = np.array([btr[m] for m in c], float)
    x = np.array([glf[m] for m in c], float)
    from statsmodels.api import OLS
    from scipy import stats as sps
    pvals = {}
    for chosen in range(1, max_lag+1):
        Y = y[chosen:]
        Xf = [np.ones(n-chosen)]; Xr = [np.ones(n-chosen)]
        for k in range(1, chosen+1):
            Xf.append(y[chosen-k:n-k]); Xr.append(y[chosen-k:n-k])
        for k in range(1, chosen+1):
            Xf.append(x[chosen-k:n-k])
        rf = OLS(Y, np.column_stack(Xf)).fit()
        rr = OLS(Y, np.column_stack(Xr)).fit()
        rss_f = float(np.sum(rf.resid**2)); rss_r = float(np.sum(rr.resid**2))
        k1, k0 = len(Xf), len(Xr)
        F = ((rss_r-rss_f)/(k1-k0))/(rss_f/(n-k1))
        p = float(1 - sps.f.cdf(F, k1-k0, n-k1))
        pvals[chosen] = p
    return pvals

pvals = exclusion_p_values(glf, btr)
print("\n(a) Per-lag GLF->BTC exclusion p-values (no lag selection):")
for lag, p in pvals.items():
    print(f"    lag {lag}: p={p:.4f} {'*' if p<0.05 else ''}")

# BH-FDR across the 6 lags
p = np.array(sorted(pvals.values()))
m = len(p)
qvals = np.minimum.accumulate(p[::-1] * m / np.arange(m,0,-1))[::-1]
sig_bh = [lag for lag, pv in pvals.items() if pv < 0.05]
print(f"\n    Lags passing raw p<0.05: {sig_bh or 'none'}")
print(f"    Min p={min(pvals.values()):.4f}; BH critical p (m={m}) = 0.05*rank/m -> "
      f"need p <= {0.05/m:.4f} for the most extreme")
print(f"    Verdict: survives BH-FDR = {min(pvals.values()) <= 0.05/m}")

# --- (b) sub-period split (first vs second half of era) ---
half = len(common)//2
def split(d, cut):
    return {m:v for m,v in d.items() if m <= cut}
def var_on(glf_s, btr_s, label):
    r = var_granger(glf_s, btr_s)
    if "error" in r:
        print(f"    {label}: {r['error']}"); return
    print(f"    {label}: n={r['n_months']} lag={r['lag_chosen_bic']} "
          f"GLF->BTC p={r['GLF_to_BTC']['p']} {'SIG' if r['GLF_to_BTC']['p']<0.05 else 'n.s.'}")
print("\n(b) Sub-period split:")
cut = common[half-1]
var_on({m:v for m,v in glf.items() if m<=cut}, {m:v for m,v in btr.items() if m<=cut}, "first half")
var_on({m:v for m,v in glf.items() if m>cut}, {m:v for m,v in btr.items() if m>cut}, "second half")

# --- (c) drop-outlier (|z|>1.5 on BTC returns) ---
rets = np.array([btr[m] for m in common])
mu, sd = rets.mean(), rets.std()
mask = np.abs((rets-mu)/sd) <= 1.5
common_dr = [m for i,m in enumerate(common) if mask[i]]
glf_dr = {m:glf[m] for m in common_dr}
btr_dr = {m:btr[m] for m in common_dr}
print(f"\n(c) Drop-outlier (|z|>1.5): kept {len(common_dr)} of {len(common)} months")
r = var_granger(glf_dr, btr_dr)
if "error" not in r:
    print(f"    GLF->BTC p={r['GLF_to_BTC']['p']} {'SIG' if r['GLF_to_BTC']['p']<0.05 else 'n.s.'} "
          f"(lag {r['lag_chosen_bic']})")

print("\n" + "="*60)
print("ROBUSTNESS VERDICT")
print("="*60)
print(f"  Raw best p   = {min(pvals.values()):.4f} @ lag {min(pvals, key=pvals.get)}")
print(f"  BH-FDR gate  = {0.05/m:.4f} -> survives: {min(pvals.values()) <= 0.05/m}")
print(f"  Sub-period   = (see above; needs BOTH halves consistent)")
print(f"  Drop-outlier = (see above; needs to persist)")
