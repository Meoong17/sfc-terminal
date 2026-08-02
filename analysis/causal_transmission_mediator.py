#!/usr/bin/env python3
"""
causal_transmission_mediator.py — GLF -> BankCredit -> BTC (mediation)
======================================================================
Tests the reframed SFC hypothesis from /home/ubuntu/A (indirect causality):
instead of GLF -> BTC directly, liquidity flows through an intermediate
transmission channel. This script tests the CREDIT channel:

    GLF (global liquidity)  ->  TOTLL (bank credit)  ->  BTC

Why TOTLL: it is the one transmission mediator with a LONG history available
from FRED ("Loans and Leases in Bank Credit, All Commercial Banks", weekly,
2010+). The other candidate mediators have data constraints:
  - Stablecoin supply: only ~1yr of local cache, CoinGecko rate-limited
  - ETF flow: spot ETFs only exist since Jan 2024 (~2.5yr window)
So bank credit is the defensible, sample-sufficient credit-transmission test.

METHOD — classic mediation (Baron & Kenny 1986), on monthly series:
  Step 1: BTC ~ GLF            (total effect c)
  Step 2: BankCredit ~ GLF     (path a)
  Step 3: BTC ~ GLF + BankCredit  (direct c' + path b)
  Indirect effect  = a * b
  Proportion mediated = a*b / c
  Sobel test + bootstrap CI for the indirect effect.
Control: BTC prior return in all steps (autoregressive).

HONEST framing: if the indirect effect is significant, credit transmission is
supported; if not, the relationship is not explained by the credit channel
either. Reported as-is. Display-only research.

USAGE:
    cd ~/sfc && export FRED_API_KEY=...
    .venv/bin/python analysis/causal_transmission_mediator.py
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

try:
    from statsmodels.api import OLS, add_constant
except ImportError as e:
    print(f"[Mediator] statsmodels unavailable: {e}", file=sys.stderr)
    sys.exit(1)

from causal_liquidity_btc import (
    build_monthly_glf, monthly_btc, fetch_series_dict, _z,
)

OUTPUT = os.path.join(SFC_ROOT, ".causal_transmission_mediator.json")
N_BOOTSTRAP = 2000


def bank_credit_monthly():
    """TOTLL weekly -> monthly YoY % change (bank credit growth)."""
    from historical_backtest_m1m6 import fetch_fred_series
    raw = fetch_fred_series("TOTLL", start_date="2002-01-01")
    # monthly avg of weekly
    month_avg = {}
    count = {}
    for d, v in raw.items():
        m = d[:7]
        month_avg[m] = month_avg.get(m, 0) + v
        count[m] = count.get(m, 0) + 1
    monthly = {m: month_avg[m] / count[m] for m in month_avg}
    mons = sorted(monthly)
    yoy = {}
    for i in range(12, len(mons)):
        p0, p1 = monthly[mons[i - 12]], monthly[mons[i]]
        if p0 and p0 != 0:
            yoy[mons[i]] = (p1 - p0) / p0 * 100
    return yoy


def fit(y, X):
    m = OLS(y, add_constant(np.array(X))).fit()
    return m


def bootstrap_indirect(rows, a_path, n_boot=N_BOOTSTRAP):
    """rows: list of (btc, glf, bc, btc_prev). Bootstrap the a*b product."""
    inds = []
    n = len(rows)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        s = [rows[i] for i in idx]
        btc = np.array([r[0] for r in s])
        glf = np.array([r[1] for r in s])
        bc = np.array([r[2] for r in s])
        prev = np.array([r[3] for r in s])
        # path a: bc ~ glf (+ prev)
        ma = fit(bc, [[g, p] for g, p in zip(glf, prev)])
        a_ = ma.params[1]
        # path b: btc ~ glf + bc (+ prev)
        mb = fit(btc, [[g, c, p] for g, c, p in zip(glf, bc, prev)])
        b_ = mb.params[2]
        inds.append(a_ * b_)
    inds = np.array(inds)
    return round(float(a_path), 4), round(float(inds.mean()), 4), \
        round(float(np.percentile(inds, 5)), 4), round(float(np.percentile(inds, 95)), 4)


def main():
    print("=" * 72)
    print("MEDIATION — GLF -> Bank Credit (TOTLL) -> BTC")
    print("Credit-transmission channel (indirect causality)")
    print("=" * 72)

    print("\n[1/4] Fetching data...")
    glf = build_monthly_glf(full=True)
    bc = bank_credit_monthly()
    btc_daily = fetch_series_dict("CBBTCUSD")
    btc_ret = monthly_btc(btc_daily)
    print(f"      GLF {len(glf)}m, BankCredit {len(bc)}m, BTC {len(btc_ret)}m")

    common = sorted(set(glf) & set(bc) & set(btc_ret))
    print(f"      {len(common)} aligned months")
    if len(common) < 40:
        print("⚠ insufficient data"); return

    rows = []
    for i, m in enumerate(common):
        if i == 0:
            prev = 0.0
        else:
            prev = btc_ret.get(common[i - 1], 0.0)
        rows.append([btc_ret[m], glf[m], bc[m], prev])

    btc = np.array([r[0] for r in rows])
    glf_a = np.array([r[1] for r in rows])
    bc_a = np.array([r[2] for r in rows])
    prev = np.array([r[3] for r in rows])

    print("\n[2/4] Path coefficients...")
    # Step 1: total effect c — btc ~ glf (+ prev)
    m1 = fit(btc, [[g, p] for g, p in zip(glf_a, prev)])
    c = m1.params[1]; c_p = m1.pvalues[1]
    # Step 2: path a — bc ~ glf (+ prev)
    m2 = fit(bc_a, [[g, p] for g, p in zip(glf_a, prev)])
    a_ = m2.params[1]; a_p = m2.pvalues[1]
    # Step 3: direct c' & path b — btc ~ glf + bc (+ prev)
    m3 = fit(btc, [[g, c2, p] for g, c2, p in zip(glf_a, bc_a, prev)])
    cp_ = m3.params[1]; cp_p = m3.pvalues[1]
    b_ = m3.params[2]; b_p = m3.pvalues[2]

    print(f"  Step1 TOTAL  (BTC~GLF):   c ={c:+.3f}  p={c_p:.4f}")
    print(f"  Step2 path-a (BC~GLF):    a ={a_:+.3f}  p={a_p:.4f}")
    print(f"  Step3 path-b (BTC~BC):    b ={b_:+.3f}  p={b_p:.4f}")
    print(f"  Step3 DIRECT (BTC~GLF|BC):c'={cp_:+.3f}  p={cp_p:.4f}")

    indirect = a_ * b_
    total = c
    prop = indirect / total if total != 0 else None
    print(f"\n  Indirect effect = a*b = {indirect:+.3f}")
    print(f"  Proportion mediated = {prop if prop is None else round(prop,3)}")

    # Sobel z-test
    # se(a*b) = sqrt(b^2*se_a^2 + a^2*se_b^2)
    se_a = m2.bse[1]; se_b = m3.bse[2]
    se_ab = np.sqrt((b_ ** 2) * (se_a ** 2) + (a_ ** 2) * (se_b ** 2))
    z = indirect / se_ab if se_ab else np.nan
    from scipy import stats as _st
    p_sobel = 2 * (1 - _st.norm.cdf(abs(z)))
    print(f"  Sobel z = {z:.3f}  p = {p_sobel:.4f}")

    # bootstrap CI
    a_b, ind_mean, ci_lo, ci_hi = bootstrap_indirect(rows, a_)
    print(f"  Bootstrap indirect 5-95% CI: [{ci_lo}, {ci_hi}]")
    sig_boot = not (ci_lo <= 0 <= ci_hi)

    print("\n[3/4] VERDICT")
    # Mediation requires: (1) a & b significant, (2) indirect sig, (3) c' < c.
    cond_a = a_p < 0.05
    cond_b = b_p < 0.05
    cond_ind = p_sobel < 0.05 or sig_boot
    cond_reduce = (cp_ if cp_ is not None else 0) < (c if c is not None else 0)
    verdict = []
    if cond_a and cond_b and cond_ind and cond_reduce:
        verdict = "MEDIATION SUPPORTED — GLF affects BTC substantially through bank credit."
    elif cond_a and cond_b and cond_ind:
        verdict = "PARTIAL — indirect path significant but direct effect not reduced (inconsistent mediation)."
    else:
        verdict = ("NOT SUPPORTED — the credit channel does not explain a GLF->BTC link. "
                   "The (already weak) GLF->BTC effect is not mediated by bank credit either.")
    print("  " + verdict)

    result = {
        "generated_at": datetime.now().isoformat(),
        "method": "Baron-Kenny mediation + Sobel + bootstrap (monthly, control BTC autoregressive)",
        "note": "Credit channel: GLF -> TOTLL(bank credit) -> BTC. Stablecoin/ETF excluded: "
                "no long history. Display-only research.",
        "n_months": len(common),
        "total_effect_c": round(float(c), 4), "total_p": round(float(c_p), 4),
        "path_a": round(float(a_), 4), "path_a_p": round(float(a_p), 4),
        "path_b": round(float(b_), 4), "path_b_p": round(float(b_p), 4),
        "direct_cp": round(float(cp_), 4), "direct_p": round(float(cp_p), 4),
        "indirect_ab": round(float(indirect), 4),
        "proportion_mediated": round(float(prop), 4) if prop is not None else None,
        "sobel_z": round(float(z), 3), "sobel_p": round(float(p_sobel), 4),
        "bootstrap_indirect_ci_5_95": [ci_lo, ci_hi],
        "verdict": verdict,
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
