#!/usr/bin/env python3
"""
causal_glf_btc_early_era.py — GLF -> BTC causal battery RESTRICTED to era 2012-2017
===================================================================================
Extends the existing full-sample GLF->BTC causal suite (causal_glf_btc_full.py) to
the EARLY era. The existing suite uses FRED CBBTCUSD (which only starts 2014). This
script injects Kaggle Bitstamp 1-min->daily BTC (2012-01 onward) so the early era
2012-2017 becomes testable — previously impossible with the 2014+ FRED BTC series.

Reuses the reconstruction + test functions from causal_liquidity_btc.py (no dup).
Runs the SAME battery (Granger, conditional-regime, weekly OLS, VAR-Granger,
walk-forward OOS, power, posterior) but on months within [2012-01, 2017-12] only.

Honest framing (project standard): negative results reported as negative. This is a
CAUSAL CHECK before anything is blended; the full-sample suite already showed NO
confirmed GLF->BTC causality at monthly cadence (2026-08). This tests whether the
EARLY era was different.

USAGE:
    cd ~/sfc && export FRED_API_KEY=$(grep -oP '(?<=FRED_API_KEY=).*' .env | tr -d '"')
    .venv/bin/python analysis/causal_glf_btc_early_era.py
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

from causal_liquidity_btc import (
    build_monthly_glf, monthly_btc, granger,
    conditional_regime_granger, lagged_ols, LAGS_OLS_WEEKS,
)
# Reuse the hard tests from the full battery (they take (glf, btc_ret) dicts).
from causal_glf_btc_full import var_granger, walk_forward_oos, power_analysis, posterior_prob

OUTPUT = os.path.join(SFC_ROOT, ".causal_glf_btc_early_era.json")

ERA_START = "2012-01"
ERA_END   = "2017-12"


def load_kaggle_daily(path="/tmp/btc_2012_2017_daily.csv"):
    """Load Kaggle daily BTC as {date: close} dict (date = YYYY-MM-DD)."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['dt'])
    out = {}
    for _, r in df.iterrows():
        out[r['dt'].strftime('%Y-%m-%d')] = float(r['close'])
    return out


def restrict(glf, btc_ret, start=ERA_START, end=ERA_END):
    """Keep only months within [start, end]."""
    g = {m: v for m, v in glf.items() if start <= m <= end}
    b = {m: v for m, v in btc_ret.items() if start <= m <= end}
    return g, b


def main():
    print("=" * 74)
    print("CAUSAL BATTERY GLF -> BTC — EARLY ERA 2012-2017 (Kaggle BTC)")
    print("=" * 74)

    print("\n[0] Loading BTC daily (Kaggle Bitstamp) + reconstructing GLF...")
    btc_daily = load_kaggle_daily()
    glf_full = build_monthly_glf(full=True)
    btc_ret = monthly_btc(btc_daily)
    print(f"    Kaggle BTC daily={len(btc_daily)}  GLF monthly={len(glf_full)}  "
          f"BTC monthly ret={len(btc_ret)}")
    print(f"    GLF range: {min(glf_full)} .. {max(glf_full)}")
    print(f"    BTC ret range: {min(btc_ret)} .. {max(btc_ret)}")

    glf, btr = restrict(glf_full, btc_ret)
    common = sorted(set(glf) & set(btr))
    print(f"\n    ERA {ERA_START}..{ERA_END}: GLF months={len(glf)}  BTC ret={len(btr)}  "
          f"common={len(common)} ({common[0]}..{common[-1]})")
    if len(common) < 24:
        print("    !! Too few overlapping months for a causal battery — STOP.")
        sys.exit(1)

    result = {
        "generated_at": datetime.now().isoformat(),
        "era": f"{ERA_START}..{ERA_END}",
        "btc_source": "Kaggle Bitstamp 1-min->daily",
        "n_common_months": len(common),
    }

    print("\n[1] Granger (4 lags), GLF<->BTC, era 2012-2017...")
    g = granger(glf, btr)
    result["granger"] = g
    print(f"   GLF->BTC min_p={g.get('GLF->BTC',{}).get('min_p')}  "
          f"| BTC->GLF min_p={g.get('BTC->GLF',{}).get('min_p')}")

    print("[2] Conditional-regime Granger (bull/bear)...")
    g_regime = conditional_regime_granger(glf, btr, btc_daily)
    result["granger_conditional_regime"] = g_regime
    for k, v in g_regime.items():
        print(f"   {k}: n={v.get('n_months')} min_p={v.get('min_p')}")

    print("[3] Weekly lagged OLS (level & change)...")
    series = None
    from causal_liquidity_btc import weekly_series
    series = weekly_series(btc_daily, glf_full)
    # restrict weekly series to era
    series_era = [s for s in series if ERA_START <= s[0][:7] <= ERA_END]
    print(f"   weekly points in era: {len(series_era)} (of {len(series)})")
    result["ols_weekly_level"] = lagged_ols(series_era, LAGS_OLS_WEEKS, use_change=False)
    result["ols_weekly_change"] = lagged_ols(series_era, LAGS_OLS_WEEKS, use_change=True)
    sig = [k for k in LAGS_OLS_WEEKS if result["ols_weekly_level"].get(k, {}).get("sig")]
    print(f"   weekly level significant lags: {sig or 'none'}")

    print("[4] VAR-based Granger, BIC-chosen lag...")
    va = var_granger(glf, btr)
    result["var_granger"] = va
    if "error" not in va:
        print(f"   lag={va['lag_chosen_bic']}  GLF->BTC p={va['GLF_to_BTC']['p']}  "
              f"LjungBox_resid_p={va['ljung_box_resid_p']}")

    print("[5] Walk-forward OOS (AR vs AR+GLF)...")
    for p in (1, 2, 3):
        wf = walk_forward_oos(glf, btr, p=p, min_train=18)
        result.setdefault("walk_forward_oos", {})[p] = wf
        if "error" not in wf:
            print(f"   p={p}: OOS R^2={wf['oos_r2_vs_ar']}  "
                  f"MSE ratio={wf['mse_ratio']}  DM p={wf['dm_p']}  -> {wf['verdict']}")

    print("[6] Power analysis...")
    pw = power_analysis(glf, btr, p=2)
    result["power_analysis"] = pw
    if "error" not in pw:
        print(f"   achieved power={pw['achieved_power_observed_effect']}  "
              f"min detectable f^2@80%={pw['min_detectable_f2_at_80pct']}  "
              f"n_needed={pw['n_needed_to_detect_observed_f2']}")

    print("[7] Posterior probability of a real effect (BIC)...")
    pp = posterior_prob(glf, btr, p=2)
    result["posterior_prob_H1"] = pp
    if "error" not in pp:
        print(f"   P(H1|data) = {pp['posterior_prob_H1']}  "
              f"(BF_10={pp['bayes_factor_10']})  -> {pp['label']}")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
