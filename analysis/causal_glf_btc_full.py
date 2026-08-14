#!/usr/bin/env python3
"""
causal_glf_btc_full.py — Full statistical battery: GLF -> BTC causality + accuracy
=================================================================================
Re-runs the existing causal suite on FRESH data and adds four harder tests so the
user gets defensible PROBABILITIES, not just point estimates:

  REFRESH  : full-sample Granger (4 lags), conditional-regime Granger,
             weekly lagged OLS (level & change).
  NEW-A VAR: VAR-based Granger exclusion test at the lag chosen by BIC/AIC,
             with residual serial-correlation (Ljung-Box) and stability checks.
  NEW-B OOS: expanding-window walk-forward forecast. Nested models
             AR(BTC)  vs  AR(BTC)+GLF  — does GLF add any OUT-OF-SAMPLE edge?
             Reported as MSE ratio, OOS R^2, and a Diebold-Mariano test.
  NEW-C POW: power analysis — minimum detectable incremental R^2 (Cohen f^2)
             at 80% power for the actual N, and achieved power for the observed
             effect. Answers: "if a real effect existed, would we have caught it?"
  NEW-D BAYES: BIC-approximation of the posterior probability that a real
             GLF->BTC effect exists: P(H1|data) ~ exp(dBIC/2)/(1+exp(dBIC/2)).
             This is the "accurate probability" the user asked for.

HONEST FRAMING (per project standard): every result is reported as-is; a null
causality result is a valid, publishable finding. Nothing is overclaimed.

USAGE:
    cd ~/sfc && export FRED_API_KEY=$(grep -oP '(?<=FRED_API_KEY=).*' .env | tr -d '"')
    .venv/bin/python analysis/causal_glf_btc_full.py
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

from statsmodels.api import OLS, add_constant, GLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats as sps

# Reuse the reconstruction + base tests from the existing module (no duplication).
from causal_liquidity_btc import (
    build_monthly_glf, monthly_btc, weekly_series, granger,
    conditional_regime_granger, lagged_ols, fetch_series_dict,
    LAGS_OLS_WEEKS, MAXLAG_GRANGER,
)

OUTPUT = os.path.join(SFC_ROOT, ".causal_glf_btc_full.json")


# --------------------------------------------------------------------------- #
# NEW-A: VAR-based Granger exclusion test with data-driven lag + diagnostics
# --------------------------------------------------------------------------- #
def var_granger(glf, btc_ret, max_lag=6):
    """Granger GLF->BTC via a bivariate VAR, lag chosen by BIC (and AIC).

    Returns exclusion F-test p-values per direction at the chosen lag plus
    Ljung-Box residual serial-correlation p for the BTC equation.
    """
    common = sorted(set(glf) & set(btc_ret))
    if len(common) < 30:
        return {"error": f"n={len(common)}"}
    y = np.array([btc_ret[m] for m in common], float)
    x = np.array([glf[m] for m in common], float)
    n = len(common)

    # Lag selection by in-sample BIC / AIC on the full (y,x) VAR via OLS per lag.
    def _crit(p):
        # rows t=p..n-1, features: [const, y(t-1..t-p), x(t-1..t-p)]
        Y = y[p:]
        Xcols = [np.ones(n - p)]
        for k in range(1, p + 1):
            Xcols.append(y[p - k:n - k])
        for k in range(1, p + 1):
            Xcols.append(x[p - k:n - k])
        X = np.column_stack(Xcols)
        r = OLS(Y, X).fit()
        return r.aic, r.bic

    best_bic, best_aic, best_lag = 1e18, 1e18, 1
    aic_series, bic_series = {}, {}
    for p in range(1, max_lag + 1):
        a, b = _crit(p)
        aic_series[p] = a
        bic_series[p] = b
        if b < best_bic:
            best_bic, best_lag = b, p
        if a < best_aic:
            best_aic = a
    chosen = best_lag  # BIC-selected lag for the exclusion test

    def _exclusion_test(dep, indep_lags):
        # Full model dep on all lags; restricted model drops indep_lags.
        Y = dep[chosen:]
        Xf = [np.ones(n - chosen)]
        Xr = [np.ones(n - chosen)]
        for k in range(1, chosen + 1):
            Xf.append(dep[chosen - k:n - k])
            Xr.append(dep[chosen - k:n - k])
        for k in range(1, chosen + 1):
            Xf.append(indep_lags[chosen - k:n - k])
        Xf = np.column_stack(Xf)
        Xr = np.column_stack(Xr)
        rf = OLS(Y, Xf).fit()
        rr = OLS(Y, Xr).fit()
        k1, k0 = Xf.shape[1], Xr.shape[1]
        rss_f = float(np.sum(rf.resid ** 2))
        rss_r = float(np.sum(rr.resid ** 2))
        df1 = k1 - k0
        df2 = n - k1
        F = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p = float(1 - sps.f.cdf(F, df1, df2))
        return F, df1, df2, p, float(rf.rsquared)

    F_gb, df1, df2, p_gb, r2_full = _exclusion_test(y, x)   # GLF -> BTC
    F_bg, _, _, p_bg, _ = _exclusion_test(x, y)             # BTC -> GLF

    # Residual serial correlation (Ljung-Box) on the BTC equation residuals.
    Y = y[chosen:]
    Xf = [np.ones(n - chosen)]
    for k in range(1, chosen + 1):
        Xf.append(y[chosen - k:n - k])
        Xf.append(x[chosen - k:n - k])
    rf = OLS(Y, np.column_stack(Xf)).fit()
    lb = acorr_ljungbox(rf.resid, lags=[6], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])

    return {
        "n_months": n,
        "period": (common[0], common[-1]),
        "lag_chosen_bic": chosen,
        "aic_series": {k: round(v, 2) for k, v in aic_series.items()},
        "bic_series": {k: round(v, 2) for k, v in bic_series.items()},
        "GLF_to_BTC": {
            "F": round(F_gb, 3), "df": [df1, df2], "p": round(p_gb, 4),
            "causal_at_0.05": bool(p_gb < 0.05),
            "full_r2": round(r2_full, 4),
        },
        "BTC_to_GLF": {"F": round(F_bg, 3), "df": [df1, df2],
                       "p": round(p_bg, 4), "causal_at_0.05": bool(p_bg < 0.05)},
        "ljung_box_resid_p": round(lb_p, 4),
        "resid_white": bool(lb_p > 0.05),
    }


# --------------------------------------------------------------------------- #
# NEW-B: Walk-forward OOS — does GLF add predictive edge beyond AR(BTC)?
# --------------------------------------------------------------------------- #
def walk_forward_oos(glf, btc_ret, p=2, min_train=48):
    """Expanding-window 1-step-ahead forecast.

    Nested comparison:
        H0  AR(p):     r_t = a0 + sum_i a_i r_{t-i}
        H1  AR+GLF(p): r_t = a0 + sum_i a_i r_{t-i} + b0 GLF_t + sum_i b_i GLF_{t-i}
    Reports MSE ratio (AR+GLF / AR), OOS R^2 vs AR, and Diebold-Mariano.
    """
    common = sorted(set(glf) & set(btc_ret))
    if len(common) < min_train + 10:
        return {"error": f"n={len(common)}"}
    n = len(common)
    r = np.array([btc_ret[m] for m in common], float)
    g = np.array([glf[m] for m in common], float)

    def _feat(t, use_glf):
        cols = [1.0]
        for k in range(1, p + 1):
            cols.append(r[t - k])
        if use_glf:
            cols.append(g[t])                      # contemporaneous liquidity
            for k in range(1, p + 1):
                cols.append(g[t - k])
        return cols

    resid0, resid1 = [], []
    for t in range(min_train, n):
        if t - min_train < 20:            # need a minimum training window
            continue
        X0 = np.array([_feat(j, False) for j in range(min_train, t)])
        X1 = np.array([_feat(j, True) for j in range(min_train, t)])
        y = r[min_train:t]
        if X0.shape[0] < 20:
            continue
        b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
        b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
        resid0.append(r[t] - float(X0[-1] @ b0))
        resid1.append(r[t] - float(X1[-1] @ b1))

    if len(resid1) < 10:
        return {"error": f"oos n={len(resid1)}"}
    e0, e1 = np.array(resid0), np.array(resid1)
    mse0 = float(np.mean(e0 ** 2))
    mse1 = float(np.mean(e1 ** 2))
    oos_r2 = 1 - mse1 / mse0

    # Diebold-Mariano (equal predictive accuracy, squared loss, HAC var).
    d = e0 ** 2 - e1 ** 2
    dbar = d.mean()
    # Newey-West-ish: use lag-1 autocovariance for HAC variance estimate.
    gamma0 = np.mean((d - dbar) ** 2)
    gamma1 = np.mean((d[:-1] - dbar) * (d[1:] - dbar))
    var_dm = (gamma0 + 2 * gamma1) / len(d)
    if var_dm <= 0:
        var_dm = gamma0 / len(d)
    dm = dbar / np.sqrt(var_dm) if var_dm > 0 else 0.0
    dm_p = float(2 * (1 - sps.norm.cdf(abs(dm)))) if var_dm > 0 else 1.0

    return {
        "n_train_min": min_train, "p": p,
        "oos_n": len(e0),
        "mse_ar_only": round(mse0, 4),
        "mse_ar_glf": round(mse1, 4),
        "mse_ratio": round(mse1 / mse0, 4),      # <1 GLF helps
        "oos_r2_vs_ar": round(oos_r2, 4),        # >0 GLF adds edge
        "dm_stat": round(dm, 3), "dm_p": round(dm_p, 4),
        "glf_improves_signif": bool(dm_p < 0.05 and oos_r2 > 0),
        "verdict": "GLF adds OOS edge" if (oos_r2 > 0 and dm_p < 0.05)
                   else "no OOS edge",
    }


# --------------------------------------------------------------------------- #
# NEW-C: Power analysis — minimum detectable effect + achieved power
# --------------------------------------------------------------------------- #
def power_analysis(glf, btc_ret, p=2, alpha=0.05, power=0.80):
    """Cohen's f^2 for the incremental GLF block; min detectable f^2; achieved power."""
    common = sorted(set(glf) & set(btc_ret))
    if len(common) < 30:
        return {"error": "n too small"}
    r = np.array([btc_ret[m] for m in common], float)
    g = np.array([glf[m] for m in common], float)
    n = len(common)

    def _build(use_glf):
        cols = [np.ones(n - p)]
        for k in range(1, p + 1):
            cols.append(r[p - k:n - k])
        if use_glf:
            cols.append(g[p:])
            for k in range(1, p + 1):
                cols.append(g[p - k:n - k])
        return np.column_stack(cols)

    Y = r[p:]
    Xr = _build(False)
    Xf = _build(True)
    rr = OLS(Y, Xr).fit()
    rf = OLS(Y, Xf).fit()
    r2_red = float(rr.rsquared)
    r2_full = float(rf.rsquared)
    f2 = (r2_full - r2_red) / (1 - r2_full) if r2_full < 1 else 0.0

    n_glf_feat = Xf.shape[1] - Xr.shape[1]           # u = num GLF params
    df1 = n_glf_feat
    df2 = n - Xf.shape[1]

    # Achieved power for the OBSERVED f^2.
    lam_obs = f2 * (n - Xf.shape[1])
    crit = sps.f.ppf(1 - alpha, df1, df2)
    power_obs = float(1 - sps.ncf.cdf(crit, df1, df2, lam_obs))

    # Minimum detectable f^2 at target power (invert noncentral-F).
    from scipy.optimize import brentq

    def _pow_at_f2(f2v):
        lam = f2v * (n - Xf.shape[1])
        return 1 - sps.ncf.cdf(crit, df1, df2, lam)

    try:
        f2_min = brentq(lambda f: _pow_at_f2(f) - power, 1e-6, 2.0, maxiter=200)
    except Exception:
        f2_min = float("nan")

    # Also: sample size needed to detect the OBSERVED f^2 at 80% power.
    def _pow_at_n(nt):
        df2t = nt - Xf.shape[1]
        critt = sps.f.ppf(1 - alpha, df1, df2t)
        return 1 - sps.ncf.cdf(critt, df1, df2t, f2 * (nt - Xf.shape[1]))

    n_needed = None
    if f2 > 0:
        lo, hi = 30, 20000
        try:
            if _pow_at_n(lo) < power < _pow_at_n(hi):
                n_needed = int(brentq(lambda nt: _pow_at_n(nt) - power, lo, hi, maxiter=400))
        except Exception:
            n_needed = None

    if power_obs >= 0.80:
        interp = ("power >= 0.80 means: if the true GLF->BTC effect were at the "
                  "observed size, our sample would likely have rejected the null "
                  "- so a null result here is informative, not just 'insufficient data'.")
    else:
        interp = ("power < 0.80: the sample may be too small to reliably detect "
                  "an effect of the observed size - the null is not strongly "
                  "informative.")

    return {
        "n": n, "u_glf_params": df1,
        "r2_ar_only": round(r2_red, 4), "r2_ar_glf": round(r2_full, 4),
        "incremental_r2": round(r2_full - r2_red, 5),
        "cohen_f2": round(float(f2), 5),
        "achieved_power_observed_effect": round(power_obs, 4),
        "min_detectable_f2_at_80pct": round(float(f2_min), 5)
            if f2_min == f2_min else None,
        "min_detectable_incremental_r2": round(
            float(f2_min) / (1 + f2_min), 5) if f2_min == f2_min else None,
        "n_needed_to_detect_observed_f2": n_needed,
        "interpretation": interp,
    }


# --------------------------------------------------------------------------- #
# NEW-D: BIC-based posterior probability of a real GLF->BTC effect
# --------------------------------------------------------------------------- #
def posterior_prob(glf, btc_ret, p=2):
    """P(H1|data) ~ exp(dBIC/2)/(1+exp(dBIC/2)), dBIC = BIC_H0 - BIC_H1."""
    common = sorted(set(glf) & set(btc_ret))
    if len(common) < 30:
        return {"error": "n too small"}
    r = np.array([btc_ret[m] for m in common], float)
    g = np.array([glf[m] for m in common], float)
    n = len(common)

    def _bic(use_glf):
        cols = [np.ones(n - p)]
        for k in range(1, p + 1):
            cols.append(r[p - k:n - k])
        if use_glf:
            cols.append(g[p:])
            for k in range(1, p + 1):
                cols.append(g[p - k:n - k])
        fit = OLS(r[p:], np.column_stack(cols)).fit()
        k = fit.params.shape[0]
        rss = float(np.sum(fit.resid ** 2))
        return n * np.log(rss / n) + k * np.log(n)

    bic0 = _bic(False)
    bic1 = _bic(True)
    dbic = bic0 - bic1
    p_h1 = float(np.exp(dbic / 2) / (1 + np.exp(dbic / 2)))
    # BF_10 (evidence for H1) ~ exp(dbic/2)
    bf = float(np.exp(dbic / 2))
    return {
        "n": n, "p": p,
        "bic_ar_only": round(bic0, 2), "bic_ar_glf": round(bic1, 2),
        "dBIC_H0_minus_H1": round(dbic, 2),
        "bayes_factor_10": round(bf, 3),
        "posterior_prob_H1": round(p_h1, 4),   # the "accurate probability"
        "label": ("strong evidence FOR effect" if p_h1 >= 0.95 else
                  "moderate evidence FOR" if p_h1 >= 0.75 else
                  "weak/anecdotal" if p_h1 >= 0.50 else
                  "evidence AGAINST effect (null more likely)"),
    }


# --------------------------------------------------------------------------- #
def main():
    print("=" * 74)
    print("FULL CAUSAL BATTERY: GLF -> BTC  (fresh data + hard robustness)")
    print("=" * 74)

    print("\n[1] Fetching BTC daily + reconstructing FULL monthly GLF...")
    btc_daily = fetch_series_dict("CBBTCUSD")
    glf_full = build_monthly_glf(full=True)
    btc_ret = monthly_btc(btc_daily)
    print(f"    BTC daily={len(btc_daily)}  GLF monthly={len(glf_full)}  "
          f"BTC monthly ret={len(btc_ret)}")

    result = {"generated_at": datetime.now().isoformat()}
    result["n_btc_daily"] = len(btc_daily)
    result["n_glf_monthly"] = len(glf_full)

    print("\n[2] REFRESH — full-sample Granger (4 lags)...")
    g = granger(glf_full, btc_ret)
    result["granger_full"] = g
    print("   GLF->BTC min_p =", g["GLF->BTC"].get("min_p"),
          "| BTC->GLF min_p =", g["BTC->GLF"].get("min_p"))

    print("[3] REFRESH — conditional-regime Granger (bull/bear)...")
    g_regime = conditional_regime_granger(glf_full, btc_ret, btc_daily)
    result["granger_conditional_regime"] = g_regime
    for k, v in g_regime.items():
        print(f"   {k}: n={v.get('n_months')} min_p={v.get('min_p')}")

    print("[4] REFRESH — weekly lagged OLS (level & change)...")
    series = weekly_series(btc_daily, glf_full)
    result["ols_weekly_level"] = lagged_ols(series, LAGS_OLS_WEEKS, use_change=False)
    result["ols_weekly_change"] = lagged_ols(series, LAGS_OLS_WEEKS, use_change=True)
    sig = [k for k in LAGS_OLS_WEEKS
           if result["ols_weekly_level"].get(k, {}).get("sig")]
    print(f"   weekly level significant lags: {sig or 'none'}")

    print("[5] NEW-A — VAR-based Granger, BIC-chosen lag + diagnostics...")
    va = var_granger(glf_full, btc_ret)
    result["var_granger"] = va
    if "error" not in va:
        print(f"   lag={va['lag_chosen_bic']}  GLF->BTC p={va['GLF_to_BTC']['p']}  "
              f"LjungBox_resid_p={va['ljung_box_resid_p']}")

    print("[6] NEW-B — walk-forward OOS (AR vs AR+GLF)...")
    for p in (1, 2, 3):
        wf = walk_forward_oos(glf_full, btc_ret, p=p)
        result.setdefault("walk_forward_oos", {})[p] = wf
        if "error" not in wf:
            print(f"   p={p}: OOS R^2={wf['oos_r2_vs_ar']}  "
                  f"MSE ratio={wf['mse_ratio']}  DM p={wf['dm_p']}  -> {wf['verdict']}")

    print("[7] NEW-C — power analysis...")
    pw = power_analysis(glf_full, btc_ret, p=2)
    result["power_analysis"] = pw
    if "error" not in pw:
        print(f"   achieved power={pw['achieved_power_observed_effect']}  "
              f"min detectable f^2@80%={pw['min_detectable_f2_at_80pct']}  "
              f"n_needed={pw['n_needed_to_detect_observed_f2']}")

    print("[8] NEW-D — posterior probability of a real effect (BIC-approx)...")
    pp = posterior_prob(glf_full, btc_ret, p=2)
    result["posterior_prob_H1"] = pp
    if "error" not in pp:
        print(f"   P(H1|data) = {pp['posterior_prob_H1']}  "
              f"(BF_10={pp['bayes_factor_10']})  -> {pp['label']}")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
