#!/usr/bin/env python3
"""
inflation_assumptions_tests.py
==============================
Uji statistik lengkap untuk pertanyaan: apakah Bitcoin benar-benar merespons
inflation surprise, atau lebih dominan merespons liquidity / real yields / DXY /
risk appetite? Membangun Model A-D (sama dengan inflation_transmission_models.py)
lalu menambah tiga lapis pengujian:

(1) UJI INSTRUMEN
    - Stationarity (ADF) tiap regressor + BTC return  ->  syarat regresi tidak-spurious.
    - Relevance: t-test individual koefisien.
    - Granger causality (monthly) surprise/kanal -> BTC, dengan robustness battery
      (BH-FDR, sub-period, drop-outlier) supaya lag signifikan tunggal tidak di-overclaim.

(2) UJI ASUMSI KLASIK (pada Model A & Model D)
    - Normalitas residual : Jarque-Bera, Shapiro-Wilk, Omnibus
    - Homoskedastisitas   : Breusch-Pagan, White
    - Autokorelasi        : Durbin-Watson, Ljung-Box (lags 1,3,6,12)
    - Multikolinearitas   : VIF semua regressor
    - Spesifikasi         : Ramsey RESET (fitted^2, fitted^3)

(3) UJI HIPOTESIS
    - F-joint: apakah blok inflation surprise (CPI,PPI,PCE) serentak = 0 di tiap model.
    - Partial-F nested: A vs B (monetary), B vs C (liquidity), C vs D (risk) —
      kontribusi marginal tiap kanal, bukan sekadar delta AdjR2.
    - Wald: inflation block = 0 di Model D penuh (survive controls?).
    - TES KUNCI: koefisien surprise Model A vs Model D (apakah hilang setelah kontrol).

BATASAN JUJUR (sama dengan script asli): surprise = PROXY BERBASIS MODEL
(MoM vs mean 12-bln), BUKAN analyst consensus (ForexFactory/Investing.com = HTTP 403
Cloudflare; Bloomberg berbayar). Frekuensi bulanan kasar; event-study intraday
(08:30 ET) tidak bisa dijalankan di sini. Hasil = indikasi arah transmission, bukan
estimasi efek surprise sesungguhnya. TIDAK ada perubahan scoring SFC.

Usage: cd ~/sfc && .venv/bin/python analysis/inflation_assumptions_tests.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflation_transmission_models import (
    fred_series, monthly, yoy, model_surprise, btc_monthly_return, FRED_KEY,
)

SIG = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# ---------------------------------------------------------------- data build
def build_panel():
    print("== Pull FRED monthly series ==")
    cpi_lvl = monthly(fred_series("CPIAUCSL"))
    ppi_lvl = monthly(fred_series("PPIACO"))
    pce_lvl = monthly(fred_series("PCEPI"))
    realy   = monthly(fred_series("DFII10"))
    dxy     = monthly(fred_series("DTWEXBGS"))
    m2_lvl  = monthly(fred_series("M2SL"))
    vix     = monthly(fred_series("VIXCLS"))
    s_cpi = model_surprise(cpi_lvl); s_cpi.name = "CPI_surp"
    s_ppi = model_surprise(ppi_lvl); s_ppi.name = "PPI_surp"
    s_pce = model_surprise(pce_lvl); s_pce.name = "PCE_surp"
    btc = btc_monthly_return()
    df = pd.DataFrame({
        "BTC": btc, "CPI_surp": s_cpi, "PPI_surp": s_ppi, "PCE_surp": s_pce,
        "RealYield": realy, "DXY": dxy, "M2yoy": yoy(m2_lvl), "VIX": vix,
    })
    df = df.dropna(subset=["BTC"]).dropna(how="all")
    df["CPI_yoy"] = yoy(cpi_lvl); df["PPI_yoy"] = yoy(ppi_lvl); df["PCE_yoy"] = yoy(pce_lvl)
    panel = df.dropna().copy()
    panel = panel[panel.index >= "2017-09-01"]
    return panel


MODEL_DEFS = {
    "A: inflation-only": ["CPI_surp", "PPI_surp", "PCE_surp"],
    "B: +monetary": ["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY"],
    "C: +liquidity(M2)": ["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy"],
    "D: FULL(+risk VIX)": ["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy", "VIX"],
}


# ------------------------------------------------------- classical assumptions
def assump_ols(resid, exog, exog_names, label):
    print(f"\n=== ASUMSI KLASIK: {label} ===")
    # normality
    jb_res = jarque_bera(resid)
    jb = jb_res[0]; jb_p = jb_res[1]  # (stat, p, skew, kurt) in newer statsmodels
    sh_w, sh_p = sps.shapiro(resid)
    _, om_p = sps.normaltest(resid) if len(resid) > 8 else (np.nan, np.nan)
    print(f"  [Normalitas] Jarque-Bera p={jb_p:.3f} {SIG(jb_p)} | "
          f"Shapiro p={sh_p:.3f} {SIG(sh_p)} | Omnibus p={om_p:.3f} {SIG(om_p)}")
    # homoskedasticity
    lm, lm_p, f, f_p = het_breuschpagan(resid, exog)
    print(f"  [Homosked.] Breusch-Pagan LM={lm:.2f} p={lm_p:.3f} {SIG(lm_p)} | F-p={f_p:.3f}")
    try:
        _, w_p, _, w_fp = het_white(resid, exog)
        print(f"              White p={w_p:.3f} {SIG(w_p)} | F-p={w_fp:.3f}")
    except Exception:
        print("              White: n/a (exog rank)")
    # autocorrelation
    dw = durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[1, 3, 6, 12], return_df=True)
    print(f"  [Autokorelasi] Durbin-Watson={dw:.3f} (target ~2)")
    for lag in [1, 3, 6, 12]:
        if lag in lb.index:
            print(f"               Ljung-Box lag{lag:<2} stat={lb.loc[lag,'lb_stat']:.2f} "
                  f"p={lb.loc[lag,'lb_pvalue']:.3f} {SIG(lb.loc[lag,'lb_pvalue'])}")
    # multicollinearity
    vifs = [variance_inflation_factor(exog, i) for i in range(exog.shape[1])]
    print("  [Multikolinearitas] VIF (intercept~infinite jika konstanta):")
    for nm, v in zip(exog_names, vifs):
        flag = "  <-- >10" if v > 10 else ""
        print(f"               {nm:12s} VIF={v:.2f}{flag}")


def reset_test(Y, X, label):
    """Ramsey RESET: tambahkan fitted^2, fitted^3, joint F-test."""
    Xc = sm.add_constant(X)
    base = sm.OLS(Y, Xc).fit()
    fh = base.fittedvalues
    Xr = Xc.copy()
    Xr["f2"] = fh ** 2
    Xr["f3"] = fh ** 3
    full = sm.OLS(Y, Xr).fit()
    # partial F: H0: f2=f3=0  (returns F, p, df_diff)
    F, p, _ = full.compare_f_test(base)
    print(f"\n  [Spesifikasi] Ramsey RESET ({label}) F={F:.3f} p={p:.3f} {SIG(p)}"
          f"  {'-> linieritas cukup' if p > 0.05 else '-> kemungkinan misspesifikasi'}")
    return p


# ------------------------------------------------------------------ main
def main():
    if not FRED_KEY:
        print("FATAL: no FRED key"); sys.exit(1)
    panel = build_panel()
    print(f"Panel n={len(panel)}  span {panel.index.min().date()}..{panel.index.max().date()}\n")

    Y = panel["BTC"]

    # ---- (1) UJI INSTRUMEN: stationarity ----
    print("=" * 74)
    print("(1) UJI INSTRUMEN — STATIONARITY (ADF, H0: unit root)")
    print("=" * 74)
    for c in ["BTC", "CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy", "VIX"]:
        try:
            s = panel[c].dropna()
            adf, p, _, _, crit, _ = adfuller(s, autolag="AIC")
            print(f"  {c:9s} ADF={adf:+.3f} p={p:.3f} {'STATIONARY' if p < 0.05 else 'UNIT ROOT'}")
        except Exception as e:
            print(f"  {c:9s} ADF err {e}")

    # ---- models ----
    models = {}
    print("\n" + "=" * 74)
    print("MODEL A-D  (HC1 robust SE)")
    print("=" * 74)
    for label, cols in MODEL_DEFS.items():
        X = panel[cols]
        Xc = sm.add_constant(X)
        m = sm.OLS(Y, Xc).fit(cov_type="HC1")
        models[label] = (m, cols)
        print(f"\n[{label}] n={int(m.nobs)} AdjR2={m.rsquared_adj:.4f} AIC={m.aic:.1f} BIC={m.bic:.1f}")
        for k in Xc.columns:
            print(f"      {k:12s} b={m.params[k]:+.4f} se={m.bse[k]:.4f} p={m.pvalues[k]:.3f} {SIG(m.pvalues[k])}")

    # ---- (2) ASUMSI KLASIK ----
    print("\n" + "=" * 74)
    print("(2) UJI ASUMSI KLASIK")
    print("=" * 74)
    for label in ["A: inflation-only", "D: FULL(+risk VIX)"]:
        m, cols = models[label]
        assump_ols(m.resid, m.model.exog, m.model.exog_names, label)
        reset_test(Y, panel[cols], label)

    # ---- (3) HIPOTESIS ----
    print("\n" + "=" * 74)
    print("(3) UJI HIPOTESIS")
    print("=" * 74)

    # 3a. F-joint inflation block per model
    print("\n-- F-joint: blok inflation surprise (CPI,PPI,PCE) serentak = 0 --")
    for label, (m, cols) in models.items():
        hyp = "CPI_surp = PPI_surp = PCE_surp = 0"
        try:
            f = m.f_test(hyp)
            print(f"  {label:20s} F={float(f.fvalue):.3f} p={float(f.pvalue):.4f} {SIG(float(f.pvalue))}")
        except Exception as e:
            print(f"  {label:20s} f_test err {e}")

    # 3b. Nested partial-F: A->B->C->D
    print("\n-- Partial-F nested (kontribusi marginal tiap kanal) --")
    order = list(MODEL_DEFS.keys())
    for i in range(3):
        red_l, full_l = order[i], order[i + 1]
        m_red = models[red_l][0]
        m_full = models[full_l][0]
        try:
            F, p, _ = m_full.compare_f_test(m_red)
            added = set(MODEL_DEFS[full_l]) - set(MODEL_DEFS[red_l])
            print(f"  {red_l:20s} -> {full_l:22s} F={F:.3f} p={p:.4f} {SIG(p)}  [+{','.join(added)}]")
        except Exception as e:
            print(f"  {red_l}->{full_l} err {e}")

    # 3c. TES KUNCI: inflation surprise survive controls?
    print("\n-- TES KUNCI: koefisien surprise A vs D (survive kontrol?) --")
    mA = models["A: inflation-only"][0]
    mD = models["D: FULL(+risk VIX)"][0]
    for v in ["CPI_surp", "PPI_surp", "PCE_surp"]:
        a = mA.params[v]; d = mD.params[v]
        ap = mA.pvalues[v]; dp = mD.pvalues[v]
        verdict = "HILANG" if (ap < 0.05 and dp >= 0.05) else ("BERTAHAN" if (ap < 0.05 and dp < 0.05) else "n/a")
        print(f"  {v:9s}: A b={a:+.4f} p={ap:.3f}  ->  D b={d:+.4f} p={dp:.3f}  [{verdict}]")

    # 3d. Granger causality (monthly) with robustness battery (BH-FDR)
    print("\n-- Granger causality (monthly) -> BTC, dengan robustness battery --")
    GRANGER_VARS = ["CPI_surp", "PPI_surp", "PCE_surp", "VIX", "M2yoy", "RealYield", "DXY"]
    cells = {}  # (var,lag) -> p
    for var in GRANGER_VARS:
        sub = panel[[var, "BTC"]].dropna()
        if len(sub) < 30:
            continue
        try:
            g = grangercausalitytests(sub, maxlag=3, verbose=False)
            for lag in range(1, 4):
                cells[(var, lag)] = float(g[lag][0]["ssr_ftest"][1])
        except Exception as e:
            print(f"  {var:9s}: granger err {e}")
    # BH-FDR over all 21 cells
    all_cells = sorted(cells, key=lambda k: cells[k])
    m = len(all_cells)
    import statsmodels.stats.multitest as smm
    _, qvals, _, _ = smm.multipletests([cells[k] for k in all_cells], method="fdr_bh")
    print(f"  BH-FDR gate (q<0.10) over {m} cells (7 vars x 3 lags):")
    n_sig = 0
    for k, q in zip(all_cells, qvals):
        if q < 0.10:
            n_sig += 1
            print(f"    {k[0]:9s} lag{k[1]} p={cells[k]:.4f} q={q:.3f}  ***")
    if n_sig == 0:
        print("    -> TIDAK ada sel yang lolos BH-FDR (semua q>=0.10).")
    # Robustness for any var with a nominal l1 p<0.05
    print("  Robustness (sub-period + drop-outlier) untuk var dgn l1 p<0.05:")
    for var in GRANGER_VARS:
        if cells.get((var, 1), 1.0) >= 0.05:
            continue
        sub = panel[[var, "BTC"]].dropna()
        full = cells[(var, 1)]
        # sub-period halves
        half = len(sub) // 2
        p_h1 = grangercausalitytests(sub.iloc[:half], maxlag=1, verbose=False)[1][0]["ssr_ftest"][1]
        p_h2 = grangercausalitytests(sub.iloc[half:], maxlag=1, verbose=False)[1][0]["ssr_ftest"][1]
        # drop-outlier: drop |z|>1.5 on the predictor
        z = (sub[var] - sub[var].mean()) / sub[var].std()
        clean = sub[z.abs() <= 1.5]
        p_c = grangercausalitytests(clean, maxlag=1, verbose=False)[1][0]["ssr_ftest"][1]
        ok = (p_h1 < 0.05 and p_h2 < 0.05 and p_c < 0.05)
        print(f"    {var:9s} l1 full_p={full:.4f} | half1={p_h1:.3f} half2={p_h2:.3f} "
              f"drop-outlier={p_c:.3f} -> {'ROBUST' if ok else 'ARTIFACT'}")

    # 3e. Era-split Model D (stabilitas)
    print("\n-- ERA-SPLIT Model D --")
    for ename, lo, hi in [("era2 2018-21", "2018-01-01", "2022-01-01"),
                          ("era3 2022-26", "2022-01-01", "2030-01-01")]:
        e = panel[(panel.index >= lo) & (panel.index < hi)]
        if len(e) < 20:
            continue
        cols = MODEL_DEFS["D: FULL(+risk VIX)"]
        Xc = sm.add_constant(e[cols])
        m = sm.OLS(e["BTC"], Xc).fit(cov_type="HC1")
        sig = " ".join(f"{k}:b={m.params[k]:+.3f}(p={m.pvalues[k]:.2f})" for k in Xc.columns)
        print(f"  {ename:14s} n={len(e)} AdjR2={m.rsquared_adj:.3f} | {sig}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
