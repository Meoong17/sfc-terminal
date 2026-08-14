#!/usr/bin/env python3
"""
model_Dprime_oos.py
===================
Tes wajib terakhir sebelum mengunci Model D sebagai baseline resmi SFC.

Model D' = BTC ~ const + CPI_surp + PPI_surp + PCE_surp
                    + ΔRealYield + ΔDXY + ΔM2 + VIX      (Newey-West HAC)

Perubahan dari Model D: RealYield/DXY/M2 masuk sebagai FIRST-DIFFERENCE (Δ),
bukan level — ADF sebelumnya menunjukkan level ketiganya I(1); Δ menjadikannya
stationary (hindari regresi spurious). Surprise tetap proxy model-based.

Dua pengujian:
(1) FULL-SAMPLE Model D' + HAC — VIX harus signifikan, blok inflation harus null,
    ΔRealYield/ΔDXY/ΔM2 harus null (kanal weak).
(2) EXPANDING OUT-OF-SAMPLE VALIDATION (min train 60 bln, +1 bln per langkah):
    * OOS R² Model D' vs VIX-only vs inflation-only vs naive(const).
    * Sign accuracy vs base rate.
    * Frekuensi OOS: VIX signifikan (p<0.05) berapa % window; blok inflation
      signifikan berapa % window. Kriteria lulus: VIX sig di mayoritas window &
      R²_OOS > inflation-only & > naive; inflation null hampir semua window.

BATASAN: n=105 bulan (2017-09..2026-06); surprise bukan analyst consensus
(moM-vs-12bln proxy). TIDAK ada perubahan scoring. Murni adjudikasi.

Usage: cd ~/sfc && .venv/bin/python analysis/model_Dprime_oos.py
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflation_assumptions_tests import build_panel, SIG, FRED_KEY

MIN_TRAIN = 60
HAC_LAGS = 3


def fit(Y, X, maxlags=HAC_LAGS):
    Xc = sm.add_constant(X)
    return sm.OLS(Y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def joint_p(model, cols):
    """Joint Wald p that all cols = 0 (robust covariance). nan on failure."""
    if not cols:
        return np.nan
    hyp = " = ".join([f"{c}=0" for c in cols]) + " = 0"
    try:
        return float(model.wald_test(hyp).pvalue)
    except Exception:
        return np.nan


def jp_fmt(p):
    return "n/a" if p != p else (f"{p:.4f} {'NULL ✓' if p >= 0.05 else 'SIG'}")


def main():
    if not FRED_KEY:
        print("FATAL: no FRED key"); sys.exit(1)
    panel = build_panel()
    panel["dRealYield"] = panel["RealYield"].diff()
    panel["dDXY"] = panel["DXY"].diff()
    panel["dM2"] = panel["M2yoy"].diff()
    panel = panel.dropna()
    print(f"Panel n={len(panel)}  span {panel.index.min().date()}..{panel.index.max().date()}\n")

    SURP = ["CPI_surp", "PPI_surp", "PCE_surp"]
    DLT = ["dRealYield", "dDXY", "dM2"]
    Y = panel["BTC"]
    X_D = pd.concat([panel[SURP], panel[DLT], panel[["VIX"]]], axis=1)

    # ---- (1) FULL-SAMPLE Model D' ----
    print("=" * 78)
    print("(1) MODEL D' FULL-SAMPLE (Newey-West HAC, maxlags=%d)" % HAC_LAGS)
    print("=" * 78)
    m = fit(Y, X_D)
    print(f"n={int(m.nobs)}  AdjR2={m.rsquared_adj:.4f}  AIC={m.aic:.1f}  BIC={m.bic:.1f}")
    for c in X_D.columns:
        print(f"  {c:12s} b={m.params[c]:+.4f} se={m.bse[c]:+.4f} p={m.pvalues[c]:.3f} {SIG(m.pvalues[c])}")
    print(f"\n  Joint Wald inflation block (CPI,PPI,PCE)=0 : {jp_fmt(joint_p(m, SURP))}")
    print(f"  Joint Wald delta block (ΔRY,ΔDXY,ΔM2)=0     : {jp_fmt(joint_p(m, DLT))}")
    vix_ok = m.pvalues["VIX"] < 0.05
    print(f"  VIX survival: b={m.params['VIX']:+.4f} p={m.pvalues['VIX']:.4f} -> "
          f"{'SURVIVE ✓' if vix_ok else 'GAGAL'}")

    # ---- (2) EXPANDING OOS ----
    print("\n" + "=" * 78)
    print("(2) EXPANDING OUT-OF-SAMPLE VALIDATION (min train %d)" % MIN_TRAIN)
    print("=" * 78)
    cols_D = X_D.columns.tolist()
    cols_VIX = ["VIX"]
    cols_INF = SURP
    oos_idx = panel.index[MIN_TRAIN:]
    print(f"OOS window: {MIN_TRAIN}..{len(panel)} -> {len(oos_idx)} out-of-sample months\n")

    preds = {"Dprime": [], "VIXonly": [], "INFlonly": [], "naive": []}
    vix_sig_windows = 0; inf_sig_windows = 0; vix_betas = []; vix_ps = []
    y_oos = []
    for i in range(MIN_TRAIN, len(panel)):
        tr = panel.iloc[:i]
        te_y = panel["BTC"].iloc[i]
        y_oos.append(te_y)
        # naive baseline = expanding mean of y
        preds["naive"].append(tr["BTC"].mean())
        for name, cols in [("Dprime", cols_D), ("VIXonly", cols_VIX), ("INFlonly", cols_INF)]:
            mm = fit(tr["BTC"], tr[cols])
            xnew = np.concatenate([np.ones(1), panel[cols].iloc[[i]].values.ravel()]).reshape(1, -1)
            preds[name].append(float(mm.predict(xnew)[0]))
        # record VIX / inflation significance from the D' fit this window
        md = fit(tr["BTC"], tr[cols_D])
        vp = md.pvalues["VIX"]
        vix_ps.append(vp)
        vix_betas.append(md.params["VIX"])
        if vp < 0.05:
            vix_sig_windows += 1
        if joint_p(md, SURP) < 0.05:
            inf_sig_windows += 1

    y = np.array(y_oos)
    n_oos = len(y)
    def oos_r2(name):
        e = y - np.array(preds[name])
        return 1.0 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    def sign_acc(name):
        return np.mean((np.array(preds[name]) > 0) == (y > 0))
    base_rate = np.mean(y > 0)

    print(f"{'model':<12}{'OOS R2':>10}{'sign_acc':>10}")
    for name in ["Dprime", "VIXonly", "INFlonly", "naive"]:
        print(f"{name:<12}{oos_r2(name):>+10.4f}{sign_acc(name):>10.3f}")
    print(f"\n  base rate (P(up)) = {base_rate:.3f}")

    nw = n_oos
    print(f"\n  OOS windows VIX signifikan (p<0.05): {vix_sig_windows}/{nw} = "
          f"{100*vix_sig_windows/nw:.0f}%   | mean VIX b={np.mean(vix_betas):+.4f} "
          f"(median {np.median(vix_betas):+.4f})")
    print(f"  OOS windows inflation block signifikan: {inf_sig_windows}/{nw} = "
          f"{100*inf_sig_windows/nw:.0f}%")
    print(f"  VIX sign-fraction OOS (b<0): {np.mean(np.array(vix_betas)<0)*100:.0f}%")

    # ---- VERDICT ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    crit1 = oos_r2("Dprime") > oos_r2("INFlonly")
    crit2 = oos_r2("Dprime") > 0
    crit3 = (vix_sig_windows / nw) > 0.5 and np.mean(np.array(vix_betas) < 0) > 0.5
    crit4 = (inf_sig_windows / nw) < 0.2
    print(f"  D' OOS R2 > inflation-only   : {crit1}  ({oos_r2('Dprime'):+.4f} vs {oos_r2('INFlonly'):+.4f})")
    print(f"  D' OOS R2 > 0 (beat naive)    : {crit2}  ({oos_r2('Dprime'):+.4f})")
    print(f"  VIX sig >50% window & arah neg : {crit3}")
    print(f"  inflation sig <20% window     : {crit4}  ({100*inf_sig_windows/nw:.0f}%)")
    if crit1 and crit2 and crit3 and crit4:
        print("\n  => MODEL D' LOLOS. VIX survive OOS, inflation null OOS. Layak baseline resmi SFC.")
    else:
        print("\n  => BELUM sepenuhnya lolos. Tabel di atas menunjukkan kriteria mana yang gagal.")


if __name__ == "__main__":
    main()
