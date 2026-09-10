#!/usr/bin/env python3
"""
tp_real_test.py — re-run the term-premium tests with REAL TP estimates
    (ACM 10y from NY Fed; Kim-Wright THREEFYTP10 from FRED) instead of proxies.
================================================================================
Replaces the spread proxies (TP2=10Y-FFR, TP3=30Y-10Y) with genuine term-premium
estimates, and cross-checks the two estimates against each other (construct-validity
point raised in 2.docx). Lays all the layers used before on top of the real series:
  (G) ACM vs KW agreement
  (A) contemporaneous co-move with BTC return / |return|
  (B) predictive fwd return, pooled + era dummies, per-era signs
  (V) forward 30d realized vol (TP -> vol channel)
  (D) incremental over VIX + SLOPE
  (E) regime interaction TP x hike (2.docx suggestion), within-regime era stability
  (F) purged walk-forward gate vs FFR-only and baseline
Report-only; no blending.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [1, 3, 7, 30]
ERAS = ["era1_17-20", "era2_21-23", "era3_24-26"]


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    tp = json.load(open(os.path.join(REPO, "data/term_premium_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b, t = macro[d], btc[d], tp.get(d, {})
        if b.get("close") is None or "ACM10" not in t:
            continue
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "ACM10": t.get("ACM10"), "KW10": t.get("KW10"),
                     "VIX": m.get("VIX"), "SLOPE": m.get("SPREAD_10_2"), "FFR": m.get("FEDFUNDS")})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["ret1"] = np.log(df["close"]).diff()
    df["absret"] = df["ret1"].abs()
    for h in HORIZONS:
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    r = df["ret1"]
    df["fwd_vol30"] = r.shift(-1).rolling(30).std().shift(-29) * np.sqrt(365) * 100
    for c in ["ACM10", "KW10"]:
        df[f"d{c}_30"] = df[c] - df[c].shift(30)
    df["hike"] = (df["FFR"] - df["FFR"].shift(90) > 0.05).astype(int)
    return df


def z(s):
    return (s - s.mean()) / s.std()


def era_signs(df, c, y):
    out = []
    for e in ERAS:
        de = df[(df["era"] == e)][[c, y]].dropna()
        if len(de) < 40 or de[c].std() == 0:
            out.append("."); continue
        r = sm.OLS(de[y], sm.add_constant(de[[c]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        out.append(("+" if r.params[c] > 0 else "-") + ("*" if r.pvalues[c] < 0.10 else " "))
    return out


def pooled(df, c, y):
    d = pd.DataFrame({"y": df[y], "x": z(df[c]), "era": df["era"]}).dropna()
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    X["x"] = d["x"]; X = sm.add_constant(X)
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    return r.params["x"], r.pvalues["x"]


def auc(y, score):
    yy = (y > 0).astype(int)
    if yy.nunique() < 2:
        return np.nan
    return stats.mannwhitneyu(score[yy == 1], score[yy == 0]).statistic / (yy.sum() * (len(yy) - yy.sum()))


def main():
    df = load()
    print(f"sample n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")

    print("\n=== (G) construct validity: do ACM and Kim-Wright agree? ===")
    s = df[["ACM10", "KW10"]].dropna()
    print(f"  corr(ACM10,KW10) levels={np.corrcoef(s.ACM10,s.KW10)[0,1]:+.3f}  n={len(s)}")
    d = df[["dACM10_30", "dKW10_30"]].dropna()
    print(f"  corr(dACM,dKW) changes={np.corrcoef(d.dACM10_30,d.dKW10_30)[0,1]:+.3f}  n={len(d)}")

    print("\n=== (A) contemporaneous corr with ret1 / |ret1| ===")
    for c in ["ACM10", "KW10", "dACM10_30", "dKW10_30"]:
        s = df[[c, "ret1", "absret"]].dropna()
        print(f"  {c:10s} corr(ret1)={np.corrcoef(s[c],s.ret1)[0,1]:+.3f}  corr(|ret1|)={np.corrcoef(s[c],s.absret)[0,1]:+.3f}")

    print("\n=== (B) predictive fwd return (pooled HAC30, + era dummies) | per-era signs ===")
    for c in ["ACM10", "KW10", "dACM10_30", "dKW10_30"]:
        line = f"  {c:10s}"
        for h in HORIZONS:
            coef, p = pooled(df, c, f"fwd{h}")
            line += f" | h{h}: {coef:+.2f}{'*' if p<0.10 else ' '}[{''.join(era_signs(df,c,f'fwd{h}'))}]"
        print(line)
    print("  (* pooled p<0.10; [] per-era sign, '*' if era p<0.10)")

    print("\n=== (V) fwd 30d realized vol (pooled + era) ===")
    for c in ["ACM10", "KW10", "dACM10_30", "dKW10_30"]:
        coef, p = pooled(df, c, "fwd_vol30")
        print(f"  {c:10s} coef={coef:+7.3f} p={p:.4f}  eras[{''.join(era_signs(df,c,'fwd_vol30'))}]")

    print("\n=== (D) incremental over VIX+SLOPE, fwd30 ===")
    for c in ["ACM10", "KW10"]:
        d = pd.DataFrame({"y": df["fwd30"], "vix": z(df["VIX"]), "slope": z(df["SLOPE"]), "x": z(df[c])}).dropna()
        base = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        full = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope", "x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        print(f"  {c}: ΔAdjR2={full.rsquared_adj-base.rsquared_adj:+.4f} p={full.pvalues['x']:.4f}")

    print("\n=== (E) regime interaction TP x hike (2.docx), target fwd30 ===")
    for c in ["ACM10", "dACM10_30", "KW10", "dKW10_30"]:
        d = pd.DataFrame({"y": df["fwd30"], "x": z(df[c]).values, "reg": df["hike"].values,
                          "era": df["era"].values}).dropna()
        d["xr"] = d["x"] * d["reg"]
        X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
        X["x"] = d["x"]; X["xr"] = d["xr"]; X = sm.add_constant(X)
        r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        er = []
        for rv in (0, 1):
            sub = d[d["reg"] == rv]
            wr = era_signs(sub.assign(**{"era": sub["era"]}), "x", "y")
            er.append(f"reg{rv}[{''.join(wr)}]")
        print(f"  {c:10s} x={r.params['x']:+7.3f}(p={r.pvalues['x']:.3f})  x:hike={r.params['xr']:+7.3f}(p={r.pvalues['xr']:.3f})  {' '.join(er)}")

    print("\n=== (F) purged walk-forward gate (expanding, embargo 30) — sign of fwd30 ===")
    df2 = df.dropna(subset=["fwd30"]).reset_index(drop=True)
    n = len(df2); fold = n // 7
    def wf(fitcols, regime_cond=False, label=""):
        ics, aucs = [], []
        for k in range(1, 7):
            tr_end = fold * k; te0 = tr_end + 30; te1 = min(te0 + fold, n)
            tr, te = df2.iloc[:tr_end], df2.iloc[te0:te1]
            if len(te) < 50 or len(tr) < 200: continue
            if regime_cond:
                pr = pd.Series(index=te.index, dtype=float)
                for rv in (0, 1):
                    a = tr[tr["hike"] == rv].dropna(subset=fitcols + ["fwd30"])
                    c_ = te[te["hike"] == rv].dropna(subset=fitcols)
                    if len(a) < 40 or len(c_) == 0: continue
                    r = sm.OLS(a["fwd30"], sm.add_constant(a[fitcols])).fit()
                    pr.loc[c_.index] = r.predict(sm.add_constant(c_[fitcols]))
            else:
                a = tr.dropna(subset=fitcols + ["fwd30"])
                c_ = te.dropna(subset=fitcols)
                r = sm.OLS(a["fwd30"], sm.add_constant(a[fitcols])).fit()
                pr = r.predict(sm.add_constant(c_[fitcols]))
            m = pr.notna()
            if m.sum() < 30: continue
            ics.append(stats.spearmanr(pr[m], te.loc[m, "fwd30"]).statistic)
            aucs.append(auc(te.loc[m, "fwd30"], pr[m].values))
        return np.mean(ics), np.std(ics), np.mean(aucs), np.std(aucs), len(ics)

    for cols, rc, lab in [(["ACM10"], False, "ACM level"),
                          (["dACM10_30"], False, "dACM"),
                          (["ACM10", "dACM10_30"], False, "ACM level+change"),
                          (["ACM10", "dACM10_30"], True, "ACM level+change REGIME-cond")]:
        icm, icsd, aum, ausd, nf = wf(cols, rc, lab)
        print(f"  {lab:32s} IC={icm:+.3f}±{icsd:.3f}  AUC={aum:.3f}±{ausd:.3f} (folds={nf})")
    print("  GATE: pass only if AUC>0.55, IC>0, low fold variance.")


if __name__ == "__main__":
    main()
