#!/usr/bin/env python3
"""
term_premium_regime_interaction.py — 2.docx recommended follow-up:
    is the term-premium relation STATE-DEPENDENT (regime-conditional) rather than null?
================================================================================
2.docx: era-flip may reflect two opposite real mechanisms pooled without an
interaction term, not absence of relation. Test explicitly:
    fwd ~ z(dTP) + z(dTP):hike_dummy + era dummies
and check WITHIN-regime stability of the slope.

Regime definitions (monetary stance, two variants for definition-robustness):
  REG_A  hike   = FFR_t - FFR_{t-90} > +0.05   (policy tightening over 90d)
  REG_B  tight  = FFR_t > FFR rolling-365d mean (policy above its own 1y mean)

Targets: fwd 30d return and fwd 30d realized vol.
Predictors: dTP2_30 = Δ30(US10Y-FFR), dTP3_30 = Δ30(US30Y-US10Y)  (changes, per 2.docx)
Report-only; screen (overlapping horizons). HAC(30).
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERAS = ["era1_17-20", "era2_21-23", "era3_24-26"]


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if None in (m.get("US10Y"), m.get("FEDFUNDS"), m.get("US30Y"), m.get("VIX")) or b.get("close") is None:
            continue
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "US10Y": m["US10Y"], "FFR": m["FEDFUNDS"], "US30Y": m["US30Y"], "VIX": m["VIX"]})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["TP2"] = df["US10Y"] - df["FFR"]
    df["TP3"] = df["US30Y"] - df["US10Y"]
    df["ret1"] = np.log(df["close"]).diff()
    df["fwd30"] = np.log(df["close"].shift(-30) / df["close"]) * 100
    r = df["ret1"]
    df["fwd_vol30"] = r.shift(-1).rolling(30).std().shift(-29) * np.sqrt(365) * 100
    for c in ["TP2", "TP3"]:
        df[f"d{c}_30"] = df[c] - df[c].shift(30)
    # regimes
    df["hike_A"] = (df["FFR"] - df["FFR"].shift(90) > 0.05).astype(int)
    df["tight_B"] = (df["FFR"] > df["FFR"].rolling(365).mean()).astype(int)
    return df


def z(s):
    return (s - s.mean()) / s.std()


def within_slope(sub, xcol, ycol):
    s = sub[[xcol, ycol]].dropna()
    if len(s) < 40 or s[xcol].std() == 0:
        return None, None, len(s)
    r = sm.OLS(s[ycol], sm.add_constant(s[[xcol]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    return r.params[xcol], r.pvalues[xcol], len(s)


def interaction(df, xcol, ycol, regcol, regname):
    d = pd.DataFrame({"y": df[ycol], "x": z(df[xcol]).values, "reg": df[regcol].values,
                      "era": df["era"].values}).dropna()
    d["x_reg"] = d["x"] * d["reg"]
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    X["x"] = d["x"]; X["x_reg"] = d["x_reg"]
    X = sm.add_constant(X)
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    print(f"  {xcol:9s} x {regname:8s}: coef(x, baseline)={r.params['x']:+7.3f} (p={r.pvalues['x']:.3f})"
          f"  coef(x:{regname})={r.params['x_reg']:+7.3f} (p={r.pvalues['x_reg']:.3f})")
    # within-regime stability across eras
    for rv, rl in [(0, "reg0"), (1, "reg1")]:
        sub = d[d["reg"] == rv]
        erastr = []
        for e in ERAS:
            se = sub[sub["era"] == e]
            c, p, n = within_slope(se, "x", "y")
            erastr.append("." if c is None else ("+" if c > 0 else "-") + ("*" if p < 0.10 else " "))
        c, p, n = within_slope(sub, "x", "y")
        base = "" if c is None else f"{c:+.3f}"
        print(f"      within {rl} n={n:4d} slope={base} p={'na' if p is None else f'{p:.3f}'}  eras[{' '.join(erastr)}]")
    return r


def main():
    df = load()
    print(f"sample n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    print(f"regime A (hike=ΔFFR90>0.05): share={df['hike_A'].mean():.2f}  "
          f"regime B (tight=FFR>MA365): share={df['tight_B'].mean():.2f}")

    for target, label in [("fwd30", "FWD 30d RETURN (%)"), ("fwd_vol30", "FWD 30d REALIZED VOL (%)")]:
        print(f"\n=== {label}: interaction TP_change x monetary regime (HAC 30, + era dummies) ===")
        for xc in ["dTP2_30", "dTP3_30"]:
            for rc, rn in [("hike_A", "hikeA"), ("tight_B", "tightB")]:
                interaction(df, xc, target, rc, rn)

    # also try LEVEL predictors (not just change)
    print(f"\n=== (supplementary) LEVEL predictors x regime, target fwd30 ===")
    for xc in ["TP2", "TP3"]:
        interaction(df, xc, "fwd30", "hike_A", "hikeA")


if __name__ == "__main__":
    main()
