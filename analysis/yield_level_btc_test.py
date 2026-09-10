#!/usr/bin/env python3
"""
yield_level_btc_test.py — does the YIELD LEVEL (yield-equilibrium-model "liquidity
    layer") carry era-stable information for BTC, on top of what SFC's GLF already has?
================================================================================
Motivation: C/"yield equilibrium model.py" decomposes Y10 into
    real + E[inflation] + term_premium + fiscal_premium
and argues the yield LEVEL reflects global-liquidity conditions that drive BTC.
SFC currently uses yield-curve SLOPE (M8/GSLS) only; GLF has no yield term.

Question tested here (empirical, no blending):
  (A) Contemporaneous: does the yield LEVEL / real yield / term-premium proxy move
      with BTC daily returns?
  (B) Predictive + era-stable: does it predict BTC forward 1/3/7/30d returns with a
      CONSISTENT sign across eras? (pooled OLS w/ era dummies + per-era coefficients)
  (C) Incremental: is it redundant with VIX (risk appetite) / M2 / curve already used?

Data: data/cleaned/macro_daily_clean.json (FRED-derived daily) + Binance Vision BTC.
Term-premium proxy = US10Y - REAL_Y10 - BE10  (nominal - TIPS real - breakeven).
Report-only; nothing blended.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [1, 3, 7, 30]


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        y10, real, be = m.get("US10Y"), m.get("REAL_Y10"), m.get("BE10")
        if None in (y10, real, be) or b.get("close") is None:
            continue
        rows.append({
            "date": d, "era": era_of(d), "close": b["close"],
            "Y10": y10, "REAL10": real, "BE10": be,
            "TP": y10 - real - be,                       # term-premium proxy
            "VIX": m.get("VIX"), "M2": m.get("M2_US"),
            "SLOPE": m.get("SPREAD_10_2"), "FFR": m.get("FEDFUNDS"),
        })
    df = pd.DataFrame(rows).dropna(subset=["close"]).reset_index(drop=True)
    df["ret1"] = np.log(df["close"]).diff()
    for h in HORIZONS:
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100  # % fwd return
    for c in ["Y10", "REAL10", "BE10", "TP", "VIX", "SLOPE", "M2"]:
        df[f"d{c}_30"] = df[c] - df[c].shift(30)
    return df


def z(s):
    return (s - s.mean()) / s.std()


def main():
    df = load()
    print(f"sample n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    print("era counts:", df["era"].value_counts().to_dict())

    # ---- (A) contemporaneous ----
    print("\n=== (A) Contemporaneous corr with same-day BTC log-return ===")
    for c in ["Y10", "REAL10", "BE10", "TP", "dY10_30", "dREAL10_30", "dTP_30", "VIX", "SLOPE"]:
        s = df[[c, "ret1"]].dropna()
        r = np.corrcoef(s[c], s["ret1"])[0, 1]
        print(f"  corr({c:11s}, ret1) = {r:+.3f}   n={len(s)}")

    # ---- (B) pooled fwd-return regression, era dummies; per-era sign ----
    print("\n=== (B) forward BTC return: pooled OLS w/ era dummies (HC1) + per-era coef sign ===")
    print("    coef = pp of fwd return per +1 sd of the variable\n")
    vars_ = ["Y10", "REAL10", "TP", "dY10_30", "dTP_30"]
    for c in vars_:
        zc = z(df[c])
        line = f"  {c:10s}"
        for h in HORIZONS:
            d = pd.DataFrame({"y": df[f"fwd{h}"], "x": zc, "era": df["era"]}).dropna()
            X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
            X["x"] = d["x"]
            X = sm.add_constant(X)
            r = sm.OLS(d["y"], X).fit(cov_type="HC1")
            coef, p = r.params["x"], r.pvalues["x"]
            # per-era sign
            signs = []
            for e in ["era1_17-20", "era2_21-23", "era3_24-26"]:
                de = d[d["era"] == e]
                if len(de) < 40 or de["x"].std() == 0:
                    signs.append(".")
                    continue
                Xe = sm.add_constant(de[["x"]])
                re = sm.OLS(de["y"], Xe).fit(cov_type="HC1")
                s = "+" if re.params["x"] > 0 else "-"
                s += "*" if re.pvalues["x"] < 0.10 else " "
                signs.append(s)
            flag = "  <-- SIGN-CONSISTENT" if len(set(x[0] for x in signs if x != ".")) == 1 else ""
            line += f" | h{h}: {coef:+.2f}{'*' if p<0.10 else ' '} eras[{''.join(signs)}]"
        print(line)

    print("\n  (* = pooled p<0.10; era columns: sign of per-era coef, '*' if p<0.10)")

    # ---- (C) incremental vs VIX / M2 / slope ----
    print("\n=== (C) does Y10 LEVEL add beyond VIX (risk) & SLOPE (curve, already used)? ===")
    for h in [7, 30]:
        d = pd.DataFrame({"y": df[f"fwd{h}"], "y10": z(df["Y10"]), "vix": z(df["VIX"]),
                          "slope": z(df["SLOPE"])})
        d = d.dropna()
        base = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope"]])).fit(cov_type="HC1")
        full = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope", "y10"]])).fit(cov_type="HC1")
        print(f"  h={h}: AdjR2 base(vix+slope)={base.rsquared_adj:.4f} "
              f"-> +Y10={full.rsquared_adj:.4f}  Δ={full.rsquared_adj-base.rsquared_adj:+.4f}  "
              f"Y10 coef p={full.pvalues['y10']:.3f}")

    # ---- overlap caveat ----
    print("\nNOTE: forward returns overlap (h>1) -> effective indep obs ~ n/(h/2+1);")
    print("      treated as screen only, per era-split-validation skill.")


if __name__ == "__main__":
    main()
