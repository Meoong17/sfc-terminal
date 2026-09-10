#!/usr/bin/env python3
"""
tp_encompassing_test.py — is the regime-conditional "TP2" signal anything more
    than the policy rate (FFR) itself?
================================================================================
TP2 = US10Y - FFR, so dTP2_30 = d10y - dFFR. The regime-interaction result could be
entirely driven by dFFR (policy change), not by the long yield / term structure.
Encompassing test: put d10y and dFFR in SEPARATELY, then with their hike-interactions,
and see which carries the predictive info for fwd30 BTC return.

Also: does TP2 LEVEL survive controlling FFR LEVEL directly?
Report-only, HAC(30), era dummies.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if None in (m.get("US10Y"), m.get("FEDFUNDS"), m.get("US30Y")) or b.get("close") is None:
            continue
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "US10Y": m["US10Y"], "FFR": m["FEDFUNDS"], "US30Y": m["US30Y"]})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["d10y"] = df["US10Y"] - df["US10Y"].shift(30)
    df["dffr"] = df["FFR"] - df["FFR"].shift(30)
    df["TP2"] = df["US10Y"] - df["FFR"]
    df["TP2_level"] = df["TP2"]
    df["fwd30"] = np.log(df["close"].shift(-30) / df["close"]) * 100
    df["hike"] = (df["FFR"] - df["FFR"].shift(90) > 0.05).astype(int)
    return df


def z(s):
    return (s - s.mean()) / s.std()


def fit(df, terms, label):
    d = df[["fwd30", "era"] + terms].dropna().copy()
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    for t in terms:
        X[t] = z(d[t]).values
    X = sm.add_constant(X)
    r = sm.OLS(d["fwd30"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    print(f"\n[{label}]  n={len(d)}  AdjR2={r.rsquared_adj:.4f}")
    for t in terms:
        print(f"    {t:16s} coef={r.params[t]:+7.3f}  p={r.pvalues[t]:.3f}")
    return r


def main():
    df = load()
    print(f"n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")

    print("\n############ (1) CHANGE predictors: is the TP2 change just dFFR? ############")
    fit(df, ["d10y"], "A: d10y alone")
    fit(df, ["dffr"], "B: dFFR alone")
    fit(df, ["d10y", "dffr"], "C: d10y + dFFR  (encompassing)")
    df["d10y_hike"] = df["d10y"] * df["hike"]
    df["dffr_hike"] = df["dffr"] * df["hike"]
    fit(df, ["d10y", "dffr", "d10y_hike", "dffr_hike"], "D: + hike-interactions")

    print("\n############ (2) LEVEL predictors: does TP2 level survive FFR level? ############")
    fit(df, ["FFR"], "E: FFR level alone")
    fit(df, ["TP2_level"], "F: TP2 level alone")
    fit(df, ["TP2_level", "FFR"], "G: TP2 level + FFR level")
    df["TP2_hike"] = df["TP2_level"] * df["hike"]
    fit(df, ["TP2_level", "FFR", "TP2_hike"], "H: + TP2:hike")

    print("\nInterpretation guide: if a TP term loses significance once FFR (or its")
    print("change) is included, its information was the policy rate, not term structure.")


if __name__ == "__main__":
    main()
