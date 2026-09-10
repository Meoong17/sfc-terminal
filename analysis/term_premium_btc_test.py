#!/usr/bin/env python3
"""
term_premium_btc_test.py — focused test of the TERM PREMIUM specifically
    (the only piece of the yield-equilibrium model that is not just the policy rate)
================================================================================
Rationale: yield LEVEL ~ policy rate (corr(Y10,FFR)=+0.90) and is already reflected
in GLF (Fed BS/M2/DXY). The one genuinely distinct, supply/expectations-driven term is
the TERM PREMIUM (TP). We test defensible daily TP proxies and ask:

  (A) contemporaneous co-move with BTC daily return AND |return| (vol proxy)
  (B) predictive of BTC forward 1/3/7/30d returns, era-stable? (pooled + per-era)
  (D) does TP predict BTC STRESS (forward 30d realized vol)? era-stable? incremental vs VIX?
  (C) incremental over VIX (risk) & SLOPE (curve) already in SFC
  (E) proxy-definition fragility

TP proxies (free/daily, FRED):
  TP2 = US10Y - FEDFUNDS    (term spread over the policy rate)
  TP3 = US30Y - US10Y       (long-end steepness; supply/TP-region, no policy)
  (TP1 = US10Y - REAL_Y10 - BE10 is DROPPED — it is an algebraic ~0: BE10 is itself
   defined as nominal - real, so the expression is a Fisher-identity residual, not TP.)
Report-only; nothing blended. Overlapping horizons -> screen only.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [1, 3, 7, 30]
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
        if None in (m.get("US10Y"), m.get("REAL_Y10"), m.get("BE10"),
                    m.get("FEDFUNDS"), m.get("US30Y")) or b.get("close") is None:
            continue
        rows.append({
            "date": d, "era": era_of(d), "close": b["close"],
            "TP1": m["US10Y"] - m["REAL_Y10"] - m["BE10"],
            "TP2": m["US10Y"] - m["FEDFUNDS"],
            "TP3": m["US30Y"] - m["US10Y"],
            "VIX": m["VIX"], "SLOPE": m["SPREAD_10_2"], "FFR": m["FEDFUNDS"],
        })
    df = pd.DataFrame(rows).dropna(subset=["close"]).reset_index(drop=True)
    df["ret1"] = np.log(df["close"]).diff()
    df["absret"] = df["ret1"].abs()
    for h in HORIZONS:
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    r = df["ret1"]
    df["fwd_vol30"] = r.shift(-1).rolling(30).std().shift(-29) * np.sqrt(365) * 100
    df["vol30"] = r.rolling(30).std() * np.sqrt(365) * 100
    for c in ["TP2", "TP3"]:
        df[f"d{c}_30"] = df[c] - df[c].shift(30)
    return df


def z(s):
    return (s - s.mean()) / s.std()


def nw_fit(y, x):
    d = pd.DataFrame({"y": y, "x": z(x)}).dropna()
    r = sm.OLS(d["y"], sm.add_constant(d[["x"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 30})
    return r.params["x"], r.pvalues["x"]


def run_column(df, c, h):
    d = pd.DataFrame({"y": df[f"fwd{h}"], "x": z(df[c]), "era": df["era"]}).dropna()
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    X["x"] = d["x"]; X = sm.add_constant(X)
    r = sm.OLS(d["y"], X).fit(cov_type="HC1")
    signs = []
    for e in ERAS:
        de = d[d["era"] == e]
        if len(de) < 40 or de["x"].std() == 0:
            signs.append("."); continue
        re = sm.OLS(de["y"], sm.add_constant(de[["x"]])).fit(cov_type="HC1")
        signs.append(("+" if re.params["x"] > 0 else "-") + ("*" if re.pvalues["x"] < 0.10 else " "))
    return r.params["x"], r.pvalues["x"], signs


def main():
    df = load()
    print(f"sample n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")

    print("\n=== (E) TP1 degeneracy check + proxy agreement ===")
    for c in ["TP1", "TP2", "TP3"]:
        print(f"  {c}: mean={df[c].mean():+.4f} sd={df[c].std():.4f} range[{df[c].min():+.3f},{df[c].max():+.3f}]")
    for a, b in [("TP2", "TP3")]:
        s = df[[a, b]].dropna()
        print(f"  corr({a},{b}) = {np.corrcoef(s[a], s[b])[0,1]:+.3f}")

    print("\n=== (A) contemporaneous corr with ret1 / |ret1| ===")
    for c in ["TP2", "TP3", "dTP2_30", "dTP3_30", "vol30"]:
        s = df[[c, "ret1", "absret"]].dropna()
        print(f"  {c:9s} corr(ret1)={np.corrcoef(s[c],s.ret1)[0,1]:+.3f}  corr(|ret1|)={np.corrcoef(s[c],s.absret)[0,1]:+.3f}")

    print("\n=== (B) predictive fwd BTC return: pooled OLS+era dummies (HC1) | per-era signs ===")
    for c in ["TP2", "TP3", "dTP2_30", "dTP3_30"]:
        line = f"  {c:9s}"
        for h in HORIZONS:
            coef, p, signs = run_column(df, c, h)
            line += f" | h{h}: {coef:+.2f}{'*' if p<0.10 else ' '}[{''.join(signs)}]"
        print(line)
    print("  (* pooled p<0.10; [] per-era sign, '*' if era p<0.10)")

    print("\n=== (D) fwd 30d realized vol on TP (pooled, era dummies, HC1) + per-era ===")
    for c in ["TP2", "TP3", "dTP2_30", "dTP3_30"]:
        d = pd.DataFrame({"y": df["fwd_vol30"], "x": z(df[c]), "era": df["era"]}).dropna()
        X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
        X["x"] = d["x"]; X = sm.add_constant(X)
        r = sm.OLS(d["y"], X).fit(cov_type="HC1")
        signs = []
        for e in ERAS:
            de = d[d["era"] == e]
            if len(de) < 40:
                signs.append("."); continue
            re = sm.OLS(de["y"], sm.add_constant(de[["x"]])).fit(cov_type="HC1")
            signs.append(("+" if re.params["x"] > 0 else "-") + ("*" if re.pvalues["x"] < 0.10 else " "))
        print(f"  {c:9s} coef={r.params['x']:+7.3f} pp_vol p={r.pvalues['x']:.4f}  eras[{''.join(signs)}]")

    print("\n=== (D2) incremental: does TP add to forward vol over CURRENT vol30 and VIX? ===")
    for c in ["TP2", "TP3"]:
        d = pd.DataFrame({"y": df["fwd_vol30"], "vol30": z(df["vol30"]), "vix": z(df["VIX"]),
                          "x": z(df[c])}).dropna()
        base = sm.OLS(d["y"], sm.add_constant(d[["vol30", "vix"]])).fit(cov_type="HC1")
        full = sm.OLS(d["y"], sm.add_constant(d[["vol30", "vix", "x"]])).fit(cov_type="HC1")
        print(f"  {c}: base AdjR2={base.rsquared_adj:.3f} -> +TP={full.rsquared_adj:.3f} "
              f"(Δ={full.rsquared_adj-base.rsquared_adj:+.3f}, p={full.pvalues['x']:.4f})")

    print("\n=== (C) incremental fwd return over VIX+SLOPE ===")
    for h in [7, 30]:
        for c in ["TP2", "TP3"]:
            d = pd.DataFrame({"y": df[f"fwd{h}"], "vix": z(df["VIX"]), "slope": z(df["SLOPE"]),
                              "x": z(df[c])}).dropna()
            base = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope"]])).fit(cov_type="HC1")
            full = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope", "x"]])).fit(cov_type="HC1")
            print(f"  h={h} {c}: ΔAdjR2={full.rsquared_adj-base.rsquared_adj:+.4f} p={full.pvalues['x']:.4f}")

    print("\n=== (G) Newey-West (maxlags=30) on the 30d forward-return head ===")
    for c in ["TP2", "TP3", "dTP2_30", "dTP3_30"]:
        coef, p = nw_fit(df["fwd30"], df[c])
        print(f"  {c:9s} coef={coef:+7.3f} pp  NW p={p:.4f}")


if __name__ == "__main__":
    main()
