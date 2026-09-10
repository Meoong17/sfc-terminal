#!/usr/bin/env python3
"""
net_liquidity_btc_test.py — the canonical "marginal liquidity" claim:
    BTC tracks NET LIQUIDITY in dollar terms (Fed BS - TGA - RRP), and its CHANGE.
================================================================================
GLF already z-scores the same components (YoY); this tests the raw-dollar LEVEL and
the CHANGE (marginal liquidity), US and global aggregate, as a BTC driver:
  net_us   = FED_BS - TGA - RRP*1000          (all -> $bn: /1000)
  net_glob = FED_BS + ECB_BS + BOJ_BS - TGA - RRP*1000
Layers: contemporaneous, predictive (level & d30 & d90, pooled + era signs),
incremental over VIX+SLOPE, and a purged walk-forward gate. Report-only.
"""
import os, json
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
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if b.get("close") is None:
            continue
        fb, tga, rrp = m.get("FED_BS"), m.get("TGA"), m.get("RRP")
        ecb, boj = m.get("ECB_BS"), m.get("BOJ_BS")
        if None in (fb, tga, rrp):
            continue
        net_us = fb - tga - rrp * 1000.0
        net_glob = (fb + (ecb or 0) + (boj or 0)) - tga - rrp * 1000.0
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "net_us": net_us / 1000.0, "net_glob": net_glob / 1000.0,   # $bn
                     "FED_BS": fb / 1000.0, "VIX": m.get("VIX"), "SLOPE": m.get("SPREAD_10_2")})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["ret1"] = np.log(df["close"]).diff()
    df["absret"] = df["ret1"].abs()
    for h in HORIZONS:
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    for c in ["net_us", "net_glob", "FED_BS"]:
        df[f"d{c}_30"] = df[c] - df[c].shift(30)
        df[f"d{c}_90"] = df[c] - df[c].shift(90)
        df[f"yoy_{c}"] = (df[c] / df[c].shift(365) - 1.0) * 100
    return df


def z(s):
    return (s - s.mean()) / s.std()


def pooled(df, c, y):
    d = pd.DataFrame({"y": df[y], "x": z(df[c]), "era": df["era"]}).dropna()
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    X["x"] = d["x"]; X = sm.add_constant(X)
    r = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    ss = []
    for e in ERAS:
        de = d[d["era"] == e]
        if len(de) < 40 or de["x"].std() == 0:
            ss.append("."); continue
        re = sm.OLS(de["y"], sm.add_constant(de[["x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        ss.append(("+" if re.params["x"] > 0 else "-") + ("*" if re.pvalues["x"] < 0.10 else " "))
    return r.params["x"], r.pvalues["x"], ss


def auc(y, score):
    yy = (np.asarray(y) > 0).astype(int)
    if len(np.unique(yy)) < 2:
        return np.nan
    return stats.mannwhitneyu(score[yy == 1], score[yy == 0]).statistic / (yy.sum() * (len(yy) - yy.sum()))


def main():
    df = load()
    print(f"n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    print(f"net_us $bn range [{df.net_us.min():.0f},{df.net_us.max():.0f}]  "
          f"net_glob [{df.net_glob.min():.0f},{df.net_glob.max():.0f}]")

    print("\n=== contemporaneous corr ===")
    for c in ["net_us", "net_glob", "dnet_us_30", "dnet_us_90", "yoy_net_us"]:
        s = df[[c, "ret1", "absret"]].dropna()
        print(f"  {c:12s} corr(ret1)={np.corrcoef(s[c],s.ret1)[0,1]:+.3f}  corr(|ret1|)={np.corrcoef(s[c],s.absret)[0,1]:+.3f}")

    print("\n=== predictive fwd return (pooled HAC30 + era dummies) | per-era signs ===")
    for c in ["net_us", "dnet_us_30", "dnet_us_90", "yoy_net_us",
              "net_glob", "dnet_glob_30", "yoy_net_glob"]:
        line = f"  {c:13s}"
        for h in HORIZONS:
            coef, p, ss = pooled(df, c, f"fwd{h}")
            line += f" | h{h}: {coef:+.2f}{'*' if p<0.10 else ' '}[{''.join(ss)}]"
        print(line)
    print("  (* p<0.10; [] per-era sign, '*' if era p<0.10)")

    print("\n=== incremental over VIX+SLOPE (fwd30) ===")
    for c in ["dnet_us_30", "dnet_us_90", "yoy_net_us"]:
        d = pd.DataFrame({"y": df["fwd30"], "vix": z(df["VIX"]), "slope": z(df["SLOPE"]), "x": z(df[c])}).dropna()
        base = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        full = sm.OLS(d["y"], sm.add_constant(d[["vix", "slope", "x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        print(f"  {c:13s} ΔAdjR2={full.rsquared_adj-base.rsquared_adj:+.4f} p={full.pvalues['x']:.4f}")

    print("\n=== purged walk-forward gate (expanding, embargo 30) — sign of fwd30 ===")
    df2 = df.dropna(subset=["fwd30"]).reset_index(drop=True)
    n = len(df2); fold = n // 7
    def wf(c):
        ics, aucs = [], []
        for k in range(1, 7):
            tr_end = fold * k; te0 = tr_end + 30; te1 = min(te0 + fold, n)
            tr = df2.iloc[:tr_end].dropna(subset=[c, "fwd30"])
            te = df2.iloc[te0:te1].dropna(subset=[c, "fwd30"])
            if len(te) < 50 or len(tr) < 200: continue
            r = sm.OLS(tr["fwd30"], sm.add_constant(tr[[c]])).fit()
            pr = r.predict(sm.add_constant(te[[c]]))
            ics.append(stats.spearmanr(pr, te["fwd30"]).statistic)
            aucs.append(auc(te["fwd30"].values, pr.values))
        return np.mean(ics), np.std(ics), np.mean(aucs), np.std(aucs), len(ics)
    for c in ["dnet_us_30", "dnet_us_90", "yoy_net_us", "dnet_glob_30"]:
        icm, icsd, aum, ausd, nf = wf(c)
        print(f"  {c:13s} IC={icm:+.3f}±{icsd:.3f}  AUC={aum:.3f}±{ausd:.3f} (folds={nf})")
    print("  GATE: pass only if AUC>0.55, IC>0, low fold variance.")


if __name__ == "__main__":
    main()
