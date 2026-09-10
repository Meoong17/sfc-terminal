#!/usr/bin/env python3
"""
net_liquidity_incremental.py — is dnet_us_30 novel vs a GLF proxy, and is its OOS AUC real?
================================================================================
GLF is reconstructed locally from the cleaned macro components (no network):
  fed/ecb/boj/M2 YoY + TGA/RRP level z, weighted as in global_liquidity_engine.py
  (0.30/0.15/0.03/0.15/0.10/0.10, normalized; DXY omitted - not in cleaned file).
Then:
  1) incremental: fwd30 ~ z(GLF_proxy) + z(dnet_us_30). Does net-liquidity add beyond GLF?
  2) robustness of the WF AUC: fold counts 4..8, horizons 7/30, per-fold, and a
     label-shuffle permutation p-value (200 draws).
Report-only.
"""
import os, json, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(12345)
W = {"fed": 0.30, "ecb": 0.15, "boj": 0.03, "m2": 0.15, "tga": 0.10, "rrp": 0.10}


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def z(s):
    return (s - s.mean()) / s.std()


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if b.get("close") is None or None in (m.get("FED_BS"), m.get("TGA"), m.get("RRP")):
            continue
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "net_us": (m["FED_BS"] - m["TGA"] - m["RRP"] * 1000.0) / 1000.0,
                     "FED_BS": m.get("FED_BS"), "ECB_BS": m.get("ECB_BS"), "BOJ_BS": m.get("BOJ_BS"),
                     "M2_US": m.get("M2_US"), "TGA": m.get("TGA"), "RRP": m.get("RRP")})
    df = pd.DataFrame(rows).reset_index(drop=True)
    # GLF proxy (ff monthly-ish): YoY for stock series, level for TGA/RRP
    for k, col in [("fed", "FED_BS"), ("ecb", "ECB_BS"), ("boj", "BOJ_BS"), ("m2", "M2_US")]:
        df[f"{k}_yoy"] = df[col].pct_change(365) * 100
    glf = pd.Series(0.0, index=df.index)
    tw = 0.0
    for k in ["fed", "ecb", "boj", "m2"]:
        glf = glf.add(z(df[f"{k}_yoy"]) * W[k], fill_value=0); tw += W[k]
    for k, col in [("tga", "TGA"), ("rrp", "RRP")]:
        glf = glf.add(z(df[col]) * W[k], fill_value=0); tw += W[k]
    df["glf_proxy"] = glf / tw
    df["dnet_us_30"] = df["net_us"] - df["net_us"].shift(30)
    for h in (7, 30):
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    return df


def auc(y, score):
    yy = (np.asarray(y) > 0).astype(int)
    if len(np.unique(yy)) < 2:
        return np.nan
    return stats.mannwhitneyu(score[yy == 1], score[yy == 0]).statistic / (yy.sum() * (len(yy) - yy.sum()))


def wf_auc(df, cols, horizon, folds=6, embargo=30, d=None):
    d = (d if d is not None else df.dropna(subset=cols + [f"fwd{horizon}"])).reset_index(drop=True)
    n = len(d); fold = n // (folds + 1)
    per = []
    for k in range(1, folds + 1):
        tr_end = fold * k; te0 = tr_end + embargo; te1 = min(te0 + fold, n)
        tr, te = d.iloc[:tr_end], d.iloc[te0:te1]
        if len(te) < 50 or len(tr) < 200:
            continue
        r = sm.OLS(tr[f"fwd{horizon}"], sm.add_constant(tr[cols])).fit()
        pr = r.predict(sm.add_constant(te[cols]))
        per.append(auc(te[f"fwd{horizon}"].values, pr.values))
    return per


def main():
    df = load()
    print(f"n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    cc = df[["glf_proxy", "dnet_us_30"]].dropna()
    print(f"corr(glf_proxy, dnet_us_30) = {np.corrcoef(cc['glf_proxy'], cc['dnet_us_30'])[0,1]:+.3f} (n={len(cc)})")

    print("\n=== (1) incremental over GLF proxy, fwd30 (HAC30 + era dummies) ===")
    d = df.dropna(subset=["glf_proxy", "dnet_us_30", "fwd30"]).copy()
    X = pd.get_dummies(d["era"], prefix="era", drop_first=True).astype(float)
    X["glf"] = z(d["glf_proxy"]).values.astype(float)
    X["x"] = z(d["dnet_us_30"]).values.astype(float)
    X = sm.add_constant(X).astype(float)
    r = sm.OLS(d["fwd30"].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
    print(f"  n={len(d)}  glf coef={r.params['glf']:+.3f}(p={r.pvalues['glf']:.3f})  "
          f"dnet coef={r.params['x']:+.3f}(p={r.pvalues['x']:.3f})")
    d2 = df.dropna(subset=["glf_proxy", "dnet_us_30", "fwd30"])
    per = wf_auc(df, ["glf_proxy", "dnet_us_30"], 30, d=d2)
    print(f"  WF AUC (glf_proxy + dnet_us_30): mean={np.nanmean(per):.3f} folds={[round(x,3) for x in per]}")

    print("\n=== (2) robustness of dnet_us_30 WF AUC ===")
    for h in (7, 30):
        for nf in (4, 5, 6, 7, 8):
            per = wf_auc(df, ["dnet_us_30"], h, folds=nf)
            print(f"  h={h:2d} folds={nf}: AUC={np.nanmean(per):.3f}  per-fold {[round(x,2) for x in per]}")

    print("\n=== (2b) label-shuffle permutation (h=30, folds=6, 200 draws) ===")
    d = df.dropna(subset=["dnet_us_30", "fwd30"]).reset_index(drop=True)
    obs = np.nanmean(wf_auc(df, ["dnet_us_30"], 30, folds=6, d=d))
    null = []
    for _ in range(200):
        tmp = d.copy()
        tmp["shuf"] = rng.permutation(tmp["dnet_us_30"].values)
        null.append(np.nanmean(wf_auc(df, ["shuf"], 30, folds=6, d=tmp)))
    null = np.array([x for x in null if not np.isnan(x)])
    print(f"  observed OOS AUC={obs:.3f}  null mean={null.mean():.3f} sd={null.std():.3f}  "
          f"p(>=obs)={(null >= obs).mean():.3f}  (n={len(null)})")


if __name__ == "__main__":
    main()
