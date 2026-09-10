#!/usr/bin/env python3
"""
tp_purged_walkforward.py — final gating: out-of-sample purged walk-forward of the
    regime-conditional term-premium rule vs FFR-only and vs a naive baseline.
================================================================================
Rule under test (from the regime-interaction screen):
    in NON-hike regime : signal = dTP2_30  (positive predictor)
    in HIKE regime     : signal = -TP2_level (level high -> fwd return low)
Predicts the SIGN of the next 30d BTC return. Purged expanding walk-forward with a
30-day embargo between train and test (target overlaps 30d).

Metrics per fold + pooled: AUC (sign of fwd30), IC (Spearman signal vs fwd30),
directional accuracy. Baselines: FFR-only and always-long (AUC=0.5).
Gate passes only if OOS AUC > 0.55 AND IC>0 with low fold variance.
Report-only; no blending.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 30            # forward horizon
EMBARGO = 30      # purge gap
N_FOLDS = 6


def era_of(d):
    y = int(str(d)[:4])
    return "era1_17-20" if y < 2021 else "era2_21-23" if y < 2024 else "era3_24-26"


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if None in (m.get("US10Y"), m.get("FEDFUNDS")) or b.get("close") is None:
            continue
        rows.append({"date": d, "era": era_of(d), "close": b["close"],
                     "US10Y": m["US10Y"], "FFR": m["FEDFUNDS"]})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["dTP2_30"] = (df["US10Y"] - df["FFR"]) - (df["US10Y"] - df["FFR"]).shift(30)
    df["TP2_level"] = df["US10Y"] - df["FFR"]
    df["FFR"] = df["FFR"]
    df["fwd30"] = np.log(df["close"].shift(-H) / df["close"]) * 100
    df["hike"] = (df["FFR"] - df["FFR"].shift(90) > 0.05).astype(int)
    df = df.dropna(subset=["dTP2_30", "TP2_level", "fwd30", "hike"]).reset_index(drop=True)
    return df


def auc(y_true_sign, score):
    """AUC of score for predicting positive fwd return (sign)."""
    y = (y_true_sign > 0).astype(int)
    if y.nunique() < 2:
        return np.nan
    return stats.mannwhitneyu(score[y == 1], score[y == 0]).statistic / (y.sum() * (len(y) - y.sum()))


def main():
    df = load()
    n = len(df)
    print(f"n={n}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}  hike share={df['hike'].mean():.2f}")

    fold = n // (N_FOLDS + 1)
    res = {"rule": [], "ffr": [], "baseline": []}
    print(f"\n{'fold':>4} {'n_test':>6}  {'rule_IC':>8} {'rule_AUC':>8} | {'ffr_IC':>8} {'ffr_AUC':>8} | {'naive_up':>8}")
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_start = tr_end + EMBARGO
        te_end = min(te_start + fold, n)
        if te_start >= n:
            break
        tr = df.iloc[:tr_end]
        te = df.iloc[te_start:te_end]
        if len(te) < 50 or len(tr) < 200:
            continue

        # --- RULE model: fit per-regime OLS on train, predict test ---
        def fit_regime(trx, tex):
            pred = pd.Series(index=tex.index, dtype=float)
            for rv in (0, 1):
                trr = trx[trx["hike"] == rv]
                ter = tex[tex["hike"] == rv]
                if len(trr) < 40 or len(ter) == 0:
                    continue
                # regressors: dTP2_30 and TP2_level (rule uses both, regime picks behaviour)
                Xtr = sm.add_constant(trr[["dTP2_30", "TP2_level"]])
                r = sm.OLS(trr["fwd30"], Xtr).fit()
                Xte = sm.add_constant(ter[["dTP2_30", "TP2_level"]])
                pred.loc[ter.index] = r.predict(Xte)
            return pred
        pred_rule = fit_regime(tr, te)

        # --- FFR-only comparison: level + change of FFR ---
        tr2 = tr.copy(); te2 = te.copy()
        tr2["dffr"] = tr2["FFR"].diff(30); te2["dffr"] = te2["FFR"].diff(30)
        tr2 = tr2.dropna(subset=["dffr"]); te2 = te2.dropna(subset=["dffr"])
        rf = sm.OLS(tr2["fwd30"], sm.add_constant(tr2[["FFR", "dffr"]])).fit()
        pred_ffr = rf.predict(sm.add_constant(te2[["FFR", "dffr"]]))

        m = pred_rule.notna()
        te_r = te.loc[m]; pr = pred_rule[m]
        ic_rule = stats.spearmanr(pr, te_r["fwd30"]).statistic
        auc_rule = auc(te_r["fwd30"], pr.values)
        m2 = pred_ffr.notna()
        ic_ffr = stats.spearmanr(pred_ffr[m2], te2.loc[m2, "fwd30"]).statistic
        auc_ffr = auc(te2.loc[m2, "fwd30"], pred_ffr[m2].values)
        naive = (te_r["fwd30"] > 0).mean()
        res["rule"].append((ic_rule, auc_rule)); res["ffr"].append((ic_ffr, auc_ffr)); res["baseline"].append(naive)
        print(f"{k:>4} {len(te_r):>6}  {ic_rule:>8.3f} {auc_rule:>8.3f} | {ic_ffr:>8.3f} {auc_ffr:>8.3f} | {naive:>8.2f}")

    def agg(lst, i):
        v = [x[i] for x in lst if x[i] is not None and not (isinstance(x[i], float) and np.isnan(x[i]))]
        return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)

    print("\n=== pooled OOS (mean ± sd over folds) ===")
    for name, key in [("RULE(regime-cond TP2)", "rule"), ("FFR-only", "ffr")]:
        icm, icsd, _ = agg(res[key], 0)
        aum, ausd, nf = agg(res[key], 1)
        print(f"  {name:24s} IC={icm:+.3f} ±{icsd:.3f}   AUC={aum:.3f} ±{ausd:.3f}  (folds={nf})")
    bm = np.mean(res["baseline"])
    print(f"  {'baseline P(up) per fold':24s} = {bm:.3f}")
    print("\nGATE: pass only if RULE AUC>0.55, IC>0, and LOW fold variance.")


if __name__ == "__main__":
    main()
