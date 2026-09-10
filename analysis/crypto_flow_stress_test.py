#!/usr/bin/env python3
"""
crypto_flow_stress_test.py — P1: do crypto-native MARGINAL-FLOW features improve
    REGIME (trend / stress) detection out-of-sample, beyond the era-stable core?
================================================================================
Rationale: macro-liquidity channels are structurally exhausted for BTC; the only
era-stable signal found so far is funding (crypto-native). Pivot: test the flow family
(funding, taker imbalance, trade-size skew, whale share) as REGIME detectors with a
purged walk-forward gate, and measure what each adds over a price/vol core.

Targets (forward 30d, so this is DETECTION not hindsight):
  stress_fwd = fwd30 realized vol > trailing-365d median
  bull_fwd   = close_{t+30} > SMA200_{t+30}
Models (logistic, standardized on TRAIN only), purged expanding WF (embargo 30):
  base      : vol20, mom30
  +funding  : base + funding_mean, df7, df30
  +flow     : +funding + taker_imbalance, tbr, whale_share, p99_notional, med_notional, buy_sell_ratio
Gate: AUC > 0.55 with low fold variance.
Report-only.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBARGO = 30


def load():
    vis = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    ofl = json.load(open(os.path.join(REPO, "data/binance_orderflow_daily.json")))
    rows = []
    for d in sorted(set(vis) & set(ofl)):
        v, o = vis[d], ofl[d]
        if v.get("close") is None or o.get("total_quote") in (None, 0):
            continue
        tq = o["total_quote"]
        rows.append({
            "date": d, "close": v["close"],
            "funding_mean": v.get("funding_mean"),
            "taker_imbalance": o.get("taker_imbalance_quote"),
            "tbr": o.get("taker_buy_ratio"),
            "whale_share": (o.get("whale_hi_quote", 0.0) or 0.0) / tq,
            "p99_notional": o.get("p99_notional"), "med_notional": o.get("med_notional"),
            "buy_sell_ratio": (o.get("n_buy", 0) or 0) / max(1, (o.get("n_sell", 1) or 1)),
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["ret"] = np.log(df["close"]).diff()
    df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(365) * 100
    df["mom30"] = df["close"] / df["close"].shift(30) - 1.0
    df["df7"] = df["funding_mean"] - df["funding_mean"].shift(7)
    df["df30"] = df["funding_mean"] - df["funding_mean"].shift(30)
    df["tk_imb7"] = df["taker_imbalance"].rolling(7).mean()
    # forward targets
    df["fwd_vol30"] = df["ret"].shift(-1).rolling(30).std().shift(-29) * np.sqrt(365) * 100
    med = df["fwd_vol30"].rolling(365, min_periods=120).median()
    df["stress_fwd"] = (df["fwd_vol30"] > med).astype(float)
    sma200 = df["close"].rolling(200).mean()
    df["bull_fwd"] = (df["close"].shift(-30) > sma200.shift(-30)).astype(float)
    return df


CORE = ["vol20", "mom30"]
FUNDING = ["funding_mean", "df7", "df30"]
FLOW = ["taker_imbalance", "tk_imb7", "tbr", "whale_share", "p99_notional", "med_notional", "buy_sell_ratio"]


def purged_wf(df, feats, target, folds=6):
    d = df.dropna(subset=feats + [target]).reset_index(drop=True)
    n = len(d); step = n // (folds + 1)
    aucs = []
    for k in range(1, folds + 1):
        tr_end = step * k; te0 = tr_end + EMBARGO; te1 = min(te0 + step, n)
        tr, te = d.iloc[:tr_end], d.iloc[te0:te1]
        if len(te) < 50 or len(tr) < 300:
            continue
        if tr[target].nunique() < 2 or te[target].nunique() < 2:
            continue
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        Xtr = (tr[feats] - mu) / sd
        Xte = (te[feats] - mu) / sd
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, tr[target].astype(int))
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(te[target].astype(int), p))
    return aucs


def main():
    df = load()
    print(f"n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")
    print(f"stress base rate={df['stress_fwd'].mean():.2f}  bull base rate={df['bull_fwd'].mean():.2f}")

    for target in ["stress_fwd", "bull_fwd"]:
        print(f"\n===== target: {target} (purged WF, folds=6, embargo 30) =====")
        contrib = []
        for label, feats in [("base(vol+mom)", CORE),
                             ("+funding", CORE + FUNDING),
                             ("+flow", CORE + FUNDING + FLOW),
                             ("funding ONLY", FUNDING)]:
            aucs = purged_wf(df, feats, target)
            if aucs:
                print(f"  {label:16s} AUC={np.mean(aucs):.3f} ±{np.std(aucs):.3f}  "
                      f"folds={[round(a,2) for a in aucs]}")
            contrib.append((label, np.mean(aucs) if aucs else np.nan))
        # incremental
        b = dict(contrib)
        print(f"  -> ΔAUC funding over base = {b['+funding']-b['base(vol+mom)']:+.3f}   "
              f"ΔAUC flow over +funding = {b['+flow']-b['+funding']:+.3f}")

    print("\nGate: pass only if AUC>0.55 with low fold variance.")

    # permutation control: is the funding contribution beyond chance?
    rng = np.random.default_rng(0)
    for target in ["bull_fwd", "stress_fwd"]:
        base_auc = np.mean(purged_wf(df, CORE, target))
        obs = np.mean(purged_wf(df, CORE + FUNDING, target))
        null = []
        dd0 = df.copy()
        for _ in range(100):
            dd = dd0.copy()
            for c in FUNDING:
                dd[c] = rng.permutation(dd[c].values)
            null.append(np.mean(purged_wf(dd, CORE + FUNDING, target)))
        null = np.array([x for x in null if not np.isnan(x)])
        p = (null >= obs).mean()
        print(f"\n[perm] {target}: base={base_auc:.3f} obs(+funding)={obs:.3f}  "
              f"null mean={null.mean():.3f} sd={null.std():.3f}  p(>=obs)={p:.3f}")


if __name__ == "__main__":
    main()
