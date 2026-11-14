#!/usr/bin/env python3
"""Baseline test: does perp funding predict forward BTC returns beyond momentum/vol?

This is the first cross-layer candidate from the V2 blueprint to get a real test.
The house rule is that no new weight enters the composite before it beats a
price/vol baseline on held-out data, so this script answers exactly that question
and nothing more:

    target  : sign of the forward 7-day return (1 = up, 0 = down/flat)
    baseline: 7d momentum, 14d realised vol, 1d return
    candidate: baseline + funding level (7d mean) and funding z-score (63d)

Protocol
  - daily frame 2019-09 (Binance perp funding starts) .. latest common date;
  - PURGED expanding-window walk-forward: train on everything before the fold,
    drop the last 7 days of train (embargo) so labels cannot overlap the test
    block, then test the next 60 days;
  - logistic regression on standardised features, identical for both models;
  - metrics: AUC and Brier; a candidate only counts if it improves AUC in a
    majority of folds AND does not worsen Brier;
  - era split (2019-2022 / 2022-2026) reported separately, because a pooled
    number can hide a sign flip;
  - permutation test on the AUC delta of the pooled out-of-sample predictions.

Output: analysis/funding_signal_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "funding_signal_results.json"

sys.path.insert(0, str(ROOT / "analysis"))

HORIZON = 7          # forward days
EMBARGO = 7          # days dropped between train and test
TEST_BLOCK = 60      # days per fold
MIN_TRAIN = 365      # days before the first fold
N_PERM = 200


def load_frame():
    daily = json.load(open(ROOT / "data" / "binance_vision_daily.json"))
    funding = json.load(open(ROOT / "data" / "binance_funding_daily.json"))
    px = pd.DataFrame.from_dict(daily, orient="index")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    close = pd.to_numeric(px["close"], errors="coerce")
    fund = pd.Series({pd.Timestamp(k): v.get("rate_mean") for k, v in funding.items()
                      if isinstance(v, dict)}).sort_index()
    df = pd.DataFrame({"close": close})
    df["funding"] = fund.reindex(df.index)
    df["funding"] = df["funding"].ffill(limit=3)
    df["ret_1d"] = df["close"].pct_change()
    df["ret_7d"] = df["close"].pct_change(HORIZON)
    df["vol_14d"] = df["ret_1d"].rolling(14).std()
    df["funding_7d"] = df["funding"].rolling(7).mean()
    mu = df["funding"].rolling(63).mean()
    sd = df["funding"].rolling(63).std()
    df["funding_z"] = (df["funding"] - mu) / sd.replace(0, np.nan)
    df["fwd"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df["y"] = (df["fwd"] > 0).astype(float)
    df.loc[df["fwd"].isna(), "y"] = np.nan
    return df


def fit_predict(train, test, feats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = train[feats].to_numpy(dtype=float)
    ytr = train["y"].to_numpy(dtype=float)
    Xte = test[feats].to_numpy(dtype=float)
    if len(np.unique(ytr)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    if len(np.unique(y)) < 2:
        return None
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "n": int(len(y))}


def main():
    df = load_frame()
    base_feats = ["ret_7d", "vol_14d", "ret_1d"]
    cand_feats = base_feats + ["funding_7d", "funding_z"]
    data = df.dropna(subset=base_feats + ["y"]).copy()
    data_c = df.dropna(subset=cand_feats + ["y"]).copy()
    if len(data_c) < MIN_TRAIN + TEST_BLOCK + 30:
        print("[funding] not enough data", file=sys.stderr)
        return 0

    idx = data_c.index
    folds = []
    oos_y, oos_b, oos_c, oos_idx = [], [], [], []
    start = MIN_TRAIN
    while start + TEST_BLOCK <= len(idx):
        tr_end = start - EMBARGO
        if tr_end < 30:
            start += TEST_BLOCK
            continue
        tr = data_c.iloc[:tr_end]
        te = data_c.iloc[start:start + TEST_BLOCK]
        p_base = fit_predict(tr, te, base_feats)
        p_cand = fit_predict(tr, te, cand_feats)
        if p_base is None or p_cand is None:
            start += TEST_BLOCK
            continue
        m_b = metrics(te["y"].to_numpy(), p_base)
        m_c = metrics(te["y"].to_numpy(), p_cand)
        if m_b and m_c:
            folds.append({"test_from": str(te.index[0].date()), "test_to": str(te.index[-1].date()),
                          "base": m_b, "cand": m_c,
                          "d_auc": round(m_c["auc"] - m_b["auc"], 4),
                          "d_brier": round(m_c["brier"] - m_b["brier"], 4)})
            oos_y.extend(te["y"].to_numpy()); oos_b.extend(p_base); oos_c.extend(p_cand)
            oos_idx.extend(list(te.index))
        start += TEST_BLOCK

    if not folds:
        print("[funding] no usable folds", file=sys.stderr)
        return 0

    y = np.array(oos_y); pb = np.array(oos_b); pc = np.array(oos_c)
    pooled_b = metrics(y, pb); pooled_c = metrics(y, pc)
    win_auc = sum(1 for f in folds if f["d_auc"] > 0)
    worse_brier = sum(1 for f in folds if f["d_brier"] > 0)

    # permutation test on the pooled AUC delta
    rng = np.random.default_rng(11)
    from sklearn.metrics import roc_auc_score
    obs = pooled_c["auc"] - pooled_b["auc"]
    deltas = []
    for _ in range(N_PERM):
        mix = np.where(rng.random(len(pc)) < 0.5, pb, pc)
        mix2 = np.where(rng.random(len(pc)) < 0.5, pb, pc)
        try:
            deltas.append(roc_auc_score(y, mix2) - roc_auc_score(y, mix))
        except Exception:
            continue
    p_val = float(np.mean(np.abs(np.array(deltas)) >= abs(obs))) if deltas else None

    era = []
    oos_dates = pd.to_datetime(oos_idx)
    for lo, hi, name in [(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01"), "era1 2019-2022"),
                         (pd.Timestamp("2022-01-01"), pd.Timestamp("2030-01-01"), "era2 2022-2026")]:
        mask = (oos_dates >= lo) & (oos_dates < hi)
        if mask.sum() > 60:
            mb = metrics(y[mask], pb[mask]); mc = metrics(y[mask], pc[mask])
            if mb and mc:
                era.append({"era": name, "base_auc": round(mb["auc"], 4),
                            "cand_auc": round(mc["auc"], 4),
                            "d_auc": round(mc["auc"] - mb["auc"], 4), "n": mb["n"]})

    if win_auc > len(folds) / 2 and worse_brier <= len(folds) / 2 and (p_val is None or p_val < 0.05):
        verdict = "CANDIDATE — funding improves AUC in most folds, Brier not worse, permutation p<0.05"
        accepted = True
    else:
        verdict = ("REJECT — funding does not add reliable information over "
                   "momentum+vol out of sample")
        accepted = False

    report = {
        "protocol": {"horizon_days": HORIZON, "embargo_days": EMBARGO,
                     "test_block": TEST_BLOCK, "min_train": MIN_TRAIN,
                     "baseline_features": base_feats, "candidate_features": cand_feats,
                     "model": "LogisticRegression(C=0.5), standardised",
                     "target": "sign of forward 7d return"},
        "n_obs": int(len(y)), "n_folds": len(folds),
        "first_test": folds[0]["test_from"], "last_test": folds[-1]["test_to"],
        "pooled": {"base": pooled_b, "cand": pooled_c,
                   "d_auc": round(pooled_c["auc"] - pooled_b["auc"], 4),
                   "d_brier": round(pooled_c["brier"] - pooled_b["brier"], 4)},
        "folds_cand_wins_auc": win_auc,
        "folds_cand_worse_brier": worse_brier,
        "perm_p_value": p_val,
        "era_split": era,
        "accepted": accepted, "verdict": verdict, "folds": folds,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)

    print(f"[funding] obs={len(y)} folds={len(folds)} "
          f"AUC base={pooled_b['auc']:.4f} cand={pooled_c['auc']:.4f} "
          f"d={report['pooled']['d_auc']:+.4f}", file=sys.stderr)
    print(f"[funding] wins {win_auc}/{len(folds)} folds, worse-Brier {worse_brier}/{len(folds)}, "
          f"perm p={p_val}", file=sys.stderr)
    print(json.dumps({"pooled": report["pooled"], "era_split": era, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
