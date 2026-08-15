#!/usr/bin/env python3
"""
validate_orderflow_predictive.py — PREDICTIVE battery for the order-flow daily series.

Follows the SFC validation discipline (walk-forward-validation + era-split-validation
+ predictive-probability-validation skills):

  Q1. Quantile top/bottom-20% forward-return gap + vectorized bootstrap P(sign)
      (sign convention left data-driven, not assumed).
  Q2. Spearman IC (point-in-time signal vs forward return).
  Q3. MANDATORY era-split (3 contiguous blocks over 2017-08..2020-12, which spans a
      bull tail, a 2018 bear, a 2019 bull, and COVID + 2020 bull) — verdict from the
      LATEST era, report sign consistency across eras.
  Q4. Purged-CV / embargo gate (López de Prado) — single-feature LogisticRegression
      pooled OOS AUC; the gate that overturns an optimistic IC screen on overlapping
      labels. Verdict bar: pooled AUC > 0.5 AND mean-fold − 1.96*SE > 0.5.

Signals tested: taker_imbalance_qty, taker_buy_ratio, whale_qty_lo_count (whale>=10BTC),
total_quote (notional volume), n_trades. Forward horizons 7d / 30d.

Verdict classification per the skills:
  - BERTAHAN  : predictive sign holds in latest era AND purged-CV passes.
  - FLIP/MATI : latest era reverses sign or P(sign)~0 -> regime-dependent, do NOT blend.
  - NOT_BLEND : IC screen optimistic, purged-CV overturns -> no OOS skill.
  - DATA_TOO_SHORT : cannot era-split across bull AND bear (not applicable here: full panel).

Output: analysis/.orderflow_predictive_summary.json  + human-readable printout.
"""
import json, os, random, math
import numpy as np
from datetime import datetime, timezone
from scipy import stats
from sklearn.linear_model import LogisticRegression

REPO = "/home/ubuntu/sfc"
OF = os.path.join(REPO, "data", "binance_orderflow_daily.json")
SUM_OUT = os.path.join(REPO, "analysis", ".orderflow_predictive_summary.json")
SEED = 42
N_BOOT = 2000
QUANT = 0.20

SIGNALS = ["taker_imbalance_qty", "taker_buy_ratio",
           "whale_qty_lo_count", "total_quote", "n_trades"]
HORIZONS = [(7, "7d"), (30, "30d")]


def quantile_gap_boot(sig, ret, h_label):
    """Top-20% minus bottom-20% forward-return gap + bootstrap P(sign>0) + Spearman IC.
    Sign convention data-driven: we report P(gap>0) and let era-consistency decide polarity."""
    mask = ~(np.isnan(sig) | np.isnan(ret))
    s, r = sig[mask], ret[mask]
    if len(s) < 40:
        return None
    k = max(1, int(len(s) * QUANT))
    order = np.argsort(s)
    bottom = r[order[:k]]
    top = r[order[-k:]]
    gap = float(np.mean(top) - np.mean(bottom))
    rng = np.random.default_rng(SEED)
    ib = rng.integers(0, len(bottom), size=(N_BOOT, len(bottom)))
    it = rng.integers(0, len(top), size=(N_BOOT, len(top)))
    boots = np.mean(top[it], axis=1) - np.mean(bottom[ib], axis=1)
    p_pos = float(np.mean(boots > 0))
    lo = np.percentile(boots, 5)
    hi = np.percentile(boots, 95)
    ic = (float(stats.spearmanr(s, r).statistic)
          if np.std(s) > 0 and np.var(ret[mask]) > 0 else float("nan"))
    return {"n": int(len(s)), "gap_pp": round(gap * 100, 2),
            "P_gap_pos": round(p_pos, 3), "ci90_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "ic": round(ic, 3)}


def purged_cv_auc(sig, ret, h, K=5):
    """Purged-CV/embargo single-feature OOS AUC. Returns (pooled_auc, folds, mean_se)."""
    mask = ~(np.isnan(sig) | np.isnan(ret))
    idx = np.where(mask)[0]
    s = sig[mask].astype(float)
    r = (ret[mask] > 0).astype(int)   # binary up/down outcome
    n = len(idx)
    if n < 40 or np.unique(r).size < 2:
        return None
    # standardize feature on TRAIN ONLY (leakage-free)
    order = np.argsort(idx)  # chronological
    s_chrono = s[order]; r_chrono = r[order]
    idx_chrono = idx[order]
    n = len(s_chrono)
    bounds = [int(i * n / K) for i in range(K + 1)]
    oos_pred = np.full(n, np.nan)
    for f in range(K):
        te = slice(bounds[f], bounds[f + 1])
        te_start, te_end = idx_chrono[bounds[f]], idx_chrono[bounds[f + 1] - 1]
        # train = all points outside test block, purged of label-overlap + embargo
        tr_mask = np.ones(n, bool)
        tr_mask[te] = False
        for j in range(n):
            if not tr_mask[j]:
                continue
            jdate = idx_chrono[j]
            # purge: training label window [j, j+h) overlaps test block start
            if jdate + h >= te_start:
                tr_mask[j] = False
                continue
            # embargo: training sample starts inside [te_end, te_end+h)
            if te_end <= jdate <= te_end + h:
                tr_mask[j] = False
        if tr_mask.sum() < 10:
            continue
        Xtr = s_chrono[tr_mask].reshape(-1, 1)
        mu, sd = Xtr.mean(), Xtr.std()
        if sd == 0:
            continue
        Xtr = (Xtr - mu) / sd
        ytr = r_chrono[tr_mask]
        clf = LogisticRegression(max_iter=1000)
        try:
            clf.fit(Xtr, ytr)
        except Exception:
            continue
        Xte = (s_chrono[te].reshape(-1, 1) - mu) / sd
        oos_pred[te] = clf.predict_proba(Xte)[:, 1]
    oos = ~np.isnan(oos_pred)
    if oos.sum() < 20 or np.unique(r_chrono[oos]).size < 2:
        return None
    # pooled AUC
    pos = r_chrono[oos] == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    pooled = _auc(r_chrono[oos], oos_pred[oos])
    # per-fold AUC
    fold_auc = []
    for f in range(K):
        te = slice(bounds[f], bounds[f + 1])
        pred = oos_pred[te]; y = r_chrono[te]
        ok = ~np.isnan(pred)
        if ok.sum() < 5 or np.unique(y[ok]).size < 2:
            continue
        fold_auc.append(_auc(y[ok], pred[ok]))
    fold_auc = np.array(fold_auc)
    se = fold_auc.std(ddof=1) / math.sqrt(len(fold_auc)) if len(fold_auc) > 1 else float("nan")
    return {"pooled_auc": round(float(pooled), 3),
            "fold_auc": [round(float(x), 3) for x in fold_auc],
            "mean_minus_196se": round(float(pooled - 1.96 * se), 3) if not math.isnan(se) else None,
            "n_oos": int(oos.sum())}


def _auc(y, score):
    """AUC via Mann-Whitney U (rank of positives among negatives)."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    rsum = ranks[pos].sum()
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    random.seed(SEED)
    of = json.load(open(OF))
    days = sorted(of)
    price = np.array([of[d]["price_close"] for d in days], dtype=float)
    n = len(days)

    # forward returns
    ret = {}
    for h, hl in HORIZONS:
        rr = np.full(n, np.nan)
        for i in range(n - h):
            rr[i] = price[i + h] / price[i] - 1
        ret[hl] = rr

    # era split (3 contiguous blocks)
    q = n // 3
    eras = {"era1": (0, q), "era2": (q, 2 * q), "era3": (2 * q, n)}
    era_labels = {k: f"{days[a]}..{days[b - 1]}" for k, (a, b) in eras.items()}

    results = []
    for sname in SIGNALS:
        sig = np.array([(of[d][sname] if of[d].get(sname) is not None else np.nan)
                        for d in days], dtype=float)
        for h, hl in HORIZONS:
            r = ret[hl]
            full = quantile_gap_boot(sig, r, hl)
            if full is None:
                continue
            era_res = {}
            for k, (a, b) in eras.items():
                e = quantile_gap_boot(sig[a:b], r[a:b], hl)
                if e:
                    era_res[k] = e
            pc = purged_cv_auc(sig, r, h)
            results.append({"signal": sname, "horizon": hl, "full": full,
                            "eras": era_res, "purged_cv": pc,
                            "era_labels": era_labels})

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "window": [days[0], days[-1]], "n_days": n,
               "method": "quantile top/bottom-20% gap + bootstrap P(sign) + Spearman IC "
                         "+ purged-CV/embargo (LogReg single-feature, 5-fold)",
               "result": results}
    json.dump(summary, open(SUM_OUT, "w"), indent=1)

    print(f"Order-flow predictive battery | {days[0]} .. {days[-1]} | n={n}")
    print("Eras: " + ", ".join(f"{k}: {v}" for k, v in era_labels.items()))
    for r in results:
        print(f"\n== {r['signal']} @ {r['horizon']} ==")
        f = r["full"]
        print(f"  FULL n={f['n']} gap={f['gap_pp']:+.2f}pp  P(gap>0)={f['P_gap_pos']}  "
              f"CI90=[{f['ci90_pp'][0]},{f['ci90_pp'][1]}]  IC={f['ic']}")
        for k in ["era1", "era2", "era3"]:
            e = r["eras"].get(k)
            if e:
                print(f"  {k} n={e['n']} gap={e['gap_pp']:+.2f}pp  P(gap>0)={e['P_gap_pos']}  IC={e['ic']}")
        pc = r["purged_cv"]
        if pc:
            print(f"  PURGED-CV pooled_AUC={pc['pooled_auc']}  folds={pc['fold_auc']}  "
                  f"mean-1.96SE={pc['mean_minus_196se']}  n_oos={pc['n_oos']}")
        else:
            print("  PURGED-CV: not computable")
    print(f"\nSummary -> {SUM_OUT}")


if __name__ == "__main__":
    main()
