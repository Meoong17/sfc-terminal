#!/usr/bin/env python3
"""
validate_etf_demand_quality.py — ETF-flow "Demand Quality" predictive battery.

Validates the article-derived claim that ETF spot/institutional demand (and the
"Same Price, Different Flow" distinction) carries OUT-OF-SAMPLE information about
forward BTC returns — separate from price alone.

Why this signal, now:
  - Funding / premium          -> already validated (purged_cv_funding.py).
  - Order-flow imbalance (CVD) -> already validated (validate_orderflow_predictive.py).
  - ETF spot demand (DQS)      -> NOT yet validated, and it is the NEW variable the
                                  SFC Behavioral Flow Engine article puts at the centre
                                  ("Same Price, Different Flow": ETF+spot vs leverage).
  - ETF data window: 671 daily flows 2024-01-11..2026-08-21 (from .etf_cache.json,
    sourced via Farside browser scrape by cron). Long enough for an honest test,
    short enough that era diversity is LIMITED (mostly one regime) -> verdict
    classified carefully, not forced.

Signals tested (point-in-time, trailing only):
  - etf_flow_1d : same-day net ETF flow (USD).
  - etf_net_5d  : 5-session rolling SUM of net ETF flow (persistence, DQS-adjacent).
  - etf_net_10d : 10-session rolling sum (FlowPersistence dimension of RTS).
  - dq_comp     : DQS-like composite = z(etf_net_5d) + z(taker_imbalance_qty)  (spot-CVD).
                  ETF demand + spot aggression, minus leverage is deliberately deferred
                  (funding already validated separately; OI history is data-limited).

Method (identical discipline to validate_orderflow_predictive.py / skills):
  Q1 quantile top/bottom-20% forward-return gap + bootstrap P(sign) + Spearman IC.
  Q2 era-split (3 contiguous blocks over 2024-2026) — sign consistency, verdict from
     LATEST era (caveat: single-regime window, so "era-flip" here is weak evidence).
  Q3 purged-CV / embargo (Lopez de Prado) single-feature LogisticRegression pooled OOS AUC.

Verdict rules (per skill, SFC standing rule — DO NOT blend into scoring either way):
  - pooled AUC > 0.5 AND mean-fold - 1.96*SE > 0.5  AND latest-era sign consistent -> BERTAHAN
  - latest era reverses sign / P(sign)~0            -> FLIP/MATI (regime-dependent, do NOT blend)
  - IC optimistic but purged-CV overturns           -> NOT_BLEND (no OOS skill)
  - window too short to era-split bull AND bear     -> DATA_TOO_SHORT

Pure analysis. No production change. No scoring change.
"""
import json, os, math, random
import numpy as np
from datetime import datetime, timezone
from scipy import stats
from sklearn.linear_model import LogisticRegression

REPO = "/home/ubuntu/sfc"
ETF_CACHE = os.path.join(REPO, ".etf_cache.json")
VISION = os.path.join(REPO, "data", "binance_vision_daily.json")
ORDERFLOW = os.path.join(REPO, "data", "binance_orderflow_daily.json")
SUM_OUT = os.path.join(REPO, "analysis", ".etf_dq_summary.json")
SEED = 42
N_BOOT = 2000
QUANT = 0.20

# reuse the exact helpers from the order-flow validator (single source of truth)
sys_path_ok = None
try:
    from validate_orderflow_predictive import quantile_gap_boot, purged_cv_auc
    sys_path_ok = True
except ImportError:
    # fallback: re-declare minimal local copies so this file runs standalone
    sys_path_ok = False


# ── local fallbacks (only used if import failed) ────────────────────────────
def quantile_gap_boot(sig, ret, h_label):
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


def _auc(y, score):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    rsum = ranks[pos].sum()
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def purged_cv_auc(sig, ret, h, K=5):
    mask = ~(np.isnan(sig) | np.isnan(ret))
    idx = np.where(mask)[0]
    s = sig[mask].astype(float)
    r = (ret[mask] > 0).astype(int)
    n = len(idx)
    if n < 40 or np.unique(r).size < 2:
        return None
    order = np.argsort(idx)
    s_chrono = s[order]; r_chrono = r[order]; idx_chrono = idx[order]
    n = len(s_chrono)
    bounds = [int(i * n / K) for i in range(K + 1)]
    oos_pred = np.full(n, np.nan)
    for f in range(K):
        te = slice(bounds[f], bounds[f + 1])
        te_start, te_end = idx_chrono[bounds[f]], idx_chrono[bounds[f + 1] - 1]
        tr_mask = np.ones(n, bool)
        tr_mask[te] = False
        for j in range(n):
            if not tr_mask[j]:
                continue
            jdate = idx_chrono[j]
            if jdate + h >= te_start:
                tr_mask[j] = False
                continue
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
    pos = r_chrono[oos] == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    pooled = _auc(r_chrono[oos], oos_pred[oos])
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


def main():
    random.seed(SEED)
    etf = json.load(open(ETF_CACHE))
    flows = etf.get("flows", [])
    flow_by_date = {}
    for f in flows:
        d = f.get("date")
        u = f.get("total_usd")
        if d and u is not None:
            flow_by_date[d] = float(u)
    print(f"[ETF] flows={len(flow_by_date)} dates={min(flow_by_date)}..{max(flow_by_date)}")

    vision = json.load(open(VISION))
    of = json.load(open(ORDERFLOW))

    # build aligned series on the ETF window
    days = sorted(flow_by_date)
    n = len(days)
    etf_1d = np.array([flow_by_date[d] for d in days], dtype=float)
    # rolling sums
    def roll(a, w):
        out = np.full(len(a), np.nan)
        for i in range(w - 1, len(a)):
            out[i] = np.sum(a[i - w + 1:i + 1])
        return out
    etf_5d = roll(etf_1d, 5)
    etf_10d = roll(etf_1d, 10)

    # BTC close on same dates (from vision daily)
    closes = []
    taker_imb = []
    for d in days:
        v = vision.get(d)
        closes.append(v.get("close") if v else np.nan)
        o = of.get(d)
        taker_imb.append(o.get("taker_imbalance_qty") if o else np.nan)
    closes = np.array(closes, dtype=float)
    taker_imb = np.array(taker_imb, dtype=float)

    # forward returns from actual BTC close
    ret = {}
    for h, hl in [(7, "7d"), (30, "30d")]:
        rr = np.full(n, np.nan)
        for i in range(n - h):
            if not np.isnan(closes[i]) and not np.isnan(closes[i + h]) and closes[i] > 0:
                rr[i] = closes[i + h] / closes[i] - 1
        ret[hl] = rr

    # DQS-like composite: z(etf_5d) + z(taker_imbalance)  (each z-scored on full window, trailing values)
    def zscore(a):
        m = ~np.isnan(a)
        out = np.full(len(a), np.nan)
        if m.sum() > 1 and np.std(a[m]) > 0:
            out[m] = (a[m] - np.mean(a[m])) / np.std(a[m])
        return out
    z_etf5 = zscore(etf_5d)
    z_imb = zscore(taker_imb)
    dq = z_etf5 + z_imb

    signals = {
        "etf_flow_1d": etf_1d,
        "etf_net_5d": etf_5d,
        "etf_net_10d": etf_10d,
        "dq_comp_5d": dq,
    }

    # era split: 3 contiguous blocks over ETF window (NOTE: single-regime caveat)
    q = n // 3
    eras = {"era1": (0, q), "era2": (q, 2 * q), "era3": (2 * q, n)}
    era_labels = {k: f"{days[a]}..{days[b - 1]}" for k, (a, b) in eras.items()}

    results = []
    for sname, sig in signals.items():
        for h, hl in [(7, "7d"), (30, "30d")]:
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
                            "eras": era_res, "purged_cv": pc, "era_labels": era_labels})

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "window": [days[0], days[-1]], "n_days": n,
               "method": "ETF-demand-quality battery: quantile top/bottom-20% gap + bootstrap "
                         "P(sign) + Spearman IC + purged-CV/embargo (LogReg single-feature, 5-fold)",
               "note": "ETF window 2024-2026 is single-regime (mostly bull); era-flip here is weak evidence.",
               "result": results}
    json.dump(summary, open(SUM_OUT, "w"), indent=1)

    print(f"\nETF Demand-Quality predictive battery | {days[0]} .. {days[-1]} | n={n}")
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
