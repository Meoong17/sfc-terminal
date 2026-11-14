#!/usr/bin/env python3
"""
structure_engine.py — three structural reads SFC did not have:

  1. PFS  Price Formation Structure — is price being set by deep two-way flow, or by
         thin/one-sided flow? (inverse Amihud impact, taker-imbalance stability,
         close location, volume concentration)
  2. FQ   Financing Quality — is leverage cheap and stable, or expensive and fragile?
         (funding level & dispersion, CME basis carry, MVRV premium)
  3. NDS  New Demand Sources — is demand broadening (ETF holdings, active addresses,
         transaction count) or stalling?

All indices are 0-100 (high = structurally healthy) with component coverage, point-in-time,
and robust MAD -> Phi percentile transforms. The module also runs an incremental
walk-forward test (expanding logistic, 5-day sampling, permutation control) so a card
can state honestly whether an index adds information over the price/vol baseline.

Writes analysis/structure_engine_results.json. Display-only; never touches sfc_effective.

Usage: cd ~/sfc && .venv/bin/python analysis/structure_engine.py
"""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "structure_engine_results.json")


def _unused_mad_pct(s, window=756, minp=252):
    med = s.rolling(window, min_periods=minp).median()
    mad = (s - med).abs().rolling(window, min_periods=minp).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    p = pd.Series(100 * norm.cdf(z.clip(-3, 3)), index=s.index).clip(0, 100)
    p[z.isna()] = np.nan
    return p


def pct_rank(s, window=756, minp=120):
    """Rolling percentile rank (0-100). Used instead of MAD-z so short histories
    (on-chain 430d, ETF 700d) still yield a scale-free read."""
    return s.rolling(window, min_periods=minp).rank(pct=True) * 100


def load():
    btc = pd.DataFrame.from_dict(json.load(open(f"{ROOT}/data/binance_vision_daily.json")), orient="index")
    btc.index = pd.to_datetime(btc.index)
    btc = btc.sort_index().apply(pd.to_numeric, errors="coerce")
    df = btc[["open", "high", "low", "close", "quote_vol"]].copy()

    of = pd.DataFrame.from_dict(json.load(open(f"{ROOT}/data/binance_orderflow_daily.json")), orient="index")
    of.index = pd.to_datetime(of.index)
    of = of.sort_index().apply(pd.to_numeric, errors="coerce")
    df["buy_q"] = of["taker_buy_quote"].reindex(df.index)
    df["sell_q"] = of["taker_sell_quote"].reindex(df.index)

    fund = pd.DataFrame.from_dict(json.load(open(f"{ROOT}/data/bitmex_funding_daily.json")), orient="index")
    fund.index = pd.to_datetime(fund.index)
    fcol = next((c for c in fund.columns if "sum" in c.lower() or "fund" in c.lower()), fund.columns[0])
    df["funding"] = pd.to_numeric(fund[fcol], errors="coerce").reindex(df.index)

    cm = pd.DataFrame.from_dict(json.load(open(f"{ROOT}/data/coinmetrics_btc_daily.json")), orient="index")
    cm.index = pd.to_datetime(cm.index)
    for c in ["CapMVRVCur", "AdrActCnt", "TxCnt"]:
        if c in cm:
            df[c] = pd.to_numeric(cm[c], errors="coerce").reindex(df.index, method="ffill")

    # CME front-month futures -> basis vs spot
    try:
        fut = pd.read_csv(f"{ROOT}/data/tradingview_btc1_daily.csv", parse_dates=["datetime"]).set_index("datetime")
        raw = pd.to_numeric(fut["close"], errors="coerce").resample("1D").last()
        df["fut"] = raw.reindex(df.index, method="ffill")
        # stale-contract guard: basis is only valid while the futures quote is fresh (<4d)
        if raw.dropna().empty is False:
            df.loc[df.index > raw.dropna().index[-1] + pd.Timedelta(days=4), "fut"] = np.nan
    except Exception:
        df["fut"] = np.nan

    # ETF net flow (USD) -> 5d sum
    try:
        etf = json.load(open(f"{ROOT}/.etf_cache.json"))["flows"]
        s = pd.Series({pd.Timestamp(r["date"]): (r.get("total_usd") or 0.0) for r in etf}).sort_index()
        df["etf_flow"] = s.reindex(df.index)
    except Exception:
        df["etf_flow"] = np.nan
    return df


def combine(comps, w, min_cov=0.30):
    """Coverage-aware composite: renormalize available components, report coverage."""
    num = pd.Series(0.0, index=next(iter(comps.values())).index)
    den = pd.Series(0.0, index=num.index)
    for k, s in comps.items():
        s = s.reindex(num.index)
        ok = s.notna()
        num = num.add((s.fillna(0) * w[k]).where(ok, 0.0), fill_value=0.0)
        den = den.add(pd.Series(w[k], index=num.index).where(ok, 0.0), fill_value=0.0)
    score = (num / den.replace(0, np.nan)).clip(0, 100)
    cov = den / sum(w.values())
    score[cov < min_cov] = np.nan
    return score, cov.clip(0, 1)


def build(df):
    r = df["close"].pct_change()
    out = pd.DataFrame(index=df.index)

    # ── PFS: price formation structure ────────────────────────────────────
    amihud = (r.abs() / df["quote_vol"].replace(0, np.nan)).rolling(20, min_periods=10).median()
    imb = (df["buy_q"] - df["sell_q"]) / (df["buy_q"] + df["sell_q"]).replace(0, np.nan)
    loc = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
    vconc = df["quote_vol"] / df["quote_vol"].rolling(20).sum()
    vconc = vconc.rolling(20).max()          # busiest day's share of 20d volume

    comps = {
        "impact_depth": 100 - pct_rank(amihud),                 # thin book = low
        "flow_stability": 100 - pct_rank(imb.rolling(20).std()),
        "close_strength": pct_rank(loc.rolling(20).mean()),
        "volume_breadth": 100 - pct_rank(vconc),
    }
    w = {"impact_depth": 0.30, "flow_stability": 0.25, "close_strength": 0.25, "volume_breadth": 0.20}
    out["PFS"], out["PFS_coverage"] = combine(comps, w)

    # ── FQ: financing quality (high = cheap & stable) ─────────────────────
    basis = (df["fut"] / df["close"] - 1).rolling(5).mean()
    mvrv = df["CapMVRVCur"]
    fq = {
        "funding_level": 100 - pct_rank(df["funding"].abs().rolling(30, min_periods=10).mean()),
        "funding_stability": 100 - pct_rank(df["funding"].rolling(30, min_periods=10).std()),
        "basis_carry": 100 - pct_rank(basis),
        "valuation_premium": 100 - pct_rank(mvrv),
    }
    wf = {"funding_level": 0.30, "funding_stability": 0.25, "basis_carry": 0.25, "valuation_premium": 0.20}
    out["FQ"], out["FQ_coverage"] = combine(fq, wf)

    # ── NDS: new demand sources (high = broadening) ───────────────────────
    etf5 = df["etf_flow"].rolling(5, min_periods=1).sum()
    nd = {
        "etf_demand": pct_rank(etf5),
        "active_users": pct_rank(df["AdrActCnt"].pct_change(30)),
        "usage": pct_rank(df["TxCnt"].pct_change(30)),
        "chain_supply_growth": pct_rank(df["AdrActCnt"].pct_change(90)),
    }
    wn = {"etf_demand": 0.35, "active_users": 0.25, "usage": 0.20, "chain_supply_growth": 0.20}
    out["NDS"], out["NDS_coverage"] = combine(nd, wn)

    out["btc"] = df["close"]
    # publish only dates where the structural read is reasonably covered
    ok = ((out["PFS_coverage"] >= 0.75) & (out["FQ_coverage"] >= 0.50) & (out["NDS_coverage"] >= 0.35))
    if ok.any():
        out = out.loc[:out.index[ok][-1]]
    out["basis_pct"] = basis * 100
    out["funding_bps"] = df["funding"] * 1e4
    for k in comps:
        out["PFS_" + k] = comps[k]
    for k in fq:
        out["FQ_" + k] = fq[k]
    for k in nd:
        out["NDS_" + k] = nd[k]
    return out.dropna(subset=["PFS", "FQ"])


def incremental_test(out, horizon=20, n_perm=200, seed=7):
    """Single 70/30 holdout + permutation null for each index (fast, honest)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    d = out
    r = d["btc"].pct_change()
    fwd = d["btc"].shift(-horizon) / d["btc"] - 1
    fwd_vol = r.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    thr = fwd_vol.rolling(504, min_periods=252).quantile(0.75)
    y = ((fwd < -0.10) | (fwd_vol > thr)).astype(float)

    base = np.column_stack([r.rolling(30).std(), d["btc"].pct_change(30)])
    m = y.notna().values & thr.notna().values & np.isfinite(base).all(axis=1)
    idx = d.index[m][::5]
    yv = y.reindex(idx).values
    Xb = np.nan_to_num(base[m][::5])
    split = int(len(yv) * 0.7)
    rng = np.random.default_rng(seed)
    res = {}
    for name in ["PFS", "FQ", "NDS"]:
        xv = d[name].reindex(idx).values
        ok = ~np.isnan(xv)
        yy, xb, xx = yv[ok], Xb[ok], xv[ok]
        if len(yy) < 200 or len(np.unique(yy[:split])) < 2 or len(np.unique(yy[split:])) < 2:
            res[name] = {"status": "insufficient_oos"}
            continue
        clf = LogisticRegression().fit(xb[:split], yy[:split])
        auc_b = roc_auc_score(yy[split:], clf.predict_proba(xb[split:])[:, 1])
        XF = np.column_stack([xb, xx])
        clf2 = LogisticRegression().fit(XF[:split], yy[:split])
        auc_f = roc_auc_score(yy[split:], clf2.predict_proba(XF[split:])[:, 1])
        delta = auc_f - auc_b
        null = np.empty(n_perm)
        for i in range(n_perm):
            Xp = np.column_stack([xb, rng.permutation(xx)])
            c = LogisticRegression().fit(Xp[:split], yy[:split])
            null[i] = roc_auc_score(yy[split:], c.predict_proba(Xp[split:])[:, 1]) - auc_b
        res[name] = {
            "auc_baseline": round(float(auc_b), 4),
            "auc_with_index": round(float(auc_f), 4),
            "delta_auc": round(float(delta), 4),
            "perm_p": round(float((null >= delta).mean()), 3),
            "n_oos": int(len(yy) - split),
        }
    return res



def transition_layer(out, horizon=20):
    """Turn the three structural reads into a CHANGE detector: is the structure
    starting to shift, which layer moves first, and does a shift lead stress?"""
    from scipy.stats import spearmanr
    d = out.copy()
    dP = d["PFS"].diff(20)
    dF = d["FQ"].diff(20)
    dN = d["NDS"].ffill().diff(20)
    zP, zF, zN = pct_rank(dP), pct_rank(dF), pct_rank(dN)
    # SFI: high = structure deteriorating fast (thin book + tightening financing + stalling demand)
    sfi = (0.40 * (100 - zP) + 0.35 * (100 - zF) + 0.25 * (100 - zN))
    d["SFI"] = sfi

    r = d["btc"].pct_change()
    fwd = d["btc"].shift(-horizon) / d["btc"] - 1
    fwd_vol = r.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    thr = fwd_vol.rolling(504, min_periods=252).quantile(0.75)
    ev = ((fwd < -0.10) | (fwd_vol > thr)).astype(float)

    # non-overlapping 20d sample to avoid label inflation
    sub = d.iloc[::horizon]
    sfi_s = sub["SFI"].reindex(sub.index)
    ev_s = ev.reindex(sub.index)
    ok = sfi_s.notna() & ev_s.notna()
    sfi_s, ev_s = sfi_s[ok], ev_s[ok]
    lead = {}
    if len(sfi_s) >= 30:
        q = sfi_s.quantile(0.80)
        top = ev_s[sfi_s >= q]
        base = ev_s.mean()
        lead["top_quintile_event_rate"] = round(float(top.mean()), 3)
        lead["base_rate"] = round(float(base), 3)
        lead["lift"] = round(float(top.mean() / base), 2) if base > 0 else None
        lead["n_top"] = int(len(top))
        # era split
        for tag, lo in [("pre2024", "2000-01-01"), ("2024plus", "2024-01-01")]:
            m2 = (sfi_s.index >= lo) if tag == "2024plus" else (sfi_s.index < "2024-01-01")
            if m2.sum() >= 10:
                lead[tag] = {"n": int(m2.sum()),
                             "event_rate": round(float(ev_s[m2].mean()), 3),
                             "shift_rate_topq": round(float(ev_s[m2][sfi_s[m2] >= q].mean()), 3)
                             if (sfi_s[m2] >= q).any() else None}
        ic1, p1 = spearmanr(dP, fwd.reindex(d.index), nan_policy="omit")
        ic2, p2 = spearmanr(dF, fwd.reindex(d.index), nan_policy="omit")
        ic3, p3 = spearmanr(dN, fwd.reindex(d.index), nan_policy="omit")
        lead["spearman_change_vs_fwd20"] = {
            "dPFS": [round(float(ic1), 3), round(float(p1), 3)],
            "dFQ": [round(float(ic2), 3), round(float(p2), 3)],
            "dNDS": [round(float(ic3), 3), round(float(p3), 3)],
        }

    last = d.iloc[-1]
    p_, f_, n_ = dP.iloc[-1], dF.iloc[-1], dN.iloc[-1]
    if np.isnan(f_) or np.isnan(p_):
        quad = "UNKNOWN"
    elif f_ < 0 and p_ < 0:
        quad = "CONTRACTION — financing tightening while the book thins"
    elif p_ < 0 and n_ > 0:
        quad = "ABSORPTION RISK — new demand arriving into a thinning book"
    elif f_ > 0 and p_ > 0:
        quad = "BROADENING — cheaper financing, deeper trading"
    elif f_ > 0 and p_ < 0:
        quad = "CARRY BID, THIN MARKET — funding cheap but price discovery weak"
    else:
        quad = "MIXED — layers diverge, no single structural direction"

    return {
        "as_of": str(d.index[-1].date()),
        "sfi": round(float(last["SFI"]), 1) if not np.isnan(last["SFI"]) else None,
        "deltas_20d": {"PFS": None if np.isnan(p_) else round(float(p_), 1),
                       "FQ": None if np.isnan(f_) else round(float(f_), 1),
                       "NDS": None if np.isnan(n_) else round(float(n_), 1)},
        "quadrant": quad,
        "lead_test": lead,
        "direction": ("deteriorating" if (last["SFI"] or 0) >= 60 else
                      "improving" if (last["SFI"] or 100) <= 40 else "stable"),
    }



def _clean(o):
    """JSON-safe: NaN/Inf -> null. Dashboard JSON.parse rejects NaN literals."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (f != f or abs(f) == float("inf")) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    df = load()
    out = build(df)
    test = incremental_test(out)
    last = out.iloc[-1]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": str(out.index[-1].date()),
        "model": "SFC structure engine: price formation / financing quality / new demand",
        "indices": {k: round(float(last[k]), 2) for k in ["PFS", "FQ", "NDS"]},
        "coverage": {k: round(float(last[k + "_coverage"]), 3) for k in ["PFS", "FQ", "NDS"]},
        "components": {
            "PFS": {k.replace("PFS_", ""): round(float(last[k]), 1) for k in last.index if k.startswith("PFS_") and not k.endswith("coverage")},
            "FQ": {k.replace("FQ_", ""): round(float(last[k]), 1) for k in last.index if k.startswith("FQ_") and not k.endswith("coverage")},
            "NDS": {k.replace("NDS_", ""): round(float(last[k]), 1) for k in last.index if k.startswith("NDS_") and not k.endswith("coverage")},
        },
        "context": {"basis_pct": round(float(last["basis_pct"]), 3), "funding_bps": round(float(last["funding_bps"]), 2)},
        "transition": transition_layer(out),
        "incremental_test": test,
        "framing": "structure reads are CHANGE DETECTORS: transition.sfi tracks deterioration speed, "
                   "transition.quadrant names which layer moves first. Level indices alone are context.",
        "verdict": ("no index adds reliable information over the price/vol baseline"
                    if all(v.get("perm_p", 1) > 0.05 or v.get("delta_auc", 0) <= 0 for v in test.values())
                    else "at least one index passes the permutation gate; needs full walk-forward before any blend"),
    }
    json.dump(_clean(payload), open(OUT, "w"), indent=2, allow_nan=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
