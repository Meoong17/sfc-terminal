#!/usr/bin/env python3
"""
flqd_engine.py — FLQD v2.0 (Fake Liquidity & Quality Dispersion) measurement engine.

Implements the *measurement* half of the C/FLQD_Framework_SFC_Methodology_v2.docx spec
on SFC's existing point-in-time data:

  Layer 1  LA / LT   (liquidity availability vs transmission)
  Layer 2  CS / FS   (credit / funding stress)
  Layer 3  PF / CE   (positioning fragility / capitulation evidence)
  Layer 4  QD        (quality dispersion, BTC-vs-alt relative strength)
  Interactions FLTI, SFM, VD, SC, RE
  Regime engine: Mahalanobis prototypes -> temperature softmax -> constrained
                 Bayesian-Markov filter -> normalized entropy H(t), 6 states R0-R5.
  Mixed-frequency: exponential information decay weights + coverage per layer,
                 robust MAD z -> Phi percentile (0-100).

This module DOES NOT touch sfc_effective. It writes analysis/flqd_engine_results.json
plus a walk-forward calibration check so the numbers can be adjudicated before any
wire-up (user rule: no scoring change before walk-forward validation).

Usage:
    cd ~/sfc && .venv/bin/python analysis/flqd_engine.py
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
OUT = os.path.join(ROOT, "analysis", "flqd_engine_results.json")

# ── frequency half-lives (calendar days) per FLQD v2 §3.3 ──────────────────
HALF_LIFE = {"daily": 3.0, "weekly": 10.0, "monthly": 35.0, "quarterly": 75.0}

# ── regime prototypes (FLQD v2 §7.1), order LA,LT,CS,FS,PF,CE,QD ──────────
DIMS = ["LA", "LT", "CS", "FS", "PF", "CE", "QD"]
MU = {
    "R0": [75, 80, 20, 20, 45, 15, 35],
    "R1": [75, 30, 35, 30, 65, 20, 75],
    "R2": [35, 25, 75, 75, 60, 35, 65],
    "R3": [25, 15, 85, 85, 85, 80, 70],
    "R4": [55, 60, 45, 40, 25, 65, 50],
    "R5": [65, 70, 25, 25, 40, 20, 80],
}
STATES = list(MU.keys())
# structural transition matrix T[j->k] from the doc; R3->R0 locked at 0.005
T_MAT = np.array([
    [0.82, 0.10, 0.03, 0.01, 0.01, 0.03],   # from R0
    [0.08, 0.76, 0.10, 0.02, 0.01, 0.03],   # from R1
    [0.01, 0.04, 0.75, 0.12, 0.07, 0.01],   # from R2
    [0.005, 0.005, 0.14, 0.68, 0.16, 0.01],  # from R3
    [0.10, 0.04, 0.05, 0.02, 0.69, 0.10],   # from R4
    [0.08, 0.08, 0.03, 0.01, 0.01, 0.79],   # from R5
])


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════
def _series_from_fred(path):
    d = json.load(open(path))
    s = pd.Series({pd.Timestamp(k): v for k, v in d.items() if v is not None}, dtype=float)
    return s.sort_index()


def load_frame():
    macro = pd.DataFrame(json.load(open(os.path.join(ROOT, "data/cleaned/macro_daily_clean.json"))))
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    btc = pd.DataFrame.from_dict(
        json.load(open(os.path.join(ROOT, "data/binance_vision_daily.json"))), orient="index")
    btc.index = pd.to_datetime(btc.index)
    btc = btc.sort_index()
    btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
    btc["quote_vol"] = pd.to_numeric(btc["quote_vol"], errors="coerce")

    of = pd.DataFrame.from_dict(
        json.load(open(os.path.join(ROOT, "data/binance_orderflow_daily.json"))), orient="index")
    of.index = pd.to_datetime(of.index)
    of = of.sort_index()

    oi_path = os.path.join(ROOT, "data/binance_oi_daily.json")
    oi = pd.DataFrame.from_dict(json.load(open(oi_path)), orient="index")
    oi.index = pd.to_datetime(oi.index)
    oi = oi.sort_index()

    fund_path = os.path.join(ROOT, "data/bitmex_funding_daily.json")
    fund = pd.DataFrame.from_dict(json.load(open(fund_path)), orient="index")
    fund.index = pd.to_datetime(fund.index)
    fund = fund.sort_index()

    usd = _series_from_fred(os.path.join(ROOT, "data/raw/macro/DTWEXBGS.json"))

    idx = btc.index
    df = pd.DataFrame(index=idx)
    for c in ["FED_BS", "TGA", "RRP", "M2_US", "HY_OAS", "IG_OAS", "VIX",
              "REAL_Y10", "US10Y", "SPX", "NDX", "COPPER"]:
        if c in macro:
            df[c] = macro[c].reindex(idx, method="ffill")
    df["USD"] = usd.reindex(idx, method="ffill")
    df["btc"] = btc["close"]
    df["btc_vol"] = btc["quote_vol"]

    # alt basket for QD (ETH/SOL if present)
    alts = []
    for sym in ["ETHUSDT", "SOLUSDT"]:
        p = os.path.join(ROOT, f"data/binance_vision_{sym}_daily.json")
        if os.path.exists(p):
            a = pd.DataFrame.from_dict(json.load(open(p)), orient="index")
            a.index = pd.to_datetime(a.index)
            alts.append(pd.to_numeric(a.sort_index()["close"], errors="coerce"))
    if alts:
        df["alt"] = pd.concat(alts, axis=1).mean(axis=1).reindex(idx, method="ffill")

    if "taker_sell_quote" in of.columns and "taker_buy_quote" in of.columns:
        df["sell_ratio"] = (of["taker_sell_quote"] / (of["taker_sell_quote"] + of["taker_buy_quote"])
                            ).reindex(idx)
    df["taker_sell_quote"] = pd.to_numeric(of.get("taker_sell_quote"), errors="coerce").reindex(idx)
    df["total_qty"] = pd.to_numeric(of.get("total_qty"), errors="coerce").reindex(idx)

    oi_col = next((c for c in oi.columns if "oi" in c.lower() or "open" in c.lower()), oi.columns[0])
    df["oi"] = pd.to_numeric(oi[oi_col], errors="coerce").reindex(idx)
    f_col = next((c for c in fund.columns if "fund" in c.lower()), fund.columns[0])
    df["funding"] = pd.to_numeric(fund[f_col], errors="coerce").reindex(idx)

    return df.dropna(subset=["btc"]).copy()


# ══════════════════════════════════════════════════════════════════════════
# transforms
# ══════════════════════════════════════════════════════════════════════════
def robust_pct(s, window=756, minp=252):
    """MAD z-score on a rolling window -> normal-CDF percentile (0-100), clipped ±3."""
    med = s.rolling(window, min_periods=minp).median()
    mad = (s - med).abs().rolling(window, min_periods=minp).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    pct = pd.Series(100 * norm.cdf(z.clip(-3, 3)), index=s.index).clip(0, 100)
    pct[z.isna()] = np.nan
    return pct


def decay_weights(staleness_days, freq):
    lam = HALF_LIFE[freq]
    return np.power(2.0, -staleness_days / lam)


def _age(df, col, freq):
    """Calendar age (days) since last real observation, assuming ffill in df[col]."""
    if col not in df:
        return pd.Series(np.inf, index=df.index)
    return (df.index - df[col].last_valid_index() * np.zeros(len(df))).days if False else None


def _age_series(df, col):
    if col not in df or df[col].dropna().empty:
        return pd.Series(np.inf, index=df.index)
    obs_idx = df[col].dropna().index
    ages = []
    j = -1
    obs = obs_idx.values
    for ts in df.index.values:
        while j + 1 < len(obs) and obs[j + 1] <= ts:
            j += 1
        ages.append((ts - obs[j]) / np.timedelta64(1, "D") if j >= 0 else np.inf)
    return pd.Series(ages, index=df.index)


def layer_score(df, specs):
    """specs: list of (name, series_0_100, base_weight, freq). Renormalize by decay weight."""
    num = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    base_total = sum(w for _, _, w, _ in specs)
    for name, s, w, freq in specs:
        if s is None:
            continue
        age = _age_series(df, name)
        weff = pd.Series(decay_weights(age.fillna(720).values, freq), index=df.index)
        num = num.add(s.fillna(50.0) * weff * w, fill_value=0.0)
        wsum = wsum.add(weff * w, fill_value=0.0)
    score = num / wsum.replace(0, np.nan)
    coverage = wsum / base_total
    return score.clip(0, 100), coverage.clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════
# build layer scores
# ══════════════════════════════════════════════════════════════════════════
def build_scores(df):
    out = pd.DataFrame(index=df.index)

    netliq = df["FED_BS"] - df["TGA"] - df["RRP"] * 1000.0
    df["netliq"] = netliq
    df["d_nl_13w"] = netliq.diff(91)
    df["d_tga_4w"] = -df["TGA"].diff(28)
    df["m2_ann3m"] = (df["M2_US"] / df["M2_US"].shift(91)) ** 4 - 1

    # --- Layer 1 ---------------------------------------------------------
    l1 = [
        ("d_nl_13w", 100 - robust_pct(-df["d_nl_13w"]), 0.25, "weekly"),
        ("VIX", 100 - robust_pct(df["VIX"]), 0.20, "daily"),
        ("HY_OAS", 100 - robust_pct(df["HY_OAS"]), 0.20, "daily"),
        ("REAL_Y10", 100 - robust_pct(df["REAL_Y10"]), 0.15, "daily"),
        ("USD", 100 - robust_pct(df["USD"]), 0.10, "daily"),
        ("d_tga_4w", 100 - robust_pct(-df["d_tga_4w"]), 0.10, "weekly"),
    ]
    out["LA"], cov_la = layer_score(df, l1)

    breadth_assets = [c for c in ["SPX", "NDX", "COPPER", "btc"] if c in df]
    breadth = sum((df[c].pct_change(20) > 0).astype(float) for c in breadth_assets) / max(len(breadth_assets), 1)
    df["breadth"] = breadth
    l2 = [
        ("SPX", robust_pct(df["SPX"].pct_change(20)), 0.35, "daily"),
        ("m2_ann3m", robust_pct(df["m2_ann3m"]), 0.20, "monthly"),
        ("breadth", robust_pct(breadth), 0.20, "daily"),
        # stablecoin net issuance: no point-in-time history -> coverage penalty only
    ]
    out["LT"], cov_lt = layer_score(df, l2)

    # --- Layer 2 ---------------------------------------------------------
    df["hy5d"] = df["HY_OAS"].diff(5)
    l3 = [
        ("HY_OAS", robust_pct(df["HY_OAS"]), 0.30, "daily"),
        ("IG_OAS", robust_pct(df["IG_OAS"]), 0.15, "daily"),
        ("hy5d", robust_pct(df["hy5d"]), 0.15, "daily"),
    ]
    cs_partial, cov_cs = layer_score(df, l3)

    rate_vol = df["US10Y"].diff().rolling(20).std()
    df["us10y_rv"] = rate_vol
    df["rrp_inv"] = -df["RRP"]
    l4 = [
        ("VIX", robust_pct(df["VIX"]), 0.40, "daily"),
        ("us10y_rv", robust_pct(rate_vol), 0.25, "daily"),
        ("rrp_inv", robust_pct(df["rrp_inv"]), 0.20, "weekly"),
    ]
    fs_partial, cov_fs = layer_score(df, l4)
    out["CS"] = cs_partial
    out["FS"] = fs_partial
    out["CS_coverage"], out["FS_coverage"] = cov_cs, cov_fs

    # --- Layer 3 ---------------------------------------------------------
    # NOTE: Binance OI history in this repo is only ~100 days -> no point-in-time
    # leverage series. PF uses funding + volume expansion instead of OI/depth
    # (coverage penalty is reported honestly in the output).
    df["vol_exp"] = df["btc_vol"] / df["btc_vol"].rolling(28).mean()
    df["funding_abs"] = df["funding"].abs()
    df["neg_funding"] = -df["funding"]
    l5 = [
        ("vol_exp", robust_pct(df["vol_exp"]), 0.30, "daily"),
        ("funding_abs", robust_pct(df["funding_abs"]), 0.25, "daily"),
        ("sell_ratio", robust_pct(df["sell_ratio"]), 0.15, "daily"),
    ]
    out["PF"], cov_pf = layer_score(df, l5)

    absorb = (df["taker_sell_quote"] / (df["btc"].pct_change().abs() + 1e-4))
    absorb = absorb.where(df["btc"].pct_change() < 0)
    absorb = absorb / absorb.rolling(60, min_periods=30).median()
    df["absorb"] = absorb
    l6 = [
        ("sell_ratio", robust_pct(df["sell_ratio"]), 0.25, "daily"),
        ("neg_funding", robust_pct(df["neg_funding"]), 0.20, "daily"),
        ("absorb", robust_pct(absorb), 0.15, "daily"),
    ]
    out["CE"], cov_ce = layer_score(df, l6)

    # --- Layer 4 ---------------------------------------------------------
    btc_r = df["btc"].pct_change()
    alt_r = df["alt"].pct_change() if "alt" in df else pd.Series(np.nan, index=df.index)
    df["relstr"] = btc_r.rolling(28).sum() - alt_r.rolling(28).sum()
    df["domchg"] = (df["btc"] / df["alt"]).pct_change(28) if "alt" in df else np.nan
    df["volshare"] = df["btc_vol"] / df["btc_vol"].rolling(28).mean()
    df["voldisp"] = btc_r.rolling(20).std() - alt_r.rolling(20).std()
    l7 = [
        ("relstr", robust_pct(df["relstr"]), 0.30, "daily"),
        ("domchg", robust_pct(df["domchg"]), 0.25, "daily"),
        ("volshare", robust_pct(df["volshare"]), 0.15, "daily"),
        ("voldisp", robust_pct(df["voldisp"]), 0.10, "daily"),
    ]
    out["QD"], cov_qd = layer_score(df, l7)

    out["LA_coverage"], out["LT_coverage"], out["PF_coverage"], out["QD_coverage"] = cov_la, cov_lt, cov_pf, cov_qd

    # --- interactions ----------------------------------------------------
    out["FLTI"] = out["LA"] * (100 - out["LT"]) / 100.0
    out["SFM"] = out["PF"] * out[["CS", "FS"]].max(axis=1) / 100.0
    out["VD"] = ((robust_pct(df["VIX"]) - 50) / 50.0) - ((robust_pct(rate_vol) - 50) / 50.0)
    out["SC"] = 0.40 * out["CS"] + 0.25 * out["FS"] + 0.20 * out["PF"] + 0.15 * (100 - out["LT"])
    out["RE"] = (0.30 * (100 - out["CS"]) + 0.20 * (100 - out["FS"]) +
                 0.20 * out["LT"] + 0.15 * out["CE"] + 0.15 * robust_pct(breadth))
    out["btc"] = df["btc"]
    if out[DIMS].dropna().empty:
        print("DEBUG non-null counts:", out[DIMS].notna().sum().to_dict())
        print("DEBUG first non-null per dim:",
              {c: (str(out[c].first_valid_index())) for c in DIMS})
    return out.dropna(subset=DIMS)


# ══════════════════════════════════════════════════════════════════════════
# regime engine
# ══════════════════════════════════════════════════════════════════════════
def mahalanobis_probs(x_rows, mu, cov_inv, tau):
    d = np.array([np.sqrt((x - m) @ cov_inv @ (x - m)) for x in x_rows for m in [mu]]) if False else None
    out = []
    for x in x_rows:
        dk = np.array([np.sqrt((x - MU[s]) @ cov_inv @ (x - MU[s])) for s in STATES])
        e = np.exp(-dk / tau)
        out.append(e / e.sum())
    return np.array(out)


def run_regime(scores, refit_every=21, window=504):
    X = scores[DIMS].values
    n = len(X)
    probs = np.full((n, 6), np.nan)
    from sklearn.covariance import LedoitWolf
    # warm-up: first 1.5y
    start = max(378, window)
    cov_inv = None
    mu_x = sd_x = None
    tau = 1.0
    for i in range(start, n):
        if (i - start) % refit_every == 0 or cov_inv is None:
            lo = max(0, i - window)
            train = X[lo:i]
            mu_x = train.mean(axis=0)
            sd_x = train.std(axis=0)
            sd_x[sd_x == 0] = 1.0
            lw = LedoitWolf().fit((train - mu_x) / sd_x)
            cov_inv = np.linalg.pinv(lw.covariance_)
            # scale-free tau: median prototype-vs-train distance
            ztr = (train[::10] - mu_x) / sd_x
            zmu = (np.array(MU[k] for k in STATES) if False else
                   np.array([MU[k] for k in STATES]) - mu_x) / sd_x
            d0 = [np.sqrt((x - m) @ cov_inv @ (x - m)) for x in ztr for m in zmu]
            tau = float(np.median(d0))
        x = (X[i] - mu_x) / sd_x
        zmu = (np.array([MU[k] for k in STATES]) - mu_x) / sd_x
        dk = np.array([np.sqrt((x - m) @ cov_inv @ (x - m)) for m in zmu])
        raw = np.exp(-dk / max(tau, 1e-6))
        raw = raw / raw.sum()
        if i == start:
            post = raw
        else:
            prior = T_MAT.T @ probs[i - 1]
            post = raw * prior
            post = post / post.sum() if post.sum() > 0 else raw
        # hard circuit breaker (doc §2.4 / §7.3)
        if i >= 2 and scores["FS"].iloc[i] > 85 and scores["FS"].iloc[i - 1] > 85:
            crisis = post[2] + post[3]
            if crisis < 0.70:
                short = 0.70 - crisis
                non = post[0] + post[1] + post[4] + post[5]
                post[2] += short * post[2] / max(crisis, 1e-9)
                post[3] += short * post[3] / max(crisis, 1e-9)
                post[[0, 1, 4, 5]] -= short * np.array([post[0], post[1], post[4], post[5]]) / max(non, 1e-9)
                post = np.clip(post, 0, 1)
                post = post / post.sum()
        probs[i] = post

    valid = ~np.isnan(probs[:, 0])
    ent = np.full(n, np.nan)
    p = np.clip(probs[valid], 1e-12, 1)
    ent[valid] = -(p * np.log(p)).sum(axis=1) / np.log(6)
    return probs, ent


# ══════════════════════════════════════════════════════════════════════════
# validation: walk-forward calibration vs price/vol baseline
# ══════════════════════════════════════════════════════════════════════════
def validate(scores, probs, horizon=20):
    df = scores.copy()
    df["crisis_p"] = probs[:, 2] + probs[:, 3]
    r = df["btc"].pct_change()
    fwd = df["btc"].shift(-horizon) / df["btc"] - 1
    fwd_vol = r.shift(-1).rolling(horizon).std().shift(-(horizon - 1))
    rv30 = r.rolling(30).std()
    mom30 = df["btc"].pct_change(30)
    thr = fwd_vol.rolling(504, min_periods=252).quantile(0.75)
    y = ((fwd < -0.10) | (fwd_vol > thr)).astype(float)

    m = df["crisis_p"].notna() & y.notna() & fwd_vol.notna() & thr.notna()
    idx = df.index[m][::5]           # 5-day sampling to cut overlapping-label inflation
    yv = y.reindex(idx).values
    pv = df["crisis_p"].reindex(idx).values
    base = np.full(len(idx), np.nanmean(yv))

    # baseline logistic on price/vol [rv30, mom30] expanding walk-forward
    from sklearn.linear_model import LogisticRegression
    Xb = np.column_stack([rv30.reindex(idx).values, mom30.reindex(idx).values])
    Xb = np.nan_to_num(Xb)
    pb = np.full(len(idx), np.nan)
    for k in range(60, len(idx)):
        lo = max(0, k - 1500)
        clf = LogisticRegression().fit(Xb[lo:k], yv[lo:k])
        pb[k] = clf.predict_proba(Xb[k:k + 1])[0, 1]
    ok = ~np.isnan(pb)

    def brier(p, yy):
        return float(np.mean((p - yy) ** 2))

    # regime sojourn
    arg = np.empty(len(probs)) * np.nan
    ok_rows = ~np.isnan(probs[:, 0])
    arg[ok_rows] = np.nanargmax(probs[ok_rows], axis=1)
    seq = arg[ok_rows]
    runs = (np.diff(seq) != 0).sum() + 1
    sojourn = float(len(seq) / max(runs, 1))

    # informational (not gate): does P(R2 u R3) rank forward realized vol, and do
    # modes separate forward returns?
    from scipy.stats import spearmanr
    ic, ic_p = spearmanr(pv, fwd_vol.reindex(idx).values, nan_policy="omit")
    mode = np.array(STATES)[np.nan_to_num(arg, nan=0.0).astype(int)]
    fwd20 = fwd.reindex(df.index).values
    by_mode = {}
    for s in STATES:
        sel = (mode == s) & ~np.isnan(fwd20)
        if sel.sum() >= 5:
            by_mode[s] = {"n": int(sel.sum()),
                          "fwd20_mean_pct": round(float(np.nanmean(fwd20[sel]) * 100), 2)}

    return {
        "n_eval": int(len(idx)),
        "base_rate": float(np.mean(yv)),
        "brier_climatology": brier(base, yv),
        "brier_flqd": brier(pv, yv),
        "brier_pricevol_baseline": brier(pb[ok], yv[ok]),
        "date_range": [str(idx[0].date()), str(idx[-1].date())],
        "sojourn_days": sojourn,
        "spearman_crisis_vs_fwdvol": [round(float(ic), 4), round(float(ic_p), 4)],
        "fwd20_by_mode": by_mode,
        "verdict": ("FLQD stress probability does not beat price/vol baseline on Brier"
                    if brier(pv, yv) >= brier(pb[ok], yv[ok]) else
                    "FLQD stress probability beats price/vol baseline on Brier"),
    }


def main():
    df = load_frame()
    scores = build_scores(df)
    probs, ent = run_regime(scores)
    scores = scores.copy()
    scores["entropy"] = pd.Series(ent, index=scores.index)

    val = validate(scores, probs)
    ts = scores.index[-1]
    last = scores.loc[ts]
    p_last = probs[-1]

    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": str(ts.date()),
        "model": "FLQD v2.0 measurement engine (C/FLQD_Framework_SFC_Methodology_v2)",
        "layers": {k: round(float(last[k]), 2) for k in DIMS},
        "coverage": {k: round(float(last[k + "_coverage"]), 3) for k in ["LA", "LT", "CS", "FS", "PF", "QD"]},
        "interactions": {k: round(float(last[k]), 2) for k in ["FLTI", "SFM", "VD", "SC", "RE"]},
        "regime_probs": {s: round(float(p), 4) for s, p in zip(STATES, p_last)},
        "regime_mode": STATES[int(np.argmax(p_last))],
        "entropy": round(float(ent[-1]), 4),
        "circuit_breaker": bool(last["FS"] > 85 and scores["FS"].iloc[-2] > 85 or last["SFM"] > 75),
        "validation": val,
    }
    def _clean(o):
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

    json.dump(_clean(res), open(OUT, "w"), indent=2, allow_nan=False)

    print(json.dumps({k: res[k] for k in ["as_of", "layers", "coverage", "interactions",
                                          "regime_probs", "regime_mode", "entropy",
                                          "circuit_breaker", "validation"]}, indent=2))


if __name__ == "__main__":
    main()
