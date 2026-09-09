#!/usr/bin/env python3
"""
SFC Divergence Test — "TrendStrength tinggi + MomentumOverlay bearish"
======================================================================
Empirical screen for the research idea in /home/ubuntu/C/1.docx: does the
divergence state (structural trend STRONG while the short/medium momentum
read is BEARISH) carry predictive information for forward BTC returns at
1D / 3D / 7D?

SCOPE (agreed Option A — clean price-only reconstruction, NOT a live replay
of data_sources.trend_strength.py / momentum_overlay.py):
  - Canonical daily OHLCV from data/binance_vision_daily.json (2017-08-17+).
  - TrendStrength is reconstructed with the SFC momentum_domain / alignment /
    structure semantics and the module's default weights (0.40 / 0.35 / 0.25)
    and 0-100 score; momentum/alignment/structure rebuilt from PRICE ONLY
    (no HMM / MTF-alignment / DFS-regime replay).
  - "MomentumOverlay bearish" is read as the trend_strength *momentum domain*
    falling in the module's BEARISH band (<= 0.45; MOM_BEAR), which under a
    regime-FOLLOW gate yields a bearish overlay.
  - Regime gate is NOT applied per-day (no daily sfc bucket series is kept),
    so this is a first screen of the raw divergence hypothesis, not the full
    regime-gated overlay. Flagged as caveat.

HYPOTHESIS STATE:
    DIVERGENCE  = TrendStrength >= 65 (STRONG)  AND  momentum_domain <= 0.45
    (bearish short-momentum inside a structurally strong uptrend).

CONTROL STATES:
    AGREE_BULL    = TrendStrength >= 65 AND momentum_domain >= 0.55
    AGREE_BEAR    = TrendStrength <  45 AND momentum_domain <= 0.45
    REVERSE_BULL  = TrendStrength <  45 AND momentum_domain >= 0.55
                     (weak/broken structure, momentum turning bullish)

STATISTICS (per state, per horizon 1/3/7d):
    n days, mean forward return, P(return>0), 90% bootstrap CI on the mean,
    and the gap vs ALL-days baseline with a two-tailed bootstrap CI on the
    difference (CI excluding zero => significant). Repeated on 3 calendar eras
    (2017-2020 / 2021-2023 / 2024-2026).

CAVEAT: overlapping forward-return labels inflate the independence of the
bootstrap CIs — treat these as a SCREEN only, never as a standalone reason to
blend a factor into scoring. A promising signal must still pass a purged /
era-split walk-forward before touching sfc_effective / signal / kelly.

Not blended anywhere. Display-only research.
"""
import json, os
import numpy as np
import pandas as pd

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SFC_DIR, "data", "binance_vision_daily.json")
OUT = os.path.join(SFC_DIR, "analysis", ".divergence_trend_momentum.json")

HORIZONS = [1, 3, 7]
MOM_BULL, MOM_BEAR = 0.55, 0.45
TS_STRONG, TS_WEAK = 65.0, 45.0
W = {"momentum": 0.40, "alignment": 0.35, "structure": 0.25}

ERA_EDGES = [
    ("era1_2017-2020", "2017-01-01", "2020-12-31"),
    ("era2_2021-2023", "2021-01-01", "2023-12-31"),
    ("era3_2024-2026", "2024-01-01", "2099-12-31"),
]


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    r = 100.0 - 100.0 / (1.0 + rs)
    return r


def rsi_to_strength(rsi):
    """Mirror data_sources/trend_strength.py `_rsi_to_strength` (0-1)."""
    if rsi is None or np.isnan(rsi):
        return 0.5
    if rsi < 30:
        return 0.30
    if rsi <= 50:
        return 0.35 + 0.30 * ((rsi - 30) / 20.0)
    if rsi <= 65:
        return 0.65 + 0.30 * ((rsi - 50) / 15.0)
    if rsi <= 80:
        return 0.95 - 0.35 * ((rsi - 65) / 15.0)
    return 0.50


def macd_to_strength(macd_frac, bb_frac):
    """Mirror `_macd_to_strength`: sign+size of MACD, wide-BB lowers confidence."""
    if macd_frac is None or np.isnan(macd_frac):
        return 0.5
    base = _clamp(0.5 + 2.0 * _clamp(macd_frac, -0.1, 0.1) / 0.1 * 0.5)
    if not np.isnan(bb_frac):
        base *= (1.0 - 0.25 * _clamp(bb_frac / 0.01, 0, 1))
    return _clamp(base)


def bootstrap_ci(a, n_boot=10000, alpha=0.10, seed=1):
    """90% bootstrap CI on the mean of array a (percentile method)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.10, seed=2):
    """Bootstrap CI on mean(a)-mean(b); excludes zero => significant (2-tailed)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diff = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def load_prices():
    raw = json.load(open(DATA))
    dates = sorted(raw.keys())
    rows = [(d, float(raw[d]["close"]), float(raw[d]["volume"])) for d in dates]
    df = pd.DataFrame(rows, columns=["date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df):
    c = df["close"]
    # --- RSI / MACD / BB / OBV (momentum domain inputs) ---
    df["rsi"] = rsi_wilder(c)
    macd = _ema(c, 12) - _ema(c, 26)
    macd_sig = _ema(macd, 9)
    macd_hist = macd - macd_sig
    df["macd_frac"] = (macd_hist / c).clip(-0.10, 0.10)  # normalised to module scale
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_width"] = ((mid + 2 * sd) - (mid - 2 * sd)) / mid  # ~0..+
    obv = (np.sign(c.diff()) * df["volume"]).fillna(0.0).cumsum()
    # obv normalised to ~[-1,1] via 60d rolling z of its diff
    obv_diff = obv.diff().rolling(60).sum()
    obv_std = obv_diff.rolling(120).std()
    df["obv_norm"] = (obv_diff / obv_std.replace(0, np.nan)).clip(-1.0, 1.0)

    # --- momentum domain (0-1), mirroring trend_strength momentum composition ---
    rsi_s = df["rsi"].apply(rsi_to_strength)
    macd_s = [macd_to_strength(m, b) for m, b in zip(df["macd_frac"], df["bb_width"])]
    dom_m = 0.6 * rsi_s + 0.4 * pd.Series(macd_s, index=df.index)
    dom_m = dom_m + df["obv_norm"].fillna(0.0) * 0.1
    df["momentum_domain"] = dom_m.clip(0.0, 1.0)

    # --- alignment domain (0-1): multi-horizon trend agreement ---
    hz = [3, 7, 15, 30, 60, 120]
    wts = np.array([0.05, 0.08, 0.12, 0.15, 0.25, 0.35])
    align = pd.Series(0.0, index=df.index)
    for k, w in zip(hz, wts):
        r = c.pct_change(k)
        align = align + (r > 0).astype(float) * w
    df["alignment_domain"] = align.clip(0.0, 1.0)

    # --- structure domain (0-1): long-MA regime + slope ---
    ma100 = c.rolling(100).mean()
    ma100_slope = ma100.pct_change(60)
    above = (c > ma100).astype(float)
    rising = (ma100_slope > 0).astype(float)
    # base: above & rising -> 0.9 ; above only -> 0.7 ; below & falling -> 0.2
    struct = 0.5 + 0.4 * above - 0.3 * (~(above.astype(bool))).astype(float)
    struct = struct.where(~np.isnan(ma100), 0.5)
    df["structure_domain"] = struct.clip(0.0, 1.0)

    # --- composite TrendStrength 0-100 (module default weights, redistribute) ---
    avail = ~df[["momentum_domain", "alignment_domain", "structure_domain"]].isna().all(axis=1)
    total_w = (W["momentum"] * (~df["momentum_domain"].isna())
               + W["alignment"] * (~df["alignment_domain"].isna())
               + W["structure"] * (~df["structure_domain"].isna()))
    num = (df["momentum_domain"].fillna(0) * W["momentum"]
           + df["alignment_domain"].fillna(0) * W["alignment"]
           + df["structure_domain"].fillna(0) * W["structure"])
    ts = (num / total_w.replace(0, np.nan)).clip(0, 1) * 100.0
    df["trend_strength"] = ts
    # Decoupled structural-only trend (alignment+structure re-weighted to sum 1)
    # so a "strong structural uptrend" can be measured independently of the
    # momentum domain — makes the divergence state populated enough to test.
    denom_s = W["alignment"] * (~df["alignment_domain"].isna()) \
        + W["structure"] * (~df["structure_domain"].isna())
    num_s = (df["alignment_domain"].fillna(0) * W["alignment"]
             + df["structure_domain"].fillna(0) * W["structure"])
    df["ts_struct"] = (num_s / denom_s.replace(0, np.nan)).clip(0, 1) * 100.0
    return df


def classify(df):
    # Original: divergence vs controls on the module composite TrendStrength
    # (momentum is 40% of it -> divergence state near-empty; kept for reference).
    ts = df["trend_strength"]
    md = df["momentum_domain"]
    state = pd.Series("OTHER", index=df.index)
    div = (ts >= TS_STRONG) & (md <= MOM_BEAR)
    ab = (ts >= TS_STRONG) & (md >= MOM_BULL)
    abear = (ts < TS_WEAK) & (md <= MOM_BEAR)
    rev = (ts < TS_WEAK) & (md >= MOM_BULL)
    state[div] = "DIVERGENCE"
    state[ab] = "AGREE_BULL"
    state[abear] = "AGREE_BEAR"
    state[rev] = "REVERSE_BULL"
    df["state"] = state

    # Decoupled: divergence between a STRONG STRUCTURAL uptrend (ts_struct,
    # momentum removed) and a BEARISH momentum read. This is the populated,
    # testable form of the 1.docx hypothesis ("structural up, momentum down").
    ts2 = df["ts_struct"]
    sd = pd.Series("OTHER", index=df.index)
    d_div = (ts2 >= TS_STRONG) & (md <= MOM_BEAR)
    d_ab = (ts2 >= TS_STRONG) & (md >= MOM_BULL)
    d_abear = (ts2 < TS_WEAK) & (md <= MOM_BEAR)
    d_rev = (ts2 < TS_WEAK) & (md >= MOM_BULL)
    sd[d_div] = "DIVERGENCE"
    sd[d_ab] = "AGREE_BULL"
    sd[d_abear] = "AGREE_BEAR"
    sd[d_rev] = "REVERSE_BULL"
    df["state_dec"] = sd
    return df


def analyze_subset(sub, baseline_close, baseline_ret):
    res = {}
    for h in HORIZONS:
        key = f"fwd_{h}d"
        if key not in sub:
            sub[key] = np.nan
        ret = sub[key].dropna()
        base = baseline_ret[f"fwd_{h}d"].dropna()
        n = len(ret)
        if n == 0:
            res[h] = {"n": 0}
            continue
        mean_r = float(ret.mean())
        ppos = float((ret > 0).mean())
        lo, hi = bootstrap_ci(ret.values)
        glo, ghi = bootstrap_diff_ci(ret.values, base.values)
        res[h] = {
            "n": int(n),
            "mean_fwd_return_pct": round(mean_r * 100, 3),
            "p_return_gt_0": round(ppos, 4),
            "ci90_mean_lo_pct": round(lo * 100, 3),
            "ci90_mean_hi_pct": round(hi * 100, 3),
            "gap_vs_baseline_pct": round((mean_r - base.mean()) * 100, 3),
            "gap_ci90_lo_pct": round(glo * 100, 3),
            "gap_ci90_hi_pct": round(ghi * 100, 3),
            "significant_2tail": bool((glo > 0) or (ghi < 0)),
            "baseline_n": int(len(base)),
            "baseline_mean_pct": round(float(base.mean()) * 100, 3),
        }
    return res


def main():
    df = load_prices()
    df = build_features(df)
    df = classify(df)
    # warmup: structure uses MA100+slope60 => ~160d; keep rows with trend_strength
    df = df.dropna(subset=["trend_strength"]).reset_index(drop=True)
    # forward returns on business-day close series
    close = df["close"]
    for h in HORIZONS:
        df[f"fwd_{h}d"] = close.shift(-h) / close - 1.0
    # baseline rets from all valid rows that have the horizon available
    base_ret = {}
    for h in HORIZONS:
        base_ret[f"fwd_{h}d"] = df[f"fwd_{h}d"].dropna()

    states = ["DIVERGENCE", "AGREE_BULL", "AGREE_BEAR", "REVERSE_BULL"]
    out = {"meta": {
        "note": "Option-A price-only proxy screen of 'TrendStrength high + "
                "MomentumOverlay bearish' divergence. NOT a live module replay; "
                "no per-day regime gate (no daily sfc bucket series). Overlapping "
                "labels inflate CI independence => screen only, not feature blend.",
        "data": DATA, "n_days": int(len(df)),
        "date_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
        "era_edges": [e[1:] for e in ERA_EDGES],
        "strong_ts": TS_STRONG, "weak_ts": TS_WEAK,
        "mom_bull": MOM_BULL, "mom_bear": MOM_BEAR,
        "weights": W,
    }, "prevalence": {}, "states": {}, "eras": {}}

    # full-sample prevalence
    for s in states:
        out["prevalence"][s] = {"n": int((df["state"] == s).sum()),
                                "pct": round(float((df["state"] == s).mean()) * 100, 2)}

    # full-sample stats per state (all-horizon baseline)
    for s in states:
        sub = df[df["state"] == s].copy()
        out["states"][s] = analyze_subset(sub, None, base_ret)

    # decoupled (structural-vs-momentum) analysis — the populated, testable form
    out["prevalence_dec"] = {}
    out["states_dec"] = {}
    for s in states:
        out["prevalence_dec"][s] = {"n": int((df["state_dec"] == s).sum()),
                                    "pct": round(float((df["state_dec"] == s).mean()) * 100, 2)}
        sub = df[df["state_dec"] == s].copy()
        out["states_dec"][s] = analyze_subset(sub, None, base_ret)
    out["eras_dec"] = {}

    # per-era
    for ename, d0, d1 in ERA_EDGES:
        m = (df["date"] >= d0) & (df["date"] <= d1)
        sub_all = df[m]
        base_e = {f"fwd_{h}d": sub_all[f"fwd_{h}d"].dropna() for h in HORIZONS}
        era_rec = {}
        for s in states:
            sub = sub_all[sub_all["state"] == s].copy()
            rec = analyze_subset(sub, None, base_e)
            rec["_n_days_total"] = int(len(sub_all))
            era_rec[s] = rec
        out["eras"][ename] = {"baseline": {f"fwd_{h}d": {
            "n": int(len(base_e[f"fwd_{h}d"])),
            "mean_fwd_pct": round(float(base_e[f"fwd_{h}d"].mean()) * 100, 3)}
            for h in HORIZONS}, "states": era_rec}
        era_rec_dec = {}
        for s in states:
            sub = sub_all[sub_all["state_dec"] == s].copy()
            era_rec_dec[s] = analyze_subset(sub, None, base_e)
        out["eras_dec"][ename] = era_rec_dec

    json.dump(out, open(OUT, "w"), indent=2, default=float)
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, indent=2, default=float))
