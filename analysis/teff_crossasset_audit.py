#!/usr/bin/env python3
"""
teff_crossasset_audit.py — Audit kelayakan "Rolling Cross-Asset Temperature Metric"
(T_eff = mean(z^2), Maxwell-Boltzmann) sebagai LAYER TAMBAHAN model SFC.

Sumber ide: /home/ubuntu/C/Rolling Market Temperature Metric BTC (1).pdf
(Research note 26 Sep 2026, kalibrasi 24 Sep 2026).

ATURAN YANG DIPATUHI (standing rule SFC):
  - TIDAK menyentuh scoring (sfc_effective / composite / signal). Ini uji kelayakan.
  - Uji deskriptif vs prediktif DIPISAH.
  - Uji incumbent: kandidat harus mengalahkan apa yang SUDAH dipakai SFC
    (EWMA vol BTC / VIX), bukan strawman.
  - Era-stability: pooled + per-era (bukan menolak hanya karena era-flip).
  - Data kanonik: Binance Vision (BTC, PAXG) + FRED (macro).

Output: analysis/teff_audit_results.json + ringkasan stdout.
Pemakaian: .venv/bin/python analysis/teff_crossasset_audit.py
"""
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(SFC_DIR, "data", "raw", "macro")
MERGED = os.path.join(SFC_DIR, "data", "merged", "sfc_research_daily.json")
OUT = os.path.join(SFC_DIR, "analysis", "teff_audit_results.json")

LAMBDA = 0.94          # RiskMetrics daily decay (paper §2.5)
WARMUP = 60            # hari dibuang agar EWMA tidak look-ahead
ERAS = [("era1", "2017-08-01", "2020-12-31"),
        ("era2", "2021-01-01", "2023-12-31"),
        ("era3", "2024-01-01", "2026-12-31")]

# Cross-section paper §2.5 (9 aset). Yang tidak tersedia diganti/ ditandai.
SET_A = {  # inti, sampel penuh
    "BTC": "return", "SPX": "return", "VIX": "return",
    "US10Y": "bp", "WTI": "return", "DXY": "return", "USDJPY": "return",
}
SET_B = dict(SET_A, GOLD="return", COPPER="return")  # + logam (silver = gap)


# ── data ─────────────────────────────────────────────────────────────────────
def fred_series(sid):
    p = os.path.join(RAW, f"{sid}.json")
    return pd.Series(json.load(open(p))) if os.path.exists(p) else None


def load_local_frame():
    rows = json.load(open(MERGED))
    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    out = pd.DataFrame(index=df.index)
    out["BTC"] = df["BTC_close"]
    out["SPX"] = df["SPX"]
    out["VIX"] = df["VIX"]
    out["US10Y"] = df["US10Y"]
    out["WTI"] = df["WTI"]
    out["COPPER"] = df.get("COPPER")
    dxy = fred_series("DTWEXBGS")
    jpy = fred_series("DEXJPUS")
    out["DXY"] = dxy.reindex(out.index)
    out["USDJPY"] = jpy.reindex(out.index)
    paxg = json.load(open(os.path.join(SFC_DIR, "data", "binance_vision_PAXGUSDT_daily.json")))
    out["GOLD"] = pd.Series({k: v["close"] for k, v in paxg.items()}).reindex(out.index)
    return out.sort_index()


def load_live_extension(days=120):
    """Perpanjang frame lokal dgn observasi FRED live + klines Binance live
    (agar tanggal referensi paper 24 Sep 2026 masuk sampel)."""
    import requests
    key = [l.strip().split("=", 1)[1] for l in open(os.path.join(SFC_DIR, ".env"))
           if l.startswith("FRED_API_KEY=")][0]
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    cols = {}
    for sid, name in [("SP500", "SPX"), ("VIXCLS", "VIX"), ("DGS10", "US10Y"),
                      ("DCOILWTICO", "WTI"), ("DTWEXBGS", "DXY"), ("DEXJPUS", "USDJPY")]:
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": key, "file_type": "json",
                                     "observation_start": start}, timeout=40)
            r.raise_for_status()
            cols[name] = pd.Series({o["date"]: float(o["value"]) for o in r.json()["observations"]
                                    if o["value"] not in (".", "")})
        except Exception as e:                                   # noqa: BLE001
            print(f"[warn] FRED live {sid}: {type(e).__name__}", file=sys.stderr)
    for sym, name in [("BTCUSDT", "BTC"), ("PAXGUSDT", "GOLD")]:
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": sym, "interval": "1d", "limit": days}, timeout=40)
            r.raise_for_status()
            cols[name] = pd.Series({
                datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"): float(k[4])
                for k in r.json()})
        except Exception as e:                                   # noqa: BLE001
            print(f"[warn] binance live {sym}: {type(e).__name__}", file=sys.stderr)
    return pd.DataFrame(cols)


def build_frame():
    local = load_local_frame()
    live = load_live_extension()
    if not live.empty:
        common = [c for c in local.columns if c in live.columns]
        merged = pd.concat([local, live[common]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = local
    # grid sesi: hanya hari di mana aset macro BENAR-BENAR bergerak (hari bursa AS)
    macro_core = ["SPX", "VIX", "US10Y", "WTI", "DXY", "USDJPY"]
    grid = merged[macro_core + ["BTC"]].dropna().index
    return merged.loc[grid]


# ── T_eff ────────────────────────────────────────────────────────────────────
def ewma_z(x, lam=LAMBDA, warmup=WARMUP):
    """z_t = x_t / sigma_{t-1}, sigma EWMA(lambda) — vol DILAG supaya tidak look-ahead."""
    x = x.dropna()
    v = x.iloc[:warmup].std() ** 2
    z = pd.Series(index=x.index, dtype=float)
    sig = pd.Series(index=x.index, dtype=float)
    for i, (d, val) in enumerate(x.items()):
        sig[d] = np.sqrt(v)
        z[d] = val / np.sqrt(v) if v > 0 else np.nan
        v = lam * v + (1 - lam) * val ** 2
    return z.iloc[warmup:], sig.iloc[warmup:]


def deltas(df, cols):
    d = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        kind = (SET_B if c in SET_B else SET_A)[c]
        s = df[c].astype(float)
        d[c] = s.diff() * 100.0 if kind == "bp" else np.log(s).diff()
    return d


def teff_frame(df, cols, drop=()):
    cols = [c for c in cols if c not in drop and c in df.columns]
    d = deltas(df, cols)
    zs = {}
    for c in cols:
        z, _ = ewma_z(d[c])
        zs[c] = z
    Z = pd.DataFrame(zs).dropna()
    z2 = Z ** 2
    out = pd.DataFrame(index=Z.index)
    out["n"] = len(cols)
    out["sum_z2"] = z2.sum(axis=1)
    out["teff"] = z2.mean(axis=1)
    out["btc_share"] = z2["BTC"] / out["sum_z2"] if "BTC" in z2 else np.nan
    out["rates_share"] = z2[[c for c in ("US10Y", "DXY", "USDJPY") if c in z2]].sum(axis=1) / out["sum_z2"]
    # n_eff koreksi korelasi (paper §2.4), rata-rata |r| pairwise rolling 60d
    r60 = Z.rolling(60).corr()
    def avg_pair(dt):
        try:
            m = r60.loc[dt].to_numpy(dtype=float)
        except KeyError:
            return np.nan
        n = m.shape[0]
        if n < 2:
            return np.nan
        return (np.nansum(m) - np.nansum(np.diag(m))) / (n * (n - 1))
    out["r_avg"] = [avg_pair(dt) for dt in out.index]
    out["n_eff"] = out["n"] / (1 + (out["n"] - 1) * out["r_avg"])
    out["teff_corr"] = out["sum_z2"] / out["n_eff"]
    out["regime"] = np.where(out["teff"] >= 2.0, "Hot",
                             np.where(out["teff"] >= 1.0, "Warm", "Cold"))
    out["regime_corr"] = np.where(out["teff_corr"] >= 2.0, "Hot",
                                  np.where(out["teff_corr"] >= 1.0, "Warm", "Cold"))
    return out, Z


# ── statistik ────────────────────────────────────────────────────────────────
def ols(y, X, names=None):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 30:
        return {"n": int(len(d)), "note": "terlalu sedikit"}
    Xm = sm.add_constant(d[X.columns])
    r = sm.OLS(d["y"], Xm).fit(cov_type="HAC", cov_kwds={"maxlags": max(2, int(len(d) ** (1 / 3)))})
    res = {"n": int(len(d)), "r2": round(float(r.rsquared), 4)}
    names = names or list(X.columns)
    for nm, c in zip(names, X.columns):
        res[f"beta_{nm}"] = round(float(r.params[c]), 4)
        res[f"t_{nm}"] = round(float(r.tvalues[c]), 2)
        res[f"p_{nm}"] = round(float(r.pvalues[c]), 4)
    return res


def rv(ret, w):
    return ret.rolling(w).std() * np.sqrt(252)


def main():
    df = build_frame()
    res = {"generated": datetime.utcnow().isoformat(),
           "frame": {"start": df.index[0], "end": df.index[-1], "n_sessions": int(len(df))}}

    btc_ret = np.log(df["BTC"].astype(float)).diff()
    rv20 = rv(btc_ret, 20)

    sets = {"A_7aset": teff_frame(df, list(SET_A)), "B_9aset": teff_frame(df, list(SET_B))}
    # varian tanpa BTC (uji tautologi)
    sets["A_exBTC"] = teff_frame(df, list(SET_A), drop=("BTC",))

    res["sets"] = {}
    for k, (t, Z) in sets.items():
        res["sets"][k] = {
            "n": int(len(t)), "n_assets": int(t["n"].iloc[0]),
            "start": t.index[0], "end": t.index[-1],
            "teff_mean": round(float(t["teff"].mean()), 3),
            "teff_median": round(float(t["teff"].median()), 3),
            "teff_p95": round(float(t["teff"].quantile(0.95)), 2),
            "share_cold": round(float((t["regime"] == "Cold").mean()), 3),
            "share_warm": round(float((t["regime"] == "Warm").mean()), 3),
            "share_hot": round(float((t["regime"] == "Hot").mean()), 3),
            "teff_corr_mean": round(float(t["teff_corr"].mean()), 3),
            "r_avg_mean": round(float(t["r_avg"].mean()), 3),
            "n_eff_mean": round(float(t["n_eff"].mean()), 2),
            "regime_disagree_raw_vs_corrPct": round(float((t["regime"] != t["regime_corr"]).mean()), 3),
        }

    A, ZA = sets["A_7aset"]
    B, ZB = sets["B_9aset"]
    AX, ZX = sets["A_exBTC"]

    # ── T1. replikasi tanggal referensi paper (24 Sep 2026) ──
    def at(t, dt):
        s = t.loc[:(dt)] if dt in t.index else t.loc[:dt]
        if s.empty:
            return None
        row = s.iloc[-1]
        return {"date": s.index[-1], "teff": round(float(row["teff"]), 3),
                "regime": row["regime"], "teff_corr": round(float(row["teff_corr"]), 3),
                "btc_share": round(float(row["btc_share"]), 4)}
    res["paper_reference_date_2026_09_24"] = {
        "paper_teff": 0.45, "paper_regime": "Cold", "paper_btc_share": 0.0035,
        "replicated_A": at(A, "2026-09-24"), "replicated_B": at(B, "2026-09-24"),
        "note": "paper pakai implied vol (VIX/DVOL) + asumsi vol statis utk 6 aset; "
                "di sini semua aset pakai EWMA(0.94) realised, vol dilag, grid hari bursa AS.",
    }

    # ── T2. episode krisis yang diketahui ──
    episodes = {"2018-02-05": "VIX spike / volmageddon", "2020-03-12": "COVID crash",
                "2021-05-19": "crypto deleveraging", "2022-06-13": "CPI shock / risk-off",
                "2023-03-13": "SVB", "2024-08-05": "yen carry unwind",
                "2025-04-07": "tariff shock (jika ada data)"}
    res["episodes"] = {}
    for dt, lbl in episodes.items():
        e = at(A, dt)
        if e:
            e["label"] = lbl
            e["btc_ret_1d_pct"] = round(float(btc_ret.loc[:dt].iloc[-1] * 100), 2)
            res["episodes"][dt] = e

    # ── T3. tautologi / dekomposisi energi BTC ──
    res["tautology"] = {
        "btc_share_mean": round(float(A["btc_share"].mean()), 4),
        "btc_share_median": round(float(A["btc_share"].median()), 4),
        "corr_teff_btc_rv20": round(float(A["teff"].corr(rv20, method="spearman")), 3),
        "corr_teff_exbtc_btc_rv20": round(float(AX["teff"].corr(rv20, method="spearman")), 3),
        "corr_teff_btc_rv20_pearson": round(float(A["teff"].corr(rv20)), 3),
        "corr_teff_exbtc_btc_rv20_pearson": round(float(AX["teff"].corr(rv20)), 3),
    }

    # ── T4. deskriptif vs prediktif BTC ──
    X = pd.DataFrame({
        "teff_lag1": A["teff"].shift(1),
        "teff_corr_lag1": A["teff_corr"].shift(1),
        "teff_exbtc_lag1": AX["teff"].shift(1).reindex(A.index),
        "btc_share_lag1": A["btc_share"].shift(1),
        "rates_share_lag1": A["rates_share"].shift(1),
    })
    targets = {}
    for h in (1, 5, 20):
        targets[f"fwd_ret_{h}d"] = (np.log(df["BTC"].astype(float)).shift(-h) -
                                    np.log(df["BTC"].astype(float)))
    targets["fwd_rv20"] = rv20.shift(-20)
    targets["fwd_dlog_rv20"] = np.log(rv20.shift(-20)) - np.log(rv20)
    targets["fwd_maxdd20"] = (np.log(df["BTC"].astype(float)).rolling(20).min().shift(-20) -
                              np.log(df["BTC"].astype(float)))
    res["predictive"] = {}
    for tn, y in targets.items():
        res["predictive"][tn] = {c: ols(y, X[[c]]) for c in X.columns}
    # incumbent: EWMA vol BTC saja, lalu + teff (ΔR²)
    ew_vol = ewma_z(btc_ret)[1]
    inc = pd.DataFrame({"ewma_vol": ew_vol.shift(1).reindex(A.index)})
    res["incumbent"] = {}
    for tn in ("fwd_dlog_rv20", "fwd_ret_20d", "fwd_maxdd20"):
        y = targets[tn]
        base = ols(y, inc)
        comb = ols(y, pd.concat([inc, X[["teff_lag1"]]], axis=1))
        res["incumbent"][tn] = {"baseline_ewma_vol_only": base,
                                "plus_teff": comb,
                                "delta_r2": round(comb.get("r2", np.nan) - base.get("r2", np.nan), 4)}
    # VIX sebagai incumbent kedua
    vix = df["VIX"].astype(float).reindex(A.index)
    base_vix = ols(targets["fwd_dlog_rv20"], pd.DataFrame({"vix": np.log(vix).shift(1)}))
    comb_vix = ols(targets["fwd_dlog_rv20"], pd.DataFrame({"vix": np.log(vix).shift(1),
                                                           "teff": X["teff_lag1"]}))
    res["incumbent"]["fwd_dlog_rv20_vs_VIX"] = {
        "baseline_vix_only": base_vix, "plus_teff": comb_vix,
        "delta_r2": round(comb_vix.get("r2", np.nan) - base_vix.get("r2", np.nan), 4)}

    # ── T5. era-stability (pooled + per era) ──
    res["era_stability"] = {}
    for tn in ("fwd_dlog_rv20", "fwd_ret_20d", "fwd_maxdd20"):
        y = targets[tn].reindex(A.index)
        res["era_stability"][tn] = {"pooled": ols(y, X[["teff_lag1"]])}
        for nm, s, e in ERAS:
            m = (A.index >= s) & (A.index <= e)
            sub = ols(y[m], X.loc[m, ["teff_lag1"]])
            res["era_stability"][tn][nm] = sub

    # ── T6. trigger yang paper usulkan: BTC energy share ──
    res["btc_share_trigger"] = {
        "corr_share_rates_share": round(float(A["btc_share"].corr(A["rates_share"])), 3),
        "fwd_dlog_rv20": ols(targets["fwd_dlog_rv20"], X[["btc_share_lag1"]]),
        "fwd_ret_20d": ols(targets["fwd_ret_20d"], X[["btc_share_lag1"]]),
    }

    # ── T7. downside conditioning: P(fwd 20d return < -10%) per regime ──
    y = targets["fwd_ret_20d"]
    base_p = float((y < -0.10).mean())
    res["downside_by_regime"] = {"base_rate": round(base_p, 4)}
    for rg in ("Cold", "Warm", "Hot"):
        mm = A["regime"] == rg
        yy = y.reindex(A.index)[mm].dropna()
        res["downside_by_regime"][rg] = {
            "n": int(len(yy)), "p_drop10": round(float((yy < -0.10).mean()), 4),
            "mean_fwd_ret_pct": round(float(yy.mean() * 100), 2),

        }
    # ambang Warm-hold (paper §4.1): teff>=1 tiga sesi berturut
    warm_hold = ((A["teff"] >= 1.0).rolling(3).sum() == 3)
    yy = y.reindex(A.index)[warm_hold].dropna()
    res["watch_condition_warm_hold3"] = {
        "n_days": int(warm_hold.sum()),
        "p_drop10": round(float((yy < -0.10).mean()), 4) if len(yy) else None,
        "mean_fwd_ret_pct": round(float(yy.mean() * 100), 2) if len(yy) else None,
        "fwd_dlog_rv20_beta": ols(targets["fwd_dlog_rv20"], pd.DataFrame({"warmH": warm_hold.astype(float)})),
    }

    # ── T8. sensitivitas estimator vol: apakah ambang Cold/Warm/Hot invariant? ──
    cols_a = list(SET_A)

    def teff_raw(df, cols, kind, win=30):
        d = deltas(df, cols)[cols]
        zs = {}
        for c in cols:
            if kind == "ewma":
                z, _ = ewma_z(d[c])
            else:
                z = d[c] / d[c].rolling(win).std().shift(1)
            zs[c] = z
        Z = pd.DataFrame(zs).dropna()
        return (Z ** 2).mean(axis=1)

    base = A["teff"]
    variants = {}
    for nm, kind, win in [("ewma94", "ewma", 0), ("rw30", "rw", 30), ("rw60", "rw", 60),
                          ("rw252", "rw", 252)]:
        s = teff_raw(df, cols_a, kind, win) if win else base
        v = {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
             "share_hot": round(float((s >= 2).mean()), 3),
             "share_warm": round(float(((s >= 1) & (s < 2)).mean()), 3),
             "share_cold": round(float((s < 1).mean()), 3)}
        if win:
            reg_b = pd.Series(np.where(base >= 2, "Hot", np.where(base >= 1, "Warm", "Cold")),
                              index=base.index)
            reg_s = pd.Series(np.where(s >= 2, "Hot", np.where(s >= 1, "Warm", "Cold")), index=s.index)
            j = pd.concat([reg_b, reg_s], axis=1, keys=["a", "b"]).dropna()
            v["spearman_vs_ewma94"] = round(float(pd.concat([base, s], axis=1).dropna().corr(method="spearman").iloc[0, 1]), 3)
            v["label_disagreement_frac"] = round(float((j["a"] != j["b"]).mean()), 3)
        variants[nm] = v
    # benchmark chi: E[S] vs sqrt(n)
    S = np.sqrt(A["sum_z2"])
    res["vol_estimator_sensitivity"] = variants
    res["chi_benchmark"] = {
        "n_assets": int(A["n"].iloc[0]),
        "sqrt_n": round(float(np.sqrt(A["n"].iloc[0])), 3),
        "E_chi_n_approx": round(float(np.sqrt(A["n"].iloc[0]) - 0.5 / np.sqrt(A["n"].iloc[0])), 3),
        "obs_mean_S": round(float(S.mean()), 3),
        "obs_median_S": round(float(S.median()), 3),
        "frac_days_S_above_sqrt_n": round(float((S > np.sqrt(A["n"].iloc[0])).mean()), 3),
    }

    # ── T9. koreksi n_eff (paper §2.4): dari z terstandardisasi vs dari raw return ──
    d_raw = deltas(df, cols_a)[cols_a].dropna()
    ZA_al = ZA.reindex(d_raw.index)

    def avg_pair(frame, win=60):
        rr = frame.rolling(win).corr()
        out = []
        for dt in frame.index:
            try:
                m = rr.loc[dt].to_numpy(dtype=float)
            except KeyError:
                out.append(np.nan)
                continue
            n = m.shape[0]
            out.append((np.nansum(m) - np.nansum(np.diag(m))) / (n * (n - 1)) if n > 1 else np.nan)
        return pd.Series(out, index=frame.index)

    r_raw = avg_pair(d_raw)
    r_z = avg_pair(ZA_al)
    n = len(cols_a)
    neff_raw = n / (1 + (n - 1) * r_raw)
    sum_z2 = (ZA_al ** 2).sum(axis=1)
    teff_neff_raw = sum_z2 / neff_raw
    res["n_eff_correction"] = {
        "n_assets": n,
        "r_avg_raw_returns_mean": round(float(r_raw.mean()), 3),
        "r_avg_standardized_z_mean": round(float(r_z.mean()), 3),
        "n_eff_from_raw_mean": round(float(neff_raw.mean()), 2),
        "n_eff_from_z_mean": round(float(A["n_eff"].mean()), 2),
        "teff_neff_raw_mean": round(float(teff_neff_raw.mean()), 3),
        "teff_neff_raw_share_hot": round(float((teff_neff_raw >= 2).mean()), 3),
        "teff_raw_mean_same_window": round(float((sum_z2 / n).mean()), 3),
        "teff_raw_share_hot_same_window": round(float(((sum_z2 / n) >= 2).mean()), 3),
    }

    # ── T10. elastisitas BTC ke yield (klaim paper §5.2) ──
    dy_bp = (df["US10Y"].astype(float).diff() * 100).reindex(A.index)
    btc_ret_al = btc_ret.reindex(A.index)
    Xr = pd.DataFrame({"dy_bp": dy_bp, "dy_bp_lag1": dy_bp.shift(1)})
    res["btc_rate_elasticity"] = {
        "contemporaneous": ols(btc_ret_al, Xr[["dy_bp"]]),
        "lagged_1d": ols(btc_ret_al, Xr[["dy_bp_lag1"]]),
        "cov_type": "HAC",
    }
    for nm, s, e in ERAS:
        m = (A.index >= s) & (A.index <= e)
        res["btc_rate_elasticity"][nm] = ols(btc_ret_al[m], Xr.loc[m, ["dy_bp"]])
    beta = res["btc_rate_elasticity"]["contemporaneous"].get("beta_dy_bp")
    res["btc_rate_elasticity"]["implied_pct_for_plus17bp"] = (
        round(beta * 17 * 100, 2) if beta is not None else None)
    res["btc_rate_elasticity"]["paper_claim_range_pct"] = [-1.36, -0.51]
    # replikasi jendela 3 sesi paper
    try:
        b = df["BTC"].astype(float)
        w = b.loc[:"2026-09-24"].iloc[-3:]
        res["btc_rate_elasticity"]["paper_window_btc"] = {
            "dates": list(w.index), "ret_3d_pct": round(float((w.iloc[-1] / w.iloc[0] - 1) * 100), 2)}
    except Exception:                                              # noqa: BLE001
        pass

    # ── T11. deskriptor state BTC per regime (bukan prediksi) ──
    ma200 = df["BTC"].astype(float).rolling(200).mean().reindex(A.index)
    px = df["BTC"].astype(float).reindex(A.index)
    dd = (px / px.rolling(252).max() - 1)
    st = {}
    for rg in ("Cold", "Warm", "Hot"):
        m = A["regime"] == rg
        st[rg] = {"n": int(m.sum()),
                  "pct_below_ma200": round(float((px[m] < ma200[m]).mean()), 3),
                  "median_drawdown_from_1y_high_pct": round(float(dd[m].median() * 100), 1),
                  "median_btc_rv20_ann_pct": round(float(rv20.reindex(A.index)[m].median() * 100), 1)}
    res["state_descriptor_by_regime"] = st

    json.dump(res, open(OUT, "w"), indent=1)

    # ── ringkasan stdout ──
    print(json.dumps({k: res[k] for k in ("frame", "sets")}, indent=1))
    print("\n== referensi paper 24 Sep 2026 ==")
    print(json.dumps(res["paper_reference_date_2026_09_24"], indent=1))
    print("\n== episode ==")
    for k, v in res["episodes"].items():
        print(k, v["label"], "| teff", v["teff"], v["regime"], "| btc_1d%", v["btc_ret_1d_pct"])
    print("\n== tautologi ==")
    print(json.dumps(res["tautology"], indent=1))
    print("\n== incumbent (ΔR2 teff di atas baseline) ==")
    for k, v in res["incumbent"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk == "delta_r2"}, flush=True)
    print("\n== era stability: fwd_dlog_rv20 ==")
    print(json.dumps(res["era_stability"]["fwd_dlog_rv20"], indent=1))
    print("\n== downside by regime ==")
    print(json.dumps(res["downside_by_regime"], indent=1))
    print("\n== watch condition (teff>=1 3 sesi) ==")
    print(json.dumps(res["watch_condition_warm_hold3"], indent=1))
    print("\n== sensitivitas estimator vol ==")
    print(json.dumps(res["vol_estimator_sensitivity"], indent=1))
    print("\n== chi benchmark ==", json.dumps(res["chi_benchmark"]))
    print("\n== n_eff correction ==")
    print(json.dumps(res["n_eff_correction"], indent=1))
    print("\n== elastisitas BTC ke yield ==")
    print(json.dumps(res["btc_rate_elasticity"], indent=1))
    print("\n== state descriptor per regime ==")
    print(json.dumps(res["state_descriptor_by_regime"], indent=1))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
