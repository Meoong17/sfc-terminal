#!/usr/bin/env python3
"""
institutional_flow_incremental.py
=================================
Pertanyaan: apakah INSTITUTIONAL FLOW (ETF flow) memiliki incremental explanatory
power atas return BTC setelah VIX (risk appetite) dan M2 (likuiditas) dikontrol?

Metode (lanjutan inflation_assumptions_tests.py):
  - Panel BTC return ~ VIX + M2_yoy (baseline) lalu + ETF_flow (nested), uji partial-F.
  - Newey-West robust SE (autocorrelation lag-1 diketahui ringan pada return bulanan).
  - Dua frekuensi:
      * BULANAN (sejajar analisis inflasi, 2024-01..2026-08, n~31) — n kecil.
      * HARIAN (2024-01..2026-08, n~950) — lebih power, return noisy.
  - ETF_flow = sum(etfs.values()) harian (rekonstruksi dari cache .etf_cache.json).
  - UJI kontemporer DAN lagged (flow_t -> return_{t+1}) utk pisahkan arah kausal
    (flow dan return kontemporer saling memicu; lagged memproksi prediksi).

BATASAN JUJUR: ETF flow hanya ada sejak Jan 2024 (spot ETF launch) -> ~2.6 th.
Bulanan n~31 = data-too-short utk verdict tegas; harian memberi power tapi return
harian sangat noisy. Seri cumulative (stock) I(1) -> tak dipakai sbg regressor return
(risk spurious). TIDAK ada perubahan scoring.

Usage: cd ~/sfc && .venv/bin/python analysis/institutional_flow_incremental.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inflation_transmission_models import fred_series, monthly, yoy, FRED_KEY

SIG = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def etf_daily_flow():
    c = json.load(open(os.path.join(SFC_DIR, ".etf_cache.json")))
    rows = []
    for f in c["flows"]:
        try:
            total_btc = sum(v for k, v in (f.get("etfs") or {}).items() if isinstance(v, (int, float)))
            rows.append({"date": pd.to_datetime(f["date"]), "etf_flow_btc": total_btc})
        except Exception:
            continue
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df["etf_flow_btc"]


def btc_daily_close():
    d = json.load(open(os.path.join(SFC_DIR, "data", "binance_vision_daily.json")))
    s = pd.Series({pd.to_datetime(x): d[x]["close"] for x in d}, name="close")
    return s.sort_index()


def build_daily():
    vix = fred_series("VIXCLS")          # daily
    m2 = monthly(fred_series("M2SL"))    # monthly
    m2yoy = yoy(m2)                      # monthly %
    btc = btc_daily_close()
    btc_ret = btc.pct_change() * 100.0
    btc_ret.name = "BTC"
    etf = etf_daily_flow()
    df = pd.DataFrame({"BTC": btc_ret, "VIX": vix, "M2yoy": m2yoy, "etf_flow_btc": etf})
    df["M2yoy"] = df["M2yoy"].ffill()    # forward-fill monthly -> daily
    df["VIX"] = df["VIX"].ffill()
    df = df.dropna(subset=["BTC"])
    df = df[(df.index >= "2024-01-01") & (df.index <= "2026-08-31")]
    return df


def build_monthly(daily):
    mo = daily.resample("ME").agg({
        "BTC": lambda x: x.sum() if x.notna().sum() > 0 else np.nan,   # cumulative monthly ret (approx)
        "VIX": "last",
        "M2yoy": "last",
        "etf_flow_btc": "sum",
    })
    # proper monthly return from month-end close instead
    d = json.load(open(os.path.join(SFC_DIR, "data", "binance_vision_daily.json")))
    closes = pd.Series({pd.to_datetime(x): d[x]["close"] for x in d}).sort_index()
    me = closes.resample("ME").last()
    ret = me.pct_change() * 100.0
    mo["BTC"] = ret.reindex(mo.index)
    mo["VIX"] = daily["VIX"].resample("ME").last()
    mo["M2yoy"] = daily["M2yoy"].resample("ME").last()
    mo["etf_flow_btc"] = daily["etf_flow_btc"].resample("ME").sum()
    return mo.dropna()


def incr_test(Y, X_base, X_full, label):
    """Nested incremental test: does X_full (adds regressors to X_base) add
    explanatory power? Classical partial-F + HAC (Newey-West) t-test on the
    added regressor(s) — robust to autocorrelation/heteroskedasticity."""
    def fit(X):
        Xc = sm.add_constant(X)
        return sm.OLS(Y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    m0 = fit(X_base)
    m1 = fit(X_full)
    added = [c for c in X_full.columns if c not in X_base.columns]
    # classical partial-F (from RSS; valid if homoskedastic)
    rss0 = float(np.sum(m0.resid ** 2))
    rss1 = float(np.sum(m1.resid ** 2))
    q = len(added); n = int(m1.nobs); k = len(X_full.columns) + 1
    F_cl = ((rss0 - rss1) / q) / (rss1 / (n - k))
    p_cl = 1 - __import__("scipy").stats.f.cdf(F_cl, q, n - k)
    # robust Wald test on added regressor(s) — valid with HAC covariance
    if len(added) == 1:
        wp = m1.pvalues[added[0]]   # HAC t-test p-value on the single added regressor
    else:
        try:
            wt = m1.wald_test(" = ".join([f"{c}=0" for c in added]))
            wp = float(wt.pvalue)
        except Exception:
            wp = np.nan
    print(f"\n  [{label}]  n={n}")
    print(f"    baseline AdjR2={m0.rsquared_adj:.4f}  +{added} AdjR2={m1.rsquared_adj:.4f}")
    print(f"    classical partial-F={F_cl:.3f} p={p_cl:.4f} {SIG(p_cl)} | "
          f"HAC-test p={wp:.4f} {SIG(wp)}  "
          f"{'INCREMENTAL SIGNIFIKAN' if (p_cl < 0.05 and (wp == wp and wp < 0.05)) else 'TIDAK robust'}")
    for c in X_full.columns:
        print(f"      {c:12s} b={m1.params[c]:+.4f} se={m1.bse[c]:.4f} p={m1.pvalues[c]:.3f} {SIG(m1.pvalues[c])}")
    return p_cl, wp


def main():
    if not FRED_KEY:
        print("FATAL: no FRED key"); sys.exit(1)
    daily = build_daily()
    monthly = build_monthly(daily)
    print(f"ETF daily: {daily['etf_flow_btc'].notna().sum()} obs, "
          f"{daily.index.min().date()}..{daily.index.max().date()}")
    print(f"Daily panel n={len(daily.dropna(subset=['etf_flow_btc','VIX']))}  "
          f"Monthly panel n={len(monthly)}")

    print("\n" + "=" * 78)
    print("BULANAN (2024-01..2026-08) — sejajar analisis inflasi, n kecil")
    print("=" * 78)
    mo = monthly
    Y = mo["BTC"]
    # contemporaneous
    Xb = mo[["VIX", "M2yoy"]]
    Xf = mo[["VIX", "M2yoy", "etf_flow_btc"]]
    incr_test(Y, Xb, Xf, "Bulanan KONTEMPORER: +ETF_flow")
    # lagged (flow_{t-1} -> ret_t)
    mo_lag = mo.dropna(subset=["VIX", "M2yoy"]).copy()
    mo_lag["etf_flow_btc_l1"] = mo_lag["etf_flow_btc"].shift(1)
    mo_lag = mo_lag.dropna()
    Yl = mo_lag["BTC"]
    incr_test(Yl, mo_lag[["VIX", "M2yoy"]], mo_lag[["VIX", "M2yoy", "etf_flow_btc_l1"]],
              "Bulanan LAGGED(flow_l1): +ETF_flow")

    print("\n" + "=" * 78)
    print("HARIAN (2024-01..2026-08) — lebih power, return noisy")
    print("=" * 78)
    dd = daily.dropna(subset=["BTC", "VIX", "M2yoy"])
    # limit to rows where etf_flow available for the full test
    dfull = dd.dropna(subset=["etf_flow_btc"]).copy()
    dfull_l = dfull.copy()
    dfull_l["etf_flow_btc_l1"] = dfull_l["etf_flow_btc"].shift(1)
    dfull_l["BTC"] = dfull_l["BTC"].shift(-1)   # return_{t+1} vs flow_t
    dfull_l = dfull_l.dropna()
    Yd = dfull["BTC"]
    incr_test(Yd, dfull[["VIX", "M2yoy"]], dfull[["VIX", "M2yoy", "etf_flow_btc"]],
              "Harian KONTEMPORER: +ETF_flow")
    incr_test(dfull_l["BTC"], dfull_l[["VIX", "M2yoy"]],
              dfull_l[["VIX", "M2yoy", "etf_flow_btc_l1"]],
              "Harian LAGGED(flow_t->ret_t+1): +ETF_flow")

    print("\nDONE.")


if __name__ == "__main__":
    main()
