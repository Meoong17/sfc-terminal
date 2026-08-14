#!/usr/bin/env python3
"""
inflation_transmission_models.py
================================
Adjudikasi empiris Model A-D (1.docx §10): apakah inflation surprise -> BTC
adalah direct driver (H1) ATAU upstream shock yang ditransmisikan via
real yield / DXY / liquidity / risk-appetite (H2-H4).

METODOLOGI & BATASAN JUJUR
--------------------------
* ANALYST CONSENSUS TIDAK TERSEDIA. ForexFactory & Investing.com memblokir
  scraping (HTTP 403, Cloudflare) pada host ini. Bloomberg berbayar. Maka
  seri surprise memakai PROXY BERBASIS MODEL, bukan analyst consensus:

      Surprise_t = Actual_MoM_t - mean(Actual_MoM atas 12 bulan sebelumnya)

  Proxy ini menangkap "apakah bulan ini inflasi lebih panas/dingin dari
  tren baru-baru ini". Ini BUKAN Actual-Consensus. Hasil harus dibaca
  sebagai indikasi arah transmission, bukan estimasi efek dari surprise
  yang sesungguhnya (yang butuh seri consensus).

* FREKUENSI BULANAN. Analisis memakai return BTC bulanan. Ini tes kasar
  Model A-D (incremental explanatory power) seperti di 1.docx poin 10.
  Event-study intraday (08:30 ET, -30m..+1D) adalah lapisan kausal yang
  LEBIH halus dan TIDAK bisa dijalankan di sini (butuh release timestamp
  + consensus intraday). Dictata sebagai keterbatasan.

* TES KUNCI (1.docx §10): kalau koefisien inflation surprise signifikan di
  Model A tapi HILANG setelah RealYield/DXY/Liquidity/Risk dimasukkan ->
  inflasi BUKAN direct driver; ia cuma shock hulu yang lewat kanal
  moneter/finansial.

TIDAK ADA pengubahan scoring SFC. File ini eksplorasi/adjudikasi murni.
Sampel: BTC Binance Vision daily (2017-08 -> now), ~9 tahun / ~108 bulan.
"""
import os, json, sys
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FRED key from repo .env (loaded by pipeline)
def _load_fred_key():
    p = os.path.join(SFC_DIR, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv("FRED_API_KEY", "")

FRED_KEY = _load_fred_key()
BTC_DAILY = os.path.join(SFC_DIR, "data", "binance_vision_daily.json")

# ---- FRED pull ----------------------------------------------------------
def fred_series(sid):
    """Return DataFrame[date(ds) -> value] for a FRED series (ascending)."""
    r = requests.get(
        f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
        f"&api_key={FRED_KEY}&file_type=json&sort_order=asc",
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"FRED {sid} -> HTTP {r.status_code}")
    obs = [o for o in r.json().get("observations", []) if o["value"] != "."]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    return df.set_index("date")["value"].sort_index()

def monthly(df):
    """Aggregate daily FRED series to month-end (last value of month)."""
    return df.resample("ME").last().dropna()

def yoy(monthly_series):
    """YoY % for a level series indexed by month."""
    return (monthly_series / monthly_series.shift(12) - 1.0) * 100.0

# ---- Surprise proxy (model-based, honest) ------------------------------
def model_surprise(level_monthly):
    """Surprise_t = Actual_MoM_t - mean(MoM over prior 12 months), in %.
    Level series indexed by month. Returns standardized surprise."""
    mom = level_monthly.pct_change() * 100.0          # MoM %
    exp = mom.rolling(12, min_periods=6).mean().shift(1)  # expected = prior trend
    surprise = mom - exp
    surprise = (surprise - surprise.mean()) / surprise.std()
    return surprise

# ---- Load BTC -----------------------------------------------------------
def btc_monthly_return():
    d = json.load(open(BTC_DAILY))
    # dict date->{close,...}
    dates = sorted(d.keys())
    closes = pd.Series({pd.to_datetime(x): d[x]["close"] for x in dates})
    closes = closes.sort_index()
    # month-end close
    me = closes.resample("ME").last()
    ret = me.pct_change() * 100.0
    ret.name = "BTC"
    return ret

# ---- Model battery ------------------------------------------------------
def run_model(Y, X, label):
    """OLS with HC1 robust SE. Returns summary dict."""
    Xc = sm.add_constant(X)
    model = sm.OLS(Y, Xc).fit(cov_type="HC1")
    out = {
        "label": label,
        "n": int(model.nobs),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {},
    }
    for k in Xc.columns:
        out["params"][k] = {
            "beta": float(model.params[k]),
            "se": float(model.bse[k]),
            "p": float(model.pvalues[k]),
        }
    return out

def fmt_params(out):
    lines = []
    for k, v in out["params"].items():
        sig = "***" if v["p"] < 0.01 else "**" if v["p"] < 0.05 else "*" if v["p"] < 0.1 else ""
        lines.append(f"      {k:14s} b={v['beta']:+.4f}  se={v['se']:.4f}  p={v['p']:.3f} {sig}")
    return "\n".join(lines)

# ---- MAIN ---------------------------------------------------------------
def main():
    if not FRED_KEY:
        print("FATAL: no FRED key"); sys.exit(1)

    print("== Pull FRED monthly series ==")
    cpi_lvl = monthly(fred_series("CPIAUCSL"))     # CPI level
    ppi_lvl = monthly(fred_series("PPIACO"))       # PPI (final demand) level
    pce_lvl = monthly(fred_series("PCEPI"))        # PCE level
    realy   = monthly(fred_series("DFII10"))       # 10Y real yield
    dxy     = monthly(fred_series("DTWEXBGS"))     # trade-weighted dollar
    m2_lvl  = monthly(fred_series("M2SL"))         # M2 level
    vix     = monthly(fred_series("VIXCLS"))       # VIX (month-end)

    print("== Surprise proxies (model-based: MoM vs trailing-12m mean) ==")
    s_cpi = model_surprise(cpi_lvl); s_cpi.name = "CPI_surp"
    s_ppi = model_surprise(ppi_lvl); s_ppi.name = "PPI_surp"
    s_pce = model_surprise(pce_lvl); s_pce.name = "PCE_surp"

    btc = btc_monthly_return()
    print(f"BTC monthly sample: {btc.index.min().date()} .. {btc.index.max().date()}  n={btc.notna().sum()}")

    # Build panel (contemporaneous alignment, month m)
    df = pd.DataFrame({
        "BTC": btc,
        "CPI_surp": s_cpi, "PPI_surp": s_ppi, "PCE_surp": s_pce,
        "RealYield": realy, "DXY": dxy,
        "M2yoy": yoy(m2_lvl),
        "VIX": vix,
    })
    df = df.dropna(subset=["BTC"]).dropna(how="all")
    df["CPI_yoy"] = yoy(cpi_lvl)
    df["PPI_yoy"] = yoy(ppi_lvl)
    df["PCE_yoy"] = yoy(pce_lvl)

    # restrict to full panel rows
    panel = df.dropna().copy()
    panel = panel[panel.index >= "2017-09-01"]
    print(f"Panel n = {len(panel)}  span {panel.index.min().date()}..{panel.index.max().date()}")
    print()

    # ---- Model A-D (contemporaneous) ----
    print("=" * 74)
    print("MODEL A-D  (BTC bulanan, surprise proxy model-based, HC1 robust SE)")
    print("=" * 74)
    Y = panel["BTC"]
    XA = panel[["CPI_surp", "PPI_surp", "PCE_surp"]]
    XB = panel[["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY"]]
    XC = panel[["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy"]]
    XD = panel[["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy", "VIX"]]

    mA = run_model(Y, XA, "A: inflation-only")
    mB = run_model(Y, XB, "B: + monetary (realyield,dxy)")
    mC = run_model(Y, XC, "C: + liquidity (M2)")
    mD = run_model(Y, XD, "D: FULL (..+risk VIX)")
    for m in (mA, mB, mC, mD):
        print(f"\n[{m['label']}]  n={m['n']}  AdjR2={m['adj_r2']:.4f}  AIC={m['aic']:.1f}  BIC={m['bic']:.1f}")
        print(fmt_params(m))

    # Incremental R2 of adding controls
    def inc_r2(base_adj, full_adj):
        return full_adj - base_adj
    print("\n-- Incremental AdjR2 (menambahkan kanal) --")
    print(f"  A->B (monetary) : {inc_r2(mA['adj_r2'], mB['adj_r2']):+.4f}")
    print(f"  B->C (liquidity): {inc_r2(mB['adj_r2'], mC['adj_r2']):+.4f}")
    print(f"  C->D (risk)     : {inc_r2(mC['adj_r2'], mD['adj_r2']):+.4f}")
    print(f"  A->D (total)    : {inc_r2(mA['adj_r2'], mD['adj_r2']):+.4f}")

    # ---- KEY TEST: does inflation surprise survive controls? ----
    print("\n-- TES KUNCI: signifikansi inflation surprise di A vs D --")
    for v in ["CPI_surp", "PPI_surp", "PCE_surp"]:
        a = mA["params"][v]; d = mD["params"][v]
        print(f"  {v:9s}: ModelA b={a['beta']:+.4f} p={a['p']:.3f}  ->  ModelD b={d['beta']:+.4f} p={d['p']:.3f}")

    # ---- Lagged predictive check (no lookahead: surprise_m -> BTC_{m+1}) ----
    print("\n-- CEK PREDIKTIF 1-BULAN-LAG (surprise_m -> BTC_{m+1}; tanpa lookahead) --")
    lag = panel.shift(1).dropna()
    Ylag = lag["BTC"]
    XA_l = lag[["CPI_surp", "PPI_surp", "PCE_surp"]]
    XD_l = lag[["CPI_surp", "PPI_surp", "PCE_surp", "RealYield", "DXY", "M2yoy", "VIX"]]
    mAl = run_model(Ylag, XA_l, "A-lag")
    mDl = run_model(Ylag, XD_l, "D-lag")
    for m in (mAl, mDl):
        print(f"\n[{m['label']}] AdjR2={m['adj_r2']:.4f}")
        print(fmt_params(m))

    # ---- Era split (SFC era2 2018-21 vs era3 2022-26) ----
    print("\n-- ERA-SPLIT (Model A & D) --")
    era2 = panel[(panel.index >= "2018-01-01") & (panel.index < "2022-01-01")]
    era3 = panel[panel.index >= "2022-01-01"]
    for ename, e in [("era2 2018-2021", era2), ("era3 2022-2026", era3)]:
        print(f"\n  == {ename}  n={len(e)} ==")
        for mname, Xcols in [("A", ["CPI_surp","PPI_surp","PCE_surp"]),
                             ("D", ["CPI_surp","PPI_surp","PCE_surp","RealYield","DXY","M2yoy","VIX"])]:
            m = run_model(e["BTC"], e[Xcols], mname)
            inv = {k: v for k, v in m["params"].items()}
            sig = " ".join(f"{k}:b={v['beta']:+.3f}(p={v['p']:.2f})" for k, v in inv.items())
            print(f"    [{mname}] AdjR2={m['adj_r2']:.3f} | {sig}")

    # ---- Raw level check (inflation LEVEL vs surprise proxy) ----
    print("\n-- Cek: inflation LEVEL (YoY) vs surprise proxy di Model A --")
    dfL = df.dropna(subset=["BTC", "CPI_yoy", "PPI_yoy", "PCE_yoy"])
    Xlvl = dfL[["CPI_yoy", "PPI_yoy", "PCE_yoy"]]
    mlvl = run_model(dfL["BTC"], Xlvl, "A-level(YoY)")
    print(f"[A-level(YoY)] AdjR2={mlvl['adj_r2']:.4f}")
    print(fmt_params(mlvl))

    print("\nDONE.")

if __name__ == "__main__":
    main()
