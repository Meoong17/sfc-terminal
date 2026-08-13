#!/usr/bin/env python3
"""
intraday_2012_2017.py — Intraday microstructure + volatility clustering, era 2012-2017
======================================================================================
Loads the Kaggle Bitstamp 1-min OHLCV for 2012-2017 and analyzes:

  MICROSTRUCTURE
   - Intraday seasonality: hourly (UTC) mean |return| and volume profile -> is there
     a recurring time-of-day structure (vol/volume curve)?
   - Realized volatility (1-min) vs close-to-close vol -> how much of daily vol is
     captured intraday; is RV serially correlated (clustering)?
   - Volume-price link: daily corr(volume, |return|) and leverage (vol of vol).

  VOLATILITY CLUSTERING
   - ACF of 1-min |returns| and squared returns at many lags (long-memory?).
   - GARCH(1,1) fit (arch) on daily returns: persistence alpha+beta.
   - Vol-of-vol: daily RV autocorrelation.

  FUNDING-PROXY
   - Assess whether ANY funding proxy is extractable from spot OHLCV (honest: mostly no;
     volume asymmetry needs tick direction, funding needs perp futures). Report what
     IS and ISN'T possible.
"""
import zipfile, os, sys, json, math, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ZIP = "/tmp/kg.zip"
OUT = "/home/ubuntu/sfc/.intraday_2012_2017.json"

def load():
    z = zipfile.ZipFile(ZIP)
    df = pd.read_csv(z.open('btcusd_1-min_data.csv'))
    df['dt'] = pd.to_datetime(df.Timestamp, unit='s')
    df = df.set_index('dt')
    era = df.loc['2012-01-01':'2017-12-31 23:59'].copy()
    # 1-min return (log)
    era['ret'] = np.log(era['Close']).diff()
    era['absret'] = era['ret'].abs()
    return era

era = load()
print(f"loaded {len(era)} 1-min rows, {era.index.min()} .. {era.index.max()}")

res = {"era": "2012-01..2017-12", "n_1min": int(len(era)),
       "btc_source": "Kaggle Bitstamp spot OHLCV"}

# ================= MICROSTRUCTURE =================
# --- 1. Intraday hourly seasonality (UTC) ---
era['hour'] = era.index.hour
hourly = era.groupby('hour').agg(
    mean_absret=('absret', 'mean'),
    med_absret=('absret', 'median'),
    mean_vol=('Volume', 'mean'),
    med_vol=('Volume', 'median'),
)
# normalize to 100 baseline
hourly['vol_index'] = hourly['mean_vol'] / hourly['mean_vol'].mean() * 100
hourly['volat_index'] = hourly['mean_absret'] / hourly['mean_absret'].mean() * 100
print("\n=== HOURLY PROFILE (UTC) ===")
print(hourly[['volat_index','vol_index']].round(1).to_string())
res['hourly_profile'] = {int(h): {"volat_idx": round(float(r['volat_index']),1),
                                  "vol_idx": round(float(r['vol_index']),1)}
                         for h, r in hourly.iterrows()}
# peak/trough hours
print(f"\n  Peak vol hour: {int(hourly.volat_index.idxmax())} UTC ({hourly.volat_index.max():.0f} idx)")
print(f"  Trough vol hour: {int(hourly.volat_index.idxmin())} UTC ({hourly.volat_index.min():.0f} idx)")

# --- 2. Realized volatility daily + clustering ---
daily = era.groupby(era.index.date).agg(
    rv=('ret', lambda s: np.sum(s**2)),   # realized variance (1-min)
    cc_ret=('Close', lambda s: np.log(s.iloc[-1]/s.iloc[0]) if s.iloc[0]>0 else 0),
    vol=('Volume', 'sum'),
    n=('ret', 'count'),
).dropna()
daily['rv_ann'] = np.sqrt(daily['rv']*1440)*100  # annualized % vol from 1-min RV
# close-to-close daily vol
cc = np.log(era['Close'].resample('D').last()).diff().dropna()*100
print(f"\n=== REALIZED VOL ===")
print(f"  Median daily RV-ann vol: {daily['rv_ann'].median():.1f}% | CC vol std: {cc.std():.1f}%")
print(f"  Mean realized variance (1-min): {daily['rv'].mean():.6f}")
res['realized_vol'] = {
    "median_rv_ann_pct": round(float(daily['rv_ann'].median()),1),
    "cc_daily_vol_std_pct": round(float(cc.std()),1),
    "rv_autocorr_lag1": round(float(np.corrcoef(daily['rv'][1:], daily['rv'][:-1])[0,1]),3),
    "rv_autocorr_lag5": round(float(np.corrcoef(daily['rv'][5:], daily['rv'][:-5])[0,1]),3),
}
print(f"  RV daily autocorr lag1={res['realized_vol']['rv_autocorr_lag1']}  lag5={res['realized_vol']['rv_autocorr_lag5']}")

# --- 3. Volume-price link ---
d_vol = daily['vol']; d_abscc = daily['cc_ret'].abs()
vc = np.corrcoef(d_vol, d_abscc)[0,1]
print(f"\n=== VOLUME-PRICE ===")
print(f"  corr(daily volume, |cc return|) = {vc:.3f}")
res['volume_price'] = {"corr_daily_vol_absret": round(float(vc),3)}

# ================= VOLATILITY CLUSTERING =================
# --- 4. ACF of 1-min |return| ---
def acf(x, maxlag):
    x = x - x.mean()
    n = len(x)
    a0 = np.sum(x*x)
    out = []
    for l in range(1, maxlag+1):
        out.append(np.sum(x[l:]*x[:-l])/a0)
    return out
# subsample to first 200k for ACF speed
sub = era['absret'].dropna().values[:200000]
lags_1m = [1,2,5,10,30,60,144,288,720,1440]
acf_1m = {f"{l}min": round(v,3) for l,v in zip(lags_1m, acf(sub, max(lags_1m)))}
print("\n=== ACF of 1-min |return| ===")
for l in lags_1m: print(f"  lag {l}min: {acf_1m[f'{l}min']}")
res['acf_1min_absret'] = acf_1m

# --- 5. GARCH(1,1) on daily CC returns ---
try:
    from arch import arch_model
    am = arch_model(cc*100, vol='GARCH', p=1, q=1, mean='Constant', rescale=False).fit(disp='off')
    omega = float(am.params.get('omega', np.nan))
    alpha = float(am.params.get('alpha[1]', np.nan))
    beta  = float(am.params.get('beta[1]', np.nan))
    pers = alpha+beta
    print("\n=== GARCH(1,1) daily ===")
    print(f"  alpha={alpha:.3f}  beta={beta:.3f}  persistence={pers:.3f}  omega={omega:.4f}")
    res['garch'] = {"alpha": round(alpha,3), "beta": round(beta,3),
                    "persistence": round(float(pers),3),
                    "half_life_days": round(float(math.log(0.5)/math.log(pers)),1) if 0<pers<1 else None}
    if 0<pers<1:
        print(f"  half-life: {res['garch']['half_life_days']} days")
except Exception as e:
    print("  GARCH err:", e); res['garch'] = {"error": str(e)}

# --- 6. daily RV autocorrelation (vol-of-vol clustering) ---
rv_acf1 = res['realized_vol']['rv_autocorr_lag1']
print(f"\n=== VOL-OF-VOL ===")
print(f"  daily RV autocorr lag1 = {rv_acf1} (>0.3 = strong clustering)")

# ================= FUNDING-PROXY =================
# Honest assessment from spot OHLCV
print("\n=== FUNDING-PROXY FEASIBILITY ===")
proxy_note = (
    "NOT FEASIBLE from this dataset. (1) Funding rate is a perp-futures instrument "
    "and is NOT in spot OHLCV. (2) Buy/sell initiated volume asymmetry (the standard "
    "short-term funding/crowding proxy) requires tick-level trade direction, which "
    "aggregate 1-min OHLCV cannot recover (only total volume per bar). (3) A basis "
    "proxy (futures-vs-spot) needs a futures price series, absent here. What IS "
    "possible: volume surge + |return| asymmetry as a rough 'crowding' signal, but "
    "that is not funding and would be an unvalidated heuristic."
)
print("  " + proxy_note.replace("\n","\n  "))
res['funding_proxy'] = {"feasible": False, "reason": proxy_note}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nSaved -> {OUT}")
print("DONE.")
