#!/usr/bin/env python3
"""
SFC v2.1 REALTIME COLLECTOR — No LLM Version
5-minute auto-update for GitHub Pages
Multi-source news aggregator (20+ free sources)
"""

import json, os, sys, subprocess, math, requests, time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Import news aggregator
sys.path.insert(0, os.path.dirname(__file__))
try:
    from news_sources import get_news_stress_v2, detect_black_swan_v2
except ImportError:
    def get_news_stress_v2(*a, **k): return 0.0, [], 0.0, [], {}
    def detect_black_swan_v2(*a, **k): return 0.0, None, "NONE"

# ============================================================
# 1. LIVE DATA COLLECTION
# ============================================================

def get_btc():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_market_cap=true", timeout=10)
        d = r.json()["bitcoin"]
        return d["usd"], d.get("usd_24h_change", 0), d.get("usd_market_cap", 0)
    except:
        return None, None, None

def get_ath():
    """Fetch dynamic ATH from CoinGecko — fallback to hardcoded 126272"""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&community_data=false&developer_data=false", timeout=10)
        d = r.json()
        ath = d.get("market_data", {}).get("ath", {}).get("usd")
        ath_date = d.get("market_data", {}).get("ath_date", {}).get("usd", "")
        if ath and ath > 0:
            print(f"[SFC] ATH fetched: ${ath:,.0f} on {ath_date}", file=sys.stderr)
            return int(ath), ath_date
    except Exception as e:
        print(f"[SFC] ATH fetch failed, using fallback: {e}", file=sys.stderr)
    FALLBACK_ATH = 126272
    return FALLBACK_ATH, ""

def get_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except:
        return None, None

def get_dom():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        return r.json()["data"]["market_cap_percentage"]["btc"]
    except:
        return None

def get_dvol():
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        ivs = [(o.get("mark_iv", 0), o.get("open_interest", 0)) for o in opts if o.get("mark_iv")]
        if not ivs:
            return None
        oi = sum(x[1] for x in ivs)
        if oi == 0:
            return None
        return round(sum(x[0] * x[1] for x in ivs) / oi, 2)
    except:
        return None

def get_m2_data():
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return None, None, None
    try:
        r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=M2SL&api_key={key}&file_type=json&sort_order=desc&limit=13", timeout=15)
        obs = r.json().get("observations", [])
        if len(obs) >= 13:
            current = float(obs[0]["value"])
            year_ago = float(obs[12]["value"])
            return current, round((current - year_ago) / year_ago * 100, 2), None
    except:
        pass
    return None, None, None

def get_dxy():
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&api_key={key}&file_type=json&sort_order=desc&limit=2", timeout=15)
        obs = r.json().get("observations", [])
        if obs and obs[0]["value"] != ".":
            return round(float(obs[0]["value"]), 2)
    except:
        pass
    return None

def get_put_call_ratio():
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        puts_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-P"))
        calls_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-C"))
        puts_vol = sum(o.get("volume", 0) for o in opts if o.get("instrument_name", "").endswith("-P"))
        calls_vol = sum(o.get("volume", 0) for o in opts if o.get("instrument_name", "").endswith("-C"))
        oi_ratio = round(puts_oi / calls_oi, 2) if calls_oi else None
        vol_ratio = round(puts_vol / calls_vol, 2) if calls_vol else None
        return oi_ratio, vol_ratio
    except:
        return None, None

def get_btc_ohlcv_daily(days=30):
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily", timeout=15)
        d = r.json()
        prices = d.get("prices", [])
        volumes = d.get("total_volumes", [])
        return [{"time": prices[i][0], "close": prices[i][1], "volume": volumes[i][1] if i < len(volumes) else 0} for i in range(len(prices))]
    except:
        return []

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_sopr_proxy(closes_7d, closes_30d, btc_spot):
    if not closes_7d or not closes_30d or not btc_spot:
        return None, "UNKNOWN", 0.5
    avg_7d = sum(closes_7d) / len(closes_7d)
    avg_30d = sum(closes_30d) / len(closes_30d)
    weighted_cost = 0.40 * avg_7d + 0.60 * avg_30d
    sopr_proxy = round(btc_spot / weighted_cost, 4) if weighted_cost > 0 else None
    if sopr_proxy is None:
        return None, "UNKNOWN", 0.5
    if sopr_proxy < 0.93:
        signal, score = "EXTREME_CAPITULATION", 0.95
    elif sopr_proxy < 0.97:
        signal, score = "CAPITULATION", 0.80
    elif sopr_proxy < 0.995:
        signal, score = "MILD_DISTRESS", 0.65
    elif sopr_proxy < 1.005:
        signal, score = "BREAKEVEN", 0.50
    elif sopr_proxy < 1.03:
        signal, score = "MILD_PROFIT", 0.40
    elif sopr_proxy < 1.08:
        signal, score = "DISTRIBUTION", 0.25
    else:
        signal, score = "EXTREME_DISTRIBUTION", 0.10
    return sopr_proxy, signal, score

# ============================================================
# 2. SFC v2.1 CALCULATION (NO LLM - Rule Based)
# ============================================================

def score_factors_from_market(btc, btc_24h, dom, dvol, fng, pc_oi, m2_yoy, dxy):
    """Score 5 factors from market data. Range -3 to +3"""
    factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}
    
    # Lt (Liquidity) - based on M2 and BTC momentum
    if m2_yoy:
        if m2_yoy < 2: factors["Lt"] -= 1.5
        elif m2_yoy < 4: factors["Lt"] -= 0.5
        elif m2_yoy > 7: factors["Lt"] += 1.0
        elif m2_yoy > 5: factors["Lt"] += 0.5
    if btc_24h:
        if btc_24h < -10: factors["Lt"] -= 1.0
        elif btc_24h < -5: factors["Lt"] -= 0.5
        elif btc_24h > 10: factors["Lt"] += 1.0
        elif btc_24h > 5: factors["Lt"] += 0.5
    
    # St (Structural) - based on dominance and put/call
    if dom:
        if dom > 65: factors["St"] -= 1.0
        elif dom < 45: factors["St"] += 0.5
    if pc_oi:
        if pc_oi > 1.2: factors["St"] -= 1.5
        elif pc_oi > 1.0: factors["St"] -= 0.5
        elif pc_oi < 0.6: factors["St"] += 0.5
    
    # Rt (Sentiment) - based on Fear & Greed
    if fng:
        if fng <= 15: factors["Rt"] = -2.0
        elif fng <= 30: factors["Rt"] = -1.0
        elif fng <= 45: factors["Rt"] = -0.5
        elif fng <= 55: factors["Rt"] = 0.0
        elif fng <= 75: factors["Rt"] = 1.0
        else: factors["Rt"] = 2.0
    
    # Ft (Systemic) - based on DVOL
    if dvol:
        if dvol >= 120: factors["Ft"] = -2.5
        elif dvol >= 100: factors["Ft"] = -1.5
        elif dvol >= 80: factors["Ft"] = -0.5
        elif dvol >= 60: factors["Ft"] = 0.0
        elif dvol < 40: factors["Ft"] = 1.0
        else: factors["Ft"] = 0.5
    
    # Sc (External) - based on DXY
    if dxy:
        if dxy > 110: factors["Sc"] = -1.5
        elif dxy > 105: factors["Sc"] = -0.5
        elif dxy < 95: factors["Sc"] = 1.0
    if dom and dom > 70:
        factors["Sc"] -= 0.5
    
    # Clamp all factors to [-3, 3]
    for k in factors:
        factors[k] = max(-3.0, min(3.0, factors[k]))
    
    return factors

def calculate_sfc_ensemble(factors):
    """Calculate 6-method ensemble from factors"""
    norm = {k: v/6 for k, v in factors.items()}
    z_score = sum(norm.values())
    
    # M1: KLR
    ns_r = {"Lt":0.35, "St":0.50, "Rt":0.40, "Ft":0.25, "Sc":0.80}
    w = {k:1/v for k,v in ns_r.items()}
    sig = sum((1.0 if norm[k]<-2 else 0.7 if norm[k]<-1 else 0.3 if norm[k]<0 else 0) * w[k] for k in factors)
    p_klr = sig / sum(w.values())
    
    # M2: Logit
    zc = [-1.0, -2.0, -3.0, -4.0, -8.0]
    pc = [0.08, 0.20, 0.55, 0.75, 0.95]
    yc = [math.log(p/(1-p)) for p in pc]
    n_z = len(zc)
    zm = sum(zc)/n_z
    ym = sum(yc)/n_z
    b1 = sum((zc[i]-zm)*(yc[i]-ym) for i in range(n_z)) / sum((z-zm)**2 for z in zc)
    b0 = ym - b1*zm
    z_l = b0 + b1*z_score
    p_logit = 1/(1+math.exp(-z_l))
    
    # M3: Bayesian
    prior = 0.04
    odds = prior/(1-prior)
    bayes_mult = [2.5, 2.0, 2.0, 3.0, 1.5]
    for i, k in enumerate(factors):
        if norm[k] < -0.5:
            odds *= bayes_mult[i]
    p_bayes = odds/(1+odds)
    
    # M4: ECB Composite
    w_ad = {"Lt":0.25, "St":0.20, "Rt":0.20, "Ft":0.30, "Sc":0.05}
    ewc = sum(w_ad[k] * abs(norm[k]) for k in factors)
    p_ewc = ewc/3.0
    
    # M5: Quantile Regression
    qr_anchors = [(-8.0, 0.95), (-5.0, 0.75), (-3.0, 0.50), (-1.5, 0.25), (-0.5, 0.10), (0.5, 0.04), (2.0, 0.01)]
    def quantile_stress(z):
        anchors = sorted(qr_anchors, key=lambda x: x[0])
        if z <= anchors[0][0]: return anchors[0][1]
        if z >= anchors[-1][0]: return anchors[-1][1]
        for i in range(len(anchors)-1):
            z0, p0 = anchors[i]
            z1, p1 = anchors[i+1]
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                return p0 + t * (p1 - p0)
        return 0.04
    p_quantile = quantile_stress(z_score)
    
    # M6: Composite Regime Score
    vals = list(norm.values())
    n = len(vals)
    extreme_count = sum(1 for v in vals if v < -1.0)
    severe_count = sum(1 for v in vals if v < -2.0)
    p_extremity = (extreme_count * 0.15 + severe_count * 0.20)
    mean_v = sum(vals) / n
    variance = sum((v - mean_v)**2 for v in vals) / n
    coherence_bonus = 0.10 * (1.0 - variance) if mean_v < -0.5 and variance < 1.0 else 0.0
    ft_val = norm.get("Ft", 0)
    lt_val = norm.get("Lt", 0)
    tail_contribution = (0.15 if ft_val < -1.5 else 0.0) + (0.10 if lt_val < -1.5 else 0.0)
    p_baseline = max(0.0, min((-mean_v) * 0.12, 0.50))
    p_regime = min(p_baseline + p_extremity + coherence_bonus + tail_contribution, 0.99)
    p_regime = max(p_regime, 0.01)
    
    # Ensemble
    p_ens = 0.20*p_klr + 0.25*p_logit + 0.20*p_bayes + 0.10*p_ewc + 0.15*p_quantile + 0.10*p_regime
    
    # Method agreement
    method_probs = [p_klr, p_logit, p_bayes, p_ewc, p_quantile, p_regime]
    mean_p = sum(method_probs) / len(method_probs)
    variance_p = sum((p - mean_p)**2 for p in method_probs) / len(method_probs)
    std_p = math.sqrt(variance_p)
    method_agreement = max(0.0, min(1.0, 1.0 - (std_p / 0.20)))
    
    zone = "CRITICAL" if p_ens > 0.75 else "HIGH" if p_ens > 0.5 else "ELEVATED" if p_ens > 0.25 else "NORMAL"
    
    return p_ens * 100, zone, factors, norm, p_klr, p_logit, p_bayes, p_ewc, p_quantile * 100, p_regime * 100, method_agreement

# ============================================================
# 3. MAIN
# ============================================================

DVOL_BASELINE = 50

def compute_floor(dvol, sfc_pct, ath):
    if sfc_pct is not None:
        dv = 0.10 + (sfc_pct / 100.0) * 0.75
    else:
        dv = 0.80
    phi = DVOL_BASELINE / max(dvol or DVOL_BASELINE, DVOL_BASELINE)
    buf = int(ath * (1 - dv))
    total = int(ath * (1 - dv) * phi)
    return buf, total, round(dv, 3), round(phi, 3)

def determine_state(dvol, sfc_pct, btc=None, floor_total=None):
    if dvol is None or sfc_pct is None:
        return "NORMAL OSCILLATION", "Standard allocation"
    if dvol >= 100 and btc and floor_total and btc <= floor_total * 1.05:
        return "⚠ TURNING POINT", "BUY / REVERSAL FLOOR"
    if dvol < 40 and sfc_pct > 30:
        return "DISSIPATIVE SLIDE", "STAY IN CASH / GOLD"
    if sfc_pct > 50:
        return "STRUCTURAL DECAY", "REDUCE LEVERAGE"
    if dvol > 80:
        return "VOLATILITY BREAKDOWN", "HEDGE / REDUCE"
    return "NORMAL OSCILLATION", "Standard allocation"

def detect_regime(dvol, sfc_pct, news_stress, sentiment):
    if dvol is None or sfc_pct is None:
        return "NORMAL", 0.9, 0.1
    p_capitulation = 0.0
    if dvol:
        p_capitulation += min(dvol / 120.0, 1.0) * 0.4
    if sfc_pct:
        p_capitulation += min(sfc_pct / 60.0, 1.0) * 0.3
    if sentiment and sentiment < -0.3:
        p_capitulation += min(abs(sentiment), 1.0) * 0.3
    p_stress = 0.0
    if dvol:
        p_stress += min(max(dvol - 50, 0) / 50.0, 1.0) * 0.3
    if sfc_pct:
        p_stress += min(sfc_pct / 30.0, 1.0) * 0.3
    if news_stress:
        p_stress += min(news_stress / 20.0, 1.0) * 0.4
    p_normal = max(0, 1.0 - p_capitulation - p_stress)
    p_capitulation = min(p_capitulation, 1.0)
    p_stress = min(p_stress, 1.0 - p_capitulation)
    regimes = [("NORMAL", p_normal), ("STRESS", p_stress), ("CAPITULATION", p_capitulation)]
    dominant = max(regimes, key=lambda x: x[1])
    if dominant[0] == "NORMAL":
        transition_risk = round(p_stress + p_capitulation, 3)
    elif dominant[0] == "STRESS":
        transition_risk = round(p_capitulation, 3)
    else:
        transition_risk = 0.0
    return dominant[0], round(dominant[1], 3), transition_risk

print("[SFC] Starting data collection...", file=sys.stderr)

btc, chg, mcap = get_btc()
fng, fcls = get_fng()
dom = get_dom()
dvol = get_dvol()
m2_val, m2_yoy, _ = get_m2_data()
dxy = get_dxy()
pc_oi, pc_vol = get_put_call_ratio()
ath, ath_date = get_ath()

ohlcv = get_btc_ohlcv_daily(days=30)
closes_all = [c["close"] for c in ohlcv] if ohlcv else []
closes_7d = closes_all[-7:] if len(closes_all) >= 7 else closes_all
closes_30d = closes_all[-30:] if len(closes_all) >= 30 else closes_all
rsi_14 = compute_rsi(closes_all, period=14)

# Score factors from market data
factors = score_factors_from_market(btc, chg, dom, dvol, fng, pc_oi, m2_yoy, dxy)

# Calculate SFC ensemble
sfc_pct, zone, factors_raw, norm_factors, m1_klr, m2_logit, m3_bayes, m4_ewc, m5_qreg, m6_regime, method_agreement = calculate_sfc_ensemble(factors)

# News aggregation
print("[SFC] Aggregating news...", file=sys.stderr)
cp_key = os.getenv("CRYPTOPANIC_KEY", "")
news_stress, news_headlines, news_sentiment, articles_scored, news_stats = get_news_stress_v2(cp_key, max_workers=6)

# Black swan detection
shock_factor, shock_event, shock_severity = detect_black_swan_v2(articles_scored)

# Compute effective SFC
liq_mod = 0.0
if m2_yoy is not None:
    liq_mod = round((7.0 - m2_yoy) * 0.8, 1)
    liq_mod = max(-5.0, min(10.0, liq_mod))
effective_sfc = min(sfc_pct + news_stress + liq_mod, 100.0) if sfc_pct is not None else None
effective_sfc = max(effective_sfc, 0.0) if effective_sfc else None

# Floor and state (dynamic ATH)
fb, ft, dv_sfc, phi = compute_floor(dvol, effective_sfc, ath)
state, signal = determine_state(dvol, effective_sfc, btc, ft)

regime, regime_prob, transition_risk = detect_regime(dvol, effective_sfc, news_stress, news_sentiment)

# Technical indicators
sopr_proxy, sopr_signal, sopr_score = compute_sopr_proxy(closes_7d, closes_30d, btc)

# Composite confidence — dynamic components
# RSI confidence: extremes = momentum/extreme = unpredictable = lower confidence
if rsi_14 is not None:
    if rsi_14 < 20:
        rsi_conf = -0.10   # severely oversold → very uncertain
    elif rsi_14 < 30:
        rsi_conf = -0.07   # oversold → uncertain
    elif rsi_14 < 40:
        rsi_conf = -0.03   # approaching oversold → slightly uncertain
    elif rsi_14 > 80:
        rsi_conf = -0.10   # severely overbought → very uncertain
    elif rsi_14 > 70:
        rsi_conf = -0.07   # overbought → uncertain
    elif rsi_14 > 60:
        rsi_conf = -0.03   # approaching overbought → slightly uncertain
    else:
        rsi_conf = 0.03    # neutral RSI = calm = confident
else:
    rsi_conf = 0.0

# DVOL confidence: high vol = chaos = low confidence, low vol = calm = confident
if dvol is not None:
    if dvol > 100:
        dvol_conf = -0.10  # extreme vol → chaos
    elif dvol > 80:
        dvol_conf = -0.06  # high vol → uncertain
    elif dvol < 40:
        dvol_conf = 0.05   # low vol → calm, confident
    else:
        dvol_conf = 0.0
else:
    dvol_conf = 0.0

# Dynamic liquidation indicators
if dvol is not None:
    liq_density = round(min(dvol / 150.0, 1.0), 3)
    cascade_risk = round(min((sopr_score or 0.5) * 0.3 + (dvol / 200.0), 0.95), 3)
else:
    liq_density = 0.15
    cascade_risk = 0.1

# Liquidation pressure based on RSI + trend
if rsi_14 is not None:
    if rsi_14 < 25 and sopr_proxy and sopr_proxy < 0.97:
        liq_pressure = "LONG_SQUEEZE"
    elif rsi_14 > 70 and sopr_proxy and sopr_proxy > 1.03:
        liq_pressure = "SHORT_SQUEEZE"
    else:
        liq_pressure = "BALANCED"
else:
    liq_pressure = "BALANCED"

# Composite confidence — multi-factor penalized model
# Base: method agreement + market calmness
cc_base = 0.30
cc_base += method_agreement * 0.15   # methods agree → more reliable
cc_base += max(0, 1.0 - (effective_sfc/100)) * 0.08  # low stress → more reliable

# Penalties — reduce confidence when conditions contradict SFC signal
cc_penalty = 0.0

# Cascade risk — high cascade = signal less reliable
if cascade_risk > 0.5:
    cc_penalty += 0.10
elif cascade_risk > 0.35:
    cc_penalty += 0.05

# RSI extremes — extreme momentum = unpredictable
if rsi_14 is not None:
    if rsi_14 < 25:
        cc_penalty += 0.08
    elif rsi_14 < 35:
        cc_penalty += 0.04
    elif rsi_14 > 75:
        cc_penalty += 0.08
    elif rsi_14 > 65:
        cc_penalty += 0.04

# Liquidation squeeze — one-sided pressure = unreliable signal
if liq_pressure in ('LONG_SQUEEZE', 'SHORT_SQUEEZE'):
    cc_penalty += 0.06

# Extreme fear/greed — emotional market = unpredictable
if fng is not None and fng < 15:
    cc_penalty += 0.06
elif fng is not None and fng > 85:
    cc_penalty += 0.04

# SOPR capitulation — on-chain stress
if sopr_proxy is not None and sopr_proxy < 0.98:
    cc_penalty += 0.05

# Very negative news sentiment
if news_sentiment < -0.5:
    cc_penalty += 0.04
elif news_sentiment < -0.3:
    cc_penalty += 0.02

# High volatility
if dvol is not None and dvol > 80:
    cc_penalty += 0.05

# Regime transition — risk of regime flip
if transition_risk > 0.5:
    cc_penalty += 0.05

composite_confidence = max(0.05, min(cc_base - cc_penalty, 0.95))
composite_confidence = round(composite_confidence, 3)

# Build output
out = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "btc": btc,
    "btc_24h": chg,
    "btc_mcap": mcap,
    "fng": fng,
    "fng_cls": fcls,
    "dom": round(dom, 1) if dom else None,
    "dvol": dvol,
    "sfc_base": sfc_pct,
    "sfc_effective": effective_sfc,
    "news_stress": news_stress,
    "news_headlines": news_headlines[:8],
    "news_sentiment": news_sentiment,
    "news_stats": news_stats,
    "m2_yoy": m2_yoy,
    "liq_mod": liq_mod,
    "zone": zone,
    "floor_buffer": fb,
    "floor_total": ft,
    "regime": regime,
    "regime_prob": regime_prob,
    "transition_risk": transition_risk,
    "dxy": dxy,
    "pc_oi": pc_oi,
    "pc_vol": pc_vol,
    "dv_sfc": dv_sfc,
    "phi": phi,
    "state": state,
    "signal": signal,
    "factors": factors_raw,
    "ath": ath,
    "ath_date": ath_date,
    "rsi_14": rsi_14,
    "rsi_regime": "OVERSOLD" if rsi_14 is not None and rsi_14 < 30 else "OVERBOUGHT" if rsi_14 is not None and rsi_14 > 70 else "NEUTRAL",
    "sopr_proxy": sopr_proxy,
    "sopr_signal": sopr_signal,
    "sopr_score": sopr_score,
    "cascade_risk": cascade_risk,
    "liq_density": liq_density,
    "liq_pressure": liq_pressure,
    "composite_confidence": composite_confidence,
    "confidence_components": {
        "method_agree": round(method_agreement, 3),
        "rsi": round(rsi_conf, 3),
        "sopr": round(-0.05 if sopr_proxy is not None and sopr_proxy < 0.98 else 0.0, 3),
        "dvol": round(dvol_conf, 3),
        "cascade_penalty": round(-(0.10 if cascade_risk > 0.5 else 0.05 if cascade_risk > 0.35 else 0.0), 3),
        "fear_penalty": round(-(0.06 if fng is not None and fng < 15 else 0.0), 3)
    },
    "m1_klr": round(m1_klr * 100, 1),
    "m2_logit": round(m2_logit * 100, 1),
    "m3_bayes": round(m3_bayes * 100, 1),
    "m4_ewc": round(m4_ewc * 100, 1),
    "m5_qreg": round(m5_qreg, 1),
    "m6_regime_score": round(m6_regime, 1),
    "method_agreement": round(method_agreement, 3),
    "readiness_score": round(composite_confidence * (1.0 - min(effective_sfc/100 if effective_sfc else 0, 0.5)), 3),
    "shock_factor": shock_factor,
    "shock_event": shock_event,
    "shock_severity": shock_severity,
}

print(json.dumps(out, indent=2))
btc_str = f"${btc:,.0f}" if btc is not None else "N/A"
rsi_str = f"{rsi_14}" if rsi_14 is not None else "N/A"
sopr_str = f"{sopr_proxy}" if sopr_proxy is not None else "N/A"
print(f"\n✅ BTC={btc_str} | SFC={effective_sfc:.1f}% | Zone={zone} | RSI={rsi_str} | SOPR={sopr_str} | News={news_stress:.1f} | {regime}", file=sys.stderr)
