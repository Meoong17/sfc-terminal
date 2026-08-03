#!/usr/bin/env python3
"""
SFC Institutional Methods (M20-M31) — Free Data Sources
======================================================
M20-M23: Microstructure (Binance public REST — free, no API key)
M24-M27: Behavioral/Tail Risk (CoinGecko + Deribit — free)
M28-M31: Macro/Debt Cycle (FRED API — user key)
"""

import json, os, sys, math, time, statistics
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY", "")

# ── MICROSTRUCTURE CHANGE DETECTION CACHE ──
_MICRO_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.micro_cache.json')

def _load_micro_cache():
    try:
        with open(_MICRO_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_micro_cache(state):
    with open(_MICRO_CACHE_FILE, 'w') as f:
        json.dump(state, f)

# ──────────────────────────────────────────────────────
# TIER 0: MICROSTRUCTURE (M20-M23)
# Binance public REST API — free, no key required
# ──────────────────────────────────────────────────────

BINANCE_BASE = "https://api.binance.com"

# ── CoinGecko API key ────────────────────────────────────
CG_API_PARAM = "x_cg_demo_api_key=" + os.getenv("COINGECKO_API_KEY", "")

# ── Shared CoinGecko cache (M24-M27, M31 all need market data) ──
_CG_CACHE = {}

def _cg_fetch_all():
    """Fetch all CoinGecko data in parallel. Falls back to Binance klines on rate limit."""
    if _CG_CACHE:
        return _CG_CACHE

    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    cg_key = CG_API_PARAM
    urls = [
        ("coin_info", f"https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&community_data=false&developer_data=false&{cg_key}"),
        ("prices_365", f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily&{cg_key}"),
        ("prices_90", f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily&{cg_key}"),
        ("prices_200", f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=200&interval=daily&{cg_key}"),
    ]
    
    results = {}
    def _fetch_one(name, url):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return name, r.json()
            return name, None
        except:
            return name, None
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_one, n, u): n for n, u in urls}
        for f in as_completed(futures):
            name, data = f.result()
            results[name] = data
    
    # Process results
    coin = results.get("coin_info")
    if coin:
        md = coin.get("market_data", {})
        _CG_CACHE["ath"] = float(md.get("ath", {}).get("usd", 126272))
        _CG_CACHE["ath_date"] = md.get("ath_date", {}).get("usd", "")
        _CG_CACHE["mcap"] = float(md.get("market_cap", {}).get("usd", 0))
        _CG_CACHE["vol_24h"] = float(md.get("total_volume", {}).get("usd", 0))
    
    for key in ["prices_365", "prices_90", "prices_200"]:
        data = results.get(key)
        if data:
            _CG_CACHE[key] = [p[1] for p in data.get("prices", [])]
    
    # If all CoinGecko calls failed, fall back to Binance klines
    if not _CG_CACHE:
        print("[CG Cache] All CoinGecko calls failed, trying Binance fallback...", file=sys.stderr)
        _binance_klines_fallback()
    
    return _CG_CACHE

def _cg_get(key, default=None):
    """Get from cache, auto-fetching if empty."""
    if not _CG_CACHE:
        _cg_fetch_all()
    return _CG_CACHE.get(key, default)


def _binance_klines_fallback():
    """Fallback when CoinGecko rate-limited: use Binance klines."""
    try:
        # Binance daily klines: [open, high, low, close, volume, ...]
        r = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=365", timeout=10)
        if r.status_code != 200:
            return
        klines = r.json()
        prices = [float(k[4]) for k in klines]  # close prices
        if prices:
            _CG_CACHE["prices_365"] = prices
            _CG_CACHE["prices_90"] = prices[-90:] if len(prices) >= 90 else prices
            _CG_CACHE["prices_200"] = prices[-200:] if len(prices) >= 200 else prices
            _CG_CACHE["ath"] = max(prices)  # Use highest close as ATH approximation
            _CG_CACHE["mcap"] = prices[-1] * 19_500_000  # Approximate: price * circulating supply
            _CG_CACHE["vol_24h"] = float(klines[-1][5]) * float(klines[-1][4])  # volume * close
            print(f"[Binance Fallback] Used Binance klines ({len(prices)} days)", file=sys.stderr)
    except Exception as e:
        print(f"[Binance Fallback] Error: {e}", file=sys.stderr)

def _binance_depth(symbol="BTCUSDT", limit=20):
    """Fetch order book depth from Binance (free)."""
    try:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/depth?symbol={symbol}&limit={limit}",
            timeout=10
        )
        if r.status_code != 200:
            return None, None
        d = r.json()
        bids = [(float(b[0]), float(b[1])) for b in d.get("bids", [])]
        asks = [(float(a[0]), float(a[1])) for a in d.get("asks", [])]
        return bids, asks
    except:
        return None, None

def _binance_trades(symbol="BTCUSDT", limit=100):
    """Fetch recent trades from Binance (free)."""
    try:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/trades?symbol={symbol}&limit={limit}",
            timeout=10
        )
        if r.status_code != 200:
            return []
        trades = r.json()
        return [{"price": float(t["price"]), "qty": float(t["qty"]),
                  "time": t["time"], "isBuyerMaker": t["isBuyerMaker"]}
                for t in trades]
    except:
        return []

def _binance_ticker(symbol="BTCUSDT"):
    """Fetch 24hr ticker from Binance (free)."""
    try:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/ticker/bookTicker?symbol={symbol}",
            timeout=10
        )
        if r.status_code != 200:
            return None
        d = r.json()
        bid = float(d.get("bidPrice", 0))
        ask = float(d.get("askPrice", 0))
        # Also get 24hr stats for range
        r2 = requests.get(
            f"{BINANCE_BASE}/api/v3/ticker/24hr?symbol={symbol}",
            timeout=10
        )
        if r2.status_code == 200:
            d2 = r2.json()
            return {
                "bidPrice": bid,
                "askPrice": ask,
                "lastPrice": float(d2.get("lastPrice", 0)),
                "volume": float(d2.get("volume", 0)),
                "quoteVolume": float(d2.get("quoteVolume", 0)),
                "highPrice": float(d2.get("highPrice", 0)),
                "lowPrice": float(d2.get("lowPrice", 0)),
                "count": int(d2.get("count", 0))
            }
        return {"bidPrice": bid, "askPrice": ask, "lastPrice": bid, "volume": 0,
                "quoteVolume": 0, "highPrice": max(bid, ask), "lowPrice": min(bid, ask), "count": 0}
    except:
        return None


def calculate_m20_order_book_imbalance(prev=None):
    """
    M20: Order Book Imbalance (OBI)
    OBI = (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
    Data: Binance depth (free REST, no key)
    """
    bids, asks = _binance_depth(limit=20)
    if bids is None or asks is None:
        return None, None, {}

    bid_vol = sum(v for _, v in bids)
    ask_vol = sum(v for _, v in asks)
    total = bid_vol + ask_vol
    if total == 0:
        return None, None, {}

    obi = (bid_vol - ask_vol) / total  # -1 to +1
    obi_norm = (obi + 1) / 2           # 0 to 1

    # Score: extreme imbalance = stress
    if obi_norm > 0.75:   # Heavy buy side = bullish but frothy
        score = 0.35
    elif obi_norm < 0.25: # Heavy sell side = bearish stress
        score = 0.75
    elif obi_norm < 0.35: # Moderate sell pressure
        score = 0.55
    elif obi_norm > 0.65: # Moderate buy pressure
        score = 0.40
    else:
        score = 0.25       # Balanced

    # ── Change detection ──
    change = {}
    if prev and "obi_norm" in prev:
        prev_obi = prev["obi_norm"]
        obi_delta = obi_norm - prev_obi
        if obi_delta < -0.15:
            change["obi_change"] = "SELL_SURGE"      # Sell pressure meningkat drastis
        elif obi_delta < -0.08:
            change["obi_change"] = "SELLING_UP"
        elif obi_delta > 0.15:
            change["obi_change"] = "BUY_SURGE"
        elif obi_delta > 0.08:
            change["obi_change"] = "BUYING_UP"
        else:
            change["obi_change"] = "STABLE"
        change["obi_delta"] = round(obi_delta, 4)
    else:
        change["obi_change"] = "FIRST_RUN"
        change["obi_delta"] = 0

    change["obi"] = round(obi, 4)
    change["obi_norm"] = round(obi_norm, 4)
    change["bid_vol"] = round(bid_vol, 2)
    change["ask_vol"] = round(ask_vol, 2)

    return score, change


def calculate_m21_large_trade_flow(prev=None):
    """
    M21: Large Trade Flow Ratio
    Track large trades (>$50K notional) vs small retail
    Data: Binance recent trades (free REST, no key)
    """
    trades = _binance_trades(limit=100)
    if not trades:
        return None, None, {}

    # Estimate large trade threshold: $50K notional
    large_threshold = 50000  # USD
    large_sells, large_buys = 0, 0
    small_sells, small_buys = 0, 0

    for t in trades:
        notional = t["price"] * t["qty"]
        if t["isBuyerMaker"]:
            # BuyerMaker=True → sell order (aggressive sell)
            if notional >= large_threshold:
                large_sells += notional
            else:
                small_sells += notional
        else:
            if notional >= large_threshold:
                large_buys += notional
            else:
                small_buys += notional

    total_large = large_buys + large_sells
    total_small = small_buys + small_sells
    if total_large == 0:
        # No large trades = retail dominated = neutral
        return 0.30, {"large_sell_notional": 0, "large_buy_notional": 0,
                       "large_ratio": 0, "large_sell_ratio": 0.5,
                       "large_share": 0}

    large_sell_ratio = large_sells / total_large if total_large > 0 else 0.5

    if large_sell_ratio > 0.65:
        score = 0.75   # Heavy institutional selling
    elif large_sell_ratio > 0.55:
        score = 0.55
    elif large_sell_ratio < 0.35:
        score = 0.25   # Heavy institutional buying = bullish
    elif large_sell_ratio < 0.45:
        score = 0.35
    else:
        score = 0.45

    # ── Change detection ──
    detail = {"large_sell_notional": round(large_sells, 0),
              "large_buy_notional": round(large_buys, 0),
              "large_sell_ratio": round(large_sell_ratio, 4),
              "large_share": round(total_large / (total_large + total_small), 4) if (total_large + total_small) > 0 else 0}
    if prev and "large_sell_ratio" in prev:
        prev_ratio = prev["large_sell_ratio"]
        delta = large_sell_ratio - prev_ratio
        if delta > 0.20:
            detail["flow_change"] = "SELL_SURGE"
        elif delta > 0.10:
            detail["flow_change"] = "SELLING_UP"
        elif delta < -0.20:
            detail["flow_change"] = "BUY_SURGE"
        elif delta < -0.10:
            detail["flow_change"] = "BUYING_UP"
        else:
            detail["flow_change"] = "STABLE"
        detail["flow_delta"] = round(delta, 4)
    else:
        detail["flow_change"] = "FIRST_RUN"
        detail["flow_delta"] = 0

    return score, detail


def calculate_m22_spread_momentum(prev=None):
    """
    M22: Bid-Ask Spread Momentum
    Spread = market stress barometer
    Data: Binance 24hr ticker (free REST, no key)
    """
    ticker = _binance_ticker()
    if not ticker:
        return None, None, {}

    bid = ticker.get("bidPrice", 0)
    ask = ticker.get("askPrice", 0)
    mid = ticker.get("lastPrice", 0)
    if bid <= 0 or ask <= 0 or mid <= 0:
        return None, None, {}

    spread_bps = (ask - bid) / mid * 10000  # Spread in bps
    high = ticker.get("highPrice", mid)
    low = ticker.get("lowPrice", mid)
    range_bps = (high - low) / mid * 10000 if mid > 0 else 0

    # Wider spread = stress
    if spread_bps > 10:    # >10 bps = very wide for BTC
        spread_score = 0.80
    elif spread_bps > 5:   # 5-10 bps = elevated
        spread_score = 0.55
    elif spread_bps > 2:   # 2-5 bps = normal
        spread_score = 0.30
    elif spread_bps > 0.5: # 0.5-2 bps = tight
        spread_score = 0.15
    else:                  # <0.5 bps = very tight/confident
        spread_score = 0.10

    # 24h range expansion amplifies stress
    if range_bps > 500:    # 5%+ intraday range
        range_penalty = 0.15
    elif range_bps > 300:  # 3-5% range
        range_penalty = 0.08
    else:
        range_penalty = 0.0

    score = min(spread_score + range_penalty, 0.95)

    # ── Change detection ──
    detail = {"spread_bps": round(spread_bps, 4), "range_bps": round(range_bps, 2),
              "bid": bid, "ask": ask, "mid": mid}
    if prev and "spread_bps" in prev:
        prev_spread = prev["spread_bps"]
        delta = spread_bps - prev_spread
        if delta > 3:
            detail["spread_change"] = "WIDENING_FAST"
        elif delta > 1:
            detail["spread_change"] = "WIDENING"
        elif delta < -3:
            detail["spread_change"] = "NARROWING_FAST"
        elif delta < -1:
            detail["spread_change"] = "NARROWING"
        else:
            detail["spread_change"] = "STABLE"
        detail["spread_delta"] = round(delta, 4)
    else:
        detail["spread_change"] = "FIRST_RUN"
        detail["spread_delta"] = 0

    return score, detail


def calculate_m23_liquidity_fractals(prev=None):
    """
    M23: Liquidity Fractals — is there enough depth to absorb large orders?
    Data: Binance order book depth (free REST, no key)
    """
    bids, asks = _binance_depth(limit=100)
    if bids is None or asks is None:
        return None, None, {}

    # Slippage estimate: how far does $1M market buy move the price?
    target = 1_000_000  # $1M USD notional
    mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
    if mid <= 0:
        return None, None, {}

    # Simulate a $1M market buy (hitting asks)
    cum_buy = 0
    levels_used_buy = 0
    avg_price_buy = 0
    for price, qty in asks:
        need = target - cum_buy
        if need <= 0:
            break
        available = price * qty
        take = min(available, need)
        cum_buy += take
        avg_price_buy += price * take
        levels_used_buy += 1
        if cum_buy >= target:
            break
    else:
        levels_used_buy = 999  # Insufficient liquidity

    # Simulate a $1M market sell (hitting bids)
    cum_sell = 0
    levels_used_sell = 0
    avg_price_sell = 0
    for price, qty in bids:
        need = target - cum_sell
        if need <= 0:
            break
        available = price * qty
        take = min(available, need)
        cum_sell += take
        avg_price_sell += price * take
        levels_used_sell += 1
        if cum_sell >= target:
            break
    else:
        levels_used_sell = 999

    # Slippage percentage
    if cum_buy >= target and levels_used_buy < 999:
        slippage_buy = (avg_price_buy / cum_buy - mid) / mid * 100
    else:
        slippage_buy = 5.0  # High slippage if insufficient liquidity

    if cum_sell >= target and levels_used_sell < 999:
        slippage_sell = (mid - avg_price_sell / cum_sell) / mid * 100
    else:
        slippage_sell = 5.0

    avg_slippage = (slippage_buy + slippage_sell) / 2
    min_levels = min(levels_used_buy, levels_used_sell)

    if avg_slippage > 2.0:
        score = 0.80   # Very thin book
    elif avg_slippage > 1.0:
        score = 0.60
    elif avg_slippage > 0.5:
        score = 0.40
    elif min_levels > 20:
        score = 0.15   # Deep book, safe
    else:
        score = 0.25

    # ── Change detection ──
    detail = {"slippage_pct": round(avg_slippage, 3),
              "levels_used_buy": levels_used_buy,
              "levels_used_sell": levels_used_sell,
              "mid_price": round(mid, 2)}
    if prev and "slippage_pct" in prev:
        prev_slippage = prev["slippage_pct"]
        delta = avg_slippage - prev_slippage
        if delta > 1.0:
            detail["liq_change"] = "THINNING_FAST"
        elif delta > 0.3:
            detail["liq_change"] = "THINNING"
        elif delta < -1.0:
            detail["liq_change"] = "DEEPENING_FAST"
        elif delta < -0.3:
            detail["liq_change"] = "DEEPENING"
        else:
            detail["liq_change"] = "STABLE"
        detail["slippage_delta"] = round(delta, 3)
    else:
        detail["liq_change"] = "FIRST_RUN"
        detail["slippage_delta"] = 0

    return score, detail


# ──────────────────────────────────────────────────────
# TIER 1: BEHAVIORAL ECONOMICS (M24-M28)
# CoinGecko (free) + Deribit (free) + FRED
# ──────────────────────────────────────────────────────

def _fred(series, limit=2):
    """Helper: fetch FRED series (free, needs API key)."""
    if not FRED_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}",
            timeout=15
        )
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        return vals if vals else None
    except:
        return None


def calculate_m24_cape():
    """
    M24: Shiller CAPE Ratio (Bitcoin adaptation)
    CAPE = Price / (avg realized price ~10 years)
    Uses shared CoinGecko cache (1 call for all methods).
    """
    prices = _cg_get("prices_365")
    if not prices or len(prices) < 30:
        return None, None

    current_price = prices[-1]
    sma_1y = statistics.mean(prices)

    if sma_1y <= 0:
        return None, None

    cape = current_price / sma_1y

    if cape > 3.0:
        score = 0.80
    elif cape > 2.5:
        score = 0.65
    elif cape > 2.0:
        score = 0.50
    elif cape < 1.0:
        score = 0.10
    elif cape < 1.5:
        score = 0.15
    else:
        score = 0.30

    return score, {"cape": round(cape, 4), "current_price": round(current_price, 2),
                   "sma_1y_realized": round(sma_1y, 2)}


def calculate_m25_minsky_moment():
    """
    M25: Minsky Moment Detection — leverage cycle analysis
    Uses Deribit perpetual order book for funding rate.
    Free: Deribit REST API.
    """
    try:
        # Get current funding from BTC-PERPETUAL order book
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-PERPETUAL",
            timeout=10
        )
        if r.status_code != 200:
            return None, None
        result = r.json().get("result", {})
        current_funding = result.get("current_funding", 0)
        funding_8h = result.get("funding_8h", 0)

        # Get funding rate history with instrument_name parameter
        now_ms = int(time.time() * 1000)
        seven_days_ago = now_ms - (7 * 24 * 60 * 60 * 1000)
        r2 = requests.get(
            f"https://www.deribit.com/api/v2/public/get_funding_rate_history?currency=BTC&instrument_name=BTC-PERPETUAL&start_timestamp={seven_days_ago}&end_timestamp={now_ms}",
            timeout=10
        )
        rates_hist = []
        if r2.status_code == 200:
            hist = r2.json().get("result", [])
            rates_hist = [d["interest_8h"] for d in hist[:8]] if hist else []

        # Use current funding + historical if available
        fr_now = current_funding or 0
        if len(rates_hist) >= 3:
            fr_1 = rates_hist[1] if len(rates_hist) > 1 else 0
            fr_2 = rates_hist[2] if len(rates_hist) > 2 else 0
            accel = (fr_now - fr_1) - (fr_1 - fr_2)
        else:
            accel = 0

        # OI from futures
        r3 = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=future",
            timeout=10
        )
        futures = r3.json().get("result", [])
        total_oi = sum(f.get("open_interest", 0) for f in futures)

        if fr_now > 0.01 and accel > 0.0005:
            stage = "PEAK"
            score = 0.85
        elif fr_now > 0.005 and accel > 0:
            stage = "BOOM"
            score = 0.65
        elif fr_now > 0:
            stage = "DISPLACEMENT"
            score = 0.35
        elif fr_now < -0.005:
            stage = "REVULSION"
            score = 0.70
        elif fr_now < -0.002:
            stage = "CRISIS"
            score = 0.85
        else:
            stage = "NORMAL"
            score = 0.15

        return score, {"funding_rate": round(fr_now, 6), "funding_8h": round(funding_8h, 8),
                       "accel": round(accel, 6),
                       "oi_futures": round(total_oi, 0), "minsky_stage": stage}
    except Exception as e:
        print(f"[M25] Error: {e}", file=sys.stderr)
        return None, None


def calculate_m26_kahneman_bias(current_price=None):
    """
    M26: Kahneman Behavioral Bias Index
    Loss Aversion + Anchoring bias using shared CoinGecko cache + Deribit IV.
    """
    if current_price is None or current_price <= 0:
        return None, None

    try:
        prices = _cg_get("prices_365")
        if not prices or len(prices) < 30:
            return None, None

        recent = prices[-60:] if len(prices) >= 60 else prices
        realized_price = statistics.mean(recent[-30:])
        ath = _cg_get("ath", 126272)

        # Get put/call IV skew from Deribit
        r3 = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
            timeout=10
        )
        opts = r3.json().get("result", [])
        puts_iv = [o["mark_iv"] for o in opts if o.get("instrument_name","").endswith("-P") and o.get("mark_iv")]
        calls_iv = [o["mark_iv"] for o in opts if o.get("instrument_name","").endswith("-C") and o.get("mark_iv")]

        # Loss aversion
        if current_price < realized_price:
            loss_pct = (realized_price - current_price) / realized_price
            if loss_pct > 0.30:
                la_score = 0.75
            elif loss_pct > 0.15:
                la_score = 0.55
            elif loss_pct > 0.05:
                la_score = 0.35
            else:
                la_score = 0.25
            underwater_pct = -loss_pct
        else:
            profit_pct = (current_price - realized_price) / realized_price
            if profit_pct > 0.50:
                la_score = 0.15
            elif profit_pct > 0.25:
                la_score = 0.20
            else:
                la_score = 0.25
            underwater_pct = profit_pct

        # Anchoring to ATH
        ath_distance = (ath - current_price) / ath if ath > 0 else 0
        if ath_distance < 0.05:
            anchor_score = 0.70
        elif ath_distance < 0.15:
            anchor_score = 0.45
        elif ath_distance < 0.30:
            anchor_score = 0.25
        else:
            anchor_score = 0.15

        # Fear from options skew
        if puts_iv and calls_iv:
            put_iv = statistics.mean(puts_iv)
            call_iv = statistics.mean(calls_iv)
            atm = (put_iv + call_iv) / 2
            skew = (put_iv - call_iv) / atm if atm > 0 else 0
            fear_score = 0.60 if skew > 0.20 else 0.20
        else:
            fear_score = 0.30

        score = 0.40 * la_score + 0.30 * anchor_score + 0.30 * fear_score

        return score, {
            "loss_aversion": round(la_score, 3),
            "anchoring": round(anchor_score, 3),
            "fear": round(fear_score, 3),
            "realized_price": round(realized_price, 2),
            "ath": round(ath, 2),
            "pl_pct": round(underwater_pct * 100, 2)
        }
    except Exception as e:
        print(f"[M26] Error: {e}", file=sys.stderr)
        return None, None


def calculate_m27_taleb_tail_risk():
    """
    M27: Taleb Tail Risk (Black Swan)
    Uses Deribit DVOL + shared CoinGecko cache for returns.
    """
    try:
        # Get DVOL from Deribit options
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
            timeout=10
        )
        if r.status_code != 200:
            return None, None
        opts = r.json().get("result", [])
        ivs = [(o.get("mark_iv", 0), o.get("open_interest", 0)) for o in opts if o.get("mark_iv")]
        if not ivs:
            return None, None

        oi_total = sum(x[1] for x in ivs)
        if oi_total <= 0:
            return None, None
        dvol = sum(x[0] * x[1] for x in ivs) / oi_total

        # Get 90-day returns from cache
        prices = _cg_get("prices_90")
        if not prices or len(prices) < 30:
            return None, None

        rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        n = len(rets)

        # Return skewness
        mean = sum(rets) / n
        var = sum((r - mean)**2 for r in rets) / n
        std = math.sqrt(var)
        skewness = (sum((r - mean)**3 for r in rets) / n) / (std**3) if std > 0 else 0

        # Max drawdown in period
        peak = prices[0]
        max_dd = 0
        for p in prices[1:]:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            max_dd = max(max_dd, dd)

        # Short-term vol from 7-day data via cache 365
        prices_365 = _cg_get("prices_365")
        if prices_365 and len(prices_365) > 7:
            p7 = prices_365[-7:]
            rets_7d = [(p7[i] - p7[i-1]) / p7[i-1] for i in range(1, len(p7))]
            vol_7d_annual = statistics.stdev(rets_7d) * math.sqrt(365) if len(rets_7d) > 1 else dvol/100
        else:
            vol_7d_annual = dvol/100

        vol_90d_annual = dvol / 100

        if vol_7d_annual > vol_90d_annual * 1.2:
            term_stress = 0.75
        elif vol_90d_annual > vol_7d_annual * 1.3:
            term_stress = 0.50
        else:
            term_stress = 0.30

        if skewness < -1.0:
            skew_stress = 0.80
        elif skewness < -0.5:
            skew_stress = 0.55
        elif skewness > 1.0:
            skew_stress = 0.50
        else:
            skew_stress = 0.25

        if max_dd > 0.20:
            dd_stress = 0.80
        elif max_dd > 0.10:
            dd_stress = 0.55
        else:
            dd_stress = 0.25

        score = 0.35 * term_stress + 0.35 * skew_stress + 0.30 * dd_stress

        return score, {
            "dvol": round(dvol, 2),
            "skewness": round(skewness, 4),
            "max_drawdown": round(max_dd, 4),
            "vol_7d_ann": round(vol_7d_annual, 4),
            "vol_90d_ann": round(vol_90d_annual, 4),
            "term": "1W>3M" if vol_7d_annual > vol_90d_annual else "3M>1W",
            "tail_danger": "CRITICAL" if score > 0.70 else "ELEVATED" if score > 0.50 else "NORMAL"
        }
    except Exception as e:
        print(f"[M27] Error: {e}", file=sys.stderr)
        return None, None


def calculate_m28_summers_stagnation():
    """
    M28: Summers Secular Stagnation Indicator
    r* vs g analysis using FRED data.
    Free: FRED API.
    """
    vals_fed = _fred("FEDFUNDS", 1)
    vals_gdp = _fred("GDPC1", 5)  # Real GDP
    if not vals_fed or not vals_gdp:
        return None, None

    fed_rate = vals_fed[0]

    # GDP growth (YoY)
    if len(vals_gdp) >= 5:
        gdp_now = vals_gdp[0]
        gdp_yr = vals_gdp[4]
        gdp_growth = (gdp_now - gdp_yr) / gdp_yr if gdp_yr > 0 else 2.0
    else:
        gdp_growth = 2.0

    # Get 10Y Treasury yield for real rate estimate
    vals_10y = _fred("DGS10", 1)
    vals_cpi = _fred("CPIAUCSL", 13)
    if vals_10y and vals_cpi and len(vals_cpi) >= 13:
        cpi_now = vals_cpi[0]
        cpi_yr = vals_cpi[12]
        cpi_yoy = (cpi_now - cpi_yr) / cpi_yr
        real_rate = vals_10y[0] - (cpi_yoy * 100)
    else:
        real_rate = fed_rate - 3.0  # Rough estimate

    if real_rate < gdp_growth * 0.5:
        stagnation = 0.75   # r* much lower than g
    elif real_rate < gdp_growth:
        stagnation = 0.50
    else:
        stagnation = 0.20

    return stagnation, {
        "real_rate": round(real_rate, 2),
        "gdp_growth": round(gdp_growth * 100, 2),
        "fed_rate": round(fed_rate, 2),
        "regime": "STAGNATION" if stagnation > 0.60 else "NORMAL"
    }


# ──────────────────────────────────────────────────────
# TIER 2: DEBT CYCLE & CREDIT (M29-M31)
# ──────────────────────────────────────────────────────

def calculate_m29_debt_crisis():
    """
    M29: Reinhart-Rogoff Debt Threshold Framework
    Debt/GDP analysis. Also applies to stablecoin/DeFi leverage.
    Free: FRED (GFDEGDQ188S = Federal Debt/GDP).
    """
    try:
        # Federal debt to GDP
        vals_debt = _fred("GFDEGDQ188S", 1)
        if not vals_debt:
            # Fallback: use M2/GDP proxy
            r = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=M2SL&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit=1",
                timeout=10
            )
            m2_data = r.json().get("observations", [])
            r2 = requests.get(
                f"https://api.stlouisfed.org/fred/series/observations?series_id=GDPC1&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit=1",
                timeout=10
            )
            gdp_data = r2.json().get("observations", [])
            if m2_data and gdp_data:
                m2 = float(m2_data[0]["value"])
                gdp = float(gdp_data[0]["value"])
                debt_ratio = m2 / gdp if gdp > 0 else 0.70
            else:
                debt_ratio = 0.70
        else:
            debt_ratio = vals_debt[0] / 100  # Convert from % to ratio

        if debt_ratio > 1.20:
            score = 0.85
        elif debt_ratio > 0.90:
            score = 0.70
        elif debt_ratio > 0.60:
            score = 0.45
        else:
            score = 0.20

        return score, {"debt_gdp_ratio": round(debt_ratio, 3),
                       "crisis_risk": "HIGH" if debt_ratio > 0.90 else "ELEVATED" if debt_ratio > 0.60 else "LOW"}
    except Exception as e:
        print(f"[M29] Error: {e}", file=sys.stderr)
        return None, None


def calculate_m30_rajan_fsi():
    """
    M30: Rajan Financial Stability Indicators
    6 indicators: credit, asset volatility, concentration, capital flows, CDS, yield curve.
    Free: FRED + crypto market data.
    """
    try:
        score = 0.0
        components = {}

        # 1. Credit growth (M2 YoY%)
        vals_m2 = _fred("M2SL", 13)
        if vals_m2 and len(vals_m2) >= 13:
            m2_growth = (vals_m2[0] - vals_m2[-1]) / vals_m2[-1]
            if m2_growth > 0.15:
                score += 0.20
            elif m2_growth > 0.10:
                score += 0.10
            components["credit_growth"] = round(m2_growth * 100, 2)
        else:
            components["credit_growth"] = None

        # 2. Asset volatility (BTC DVOL from Deribit)
        try:
            r = requests.get(
                "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
                timeout=10
            )
            opts = r.json().get("result", [])
            ivs = [o.get("mark_iv", 0) for o in opts if o.get("mark_iv")]
            if ivs:
                avg_iv = statistics.mean(ivs)
                if avg_iv > 80:
                    score += 0.20
                elif avg_iv > 60:
                    score += 0.10
                components["asset_vol"] = round(avg_iv, 2)
            else:
                components["asset_vol"] = None
        except:
            components["asset_vol"] = None

        # 3. Yield curve (10Y-2Y)
        vals_10y = _fred("DGS10", 1)
        vals_2y = _fred("DGS2", 1)
        if vals_10y and vals_2y:
            term = vals_10y[0] - vals_2y[0]
            if term < 0.5:
                score += 0.15  # Flat/inverted = concern
            components["yield_curve"] = round(term, 2)
        else:
            components["yield_curve"] = None

        # 4. CDS-like proxy: HY spread
        vals_hy = _fred("BAMLH0A0HYM2", 1)
        if vals_hy:
            if vals_hy[0] > 400:
                score += 0.15
            elif vals_hy[0] > 300:
                score += 0.10
            components["hy_spread"] = round(vals_hy[0], 1)
        else:
            components["hy_spread"] = None

        # 5. Crypto-specific: concentration in options OI
        try:
            r2 = requests.get(
                "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
                timeout=10
            )
            opts2 = r2.json().get("result", [])
            ois = sorted([o.get("open_interest", 0) for o in opts2 if o.get("open_interest", 0) > 0], reverse=True)
            if ois and len(ois) > 1:
                total = sum(ois)
                if total > 0:
                    top3_share = sum(ois[:3]) / total
                    if top3_share > 0.60:
                        score += 0.10
                    components["top3_concentration"] = round(top3_share, 3)
            else:
                components["top3_concentration"] = None
        except:
            components["top3_concentration"] = None

        # 6. Capital flows proxy: stablecoin supply growth (approximate via market sentiment)
        try:
            r3 = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            fng_val = r3.json()["data"][0]["value"]
            # Rapid sentiment shift = capital flow volatility
            components["fng"] = int(fng_val)
        except:
            components["fng"] = None

        return min(score, 0.95), {
            "fsi_score": round(score, 3),
            "components": components,
            "stability": "VULNERABLE" if score > 0.70 else "ELEVATED" if score > 0.40 else "NORMAL"
        }
    except Exception as e:
        print(f"[M30] Error: {e}", file=sys.stderr)
        return None, None


def calculate_m31_altman_zscore(btc_current=None):
    """
    M31: Altman Z-Score (adapted for crypto)
    Uses shared CoinGecko cache + Binance + Deribit.
    """
    if btc_current is None or btc_current <= 0:
        return None, None

    try:
        mcap = _cg_get("mcap", 0)
        vol_24h = _cg_get("vol_24h", 0)

        # Get Deribit futures OI
        r2 = requests.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=future",
            timeout=10
        )
        futures_oi = sum(f.get("open_interest", 0) for f in r2.json().get("result", []))

        # Get Binance order book depth
        bids, asks = _binance_depth(limit=5)
        depth = sum(v for _, v in bids) + sum(v for _, v in asks) if (bids and asks) else 0

        # Adapted Z-score components
        vol_btc = vol_24h / btc_current if btc_current > 0 else 1
        x1 = min(depth / max(vol_btc, 1), 1.0) * 10

        # X2: Network health (using cached price as proxy)
        x2 = 4.0  # Default neutral

        # X3: Efficiency (price / 200-day SMA as cost basis)
        prices_200 = _cg_get("prices_200")
        if prices_200 and len(prices_200) > 5:
            sma_200 = statistics.mean(prices_200)
            x3 = btc_current / sma_200 if sma_200 > 0 else 1
        else:
            x3 = 1
        x3 = max(min(x3, 5), -5)

        # X4: futures OI / spot volume
        x4 = (futures_oi / vol_24h) if vol_24h > 0 else 1
        x4 = max(min(x4 * 10, 10), -10)

        # X5: Turnover
        x5 = (vol_24h / mcap) * 100 if mcap > 0 else 1
        x5 = max(min(x5 * 2, 10), -10)

        z = 0.6 * x1 + 0.7 * x2 + 1.65 * x3 + 0.3 * x4 + 0.5 * x5

        if z < -5:
            score = 0.85
        elif z < -1:
            score = 0.70
        elif z < 2:
            score = 0.50
        elif z < 5:
            score = 0.30
        else:
            score = 0.15

        return score, {
            "z_score": round(z, 3),
            "x1_liquidity": round(x1, 3),
            "x2_network": round(x2, 3),
            "x3_efficiency": round(x3, 3),
            "x4_oi_ratio": round(x4, 3),
            "x5_turnover": round(x5, 3),
            "zone": "DISTRESS" if z < -1 else "GRAY" if z < 3 else "SAFE"
        }
    except Exception as e:
        print(f"[M31] Error: {e}", file=sys.stderr)
        return None, None


# ──────────────────────────────────────────────────────
# BATCH COMPUTATION
# ──────────────────────────────────────────────────────

def compute_all_institutional(btc_current=None):
    """
    Compute all 12 institutional methods (M20-M31).
    Returns list of (score, detail, name) for successful methods.
    """
    results = {}
    details = {}
    active = 0
    micro_change_flags = {}  # Change detection flags
    micro_deteriorating = False

    # ── Load microstructure state from previous run ──
    micro_prev = _load_micro_cache()
    micro_curr = {}

    # M20-M23: Microstructure (with change detection)
    s, d = calculate_m20_order_book_imbalance(prev=micro_prev.get("m20"))
    if s is not None:
        results["m20_obi"] = s; details["m20_detail"] = d; active += 1
        micro_curr["m20"] = {"obi_norm": d.get("obi_norm", 0.5)}
        if d.get("obi_change") in ("SELL_SURGE",):
            micro_change_flags["obi"] = "SELL_SURGE"
            micro_deteriorating = True
        print(f"  ✓ M20 (OBI): {s:.3f} — OBI={d.get('obi_norm', 0):.3f} | chg={d.get('obi_change', '?')}", file=sys.stderr)

    s, d = calculate_m21_large_trade_flow(prev=micro_prev.get("m21"))
    if s is not None:
        results["m21_trade_flow"] = s; details["m21_detail"] = d; active += 1
        micro_curr["m21"] = {"large_sell_ratio": d.get("large_sell_ratio", 0.5)}
        if d.get("flow_change") in ("SELL_SURGE",):
            micro_change_flags["flow"] = "SELL_SURGE"
            micro_deteriorating = True
        print(f"  ✓ M21 (TradeFlow): {s:.3f} — sell_ratio={d.get('large_sell_ratio', 0):.3f} | chg={d.get('flow_change', '?')}", file=sys.stderr)

    s, d = calculate_m22_spread_momentum(prev=micro_prev.get("m22"))
    if s is not None:
        results["m22_spread"] = s; details["m22_detail"] = d; active += 1
        micro_curr["m22"] = {"spread_bps": d.get("spread_bps", 0)}
        if d.get("spread_change") in ("WIDENING_FAST",):
            micro_change_flags["spread"] = "WIDENING_FAST"
            micro_deteriorating = True
        print(f"  ✓ M22 (Spread): {s:.3f} — spread_bps={d.get('spread_bps', 0):.1f} | chg={d.get('spread_change', '?')}", file=sys.stderr)

    s, d = calculate_m23_liquidity_fractals(prev=micro_prev.get("m23"))
    if s is not None:
        results["m23_liquidity"] = s; details["m23_detail"] = d; active += 1
        micro_curr["m23"] = {"slippage_pct": d.get("slippage_pct", 0)}
        if d.get("liq_change") in ("THINNING_FAST",):
            micro_change_flags["liq"] = "THINNING_FAST"
            micro_deteriorating = True
        print(f"  ✓ M23 (Liquidity): {s:.3f} — slippage={d.get('slippage_pct', 0):.3f}% | chg={d.get('liq_change', '?')}", file=sys.stderr)

    # ── Save current microstructure state for next run ──
    _save_micro_cache(micro_curr)

    # Aggregate micro trend: count deteriorating signals
    micro_trend_score = sum(1 for f in micro_change_flags.values() if f in ("SELL_SURGE", "WIDENING_FAST", "THINNING_FAST")) / 3.0

    # M24-M28: Behavioral/Tail Risk
    s, d = calculate_m24_cape()
    if s is not None:
        results["m24_cape"] = s; details["m24_detail"] = d; active += 1
        print(f"  ✓ M24 (CAPE): {s:.3f} — CAPE={d.get('cape', 0):.2f}", file=sys.stderr)

    s, d = calculate_m25_minsky_moment()
    if s is not None:
        results["m25_minsky"] = s; details["m25_detail"] = d; active += 1
        print(f"  ✓ M25 (Minsky): {s:.3f} — stage={d.get('minsky_stage', '?')}", file=sys.stderr)

    s, d = calculate_m26_kahneman_bias(btc_current)
    if s is not None:
        results["m26_kahneman"] = s; details["m26_detail"] = d; active += 1
        print(f"  ✓ M26 (Kahneman): {s:.3f} — P/L={d.get('pl_pct', 0):.1f}%", file=sys.stderr)

    s, d = calculate_m27_taleb_tail_risk()
    if s is not None:
        results["m27_taleb"] = s; details["m27_detail"] = d; active += 1
        print(f"  ✓ M27 (Taleb): {s:.3f} — tail={d.get('tail_danger', '?')}", file=sys.stderr)

    s, d = calculate_m28_summers_stagnation()
    if s is not None:
        results["m28_summers"] = s; details["m28_detail"] = d; active += 1
        print(f"  ✓ M28 (Summers): {s:.3f} — regime={d.get('regime', '?')}", file=sys.stderr)

    # M29-M31: Debt Cycle
    s, d = calculate_m29_debt_crisis()
    if s is not None:
        results["m29_debt"] = s; details["m29_detail"] = d; active += 1
        print(f"  ✓ M29 (Debt): {s:.3f} — debt_gdp={d.get('debt_gdp_ratio', 0):.2f}", file=sys.stderr)

    s, d = calculate_m30_rajan_fsi()
    if s is not None:
        results["m30_rajan"] = s; details["m30_detail"] = d; active += 1
        print(f"  ✓ M30 (Rajan FSI): {s:.3f} — stability={d.get('stability', '?')}", file=sys.stderr)

    s, d = calculate_m31_altman_zscore(btc_current)
    if s is not None:
        results["m31_altman"] = s; details["m31_detail"] = d; active += 1
        print(f"  ✓ M31 (Altman): {s:.3f} — z={d.get('z_score', 0):.2f}", file=sys.stderr)

    avg_score = sum(results.values()) / len(results) if results else None
    if micro_change_flags:
        print(f"  → Micro change flags: {micro_change_flags} | trend_score={micro_trend_score:.2f} | deteriorating={micro_deteriorating}", file=sys.stderr)
    print(f"  → Institutional methods active: {active}/12 | avg: {avg_score:.3f}", file=sys.stderr)

    return results, details, active, avg_score, micro_change_flags, micro_trend_score, micro_deteriorating


if __name__ == "__main__":
    print("Testing Institutional Methods (M20-M31)...", file=sys.stderr)
    results, details, active, avg, micro_flags, micro_trend, micro_bad = compute_all_institutional(btc_current=95000)
    for k, v in results.items():
        print(f"  {k}: {round(v, 4)}")
    print(f"\nActive: {active}/12 | Average: {round(avg, 4) if avg else 'N/A'}")
    print(f"Micro flags: {micro_flags} | trend={micro_trend:.2f} | bad={micro_bad}")
