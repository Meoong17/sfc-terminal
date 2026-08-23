"""
market_data_fetcher.py — Cross-asset data for GNN Systemic Risk (M69)
=======================================================================

Fetches real return/volatility/momentum data for ETH, Gold, and SPX so
calculate_systemic_risk() in risk/gnn_module.py can run on live market
conditions instead of the hardcoded _default_simulated_data() fallback.

Sources:
    ETH  — Binance REST API (no key required, same exchange as binance_ws.py)
    Gold — Binance PAXGUSDT daily klines (no key required; PAX Gold tracks
           spot XAU 1:1), falls back to Twelve Data XAU/USD (TWELVEDATA_KEY)
    SPX  — Twelve Data (key required: TWELVEDATA_KEY), falls back to
           Alpha Vantage (key required: ALPHAVANTAGE_KEY) if Twelve Data
           fails or is rate-limited

Each fetcher returns a dict: {"return": float, "volatility": float, "momentum": float}
  - return:     most recent daily % change, as a decimal (e.g. 0.02 = +2%)
  - volatility: rolling stdev of recent daily returns, as a decimal
  - momentum:   short-term trend (recent avg return vs longer avg), as a decimal
This matches the format SystemicRiskCalculator expects (see gnn_module.py
ASSET_NAMES / calculate_systemic_risk signature) and is scale-consistent
with how BTC/DXY data is already computed elsewhere in collect.py.

All fetchers are wrapped in try/except and cache results for CACHE_TTL
seconds — a genuine network failure or missing key returns None (not a
guessed value), and the caller (collect.py) is responsible for deciding
whether to fall back to simulated data or skip the asset. Silently
guessing a plausible-looking number here would just reintroduce a
differently-shaped version of the same "looks real but isn't" problem
this module exists to fix.
"""
import os
import time
import requests

CACHE_TTL = 900  # 15 min — these assets don't need BTC-pipeline-level freshness
_CACHE = {}


def _cached(key):
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(key, data):
    _CACHE[key] = {"data": data, "ts": time.time()}


def _compute_rvm(closes):
    """
    Compute {return, volatility, momentum} from a list of closing prices
    (oldest first). Requires at least 8 points for a stable momentum
    signal; returns None if insufficient data.
    """
    if not closes or len(closes) < 8:
        return None
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1]
    ]
    if len(returns) < 7:
        return None

    latest_return = returns[-1]

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    volatility = variance ** 0.5

    # Momentum: short window (last 3) vs longer window (last 7) average
    short_avg = sum(returns[-3:]) / 3
    long_avg = sum(returns[-7:]) / 7
    momentum = short_avg - long_avg

    return {
        "return": round(latest_return, 6),
        "volatility": round(volatility, 6),
        "momentum": round(momentum, 6),
    }


def fetch_eth_data():
    """
    ETH daily klines from Binance REST API (public, no key required —
    same exchange binance_ws.py already streams BTC from).
    """
    cached = _cached("eth")
    if cached is not None:
        return cached

    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": "1d", "limit": 10},
            timeout=10,
        )
        r.raise_for_status()
        klines = r.json()
        # kline[4] = close price, as string
        closes = [float(k[4]) for k in klines]
        result = _compute_rvm(closes)
        _set_cache("eth", result)
        return result
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"[MarketData] ETH fetch failed: {e}", file=__import__("sys").stderr)
        return None


def fetch_gold_data():
    """
    Gold spot price history, no key required.

    Primary: Binance PAXGUSDT daily klines (PAX Gold is a 1:1 gold-backed
    token, so its price tracks spot XAU closely; public endpoint, same
    pattern as fetch_eth_data). Fallback: Twelve Data XAU/USD spot when a
    TWELVEDATA_KEY is present. The historical free tier only exposes the
    current spot price, so this builds a rolling window from repeated calls
    over time via an on-disk cache rather than one historical API call —
    see _rolling_gold_cache below.
    """
    cached = _cached("gold")
    if cached is not None:
        return cached

    # Primary: Binance PAXGUSDT daily klines (no key).
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "PAXGUSDT", "interval": "1d", "limit": 10},
            timeout=10,
        )
        r.raise_for_status()
        klines = r.json()
        closes = [float(k[4]) for k in klines]
        result = _compute_rvm(closes)
        if result is not None:
            _set_cache("gold", result)
            return result
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"[MarketData] Gold PAXG fetch failed: {e}", file=__import__("sys").stderr)

    # Fallback: Twelve Data XAU/USD spot (if key present).
    try:
        td_key = os.getenv("TWELVEDATA_KEY", "")
        if td_key:
            r = requests.get(
                "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": td_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            price = data.get("price")
            if price is not None:
                closes = _update_rolling_series("gold", float(price))
                result = _compute_rvm(closes)
                if result is not None:
                    _set_cache("gold", result)
                    return result
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[MarketData] Gold TwelveData fetch failed: {e}", file=__import__("sys").stderr)

    return None



def fetch_spx_data():
    """
    SPX (S&P 500) from Twelve Data, falling back to Alpha Vantage if
    Twelve Data fails (both have restrictive free-tier rate limits, so
    having a fallback matters more here than for ETH/Gold).
    """
    cached = _cached("spx")
    if cached is not None:
        return cached

    result = _fetch_spx_twelvedata()
    if result is None:
        result = _fetch_spx_alphavantage()

    if result is not None:
        _set_cache("spx", result)
    return result


def _fetch_spx_twelvedata():
    key = os.getenv("TWELVEDATA_KEY", "")
    if not key:
        return None
    # SPX index symbol is only available on paid Twelve Data plans (Grow/Venture);
    # on the free tier it returns 404 ("This symbol is available starting with the
    # Grow or Venture plan"). SPY (S&P 500 ETF) is a close proxy and works on the
    # free plan — the same proxy the Alpha Vantage fallback already uses. Try the
    # exact index first (in case the plan is upgraded), then fall back to SPY.
    for symbol in ("SPX", "SPY"):
        try:
            r = requests.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": symbol, "interval": "1day", "outputsize": 10, "apikey": key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            values = data.get("values")
            if not values:
                print(f"[MarketData] Twelve Data {symbol}: no values in response ({data.get('message', data.get('status'))})",
                      file=__import__("sys").stderr)
                continue
            # Twelve Data returns newest-first — reverse to oldest-first
            closes = [float(v["close"]) for v in reversed(values)]
            return _compute_rvm(closes)
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"[MarketData] Twelve Data {symbol} fetch failed: {e}", file=__import__("sys").stderr)
            continue
    return None


def _fetch_spx_alphavantage():
    key = os.getenv("ALPHAVANTAGE_KEY", "")
    if not key:
        return None
    try:
        # Alpha Vantage has no direct SPX index endpoint on the free tier;
        # SPY (S&P 500 ETF) tracks the index closely enough for a
        # return/volatility/momentum proxy at this resolution.
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_DAILY", "symbol": "SPY", "apikey": key, "outputsize": "compact"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        series = data.get("Time Series (Daily)")
        if not series:
            print(f"[MarketData] Alpha Vantage SPX: no series in response ({data.get('Note', data.get('Information'))})",
                  file=__import__("sys").stderr)
            return None
        dates_sorted = sorted(series.keys())[-10:]
        closes = [float(series[d]["4. close"]) for d in dates_sorted]
        return _compute_rvm(closes)
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[MarketData] Alpha Vantage SPX fetch failed: {e}", file=__import__("sys").stderr)
        return None


# ── Rolling series cache for single-point APIs (Gold) ──
# GoldAPI's free tier returns only the current spot price, not a
# historical series. To compute volatility/momentum we build our own
# rolling window on disk, appended to once per call (i.e. once per
# collect.py cycle), decayed to the most recent N points.
def _rolling_series_path(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f".{name}_rolling.json")


def _update_rolling_series(name, latest_price, max_points=10):
    import json
    path = _rolling_series_path(name)
    try:
        with open(path) as f:
            series = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        series = []

    series.append(latest_price)
    series = series[-max_points:]

    try:
        with open(path, "w") as f:
            json.dump(series, f)
    except OSError:
        pass

    return series


def fetch_all_cross_asset_data(btc_return=None, btc_volatility=None, btc_momentum=None, dxy_price=None):
    """
    Convenience wrapper: fetch ETH/Gold/SPX and combine with BTC/DXY data
    the caller already has (collect.py computes these from its own BTC/DXY
    pipeline — no need to refetch them here).

    Args:
        btc_return, btc_volatility, btc_momentum: pre-computed BTC values
            from collect.py's existing pipeline (chg, dvol, etc.)
        dxy_price: current DXY level — used to build a rolling series the
            same way Gold is, since collect.py only tracks the current
            DXY level, not a return/volatility/momentum triple.

    Returns:
        dict with keys btc_data, eth_data, spx_data, gold_data, dxy_data —
        each either a {"return","volatility","momentum"} dict or None if
        that asset's data couldn't be obtained. Passing None for any asset
        to calculate_systemic_risk() makes it fall back to that asset's
        simulated default individually (see gnn_module.py) rather than
        failing the whole calculation — so partial real data is still an
        improvement over the all-simulated baseline.
    """
    btc_data = None
    if btc_return is not None and btc_volatility is not None and btc_momentum is not None:
        btc_data = {"return": btc_return, "volatility": btc_volatility, "momentum": btc_momentum}

    dxy_data = None
    if dxy_price is not None:
        dxy_closes = _update_rolling_series("dxy", dxy_price)
        dxy_data = _compute_rvm(dxy_closes)

    return {
        "btc_data": btc_data,
        "eth_data": fetch_eth_data(),
        "spx_data": fetch_spx_data(),
        "gold_data": fetch_gold_data(),
        "dxy_data": dxy_data,
    }
