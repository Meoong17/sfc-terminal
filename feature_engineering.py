"""
feature_engineering.py — Fetches daily BTCUSDT klines from Binance and computes technical indicators.

All prints go to stderr. The module is importable from collect.py and exposes:
    get_features() -> dict[str, float]  (empty dict on any failure)

Indicators computed (reduced from 25+ to ~17 quality features):
    Momentum:   RSI-7, RSI-14, Stochastic K/D
    Trend:      MACD line/signal/histogram, EMA21, EMA21-EMA200 crossover, EMA200 slope
    Volatility: ATR, Bollinger Band Width, %B, Realized Volatility (30d)
    Volume:     VWAP, OBV, Chaikin Money Flow
"""

import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import ta

# ---------- cache ----------
_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 300  # seconds

# ---------- helpers ----------


def _eprint(*args, **kwargs):
    """Print to stderr so stdout stays clean for JSON / structured output."""
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _normalize_01(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp and min-max scale value to [0, 1]."""
    if hi - lo < 1e-12:
        return 0.5
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo)


def _normalize_n11(value: float, bound: float = 100.0) -> float:
    """Clamp and scale value to [-1, 1] given a symmetric bound."""
    clipped = max(-bound, min(bound, value))
    if bound < 1e-12:
        return 0.0
    return clipped / bound


def _symmetric_bound(series: pd.Series, percentile: float = 95.0) -> float:
    """Compute a symmetric bound for a series using the given percentile of absolute values."""
    if series.empty:
        return 100.0
    vals = series.dropna().abs().values
    if len(vals) == 0:
        return 100.0
    return float(np.percentile(vals, percentile)) + 1e-12


def _normalize_series_last(series: pd.Series) -> float:
    """Min-max normalize the last value of a series against its own range -> [0, 1]."""
    vals = series.dropna()
    if len(vals) < 2:
        return 0.5
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax - vmin < 1e-12:
        return 0.5
    return (float(vals.iloc[-1]) - vmin) / (vmax - vmin)


# ---------- fetch ----------


def _fetch_klines() -> pd.DataFrame | None:
    """Fetch 365 daily BTCUSDT candles from Binance. Returns a DataFrame or None."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 365}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _eprint(f"[feature_engineering] Binance fetch failed: {exc}")
        return None

    if not isinstance(data, list) or len(data) == 0:
        _eprint("[feature_engineering] Empty response from Binance")
        return None

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _get_data() -> pd.DataFrame | None:
    """Return cached DataFrame or fetch fresh one."""
    global _cache
    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]
    df = _fetch_klines()
    if df is not None:
        _cache = {"data": df, "ts": now}
    return df


# ---------- indicators ----------


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute technical indicators (reduced set — 17 quality features)."""
    features: dict[str, float] = {}

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    last_c = float(c.iloc[-1])

    def _safe(key: str, func):
        """Wrapper: compute an indicator, catch any exception, log to stderr, fallback to 0.0."""
        try:
            val = func()
            if val is not None:
                features[key] = val
                return
        except Exception as exc:
            _eprint(f"[feature_engineering] Indicator '{key}' failed: {exc}")
        features[key] = 0.0

    # ---- Momentum (4) ----
    # RSI-7: short-term momentum
    _safe("rsi_7", lambda: _normalize_01(float(
        ta.momentum.RSIIndicator(close=c, window=7).rsi().iloc[-1])))
    # RSI-14: medium-term momentum
    _safe("rsi_14", lambda: _normalize_01(float(
        ta.momentum.RSIIndicator(close=c, window=14).rsi().iloc[-1])))

    # Stochastic K/D: overbought/oversold with different sensitivity
    _safe("stoch_k", lambda: _normalize_01(float(
        ta.momentum.StochasticOscillator(
            high=h, low=l, close=c, window=14, smooth_window=3
        ).stoch().iloc[-1])))
    _safe("stoch_d", lambda: _normalize_01(float(
        ta.momentum.StochasticOscillator(
            high=h, low=l, close=c, window=14, smooth_window=3
        ).stoch_signal().iloc[-1])))

    # ---- Trend (5) ----
    macd_obj = ta.trend.MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    _safe("macd_line", lambda: _normalize_n11(
        float(macd_obj.macd().iloc[-1]), _symmetric_bound(macd_obj.macd())))
    _safe("macd_signal", lambda: _normalize_n11(
        float(macd_obj.macd_signal().iloc[-1]), _symmetric_bound(macd_obj.macd_signal())))
    _safe("macd_histogram", lambda: _normalize_n11(
        float(macd_obj.macd_diff().iloc[-1]), _symmetric_bound(macd_obj.macd_diff())))

    # EMA21 (short-term trend) + EMA21-EMA200 crossover
    ema21 = ta.trend.EMAIndicator(close=c, window=21).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close=c, window=min(200, len(c))).ema_indicator()

    _safe("ema21_price_ratio", lambda: max(0.0, min(1.0, (
        float(ema21.iloc[-1]) / last_c - 0.8) / 0.4)) if last_c > 0 else 0.5)

    # EMA21-EMA200 crossover: positive = bullish trend, negative = bearish
    _safe("ema_crossover", lambda: max(-1.0, min(1.0, (
        float(ema21.iloc[-1]) - float(ema200.iloc[-1])) / last_c * 10.0)) if last_c > 0 else 0.0)

    # EMA200 slope (rate of change over 5 periods): 0 to 1, high = steep uptrend
    if len(ema200.dropna()) >= 6:
        ema200_vals = ema200.dropna().values
        slope = (ema200_vals[-1] - ema200_vals[-6]) / ema200_vals[-6] * 100 if ema200_vals[-6] > 0 else 0
        _safe("ema200_slope", lambda: max(0.0, min(1.0, (slope + 2.0) / 4.0)))
    else:
        features["ema200_slope"] = 0.5

    # ---- Volatility (4) ----
    _safe("atr", lambda: max(0.0, min(1.0,
        float(ta.volatility.AverageTrueRange(
            high=h, low=l, close=c, window=14
        ).average_true_range().iloc[-1]) / last_c * 20.0)) if last_c > 0 else 0.0)

    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    _safe("bb_width", lambda: max(0.0, min(1.0,
        float(bb.bollinger_wband().iloc[-1]) / last_c * 20.0)) if last_c > 0 else 0.0)
    _safe("bb_pct_b", lambda: _normalize_01(float(bb.bollinger_pband().iloc[-1])))

    # Realized Volatility (30-day rolling std of daily returns)
    returns = c.pct_change().dropna()
    if len(returns) >= 30:
        realized_vol = returns.tail(30).std() * np.sqrt(365)  # annualized
        _safe("realized_vol", lambda: max(0.0, min(1.0,
            float(realized_vol) / 2.0)))  # 100% annualized → 0.5, 200% → 1.0
    else:
        features["realized_vol"] = 0.5

    # ---- Volume (3) ----
    _safe("vwap", lambda: max(0.0, min(1.0, (
        last_c / float(ta.volume.VolumeWeightedAveragePrice(
            high=h, low=l, close=c, volume=v, window=14
        ).volume_weighted_average_price().iloc[-1]) - 0.95) / 0.1))
        if last_c > 0 and float(
            ta.volume.VolumeWeightedAveragePrice(
                high=h, low=l, close=c, volume=v, window=14
            ).volume_weighted_average_price().iloc[-1]) > 0 else 0.5)

    _safe("obv", lambda: _normalize_series_last(
        ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()))

    _safe("cmf", lambda: max(-1.0, min(1.0, float(
        ta.volume.ChaikinMoneyFlowIndicator(
            high=h, low=l, close=c, volume=v, window=20
        ).chaikin_money_flow().iloc[-1]))))

    return features


# ---------- public API ----------


def get_features() -> dict[str, float]:
    """
    Fetch BTCUSDT data and compute quality technical indicators.

    Returns:
        dict[str, float]: Feature name -> normalized value.
        Returns an empty dict on any failure (network, parse, compute).
    """
    try:
        df = _get_data()
        if df is None or df.empty:
            _eprint("[feature_engineering] No data available")
            return {}
        return _compute_features(df)
    except Exception as exc:
        _eprint(f"[feature_engineering] Unexpected error: {exc}")
        return {}


# ---------- main / self-test ----------


def main():
    """Fetch, compute, and print features (to stderr), then print a summary to stderr."""
    features = get_features()
    count = len(features)
    _eprint(f"[feature_engineering] Computed {count} features")
    for k, v in sorted(features.items()):
        _eprint(f"  {k}: {v:.4f}")
    if count == 0:
        _eprint("[feature_engineering] WARNING: No features computed")
    _eprint(f"[feature_engineering] JSON: {json.dumps(features)}")
    return features


if __name__ == "__main__":
    main()
