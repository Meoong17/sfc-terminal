#!/usr/bin/env python3
"""
multi_timeframe.py — Multi-timeframe feature fusion for BTCUSDT.

Fetches data from Binance at 1h, 4h, 1d, 1w intervals, computes technical
indicators per timeframe, then computes cross-timeframe alignment and
divergence signals.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"
TIMEFRAMES = ["1h", "4h", "1d", "1w"]
LIMIT = 100

# Cache
_cache: Dict[str, Any] = {"data": None, "ts": 0.0}
CACHE_TTL = 60.0  # seconds

# Default fallback
DEFAULT_RESULT: Dict[str, Any] = {
    "alignment_score": 0.0,
    "divergence_detected": False,
    "signals": {},
    "error": None,
}


# ---------------------------------------------------------------------------
# Binance API helpers
# ---------------------------------------------------------------------------

BINANCE_INTERVAL_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


def fetch_klines(symbol: str, interval: str, limit: int = 100) -> List[List]:
    """Fetch kline/candlestick data from Binance public API.

    Returns raw API response (list of lists). Each inner list:
    [open_time, open, high, low, close, volume, close_time, qav, num_trades,
     taker_base_vol, taker_quote_vol, ignore]
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def klines_to_arrays(klines: List[List]) -> Dict[str, np.ndarray]:
    """Convert raw kline data to numpy arrays keyed by field name."""
    arr = np.array(klines, dtype=np.float64)
    return {
        "open": arr[:, 1],
        "high": arr[:, 2],
        "low": arr[:, 3],
        "close": arr[:, 4],
        "volume": arr[:, 5],
    }


# ---------------------------------------------------------------------------
# Indicator computations  (vectorised with numpy)
# ---------------------------------------------------------------------------


def emma(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.empty_like(values)
    alpha = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def compute_ema_cross(close: np.ndarray) -> int:
    """
    EMA 9 vs EMA 21 cross signal.

    Returns:
        +1 if EMA9 > EMA21 (bullish)
        -1 if EMA9 < EMA21 (bearish)
         0 if equal (rare edge case)
    """
    ema9 = emma(close, 9)
    ema21 = emma(close, 21)
    if ema9[-1] > ema21[-1]:
        return 1
    elif ema9[-1] < ema21[-1]:
        return -1
    return 0


def compute_momentum(close: np.ndarray, period: int = 10) -> float:
    """Short-term return over the last `period` bars (fractional, e.g. 0.05 = 5%)."""
    if len(close) < period + 1:
        return 0.0
    return float((close[-1] - close[-period - 1]) / close[-period - 1])


def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index (RSI-14)."""
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> float:
    """Average True Range."""
    if len(high) < period + 1:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    atr = np.mean(tr[-period:])
    return float(atr)


def compute_volatility(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Volatility = ATR / close price (normalised)."""
    atr = compute_atr(high, low, close)
    if close[-1] == 0:
        return 0.0
    return atr / close[-1]


# ---------------------------------------------------------------------------
# Single-timeframe feature extraction
# ---------------------------------------------------------------------------


def extract_features(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute all features for one timeframe.

    Returns dict with keys: trend, momentum, rsi, volatility
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]

    return {
        "trend": compute_ema_cross(close),
        "momentum": compute_momentum(close),
        "rsi": compute_rsi(close),
        "volatility": compute_volatility(high, low, close),
    }


# ---------------------------------------------------------------------------
# Multi-timeframe fusion
# ---------------------------------------------------------------------------


def compute_alignment(signals: Dict[str, Dict[str, Any]]) -> float:
    """
    Compute alignment score: fraction of timeframes agreeing on trend direction.

    Returns float in [-1, +1]:
        +1 = all timeframes bullish
        -1 = all timeframes bearish
         0 = fully mixed
    """
    if not signals:
        return 0.0

    trends = [s["trend"] for s in signals.values()]
    # Exclude neutral (0) from count here — they don't contribute to agreement
    non_neutral = [t for t in trends if t != 0]

    if not non_neutral:
        return 0.0

    # +1 if most are bullish, -1 if most are bearish
    mean_trend = np.mean(non_neutral)
    # Clamp to [-1, 1]
    return float(np.clip(mean_trend, -1.0, 1.0))


def compute_divergence(signals: Dict[str, Dict[str, Any]]) -> bool:
    """
    Detect divergence: short-term (1h, 4h) disagree with longer-term (1d, 1w) trend.

    Short-term is considered divergent when both short timeframes have the
    same sign and the sign differs from the majority of long timeframes.
    """
    short_tfs = ["1h", "4h"]
    long_tfs = ["1d", "1w"]

    short_trends = [
        signals[tf]["trend"] for tf in short_tfs if tf in signals
    ]
    long_trends = [
        signals[tf]["trend"] for tf in long_tfs if tf in signals
    ]

    # Need at least one short and one long to compare
    if not short_trends or not long_trends:
        return False

    # Consensus within each group (mean then sign)
    short_consensus = np.sign(np.mean(short_trends))
    long_consensus = np.sign(np.mean(long_trends))

    # Divergence when both have clear direction and they disagree
    return short_consensus != 0 and long_consensus != 0 and short_consensus != long_consensus


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------


def multi_timeframe_fusion(
    symbol: str = SYMBOL,
    timeframes: Optional[List[str]] = None,
    limit: int = LIMIT,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Fetch data from Binance and compute multi-timeframe fusion result.

    Args:
        symbol: Trading pair (e.g. BTCUSDT).
        timeframes: List of intervals to fetch (default: 1h, 4h, 1d, 1w).
        limit: Number of candles per timeframe.
        use_cache: Whether to use the 60-second in-memory cache.

    Returns:
        Dict with keys:
            alignment_score: float in [-1, +1]
            divergence_detected: bool
            signals: {tf_name: {trend, momentum, rsi, volatility}}
            error: str | None
    """
    global _cache

    if timeframes is None:
        timeframes = TIMEFRAMES

    # --- Cache check ---
    cache_key = (symbol, tuple(timeframes), limit)
    if use_cache:
        cached = _cache["data"]
        cache_ts = _cache["ts"]
        if cached is not None and (time.time() - cache_ts) < CACHE_TTL:
            # Verify cache is for the same parameters (optional safety)
            if cached.get("_cache_key") == cache_key:
                logger.debug("Returning cached multi-timeframe result")
                return cached

    # --- Fetch ---
    signals: Dict[str, Any] = {}
    fetch_errors: List[str] = []

    for tf in timeframes:
        try:
            raw = fetch_klines(symbol, BINANCE_INTERVAL_MAP[tf], limit)
            data = klines_to_arrays(raw)
            features = extract_features(data)
            signals[tf] = features
            logger.debug("Fetched and computed %s OK", tf)
        except Exception as exc:
            msg = f"{tf}: {exc}"
            fetch_errors.append(msg)
            logger.warning("Failed to fetch/compute %s: %s", tf, exc)

    # --- Fusion ---
    alignment_score = compute_alignment(signals)
    divergence_detected = compute_divergence(signals)

    result: Dict[str, Any] = {
        "alignment_score": alignment_score,
        "divergence_detected": divergence_detected,
        "signals": signals,
        "error": "; ".join(fetch_errors) if fetch_errors else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_cache_key": cache_key,
    }

    # --- Cache ---
    _cache["data"] = result
    _cache["ts"] = time.time()

    return result


# ---------------------------------------------------------------------------
# Graceful fallback wrapper
# ---------------------------------------------------------------------------


def safe_multi_timeframe_fusion(**kwargs) -> Dict[str, Any]:
    """Wrapper that catches all exceptions and returns a default dict on failure.

    Returns:
        On success: same as multi_timeframe_fusion()
        On failure: dict with alignment_score=0.0, divergence_detected=False,
                    signals={}, and error description.
    """
    try:
        return multi_timeframe_fusion(**kwargs)
    except Exception as exc:
        logger.exception("Unhandled error in multi_timeframe_fusion")
        return {
            **DEFAULT_RESULT,
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone entry point: fetch, compute, and print the fusion result."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 65)
    print("Multi-timeframe Feature Fusion — BTCUSDT")
    print("=" * 65)

    result = multi_timeframe_fusion()
    signals = result.get("signals", {})

    print(f"\nAlignment score:  {result['alignment_score']:+.4f}   "
          f"(range -1 to +1, +1 = fully aligned bullish)")
    print(f"Divergence:       {result['divergence_detected']}")
    print(f"Error:            {result.get('error', 'none')}")
    print(f"Timestamp:        {result.get('timestamp', 'N/A')}")
    print()

    if signals:
        print(f"{'TF':>5}  {'Trend':>7}  {'Momentum':>10}  {'RSI':>8}  {'Volatility':>12}")
        print("-" * 50)
        for tf in TIMEFRAMES:
            s = signals.get(tf)
            if s is None:
                print(f"{tf:>5}  {'N/A':>7}  {'N/A':>10}  {'N/A':>8}  {'N/A':>12}")
                continue
            trend_label = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(s["trend"], "?")
            print(
                f"{tf:>5}  {trend_label:>7}  {s['momentum']:>+10.4f}  "
                f"{s['rsi']:>8.2f}  {s['volatility']:>12.4f}"
            )
    else:
        print("No signals computed.")

    print("\n" + "=" * 65)
    print("Done.")
    return


if __name__ == "__main__":
    main()
