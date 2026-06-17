#!/usr/bin/env python3
"""
fetch_historical_btc.py — Download historical BTC market data + compute feature vectors
====================================================================================
Fetches:
  - Binance daily klines (BTCUSDT, 1d) from 2021 to present (~1500+ candles)
  - Fear & Greed Index historical data (Alternative.me)
  - Computes 39-dim feature vectors compatible with Mamba training

Output:
  - .historical_features.npy — numpy array of (n_days, 39) feature vectors
  - .historical_dates.npy — corresponding dates as ISO strings
  - historical_data.json — raw combined data (for inspection)

Usage:
  cd /home/ubuntu/sfc && python3 fetch_historical_btc.py
"""

import json, os, sys, time, math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_DIR = Path(__file__).parent
OUTPUT_FEATURES = BASE_DIR / ".historical_features.npy"
OUTPUT_DATES = BASE_DIR / ".historical_dates.npy"
OUTPUT_RAW = BASE_DIR / "historical_data.json"
CACHE_FILE = BASE_DIR / ".historical_btc_cache.json"

# ── Config ──
START_YEAR = 2021        # Fetch from Jan 1, 2021
BINANCE_LIMIT = 1000     # Max candles per Binance API call


def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_ts=None, limit=1000):
    """Fetch klines from Binance. Returns list of candles."""
    all_candles = []
    while True:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_ts:
            url += f"&startTime={int(start_ts * 1000)}"

        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urlopen(req, timeout=30)
            candles = json.loads(resp.read())
        except Exception as e:
            print(f"[Fetch] Binance error: {e}", file=sys.stderr)
            break

        if not candles:
            break

        all_candles.extend(candles)
        print(f"  Fetched {len(candles)} candles (total: {len(all_candles)})", file=sys.stderr)

        if len(candles) < limit:
            break

        # Next batch: start after last candle's open time
        last_ts = candles[-1][0] / 1000
        start_ts = last_ts + 86400  # 1 day later
        time.sleep(0.3)  # Rate limit

    return all_candles


def fetch_fng_historical(limit=2000):
    """Fetch Fear & Greed Index historical data."""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urlopen(req, timeout=15)
        data = json.loads(resp.read())
        result = {}
        for entry in data.get("data", []):
            ts = int(entry["timestamp"])
            result[ts] = int(entry["value"])
        print(f"[Fetch] FNG: {len(result)} data points", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[Fetch] FNG error: {e}", file=sys.stderr)
        return {}


def ohlcv_to_features(candles, fng_data):
    """
    Convert Binance OHLCV candles + FNG data to 39-dim feature vectors.
    Missing features default to 0.0 (handled by Mamba's get_val()).
    """
    feature_vectors = []
    dates = []

    for c in candles:
        open_ts = c[0] / 1000  # ms to seconds
        dt = datetime.fromtimestamp(open_ts, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%dT%H:%M:%S")

        o = float(c[1])
        h = float(c[2])
        l_ = float(c[3])
        c_close = float(c[4])
        vol = float(c[5])

        # --- Compute features ---

        # 1. Price & Market (5 features)
        btc_price = c_close
        btc_norm = btc_price / 100000.0

        # Find previous close from previous candle
        # We'll compute btc_24h_change from the previous stored value
        # For now, use the previous candle if available
        features = [0.0] * 39

        features[0] = btc_price / 100000.0  # btc/100k
        # btc_24h will be filled during sequence building from adjacent candles
        features[1] = 0.0
        features[2] = 0.0  # market cap - not available from klines alone
        features[3] = 0.0  # dominance
        # DVOL - estimate from high/low range
        daily_range = (h - l_) / o if o > 0 else 0
        features[4] = min(daily_range * 5, 1.0)  # rough DVOL proxy

        # 2. Technical Indicators (5 features)
        features[5] = 0.0  # RSI - computed in build_features

        # Put/ Call ratios - not available historically
        features[6] = 0.0
        features[7] = 0.0

        # F&G - look up by date
        date_key = int(open_ts)
        fng_val = 50  # default neutral
        if date_key in fng_data:
            fng_val = fng_data[date_key]
        # Also try within 1 day tolerance
        else:
            for fng_ts, fng_v in fng_data.items():
                if abs(fng_ts - date_key) < 86400:
                    fng_val = fng_v
                    break
        features[8] = fng_val / 100.0

        features[9] = 0.0  # sopr_proxy

        # 3. On-chain & Risk (5 features)
        features[10] = 0.0  # cascade_risk
        features[11] = 0.0  # liq_density
        features[12] = 0.0  # liq_mod
        features[13] = 0.0  # sopr_score
        features[14] = 0.0  # regime_prob

        # 4. Macro (4 features)
        features[15] = 0.0  # m2_yoy
        features[16] = 0.0  # dxy
        features[17] = 0.0  # transition_risk
        features[18] = 0.0  # dv_sfc

        # 5. SFC State (5 features)
        features[19] = 0.0  # sfc_base
        features[20] = 0.0  # sfc_effective
        features[21] = 0.0  # zone (NORMAL=0)
        features[22] = 0.25  # regime (NORMAL)
        features[23] = 0.0  # phi

        # 6. SFC Factors (5 features)
        features[24] = 0.0  # Lt
        features[25] = 0.0  # St
        features[26] = 0.0  # Rt
        features[27] = 0.0  # Ft
        features[28] = 0.0  # Sc

        # 7. Method Ensemble (6 features)
        features[29] = 0.0  # method_agreement
        features[30] = 0.0  # composite_confidence
        features[31] = 0.0  # m1_klr
        features[32] = 0.0  # m2_logit
        features[33] = 0.0  # m4_ewc
        features[34] = 0.0  # m5_qreg

        # 8. Q10 On-Chain (4 features)
        features[35] = 0.0  # whale_pressure
        features[36] = 0.0  # onchain_value
        features[37] = 0.0  # buying_power
        features[38] = 0.0  # market_structure

        feature_vectors.append(features)
        dates.append(date_str)

    return np.array(feature_vectors, dtype=np.float32), dates


def compute_returns(vectors, dates):
    """
    Post-process: compute btc_24h_change from adjacent feature vectors.
    btc_price is at feature[0] (scaled by /100k).
    """
    for i in range(1, len(vectors)):
        prev_btc = vectors[i-1][0] * 100000.0  # de-scale
        curr_btc = vectors[i][0] * 100000.0
        if prev_btc > 0:
            change_pct = (curr_btc - prev_btc) / prev_btc
            vectors[i][1] = change_pct / 20.0  # normalize to [-1,1] range
    return vectors


def compute_rsi(vectors, period=14):
    """Compute RSI from close prices and store in feature[5]."""
    prices = vectors[:, 0] * 100000.0  # de-scale BTC prices
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = np.full(len(prices), 50.0)  # default neutral

    if avg_loss == 0:
        rsi_values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    # Store normalized RSI
    vectors[:, 5] = rsi_values / 100.0
    return vectors


def compute_historical_volatility(vectors, window=20):
    """Compute historical volatility proxy from daily returns."""
    prices = vectors[:, 0] * 100000.0
    log_returns = np.diff(np.log(prices + 1e-10))
    vol = np.zeros(len(prices))
    vol[:window] = 0.2  # default 20% vol

    for i in range(window, len(prices)):
        vol[i] = np.std(log_returns[i-window:i]) * math.sqrt(365)

    # Normalize to ~0-1 range (typical crypto vol: 30%-100%)
    vectors[:, 4] = np.clip(vol / 100.0, 0.0, 1.0)
    return vectors


def main():
    print("=" * 60, file=sys.stderr)
    print("SFC Historical BTC Data Fetcher", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # 1. Fetch F&G data first
    print("\n[1/4] Fetching Fear & Greed historical...", file=sys.stderr)
    fng_data = fetch_fng_historical()
    if not fng_data:
        print("  ⚠ No F&G data — continuing with defaults", file=sys.stderr)

    # 2. Fetch Binance klines
    print("\n[2/4] Fetching Binance BTCUSDT daily klines...", file=sys.stderr)
    start_dt = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
    start_ts = start_dt.timestamp()
    candles = fetch_binance_klines(start_ts=start_ts)

    if not candles:
        print("❌ No Binance data fetched!", file=sys.stderr)
        return

    print(f"  Total candles: {len(candles)}", file=sys.stderr)
    first_dt = datetime.fromtimestamp(candles[0][0]/1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc)
    print(f"  Range: {first_dt.date()} to {last_dt.date()}", file=sys.stderr)

    # 3. Convert to feature vectors
    print("\n[3/4] Computing feature vectors...", file=sys.stderr)
    vectors, dates = ohlcv_to_features(candles, fng_data)
    print(f"  Initial vectors: {vectors.shape}", file=sys.stderr)

    # Post-process: compute returns, RSI, volatility
    vectors = compute_returns(vectors, dates)
    vectors = compute_rsi(vectors)
    vectors = compute_historical_volatility(vectors)

    # Compute btc_mcap proxy from volume + typical ratios
    # Use volume * 20 as rough mcap proxy (very rough)
    for i, c in enumerate(candles):
        vol = float(c[5])
        close = float(c[4])
        # Estimate mcap: typical BTC circ ~19.5M
        vectors[i][2] = (close * 19_500_000) / 2e12  # normalize

    # Save
    print(f"\n[4/4] Saving...", file=sys.stderr)
    np.save(OUTPUT_FEATURES, vectors)
    np.save(OUTPUT_DATES, np.array(dates, dtype=object))

    print(f"  Features: {OUTPUT_FEATURES}", file=sys.stderr)
    print(f"  Dates:    {OUTPUT_DATES}", file=sys.stderr)
    print(f"  Shape:    {vectors.shape}", file=sys.stderr)

    # Also save raw combined data for inspection
    raw_data = []
    for i, c in enumerate(candles):
        dt = datetime.fromtimestamp(c[0]/1000, tz=timezone.utc)
        date_key = int(c[0] / 1000)
        fng_val = fng_data.get(date_key, 50)
        raw_data.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
            "fng": fng_val,
        })
    with open(OUTPUT_RAW, "w") as f:
        json.dump(raw_data, f, indent=1)
    print(f"  Raw data: {OUTPUT_RAW}", file=sys.stderr)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("✅ Historical BTC data ready!", file=sys.stderr)
    print(f"   {len(vectors)} daily snapshots ({first_dt.date()} to {last_dt.date()})", file=sys.stderr)
    print(f"   {39} features per snapshot", file=sys.stderr)
    print(f"   FNG coverage: {sum(1 for v in fng_data.values() if v != 50)}/{len(candles)} days", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
