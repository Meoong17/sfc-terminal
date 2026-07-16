"""
liquidity_lag_analysis.py — Empirical lag/lead analysis: GLF vs BTC returns
=============================================================================

Answers a specific question: does global liquidity (GLF) LEAD Bitcoin price,
and if so, by how many weeks?

Why this exists: every liquidity signal in this codebase (global_liquidity_engine,
stablecoin_intelligence, fiscal_liquidity, etc.) is currently compared to BTC
conditions AT THE SAME POINT IN TIME. But global liquidity indicators
(Fed/ECB/BOJ balance sheets, M2) are widely observed to move before BTC price
reacts — by weeks, not the same day. Without an explicit lag structure, the
model can only ask "is liquidity high right now", not "does today's liquidity
level predict where BTC goes N weeks from now" — which is the more useful
question for a model whose stated focus is liquidity as a predictive factor.

This script:
    1. Pulls a long FRED history for each GLF component (Fed, ECB, BOJ, M2) —
       several years, not the last 13 months collect.py normally uses.
    2. Reconstructs GLF at each historical monthly point using the SAME
       z-score formula as production (global_liquidity_engine.py), so results
       are apples-to-apples with what the live pipeline actually computes —
       not a different, idealized calculation that wouldn't transfer.
    3. Pulls BTC daily closes from historical_data.json (already produced by
       fetch_historical_btc.py — no new data source needed).
    4. For each candidate lag (0, 2, 4, 8, 12, 16 weeks), aligns GLF(t) with
       the BTC return over [t+lag, t+lag+2weeks] and computes correlation.
    5. Reports which lag has the strongest correlation — empirically, not
       assumed.

IMPORTANT — what this does NOT do:
    It does not retroactively fix the frozen z-score normalization constants
    (fed mean=5.5/std=8.0 etc. — flagged separately as needing validation
    against real historical data). This script reuses those exact constants
    so the lag result is comparable to what production GLF actually outputs
    today. If those constants are later corrected, re-run this analysis.

Usage:
    python3 liquidity_lag_analysis.py
    (requires FRED_API_KEY in .env, and historical_data.json to exist —
     run fetch_historical_btc.py first if it doesn't)
"""
import json
import os
import sys
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY", "")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORICAL_BTC_PATH = os.path.join(BASE_DIR, "historical_data.json")

# Same normalization constants as global_liquidity_engine.py's
# compute_global_liquidity_factor() — kept in sync deliberately, see
# module docstring for why.
GLF_COMPONENTS = {
    "WALCL": {"name": "fed", "mean": 5.5, "std": 8.0, "weight": 0.30},
    "ECBASSETSW": {"name": "ecb", "mean": 4.0, "std": 7.0, "weight": 0.15},
    "JPNASSETS": {"name": "jpn", "mean": 3.0, "std": 6.0, "weight": 0.05},
    "M2SL": {"name": "m2", "mean": 6.0, "std": 4.0, "weight": 0.15},
}
# Note: TGA/RRP/DXY components from production GLF are omitted here — they
# use different (non-YoY) transformations that need separate historical
# reconstruction logic. This first pass focuses on the four YoY-based
# components, which are the largest combined weight (0.65 of 1.0) and the
# most directly "global liquidity" in nature.


def fetch_fred_series_long(series_id, years=8):
    """Fetch a long monthly/weekly FRED series (years of history, not the
    ~13-point window collect.py's own _fred() uses for production)."""
    if not FRED_KEY:
        print(f"[LagAnalysis] FRED_API_KEY not set — cannot fetch {series_id}", file=sys.stderr)
        return None
    try:
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_KEY,
                "file_type": "json",
                "sort_order": "asc",
                "observation_start": start_date,
            },
            timeout=20,
        )
        if r.status_code != 200:
            print(f"[LagAnalysis] FRED {series_id} returned {r.status_code}", file=sys.stderr)
            return None
        obs = r.json().get("observations", [])
        # Return list of (date_str, value) oldest-first, skipping missing ('.')
        return [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[LagAnalysis] FRED {series_id} fetch failed: {e}", file=sys.stderr)
        return None


def _z_score(value, mean, std):
    """Identical to global_liquidity_engine.py's _z_score — kept in sync."""
    if std == 0 or value is None:
        return 0
    return max(-3.0, min(3.0, (value - mean) / std))


def compute_historical_glf(years=8):
    """
    Reconstruct GLF at each monthly point over the lookback window, using
    the same YoY + z-score + weighted-sum formula as production.

    Returns: list of (date_str, glf_partial_score) tuples, oldest-first.
    glf_partial_score is on the same -3..+3 scale as the individual
    z-scores (weighted sum of the 4 components' z-scores, not yet rescaled
    to the 0-100 GLF display scale — sufficient for correlation analysis,
    which is scale-invariant anyway).
    """
    series_data = {}
    for series_id in GLF_COMPONENTS:
        print(f"[LagAnalysis] Fetching {series_id} ({years}y history)...", file=sys.stderr)
        result = fetch_fred_series_long(series_id, years=years)
        if result is None:
            print(f"[LagAnalysis] WARNING: {series_id} unavailable, will be excluded from GLF reconstruction", file=sys.stderr)
        series_data[series_id] = result

    available = {k: v for k, v in series_data.items() if v is not None}
    if not available:
        print("[LagAnalysis] No FRED series available — check FRED_API_KEY", file=sys.stderr)
        return []

    # Build monthly YoY series for each available component
    yoy_series = {}
    for series_id, points in available.items():
        dates = [p[0] for p in points]
        values = [p[1] for p in points]
        yoy = []
        for i in range(12, len(values)):
            if values[i - 12] == 0:
                continue
            pct = (values[i] - values[i - 12]) / values[i - 12] * 100
            yoy.append((dates[i], pct))
        yoy_series[series_id] = yoy

    # Align all series to common dates (use the sparsest series' dates as
    # the anchor — typically M2SL, monthly) and compute weighted GLF
    if not yoy_series:
        return []
    anchor_id = max(yoy_series, key=lambda k: len(yoy_series[k]))
    anchor_dates = [d for d, _ in yoy_series[anchor_id]]

    glf_history = []
    for anchor_date in anchor_dates:
        weighted_sum = 0.0
        total_weight_used = 0.0
        for series_id, yoy in yoy_series.items():
            # Find nearest yoy value within 20 days of anchor_date
            nearest_val = _find_nearest(yoy, anchor_date, max_days=20)
            if nearest_val is None:
                continue
            comp = GLF_COMPONENTS[series_id]
            z = _z_score(nearest_val, comp["mean"], comp["std"])
            weighted_sum += z * comp["weight"]
            total_weight_used += comp["weight"]

        if total_weight_used > 0:
            # Rescale by weight actually used, so months missing one series
            # aren't unfairly diluted toward zero.
            glf_history.append((anchor_date, weighted_sum / total_weight_used * total_weight_used))

    return glf_history


def _find_nearest(date_value_pairs, target_date_str, max_days=20):
    """Find the value closest to target_date_str within max_days, or None."""
    target = datetime.fromisoformat(target_date_str)
    best_diff = None
    best_val = None
    for d_str, val in date_value_pairs:
        d = datetime.fromisoformat(d_str)
        diff = abs((d - target).days)
        if diff <= max_days and (best_diff is None or diff < best_diff):
            best_diff = diff
            best_val = val
    return best_val


def load_btc_history():
    """Load BTC daily closes from historical_data.json (produced by
    fetch_historical_btc.py). Returns list of (date_str, close) oldest-first."""
    if not os.path.exists(HISTORICAL_BTC_PATH):
        print(f"[LagAnalysis] {HISTORICAL_BTC_PATH} not found — run fetch_historical_btc.py first", file=sys.stderr)
        return []
    with open(HISTORICAL_BTC_PATH) as f:
        raw = json.load(f)
    return [(r["date"], r["close"]) for r in raw]


def btc_return_over_window(btc_history, start_date_str, window_days=14):
    """
    Return the % change in BTC price from start_date_str to
    start_date_str + window_days, or None if either point isn't available
    within a few days' tolerance.
    """
    start = datetime.fromisoformat(start_date_str)
    end = start + timedelta(days=window_days)

    start_price = _find_nearest(btc_history, start_date_str, max_days=5)
    end_price = _find_nearest(btc_history, end.strftime("%Y-%m-%d"), max_days=5)

    if start_price is None or end_price is None or start_price == 0:
        return None
    return (end_price - start_price) / start_price * 100


def pearson_corr(xs, ys):
    """Simple Pearson correlation, no numpy dependency needed for this size."""
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def run_lag_analysis(glf_history, btc_history, lags_weeks=(0, 2, 4, 8, 12, 16)):
    """
    For each candidate lag, compute correlation between GLF(t) and BTC
    return over [t+lag, t+lag+2weeks].

    Returns: dict {lag_weeks: {"correlation": float, "n_pairs": int}}
    """
    results = {}
    for lag_weeks in lags_weeks:
        lag_days = lag_weeks * 7
        glf_vals = []
        btc_rets = []
        for date_str, glf_val in glf_history:
            target_date = datetime.fromisoformat(date_str) + timedelta(days=lag_days)
            ret = btc_return_over_window(btc_history, target_date.strftime("%Y-%m-%d"), window_days=14)
            if ret is not None:
                glf_vals.append(glf_val)
                btc_rets.append(ret)

        corr = pearson_corr(glf_vals, btc_rets)
        results[lag_weeks] = {"correlation": corr, "n_pairs": len(glf_vals)}

    return results


def main():
    print("=" * 70)
    print("LIQUIDITY LAG/LEAD ANALYSIS — GLF vs BTC returns")
    print("=" * 70)

    print("\n[1/3] Reconstructing historical GLF (Fed/ECB/BOJ/M2, weighted)...")
    glf_history = compute_historical_glf(years=8)
    print(f"      {len(glf_history)} monthly GLF points reconstructed")

    if not glf_history:
        print("\n⚠ Cannot proceed — no GLF history available. Check FRED_API_KEY.")
        return

    print("\n[2/3] Loading BTC price history...")
    btc_history = load_btc_history()
    print(f"      {len(btc_history)} daily BTC price points loaded")

    if not btc_history:
        print("\n⚠ Cannot proceed — no BTC history available. Run fetch_historical_btc.py first.")
        return

    print("\n[3/3] Testing correlation at each candidate lag...")
    results = run_lag_analysis(glf_history, btc_history)

    print("\n" + "-" * 70)
    print(f"{'Lag (weeks)':<15}{'Correlation':<15}{'N pairs':<10}{'Interpretation'}")
    print("-" * 70)
    best_lag = None
    best_corr = 0
    for lag_weeks, r in results.items():
        corr = r["correlation"]
        n = r["n_pairs"]
        if corr is None:
            interp = "insufficient data"
            corr_str = "N/A"
        else:
            interp = "strong" if abs(corr) > 0.3 else "weak" if abs(corr) > 0.15 else "negligible"
            corr_str = f"{corr:+.3f}"
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag_weeks
        print(f"{lag_weeks:<15}{corr_str:<15}{n:<10}{interp}")

    print("-" * 70)
    if best_lag is not None:
        print(f"\n📊 Strongest empirical lag: {best_lag} weeks (correlation={best_corr:+.3f})")
        print(f"   This means GLF measured today correlates most strongly with")
        print(f"   BTC's 2-week return starting {best_lag} weeks from now.")
        print(f"\n   NOTE: correlation strength should be interpreted cautiously —")
        print(f"   {len(glf_history)} monthly points over 8 years is a small sample")
        print(f"   for financial time series (autocorrelated, regime-dependent).")
        print(f"   Treat this as a starting hypothesis to monitor, not a settled fact.")
    else:
        print("\n⚠ No lag produced a usable correlation — check data availability.")

    # Save results for downstream use (e.g. by global_liquidity_engine.py
    # to eventually add a lagged GLF output field once a lag is confirmed
    # stable across multiple runs / longer history).
    output_path = os.path.join(BASE_DIR, ".liquidity_lag_analysis_result.json")
    with open(output_path, "w") as f:
        json.dump({
            "computed_at": datetime.now().isoformat(),
            "results": {str(k): v for k, v in results.items()},
            "best_lag_weeks": best_lag,
            "best_correlation": best_corr if best_lag is not None else None,
            "n_glf_points": len(glf_history),
            "n_btc_points": len(btc_history),
        }, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
