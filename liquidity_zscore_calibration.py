"""
liquidity_zscore_calibration.py — Verify GLF z-score constants against real history
======================================================================================

global_liquidity_engine.py normalizes each YoY liquidity series with a
hardcoded (mean, std) pair, e.g.:

    fed_z = _z_score(fed_yoy, 5.5, 8.0)   # "historical mean ~5.5%, std ~8%"
    ecb_z = _z_score(ecb_yoy, 4.0, 7.0)
    jpn_z = _z_score(jpn_yoy, 3.0, 6.0)
    m2_z  = _z_score(m2_yoy,  6.0, 4.0)

These numbers were written as approximate comments, never verified against
an actual computed mean/std from FRED's own history. If they're off, every
z-score built from them is biased — e.g. if the real Fed YoY mean over the
last decade is actually 8% (not 5.5%), the model will systematically read
"expansion" when balance sheet growth is merely average, inflating GLF
during genuinely unremarkable periods.

This script:
    1. Pulls N years of history for each series from FRED (same series IDs
       as production: WALCL, ECBASSETSW, JPNASSETS, M2SL).
    2. Computes the actual YoY % change at every point in that history.
    3. Reports the real mean/std, compared side-by-side with the hardcoded
       production values.
    4. Flags any component where the discrepancy is large enough to matter
       (a rule of thumb: >30% relative difference in mean, or >40% in std).

This intentionally reuses fetch logic consistent with liquidity_lag_analysis.py
(same FRED series, same YoY formula) so results from both scripts describe
the same underlying data, not two different reconstructions that could
silently diverge.

Usage:
    python3 liquidity_zscore_calibration.py
    (requires FRED_API_KEY in .env)
"""
import os
import sys
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY", "")

# (series_id, display_name, current_hardcoded_mean, current_hardcoded_std, weight_in_glf)
# Values must match global_liquidity_engine.py's compute_global_liquidity_factor()
# exactly — if that file's constants are ever edited, update this table too,
# or this script will compare against stale numbers.
CURRENT_CONSTANTS = [
    ("WALCL", "Fed Balance Sheet", 5.5, 8.0, 0.30),
    ("ECBASSETSW", "ECB Balance Sheet", 4.0, 7.0, 0.15),
    ("JPNASSETS", "BOJ Balance Sheet", 3.0, 6.0, 0.05),
    ("M2SL", "US M2 Money Supply", 6.0, 4.0, 0.15),
]

# Relative-difference thresholds for flagging a constant as needing update.
MEAN_FLAG_THRESHOLD = 0.30   # 30% relative difference
STD_FLAG_THRESHOLD = 0.40    # 40% relative difference


def fetch_fred_series_long(series_id, years=10):
    """Fetch a long FRED series. Returns list of (date_str, value), oldest-first."""
    if not FRED_KEY:
        print(f"[Calibration] FRED_API_KEY not set — cannot fetch {series_id}", file=sys.stderr)
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
            print(f"[Calibration] FRED {series_id} returned {r.status_code}", file=sys.stderr)
            return None
        obs = r.json().get("observations", [])
        return [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[Calibration] FRED {series_id} fetch failed: {e}", file=sys.stderr)
        return None


def compute_yoy_series(points):
    """Compute YoY % change at every point where a 12-observations-earlier
    value exists. Works for both monthly (WALCL/M2SL/JPNASSETS are monthly
    or weekly-resampled-to-monthly by FRED) series — assumes points are
    evenly enough spaced that index-12 approximates "1 year earlier"; FRED
    series used here (WALCL, ECBASSETSW are weekly; M2SL, JPNASSETS are
    monthly) — weekly series need index-52, not index-12. Handled below by
    inferring spacing from the actual date deltas rather than assuming.
    """
    if not points or len(points) < 20:
        return []

    dates = [datetime.fromisoformat(p[0]) for p in points]
    values = [p[1] for p in points]

    # Infer approximate observation frequency from median day-gap
    gaps = sorted((dates[i] - dates[i - 1]).days for i in range(1, min(50, len(dates))))
    median_gap = gaps[len(gaps) // 2] if gaps else 30
    # ~365 days / median_gap ≈ observations per year
    obs_per_year = max(1, round(365 / median_gap)) if median_gap > 0 else 12

    yoy = []
    for i in range(obs_per_year, len(values)):
        if values[i - obs_per_year] == 0:
            continue
        pct = (values[i] - values[i - obs_per_year]) / values[i - obs_per_year] * 100
        yoy.append(pct)
    return yoy


def mean_std(values):
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def main():
    print("=" * 78)
    print("LIQUIDITY Z-SCORE CALIBRATION — hardcoded constants vs real FRED history")
    print("=" * 78)

    print(f"\n{'Series':<22}{'HC Mean':>10}{'Real Mean':>12}{'HC Std':>9}{'Real Std':>11}   Flag")
    print("-" * 78)

    flagged = []
    for series_id, name, hc_mean, hc_std, weight in CURRENT_CONSTANTS:
        points = fetch_fred_series_long(series_id, years=10)
        if points is None:
            print(f"{name:<22}{'—':>10}{'FETCH FAILED':>12}{'—':>9}{'—':>11}   ⚠")
            continue

        yoy = compute_yoy_series(points)
        if not yoy:
            print(f"{name:<22}{'—':>10}{'INSUFFICIENT DATA':>18}")
            continue

        real_mean, real_std = mean_std(yoy)

        mean_diff = abs(real_mean - hc_mean) / abs(hc_mean) if hc_mean != 0 else float("inf")
        std_diff = abs(real_std - hc_std) / abs(hc_std) if hc_std != 0 else float("inf")

        flag = ""
        if mean_diff > MEAN_FLAG_THRESHOLD or std_diff > STD_FLAG_THRESHOLD:
            flag = "🔴 UPDATE"
            flagged.append((series_id, name, hc_mean, hc_std, real_mean, real_std, weight))
        elif mean_diff > MEAN_FLAG_THRESHOLD / 2 or std_diff > STD_FLAG_THRESHOLD / 2:
            flag = "🟡 minor drift"
        else:
            flag = "✅ OK"

        print(f"{name:<22}{hc_mean:>10.2f}{real_mean:>12.2f}{hc_std:>9.2f}{real_std:>11.2f}   {flag}")

    print("-" * 78)

    if flagged:
        print(f"\n⚠ {len(flagged)} component(s) need updated constants:\n")
        for series_id, name, hc_mean, hc_std, real_mean, real_std, weight in flagged:
            print(f"  {name} (weight={weight} in GLF):")
            print(f"    Current:  _z_score(value, {hc_mean}, {hc_std})")
            print(f"    Suggested: _z_score(value, {real_mean:.2f}, {real_std:.2f})")
            print()
        print("To apply: update the corresponding line in global_liquidity_engine.py's")
        print("compute_global_liquidity_factor() with the suggested values above.")
        print()
        print("NOTE: mean/std computed here reflect the last 10 years, which includes")
        print("both the 2020-2021 QE surge and 2022-2023 QT — a genuinely unusual decade")
        print("for central bank balance sheets. Consider whether that period should be")
        print("weighted, excluded, or kept as-is before blindly replacing the constants —")
        print("this script reports what IS in the data, not what SHOULD define \"normal\".")
    else:
        print("\n✅ All hardcoded constants are within tolerance of real historical values.")
        print("   No updates needed at this time.")


if __name__ == "__main__":
    main()
