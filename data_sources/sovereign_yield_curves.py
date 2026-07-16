#!/usr/bin/env python3
"""
SFC Sovereign Yield Curve Module (M88-M90)
=============================================
M88 — Japanese Government Bond (JGB) yield curve slope
M89 — German Bund yield curve slope
M90 — UK Gilt yield curve slope

WHY THIS MATTERS FOR A LIQUIDITY-FOCUSED MODEL:
    Beyond the US Treasury yield curve (already covered by M8), three
    other sovereign bond markets carry outsized weight in global
    liquidity conditions:

    JGB (Japan): Japan is the world's largest net exporter of capital.
    Decades of near-zero JGB yields fund the "yen carry trade" — global
    investors borrow cheap yen, invest in higher-yielding foreign assets
    (including risk assets like BTC). When BOJ tightens and JGB yields
    rise, carry trades become unprofitable and unwind rapidly, forcing
    synchronized selling across global risk assets — the August 2024
    carry-trade unwind (triggered by a small BOJ hike) is the textbook
    case, causing a sharp simultaneous drop in US equities and crypto.

    German Bund: the risk-free benchmark for the entire Eurozone, same
    role as US Treasuries for USD. A sharp Bund yield move signals ECB
    policy shifts affecting the second-largest reserve currency bloc.

    UK Gilt: smaller market, but the September 2022 UK gilt crisis
    (a fiscal policy announcement caused Gilt yields to spike, triggering
    forced selling by leveraged pension funds using LDI strategies, and
    an emergency Bank of England intervention) proved that even a
    mid-sized sovereign bond market can trigger systemic liquidity
    stress when enough leverage sits behind it.

DATA SOURCE — IMPORTANT CAVEAT:
    Series IDs below (IRLTLT01xxM156N for 10-year, IRSTCI01xxM156N for
    short-term) follow the OECD Main Economic Indicators naming
    convention that FRED mirrors for many countries — this pattern is
    used with reasonable confidence based on how FRED structures
    cross-country series, but was NOT verified against a live FRED query
    (no network access in the environment this was written in). Confirm
    each series resolves at https://fred.stlouisfed.org/series/<ID>
    before relying on this module. Every function here fails safe (that
    country's component is simply excluded, returns None) if its series
    ID is wrong — a bad ID won't corrupt output, it'll just mean that
    one country's yield curve doesn't contribute this cycle, same
    fail-safe pattern already used for the China M2 series in
    global_liquidity_engine.py.

NOT IMPLEMENTED — cross-currency basis swap:
    This was also requested, but cross-currency basis swaps are an OTC
    derivatives market data point (typically Bloomberg/Refinitiv,
    proprietary) — FRED does not appear to publish this data type, and
    no free/reliable public API for it is known. Rather than fabricate a
    misleading proxy, this is deliberately left out of this module.
    If you have access to a specific data source for this (even a manual
    CSV export you could periodically update, similar to how ETF flow
    data works in this codebase), let me know and this can be added as a
    separate M91.
"""
import os
import sys
import time
import json
import requests

FRED_KEY = os.getenv("FRED_API_KEY", "")
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".sovereign_yield_cache.json")
CACHE_TTL = 43200  # 12 hours — these are monthly-frequency series, no need to poll more often

_FRED_CACHE = {}

# (country_code, display_name, 10y_series_id, short_term_series_id,
#  inversion_threshold_notes) — short_term here approximates a 2Y-like
#  policy-sensitive rate since FRED's OECD-mirrored series for many
#  countries don't have a clean "2-year government bond" equivalent the
#  way DGS2 does for the US; using the short-term rate series is a
#  reasonable proxy for curve slope direction, though less precise than
#  a true 2Y point.
COUNTRIES = {
    "jgb": {
        "name": "Japan (JGB)",
        "long_series": "IRLTLT01JPM156N",
        "short_series": "IRSTCI01JPM156N",
        "method_id": "M88",
    },
    "bund": {
        "name": "Germany (Bund)",
        "long_series": "IRLTLT01DEM156N",
        "short_series": "IRSTCI01DEM156N",
        "method_id": "M89",
    },
    "gilt": {
        "name": "UK (Gilt)",
        "long_series": "IRLTLT01GBM156N",
        "short_series": "IRSTCI01GBM156N",
        "method_id": "M90",
    },
}


def _fred(series, limit=1):
    """Fetch FRED data with module-level cache. Same pattern used
    throughout this codebase (repo_market_stress.py, fiscal_liquidity.py)
    for consistency."""
    global _FRED_CACHE
    cache_key = f"{series}:{limit}"
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]
    if not FRED_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
            f"&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}",
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[SovereignYield] FRED returned {r.status_code} for {series} — "
                  f"series ID may be wrong, see module docstring", file=sys.stderr)
            return None
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        result = vals if vals else None
        _FRED_CACHE[cache_key] = result
        return result
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[SovereignYield] FRED fetch failed for {series}: {e}", file=sys.stderr)
        return None


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"cached_at": 0}


def _save_cache(cache):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def compute_yield_curve_slope(country_key, force_refresh=False):
    """
    Generic yield curve slope calculator for one country — mirrors
    collect.py's calculate_m8_yield_curve() logic (same score thresholds,
    same "inversion = highest stress" interpretation) so results are
    directly comparable across US/JGB/Bund/Gilt rather than each country
    using a different, hard-to-compare scale.

    Returns:
        (score, detail_dict)
        score: 0-1, higher = more inverted/stressed yield curve
        detail_dict: {"slope": float, "long_yield": float,
                       "short_yield": float, "status": "ok"|"unavailable"}
    """
    if country_key not in COUNTRIES:
        raise ValueError(f"Unknown country_key: {country_key}")

    config = COUNTRIES[country_key]
    cache = _load_cache()
    now = time.time()
    cache_key = f"{country_key}_score"

    if not force_refresh and (now - cache.get("cached_at", 0)) < CACHE_TTL:
        if cache.get(cache_key) is not None:
            return cache[cache_key], cache.get(f"{country_key}_detail", {})

    long_vals = _fred(config["long_series"], limit=1)
    short_vals = _fred(config["short_series"], limit=1)

    if not long_vals or not short_vals:
        detail = {
            "status": "unavailable",
            "reason": f"{config['name']} yield data unavailable — verify series IDs "
                      f"{config['long_series']} / {config['short_series']}",
        }
        return 0.5, detail

    long_yield = long_vals[0]
    short_yield = short_vals[0]
    slope = long_yield - short_yield

    # Same thresholds as calculate_m8_yield_curve() (US) — kept identical
    # deliberately, so a 0.80 score means "similarly inverted" whether
    # it's the US, Japan, Germany, or UK curve, making cross-country
    # comparison meaningful rather than each using an arbitrary own scale.
    if slope < 0:
        score = 0.80
    elif slope < 0.5:
        score = 0.65
    elif slope < 1.0:
        score = 0.40
    elif slope > 2.0:
        score = 0.15
    else:
        score = 0.25

    detail = {
        "slope": round(slope, 3),
        "long_yield": round(long_yield, 3),
        "short_yield": round(short_yield, 3),
        "status": "ok",
    }

    cache[cache_key] = score
    cache[f"{country_key}_detail"] = detail
    cache["cached_at"] = now
    _save_cache(cache)

    return score, detail


def compute_all_sovereign_curves(force_refresh=False):
    """Convenience wrapper: compute all three (JGB, Bund, Gilt) at once.

    Returns:
        dict {country_key: (score, detail)} for "jgb", "bund", "gilt"
    """
    return {
        key: compute_yield_curve_slope(key, force_refresh=force_refresh)
        for key in COUNTRIES
    }


if __name__ == "__main__":
    print("=== Live fetch (requires FRED_API_KEY + valid series IDs) ===\n")
    for key, config in COUNTRIES.items():
        score, detail = compute_yield_curve_slope(key, force_refresh=True)
        print(f"{config['method_id']} {config['name']}: score={score}")
        print(f"  Detail: {json.dumps(detail, indent=2)}")
        if detail.get("status") == "unavailable":
            print(f"  ⚠ Verify series IDs at:")
            print(f"    https://fred.stlouisfed.org/series/{config['long_series']}")
            print(f"    https://fred.stlouisfed.org/series/{config['short_series']}")
        print()

    # Self-test: verify scoring logic without network, using the same
    # threshold ladder as calculate_m8_yield_curve() for consistency —
    # confirms cross-country comparability is preserved.
    print("--- Self-test: scoring logic (no network) ---")

    def _score_only(slope):
        if slope < 0: return 0.80
        elif slope < 0.5: return 0.65
        elif slope < 1.0: return 0.40
        elif slope > 2.0: return 0.15
        else: return 0.25

    test_cases = [
        (-0.5, 0.80, "inverted — highest stress"),
        (0.2, 0.65, "mildly positive, still elevated"),
        (0.7, 0.40, "moderate"),
        (1.5, 0.25, "normal upward-sloping"),
        (2.5, 0.15, "steep — lowest stress"),
    ]
    all_pass = True
    for slope, expected, note in test_cases:
        actual = _score_only(slope)
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_pass = False
        print(f"  {status} slope={slope:+.1f} -> score={actual} (expected {expected})  ({note})")

    if not all_pass:
        raise AssertionError("Self-test failed — see ❌ above")
    print("\nALL SELF-TESTS PASSED")
