#!/usr/bin/env python3
"""
SFC Expectations Engine (L6) — Market Expectation Gap
=====================================================
Layer 6 of the IMBS / Macro-Intelligence blueprint: the market moves on
CHANGES IN EXPECTATION, not just levels of the economy. This module
measures how far MARKET PRICING of the future sits from the RECENTLY
REALIZED economy.

WHAT THIS COMPUTES (honest scope — no fabricated consensus data):
    A genuine "surprise" score (actual vs analyst consensus) needs
    Bloomberg/Refinitiv consensus forecasts, which are proprietary and
    not freely available. FRED only carries actuals. Rather than fake a
    consensus, this engine computes a defensible Expectation Gap from
    data FRED genuinely publishes:

      1. Inflation expectation gap
         = T10YIE (10Y breakeven inflation = the MARKET's priced-in
           average inflation expectation over 10Y) MINUS realized CPI
           YoY.  Positive => market prices HIGHER future inflation than
           recent realized; negative => market prices DISINFLATION.
      2. Real rate pressure (monetary expectation)
         = DGS10 - T10YIE (10Y real yield). High real yield = tight
           financial conditions priced in (expectation of restrictive
           policy).
      3. Policy path / growth expectation (curve shape)
         = T10Y2Y yield-curve slope. Inverted (negative) is a classic
           forward-looking recession / easing-expectation signal.
      4. Labour-market expectation backdrop
         = UNRATE level & its short trend (recent realized, used as the
           "reality" anchor alongside CPI).

OUTPUT:
    expectation_gap (signed, can be negative — the single headline)
        = inflation expectation gap (T10YIE - CPI YoY), scaled to
          percentage points. This is the closest defensible proxy to the
          blueprint's "Expectation Gap = Reality - Market Pricing".
    gap_score (0-100, stress-oriented)
        Normalized composite where HIGH = negative / fragile expectation
        regime (disinflation surprise risk + inverted curve + tight real
        rates). Display-only; NOT blended into sfc_effective/signal.

DESIGN / CAVEATS (honesty markers):
    - This is a PROXY, not a true consensus-surprise engine. Fields that
      would need consensus data are explicitly set to None with a reason
      in `details['unavailable']` rather than being fabricated.
    - Follows the cautious-rollout pattern of M86/M90/reflexivity:
      exposed as its own field, observed vs live data before any decision
      to fold it into the core ensemble.
    - Fails safe: any missing FRED series excludes that component; the
      module returns a neutral/partial result rather than corrupting output.

Usage:
    from data_sources.expectations_engine import compute_expectations
    score, details = compute_expectations()
"""
import os, json, time, requests

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".expectations_cache.json")
CACHE_TTL = 21600  # 6 hours — CPI is monthly, breakeven/daily series move a bit
FRED_KEY = os.getenv("FRED_API_KEY", "")
_FRED_CACHE = {}


def _fred_obs(series, limit=15):
    """Fetch FRED observations as list of {date, value} (newest-first),
    skipping missing ('.') values, with module-level cache. Same pattern
    as sovereign_yield_curves.py / repo_market_stress.py but retains dates
    so monthly YoY can be computed by date-matching (robust to missing
    prints like CPIAUCSL's occasional '.')."""
    global _FRED_CACHE
    key = f"obs:{series}:{limit}"
    if key in _FRED_CACHE:
        return _FRED_CACHE[key]
    if not FRED_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.stlouisfed.org/fred/series/observations?series_id={series}"
            f"&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}",
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[Expectations] FRED returned {r.status_code} for {series}", file=sys.stderr)
            return None
        rows = [{"date": o["date"], "value": float(o["value"])}
                for o in r.json().get("observations", []) if o["value"] != "."]
        result = rows if rows else None
        _FRED_CACHE[key] = result
        return result
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[Expectations] FRED fetch failed for {series}: {e}", file=sys.stderr)
        return None


def _fred(series, limit=13):
    """Numeric values only (newest-first) — convenience for point-in-time
    daily series (T10YIE, DGS10, T10Y2Y, UNRATE)."""
    rows = _fred_obs(series, limit=max(limit, 3))
    return [r["value"] for r in rows] if rows else None


def _cpi_yoy(limit=15):
    """CPI level series -> YoY % computed by DATE-MATCHING (value at month
    T vs value at month T-12), robust to FRED's occasional missing print
    ('.') that would otherwise shift a naive index-based lookup."""
    rows = _fred_obs("CPIAUCSL", limit)
    if not rows:
        return None
    # Newest-first; latest valid observation:
    latest = rows[0]
    latest_ym = latest["date"][:7]  # 'YYYY-MM'
    y, m = int(latest_ym[:4]), int(latest_ym[5:7])
    target = f"{y-1:04d}-{m:02d}"
    for r in rows[1:]:
        if r["date"][:7] == target:
            if r["value"] <= 0:
                return None
            return (latest["value"] / r["value"] - 1.0) * 100.0
    return None


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(state):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def _slope_to_stress(yoy_gap):
    """Map the signed inflation-expectation gap (pp) to a 0-100 stress.
    A large NEGATIVE gap (market prices much lower inflation than just
    realized = deflation-surprise risk) is fragile for risk assets -> high
    stress. A small positive/near-zero gap is benign. Moderate positive
    (modest reflation) is neutral. Very large positive (runaway inflation
    expectation) is also elevated. Heuristic scale, honest as provisional."""
    g = yoy_gap if yoy_gap is not None else 0.0
    # Deflation surprise risk dominates (curve inverted era, 2022-24).
    if g <= -2.5:
        return 85.0
    if g <= -1.0:
        return 65.0
    if g <= 0.0:
        return 50.0
    if g <= 1.0:
        return 35.0
    if g <= 2.5:
        return 45.0
    if g <= 4.0:
        return 60.0
    return 75.0


def compute_expectations():
    """Compute L6 Expectation Gap. Returns (gap_score 0-100, details dict)."""
    cached = _load_cache()
    now = time.time()
    if cached.get("ts") and now - cached.get("ts", 0) < CACHE_TTL:
        return cached.get("gap_score", 50.0), cached.get("details", {"status": "cached"})

    unavailable = []

    # --- Component 1: inflation expectation gap (signed) ---
    cpi_yoy = _cpi_yoy()
    t10yie = _fred("T10YIE", 1)
    brk = t10yie[0] if t10yie else None

    if cpi_yoy is not None and brk is not None:
        infl_gap = brk - cpi_yoy  # signed, pp (can be negative)
    else:
        infl_gap = None
        if cpi_yoy is None:
            unavailable.append("CPI_YoY (CPIAUCSL)")
        if brk is None:
            unavailable.append("10Y breakeven (T10YIE)")

    # --- Component 2: real rate pressure (monetary expectation) ---
    dgs10 = _fred("DGS10", 1)
    real_rate = (dgs10[0] - brk) if (dgs10 and brk is not None) else None
    if real_rate is None and (dgs10 is None or brk is None):
        unavailable.append("Real yield (DGS10-T10YIE)")

    # --- Component 3: curve shape / growth expectation ---
    t10y2y = _fred("T10Y2Y", 1)
    curve = t10y2y[0] if t10y2y else None
    if curve is None:
        unavailable.append("Yield curve (T10Y2Y)")

    # --- Component 4: labour-market reality anchor ---
    unrate = _fred("UNRATE", 3)
    u = unrate[0] if unrate else None
    u_trend = (unrate[0] - unrate[1]) if (unrate and len(unrate) >= 2) else None
    if u is None:
        unavailable.append("Unemployment (UNRATE)")

    # --- Headline expectation gap ---
    # Signed headline = inflation expectation gap (the most defensible
    # "market pricing vs realized reality" measure available for free).
    expectation_gap = round(infl_gap, 2) if infl_gap is not None else None

    # --- Normalized stress score (0-100) ---
    comps = []
    if infl_gap is not None:
        comps.append(_slope_to_stress(infl_gap))
    if real_rate is not None:
        # real rate > 3% = tight, elevated; ~0-1.5% neutral
        comps.append(25.0 if real_rate < 1.0 else 50.0 if real_rate < 2.0 else 65.0 if real_rate < 3.0 else 80.0)
    if curve is not None:
        comps.append(80.0 if curve < 0 else 45.0 if curve < 0.5 else 30.0)
    if u is not None:
        comps.append(75.0 if u >= 6.5 else 55.0 if u >= 5.5 else 40.0 if u >= 4.0 else 30.0)

    if comps:
        gap_score = round(sum(comps) / len(comps), 1)
        available = True
    else:
        gap_score = 50.0
        available = False

    details = {
        "status": "ok" if available else "partial",
        "available": available,
        "expectation_gap": expectation_gap,   # signed headline (pp)
        "inflation_expect_gap_pp": round(infl_gap, 2) if infl_gap is not None else None,
        "cpi_yoy_pct": round(cpi_yoy, 2) if cpi_yoy is not None else None,
        "breakeven_inflation": round(brk, 2) if brk is not None else None,
        "real_yield_10y": round(real_rate, 2) if real_rate is not None else None,
        "curve_10y2y": round(curve, 2) if curve is not None else None,
        "unemployment": round(u, 1) if u is not None else None,
        "unemployment_1m_delta": round(u_trend, 1) if u_trend is not None else None,
        "label": "DEFLATION-SURPRISE RISK" if expectation_gap is not None and expectation_gap < -1.0 else
                 "REFLATION-PRICING" if expectation_gap is not None and expectation_gap > 1.5 else
                 "BENIGN EXPECTATIONS" if expectation_gap is not None else "UNAVAILABLE",
        "unavailable": unavailable,
        "caveat": "Proxy from FRED actuals vs market pricing (T10YIE), NOT a consensus-surprise score.",
        "ts": now,
    }

    state = {"gap_score": gap_score, "details": details, "ts": now}
    _save_cache(state)
    return gap_score, details


if __name__ == "__main__":
    import sys
    s, d = compute_expectations()
    print(f"gap_score={s}")
    for k in ("expectation_gap", "cpi_yoy_pct", "breakeven_inflation",
              "real_yield_10y", "curve_10y2y", "unemployment", "label",
              "status", "unavailable"):
        print(f"  {k}: {d.get(k)}")
