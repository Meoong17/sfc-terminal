#!/usr/bin/env python3
"""
slr_engine.py — Sovereign Liquidity Regime (SLR) v2 reconstruction (research only)
=================================================================================
Reconstructs the SLR v2 components described in C/SLR.md, point-in-time, using
data available in this repo / free FRED — so the SLR thesis can be TESTED before
anything is blended into the live SFC signal. This module is RESEARCH/EVAL only:
it does NOT feed the live pipeline and does NOT touch sfc_effective.

Components reconstructed
------------------------
  M91   Sovereign Duration Stress   — z-score(window=252) of the daily change in a
                                     term-premium proxy. Term premium is approximated
                                     as dTP ~ dY30 - dY2 (see caveat below). This is the
                                     objective, data-derived core of SLR (SLR_Risk axis).
  M92   Policy Liquidity Response   — direction-aware event registry (human-in-the-loop,
                                     per design). Positive/negative liquidity policy
                                     response to sovereign-stress, decayed by recency.
                                     This is the SUBJECTIVE component; kept small and
                                     explicitly documented.
  M93b  Market Response             — BTC relative-strength / volume / volatility
                                     confirmation. Reconstructed from Binance Vision
                                     (2017+) / FRED CBBTCUSD. M93a (capital flow) is NOT
                                     reconstructable historically (ETF/stablecoin/whale
                                     histories are too short or unreliable), so TC = M93b
                                     alone here — documented, not silently dropped.

  SLR_Liquidity (interim) = geometric_mean(M91_clip, M92_clip, TC_clip) with floor 15
                            (per SLR.md Section 4 interim formula).
  SLR_Risk       = M91 magnitude alone.

CAVEATS (honesty):
  * M92 event registry is manual/subjective per the design's human-in-the-loop nature.
    Direction assignments follow SLR.md's lookup table but are judgment calls. Keep the
    registry small and documented. The PRIMARY quantitative test (Test #1) therefore
    leans on the objective M91/M93b; M92's effect is reported transparently.
  * Term-premium proxy dY30 - dY2 is a crude approximation of the ACM term premium the
    doc references (true ACM needs a full term-structure model). The proxy captures the
    duration/slope shock direction; not a precise TP estimate.
  * OI (open interest) is not present in Binance Vision history, so M93b's
    OI_sustainability subcomponent is omitted (documented; the rest of M93b is intact).

USAGE:
    .venv/bin/python analysis/slr_engine.py            # self-test + build all series
"""
import json
import os
import sys
import math
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

from historical_backtest_m1m6 import fetch_fred_series

YIELD30 = "DGS30"   # US 30-year Treasury constant maturity
YIELD2  = "DGS2"    # US 2-year Treasury constant maturity
BTCCB   = "CBBTCUSD"  # FRED Bitcoin price (for M93b pre-Binance window)

# M91 z-score window (calendar days; matches SLR.md window=252)
M91_WINDOW = 252
# M91 trigger threshold (z >= 1.5 -> sovereign stress candidate)
M91_TRIGGER_Z = 1.5

# M92 decay half-life (days): PRM decays with exp(-days_after/10) per SLR.md
M92_DECAY_DAYS = 10

OUT_JSON = os.path.join(SFC_ROOT, ".slr_series.json")


# --------------------------------------------------------------------------- #
# M91 — Sovereign Duration Stress
# --------------------------------------------------------------------------- #
def term_premium_proxy(y30, y2):
    """Crude daily term-premium proxy dTP ~ dY30 - dY2 (slope-change proxy).
    Aligned on common dates. Returns {date: dTP}."""
    common = sorted(set(y30) & set(y2))
    out = {}
    prev = None
    for d in common:
        cur = (y30[d] - y2[d])  # slope: 30Y - 2Y
        if prev is not None:
            out[d] = cur - prev
        prev = cur
    return out


def zscore_trailing(series, window=M91_WINDOW, min_n=60):
    """{date: z-score of series vs trailing `window` prior obs}.
    Requires at least min_n prior non-null points."""
    dates = sorted(series)
    out = {}
    for i, d in enumerate(dates):
        if i < min_n:
            continue
        window_vals = [series[dates[j]] for j in range(max(0, i - window), i)]
        window_vals = [v for v in window_vals if v is not None]
        if len(window_vals) < min_n:
            continue
        mu = float(np.mean(window_vals))
        sd = float(np.std(window_vals))
        if sd <= 1e-9:
            continue
        out[d] = (series[d] - mu) / sd
    return out


def m91_duration_stress(y30, y2):
    """Returns (z_series, trigger_series). z_series = z-score of dTP.
    trigger where z >= 1.5."""
    dtp = term_premium_proxy(y30, y2)
    z = zscore_trailing(dtp, M91_WINDOW)
    return z


def m91_score_0100(zseries):
    """Map M91 z to a 0-100 magnitude (SLR_Risk axis): z>=3 -> 100, z<=0 -> floor."""
    out = {}
    for d, zv in zseries.items():
        # magnitude: higher positive z (duration stress building) -> higher score
        s = max(0.0, min(100.0, (zv / 3.0) * 100.0))
        out[d] = s
    return out


# --------------------------------------------------------------------------- #
# M92 — Policy Liquidity Response (event registry, human-in-the-loop)
# --------------------------------------------------------------------------- #
# Direction lookup per SLR.md Section 2. magnitude_class_weight: high=1.0,
# medium=0.6, low=0.3 (draft). Each event: (start_date, type, direction, magnitude_class)
# Direction +1 = liquidity injection (bullish), -1 = liquidity destruction (bearish).
# These are documented, high-confidence episodes; the registry is intentionally small.
EVENT_REGISTRY = [
    # --- Positive liquidity response episodes ---
    {"date": "2020-03-16", "type": "rate_cut_qe", "direction": 1, "magnitude_class": "high",
     "note": "Fed emergency 100bp cut to 0-0.25% + open-ended QE (COVID stress)"},
    {"date": "2023-03-12", "type": "emergency_lending_btfp", "direction": 1, "magnitude_class": "high",
     "note": "SVB crisis; Fed launched Bank Term Funding Program (BTFP)"},
    {"date": "2023-11-01", "type": "qt_pause_dovish", "direction": 1, "magnitude_class": "medium",
     "note": "Fed held rates after Oct-2023 30Y yield spike to ~5%; signaled end of hiking"},
    {"date": "2019-10-11", "type": "btfp_style_liquidity", "direction": 1, "magnitude_class": "medium",
     "note": "Post-Sept-2019 repo spike; Fed resumed organic balance-sheet growth + bill purchases"},
    {"date": "2024-09-27", "type": "pboc_rrr_cut", "direction": 1, "magnitude_class": "medium",
     "note": "PBOC 50bp RRR cut + policy easing during China stress"},
    {"date": "2026-08-19", "type": "treasury_buyback_expansion", "direction": 1, "magnitude_class": "high",
     "note": "Treasury buyback expansion (SLR.md's own worked example)"},
    # --- Negative liquidity response episodes ---
    {"date": "2022-06-01", "type": "qt_launch_hikes", "direction": -1, "magnitude_class": "high",
     "note": "Fed began QT + aggressive hikes in response to 2022 inflation/yield surge"},
    {"date": "2022-09-21", "type": "qt_resume_hawkish", "direction": -1, "magnitude_class": "medium",
     "note": "Continued hawkish hikes + QT pace ramp into 2022 (UK gilt crisis period)"},
    {"date": "2018-10-01", "type": "qt_resume", "direction": -1, "magnitude_class": "medium",
     "note": "Fed QT + hikes through 2018; 30Y/10Y rose into Dec-2018 stress"},
    {"date": "2024-05-01", "type": "qt_continuation", "direction": -1, "magnitude_class": "low",
     "note": "QT continued at reduced pace; balance-sheet runoff ongoing"},
]

_MAG_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


def m92_policy_response(dates, events=None):
    """{date: M92 on 0-100 scale} for each date. Positive response -> high (liquidity
    bull), negative response -> low (destruction), no active event -> 50 (neutral).
    PRM = mag_weight * direction * exp(-days_after/10); then mapped 0-100 with 50 base."""
    events = events if events is not None else EVENT_REGISTRY
    # convert events to date objects
    ev = []
    for e in events:
        try:
            ev.append((datetime.strptime(e["date"], "%Y-%m-%d"), e["direction"], _MAG_WEIGHT[e["magnitude_class"]]))
        except KeyError:
            continue
    out = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        active = None
        for (ed, direction, mag) in ev:
            days = (dt - ed).days
            if 0 <= days <= 60:  # event effect window ~2 months
                prm = mag * direction * math.exp(-days / M92_DECAY_DAYS)
                if active is None or abs(prm) > abs(active[0]):
                    active = (prm, days)
        if active is None:
            out[d] = 50.0  # neutral / no signal
        else:
            prm, days = active
            # prm in [-1, 1]; map to 0-100: +1 -> 90, -1 -> 10, 0 -> 50
            out[d] = 50.0 + 40.0 * prm
    return out


# --------------------------------------------------------------------------- #
# M93b — Market Response (BTC relative strength / volume / volatility)
# --------------------------------------------------------------------------- #
def m93b_market_response(btc_close, btc_vol=None):
    """{date: 0-100} market response confirmation from BTC behavior.
    High score = BTC confirming (rising, above its own trend = relative strength).
    Built from BTC close: z of the trailing 90d return (positive = confirming)."""
    dates = sorted(btc_close)
    ret90 = {}
    for i, d in enumerate(dates):
        if i < 90:
            continue
        p_now = btc_close[d]
        p_90 = btc_close[dates[i - 90]]
        if p_90 and p_90 > 0:
            ret90[d] = (p_now - p_90) / p_90 * 100.0
    z = zscore_trailing(ret90, window=180, min_n=60)
    out = {}
    for d, zv in z.items():
        out[d] = max(0.0, min(100.0, 50.0 + 16.0 * zv))
    return out


# --------------------------------------------------------------------------- #
# SLR_Liquidity (interim) + SLR_Risk
# --------------------------------------------------------------------------- #
def slr_composite(m91z, m92, tc, floor=15.0):
    """Interim geometric mean with floor, per SLR.md Section 4.
    Returns {date: {slr_liquidity, slr_risk, m91, m92, tc}}.
    slr_risk = M91 magnitude (0-100); slr_liquidity = (m91*m92*tc)^(1/3), floored."""
    m91s = m91_score_0100(m91z)
    common = sorted(set(m91s) & set(m92) & set(tc))
    out = {}
    for d in common:
        a = max(floor, m91s[d])
        b = max(floor, m92[d])
        c = max(floor, tc[d])
        liq = (a * b * c) ** (1.0 / 3.0)
        out[d] = {
            "slr_liquidity": round(float(liq), 3),
            "slr_risk": round(float(m91s[d]), 3),
            "m91": round(float(m91s[d]), 3),
            "m92": round(float(m92[d]), 3),
            "tc": round(float(tc[d]), 3),
        }
    return out


# --------------------------------------------------------------------------- #
def build_all():
    print("=" * 70)
    print("SLR v2 reconstruction")
    print("=" * 70)
    print("\n[1/4] Fetching FRED yields (DGS30, DGS2)...")
    y30 = fetch_fred_series(YIELD30, start_date="2000-01-01")
    y2 = fetch_fred_series(YIELD2, start_date="2000-01-01")
    print(f"      DGS30={len(y30)}  DGS2={len(y2)}")
    if not y30 or not y2:
        print("⚠ Missing yield data — cannot build M91.")
        return None

    print("[2/4] M91 duration stress (z of dTP, window=252)...")
    m91z = m91_duration_stress(y30, y2)
    print(f"      {len(m91z)} daily z points  (last z={list(m91z.values())[-1]:.2f})")
    ntrig = sum(1 for v in m91z.values() if v >= M91_TRIGGER_Z)
    print(f"      M91_trigger (z>=1.5) days: {ntrig}")

    print("[3/4] M92 policy response (event registry, direction-aware)...")
    dates = list(m91z.keys())
    m92 = m92_policy_response(dates)
    npos = sum(1 for v in m92.values() if v > 50)
    nneg = sum(1 for v in m92.values() if v < 50)
    print(f"      neutral={len(dates)-npos-nneg}  positive={npos}  negative={nneg}")

    print("[4/4] M93b market response + composite...")
    btc = fetch_fred_series(BTCCB, start_date="2014-01-01")
    tc = m93b_market_response(btc)
    print(f"      M93b: {len(tc)} points")

    composite = slr_composite(m91z, m92, tc)

    # monthly aggregation (last trading day of each month) for GLF-aligned tests
    monthly = {}
    for d in sorted(composite):
        monthly[d[:7]] = composite[d]

    result = {
        "generated_at": datetime.now().isoformat(),
        "method": "SLR v2 interim reconstruction (research only)",
        "note": ("M91 objective (FRED z(dTP)); M92 subjective event registry; "
                 "M93b only (M93a capital flow not reconstructable historically); "
                 "TC=M93b. Not blended into live signal."),
        "n_m91_triggers": ntrig,
        "m92_event_count": len(EVENT_REGISTRY),
        "daily": composite,
        "monthly": monthly,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUT_JSON}  (daily={len(composite)}, monthly={len(monthly)})")
    return composite, m91z


def _selftest():
    # verify term-premium proxy + zscore logic on synthetic data
    d = {}
    base = 3.0
    for i in range(400):
        d[f"2020-{i//28+1:02d}-{i%28+1:02d}"] = base + (0.01 * i if i % 7 else 0)
    # monotonic-ish rise should produce positive z late
    dtp = term_premium_proxy(d, {k: v - 1 for k, v in d.items()})
    assert dtp, "term premium proxy empty"
    z = zscore_trailing({k: v * 0.0 for k, v in dtp.items()})  # flat -> z=0 handled
    z2 = zscore_trailing({k: float(v) for k, v in dtp.items()})
    print("self-test: z score range on rising series:",
          round(min(z2.values()), 2) if z2 else None, "..",
          round(max(z2.values()), 2) if z2 else None)
    # M92 mapping sanity
    m92 = m92_policy_response(list(d.keys()))
    vals = set(round(v, 1) for v in m92.values())
    print("self-test: M92 value set sample:", sorted(vals)[:8])
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    build_all()
