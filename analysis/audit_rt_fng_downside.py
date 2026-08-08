#!/usr/bin/env python3
"""
audit_rt_fng_downside.py — the RIGHT test for a defensive/stress input.
FNG (Rt) showed no edge at predicting mean forward RETURN (up/down), but SFC is a
STRESS score. A defensive input earns its large weight if it predicts forward
DOWNSIDE / drawdown / tail risk, not mean return. Test FNG (and the SFC core
replay) against:
  - forward max drawdown (peak-to-trough) over next h days
  - forward worst single-day return
  - forward downside semi-deviation of daily returns
Correct polarity for a defensive input: LOW FNG (fear) -> HIGHER forward drawdown,
i.e. NEGATIVE IC(fng, drawdown). Era-split for stability. Pure analysis.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily, compute_features

HORIZONS = [30, 90]
ERA_CUT = "2022-01-01"


def spearman_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 40:
        return None
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def forward_downside(closes, h):
    """Return arrays over the aligned series:
      maxdd   : max peak-to-trough drawdown magnitude over next h days
      worstday: worst single daily return over next h days
      dsdev   : downside semi-deviation of daily returns over next h days
    np.nan for the last h days (no future window)."""
    n = len(closes)
    rets = np.diff(np.log(closes))  # n-1 daily log returns
    maxdd = np.full(n, np.nan)
    worstday = np.full(n, np.nan)
    dsdev = np.full(n, np.nan)
    for i in range(n - h):
        w = closes[i:i + h + 1]
        peak = np.maximum.accumulate(w)
        dd = (w - peak) / peak          # negative
        maxdd[i] = -float(dd.min())      # positive magnitude of max drawdown
        wr = rets[i:i + h]
        worstday[i] = -float(wr.min())   # positive magnitude of worst day
        neg = wr[wr < 0]
        dsdev[i] = float(np.sqrt(np.mean(neg ** 2))) if len(neg) else 0.0
    return maxdd, worstday, dsdev


def run():
    data = compute_features(load_daily())
    days = np.array([d[:10] for d in data["days"]])
    closes = np.array(data["close"], dtype=float)
    n = len(days)

    # FNG aligned from trend cache (2014-2026), sfc_pct core replay too
    trend = {p["date"]: p for p in json.load(open(".walk_forward_trend_continuation.json"))}
    fng = np.array([trend.get(ds, {}).get("fng") for ds in days], dtype=float)
    sfc = np.array([trend.get(ds, {}).get("sfc_pct") for ds in days], dtype=float)

    # Restrict to canonical window where both exist & close available
    for h in HORIZONS:
        maxdd, worstday, dsdev = forward_downside(closes, h)
        print(f"=== FNG (Rt) vs forward downside, horizon {h}d "
              f"({days[0]} -> {days[-1]}) ===")
        print("  Correct defensive polarity: LOW fng -> HIGHER drawdown => IC(fng,dd) < 0")
        for name, y in (("max drawdown", maxdd), ("worst-day", worstday),
                        ("downside semidev", dsdev)):
            m1 = days < ERA_CUT
            m2 = days >= ERA_CUT
            full = spearman_ic(fng, y)
            e1 = spearman_ic(fng[m1], y[m1])
            e2 = spearman_ic(fng[m2], y[m2])
            fstr = f"{full:+.3f}" if full is not None else "-"
            e1s = f"{e1:+.3f}" if e1 is not None else "-"
            e2s = f"{e2:+.3f}" if e2 is not None else "-"
            flip = "FLIP" if (e1 is not None and e2 is not None and (e1 > 0) != (e2 > 0)) else "cons"
            pol = "DEFENSIVE-OK" if full is not None and full < -0.03 else "weak/none"
            print(f"  {name:16s} IC={fstr} era1={e1s} era2={e2s} [{flip}] -> {pol}")
        # SFC core replay comparison
        sfci = spearman_ic(sfc, maxdd)
        print(f"  [SFC core replay] IC(sfc_pct, maxdd): {sfci:+.3f}" if sfci is not None else "-")
        print()


if __name__ == "__main__":
    run()
