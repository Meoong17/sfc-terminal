#!/usr/bin/env python3
"""
validate_liquidity_momentum.py
==============================
Empirical validation of the liquidity-momentum (LM) stress-adjustment buckets
(analysis/liquidity_momentum.py) against BTC forward returns.

QUESTION UNDER TEST
  Is positive LM (GLF improving over 30d) associated with HIGHER forward BTC
  returns, and negative LM with LOWER returns? Are the hard-coded bucket
  cutoffs (lm_change >3 -> -0.05, >1 -> -0.02, >-1 -> 0, >-3 -> +0.03,
  >-10 -> +0.08, else +0.15) sensible against actual forward returns?

DATA AVAILABILITY — honest limits (investigated first)
  GLF daily history is SHORT: .liq_momentum_cache.json holds only 55 daily
  values (2026-06-28 .. 2026-08-21). Reconstructing a longer GLF from the
  raw macro components in data/merged/sfc_research_daily.json was attempted
  but REJECTED as unreliable: the reconstruction correlates only ~0.29 with
  the real 55-day GLF and has ~6x too little variance (missing DXY / China M2
  components that are live-fetched in the real engine). Using that would
  FABRICATE a GLF series, so only the real 55-day history is used.

  Consequences of the short sample:
    * LM (30-day GLF change) is computable for only ~21 days.
    * forward 30d BTC returns are NOT available for ANY LM day (the GLF
      history is too recent to have 30d of BTC forward data yet).
    * forward 7d returns are available for ~18 LM days.
  This is far too small for any statistically robust conclusion; results are
  reported as indicative only.

Output: analysis/.validate_liquidity_momentum.json
"""
import json, os
import numpy as np
from datetime import date as _D, timedelta
from collections import Counter

SFC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_glf_history():
    with open(os.path.join(SFC, ".liq_momentum_cache.json")) as f:
        return json.load(f)["history"]  # list of {date, glf}

def load_btc_closes():
    with open(os.path.join(SFC, "data/binance_vision_daily.json")) as f:
        bd = json.load(f)
    close = {d: bd[d]["close"] for d in bd}
    # extend with the behavioral-divergence tracker's recorded BTC prices
    try:
        with open(os.path.join(SFC, ".behavioral_divergence_history.json")) as f:
            bh = json.load(f)
        from datetime import datetime
        for e in bh:
            d = datetime.fromisoformat(e["ts"]).strftime("%Y-%m-%d")
            if d not in close:
                close[d] = e["btc_price"]
    except Exception:
        pass
    return close

def glf_on(glf_by, target):
    t = _D.fromisoformat(target)
    for k in sorted(glf_by, reverse=True):
        if _D.fromisoformat(k) <= t:
            return k, glf_by[k]
    return None, None

def close_on(close, target):
    t = _D.fromisoformat(target)
    for k in sorted(close):
        if _D.fromisoformat(k) >= t:
            return k, close[k]
    return None, None

BUCKETS = [
    (3.0, -0.05), (1.0, -0.02), (-1.0, 0.0),
    (-3.0, 0.03), (-10.0, 0.08), (float("-inf"), 0.15),
]

def stress_adjustment(lm):
    for cutoff, adj in BUCKETS:
        if lm > cutoff:
            return adj
    return 0.15

def run():
    hist = load_glf_history()
    glf_by = {h["date"]: h["glf"] for h in hist}
    dates = [h["date"] for h in hist]
    glfs = [h["glf"] for h in hist]
    close = load_btc_closes()

    rows = []
    for i, dt in enumerate(dates):
        if i < 30:
            continue
        glf_now = glfs[i]
        _, glf_prev30 = glf_on(glf_by, (_D.fromisoformat(dt) - timedelta(days=30)).isoformat())
        if glf_prev30 is None:
            continue
        lm = glf_now - glf_prev30
        c_now = close.get(dt)
        if c_now is None:
            continue
        _, p7 = close_on(close, (_D.fromisoformat(dt) + timedelta(days=7)).isoformat())
        _, p30 = close_on(close, (_D.fromisoformat(dt) + timedelta(days=30)).isoformat())
        rows.append({
            "date": dt,
            "lm": round(lm, 3),
            "glf_now": glf_now,
            "glf_prev30": glf_prev30,
            "stress_adj": stress_adjustment(lm),
            "fwd7": (p7 / c_now - 1) * 100 if p7 else None,
            "fwd30": (p30 / c_now - 1) * 100 if p30 else None,
        })

    result = {
        "data_availability": {
            "glf_history_days": len(hist),
            "glf_history_span": (hist[0]["date"], hist[-1]["date"]),
            "lm_observations": len(rows),
            "with_fwd7": sum(1 for r in rows if r["fwd7"] is not None),
            "with_fwd30": sum(1 for r in rows if r["fwd30"] is not None),
            "note": (
                "GLF history too short for any 30d forward validation; only ~18 "
                "7d-forward LM days available. Reconstruction of a longer GLF "
                "series from raw macro components was attempted and REJECTED "
                "(corr ~0.29 vs real GLF, variance ~6x too small)."
            ),
        },
        "lm_to_fwd7_correlation": {},
        "lm_sign_analysis_fwd7": {},
        "bucket_analysis_fwd7": {},
        "rows": rows,
    }

    # correlation LM <-> fwd7
    pairs = [(r["lm"], r["fwd7"]) for r in rows if r["fwd7"] is not None]
    if len(pairs) >= 3:
        lms = np.array([p[0] for p in pairs]); r7 = np.array([p[1] for p in pairs])
        corr = np.corrcoef(lms, r7)[0, 1]
        result["lm_to_fwd7_correlation"] = {
            "n": len(pairs),
            "pearson": round(float(corr), 3),
        }

    # sign analysis: LM>0 vs LM<0 forward 7d
    pos = [r["fwd7"] for r in rows if r["fwd7"] is not None and r["lm"] > 0]
    neg = [r["fwd7"] for r in rows if r["fwd7"] is not None and r["lm"] < 0]
    zero = [r["fwd7"] for r in rows if r["fwd7"] is not None and r["lm"] == 0]
    def stats(vals):
        return {"n": len(vals),
                "mean_pct": round(float(np.mean(vals)), 3) if vals else None,
                "median_pct": round(float(np.median(vals)), 3) if vals else None}
    result["lm_sign_analysis_fwd7"] = {"lm_pos": stats(pos), "lm_neg": stats(neg), "lm_zero": stats(zero)}

    # bucket analysis
    for cutoff, adj in BUCKETS:
        if cutoff == float("-inf"):
            name = "lm_<=-10"
            group = [r["fwd7"] for r in rows if r["fwd7"] is not None and r["lm"] <= -10]
        else:
            # find bucket (cutoff, next_higher_cutoff]
            higher = [c for c, _ in BUCKETS if c > cutoff]
            lo = cutoff
            hi = min(higher) if higher else float("inf")
            name = f"{lo}<lm<={hi}" if hi != float("inf") else f"lm>{lo}"
            group = [r["fwd7"] for r in rows if r["fwd7"] is not None and lo < r["lm"] <= hi]
        result["bucket_analysis_fwd7"][name] = {
            "stress_adj": adj, **stats(group)
        }

    out_path = os.path.join(SFC, "analysis", ".validate_liquidity_momentum.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
