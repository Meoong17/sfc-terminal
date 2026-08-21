#!/usr/bin/env python3
"""
validate_behavioral_divergence.py
=================================
Empirical validation of the behavioral-divergence detector's core hypothesis
(analysis/behavioral_divergence.py) against BTC forward returns.

QUESTION UNDER TEST
  Does a HIDDEN_ACCUMULATION call (price DOWN + flow BULLISH) precede HIGHER
  than-baseline 7d/30d forward BTC returns, and does HIDDEN_DISTRIBUTION
  (price UP + flow BEARISH) precede LOWER forward returns? And is the
  hard-coded DIVERGENCE_THRESHOLD=0.15 sensible?

DATA AVAILABILITY — honest limits (investigated first)
  * The full 3-component composite (M81 ETF flow + Q10 whale pressure + SLI)
    has NO multi-month point-in-time history:
      - .behavioral_divergence_history.json holds only 3 calendar days
        (2026-08-19..21), far too short for 7d/30d forward-return validation.
      - SLI (.stablecoin_intel_cache.json / .stablecoin_cache.json) is a
        single current snapshot + 3 sparse supply_history points -> NOT
        reconstructable point-in-time. SLI leg CANNOT be validated.
  * Two of the three legs ARE reconstructable point-in-time and are
    validated here:
      (1) M81 ETF-flow score  (0-1) — rebuilt daily from the 670-day
          .etf_cache.json net-flow series using the exact bucket logic in
          data_sources/etf_flow.py (5-trading-day avg flow -> 0.15..0.85).
      (2) Q10 whale_pressure  (0-100) — rebuilt daily from the ~1357-day
          .onchain_cache.json raw series using the exact trailing-365d
          percentile scoring in data_sources/onchain_fetch.py (weighted
          group: whale_ratio, exchange_supply_ratio, funding_rates,
          exchange_inflow_total, exchange_outflow_total).
  So this validates the detector on the M81 and M81+Q10 legs. The SLI leg
  and the full composite remain unvalidated for lack of history.

SIGN-CONVENTION NOTE (critical finding)
  In analysis/behavioral_divergence.py the ETF-flow component is mapped as
      components["etf_flow"] = (m81_etf_flow - 0.5) * 2
  But a HIGH m81 score means ETF OUTFLOWS / stress (data_sources/etf_flow.py:
  "0-1 where high = high stress (large outflows)"). With the module's mapping,
  m81=0.85 (heavy outflows, bearish) yields etf_flow=+0.7, i.e. it is ADDED to
  flow_direction_score as if it were BULLISH. That inverts the ETF-flow leg's
  polarity relative to the detector's own convention (positive flow_direction =
  bullish). We therefore report results under BOTH mappings:
    * "module-as-written" (etf_flow = (m81-0.5)*2)
    * "economically-corrected" (etf_flow = (0.5-m81)*2, outflows->bearish)

Output: cache/analysis_behavioral_divergence.json (see results_paths)
"""
import json, os
import numpy as np
from datetime import date as _D
from collections import Counter

SFC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- loaders
def load_etf_flows():
    with open(os.path.join(SFC, ".etf_cache.json")) as f:
        return json.load(f)["flows"]

def _entry_total_btc(f):
    if f.get("total_btc") is not None:
        return f["total_btc"]
    etfs = f.get("etfs")
    if isinstance(etfs, dict) and etfs:
        return sum(v for v in etfs.values() if v is not None)
    return None

def m81_score_for_idx(flows, idx):
    """Replicate data_sources/etf_flow.py M81 (last 5 flow days avg -> bucket)."""
    recent = []
    for j in range(idx, -1, -1):
        tb = _entry_total_btc(flows[j])
        if tb is not None:
            recent.append(tb)
            if len(recent) >= 5:
                break
    if not recent:
        return None
    avg = sum(recent) / len(recent)
    if avg < -1000: return 0.85
    elif avg < -500: return 0.70
    elif avg < -100: return 0.60
    elif avg < 100:  return 0.50
    elif avg < 500:  return 0.35
    elif avg < 1000: return 0.25
    else:            return 0.15

def load_onchain_raw():
    with open(os.path.join(SFC, ".onchain_cache.json")) as f:
        return json.load(f).get("raw", {})

def percentile_score(value, data_vals, direction="pos"):
    if not data_vals or len(data_vals) < 10:
        return 50.0
    sv = sorted(data_vals)
    n = len(sv)
    below = sum(1 for v in sv if v < value)
    pct = below / n
    if direction == "pos":
        return pct * 100
    elif direction == "neg":
        return (1 - pct) * 100
    else:
        return (1 - abs(pct - 0.5) * 2) * 100

# whale_pressure group: (metric, direction, weight)
WHALE_GROUP = [
    ("whale_ratio", "pos", 0.25),
    ("exchange_supply_ratio", "neg", 0.1875),
    ("funding_rates", "neutral", 0.125),
    ("exchange_inflow_total", "pos", 0.25),
    ("exchange_outflow_total", "neg", 0.1875),
]

def load_btc_closes():
    with open(os.path.join(SFC, "data/binance_vision_daily.json")) as f:
        bd = json.load(f)
    return {d: bd[d]["close"] for d in bd}

def btc_on(close, target):
    t = _D.fromisoformat(target)
    for k in sorted(close, reverse=True):
        if _D.fromisoformat(k) <= t:
            return k, close[k]
    return None, None

# ---------------------------------------------------------------- build panel
def build_m81_series(flows):
    out = []
    for i in range(len(flows)):
        m = m81_score_for_idx(flows, i)
        out.append((flows[i]["date"], m))
    return out

def build_q10_series(raw):
    """Rebuild whale_pressure daily using trailing-365d percentile per metric."""
    # Prepare per-metric daily value arrays {date: value}
    metric_series = {}
    for name, _dir, _w in WHALE_GROUP:
        entry = raw.get(name)
        if not entry or not entry.get("data"):
            continue
        s = {}
        for pt in entry["data"]:
            if pt.get("value") is None:
                continue
            d = _D.fromtimestamp(pt["timestamp"] / 1000).isoformat()
            s[d] = pt["value"]
        metric_series[name] = s
    # iterate over union of dates
    all_dates = sorted(set().union(*[set(s) for s in metric_series.values()]) if metric_series else set())
    out = []
    for d in all_dates:
        scores, weights = [], []
        for name, direction, w in WHALE_GROUP:
            if name not in metric_series:
                continue
            vals = sorted([v for dd, v in metric_series[name].items() if dd <= d])[-365:]
            v = metric_series[name].get(d)
            if v is None or len(vals) < 10:
                continue
            scores.append(percentile_score(v, vals, direction))
            weights.append(w)
        if not scores:
            continue
        out.append((d, round(sum(s * w for s, w in zip(scores, weights)) / sum(weights), 1)))
    return out

def classify(etf_comp, whale_comp, btc24h, threshold):
    """Apply compute_behavioral_divergence() logic for available components.
    Returns regime."""
    comps = []
    if etf_comp is not None:
        comps.append(etf_comp)
    if whale_comp is not None:
        comps.append(whale_comp)
    if not comps or btc24h is None:
        return "INSUFFICIENT"
    flow_direction_score = sum(comps) / len(comps)
    price_dir = 1 if btc24h > 0 else -1 if btc24h < 0 else 0
    divergence_raw = max(-1.0, min(1.0, -price_dir * flow_direction_score))
    if divergence_raw > threshold and price_dir < 0:
        return "HIDDEN_ACCUMULATION"
    if divergence_raw > threshold and price_dir > 0:
        return "HIDDEN_DISTRIBUTION"
    if price_dir == 0:
        return "NO_SIGNAL"
    return "NO_DIVERGENCE"

def fwd_ret(close, cdate, close_arr, dates_arr, horizon):
    """Forward simple return horizon trading days later."""
    try:
        i = dates_arr.index(cdate)
    except ValueError:
        return None
    if i + horizon >= len(close_arr):
        return None
    c0 = close_arr[i]
    if c0 == 0:
        return None
    return close_arr[i + horizon] / c0 - 1.0

def analyze_legs():
    flows = sorted(load_etf_flows(), key=lambda f: f["date"])
    raw = load_onchain_raw()
    close = load_btc_closes()
    dates_arr = sorted(close)
    close_arr = np.array([close[d] for d in dates_arr])

    m81_series = build_m81_series(flows)
    q10_series = build_q10_series(raw)
    q10_by = dict(q10_series)

    # Merge into one panel keyed by date (use ETF flow date as the anchor;
    # Q10 value is the trailing-window score as of that same day).
    panel = []
    for date, m81 in m81_series:
        if m81 is None:
            continue
        k, c = btc_on(close, date)
        if k is None:
            continue
        q10 = q10_by.get(k)
        if q10 is None:  # nearest <= k
            for qd in sorted(q10_by, reverse=True):
                if qd <= k:
                    q10 = q10_by[qd]
                    break
        ki = dates_arr.index(k)
        if ki == 0:
            continue
        btc24h = (close_arr[ki] / close_arr[ki - 1] - 1) * 100
        row = {
            "date": date,
            "m81": m81,
            "q10": q10,
            "btc24h": float(btc24h),
            "close": float(c),
        }
        for h in (7, 30):
            row[f"fwd{h}"] = fwd_ret(close, k, close_arr, dates_arr, h)
        panel.append(row)
    return panel

def run(thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    panel = analyze_legs()
    results = {
        "data_availability": {
            "behavioral_divergence_history_days": 3,
            "behavioral_divergence_history_span": "2026-08-19..2026-08-21",
            "sl_leg_reconstructable": False,
            "sl_leg_reason": "stablecoin intel is a single snapshot; stablecoin_cache has only 3 supply_history points",
            "m81_etf_leg_reconstructable": True,
            "m81_leg_days": len([p for p in panel if p["m81"] is not None]),
            "q10_whale_leg_reconstructable": True,
            "q10_leg_days": len([p for p in panel if p["q10"] is not None]),
            "full_composite_validatable": False,
            "full_composite_reason": "no multi-month point-in-time SLI/M81+Q10+SLI panel; tracker only 3 days",
        },
        "n_panel_days": len(panel),
        "panel_span": (panel[0]["date"], panel[-1]["date"]) if panel else None,
        "baseline_forward_returns": {},
        "sign_convention_note": (
            "Module maps etf_flow=(m81-0.5)*2; high m81=outflows/stress, so module-as-written "
            "counts ETF outflows as bullish (inverted polarity). Reported under both mappings."
        ),
    }
    # baseline
    for h in (7, 30):
        vals = [p[f"fwd{h}"] for p in panel if p.get(f"fwd{h}") is not None]
        results["baseline_forward_returns"][str(h)] = {
            "n": len(vals), "mean_pct": round(float(np.mean(vals) * 100), 3) if vals else None,
            "median_pct": round(float(np.median(vals) * 100), 3) if vals else None,
        }

    for label, m81_map in [
        ("module_as_written", lambda m: (m - 0.5) * 2),
        ("economically_corrected", lambda m: (0.5 - m) * 2),
    ]:
        results[label] = {"threshold_sweep": {}}
        for thr in thresholds:
            acc, dis = [], []
            for p in panel:
                etf_comp = m81_map(p["m81"])
                whale_comp = (p["q10"] - 50) / 50 if p["q10"] is not None else None
                reg = classify(etf_comp, whale_comp, p["btc24h"], thr)
                if reg == "HIDDEN_ACCUMULATION":
                    acc.append(p)
                elif reg == "HIDDEN_DISTRIBUTION":
                    dis.append(p)
            entry = {}
            for h in (7, 30):
                for name, group in (("acc", acc), ("dis", dis)):
                    vals = [p[f"fwd{h}"] for p in group if p.get(f"fwd{h}") is not None]
                    entry[f"{name}_{h}d"] = {
                        "n": len(vals),
                        "mean_pct": round(float(np.mean(vals) * 100), 3) if vals else None,
                        "median_pct": round(float(np.median(vals) * 100), 3) if vals else None,
                    }
            entry["n_acc"] = len(acc)
            entry["n_dis"] = len(dis)
            results[label]["threshold_sweep"][str(thr)] = entry

    # Full-width regime breakdown at 0.15 (both legs combined: M81+Q10)
    results["m81q10_leg_combined_threshold_0.15"] = {}
    for label, m81_map in [
        ("module_as_written", lambda m: (m - 0.5) * 2),
        ("economically_corrected", lambda m: (0.5 - m) * 2),
    ]:
        regimes = Counter()
        by_regime = {}
        for p in panel:
            etf_comp = m81_map(p["m81"])
            whale_comp = (p["q10"] - 50) / 50 if p["q10"] is not None else None
            reg = classify(etf_comp, whale_comp, p["btc24h"], 0.15)
            regimes[reg] += 1
            by_regime.setdefault(reg, []).append(p)
        detail = {"regime_counts": dict(regimes), "regimes": {}}
        for reg, group in by_regime.items():
            detail["regimes"][reg] = {}
            for h in (7, 30):
                vals = [p[f"fwd{h}"] for p in group if p.get(f"fwd{h}") is not None]
                detail["regimes"][reg][f"fwd{h}"] = {
                    "n": len(vals),
                    "mean_pct": round(float(np.mean(vals) * 100), 3) if vals else None,
                    "median_pct": round(float(np.median(vals) * 100), 3) if vals else None,
                }
        results["m81q10_leg_combined_threshold_0.15"][label] = detail

    out_path = os.path.join(SFC, "analysis", ".validate_behavioral_divergence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
