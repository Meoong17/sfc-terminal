#!/usr/bin/env python3
"""
slr_event_study_daily.py — SLR event study at DAILY frequency (Test #2 + #3)
=============================================================================
The monthly Test #1 (SLR.md Section 8) was NULL: SLR adds no monthly predictive
edge over GLF. But the short-horizon signal survived (weekly lag 1-2, event
windows 7-30d). This script tests the SLR thesis at the FREQUENCY IT WAS DESIGNED
FOR — daily/event-driven — rigorously:

  Test #2 (policy-response attribution):
      Detect sovereign-duration-stress events OBJECTIVELY from M91 (z>=1.5
      threshold), classify each by the policy-response score (M92), and compare
      BTC forward returns (1/3/7/14/30d) across Positive vs Neutral vs Negative.

  Test #3 (policy placebo / baseline control):
      Does the same BTC response happen after (a) random non-stress dates, or
      (b) unconditional baseline? If yes, the stress-event response is just BTC
      drift, not SLR-specific. This is the placebo SLR.md mandates.

HONEST FRAMING: tiny n on a manual event registry is a hypothesis, not a
conclusion. Any verdict is reported as-is.

USAGE:
    cd ~/sfc && .venv/bin/python analysis/slr_event_study_daily.py
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_ROOT, "analysis"))

from historical_backtest_m1m6 import fetch_fred_series

SLR_JSON = os.path.join(SFC_ROOT, ".slr_series.json")
OUTPUT = os.path.join(SFC_ROOT, ".slr_event_study.json")

M91_TRIGGER_SCORE = 50.0     # z>=1.5 -> m91 score (1.5/3*100)
EVENT_GAP_DAYS = 10          # coalesce trigger days within 10d into one event
M92_POS = 55.0               # policy score > 55 = positive response
M92_NEG = 45.0               # policy score < 45 = negative response
HORIZONS = [1, 3, 7, 14, 30]


def load_slr_daily():
    with open(SLR_JSON) as f:
        data = json.load(f)
    return data["daily"]


def detect_stress_events(slr_daily):
    """Consecutive days with m91>=50 (z>=1.5) coalesced into distinct events.
    Returns list of (event_start_date, m91_score, m92_score)."""
    dates = sorted(slr_daily)
    triggers = [d for d in dates if slr_daily[d]["m91"] >= M91_TRIGGER_SCORE]
    if not triggers:
        return []
    events = []
    cur = [triggers[0]]
    for d in triggers[1:]:
        if (datetime.strptime(d, "%Y-%m-%d") -
                datetime.strptime(cur[-1], "%Y-%m-%d")).days <= EVENT_GAP_DAYS:
            cur.append(d)
        else:
            start = cur[0]
            events.append((start, slr_daily[start]["m91"], slr_daily[start]["m92"]))
            cur = [d]
    start = cur[0]
    events.append((start, slr_daily[start]["m91"], slr_daily[start]["m92"]))
    return events


def fwd_returns(btc_daily, start_date, horizons):
    """{h: forward pct return over h days from start_date} or None."""
    pdates = sorted(btc_daily)
    # find index of first price on/after start_date
    si = None
    for i, d in enumerate(pdates):
        if d >= start_date:
            si = i
            break
    if si is None:
        return {h: None for h in horizons}
    p0 = btc_daily[pdates[si]]
    out = {}
    for h in horizons:
        ei = si + h
        if ei >= len(pdates):
            out[h] = None
            continue
        p1 = btc_daily[pdates[ei]]
        out[h] = (p1 - p0) / p0 * 100.0 if p0 else None
    return out


def classify_policy(m92):
    if m92 >= M92_POS:
        return "pos"
    if m92 <= M92_NEG:
        return "neg"
    return "neutral"


def bootstrap_ci(arr, nboot=10000, alpha=0.10):
    arr = np.asarray([x for x in arr if x is not None], float)
    if len(arr) < 2:
        return None
    rng = np.random.default_rng(42)
    means = np.empty(nboot)
    for i in range(nboot):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(arr.mean()), float(np.percentile(means, alpha / 2 * 100)), \
        float(np.percentile(means, (1 - alpha / 2) * 100))


def placebo_control(btc_daily, n_draws, horizons, n_samples=2000, seed=42):
    """Random non-stress dates: mean forward return over n_draws random dates.
    Returns {h: (mean, ci_lo, ci_hi)} over bootstrap draws of n_draws dates.
    Precomputes all forward returns once (fast)."""
    pdates = sorted(btc_daily)
    maxh = max(horizons)
    # precompute forward return for every usable date
    pre = {h: [] for h in horizons}   # aligned with usable dates
    usable = pdates[: len(pdates) - maxh]
    for i, d in enumerate(usable):
        p0 = btc_daily[d]
        if not p0:
            continue
        for h in horizons:
            p1 = btc_daily[pdates[i + h]]
            pre[h].append((p1 - p0) / p0 * 100.0 if p0 else None)
    usable_clean = usable[: len(pre[horizons[0]])]
    rng = np.random.default_rng(seed)
    out = {}
    idx_range = np.arange(len(usable_clean))
    for h in horizons:
        arr = np.asarray([x for x in pre[h] if x is not None], float)
        means = np.empty(n_samples)
        for s in range(n_samples):
            picks = rng.choice(arr, size=n_draws, replace=True)
            means[s] = picks.mean()
        out[h] = (float(means.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95)))
    return out


def main():
    print("=" * 74)
    print("SLR DAILY EVENT STUDY (Test #2 attribution + Test #3 placebo)")
    print("=" * 74)

    slr_daily = load_slr_daily()
    print(f"\nLoaded SLR daily: {len(slr_daily)} days")

    print("\n[1] Fetching BTC daily closes (FRED CBBTCUSD)...")
    btc_daily = fetch_fred_series("CBBTCUSD")
    print(f"    {len(btc_daily)} closes ({min(btc_daily)}..{max(btc_daily)})")

    print("\n[2] Detecting sovereign-duration-stress events (M91 z>=1.5)...")
    events = detect_stress_events(slr_daily)
    print(f"    {len(events)} distinct stress events")
    for e in events[:15]:
        print(f"      {e[0]}  M91={e[1]:.0f}  M92={e[2]:.0f} ({classify_policy(e[2])})")
    if len(events) > 15:
        print(f"      ... and {len(events)-15} more")

    print("\n[3] Classifying by policy response & computing forward returns...")
    groups = {"pos": [], "neg": [], "neutral": []}
    ev_details = []
    for (start, m91s, m92s) in events:
        pol = classify_policy(m92s)
        fr = fwd_returns(btc_daily, start, HORIZONS)
        groups[pol].append(fr)
        ev_details.append({"date": start, "policy": pol, "m91": m91s, "m92": m92s, "fwd": fr})

    result = {"generated_at": datetime.now().isoformat(),
              "n_events": len(events),
              "n_by_policy": {k: len(v) for k, v in groups.items()},
              "events": ev_details,
              "horizons": HORIZONS}

    print("\n[4] Forward BTC return by policy response (Test #2)...")
    print(f"    {'group':<10}" + "".join(f"{h:>9}d" for h in HORIZONS))
    for pol in ("pos", "neg", "neutral"):
        cells = []
        for h in HORIZONS:
            vals = [g[h] for g in groups[pol] if g[h] is not None]
            cells.append((vals, bootstrap_ci(vals)))
        row = [f"{pol:<10}"]
        for h, (vals, ci) in zip(HORIZONS, cells):
            if ci is None:
                row.append(f"{'n/a':>9}")
            else:
                row.append(f"{ci[0]:>6.1f}±{ci[2]-ci[1]:<3.1f}")
        print("  " + "".join(row))
        result.setdefault("by_policy", {})[pol] = {
            h: {"n": len([g[h] for g in groups[pol] if g[h] is not None]),
                "mean": round(bootstrap_ci([g[h] for g in groups[pol] if g[h] is not None])[0], 3)
                if bootstrap_ci([g[h] for g in groups[pol] if g[h] is not None]) else None,
                "ci90": [round(bootstrap_ci([g[h] for g in groups[pol] if g[h] is not None])[1], 3),
                         round(bootstrap_ci([g[h] for g in groups[pol] if g[h] is not None])[2], 3)]
                if bootstrap_ci([g[h] for g in groups[pol] if g[h] is not None]) else None}
            for h in HORIZONS}

    # Pos vs Neg difference test
    print("\n[5] Pos vs Neg difference (direct bootstrap)...")
    for h in HORIZONS:
        pv = [g[h] for g in groups["pos"] if g[h] is not None]
        nv = [g[h] for g in groups["neg"] if g[h] is not None]
        if len(pv) < 3 or len(nv) < 3:
            print(f"    h={h}d: insufficient n (pos={len(pv)}, neg={len(nv)})")
            continue
        rng = np.random.default_rng(42)
        nboot = 20000
        diffs = np.empty(nboot)
        for i in range(nboot):
            a = rng.choice(pv, size=len(pv), replace=True).mean()
            b = rng.choice(nv, size=len(nv), replace=True).mean()
            diffs[i] = a - b
        lo, hi = np.percentile(diffs, [5, 95])
        print(f"    h={h}d: pos-neg = {pv_mean if False else np.mean(pv)-np.mean(nv):+.1f}pp  "
              f"CI90=[{lo:+.1f},{hi:+.1f}]  {'SIG' if lo>0 or hi<0 else 'n.s.'}")
        result.setdefault("pos_vs_neg", {})[h] = {
            "n_pos": len(pv), "n_neg": len(nv),
            "diff_mean": round(float(np.mean(pv) - np.mean(nv)), 3),
            "ci90": [round(float(lo), 3), round(float(hi), 3)],
            "significant": bool(lo > 0 or hi < 0)}

    print("\n[6] Placebo — random non-stress dates (Test #3)...")
    placebo = placebo_control(btc_daily, n_draws=len(events), horizons=HORIZONS)
    print(f"    {'horizon':<10}{'random-date mean':<18}{'stress-event mean':<18}")
    # stress-event unconditional mean
    all_events_fwd = {h: [e["fwd"][h] for e in ev_details if e["fwd"].get(h) is not None] for h in HORIZONS}
    for h in HORIZONS:
        pm, pl, ph = placebo[h]
        se = np.mean(all_events_fwd[h]) if all_events_fwd[h] else None
        se_s = f"{se:+.1f}" if se is not None else "n/a"
        print(f"    {h:<10}{pm:+.1f}  [{pl:+.1f},{ph:+.1f}]{'':<4}{se_s}")
        result.setdefault("placebo", {})[h] = {
            "random_mean": round(pm, 3), "random_ci90": [round(pl, 3), round(ph, 3)],
            "stress_mean": round(se, 3) if se is not None else None}

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
