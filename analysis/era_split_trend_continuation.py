#!/usr/bin/env python3
"""
era_split_trend_continuation.py — era-split the live trend-continuation probabilities.

The dashboard displays cont_prob_30d/90d/180d = P(forward return > 0) conditioned on the
current SFC signal bucket (from walk_forward_trend_continuation.py). That cache is a
FULL-SAMPLE frequency (calibrated by construction), but it has NO era-split — so a bucket
probability can be driven by old eras while the LATEST era behaves differently (the exact
"aggregate probability ≠ today's probability" trap). This script era-splits it offline by
reusing the cached daily series (NO FRED re-fetch), reports per-era per-bucket P(cont) + n,
and sets an era_stable flag mirroring the stress-gap card discipline (era2 & era3 both
significantly on the same side of baseline).

Pure analysis. No production change. Run from repo root:
    cd ~/sfc && .venv/bin/python analysis/era_split_trend_continuation.py
"""
import json, os, random, sys
from datetime import datetime, timezone

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = os.path.join(SFC_ROOT, ".walk_forward_trend_continuation.json")
OUT = os.path.join(SFC_ROOT, ".trend_continuation_era.json")

HORIZONS = [30, 90, 180]
BUCKET_EDGES = [(0, 25, "CALM"), (25, 45, "ELEVATED"), (45, 101, "STRESS")]


def era_of(date_str):
    y = int(date_str[:4])
    return "era1" if y < 2018 else ("era2" if y < 2022 else "era3")


def bucket_label(pct):
    for lo, hi, lbl in BUCKET_EDGES:
        if lo <= pct < hi:
            return lbl
    return "STRESS"


def bootstrap_prob(vals, n_boot=2000, ci=0.90):
    vals = [v for v in vals if v is not None]
    if len(vals) < 10:
        return None, None, None, len(vals)
    p = sum(1 for v in vals if v > 0) / len(vals)
    probs = sorted(sum(1 for _ in range(len(vals)) if vals[random.randrange(len(vals))] > 0) / len(vals)
                   for _ in range(n_boot))
    lo = int((1 - ci) / 2 * n_boot); hi = int((1 + ci) / 2 * n_boot) - 1
    return round(p, 3), round(probs[lo], 3), round(probs[hi], 3), len(vals)


def main():
    random.seed(42)
    rows = json.load(open(SERIES))
    print(f"Loaded {len(rows)} daily rows ({rows[0]['date']} -> {rows[-1]['date']})")

    # unconditional baseline per era per horizon
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "source": os.path.basename(SERIES),
               "caveat": "Reduced-set replay (price/DXY/M2/FNG). Per-era frequencies; "
                         "era_stable = era2 AND era3 both on same side of baseline."}
    for h in HORIZONS:
        by_era = {"era1": [], "era2": [], "era3": []}
        for r in rows:
            if r.get(f"fwd_{h}d") is not None:
                by_era[era_of(r["date"])].append(r[f"fwd_{h}d"])
        print(f"\n[{h}d] Unconditional P(trend continues) by era:")
        for e in ["era1", "era2", "era3"]:
            p, lo, hi, n = bootstrap_prob(by_era[e])
            summary.setdefault(f"baseline_p_cont_{h}d_era", {})[e] = p
            summary.setdefault(f"baseline_n_{h}d_era", {})[e] = n
            print(f"    {e}: P(cont)={p} [CI {lo},{hi}] n={n}")

        # per-bucket per-era
        for _, _, lbl in BUCKET_EDGES:
            key = f"{lbl.lower()}_p_cont_{h}d"
            buckets_era = {"era1": [], "era2": [], "era3": []}
            for r in rows:
                if r.get(f"fwd_{h}d") is None:
                    continue
                if bucket_label(r["sfc_pct"]) == lbl:
                    buckets_era[era_of(r["date"])].append(r[f"fwd_{h}d"])
            per_era = {}
            for e in ["era1", "era2", "era3"]:
                p, lo, hi, n = bootstrap_prob(buckets_era[e])
                per_era[e] = {"p": p, "ci": [lo, hi] if lo is not None else None, "n": n}
                print(f"    {lbl:<9} {e}: P(cont)={p} [CI {lo},{hi}] n={n}")
            summary[key] = per_era

            # era-stability: CALM bucket is the live-relevant one. Use CALM vs baseline
            # spread sign across era2 & era3.
            if lbl == "CALM":
                b2 = summary.get(f"baseline_p_cont_{h}d_era", {}).get("era2")
                b3 = summary.get(f"baseline_p_cont_{h}d_era", {}).get("era3")
                c2 = (per_era["era2"]["p"] - b2) if per_era["era2"]["p"] is not None and b2 else None
                c3 = (per_era["era3"]["p"] - b3) if per_era["era3"]["p"] is not None and b3 else None
                same_side = (c2 is not None and c3 is not None and (c2 > 0) == (c3 > 0) and abs(c2) > 0.005 and abs(c3) > 0.005)
                summary[f"calm_p_cont_{h}d_era_stable"] = bool(same_side)
                summary[f"calm_p_cont_{h}d_calm_minus_base_era2"] = round(c2, 3) if c2 is not None else None
                summary[f"calm_p_cont_{h}d_calm_minus_base_era3"] = round(c3, 3) if c3 is not None else None
                print(f"    -> CALM-vs-baseline era2={c2:+.3f} era3={c3:+.3f} "
                      f"era_stable={bool(same_side)}")

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {OUT}")


if __name__ == "__main__":
    main()
