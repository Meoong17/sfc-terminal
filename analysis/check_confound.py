#!/usr/bin/env python3
"""
check_confound.py — Time-period confounding check for walk-forward validation.

WHEN TO USE:
    After any change to the SFC factor set (new features added, threshold
    recalibration, factor weighting changes), re-run walk_forward_validation.py
    then run this script to verify that the observed predictive pattern
    (high sfc_pct → worse forward returns) is not an artefact of uneven
    time-period distribution across deciles.

WHAT IT CHECKS:
    1. Year distribution per sfc_pct decile — does one decile cluster in
       a specific bull/bear era that explains its forward returns?
    2. Date ranges, BTC price ranges per decile
    3. Detailed comparison of specific deciles of interest

USAGE:
    python3 analysis/walk_forward_validation.py   # generate data first
    python3 analysis/check_confound.py             # check confounds

INPUT:
    Reads .walk_forward_validation.json (written by walk_forward_validation.py)

OUTPUT:
    Year histogram per decile showing distribution density across time,
    plus detailed stats for selected decile comparisons.
"""
import json
from collections import Counter

DATA_FILE = ".walk_forward_validation.json"


def main():
    with open(DATA_FILE) as f:
        data = json.load(f)

    sorted_data = sorted(data, key=lambda x: x["sfc_pct"])
    n = len(sorted_data)
    bin_size = n // 10
    remainder = n % 10

    print("=" * 75)
    print("DATE DISTRIBUTION PER DECILE — Time-Period Confound Check")
    print("=" * 75)

    all_chunks = []
    start = 0
    for q in range(10):
        size = bin_size + (1 if q < remainder else 0)
        chunk = sorted_data[start : start + size]
        all_chunks.append(chunk)
        start += size

        dates = [d["date"] for d in chunk]
        years = [d[:4] for d in dates]
        year_counts = Counter(years)

        lo_pct = chunk[0]["sfc_pct"]
        hi_pct = chunk[-1]["sfc_pct"]

        fwd_7d = [d.get("fwd_return_7d") for d in chunk if d.get("fwd_return_7d") is not None]
        fwd_30d = [d.get("fwd_return_30d") for d in chunk if d.get("fwd_return_30d") is not None]
        m7 = sum(fwd_7d) / len(fwd_7d) if fwd_7d else 0
        m30 = sum(fwd_30d) / len(fwd_30d) if fwd_30d else 0

        print(f"\nQ{q+1}  sfc_pct [{lo_pct:5.1f}, {hi_pct:5.1f}]  7d={m7:+.2f}%  30d={m30:+.2f}%")
        print(f"    Date range: {dates[0]} to {dates[-1]}")

        all_years = sorted(set(year_counts.keys()))
        max_count = max(year_counts.values()) if year_counts else 1
        for y in all_years:
            c = year_counts[y]
            bar = "█" * int(c / max_count * 20) + f" {c}"
            print(f"    {y}: {bar}")

    # Q1 vs Q4 detailed compare
    print("\n" + "=" * 75)
    print("Q1 vs Q4 — DETAILED COMPARISON")
    print("=" * 75)

    for label, q_idx in [("Q1 (lowest sfc_pct)", 0), ("Q4 (highest returns)", 3)]:
        chunk = all_chunks[q_idx]
        dates = [d["date"] for d in chunk]
        years = set(d[:4] for d in dates)
        prices = [d["price"] for d in chunk]

        print(f"\n  {label} (n={len(chunk)}):")
        print(f"    sfc_pct range: [{chunk[0]['sfc_pct']:.1f}, {chunk[-1]['sfc_pct']:.1f}]")
        print(f"    Years covered: {sorted(years)}")
        print(f"    Date range: {dates[0]} to {dates[-1]}")
        print(f"    BTC price range: ${min(prices):.0f} — ${max(prices):.0f}")
        print(f"    BTC price median: ${sorted(prices)[len(prices)//2]:.0f}")


if __name__ == "__main__":
    main()
