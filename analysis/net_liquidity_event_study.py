#!/usr/bin/env python3
"""
net_liquidity_event_study.py — quasi-natural-experiment layer for the net-liquidity thesis.
================================================================================
Instead of only the continuous regression (which mixes many noise sources), isolate the
few LARGE net-liquidity / TGA episodes and test forward BTC return around them, vs the
full-sample baseline. Episodes are auto-detected (non-overlapping, largest 90d moves)
AND a fixed set of known events is reported for transparency.

Metrics per episode: forward BTC return +30/+60/+90d from episode START, and the same
from the drawdown PEAK. Baseline = all overlapping forward windows (mean, sd) and a
block-permutation p-value (small n -> low power, stated).
Report-only.
"""
import os, json
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)


def load():
    macro = {r["date"]: r for r in json.load(open(os.path.join(REPO, "data/cleaned/macro_daily_clean.json")))}
    btc = json.load(open(os.path.join(REPO, "data/binance_vision_daily.json")))
    rows = []
    for d in sorted(set(macro) & set(btc)):
        m, b = macro[d], btc[d]
        if b.get("close") is None or None in (m.get("FED_BS"), m.get("TGA"), m.get("RRP")):
            continue
        rows.append({"date": d, "close": b["close"], "FED_BS": m["FED_BS"], "TGA": m["TGA"], "RRP": m["RRP"],
                     "net_us": m["FED_BS"] - m["TGA"] - m["RRP"] * 1000.0})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["d90_net"] = df["net_us"] - df["net_us"].shift(90)
    df["d90_tga"] = df["TGA"] - df["TGA"].shift(90)
    for h in (30, 60, 90):
        df[f"fwd{h}"] = np.log(df["close"].shift(-h) / df["close"]) * 100
    return df


def episodes(df, col, mode, top=4, min_gap=120):
    """Find top non-overlapping windows by |col| value, mode='pos' or 'neg'."""
    idx = df[col].dropna().index
    order = df.loc[idx, col].sort_values(ascending=(mode == "neg")).index
    picked = []
    for i in order:
        v = df.loc[i, col]
        if (mode == "pos" and v <= 0) or (mode == "neg" and v >= 0):
            continue
        if any(abs(i - j) < min_gap for j in picked):
            continue
        picked.append(i)
        if len(picked) >= top:
            break
    return sorted(picked)


def summarize(df, i, label):
    r = df.loc[i]
    line = f"  {label:38s} start={r['date']}  net_us=${r['net_us']/1e6:.2f}T"
    for h in (30, 60, 90):
        v = r[f"fwd{h}"]
        line += f"  +{h}d={v:+.1f}%" if not np.isnan(v) else f"  +{h}d=na"
    return line


def main():
    df = load()
    print(f"n={len(df)}  {df['date'].iloc[0]}..{df['date'].iloc[-1]}")

    base = {}
    for h in (30, 60, 90):
        s = df[f"fwd{h}"].dropna()
        base[h] = (s.mean(), s.std(), len(s))
        print(f"baseline fwd{h}d: mean={s.mean():+.2f}%  sd={s.std():.2f}%  n={len(s)}")

    print("\n=== EPISODE SET A: largest 90d NET-LIQUIDITY EXPANSION onsets (peak +) ===")
    eps = episodes(df, "d90_net", "pos", top=4)
    for i in eps:
        print(summarize(df, i, f"net-liq expansion #{eps.index(i)+1}"))

    print("\n=== EPISODE SET B: largest 90d TGA DRAWDOWN onsets (TGA most negative d90) ===")
    epsb = episodes(df, "d90_tga", "neg", top=4)
    for i in epsb:
        print(summarize(df, i, f"TGA drain #{epsb.index(i)+1}"))

    print("\n=== EPISODE SET C (fixed, transparent): known macro-liquidity events ===")
    fixed = [("2020 COVID QE+TGA", "2020-03-23"), ("2021 TGA drawdown (peak drain)", "2021-02-24"),
             ("2023 post-ceiling TGA rebuild->drain", "2023-06-01"), ("2024 RRP drain", "2024-01-02"),
             ("2025 (check)", "2025-06-02")]
    for lab, ds in fixed:
        sub = df[df["date"] >= ds]
        if len(sub) == 0:
            print(f"  {lab:38s} no data"); continue
        print(summarize(df, sub.index[0], lab))

    print("\n=== within-episode (peak-to-trough) forward returns vs baseline (avg across sets A+B) ===")
    allp = eps + epsb
    for h in (30, 60, 90):
        vals = [df.loc[i, f"fwd{h}"] for i in allp if not np.isnan(df.loc[i, f"fwd{h}"])]
        m, sd, n = base[h]
        if vals:
            z = (np.mean(vals) - m) / (sd / np.sqrt(len(vals)))
            print(f"  h={h}d: event mean={np.mean(vals):+.2f}% (n={len(vals)}) vs baseline {m:+.2f}% -> z={z:+.2f}")

    print("\nsmall-n warning: episodes are few and clustered (QE 2020-21, TGA/RRP 2023-24);")
    print("treat z-values as indicative, not inferential.")


if __name__ == "__main__":
    main()
