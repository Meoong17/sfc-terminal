#!/usr/bin/env python3
"""
revalidate_canonical.py — re-validate a cached macro walk-forward factor using the
CANONICAL Binance BTCUSDT price (2017-2026, 9 yr) instead of the original source,
with era-split stability check.

Motivation: original walk-forwards used FRED CBBTCUSD (2014+) or mixed sources.
The canonical Binance close is exchange-traded, consistent with SFC live, and
clean over 2017+. Re-running on it tests whether the factor's predictive sign
survives a cleaner 9-yr series AND across two eras (2017-21 vs 2022-26) — the
era-stability test that rejected China M2 / JPY carry / HY.

Usage:
    .venv/bin/python analysis/revalidate_canonical.py \
        --factor .walk_forward_imbs_l8.json --signal l8_subset \
        --factor-name "L8 subset (GLF liq + expect)"
    .venv/bin/python analysis/revalidate_canonical.py \
        --factor .walk_forward_trend_continuation.json --signal sfc_pct \
        --factor-name "SFC pct (trend continuation)"

Pure analysis. Does not touch production.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_sources.binance_features import load_daily

HORIZONS = [7, 30, 90, 180, 365]
ERA_SPLIT = "2022-01-01"  # era1 < split <= era2


def load_canonical():
    daily = load_daily()
    return {d: r["close"] for d, r in daily.items() if "close" in r}


def join(factor, canon):
    """Return rows: (date, signal, close). Only dates in canonical window."""
    rows = []
    for p in factor:
        d = p.get("date")
        sig = p.get("signal_key")
        if d in canon and sig is not None:
            rows.append((d, sig, canon[d]))
    rows.sort(key=lambda x: x[0])
    return rows


def quantile_gap(x, fwd, q=0.20, nboot=3000, seed=42, boot=True):
    n = len(x)
    if n < 40:
        return None, (None, None), None
    pts = sorted(zip(x, fwd))
    xs = np.array([a for a, _ in pts]); fs = np.array([b for _, b in pts])
    tail = int(n * q)
    bot = fs[:tail].mean(); top = fs[-tail:].mean()
    obs = bot - top  # low-signal fwd - high-signal fwd (stress high -> return low => positive)
    if not boot:
        return obs, None, n
    rng = np.random.default_rng(seed)
    idx = np.arange(n); gaps = []
    for _ in range(nboot):
        b = rng.choice(idx, n, replace=True)
        p = sorted(zip(x[b], fwd[b]))
        f = np.array([v for _, v in p]); t = int(n * q)
        gaps.append(f[:t].mean() - f[-t:].mean())
    gaps = np.array(gaps)
    return obs, (np.percentile(gaps, 5), np.percentile(gaps, 95)), n


def run(factor_path, signal_key, name):
    canon = load_canonical()
    factor = json.load(open(factor_path))
    rows = join([{**p, "signal_key": p.get(signal_key)} for p in factor], canon)
    if not rows:
        print(f"  no overlap for {name}"); return
    dates = [d for d, _, _ in rows]
    closes = np.array([c for _, _, c in rows])
    n = len(rows)
    sig = np.array([s for _, s, _ in rows])
    print(f"\n{'='*70}\nREVALIDATE CANONICAL: {name}\n"
          f"canonical window {dates[0]} -> {dates[-1]} ({n} days, 9yr Binance close)")
    print(f"(original source: {factor_path})")

    # full-window quantile gap
    for h in HORIZONS:
        if h >= n: continue
        fwd = np.array([closes[i+h]/closes[i]-1.0 for i in range(n-h)])
        x, f = sig[:len(fwd)], fwd
        obs, ci, nn = quantile_gap(x, f)
        if obs is None: continue
        sig_ = "***" if (ci[0] > 0 and ci[1] > 0) else ("+++" if (ci[0] < 0 and ci[1] < 0) else "")
        print(f"  [{h:>3}d] low-signal fwd - high-signal fwd = {obs:+.4f} "
              f"[CI90 {ci[0]:+.3f},{ci[1]:+.3f}] {sig_}  (n={nn})")

    # era-split stability (bottom-top gap must keep SAME SIGN in both eras)
    print(f"\n  ERA-SPLIT (split {ERA_SPLIT}): low-signal - high-signal fwd gap, 30d/90d")
    for h in [30, 90]:
        if h >= n: continue
        line = f"    {h:>3}d: "
        era_res = []
        for lbl, (d0, d1) in {"era1(2017-21)": ("2010-01-01", ERA_SPLIT),
                              "era2(2022-26)": (ERA_SPLIT, "2030-01-01")}.items():
            idx = [i for i, d in enumerate(dates) if d0 <= d < d1 and i + h < n]
            if len(idx) < 20:
                line += f"{lbl}=n<20  "; era_res.append(None); continue
            fwd = np.array([closes[i+h]/closes[i]-1.0 for i in idx])
            x = sig[[i for i in idx]]
            obs, _, _ = quantile_gap(x, fwd, boot=False)
            g = f"{obs:+.4f}" if obs is not None else "NA"
            line += f"{lbl}={g}  "
            era_res.append(obs)
        line += f"  sign_consistent={'YES' if era_res[0] and era_res[1] and era_res[0]*era_res[1]>0 else 'NO'}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--factor-name", default="factor")
    a = ap.parse_args()
    run(a.factor, a.signal, a.factor_name)
