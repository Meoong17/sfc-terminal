#!/usr/bin/env python3
"""
validate_signal_prob.py — REUSABLE predictive-probability validator for ANY
reconstructable signal series (bootstrap P + BIC posterior + era-split).

Reads the point-in-time reconstructed series cached by
walk_forward_imbs_l8.py (date, price, liquidity_stress, expectation_shock,
l8_subset) so NO new FRED fetches are needed, and validates a chosen signal
column against actual forward BTC returns.

Output per signal:
  - QUANTILE gap (bottom-20% vs top-20% of the signal) + bootstrap P(predictive)
  - BIC posterior P(signal improves forecast | data)
  - ERA-SPLIT (3 blocks): does the edge BERTAHAN or FLIP sign?
  - VERDICT per the project's era-stability rule.

USAGE:
    .venv/bin/python analysis/validate_signal_prob.py [--signal liquidity_stress|expectation_shock] [--invert]
    --invert  : treat HIGH signal as LOW stress (e.g. GLF score where high=liquid).
"""
import json
import os
import random
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statsmodels.api import OLS
from walk_forward_imbs_l8 import add_forward_returns, bootstrap_diff_ci, N_BOOTSTRAP

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED = os.path.join(SFC_ROOT, ".walk_forward_imbs_l8.json")
OUTPUT = os.path.join(SFC_ROOT, ".validate_signal_prob.json")

QUANTILE_TAIL = 0.20
HORIZONS = [7, 30]


def bootstrap_probability(lo_g, hi_g, n_boot=N_BOOTSTRAP):
    if len(lo_g) < 2 or len(hi_g) < 2:
        return None
    nl, nh = len(lo_g), len(hi_g)
    n_neg = 0
    for _ in range(n_boot):
        sl = [lo_g[random.randrange(nl)] for _ in range(nl)]
        sh = [hi_g[random.randrange(nh)] for _ in range(nh)]
        if (sum(sh) / nh - sum(sl) / nl) < 0:  # high signal -> lower fwd return
            n_neg += 1
    return n_neg / n_boot


def bic_posterior(values, forward, n_boot=3000):
    x = np.array(values, float); y = np.array(forward, float); n = len(x)
    if n < 20:
        return {"error": "n too small"}
    b0 = float(np.sum((y - y.mean()) ** 2))
    bic0 = n * np.log(b0 / n) + 1 * np.log(n)
    X = np.column_stack([np.ones(n), x])
    m = OLS(y, X).fit()
    rss = float(np.sum(m.resid ** 2))
    bic1 = n * np.log(rss / n) + m.params.shape[0] * np.log(n)
    dbic = bic0 - bic1
    p_h1 = float(np.exp(dbic / 2) / (1 + np.exp(dbic / 2)))
    bf = float(np.exp(dbic / 2))
    from scipy import stats as sps
    ic = sps.spearmanr(x, y).correlation
    signs = []
    for _ in range(n_boot):
        idx = random.sample(range(n), n)
        signs.append(sps.spearmanr(x[idx], y[idx]).correlation)
    return {"n": n, "bic_null": round(bic0, 2), "bic_signal": round(bic1, 2),
            "dBIC": round(dbic, 2), "bayes_factor_10": round(bf, 3),
            "posterior_prob_H1": round(p_h1, 4),
            "spearman_ic": round(float(ic), 4),
            "p_ic_sign": round(sum(1 for s in signs if s < 0) / n_boot, 4),
            "label": ("strong FOR predictive" if p_h1 >= 0.95 else
                      "moderate FOR" if p_h1 >= 0.75 else "weak" if p_h1 >= 0.5
                      else "AGAINST (no edge)")}


def era_split(series, signal_col, invert, horizons):
    pts = [(p["date"], p.get(signal_col), p) for p in series
           if p.get(signal_col) is not None]
    pts.sort(key=lambda t: t[0])
    n = len(pts)
    if n < 30:
        return {"error": "too few"}
    thirds = [pts[:n // 3], pts[n // 3: 2 * n // 3], pts[2 * n // 3:]]
    names = ["era1", "era2", "era3(latest)"]
    out = {}
    for h in horizons:
        fk = f"fwd_return_{h}d"
        out[str(h)] = {}
        for name, block in zip(names, thirds):
            paired = [(p[signal_col], p.get(fk)) for _, _, p in block
                      if p.get(fk) is not None]
            if len(paired) < 10:
                continue
            paired.sort(key=lambda z: z[0])
            tn = max(1, int(len(paired) * QUANTILE_TAIL))
            lo_g = [v for _, v in paired[:tn]]
            hi_g = [v for _, v in paired[-tn:]]
            est, lo_, hi_ = bootstrap_diff_ci(lo_g, hi_g)  # high - low
            prob = bootstrap_probability(lo_g, hi_g)
            # predictive direction: if invert, low signal = high stress
            pred_ok = (est < 0 and not invert) or (est > 0 and invert)
            out[str(h)][name] = {
                "n": len(paired),
                "gap_high_minus_low": round(est, 2) if est is not None else None,
                "ci90": [round(lo_, 2), round(hi_, 2)] if lo_ is not None else None,
                "prob_predictive": round(prob, 3) if prob is not None else None,
                "sig": bool(est is not None and (hi_ < 0 or lo_ > 0)),
                "predictive_dir_ok": bool(pred_ok),
            }
    return out


def validate(series, signal_col, invert=False):
    name = f"{signal_col}{' (inverted)' if invert else ''}"
    print("\n" + "=" * 70)
    print(f"SIGNAL: {name}")
    print("=" * 70)
    res = {"signal": name}
    for h in HORIZONS:
        fk = f"fwd_return_{h}d"
        paired = [(p[signal_col], p.get(fk)) for p in series
                  if p.get(signal_col) is not None and p.get(fk) is not None]
        if len(paired) < 20:
            print(f"  [{h}d] insufficient n={len(paired)}")
            continue
        paired.sort(key=lambda z: z[0])
        tn = max(1, int(len(paired) * QUANTILE_TAIL))
        lo_g = [v for _, v in paired[:tn]]
        hi_g = [v for _, v in paired[-tn:]]
        est, lo_, hi_ = bootstrap_diff_ci(lo_g, hi_g)
        prob = bootstrap_probability(lo_g, hi_g)
        bic = bic_posterior([p[signal_col] for p in series if p.get(fk) is not None
                             and p.get(signal_col) is not None],
                            [p[fk] for p in series if p.get(fk) is not None
                             and p.get(signal_col) is not None])
        pred_ok = (est < 0 and not invert) or (est > 0 and invert)
        print(f"  [{h}d] quantile bottom(n={len(lo_g)}) vs top(n={len(hi_g)}): "
              f"gap={est:+.2f}pp [90% {lo_:+.2f},{hi_:+.2f}] P(predictive)={prob:.3f}")
        print(f"         BIC posterior P(predictive|data)={bic['posterior_prob_H1']} "
              f"(BF={bic['bayes_factor_10']}) {bic['label']}")
        res[str(h)] = {"quantile_gap": round(est, 2), "ci90": [round(lo_, 2), round(hi_, 2)],
                       "prob_predictive": round(prob, 3), "bic_posterior": bic,
                       "predictive_dir_ok": bool(pred_ok)}
    print("  ERA-SPLIT:")
    eras = era_split(series, signal_col, invert, HORIZONS)
    res["era_split"] = eras
    for h, per in eras.items():
        line = " ".join(f"{k}:{v['gap_high_minus_low']}(P={v['prob_predictive']})"
                        for k, v in per.items())
        print(f"    [{h}d] {line}")
    return res


def main():
    if not os.path.exists(CACHED):
        print(f"No cached series at {CACHED}. Run walk_forward_imbs_l8.py first.")
        return
    with open(CACHED) as f:
        series = json.load(f)
    series = add_forward_returns(series)
    print(f"Loaded {len(series)} observations from cache")

    signals = []
    if "--signal" in sys.argv:
        idx = sys.argv.index("--signal")
        col = sys.argv[idx + 1]
        signals.append((col, "--invert" in sys.argv))
    else:
        signals = [("liquidity_stress", False),   # GLF stress: high = tight = stress
                   ("expectation_shock", False),   # L6: high = fragile
                   ("l8_subset", False)]           # composite (redundant with above)

    result = {"generated_at": datetime.now(timezone.utc).isoformat()}
    for col, inv in signals:
        result[col] = validate(series, col, invert=inv)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    random.seed(42)
    main()
