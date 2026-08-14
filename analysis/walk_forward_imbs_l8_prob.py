#!/usr/bin/env python3
"""
walk_forward_imbs_l8_prob.py — Predictive-probability validation of L8 Tail Risk
=================================================================================
Extends walk_forward_imbs_l8.py (reconstructable 2/5-dim subset: GLF liquidity
stress + L6 expectation shock) to deliver DEFENSIBLE PROBABILITIES, not just CIs:

  1. BUCKET gap (LOW-MOD vs HIGH) + quantile (bottom-20% vs top-20%) — as before.
  2. BOOTSTRAP POSTERIOR PROBABILITY: P(gap is in the predictive direction | data)
     = fraction of bootstrap draws where the gap has the correct sign
     (higher L8 stress -> lower forward return). This is a direct, intuitive
     "how likely is this indicator genuinely predictive?" probability.
  3. BIC-based posterior P(H1 | data) that l8_subset improves the forecast of
     forward return over an intercept-only model.
  4. ERA-SPLIT (3 contiguous blocks + pre/post): does the edge BERTAHAN across
     regimes, or is it one-era luck?

HONEST SCOPE: reconstructable 2/5-dim subset only (behavior/leverage/correlation
unavailable pre-2021). Verdict applies to the SUBSET's direction being predictive,
NOT to the live 4-5-dim cutoff (that needs re-calibration on live data).

USAGE:
    cd ~/sfc && export FRED_API_KEY=$(grep -oP '(?<=FRED_API_KEY=).*' .env | tr -d '"')
    .venv/bin/python analysis/walk_forward_imbs_l8_prob.py
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
from walk_forward_imbs_l8 import (
    compute_l8_time_series, add_forward_returns, bootstrap_diff_ci,
    bucket_label, BUCKET_EDGES, FORWARD_HORIZONS_DAYS, QUANTILE_TAIL,
    N_BOOTSTRAP,
)

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(SFC_ROOT, ".walk_forward_imbs_l8_prob.json")


def _bootstrap_probability(group_bottom, group_top, n_boot=N_BOOTSTRAP):
    """P(predictive direction | data): fraction of bootstrap draws where
    the top-stress group has LOWER mean forward return than the bottom group.
    Predictive sign => gap = mean_top - mean_bottom < 0."""
    if len(group_bottom) < 2 or len(group_top) < 2:
        return None, None
    nb, nt = len(group_bottom), len(group_top)
    n_neg = 0
    for _ in range(n_boot):
        sb = [group_bottom[random.randrange(nb)] for _ in range(nb)]
        st = [group_top[random.randrange(nt)] for _ in range(nt)]
        if (sum(st) / nt - sum(sb) / nb) < 0:
            n_neg += 1
    return n_neg / n_boot, n_boot


def _bic_posterior(values, forward, n_boot=3000):
    """P(H1|data): BIC-approx posterior that the signal improves the forecast
    of forward return over an intercept-only model. Also a bootstrap-based
    Spearman IC and its sign-probability."""
    x = np.array(values, float)
    y = np.array(forward, float)
    n = len(x)
    if n < 20:
        return {"error": "n too small"}
    # intercept-only
    b0 = float(np.sum((y - y.mean()) ** 2))
    k0, n0 = 1, n
    bic0 = n * np.log(b0 / n) + k0 * np.log(n)
    # level + signal
    X = np.column_stack([np.ones(n), x])
    m = OLS(y, X).fit()
    rss = float(np.sum(m.resid ** 2))
    bic1 = n * np.log(rss / n) + m.params.shape[0] * np.log(n)
    dbic = bic0 - bic1
    p_h1 = float(np.exp(dbic / 2) / (1 + np.exp(dbic / 2)))
    bf = float(np.exp(dbic / 2))
    # Spearman IC with bootstrap sign probability
    from scipy import stats as sps
    ic = sps.spearmanr(x, y).correlation
    signs = []
    for _ in range(n_boot):
        idx = random.sample(range(n), n)
        xs = x[idx]; ys = y[idx]
        signs.append(sps.spearmanr(xs, ys).correlation)
    p_neg_ic = sum(1 for s in signs if s < 0) / n_boot
    return {
        "n": n,
        "bic_null": round(bic0, 2), "bic_signal": round(bic1, 2),
        "dBIC_null_minus_signal": round(dbic, 2),
        "bayes_factor_10": round(bf, 3),
        "posterior_prob_H1_predictive": round(p_h1, 4),
        "spearman_ic": round(float(ic), 4),
        "p_ic_negative_bootstrap": round(p_neg_ic, 4),
        "label": ("strong evidence FOR predictive" if p_h1 >= 0.95 else
                  "moderate evidence FOR" if p_h1 >= 0.75 else
                  "weak/anecdotal" if p_h1 >= 0.5 else
                  "evidence AGAINST (no predictive edge)"),
    }


def _era_split(series, horizons):
    """Split into 3 contiguous time blocks + measure bucket & quantile gap per era."""
    pts = [(p["date"], p["l8_subset"], p) for p in series
           if p.get("l8_subset") is not None]
    if len(pts) < 30:
        return {"error": "too few points"}
    pts.sort(key=lambda t: t[0])
    n = len(pts)
    thirds = [pts[: n // 3], pts[n // 3: 2 * n // 3], pts[2 * n // 3:]]
    names = ["era1(earliest)", "era2", "era3(latest)"]
    out = {}
    for h in horizons:
        fk = f"fwd_return_{h}d"
        out[str(h)] = {}
        for name, block in zip(names, thirds):
            bottom = [p.get(fk) for _, _, p in block if p.get(fk) is not None]
            vals = [p["l8_subset"] for _, _, p in block if p.get(fk) is not None]
            if len(bottom) < 10:
                continue
            # quantile split within the era
            paired = sorted(zip(vals, bottom))
            tn = max(1, int(len(paired) * QUANTILE_TAIL))
            lo_g = [v for _, v in paired[:tn]]
            hi_g = [v for _, v in paired[-tn:]]
            est, lo_, hi_ = bootstrap_diff_ci(lo_g, hi_g)  # hi(high stress) - lo
            prob, _ = _bootstrap_probability(lo_g, hi_g)
            out[str(h)][name] = {
                "n": len(bottom),
                "quantile_gap_top_minus_bottom": round(est, 2) if est is not None else None,
                "ci90": [round(lo_, 2), round(hi_, 2)] if lo_ is not None else None,
                "prob_predictive": round(prob, 3) if prob is not None else None,
                "sig": bool(est is not None and (hi_ < 0 or lo_ > 0)),
            }
    return out


def main():
    print("=" * 70)
    print("L8 TAIL-RISK SUBSET — PREDICTIVE-PROBABILITY VALIDATION")
    print("(reconstructable 2/5 dims: GLF liquidity stress + L6 expectation shock)")
    print("=" * 70)

    print("\nFetching historical series (~1 min)...")
    series = compute_l8_time_series()
    if not series:
        print("No data — abort.")
        return
    series = add_forward_returns(series)
    print(f"Computed {len(series)} daily observations")

    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "n_periods": len(series)}

    for h in FORWARD_HORIZONS_DAYS:
        fk = f"fwd_return_{h}d"
        # Bucket LOW-MOD vs HIGH
        low, high = [], []
        quant_low, quant_high = [], []
        for p in series:
            if p.get(fk) is None or p.get("l8_subset") is None:
                continue
            lbl = bucket_label(p["l8_subset"])
            if lbl == "LOW-MOD":
                low.append(p[fk])
            elif lbl == "HIGH":
                high.append(p[fk])
        # quantile split across full sample
        pts = sorted((p["l8_subset"], p.get(fk)) for p in series
                     if p.get(fk) is not None and p.get("l8_subset") is not None)
        tn = max(1, int(len(pts) * QUANTILE_TAIL))
        q_low = [v for _, v in pts[:tn]]
        q_high = [v for _, v in pts[-tn:]]

        print(f"\n[{h}d forward]")
        be = None
        if len(low) >= 2 and len(high) >= 2:
            be, blo, bhi = bootstrap_diff_ci(low, high)  # high - low
            bprob, _ = _bootstrap_probability(low, high)
            print(f"  BUCKET LOW-MOD(n={len(low)}) vs HIGH(n={len(high)}): "
                  f"gap={be:+.2f}pp [90% {blo:+.2f},{bhi:+.2f}] "
                  f"prob_predictive={bprob:.3f}")
        qe, qlo, qhi = bootstrap_diff_ci(q_low, q_high)  # high stress - low stress
        qprob, _ = _bootstrap_probability(q_low, q_high)
        print(f"  QUANTILE bottom(n={len(q_low)}) vs top(n={len(q_high)}): "
              f"gap={qe:+.2f}pp [90% {qlo:+.2f},{qhi:+.2f}] "
              f"prob_predictive={qprob:.3f}")

        # BIC posterior on the level signal
        vals = [p["l8_subset"] for p in series if p.get(fk) is not None
                and p.get("l8_subset") is not None]
        fwd = [p[fk] for p in series if p.get(fk) is not None
               and p.get("l8_subset") is not None]
        bic = _bic_posterior(vals, fwd)
        print(f"  BIC posterior P(predictive|data): {bic.get('posterior_prob_H1_predictive')} "
              f"(BF_10={bic.get('bayes_factor_10')}) -> {bic.get('label')}")
        print(f"  Spearman IC={bic.get('spearman_ic')}  "
              f"P(IC<0)={bic.get('p_ic_negative_bootstrap')}")

        result[str(h)] = {
            "bucket": {"n_low": len(low), "n_high": len(high),
                       "gap_high_minus_low": round(be, 2) if be is not None else None,
                       "ci90": [round(blo, 2), round(bhi, 2)] if blo is not None else None,
                       "prob_predictive": round(bprob, 3) if bprob is not None else None},
            "quantile": {"n_bottom": len(q_low), "n_top": len(q_high),
                         "gap_top_minus_bottom": round(qe, 2) if qe is not None else None,
                         "ci90": [round(qlo, 2), round(qhi, 2)] if qlo is not None else None,
                         "prob_predictive": round(qprob, 3) if qprob is not None else None},
            "bic_posterior": bic,
        }

    print("\n" + "=" * 70)
    print("ERA-SPLIT (temporal stability) — quantile gap per 1/3 period")
    print("=" * 70)
    eras = _era_split(series, FORWARD_HORIZONS_DAYS)
    result["era_split"] = eras
    for h, per in eras.items():
        print(f"  [{h}d]")
        for name, d in per.items():
            print(f"    {name:16s} n={d['n']:<6} gap={d['quantile_gap_top_minus_bottom']} "
                  f"prob_predictive={d['prob_predictive']} sig={d['sig']}")

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    random.seed(42)
    main()
