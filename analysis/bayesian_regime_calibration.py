#!/usr/bin/env python3
"""
bayesian_regime_calibration.py — implement the Bisa.docx proposal's MANDATORY test
====================================================================================
The doc argues SFC should output P(Bullish regime) + separate P(Tail Risk), and the
wajib test is CALIBRATION: "when SFC says 70% bullish, BTC must actually be bullish
~70% of the time on that horizon."

This builds that framework on RECONSTRUCTABLE evidence (point-in-time, no look-ahead):

  OUTCOME (objective, calibratable) : y_h = 1 if forward BTC return over h days > 0
  EVIDENCE  (reconstructable macro) : liquidity_stress (GLF) + expectation_shock (L6),
                                      from the cached point-in-time series.
  MODEL     : logistic regression P(y=1 | evidence), EXPANDING-WINDOW WALK-FORWARD OOS
              (no look-ahead — the model only sees data before each prediction day).
  TEST      : CALIBRATION on OOS probabilities — reliability binning, Brier score,
              Hosmer-Lemeshow, and the doc's headline check (does a 60-80% bucket
              really produce ~60-80% positives?). Era-split to reveal regime drift.

HONEST SCOPE: evidence = 2/5-dim reconstructable subset (behavior/leverage/correlation
unavailable pre-2021). Verdict is about whether THIS evidence yields a CALIBRATED
probability today. It is a demonstration of the framework, not the full live model.

USAGE:
    cd ~/sfc && .venv/bin/python analysis/bayesian_regime_calibration.py
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from scipy import stats as sps

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED = os.path.join(SFC_ROOT, ".walk_forward_imbs_l8.json")
OUTPUT = os.path.join(SFC_ROOT, ".bayesian_regime_calibration.json")

HORIZONS = [7, 30, 90]
MIN_TRAIN = 48
N_BINS = 8          # reliability bins (quantile-based)
FEATURES = ["liquidity_stress", "expectation_shock"]


def load_series():
    with open(CACHED) as f:
        s = json.load(f)
    # ensure forward returns for all horizons exist
    prices = [p["price"] for p in s]
    for i, p in enumerate(s):
        for h in HORIZONS:
            if p.get(f"fwd_return_{h}d") is None and i + h < len(prices):
                p[f"fwd_return_{h}d"] = (prices[i + h] - p["price"]) / p["price"] * 100
    return s


def walk_forward_oos(series, h):
    """Rolling-window logistic: predict P(y=1 | features) out-of-sample at a
    weekly stride (predictions are 7d apart; enough for calibration since
    forward-return labels overlap heavily). Fixed rolling window for speed.
    Returns (dates, y_true, p_pred, era_index)."""
    fk = f"fwd_return_{h}d"
    rows = []
    for i, p in enumerate(series):
        if all(p.get(f) is not None for f in FEATURES) and p.get(fk) is not None:
            rows.append((i, p["date"], [p[f] for f in FEATURES], 1.0 if p[fk] > 0 else 0.0))
    dates, X, y = [], [], []
    for _, d, x, yy in rows:
        dates.append(d); X.append(x); y.append(yy)
    X = np.array(X, float); y = np.array(y, float); n = len(X)

    ROLL = 1000      # fixed rolling window
    STRIDE = 7       # predict weekly
    dates_o, y_o, p_o, era_o, coefs = [], [], [], [], []
    for t in range(MIN_TRAIN, n, STRIDE):
        lo = max(0, t - ROLL)
        Xtr, ytr = X[lo:t], y[lo:t]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        prob = float(clf.predict_proba(X[t:t + 1])[0, 1])
        dates_o.append(dates[t]); y_o.append(y[t]); p_o.append(prob)
        era_o.append(t / n); coefs.append(clf.coef_[0])
    return (np.array(dates_o), np.array(y_o), np.array(p_o),
            np.array(era_o), np.array(coefs))


def calibration_report(y, p, dates, era_label=""):
    n = len(y)
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    n_bin = n // N_BINS
    bins = []
    for b in range(N_BINS):
        lo = b * n_bin
        hi = n if b == N_BINS - 1 else (b + 1) * n_bin
        seg = slice(lo, hi)
        mb = float(ps[seg].mean())
        freq = float(ys[seg].mean())
        nb = hi - lo
        bins.append({"range": f"{mb-1/(2*nb):.2f}-{mb+1/(2*nb):.2f}",
                     "n": nb, "mean_p": round(mb, 3), "actual_freq": round(freq, 3),
                     "abs_err": round(abs(freq - mb), 3)})
    # reliability metric: weighted mean abs error
    rel_err = sum(b["n"] * b["abs_err"] for b in bins) / n
    brier = float(np.mean((p - y) ** 2))
    brier_baseline = float(np.mean(y) * (1 - np.mean(y)))  # predict overall mean
    # Hosmer-Lemeshow (decile groups, standard formulation)
    hl = 0.0; df = N_BINS - 2
    try:
        for b in range(N_BINS):
            seg_lo = b * n_bin
            seg_hi = n if b == N_BINS - 1 else (b + 1) * n_bin
            seg = slice(seg_lo, seg_hi)
            obs_g = float(ys[seg].sum())
            exp_g = float(ps[seg].sum())
            g_n = seg_hi - seg_lo
            var_g = exp_g * (1 - exp_g / g_n)
            hl += ((obs_g - exp_g) ** 2) / max(1e-9, var_g)
        hl_p = float(1 - sps.chi2.cdf(hl, df))
    except Exception:
        hl, hl_p = None, None
    # doc's headline: top third (p in ~0.66-1.0) vs realized
    top = ps[-max(1, n // 3):]; top_y = ys[-max(1, n // 3):]
    top_diff = abs(top.mean() - top_y.mean())
    skill = 1 - brier / max(1e-9, brier_baseline)
    hl_ok = hl_p is not None and hl_p > 0.05
    # Honest verdict: skill (predictive value) + calibration (P behaves like P).
    if skill < -0.02:
        verdict = ("NO SKILL + ANTI-CALIBRATED: worse than base rate — "
                   "probability is not useful (and over-confident in one direction)")
    elif skill < 0.05:
        verdict = ("NO PREDICTIVE SKILL (Brier ~ baseline) — probability is "
                   "roughly 'calibrated' only in the weak sense of matching the "
                   "base rate, NOT useful for decisions")
    elif top_diff < 0.10 and hl_ok:
        verdict = "CALIBRATED + HAS SKILL: P behaves like P and beats base rate"
    else:
        verdict = (f"CALIBRATED-ISH BUT IMPERFECT: skill={skill:+.3f}, "
                   f"HL_p={hl_p}, top-third off by {top_diff:.2f} — treat with caution")
    return {
        "n": n, "era": era_label,
        "mean_pred_p": round(float(p.mean()), 3),
        "base_rate_y": round(float(y.mean()), 3),
        "bins": bins,
        "reliability_mean_abs_err": round(rel_err, 3),
        "brier": round(brier, 4),
        "brier_baseline": round(brier_baseline, 4),
        "brier_skill": round(skill, 3),
        "hosmer_lemershow_p": round(hl_p, 4) if hl_p is not None else None,
        "top_third_mean_p": round(float(top.mean()), 3),
        "top_third_actual_freq": round(float(top_y.mean()), 3),
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("BAYESIAN REGIME CALIBRATION (Bisa.docx mandatory test)")
    print("Evidence: liquidity_stress(GLF)+expectation_shock(L6) | logistic P(bullish)")
    print("=" * 70)
    series = load_series()
    print(f"Loaded {len(series)} point-in-time observations\n")

    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "evidence": FEATURES, "method": "expanding-window walk-forward logistic OOS"}
    for h in HORIZONS:
        print("-" * 70)
        print(f"HORIZON {h}d  —  outcome y = forward BTC return > 0")
        print("-" * 70)
        dates, y, p, era, coefs = walk_forward_oos(series, h)
        full = calibration_report(y, p, dates)
        print(f"  OOS n={full['n']}  base_rate(y=1)={full['base_rate_y']}  mean_P={full['mean_pred_p']}")
        print(f"  Brier={full['brier']} (baseline {full['brier_baseline']}) "
              f"skill={full['brier_skill']} | H-L p={full['hosmer_lemershow_p']}")
        print(f"  Verdict: {full['verdict']}  (top-third P={full['top_third_mean_p']} "
              f"vs actual {full['top_third_actual_freq']})")
        # era-split calibration
        eras = {}
        labels = ["era1", "era2", "era3(latest)"]
        for lab in labels:
            m = (era < 1/3) if lab == "era1" else ((era < 2/3) if lab == "era2" else (era >= 2/3))
            if m.sum() < 20:
                continue
            eras[lab] = calibration_report(y[m], p[m], np.array(dates)[m], era_label=lab)
            er = eras[lab]
            print(f"    {lab:12s} n={er['n']:<5} Brier={er['brier']:<7} "
                  f"skill={er['brier_skill']:<6} | top-third P={er['top_third_mean_p']} "
                  f"vs actual={er['top_third_actual_freq']} -> {er['verdict']}")
        # mean GLF coef sign per era (how evidence maps to regime)
        c_era = []
        for lab in labels:
            m = (era < 1/3) if lab == "era1" else ((era < 2/3) if lab == "era2" else (era >= 2/3))
            if m.sum() > 0:
                c_era.append({lab: [round(x, 4) for x in np.mean(coefs[m], axis=0)]})
        result[str(h)] = {"full": full, "era_split": eras, "mean_coef_by_era": c_era,
                          "base_rate": full["base_rate_y"]}

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")
    print("DONE.")


if __name__ == "__main__":
    main()
