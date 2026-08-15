#!/usr/bin/env python3
"""
walk_forward_alphractal.py — predictive validation of Alphractal BTC metrics.

For each Alphractal metric (data/alphractal_daily.json), test whether it has
genuine OUT-OF-SAMPLE forward predictive power for BTC price return. This is the
exact discipline the SFC rulebook requires BEFORE any metric may enter scoring:
walk-forward validation, no look-ahead, honest uncertainty.

Two-stage gate (matches skill walk-forward-validation Pitfalls 25/26):

STAGE 1 — rank-IC screen (many features at once)
  IC = Spearman(metric_t, fwd_return_{t..t+h}) per metric × horizon (7/30/90d).
  Bootstrap p + 90% CI, then Benjamini-Hochberg FDR across ALL metric×horizon
  cells (one p<0.05 in a few cells does NOT survive). MANDATORY era-split
  (3 equal blocks) requiring BOTH eras same sign. Verdict VALID only if
  q<0.10 AND era-consistent.

STAGE 2 — purged-CV / embargo gate (Lopez de Prado) for screen-survivors
  Because consecutive forward-return labels OVERLAP (~h−1 shared days), an IC
  screen over-estimates skill. Any screen-survivor is confirmed with K-fold
  purged-CV: purge training samples whose label window overlaps the test block,
  add an embargo gap, fit a 1-feature logistic regression, evaluate OOS AUC.
  Verdict bar: pooled AUC > 0.5 and mean-fold − 1.96·SE > 0.5.

Also reports BOTH raw-level and point-in-time expanding-z transforms (they can
reverse polarity — Pitfall 19), and flags series too short to era-split across a
bull AND a bear (Pitfall 22: DATA-TOO-SHORT, not validated).

SFC status: RESEARCH ONLY. No metric is blended into any score here. A VERDICT
is the input to a human decision to add/ignore, not an automatic wiring.

Output: .walk_forward_alphractal.json  (gitignored runtime cache)
Usage:  python3 analysis/walk_forward_alphractal.py
        python3 analysis/walk_forward_alphractal.py --json
"""

import os, sys, json, math, random, time
import numpy as np

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SFC_DIR, "data", "alphractal_daily.json")
OUTPUT = os.path.join(SFC_DIR, ".walk_forward_alphractal.json")

HORIZONS = [7, 30, 90]
N_ERA = 3                 # equal-size era blocks for era-split sign-consistency
QUANTILE_TAIL = 0.20      # top/bottom 20% tail gap
N_BOOT = 2000
N_FOLDS = 5
ALPHA = 0.10

# outcome metric (predicting BTC price return) — excluded from predictors
OUTCOME = "PriceUSD"

PREDICTORS = [
    "SplyCur", "supply_in_profit_pct", "TxCnt", "AdrActCnt",
    "HashRate", "DiffLast", "Funding_Rate", "Open_Interest",
    "Long_short_ratio", "Taker_long_short", "Liquidations",
]


def load_panel():
    """Return dict date -> {metric: value}, sorted dates."""
    with open(DATA) as f:
        raw = json.load(f)
    dates = sorted(raw.keys())
    return raw, dates


def expanding_z(series):
    """Point-in-time expanding-window z-score (uses only data up to each t)."""
    out = np.full(len(series), np.nan, dtype=float)
    vals = series.astype(float)
    mean = 0.0
    m2 = 0.0
    n = 0
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            continue
        n += 1
        # Welford
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)
        if n >= 30 and m2 > 0:
            std = math.sqrt(m2 / n)
            if std > 0:
                out[i] = (v - mean) / std
    return out


def spearman_ic(x, y):
    """Spearman rank correlation; returns (rho, n_valid)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.nan, int(mask.sum())
    xv, yv = x[mask], y[mask]
    rx = np.argsort(np.argsort(xv)).astype(float)
    ry = np.argsort(np.argsort(yv)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return np.nan, int(mask.sum())
    return float((rx * ry).sum() / denom), int(mask.sum())


def bootstrap_ic_p(x, y, rho_obs, n_boot=N_BOOT, seed=42):
    """Two-sided bootstrap p + 90% CI via NUMPY-VECTORIZED resampling.

    Vectorized (skill Pitfall 19): build an (n_boot, n) index matrix and
    compute all bootstrap Spearman rho in bulk — ~100x faster than the pure-
    Python per-draw loop.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    xv = x[mask].copy(); yv = y[mask].copy()
    n = len(xv)
    if n < 30:
        return np.nan
    rng = np.random.default_rng(seed)

    def _rho(xv, yv):
        rx = np.argsort(np.argsort(xv)).astype(float)
        ry = np.argsort(np.argsort(yv)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return (rx * ry).sum() / d if d else 0.0

    rx = np.argsort(np.argsort(xv)).astype(float); rx -= rx.mean()
    denom = np.sqrt((rx * rx).sum())

    # null: rho with y permuted across all bootstraps at once
    # Random permutation per row via argsort of iid uniforms (no Python loop)
    perm = np.argsort(np.argsort(rng.random((n_boot, n)), axis=1), axis=1)
    ry = np.argsort(np.argsort(np.take_along_axis(yv[None, :], perm, axis=1),
                                axis=1), axis=1).astype(float)
    ry -= ry.mean(axis=1, keepdims=True)
    d = np.sqrt((ry * ry).sum(axis=1))
    nulls = (rx[None, :] * ry).sum(axis=1) / np.where(d > 0, d * denom, 1.0)
    p = float((np.abs(nulls) >= abs(rho_obs)).mean())

    # 90% CI on rho via bootstrap resample (sample with replacement)
    bi = rng.integers(0, n, size=(n_boot, n))
    xb, yb = xv[bi], yv[bi]
    rxb = np.argsort(np.argsort(xb, axis=1), axis=1).astype(float)
    ryb = np.argsort(np.argsort(yb, axis=1), axis=1).astype(float)
    rxb -= rxb.mean(axis=1, keepdims=True)
    ryb -= ryb.mean(axis=1, keepdims=True)
    d1 = np.sqrt((rxb * rxb).sum(axis=1)); d2 = np.sqrt((ryb * ryb).sum(axis=1))
    dd = np.where((d1 > 0) & (d2 > 0), d1 * d2, np.nan)
    rhos = (rxb * ryb).sum(axis=1) / dd
    lo, hi = np.percentile(rhos, [5, 95])
    return float(p), float(lo), float(hi)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values for an array of p-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    q = np.full(n, np.nan)
    order = np.argsort(p)
    ranks = np.empty(n)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    for i in range(n):
        q[i] = p[i] * n / max(ranks[i], 1)
    # enforce monotonicity
    q = np.minimum.accumulate(q[order])[np.argsort(order)]
    return np.clip(q, 0, 1)


def quantile_tail_gap(x, y, tail=QUANTILE_TAIL, seed=42):
    """bottom-20% vs top-20% of metric vs forward return; bootstrap diff CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 100:
        return None
    order = np.argsort(xv)
    tail_n = max(1, int(n * tail))
    bottom = yv[order[:tail_n]]   # low metric
    top = yv[order[-tail_n:]]     # high metric
    gap = float(bottom.mean() - top.mean())
    rng = np.random.default_rng(seed)
    nb = len(bottom); nt = len(top)
    bi = rng.integers(0, nb, size=(N_BOOT, nb))
    ti = rng.integers(0, nt, size=(N_BOOT, nt))
    diffs = bottom[bi].mean(axis=1) - top[ti].mean(axis=1)
    lo, hi = np.percentile(diffs, [5, 95])
    return {"gap": gap, "lo": float(lo), "hi": float(hi),
            "n_bottom": int(nb), "n_top": int(nt)}


def purged_cv_auc(metric_series, dates, fwd, horizon, n_folds=N_FOLDS, seed=42):
    """
    Purged-CV / embargo (Lopez de Prado): K contiguous folds over TIME.
    Test block [i0,i1). Purge every train sample j with j < i0 and j+h >= i0
    (its label window overlaps the test), add embargo j in [i1, i1+h].
    Fit 1-feature logistic regression, evaluate OOS AUC on test block.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mask = ~(np.isnan(metric_series) | np.isnan(fwd))
    idx = np.where(mask)[0]
    x = metric_series[idx]
    y = (fwd[idx] > 0).astype(int)
    n = len(x)
    if n < n_folds * 100:
        return None
    h = horizon
    rng = np.random.default_rng(seed)
    fold_auc = []
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    for f in range(n_folds):
        i0, i1 = bounds[f], bounds[f + 1]
        # purge + embargo
        keep = np.ones(n, dtype=bool)
        for j in range(i0):
            if j + h >= i0:      # train label overlaps test start
                keep[j] = False
        keep[i0:i1] = False       # test block
        keep[i1:min(i1 + h, n)] = False  # embargo after test
        tr_x, tr_y = x[keep], y[keep]
        te_x, te_y = x[i0:i1], y[i0:i1]
        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(tr_x.reshape(-1, 1), tr_y)
        if len(np.unique(te_y)) == 2:
            auc = roc_auc_score(te_y, clf.predict_proba(te_x.reshape(-1, 1))[:, 1])
            fold_auc.append(auc)
    if not fold_auc:
        return None
    pooled_auc = float(np.mean(fold_auc))
    se = float(np.std(fold_auc) / math.sqrt(len(fold_auc)))
    return {
        "pooled_auc": round(pooled_auc, 4),
        "se": round(se, 4),
        "ci_lower": round(pooled_auc - 1.96 * se, 4),
        "n_folds": len(fold_auc),
        "n_test": int(n),
        "verdict": "PASS" if (pooled_auc > 0.5 and pooled_auc - 1.96 * se > 0.5) else "FAIL",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)

    raw, dates = load_panel()
    n_days = len(dates)

    # Build arrays
    arr = {m: np.array([raw.get(d, {}).get(m) for d in dates], dtype=float) for m in PREDICTORS + [OUTCOME]}
    price = arr[OUTCOME]

    # forward returns for each horizon
    fwd = {}
    for h in HORIZONS:
        fwd[h] = np.full(n_days, np.nan)
        for i in range(n_days - h):
            p0, p1 = price[i], price[i + h]
            if not (np.isnan(p0) or np.isnan(p1)) and p0 > 0:
                fwd[h][i] = (p1 - p0) / p0 * 100.0  # pct

    # ── STAGE 1: rank-IC screen ─────────────────────────────────────────
    cells = []  # (metric, horizon, transform)
    results = {}
    for m in PREDICTORS:
        results[m] = {}
        for h in HORIZONS:
            for transform, series in (("raw", arr[m]), ("z", expanding_z(arr[m]))):
                rho, nv = spearman_ic(series, fwd[h])
                if nv < 30 or math.isnan(rho):
                    cells.append((m, h, transform, None, nv, None, None, None))
                    continue
                p, lo, hi = bootstrap_ic_p(series, fwd[h], rho, seed=42 + hash((m, h, transform)) % 1000)
                cells.append((m, h, transform, rho, nv, p, lo, hi))

    # BH-FDR across all valid cells
    valid = [c for c in cells if c[3] is not None]
    qvals = bh_fdr([c[5] for c in valid]) if valid else []
    qmap = {id(c): q for c, q in zip(valid, qvals)}

    # era-split sign-consistency (per metric, raw transform, each horizon)
    era_len = n_days // N_ERA
    for m in PREDICTORS:
        for h in HORIZONS:
            era_rho = []
            for e in range(N_ERA):
                s = e * era_len
                e_ = s + era_len if e < N_ERA - 1 else n_days
                r, _ = spearman_ic(arr[m][s:e_], fwd[h][s:e_])
                era_rho.append(r)
            results[m].setdefault("era_ic", {})[h] = era_rho

    screen = {}
    for c in valid:
        m, h, transform, rho, nv, p, lo, hi = c
        q = qmap[id(c)]
        er = results[m]["era_ic"][h]
        signs = [np.sign(r) for r in er if not math.isnan(r)]
        era_consistent = len(set(signs)) == 1
        # data-too-short: not enough overlap across bull AND bear — flag when
        # any era block is empty (NaN) OR series spans < ~2 bull/bear cycles
        any_nan_era = any(math.isnan(r) for r in er)
        data_too_short = any_nan_era or nv < N_ERA * 200
        key = (m, h, transform)
        screen[key] = {
            "metric": m, "horizon": h, "transform": transform,
            "ic": round(rho, 4), "n": nv, "p": round(p, 4), "q": round(q, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "era_ic": [None if math.isnan(r) else round(r, 4) for r in er],
            "era_consistent": era_consistent,
            "data_too_short": data_too_short,
            "screen_pass": (q < ALPHA and era_consistent and not data_too_short),
        }

    # ── STAGE 2: purged-CV gate for screen survivors (raw transform) ────
    purged = {}
    for key, s in screen.items():
        if not s["screen_pass"]:
            continue
        m, h, transform = key
        if transform != "raw":
            continue
        res = purged_cv_auc(arr[m], dates, fwd[h], h, seed=42 + h)
        if res:
            purged[(m, h)] = res
            s["purged_cv"] = res
        else:
            s["purged_cv"] = None

    # final verdict per metric (any horizon passing BOTH gates)
    final = {}
    for m in PREDICTORS:
        best = None
        for (mm, h) in purged:
            if mm == m and purged[(mm, h)]["verdict"] == "PASS":
                if best is None or h < best:
                    best = h
        qtail = {}
        for h in HORIZONS:
            t = quantile_tail_gap(arr[m], fwd[h])
            if t: qtail[h] = t
        final[m] = {
            "verdict": "VALIDATED" if best else (
                "DATA_TOO_SHORT" if all(screen[(m, h, "raw")]["data_too_short"] for h in HORIZONS) else "REJECTED"),
            "best_horizon_purged_pass": best,
            "quantile_tail_gap_30d": qtail.get(30),
            "screen_cells": [s for (mm, h, tr), s in screen.items() if mm == m],
            "purged_cv": {f"{h}d": purged[(m, h)] for (mm, h) in purged if mm == m},
        }

    out = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "data_file": "data/alphractal_daily.json",
        "n_days": n_days,
        "horizons": HORIZONS,
        "verdict_bar": "screen: q<0.10 AND era-consistent AND not data-too-short; "
                       "then purged-CV pooled AUC>0.5 and CI_lower>0.5",
        "final_verdicts": final,
    }
    os.makedirs(SFC_DIR, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=1)

    # ── console report ──────────────────────────────────────────────────
    print(f"Alphractal walk-forward validation  ({n_days} days, {DATA})")
    print(f"stage1 = rank-IC + BH-FDR + era-split; stage2 = purged-CV/embargo\n")
    print(f"{'metric':<22}{'verdict':<16}{'best_hz':<8}{'q_30d_gap':<22}{'pucv_30d'}")
    for m, v in final.items():
        gap = v["quantile_tail_gap_30d"]
        gstr = f"{gap['gap']:+.3f} ({gap['lo']:+.2f},{gap['hi']:+.2f})" if gap else "n/a"
        p30 = v["purged_cv"].get("30d")
        pstr = f"{p30['pooled_auc']} ciLo={p30['ci_lower']}" if p30 else "not-survived"
        print(f"{m:<22}{v['verdict']:<16}{str(v['best_horizon_purged_pass']):<8}{gstr:<22}{pstr}")
    print("\nscreen survivors with purged-CV detail:")
    for (mm, h), r in purged.items():
        print(f"  {mm} {h}d: AUC={r['pooled_auc']} CI_lower={r['ci_lower']} folds={r['n_folds']} -> {r['verdict']}")

    if args.json:
        print(json.dumps(out))


if __name__ == "__main__":
    main()
