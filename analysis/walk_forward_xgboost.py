#!/usr/bin/env python3
"""
walk_forward_xgboost.py — Walk-forward validation of the XGBoost meta-ensemble.

Determines EMPIRICALLY whether the XGBoost meta-ensemble's "6h forward price-drop
probability" is calibrated against realized 6h BTC price outcomes, using
chronological expanding-window folds (mirroring analysis/walk_forward_validation.py).

This is the evidence required before EVER re-enabling the XGBoost blend into
effective_sfc (disabled 2026-08-07 — see collect.py "XGBoost Blend: DISABLED").

Verdict rule (empirical, no verbal claims):
  Re-enable blend only if ALL hold on out-of-sample (OOS) predictions:
    (a) AUC meaningfully > 0.5 (model discriminates drop risk), AND
    (b) Brier < naive baseline (always predict base rate), AND
    (c) calibration monotonic / predictions track realized drop rates, AND
    (d) results stable across expanding folds.
  Otherwise keep the blend disabled.

Run:  cd ~/sfc && .venv/bin/python analysis/walk_forward_xgboost.py
Output: prints report + writes .walk_forward_xgboost.json
"""
import sys, os, json, logging
from datetime import datetime, timezone
import numpy as np

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SFC_DIR)

from models import ensemble_meta as em  # noqa: E402  (SFC_DIR on path above)

logging.basicConfig(level=logging.INFO, format="[XGB-WF] %(message)s", stream=sys.stderr)
log = logging.getLogger("xgb_wf")

OUT_PATH = os.path.join(SFC_DIR, ".walk_forward_xgboost.json")
# Expanding-window train/future cut fractions (chronological).
FOLDS = [0.60, 0.75, 0.85, 0.92]


def _auc(y_true, y_score):
    """Area under ROC via rank (Mann-Whitney U)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _brier(y_true, y_score):
    """Brier score vs graded target (proper scoring rule)."""
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))


def _calibration_bins(y_bin, y_score, n_bins=5):
    """Mean predicted vs observed event rate in score quantiles."""
    y_score = np.asarray(y_score)
    y_bin = np.asarray(y_bin, dtype=bool)
    idx = np.argsort(y_score)
    edges = np.linspace(0, len(y_score), n_bins + 1).astype(int)
    rows = []
    for b in range(n_bins):
        if edges[b] == edges[b + 1]:
            continue
        sel = idx[edges[b]:edges[b + 1]]
        rows.append({
            "pred_mid": round(float(y_score[sel].mean()), 3),
            "obs_rate": round(float(y_bin[sel].mean()), 3),
        })
    return rows


def run():
    log.info("Extracting historical snapshots from git...")
    snapshots = em.extract_historical_snapshots()
    log.info("Snapshots: %d", len(snapshots))
    if len(snapshots) < 200:
        log.error("Too few snapshots (%d) to walk-forward validate.", len(snapshots))
        return None

    results = {}
    all_pred, all_y, all_ybin = [], [], []
    n = len(snapshots)
    for f in FOLDS:
        cut = int(n * f)
        train_snaps = snapshots[:cut]
        test_snaps = snapshots[cut:]
        Xtr, ytr = em.build_method_scores_array(train_snaps)
        Xte, yte = em.build_method_scores_array(test_snaps)
        if len(Xtr) < 100 or len(Xte) < 50:
            log.warning("Fold f=%.2f too small (tr=%d te=%d), skipping", f, len(Xtr), len(Xte))
            continue
        # early-stopping split within training
        val_n = max(1, int(len(Xtr) * 0.15))
        m = em.XGBoostMetaEnsemble()
        m.fit(Xtr[:-val_n], ytr[:-val_n],
              eval_set=[(Xtr[-val_n:], ytr[-val_n:])], verbose=False)
        pred = m.predict(Xte) / 100.0             # predict() returns 0-100; /100 -> 0-1
        ybin = (yte > 0.5).astype(float)          # material-drop event
        auc = _auc(ybin, pred)
        brier = _brier(yte, pred)
        base = float(ybin.mean())
        naive_brier = float(np.mean((yte - base) ** 2))  # always-predict-base
        calib = _calibration_bins(ybin, pred)
        results[f"{f:.2f}"] = {
            "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
            "base_rate": round(base, 4), "auc": round(auc, 4),
            "brier": round(brier, 4), "naive_brier": round(naive_brier, 4),
            "brier_delta": round(naive_brier - brier, 4),  # >0 = better than naive
            "calibration": calib,
        }
        all_pred.extend(pred.tolist())
        all_y.extend(yte.tolist())
        all_ybin.extend(ybin.tolist())
        log.info("fold %.2f: auc=%.3f brier=%.4f (naive %.4f) n_test=%d",
                 f, auc, brier, naive_brier, len(Xte))

    if not results:
        log.error("No folds validated — cannot form verdict.")
        return None

    all_pred = np.array(all_pred)
    all_y = np.array(all_y)
    all_ybin = np.array(all_ybin)
    pooled_auc = _auc(all_ybin, all_pred)
    pooled_brier = _brier(all_y, all_pred)
    base = float(all_ybin.mean())
    naive = float(np.mean((all_y - base) ** 2))
    pooled_calib = _calibration_bins(all_ybin, all_pred)

    # Verdict
    aucs = [r["auc"] for r in results.values() if not np.isnan(r["auc"])]
    auc_ok = pooled_auc > 0.55 and all(a > 0.5 for a in aucs)
    brier_ok = pooled_brier < naive and all(r["brier_delta"] > 0 for r in results.values())
    calib_ok = _calibration_ok(pooled_calib)

    verdict = "RE-ENABLE_OK" if (auc_ok and brier_ok and calib_ok) else "STAY_DISABLED"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_folds": len(results),
        "pooled": {
            "n": int(len(all_pred)), "base_rate": round(base, 4),
            "auc": round(pooled_auc, 4), "brier": round(pooled_brier, 4),
            "naive_brier": round(naive, 4),
            "calibration": pooled_calib,
        },
        "folds": results,
        "checks": {"auc_ok": bool(auc_ok), "brier_ok": bool(brier_ok),
                   "calib_ok": bool(calib_ok)},
        "verdict": verdict,
        "recommendation": (
            "XGBoost OOS predictions are calibrated & discriminative across folds — "
            "blend MAY be re-enabled after scale calibration." if verdict == "RE-ENABLE_OK"
            else "XGBoost OOS predictions are NOT consistently calibrated/discriminative — "
                 "KEEP the blend disabled (display-only)."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _calibration_ok(rows):
    """Calibration is 'ok' if predicted rate roughly tracks observed rate
    (non-decreasing-ish and not wildly over/under confident)."""
    if not rows:
        return False
    preds = [r["pred_mid"] for r in rows]
    obs = [r["obs_rate"] for r in rows]
    # Predicted should generally track observed direction
    monotone = all(b >= a for a, b in zip(obs, obs[1:])) or all(a >= b for a, b in zip(obs, obs[1:]))
    # Mean absolute error between pred_mid and obs_rate across bins
    mae = float(np.mean([abs(p - o) for p, o in zip(preds, obs)]))
    # accept if roughly monotone AND not absurdly miscalibrated (>0.15 MAE)
    return bool(monotone and mae <= 0.15)


if __name__ == "__main__":
    s = run()
    if s:
        print(json.dumps(s, indent=2))
