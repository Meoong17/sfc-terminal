#!/usr/bin/env python3
"""
walk_forward_xgboost_purged.py — Tahap 3: purged/embargo CV + bootstrap CI utk XGBoost.

Upgrade dari walk_forward_xgboost.py (single-path expanding) dgn:
  - EMBARGO: buang K sampel terakhir train sebelum cut, agar label forward-6h
    tidak bocor antar train/test (Lopez de Prado 2018).
  - BOOTSTRAP CI pada AUC pooled (2.5/97.5 persentil).
  - CALIBRATION GATE (D2): Brier decomposition utk keputusan formal.

Pertanyaan: apakah verdict STAY_DISABLED (dari single-path) bertahan setelah
embargo + CI? Jika CI AUC pooled menyentuh/lewat 0.5 dan gate kalibrasi gagal,
verdict tetap STAY_DISABLED dengan bukti lebih kuat.

Jalankan (background, ~5 menit):  cd ~/sfc && .venv/bin/python analysis/walk_forward_xgboost_purged.py
"""
import json, os, sys, logging
import numpy as np
sys.path.insert(0, "/home/ubuntu/sfc")
from models import ensemble_meta as em
from analysis.sfc_methods_academic import brier_decompose, calibration_gate

logging.basicConfig(level=logging.INFO, format="[XGB-P] %(message)s", stream=sys.stderr)

FOLDS = [0.60, 0.75, 0.85, 0.92]
EMBARGO = 6          # buang 6 sampel train terakhir sebelum cut (antisipasi overlap 6h)


def _auc(ybin, p):
    p = np.asarray(p); y = np.asarray(ybin)
    order = np.argsort(p)
    y = y[order]; p = p[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    # trapezoid
    return float(np.trapezoid(tpr, fpr))


def _brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bootstrap_auc(y, p, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs[i] = _auc(y[idx], p[idx])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    print("[XGB-P] extract snapshots (git history)...", file=sys.stderr)
    snaps = em.extract_historical_snapshots()
    n = len(snaps)
    print(f"[XGB-P] {n} snapshots", file=sys.stderr)

    folds = []
    pooled_y, pooled_p = [], []
    for f in FOLDS:
        cut = int(n * f)
        train = snaps[: max(cut - EMBARGO, 1)]
        test = snaps[cut:]
        Xtr, ytr = em.build_method_scores_array(train)
        Xte, yte = em.build_method_scores_array(test)
        if len(Xtr) < 100 or len(Xte) < 100:
            continue
        m = em.XGBoostMetaEnsemble()
        val_n = max(1, int(len(Xtr) * 0.15))
        m.fit(Xtr[:-val_n], ytr[:-val_n],
              eval_set=[(Xtr[-val_n:], ytr[-val_n:])], verbose=False)
        p = m.predict(Xte) / 100.0
        ybin = (yte > 0.5).astype(float)
        folds.append({"fold": f, "n_test": len(yte), "auc": _auc(ybin, p),
                      "brier": _brier(yte, p),
                      "base_rate": float(ybin.mean())})
        pooled_y.extend(ybin.tolist()); pooled_p.extend(p.tolist())
        print(f"[XGB-P] fold {f}: auc={_auc(ybin,p):.3f} brier={_brier(yte,p):.4f} "
              f"n_test={len(yte)}", file=sys.stderr)

    py = np.array(pooled_y); pp = np.array(pooled_p)
    auc_lo, auc_hi = bootstrap_auc(py, pp)
    gate = calibration_gate(py, pp, min_resolution=0.0, max_reliability=0.05)
    auc_ok = auc_lo > 0.5
    verdict = "RE-ENABLE_OK" if (auc_ok and gate["passed"]) else "STAY_DISABLED"

    res = {
        "embargo": EMBARGO,
        "folds": folds,
        "pooled": {"n": len(py), "auc": _auc(py, pp),
                   "auc_ci95": [auc_lo, auc_hi],
                   "brier": float(np.mean((pp - py) ** 2)),
                   "base_rate": float(py.mean()),
                   "calibration_gate": gate},
        "checks": {"auc_ok": bool(auc_ok), "calib_gate_ok": bool(gate["passed"])},
        "verdict": verdict,
    }
    with open("/home/ubuntu/sfc/.walk_forward_xgboost_purged.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[XGB-P] pooled auc={res['pooled']['auc']:.3f} CI95=[{auc_lo:.3f},{auc_hi:.3f}]")
    print(f"[XGB-P] gate: resolution={gate['resolution']:.4f} reliability={gate['reliability']:.4f} "
          f"passed={gate['passed']}")
    print(f"[XGB-P] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
