#!/usr/bin/env python3
"""Walk-forward validation of the confidence calibration map.

Why: a single train/test split (analysis/calibration_isotonic.py) fitted a Platt
map that passed the OOS gate on 56 test snapshots but was FLAT — it mapped every
raw confidence into 0.395-0.443, i.e. it predicted the base rate and carried no
ordinal information. This script asks the question the single split cannot:

    does a map fitted on the past beat BOTH the raw confidence and the
    climatology base rate, repeatedly, as time advances?

Protocol
  - labeled snapshots ordered by time (price-outcome labels from
    confidence_calibration.price_label);
  - expanding-window walk-forward: train on everything before the fold, test on
    the next block;
  - per fold compare raw / calibrated (Platt on train) / climatology (mean of
    train labels) on ECE and Brier;
  - verdict requires beating raw AND climatology on Brier in a majority of folds
    (a map that only beats raw by predicting the base rate is not useful).

Output: analysis/calibration_walkforward_results.json
Exit 0 always; the verdict is published, not enforced.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from calibration_isotonic import (  # noqa: E402
    load_pairs, beta_fit, beta_apply, ece, brier,
)

OUT = ROOT / "analysis" / "calibration_walkforward_results.json"
MIN_TRAIN = 50
TEST_BLOCK = 15


def metrics(y, p):
    return {"ece": float(ece(y, p)), "brier": float(brier(y, p))}


def main():
    pairs = load_pairs(max_count=800)
    if not pairs:
        print("[wf] no labeled snapshots", file=sys.stderr)
        return 0
    # load_pairs -> (confidence, label, date)
    pairs = sorted(pairs, key=lambda r: str(r[2]))
    x = np.array([float(p[0]) for p in pairs], dtype=float)
    y = np.array([int(bool(p[1])) for p in pairs], dtype=int)
    dates = [str(p[2]) for p in pairs]
    n = len(pairs)

    folds = []
    start = MIN_TRAIN
    while start + TEST_BLOCK <= n:
        tr = slice(0, start)
        te = slice(start, start + TEST_BLOCK)
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        try:
            params = beta_fit(xtr, ytr, platt_only=True)
            p_cal = np.clip(beta_apply(xte, params), 1e-6, 1 - 1e-6)
        except Exception as exc:  # degenerate fold
            folds.append({"test_from": dates[start], "error": str(exc)})
            start += TEST_BLOCK
            continue
        base = float(np.mean(ytr))
        p_raw = np.clip(xte, 1e-6, 1 - 1e-6)
        p_clim = np.full_like(p_raw, base)
        m = {"raw": metrics(yte, p_raw), "cal": metrics(yte, p_cal),
             "clim": metrics(yte, p_clim)}
        spread = float(np.nanmax(p_cal) - np.nanmin(p_cal))
        folds.append({
            "test_from": dates[start], "test_to": dates[start + TEST_BLOCK - 1],
            "n_train": int(start), "n_test": int(TEST_BLOCK),
            "base_rate_train": round(base, 4), "cal_spread": round(spread, 4),
            "raw": m["raw"], "cal": m["cal"], "clim": m["clim"],
            "cal_beats_raw_brier": bool(m["cal"]["brier"] < m["raw"]["brier"]),
            "cal_beats_clim_brier": bool(m["cal"]["brier"] < m["clim"]["brier"]),
        })
        start += TEST_BLOCK

    ok = [f for f in folds if "error" not in f]
    if not ok:
        print("[wf] no usable folds", file=sys.stderr)
        return 0

    def mean(key, sub):
        return float(np.mean([f[key][sub] for f in ok]))

    agg = {
        "n_labeled": n, "n_folds": len(ok),
        "first_test": ok[0]["test_from"], "last_test": ok[-1]["test_to"],
        "mean_raw_ece": round(mean("raw", "ece"), 4),
        "mean_cal_ece": round(mean("cal", "ece"), 4),
        "mean_clim_ece": round(mean("clim", "ece"), 4),
        "mean_raw_brier": round(mean("raw", "brier"), 4),
        "mean_cal_brier": round(mean("cal", "brier"), 4),
        "mean_clim_brier": round(mean("clim", "brier"), 4),
        "folds_cal_beats_raw": sum(f["cal_beats_raw_brier"] for f in ok),
        "folds_cal_beats_clim": sum(f["cal_beats_clim_brier"] for f in ok),
        "mean_cal_spread": round(float(np.mean([f["cal_spread"] for f in ok])), 4),
    }
    frac_raw = agg["folds_cal_beats_raw"] / agg["n_folds"]
    frac_clim = agg["folds_cal_beats_clim"] / agg["n_folds"]

    if agg["mean_cal_brier"] < agg["mean_raw_brier"] and frac_clim >= 0.6 and agg["mean_cal_spread"] >= 0.05:
        verdict = "ACCEPT — calibration beats raw and climatology walk-forward"
        accepted = True
    elif frac_clim < 0.4 and agg["mean_cal_brier"] < agg["mean_raw_brier"]:
        verdict = ("REJECT AS CLIMATOLOGY — map improves on raw only because it predicts "
                   "the base rate; climatology is just as good")
        accepted = False
    else:
        verdict = "REJECT — no reliable edge over raw confidence walk-forward"
        accepted = False

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"min_train": MIN_TRAIN, "test_block": TEST_BLOCK,
                     "folds_expanding": True,
                     "label": "price_outcome (confidence_calibration.price_label)"},
        "aggregate": agg,
        "frac_folds_cal_beats_raw": round(frac_raw, 3),
        "frac_folds_cal_beats_clim": round(frac_clim, 3),
        "accepted": accepted, "verdict": verdict, "folds": folds,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[wf] folds={agg['n_folds']} labeled={n} "
          f"Brier raw={agg['mean_raw_brier']} cal={agg['mean_cal_brier']} "
          f"clim={agg['mean_clim_brier']}", file=sys.stderr)
    print(f"[wf] beats raw {agg['folds_cal_beats_raw']}/{agg['n_folds']}, "
          f"beats clim {agg['folds_cal_beats_clim']}/{agg['n_folds']}, "
          f"spread={agg['mean_cal_spread']}", file=sys.stderr)
    print(json.dumps({"aggregate": agg, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
