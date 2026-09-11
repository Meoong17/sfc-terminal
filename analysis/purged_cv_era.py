#!/usr/bin/env python3
"""
purged_cv_era.py — Purged-CV/embargo TERBATAS ERA.

MENGAPA: purged-CV pooled (2012-2026) di vol_stress_estimator_test.py memberi
AUC tinggi justru ke sinyal yang skill-nya hidup di era0/era1 — era di mana
polaritas stress->return TERBALIK. Jadi pooled-AUC adalah metrik yang salah
untuk model yang harus bekerja di struktur pasar SEKARANG: ia memberi hadiah
atas skill rezim lama.

Skrip ini mengulang purged-CV + kontrol permutasi DI DALAM satu era saja,
sehingga jawabannya langsung relevan: "apakah ada yang memperkuat inti di
struktur sekarang?" — dan sekaligus memperlihatkan apakah AUC pooled memang
didorong era lama (kalau ya, AUC per-era3b harus turun/berbalik).

Era: era0/era1 (diagnostik), era2 (penguat), era3a/era3b (gate = struktur kini).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_DIR, "analysis"))
from vol_stress_estimator_test import (  # noqa: E402
    BASE_FEATURES, INCUMBENT, STRESS_CANDIDATES, SEED, HORIZONS, MIN_Z_HIST,
    ERAS, build_signals, load_prices, purged_folds, _fit_auc,
)

N_FOLDS_ERA = 5
N_PERM = 200
OUT = os.path.join(SFC_DIR, ".purged_cv_era.json")


def purged_auc_arr(X, y, h, folds):
    aucs = []
    for tr, te in folds:
        a, _ = _fit_auc(X[tr], y[tr], X[te], y[te])
        if a is not None:
            aucs.append(a)
    if not aucs:
        return None
    aucs = np.array(aucs)
    return {"pooled": float(aucs.mean()), "folds": [round(a, 3) for a in aucs],
            "se": float(aucs.std(ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else float("nan")}


def main():
    t0 = time.time()
    quick = "--quick" in sys.argv
    n_perm = 20 if quick else N_PERM

    df = load_prices()
    sig = build_signals(df)
    for h in HORIZONS:
        sig[f"fwd_{h}d"] = (sig["close"].shift(-h) / sig["close"] - 1.0) * 100.0

    # sfc_pct (tolok ukur inti) — hanya tersedia 2014-12+ dari cache WFV
    wfv_path = os.path.join(SFC_DIR, ".walk_forward_validation.json")
    if os.path.exists(wfv_path):
        bs = pd.DataFrame(json.load(open(wfv_path)))
        bs["date"] = pd.to_datetime(bs["date"])
        bs = bs.set_index("date")[["sfc_pct"]]
        sig = sig.join(bs, how="left")
        sig["sfc_pct_z"] = ((sig["sfc_pct"] - sig["sfc_pct"].expanding(MIN_Z_HIST).mean())
                            / sig["sfc_pct"].expanding(MIN_Z_HIST).std())

    out = {}
    for ename, d0, d1 in ERAS:
        seg = sig.loc[(sig.index >= d0) & (sig.index < d1)]
        out[ename] = {"n_days": int(len(seg))}
        for h in HORIZONS:
            fwd = f"fwd_{h}d"
            sub = seg.dropna(subset=[f + "_z" for f in BASE_FEATURES] + [fwd]).copy()
            if len(sub) < 200:
                out[ename][f"{h}d"] = {"insufficient": True, "n": int(len(sub))}
                continue
            sub["y"] = (sub[fwd] < 0).astype(int)
            Xb = sub[[f + "_z" for f in BASE_FEATURES]].to_numpy(dtype=float)
            y = sub["y"].to_numpy()
            folds = list(purged_folds(len(sub), N_FOLDS_ERA, h))
            base = purged_auc_arr(Xb, y, h, folds)
            rec = {"n": int(len(sub)), "base_rate_neg": round(float(y.mean()), 4),
                   "baseline": base, "signals": {}}
            sig_list = [INCUMBENT] + STRESS_CANDIDATES
            if "sfc_pct_z" in sub.columns and sub["sfc_pct_z"].notna().sum() > 200:
                sig_list = sig_list + ["sfc_pct"]
            for s in sig_list:
                col = s + "_z"
                if col not in sub.columns or sub[col].notna().sum() < 200:
                    continue
                m = sub[col].notna().to_numpy()
                Xb_m, y_m = Xb[m], y[m]
                Xc = sub.loc[m, col].to_numpy(dtype=float)
                folds_m = list(purged_folds(len(y_m), N_FOLDS_ERA, h))
                uni = purged_auc_arr(Xc.reshape(-1, 1), y_m, h, folds_m)
                aug = purged_auc_arr(np.column_stack([Xb_m, Xc]), y_m, h, folds_m)
                delta = None
                if uni and aug and base:
                    delta = round(aug["pooled"] - base["pooled"], 4)
                p_perm = None
                if delta is not None and n_perm > 0:
                    rng = np.random.default_rng(SEED)
                    null = []
                    for _ in range(n_perm):
                        Xs = np.column_stack([Xb_m, rng.permutation(Xc)])
                        r = purged_auc_arr(Xs, y_m, h, folds_m)
                        if r:
                            null.append(r["pooled"])
                    if null:
                        thr = base["pooled"] + delta
                        p_perm = round(float((np.array(null) >= thr).mean()), 4)
                rec["signals"][s] = {
                    "uni_auc": round(uni["pooled"], 4) if uni else None,
                    "uni_ci_lo": round(uni["pooled"] - 1.96 * uni["se"], 4) if uni else None,
                    "uni_folds": uni["folds"] if uni else None,
                    "aug_delta": delta, "perm_p": p_perm}
            out[ename][f"{h}d"] = rec

    res = {"generated_at": pd.Timestamp.utcnow().isoformat(), "seed": SEED,
           "n_perm": n_perm, "n_folds": N_FOLDS_ERA,
           "note": "purged-CV terbatas era; baseline = [rv30_z, mom30_z]; "
                   "perm_p = fraksi null (kolom kandidat diacak) >= delta teramati",
           "eras": out}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[EraPurgedCV] {time.time()-t0:.1f}s -> {OUT}", file=sys.stderr)
    _print(res)


def _print(res):
    for ename, ed in res["eras"].items():
        print(f"\n=== {ename} (n_days={ed['n_days']}) ===")
        for h in [f"{x}d" for x in HORIZONS]:
            r = ed.get(h)
            if not r or r.get("insufficient"):
                print(f"  {h}: data tidak cukup"); continue
            print(f"  {h}: n={r['n']} base_rate={r['base_rate_neg']} "
                  f"baseline AUC={r['baseline']['pooled']:.4f} folds={r['baseline']['folds']}")
            for s, v in r["signals"].items():
                print(f"    {s:13s} uni={v['uni_auc']} ci_lo={v['uni_ci_lo']} "
                      f"dAUC={v['aug_delta']} perm_p={v['perm_p']}")


if __name__ == "__main__":
    main()
