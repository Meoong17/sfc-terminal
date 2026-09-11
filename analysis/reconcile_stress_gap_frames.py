#!/usr/bin/env python3
"""
reconcile_stress_gap_frames.py — Menjelaskan kenapa `sfc_pct` TIDAK signifikan di
era3b pada kerangka uji vol_stress_estimator_test.py, padahal laporan live
(.walk_forward_summary.json) menyatakan gap era3 signifikan.

Dua perbedaan kerangka yang mungkin jadi penyebab, diuji silang 2x2:
  (i)  ATURAN BUCKET : ambang live (CALM <25, STRESS >=45) vs kuantil-z ekstrem
       (top-20% vs bottom-20% dari expanding-z sfc_pct).
  (ii) SUMBER HARGA untuk return forward: return yang tersimpan di cache WFV
       (berbasis FRED CBBTCUSD, dihitung walk_forward_validation.py) vs seri
       gabungan kanonik Binance + Bitstamp (dipakai uji ini).

Kalau selisih verdict hilang saat (i) disamakan -> penyebabnya aturan bucket.
Kalau hilang saat (ii) disamakan -> penyebabnya sumber harga.

Batas era: era3 = 2022-01-01.. dan era3b = 2024-01-01.. (structural break ETF).
CI: moving-block bootstrap, blok = h (label forward tumpang tindih).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_DIR, "analysis"))
from vol_stress_estimator_test import boot_diff_ci, load_prices, MIN_Z_HIST  # noqa: E402

WFV = os.path.join(SFC_DIR, ".walk_forward_validation.json")
BUCKETS = [(0, 25, "CALM"), (25, 45, "ELEV"), (45, 101, "STRESS")]
SEED = 42


def summarize(stress, calm, h):
    if len(stress) < 10 or len(calm) < 10:
        return {"insufficient": True, "n_stress": len(stress), "n_calm": len(calm)}
    ci = boot_diff_ci(stress.values, calm.values, block=h)
    ci_iid = boot_diff_ci(stress.values, calm.values, block=1)
    out = {"n_stress": int(len(stress)), "n_calm": int(len(calm)),
           "mean_stress": round(float(stress.mean()), 2), "mean_calm": round(float(calm.mean()), 2)}
    if ci is not None:
        out.update(gap=round(ci[0], 2), ci_lo=round(ci[1], 2), ci_hi=round(ci[2], 2),
                   sig_block=bool(ci[2] < 0 or ci[1] > 0))
    if ci_iid is not None:
        out["sig_iid"] = bool(ci_iid[2] < 0 or ci_iid[1] > 0)
    return out


def main():
    wfv = pd.DataFrame(json.load(open(WFV)))
    wfv["date"] = pd.to_datetime(wfv["date"])
    wfv = wfv.set_index("date")

    px = load_prices()
    panel = px[["close"]].copy()
    for h in (7, 30):
        panel[f"px_fwd_{h}d"] = (panel["close"].shift(-h) / panel["close"] - 1.0) * 100.0
    panel = panel.join(wfv[["sfc_pct", "fwd_return_7d", "fwd_return_30d"]], how="inner")
    panel["sfc_pct_z"] = ((panel["sfc_pct"] - panel["sfc_pct"].expanding(MIN_Z_HIST).mean())
                          / panel["sfc_pct"].expanding(MIN_Z_HIST).std())
    print(f"[Reconcile] panel bersama: {len(panel)} hari "
          f"{panel.index[0].date()} .. {panel.index[-1].date()}", file=sys.stderr)

    WINDOWS = {"era3 (2022-)": ("2022-01-01", "2099-01-01"),
               "era3b (2024-)": ("2024-01-01", "2099-01-01")}
    out = {}
    for wname, (d0, d1) in WINDOWS.items():
        seg = panel.loc[(panel.index >= d0) & (panel.index < d1)]
        for h in (7, 30):
            fwd_wfv = f"fwd_return_{h}d"
            fwd_px = f"px_fwd_{h}d"

            # (i) aturan bucket live  x  (ii) sumber harga
            for src, col in (("WFV/FRED", fwd_wfv), ("Binance+Bitstamp", fwd_px)):
                sub = seg[[col, "sfc_pct"]].dropna()
                stress = sub.loc[sub["sfc_pct"] >= 45, col]
                calm = sub.loc[sub["sfc_pct"] < 25, col]
                out[f"{wname} | {h}d | bucket-live | {src}"] = summarize(stress, calm, h)

            # (i) kuantil-z  x  (ii) sumber harga
            for src, col in (("WFV/FRED", fwd_wfv), ("Binance+Bitstamp", fwd_px)):
                sub = seg[[col, "sfc_pct_z"]].dropna()
                if len(sub) < 60:
                    out[f"{wname} | {h}d | quantile-z | {src}"] = {"insufficient": True}
                    continue
                lo, hi = sub["sfc_pct_z"].quantile(0.20), sub["sfc_pct_z"].quantile(0.80)
                stress = sub.loc[sub["sfc_pct_z"] >= hi, col]
                calm = sub.loc[sub["sfc_pct_z"] <= lo, col]
                out[f"{wname} | {h}d | quantile-z | {src}"] = summarize(stress, calm, h)

    res = {"generated_at": pd.Timestamp.utcnow().isoformat(), "seed": SEED, "cells": out}
    with open(os.path.join(SFC_DIR, ".reconcile_stress_gap.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\nAturan bucket / sumber harga -> gap STRESS-CALM (negatif = benar); "
          "sig: * block-bootstrap, ~ hanya iid")
    print(f"{'window':14s} {'h':>3s} {'aturan':12s} {'harga':17s} {'gap':>8s} {'CI95':>18s} "
          f"{'n_str':>6s} {'n_calm':>6s}  sig")
    for k, v in out.items():
        if v.get("insufficient"):
            print(f"  {k:58s} n/a")
            continue
        mark = "*" if v.get("sig_block") else ("~" if v.get("sig_iid") else "")
        ci = f"[{v['ci_lo']}, {v['ci_hi']}]"
        wname, h, rule, src = [x.strip() for x in k.split("|")]
        print(f"{wname:14s} {h:>3s} {rule:12s} {src:17s} {v['gap']:>+8.2f} {ci:>18s} "
              f"{v['n_stress']:>6d} {v['n_calm']:>6d}  {mark}")


if __name__ == "__main__":
    main()
