#!/usr/bin/env python3
"""
cross_asset_universality.py — Uji keempat universalitas: LINTAS-ASET.

Tiga sumbu lain sudah diuji dan jawabannya "tidak universal":
  lintas-waktu (era0/era1 tanda terbalik), lintas-definisi (12 estimator berbeda
  tanda di era yang sama), lintas-horizon (7d vs 30d berbeda). Sumbu keempat —
  apakah edge stress->return ini properti pasar kripto pada umumnya, atau
  kekhususan BTC — belum tersentuh. Inilah ujinya.

DEFINISI ESTIMATOR IDENTIK dengan uji BTC (impor build_signals yang sama), supaya
perbandingan antar-aset bukan perbandingan apel-jeruk. Harga semua aset dari
sumber yang sama (Binance Vision klines 1d).

Tafsir:
  - Kalau ETH/SOL juga negatif di era3a/era3b -> edge ini universal lintas-aset
    (properti pasar, bukan keanehan BTC) -> argumen lebih kuat untuk mempercayainya.
  - Kalau hanya BTC -> edge ini spesifik BTC; menaikkan bobot/klaim jadi lebih
    sulit dibenarkan, dan "universal" harus dijawab TIDAK di keempat sumbu.

Cakupan era berbeda per aset (ETH listing 2017-08, SOL 2020-08) -> era tanpa data
dilaporkan n/a, bukan diisi paksa.
"""
import json
import os
import sys

import pandas as pd

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SFC_DIR, "analysis"))
from vol_stress_estimator_test import (  # noqa: E402
    ERAS, HORIZONS, build_signals, era_gap, load_prices,
)

ASSETS = ["BTC", "ETH", "SOL"]
SIGNALS = ["rv30", "semidev30", "parkinson30", "maxdd90", "jump_ratio30", "tail30", "dvol30"]
OUT = os.path.join(SFC_DIR, ".cross_asset_universality.json")


def load_asset(name):
    """BTC = seri gabungan kanonik (Kaggle+Binance); lain = klines Binance Vision."""
    if name == "BTC":
        return load_prices()
    path = os.path.join(SFC_DIR, "data", f"binance_vision_{name}USDT_daily.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df[(df["close"] > 0) & df["high"].notna() & df["low"].notna() & df["open"].notna()]


def main():
    res = {"assets": {}, "signals": SIGNALS,
           "note": "gap top-20% vs bottom-20% expanding-z per era; CI moving-block "
                   "bootstrap blok=h; * block-sig, ~ hanya iid"}
    panels = {}
    for a in ASSETS:
        px = load_asset(a)
        if px is None or len(px) < 400:
            res["assets"][a] = {"unavailable": True}
            print(f"[CrossAsset] {a}: data tidak tersedia", file=sys.stderr)
            continue
        sig = build_signals(px)
        for h in HORIZONS:
            sig[f"fwd_{h}d"] = (sig["close"].shift(-h) / sig["close"] - 1.0) * 100.0
        panels[a] = sig
        res["assets"][a] = {"n_days": int(len(sig)),
                            "start": str(sig.index[0].date()), "end": str(sig.index[-1].date())}
        print(f"[CrossAsset] {a}: {len(sig)} hari {sig.index[0].date()}..{sig.index[-1].date()}",
              file=sys.stderr)

    for a, sig in panels.items():
        for s in SIGNALS:
            zc = s + "_z"
            if zc not in sig.columns:
                continue
            for h in HORIZONS:
                for ename, d0, d1 in ERAS:
                    seg = sig.loc[(sig.index >= d0) & (sig.index < d1)]
                    key = f"{s}|{h}d|{ename}"
                    res["assets"][a].setdefault("gaps", {})[key] = era_gap(seg, zc, f"fwd_{h}d", h=h)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[CrossAsset] -> {OUT}", file=sys.stderr)
    _print(res)


def _print(res):
    eras = [e[0] for e in ERAS]
    for a, ad in res["assets"].items():
        if ad.get("unavailable"):
            print(f"\n=== {a}: TIDAK TERSEDIA ==="); continue
        print(f"\n=== {a} ({ad['n_days']} hari, {ad['start']}..{ad['end']}) ===")
        print(f"{'sinyal':13s} {'h':>3s} " + " ".join(f"{e:>10s}" for e in eras))
        for s in SIGNALS:
            for h in HORIZONS:
                cells = []
                for e in eras:
                    g = ad.get("gaps", {}).get(f"{s}|{h}d|{e}", {})
                    if "gap" not in g:
                        cells.append(f"{'n/a':>10s}")
                    else:
                        m = "*" if g.get("significant") else ("~" if g.get("significant_iid") else "")
                        cells.append(f"{g['gap']:+8.2f}{m}")
                print(f"{s:13s} {h:>3d} " + " ".join(cells))


if __name__ == "__main__":
    main()
