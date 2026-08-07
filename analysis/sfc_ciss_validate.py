#!/usr/bin/env python3
"""
sfc_ciss_validate.py — Tahap 2: Uji empiris CISS-composite (A1) vs composite sekarang.

Gagasan CISS (Hollo, Kremer, Lo Duca 2012): ketika sub-faktor stres co-move
(mis. macro lemah DAN exec_risk tinggi), stres sistemik lebih berbahaya daripada
perkalian linear. Sketsa A1 menambah penalti "penyebaran stres" + korelasi.

Pertanyaan empiris yang diuji pada data NYATA (panel harian dari git data.json):
  Q1. Seberapa besar CISS mengubah composite vs formula sekarang?
  Q2. Apakah composite (versi sekarang ATAU CISS) berkorelasi negatif dgn
      return BTC forward 30d? (komposit tinggi -> return masa depan lebih rendah)
  Q3. Gap test: mean return forward 30d pada kuartil composite tertinggi vs
      terendah — apakah beda signifikan (bootstrap CI)?

KETERBATASAN DAYA (dilaporkan jujur):
  - Panel harian hanya ~30-60 titik (snapshot per hari, data.json mulai 2026-06-09),
    dan hanya ~30 yang punya forward-30d penuh (BTC dari snapshot git s/d 08-08).
  - Komponen nyaris konstan di rezim tenang (macro ~0.41, exec ~0.23) -> variasi kecil.
  - Batas kode (exec_risk 0.393 sebelum fix vs 0.232 sesudah) -> panel tak homogen.
  Kesimpulan dibuat dengan mempertimbangkan keterbatasan ini.

Jalankan:  cd ~/sfc && .venv/bin/python analysis/sfc_ciss_validate.py
"""
import json, subprocess, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/sfc")
from analysis.sfc_methods_academic import ciss_composite_confidence


def daily_snapshots():
    """Latest data.json per day dari git history. Return list of (date, dict)."""
    out = subprocess.run(
        ["git", "log", "--format=%h %ad", "--date=short", "--", "data.json"],
        capture_output=True, text=True, cwd="/home/ubuntu/sfc",
    ).stdout.splitlines()
    by_day = {}
    for line in out:
        h, day = line.split()
        by_day[day] = h  # log urut newest -> keep newest hash per day
    snaps = []
    for day in sorted(by_day):
        h = by_day[day]
        raw = subprocess.run(
            ["git", "show", f"{h}:data.json"], capture_output=True, text=True,
            cwd="/home/ubuntu/sfc").stdout
        try:
            d = json.loads(raw)
        except Exception:
            continue
        cc = d.get("confidence_components", {})
        snaps.append({
            "day": day, "btc": d.get("btc"),
            "composite": d.get("composite_confidence"),
            "exec_risk": cc.get("execution_risk"),
            "macro": cc.get("macro_confidence"),
            "agree": cc.get("method_agree"),
        })
    return snaps


def forward_30d(day, price_by_day, horizon_days=30):
    """Return BTC forward `horizon_days` hari dari `day` (dari deret harga harian)."""
    # price_by_day: dict day->btc (latest per day). cari harga ~30 hari setelah.
    days = sorted(price_by_day)
    if day not in price_by_day:
        return None
    p0 = price_by_day[day]
    # index hari sekarang
    idx = days.index(day)
    target_day = days[min(idx + horizon_days, len(days) - 1)]
    if target_day == day:
        return None
    p1 = price_by_day[target_day]
    return (p1 - p0) / p0


def bootstrap_ci(diff_vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(diff_vals)
    means = [(rng.choice(a, len(a), replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    snaps = daily_snapshots()
    # deret harga harian (untuk forward return)
    price = {s["day"]: s["btc"] for s in snaps if s["btc"]}
    # panel: snapshot yang punya semua komponen + forward 30d
    rows = []
    for s in snaps:
        if None in (s["composite"], s["exec_risk"], s["macro"], s["agree"]):
            continue
        fwd = forward_30d(s["day"], price, 30)
        if fwd is None:
            continue
        ciss = ciss_composite_confidence(
            s["macro"], s["exec_risk"], s["agree"], recent_corr=0.5)
        rows.append({**s, "fwd": fwd, "ciss": ciss, "diff": ciss - s["composite"]})

    print(f"Panel harian: {len(rows)} snapshot dengan forward-30d lengkap "
          f"({rows[0]['day']} s/d {rows[-1]['day']})\n")
    if len(rows) < 15:
        print("TERLALU SEDIKIT titik utk gap test bermakna. Laporkan deskriptif saja.\n")

    comps = np.array([r["composite"] for r in rows])
    ciss = np.array([r["ciss"] for r in rows])
    fwd = np.array([r["fwd"] for r in rows])
    diffs = np.array([r["diff"] for r in rows])

    # Q1: seberapa besar CISS mengubah composite
    print("Q1 — perubahan composite oleh CISS:")
    print(f"  mean diff = {diffs.mean():+.4f}  (std {diffs.std():.4f}, "
          f"range [{diffs.min():+.4f}, {diffs.max():+.4f}])")
    print(f"  korelasi (Spearman) CISS vs composite sekarang = "
          f"{np.corrcoef(ciss, comps)[0,1]:.3f}\n")

    # Q2: korelasi komposit -> forward return
    from scipy.stats import spearmanr
    r_now, p_now = spearmanr(comps, fwd)
    r_ciss, p_ciss = spearmanr(ciss, fwd)
    print("Q2 — korelasi komposit vs return forward 30d (negatif = semakin stress semakin turun):")
    print(f"  composite sekarang: rho={r_now:+.3f}  p={p_now:.3f}")
    print(f"  composite CISS    : rho={r_ciss:+.3f}  p={p_ciss:.3f}\n")

    # Q3: gap test kuartil atas vs bawah
    print("Q3 — gap test (mean forward 30d, kuartil komposit atas minus bawah):")
    for name, arr in [("sekarang", comps), ("CISS", ciss)]:
        q1, q3 = np.quantile(arr, 0.25), np.quantile(arr, 0.75)
        lo = fwd[arr <= q1]; hi = fwd[arr >= q3]
        if len(lo) >= 3 and len(hi) >= 3:
            gap = hi.mean() - lo.mean()
            ci = bootstrap_ci(np.concatenate([hi - lo]))
            # interpretasi: komposit TINGGI seharusnya return RENDAH -> gap NEGATIF
            print(f"  {name}: n_low={len(lo)} n_high={len(hi)} | "
                  f"fwd_low={lo.mean():+.3%} fwd_high={hi.mean():+.3%} | "
                  f"gap(high-low)={gap:+.3%}  CI95=[{ci[0]:+.3%},{ci[1]:+.3%}]  "
                  f"{'SIG' if (ci[0]<0<ci[1] is False and gap<0) else ''}")
        else:
            print(f"  {name}: titik terlalu sedikit untuk gap test")
    print("\nInterpretasi: gap negatif + CI tak lintas 0 = komposit tinggi memprediksi "
          "return lebih rendah (valid). Hati-hati dgn n kecil & variasi komponen rendah.")


if __name__ == "__main__":
    main()
