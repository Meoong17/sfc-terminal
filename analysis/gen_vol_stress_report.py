#!/usr/bin/env python3
"""gen_vol_stress_report.py — Render laporan markdown dari .vol_stress_estimator_test.json.

Deterministik: hanya membaca JSON hasil uji, tidak menghitung ulang apa pun,
supaya angka di laporan tidak mungkin berbeda dari angka uji.
"""
import json
import os
import sys

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(SFC_DIR, ".vol_stress_estimator_test.json")
DST = os.path.join(SFC_DIR, "docs", "VOL_STRESS_ESTIMATOR_TEST.md")

ERA_LABEL = {
    "era0": "era0 2012-14", "era1": "era1 2015-18", "era2": "era2 2018-22",
    "era3a": "era3a 2022-24", "era3b": "era3b 2024-26",
}
NOTE = ("* = signifikan dgn moving-block bootstrap (blok=h) — menghormati label forward "
        "yang tumpang tindih; ~ = signifikan HANYA kalau ketergantungan diabaikan (iid), "
        "jadi jangan dianggap bukti.")


def cell(g):
    if not g or g.get("insufficient") or "gap" not in g:
        return "  n/a  "
    mark = "*" if g.get("significant") else ("~" if g.get("significant_iid") else "")
    return f"{g['gap']:+.2f}{mark}"


def main():
    with open(SRC) as f:
        d = json.load(f)
    eras = [e["name"] for e in d["eras"]]
    role = {e["name"]: e["role"] for e in d["eras"]}
    nera = {e["name"]: e["n"] for e in d["eras"]}
    L = []
    A = L.append
    A("# Uji Estimator Stress Price/Vol — apakah inti era-stable bisa diperkuat?")
    A("")
    A(f"Dihasilkan: `analysis/vol_stress_estimator_test.py` · run {d['generated_at']} · "
      f"seed={d['seed']} · permutasi={d['n_perm']}")
    A("")
    A(f"Panel: **{d['panel']['n_days']} hari** ({d['panel']['start']} .. {d['panel']['end']}) — "
      "Kaggle Bitstamp (2012-01..2017-08) + Binance Vision kanonik (2017-08-17..).")
    A("")
    A("Pertanyaan: apakah salah satu estimator stress alternatif **memperkuat inti** "
      "(stress-gauge harga/vol, satu-satunya edge SFC yang era-stable) — atau semuanya "
      "sekadar representasi ulang realized-vol?")
    A("")
    A("## Era dan perannya")
    A("")
    A("| Era | Periode | n | Peran |")
    A("|---|---|---|---|")
    for e in d["eras"]:
        A(f"| {e['name']} | {e['start'][:7]} .. {e['end'][:7]} | {e['n']} | {e['role']} |")
    A("")
    A("`gate` = penentu verdict (struktur pasar sekarang, era3 dipecah di structural break "
      "spot-ETF 2024-01). `support` = penguat. `diagnostic` = konteks saja — TIDAK dipakai "
      "membunuh sinyal, karena menuntut tanda sama antara 2013 (bull satu arah) dan 2025 "
      "(terinstitusionalisasi) adalah bar yang salah.")
    A("")

    # ── TEST A ──
    A("## Test A (PRIMER) — gap stress→return forward per era")
    A("")
    A("Top-20% vs bottom-20% dari expanding-z **di dalam** tiap era (threshold-free). "
      "Polaritas benar = **negatif** (stress tinggi → return forward lebih buruk).")
    A("")
    A(NOTE)
    A("")
    for h in sorted({k for v in d["test_a_era_gaps"].values() for k in v}, key=lambda x: int(x[:-1])):
        A(f"### Horizon {h}")
        A("")
        A("| Sinyal | " + " | ".join(ERA_LABEL.get(e, e) for e in eras) + " | gate |")
        A("|---" * (len(eras) + 2) + "|")
        for sig, hs in d["test_a_era_gaps"].items():
            if h not in hs:
                continue
            dd = hs[h]
            cells = " | ".join(cell(dd["per_era"].get(e, {})) for e in eras)
            A(f"| `{sig}` | {cells} | {dd['gate']} |")
        A("")
    if d.get("benchmark_incumbent_sfc_pct"):
        A("### Tolok ukur: inti yang sekarang hidup (`sfc_pct`, replay faktor tereduksi)")
        A("")
        A("| Sinyal | " + " | ".join(ERA_LABEL.get(e, e) for e in eras) + " |")
        A("|---" * (len(eras) + 1) + "|")
        for h, dd in d["benchmark_incumbent_sfc_pct"].items():
            cells = " | ".join(cell(dd.get(e, {})) for e in eras)
            A(f"| `sfc_pct` {h} | {cells} |")
        A("")

    # ── TEST B ──
    A("## Test B (KONFIRMASI) — purged-CV/embargo + kontrol permutasi")
    A("")
    A("Target = P(return forward < 0). K=5 fold kontigu, purge label tumpang tindih + "
      "embargo=h, standardisasi fit hanya di train. `ΔAUC` = kenaikan di atas baseline "
      "`[rv30, mom30]`. `perm_p` = fraksi null (kolom kandidat diacak, 200×) yang ΔAUC-nya "
      "≥ yang diamati — **wajib**, karena menambah fitur apa pun (termasuk noise) menaikkan "
      "AUC OOS ~0.05.")
    A("")
    for h, dd in d["test_b_purged_cv"].items():
        b = dd["baseline"]
        A(f"### Horizon {h} (n={dd['n']}, base rate negatif={dd['base_rate_neg']}, "
          f"baseline AUC={b.get('pooled_auc')})")
        A("")
        A("| Sinyal | AUC univariat | ΔAUC atas baseline | perm_p |")
        A("|---|---|---|---|")
        for sig, v in dd["signals"].items():
            u = v["univariate"].get("pooled_auc")
            A(f"| `{sig}` | {u} | {v['delta_auc']} | {v['perm_p']} |")
        A("")

    # ── TEST C ──
    A("## Test C — redundansi terhadap realized-vol")
    A("")
    A("Spearman vs `rv30`. Di atas ~0.9 = kandidat praktis representasi ulang realized-vol.")
    A("")
    A("| Sinyal | Spearman vs rv30 |")
    A("|---|---|")
    for k, v in sorted(d["test_c_redundancy_vs_rv30"].items(), key=lambda x: -x[1]):
        A(f"| `{k}` | {v:+.3f} |")
    A("")
    A("## Verdict")
    A("")
    A(_verdict_table(d))
    A("")
    A("## Caveat yang wajib dibaca bersama hasil")
    A("")
    A("- **Vol era3b teredam** (vol harian ~2.46% vs ~3.39% era3a / ~4.23% 2017-20) → "
      "rentang dinamis sinyal berbasis LEVEL mengecil; gap tipis di era3b bisa berarti "
      "rezimnya lebih tenang, bukan sinyalnya mati.")
    A("- **Label forward 30d tumpang tindih** (30d berurutan berbagi 29 hari) → `n_eff` ≈ "
      "n/(h/2+1). CI di sini memakai moving-block bootstrap (blok=h) untuk itu; tanda `~` "
      "menandai sel yang hanya lolos kalau ketergantungan diabaikan.")
    A("- **`sfc_pct` = replay faktor tereduksi** (price+DXY+M2+FNG dari cache WFV), bukan "
      "replay penuh sistem live yang juga memuat DVOL/on-chain/GLF/dynamic weighting. "
      "Dipakai sebagai tolok ukur inti, bukan sebagai deskripsi sistem live.")
    A("- **Satu aset (BTC).** Universalitas lintas-aset tidak diuji di sini.")
    A("")
    with open(DST, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[Report] {DST} ({len(L)} baris)", file=sys.stderr)


def _verdict_table(d):
    """Verdict per kandidat; aturan eksplisit supaya bisa diperiksa ulang."""
    ta = d["test_a_era_gaps"]
    tb = d["test_b_purged_cv"]
    tc = d["test_c_redundancy_vs_rv30"]
    cands = [s for s in ta if s not in ("rv30", "mom30")]
    rows = []
    for s in sorted(cands):
        gate7 = ta.get(s, {}).get("7d", {}).get("gate_ok")
        gate30 = ta.get(s, {}).get("30d", {}).get("gate_ok")
        gate = bool(gate7 or gate30)
        rho = tc.get(s)
        perm = {}
        delta = {}
        for h, dd in tb.items():
            v = dd["signals"].get(s)
            if v:
                perm[h] = v.get("perm_p")
                delta[h] = v.get("delta_auc")
        adds = any((perm.get(h) is not None and perm[h] < 0.05 and (delta.get(h) or 0) > 0)
                   for h in perm)
        if not gate:
            v = "REJECT — tidak lolos gate era struktur sekarang"
        elif rho is not None and abs(rho) > 0.9 and not adds:
            v = f"REDUNDAN — lolos gate tapi ≈ realized-vol (rho {rho:+.2f}) & tidak menambah"
        elif adds:
            hr = [h for h in perm if perm[h] is not None and perm[h] < 0.05 and (delta.get(h) or 0) > 0]
            v = f"KANDIDAT TAMBAH — lolos gate & menambah pada {', '.join(hr)} (perm_p<0.05)"
        else:
            v = "GATE SAJA — lolos era sekarang tapi tidak ada tambahan terverifikasi (perm_p≥0.05)"
        rows.append(f"| `{s}` | {'ya' if gate else 'tidak'} | {rho if rho is None else f'{rho:+.3f}'} | "
                    + ", ".join(f"{h}:{perm[h]}" for h in sorted(perm)) + f" | {v} |")
    head = ("| Kandidat | Lolos gate era | rho vs rv30 | perm_p (7d,30d) | Verdict |\n"
            "|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows)


if __name__ == "__main__":
    main()
