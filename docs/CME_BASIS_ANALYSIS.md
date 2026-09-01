# CME Basis Analysis — Hasil (institutional carry signal, basis futures vs spot)

Uji nilai sinyal carry institusional: basis CME BTC1! (close futures / close spot − 1),
menggunakan `data/tradingview_btc1_daily.csv`. Script: `analysis/cme_basis_analysis.py`.

## Level & karakter
- n=2189 hari, 2017-12-18 → 2026-08-29.
- basis mean +0.0018, median +0.0017, contango 56% hari. Min −0.179 / max +0.207
  (fluktuasi besar — artefak roll/thin-liquidity awal CME).

## State-discrimination (tercile basis → perilaku)
- bull%: low=47.6% mid=52.0% high=67.4% — monoton lemah (LEBIH LEMAH dari funding 41→82).
- stress%: low=28.1% mid=22.3% high=18.1% — basis tinggi ↔ sedikit lebih tenang.

## Era-stability corr(basis, regime)
- corr(basis,bull): 2017-20 +0.105, 2020-23 +0.112, **2023-26 +0.022 (≈nol — MELURUH)**.
- corr(basis,stress): −0.05, −0.09, −0.01 (lemah).
- Verdict: TIDAK robust era-stable — memburuk ke ~0 di era terbaru (beda dari funding
  yang +0.16/+0.41/+0.26 stabil).

## Nilai incremental atas baseline harga/vol (CV AUC) — UJI KEPUTUSAN
| target | baseline | +basis | Δ |
|---|---|---|---|
| BULL/BEAR | 0.743 | 0.744 | **+0.001** |
| STRESS | 0.838 | 0.838 | **+0.000** |

Basis univariat sendiri 0.590/0.571 (moderat) — tapi terserap baseline harga/vol.

## Kesimpulan
1. **CME basis redundan total dengan perilaku harga/vol** (Δ≈0) — sama seperti funding.
2. Diskriminasi state **lebih lemah** dari funding, dan **era-stability meluruh ke ~0**
   di 2023-26 — tidak robust.
3. Implikasi SFC: **JANGAN tambahkan** CME basis — tidak ada info unik, tidak era-stable
   di era terbaru. Data CME BTC1! tetap berguna hanya sebagai referensi harga
   institusional/cross-check spot, bukan sinyal regime.

## Relasi ke temuan besar
Konsisten: sinyal hidup di harga/vol; carry/derivatif (funding, basis) = valid tapi
redundan. Basis bahkan lebih lemah dari funding (era-decay). SFC tidak butuh input baru
dari sini.
