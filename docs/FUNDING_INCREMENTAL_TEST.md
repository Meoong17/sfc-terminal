# Funding Incremental Test — Hasil (apakah funding menambah deteksi regime?)

Pertanyaan: BitMEX funding yang era-stable itu **redundan dengan harga/vol**, atau
menambah daya deteksi regime unik? (framing sama dengan regime_conditioning_test
yang menolak macro). Script: `analysis/funding_incremental_test.py`.

## [1] Lintas-exchange (BitMEX vs Binance funding, 2020+, n=2403)
- pearson **+0.687** (co-move moderat), spearman **+0.130** (peringkat harian beda nyata).
- BitMEX sd 0.00022 vs Binance sd 0.00020 (skala mirip).
- Verdict: BitMEX cuma SATU venue — cukup representatif untuk skala/ko-move, tapi
  tidak sempurna untuk peringkat harian. Caveat bila dipakai.

## [2] Nilai incremental atas baseline harga/vol (CV AUC)
| target | baseline (ret20+vol) | +funding | Δ |
|---|---|---|---|
| BULL/BEAR | 0.741 | 0.741 | **−0.000** |
| STRESS | 0.814 | 0.814 | **+0.000** |

Funding univariat sendiri: bull 0.588, stress 0.570 (moderat) — tapi SEPENUHNYA
terserap baseline harga/vol.

## [3] Increment per era (Δ selalu nol)
2016-19, 2019-22, 2022-26: Δ = 0.000 untuk BULL & STRESS di SEMUA era.

## Kesimpulan
1. **Funding = valid & era-stable sebagai pembacaan regime** (confirmation) — menguatkan
   analisis sebelumnya (corr bull +0.16/0.41/0.26, tercile bull% 41→82, negatif di krisis).
2. **TAPI funding REDUNDAN total dengan perilaku harga/vol** — menambah Δ=0.000 AUC
   terhadap baseline (ret20+vol) di semua era. Tidak ada info unik.
3. Implikasi SFC: funding layak sebagai **lapisan CONTEXT/CONFIRMATION display**
   (state-read yang valid & era-stable, seperti core stress-gauge), **BUKAN sebagai
   input baru ke classifier regime** (tidak menambah apa pun atas baseline perilaku).

## Relasi ke temuan besar
Ini mengonfirmasi tesis inti SFC: **sinyal hidup di perilaku harga/vol**. Funding —
seperti macro-liquidity — adalah KONFIRMASI state yang valid tapi redundan; bedanya
funding era-stable (macro era-flip). Jadi funding = context/confirmation, bukan driver.
