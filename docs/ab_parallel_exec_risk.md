# A/B/C/D Parallel Comparison — execution_risk Formula Variants (2026-08-08)

## Tujuan
Buktikan secara OUT-OF-SAMPLE kandidat formula `execution_risk` mana yang memberi
informasi prediktif lebih baik, sebelum menyentuh kode production. Tidak ada
perubahan collect.py — murni rekonstruksi dari snapshot historis.

Skrip: `analysis/ab_parallel_exec_risk.py` (git data.json, 11.702 snapshot,
2026-06-09 → 2026-08-08, daily-resample → 59 titik).

## Varian yang diuji
| Varian | cascade | squeeze | funding | exec_risk |
|---|---|---|---|---|
| A (OLD, double-count) | imbalance×0.5 + total/5e9 | imbalance×density | funding | 0.40A + 0.30B + 0.30C |
| B (de-dup orthogonal, LIVE) | imbalance (direction) | density (magnitude) | funding | 0.40A + 0.30B + 0.30C |
| C (no squeeze) | imbalance | — (dropped) | funding | 0.40A + 0.30C |
| D (funding-heavy) | imbalance | density | funding | 0.30A + 0.20B + 0.50C |

imbalance = |L−S|/T dari liq_long/short/total. funding = `funding_imbalance` real
(Deribit m13). Semua di-cap 0.95.

## Ketersediaan data (temuan penting)
- Likuidasi mentah (L/S/T): ~2026-06-25 → sekarang (~44 hari harian).
- `funding_imbalance` REAL nonzero: HANYA ~17 hari (2026-07-22 → 08-08).
  Field `m13_funding` = None di SELURUH riwayat — funding hanya tersedia via
  `confidence_components.funding_imbalance`, dan cuma ~17 hari.
- → Empat-varian penuh (C/D bergantung bobot funding) hanya bisa diuji pada
  **18 titik harian**. 30d forward tak mungkin (jendela < 30 hari).

## Hasil
### TIER 1 — empat-varian pada jendela funding real (18 titik)
| var | std | fwd7d gap [CI90] | IC7 | fwd14d gap [CI90] | IC14 |
|---|---|---|---|---|---|
| A | 0.261 | +0.021 [-0.018,+0.039] | +0.23 | +0.010 [-0.047,+0.032] | 0.00 |
| B | 0.268 | +0.021 [-0.019,+0.039] | +0.22 | +0.010 [-0.047,+0.032] | 0.00 |
| C | 0.238 | +0.013 [-0.025,+0.036] | +0.09 | +0.010 [-0.047,+0.032] | 0.00 |
| D | 0.294 | +0.002 [-0.021,+0.034] | +0.16 | +0.010 [-0.047,+0.032] | 0.00 |

Semua CI mencakup nol → **tidak ada varian yang terbukti lebih baik di OOS.**

### TIER 2 — A vs B robustness di jendela penuh (59 titik)
| var | std | 7d gap [CI90] | IC7 | 14d gap [CI90] | 30d gap [CI90] | IC30 |
|---|---|---|---|---|---|---|
| A | 0.207 | −0.024 [-0.05,+0.01] | −0.21 | +0.016 [-0.02,+0.03] | −0.014 [-0.04,+0.01] | −0.17 |
| B | 0.212 | −0.024 [-0.05,+0.01] | −0.21 | +0.016 [-0.02,+0.04] | −0.014 [-0.04,+0.02] | −0.13 |

std ratio A/B = **0.977** (de-dup hampir tidak mengubah amplitudo — B hanya ~2% lebih kecil).
Pearson corr antar varian: A–B = 0.991; A–C = 0.909; B–C = 0.929; semua ≥ 0.91.

## Verdict: EMPIRIS TAK TERBEDAKAN (data-too-short) — BERTAHAN pada B
1. **Keempat varian nyaris kolinear** (corr 0.91–0.99). Pada ukuran sampel ini,
   tidak ada formula yang menghasilkan informasi OOS yang beda secara statistik —
   semua CI mencakup nol, IC tak signifikan.
2. **Polaritas benar (negatif) muncul di jendela penuh** untuk A dan B di 7d & 30d
   (−0.024, −0.014), tapi CI tetap mencakup nol → ini bukti arah, bukan bukti signifikan.
3. **C dan D tidak bisa dievaluasi**: hanya ~17 hari funding real → bobot funding
   yang diubah oleh C/D tidak punya data untuk diverifikasi.
4. **Tidak ada alasan empiris untuk mengganti B (fix de-duplikasi live)**. B setara
   A secara prediktif, amplitudo stabil (std ratio 0.977), dan secara STRUKTURAL benar
   (tidak double-count satu sinyal likuidasi — lihat redundancy-dedup-validation skill).
5. Rekomendasi tetap: biarkan B (sudah live). Jangan naikkan bobot exec_risk atas
   dasar 18–59 hari. Re-run bulanan saat funding real mengumpul >6 bulan.

## Cara run
```
.venv/bin/python analysis/fetch_binance_vision.py   # download klines+funding -> data/binance_vision_daily.json
.venv/bin/python analysis/ab_parallel_exec_risk.py
```

## Ekstensi Binance Vision (2026-08-08) — funding & price historis
Untuk menutup gap funding 17 hari, funding diambil dari Binance Vision:
`data/futures/um/monthly/fundingRate/BTCUSDT/` (sejak 2020-01, 8-jam, format
`calc_time, interval_hours, last_funding_rate`). Kline daily BTCUSDT untuk
forward return. Sumber: `analysis/fetch_binance_vision.py`, cache
`data/binance_vision_daily.json` (2026-06-01 → 07-31; Agustus monthly belum
terbit, daily 404 → 08-01..08-08 pakai fallback funding SFC).
- Konversi ke skala sama: `funding_imbalance = min(|fr|*10, 1)`.
- CATATAN: Binance = futures USDT-perp, beda sumber dari SFC (Deribit m13) →
  proksi sinyal funding yang sah, bukan sumber identik.

### Hasil ulang — jendela likuidasi 57 titik harian (2026-06-11 → 08-08, funding real 56/57)
| var | std | fwd7d gap [CI90] | fwd14d gap [CI90] | fwd30d gap [CI90] |
|---|---|---|---|---|
| A (OLD) | 0.165 | −0.009 [-0.032,+0.017] | −0.008 [-0.039,+0.020] | −0.007 [-0.041,+0.022] |
| B (LIVE de-dup) | 0.173 | −0.007 [-0.030,+0.018] | +0.002 [-0.035,+0.023] | +0.008 [-0.036,+0.026] |
| **C (no squeeze)** | 0.157 | −0.028 [-0.046,+0.004] | **−0.049 [-0.073,−0.005]*** | −0.025 [-0.056,+0.007] |
| D (funding-heavy) | 0.170 | −0.007 [-0.031,+0.016] | +0.002 [-0.037,+0.021] | −0.007 [-0.042,+0.023] |

std ratio A/B = 0.956. Pearson corr: A–B 0.986, B–C 0.890, A–C 0.856 (C paling berbeda).

### Pembacaan jujur (HYPOTHESIS-GENERATING, BUKAN verdict deploy)
1. **Varian C (buang squeeze/liq-density) menunjukkan polaritas prediktif terkuat &
   satu-satunya signifikan**: 14d gap −0.049 [CI −0.073,−0.005]***, IC −0.238.
   Petunjuk bahwa komponen MAGNITUDE likuidasi (`squeeze = liq_density`) menambah
   NOISE, bukan informasi — buangnya memperbaiki info OOS. Ini arah yang BERLAWANAN
   dengan formula live B (yang justru memasukkan squeeze=density).
2. **TAPI tidak konklusif**: (a) cuma 1 dari 12 uji (4 varian × 3 horizon) yang
   signifikan — dengan α=0.10, ~1.2 false positive diharapkan; bisa jadi kebetulan;
   (b) jendela masih pendek (57 titik ~7 minggu); (c) 30d C tak signifikan.
3. **Karena itu JANGAN ubah production ke C sekarang.** C layak jadi hipotesis
   yang diuji ulang setelah data likuidasi terkumpul lebih lama (≥6 bulan), atau
   diverifikasi purged-CV vs incumbent.
4. A vs B tetap tak terbukti beda secara prediktif (gap & IC nyaris sama, std ratio
   0.956) → fix de-duplikasi B tetap berdiri atas dasar kebenaran struktural.

## Robustness empiris (2026-08-08) — `analysis/ab_empirical_robustness.py`
Uji objektif sebelum keputusan produksi. Window 57 titik.

### T1/T2 — p-value bootstrap + koreksi multiple-comparison (12 uji)
| var | hor | gap | p | q (BH) |
|---|---|---|---|---|
| C | 7d | −0.028 | 0.064 | 0.383 |
| C | **14d** | **−0.049** | **0.014** | **0.169** |
| C | 30d | −0.025 | 0.187 | 0.747 |
| A/B/D | semua | ~±0.01 | >0.5 | >0.84 |

C 14d p=0.014 **TIDAK lolos** Benjamini-Hochberg (q 0.169 > 0.10) maupun Bonferroni
(0.014 > 0.0083). Tanda "***" di sesi pertama adalah artefak multiple-comparison.

### T3/T4/T5 — stabilitas
- **Era-split** (14d): C negatif di kedua paruh (−0.038, −0.020), keduanya tak sig (n kecil).
- **Jackknife**: C 14d stabil; 0/43 hari membalik tanda; dampak satu hari maks 0.013 → bukan outlier.
- **Funding-source**: Binance-only (49 hari) C 14d = −0.045 p=0.050 → sinyal bertahan, bukan artefak fallback.

### Verdict akhir: TIDAK ADA PERUBAHAN PRODUKSI
Arah C konsisten (bukan outlier), TAPI tidak signifikan secara statistik setelah
koreksi multiple-comparison di 57 hari. A vs B tak terbedakan. **Pertahankan B**.
C = hipotesis terlacak (tracked), uji ulang setelah likuidasi >6 bulan. Tidak menambah
flag config (YAGNI — kode untuk varian belum terbukti). Skrip + cache siap re-run bulanan.

