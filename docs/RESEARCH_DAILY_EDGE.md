# SFC Research Dataset — Daily Macro Predictive-Edge Audit
**Tanggal:** 2026-08-15 · **Sumber:** `data/merged/sfc_research_daily.json` (3285 hari, 2017-08→2026-08-14)
**Script:** `analysis/research_daily_edge.py` · **Output:** `.research_daily_edge.json`

## Pertanyaan (dari proposal Bisa.docx)
> Apakah variabel makro harian benar-benar menghasilkan alpha untuk prediksi
> return BTC, atau hanya klasifikasi yang terlihat bagus setelah hasil diketahui?

## Metode (standar SFC, skill predictive-probability-validation)
- Forward return BTC 7d & 30d.
- Quantile gap: bottom-20% vs top-20% signal → selisih forward return (pp).
- Bootstrap P(predictive) (2000 draw), BIC posterior / BF_10, Spearman IC.
- **ERA-SPLIT wajib** (3 blok): verdict dari era TERBARU, bukan agregat.

## Hasil ringkas

### 1. Era-STABLE (arah konsisten di ketiga era) — temuan robust, tapi hati-hati arahnya
| Kolom | 30d gap | era1 | era2 | era3 | Verdict |
|---|---|---|---|---|---|
| **SPX** | −8.3 | −27.9 | −26.5 | −19.5 | **STABLE negatif**: SPX tinggi → return BTC RENDAH |
| **NDX** | −7.9 | −19.2 | −3.7 | −19.9 | STABLE negatif (era2 melemah) |
| **M2_US** | −11.8 | −13.5 | −34.7 | −17.6 | STABLE negatif |
| **BE10** (10y breakeven) | −13.4 | −13.4 | −39.0 | −9.6 | STABLE negatif |
| **US30Y** | −10.4 | −16.7 | −19.4 | −5.0 | STABLE negatif |
| **HY_OAS / IG_OAS** | +20.5/+18.3 | +24.3/+18.3 | +14.9/+10.8 | +11.1/+14.3 | STABLE **positif** (hanya dr 2023-08, 33% coverage) |

> **Peringatan interpretasi:** gap NEGATIF utk SPX/NDX berarti ketika risk-asset
> tinggi, return BTC berikutnya cenderung RENDAH. Ini pola **mean-reversion /**
> BTC-sudah-rally, BUKAN "makro memprediksi BTC ke arah naik". Arah ini justru
> kebalikan dari intuisi risk-on naif. Layak dicatat, tapi jangan dijadikan
> driver tanpa uji tambahan (confound potensial: level vs momentum).

### 2. Era-FLIP / MATI di era terbaru — TIDAK stabil (jangan jadikan driver)
| Kolom | 30d gap | era1 | era2 | era3 | Verdict |
|---|---|---|---|---|---|
| **US10Y** | −7.4 | −16.3 | −21.9 | **−0.2 (P=0.56)** | FLIP → lemah di era3 |
| **US2Y / US5Y** | −6.3/−8.4 | −14.8/−14.6 | −13.5/−22.0 | **+11.1/+7.6** | FLIP → era3 positif |
| **REAL_Y10** (TIPS) | −6.0 | −11.8 | −3.5 | **+4.5** | FLIP |
| **VIX** | −2.2 | −12.1 | +7.6 | **+3.7** | FLIP (era2 sudah balik) |
| **BRENT / WTI** | −16.7/−18.1 | −8.8/−12.8 | −33.9/−34.3 | **+7.4/+6.2** | FLIP era3 |
| **FED_BS / ECB_BS / BOJ_BS** | −3.4/−1.4/−2.5 | +14.3/−3.0/−14.9 | −34.9/−36.6/−20.9 | **+20.1/+13.5/+15.5** | FLIP parah |
| **FEDFUNDS** | −3.1 | −0.3 | +1.2 | **+4.9** | FLIP |
| **RRP** | −12.7 | +14.2 | −24.5 | **+16.4** | FLIP parah |
| **TGA** | +0.7 | −14.0 | +28.1 | **−2.6** | FLIP |
| **SPREAD_10_2 / SPREAD_5_30** | +7.8/+8.1 | +15.8/+25.1 | −6.2/+9.5 | **−6.1/−7.5** | FLIP |

## Verdict (per aturan era-stability)
- **TIDAK ada variabel makro harian yang era-stable dengan arah intuitif
  "stress tinggi → BTC turun" di era terbaru.** Hampir semua variabel yang
  terlihat kuat secara agregat (BF hingga 10^30+) adalah artefak era1/era2 dan
  **FLIP sign di era3 (2022-26)** — persis pola yang sudah kita temukan utk
  GLF/liquidity (memory: era-flip 2026-08).
- Konsistensi era-flip di banyak kolom independen (rates, credit via FED_BS,
  VIX, commodities, liquidity) = **struktural, bukan noise**: hubungan
  makro→BTC berubah total setelah 2022.
- **Satu-satunya temuan era-stable** adalah pola mean-reversion (SPX/NDX/M2/BE10/
  US30Y tinggi → return BTC rendah) dan HY/IG OAS positif (tapi coverage 33%).
  Ini bukan edge prediktif directional yang bisa dijadikan driver; paling jauh
  layak sebagai catatan regime-context display-only.

## Rekomendasi
1. **JANGAN build** driver baru dari variabel makro harian ini tanpa walk-forward
   purged-CV yang lolos di era3. Full-sample BF astronomis terbukti menyesatkan.
2. Data harian tetap berguna utk **diagnostik regime** (monitoring), bukan prediksi.
3. Gap GOLD tetap terbuka; utk kelengkapan penelitian (bukan driver) bisa diisi
   via TradingView GC1! export.
4. Tindak lanjut opsional: uji momentum/level (bukan level polos) utk SPX/NDX
   mean-reversion, dan periksa confound level-vs-momentum sebelum menarik
   kesimpulan.
