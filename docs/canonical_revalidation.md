# Canonical 9-Year Re-validation — Macro Factors (2026-08-08)

Re-run walk-forward factor validasi pada HARGA KANONIK Binance BTCUSDT (2017-08 →
2026-07, 9 tahun) sebagai pengganti sumber asli (FRED CBBTCUSD / campuran), PLUS
era-split (2017-21 vs 2022-26) untuk uji stabilitas lintas rezim.

Alasan: seri kanonik exchange-traded, konsisten dengan harga live SFC, bersih 9 tahun.
Uji era-split adalah standar yang sama yang menolak China M2 / JPY carry / HY.

Skrip: `analysis/revalidate_canonical.py` (murni analisis; baca cache walk-forward
yang sudah ada + cache Binance — TIDAK re-fetch FRED, TIDAK sentuh produksi).

## Hasil

### L8 subset (GLF liquidity stress + L6 expectation shock) — dari .walk_forward_imbs_l8.json
Signal tinggi = stress tinggi. Gap = fwd(low-signal) − fwd(high-signal); positif = benar.

| Horizon | Full 9yr [CI90] | era1(2017-21) | era2(2022-26) | Verdict |
|---|---|---|---|---|
| 7d | +0.021 [0.011,0.029]*** | — | — | kuat |
| 30d | +0.072 [0.048,0.094]*** | +0.204 | **−0.109** | **SIGN-FLIP** |
| 90d | +0.264 [0.197,0.329]*** | +0.609 | **−0.374** | **SIGN-FLIP** |
| 180d | +0.473 [0.356,0.591]*** | — | — | kuat |
| 365d | +0.510 [0.341,0.730]*** | — | — | kuat |

Signifikan kuat di window penuh, TAPI era-split membalik tanda: 2017-21 (+0.20/+0.61)
vs 2022-26 (−0.11/−0.37). **ERA-UNSTABLE.** Signifikansi window penuh didorong periode
2017-21 (termasuk bull-run 2020-21); efek INVERT di era terbaru. → Jangan diandalkan
untuk rezim saat ini. Ini memperkuat caveat sebelumnya (L8 2-dim subset, cutoff tak
portable).

### SFC pct (trend continuation) — dari .walk_forward_trend_continuation.json
| Horizon | Full 9yr [CI90] | era1(2017-21) | era2(2022-26) | Verdict |
|---|---|---|---|---|
| 7d | +0.021 [0.012,0.029]*** | — | — | kuat |
| 30d | +0.071 [0.050,0.089]*** | +0.047 | +0.056 | **KONSISTEN** |
| 90d | +0.258 [0.210,0.304]*** | +0.335 | +0.139 | **KONSISTEN** |
| 180d | +0.191 [0.122,0.261]*** | — | — | kuat |
| 365d | −0.483 [−0.636,−0.312]+++ | — | — | membalik |

sfc_pct ERA-KONSISTEN di 30d & 90d (kedua era positif = polaritas benar stabil).
TAPI 365d membalik negatif. → Sinyal SFC stress lebih stabil lintas era di horizon
pendek-menengah daripada L8, meski long-horizon berbalik.

## Implikasi jujur
1. **L8 subset tidak era-stabil** di seri kanonik — perkuat keputusan untuk TIDAK
   menaikkan peran L8 / cutoff-nya di live. Arahnya bergantung era (dominan 2017-21).
2. **sfc_pct lebih robust** (era-konsisten 30d/90d) — kandidat yang lebih layak jika
   mau memakai komponen stress SFC sebagai sinyal, dengan caveat long-horizon.
3. Semua ini ANALISIS. Tidak ada perubahan produksi. Seri kanonik kini tersedia untuk
   re-run bulanan otomatis (cron) guna memantau stabilitas lintas rezim.

## Cara run
```
.venv/bin/python analysis/revalidate_canonical.py --factor .walk_forward_imbs_l8.json \
    --signal l8_subset --factor-name "L8 subset"
.venv/bin/python analysis/revalidate_canonical.py --factor .walk_forward_trend_continuation.json \
    --signal sfc_pct --factor-name "SFC pct"
```
