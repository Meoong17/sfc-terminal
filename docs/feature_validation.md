# Feature Validation — Binance-derived factors (2026-08-08)

Layar OOS fitur kandidat dari data Binance kanonik (9 tahun) terhadap forward BTC
return. Skrip: `analysis/validate_features_purged.py`. Pure analysis, no production.

Metode: Information Coefficient (Spearman) antara fitur di t dan forward-return
t..t+h, pada seri kanonik 2017-2026, dengan bootstrap 90% CI + p-value dua sisi,
koreksi Benjamini-Hochberg (q) lintas semua uji, dan era-split (2017-21 vs 2022-26).

## Hasil
| Fitur | Horizon | IC | q(BH) | era1 | era2 | Stabilitas era |
|---|---|---|---|---|---|---|
| **mom_30** | 7d | +0.094 | <0.001 | +0.10 | +0.05 | STABIL |
| **mom_30** | 30d | +0.108 | <0.001 | +0.15 | +0.03 | STABIL |
| **mom_30** | 90d | +0.119 | <0.001 | +0.12 | +0.11 | STABIL |
| **mom_30** | 180d | +0.050 | 0.013 | +0.01 | +0.07 | STABIL |
| **mom_90** | 7d | +0.058 | 0.004 | +0.08 | +0.01 | STABIL |
| **mom_90** | 30d | +0.075 | <0.001 | +0.07 | +0.07 | STABIL |
| rvol_30 | 30d | +0.038 | 0.072 | +0.04 | +0.01 | STABIL (positif) |
| rvol_30 | 90d | −0.074 | <0.001 | −0.07 | −0.14 | STABIL (negatif) |
| rvol_30 | 180d | −0.181 | <0.001 | −0.22 | −0.17 | STABIL (negatif) |
| rvol_90 | 90d | −0.217 | <0.001 | −0.27 | −0.23 | STABIL (negatif) |
| vol_30 | 90d | −0.095 | <0.001 | −0.03 | −0.12 | STABIL (negatif) |
| funding | 30d | −0.056 | 0.011 | −0.30 | −0.02 | STABIL (negatif) |
| funding | 90d | −0.044 | 0.068 | −0.30 | +0.02 | **FLIP** |
| premium | 30d | −0.012 | 0.67 | −0.29 | +0.04 | **FLIP** |
| premium | 90d | −0.001 | 0.95 | −0.34 | +0.06 | **FLIP** |
| taker_ratio | 90d | −0.081 | <0.001 | −0.16 | −0.01 | STABIL (negatif) |

(Semua uji: n≈2400-3200; baris non-signifikan dihilangkan sebagian.)

## Interpretasi jujur (baca manual — label auto menyesatkan utk tipe-vol)
1. **MOMENTUM (mom_30/mom_90) = fitur paling robust**: IC positif, era-stabil, lolos
   BH di horizon pendek-menengah. Efek persisten lintas rezim. Ini kandidat sinyal yang
   sah untuk divalidasi lanjut bila mau dipakai.
2. **REALIZED VOL**: signifikan & era-stabil TAPI TANDA BERGANDA horizon. rvol_30 positif
   di 30d (vol tinggi → return tinggi pendek), NEGATIF di 90d/180d (vol tinggi → return
   rendah panjang, efek risk-premium). Perlu spesifikasi tanda per-horizon; tidak bisa
   dipakai sebagai satu sinyal satu-arah.
3. **FUNDING & PREMIUM: ERA-UNSTABLE** (premium jelas flip; funding flip di 90d). Meski
   funding negatif (tinggi→return rendah) stabil di 7d/30d/180d, flip di 90d + premium
   flip → JANGAN diandalkan. Konsisten dengan temuan demo sebelumnya.
4. **VOLUME & TAKER_RATIO**: lemah; hanya signifikan di horizon panjang, taker_ratio
   satu-arah tapi efek kecil.

## Implikasi
- Momentum layak dipertimbangkan sebagai fitur tambahan (setelah validasi purged-CV
  formal / masuk pipeline). Belum dipakai di produksi.
- Funding/premium/vol tidak boleh dijadikan andalan tanpa spesifikasi tanda & horizon
  yang ketat — pola yang sama yang menolak China M2/JPY carry/HY.
- Tidak ada perubahan produksi.

## Cara run
```
.venv/bin/python analysis/validate_features_purged.py
```
