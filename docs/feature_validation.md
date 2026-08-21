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

## KOREKSI (2026-08-08) — momentum BUKAN prediktor andal di bawah purged-CV
Layar IC DI ATAS MENYESATKAN. Karena label forward-return antar hari bertumpang
tindih, observasi tidak independen → ukuran sampel efektif menggelembung → p-value
IC terlalu optimis. Validasi purged-CV (López de Prado, embargo=h) yang leakage-free:

| Fitur | h | pooled AUC | mean-fold AUC | Verdict |
|---|---|---|---|---|
| mom_30 | 7d | 0.520 | 0.526 ± 0.008 | edge kecil (marginal) |
| mom_30 | 30d | **0.413** | 0.494 ± 0.025 | TIDAK ada edge |
| mom_30 | 90d | **0.371** | 0.510 ± 0.031 | TIDAK ada edge |
| mom_90 | 7d/30d/90d | <0.5 | 0.46-0.49 | TIDAK ada edge |

Skrip `analysis/purged_cv_momentum.py`. Kesimpulan: **momentum TIDAK boleh ditambahkan
ke sinyal/pipeline** — edge-nya hilang (atau terbalik) setelah koreksi leakage label.
Ini menegaskan kenapa purged-CV/embargo wajib (lihat PROJECT_STATUS): IC non-purged
over-estimates skill.

## KOREKSI 2 (2026-08-21) — ALTERNATIVE momentum SPECIFICATIONS also fail purged-CV
The prior section rejected raw `mom_30`/`mom_90`. This section reports whether
BETTER momentum *specifications* have a genuine leakage-free OOS edge (previously
untested). Same purged-CV method (López de Prado, embargo=h, 5 folds,
single-feature LogisticRegression, OOS AUC). Script:
`analysis/purged_cv_momentum_alt.py`. Data: 2017-08-17 → 2026-08-14 (3285 days).

Specifications tested (label = 1 if forward return over h > 0, h ∈ {7,30,90}):
- **risk_managed** — Moskowitz-Ooi-Pedersen TSMOM: `mom_h / rvol_20` (vol-normalized).
- **composite** — multi-horizon blend: mean of z-scored `mom_21/63/126/252`.
- **voltarget** — vol-targeted, CAPPED: `mom_h * min(target/rvol_20, 1)`.
- **ema** — smoothed momentum: EMA(α=2/31) of daily log returns.

Pooled OOS AUC (pooled) / per-fold mean ± SE; `edge` = mean-fold − 1.96·SE > 0.5.
Per-era pooled OOS AUC: era1 <2020, era2 2020-2022, era3 ≥2023. `stable` = era2 &
era3 both on the SAME side of 0.5 (note: here it flags a stable BELOW-chance
relation, i.e. a consistent anti-edge, NOT a positive edge).

| Spec | h | pooled | mean-fold ± SE | base | edge | era1 | era2 | era3 | stable |
|---|---|---|---|---|---|---|---|---|---|
| risk_managed | 7d | 0.479 | 0.501 ± 0.012 | 0.53 | NO | 0.388 | 0.377 | 0.445 | Y (below) |
| risk_managed | 30d | 0.435 | 0.503 ± 0.025 | 0.54 | NO | 0.271 | 0.313 | 0.430 | Y (below) |
| risk_managed | 90d | 0.393 | 0.491 ± 0.024 | 0.52 | NO | 0.245 | 0.180 | 0.206 | Y (below) |
| composite | 7d | 0.458 | 0.482 ± 0.015 | 0.53 | NO | 0.383 | 0.374 | 0.438 | Y (below) |
| composite | 30d | 0.418 | 0.510 ± 0.040 | 0.54 | NO | 0.330 | 0.232 | 0.454 | Y (below) |
| composite | 90d | 0.357 | 0.466 ± 0.035 | 0.53 | NO | 0.276 | 0.185 | 0.182 | Y (below) |
| voltarget | 7d | 0.461 | 0.501 ± 0.012 | 0.53 | NO | 0.367 | 0.369 | 0.441 | Y (below) |
| voltarget | 30d | 0.413 | 0.503 ± 0.025 | 0.54 | NO | 0.260 | 0.263 | 0.417 | Y (below) |
| voltarget | 90d | 0.362 | 0.491 ± 0.024 | 0.52 | NO | 0.243 | 0.161 | 0.217 | Y (below) |
| ema | 7d | 0.463 | 0.514 ± 0.011 | 0.53 | NO | 0.356 | 0.371 | 0.444 | Y (below) |
| ema | 30d | 0.400 | 0.498 ± 0.021 | 0.54 | NO | 0.244 | 0.259 | 0.396 | Y (below) |
| ema | 90d | 0.304 | 0.462 ± 0.023 | 0.54 | NO | 0.323 | 0.181 | 0.265 | Y (below) |

Raw per-fold AUCs and full JSON: `.purged_cv_momentum_alt.json`.

**VERDICT — NOT CONFIRMED. NO alternative momentum specification has a genuine OOS
edge.** Every one of the 12 spec×horizon cells fails the purged-CV test (no cell has
mean-fold − 1.96·SE > 0.5), and every pooled OOS AUC is at or below chance
(0.304–0.479). In fact most are *below* 0.5 while the base rate is ~0.53 up —
i.e. these momentum formulations are mildly ANTI-predictive of up/down, not
predictive, and the effect is too weak/volatile to be tradeable. The `stable=Y`
flags are misleading: they reflect era2 & era3 being consistently BELOW chance
(a stable reversal tendency), which is not a positive edge and does not overturn
the rejection. Honest conclusion: **the momentum family (raw and all respecified
forms) does not survive leakage-free validation — keep momentum DISPLAY-ONLY and
do not add it to any signal/pipeline.** This is consistent with the prior raw
momentum rejection; no specification here is an exception to the leakage-free
requirement (see PROJECT_STATUS).

## Cara run
```
.venv/bin/python analysis/validate_features_purged.py
```
