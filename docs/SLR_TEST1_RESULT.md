# SLR v2 — Hasil Uji (Test #1 gating)

Dokumen ini mencatat hasil validasi arsitektur Sovereign Liquidity Regime (SLR) v2
(C/SLR.md) terhadap data nyata. Rekonstruksi + uji dibuat di
`analysis/slr_engine.py` dan `analysis/slr_test1_incremental.py` (RESEARCH ONLY,
tidak menyentuh scoring live). Hasil mentah: `.slr_series.json`, `.slr_test1.json`.

## Rekonstruksi (point-in-time)
- M91 Sovereign Duration Stress: z-score(window=252) dari Δ(term-premium proxy
  dY30 − dY2), FRED DGS30/DGS2 sejak 2000. Objektif.
- M92 Policy Liquidity Response: event registry 10 episode (6 positif/4 negatif),
  direction-aware, human-in-the-loop per desain. Subjektif — dokumentasi eksplisit.
- M93b Market Response: dari FRED CBBTCUSD. M93a (capital flow) TIDAK bisa
  direkonstruksi historis (ETF/stablecoin/whale history terlalu pendek) → TC=M93b,
  dicatat jujur. OI tidak ada di history Binance Vision → sub-komponen OI_sustainability
  diomisikan.
- SLR_Liquidity (interim) = geometric_mean(M91,M92,TC) floor 15; SLR_Risk = M91.
- Keluaran: 2.810 hari sejajar / 135 bulan (2014-2026).

## Test #1 (gating): SLR menambah predictive power di atas GLF?
Definitif **TIDAK**, pada frekuensi monthly yang dipakai SLR.md:
- Walk-forward OOS (expanding, lagged): OOS R2 SLR-incremental NEGATIF di semua p
  (-0.011 / -0.005 / -0.006), DM SLR-vs-GLF p>0.55.
- BIC posterior: P(H1|data)=0.081, BF=0.088 → **evidence AGAINST** SLR menambah edge.
- Spearman (SLR vs arah return 3-bulan ke depan) ≈ 0.12, p=0.17; GLF ≈ 0.10, p=0.24.
  Mean SLR saat return-fwd>0 (36.8) ≈ saat <=0 (35.2) → tanpa separasi.
- Purged-CV AUC sempat <0.5 untuk SEMUA fitur (GLF/SLR/GLF+SLR) — diperiksa ulang,
  ini artefak sample monthly kecil (n≈130), bukan sinyal anti-predictive.

Konsisten dengan temuan lama repo: GLF→BTC juga TIDAK punya edge monthly
(causal battery, BF≈0.003). SLR bukan pengecualian.

## Namun: sinyal pendek (event-driven) ADA — TAPI itu ILUSI (momentum)
Weekly lagged OLS awalnya menunjukkan SLR_Liquidity signifikan di lag 1-2 minggu
(p≈0.003/0.016) bahkan setelah kontrol GLF. Tapi DEKOMPOSISI komponen (lag-2, dengan
kontrol GLF) membuktikan sinyal itu BUKAN mekanisme SLR:
- **M91 (sovereign duration stress): coef=-0.031, p=0.20 → TIDAK signifikan, tanda NEGATIF.**
  Komponen inti SLR tidak punya edge, dan arahnya malah berlawanan tesis.
- **TC (market response = momentum BTC): coef=+0.081, p=0.0000 → paling dominan.**
  korelasi SLR~TC = 0.61 → SLR_Liquidity signifikan (p=0.016) hanya karena menyerap
  komponen momentum BTC. Ini sirkuler (BTC yang sudah naik cenderung terus naik).
- M92 (policy): coef=+0.359, p=0.038, tapi ~86% hari netral — didorong segelintir
  hari event.

## Event study daily (analysis/slr_event_study_daily.py)
- 92 event sovereign-duration-stress terdeteksi objektif (M91 z>=1.5).
- **Stress events TIDAK menambah apa-apa di atas base rate (netral, bukan aktif buruk):**
  mean unconditional 30d forward return = +6.30%, stress-event = +3.81% (CI [-0.10,+7.84]
  mencakup base rate). Bootstrap diff (stress − unconditional) di SEMUA horizon (1/3/7/14/30d)
  n.s. — 30d = -2.50pp, CI [-6.44,+1.65] memuat nol. Placebo random-date (+6.4%) cocok dengan
  mean unconditional → estimasi base rate yang tak bias. Hint lemah arah negatif (semua diff
  negatif) tapi tak signifikan.
- Split kebijakan (Test #2) **tidak teruji**: registry 10 event tidak sejajar trigger
  M91 → 0 event positif, 3 negatif yang kena. n terlalu kecil.

## Verdict (per gating rule SLR.md + prinsip repo)
Test #1 GAGAL (monthly) + event study daily tidak mendukung + dekomposisi membuktikan
"sinyal short-horizon" hanyalah momentum BTC yang tertanam di komponen TC, BUKAN
mekanisme sovereign-stress (M91 n.s. & wrong-sign). **SLR DITOLAK sebagai sinyal
independen — JANGAN blend ke composite SFC.** Ini pola narasi→uji→tolak yang sama
dengan China M2/JPY carry/HY/momentum/Alphractal/WWI/Kronos. Momentum sendiri sudah
pernah DITOLAK repo via purged-CV (AUC<0.5). Berhenti membangun (Pitfall 18).

## Aset
- `analysis/slr_engine.py` — rekonstruksi (reusable, point-in-time, no scoring impact)
- `analysis/slr_test1_incremental.py` — test suite monthly (walk-forward, BIC, weekly OLS, purged-CV, event study)
- `analysis/slr_event_study_daily.py` — event study daily (Test #2 attribution + Test #3 placebo) + dekomposisi komponen
- `.slr_series.json`, `.slr_test1.json`, `.slr_event_study.json` — hasil (gitignored runtime cache)
