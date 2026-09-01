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

## Namun: sinyal pendek (event-driven) ADA
- Weekly lagged OLS: SLR_Liquidity signifikan di lag 1-2 minggu (p≈0.003 / 0.016),
  **dan bertahan setelah kontrol GLF** (p≈0.003 / 0.016). Lenyap di lag ≥4 mgg.
- Test #2 (policy attribution, event study): POS events → +10/+12/+18% return fwd
  7/14/30d; NEG events → +3%/-0.9%/-5.0%. Arah sesuai hipotesis (Pos>Neutral>Neg),
  tapi n sangat kecil (5-6 vs 4) — hipotesis, bukan kesimpulan.

## Verdict (per gating rule SLR.md)
Test #1 GAGAL pada frekuensi yang ditentukan → **JANGAN blend SLR ke composite SFC**.
Sesuai prinsip repo (Pitfall 18 walk-forward-validation: tahu kapan berhenti build).
Nilai SLR adalah sebagai sinyal EVENT/SHORT-HORIZON, bukan prediktor macro monthly;
untuk itu butuh event registry yang dikurasi lebih besar + sample daily — effort
terpisah, bukan alasan untuk wiring SLR sekarang.

## Aset
- `analysis/slr_engine.py` — rekonstruksi (reusable, point-in-time, no scoring impact)
- `analysis/slr_test1_incremental.py` — test suite (walk-forward, BIC, weekly OLS, purged-CV, event study)
- `.slr_series.json`, `.slr_test1.json` — hasil (gitignored runtime cache)
