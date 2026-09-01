# Macro Lead/Transition Test — Hasil (2.docx open question: does external condition lead regime flips?)

## Pertanyaan
Apakah kondisi external (GLF, VIX, real yield, term-premium, sovereign stress) berubah
SEBELUM perilaku BTC berpindah regime — sebagai info transisi, bukan ramal harga?

## Metode
- Regime ex-post dari harga: trend bull/bear (200DMA, 90 transisi), stress/calm (vol p90, 50 transisi).
- Event-study: rata-rata tiap variable di window [-7,-1],[-14,-1],[-30,-1] sebelum transisi vs baseline sampel (z-score).
- Lead-lag: cross-correlation corr(ΔX[t-l], flip[t]) untuk l=0..30.

## Hasil — TIDAK ada info lead/transition yang signifikan
- Cross-correlation lead-lag SEMUA variable ≈ 0 (trend: |r|≤0.094, mayoritas ≤0.04; stress: ≤0.138 tapi artefak forward-fill bulanan GLF). Tidak ada yang memimpin transisi.
- Pre-transition drift LEMAH (z<1, tidak signifikan). Z terbesar: TERM_PREM/US30-US2/GLF di regime STRESS (z≈+0.78..+0.88 — term-premium & GLF lebih tinggi ~0.85 sd sebelum onset stress), REAL_Y10 di stress (z=-0.81). Semua <2 (tidak signifikan) dan dalam konteks multiple-comparison (6 var × 2 regime × 3 window) → noise.
- n_trans efektif kecil (31 untuk stress windows), dan trend via 200DMA flip sering = noise.

## Kesimpulan
Pertanyaan terbuka dari 2.docx terjawab: macro/liquidity TIDAK menunjukkan
lead/transition info yang terdeteksi untuk transisi regime BTC. Sesuai framing 2.docx,
macro/liquidity layak jadi CONTEXT (dashboard), bukan modul pemimpin transisi regime.
Caveat daya uji rendah (sedikit transisi) — tapi tidak ada sinyal lead sama sekali.

Script: analysis/macro_lead_transition_test.py
