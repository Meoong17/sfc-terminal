# Regime-Conditioning Test — Hasil (does liquidity conditioning improve regime detection?)

SFC = BTC behavior/regime detection system conditioned on liquidity & macro state.
Uji ini menjawab pertanyaan langsung: **apakah conditioning pada likuiditas/macro
menambah kemampuan mendeteksi regime BTC?** Script: `analysis/regime_conditioning_test.py`,
hasil `.regime_conditioning_test.json`.

## Metode
- Regime "sebenarnya" didefinisikan ex-post dari perilaku harga:
  - TREND: bull vs bear (BTC vs 200DMA)
  - STRESS: high-drawdown (max-drawdown 60d tercile atas) — dipilih BUKAN dari realized-vol
    agar baseline bebas leak.
- Variabel conditioning: GLF, term-premium/M91, ΔM2 impulse, order-flow, ETF-flow.
- Uji:
  1. Separasi univariat (Mann-Whitney AUC robust + Cohen's d): apakah variabel membawa
     info state?
  2. Incremental conditioning (5-fold CV logistic AUC): baseline (ret20 + realized-vol)
     vs baseline + variabel — apakah menambah V MENAIKKAN deteksi regime?

Ini uji PENGUKURAN/diskriminasi kontemporan (label dikenal, bukan forecast forward).

## Hasil
### TREND (bull/bear), n=580
| var | MW-AUC | Cohen d | incremental AUC (vs baseline 0.487) |
|---|---|---|---|
| term_prem | 0.467 | -0.08 | +0.002 |
| order_flow | 0.465 | -0.12 | +0.016 |
| etf_flow | 0.531 | +0.17 | -0.002 |
| glf | 0.424 | -0.47 | -0.017 |
| m2_impulse | 0.461 | -0.32 | -0.041 |

Semua MW-AUC ≈ 0.42-0.53 (≈0.5 = tanpa separasi). Semua incremental |Δ| ≤ 0.04, GLF/m2
justru sedikit MERUGIKAN.

### STRESS (high-drawdown), n=580
| var | MW-AUC | Cohen d | incremental AUC (vs baseline 0.759) |
|---|---|---|---|
| term_prem | 0.526 | +0.10 | -0.001 |
| order_flow | 0.496 | -0.03 | -0.010 |
| etf_flow | 0.510 | +0.01 | +0.004 |
| glf | 0.360 | -0.47 | +0.012 |
| m2_impulse | 0.519 | -0.01 | -0.051 |

Baseline (ret20+vol) sudah 0.759 — perilaku harga sendiri sudah memisahkan stress dengan
baik. GLF univariat 0.360 = INVERTED (likuiditas tinggi ↔ stress rendah; jika di-invert,
separasi ~0.64 = moderat), TAPI ditambah ke baseline hanya +0.012 → info GLF REDUNDAN
dengan realized-vol. Semua incremental lain |Δ| ≤ 0.01-0.05.

## Verdict
**Conditioning likuiditas/macro menambah SANGAT KECIL (≈0-0.012 AUC) pada deteksi regime
di atas fitur perilaku harga (ret20 + realized-vol).** Baik untuk TREND maupun STRESS,
baseline perilaku harga sudah memuat hampir seluruh informasi regime; GLF, term-premium,
ΔM2, dan flow hampir semuanya redundan atau noise saat ditambahkan.

## Interpretasi jujur & batasan
- Temuan: informasi regime BTC sebagian besar **hidup di perilaku harga/vol itu sendiri**,
  bukan di state likuiditas/macro yang menambah info unik setelahnya.
- BATASAN: (a) likuiditas GLF di-forward-fill monthly → kasar; (b) label regime didefinisikan
  dari harga, jadi conditioning tak bisa "melihat" regime yang belum muncul di harga (leading);
  (c) model logistic 1-layer ≠ HMM conditional penuh. Jadi ini bukti bahwa conditioning
  likuiditas TIDAK menambah separasi regime yang sudah terlihat di harga — bukan uji final
  HMM conditional.
- Kesimpulan yang aman: sinyal regime SFC sebenarnya dibawa oleh behavior/vol price;
  komponen likuiditas/macro bukan sumber info regime yang independen — konsisten dengan
  rangkaian temuan bahwa macro-liquidity punya nilai tambah terbatas untuk state BTC.

## Aset
- `analysis/regime_conditioning_test.py` — test suite (reusable)
- `.regime_conditioning_test.json` — hasil (gitignored runtime cache)
