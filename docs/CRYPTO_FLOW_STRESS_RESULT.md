# Crypto-native Marginal-Flow → Regime Detection (P1): Hasil

Latar: kanal macro-liquidity sudah ditolak berulang (SLR, term premium, net liquidity) dengan
hipotesis batas struktural. P1 = pivot ke family crypto-native (funding, taker flow, trade-size
skew, whale share) sebagai detektor REGIME (bukan arah harga), diuji dengan purged walk-forward.

Data: `data/binance_vision_daily.json` (funding_mean, close) + `data/binance_orderflow_daily.json`
(taker_imbalance_quote, taker_buy_ratio, whale_hi_quote, n_buy/n_sell, med/p99 notional),
n=3241 (2017-08-17..2026-07-31). OI harian hanya 39 hari → tidak dipakai.
Script: `analysis/crypto_flow_stress_test.py`.

Target (forward-30d, jadi DETECTION bukan hindsight):
- `stress_fwd` = fwd30 realized vol > median trailing-365d
- `bull_fwd`   = close_{t+30} > SMA200_{t+30}

Model logistic (standardisasi TRAIN-only), purged expanding WF (6 fold, embargo 30).

## Hasil
| fitur | stress_fwd AUC | bull_fwd AUC |
|---|---|---|
| base (vol20, mom30) | 0.535 ±0.122 | 0.730 ±0.208 |
| +funding | 0.531 ±0.088 | 0.784 ±0.093 |
| +flow (taker/whale/size) | 0.475 ±0.120 | 0.700 ±0.174 |
| funding ONLY | 0.557 ±0.130 | 0.648 ±0.118 |

## Permutation control (KUNCI)
Menambahkan fitur apa pun (termasuk yang diacak) menaikkan AUC WF — jadi "+0.054 dari funding"
harus diuji vs kebisingan:
- bull_fwd: base 0.730, obs(+funding) **0.784**, null(shuffle funding) mean **0.785** sd 0.022 → **p=0.540**
- stress_fwd: base 0.535, obs 0.531, null 0.526 → **p=0.390**

→ Keuntungan "+funding" **100% artefak penambahan dimensi fitur**, bukan informasi. Versi funding
yang diacak memberi AUC sama (0.785). Funding tidak menambah di atas peluang.

## Verdict
1. **Family crypto-native marginal-flow DITOLAK sebagai penambah deteksi regime OOS.** Flow extras
   (taker imbalance, whale share, trade-size skew) MERUSAK (ΔAUC negatif di kedua target).
2. **Funding bukan forward-predictor**: tambahannya artefak (permutation p≈0.54). Ini konsisten
   dengan temuan SFC sebelumnya: funding = *pembaca keadaan/konfirmasi kontemporer*, BUKAN pemimpin.
   (Temuan era-stable funding bull/bear terdahulu tetap sah sebagai diskriminator KONTEMPORER.)
3. Deteksi stress forward gagal untuk semua set fitur (AUC ≤0.557, varians besar).
4. Satu-satunya yang OOS-prediktif adalah inti price/vol (momentum/vol) — yang memang sudah ada di SFC.

## Pelajaran metodologis (reusable)
Saat membandingkan set fitur bersarang dengan WF AUC, **wajib** pakai kontrol permutasi: menambah
fitur (bahkan noise) menaikkan AUC OOS semu. Tanpa kontrol ini, kontribusi fitur mudah
di-over-klaim. (Ini menjelaskan kenapa banyak kandidat "tampak menambah" di screen in-sample.)

## Implikasi untuk SFC
Mendukung arah P2 (konsolidasi): inti era-stable = stress-gauge + price/vol; funding = konfirmasi
kontemporer; macro = display-context. Tidak ada fitur marginal-flow/macro baru yang additif OOS.
Arsitektur SFC saat ini sudah benar; yang perlu bukan menambah kanal, tetapi merawat & memvalidasi
inti yang ada.

## Caveat
- Target stress pakai median trailing dari fwd_vol30 (overlap jendela) — sedikit look-ahead di
  definisi threshold; namun hasilnya null di kedua arah sehingga tidak mengubah verdict.
- OI harian terlalu pendek; derivatif lain (liq, LS ratio) belum diuji panjang.
- Logistic linier; model non-linier bisa berbeda (tapi arah "flow merusak" cukup tegas).
