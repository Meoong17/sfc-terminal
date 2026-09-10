# Net Liquidity → BTC: Hasil Uji (marginal-liquidity versi kanonik)

Melanjutkan rangkaian "apakah likuiditas eksternal menggerakkan BTC". Ini bentuk paling kanonik
dari tesis marginal-liquidity: **net liquidity dolar** (Fed BS − TGA − RRP) dan perubahannya.
GLF mengagregasi komponen yang sama sebagai z-score YoY; di sini diuji level & Δ dolar apa adanya.

Data: `data/cleaned/macro_daily_clean.json` (FED_BS, TGA, RRP, ECB_BS, BOJ_BS, M2_US) + Binance
Vision BTC. `net_us = FED_BS − TGA − RRP*1000` ($mil; range ≈ $3.5T–$7.1T).
Script: `analysis/net_liquidity_btc_test.py`, `net_liquidity_incremental.py`, `net_liquidity_event_study.py`.

## 1. Regresi kontinu (level & Δ)
Prediktif fwd return (pooled HAC30 + era dummies), tanda era:
- `yoy_net_us`: h1 +0.19* [+*+*+*], h3 +0.55* [+*+*+*], h7 +1.28* [+*+ +*], h30 +4.94* [+*+ +*]
  → positif & tanda era-KONSISTEN (satu-satunya kanal macro yang begitu).
- `dnet_us_30`: h30 +2.51* [+*+ +]; `dnet_us_90`: [+ + +].
- `net_us` level: h30 +6.22* [+* −* +*] (era2 negatif → level era-flip).
Incremental di atas VIX+SLOPE (fwd30): `yoy_net_us` ΔAdjR2 **+0.0941** (p=0.0003);
`dnet_us_30` +0.0119 (p=0.031).

## 2. Incremental di atas GLF (proxy lokal, bobot global_liquidity_engine)
fwd30 ~ z(GLF_proxy) + z(dnet_us_30): GLF +3.03 (p=0.094), **dnet_us_30 +2.07 (p=0.027)**.
corr(GLF_proxy, dnet_us_30)=+0.175 → tidak kolinier; net-liquidity menambah di atas GLF.
(DXY tidak ada di file cleaned → GLF-proxy tanpa DXY; caveat.)

## 3. Purged walk-forward gate (expanding, embargo 30, prediksi tanda fwd30)
- `dnet_us_30`: IC **+0.074 ±0.055**, AUC **0.552 ±0.066** (folds=6).
- Robustness h=30, folds 4..8: AUC 0.574 / 0.558 / 0.549 / 0.566 / 0.541 → konsisten ≈0.55.
- Permutation label-shuffle (200x): observed 0.549 vs null 0.498±0.012, **p=0.000**.
- Tapi: fold-inconsistency lebar (fold 0.29 … 0.68); GLF+dnet WF AUC turun ke 0.522.
→ Ini SATU-SATUNYA kandidat sejauh ini yang AUC OOS ≥0.55 dengan IC>0 & varians rendah,
  walau ambangnya tipis.

## 4. Event study (lapisan quasi-natural-experiment) — KUNCI
Episode besar (auto: 90d Δnet/ΔTGA terbesar; + fixed events):
| episode | +30d | +60d | +90d |
|---|---|---|---|
| net-liq expansion 2020-05 | +4.5% | +9.3% | +28.4% |
| net-liq expansion 2021-04 | −30.1% | −44.4% | −65.5% |
| net-liq expansion 2023-03 | +4.2% | −1.0% | +7.1% |
| net-liq expansion 2025-04 | +12.5% | +20.6% | +23.0% |
| TGA drain 2021-05 | −2.7% | +10.1% | +27.3% |
| TGA drain 2021-10 | −2.7% | −29.6% | −36.4% |
| TGA drain 2023-05 | +8.9% | +4.0% | −7.6% |
| TGA drain 2025-04 | +12.6% | +22.6% | +25.7% |

Agregat event (n=8) vs baseline (fwd30 +2.53%, fwd60 +4.98%, fwd90 +7.35%):
- h30 event +0.90% → z=−0.22
- h60 event −1.06% → z=−0.55
- h90 event +0.24% → z=−0.51
→ Episode likuiditas-terbesar TIDAK menghasilkan forward return di atas baseline (semua z negatif),
  dan antar-episode sangat inkonsisten (2021-04 +30d −30% vs 2025-04 +12.5%). Termasuk top 2021.

## 5. Verdict
- Regresi kontinu memberi *screen* lemah-positif (AUC ~0.55, permutation p=0.000, menambah di atas GLF).
- TAPI event study quasi-natural-experiment TIDAK mengonfirmasi: episode terbesar tidak di atas baseline
  dan arahnya tak konsisten. Sinyal kontinu tidak didukung struktur event yang bersih.
- Ini kanal macro-liquidity KETIGA berturut dengan pola sama (SLR → term premium → net liquidity):
  layar in-sample/screen tampak ada, tapi gagal di uji decisive/bersih, dengan episode "bersih" yang
  sedikit & terklaster.
- Kesimpulan: **net liquidity DITOLAK sebagai fitur/gate/blend.** Tidak di-blend ke scoring.

## 6. Implikasi struktural
Episode likuiditas-macro "bersih" untuk BTC praktis hanya ~2 (QE 2020-21; TGA/RRP drain 2023-24),
dan keduanya berseberangan arah. Jadi kegagalan berulang kemungkinan adalah **batas struktural
(kekurangan episode bersih)** — bukan sekadar salah spesifikasi. Menambah kanal macro-liquidity
lebih jauh expected value-nya rendah.

## 7. Caveat
- Sampel BTC mulai 2017 (n≈3265); episode besar sedikit & terklaster → daya uji rendah.
- GLF-proxy tanpa DXY (tidak ada di file cleaned).
- WF: expanding 6 fold, target overlap-30d, model OLS linier.
- TGA/RRP level mungkin punya artefak forward-fill bulanan.
