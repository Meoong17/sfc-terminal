# Divergence Test — TrendStrength tinggi + MomentumOverlay bearish

Sumber ide: /home/ubuntu/C/1.docx. Script: `analysis/divergence_trend_momentum_test.py`.
Data: `data/binance_vision_daily.json` BTC harian 2017-08-17..2026-07-31 (3271 hari), harga-only
(sesuai scope Option A; bukan replay live modul data_sources).

## Definisi (proxy harga, formula SFC)
- momentum_domain (0-1): RSI14 + MACD-hist + BB-width + OBV, komposisi & band BEARISH (<=0.45)
  mengikuti `data_sources/trend_strength.py` (MOM_BEAR).
- TrendStrength (0-100): komposit bobot momentum .40 / alignment .35 / structure .25 (default modul),
  alignment = kesepakatan multi-horizon, structure = posisi vs MA100. STRONG >= 65.
- DIVERGENCE = TrendStrength>=65 DAN momentum_domain<=0.45.

## Hasil
### Varian literal (komposit modul) — TIDAK TERUJI, nyaris kosong
n = 10 hari / 3271 (0.31%), terbagi era 5/3/2. Penyebab struktural: momentum = 40% bobot
TrendStrength dan korrelat dengan trend, jadi "TS tinggi + momentum bearish" nyaris tak pernah
muncul. n=10 => daya statistik ~0, tak bisa menyimpulkan apa-apa (bukan bukti ada/tak-ada efek).

### Varian decoupled (struktur kuat TANPA momentum vs momentum bearish) — TERUJI, n=66
Memisahkan "struktur naik kuat" dari momentum (ts_struct = alignment+structure, momentum dibuang)
supaya konsep divergence bisa diukur. Ini bentuk hipotesis yang bisa diuji.

| State | n | mean fwd 7d | gap vs baseline (CI90) | sig |
|---|---|---|---|---|
| DIVERGENCE | 66 (2.0%) | +3.38% | +2.34pp [+0.10, +4.73] | ya (screen) |
| AGREE_BULL | 1263 | +1.89% | +0.86pp [+0.35, +1.38] | ya |
| AGREE_BEAR | 962 | +0.41% | -0.62pp [-1.18, -0.07] | ya |
| REVERSE_BULL | 209 | +1.65% | +0.61pp | tidak |

Era split 7d DIVERGENCE:
- era1 2017-2020: n=22, mean +0.06% (gap -1.8pp)  <- kontra
- era2 2021-2023: n=30, mean +6.5%  (gap +6.0pp)  <- satu-satunya pendorong
- era3 2024-2026: n=14, mean +1.8%  (gap +1.3pp)

## Verdict
1. Divergence literal (komposit modul) praktis tak pernah muncul => bukan komponen yang layak
   digali sebagai fitur; definisinya internally hampir mustahil (momentum 40% dari TrendStrength).
2. Bentuk decoupled (struktur up kuat + momentum bearish = pullback-in-uptrend) punya sinyal
   full-sample 7d POSITIF (+2.34pp), TAPI: arah berlawanan dengan narasi "momentum bearish -> turun"
   (ini mean-reversion ke atas), dan TIDAK era-stable — gap penuh ditarik era2 2021-23; era1 negatif;
   n per era kecil (22/30/14).
3. Belum walk-forward purged; label forward 1/3/7d overlapping menaikkan independensi CI => screen saja.

Kesimpulan: TIDAK ada bukti robust bahwa komponen divergence ini punya informasi prediktif
era-stable utk return 1D/3D/7D. Mengikuti aturan SFC (jangan blend sinyal yang belum walk-forward
validated; layering display-only), komponen ini TIDAK direkomendasikan menjadi fitur/blend scoring.
Bila mau digali lanjut: (a) uji dengan regime-gate harian sebenarnya (replay Option B HMM/MTF/DFS +
bucket sfc), (b) horizon intraday, atau (c) pooled-reg + kontrol era — tapi prospek awal lemah.

## Caveat metodologis
- Proxy harga, tanpa regime gate per-hari (tak ada seri bucket sfc harian disimpan). Bukan replay
  literal modul trend_strength/momentum_overlay. Tidak blended ke mana pun (display-only research).


---

## DEEP FOLLOW-UP (divergence_trend_momentum_deep.py) — pooled OLS, era-controlled
Menjawab kelemahan screen: efek full-sample apakah era2-artifact? Dipool dgn kontrol era
(HC1). Populasi = SEMUA hari berstruktur-kuat (ts_struct>=65, n=1528), momentum diuji
kontinu + band, supaya daya naik jauh di atas ambang n=66.

Dalam struktur-naik-kuat, koef momentum terhadap fwd return (pp per unit mom):
- 1d: -0.22 (p .82)  3d: -0.90 (p .59)  7d: -0.37 (p .89)  => TIDAK signifikan semua horizon
- band bear_vs_neutral: 7d +1.46pp (p .37), 3d +0.12 (p .91) => tidak signifikan
- interaksi era x bear (7d): era1 = -3.22pp, era2 = +5.44pp (p_x_era2 = 0.015),
  era3 = +0.64 => TANDA MEMBALIK signifikan antar era (era2 berlawanan dgn era1).

Era-mean 7d (bear vs nonbear): era1 0.06 vs 3.28; era2 6.55 vs 1.11; era3 1.80 vs 1.16.
Prevalensi bear dalam struktur-kuat kecil: era1 n=22, era2 n=30, era3 n=14.

### Verdict deep
Divergence (momentum bearish dalam struktur-naik-kuat) TIDAK punya kandungan prediktif yang
era-stabil utk return 1/3/7d. Efek full-sample screen semula hanyalah artefak era2 2021-23:
uji interaksi era menunjukkan tanda membalik signifikan antar era (era1 negatif vs era2
positif, p=0.015) — bukti tidak-stabil, bukan sekadar observasi era-flip. Arah juga tidak
konsisten dgn narasi "momentum bearish -> turun"; kalau pun ada sinyal lemah itu mean-reversion
ke atas yang hanya ada di era2.

REKOMENDASI: jangan jadikan komponen divergence ini fitur / blend ke scoring. Sesuai aturan SFC
(jangan blend sblm walk-forward validated; hasil ini gagal konsistensi era). Tidak ada langkah
berikut yang hemat-biaya menjanjikan — bukti cukup utk menutup jalur ini, kecuali nanti ada
definisi momentum/gate yang benar-benar baru.

Caveat: label forward overlapping menaikkan independensi (SE HC1 terlalu rendah) => kesimpulan
"tidak era-stabil + tidak robust" berlaku lebih kuat, bukan lebih lemah.
2|
3|---
4|
5|## CYCLE / PHASE SPLIT (divergence_trend_momentum_cycle.py) — dokumentasi final
6|Phase (harga-only): BULL = close>SMA200 & slope SMA200 90d>0; BEAR = lainnya.
7|Dalam struktur-naik-kuat (ts_struct>=65, n=1465):
8|
9|by phase:
10|- BULL (n=1090, bear 32): divergence 7d 1.23% vs nonbear 1.88% -> gap -0.65pp (p=.51, ns).
11|  => dalam siklus bull sehat, momentum bearish di struktur-kuat TIDAK punya edge (cenderung
12|  underperform, tidak signifikan). Berlawanan dgn narasi "pullback bull -> balik kuat ke atas".
13|- BEAR (n=375, bear 14): 7d 12.7% vs 1.63%, ols +11pp p=.0016. TAPI n=14 dan ini episode
14|  reversal dasar-bear (bottom 2018/2022), 1-2 kluster -> TIDAK robust, bukan sinyal teratur.
15|
16|by maturity:
17|- BULL_NEAR_HIGH (bull sehat dekat high, n=932, bear 24): 7d 0.37% vs 1.68% (gap -1.3pp).
18|  => divergence di dekat puncak justru FLAT/underperform. Bukan sinyal naik.
19|- BULL_OFF_HIGH (bull tapi >=15% dr high, n=158, bear 8): 7d 3.81% vs 3.13% -> tak ada edge.
20|
21|Kesimpulan phase-split: upside full-sample divergence itu KONSENTRASI di segelintir episode
22|reversal dasar-bear (artefak era2, bottom 2022 dkk), BUKAN sinyal bull-pullback yang andal.
23|Di kasus paling bersih (bull sehat dekat high: n=24), divergence tak punya edge. Verdict
24|diperkuat: regime-dependent, bukan fitur.
25|