# Term Premium → BTC: Hasil Uji Lengkap (2017-08..2026-07)

Sumber ide: `C/yield equilibrium model.py` (dekomposisi Y10 = real + E[infl] + term premium +
fiscal premium) + `C/2.docx` (review kritis + saran uji state-dependence).
Pertanyaan: apakah "liquidity layer" yield (khususnya TERM PREMIUM) menggerakkan likuiditas
global → BTC, secara stabil dan bisa dipakai sebagai fitur/gate SFC?

Verdict akhir: **DITOLAK sebagai fitur/gate/blend.** Ada struktur in-sample yang nyata dan
era-konsisten, tapi TIDAK selamat purged walk-forward OOS. Konsisten dengan pola penolakan
SLR/macro sebelumnya. Tidak ada yang di-blend ke scoring.

---

## 1. Di mana "yield/likuiditas" sudah ada di SFC (agar tidak double-count)
- GLF (`data_sources/global_liquidity_engine.py`): Fed BS 30%, ECB 15%, BOJ 3%, China M2 4%,
  US M2 15%, TGA 10%, RRP 10%, DXY 13%. **Tidak ada suku bunga/yield** — sudah menyerap
  penentu yield (balance sheet, M2, DXY).
- M90 GSLS (`global_sovereign_liquidity.py`): yang dipakai **SLOPE** kurva (10Y−2Y) US/JP/EU/UK
  + carry US10Y−JGB10Y → feed Lt. Bukan LEVEL.
- M8 / `sovereign_yield_curves.py`: slope juga. Data mentah: `data/raw/macro`, `data/cleaned/macro_daily_clean.json`.

## 2. Uji dengan PROXY (script: yield_level_btc_test.py, term_premium_btc_test.py)
- Yield LEVEL: corr kontemporer ~0 (Y10 −0.023). Prediktif era-FLIP (Y10 h30 −2.68*, era [−* +* −*]).
  Redundan: corr(Y10 level, FFR)=+0.90, corr(Y10, M2)=+0.46 → cerminan policy rate.
- Proxy TP2=10Y−FFR & TP3=30Y−10Y: **berlawanan tanda** (−3.67 vs +3.04) → tampak construct-validity
  problem. TP1=10Y−REAL10−BE10 = nol identik (sd=0) — BE10 didefinisikan nominal−real, jadi
  "term premium" TIDAK bisa diturunkan sebagai residual dari FRED (harus impor ACM/KW).
- Klaim 2.docx diuji (term_premium_regime_interaction.py): interaction TP×hike menemukan
  struktur state-dependent in-sample (dTP2_30 non-hike +3.37 [+ + +*]; TP2 level×hike −7.16).
- Encompassing (tp_encompassing_test.py): sinyal BUKAN sekadar FFR — setelah dekomposisi,
  yang signifikan adalah d10y (+3.71, p=.028) dan d10y×hike (−3.76), bukan dFFR.
- Purged WF proxy (tp_purged_walkforward.py): RULE IC −0.019±0.313, AUC 0.460±0.171 → GAGAL.

## 3. Uji dengan TERM PREMIUM ASLI (script: fetch_term_premium.py, tp_real_test.py)
Data baru `data/term_premium_daily.json`: **ACM10** (NY Fed, harian 1961-06..2026-09-08, `ACMTP10`)
+ **KW10/KW5** (FRED THREEFYTP10/5, harian 1990+).

- **(G) Construct validity:** corr(ACM10, KW10) = **+0.823**; perubahan **+0.788**. Dua estimator
  asli SALING MENDUKUNG → "sign-flip antar-proxy" tadi **artefak proxy spread**, bukan masalah
  konstruk. (Koreksi penting terhadap kesimpulan proxy.)
- **(A) Kontemporer:** ~0 (ACM ret1 −0.025, corr|ret1| −0.080).
- **(B) Prediktif (pooled HAC30 + era dummies), tanda era:**
  - ACM10 level h30 −4.85 [− − −*]; KW10 h30 −8.72* [−* − −*] → **tanda negatif KONSISTEN lintas era**
    (level TP tinggi → forward return BTC lebih rendah).
  - dACM10_30: h1 +0.12* [+ + +*], h3 +0.32* [+ + +*], h7 +0.77* [+ + +*] → **positif era-konsisten**
    (perubahan TP naik → forward return naik). Level dan change berlawanan tanda (pola wajar:
    level = regime risk-premium/uncertainty; change = reflasi/ekspektasi).
- **(V) Volatilitas:** dengan TP asli kanal vol HILANG (ACM +3.43 p=0.47; KW −1.65 p=0.71).
  → temuan "TP→vol" pada proxy adalah **spurious**.
- **(D) Incremental di atas VIX+SLOPE (fwd30):** ACM ΔAdjR2 +0.0134 (p=0.15, ns);
  KW +0.0583 (p=0.0072). Campur.
- **(E) Interaction TP×hike (2.docx, target fwd30):**
  - dACM10_30: x(baseline)=+5.42 (p=0.061), x:hike=−7.55 (p=0.081); within non-hike [+* + +],
    within hike [−* − .] → **flip antar-regime, dan era-konsisten DI DALAM tiap regime**.
  - dKW10_30: x:hike=−7.71 (p=0.044) — arah sama di estimator independen.
  → struktur state-dependent yang diprediksi 2.docx memang ada, dan bukan sekadar era-confound.
- **(F) Purged walk-forward (expanding, embargo 30, prediksi tanda fwd30) — GATE final:**
  | model | IC | AUC |
  |---|---|---|
  | ACM level | −0.157 ±0.213 | 0.383 ±0.117 |
  | dACM | −0.059 ±0.191 | 0.486 ±0.094 |
  | ACM level+change | −0.019 ±0.268 | 0.508 ±0.131 |
  | ACM level+change **regime-cond** | **+0.076 ±0.255** | **0.525 ±0.155** |

  Syarat lolos: AUC>0.55, IC>0, varians fold rendah. Versi regime-conditional adalah yang terbaik
  (IC>0) tapi AUC 0.525 < 0.55 DAN varians antar-fold besar → **GAGAL**.

## 4. Kesimpulan
1. Secara in-sample, TP asli punya relasi era-konsisten dengan BTC: level → forward return lebih
   rendah; perubahan → lebih tinggi; dan regime-conditional (non-hike vs hike) berlawanan tanda
   dengan era-konsistensi di dalam regime. Dua estimator (ACM & KW) sepakat.
2. Namun OOS purged walk-forward menolaknya: AUC terbaik 0.525, IC +0.076 dengan varians besar —
   tidak melewati gate. Tidak ada bukti edge yang dapat dipakai.
3. Kanal term-premium/fiscal → likuiditas global → BTC karenanya TIDAK terbukti independen.
   Yang benar-benar era-stable untuk regime BTC tetap sinyal crypto-native (funding), bukan macro.
4. Koreksi metodologis yang saya catat: (a) "term premium" tidak bisa = 10Y−real−breakeven (nol);
   (b) sign-flip antar-proxy sebelumnya artefak; (c) kanal TP→vol pada proxy spurious.

## 5. Caveat
- TP asli harian (ACM/KW) mulai 1990 (KW) / 1961 (ACM) tapi sampel BTC dimulai 2017 (n=2238).
- Purged WF: expanding 6 fold, embargo 30d, target overlap-30d → layar, bukan produksi.
- Model WF sederhana (OLS linier) per regime; hasil bisa berbeda dengan model non-linier.
- Data TP didapat via `analysis/fetch_term_premium.py` (FRED key + NY Fed xls). Re-pull butuh
  `xlrd` di venv (python3.12): `.venv/bin/python -m pip install xlrd`.

## 6. Script
- `analysis/fetch_term_premium.py` — pull ACM (NY Fed) + Kim-Wright (FRED) → data/term_premium_daily.json
- `analysis/yield_level_btc_test.py` — yield LEVEL sebagai driver
- `analysis/term_premium_btc_test.py` — proxy TP (level, change, vol, incremental)
- `analysis/term_premium_regime_interaction.py` — uji interaction 2.docx (proxy)
- `analysis/tp_encompassing_test.py` — TP2 vs FFR langsung (encompassing)
- `analysis/tp_purged_walkforward.py` — gate WF (proxy)
- `analysis/tp_real_test.py` — TP ASLI: semua layer + gate WF final
