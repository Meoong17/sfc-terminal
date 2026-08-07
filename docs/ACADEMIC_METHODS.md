# Metode Akademik untuk Kalkulasi Model SFC

Dokumen riset: metode matematika, ekonomi, dan coding dari literatur akademis yang
dapat dipakai untuk memperkuat kalkulasi model SFC Terminal. Disusun 2026-08-07.
Pola pembacaan: setiap subsistem SFC dipetakan ke metode akademis ber-"grounding"
(sumber asli), bukan threshold arbitrer — sesuai prinsip kerja Meong.

## 1. Ringkasan peta metode → komponen SFC

| # | Metode | Disiplin | Komponen SFC | Sumber |
|---|--------|----------|--------------|--------|
| 1 | OFR Financial Stress Index (weight co-movement) | Statistik/Finansial | composite_confidence, bobot GLF/SLI/MPI | Monin 2019, OFRwp-17-04 |
| 2 | CISS (Composite Indicator of Systemic Stress, korelasi silang time-varying) | Statistik/Finansial | composite_confidence, method_agreement | Hollo, Kremer, Lo Duca 2012, ECB WP1426 |
| 3 | Hakkio & Keeton (PCA faktor laten) | Ekonometrika | macro_confidence, agregasi 31 method | Hakkio & Keeton 2009, Kansas City Fed |
| 4 | Illing & Liu (variance-equal/factor weight) | Ekonometrika | FSI agregasi | Illing & Liu 2006, J. Financial Stability |
| 5 | Hamilton regime-switching / HMM | Ekonometrika | HMM regime (BULL/BEAR/SIDEWAYS/CRISIS) | Hamilton 1989, J. Econometrics 39 |
| 6 | Ang & Bekaert (jumlah state & stabilitas HMM) | Finansial | jumlah state regime | Ang & Bekaert 2002 |
| 7 | Extreme Value Theory (GEV/GPD/POT) | Statistik ekstrem | m11_var, m12_jump, prob_crash, es_95 | Pickands 1975; Balkema–de Haan 1974 |
| 8 | Proper scoring rules (Brier + dekomposisi CAL+REF) | Statistik | gate validasi faktor; walk_forward_xgboost | Murphy 1973; scikit-learn |
| 9 | Bayesian Model Averaging / Ensemble BMA | Statistik Bayes | bobot ensemble (pengganti bobot 1:1) | Raftery et al 2005, MWR |
| 10 | Isotonic / Platt scaling | ML kalibrasi | kalibrasi probabilitas prediksi | Niculescu-Mizil & Caruana 2005 |
| 11 | Copula & tail dependence | Statistik | dependensi ekor antar-indikator | Joe 1997; Schmidt & Stadtmüller 2006 |
| 12 | Random Matrix Theory (Marchenko–Pastur) | Statistik | bersihkan noise korelasi 31 method, cegah double-count | Bun, Bouchaud, Potters 2016, arXiv:1610.08104 |
| 13 | Purged / Combinatorial Purged CV | ML validasi | ganti walk-forward single-path → CI performa | López de Prado 2018, "Advances in Financial ML" |
| 14 | Recursive Least Squares / online gradient | Optimasi/Adaptif | EWMA state adaptif | IJAES 37836; Preprints 202104.0601 |
| 15 | Stock-Flow Consistent (Godley & Lavoie) | Ekonomi Makro | rancang GLF/SLI sebagai sistem stok-arus | Godley & Lavoie 2007; Levy wp_891 |
| 16 | Minsky Financial Fragility Index | Ekonomi Makro | m25_minsky_stage (hedge/speculative/Ponzi) | Levy wp_654; Tymoigne |
| 17 | IMF Financial Stress Index (text-analysis) | Ekonomi Makro | agregasi stres alternatif | Ahir et al 2023, IMF WP/23/217 |

## 2. Detail metode kunci yang paling berdampak

### 2.1 Agregasi komposit berbasis co-movement (CISS / OFR)
Masalah sekarang: bobot manual (`execution_risk = 0.40×cascade + 0.30×squeeze + 0.30×funding`),
bobot GLF/SLI/MPI ditebak. Akademis: bobot turun dari data.

- **CISS (Hollo et al 2012)**: bobot agregasi = fungsi korelasi silang antar-subindeks
  yang berubah terhadap waktu. Saat sub-indeks saling memperkuat (korelasi naik),
  sinyal stres sistemik naik; saat tersebar, indeks "meler" (tidak over-reaksi).
  Implementasi inti:
  ```
  C_t = sum_i w_i * (1 - rho_t(i,j)) ... atau bobot korelasi-berat
  ```
- **OFR FSI (Monin 2019)**: z-score tiap indikator; bobot diturunkan dari kovariansi
  (variabel yang co-move saat stress dapat bobot lebih). Menghindari "equal-weight noise".

Integrasi ke SFC:
- Ganti `composite_confidence = macro × (1−exec_risk)` dengan versi yang menambahkan
  istilah korelasi: jika `execution_risk` DAN `macro` naik bersamaan → confidence
  diturunkan ekstra (stres menyebar), bukan hanya perkalian linear.
- Bobot GLF/SLI/MPI diturunkan dari PCA/kovariansi historis, dengan purged-CV
  (metode 13) sebagai validasi agar tidak overfit.

### 2.2 Tail risk berbasis EVT (m11_var, prob_crash)
Masalah: `var_95`/`es_95` berbasis normal underestimate ekor.
EVT (Peaks-Over-Threshold): pasang Generalized Pareto pada ekses di atas threshold u
(kuantil ~90-95%), lalu:
```
xi, beta = fit GPD(excess)
VaR_q = u + (beta/xi) * ((n/N_u * (1-q))^(-xi) - 1)
ES_q  = VaR_q + (beta - xi*u)/(1-xi)
```
Dipakai saat vol ekstrem; fallback ke normal saat data tenang.

### 2.3 Gate kalibrasi (proper scoring rules)
Ganti "confidence heuristic" dengan metrik proper:
- Brier score + dekomposisi `Brier = Reliability + Resolution − Uncertainty` (Murphy).
- Penerimaan faktor baru HANYA jika Resolution > 0 (ada daya diskriminasi) dan
  Reliability rendah (kurva kalibrasi dekat diagonal) — diverifikasi dengan
  walk-forward (metode 13). Ini mencegah faktor tak-terkalibrasi masuk ke sinyal.

### 2.4 Validasi purged-CV (pengganti single-path WFV)
WFV sekarang single-path (satu urutan fold). Purged k-fold / Combinatorial Purged CV
menghasilkan banyak path backtest → CI untuk Sharpe/AUC, bukan satu angka.
Untuk target yang labelnya bergantung masa depan (drop-6h XGBoost), WAJIB purge
(beri embargo antar train/test) agar tidak bocor look-ahead.

## 3. Yang SUDAH dipakai & terbukti benar di SFC
- HMM regime (metode 5) — sudah, display-only.
- Walk-forward validation (metode 13 versi single-path) — sudah, dipakai untuk
  menolak China M2, JPY carry, HY spread, dan XGBoost (verdict STAY_DISABLED).
- Brier score (metode 8) — baru dipakai di analysis/walk_forward_xgboost.py.
- EWMA online (metode 14) — sudah di models/ewma_state.json.
- Minsky stage (metode 16) — sudah di m25_minsky_stage.

## 4. Referensi URL (verifikasi 2026-08-07)
- SFC model (Godley & Lavoie): https://en.wikipedia.org/wiki/Stock-Flow_consistent_model ;
  Levy survey https://www.levyinstitute.org/pubs/wp_891.pdf ; Wiley joes.12221
- OFR FSI: https://www.financialresearch.gov/financial-stress-index/ ; paper OFRwp-17-04
- IMF FSI (Ahir et al): https://www.imf.org/en/publications/wp/issues/2023/10/18/
  financial-stress-and-economic-activity-evidence-from-a-new-worldwide-index-540713
- EVT: https://kth.diva-portal.org/smash/get/diva2:1996735/FULLTEXT01.pdf ;
  https://www.mdpi.com/1099-4300/22/12/1425
- Brier/proper scoring: https://en.wikipedia.org/wiki/Brier_score ;
  https://scikit-learn.org/stable/modules/calibration.html
- BMA: https://sites.stat.washington.edu/people/raftery/Research/PDF/fadoua.pdf
- Purged CV: https://en.wikipedia.org/wiki/Purged_cross-validation ;
  https://reasonabledeviations.com/notes/adv_fin_ml/
- HMM (Hamilton): https://quantdecoded.com/en/regime-switching-models-detecting-market-regimes
- Random Matrix: https://arxiv.org/abs/1610.08104

Sketsa implementasi referensi ada di `analysis/sfc_methods_academic.py`.
