# EKSTRAKSI LOGIKA MATEMATIS EKSAK — SFC ML/Regime/Safety

Semua konstanta dikutip literal dari kode. Bahasa: Indonesia. `#baris` = nomor baris di file sumber.

---

## 1. `models/hmm_regime.py` — Gaussian HMM Regime Detection

**Fungsi utama:** `HMMRegimeDetector.fit()`, `HMMRegimeDetector.predict()`, `fit_from_git()`, `_build_feature_matrix()`

### Konfigurasi
- `N_FEATURES = 5` (L62), `n_regimes = 4` (default, L85)
- Label regime: `{0:BULL, 1:BEAR, 2:SIDEWAYS, 3:CRISIS}` (L39-40), diurutkan menurun berdasar **mean daily_return** (kolom 0)

### Input features (L43-49, 8-9)
Fitur mentah: `daily_return(=btc_24h, %per hari)`, `dvol`, `m2_yoy`, `rsi_14`, `fng`
- Normalisasi per fitur: `dvol/100`, `m2_yoy/15`, `rsi_14/100`, `fng/100` (L361-364)
- **Alasan pembagi `m2_yoy/15`** (bukan /100): YoY % tipikal -5%..+15%, /100 akan memampatkan (L342-347)
- Default saat field hilang: `btc_24h→0.0`, `dvol→0.0`, `m2_yoy→5.0`, `rsi_14→50.0`, `fng→50.0` (L360-364)

### Fit — Preprocessing + EM/Baum-Welch (L121-138)
1. `nan_to_num(nan=0, posinf=0, neginf=0)` (L121)
2. **Z-score standarisasi:**
   - `_scaler_mean = mean(features, axis=0)` (L124)
   - `_scaler_std = std(features, axis=0)` (L125)
   - `_scaler_std[_scaler_std < 1e-8] = 1.0` (anti div-by-zero) (L126)
   - `features_std = (features - mean) / std` (L127)
3. **Model GaussianHMM:** `covariance_type="diag"`, `n_iter=1000`, `random_state=42`, `tol=1e-4` (L130-137)

### Labeling regime (L141-148)
- `daily_return_means = means[:, 0]`
- `sorted_indices = argsort(daily_return_means)[::-1]` (descending)
- `_state_order[state_idx] = label` → state HMM vol terendah↔regime BULL(label0), tertinggi↔CRISIS(label3)
- Minimum sampel: `n_samples < n_regimes*10` → warning (L115); `fit_from_git` butuh ≥50 snapshots (L257,264)

### Predict — Posterior Bayesian (L202-216)
1. `log_likelihoods = model._compute_log_likelihood(features)[0]` (emission)
2. **Stationary prior** dari transmat via eigendecomposition (L206-208):
   `eigvecs = eig(transmat_.T)`, `stationary = real(eigvecs[:, isclose(eigvals,1.0)][:,0])`, `stationary /= stationary.sum()`
   - fallback (gagal): `stationary = ones(4)/4` (L210)
3. **Posterior:**
   - `log_posterior = log(stationary + 1e-15) + log_likelihoods` (L213)
   - `log_posterior -= log_posterior.max()` (stabilitas numerik)
   - `posterior = exp(log_posterior)`, `posterior /= posterior.sum()` (L215-216)
4. `state = argmax(posterior)`, `regime_label = _state_order[state]`
5. **`crisis_probability = regime_probs[3]`** (prob massa label CRISIS), di-round 4 desimal (L230,235)

### Output
- `regime` (str), `regime_label` (0-3), `crisis_probability` (0-1, round 4dp), `state_probs` (list 4 prob, di-reorder ke urutan label)
- On error / not fitted → `{"regime":"NORMAL","crisis_probability":0.0}` (L184,190,242)

---

## 2. `ml/feature_engineering.py` — Teknikal Features (17 quality features)

**Fungsi utama:** `get_features()`, `_compute_features()`. `CACHE_TTL = 300` dtk.

### Helper normalisasi (L35-70)
- `_normalize_01(v, lo=0, hi=100)` → `clip(v,lo,hi)` lalu `(v-lo)/(hi-lo)` ∈ [0,1]; fallback 0.5 jika `hi-lo<1e-12`
- `_normalize_n11(v, bound=100)` → `clip(v,-bound,bound)/bound` ∈ [-1,1]
- `_symmetric_bound(series, percentile=95)` → `percentile(|series|,95) + 1e-12`; empty→100.0
- `_normalize_series_last(s)` → min-max pada **nilai terakhir**: `(last-min)/(max-min)` ∈ [0,1]; fallback 0.5 jika <2 sampel

### Indikator & rumus literal (L145-215)
| Fitur | Formula | Rentang |
|---|---|---|
| `rsi_14` | `_normalize_01(RSI(close,14))` | [0,1] |
| `macd_line` | `_normalize_n11(MACD, _symmetric_bound(macd))`, windows `slow=26,fast=12,sign=9` | [-1,1] |
| `macd_signal` | sama, `macd_signal` | [-1,1] |
| `macd_histogram` | sama, `macd_diff` | [-1,1] |
| `ema21_price_ratio` | `clip((ema21/last_c - 0.8)/0.4, 0, 1)`; EMA windows 21 & `min(200,len)` | [0,1] |
| `ema_crossover` | `clip((ema21 - ema200)/last_c * 10.0, -1, 1)` | [-1,1] |
| `ema200_slope` | `slope=(ema200[-1]-ema200[-6])/ema200[-6]*100`; `clip((slope+2.0)/4.0,0,1)` (butuh ≥6 nilai) | [0,1] |
| `atr` | `clip(ATR(14)/last_c*20.0, 0, 1)` | [0,1] |
| `bb_width` | `clip(bollinger_wband()/last_c*20.0, 0, 1)`; windows `20, dev=2` | [0,1] |
| `bb_pct_b` | `_normalize_01(bollinger_pband())` | [0,1] |
| `realized_vol` | `std(returns.tail(30)) * sqrt(365)` lalu `/2.0` clip [0,1] → 100% ann=0.5, 200%=1.0 | [0,1] |
| `vwap` | `clip((last_c/VWAP(14) - 0.95)/0.1, 0, 1)` | [0,1] |
| `obv` | `_normalize_series_last(OBV)` | [0,1] |
| `cmf` | `clip(CMF(20), -1, 1)` | [-1,1] |

Semua fallback kegagalan = `0.0`; `ema200_slope`/`realized_vol` default `0.5` jika data kurang.

---

## 3. `ml/dynamic_feature_weighting.py` — Regime → Faktor Weights

**Fungsi utama:** `get_regime_weights`, `apply_dynamic_weights`, `get_feature_group_weights`, `get_sfc_effective_with_dynamic_weights`

### Matriks bobot 5-faktor (Lt,St,Rt,Ft,Sc) per regime (L44-80)
| Regime | Lt | St | Rt | Ft | Sc |
|---|---|---|---|---|---|
| BULL | 0.30 | 0.10 | 0.10 | 0.15 | 0.35 |
| BEAR | 0.25 | 0.20 | 0.20 | 0.25 | 0.10 |
| SIDEWAYS | 0.20 | 0.25 | 0.20 | 0.15 | 0.20 |
| CRISIS | 0.15 | 0.10 | 0.10 | 0.40 | 0.25 |
| NORMAL | 0.25 | 0.20 | 0.20 | 0.20 | 0.15 |
| STRESS | 0.25 | 0.15 | 0.15 | 0.30 | 0.15 |
| CAPITULATION | 0.15 | 0.10 | 0.10 | 0.40 | 0.25 |

### Matriks bobot grup-metode (liquidity,stablecoin,onchain,derivatives,technical) (L84-90)
- BULL: 0.45/0.30/0.10/0.10/0.05
- BEAR: 0.20/0.15/0.20/0.35/0.10
- SIDEWAYS: 0.25/0.20/0.25/0.15/0.15
- CRISIS: 0.20/0.10/0.10/0.45/0.15
- NORMAL: 0.35/0.20/0.20/0.15/0.10

### `apply_dynamic_weights(factors, regime, ...)` (L132-148)
- **Normalisasi [-3,+3] → [0,1] (stress-oriented, 1=max stress):**
  `norm[k] = clip((-factors[k] + 3.0)/6.0, 0, 1)` (L139)
- **z_score (berbobot):** `z_score = Σ_k norm[k]*weights[k]`; fallback `norm.get(k,0.5)*weights.get(k,0.20)` (L142)
- `z_score = clip(z_score, 0, 1)` (L145)
- Return: `(norm_dict, z_score, weights)`

### `get_sfc_effective_with_dynamic_weights` (L204-235) — penyesuaian SFC (0-100)
- **CRISIS/CAPITULATION:** jika `ft_weight > 0.30`: `excess = ft_weight - 0.20`; `adj = min(sfc*excess*0.3, 10.0)`; hasil `min(100, sfc+adj)` (L214-216)
- **BULL:** jika `sc_weight > 0.25`: `dampen = min(sfc*0.08, 5.0)`; hasil `max(0, sfc-dampen)` (L223-224)
- **BEAR:** `bear_amp = (Rt + Ft - 0.35)*sfc*0.2`; clamp `[0,8]`; jika `>0.5` → `min(100, sfc+bear_amp)` (L229-232)
- NORMAL/SIDEWAYS/STRESS: tidak berubah `(sfc, 0.0)`

---

## 4. `ml/dynamic_feature_selector.py` — Regime Feature Selection

**Fungsi utama:** `DynamicFeatureSelector.select`, `get_group_weights`, `get_regime_profile`, `filter_mamba_input`

### 7 grup fitur + importance per regime (0-1) (L39-160)
| Grup | BULL | BEAR | SIDEWAYS | CRISIS | NORMAL |
|---|---|---|---|---|---|
| global_liquidity | 0.95 | 0.70 | 0.60 | 0.90 | 0.80 |
| stablecoin | 0.90 | 0.50 | 0.60 | 0.70 | 0.70 |
| onchain | 0.50 | 0.80 | 0.90 | 0.40 | 0.65 |
| derivatives | 0.40 | 0.90 | 0.60 | 0.85 | 0.55 |
| volatility | 0.30 | 0.75 | 0.40 | 0.95 | 0.50 |
| technical | 0.25 | 0.30 | 0.50 | 0.20 | 0.40 |
| macro | 0.40 | 0.85 | 0.55 | 0.60 | 0.55 |

### Threshold seleksi per regime (L164-170)
- BULL: 0.35, BEAR: 0.50, SIDEWAYS: 0.45, CRISIS: 0.40, NORMAL: 0.45
- **Aturan seleksi:** grup masuk jika `importance[regime] >= threshold[regime]` (L207)

### `get_group_weights` (L239-249) — normalisasi bobot grup terpilih
`weight_k = importance_k / Σ importance`; round 3dp; jika `total==0` → uniform `1/n_groups`

### `filter_mamba_input` (L305-339)
- `active_groups = set(select(regime))`; selalu tambah `"__core__"`
- Key tak dikenal → **keep (konservatif)**; key dengan grup tak aktif → drop

---

## 5. `ml/sfc_advanced.py` — Advanced (Prioritas 2-6)

### 5a. `RegimeDetector` — Manual GMM/k-means + Markov (L24-164)
**fit(features):**
1. Butuh `n >= n_regimes*5` (L51)
2. **k-means:** `kmeans2(features, 4, minit='points', iter=50, seed=42)` (L55-56)
3. **Covariance per cluster:** `cov = np.cov(features[mask].T) + eye*n_features*1e-6` (regularisasi L65-66); cluster kosong → `eye*0.1` (L68)
4. **Urut regime by volatilitas** kolom fitur-0: `state_order = argsort(cluster_vols)` → vol terendah=BULL, tertinggi=CRISIS (L81)
5. **Transition matrix:** `transmat[i,j] += 1` untuk tiap transisi `labels[t-1]→labels[t]`; `row_sums==0 → 1.0`; `transmat /= row_sums` (L84-91)

**predict(features):** nearest-centroid via `argmin(‖f-c‖)` (L105-106); map via `state_order`

**get_regime_status (L114-149):**
- `stability = 1.0 - trans_probs[current_regime]` (prob self-loop)
- `crisis_probability = trans_probs['CRISIS']` (prob transisi ke CRISIS dari cluster raw saat ini)
- **Fix bug (L119-127):** gunakan `raw_current = state_order[regime_id]` agar indeks transmat cocok (dulu baca sel salah → crisis_prob=0.991 palsu)

**score_stress_boost (L151-164):**
- `crisis_probability > 0.3` → **+15**
- `regime == 'CRISIS'` → **+10**
- `regime == 'BEAR'` dan `stability < 0.5` → **+5**
- Total `boost` = 0..30 poin ditambahkan ke SFC

### 5b. `WalkForwardBacktest` (L171-271)
- `train_days=756`, `test_days=63` (L177)
- **Portfolio return per baris (L210-215):** `signal>0.5 → ret`; `signal<0.1 → 0.0`; else `ret*signal`
- `equity *= (1+portfolio_ret)`
- **Metrics (L236-270):**
  - `sharpe = sqrt(252) * mean(port_rets)/std(port_rets)` (0 jika std=0)
  - `max_dd = min((equity - cummax(equity))/cummax(equity))`
  - `win_rate = mean(port_rets > 0)`
  - `bh_return = prod(1+returns) - 1`; `strategy_return = equity[-1] - 1`
  - `exposure = mean(signals > 0.5)`
  - `signal_stability = 1.0 - min(signal_std, 1.0)`
  - **overfitting_risk:** `LOW` if `sharpe>1.5 and max_dd>-0.15`; `MEDIUM` if `sharpe>0.5`; else `HIGH`

### 5c. `UncertaintyQuantifier` — Platt + Bootstrap (L299-478)
**`_platt_fit` (Newton-Raphson, L314-364):**
- Model: `p = 1/(1+exp(-(a*scores + b)))` (L333)
- Prior: `prior0=max(1,count0)`, `prior1=max(1,count1)` (L327-328)
- p clip `[1e-10, 1-1e-10]`; hingga 100 iterasi
- Gradien: `da=Σ f(p-y)`, `db=Σ(p-y)`; Hessian: `w=p(1-p)`, `ha=Σf²w`, `hb=Σfw`, `hbb=Σw` (L337-344)
- Newton step via inverse `det = ha*hbb - hb²` (break jika `|det|<1e-15`); konvergensi jika `|da_new|,|db_new|<1e-5`
- `_platt_predict(scores) = 1/(1+exp(-(a·scores+b)))` (L369)

**`predict_with_uncertainty(scores, regime_info)` (L376-478):**
- `primary = scores[0]`; `n_feats = len(scores)`
- Point prediction: Platt (jika a/b ≠ 0) else `clip(primary,0,1)`
- **Dimensional bootstrap:** bobot `w[0]=2.0` (fitur utama double-weighted), rest uniform; `w /= w.sum()`; `n_boot = max(n_bootstrap, 50)` (L421); resample `choice(n_feats, replace=True)`; `boot_pred = dot(w_sub, calibrated)` (L424-432)
- `alpha = 1 - confidence_level` (default 0.9); `lower = percentile(boot_preds, 50*alpha)`, `upper = percentile(boot_preds, 100-50*alpha)`; `uncertainty = upper-lower` (L434-437)
- **`is_reliable = uncertainty < 0.30`** (L438)
- **Dynamic thresholds (L452-453):**
  - `calm_t = 0.30 * max(0.01, 1.0 - 2.0*P(crisis))`
  - `stress_t = 0.70 * max(0.25, 1.0 - 0.4*P(crisis))`
- **Action (L456-463):** `pred>stress_t and reliable→HIGH_CONFIDENCE_STRESS`; `pred<calm_t and reliable→HIGH_CONFIDENCE_CALM`; `uncertainty<0.15→MEDIUM_CONFIDENCE`; else `LOW_CONFIDENCE`
- **Safety override (L466-469):** `HIGH_CONFIDENCE_CALM` & `P(crisis)>0.5 → MEDIUM`; `(CALM|MEDIUM)` & `P(crisis)>0.85 → LOW`

### 5d. `AutoFeatureEngineer` (L485-627)
**generate(data_dict) per fitur (L518-533):**
- `v` (original)
- `v²` (kuadrat)
- `log(v)` jika `v>0`, `log(max(v,1e-10))`
- **Sigmoid:** `1/(1+exp(-v*5 + 2.5))` — konstanta gain 5.0, offset 2.5 (L532)
- **Ratio:** `btc_price/max(v,1e-10)` untuk tiap fitur positif (kecuali btc/price/btc_mcap) (L541)

**select(X,y,top_k=15) (L547-588):** pilih `argsort(mi)[-top_k:]` (MI terbesar)

**`_mutual_info(x,y,bins=10)` (L590-615):**
- `MI = Σ p_xy·log(p_xy/(p_x·p_y))` atas bins 10×10 (histogram via `digitize` pada percentiles)
- Normalisasi: `MI / log(bins)` ∈ [0,1]; butuh ≥10 sampel else 0.0

### 5e. Alternative data (L634-805)
- Google Trends **proksi FnG** (L658-665): `recession = crash = (100-fng)/100`, `inflation = fng/100`; `TRENDS_CACHE_TTL=3600`; fallback 0.5
- Reddit sentiment (L691-733): `bullish_words` 8 kata, `bearish_words` 8 kata; `score = Σ(+1 bull, -1 bear)` per judul; `sentiment = total_score/total_posts`; clip [-1,1]; label `BULLISH if >0.2`, `BEARISH if <-0.2`, else NEUTRAL; `REDDIT_CACHE_TTL=600`
- `cg_ath_dd = (current_price - ath)/max(ath,1)` (L797-798)

### 5f. `compute_all_advanced` (L812-870)
- Regime: perlu `>20` historical obs; matriks fitur ambil `feat_names[:5]` per obs (`float(obs.get(k,0.5))`)
- UQ: `UncertaintyQuantifier(n_bootstrap=50)`; `sfc_score = regime_boost/100.0` sebagai skor
- Fallback regime: `{"regime":"NORMAL","crisis_probability":0.0,"stability":0.9}`, boost 0

---

## 6. `data_sources/regime_consolidation.py` — Konsensus Regime (P0)

**Fungsi utama:** `consolidate_regime(...)`. `CACHE_TTL=300`.

### Peta severity (0-100) (L55-66)
- `_MAIN_MAP`: NORMAL=20, STRESS=55, CAPITULATION=85
- `_HMM_MAP`: BULL=20, SIDEWAYS=45, BEAR=60, CRISIS=85
- `_ADV_MAP`: BULL=20, SIDEWAYS=45, BEAR=60, CRISIS=85, NORMAL=20, STRESS=55, CAPITULATION=85 (gabungan ruang 4+3 regime)
- `_BEHAVIOR_STRESS`: PANIC=85, DISTRIBUTION=65, EUPHORIA=45, EXPANSION=20, ACCUMULATION=25

### Bucket (L69-74)
- `sev>=60 → STRESSED`; `sev>=35 → ELEVATED`; else `BULLISH`

### Konsensus (L139-150)
- `structural = [sev utk sumber selain behavior]`; `sev = max(structural)` (jika ada) else `sev = behavior_sev`
- **Modifier behavior** (hanya jika ada structural):
  - `behavior_sev>=60 and sev>=45 → sev = max(sev, 70)`
  - `behavior_sev<=30 and sev<=30 → sev = min(sev, 25)`
- `label = _bucket(sev)`

### Conflict/Agreement (L159-161)
- `matching = Σ(vote.bucket == consensus_bucket)`; `agreement = matching/len(votes)` (round 3dp)
- `conflict_sources` = sumber yang bucket-nya ≠ consensus bucket

---

## 7. `analysis/circuit_breaker.py` — Safety Guard

**Fungsi utama:** `CircuitBreaker.validate(data)`, `validate_output()`

### Konstanta (L32-39)
- `MAX_CONSECUTIVE_FAILURES = 5`
- `COOLDOWN_SECONDS = 3600` (1 jam)
- `MAX_SFC_JUMP_PP = 20.0` (pp; dikurangi dari 40)
- `JUMP_SENSITIVE_FIELDS = ["sfc_effective", "sfc_base"]`

### FIELD_RULES (min,max) — subset kunci (L44-111)
- sfc_effective/sfc_base: (0,100); composite_confidence/cascade_risk/transition_risk/signal_strength/readiness_score/kelly_*/method_agreement/hmm_crisis_prob/ml_ensemble_*: (0,1)
- m1-m6: (0,100); mtf_alignment_score: (-1,1); bt_sharpe: (0,10); bt_return: (-1,10); bt_max_dd: (-1,1)
- news_stress: (-100,100); news_sentiment: (-1,1); q10_*: (0,100); var_95/es_975/ci_90_*: (0,100); prob_*: (0,1); sharpe/sortino: (-10,10); dxy_btc_corr/m81-m85: (-1,1)

### Logika validate (L143-349)
1. **NaN/Inf sweep:** ganti dengan `_last_valid[key]` atau `0.0`; set `all_ok=False`
2. **Range:** clamp `val<lo→lo`, `val>hi→hi` (clamp = **bukan failure**, all_ok tetap True, L183-205); non-numeric/NaN/Inf = **invalid** (`all_ok=False`); `prob_quantiles` nested dict di-clamp ke [0,100] (L212-226)
3. **Jump detection:** `jump = |curr - prev|`; `jump > 20.0` (dan `prev != 0`) → warning + `all_ok=False`; **nilai TIDAK di-clamp** (L242-249)
4. **Consistency sfc (L266-295):**
   - `liq = liq_mod atau -5.0` (default asumsi max negatif)
   - `dw_adj = dw_sfc_adjustment atau 0.0`
   - `mid = sfc_base + liq + dw_adj` (regime boost +2pp dikecualikan — konservatif)
   - jika `xgb_blend_weight > 0` dan ada prediksi: `mid = (1-xgb_w)*mid + xgb_w*xgb_meta_prediction`
   - `min_expected = mid - 3.0` (margin EWMA/rounding)
   - `sfc_eff < min_expected → consistency_issue, all_ok=False`
   - `composite_confidence` di luar [0,1] → issue
5. **Trip (L308-320):** `if not all_ok: consecutive_failures+=1`; bila `>=5` → `_tripped=True`, `_tripped_at=now`, **return `({}, False, warnings)`** (output di-purge)
6. **Valid:** reset counter; simpan `_last_valid` utk semua FIELD_RULES + JUMP_SENSITIVE + `btc,btc_24h,fng,dvol,dom`
7. **Recovery (L334-343):** jika tripped & valid: butuh `elapsed >= 3600` utk reset; else return `({},False,..)` dgn sisa cooldown
8. **Persist state:** tiap `total_valid % 10 == 0` atau tripped

---

## 8. `data_sources/trend_strength.py` — Trend Strength Score (P2)

**Fungsi utama:** `compute_trend_strength(...)`. Bobot default: `momentum=0.40, alignment=0.35, structure=0.25`. `CACHE_TTL=300`.

### `_rsi_to_strength(rsi)` (L42-56) — peta RSI → [0,1]
| RSI | strength |
|---|---|
| <30 | 0.30 |
| ≤50 | `0.35 + 0.30*((rsi-30)/20)` |
| ≤65 | `0.65 + 0.30*((rsi-50)/15)` (sweet spot) |
| ≤80 | `0.95 - 0.35*((rsi-65)/15)` |
| >80 | 0.50 |
| None | 0.5 |

### `_macd_to_strength(macd, bb_width)` (L59-70)
- `base = 0.5 + 2.0*clip(macd,-0.1,0.1)/0.1*0.5` = `0.5 + 10*clip(macd,-0.1,0.1)` (≈±0.5), clip [0,1]
- `base *= (1.0 - 0.25*clip(bb_width/0.01, 0, 1))` (BB lebar kurangi confidence)

### `_alignment_to_strength(mtf_alignment, dfs_regime)` (L73-86)
- `align = clip(0.5 + mtf_alignment*0.5, 0, 1)` (dari -1..+1 → 0..1)
- DFS: BULL → `+0.10`; BEAR/CRISIS → `-0.15`; SIDEWAYS → 0

### `_structure_to_strength(hmm_regime, hmm_crisis_prob)` (L89-107)
- BULL=0.85, BEAR=0.25, SIDEWAYS=0.50, CRISIS=0.15, else=0.50
- `base -= clip(crisis_prob)*0.5`

### Kombinasi (L130-164)
- Setiap domain ada jika inputnya ada; missing domain **dikeluarkan**, bobotnya **didistribusikan** ulang
- Enhancer momentum: `dom[1] = 0.6*dom[1] + 0.4*macd_strength` (jika macd_signal ada); `dom[1] = clip(dom[1] + clip(obv_norm,-1,1)*0.1)` (jika obv ada)
- `raw = Σ(strength_i*weight_i) / Σ weights`; `score = round(clip(raw)*100, 1)`
- Label: `>=65 STRONG`, `>=45 MODERATE`, `>=25 WEAK`, else BROKEN
- Tanpa input: score=50, `available=False`

---

## 9. `data_sources/trend_continuation.py` — Continuation Prob (P3)

**Fungsi utama:** `compute_trend_continuation(sfc_effective, sfc_zone)`. `CACHE_TTL=900`.

- `HORIZONS = [30, 90, 180]` (L28)
- `BUCKET_EDGES = [(0,25,"CALM"), (25,45,"ELEVATED"), (45,101,"STRESS")]` (L30)
- **Bucket mapping (L93-102):** `NORMAL/CALM→CALM`; `HIGH/CRITICAL/STRESS/BEAR/CRISIS→STRESS`; else `ELEVATED`; tanpa zone → `_bucket_label(sfc_effective)` (per BUCKET_EDGES)
- Prob per horizon dibaca dari summary: `{bucket}_p_cont_{h}d`, CI, n, baseline
- `relative = round(p - baseline, 3)` (L121)
- **Bukan formula mandiri** — membaca cache walk-forward (`.trend_continuation_summary.json`)

---

## 10. `data_sources/transmission_divergence.py` — Liquidity→BTC Transmission (P1)

**Fungsi utama:** `classify_transmission(...)`. `CACHE_TTL=300`.

### Normalisasi (L50-60)
`_norm(v, is_pct)`: jika `is_pct or v>1.0` → `clip(v/100,0,1)`; else `clip(v,0,1)`

### Quadrant (L93-127) — `liq = _norm(liquidity_stress)`, `struct = _norm(structural_stress, is_pct=True)`
| Kondisi | Status |
|---|---|
| `liquid` (liq<0.40) & `btc_strong` (struct<0.40) | STRENGTHENING |
| `liquid` & `btc_weak` (struct>=0.60) | TRANSMISSION_GAP |
| `illiquid` (liq>=0.60) & `btc_strong` | DIVERGENCE |
| `illiquid` & `btc_weak` | DISTRESS_CASCADE |
| else (mid) | TRANSMITTING |

### Confidence (L130-136)
- `edges = int(liquid or illiquid) + int(btc_strong or btc_weak)` (0-2)
- `confidence = 0.5 + 0.25*edges`
- Koreksi tanda btc: `+0.10` jika `(sign>0 and not btc_weak)`, `-0.10` jika `(sign<0 and not btc_strong)`; clip `<=0.95`

---

## 11. `data_sources/behavior_state.py` — Behavior State Overlay (L5)

**Fungsi utama:** `compute_behavior_state(...)`. `CACHE_TTL=300`. Rule-based, **first match wins**.

### Evidence thresholds (L130-171)
- **MPI:** `>=75 mpi_crowded_long`(bull); `>=60 mpi_bullish`(bull); `<=25 mpi_bearish`(bear); `<=40 mpi_weak`(bear)
- **FNG:** `>=85 fng_extreme_greed`(bull); `>=70 fng_greed`(bull); `<=15 fng_extreme_fear`(bear); `<=25 fng_fear`(bear)
- **Cascade:** `>=0.5 cascade_high`(bear); `>=0.25 cascade_elevated`(bear)
- **ETF:** `>=0.65 etf_inflow`(bull); `<=0.35 etf_outflow`(bear)
- **Whale:** `>=0.65 whale_buying`(bull); `<=0.35 whale_selling`(bear)
- **HMM:** BEAR/CRISIS→bear; BULL→bull

### Rule priority (L175-201)
1. `PANIC` jika `cascade_high` atau HMM=CRISIS
2. `EUPHORIA` jika `(fng_extreme_greed|mpi_crowded_long)` & `div != HIDDEN_DISTRIBUTION`
3. `DISTRIBUTION` jika `div==HIDDEN_DISTRIBUTION` atau `(etf_outflow & mpi_bullish)` atau `(whale_selling & fng_greed)`
4. `ACCUMULATION` jika `div==HIDDEN_ACCUMULATION` atau `(etf_inflow & (fng_fear|fng_extreme_fear))` atau `(whale_buying & (fng_fear|fng_extreme_fear))`
5. `PANIC` (sekunder) jika `len(bear_ev)>=3` & `not bull_ev`
6. `EXPANSION` jika `len(bull_ev)>=1` & `len(bear_ev)<=1`
7. fallback: `ACCUMULATION` if `len(bull)>len(bear)`; `DISTRIBUTION` if `len(bear)>len(bull)`; else `EXPANSION`

---

## 12. `data_sources/tail_risk_engine.py` — Tail Risk (L8)

**Fungsi utama:** `compute_tail_risk(liquidity, behavior, expectation, leverage, correlation)`. `CACHE_TTL=900`.

### Normalisasi `_norm` (L60-72)
- `None → 50.0` (netral, tidak meniadakan produk)
- `v>1.0 → clip(v,0,100)`; else `clip(v*100,0,100)`

### Formula kombinasi (L160-173)
- **Geometric mean:** `prod = Π(dims[k]/100)`; `geometric = (prod^(1/N))*100` (N=5 dimensi)
- **Amplifier leverage×correlation:**
  - `amp = 1.0 + 0.25*(lev/100)*(corr/100)*(lev>=60 and corr>=60)`
  - (boolean `(lev>=60 and corr>=60)` bernilai 1/0)
  - `score = min(100, geometric*amp)`
- Tanpa input → 50.0, `available=False`

### Severity (L75-85): `>=80 CRITICAL`; `>=60 HIGH`; `>=40 ELEVATED`; `>=25 MODERATE`; else LOW

---

## 13. `data_sources/expectations_engine.py` — Expectation Gap (L6)

**Fungsi utama:** `compute_expectations()`. `CACHE_TTL=21600` (6 jam).

### Komponen (L173-221)
- **Infl gap (signed):** `infl_gap = T10YIE - cpi_yoy` (breakeven inflation − CPI YoY) (L179)
- **Real rate:** `real_rate = DGS10 - T10YIE` (L189)
- **Curve:** `curve = T10Y2Y` (slope 10y-2y)
- **Unemployment:** `u = UNRATE`, `u_trend = u[0]-u[1]` (delta 1 bulan)
- CPI YoY dihitung dengan date-matching bulan T vs T-12 (L104-121)

### `_slope_to_stress(infl_gap)` (L149-161)
| gap (pp) | stress |
|---|---|
| ≤-2.5 | 85.0 |
| ≤-1.0 | 65.0 |
| ≤0.0 | 50.0 |
| ≤1.0 | 35.0 |
| ≤2.5 | 45.0 |
| ≤4.0 | 60.0 |
| >4.0 | 75.0 |

### Komponen stress lain (L217-221)
- **Real rate:** `<1.0→25`; `<2.0→50`; `<3.0→65`; else `80`
- **Curve:** `<0→80`; `<0.5→45`; else `30`
- **UNRATE:** `>=6.5→75`; `>=5.5→55`; `>=4.0→40`; else `30`

### Agregasi (L223-228)
`gap_score = round(mean(comps), 1)`; jika tak ada komponen → 50.0, `available=False`

### Label headline (L241-243): `expectation_gap < -1.0 → "DEFLATION-SURPRISE RISK"`; `>1.5 → "REFLATION-PRICING"`; else "BENIGN EXPECTATIONS"/UNAVAILABLE

---

## 14. `models/ml_ensemble.py` — Random Forest + Online Learning

**Fungsi utama:** `train_model`, `predict_with_ml`, `resolve_pending_labels`, `_weighted_average_fallback`, `retrain_on_errors`, `evaluate_accuracy`

### Labeling harga (L158-161)
- `LABEL_LOOKAHEAD_MINUTES = 360`; `LABEL_LOOKAHEAD_TOLERANCE_MINUTES = 60`
- `LABEL_STRESS_DROP_PCT = -3.0` (BTC turun ≥3% = stress)
- `LABEL_CALM_RISE_PCT = 1.5`
- Label: `pct_change <= -3.0 → 1.0` (stress); `pct_change >= -1.5 → 0.0` (calm); antara -3..-1.5 → pending (L234-247)
- Pencocokan harga: dalam `±120 dtk` (obs), `±tolerance*60` (target) (L222-229)

### `train_model` (L283-364)
- Butuh `>=30` labeled sampel
- `X = nan_to_num(nan=0.5, posinf=1.0, neginf=0.0)` (L312)
- **Split walk-forward:** `split = int(len(X)*0.80)`; 80% tertua=train, 20% terbaru=val (L315-317)
- **Normalisasi:** `StandardScaler` fit pada train
- **RandomForestClassifier:** `n_estimators=200, max_depth=15, min_samples_split=20, min_samples_leaf=10, class_weight='balanced', n_jobs=-1, random_state=42` (L325-333)
- Collection cap: `2000` observasi (L96)

### `predict_with_ml` (L381-416)
- `stress_prob = proba[1]` (prob class 1); `confidence = |stress_prob - 0.5| * 2` (0-1) (L408)

### `_weighted_average_fallback` (L419-451) — fallback jika model tak ada
- Bobot mentah: M1-M6 (i<6) → `0.12`; M7-M19 (6≤i<19) → `0.03`; M20+ (i≥19) → `0.02` (L424-431)
- Padding/trim ke `method_count`; `weights = raw/total_w`
- `weighted = Σ(v_i*w_i)` utk v bukan None
- `confidence = min(active/method_count, 1.0) * 0.8`

### `retrain_on_errors` (L454-488)
- `mismatches = Σ(round(pred) != label)`; butuh `total_labeled>=30` dan `mismatches>0`
- `accuracy = 1 - mismatches/total_labeled`

### `evaluate_accuracy` (L491-527): `accuracy = correct/total`; per-kelas acc stress/calm

### (Deprecated) `compute_actual_stress` (L105-145): voting threshold `dvol>80`, `sfc_pct>50`, `btc_24h<-5`, `news_stress>30`; label 1 jika `stress_signals >= max(2, total//2)`. **Deprecated — jangan dipakai untuk labeling** (sifatnya sirkular).

---

## 15. `models/online_learning.py` — EWMA + Kalman

**Fungsi utama:** `OnlineEWMA.update`, `AdaptiveKalman.update`, `correct_stress`

### `OnlineEWMA` (L27-57)
- Default `alpha=0.15`, `window=30`; validasi `0<alpha<=1`, `window>=1`
- **Update:** nilai pertama seed baseline; berikutnya:
  - `baseline = alpha*value + (1-alpha)*baseline` (L50)
- History dibatasi: jika `len > window*2` → simpan `[-window:]` (L54-55)

### `AdaptiveKalman` (L128-169) — 1D Kalman
- Default: `Q = process_noise = 0.01`, `R = measurement_noise = 0.1` (validasi >0)
- `x` (state) awal 0.0; `P` (error cov) awal 1.0
- Call pertama seed: `x = m`, `P = 1.0`
- **Predict:** `x_pred = x`; `P_pred = P + Q`
- **Update:**
  - `K = P_pred/(P_pred + R)` (Kalman gain)
  - `x = x_pred + K*(m - x_pred)` (posterior estimate)
  - `P = (1 - K)*P_pred` (posterior covariance)

### `correct_stress(raw_stress, confidence=1.0, transition_risk=0.0)` (L188-227)
1. `corrected = ewma.update(raw_stress)`
2. `blended = confidence*raw_stress + (1-confidence)*corrected` (confidence rendah → lebih percaya baseline)
3. jika `transition_risk>0`: `blended = (1-transition_risk)*blended + transition_risk*raw_stress`
4. `return clip(blended, 0, 1)`

---
