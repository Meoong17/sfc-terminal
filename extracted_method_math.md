# Logika Matematis Eksak — collect.py & methods_institutional.py

Ekstraksi rumus presisi dari `/home/ubuntu/sfc/collect.py` (4570 baris) dan
`/home/ubuntu/sfc/data_sources/methods_institutional.py` (1199 baris).
Semua konstanta literal dikutip dari kode. Bahasa: Indonesia.

> Konvensi skor: **semua metode M1–M80 menghasilkan skor stres 0–1 di mana
> NILAI TINGGI = LEBIH STRES/bearish**, kecuali bila dinyatakan lain. Return shape
> setiap metode = `(score, detail)`; `score=None` bila data tak tersedia (honest).

---

## BAGIAN A — collect.py

### A1. Metode M7–M19 (macro & method scores)

#### M7 `calculate_m7_fisher()` (baris 1347)
Fisher Real Rates.
- `real_rate = fed_rate − cpi_yoy`  (FEDFUNDS − CPI YoY prefetch)
- Threshold bertingkat (tinggi real_rate = longgar/ekonomi panas → lebih berisiko):
  | Kondisi | score |
  |---|---|
  | `real_rate > 3.0` | 0.85 |
  | `> 2.0` | 0.60 |
  | `> 0.5` | 0.35 |
  | `> 0` | 0.20 |
  | `> -1.0` | 0.10 |
  | else | 0.05 |
- Return `None` bila FEDFUNDS/CPIAUCSL kosong ATAU `cpi_yoy is None` (A4: **tanpa fallback 3.0**).

#### M8 `calculate_m8_yield_curve()` (baris 1365)
- `slope = DGS10 − DGS2`
- `spread = BAMLH0A0HYM2` (default literal `300` bila tak tersedia)
- Slope score:
  | Kondisi | slope_s |
  |---|---|
  | `slope < 0` | 0.80 |
  | `< 0.5` | 0.65 |
  | `< 1.0` | 0.40 |
  | `> 2.0` | 0.15 |
  | else | 0.25 |
- Credit score:
  | Kondisi | cred_s |
  |---|---|
  | `spread > 400` | 0.85 |
  | `> 300` | 0.65 |
  | `> 200` | 0.40 |
  | else | 0.15 |
- **FINAL:** `score = 0.60*slope_s + 0.40*cred_s`

#### M9 `calculate_m9_liquidity()` (baris 1386)
- `mult = M2SL / MBCURSL` (money multiplier)
  | Kondisi | score |
  |---|---|
  | `mult < 4.0` | 0.85 |
  | `< 5.0` | 0.65 |
  | `< 6.0` | 0.40 |
  | `> 10.0` | 0.20 |
  | else | 0.30 |

#### M10 `calculate_m10_garch(closes)` (baris 1404) — GARCH(1,1)
- `rets = diff(closes)/closes[:-1]`; butuh `len(closes)≥30`, `len(rets)≥20`
- Konstanta model: `omega=0.00001, alpha=0.05, beta=0.94`
- Iterasi: `sigma2_t = omega + alpha*resid[t-1]² + beta*sigma2[t-1]`; `curr_vol=sqrt(sigma2[-1])`
- `persist = alpha + beta = 0.99`
  | Kondisi | score |
  |---|---|
  | `persist>0.98 and curr_vol>0.03` | 0.85 |
  | `persist>0.95 and curr_vol>0.02` | 0.65 |
  | `curr_vol>0.02` | 0.45 |
  | `curr_vol>0.01` | 0.25 |
  | else | 0.10 |

#### M11 `calculate_m11_var(rets)` (baris 1427) — VaR + Expected Shortfall
- `var_95 = percentile(rets, 5)`; `es = mean(rets[rets<=var_95])` (fallback `=var_95`)
  | Kondisi (es) | score |
  |---|---|
  | `es < -0.15` | 0.85 |
  | `< -0.10` | 0.65 |
  | `< -0.05` | 0.45 |
  | `< 0` | 0.25 |
  | else | 0.10 |

#### M12 `calculate_m12_jump(ohlcv)` (baris 1442) — Merton jump
- `gap = |open[t] − close[t-1]| / close[t-1]`; hanya `gap > 0.02` dihitung
- `freq = len(gaps)/len(closes)`; `avg = mean(gaps)`; `jr = freq*avg`
  | Kondisi | score |
  |---|---|
  | `jr > 0.05` | 0.85 |
  | `> 0.03` | 0.65 |
  | `> 0.01` | 0.45 |
  | else | 0.15 |

#### M13 `calculate_m13_funding()` (baris 1463) — Deribit funding acceleration
- Ambil `interest_8h` dari Deribit (data[:8]): `fr_now=rates[0], fr_1=rates[1], fr_2=rates[2]`
- `accel = (fr_now − fr_1) − (fr_1 − fr_2)`
- `FR_CAP = 0.005` (cap Deribit 0.5%/8h)
  | Kondisi | score |
  |---|---|
  | `accel > FR_CAP*0.40` (=0.002) | 0.75 |
  | `fr_now > FR_CAP*0.70` (=0.0035) | 0.65 |
  | `fr_now > FR_CAP*0.30` (=0.0015) | 0.35 |
  | else | 0.15 |

#### M14 `calculate_m14_skew()` (baris 1493) — Deribit options skew
- `put_iv=mean(P-mark_iv)`, `call_iv=mean(C-mark_iv)`, `atm=mean(all mark_iv)`
- `skew = (put_iv − call_iv)/atm`
  | Kondisi | score |
  |---|---|
  | `skew > 0.20` | 0.80 |
  | `> 0.15` | 0.65 |
  | `> 0.10` | 0.45 |
  | `> 0.05` | 0.25 |
  | else | 0.10 |

#### M15 `calculate_m15_concentration()` (baris 1515) — OI HHI
- `shares = oi/total` (sorted desc); `hhi = Σ shares²`; `top3 = Σ shares[:3]`
  | Kondisi | score |
  |---|---|
  | `hhi > 0.25` | 0.80 |
  | `> 0.20` | 0.65 |
  | `> 0.15` | 0.45 |
  | `> 0.10` | 0.25 |
  | else | 0.10 |

#### M16 `calculate_m16_regime_switch(rets)` (baris 1537) — heuristic, return `p_crisis` LANGSUNG sebagai score
- `recent=rets[-30:]`; `r_mean=mean`, `r_std=std`
  | Kondisi | p_crisis (score) |
  |---|---|
  | `r_mean<-0.005 and r_std>0.03` | 0.75 |
  | `r_std>0.025` | 0.45 |
  | `r_mean>0.01` | 0.10 |
  | else | 0.20 |

#### M17 `calculate_m17_granger(series_x, series_y)` (baris 1550) — Pearson + lag-corr
- `dx=diff(x), dy=diff(y)`; `corr=corrcoef(dx,dy)`; `lag_corr=corrcoef(dx[:-1], dy[1:])`
  | Kondisi | score |
  |---|---|
  | `lag_corr>0.5 and abs(corr)<0.3` | 0.7 |
  | `abs(corr)>0.6` | 0.5 |
  | `lag_corr<-0.3 and abs(corr)>0.3` | 0.3 |
  | else | 0.1 |

#### M18 `calculate_m18_entropy(prices)` (baris 1570) — Shannon entropy
- `rets` dari 30 harga terakhir; `n_bins=10`
- `h=entropy(hist_norm)`; `h_max=ln(10)`; `hn = h/h_max`
  | Kondisi | score |
  |---|---|
  | `hn > 0.85` | 0.75 |
  | `> 0.75` | 0.55 |
  | `> 0.65` | 0.35 |
  | else | 0.15 |

#### M19 `calculate_m19_mutual_info(btc_rets, macro_rets)` (baris 1590)
- Discretize btc & macro ke 3 bin (tertile `percentile [33,67]`); `n_bins=3`
- `mi = Σ p_j·ln(p_j/(p_b·p_m))`
- `mi_norm = min(mi/ln(3), 1.0)`
  | Kondisi | score |
  |---|---|
  | `mi_norm > 0.70` | 0.75 |
  | `> 0.50` | 0.55 |
  | `> 0.30` | 0.35 |
  | else | 0.15 |

---

### A2. Metode M72–M75 (macro liquidity)

#### M72 `calculate_m72_m2_growth()` (baris 1726)
- `m2_yoy = (m2[0]−m2[12])/m2[12]*100` (YoY %, M2SL)
- Skor (tinggi = likuiditas tinggi = bullish/stress rendah):
  | Kondisi | score |
  |---|---|
  | `m2_yoy < 0` | `max(0.05, 0.3 + m2_yoy*0.03)` |
  | `< 5` | `0.3 + (m2_yoy/5)*0.4` |
  | `< 10` | `0.7 + min(0.2, (m2_yoy−5)*0.04)` |
  | else (≥10, overheating) | `0.9` |

#### M73 `calculate_m73_m2_momentum()` (baris 1761)
- `growth_3m=(m2[0]−m2[3])/m2[3]*100`; `growth_12m=(m2[0]−m2[12])/m2[12]*100`
- `momentum = growth_3m − growth_12m`
- **Logistik:** `score = 1/(1+exp(−1.5*(momentum−0.3)))`, clamp `[0.05,0.95]`
- Label: `>1`=ACCELERATING, `>0`=STEADY, `>−1`=DECELERATING, else CONTRACTING

#### M74 `calculate_m74_fed_balance()` (baris 1799)
- `fed_yoy = (WALCL[0]−WALCL[12])/WALCL[12]*100`
  | Kondisi | score | label |
  |---|---|---|
  | `fed_yoy > 10` | 0.1 | EXPANDING |
  | `> 2` | 0.2 | EXPANDING |
  | `> −2` | 0.4 | STABLE |
  | `> −5` | 0.6 | MILD_QT |
  | else | 0.8 | AGGRESSIVE_QT |

#### M75 `calculate_m75_liquidity_composite()` (baris 1848)
- Bobot tetap: `M72=0.30, M73=0.30, M74=0.40`
- `composite = Σ(s·w)/Σw` (renormalisasi bila ada metode None)
- Regime label: `<0.2`=EXPANSIVE, `<0.4`=ACCOMMODATIVE, `<0.6`=NEUTRAL, `<0.8`=TIGHTENING, else CONTRACTIVE

#### `_macro_active` (baris 2497)
- `_macro_active = sum(1 for x in [_m72_score,_m73_score,_m74_score,_m75_score] if x is not None)` → rentang **0–4**

---

### A3. Causal Filter & Ensemble Blend (baris 2500–2649)

- `CausalFilter(max_lag=3)`; `causal_weights = get_weights()`; `causal_adjustment = get_blend_adjustment()` (dari modul CausalFilter, bukan hardcoded).
- **Fallback hardcoded bila causal tak tersedia:** `m1_m6_pct=85.0, m7_m19_pct=10.0, m20_m31_pct=5.0`
- `method_scores_dict` diisi dengan `None→0.5`.
- `m5_qreg = m5_qreg/100`, `m6_regime = m6_regime/100` (dikonversi ke 0–1) untuk dict; XGB memakai `m5_qreg` dan `m6_regime_score` tanpa konversi (sudah 0–100).
- `apply_filter(method_scores_dict)` → `filtered_scores, causal_active`

**Aktivasi per grup** — hanya metode dengan `causal_weights.get(name, 0.5) >= 0.2` dihitung:
- `m1m6_active`, `m7m19_active`, `m20m31_active` (filter threshold literal **0.2**)
- `new_active = len(m7m19_active)` (rentang 0–13)
- `inst_active_count = len(m20m31_active)` (rentang 0–12)

**Rata-rata grup** (avg hanya dari metode aktif; fallback bila kosong):
- `m1m6_avg = mean(m1m6_active)` else `sum(m1m6_scores)/6`
- `new_avg = mean(m7m19_active)` else `0.5`
- `inst_avg_value = mean(m20m31_active)` else `0.5`

**Bobot blend dinamis:**
- `causal_p1 = m1_m6_pct/100` (default 0.85), `causal_p2` (0.10), `causal_p3` (0.05)
- `active_total = p1+p2+p3`
- `p1 = causal_p1/active_total`
- `p2 = causal_p2/active_total` **hanya jika `new_active>0`**, else `0.0`
- `p3 = causal_p3/active_total` **hanya jika `inst_active_count>0`**, else `0.0`
- **Redistribusi:** `unused = 1−p1−p2−p3; p1 += unused` (sisa bobot dipindah ke M1–M6 core)
- Fallback penuh: `p1,p2,p3 = 1.0, 0.0, 0.0`

**Ensemble awal:**
```
sfc_pct = (p1*m1m6_avg + p2*new_avg + p3*inst_avg_value) * 100
```
(baris 2641; ini `sfc_pct` awal sebelum XGB/dynamic weight/QLSTM/Mamba.)

---

### A4. XGBoost Meta-Ensemble Blend (baris 2577–2604, 3097–3104)

- `_xgb_method_scores`: M1–M6 dikalikan `*100` (kecuali `m5_qreg`,`m6_regime_score` sudah 0–100), M7–M19 sudah 0–1, M20–M31 dari `inst_results` (None→0.5).
- `_xgb_result = _xgb_module.predict_ensemble(_xgb_method_scores)` → `_xgb_pred['stress']` (%), `_xgb_confidence` (default 0.5).

**Blend (baris 3098–3104):** hanya bila `_xgb_pred` tersedia DAN `_xgb_confidence > 0.3`
```
_xgb_blend_weight = 0.3 * _xgb_confidence        # 0–30%
effective_sfc = (1 − _xgb_blend_weight)*effective_sfc + _xgb_blend_weight*_xgb_pred
effective_sfc = clamp[0,100]
```

---

### A5. Mamba nudge (baris 2666–2729)

- `_mamba_data` = 30+ fitur (btc, chg, mcap, dom, dvol, rsi_14, pc_oi, pc_vol, fng, zone, regime, sfc_base, sfc_effective, m2_yoy, dxy, method_agreement, composite_confidence, m1–m5 scores, factors, sopr_proxy, cascade_risk, liq_density, liq_mod, regime_prob, transition_risk + q10 whale 4 fitur).
- DFS `filter_mamba_input(regime, data)` bisa drop fitur.
- `mamba_pred = mamba_result['combined']` (0–1); `mamba_sfc = mamba_pred*100`.
- **Status kini DISABLED:** `mamba_adjustment = 0` (komentar: SSM collapse → output semua nol), `mamba_ok=False`. Tidak ada perubahan `sfc_pct` dari Mamba.

---

### A6. `_REGIME_DRIVER_MULT` & `_regime_consensus_label` (baris 3006–3047, 3947–3970)

- `consolidate_regime(regime, regime_prob, hmm_regime, hmm_crisis_prob, adv_regime, adv_crisis_prob, behavior_state=None)` → `(_regime_consensus_label, _regime_consensus_details)`.
- `behavior_state` sengaja `None` sebagai scoring driver (display-only L5 overlay).
- `_rc_sev = _regime_consensus_details.get("severity")` (0–100).

**`_REGIME_DRIVER_MULT` (baris 3038–3047):**
```
if _rc_sev is None:         mult = 1.0
elif _rc_sev >= 60:         mult = 0.7   # STRESSED → threshold dikecilkan
elif _rc_sev < 35:          mult = 1.2   # BULLISH → threshold dinaikkan
else:                       mult = 1.0
```

**Zone (digunakan berulang, baris 3068/3092/3103/3120/3135):**
```
zone = CRITICAL jika effective_sfc/100 > 0.75*MULT
       HIGH    jika              > 0.50*MULT
       ELEVATED jika             > 0.25*MULT
       else NORMAL
```

**Single-driver guard adv boost (baris 3081–3095):** boost `adv_regime_boost` hanya diterapkan bila `_rc_sev >= 45`; jika DW aktif, boost dicap `min(boost, 2.0)`.
`effective_sfc = min(effective_sfc + boost, 100.0)`.

**Late P0 block (baris 3947–3970):** `consolidate_regime` dipanggil ulang dengan `behavior_state` + `structural_stress=effective_sfc` untuk label display.

---

### A7. Dynamic Weighting (baris 3049–3075)

- `apply_dynamic_weights(factors, regime)` → norm_factors, z_score, weights
- `get_sfc_effective_with_dynamic_weights(factors, effective_sfc, regime)` → `_dw_adjusted_sfc`, `_dw_sfc_adjustment`; bila `!=0`, `effective_sfc = clamp[0,100]`. (Formula internal di modul terpisah; kode hanya mencatat "DW adjusts +0.9pp CRISIS".)

---

### A8. Macro Confidence & Yield Curve Adjustment (baris 3403–3500)

**Penalti Layer-1 (additif):**
| Variabel | Nilai |
|---|---|
| `_pen_rsi` | `0.08` jika rsi<25 atau >75; `0.04` jika <35 atau >65; else 0 |
| `_pen_sopr` | `0.05` jika `sopr_proxy<0.97` |
| `_pen_fng` | `0.06` jika fng<15; `0.04` jika fng>85 |
| `_pen_news` | `0.04` jika news<-0.5; `0.02` jika news<-0.3 |
| `_pen_dvol_safety` | `0.05` jika dvol>80 |
| `_pen_transition` | `0.05` jika transition_risk>0.5 |
| `_pen_mpi` | `max(0,(mpi_stress−0.5)*0.08)` |

**Yield curve adj (dari `m8_d`):** slope<0 → `_pen_yield+=0.08`; slope<0.5 → `+=0.04`; slope>2.0 → `_boost_yield=0.03`. Spread>400 → `+=0.06`; >300 → `+=0.03`.

**Layer-2 Execution Risk (multiplikatif):**
- `_squeeze_magnitude = |liq_long−liq_short|/(liq_long+liq_short) * liq_density` (fallback: `rsi_ext*liq_density`)
- `_imb_funding = |liq_long−liq_short|/(liq_long+liq_short)` atau `min(abs(fr)*10, 1.0)` dari m13
- **`_execution_risk = min(0.40*cascade_risk + 0.30*squeeze + 0.30*imb, 0.95)`**

**Confidence (Layer-1 → komposit):**
```
cc_base = 0.30 + method_agreement*0.15 + max(0, 1−(effective_sfc/100))*0.08
macro_confidence = clamp(cc_base + _boost_yield − Σpenalti, 0.05, 0.95)
composite_confidence = clamp(macro_confidence * (1 − _execution_risk), 0.05, 0.95)
```

---

### A9. Kelly, Signal Type, Zone & Breakeven (baris 3502–3513, 4361–4391)

**Kelly override (baris 3505–3513):**
```
if transition_risk > 0.60: _kelly_override = 0.0   (CASH)
elif transition_risk > 0.50: _kelly_override = 0.5  (half)
else: _kelly_override = 1.0
```

**Kelly (baris 4364–4366):**
```
kelly_fraction = max(0, (p*2.0 − (1−p))/2.0) * _kelly_override   # p = composite_confidence, b=2.0
kelly_half     = max(0, (p*2.0 − (1−p))/4.0) * _kelly_override
kelly_quarter  = max(0, (p*2.0 − (1−p))/8.0) * _kelly_override
```
Rumus Kelly standar `(b·p − q)/b` dengan `b=2.0 (risk/reward)`, `q=1−p`.

**`signal_decision` (baris 4373–4378)** — mirror sigClass frontend:
- **CASH** jika `kelly_fraction_raw*_kelly_override <= 0` ATAU `effective_sfc >= 50*MULT`
- **BUY** jika `kelly>0` DAN `effective_sfc < 25*MULT`
- **else WATCH**
- Threshold regime-adjusted: buy `25*MULT`, cash `50*MULT`.

**`signal_type` (baris 4381):**
```
STRESS_TRANSITION jika transition_risk > 0.60
STRESS  jika effective_sfc > 25*MULT
else CALM
```

**Trigger 'confidence below edge breakeven' → CASH:** terjadi saat
`composite_confidence` cukup rendah sehingga `(p*2−(1−p))/2 ≤ 0`, yaitu
`3p−1 ≤ 0` → **`p ≤ 1/3`** (composite_confidence ≤ 0.333...). Karena override
bisa 0, kondisi CASH juga terpenuhi saat `_kelly_override==0` (transition_risk>0.60).

**Lain-lain:**
- `signal_strength = min(effective_sfc/50, 1.0)`
- `alert_window_days = 7 + 30*(1−composite_confidence)`
- `readiness_score = composite_confidence*(1 − min(effective_sfc/100, 0.5))`
- `timing_precision`: LOW jika conf<0.3; MEDIUM <0.6; else HIGH

---

### A10. `_sopr_signal_score(value)` (baris 885–902)

| Value SOPR | Signal | stress score |
|---|---|---|
| `< 0.93` | EXTREME_CAPITULATION | 0.95 |
| `< 0.97` | CAPITULATION | 0.80 |
| `< 0.995` | MILD_DISTRESS | 0.65 |
| `< 1.005` | BREAKEVEN | 0.50 |
| `< 1.03` | MILD_PROFIT | 0.40 |
| `< 1.08` | DISTRIBUTION | 0.25 |
| else | EXTREME_DISTRIBUTION | 0.10 |
- `None` → `("UNKNOWN", 0.5)`

---

## BAGIAN B — methods_institutional.py (M20–M31)

### B1. `compute_all_institutional(btc_current)` (baris 1085)

- Menghitung 12 metode; hanya menyimpan yang `score is not None`.
- `active = jumlah sukses` (0–12); `avg_score = mean(results.values())`.
- `micro_change_flags` & `micro_deteriorating` hanya untuk M20–M23.
- `micro_trend_score = count(flag in {SELL_SURGE,WIDENING_FAST,THINNING_FAST}) / 3.0`
- Persist cache mikro (`_MICRO_CACHE_FILE`, `.micro_cache.json`) untuk change-detection antar run.
- Return `(results, details, active, avg_score, micro_change_flags, micro_trend_score, micro_deteriorating)`.
- Di collect.py, `inst_results` memetakan ke keys `m20_obi`…`m31_altman`.

### B2. M20–M23 Microstructure (Binance)

#### M20 `calculate_m20_order_book_imbalance` (baris 191)
- Depth 20 level: `bid_vol=Σbids`, `ask_vol=Σasks`
- `obi = (bid_vol−ask_vol)/(bid_vol+ask_vol)` → −1…+1
- `obi_norm = (obi+1)/2` → 0…1
  | Kondisi obi_norm | score |
  |---|---|
  | `> 0.75` (buy-heavy, frothy) | 0.35 |
  | `< 0.25` (sell-heavy) | 0.75 |
  | `< 0.35` | 0.55 |
  | `> 0.65` | 0.40 |
  | else (balanced) | 0.25 |
- Change detection: `obi_delta = obi_norm−prev`: `<−0.15` SELL_SURGE, `<−0.08` SELLING_UP, `>0.15` BUY_SURGE, `>0.08` BUYING_UP, else STABLE.

#### M21 `calculate_m21_large_trade_flow` (baris 250)
- Threshold `large_threshold = 50000` USD; `notional = price*qty`
- `isBuyerMaker=True` → sell; hitung `large_sell_ratio = large_sells/(large_buys+large_sells)`
- Fallback bila `total_large==0`: return `0.30` (retail-dominated neutral)
  | Kondisi | score |
  |---|---|
  | `large_sell_ratio > 0.65` | 0.75 |
  | `> 0.55` | 0.55 |
  | `< 0.35` | 0.25 |
  | `< 0.45` | 0.35 |
  | else | 0.45 |

#### M22 `calculate_m22_spread_momentum` (baris 326)
- `spread_bps = (ask−bid)/mid*10000`; `range_bps = (high−low)/mid*10000`
- Spread score: `>10`→0.80, `>5`→0.55, `>2`→0.30, `>0.5`→0.15, else→0.10
- Range penalty: `range_bps>500`→+0.15; `>300`→+0.08; else 0
- **`score = min(spread_score + range_penalty, 0.95)`**

#### M23 `calculate_m23_liquidity_fractals` (baris 393)
- Simulasi $1M (`target=1_000_000`) market buy (walk asks) & sell (walk bids)
- `slippage_buy = (avg_price_buy/cum_buy − mid)/mid*100`; sell analog; fallback `5.0` bila likuiditas tak cukup (levels=999)
- `avg_slippage = (buy+sell)/2`; `min_levels = min(levels_used_buy, levels_used_sell)`
  | Kondisi | score |
  |---|---|
  | `avg_slippage > 2.0` | 0.80 |
  | `> 1.0` | 0.60 |
  | `> 0.5` | 0.40 |
  | `min_levels > 20` (deep) | 0.15 |
  | else | 0.25 |

### B3. M24–M28 Behavioral/Tail Risk

#### M24 `calculate_m24_cape` (baris 518) — Shiller CAPE adaptasi BTC
- `cape = current_price / mean(prices_365)` (1 tahun, bukan 10 tahun)
  | Kondisi | score |
  |---|---|
  | `cape > 3.0` | 0.80 |
  | `> 2.5` | 0.65 |
  | `> 2.0` | 0.50 |
  | `< 1.0` | 0.10 |
  | `< 1.5` | 0.15 |
  | else | 0.30 |

#### M25 `calculate_m25_minsky_moment` (baris 553)
- `fr_now=current_funding`; `accel = (fr_now−fr_1)−(fr_1−fr_2)` (dari history 8h)
- Stage & score:
  | Kondisi | stage | score |
  |---|---|---|
  | `fr_now>0.01 and accel>0.0005` | PEAK | 0.85 |
  | `fr_now>0.005 and accel>0` | BOOM | 0.65 |
  | `fr_now>0` | DISPLACEMENT | 0.35 |
  | `fr_now<−0.005` | REVULSION | 0.70 |
  | `fr_now<−0.002` | CRISIS | 0.85 |
  | else | NORMAL | 0.15 |

#### M26 `calculate_m26_kahneman_bias` (baris 627)
- `realized_price = mean(prices[-30:])`; `ath = _cg_get("ath", 126272)`
- **Loss aversion (`la_score`)**: jika `current<realized` → loss_pct>0.30→0.75, >0.15→0.55, >0.05→0.35, else 0.25. Jika profit → profit_pct>0.50→0.15, >0.25→0.20, else 0.25.
- **Anchoring (`anchor_score`)**: `ath_distance=(ath−current)/ath`; `<0.05`→0.70, `<0.15`→0.45, `<0.30`→0.25, else 0.15
- **Fear (`fear_score`)**: `skew=(put_iv−call_iv)/atm`; `skew>0.20`→0.60 else 0.20; bila data opsi tak ada → 0.30
- **FINAL: `score = 0.40*la + 0.30*anchor + 0.30*fear`**

#### M27 `calculate_m27_taleb_tail_risk` (baris 711)
- `dvol = Σ(mark_iv*oi)/Σoi` (OI-weighted IV)
- Return skewness 90d (standardized 3rd moment); `max_dd` 90d; `vol_7d_annual = stdev(rets_7d)*sqrt(365)`
- Term stress: `vol_7d > 1.2*vol_90d`→0.75; `vol_90d > 1.3*vol_7d`→0.50; else 0.30
- Skew stress: `<−1.0`→0.80, `<−0.5`→0.55, `>1.0`→0.50, else 0.25
- DD stress: `max_dd>0.20`→0.80, `>0.10`→0.55, else 0.25
- **FINAL: `score = 0.35*term + 0.35*skew + 0.30*dd`**
- `tail_danger`: CRITICAL jika score>0.70; ELEVATED >0.50; else NORMAL

#### M28 `calculate_m28_summers_stagnation` (baris 807)
- `gdp_growth = (GDPC1[0]−GDPC1[4])/GDPC1[4]` (YoY); fallback literal `2.0`
- `real_rate = DGS10 − (CPI YoY*100)`; fallback `fed_rate − 3.0`
  | Kondisi | stagnation |
  |---|---|
  | `real_rate < gdp_growth*0.5` | 0.75 |
  | `real_rate < gdp_growth` | 0.50 |
  | else | 0.20 |
- `regime`: STAGNATION jika >0.60, else NORMAL

### B4. M29–M31 Debt Cycle & Credit

#### M29 `calculate_m29_debt_crisis` (baris 858) — Reinhart-Rogoff
- `debt_ratio = GFDEGDQ188S/100` (%→ratio); fallback M2/GDP proxy atau literal `0.70`
  | Kondisi | score |
  |---|---|
  | `debt_ratio > 1.20` | 0.85 |
  | `> 0.90` | 0.70 |
  | `> 0.60` | 0.45 |
  | else | 0.20 |

#### M30 `calculate_m30_rajan_fsi` (baris 904) — aditif 6 komponen
- **1. Credit growth** (M2 YoY): `>0.15`→+0.20; `>0.10`→+0.10
- **2. Asset vol** (DVOL): `avg_iv>80`→+0.20; `>60`→+0.10
- **3. Yield curve** (10Y−2Y): `term<0.5`→+0.15
- **4. CDS proxy** (HY spread): `>400`→+0.15; `>300`→+0.10
- **5. OI concentration**: `top3_share>0.60`→+0.10
- **6. Capital flows proxy** (FNG): diambil tapi **tidak menambah skor**
- **FINAL: `score = min(Σ, 0.95)`**; stability: VULNERABLE >0.70, ELEVATED >0.40, else NORMAL

#### M31 `calculate_m31_altman_zscore` (baris 1007) — Z adaptasi crypto
- `vol_btc = vol_24h/btc_current`
- `x1 = min(depth/max(vol_btc,1), 1.0)*10` (depth $ = Σ bid+ask, 5 level)
- `x2 = 4.0` (konstanta netral)
- `x3 = btc/sma_200`, clamp `[−5,5]`
- `x4 = (futures_oi/vol_24h)*10`, clamp `[−10,10]` (fallback 1)
- `x5 = (vol_24h/mcap)*100*2`, clamp `[−10,10]`
- **`z = 0.6*x1 + 0.7*x2 + 1.65*x3 + 0.3*x4 + 0.5*x5`**
  | Kondisi z | score | zone |
  |---|---|---|
  | `< −5` | 0.85 | DISTRESS |
  | `< −1` | 0.70 | DISTRESS |
  | `< 2` | 0.50 | GRAY |
  | `< 5` | 0.30 | GRAY |
  | else | 0.15 | SAFE |

---

## Ringkasan Rantai Keputusan (pipeline di collect.py)

```
M1-M6 ─┐                    causal_weights>=0.2
M7-M19 ─┤  causal filter ──→ group avg ──→ sfc_pct = (p1*m1m6+p2*new+p3*inst)*100
M20-M31 ┘       (fallback 85/10/5)           p2=0 jika new_active==0; p1+=unused
M72-M75 ──────────────────→ _macro_active (0-4)
        │
        └─ dynamic weight (regime) → effective_sfc
           adv boost (if _rc_sev>=45, cap 2pp) → XGB blend (w=0.3*conf, if conf>0.3)
           → EWMA correct → effective_sfc
Regime  → _REGIME_DRIVER_MULT (0.7/1.0/1.2) → zone & signal thresholds
Kelly   → composite_confidence → kelly_fraction (b=2) × _kelly_override (0/0.5/1)
signal  → CASH jika kelly<=0 (p<=1/3) atau sfc>=50*MULT; BUY jika sfc<25*MULT; else WATCH
```

**Catatan integrasi:** bobot blend 85/10/5 hanya **fallback** di collect.py; nilai sesungguhnya datang dari `CausalFilter.get_blend_adjustment()` (modul `data_sources/causal_filter`). Rumus internal `get_sfc_effective_with_dynamic_weights`, `predict_ensemble` (XGB), `consolidate_regime`, dan EWMA `correct_stress` berada di modul terpisah (bukan hardcoded di collect.py) dan perlu diekstrak terpisah bila dibutuhkan.
