# Audit Total SFC — Skoring, Metode, Kalkulasi, Sistem (2026-09-09)

Audit 4 dimensi via subagent paralel + verifikasi independen tiap klaim CRITICAL/HIGH
ke file:line & data.json (aturan: temuan subagent = hipotesis sampai diverifikasi).
Data live: data.json ts 2026-09-09T02:40:33Z. Ringkasan per komponen diberi status verifikasi
[VERIFIED] = saya cek langsung ke kode/nilai, [REPORT] = dari subagent (belum cek ulang penuh).

## Verdict ringkas
- Rantai skoring headline (effective_sfc, composite_confidence dua-lapis, kelly, zone/signal)
  TERHITUNG KONSISTEN dan tidak ada cacat aritmetika/unit pada nilai live [VERIFIED].
- Ada SATU masalah keselamatan produksi (HIGH): circuit-breaker persistence mati.
- Ada SATU masalah metodologi scoring-affecting terbuka (MED): M4 memakai abs() pada faktor
  directional → euforia/greed ikut dihitung stress.
- ML/detektor mahal sebagian besar SUDAH di-gate OFF dengan benar (xgb blend 0, adv non-scoring);
  adv masih punya scale-mismatch 100x di fitur fit-vs-predict (tapi sudah display-only).

---

## A. Rantai Skoring & Confidence (subagent 1)
### Bersih [VERIFIED]
- effective_sfc 12.38 = sfc_base 13.45 + 0 (liq_mod) − DW 1.076 (adjust BULL). Data field
  dw_sfc_adjustment=−1.1 adalah nilai DIBULATKAN dari 0.08×base, bukan input terpisah. OK.
- composite_confidence 0.388 = macro_confidence 0.393 × (1 − execution_risk 0.014). macro_base =
  0.30 + method_agreement(0.419)·0.15 + low_stress_boost(0.0701=(1−12.38/100)·0.08) − penalty. OK.
- XGBoost TIDAK masuk effective_sfc (xgb_blend_weight=0, display-only). OK.
- adv boost TIDAK diaplikasikan. OK. kelly aritmetika benar (0.082). OK.

### Temuan
- [MED][REPORT] confidence_components.rsi/.dvol/.sopr = field DEAD: +0.03/+0.05/0.0 di-display
  seolah bagian dekomposisi, tapi TIDAK pernah masuk macro_confidence (formula 3605-3612 cuma
  pakai cc_base+boost−_macro_penalty). Sum komponen ≠ macro 0.393. → wiring atau hapus.
- [MED][VERIFIED-sebagian] M4 'ECB composite' pakai abs() pada faktor directional
  (collect.py:1215 `sum(w_ad[k]*abs(factors[k]))/3.0`). m4_ewc=33.6 vs saudara inti m1=6.8/m2=0.5/
  m3=8.0/m5=1.0/m6=1.0 — M4 4-6× lebih besar; ~94% kontribusinya dari faktor POSITIF (calm/greed).
  Metode lain (M1/M2/M5/M6) hanya sisi stress. SCORING-AFFECTING → keputusan design + walk-forward
  dulu sebelum diubah. Kemungkinan memang disengaja ("|deviasi| = instabilitas") tapi inkonsisten dgn 5 saudara.
- [LOW] composite_confidence tautologis dgn low effective_sfc yang dinilainya + dipakai ulang sebagai
  kelly_p_win → BUY procyclical di regime calm/greed (FNG 73). Pemisahan edge vs confidence disarankan.
- [LOW] calibrated_confidence non-monotonik (0.35→0.464, 0.45→0.313; 0.95→0.0); hanya field display.
- [LOW] determine_state (teks signal) pakai threshold 30/50 tanpa _REGIME_DRIVER_MULT, sementara
  zone/signal_type/signal_decision pakai mult → batas divergen di luar BULL/NEUTRAL (mis. mult 0.7).
- [LOW] prob_uncertainty_breakdown: method_disagreement_sigma & historical_noise_sigma terpaku di
  floor 3.0 (inert); ~83% final_sigma dari conf_sigma sirkular. Display-only.

---

## B. Metode M1–M86 & Faktor (subagent 2 — TIMEOUT 600s, TIDAK tuntas)
Bagian ini tidak selesai dijalankan subagent (timeout). Status: sebagian tercakup dari arsitektur
live + audit 2026-08 yang terdokumentasi + temuan M4 di atas. Yang TERVERIFIKASI sekarang:
- faktor de-dup state sesuai dokumentasi: liq_mod=0 (m2_yoy tak blend langsung), TGA/RRP hanya di
  GLF, xgb/adv display-only, m81/m82 (ETF) tidak masuk scoring. [VERIFIED untuk nilai live]
- m4_ewc=33.6 mendominasi core m1m6 (lihat A) [VERIFIED].
- Metode inti m1–m6 (causal-weighted) mendorong sfc_base; total_methods_active=40.
- Konstansi metode lintas waktu / scale-mismatch luas per-metode BELUM discan ulang penuh (perlu
  git-history scan). Rekomendasi: jalankan ulang scan ini terpisah sebelum menyimpulkan layer metode.

---

## C. Sistem / Pipeline (subagent 3)
### HIGH [VERIFIED] — Circuit breaker persistence mati di produksi
- .circuit_breaker_state.json beku sejak 2026-08-07 16:00Z (32+ hari): total_valid=0, last_valid={},
  consecutive_failures=4. data.json cb_failures=4 (konstan), cb_tripped=False.
- Root cause: collect.py buat CircuitBreaker BARU tiap run 5-menit; validate() sekali per proses.
  Disk state tiap run di-load (0), valid → total_valid 0→1, persist gate `total_valid%10` (1%10≠0)
  tak pernah terpenuhi; branch trip return {} di circuit_breaker.py:310 sebelum _save_state (:336).
  Ditambah cb_* di-EMIT di collect.py:4669-4671 (dalam `out`) yang DIBANGUN SEBELUM validate() :4760.
- Dampak: (1) cb_failures angka mati '4' (terlihat 1 lagi dari trip MAX=5); (2) last_valid kosong →
  get_last_valid() tak mengembalikan apa-apa; (3) pada trip sungguhan, jalur purge :4775-4780 restore
  0 field lalu PUBLISH out korup — justru kegagalan yang fix 2026-08-03 dimaksud mencegah.
- Fix (safety, bukan nilai skor): simpan state tiap validate (atau gate waktu); isi ulang last_valid
  tiap siklus; pindahkan emit cb_* ke bawah :4760. Regression: injeksi NaN sekali, pastikan data.json
  dapat last-good.

### MED [VERIFIED] — .probabilistic_head_history.json terkorup
442,728 byte, terpotong di EOF (JSONDecodeError char 442728), mtime 2026-08-11. Writer tak atomic.
Akumulasi history ProbHead rusak (346× 'Logging failed (non-fatal)' di log). → hapus/truncate agar
mulai bersih + tulis atomic (tmp+os.replace). Bukan scoring; logistik history.

### MED [VERIFIED] — ETF cache stale 4.7 hari
.etf_cache.json umur 113.6 jam >> TTL 12 jam (etf_flow.py CACHE_TTL=43200); data flow s/d
2026-09-03 dipakai utk m81/m82 tanpa flag staleness. (m81/m82 display-only — tak masuk scoring.)
→ refresh scripts/update_etf_cache.py + tambah field last_data_date/age + diagnosa Farside gagal.

### LOW — noise benign per-siklus (SPX 404→SPY, CoinGlass plan, calm-window neutral baseline).
Log 0 WARNING/0 ERROR karena pakai print bukan logger — grep WARNING bukan sinyal berarti.
### INFO [VERIFIED] — Baseline sehat: pipeline hidup, auto-commit tiap ~60-66 menit (reflog lokal),
throttle marker benar (hanya refresh saat commit nyata), sse_server pid 4738, cache inti segar tiap
siklus, onchain 21h<24h TTL. CPI-fallback-3.0 masquerade sudah dihapus (collect.py:1347).

---

## D. Regime / ML / Detector (subagent 4)
### HIGH [VERIFIED] — adv_regime scale-mismatch 100x fit-vs-predict (m1..m4)
feat_dict collect.py:2837-2840 membagi m1_klr..m4_ewc /100 — padahal variabel itu SUDAH internal 0-1
(di-emit ×100 di 4279-4282 & ×100 di 2693). m5_qreg memang 0-100 (jadi /100-nya benar). Hasil: fitur
prediksi m1..m4 ~100x di bawah cloud fit training → argmin k-means arbitrer (adv label noice,
flip BULL/BEAR/CRISIS/SIDEWAYS selama 5 hari calm). Konsekuensi praktis TERBATAS: adv sudah
display-only (tidak masuk consensus/boost). Fix: drop /100 pada m1..m4 di feat_dict (lalu walk-forward
sebelum pernah di-re-enable). Ini kelas bug yang sama yg didokumentasikan skill (regime-detector-artifact).

### HIGH → turunkan ke MED [VERIFIED] — prob_sharpe/prob_sortino bukan Sharpe/Sortino
= (50−mu)/sigma dengan mu=sfc, sigma sirkular dari conf (heuristik, tanpa OOS); sortino literal
=sharpe×1.5. Kode SUDAH mengakui misnomer (komentar probabilistic_output.py:227-234, rename
stress_distance_sigma) dan pertahankan nama sharpe/sortino "untuk backward-compat". Jadi isu-nya
PENAMAAN/UX pada field yang di-emit (prob_sharpe=3.72): ganti nama field/UI jadi stress_distance_sigma.
Bukan nilai salah (sudah didokumentasikan sebagai jarak sigma).

### MED [VERIFIED] — ml_accuracy=1.0 artefak calm-majority (1911 label, 0 stress)
evaluate_accuracy() sudah punya accuracy_reliable (stress_events>0) tapi TIDAK di-emit →
dashboard tampil '100%'. → surface accuracy_reliable/stress_events atau suppress saat stress_events=0.
### MED [VERIFIED-sebagian] — adv 'CRISIS'/'crisis_probability' = label kluster volatilitas tinggi +
prob transisi Markov 1-langkah dari 37 baris harian; TIDAK makna krisis. adv_crisis_prob 0.333 & boost
+15 koeksis dengan regime NORMAL. Tetap display-only. → relabel + jangan tampil sbg probability.
### LOW — HMM terpaku BULL 100% dari 92 snapshot terakhir (crisis 0.0) sementara regime NORMAL → HMM
tidak memberi informasi diskriminatif; consensus BULLISH (mult 1.2) dilonggarkan sebagian oleh label
tunggal ini. Cek pola HMM stuck/single-state. 
### LOW — behavior_state ACCUMULATION display-only TAPI dimasukkan sebagai 'source' di consensus
DISPLAY (severity 25) sementara consensus SCORING dihitung tanpa behavior → dua pemanggilan
consolidate_regime divergen; saat ini berimpit (severity 20). → single-source consensus.
### INFO [VERIFIED] — xgb blend 0 + adv exclusion: keduanya TIDAK masuk composite. Bersih.

---

## Rekomendasi prioritas (tidak ada yang dieksekusi dalam audit ini)
P0 (safety, scoring-adjacent, WAJIB sebelum insiden): perbaiki persistence circuit-breaker (C-HIGH).
  Perlu walk-forward/regression ringan (bukan ganti rumus skor).
P1 (logistik, aman, non-scoring): hapus .probabilistic_head_history.json rusak + tulis atomic;
  refresh ETF cache + flag staleness (C-MED).
P1 (display truthfulness, non-scoring): ganti nama field/UI prob_sharpe→stress_distance_sigma
  (D-MED); surface accuracy_reliable/suppress ml_accuracy=1.0 (D-MED); hapus/wire confidence
  components rsi/.dvol/.sopr dead (A-MED); single-source consensus behavior vs scoring (D-LOW).
P2 (scoring-affecting — WAJIB keputusan + walk-forward sebelum deploy):
  M4 abs() vs arah stress (A-MED) — putuskan desain (|deviasi|=instabilitas vs stress-directional
  konsisten dgn M1/M2/M5/M6) lalu validasi era-split; adv feat_dict /100 fix (D-HIGH) hanya jadi
  relevan bila adv mau di-re-enable (jangan sekarang); HMM-stuck (D-LOW) audit dulu sebelum sentuh.
P3 (UX/teks): selaraskan determine_state dgn zone/signal_decision pakai mult (A-LOW); kelly_p_win
  terpisah dari confidence stress-derived (A-LOW).

## Yang belum tercakup (jujur)
- Scan konstansi/scale tiap metode M1-M86 lintas git-history TIDAK tuntas (subagent timeout). Lapisan
  metode berdiri di: audit double-count 2026-08 terdokumentasi (state de-dup sesuai), arsitektur live
  terverifikasi, + temuan M4. Scan penuh disarankan sebagai langkah terpisah sebelum menyimpulkan
  kualitas seluruh 40 metode aktif.


---

## Fix yang SUDAH dieksekusi sesi ini (2026-09-09) — non-scoring, terverifikasi
1. .probabilistic_head_history.json korup -> di-truncate ke [] (backup di sfc_scratch_backup/) +
   writer dibuat atomic (tmp+os.replace) di analysis/probabilistic_head_tracker.py.
   Verifikasi: parse OK, append OK, tak ada tmp sisa.
2. ml_accuracy=1.0 -> emit ml_accuracy_reliable + ml_stress_events (collect.py ~4442);
   evaluate_accuracy() sudah punya accuracy_reliable (stress_events>0). Compile OK.
3. Circuit-breaker persistence (HIGH) -> FIXED: analysis/circuit_breaker.py kini _save_state()
   SETIAP validate (valid & trip), bukan gate total_valid%10 yang tak pernah menyala; cb_* di-refresh
   SETELAH validate di collect.py. Regression harness (state isolasi /tmp): 3 run valid persist
   (total_valid 3, last_valid terisi sfc_effective), 5 run NaN trip pada ke-5 + purge restore
   last-known-good 12.38, cooldown menahan output. Semua lulus.
4. ETF cache: update_etf_cache.py dijalankan — tapi Farside kini balas HTTP 403 (anti-bot), sebab
   asli data macet sejak 2026-09-03. cached_at DIKEMBALIKAN ke state lama (Sep-04) supaya staleness
   tetap terlihat & modul terus retry, bukan menutupi data 4 hari dengan cap 'fresh'. m81/m82
   display-only -> tak kena skor. Perlu workaround scraping/alternatif sumber + flag staleness.
