# Analisis Gap: Dokumen Kerangka Institusional vs Model SFC

**Tanggal:** 2026-08-02
**Sumber dokumen:** /home/ubuntu/A/apakah model sudah dapat membedakan beberapa situasi yang sangat penting, misalnya-.docx
**Status sinyal live (data.json):** regime=STRESS, sfc_effective=27.9, composite_confidence=0.308, signal_decision=CASH, behavior_state=EXPANSION, glf_regime=NEUTRAL, mpi_label=NEUTRAL

---

## 1. Ringkasan Eksekutif

Dokumen menuntut **pergeseran paradigma output**: dari "berapa stress / buy-cash" menjadi
**"berapa probabilitas tren berlanjut dalam horizon tertentu" + "deteksi divergensi transmisi likuiditas"**.
Model saat ini sudah sangat kaya dalam *input* (90+ metode, GLF, MPI, HMM, L6/L8/L5), tetapi
**belum menerjemahkan kekayaan itu ke output institusional yang diminta dokumen**:
belum ada Trend Strength Score, belum ada probabilitas kelanjutan tren forward-looking (1-3 & 6-12 bulan),
dan deteksi divergensi likuiditas-vs-BTC belum diekspos sebagai alert terpisah.

**Verdict: Gap adalah pada LAPISAN OUTPUT & INTERPRETASI, bukan pada data/mesin.** Komponen pembangunnya
sudah ada hampir semuanya; yang hilang adalah agregasi + penyajian yang sesuai kebutuhan family office.

---

## 2. Pemetaan 5 Skenario vs Kemampuan Model

| # | Skenario dokumen | Status | Field pendukung yang ada | Gap |
|---|---|---|---|---|
| 1 | Likuiditas makro naik + BTC menguat → tren naik berlanjut | ✅ Sebagian | glf_score, glf_regime, behavior_state, composite_confidence | Belum ada *score probabilitas kelanjutan tren* (hanya stress) |
| 2 | Likuiditas naik + BTC melemah → jeda transmisi / rotasi | ⚠️ **LEMAH** | glf_stress vs sfc_effective (kontradiktif), mtf_divergence, m33_glo | **Tidak ada mekanisme eksplisit "jeda transmisi"** — dua sinyal kontradiktif tidak diresolusi jadi narasi |
| 3 | Likuiditas melemah + BTC naik → divergensi waspadai | ⚠️ **LEMAH** | reflexivity_divergence_score, behavioral_divergence_score, mtf_divergence | Divergence terdeteksi tapi belum di-ekspose sebagai **alert** terpisah yang actionable |
| 4 | Likuiditas & struktur BTC sama-sama lemah → tren berubah | ✅ **KUAT** | regime=STRESS, tail_risk=42.5, kelly=0→CASH | Inilah kasus live sekarang; sudah tertangkap baik |
| 5 | Disiplin metodologi (kausal tidak absolut, uji data historis) | ✅ Sesuai | wfv_gap_30d=-7.46 sig, walk_forward_imbs_l1l2/l8 | Sudah dianut (walk-forward validation) |

---

## 3. Pemetaan Output yang Direkomendasikan Dokumen vs Ketersediaan

| Rekomendasi dokumen | Ada? | Kondisi saat ini |
|---|---|---|
| **Regime makro** (Expansion/Bubble/Distribution/Contraction) | ✅ Sebagian | `behavior_state` 5-state (ACCUMULATION/EXPANSION/EUPHORIA/DISTRIBUTION/PANIC) — **display-only, bukan output utama**; `hmm_regime` (4-state) & `adv_regime` juga ada tapi 3 sistem regime beda-beda (SIDEWAYS vs CRISIS vs STRESS) tanpa konsolidasi |
| **Trend Strength Score** | ❌ **Belum** | Tidak ada metrik kekuatan tren eksplisit. Ada `m32_mamba_short/medium/long` tapi **semuanya 0.0** (tidak aktif) |
| **Probabilitas kelanjutan tren 1-3 bln & 6-12 bln** | ❌ **Belum** | Ada `prob_stress/prob_calm/prob_crash_10pct` tapi itu probabilitas *stress*, bukan probabilitas *arah/kelanjutan tren*; `predicted_mean/std` tanpa horizon jelas |
| **Confidence dari keselarasan SFC+likuiditas+perilaku+struktur** | ✅ Sebagian | `composite_confidence=0.308` ada, tapi **alignment-based (keselarasan antar komponen) belum eksplisit**; `method_agreement=0.399` = keselarasan antar-metode, bukan antar-domain |

---

## 4. Temuan Kritis (harus diselesaikan sebelum membangun fitur baru)

### 4.1 TIGA SISTEM REGIME YANG TIDAK KONSISTEN
Saat ini model punya 3 regime yang saling bertentangan:
- `regime` = STRESS (prob 0.434) ← main ensemble
- `hmm_regime` = SIDEWAYS (HMM)
- `adv_regime` = CRISIS (advanced, LOW_CONFIDENCE)
- `behavior_state` = EXPANSION (L5, display-only)

Empat label beda untuk hari yang sama. **Ini pelanggaran prinsip satu-sumber-kebenaran** dan
membingungkan pembaca institusional. Dokumen justru meminta SATU regime makro yang jelas.

### 4.2 DETEKSI DIVERGENSI TRANSMISI (skenario #2 & #3) TIDAK DIAGREGASI
Input sudah ada: `glf_stress` (likuiditas) vs `sfc_effective` (struktur BTC), `m33_glo_score`,
`mtf_divergence`, `reflexivity_divergence_score`, `behavioral_divergence_score`.
Tapi semua terpisah — tidak ada satu *output* yang bilang "likuiditas naik tapi BTC melemah = jeda transmisi".

### 4.3 TIDAK ADA HORIZON FORWARD-LOOKING YANG SEJALAN DENGAN WALK-FORWARD
Walk-forward (wfv_gap_30d=-7.46pp) membuktikan sinyal stress memprediksi return 30d. Tapi tidak
diterjemahkan menjadi probabilitas *practical*: "peluang tren naik berlanjut dalam 1-3 bln = X%".

---

## 5. Prioritas (urutan rekomendasi, mengikuti prinsip walk-forward validation)

### P0 — KONSOLIDASI REGIME (syarat dasar, sebelum yang lain)
Gabungkan regime/hmm/adv/behavior_state → SATU label regime makro + konflik-flag bila beda.
Tanpa ini, output institusional (regime makro) tidak akan kredibel.

### P1 — DETEKSI DIVERGENSI TRANSMISI (skenario #2, #3)
Bangun modul baru yang membandingkan arah likuiditas (GLF) vs arah BTC/struktur (sfc) →
output: `transmission_status` (TRANSMITTING / TRANSMISSION_GAP / ROTATION / DIVERGENCE)
+ confidence. Ini menjawab gap paling kritis dokumen & paling dekat dengan data yang ada.

### P2 — TREND STRENGTH SCORE
Komposit dari momentum (RSI, mamba bila aktif), alignment multi-timeframe (mtf_alignment),
dan struktural (HMM trend persistence). Skala 0-100.

### P3 — PROBABILITAS KELANJUTAN TREN 1-3bln & 6-12bln
Reuse hasil walk-forward (wfv_gap_30d, bootstrap CI) sebagai kalibrator empiris:
probabilitas kelanjutan = f(kekuatan tren saat ini, keselarasan komponen, gap walk-forward).
Ini output paling bernilai untuk institusi, tapi PALING menuntut validasi.

---

## 6. Yang TIDAK boleh dilakukan (sesuai disiplin metodologi dokumen #6)

- Jangan mengklaim kausalitas absolut — model hanya menyediakan *argumen kausal yang konsisten + data historis*.
- Setiap fitur output baru (divergence, trend strength, probabilitas) **wajib walk-forward validated**
  sebelum dipajang sebagai angka kredibel di dashboard family office.
- Jangan blend L6/L8/L5 ke sfc_effective sampai sfc_effective live direkonstruksi & dikalibrasi ulang
  (precedent: rekomendasi STRESS=55 & cutoff L8 tidak di-deploy karena distribusi beda).

---

*Dokumen ini adalah analisis gap — belum ada perubahan kode.*
