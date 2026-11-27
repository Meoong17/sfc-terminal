# Audit Kode & Metode SFC — 2026-11-27

Status: audit menyeluruh (4 workstream paralel) + perbaikan yang sudah **live**.
Tanda: `[V]` = saya verifikasi sendiri ke kode/angka nyata; `[R]` = laporan subagent,
belum saya reproduksi. Laporan mentah: `~/.hermes/cache/scratch/audit_{metode,data,ml,pipeline}.md`.

Baseline yang dibaca lebih dulu: `docs/AUDIT_TOTAL_2026-09-09.md`,
`docs/AUDIT_KODE_2026-09-18.md`, `docs/PROJECT_STATUS.md` — temuan lama yang sudah
diperbaiki tidak dilaporkan ulang.

## 0. Ringkasan dampak

| Metrik | Sebelum audit | Sesudah perbaikan |
|---|---|---|
| `sfc_effective` | 48.08 | **34.99** (−13.1 pp) |
| `zone` | ELEVATED | ELEVATED |
| `m8_yield` | 0.45 (konstan) | 0.65 |
| `m9_liquidity` | `None` (diisi 0.5) | 0.65 |
| `m10_garch` | 0.85 (konstan) | 0.65 |
| `m11_var` | 0.85 (konstan) | 0.25 |
| `m13_funding` | `None` (diisi 0.5) | 0.15 |
| `m14_skew` | 0.10 (konstan) | 0.80 |
| `m28_summers` | 0.20 (konstan) | 0.75 |
| `m81_etf_flow` | 0.60 | 0.85 |
| `m32_adjustment_pp` | −1.01 | 0 (display-only) |
| `ml_accuracy_reliable` | true (0.9263) | **false** (recall stress 0.0, baseline 0.9648) |
| `ewma_corrected` | true (padahal no-op) | false (jujur) |
| `hmm_available` | false karena label NORMAL | false karena model **tidak reliabel** |

Skor turun 13 pp bukan karena "metode diganti", melainkan karena bobot kausal berhenti
mengalikan level skor (lihat A1) dan metode yang tadinya konstanta kini membaca data.

## 1. Temuan KRITIS yang diperbaiki

### A1. `[V]` Bobot kausal mengalikan skor, bukan merata-ratakan — collect.py:2660-2760
`CausalFilter.apply_filter()` mengembalikan `min(1, score * weight)`, lalu ensemble
mengambil **rata-rata polos** dari hasil kali itu:
`group_avg = mean(score·w) = mean(score) · mean(w)`.
Akibatnya: M1–M6 (bobot 1.0–1.5, dilindungi) jenuh di 1.0 dan grup naik
0.366 → 0.550; grup M7–M19 dan M20–M31 (bobot ~0.5–0.7) menyusut 0.190→0.114 dan
0.286→0.120. Blend nominal 91.8/4.2/4.0 berperilaku seperti ~98/1/1 — dua grup
"enrichment" praktis tidak berkontribusi.
Perbaikan: weighted mean `sum(s·w)/sum(w)` (`_group_weighted_avg()`).
Efek terukur: 48.08 → 34.99.

### A2. `[V]` Off-by-one kolom fitur kausal — collect.py:2968, causal_inference.py:29
`collect.py` menulis 30 kolom dan **melewatkan `m8_yield`**, sedangkan `METHOD_NAMES`
berisi 31 nama berurutan. Semua kolom ≥ indeks 7 salah label: kolom 7 (m9) dibaca
sebagai m8, kolom 18 (m20_obi) masuk grup M7–M19, m31 tidak punya kolom.
Bukti: `features.shape=(2000,30)`; kolom 7 konstan 0.5000 = nilai m9.
Perbaikan: daftar nama disamakan ke urutan 30 kolom sebenarnya + `FEATURE_ORDER_VERSION`.

### A3. `[V]` Riwayat stablecoin dihapus fallback — stablecoin_liquidity.py:192
Fallback DefiLlama **mengganti** seluruh `supply_history` (365 hari) dengan 3 titik
sintetis; saat semua sumber gagal hanya `cached_at` yang ditulis, jadi cache kosong
tampak segar 12 jam. Live saat audit: `supply_history` = 3 kunci.
Perbaikan: merge (bukan replace) + `_fetch_history_from_defillama()` memakai endpoint
`stablecoincharts/all` (CG sedang 429 untuk 0/4 koin), level diharmoniskan (scale 0.8780,
anchor 2026-11-27) agar tidak ada lompatan +14% palsu. Hasil: **400 hari** terisi kembali.
`cached_at` tidak lagi dicap saat fetch gagal.

### A4. `[V]` Nudge QLSTM dari checkpoint sirkular — collect.py:2891
`qlstm_model.pt` (mtime 2026-07-16) dilatih dengan target = kombinasi linear M1–M6
(fitur inputnya sendiri); `val_loss=8.11e-08` pada target 0–100. Model ini masih
menggeser skor 5% dari selisih tiap siklus (live −1.01 pp).
Perbaikan: nudge dimatikan (`m32_adjustment_pp=0`), nilai bakal-ada diekspos sebagai
`m32_adjustment_display_pp` + `m32_note`. Re-enable hanya setelah retrain + walk-forward.

## 2. Temuan TINGGI yang diperbaiki (skala satuan / konstanta menyamar)

| ID | Lokasi | Masalah (bukti) | Perbaikan |
|---|---|---|---|
| B1 `[V]` | collect.py:1444 | HY OAS dalam **persen** (live 3.7) dibanding ambang **bps** (200/300/400) → komponen kredit 40% tidak pernah aktif; `else 300` mengarang stres saat data hilang | konversi ke bps; data hilang = komponen dilepas |
| B2 `[V]` | collect.py:1490 | M10 GARCH: input **bulanan** (vol 0.1236), ambang **harian** 0.03 → 0.85 di 83/83 window | ambang ×√21 |
| B3 `[V]` | collect.py:1509 | M11 ES bulanan −0.2055 vs ambang harian −0.15 → 0.85 di 82/82 | ambang ×√21 |
| B4 `[V]` | collect.py:1615 | M16 std bulanan 0.129 vs 0.025 → 0.45 di 81/82 | ambang ×√21, mean ×21 |
| B5 `[V]` | methods_institutional.py:840 | M28 real_rate **persen** (0.88) vs gdp_growth **rasio** (0.0229) → selalu 0.20 di 2000 siklus | satuan disamakan (%) |
| B6 `[V]` | methods_institutional.py:1034 | M31 `x2=4.0` hardcode → z≥2.8, 3 dari 5 cabang mustahil | x2 = 0.0 netral (tidak ada seri network-health yang tersambung) |
| B7 `[V]` | collect.py:1475 | M9 FRED `MBCURSL` tidak ada (HTTP 400) → metode selalu None | `BOGMBASE` |
| B8 `[V]` | collect.py:1563 | M13 Deribit tanpa `instrument_name` + `start_timestamp=0` (HTTP 400) → selalu None padahal bobot kausal 0.3 | `BTC-PERPETUAL`, jendela 7 hari, urut terbaru |
| B9 `[V]` | methods_institutional.py:445 | M23 sentinel slippage 5.0% saat $1jt tak terpenuhi → "very thin book" 0.80 padahal sisi bid terisi 70 level | sisi tak terukur = unavailable, bukan sentinel |
| B10 `[V]` | collect.py:1587 | M14 skew = rata-rata semua put − semua call; Deribit memakai satu `mark_iv` per strike → put=call=62.064 (523 vs 523), skew 0 selalu | OTM put vs OTM call per moneyness 10–30%, normalisasi ATM |
| B11 `[V]` | collect.py:1598 | M15 HHI dibanding ambang 0.15–0.25, maksimum teoretis 1/N (live HHI 0.00468, 1/N 0.00117) → mustahil | top-3 share (live 0.049) |
| B12 `[V]` | etf_flow.py:160 | `etfs` dalam **juta-USD** dipakai sebagai **BTC** (rasio sum/total_usd = 1e6) → arus −8.136 BTC terbaca −231 "BTC"; m81 masuk skor via `etf_rt_adj` | konversi USD→BTC dengan harga yang sudah tersedia |
| B13 `[V]` | onchain_fetch.py:232 | `cached_at` dicap setiap save, termasuk saat 0 metrik terambil → data beku 20 jam tampak segar | `refresh_ttl` eksplisit + tulis atomik |
| B14 `[V]` | ml_ensemble.py:37 | `save_collection()` non-atomik untuk file 2.7 MB di bawah `timeout 150` | tmp + `os.replace` |

## 3. Temuan TINGGI pada penyajian & operasi (diperbaiki)

- **C1 `[V]` HMM: flag & gerbang.** `_hmm_available = True` hanya bila `regime != NORMAL`,
  dan flag `reliable` dari model tidak pernah dibaca. Dua cacat: (a) HMM sehat yang membaca
  NORMAL dilaporkan "unavailable"; (b) model tersaturasi bisa menimpa `regime`/`transition_risk`
  yang menggerakkan `_REGIME_DRIVER_MULT` → ambang zona. Kini keduanya digerbang pada `reliable`.
  Modelnya sendiri masih degenerasi (covars 1e-6 pada state 0/3, transmat identitas,
  posterior 1.0000) → **butuh retrain dengan `reg_covar` + label state berbasis risiko**,
  dan itu perubahan model yang wajib lewat walk-forward (belum dikerjakan).
- **C2 `[V]` EWMA.** `models/ewma_state.json` tetap `{"baseline":0.0,"history":[]}`:
  koreksi dihitung pada singleton modul tetapi `save_ewma()` menyimpan instance lain,
  jadi tidak pernah belajar — namun `ewma_corrected: true` diterbitkan (hardcode).
  Ditambah `composite_confidence ... if 'composite_confidence' in dir() else 0.5` yang
  selalu 0.5 karena variabelnya baru ada ~430 baris kemudian. Kini display-only + flag jujur.
- **C3 `[V]` `models/hmm_regime.py:644`** memakai nama `sfc` yang tidak ada → NameError di
  `main()`; diganti `m2`.
- **C4 `[V]` Kejujuran akurasi ML.** `ml_accuracy 0.9263` < baseline selalu-calm `0.9648`,
  recall stress 0/65, tetapi `accuracy_reliable=true` (syaratnya hanya `stress_events > 0`).
  Kini: `recall_stress`, `recall_calm`, `balanced_accuracy`, `majority_baseline`,
  `skill_vs_baseline`, dan `accuracy_reliable` mensyaratkan stress_events≥30 dan recall>0.
- **C5 `[V]` sse-watchdog tanpa token.** Restart manual watchdog menjalankan `sse_server.py`
  tanpa `SFC_ORIGIN_TOKEN` (token hanya di `/etc/sfc-origin.env`, root-only, di unit systemd)
  → satu restart mematikan autentikasi origin (`if not _ORIGIN_TOKEN: return  # open`).
  Kini restart lewat unit systemd; tanpa unit & tanpa token → menolak start.
- **C6 `[V]` Cron basi.**
  - `sfc-pipeline.sh` repo ketinggalan blok quant engine vs salinan cron; `ws-watchdog.sh`
    beda (`python3` vs `/usr/bin/python3`). Disinkronkan live→repo, md5 identik sekarang.
  - Cron `*/5 * * * * cd /home/ubuntu/B && python3 refresh_live_values.py` menunjuk skrip
    yang tidak ada (log error bertahun) → dihapus; crontab dibackup.
  - `sfc-daily-retrain.sh` selalu exit 0 (semua tahap `| tee`, baris akhir `echo`) →
    `set -uo pipefail` + penghitung kegagalan + `exit 1`; salinannya kini ada di repo.
- **C7 `[V]` Falsy-zero.** `effective_sfc` 0.0 dianggap kosong; `(fng or 50)`, `(dvol or 50)`,
  `sfc_pct/100 if sfc_pct else 0.5`, `signal_strength`/`readiness_score`/`signal_type`
  memakai truthiness. Diganti uji `is None`.

## 4. Temuan yang BELUM diperbaiki (butuh validasi / keputusan)

1. `[R]` HMM degenerasi & label regime terbalik (BULL = DVOL 55/RSI 30/FNG 13; CRISIS =
   paling tenang; BEAR mean return +0.122%/hari) — argsort mean return dengan selisih
   0.18 pp/hari (di dalam noise). Perlu label berbasis sumbu risiko + `reg_covar`.
2. `[R]` Kebocoran label: horizon 360 menit pada sampling ~6 menit (overlap 60×, ~47 jendela
   independen), tanpa purge/embargo di split, 65 label stress = 4 episode.
3. `[R]` Circuit breaker deadlock: `_last_valid` hanya diperbarui saat `all_ok`, pergeseran
   level >20 pp membuat siklus berikutnya gagal permanen (1183/17521 siklus pernah gagal).
4. `[R]` expectation `gap_score=50.0` dengan `status:"ok"` saat komponen CPI hilang.
5. `[R]` XGBoost `best_iteration=0`, keluaran konstan 2.86–4.78/100, tetapi masih dihitung.
6. `[R]` Ambang zona UI (25/50, `>=`, 3 kelas) vs backend (0.25/0.50/0.75, `>`, 4 zona);
   KPI confidence memakai `|| 0.5` sehingga 0/null tampil "50% Moderate trust".
7. `[R]` `worker/index.js` `/live.json` membalas 200 + `{"error":"unreachable"}` saat origin gagal.
8. `[R]` `cli`/ops: `log()` pipeline menulis ke stdout (tidak masuk `sfc-pipeline.log`);
   `index.html` masih memuat Chart.js dari CDN padahal ada `static/chart.umd.min.js`.
9. `[R]` 10–12 dari 30 fitur ML masih konstan (m17, m19, m24, m29 …) dan 15 fitur
   berimportance 0 — perlu dibersihkan setelah metode diperbaiki, lalu retrain.
10. `[R]` Kalibrasi isotonic: n test 58 dengan label tumpang tindih, verdict berbalik antar-refit.

## 5. Verifikasi yang dijalankan

```
~/.hermes/scripts/sfc-pipeline.sh            -> collect=ok, tanpa Traceback
data.json ts 2026-11-27T07:58:21Z            -> sfc_effective 34.99, zone ELEVATED,
                                                m32_adjustment_pp 0, ml_accuracy_reliable false
https://sfcterminal.xyz/data.json            -> 200, ts & angka identik dengan origin
md5sum sfc-pipeline.sh / ws-watchdog.sh / sse-watchdog.sh / sfc-daily-retrain.sh
                                             -> salinan cron == repo
```

Aturan yang dipegang: tidak ada perubahan pada definisi ambang zona, bobot faktor
`_FACTOR_WT`, atau rumus M1–M6. Semua perubahan di atas adalah perbaikan satuan,
urutan, kontrak data, dan kejujuran status. Perubahan yang menggeser level skor
(weighted mean A1, satuan metode B1–B13, ETF B12) dicatat delta-nya di §0 dan
**perlu ditinjau ulang pada walk-forward berikutnya** — ambang 25/50/75 mungkin perlu
dikalibrasi ulang karena skor kini ~13 pp lebih rendah pada kondisi yang sama.
