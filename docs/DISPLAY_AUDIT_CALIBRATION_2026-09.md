# Audit Display, Kalibrasi Confidence & Regime Detector (2026-09)

Ringkasan pekerjaan lanjutan setelah verdict IGC `NOT_BLEND` (lihat
`docs/CRITICALITY_IGC_RESULT.md`). Semua angka di bawah berasal dari eksekusi nyata
pada data repo ini; perintah reproduksinya tercantum di bagian akhir.

**Aturan yang dipatuhi:** tidak ada perubahan pada formula scoring (`sfc_effective`,
bobot metode, blend XGB/ML-RF). Yang diperbaiki adalah (a) label display yang tidak
punya dasar, (b) kalibrasi confidence, (c) kejujuran metadata komponen display.

---

## 1. Detektor regime `adv_regime` — label yang tidak punya dasar

**Gejala.** `data.json` menampilkan Advanced Regime = `CRISIS` sementara `adv_crisis_prob`
= 0.143 dan HMM/regime utama = BULL/NORMAL.

**Akar masalah (diverifikasi dari data, bukan dugaan).**
- Detektor k-means + Markov (`ml/sfc_advanced.py: RegimeDetector`) dilatih dari
  `data_collection_daily.json` yang berisi **38 baris harian** — komentar kode mengklaim
  akumulasi "bertahun-tahun", kenyataannya ≈1,3 bulan.
- Matriks transisinya dibangun dari **≤7 transisi** (baris berisi pecahan seperti 2/7 dan
  4/7, yaitu 2 dan 4 kejadian) → label adalah derau.
- **Bukti ketidakstabilan (harness 200 bootstrap refit pada data yang sama):** label untuk
  observasi terakhir = **BULL 99×, BEAR 64×, SIDEWAYS 24×, CRISIS 13×** — 49% refit
  memberi label berbeda dari titik estimasi. Label seperti ini tidak boleh ditampilkan
  seolah fakta.
- Bug semantik: `stability = 1 − P(stay)`. Nilai 0.857 yang tampil berarti **85,7% peluang
  MENINGGALKAN** regime, bukan bertahan.

**Perbaikan.**
| lokasi | perubahan |
|---|---|
| `collect.py` | guard `_ADV_MIN_FIT_ROWS = 250` (≈1 tahun observasi harian). Di bawah ambang → label & probabilitas **tidak diterbitkan** |
| `collect.py` | cache pra-guard (6 jam) ditolak: wajib `regime_reliable is True` |
| `collect.py` | field baru: `adv_regime_reliable`, `adv_regime_fit_rows`, `adv_regime_note`; `adv_regime`/`adv_crisis_prob`/`adv_regime_stability` → `None` bila tak reliabel |
| `ml/sfc_advanced.py` | `stability = P(stay)` (semantik benar) + field `exit_probability` + pencatatan `n_fit_obs` |
| `index.html` (+ master `/home/ubuntu/index.html`, `app.js`) | fallback `d.adv_regime \|\| 'NORMAL'` dihapus — label regime tidak lagi bisa bocor dari detektor mati |

**Verifikasi e2e** (`data.json` setelah pipeline berjalan):
```
adv_regime = None | adv_crisis_prob = None | adv_regime_reliable = False
adv_regime_fit_rows = 38 | adv_regime_note = "insufficient daily history for
k-means+Markov regime (38 rows < 250); label/probabilities not published"
```

---

## 2. Audit komponen display-only (null IAAFT) — 9 komponen, 0 lolos

**Alasan uji ulang yang sah (bukan p-hacking):** file orderflow yang dulu ditolak hanya
mencakup 2017-2020. `data/binance_orderflow_daily.json` kini berisi **3.241 hari
(2017-08 .. 2026-07)**, sehingga era-split penuh akhirnya mungkin — uji lama tidak bisa
menjawab regime 2021-2026.

**Metode** (`analysis/audit_display_components_iaaft.py`, seed 20260911, 200 surrogate):
purged-CV AUC dengan embargo (label = forward < 0), rank-IC 7/30 hari + BH-FDR,
era-split 3 era, dan **null IAAFT pada seri sinyal itu sendiri** (fase-randomisasi
mempertahankan amplitudo + spektrum ⇒ menguji apakah *timing* sinyal membawa informasi
melebihi persistensinya sendiri).

| komponen | n OOS | IC30 | gap30 (pp) | purged AUC30 | p_IAAFT | era1/era2/era3 (gap30, pp) | verdict |
|---|---|---|---|---|---|---|---|
| `of_taker_imbalance_qty` | 3211 | −0.018 | −1.9 | 0.419 | 0.740 | −10.9 / +0.3 / +2.9 | NO_EDGE |
| `of_taker_buy_ratio` | 3211 | −0.018 | −1.9 | 0.419 | 0.745 | −10.9 / +0.3 / +2.9 | NO_EDGE |
| `of_whale_lo_count` | 3211 | −0.135 | −8.8 | 0.517 | 0.070 | −0.1 / −7.6 / −7.1 | NO_EDGE |
| `of_n_trades` | 3211 | −0.012 | −4.1 | 0.396 | 0.890 | −1.8 / +2.7 / +2.6 | NO_EDGE |
| `of_total_quote` | 3211 | −0.092 | −9.6 | 0.412 | 0.160 | −11.5 / −3.3 / −1.7 | NO_EDGE |
| `cm_mvrv_chg30` | 340 | −0.147 | −6.2 | 0.478 | 0.700 | (n<era-split) | NO_EDGE |
| `cm_adract_chg30` | 340 | −0.082 | −2.4 | 0.447 | 0.290 | (n<era-split) | NO_EDGE |
| `cm_hashrate_chg30` | 340 | −0.021 | −1.6 | 0.409 | 0.850 | (n<era-split) | NO_EDGE |
| `cm_supply_chg30` | 340 | −0.249 | −5.9 | 0.504 | 0.160 | (n<era-split) | NO_EDGE |

**Bacaan hasil.**
- 9/9 gagal null IAAFT (p 0.07-0.89) **dan** gagal AUC purged > 0.55 → tak satu pun punya
  informasi timing yang bisa dipisahkan dari persistensi sinyalnya sendiri.
- `of_taker_imbalance`/`taker_buy_ratio` **di bawah koin** (AUC 0.419) pada 9 tahun data:
  tolakan atas orderflow pada window 2017-2020 kini dikuatkan, bukan sekadar diulang.
- `of_whale_lo_count` paling dekat: era2/era3 tanda konsisten (−7.6/−7.1 pp, p 0.91/0.96)
  tetapi AUC 0.517 dan p_IAAFT 0.070 → tetap **tidak lolos**; dicatat sebagai kandidat
  paling layak bila kelak ada uji dengan daya lebih tinggi, bukan sebagai sinyal.
- CoinMetrics hanya 340 hari OOS → era-split tidak mungkin; itu **dilaporkan apa adanya**,
  tidak dipaksakan menjadi klaim era-stability.
- Implikasi display: seluruh kartu yang memuat komponen ini tetap **display-only**, dan
  klaim prediktif apa pun pada komponen tersebut tidak didukung bukti.

---

## 3. Kalibrasi confidence — peta monoton + gate OOS (dipakai)

**Masalah sumber.** `.calibration_state.json` (dibangun 2026-08-12 dari 499 snapshot)
menyimpan peta bin-based yang **non-monoton** (0.45 → 0.313 lalu 0.55 → 0.55; diskontinuitas
dari bin kosong) dan **tak pernah di-refit** walau ada 19.019 commit `data.json`.
ECE blended 0.170 vs model-only 0.0968 → peta itu tidak membantu.

**Dua pelajaran metodologis yang mengubah desain (keduanya ditemukan dengan uji, bukan asumsi):**
1. **Isotonic/PAV GAGAL untuk n kecil.** Pada label biner dengan n≈200, PAV memprediksi 0/1
   (blok tunggal) → hold-out **memburuk**: ECE 0.246 → 0.445, Brier 0.307 → 0.445
   (uji sintetis). Isotonic butuh n besar; di sini hanya boleh dilaporkan, bukan dipakai.
2. **`a,b ≥ 0` tidak menjamin monotoni.** Turunan logit beta-calibration terhadap s adalah
   `a/s − b/(1−s)`; non-negatif untuk semua s **hanya bila a ≥ b**. Fit pertama menghasilkan
   b=1.50 > a=0.34 → peta berbentuk **punuk** (puncak s≈0.187 lalu turun), mengulang cacat
   non-monoton yang justru sedang diperbaiki. Perbaikan: reparametrisasi `b = a·u`, u∈[0,1]
   ⇒ a ≥ b terjamin secara struktural.

**Protokol (ditetapkan sebelum melihat metrik uji):**
`Platt (b=0, 1 parameter, monoton)` sebagai primer → `beta terkendala monoton (a≥b)` sebagai
fallback bila primer gagal gate → isotonic hanya diagnostik (n≥1000). Split **temporal**
60/40; peta dipakai **hanya bila ECE DAN Brier keduanya membaik** di hold-out, dan
`monotone_verified` wajib True (dicek numerik pada 201 titik grid).

**Hasil nyata** (`analysis/calibration_isotonic.py`, n=144 snapshot berlabel harga,
train 86 / test 58, base-rate stress 0.4655):

| metrik | raw | terpilih (Platt) | beta-monoton (diagnostik) | isotonic |
|---|---|---|---|---|
| ECE | 0.1822 | **0.0397** | 0.0237 | dilewati (n<1000) |
| Brier | 0.2714 | **0.2462** | 0.2494 | dilewati |
| monoton | — | True | True | — |
| lolos gate | — | ya | ya | — |

Peta aktif (raw → kalibrasi): `0.0→0.019, 0.1→0.358, 0.3→0.434, 0.5→0.471, 0.7→0.496,
0.9→0.514, 1.0→0.522` — monoton naik, dan **terkompresi** di sekitar base rate. Artinya
confidence mentah memang hanya punya sedikit informasi arah stress pada horizon pendek
(honest output), bukan sinyal kuat. Wiring terverifikasi: `collect.py:3805` memanggil
`recalibrate()` → field `calibrated_confidence` di `data.json`.

**Refit terjadwal (belum dipasang ke cron — langkah berikutnya):** jalankan
`analysis/calibration_isotonic.py` bulanan dengan pola `scripts/*.sh` yang ada; state
mencatat `fitted_at`, `n_labeled`, `metrics`, dan alasan bila ditolak.

---

## 4. Catatan operasional penting: index.html punya master di luar repo

`sfc-pipeline.sh` **memulihkan** `index.html` tiap siklus dari `/home/ubuntu/index.html`
(salinan repo di-`skip-worktree`). Akibatnya setiap edit `index.html` di dalam repo akan
**hilang** pada siklus berikutnya kecuali master itu juga diubah. Perbaikan display pada
dokumen ini diterapkan di: master `/home/ubuntu/index.html`, salinan repo, dan `app.js`
(dead code, untuk mencegah bug yang sama hidup lagi bila `app.js` diaktifkan).

---

## Reproduksi

```bash
# 1. regime detector (bukti ketidakstabilan label)
.venv/bin/python analysis/audit_display_components_iaaft.py   # komponen display + null IAAFT
#    harness adv_regime: lihat blok "bukti root-cause" (~200 bootstrap refit)

# 2. kalibrasi + gate OOS
.venv/bin/python analysis/calibration_isotonic.py --dry-run   # lapor saja
.venv/bin/python analysis/calibration_isotonic.py             # tulis state bila lolos gate

# 3. validasi IGC pada panel ter-refresh
.venv/bin/python analysis/validate_igc_criticality.py         # panel 3301 hari
```

## Yang TIDAK diubah

- `sfc_effective`, bobot metode, dan blend XGB/ML-RF: tidak disentuh (tidak ada bukti baru
  yang lolos gate).
- `adv_regime`: tidak dihapus, tetapi labelnya tidak diterbitkan sampai ada riwayat harian
  ≥250 baris; `data_collection_daily.json` akan mencapai itu secara alami.
- Semua komponen display: tetap display-only (tidak ada yang dipromosikan ke scoring).
