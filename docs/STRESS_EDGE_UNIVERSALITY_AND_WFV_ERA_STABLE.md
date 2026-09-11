# Apakah edge stress→return SFC universal? — uji empat sumbu

_Dihasilkan 2026-09-11 · skrip: `analysis/cross_asset_universality.py`,
`analysis/vol_stress_estimator_test.py`, `analysis/reconcile_stress_gap_frames.py`,
`analysis/fetch_binance_vision_symbol.py`_

## Jawaban ringkas

**Tidak — di keempat sumbu.** Edge ini bukan hukum perilaku pasar; ia properti
terkondisi struktur. Yang penting untuk SFC: **inti `sfc_pct` tetap satu-satunya
sinyal yang bertanda benar di seluruh era pasca-2015**, sesuatu yang tidak dicapai
satu estimator tunggal mana pun. Tidak ada kandidat pengganti yang memperkuatnya.

## Metode

Gap = rata-rata return forward pada top-20% vs bottom-20% dari expanding-z
**di dalam tiap era** (threshold-free; bucket absolut pincang lintas-era).
Polaritas benar = **negatif**. `*` = signifikan dengan moving-block bootstrap
(panjang blok = h, menghormati label forward yang tumpang tindih); `~` = hanya
signifikan kalau ketergantungan diabaikan (iid) — jangan dianggap bukti.

Era: era0 2012-14 · era1 2015-18 · era2 2018-22 · era3a 2022-24 · era3b 2024-26.
Era3 dipecah di structural break spot-ETF 2024-01.

## Sumbu 1 — lintas-waktu (BTC, 5.357 hari)

```
sinyal         h      era0      era1      era2     era3a     era3b
rv30          7d    +9.23*    +4.60*    -0.45     -0.44     +1.21
rv30         30d    +5.28     +8.40~    +0.03    -11.64~    +5.52~
parkinson30  30d    +9.43~    +7.77~    +0.35    -14.96*    +5.23~
tail30       30d   +34.82~    +9.30~    +7.46~   -10.07~    +4.08~
dvol30        7d   +10.29*    +2.54~    +0.37     +1.40     -1.58~
semidev30     7d    +1.80     +0.47     -3.97*    -0.84     -2.53~
maxdd90       7d   -11.14*    -2.48~    -2.49~    -1.47     -0.68
sfc_pct       7d      n/a     -3.57~    -1.94~    -1.66     -1.83~
sfc_pct      30d      n/a    -13.73~    -8.65~    -7.85~    -4.47~
```

Di era0/era1 hampir semua ukuran stress bertanda **positif** dan beberapa lolos
block-bootstrap. Di bull ~2.800x, vol tinggi = beli saat diskon. Edge ini properti
struktur pasca-2018, bukan keteraturan lintas-sejarah.

`sfc_pct` satu-satunya baris yang negatif di **keempat** era pasca-2015 di kedua
horizon — termasuk era0/era1 di mana hampir semua estimator tunggal bertanda
salah. Jadi sifat lintas-era ada pada **kombinasi**, bukan pada komponen.

## Sumbu 2 — lintas-definisi

12 estimator price/vol pada era yang sama memberi tanda berbeda (lihat tabel di
atas: era3a 30d keluarga level-vol −12 s/d −16, sementara `jump_ratio30` +9.71*
berlawanan arah). Definisi yang berbeda → kesimpulan yang berbeda; itu sendiri
bukti melawan universalitas.

## Sumbu 3 — lintas-horizon

`semidev30` lolos di 7d (era3b −2.53) tapi bertanda salah di 30d (+2.71).
`jump_ratio30` berlawanan tanda antara 7d dan 30d. Tidak ada sinyal yang konsisten
di kedua horizon.

## Sumbu 4 — lintas-aset (BTC vs ETH vs SOL)

Estimator identik (impor `build_signals` yang sama), sumber harga sama
(Binance Vision klines 1d), cakupan penuh sampai 2026-08-31
(BTC 5.357 hari, ETH 3.302, SOL 2.212).

```
sinyal         h      BTC(era3b)   ETH(era3b)   SOL(era3b)
rv30          7d      +1.21        -3.22~       -5.00*
rv30         30d      +5.52~       +2.14       -17.21~
parkinson30  30d      +5.23~       +2.07       -13.51~
tail30        7d      +1.06        -3.30~       -4.55*
tail30       30d      +4.08        -3.97       -22.62*
dvol30        7d      -1.58~       -1.79~       -3.76*
dvol30       30d      -5.61        -7.29       -20.02*
```

**Jendela bersama 2024-01-01..2024-12-31 (ketiganya punya data penuh):**

```
sinyal         h         BTC        ETH        SOL
rv30          7d      -3.61~     -8.02*    -10.16*
rv30         30d     -24.73*    -21.12*    -45.93*
parkinson30  30d     -23.43*    -20.74*    -33.21*
tail30       30d     -23.17*    -17.12*    -36.18*
```

Interpretasi:
- **Di jendela identik (2024), ketiga aset sepakat** — semuanya negatif, mayoritas
  signifikan. Pada 2024, edge-nya lintas-aset.
- **Di era3b penuh (2024-2026) kesepakatan itu pecah**: SOL tetap negatif kuat,
  BTC berbalik positif, ETH lemah/ambigu. Divergensi terjadi di 2025-2026.
- Kesimpulan: universalitas lintas-aset **bergantung periode**. Klaim sebelumnya
  ("BTC outlier di era3b") adalah **artefak jendela** — ETH/SOL waktu itu hanya
  punya 2024 sementara BTC punya 2024-2026. Koreksi ini penting dan sudah
  diperbaiki dengan mengunduh ulang cakupan penuh.

## Uji perkuatan inti — tidak ada kandidat yang lolos

Tabel lengkap: `docs/VOL_STRESS_ESTIMATOR_TEST.md`. Ringkas:

- **purged-CV pooled (2012-2026)** justru menghadiahi sinyal yang skill-nya hidup di
  era0/era1 — `jump_ratio30` (AUC univariat 0.5643, ci_lo 0.5215, ΔAUC +0.032,
  perm_p 0.00) bertanda **salah** di era3a/era3b (+9.71*, +9.41). Sama untuk
  `tail30` (+0.079), `dvol30` (+0.058), `vov30` (+0.043).
- **Pooled-AUC adalah metrik yang salah** untuk model yang harus bekerja di
  struktur sekarang: ia memberi hadiah atas skill rezim lama. Uji per-era
  disediakan di `analysis/purged_cv_era.py`.
- Di era sekarang hanya `semidev30`@7d lolos gate (−2.53 era3b, −0.84 era3a),
  dan ia **tidak menambah apa pun** di atas baseline `[rv30, mom30]`
  (ΔAUC −0.022, perm_p 1.0) serta sangat redundan (Spearman 0.853 vs rv30).
- `sfc_pct` sendiri di purged-CV pooled: uni AUC 0.493, ΔAUC −0.051.

## TEMUAN TERPISAH — label `era_stable` tidak reprodusibel (perlu tindakan)

Saat memverifikasi ulang, ditemukan bahwa badge "era-stable ✓" pada dashboard
tidak stabil antar-run.

Fakta terverifikasi (`analysis/walk_forward_validation.py`):

1. Prosedurnya: `bootstrap_diff_ci(..., ci=0.90)` → **90% satu-sisi**
   (`significant = hi < 0`), bootstrap **iid** (`random.randrange`, tanpa seed di
   jalur ini) dan **tanpa penanganan label forward yang tumpang tindih**.
2. Memanggil `w._gap_stats(series, 7, '2022-01-01', '2099-01-01')` langsung pada
   data cache, berulang:
   - 15 panggilan → `hi_90` ∈ [−0.0769, −0.0133], `sig=True` 15/15
   - 1 panggilan → `hi_90 = +0.00101`, `sig=False`
   Jadi ambang keputusan duduk **di ≈ 0** dan verdict flip mengikuti undian
   bootstrap. Marginnya −0.04 pada gap −0.68: nyaris tanpa ruang.
3. Untuk perbandingan, era2 kokoh (`hi_90 = −0.2004`, est −1.3386) dan
   full-sample kokoh (CI90 [−2.0626, −0.8933]).
4. Re-run penuh skrip (n_periods 4256 → 4266) **mereproduksi** `era_stable: True`,
   jadi ini bukan cache basi — dua kali berturut-turut keluar True, tapi
   prosedurnya sendiri tidak deterministik.
5. Dengan moving-block bootstrap (blok=h, menghormati tumpang tindih) pada data
   yang sama: p(gap ≥ 0) = 0.957 (7d) dan 1.000 (30d) untuk era3 → **tidak
   signifikan sama sekali** pada 95%. Untuk era3b: 0.909 dan 1.000.

Dampak: badge dashboard menyatakan era-stable tanpa kualifikasi, padahal margin
keputusannya ≈ 0 pada CI 90% satu-sisi dan verdict-nya dapat berubah antar-run.

## Verdict

| Pertanyaan | Jawaban |
|---|---|
| Universal lintas-waktu? | **Tidak** — tanda terbalik di era0/era1 |
| Universal lintas-definisi? | **Tidak** — 12 estimator, tanda berbeda di era yang sama |
| Universal lintas-horizon? | **Tidak** — 7d vs 30d berlawanan untuk beberapa sinyal |
| Universal lintas-aset? | **Tergantung periode** — sepakat di 2024, pecah di 2025-2026 |
| Ada kandidat yang memperkuat inti? | **Tidak** — hanya `semidev30`@7d lolos gate, dan tidak menambah |
| Badge `era_stable` layak apa adanya? | **Belum** — non-deterministik + margin ≈ 0 |

## Rekomendasi (belum ada yang diterapkan; skor tidak disentuh)

1. **Seed bootstrap** di `walk_forward_validation.py` (semua jalur) supaya verdict
   deterministik — ini juga tuntutan yang sudah tercatat di skill.
2. **Jangan jadikan `hi < 0` pada CI 90% satu-sisi sebagai bukti era-stable**
   tanpa margin minimum; laporkan `hi_90` apa adanya di kartu supaya pembaca
   melihat marginnya.
3. Pisahkan klaim "signifikan" dari "era-stable": full-sample dan era2 kokoh,
   era3 marginal — label harus mencerminkan perbedaan itu.
4. Pertahankan arsitektur inti (ensemble), jangan tukar gauge dengan satu
   estimator yang menang di satu rezim.

## Caveat

- Vol era3b teredam (vol harian ~2.46% vs ~3.39% era3a / ~4.23% 2017-20) →
  rentang dinamis sinyal berbasis level mengecil.
- BTC memakai seri gabungan Bitstamp (2012-2017) + Binance Vision (2017+);
  ETH/SOL murni Binance Vision. Overlap BTC sudah divalidasi (mean rel.diff 0.190%).
- `sfc_pct` = replay faktor tereduksi dari cache WFV, bukan replay penuh sistem
  live. Dipakai sebagai tolok ukur inti, bukan deskripsi sistem live.
- Uji lintas-aset mencakup tiga aset besar; stablecoin/aset non-kripto tidak diuji.

## PERBAIKAN DITERAPKAN (2026-09-11) — verdict deterministik + margin diekspos

1. **Determinisme.** `analysis/walk_forward_validation.py`: RNG bootstrap di-seed
   (`BOOTSTRAP_SEED = 42`, `random.Random(...)` lokal per panggilan). Sebelumnya
   tanpa seed → verdict bisa berubah antar-run pada data yang sama.
   Dibuktikan: dua run penuh berturut-turut → 62 kunci identik (kecuali `generated_at`).
2. **Uji yang menghormati label tumpang tindih.** Fungsi baru
   `bootstrap_diff_ci_block()` + parameter `block` di `_gap_stats()`: moving-block
   bootstrap (blok = horizon).
3. **Margin disimpan, bukan disembunyikan.** Cache kini memuat
   `gap_{h}d_{era}_ci_lo/_ci_hi`, `..._significant_block`, `..._ci_hi_block`,
   `..._margin_block` (margin = −ci_hi_block; **negatif = uji blok tidak lolos**).
4. **Definisi utama `era_stable` TIDAK diubah** (tetap uji standar/iid) supaya label
   historis tidak berubah diam-diam; varian konservatif `era_stable_block` disediakan
   sebagai pembanding.
5. `collect.py` meneruskan `wfv_gap_{7d,30d}_era3_significant_block`,
   `_era3_margin_block`, `_era_stable_block` ke `data.json`.
6. Kartu WFV di `index.html` menampilkan baris **Regime check (era2 vs era3)** plus
   peringatan ⚠ bila margin era3 negatif. (index.html ber-flag skip-worktree →
   berlaku live tapi tidak ikut commit.)

Hasil (dua run identik):

```
horizon  era    est(pp)   sig_iid   margin_block   era_stable   era_stable_block
7d       era2    -1.34    True        -0.91          True          False
7d       era3    -0.68    True        -0.75          True          False
30d      era2    -4.59    True        -6.51          True          False
30d      era3    -4.43    True        -1.21          True          False
```

Interpretasi: full-sample tetap kokoh; era2 dan era3 lolos uji standar TAPI margin
block-nya negatif — di bawah uji yang menghormati label forward tumpang tindih,
keduanya tidak mengecualikan nol. Karena itu klaim era-stability kini ditampilkan
**bersama marginnya**, bukan sebagai badge kosong.

**Belum diputuskan (menunggu pemilik model):** apakah `era_stable` utama harus
dipindah ke definisi blok (konsekuensinya menjadi `False` untuk era2 DAN era3).
Saya tidak melakukannya sepihak karena (a) itu membatalkan klaim era-stable inti yang
selama ini tercatat sebagai satu-satunya konfirmasi terverifikasi, dan (b) hasil blok
sangat bergantung pilihan panjang blok (= horizon) — **sensitivitas panjang blok wajib
diuji lebih dulu** sebelum dijadikan label utama. Selama itu, field `_block` dan
`_margin_block` sudah tersedia untuk pemeriksaan.

---

# KOREKSI PENTING — pooled-purged-CV menyembunyikan kandidat terbaik

Hasil uji **per-era** (`analysis/purged_cv_era.py`, 200 permutasi, output
`.purged_cv_era.json`) **membatalkan kesimpulan "tidak ada kandidat"** di bagian
verdict sebelumnya. Pooled-AUC bukan cuma "menghadiahi rezim lama" — ia membuang
kandidat yang justru spesifik struktur sekarang.

Baseline `[rv30_z, mom30_z]`, era3b (2024-2026, n≈967):

```
sinyal        h    uni_AUC   ci95_lo    ΔAUC     perm_p
maxdd90       7d    0.5786    0.5375   +0.1257   0.00
semidev30     7d    0.5063    0.4564   +0.1081   0.00
rs30          7d    0.5334    0.4504   +0.0790   0.00
maxdd90      30d    0.6752    0.5838   +0.0818   0.00
dvol30       30d    0.6013    0.4795   +0.0814   0.00
```

Bandingkan pooled (semua era, dari `docs/VOL_STRESS_ESTIMATOR_TEST.md`):
`maxdd90` ΔAUC 7d **−0.0015** / 30d **−0.0336** (perm_p 1.0) → ditolak.
Era yang sama, sinyal yang sama, verdict berlawanan.

**`maxdd90` (kedalaman drawdown dari puncak 90 hari) kini kandidat terkuat:**
- polaritas benar di SEMUA era modern (7d: era2 −2.49, era3a −1.47, era3b −0.68);
  hanya terbalik di era0/era1 (era yang polaritasnya memang terbalik untuk semua);
- satu-satunya kandidat dengan `ci95_lo` AUC univariat **> 0.5** di era3b
  (7d 0.5375, 30d 0.5838);
- ΔAUC +0.126 (7d) / +0.082 (30d) dengan perm_p 0.00;
- **tidak redundan** dengan realized-vol (Spearman hanya 0.344 vs rv30) —
  berbeda dari keluarga range-vol (0.89-0.96) yang praktis duplikat.

**Tetap JANGAN blend sebelum uji lanjut.** Caveat yang mengikat:
1. 5 era × 2 horizon × 13 sinyal = **130 sel** → sebagian signifikan bisa
   kebetulan; jumlahnya jauh melebihi ~6 yang diharapkan acak, jadi bukan
   seluruhnya noise, tapi klaim per-sel tetap perlu koreksi multiplisitas.
2. **Dispersi antar-fold besar** di era3b 30d (0.276 / 0.350 / 0.487 / 0.642 /
   0.686) → estimasi pooled 0.6752 rapuh; jangan dibaca sebagai presisi.
3. n era3b ≈ 967 hari dengan label 30d tumpang tindih → effective n jauh lebih kecil.
4. Verdict per-era lain juga bergerak (`jump_ratio30` era3a 30d AUC 0.7496) —
   pola lintas-era masih tidak konsisten, sesuai temuan universalitas di atas.

**Status: kandidat untuk walk-forward penuh (purged-CV + embargo lebih ketat,
sensitivitas panjang blok, dan uji inkremental terhadap `sfc_pct` sendiri) —
BUKAN siap blend. Skor SFC tidak disentuh.**

## Catatan operasional — edit dashboard WAJIB di master

`sfc-pipeline.sh` memulihkan `index.html` dari `/home/ubuntu/index.html` setiap
siklus (5 menit), dan `scripts/deploy.sh` menyalin master yang sama. Edit langsung
ke `index.html` repo **pasti hilang** pada siklus berikutnya. Perbaikan kartu WFV di
atas diterapkan ke master, lalu direstore + inject ke repo
(`cp /home/ubuntu/index.html index.html && python3 inject_data.py data.json index.html`).
Diverifikasi: `node --check` lolos pada dua blok JS, injeksi 451 field, dan baris
"Regime check (era2 vs era3)" ada di `index.html` hasil.
