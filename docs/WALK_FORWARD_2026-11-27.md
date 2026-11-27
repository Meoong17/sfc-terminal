# Walk-Forward Validation — 2026-11-27

Dijalankan setelah audit & perbaikan `docs/AUDIT_KODE_2026-11-27.md`.
Perintah: `.venv/bin/python analysis/walk_forward_validation.py` (dengan `.env` dimuat;
tanpa `FRED_API_KEY` skrip gagal "No BTC price data").
Artefak: `.walk_forward_validation.json` (4343 observasi harian, 2014-12-02 .. 2026-11-26),
`.walk_forward_summary.json`, log `/tmp/wf.log`.

## 0. Batas validitas — baca dulu

Yang diuji skrip ini adalah **proxy inti 3 input**: `calculate_sfc_ensemble()` dengan
faktor `Lt` (btc_24h), `Rt` (FNG), `Sc` (DXY); `St = Ft = 0`, `dom/dvol/pc_oi = None`.
Artinya:

- **Tidak** menguji skor live hari ini (yang memuat 31 metode + blend kausal berbobot +
  penyesuaian ETF/GLF/GSLS/REPO). Perbaikan A1 (weighted mean) dan B1–B13 **tidak**
  tervalidasi oleh run ini.
- **Tidak** bisa dijalankan untuk komposit live: riwayat metode M7–M31 hanya tersedia
  ~12 hari (`data_collection.json`, 2000 obs @6 menit), jauh di bawah kebutuhan
  walk-forward. Jadi pertanyaan "apakah ambang 25/50/75 masih tepat setelah skor turun
  13 pp" **belum terjawab** oleh run ini — hanya bisa dijawab setelah ada ≥1 tahun
  riwayat komposit (atau rekonstruksi metode dari data mentah).
- Angka di bawah tetap berguna: ia menguji apakah **kerangka teoretis** "stres tinggi →
  return forward lebih buruk" punya dukungan empiris, dan di rentang skor mana.

## 1. Hasil pooled (semua era, n=4343)

Threshold yang dipakai skrip: CALM <25, ELEVATED 25–45, STRESS ≥45.

| Bucket | n | mean fwd 7d | mean fwd 30d |
|---|---|---|---|
| CALM | 1791 | +1.96% [+1.58, +2.36] | +8.40% [+7.46, +9.41] |
| ELEVATED | 1294 | +1.17% [+0.75, +1.57] | +7.40% [+6.34, +8.53] |
| STRESS | 1251 | +0.43% [+0.01, +0.84] | +1.78% [+0.91, +2.69] |

Gap CALM−STRESS: **7d −1.53 pp** [−2.10, −0.95], **30d −6.62 pp** [−7.94, −5.35]
(bootstrap iid, CI 90% tidak memuat nol).

**Tetapi** dengan moving-block bootstrap (wajib, karena label 30-hari bertumpang tindih
29 hari), hasil per era menjadi **tidak signifikan di ketiga era**:
`gap_30d_era{1,2,3}_significant_block = false`. Artinya arahnya konsisten di era2/era3,
tetapi besarannya tidak stabil secara statistik. Verdict yang jujur: **arah didukung,
magnitudo tidak**.

## 2. Era split (konvensi repo: era1 2014–17, era2 2018–21, era3 2022–26)

| Era | n | CALM 30d | ELEVATED 30d | STRESS 30d | n(STRESS) | gap 30d |
|---|---|---|---|---|---|---|
| era1 | 1092 | +13.26% | +15.48% | +24.08% | **12** | **+10.82 pp** |
| era2 | 1460 | +8.64% | +2.41% | +4.05% | 506 | −4.59 pp |
| era3 | 1791 | +3.41% | +3.70% | −0.20% | 714 | −3.61 pp |

Cara membaca (aturan IERF, `era-split-validation`): era1 **tidak** membantah hipotesis —
di era1 hanya ada 12 hari STRESS (2014–17 adalah bull tanpa stres terukur pada proxy) dan
CI-nya [−17.4, +17.5] pp, yaitu tidak informatif. Yang informatif adalah era2 dan era3:
keduanya negatif, dan di era3 STRESS bahkan negatif absolut (−0.20%) sementara CALM
+3.41%. Jadi tanda konsisten di era yang punya data; era1 = *uninformative*, bukan flip.

## 3. Non-monotonisitas — temuan paling penting

Decile era3 (30d forward return):

| Desil | rentang sfc | mean 30d |
|---|---|---|
| D1 | 15.0–16.8 | +3.24% |
| D4 | 21.9–27.5 | +2.76% |
| D6 | 32.8–45.5 | +3.76% |
| **D7** | **45.5–53.2** | **−0.90%** |
| D8 | 53.2–55.2 | +0.67% |
| **D9** | **55.2–57.2** | **−1.41%** |
| D10 | 57.3–71.5 | +0.63% |

Di pooled: korelasi peringkat desil vs return **+0.378 (positif)** — hubungan bukan garis
lurus menaik-turun, melainkan **U terbalik**: return terburuk di pita **45–57**, lalu
**pulih** di atas 57. Distribusi proxy: p10 = 16.0, p50 = 28.5, p90 = 56.0, max = 72.6.

Implikasi produk: label "CRITICAL" (>75 di backend, atau >50 di UI) **tidak** boleh
dibaca "makin tinggi = makin buruk". Di atas ~57 forward return justru membaik — ini
konsisten dengan state `⚠ TURNING POINT / REVERSAL FLOOR` yang sudah ada di
`determine_state()`. Zona tinggi = risiko kapitulasi/volatilitas, bukan prediksi
penurunan lanjutan.

## 4. Konsekuensi untuk ambang setelah perbaikan skor

Skor live turun 48.1 → 34.9. Pada distribusi proxy, nilai 34.9 berada di **persentil 61**;
nilai lama 48.1 berada di sekitar persentil 85 (pita D7–D9, yaitu pita return terburuk).
Pemetaan ini **indikatif saja** (proxy ≠ komposit live), tetapi menunjukkan arah: setelah
bobot kausal diperbaiki, pembacaan live tidak lagi "menempel" di pita ekstrem.

Rekomendasi (belum diterapkan — mengubah ambang = perubahan scoring, perlu validasi):
1. Jangan ubah 25/50/75 berdasarkan run ini; yang dibutuhkan adalah WF pada komposit live.
2. Ubah **narasi**, bukan angka: zona HIGH/CRITICAL = "risiko kapitulasi / volatilitas
   tinggi", bukan "harga akan turun". Ini bisa dikerjakan tanpa menyentuh skor.
3. Saat riwayat komposit cukup, uji ulang dengan pita 45–57 sebagai kandidat zona
   "worst forward return", dan >57 sebagai "reversal watch".

## 5. Status yang dipublikasikan ke dashboard

`collect.py` membaca `.walk_forward_summary.json` → field `wfv_gap_30d = −6.62`,
`wfv_gap_7d = −1.53`, `wfv_n_stress_pct = 29.0`, label
`WALK-FORWARD VALIDATED — reduced 4-input proxy (price, DXY, M2, FNG; St/Ft=0), NOT the
full 90+ method live score`. Label itu benar dan dipertahankan: run ini tidak boleh
dipresentasikan sebagai validasi skor penuh.
