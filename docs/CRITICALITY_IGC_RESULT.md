# IGC — Information-Geometric Criticality (metode matematika baru) + hasil pengujian

_Tanggal: 2026-09-11. Status: **TIDAK di-blend** (metode baru, hasil jujur apa adanya)._

## Ringkasan satu paragraf

Kerangka matematika baru untuk SFC dibangun dari empat besaran teori-sistem-dinamik /
geometri-informasi yang **belum pernah ada di repo ini** — Fisher Information Measure
(FIM) dari densitas return, jarak geodesik **Fisher–Rao** antar-distribusi window,
**transfer entropy asimetris** (arus informasi nonlinier berarah), dan **critical
slowing down** (ρ₁, skew) — plus eksponen **Lyapunov Rosenstein** sebagai diagnostik.
Semua besaran dihitung point-in-time (window trailing, grid KDE tetap, tanpa
fitting label). Komposit pra-spesifikasi **IGC** diuji dengan baterai standar repo
(purged-CV + embargo, era-split, block bootstrap, BH-FDR) **ditambah uji yang belum
pernah dipakai di sini: null surrogate IAAFT** (100 surrogate yang mempertahankan
spektrum daya + distribusi amplitudo, menghancurkan struktur nonlinier) dan uji
inkremental atas incumbent volatilitas-21d dengan CI berpasangan.
**Hasil: komposit IGC dan semua komponennya GAGAL gate → NOT_BLEND.** Satu komponen
(fim_excess) lolos 3 dari 4 kriteria (AUC purged 0.712, p_surrogate 0.000, ΔAUC
+0.374 atas vol) tetapi **gagal era-stability** (era3 tidak signifikan) → tetap
NOT_BLEND, dicatat sebagai hipotesis, bukan sinyal.

## Definisi matematis (semua point-in-time, tidak ada look-ahead)

| Besaran | Definisi | Catatan verifikasi |
|---|---|---|
| `fim_excess` | `I[p] − 1`, `I[p] = ∫ (p'(x))²/p(x) dx`, KDE Gaussian (Silverman) atas return terstandardisasi, grid tetap [−5,5]×512 + trapesium | N(0,1) → **1.0142** (teori 1.0); N(0,9) → 0.1096 (teori 1/9=0.1111); Student-t(3) → 1.8888 (>1) |
| `fr_drift` | `d(p,q) = arccos ∫ √(pq) dx` antara densitas window t−1 dan t, standardisasi **pooled** | cocok rumus tertutup Gaussian: varians×4 → 0.4662 vs 0.4663; geser mean → 0.2471 vs 0.2490; campuran → 0.7161 vs 0.7168; invarian skala eksak |
| `te_asym` | `TE(tanda→|r|) − TE(|r|→tanda)`, bit, histogram kuantil 3-simbol, koreksi Miller–Madow, bentuk entropi `H(Y⁺,Y)−H(Y)−H(Y⁺,Y,X)+H(Y,X)` | sistem terkopel: TE(x→y)=0.2603 vs TE(y→x)=0.0023; independen 0.0032; leverage sintetis → +0.1254 |
| `rho1`, `skew` | autocorr lag-1 return; skewness (critical slowing down) | definisi standar |
| `lam1` | Lyapunov terbesar, Rosenstein (1993), delay-embedding Takens | logistic map r=4 → **0.7351** (teori ln2=0.6931) |
| `igc` | rata-rata peringkat-persentil **expanding** dari {fim_excess, fr_drift, te_asym, ρ₁, −skew}; bobot sama, pra-spesifikasi | — |
| null | `iaaft()` — 100 surrogate, spektrum daya & distribusi amplitudo identik | distribusi eksak terjaga, korelasi spektrum 1.00000, autocorr |x| asli +0.9985 vs surrogate −0.5751 |

Self-test lengkap: `.venv/bin/python analysis/criticality_math.py --selftest` → **LULUS**
(semua ambang teori/analitik).

## Baterai pengujian

Data: BTC kanonik Binance Vision, 2017-08-18 .. 2026-07-31, **3.270 hari**, window 250
(~8 bulan), label forward 7/30/90 hari, 100 surrogate IAAFT (stride 5, 599 titik).
Kriteria verdict **dipre-spesifikasi sebelum melihat hasil**: (1) purged-CV AUC > 0.55
dan CI-lower > 0.50; (2) **era2 DAN era3 signifikan dengan tanda sama**; (3)
p_surrogate_IAAFT < 0.05; (4) ΔAUC atas vol-21d > 0 dengan CI-lower > 0.

### Tabel 1 — IC / gap / purged-CV / null

| sinyal | IC30 | gap30 | purged AUC 30d | purged AUC 90d | p_surrogate IC30 |
|---|---|---|---|---|---|
| `fim_excess` | +0.341 (q=0.000) | +18.13pp | 0.627 [0.564, 0.698] | **0.712 [0.634, 0.786]** | **0.000** |
| `fr_drift` | +0.302 (q=0.000) | +17.40pp | 0.618 [0.553, 0.696] | 0.668 [0.587, 0.744] | 0.010 |
| `te_asym` | −0.102 (q=0.000) | −7.35pp | 0.441 | 0.291 | 0.370 |
| `rho1` | −0.177 (q=0.000) | −11.16pp | 0.530 | 0.485 | 0.110 |
| `skew` | +0.006 (q=0.772) | −2.85pp | 0.428 | 0.327 | — |
| `lam1` | +0.137 (q=0.000) | +3.90pp | 0.459 | 0.395 | — |
| **`igc`** (komposit) | +0.076 (q=0.000) | +1.04pp | **0.476 [0.411, 0.542]** | 0.477 | **0.290** |

(BH-FDR: 16 dari 21 sel q<0.10 — IC screen "lolos" untuk banyak sel; justru karena itu
gate purged-CV + era-split + surrogate yang menentukan.)

### Tabel 2 — era-split (gap top/bottom-20% @30d, p_sign)

| sinyal | era1 (2017-08..2020) | era2 (2021..2023) | era3 (2024..2026) | era2&3 signifikan & sama tanda |
|---|---|---|---|---|
| `fim_excess` | +32.21 (p=1.000) | +19.33 (p=0.998) | **+4.80 (p=0.782)** | **TIDAK** (era3 tidak signifikan) |
| `fr_drift` | +23.53 (p=1.000) | +14.68 (p=1.000) | +6.22 (p=0.894) | TIDAK (era3 tidak signifikan) |
| `te_asym` | −10.66 (p=0.062) | −1.29 (p=0.415) | −9.37 (p=0.068) | TIDAK |
| `rho1` | −22.27 (p=0.000) | −1.70 (p=0.445) | −3.32 (p=0.274) | TIDAK |
| `igc` | −7.70 (p=0.341) | **+8.79** (p=0.907) | **−8.15** (p=0.102) | TIDAK (tanda era2 ≠ era3) |

### Tabel 3 — inkremental atas incumbent (vol realisasi 21d, mask OOS identik)

| sinyal | ΔAUC 30d [CI] | ΔAUC 90d [CI] | partial-IC vs vol 90d |
|---|---|---|---|
| `fim_excess` | +0.171 [+0.074] | **+0.374 [+0.263]** | +0.491 (p=0.00) |
| `fr_drift` | +0.157 [+0.066] | +0.325 [+0.184] | +0.392 (p=0.00) |
| `igc` | +0.026 [−0.035] | +0.152 [+0.066] | +0.220 (p=0.00) |
| `te_asym` | −0.007 [−0.038] | −0.036 [−0.070] | −0.012 (p=0.50) |
| `rho1` | +0.064 [−0.014] | +0.118 [+0.038] | −0.245 (p=0.00) |

Redundansi: korelasi Spearman `fim_excess` vs vol-21d = **−0.012** (bukan proksi
volatilitas); vs `fr_drift` +0.230; `igc` vs `te_asym` +0.655.

### Tabel 4 — event study drawdown ≥15% (tugas early-warning)

8 event (2017-09, 2017-12, 2021-01, 2021-04, 2021-11, 2024-03, 2025-01, 2025-10; dd
0.25–0.83). Sinyal pada jendela pra-onset [onset−30, onset−5] dibanding distribusi
unconditional: `fim_excess` persentil 0.448 (p=0.461); `igc` persentil 0.224 (p=0.482);
`vol21` persentil 0.520 (p=0.522). **Tidak ada bukti early-warning** — tetapi n=6-8
event → daya uji sangat rendah; hasil ini "tidak terkONfirmasi", bukan "terbukti tidak ada".

## Verdict (kriteria pre-spesifikasi)

| sinyal | (1) AUC | (2) era2&3 | (3) surrogate | (4) ΔAUC>vol | PASS | keputusan |
|---|---|---|---|---|---|---|
| `igc` (primer) | ✗ 0.480 | ✗ | ✗ p=0.290 | ✗ | **TIDAK** | NOT_BLEND |
| `fim_excess` (sekunder) | ✓ 0.712 | **✗** (era3 p=0.78) | ✓ p=0.000 | ✓ +0.374 [+0.263] | **TIDAK** | NOT_BLEND |

**Keputusan: JANGAN blend IGC / komponennya ke `sfc_effective`, jangan tambah field
scoring, jangan ubah dashboard.** Tidak ada perubahan scoring di commit ini (sesuai
aturan: tidak ubah scoring sebelum walk-forward validated; dan di sini gate-nya gagal).

## Temuan yang layak dicatat (dan batasnya)

1. **Fisher-information excess adalah kandidat nyata, bukan noise linier.** Ia
   mengalahkan null IAAFT (IC30 real +0.336 vs surrogate +0.007±0.114 → p=0.000;
   gap30 +18.75pp vs +0.45pp → p=0.000), **tidak** berkorelasi dengan vol realisasi
   (−0.012), dan menambah ΔAUC +0.374 atas vol pada 90d (CI-lower +0.263). Namun
   efeknya **meluruh antar-era** (+32 → +19 → +4.8pp) dan **tidak signifikan di era3**
   → persis pola yang membuat repo ini menolak faktor lain. Tanpa era-stability,
   ini hipotesis, bukan sinyal.
2. **Arah efek berlawanan dengan prior teoretis saya.** Saya berhipotesis "FIM
   tinggi = rapuh → return ke depan lebih rendah". Empiris: **FIM tinggi → return ke
   depan lebih TINGGI**. Prior tidak dikonfirmasi; ini contoh kenapa orientasi
   komposit tidak boleh dipercaya tanpa uji.
3. **Komposit pra-spesifikasi (rata-rata sama) merusak sinyalnya sendiri.** Komponen
   terpecah polaritas: `te_asym` (pembawa bobot terbesar, korelasi +0.655 dengan
   komposit) **gagal null** (p=0.29-0.37) dan arahnya berlawanan dengan
   `fim_excess`, sehingga komposit turun ke AUC 0.476 (di bawah koin). Pelajaran:
   pra-spesifikasi itu benar, tapi komposit rata-sama atas komponen dengan polaritas
   berbeda = peredam informasi.
4. **`te_asym`, `rho1`, `skew`, `lam1` TIDAK lolos null IAAFT** — asosiasi
   full-sample-nya bisa direproduksi oleh surrogate yang hanya mempertahankan
   autokorelasi linier ⇒ tak ada kandungan nonlinier mandiri. Ini kelas kegagalan
   yang tidak bisa ditangkap IC-screen maupun purged-CV biasa.
5. **Tidak ada early-warning** pada event drawdown (n kecil → tak konklusif).
6. **`lam1` (Rosenstein) tervalidasi untuk sistem deterministik (0.735 vs ln2 0.693)
   tetapi pada deret stokastik ia mengukur dekorrelasi noise, bukan kontraksi
   deterministik** (AR(1) φ=0.9: teori −0.105, estimasi +0.27) ⇒ sengaja **tidak**
   dimasukkan ke komposit.

## Reproduksi

```
.venv/bin/python analysis/criticality_math.py --selftest          # verifikasi matematis (LULUS)
.venv/bin/python analysis/validate_igc_criticality.py             # baterai penuh (~11 menit; 100 surrogate)
IGC_N_SURR=20 .venv/bin/python analysis/validate_igc_criticality.py   # versi cepat
```
Output: `analysis/.validate_igc_criticality.json` (deterministik; seed 20260911).

## Yang bisa mengubah verdict (syarat eksplisit)

1. `fim_excess` signifikan di era3 pada panel yang lebih panjang / spesifikasi
   regime-conditional (mis. di-gate oleh vol atau regime HMM) **dengan** era-stability
   era2&era3 → baru boleh dibahas sebagai kandidat blend.
2. Uji ulang komposit dengan polaritas yang ditetapkan per-komponen berdasarkan teori
   yang **diuji lebih dulu** (bukan diasumsikan), dan bobot yang tidak didominasi
   komponen yang gagal null.
3. Untuk early-warning: tunggu lebih banyak event drawdown (≥15 event) agar daya uji
   memadai.
