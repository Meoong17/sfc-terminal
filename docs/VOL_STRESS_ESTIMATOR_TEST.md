# Uji Estimator Stress Price/Vol — apakah inti era-stable bisa diperkuat?

Dihasilkan: `analysis/vol_stress_estimator_test.py` · run 2026-09-11T08:30:31.672107+00:00 · seed=42 · permutasi=200

Panel: **5357 hari** (2012-01-01 .. 2026-08-31) — Kaggle Bitstamp (2012-01..2017-08) + Binance Vision kanonik (2017-08-17..).

Pertanyaan: apakah salah satu estimator stress alternatif **memperkuat inti** (stress-gauge harga/vol, satu-satunya edge SFC yang era-stable) — atau semuanya sekadar representasi ulang realized-vol?

## Era dan perannya

| Era | Periode | n | Peran |
|---|---|---|---|
| era0 | 2012-01 .. 2015-01 | 1096 | diagnostic |
| era1 | 2015-01 .. 2018-01 | 1096 | diagnostic |
| era2 | 2018-01 .. 2022-01 | 1461 | support |
| era3a | 2022-01 .. 2024-01 | 730 | gate |
| era3b | 2024-01 .. 2099-01 | 974 | gate |

`gate` = penentu verdict (struktur pasar sekarang, era3 dipecah di structural break spot-ETF 2024-01). `support` = penguat. `diagnostic` = konteks saja — TIDAK dipakai membunuh sinyal, karena menuntut tanda sama antara 2013 (bull satu arah) dan 2025 (terinstitusionalisasi) adalah bar yang salah.

## Test A (PRIMER) — gap stress→return forward per era

Top-20% vs bottom-20% dari expanding-z **di dalam** tiap era (threshold-free). Polaritas benar = **negatif** (stress tinggi → return forward lebih buruk).

* = signifikan dgn moving-block bootstrap (blok=h) — menghormati label forward yang tumpang tindih; ~ = signifikan HANYA kalau ketergantungan diabaikan (iid), jadi jangan dianggap bukti.

### Horizon 7d

| Sinyal | era0 2012-14 | era1 2015-18 | era2 2018-22 | era3a 2022-24 | era3b 2024-26 | gate |
|---|---|---|---|---|---|---|
| `rv30` | +9.23* | +4.60* | -0.45 | -0.44 | +1.21 | era3b tidak signifikan / arah salah |
| `semidev30` | +1.80 | +0.47 | -3.97* | -0.84 | -2.53~ | era3b tidak signifikan / arah salah |
| `parkinson30` | +6.55~ | +4.20~ | -1.09 | -1.40 | +0.35 | era3b tidak signifikan / arah salah |
| `gk30` | +6.80~ | +3.09~ | -1.18 | -1.88 | +0.18 | era3b tidak signifikan / arah salah |
| `rs30` | +7.01~ | +2.00 | -1.86 | -1.66 | -0.36 | era3b tidak signifikan / arah salah |
| `jump_ratio30` | -11.11* | -1.96 | -1.18 | +0.35 | +2.10~ | era3b lolos TAPI era3a berbalik arah |
| `vov30` | +4.61~ | +2.00~ | -0.42 | +2.53~ | +0.37 | era3b lolos TAPI era3a berbalik arah |
| `tail30` | +8.48* | +2.94~ | +0.13 | -0.35 | +1.06 | era3b tidak signifikan / arah salah |
| `range30` | +6.79~ | +3.96~ | -1.80 | -0.98 | -0.09 | era3b tidak signifikan / arah salah |
| `maxdd90` | -11.14* | -2.48~ | -2.49~ | -1.47 | -0.68 | era3b tidak signifikan / arah salah |
| `dvol30` | +10.29* | +2.54~ | +0.37 | +1.40 | -1.58~ | era3b lolos TAPI era3a berbalik arah |
| `dvol7` | +4.57~ | -0.24 | +2.47~ | -2.28~ | -0.50 | era3b tidak signifikan / arah salah |
| `mom30` | +15.32* | +2.95~ | +2.23~ | +0.13 | +2.13~ | era3b lolos TAPI era3a berbalik arah |

### Horizon 30d

| Sinyal | era0 2012-14 | era1 2015-18 | era2 2018-22 | era3a 2022-24 | era3b 2024-26 | gate |
|---|---|---|---|---|---|---|
| `rv30` | +5.28 | +8.40~ | +0.03 | -11.64~ | +5.52~ | era3b tidak signifikan / arah salah |
| `semidev30` | -23.25~ | +7.52~ | -6.86~ | -8.63~ | +2.71 | era3b tidak signifikan / arah salah |
| `parkinson30` | +9.43~ | +7.77~ | +0.35 | -14.96* | +5.23~ | era3b tidak signifikan / arah salah |
| `gk30` | +9.04 | +5.71~ | -1.64 | -15.65* | +3.90~ | era3b tidak signifikan / arah salah |
| `rs30` | +8.89~ | +5.56~ | -5.59~ | -15.92* | +2.01 | era3b tidak signifikan / arah salah |
| `jump_ratio30` | -12.09 | -2.29 | +7.07~ | +9.71* | +9.41~ | era3b lolos TAPI era3a berbalik arah |
| `vov30` | +26.04~ | +8.93~ | +7.71~ | -0.35 | -1.76 | era3b tidak signifikan / arah salah |
| `tail30` | +34.82~ | +9.30~ | +7.46~ | -10.07~ | +4.08~ | era3b tidak signifikan / arah salah |
| `range30` | +8.74 | +7.65~ | -3.19 | -15.34* | +5.30~ | era3b tidak signifikan / arah salah |
| `maxdd90` | -81.92* | -1.32 | -8.88~ | -8.18~ | -1.24 | era3b tidak signifikan / arah salah |
| `dvol30` | +70.21* | -2.70 | +6.42~ | +3.20 | -5.61~ | era3b lolos TAPI era3a berbalik arah |
| `dvol7` | +25.59~ | +1.41 | -0.36 | +1.02 | -1.73 | era3b lolos TAPI era3a berbalik arah |
| `mom30` | +83.68* | +3.99 | +11.16* | +5.26~ | +1.54 | era3b lolos TAPI era3a berbalik arah |

### Tolok ukur: inti yang sekarang hidup (`sfc_pct`, replay faktor tereduksi)

| Sinyal | era0 2012-14 | era1 2015-18 | era2 2018-22 | era3a 2022-24 | era3b 2024-26 |
|---|---|---|---|---|---|
| `sfc_pct` 7d |   n/a   | -3.57~ | -1.94~ | -1.66 | -1.83~ |
| `sfc_pct` 30d |   n/a   | -13.73~ | -8.65~ | -7.85~ | -4.47~ |

## Test B (KONFIRMASI) — purged-CV/embargo + kontrol permutasi

Target = P(return forward < 0). K=5 fold kontigu, purge label tumpang tindih + embargo=h, standardisasi fit hanya di train. `ΔAUC` = kenaikan di atas baseline `[rv30, mom30]`. `perm_p` = fraksi null (kolom kandidat diacak, 200×) yang ΔAUC-nya ≥ yang diamati — **wajib**, karena menambah fitur apa pun (termasuk noise) menaikkan AUC OOS ~0.05.

### Horizon 7d (n=5141, base rate negatif=0.4497, baseline AUC=0.534)

| Sinyal | AUC univariat | ΔAUC atas baseline | perm_p |
|---|---|---|---|
| `rv30` | 0.4824 | 0.0 | 0.355 |
| `semidev30` | 0.4925 | -0.0218 | 1.0 |
| `parkinson30` | 0.471 | -0.0099 | 1.0 |
| `gk30` | 0.4681 | -0.0105 | 1.0 |
| `rs30` | 0.4655 | -0.012 | 1.0 |
| `jump_ratio30` | 0.4682 | -0.0136 | 1.0 |
| `vov30` | 0.5123 | 0.007 | 0.0 |
| `tail30` | 0.5157 | 0.0178 | 0.0 |
| `range30` | 0.4755 | 0.009 | 0.005 |
| `maxdd90` | 0.5277 | -0.0015 | 1.0 |
| `dvol30` | 0.534 | 0.0081 | 0.0 |
| `dvol7` | 0.5099 | -0.0075 | 1.0 |
| `sfc_pct` | 0.5102 | -0.0139 | 0.44 |

### Horizon 30d (n=5118, base rate negatif=0.432, baseline AUC=0.5021)

| Sinyal | AUC univariat | ΔAUC atas baseline | perm_p |
|---|---|---|---|
| `rv30` | 0.5139 | -0.0001 | 0.055 |
| `semidev30` | 0.4899 | -0.02 | 0.99 |
| `parkinson30` | 0.4451 | 0.0059 | 0.0 |
| `gk30` | 0.4492 | 0.0048 | 0.005 |
| `rs30` | 0.4521 | 0.0053 | 0.0 |
| `jump_ratio30` | 0.5643 | 0.0319 | 0.0 |
| `vov30` | 0.5626 | 0.043 | 0.0 |
| `tail30` | 0.5476 | 0.0786 | 0.0 |
| `range30` | 0.4503 | 0.0482 | 0.0 |
| `maxdd90` | 0.5154 | -0.0336 | 1.0 |
| `dvol30` | 0.5557 | 0.0577 | 0.0 |
| `dvol7` | 0.4879 | -0.0177 | 0.995 |
| `sfc_pct` | 0.493 | -0.051 | 0.065 |

## Test C — redundansi terhadap realized-vol

Spearman vs `rv30`. Di atas ~0.9 = kandidat praktis representasi ulang realized-vol.

| Sinyal | Spearman vs rv30 |
|---|---|
| `parkinson30` | +0.956 |
| `range30` | +0.944 |
| `gk30` | +0.923 |
| `tail30` | +0.921 |
| `rs30` | +0.893 |
| `semidev30` | +0.853 |
| `vov30` | +0.797 |
| `dvol30` | +0.455 |
| `maxdd90` | +0.344 |
| `jump_ratio30` | -0.017 |
| `dvol7` | -0.049 |

## Verdict

| Kandidat | Lolos gate era | rho vs rv30 | perm_p (7d,30d) | Verdict |
|---|---|---|---|---|
| `dvol30` | tidak | +0.455 | 30d:0.0, 7d:0.0 | REJECT — tidak lolos gate era struktur sekarang |
| `dvol7` | tidak | -0.049 | 30d:0.995, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `gk30` | tidak | +0.923 | 30d:0.005, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `jump_ratio30` | tidak | -0.017 | 30d:0.0, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `maxdd90` | tidak | +0.344 | 30d:1.0, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `parkinson30` | tidak | +0.956 | 30d:0.0, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `range30` | tidak | +0.944 | 30d:0.0, 7d:0.005 | REJECT — tidak lolos gate era struktur sekarang |
| `rs30` | tidak | +0.893 | 30d:0.0, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `semidev30` | tidak | +0.853 | 30d:0.99, 7d:1.0 | REJECT — tidak lolos gate era struktur sekarang |
| `tail30` | tidak | +0.921 | 30d:0.0, 7d:0.0 | REJECT — tidak lolos gate era struktur sekarang |
| `vov30` | tidak | +0.797 | 30d:0.0, 7d:0.0 | REJECT — tidak lolos gate era struktur sekarang |

## Caveat yang wajib dibaca bersama hasil

- **Vol era3b teredam** (vol harian ~2.46% vs ~3.39% era3a / ~4.23% 2017-20) → rentang dinamis sinyal berbasis LEVEL mengecil; gap tipis di era3b bisa berarti rezimnya lebih tenang, bukan sinyalnya mati.
- **Label forward 30d tumpang tindih** (30d berurutan berbagi 29 hari) → `n_eff` ≈ n/(h/2+1). CI di sini memakai moving-block bootstrap (blok=h) untuk itu; tanda `~` menandai sel yang hanya lolos kalau ketergantungan diabaikan.
- **`sfc_pct` = replay faktor tereduksi** (price+DXY+M2+FNG dari cache WFV), bukan replay penuh sistem live yang juga memuat DVOL/on-chain/GLF/dynamic weighting. Dipakai sebagai tolok ukur inti, bukan sebagai deskripsi sistem live.
- **Satu aset (BTC).** Universalitas lintas-aset tidak diuji di sini.

