# Walk-Forward Validation — De-duplication TGA/RRP & liq_mod (2026-08-09)

## Perubahan di collect.py

### Fix #1 — TGA/RRP double-count
**Dihapus** block `FISCAL LIQUIDITY FACTOR ADJUSTMENT` (sebelumnya collect.py ~2303-2308)
yang menambahkan `tga_adj + rrp_adj` ke `factors["Lt"]`.

Alasan: TGA (WTREGEN) dan RRP (RRPONTSYD) SUDAH menjadi komponen di dalam GLF
(`global_liquidity_engine.py`, bobot 10% + 10%), yang juga masuk ke `Lt` via
`glf_factor_adj` (x5.927). Menambahkan M83/M84 lagi = double-count seri FRED yang
IDENTIK (WTREGEN/RRPONTSYD). TGA/RRP kini hidup HANYA di GLF; M83/M84/M85 tetap
dihitung & dilaporkan (display-only).

### Fix #2 — liq_mod (m2_yoy) double-count
**Dihapus** `liq_mod = (7.0 - m2_yoy) * 0.8` dari `effective_sfc`
(sebelumnya collect.py ~2920-2924). `liq_mod` kini = 0.

Alasan: m2_yoy (US M2 YoY, M2SL) SUDAH menjadi komponen GLF (bobot 15%) yang masuk
`sfc_pct` via `Lt`. Menambahkan `liq_mod` langsung ke `effective_sfc` = double-count
M2SL yang sama. m2_yoy juga dipakai fitur HMM (jalur regime) — jadi sebelumnya M2
terhitung 2-3x. Kini m2 hanya via GLF.

### Fix #3 — DXY: TIDAK DIUBAH (keep both)
Analisis inkremental (di bawah) menunjukkan Sc-DXY ADITIF terhadap GLF-DXY
(lensa transmisi berbeda: likuiditas vs struktur korelasi). **Bukan double-count
murni** → dipertahankan.

## Metode
- Rekonstruksi FRED long-history (2015-01-08 → 2026-07-30, n=4215; 4208 dengan fwd-return).
- Formula GLF/FISCAL/liq_mod/Sc disalin verbatim dari modul (bukan import collect.py).
- Faktor lain di-hold 0 (St=Rt=Ft=Sc=0); hanya komponen yang diuji yang bervariasi
  (apples-to-apples, pola lt_redundancy_experiment).
- DXY proxy = FRED DTWEXBGS. China excluded (bobot 0.04).
- Bootstrap numpy-vectorized, RNG seeded (random.seed(42)), nboot=20.000, CI 90%.
- Quantile top-vs-bottom 20% fwd return; polaritas benar = NEGATIF (stress → return lebih rendah).

## Hasil

### Fix #1 — TGA/RRP
| Metrik | OLD (GLF+fiscal) | NEW (GLF only) |
|---|---|---|
| std sfc_pct | 11.60 | 11.85 | 
| std-ratio OLD/NEW | **0.979** | (~1.00, amplitudo terjaga) |
| 30d gap (low−high sfc) | **+6.92 [5.51, 8.34] SIG** | **−2.23 [−3.99, −0.54] SIG** |
| 7d gap | +1.42 SIG | −0.035 ns |

Polaritas OLD **terbalik** (+6.92 = low sfc memprediksi return LEBIH TINGGI — salah
untuk skor stress). Setelah de-duplikasi, polaritas **kembali benar** (−2.23, sig).
Term FISCAL redundan tidak hanya netral — ia justru MERUSAK arah prediksi di
rekonstruksi ini. **Verdict: DEPLOY (fix memperbaiki, bukan sekadar menyamakan).**

### Fix #2 — liq_mod (m2)
| Metrik | OLD (+liq_mod) | NEW (no liq_mod) |
|---|---|---|
| std sfc_pct | 14.30 | 11.85 |
| std-ratio OLD/NEW | **1.206** (NEW ~17% kurang variabel) |
| 30d gap | **−2.30 [−3.92, −0.78] SIG** | **−2.23 [−3.99, −0.38] SIG** |
| 7d gap | −0.27 ns | −0.035 ns |

Polaritas TERJAGA (−2.30 → −2.23, hampir identik). Amplitudo menyusut ~17%
karena liq_mod adalah ~murni amplifikasi (gap prediktif tak berubah). Menghapusnya
tidak menghilangkan info unik. **Verdict: DEPLOY.**

### Fix #3 — DXY incremental (KEEP both)
| Metrik | BASE (GLF-dxy, Sc=0) | AUG (GLF-dxy + Sc-dxy) |
|---|---|---|
| std sfc_pct | 11.85 | 17.66 |
| 30d gap | −2.23 [−4.08, −0.42] SIG | **−3.14 [−5.01, −1.22] SIG** |
| range sfc_pct | [3.81, 30.6] | **[2.12, 56.96]** |

Menambahkan Sc-DXY memperlebar gap prediktif (−2.23 → −3.14) dan memperluas rentang
dinamis skor. Sc-DXY = ADITIF (info unik di luar GLF-DXY). Ini lensa transmisi berbeda
(struktur korelasi DXY-BTC via correlation gate), bukan double-count. **Verdict: JANGAN
hapus — pertahankan keduanya.**

## Caveat (jujur)
- Rekonstruksi parsial (St/Rt/Ft=0, china excluded, DTWEXBGS sbg proxy DXY) — nilai
  gap absolut TIDAK portabel ke live (live pakai 5 faktor penuh + DW + EWMA). Yang
  valid adalah PERBANDINGAN RELATIF OLD vs NEW / BASE vs AUG (same-subsets).
- De-duplikasi #1 & #2 adalah fix struktural (double-count = bug: sinyal yang sama
  diberi bobot 2x). Verdict validasi berdiri terlepas dari edge prediktif.
- Tidak ada CI yang dinaikkan/diturunkan bobotnya berdasarkan jendela ini; skala
  GLF x5.927 dan bobot faktor lain TIDAK diubah (hanya menghapus term duplikat).

## Skrip
- `analysis/dedup_walkforward_2026.py` — rekonstruksi + analisis (FRED long-history).
- Hasil mentah: `.dedup_walkforward_2026.json`
