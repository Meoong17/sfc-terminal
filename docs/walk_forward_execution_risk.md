# Walk-Forward Validation — Execution Risk De-duplication (2026-08-08)

## Perubahan yang divalidasi (collect.py)

`execution_risk = min(0.40×cascade + 0.30×squeeze + 0.30×funding, 0.95)`

Sebelum (double-count): `cascade = imbalance×0.5 + volume`, `squeeze = imbalance×density`
— asimetri likuidasi (`imbalance ≡ |L−S|/T`) dihitung DUA KALI (bobot efektif 0.70).

Sesudah (orthogonal): `cascade = imbalance` (direction), `squeeze = density` (magnitude),
`funding` (independen). Sinyal likuidasi dihitung TEPAT SEKALI.

## Metode

- Sampel: seluruh riwayat live data.json dari git (11.444 snapshot, 2026-06-10 → 2026-08-07).
- Resample harian → 57 titik (kurangi autokorelasi cron ~5-menit).
- Rekonstruksi execution_risk OLD vs NEW dari field likuidasi (liq_long/short/total) yang
  tersedia sepanjang riwayat (funding=0 — tidak tersedia historis, sama untuk kedua versi).
- Quantile tail gap (top 25% vs bottom 25% exec_risk) untuk forward return 7d/14d
  (polaritas benar = negatif) dan forward realized-vol (polaritas benar = positif).
- Bootstrap numpy-vectorized, RNG seeded, nboot=20.000, CI 90%.
- Redundancy control Version C: NEW di-rescale ke std OLD (uji amplifikasi vs info unik).

## Hasil

| Versi | Horizon | Outcome | Gap top−bottom | 90% CI | Sig |
|---|---|---|---|---|---|
| OLD | 7d | fwd return | −0.0094 | [−0.032, +0.015] | ns |
| NEW | 7d | fwd return | −0.0148 | [−0.038, +0.009] | ns |
| OLD | 14d | fwd return | −0.0091 | [−0.045, +0.028] | ns |
| NEW | 14d | fwd return | −0.0129 | [−0.050, +0.024] | ns |
| OLD | 7d | fwd vol | −0.0005 | [−0.003, +0.002] | ns |
| NEW | 7d | fwd vol | −0.0006 | [−0.003, +0.002] | ns |

- std OLD = 0.1063, std NEW = 0.1068 → **rasio 1.00** (de-duplikasi nyaris tidak mengubah
  amplitudo; kekhawatiran sinyal "mengecil" tak terbukti).
- Version C (NEW rescaled ke std OLD) identik dengan NEW → tak ada perbedaan amplifikasi.

## Verdict: BERTAHAN (no regression) — dengan caveat data-terlalu-pendek

1. **Tidak ada regresi.** NEW mempertahankan (sedikit memperkuat) polaritas prediktif
   forward-return yang benar (negatif, sedikit lebih besar dari OLD di 7d & 14d). Arah
   de-duplikasi aman — komponen ganda lama mayoritas amplifikasi aritmetik, bukan info unik.
2. **Amplitudo stabil (std ratio 1.00).** Menghapus double-count tidak mengecilkan sinyal.
3. **Caveat wajib (data terlalu pendek):** semua CI mencakup nol. Sampel hanya 57 titik harian.
   Ini BUKAN bukti sinyal mati — hanya belum bisa ditegaskan statistik di jendela 2 bulan.
4. **Composite_confidence TIDAK bisa divalidasi:** field execution_risk/macro_confidence
   hanya ada sejak 2026-07-21 (17 hari live). Komposit hanya divalidasi 17 hari = terlalu
   pendek → verdict composite-level: DATA-TOO-SHORT.
5. **Justifikasi struktural berdiri sendiri:** double-count asimetri likuidasi adalah bug
   (bobot 0.70 pada satu sinyal). Fix koreksi ini sah terlepas dari verdict — per skill
   walk-forward: fix integritas data/struktur tidak perlu menunggu bukti prediktif.

## Rekomendasi

- Deploy fix (sudah di collect.py + data.json dikoreksi). Cron pipeline memakai kode ini.
- Jangan blend/naikkan bobot exec_risk atas dasar 57 hari. Beri label validasi tetap
  ESTIMATED untuk komposit, tunggu ≥6 bulan live data untuk kalibrasi cutoff.
- Re-run walk-forward otomatis sebulan sekali (ikuti pola `.walk_forward_summary.json`).
