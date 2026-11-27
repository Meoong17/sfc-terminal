# Walk-Forward pada Komposit Live — percobaan & keputusan (2026-11-27)

Tujuan: menjawab pertanyaan yang gagal dijawab run sebelumnya — "apakah ambang
25/50/75 masih tepat setelah skor turun ~13 pp?" — dengan menguji **skor yang benar-benar
terbit**, bukan proxy 3-input.

Skrip baru: `analysis/wf_live_composite.py`. Ia mengambil fungsi **asli** dari `collect.py`
(`score_factors_from_market`, `calculate_sfc_ensemble`) lewat ekstraksi AST, jadi rumusnya
tidak diduplikasi; ditambah penyesuaian faktor ETF flow (m81) dan riwayat harga
`data_v2/btc_daily.csv` (3386 hari, 2017-08-18 .. 2026-11-24).

## 1. Hasil run (komposit core + ETF, n=3386)

Distribusi: p10 17.1 / p50 35.5 / p90 57.2 / max 74.1.

| Bucket | n | fwd 7d | fwd 30d |
|---|---|---|---|
| CALM (<25) | 1099 | +2.08% | +8.26% |
| ELEVATED (25–50) | 1324 | +0.54% | +4.06% |
| STRESS (≥50) | 933 | +0.58% | +1.71% |

Gap STRESS−CALM 30d = **−6.54 pp**, 90% moving-block CI [−13.53, −0.72] → signifikan
di pooled. Per era: era1 n=136 dengan hanya **3** hari STRESS (tidak informatif),
era2 −3.39 pp (tidak signifikan), era3 −4.47 pp (tidak signifikan; CALM +3.87% vs
STRESS −0.60%). Arah konsisten di era2/era3; magnitudo tidak stabil — sama seperti run
proxy sebelumnya.

Decile 30d: D1 +8.57% → D7 +1.16% (paling rendah) → D10 +2.28%. Non-monoton lagi di
ekor atas (57+), konsisten dengan temuan "reversal band" di `WALK_FORWARD_2026-11-27.md`.

Sweep ambang atas (informasi, tidak diterapkan): gap vs CALM stabil di −5.7..−7.1 pp
untuk ambang 40–60, jadi tidak ada ambang tunggal yang jelas optimal.

## 2. TEMUAN PENTING — rekonstruksi TIDAK melacak angka live

Dibandingkan angka live yang benar-benar terbit (6 hari tumpang tindih dari git history
`data.json`, 17–24 Nov 2026):

| Tanggal | live `sfc_effective` | rekonstruksi | selisih |
|---|---|---|---|
| 2026-11-17 | 34.61 | 61.32 | +26.71 |
| 2026-11-18 | 38.21 | 66.14 | +27.94 |
| 2026-11-19 | 36.82 | 63.20 | +26.38 |
| 2026-11-20 | 50.30 | 67.56 | +17.26 |
| 2026-11-21 | 51.88 | 54.36 | +2.48 |
| 2026-11-24 | 46.12 | 65.87 | +19.76 |

MAE **20.09 pp**, bias **+20.09 pp**, korelasi **−0.205**.

Konfirmasi dari arsip siklus live 16:18: grup core yang sebenarnya
`g_m1m6 = 0.3445` (34.5%), sedangkan rekonstruksi membaca ~63. Jadi perbedaan bukan
soal level ambang, tetapi input: komposit live memakai 31 metode + penyesuaian
on-chain/GLF/GSLS/REPO + gerbang korelasi DXY, sementara rekonstruksi hanya bisa
menyertakan harga, FNG, DXY, M2, dan ETF.

**Kesimpulan yang harus dipegang: ambang zona TIDAK BOLEH dikalibrasi dari rekonstruksi
ini.** Dua percobaan rekonstruksi (proxy 3-input dan core+ETF) sama-sama hanya
memvalidasi *kerangka*, bukan skor terbit.

## 3. Tindakan yang diambil (real fix)

Karena riwayat skor live hilang terus-menerus (`data_collection.json` bergulir 2000 baris
≈ 12 hari), `collect.py` kini menulis arsip append-only:
`data/live_sfc_history.jsonl`, satu baris per siklus, berisi
`ts, sfc, sfc_base, zone, conf, regime, g_m1m6, g_m7m19, g_m20m31, p1, p2, p3,
m1..m6, hmm_ok, wfv_gap30`.

Contoh baris pertama (2026-11-27T08:18Z):
`sfc 35.49, g_m1m6 0.3445, g_m7m19 0.4357, g_m20m31 0.4333, p1 0.885, p2 0.072, p3 0.043`.

Implikasi: 3–6 bulan lagi (≈26k–52k baris) walk-forward pada **skor live sebenarnya**
bisa dijalankan tanpa rekonstruksi — dan baru pada titik itu pertanyaan ambang 25/50/75
dapat dijawab secara sah. Tidak ada perubahan pada skor, ambang, atau bobot faktor.

## 4. Ringkasan verdict

| Pertanyaan | Jawaban |
|---|---|
| Arah "stres tinggi → return forward lebih buruk" | Didukung (gap 30d −6.5 s/d −6.6 pp di dua konstruksi berbeda) |
| Magnitudo stabil lintas era? | Tidak (block bootstrap per era tidak signifikan) |
| Hubungan monoton? | Tidak — return terburuk di pita ~42–57, pulih di atas 57 |
| Ambang 25/50/75 terbukti optimal? | Belum — dan tidak bisa diuji dengan rekonstruksi |
| Jalan keluar | arsip `data/live_sfc_history.jsonl`, WF ulang setelah ≥3 bulan |
