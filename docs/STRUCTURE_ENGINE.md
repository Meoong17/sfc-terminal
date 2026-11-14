# Structure engine: price formation · financing quality · new demand (2026-11-14)

SFC sebelumnya membaca **likuiditas** dan **perilaku**. Modul ini menambah tiga sumbu struktural —
tetapi diperlakukan sebagai **detektor perubahan**, bukan kumpulan indikator yang menjelaskan keadaan.

## Mengapa "detektor perubahan", bukan level

Level saja tidak berguna untuk keputusan: FQ 50 bisa berarti "normal" atau "sedang memburuk cepat".
Karena itu tiap indeks dipakai melalui **Δ20d**, dan digabung menjadi:

- **SFI (Shift Intensity)** = 0.40·Δ⁻PFS + 0.35·Δ⁻FQ + 0.25·Δ⁻NDS — kecepatan deteriorasi struktur.
- **Kuadran mekanisme**, dari tanda ΔPFS × ΔFQ (+ ΔNDS):
  - CONTRACTION: financing mengetat **dan** buku menipis → awal deleveraging.
  - ABSORPTION RISK: demand baru masuk **ke buku yang menipis** → harga mudah dipaksa.
  - BROADENING: pembiayaan lebih murah **dan** price discovery membaik → peluang sehat.
  - CARRY BID, THIN MARKET: funding murah tapi struktur rapuh → reli yang tidak ditopang.
  - MIXED: layer saling bertentangan (kondisi saat ini).

## Isi tiap sumbu

| Indeks | Komponen | Arah |
|---|---|---|
| PFS | inverse-Amihud impact (20d), stabilitas taker imbalance, close location, breadth volume | tinggi = price discovery dalam & dua arah |
| FQ | level & dispersi funding, CME basis carry, premi MVRV | tinggi = leverage murah & stabil |
| NDS | ETF flow 5d, active addresses, tx count, pertumbuhan jaringan 90d | tinggi = permintaan melebar |

Transformasi: rolling percentile rank (0–100, window 756d, min 120d) agar seri pendek (on-chain 430d,
ETF 700d) tetap terbaca; komposit coverage-aware (bobot direnormalisasi, coverage dilaporkan
PFS 75% / FQ 75% / NDS 100%). Basis futures diberi guard "kontrak basi >4 hari → NaN".

## Hasil (as of 2026-09-30, data harian repo)

- PFS 59 · FQ 50 · NDS 50; Δ20d: PFS **+20.8**, FQ −3.8, NDS +9.9
- SFI **34** ("stable"), kuadran MIXED
- Uji lead (sampel non-overlap 20d): event rate top-quintile SFI 37.5% vs base 21.6% (lift 1.73×)
  tetapi **n=8** → tidak layak disebut leading edge.
- Uji incremental: PFS ΔAUC **+0.068**, perm p **0.375** (gagal gerbang permutasi); FQ ΔAUC −0.033;
  NDS belum punya OOS cukup.

**Verdict: tidak ada indeks yang lolos gerbang prediktif.** Karena itu semuanya display-only dan
tidak masuk `sfc_effective`. Yang ditampilkan di kartu justru angka kegagalan itu (ΔAUC, perm p, n)
supaya pembaca tidak menyimpulkan lebih dari bukti.

## Kaitan dengan instruksi "jangan jadi tumpukan indikator"

Kerangka tetap kausal: tiap layer punya mekanisme yang bisa difalsifikasi (financing → price formation
→ demand), yang diuji adalah **apakah perubahan layer mendahului stres** (event rate + lift + era split),
bukan apakah level-nya berkorelasi dengan harga. Bila uji lead tidak lolos, kartu menyatakannya.

## Operasional

- `analysis/structure_engine.py` → `analysis/structure_engine_results.json`
- `collect.py` membangun `structure` + `structure_html` (markup pra-render, satu substitusi di template)
  dengan `cache_age_hours`/`stale` seperti FLQD.
- Engine dijalankan harian oleh `scripts/run_flqd_engine.sh` (cron 03:15) bersama engine FLQD.
