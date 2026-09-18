# Audit Kode Model SFC — 2026-09-18

Audit menyeluruh terverifikasi atas repo `/home/ubuntu/sfc` (skoring, metode, data, ML/statistik, pipeline).
Data acuan: `data.json` ts **2026-09-18T09:10–09:22Z**, 60 snapshot `git show <hash>:data.json`,
`sfc-pipeline.log` (2542 siklus), cache runtime, `.causal_cache.json`, `data_collection.json` (2000 obs).
Metode: 4 sub-audit paralel (1 timeout) → **setiap klaim berdampak skor saya verifikasi ulang sendiri**
ke `file:line` + angka artifact. Status tiap temuan ditandai:

- `[V]` = saya verifikasi langsung (kode + angka).
- `[R]` = laporan sub-audit, belum saya cek ulang penuh (severity rendah / non-scoring).

Tidak ada kode produksi yang diubah dalam audit ini. `git status` bersih kecuali `M data.json`
(pipeline live menulis siklus baru saat audit berjalan).

---

## 0. Verdict ringkas

| # | Temuan | Severity | Kena skor live? | Status |
|---|--------|----------|-----------------|--------|
| A1 | Vektor fitur kausal kehilangan `m8_s` → bobot kausal M8–M30 milik metode tetangga (off-by-one) | **CRITICAL** | YA | `[V]` |
| A2 | Metrik likuidasi on-chain bernilai 0 → skor 100 → `St` +1.191 | **CRITICAL** | YA | `[V]` |
| A3 | Checkpoint QLSTM dilatih target sirkular (`val_loss` 8.1e-08) tapi men-drive nudge 5% | HIGH | YA (−1.15pp) | `[V]` |
| A4 | Label HMM = peringkat return (BEAR = +0.12%/hari), `crisis_prob` 0.00 di 2542/2542, HMM men-override regime 743× | HIGH | YA (severity/mult) | `[V]` |
| A5 | `execution_risk` kolaps: `funding_imbalance` & `squeeze_magnitude` = 0 di 60/60 → hanya cascade | HIGH | YA | `[V]` |
| A6 | M13 funding selalu gagal (endpoint `start_timestamp=0`) → fallback 0.5 × bobot 2.0 → clamp 1.0 | HIGH | YA (~1.1pp) | `[V]` |
| A7 | Clamp `min(1.0, skor×bobot)` mengubah konstanta & data hilang jadi stres maksimum | MED | YA | `[V]` |
| A8 | Severity konsensus rezim yang ditampilkan ≠ yang dipakai untuk ambang zona | MED | Tampilan | `[V]` |
| A9 | Sinyal likuidasi/pendanaan masuk dua kanal `composite_confidence` | MED | Laten | `[V]` |
| A10 | MPI: komponen gagal diisi 0.50 berbobot penuh + penalti `_pen_mpi` belum pernah menyala | MED | YA (kecil) | `[V]` |
| A11 | `confidence_components` mencampur 3 besaran; DVOL tampil dua kali; penalti RSI tidak tampil | MED | Tampilan | `[V]` |
| A12 | `EWMA` correction inert tapi `ewma_available: true`; instance yang disimpan salah | MED | Tidak (kini) | `[V]` |
| A13 | Faktor memakai SMA 30 hari tapi UI menampilkan nilai spot (faktor tidak dapat direproduksi pembaca) | MED | Tampilan | `[V]` |
| A14 | Sensitivitas faktor dekat saturasi (Rt 0.62 dari plafon; dRt/dFNG 0.0575) → skor tidak responsif harian | LOW | Tidak langsung | `[V]` |
| B1–B10 | Metode rusak/konstan yang saat ini berbobot 0 (M8 satuan bps vs persen, M12 tanpa `open`, M10/M11/M16 ambang skala bulanan, M21/M22/M23/M25/M30/M31 ambang di luar rentang fisik) | MED–HIGH | Tidak (kini) — **jadi YA bila bobot kausal di-refit** | `[V]` |
| C1–C8 | Cache non-atomic (23 penulis) + `timeout 150` SIGKILL; stablecoin refresh TTL tanpa data; alphractal 33 hari + wipe-on-failure; 404 SPX 2540×; drift/DQ mati tapi `available:true`; sigma metode beda satuan (floor 3.0); `ml_ensemble_confidence` bukan keyakinan | MED–LOW | Tidak | `[V]`/`[R]` |

Inti: **rantai aritmetika skor benar, tapi dua cacat masuk skor live hari ini** (A1–A2), tiga cacat
lain menyentuh skor secara struktural (A3–A6), dan lapisan metode berisi ~10 metode rusak yang
"aman" hanya karena bobot kausalnya 0 — rapuh terhadap satu refit.

---

## A. Temuan yang menyentuh skor live

### A1. `[V]` CRITICAL — Vektor fitur kausal kehilangan `m8_s`, seluruh bobot bergeser satu metode
`collect.py:2963-2966` (daftar eksplisit) + `analysis/causal_inference.py:169-176` (pemetaan posisional):

```python
all_method_scores = []
for s in [m1_klr, m2_logit, ..., m6_regime/100,
          m7_s, m9_s, m10_s, ..., m19_s]:        # ← m8_s TIDAK ADA
    all_method_scores.append(s if s is not None else 0.5)
...
n_methods = len(features[0])                     # = 30
for i, name in enumerate(METHOD_NAMES[:n_methods]):   # METHOD_NAMES panjang 31
    series = [obs[i] ...]                        # nama ke-i ← kolom ke-i
```

Bukti:
- `data_collection.json`: **panjang vektor 30**, `METHOD_NAMES` **31** (`analysis/causal_inference.py:29-37`).
- Warna kolom konsisten dengan pergeseran 1 langkah: kolom 7 konstan `0.5` (= m9_liquidity null→0.5, dibaca
  sebagai m8_yield), kolom 14 konstan `0.45` (= m16, dibaca sebagai m15), kolom 24 ∈ {0.205, 0.235} (= m26,
  dibaca sebagai m25), kolom 27 ∈ {0.7, 0.85} (= m29, dibaca sebagai m28), kolom 29 ∈ {0.15, 0.3} (= m31,
  dibaca sebagai m30); kolom 18/19 berisi seri m20/m21.
- `m31_altman` tidak punya kolom sama sekali (`METHOD_NAMES[:30]`) → bobotnya jatuh ke default.

Konsekuensi terukur:
- `m13_funding` (mati) mendapat bobot **2.0** dari seri m14 → `0.5 × 2.0` → **clamp 1.0**.
- `m29_debt` (0.85) juga ter-clamp **1.0** (bobot 2.0).
- Reproduksi mandiri saya (data.json + `.causal_cache.json`) memberi `new_avg = 0.445`, `inst_avg = 0.183`
  — **sama persis** dengan `data.json` (0.445 / 0.183, aktif 6/11). Mengeluarkan `m13_funding` →
  `new_avg = 0.334` (Δ −0.111 → **≈ −1.11 pp pada sfc_base**, p2 = 0.10). Mengeluarkan `m29_debt` →
  `inst_avg = 0.101` (≈ −0.41 pp).

Perbaikan: susun vektor dari dict bernama (bukan posisi), tambahkan asersi
`len(vector) == len(METHOD_NAMES)`, dan uji regresi kolom↔nama. **Semua keputusan bobot kausal
saat ini tidak sah** sampai ini diperbaiki lalu di-refit.

### A2. `[V]` CRITICAL — Likuidasi on-chain bernilai 0 dibaca sebagai skor 100 → `St` +1.191
`data_sources/onchain_fetch.py:250-271` (`_compute_percentile`) + `collect.py:1131-1132`:

```python
below = sum(1 for v in sorted_vals if v < value)
percentile = below / n
elif direction == "neg":
    score = (1 - percentile) * 100        # value 0 → percentile 0 → score 100
...
ms_adj = (onchain_market_structure - 50) / 50 * 1.5
factors["St"] += ms_adj
```

Bukti live: `q10_details.long_liquidations_usd = {value: 0, score: 100}` dan
`short_liquidations_usd = {value: 0, score: 100}` (tanggal 2026-09-18, `status: "ok"`),
`q10_market_structure = 89.7`, log `[OnChain] Market structure=89.7 → St adj=+1.191`.
Dua angka 100 ini menyumbang bobot 0.45 dari pilar market-structure.
`_compute_percentile` hanya menjaga `len(data_vals) < 10` — nilai 0 (placeholder "tidak ada likuidasi")
tidak dibedakan dari "likuidasi nol = bullish maksimum".

Perbaikan: untuk metrik likuidasi, `value == 0` → `unavailable` (jangan dihitung sebagai skor).
Karena ini mengubah `St` (score-affecting), delta wajib diukur + walk-forward sebelum deploy
(aturan: jangan ubah skoring tanpa re-validasi).

### A3. `[V]` HIGH — Checkpoint QLSTM dari target sirkular, tapi nudge 5% aktif
`models/qlstm_model.py:14-20` menyatakan target lama bersifat sirkular
("Training target changed from a circular linear-combination-of-inputs … delete it and retrain;
the old weights are not …"), dan `collect.py:2767-2768`:
`qlstm_adjustment = qlstm_diff * 0.05; sfc_pct += qlstm_adjustment`.
Bukti: `qlstm_model.pt` (root, mtime **2026-07-16**) berisi `val_loss = 8.111320681791767e-08`
(R²≈1 = jejak target sirkular). Log live: `QLSTM enhanced: raw=4.6 garch+-5.330 adapt=0.0 adj=-1.15pp → 26.4%`.
Catatan: path yang diklaim sub-audit (`models/qlstm_model.pt`) tidak ada — checkpointnya di root repo;
substansi klaim tetap benar setelah koreksi path.

Perbaikan: matikan nudge M32 sampai checkpoint hasil retrain target harga-outcome terpasang, atau
tandai versi target di checkpoint dan tolak checkpoint lama saat inference.

### A4. `[V]` HIGH — HMM: label = peringkat return, `crisis_prob` selalu 0.00, tapi men-override regime
Label dipaksakan monotonik (`models/hmm_regime.py:141-148`, argsort mean return). Dari `models/hmm_regime.pkl`
(`_scaler_mean[0] = 0.0915`, `_scaler_std[0] = 1.7434`) saya hitung ulang mean return harian per klaster:

| Label | Mean return harian |
|-------|--------------------|
| BULL | +0.2962% |
| **BEAR** | **+0.1225%** |
| SIDEWAYS | −0.0289% |
| CRISIS | −0.0544% |

"BEAR" diberi state dengan return harian **positif**, dan CRISIS hanya 0.025 pp/hari di bawah SIDEWAYS.
Log: `crisis_prob=0.00` pada **2542/2542** siklus; distribusi label BULL 1202 / BEAR 709 / SIDEWAYS 631;
`override=BEAR` 451× dan `override=STRESS` 292× (`collect.py:3098-3100` meng-override `regime`).
Karena `regime` → `regime_consensus` → `_REGIME_DRIVER_MULT` (0.7/1.0/1.2) → ambang zona & bobot DW,
label HMM yang tidak informatif ini menyentuh skor.

Perbaikan: perlakukan HMM sebagai display (seperti `adv_regime`), atau ganti pelabelan dengan ambang
return/vol absolut + walk-forward sebelum dipakai sebagai driver.

### A5. `[V]` HIGH — `execution_risk` tinggal satu faktor
Komentar `collect.py:3589-3633` menyatakan `R = 0.40·cascade + 0.30·squeeze + 0.30·funding`.
60/60 snapshot: `squeeze_magnitude = 0.0` **selalu**, `funding_imbalance = 0.0` **selalu** (`exec_risk` ∈ 0.0–0.4,
seluruh variasinya dari `cascade_risk`). Penyebab:
- `collect.py:3601-3606`: begitu `liq_total_24h` tersedia (jalur OKX nyata), `_squeeze_magnitude = 0.0`
  hardcoded → dua cabang `elif` (liq_density, RSI-extremity) menjadi dead code di produksi.
- `collect.py:3620-3624`: `_funding_imbalance` hanya dari `m13_d['funding_rate']`, dan m13 selalu `None`.

Perbaikan: putuskan apakah squeeze/funding memang ditiadakan (hapus dari rumus + komentar) atau
dihidupkan kembali dengan sumber yang benar; jangan biarkan dokumentasi menjanjikan tiga faktor
sementara artifact hanya punya satu.

### A6. `[V]` HIGH — M13 funding tidak pernah berhasil → 0.5 → bobot 2.0 → clamp 1.0
`collect.py:1533`: `get_funding_rate_history?currency=BTC&start_timestamp=0&end_timestamp=now` (pitfall lama
Deribit: `start_timestamp=0` + tanpa `instrument_name` → data kosong/timeout), sementara M25 memakai jendela
7 hari dan berhasil. `m13_funding = None` di 40/40 snapshot → `method_scores_dict["m13_funding"] = 0.5`
(`collect.py:2640`) → bobot kausal 2.0 → **clamp 1.0** (lihat A7). Dampak terukur: −1.11 pp pada `sfc_base`
bila dikeluarkan (A1).

Perbaikan: perbaiki parameter endpoint (atau ambil `current_funding` dari order book seperti M25), dan
ubah fallback "data hilang" dari `0.5` menjadi **dikeluarkan dari rata-rata**.

### A7. `[V]` MED — Clamp per-metode mengubah konstanta & data hilang menjadi stres maksimum
`analysis/causal_inference.py:409`: `adjusted = min(1.0, max(0.0, score * w))`.
Contoh live: `m29_debt 0.85 × 2.0 = 1.7 → 1.0`; `m13_funding 0.5 × 2.0 → 1.0`. Akibatnya satu metode
konstan dapat menyumbang nilai maksimum ke rata-rata grup (`inst_avg` 0.188 → 0.107 tanpa m29).
Perbaikan: rata-rata berbobot ternormalisasi (`Σ(s·w)/Σw`) alih-alih clamp per-metode.

### A8. `[V]` MED — Severity konsensus yang ditampilkan ≠ yang menggerakkan ambang zona
`collect.py:3117` memanggil `consolidate_regime(..., behavior_state=None)` (**dipakai skor**),
`collect.py:4083` memanggil ulang dengan `behavior_state=_behavior_state` dan **menimpa** variabel yang
di-emit; `_REGIME_DRIVER_MULT` dihitung di `3140-3150` dari panggilan pertama.
Harness saya dengan input live: panggilan pertama → `ELEVATED` sev **55** → mult **1.0**;
panggilan kedua (yang di-emit) → `STRESSED` sev **70** → pembaca menyimpulkan mult 0.7.
Frekuensi: **1 dari 60** snapshot (hanya saat behavior ekstrem seperti hari ini), jadi dampaknya
adalah kejujuran tampilan, bukan bias harian.

### A9. `[V]` MED — Sinyal yang sama masuk dua kanal `composite_confidence` (laten)
- Likuidasi: `cascade_risk` → `execution_risk` (`collect.py:3628-3633`) **dan** MPI dibangun dari
  `liq_long_vol/liq_short_vol/liq_total_24h` yang sama (`collect.py:3499-3503`) → `_pen_mpi` (`3571`).
- Pendanaan: `_funding_imbalance` (m13) **dan** komponen `funding_rate` MPI.
Inert hari ini (`_pen_mpi = max(0,(0.35−0.5)·0.08) = 0`), tapi saluran kedua hidup begitu
`mpi_stress ≥ 0.70` (peta diskret 0.20/0.35/0.50/0.70/0.85 di `data_sources/market_positioning_index.py:272-282`).
Ini kelas yang sama dengan perbaikan de-dup 2026-08 — perbaikannya belum menyentuh kanal MPI.

### A10. `[V]` MED — MPI: imputasi netral berbobot penuh + penalti belum pernah menyala
`.mpi_cache.json` → `components.funding_rate = {score: 0.5, weight: 0.25}` padahal
`raw_data.funding_rate = null` (bobot terbesar diisi netral, tanpa flag `unavailable`).
Sepanjang 60 snapshot `mpi_stress` hanya bernilai 2 unik (0.35 / 0.5) → `_pen_mpi` **0/60 kali menyala**
(penalti melompat 0 → 0.016 → 0.028 karena peta bertingkat, bukan kontinu).

### A11. `[V]` MED — `confidence_components` tidak dapat dijumlahkan dan sebagian bukan besaran yang dipakai
`collect.py:4293-4308`: `"rsi": rsi_conf` dan `"dvol": dvol_conf` memakai variabel gaya "delta keyakinan"
(`3374-3401`) yang **tidak masuk** `macro_confidence`; yang masuk adalah `_pen_rsi`/`_pen_dvol_safety`
(`3551-3567`). DVOL tampil dua kali (`dvol` + `vol_penalty`); `method_agree` ditampilkan mentah (0.408)
bukan kontribusinya (0.408×0.15 = 0.061); penalti RSI tidak tampil di komponen mana pun.
Kartu UI merender dict ini (`app.js:778` `Object.entries(d.confidence_components)`).
Komentar `3549-3552` ("penalty dihitung SEKALI dan dipakai computation + display") tidak berlaku untuk RSI/DVOL.

### A12. `[V]` MED — EWMA "online learning" inert tapi dilaporkan aktif
`models/ewma_state.json` = `{"baseline": 0.0, "history": []}`; 2542/2542 baris log identik
(`[EWMA] Corrected: 12.7% → 12.7%`, `26.4% → 26.4%`); `data.json` tetap `ewma_available: true`.
`ewma_corrected: true` dengan koreksi nol = klaim adaptif yang tidak berdasar.

### A13. `[V]` MED — Faktor memakai SMA 30 hari, UI menampilkan nilai spot (tidak dapat direproduksi pembaca)
`collect.py:2135-2151` mengoper **rata-rata 30 hari** ke `score_factors_from_market`, sedangkan dashboard
menampilkan nilai spot. Bukti reproduksi persis dari `.daily_market_cache.json`:

| Faktor | Rumus aktual | Hasil hitung | data.json |
|--------|--------------|--------------|-----------|
| Rt | `sigmoid(fng_30d=72.77, 50, k=0.08)` + whale_adj(55.4 → +0.216) | **+2.3804** | +2.3804 |
| Sc | `-sigmoid(dxy_30d=99.21, 100, k=0.2)` ×0.5 (rezim MIXED, corr 0.005) | **+0.1183** | +0.1183 |

Artinya: pembaca yang mencoba mereproduksi faktor dari angka di dashboard (FNG spot 68, DXY spot 100.15)
akan meleset — Rt seharusnya +2.067 (selisih 0.314), Sc +0.047. Perbaikannya: tampilkan nilai input yang
benar-benar dipakai (30d-SMA) di samping spot, atau beri label "faktor memakai SMA 30 hari".

### A14. `[V]` LOW — Sensitivitas faktor dekat saturasi → skor tidak responsif pada rentang normal
`Rt = +2.380` dari plafon clamp ±3.0 (jarak 0.620) dengan `dRt/dFNG = 0.0575` per poin
(gerakan 10 poin FNG = 0.575 pada skala ±3). Dalam 10 entri `.factor_history.json`, rentang gerak:
Rt 0.0019, Sc 0.0000 (1 nilai unik), St 0.0006, Ft 0.0012, Lt 0.0039 — semuanya < 0.5% rentang faktor.
Ini **konsekuensi desain SMA 30 hari**, bukan input mati (lihat A13), tetapi berarti klaim dashboard
tentang perubahan sentimen harian praktis tidak tercermin di faktor.

---

## B. `[V]` Lapisan metode: rusak/konstan, "aman" hanya karena bobot kausal 0

Semua di bawah ini terverifikasi dari kode + nilai 40 snapshot. **Peringatan**: bobot kausal berasal dari
`.causal_cache.json` yang TTL-nya 24 jam dan di-refit otomatis; satu refit dapat menyalakan kembali
metode-metode ini beserta bug-nya (A1 membuat pemetaan bobot tidak sah, sehingga risiko ini nyata).

| # | Metode | Bukti (40 snapshot) | Cacat |
|---|--------|---------------------|-------|
| B1 | M8 Yield (HY spread) | `spread` ∈ {2.7 ×18, 2.8 ×20, **300** ×1, None ×1} | `BAMLH0A0HYM2` bersatuan **persen**, ambang 200/300/400 mengasumsikan **bps** → cabang kredit mati; fallback `300` masuk cabang `>200` sehingga **menaikkan** skor tanpa data (`collect.py:1439-1446`) |
| B2 | M12 Jump risk | `jump_risk = 0.0`, `gap_count = 0` di 40/40 | `_binance_klines` (`collect.py:921`) hanya menyimpan `time/close/volume`; `opens = c.get("open", closes[i-1])` → `gap ≡ 0` (`collect.py:1514`) |
| B3 | M10 GARCH | `0.85` di 40/40 | `persist = alpha + beta` **hardcode 0.99**, bukan hasil fit; ambang `curr_vol > 0.03` memakai skala harian pada return bulanan (`collect.py:1486`) |
| B4 | M11 VaR/ES | `0.85` di 40/40 (ES −0.19) | ambang `es < -0.15` skala harian pada ES bulanan (`collect.py:1502`) |
| B5 | M16 Markov regime | `p_crisis = 0.45` di 40/40 | `r_std > 0.025` skala harian vs `r_std ≈ 0.126` bulanan (`collect.py:1608`) |
| B6 | M21 Large trade flow | `0.30` di 38/40 | jendela 100 trade + ambang $50k ≈ 0.64 BTC → hampir selalu cabang default (`data_sources/methods_institutional.py:281`) |
| B7 | M22 Spread momentum | `spread_bps = 0.0013` di 40/40 | ambang 0.5–10 **bps** mustahil untuk BTCUSDT (tick $0.01 = 0.0013 bps); skor hanya bergerak dari penalti range (`methods_institutional.py:342-348`) |
| B8 | M23 Liquidity fractals | `levels_used_buy = 999` (sentinel) di **23/40**, skor 0.80 di 39/40 | target notional $1jt dibanding depth 100 level; fallback "slippage = 5.0" hardcode (`methods_institutional.py:448`) |
| B9 | M25 Minsky | `|funding|` maksimum 4 snapshot teratas = 9.6e-05 | ambang 0.01/0.005/0.002 → 20–100× di atas rentang fisik; hanya NORMAL/DISPLACEMENT tercapai (`methods_institutional.py:600`) |
| B10 | M30 Rajan, M14 Skew, M31 Altman | fsi 0.15 40/40; `skew = -0.0` 40/40 (`put_iv == call_iv`); `x2 = 4.0` & `x4` clamp 10 40/40 | ambang bertingkat tanpa gradasi / rata-rata seluruh strike put-vs-call / konstanta hardcode 5.8 dari z≈7.7 (`methods_institutional.py:917-960`, `collect.py:1569-1574`, `methods_institutional.py:1035-1074`) |

Konstansi 11 metode yang dilaporkan audit 2026-09-09 **terkonfirmasi** pada 40 snapshot terbaru
(m8, m14, m15, m17, m19, m23, m24, m28, m29, m30, m31). Klaim "m33_glo orphaned" juga terkonfirmasi:
`m33_glo` hanya ditulis (`collect.py:2652`), tidak pernah dibaca.
Klaim sub-audit bahwa **M81/M82 menggeser faktor** juga benar (`collect.py:2389`: `m81` → `Rt`,
`m82` → `Lt`), namun pada snapshot ini keduanya 0.5 sehingga tidak menggeser — jadi "display-only"
tidak sepenuhnya tepat dan label perlu dikoreksi.

---

## C. `[V]`/`[R]` Lapisan data & sistem

| # | Temuan | Severity | Status |
|---|--------|----------|--------|
| C1 | **23 penulis cache** di `data_sources/` memakai `json.dump` langsung tanpa `os.replace`/`tempfile` (0 occurrence) → cache terpotong bila proses di-SIGKILL; pipeline memakai `timeout 150` (`sfc-pipeline.sh:25,32`) | MED | `[V]` |
| C2 | `stablecoin_liquidity.py:203-207`: gagal total → `cache["cached_at"] = now` (TTL "disembuhkan") tanpa data baru; live: `supply_history` hanya **3 entri** (2026-08-19, 09-11, 09-18) padahal `cached_at` segar | MED | `[V]` |
| C3 | Provenance stablecoin hilang: guard "hari lengkap" diukur terhadap `len(all_series)` (subset yang berhasil), bukan 4 koin tetap → hari 2 koin bisa lolos dan menimpa riwayat 4 koin (`stablecoin_liquidity.py:216-235`) | MED | `[R]` |
| C4 | `data/alphractal_daily.json` umur **33.4 hari** (mtime 2026-08-16) vs TTL 6 jam, dan `fetch_all()` menulis `{}` saat gagal total → berpotensi menghapus riwayat riset 1.68 MB (`data_sources/alphractal.py:127-170`) | MED | `[V]` (umur), `[R]` (wipe) |
| C5 | `drift_detection` & `data_quality` tidak pernah menyimpan state (file state tidak ada) → `drift_available/dq_available: true`, `drift_stable: true` adalah nilai default, bukan hasil hitung | MED | `[V]` (file tidak ada), `[R]` (mekanisme) |
| C6 | `prob_uncertainty_breakdown.method_disagreement_sigma = 3.0` = floor: `_PROB_METHOD_SCORES` diskalakan 0–1 (`collect.py:3825-3837`) sedangkan `mu` 0–100 → sigma metode ≈ 0.29 lalu dipaku ke 3.0; `final_sigma` 12.33 hampir seluruhnya dari `confidence_sigma` (sirkular) | MED | `[V]` |
| C7 | `ml_ensemble_confidence = abs(p−0.5)×2` (`models/ml_ensemble.py:658`) → 0.994 untuk prediksi 0.003; bukan probabilitas terkalibrasi | LOW | `[V]` |
| C8 | Twelve Data `SPX` 404 di **2540** baris log (tiap siklus, tanpa negative-cache; SPY sebagai proxy senyap); liquidation_client hanya menandai 401-dalam-200, 403 diulang diam-diam (`market_data_fetcher.py:190-219`, `liquidation_client.py:193-202`) | LOW | `[V]` (jumlah log) |
| C9 | `hybrid_correction` masih memakai bobot QIGWO basi (M5 0.24 vs kode 0.23) + target sirkular → `m32_garch_residual = −5.33`, `m32_hybrid_pred = −0.81` (stres negatif) ditampilkan ke dashboard | LOW | `[R]` |
| C10 | `qlstm_enhanced.py:90-93` menormalisasi input dengan statistik live tiap siklus (bukan statistik training) + padding 0.5 sebelum z-score — input model yang men-drive nudge 5% (A3) | MED | `[R]` |

---

## D. Diverifikasi BERSIH (jangan "diperbaiki" tanpa bukti baru)

1. **Publikasi `data.json` atomic + tervalidasi**: `sfc-pipeline.sh:32-46` menulis ke `data.json.tmp`,
   validasi `json.load`, baru `mv -f`; gagal → "keeping previous data.json". Log menunjukkan
   **0** kejadian "keeping previous" → selama 2542 siklus tidak pernah ada data korup yang terbit.
2. **Integritas artifact**: `head -c 40 data.json` = `{`, 0 `JSONDecodeError` di log —
   jadi kekhawatiran "stdout contamination" (23 modul punya `print()` tanpa `sys.stderr`, `redirect_stdout`
   hanya ada di `models/qlstm_enhanced.py:148`) **belum termaterialisasi**: print modul-modul itu berada di
   jalur `main()`/CLI, dan pipeline tetap menggagalkan siklus bila stdout tercemar. Risiko = data basi, bukan JSON rusak.
3. **Rantai aritmetika keyakinan**: `cc_base 0.4200 → macro_confidence 0.380 → ×(1 − 0.40) = 0.228` = nilai emit.
   `_pen_yield = 0.04` (slope M8 < 0.5) satu-satunya penalti aktif.
4. **Skor ensemble dapat direproduksi persis** dari `data.json` + `.causal_cache.json`:
   `new_avg 0.445 / inst_avg 0.183 / aktif 6-11` (angka yang sama), sehingga klaim dampak A1/A6 punya dasar.
5. **Gating modul tidak tervalidasi sudah benar**: `xgb_blend_weight = 0.0` (`collect.py:3204`),
   `adv_regime` tidak dipublikasikan (`adv_regime_reliable: false`, 46 baris < 250), adv boost tidak diterapkan,
   `m65` disabled, Mamba inert (`m32_mamba_active: false`).
6. **Jalur GLF → Lt benar**: `get_glf_for_factors(0.5) = 0.0` → dengan `glf_stress = 0.5` GLF tidak menggeser Lt;
   `m90_gsls 40.8 → Lt −0.184` sesuai rumus `(score−50)/50`, keduanya di dalam clamp ±3.0 (`collect.py:2433-2587`).
7. **HMM tidak punya scale mismatch fit-vs-serve**: fitur fit == serve (`btc_24h` mentah; `scaler_mean[0]=0.0915`
   cocok dengan `btc_24h` live 2.2% pada skala std 1.74); `nan_to_num` ada di jalur prediksi.
8. **Guard staleness yang sudah ada**: ETF `ETF_MAX_STALE_DAYS=10` (flow terbaru 1 hari), China M2 3 bulan,
   `repo_market_stress` gagal → tidak menulis cache, `expectations_engine` memakai `status: partial` +
   daftar `unavailable`, `onchain_fetch` fallback per-metrik ke cache lama (tidak menghapus), 7 modul sinyal
   ber-cache `key`+`ts` dan tidak menimpa cache saat gagal.
9. **Klaim ML jujur**: `ml_accuracy = 1.0` selalu disertai `ml_accuracy_reliable: false`,
   `ml_stress_events: 0`, dan `bt_calibration_note` "calm-majority artifact"; `wfv_label` menyatakan
   dirinya proxy 4-input, bukan skor live.

---

## E. Rekomendasi berprioritas

Status: per 2026-09-18 hanya **dua perbaikan non-scoring di lapisan operasi** yang sudah diterapkan
(G1 sinkronisasi + propagasi exit kode skrip training mingguan; G2 `chown sse_server.log`). Tidak ada kode
skoring yang diubah.

**P0 — perbaikan non-scoring (bisa langsung, tanpa walk-forward):**
1. Cache atomic: semua `_save*` di `data_sources/` → pola `tmp + json.load validasi + os.replace` (C1).
2. `stablecoin_liquidity.py`: jangan perbarui `cached_at` saat gagal; ekspos `n_coins`/`source` (C2, C3).
3. `alphractal.py`: jangan menulis `{}` saat gagal; hormati TTL (C4).
4. Persist state `drift_detection`/`data_quality` atau set `available: false` selama histori < 10 (C5).
5. Perbaiki satuan/ambang metode berbobot 0 (B1, B6–B10) + hapus fallback numerik yang menyamar sebagai data.
6. `m31_altman` tidak punya kolom kausal (A1) — perbaiki pemetaan sebelum menyentuh bobot apa pun.

**P1 — kebenaran tampilan (non-scoring):**
7. Perbaiki `confidence_components` (A11): satu representasi saja, tandai mana yang masuk rumus.
8. Selaraskan severity konsensus yang di-emit dengan yang dipakai (A8): emit severity *scoring* + tandai
   severity *tampilan* secara terpisah.
9. `ml_ensemble_confidence` → `ml_margin_0_5`; `ewma_available` hanya true bila koreksi ≠ 0 (C7, A12).

**P2 — score-affecting; wajib kuantifikasi delta + walk-forward sebelum deploy:**
10. A1 (vektor kausal off-by-one) — **paling penting**: tanpa ini, seluruh bobot kausal & klaim "metode mana
    yang aktif" tidak sahih. Perbaiki pemetaan → refit → ukur ulang `new_avg`/`inst_avg`/`sfc_base`.
11. A2 (likuidasi 0 → 100 → `St` +1.191) — guard nilai 0 sebagai `unavailable`, lalu ukur delta.
12. A6/A7 (M13 endpoint + fallback 0.5 + clamp) — perbaiki endpoint, ubah fallback menjadi "keluarkan",
    ganti clamp dengan rata-rata berbobot ternormalisasi.
13. A3 (nudge QLSTM) — matikan sampai checkpoint target harga-outcome terpasang.
14. A4 (label HMM sebagai driver) — jadikan display-only atau ganti ambang pelabelan + walk-forward.
15. A5 (execution_risk) — putuskan hidup/mati squeeze & funding, lalu selaraskan komentar/rumus.

**P3 — kebersihan:** D3 (`liq_mod` mati di 60/60, masih di rumus), imputasi `0.5` untuk skor hilang
(`collect.py:2729-2731` vs `apply_filter` yang melewati `None`), 404/403 diam (C8).

---

## G. Lapisan operasi, dashboard, dan penyajian publik (`[V]`, workstream lanjutan)

Audit higiene operasi + kontrak field dashboard + lapisan publik. Dua item KRITIS sudah **saya perbaiki**
(ditandai DIPERBAIKI); sisanya laporan temuan tanpa perubahan.

### G1. `[V]` KRITIS (DIPERBAIKI) — Pelatihan mingguan gagal total tapi dilaporkan `ok` selama berbulan-bulan
Cron `7c8b06a1fddf` ("SFC Weekly Heavy Train", `0 3 * * 0`) mengeksekusi salinan Hermes
`~/.hermes/scripts/sfc-weekly-train.sh`, **Last run: 2026-09-13 ok**. Isi run terakhir:
`can't open file '/home/ubuntu/sfc/train_mamba.py'` (tidak ada; file sebenarnya `models/train_mamba.py`)
→ `[MAMBA] ❌ Training failed (exit=2)`; langkah 2 → `ModuleNotFoundError: No module named 'qlstm_model'`;
lalu skrip tetap mencetak `✅ Weekly heavy training complete` **dan tidak pernah `exit 1`** → Hermes mencatat `ok`.

Bukti tambahan (verifikasi saya): fungsi yang dipanggil langkah 2 **tidak ada** —
`from qlstm_enhanced import build_training_data` → `ImportError: cannot import name 'build_training_data'`
(`models/qlstm_enhanced.py` 10.788 B). Jadi langkah 2 rusak bukan hanya karena `sys.path`.

Perbaikan yang diterapkan: (a) salinan Hermes disinkronkan (md5 repo ↔ `~/.hermes/scripts` identik);
(b) **langkah QLSTM dipensiunkan** (keputusan user) — skrip menjadi Mamba-only; (c) pre-flight nyata
(`"$PY" -c "import torch, einops"` → terverifikasi `torch 2.12.0+cu130`, `einops 0.8.2`, plus cek keberadaan
`models/train_mamba.py`) dan `MAMBA_EXIT=${PIPESTATUS[0]}` + `exit 1` bila gagal — logika diuji dua arah
(`2` → `exit 1`, `0` → `exit 0`), `bash -n` bersih, salinan Hermes md5 `8552df696b19216df5b016dd0dcfd0e9`
identik dengan repo. Efeknya: run mingguan berikutnya melaporkan status jujur (selesai hanya bila benar
selesai). Catatan penting: Mamba sendiri sudah dikeluarkan dari retrain harian 2026-09 karena inert
(`m32_mamba: null` di `data.json`) — bila Mamba memang tidak dipakai, seluruh job `7c8b06a1fddf` layak
dipensiunkan juga (kandidat keputusan berikutnya).

### G2. `[V]` KRITIS (DIPERBAIKI) — `sse_server.log` milik root → restart watchdog akan gagal tanpa suara
`sse_server.log` = `root:root 644` (14.581.181 B) padahal `sse-watchdog.sh:32` melakukan
`nohup "$PY" sse_server.py >> "$LOG" 2>&1 &` sebagai user `ubuntu`; redirect gagal → server tidak pernah
start, tapi `:34` tetap menulis PID ke `$PID_FILE`, dan watchdog hanya bicara via stderr yang dibuang ke
`/dev/null` → kegagalan tak terlihat. Proses yang hidup (start 23 Agu) hanya bertahan karena FD lama.
Diperbaiki: `sudo chown ubuntu:ubuntu sse_server.log` (terverifikasi `-rw-r--r-- ubuntu ubuntu`).
Sisa pekerjaan: entri logrotate untuk file ini + verifikasi pasca-start (`kill -0` + `curl /health`) dengan
keluar non-zero bila gagal.

### G3. `[V]` KRITIS — `sfc-pipeline.sh` tidak punya jalur exit non-zero
`sfc-pipeline.sh:4` `set -uo pipefail` (tanpa `-e`); nilai balik `collect_with_retry` (return `1` di `:50`)
diabaikan di `:53` (`COLLECT_RESULT`/`GIT_RESULT` hanya dicetak); perintah terakhir adalah
`log "Pipeline done: ..."` (fungsi `log` = `echo` → exit 0). Akibat: kegagalan `collect.py` atau push
`data.json` **tidak pernah** terlihat sebagai error cron. Perbaikan yang disarankan: `exit 1` bila
`COLLECT_RESULT=failed` atau `GIT_RESULT` ∈ {push-failed, sync-failed}.

### G4. `[V]` HIGH — Drift salinan repo ↔ `~/.hermes/scripts` dan dua skrip menunjuk venv yang tidak ada
`ws-watchdog.sh` beda (`nohup python3 binance_ws.py` di repo vs `/usr/bin/python3` di salinan Hermes);
`sfc-pipeline.sh` dan `sse-watchdog.sh` identik. `scripts/weekly-model-train.sh:9` meng-`export
PYTHONPATH="${REPO_DIR}/sfc2/venv/..."` sedangkan `/home/ubuntu/sfc2` **tidak ada** (verifikasi `ls`),
dan `scripts/mamba-weekly-train.sh` juga memakai `python3` telanjang — keduanya skrip mati (tidak dipanggil
cron mana pun) dan pasti gagal bila dijalankan. Perbaikan: satu sumber kebenaran (symlink) + pre-flight
`"$PY" -c "import ..."` dan pemeriksaan md5.

### G5. `[V]` HIGH — Dashboard memakai 2 ambang zona, backend 4 → warna dan label saling bertentangan
`index.html:2582` (`_zoneClass`) memakai `>=50` / `>=25`, sedangkan `collect.py:3171` punya 4 zona
(`CRITICAL >0.75`, `HIGH >0.50`, `ELEVATED >0.25`, `NORMAL`). Pada skor 60 backend berkata `HIGH` (oranye)
tapi `index.html:2746` memberi kelas `zone-critical` (merah), sementara teks badge mengambil `d.zone` mentah
→ badge berlabel "HIGH" berwarna CRITICAL. Perbaikan: ekspor ambang dari backend, atau tambah kelas 0.50–0.75.

### G6. `[V]` MED — Falsy-zero di UI (`||`) menelan nilai 0 yang sah
`index.html:2369-2375`, `2759`: `d.composite_confidence || 0.5`, `(d.fng||50) < 15`, `d.rsi_14 || 50`.
Saya turunkan severity dari HIGH ke MED karena dampak praktisnya terbatas: `composite_confidence` di-clamp
≥ 0.05 dan FNG/RSI tepat 0 praktis tidak terjadi (kini fng 68 / RSI-14 bulanan 52,5 → tidak terpicu).
Tetap bug kelas: nilai sah 0 akan berubah menjadi default netral/optimistis. Perbaikan: `?? 50` / `!= null`.

### G7. `[V]` MED — Worker mengembalikan `{}` dengan status 200 saat origin gagal → service worker men-cache sebagai data segar
`worker/index.js:373-374`: kegagalan origin → `new Response('{}', { status: 200, ...})` sehingga
`res.ok === true` di `sw.js:46-51` → `{}` ikut di-`cache.put` dengan header `Date` baru → cek umur
`sw.js:58-65` lolos dan dashboard menerima JSON kosong sebagai data terkini. Jalur `/snapshot` sudah benar
(`502 Backend unreachable`, `worker/index.js:325`). Perbaikan: status non-2xx / flag `error` di body, dan SW
jangan menyimpan body kosong; jalur `catch` hanya boleh menyajikan cache bila umurnya ≤ `MAX_DATA_AGE_MS`.

### G8. `[V]` MED — `/snapshot`: cache-buster `?t=` mematikan cache, dan respons ber-CORS bisa ter-cache edge
`worker/index.js:319-322`: `const cacheKey = new Request(url.toString())` menyertakan query, jadi setiap
`?t=<baru>` melewati cache sepenuhnya; early-return `if (cached) return cached` juga melewati
`getCorsHeaders(request)`. Header `access-control-allow-origin` dinamis + `cache-control: public, max-age=30`
berarti respons ber-ACAO dapat disimpan dan disajikan ke origin lain di luar allowlist.
Perbaikan: cacheKey tanpa query + `Vary: Origin` di luar jalur cache (atau `Cache-Control: private`).

### G9. `[V]` LOW — Aset/versi service worker
`sw.js:46-51` sudah update cache, lalu `sw.js:69-73` mengulang `fetch` + `cache.put` untuk request yang sama
(dua permintaan per poll `data.json`); string versi tidak konsisten (`CACHE_NAME v13` di `:5`, komentar v11 di
`:2`, log v9 di `:171`); `/sw.js` disajikan `public, max-age=3600` (`worker/index.js:476`) sehingga update
service worker tertunda hingga 1 jam.

### G10. `[V]` LOW — Klaim PWA tidak dapat dijalankan
`manifest.json`: `"icons": []`; `index.html` tidak memuat `<link rel="manifest">`; `/manifest.json` → **404**,
`/app.js` → **404** (masih diteruskan Worker tapi tidak ada di `_PUBLIC_FILES` `sse_server.py:198-207`);
`/icon-192.png` → **200 `text/html` 359.664 B** (catch-all SPA menyajikan index.html, menyamarkan 404).
Perbaikan: sediakan ikon + tautkan manifest, atau hapus klaim PWA.

### G11. `[V]` LOW — SSE: klien kedua tidak menerima pembaruan; komentar tidak sesuai perilaku
`sse_server.py:64-71`: mtime dedup memakai variabel global (`_btc_mtime/_sfc_mtime`, `_last_broadcast_*`) →
hanya koneksi pertama yang "melihat" perubahan berkas; klien konkuren hanya menerima heartbeat. Komentar
`:147` ("Heartbeat ... also delivers initial data") tidak benar: payload heartbeat hanya `{"ts": ...}` (`:148`).
`/events` publik tanpa auth dan tanpa cap jumlah klien (sudah tercatat terbuka di `docs/SECURITY_AUDIT.md`).

### G12. `[V]` LOW — Cakupan logrotate
`/etc/logrotate.d/sfc-pipeline` hanya menangani `sfc-pipeline.log` (24,6 MB, `daily/rotate 7/size 50M`);
`sse_server.log` (14,5 MB) **tidak punya entri** → tumbuh tanpa batas. Catatan operasi lain: job
`707bb2af6b4e` ("SFC Weekly AI Analyst") ber-`deliver: local` (tidak ada kanal pengiriman) — perlu dipastikan
outputnya memang dikonsumsi dari berkas, bukan diharapkan terkirim.



## F. Yang belum tercakup (jujur)

1. **Workstream operasi/dashboard/publik** → **SUDAH DITUTUP** di bagian G (14 temuan G1-G12, dua di antaranya
   sudah diperbaiki). Yang masih terbuka dari workstream itu: keputusan nasib langkah QLSTM (G1), entri
   logrotate `sse_server.log` + verifikasi restart watchdog (G2), jalur exit `sfc-pipeline.sh` (G3),
   rate-limit `/events` (G11), identitas `worker/index.js` on-disk vs deployment aktif, config
   `cloudflared`/systemd/nanti WAF, dan master `/home/ubuntu/index.html` (mtime 11 Sep).
2. ~~Faktor `Rt`/`Sc` konstan sepanjang jendela histori `.factor_history.json`~~ → **SUDAH DITUTUP** (A13/A14):
   konstantanya adalah artefak SMA 30 hari (bukan input mati); `Rt` dan `Sc` direproduksi persis dari
   `.daily_market_cache.json`. Sisa celahnya hanya tampilan (spot vs rata-rata) — tercatat sebagai A13.
3. M65 (CNN), M68 (DRL), M69 (GNN) hanya dilihat call-site-nya (semuanya display-only).
4. Validitas isi `data_collection.json` per-era (perubahan skala historis m5/m6) belum diuji.
5. Root cause CoinGecko konsisten partial (0–3/4 koin stablecoin) belum dipastikan.
6. Dampak penuh B1–B10 bila `.causal_cache.json` di-refit belum diukur (perlu harness replay kode-konsisten).

---

## H. Status akhir 2026-09-18 (apa yang benar-benar sudah dieksekusi & diverifikasi)

Perbaikan lapisan operasi/tampilan (tidak ada kode skoring yang diubah):

| Item | Perubahan | Verifikasi nyata |
|------|-----------|------------------|
| G1 | Langkah QLSTM dipensiunkan (keputusan user); skrip jadi Mamba-only + pre-flight + exit code jujur | `bash -n` bersih; pre-flight nyata `torch 2.12.0+cu130`, `einops 0.8.2`; logika exit diuji 2 arah; salinan Hermes md5 `8552df696b19216df5b016dd0dcfd0e9` |
| G13 (baru) | — | Run nyata: **epoch 1 = 551,3 s** untuk 100 epoch → ±15 jam; job mingguan tak bisa selesai (dibatalkan setelah bukti) |
| G2 | `sudo chown ubuntu:ubuntu sse_server.log` + logrotate `/etc/logrotate.d/sse-server` (daily/rotate 7/size 20M/copytruncate) | Restart nyata: PID 4738 (hidup 25 hari) dimatikan → watchdog menyalakan PID 2047498 → `/health` 200 → log append bekerja; `logrotate -d` lulus |
| G3 | `sfc-pipeline.sh`: cek `git add` gagal (`GIT_ADD_FAILED`) + blok exit non-zero | Blok exit diuji dari teks berkas: `ok/ok`→0, `failed`→1, `push-failed`→1, `throttled`→0, `add_failed=1`→1; md5 salinan `5f8469be07bfe9883c8b6b319e060412` |
| G5 | Zona: `d.zone` jadi satu sumber kebenaran (peta `_ARC_MULT` karangan dihapus) + kelas `.zone-high`/`.kb-o`/`.pf-orange` | Harness node mengekstrak fungsi asli dari `index.html`: 6/6 PASS; 0 sisa `_arcMult` di kode; live `curl /` memuat `_badgeClass`/`zone-high` |
| G7/G8/G9 | Worker: `/data.json` gagal → 502 + `no-store`; `/snapshot` cache key tanpa query + `Vary: Origin` + CORS dihitung ulang; `/sw.js` → `no-cache, must-revalidate`. `sw.js`: guard `isRealData()`, fetch ganda dihapus, versi v14 | **Di-deploy** (Version ID `d55b349f-5619-4ec3-9429-83c7c9884511`); live: `/snapshot?t=…` → `vary: Origin` + ACAO; `/sw.js` → 6288 B, `no-cache, must-revalidate`, berisi `isRealData` + `sfc-terminal-v14`; `/data.json` 200 json; `/live.json` tetap 200 (Pine tidak terpengaruh) |

**Trap penting yang terkonfirmasi (jangan diulang):** perbaikan UI pertama saya tulis ke `sfc/index.html` dan
**hilang dalam hitungan menit** — pipeline 5-menit memulihkan berkas itu dari master `/home/ubuntu/index.html`
(`sfc-pipeline.sh:66`) lalu menyuntik data baru. Perbaikan UI harus ditulis ke **master** `/home/ubuntu/index.html`
dulu, baru disalin ke repo (kini md5 `268aeab7cc6436c63b865dd542684ace` identik di keduanya), supaya bertahan.

**Sisa yang butuh keputusan Anda (belum dikerjakan):**
1. G13 — nasib job `7c8b06a1fddf`: turunkan jumlah epoch secara drastis, atau pensiunkan job-nya (Mamba inert,
   `m32_mamba: null`), atau beri timeout di peluncur cron.
2. G10 — `/app.js` & `/manifest.json` 404: hapus klaim PWA, atau sediakan ikon + `<link rel="manifest">`.
3. G11 — rate-limit `/events` (SSE publik tanpa cap klien).
4. Identitas `worker/index.js` on-disk vs deployment aktif kini terjawab sebagian (deploy dari repo ini sukses,
   Version ID di atas); sisa: config `cloudflared`/systemd/WAF dan master `index.html` (kini sudah sinkron).
