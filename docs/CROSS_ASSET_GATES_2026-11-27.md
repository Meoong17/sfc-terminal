# Cross-Asset Gates — implementasi dari cetak biru A.docx (2026-11-27)

Sumber desain: `C/A .docx` (Cross-Asset Liquidity & Structural Regime Engine, 4 model).
Yang dieksekusi hanya bagian yang **datanya benar-benar ada** dan **tidak melanggar aturan
metodologi** SFC (macro = konteks, bukan skor).

Modul: `analysis/cross_asset_gates.py` → `data/cross_asset_state.json` →
dipublikasikan di `data.json` sebagai key `cross_asset` (dijalankan pipeline sebelum
`collect.py` tiap siklus).

## 1. Yang diimplementasikan

### Model 1 — ML-PT Gate (status rezim makro)
Input (semua dari sumber nyata): Core PCE `PCEPILFE` (YoY + MoM annualized), Real PCE
`PCEC96` (MoM), `DGS10`, `DGS20`/`DGS30` (kurva 20s30s), `BAMLH0A0HYM2` (HY OAS),
`DEXJPUS` (USDJPY), DXY (dari data.json).

Logika:
- `policy_trap`: BOXED_IN_HAWKISH bila Core PCE > 2,5% YoY **dan** Real PCE ≤ 0,5% MoM.
- `yield_oas_regime`: ΔUS10Y 30d vs ΔHY OAS 30d → DEFENSIVE_FLIGHT_TO_QUALITY (yield turun,
  OAS melebar), HEALTHY_EASING, RISK_OFF_RATES_UP, RISK_ON.
- `cross_asset_liquidity_drain`: DXY > 103 **dan** gold **dan** silver turun 30 hari.
- Status: CONTRACTION_LOCK / NEUTRAL_CHOP / EXPANSION.

Status live (2026-11-27): **CONTRACTION_LOCK** —
Core PCE 3,26% YoY (annualized MoM 3,57%), Real PCE −0,03% MoM, US10Y 4,91%
(−9 bps/30d), HY OAS 3,72% (+34 bps/30d), 20s30s −3 bps, DXY 103,76,
`drain = null` (basis `dxy_only_partial`, lihat §3).

### Model 3 — BIM-ETF 4 gerbang bottom
| Gerbang | Kriteria terukur | Status live |
|---|---|---|
| 1 ETF exhaustion | IBIT net ≥0 pada ≥3 dari 5 hari bursa | GAGAL (0/5; 15D −4,95B USD ≈ −82.9k BTC) |
| 2 Leverage flush | OI ≤ −15% dari puncak 90D **dan** funding 3D ≤ 0 | GAGAL (OI −13,3%; funding +0,0058%/8h) |
| 3 Spot absorption | harga lower low **dan** CVD higher low (7D vs 7D) | GAGAL (harga 58.952 vs 63.068; CVD −675,7M vs −417,2M) |
| 4 Structure reclaim | harga > 200W SMA | GAGAL (59.720 vs 68.201, −12,4%) |

Hasil: **0/4**, sama dengan penilaian dokumen — jadi tidak ada posisi buy-and-hold.

### Model 4 — TR-SSA (aturan alokasi, display)
Karena ML-PT = CONTRACTION_LOCK dan gates < 3/4 → mode
`CAPITAL_PRESERVATION_&_RENTALS`, kas 80–90%, rental taktis 10–20%, plus aturan sektor
(hindari token AI/DePIN bersubsidi & memecoin tanpa likuiditas; prioritas BTC + protokol
berpendapatan organik saat gates ≥3/4). Ini teks kebijakan, bukan eksekusi order.

## 2. Yang SENGAJA tidak dieksekusi

**Model 2 — ACBI: `UNAVAILABLE`.** Butuh CDS Tier-1 (MSFT/GOOGL/AMZN/META/NVDA),
Oracle 5Y CDS, dan harga obligasi tunai 144A (CRWV/TRACTC). Tidak ada sumber gratis
atau tepercaya di mesin ini. Menggantinya dengan HY OAS/LQD lalu menamainya "ACBI"
akan menghasilkan sinyal karangan yang terlihat kuantitatif — persis jenis kesalahan
yang audit 2026-11-27 berhasil hapus dari repo ini. Model ini menunggu sumber data
berbayar (Bloomberg/FactSet/LSEG) atau tidak dipakai sama sekali.

**Gold/Silver:** stooq memasang proof-of-work anti-scraping, Yahoo Finance membalas
"Edge: Too Many Requests", dan seri FRED `GOLDAMGBD228NLBM`/`GOLDPMGBD228NLBM`/`SLVPRUSD`
sudah dihentikan (HTTP 400). Karena itu `gold_usd`/`silver_usd` = null dan filter
"dash for cash" **tidak** diklaim terpenuhi.

## 3. Kejujuran status (yang membedakan modul ini dari narasi)

- `cross_asset_liquidity_drain = null` dengan `drain_basis = "dxy_only_partial"` —
  hanya level DXY yang terukur; separuh definisi (logam) tidak tersedia, jadi tidak
  dilaporkan `true` hanya karena DXY > 103.
- `acbi_status = "UNAVAILABLE"` beserta alasannya, bukan skor 0,5 netral.
- `meta.scoring_impact = "NONE"`; key `cross_asset` ditulis setelah seluruh perhitungan
  `sfc_pct` selesai, tidak dipakai di zona, confidence, Kelly, maupun bobot faktor.
  Bukti: pada dua siklus berturut-turut `sfc_effective` bergerak 35,01 → 35,15 karena data
  pasar, bukan karena gerbang (kode gerbang tidak menulis variabel skor apa pun).
- `meta.validation` menyatakan ambang (2,5% / 0,5% / 103 / −15% / 3 hari) **ditetapkan
  tangan dan belum walk-forward validated** — sama seperti yang diminta aturan repo:
  konteks boleh tampil, klaim prediktif butuh validasi.

## 4. Verifikasi

```
.venv/bin/python analysis/cross_asset_gates.py
  -> ML-PT CONTRACTION_LOCK, ACBI UNAVAILABLE, BIM 0/4, TR-SSA CAPITAL_PRESERVATION
sfc-pipeline.sh (dijalankan manual)
  -> collect=ok, data.json cross_asset terisi, zona ELEVATED, sfc 35,15
https://sfcterminal.xyz/data.json
  -> 200, ts 08:31:34Z, cross_asset: CONTRACTION_LOCK / 0 of 4 / CAPITAL_PRESERVATION_&_RENTALS
```

## 5. Langkah berikutnya (belum dikerjakan)

1. Panel UI: kartu "Cross-Asset" di dashboard (saat ini hanya tersedia di `data.json`;
   UI belum dirender karena perubahan `index.html` perlu desain terpisah).
2. Uji gerbang pada riwayat: HY OAS 1997+, ETF 2024+, OI/CVD 2026+ — untuk melihat apakah
   gerbang ini punya nilai sebelum dipakai sebagai klaim; gerbang 3 (CVD) baru punya
   ~14 hari data sehingga belum bisa diuji.
3. ACBI menunggu sumber CDS/bond; sampai itu tetap `UNAVAILABLE`.
