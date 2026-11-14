# FLQD v2.0 → SFC: integrasi pengukuran (2026-11-14)

Sumber spesifikasi: `C/FLQD_Framework_SFC_Methodology_v2.docx` (14 Nov 2026).
Aturan yang dipakai: dokumen **tidak** otomatis menjadi alasan baru untuk menguji ulang kanal yang
sudah ditolak repo (`docs/PROJECT_STATUS.md`, bagian "Kanal macro-liquidity & crypto-flow — DITUTUP").

## 1. Klasifikasi tiap komponen spesifikasi

| Komponen FLQD v2 | Data tersedia di repo | Status |
|---|---|---|
| LA / LT (availability vs transmission) | WALCL, TGA, RRP, M2SL, HY/IG OAS, VIX, REAL_Y10, USD, SPX/NDX/copper/BTC | **Diimplementasi sebagai pengukuran** (display) |
| CS / FS (credit & funding stress) | HY/IG OAS + Δ5d, VIX, realized vol UST, RRP | Pengukuran; riwayat HY OAS hanya 2023-08+ |
| PF (positioning fragility) | funding BitMEX, volume expansion, taker sell ratio | Pengukuran; OI Binance hanya ~100 hari → LiqSens/OI-leverage **tidak** tersedia (coverage 44%) |
| CE (capitulation) + Absorb(t) | taker sell quote, funding, inverse-Amihud pada hari turun | Pengukuran |
| QD (quality dispersion) | BTC vs ETH/SOL relative strength, dominance change, volume share | Pengukuran |
| FLTI / SFM / VD | turunan skor layer | Pengukuran |
| Robust MAD → Φ-percentile (0–100) | semua seri | **Menggantikan** ambang absolut di engine |
| Exponential information decay + coverage | WALCL/TGA/RRP (weekly), M2/SLOOS (monthly/quarterly) | Diimplementasi (half-life 3/10/35/75 hari) |
| Regime engine Mahalanobis + softmax + Markov + entropy | 7 vektor layer | **Diimplementasi** (menggantikan `detect_regime()` 3-state ad-hoc) |
| Hard circuit breaker FS>85 / SFM>75 | turunan | Diimplementasi di engine |
| Net liquidity / ΔM2 sebagai **faktor skor** | ada | **DITOLAK** — sudah diuji repo (event study & purged WF), tidak di-blend |
| Cross-asset playbook R0–R5 (eksekusi) | — | Tidak diimplementasi (butuh eksekusi lintas aset; di luar cakupan SFC) |

**Tidak ada perubahan `sfc_effective`.** Semua penambahan bersifat pengukuran/konteks.

## 2. Hasil engine (`analysis/flqd_engine.py` → `analysis/flqd_engine_results.json`)

As of 2026-09-30 (data harian repo berakhir di sini):

- Layer: LA 50.7, LT 24.5, CS 41.1, FS 21.3, PF 73.2, CE 50.0, QD 75.5
- Interaksi: FLTI 38.2, SFM 30.1, VD 0.34, SC 47.7, RE 49.6
- Rezim: R0 20%, **R1 30%** (mode — liquidity divergence), R2 13%, R3 4%, R4 12%, R5 21%
- Entropy H(t) 0.92 (diffuse), sojourn rata-rata 11 hari

Validasi walk-forward (sampling 5 hari, target = fwd 20d vol > q75 **atau** fwd 20d < −10%):

| Model | Brier |
|---|---|
| Klimatologi (base rate 0.29) | 0.206 |
| Baseline harga/vol (`rv30`, `mom30`, logistic expanding) | 0.249 |
| FLQD `P(R2 ∪ R3)` | **0.274** |

**Verdict: NULL.** Probabilitas stres FLQD kalah dari klimatologi dan baseline harga/vol. Spearman
P(stress) vs realized vol 20d hanya +0.19 (p=0.03). Konsisten dengan verdict repo: lapisan macro/credit
berguna sebagai **deskripsi keadaan**, bukan prediktor. Karena itu kartu dashboard berlabel
"context / not blended". Entropy 0.92 juga berarti distribusi masih hampir seragam — mode tunggal
belum layak dijadikan label keras; yang ditampilkan adalah distribusinya.

## 3. Perubahan dashboard

1. **Kartu "Signal & Action" dihapus** (Overview). Isinya duplikatif: pill sinyal, `pillReason`,
   status Kelly (heuristic), BTC-vs-Floor, transition risk, dan readiness — kelimanya bukan keputusan
   tereksekusi, dan Kelly sudah berlabel heuristic di kartu Ensemble.
2. **Kartu "Regime Distribution"** (FLQD v2, context) menggantikannya: 6 probabilitas R0–R5, status
   mode, entropy, LA/LT · CS/FS · PF/QD, coverage PF, tanggal as-of, flag STALE, dan Brier vs baseline
   di footer. Fallback text bila engine belum dijalankan.
3. **Kartu "Macro Liquidity (M72–M75)" dan "Stablecoin Liquidity (M76–M80)" dihapus** dari tab Macro
   (11.5 KB blok UI). Keduanya display-only untuk kanal yang sudah ditolak (M2/GLF dan stablecoin
   sebagai sinyal); datanya tetap ada di `data.json` bila dibutuhkan lagi.

## 4. Daemon / freshness

`collect.py` memuat `analysis/flqd_engine_results.json` (field `flqd`) saat menyusun `data.json`,
dengan `cache_age_hours` dan `stale=true` bila >48 jam. Engine dijalankan harian lewat cron
`sfc-flqd-engine` (lihat `scripts/run_flqd_engine.sh`); tanpa run baru, dashboard menampilkan STALE.

## 5. Yang masih menunggu keputusan

- Kalibrasi τ / window kovarians belum dioptimasi walk-forward (sengaja: optimasi pada sampel pendek
  = narrative fitting). Bila HY OAS historis penuh tersedia (FRED API di host ini terpotong ke 2023-08),
  ulangi validasi dengan riwayat 2017+ sebelum mempertimbangkan blend.
- LiqSens/OI leverage tidak bisa dihitung tanpa riwayat OI (Binance OI collector hanya ~100 hari).
