# Binance Vision — Multi-Timeframe Data Infrastructure (2026-08-08)

Tulang punggung data harga & funding historis untuk SFC. Canonical single series
9 tahun sehingga SEMUA walk-forward validation memakai satu seri bersih, bukan
snapshot SFC 2 bulan.

## Sumber & fetcher
`analysis/fetch_binance_vision.py` — unduh bucket publik Binance Vision (no API key),
threaded (8), verifikasi SHA256 via `.CHECKSUM`, cache zip lokal.

| Dataset | Path | Rentang |
|---|---|---|
| Spot klines 1d | `spot/monthly/klines/BTCUSDT/1d/` | 2017-08-17 → 2026-07-31 |
| Spot klines 1w | `spot/monthly/klines/BTCUSDT/1w/` | 2017-08 → 2026-06 |
| Futures funding | `futures/um/monthly/fundingRate/BTCUSDT/` | 2020-01 → 2026-07 (8h) |
| Futures mark/index/premium klines 1d | `futures/um/monthly/{mark,index,premium}PriceKlines/BTCUSDT/1d/` | 2020-01 → 2026-07 |

Catatan teknis:
- Timestamp bervariasi (ms vs mikrosekon antar era) → parser normalisasi otomatis.
- Candle bulanan TIDAK ada path `1M` → agregasi dari daily.
- Trades full-resolution di-skip (klines sudah punya volume base/quote/taker, cukup utk slippage).
- Bulan berjalan (2026-08) monthly belum terbit & daily 404 → data berakhir di 2026-07.

## Cache (data/binance_vision_*.json)
- `binance_vision_daily.json` — 3271 hari; per-hari: OHLCV, quote_vol, taker_base/quote,
  funding_last/mean, index, mark, premium(basis).
- `binance_vision_weekly.json` — 456 minggu OHLCV.
- `binance_vision_monthly.json` — 108 bulan (open/close/volume/quote_vol, agregat daily).

## Modul fitur — `data_sources/binance_features.py`
`load_daily()`, `compute_features()` → array sejajar (point-in-time-safe, hanya ret_*
yang look-forward sebagai LABEL):
- `ret_7/30/90/180/365` — forward log-return (label canonical)
- `mom_7/30/90/...` — trailing return (feature)
- `rvol_7/30/90` — realized vol (rolling std log-return, annualized √365)
- `vol_7/30/90` — rata-rata quote volume harian (USD)
- `taker_ratio`, `premium` (basis), `funding`

## Demo validasi — `analysis/binance_validate_demo.py` (6.5 tahun funding, 9 tahun harga)
| Faktor | arah | Hasil (gap top−bottom, 90% CI) | Verdict |
|---|---|---|---|
| FUNDING | neg | 7d −0.0116*** · 30d −0.0149*** · 90d ns · 365d ns | **era-unstable**: era1(20-22) −0.038 vs era2(23-26) +0.016 → SIGN-FLIP |
| PREMIUM/basis | neg | 7d/30d/90d ns · 365d −0.1112*** | era-flip (−0.053 → +0.033) |
| REALIZED VOL 90d | neg | 90d −0.1212*** · 365d −0.5802*** | stabil, kuat |
| MOMENTUM 30d | pos | 7d +0.0314*** · 30d +0.0811*** · 90d +0.1089*** | stabil, kuat |

### Implikasi jujur
1. **Funding era-unstable** (sign-flip 2020-22 vs 2023-26) meski signifikan overall.
   Ini pola yang sama yang menolak China M2 / JPY carry / HY. → **JANGAN naikkan bobot
   funding** di execution_risk. Mendukung sikap hati-hati thd varian D (funding-heavy)
   di eksperimen A/B/C/D.
2. Realized vol & momentum menunjukkan efek stabil 9 tahun — kandidat fitur yang valid
   untuk divalidasi lebih lanjut bila mau dipakai.
3. Kanonik price/funding kini memungkinkan re-run WALK-FORWARD panjang utk faktor mana
   pun (L8, trend continuation, dst) — bukan snapshot 2 bulan.

## Cara run
```
.venv/bin/python analysis/fetch_binance_vision.py   # unduh + cache (sekali; zip ter-cache utk re-run)
.venv/bin/python data_sources/binance_features.py   # fitur
.venv/bin/python analysis/binance_validate_demo.py  # demo validasi
```
