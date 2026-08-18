# Order-flow extension progress (resume note)

Status saat dihentikan: **2026-08-16 ~07:53** (sesi "lanjutkan sesi terakhir").

## Tujuan
Perpanjang `data/binance_orderflow_daily.json` (order-flow aggre-trades BTCUSDT)
dari 2022-07-31 menuju bulan lengkap terakhir **2026-07**, via `--stream`
(download->parse->delete zip per bulan, disk terpakai ~2-3GB).

## Perintah resume
```
cd /home/ubuntu/sfc
python3 analysis/fetch_orderflow.py --start <BULAN_MULAI> --end 2026-07 --stream
```
`--start` = bulan terakhir yang SUDAH ADA di JSON + 1 bulan (script merge base otomatis
memakai hari yang sudah ada di OUT, tidak re-fetch). Cek `python3 -c "import json;print(sorted(json.load(open('data/binance_orderflow_daily.json')))[-1])"`.

## Progres terakhir
- JSON `n=1902`, terakhir **2022-11-30**.
- Sudah selesai: 2022-08 .. 2022-11. Sedang jalan: 2022-12.
- Sisa: **45 bulan** (2022-12 .. 2026-07). Estimasi ±3-5 jam (~4 jam realistis).
  - Berat: 2022-12..2023-12 (13 bln) ~1.5-2 jam
  - Ringan: 2024-01..2026-07 (32 bln) ~1.5-2.5 jam
- Rata-rata laju ~6 min/bulan di bagian berat.

## Fix OOM yang SUDAH diterapkan (analysis/fetch_orderflow.py)
Sebelumnya mati OOM (kernel kill) saat parse zip 2022-08 (2.2GB) di box 3.6GB.
- `nsamp` sekarang numpy array BOUNDED (rolling downsample + halving > 6M elemen),
  bukan akumulasi `nsamp.append(notional[::4])` tanpa batas.
- `chunksize` 4_000_000 -> 1_000_000.
- Terbukti: parse 2022-08 selesai dgn RSS stabil ~394MB (VmPeak 602MB), jauh di bawah OOM.
JANGAN balik ke akumulasi penuh.

## Kendala / catatan
- Box tekanan memori: workerd(CF) + 3 chromium + gateway Hermes, swap nyaris penuh.
  Proses boleh jalan, RSS teruji aman.
- File `data/binance_vision_cache/spot_aggTrades/BTCUSDT-aggTrades-2022-09.zip` (2.25GB)
  tertinggal dari run yg gagal; 2022-09 SUDAH ter-parse, file bisa dihapus utk hemat 2GB.
- Ada kejadian `FileNotFoundError` saat rename .part (run kedua, di 2022-09) — kemungkinan
  proses sempat tertimpa/restart; run ketiga (sekarang) jalan stabil.

## Setelah selesai
1. Verifikasi sanity orderflow vs kline (analog validasi sebelumnya):
   `total_qty` hari ~ sama dgn `volume` di binance_vision_daily.json.
2. Commit `analysis/fetch_orderflow.py` + `data/binance_orderflow_daily.json`
   (file masih untracked dari sesi kemarin).
3. Kemas deliverable jadi SATU file .zip (sesuai permintaan user "compress menjadi zip saja").
