# WSELOB-2017 — LOB Methodology Sandbox

Sandi metodologi order-book/flow dari dataset WSE (Mendeley, DOI 10.17632/3g4mhdp899.1).
**BUKAN bagian scoring SFC** — ini referensi reusable untuk feature engineering
microstructure yang bisa ditransfer ke data crypto bila SFC nanti punya LOB/tick BTC.

## Dataset (CC BY 4.0, open)
- WSELOB-2017: limit order book + trades, 5 perusahaan terbesar Bursa Warsawa (WSE), setahun 2017.
- Format HDF5, struktur `dYYYYMMDD/table` = record array pesan order mentah.
- Total 1.36 GB / 15 file. Di folder ini hanya subset representatif PEKAO (LOB 153MB + trades 7.7MB).
- Full list: `https://data.mendeley.com/public-api/datasets/3g4mhdp899` (public, no key).
  File langsung: `https://data.mendeley.com/public-files/datasets/3g4mhdp899/files/{fileId}/file_downloaded`.

## Schema (kolom `table`)
- time (ns epoch), order_id, order_date, priority_date, symbol_idx
- price (scaled /100 → PLN), volume (saham), agg_volume, num_orders
- side (1=buy, 2/5=sell), order_type (S2), action_type (S1), price_level

## Semantik pesan (action_type)
- A = add order, M = modify, D = delete, F = flush/clear, Y = retransmit (mod-atau-add)
- Contoh hari PEKAO 2017-01-02: A=6095, M=189, D=5715, F=1 (dari 12k pesan)

## Metodologi (rekonstruksi + fitur)
1. Rekonstruksi order book dari aliran pesan (class `orderbook2.OrderBook` dari penulis,
   atau reimplementasi numpy untuk kecepatan).
2. Fitur LOB kanonik per snapshot interval:
   - best bid / best ask, mid, spread
   - depth (volume kumulatif bid/ask, top-N)
   - **Order Book Imbalance** `OBI = (Vbid - Vask)/(Vbid + Vask)` (top-N)
   - WAP (volume-weighted avg price)

## Hasil demo (PEKAO 2017-01-02, 12k pesan, snapshot tiap 2k)
```
msg  best_bid best_ask   mid   spread depth5b depth5a  OBI5   WAP
2000  126.10   126.30  126.20  0.20   1357    983   0.160 126.24
4000  126.45   126.75  126.60  0.30   1320    979   0.148 126.65
6000  126.90   126.95  126.93  0.05   1890    965   0.324 126.95
12000 128.15   128.55  128.35  0.40   3788   1677   0.386 128.48
```
(Mid naik 126.20→128.35 PLN; OBI bid-heavy sepanjang window → tekanan beli.)

## Transfer ke crypto (relevan untuk SFC)
Konsep fitur ini (mid, spread, depth, OBI, flow imbalance, WAP) berlaku langsung ke
order book BTC bila data tick/LOB crypto tersedia (Binance/OKX depth streams).
SFC saat ini pakai data harian (Binance Vision) — LOB hanya relevan jika nanti
menambah tick/LOB collection. Folder ini = sandi metodologi, bukan integrasi.

## Cara run
`.venv/bin/python lob_features_demo.py`  (butuh h5py, pandas, matplotlib)
