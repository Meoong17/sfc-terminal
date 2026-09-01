# Flow Regime-Conditional Test — Hasil (Ya.docx Test #4)

Menguji hipotesis dari `C/Ya.docx` Test #4: "flow tidak prediktif saat liquidity
expansion, tapi sangat prediktif saat liquidity contraction." Ini satu-satunya ide
di dokumen yang belum pernah diuji di repo. Script: `analysis/flow_regime_conditional_test.py`,
hasil mentah `.flow_regime_test.json`.

## Metode
- Flow dengan history panjang:
  - ORDER_FLOW: taker imbalance ratio (binance_orderflow_daily.json), 2017-08..2026-07 (3.2k hari)
  - ETF_FLOW: net BTC flow (.etf_cache.json), 2024-01..2026-08 (677 hari)
- Keduanya di-z-score rolling 90d (point-in-time, sesuai rekomendasi Ya.docx).
- Regime: liquidity expansion vs contraction via GLF vs **trailing median** (point-in-time;
  sign GLF fixed-calibration ternyata all-positive in-sample, tak berguna). expansion=1820d,
  contraction=2437d.
- Metrik per flow x regime x horizon [1,3,7,14,30]: tail-gap top-vs-bottom 25% (bootstrap CI),
  Spearman IC, dan purged-CV/embargo OOS AUC (P(fwd>0|flow)).

## Hasil
### Order flow (taker imbalance)
| regime | h | tail-gap (bottom-top) | purged-CV AUC |
|---|---|---|---|
| expansion | 3d | -0.28 n.s. | 0.424 |
| expansion | 7d | -0.98 n.s. | 0.394 |
| expansion | 30d | -4.39 SIG (neg) | 0.341 |
| contraction | 3d | -0.95 SIG (neg) | 0.492 |
| contraction | 7d | -1.40 SIG (neg) | 0.448 |
| contraction | 30d | -3.45 SIG (neg) | 0.395 |

Gap NEGATIF = flow rendah (taker sell) -> forward return LEBIH TINGGI. Ini **mean-reversion**,
kebalikan dari "flow bullish leading". Purged-CV AUC <0.5 di KEDUA regime -> tidak ada skill
prediktif arah yang generalizable (AUC<0.5 menunjukkan bahkan arah contrarian pun tak robust).

### ETF flow
Semua data 2024-2026 terpetakan ke contraction (periode post-ETF = contraction per trailing-median
GLF). Tail-gap negatif signifikan 14d (-2.20) & 30d (-4.92) = mean-reversion; purged-CV AUC
0.374-0.475 (di bawah chance). ETF inflow besar TIDAK mendahului return lebih tinggi.

## Verdict (Test #4)
**DITOLAK untuk kedua flow.** Conditioning per regime TIDAK memunculkan edge flow:
1. Tail-gap "signifikan" di contraction itu NEGATIF (mean-reversion), bukan bullish-lead.
2. Purged-CV OOS AUC <0.5 di semua regime/horizon -> tidak ada skill OOS untuk memprediksi arah.
3. Pola ini = Pitfall 17/26 walk-forward-validation: hasil nominal signifikan yang gugur di
   purged-CV & berlawanan polaritas = bukan sinyal deployable.

Ini MENGONFIRMASI temuan lama repo: flow mengikuti harga (ETF lagged n.s. setelah VIX+M2;
Pitfall 32/35), bukan leading indicator. Regime conditioning tidak menyelamatkannya.

## Implikasi untuk Ya.docx
Dari 13 poin arsitektur Ya.docx, yang sudah teruji kini: (1) tesis Liquidity->Flow->Price
gagal via SLR; (2) Test #4 flow regime-conditional gagal; (3) flow leading gagal. Komponen
yang belum teruji hanyalah konsep Absorption/market-response (behavior layer) — tapi itu
bergantung definisi shock yang fragil. Sesuai prinsip repo (Pitfall 18: tahu kapan berhenti
membangun), tidak ada alasan membangun flow layer SFC v3.

## Aset
- `analysis/flow_regime_conditional_test.py` — test suite (reusable, point-in-time)
- `.flow_regime_test.json` — hasil (gitignored runtime cache)
