# Funding Regime Analysis — Hasil (BitMEX XBTUSD funding 2016-2026)

Analisis regime funding sebagai sinyal perilaku/posisi crypto-native untuk SFC.
Data: `data/bitmex_funding_daily.json` (3761 hari, 2016-05-14 → 2026-08-31).
Script: `analysis/funding_regime_analysis.py`.

## Distribusi funding
mean +0.00009, median +0.00008, sd 0.00070, min −0.00545, max +0.00943.
Regime incidence: z>+1 (long-crowding) 6.4%, z<−1 (short-crowding) 5.1% (seimbang).

## Hasil — FUNDING ERA-STABLE sebagai diskriminator regime (temuan positif pertama)

**[3] State-discrimination (tercile funding → perilaku):**
- **trend (bull%): low=41% mid=47% high=82%** — monoton kuat. Funding tinggi ↔ jauh lebih mungkin di regime bull.
- stress: low=16.7% mid=4.2% high=9.8% — stress terangkat di tercile low (short-crowding).
- vol: low=0.67 mid=0.53 high=0.59 — tercile low punya realized-vol tertinggi.
- fwd30: low=+10.3% mid=+3.7% high=+4.9% — kontrarian lemah (low funding → return 30d lebih tinggi), in-sample, butuh OOS.

**[4] Era-stability (funding vs regime):**
- corr(funding, bull): era0(2016-19)=**+0.163**, era1(2019-22)=**+0.407**, era2(2023-26)=**+0.263** — **TANDA POSITIF STABIL di semua era.**
- corr(funding, stress): −0.07, −0.16 (funding tinggi ↔ stress rendah); era2 NaN (edge, minor).
- **Ini sinyal pertama yang lolos era-stability** — tidak seperti SEMUA macro-liquidity (GLF/term-prem/ΔM2/VIX/yield-spread) yang era-flip.

**[2] Crisis-elevation:** funding BERPUTAR NEGATIF di krisis (COVID −0.00031, Luna −0.00006, FTX −0.00017) vs baseline +0.00009 → indikator stress/posisi yang valid.

**[5] Lead/transition:** pre-30d funding sebelum flip trend (+0.00004) & stress (+0.00013) ≈ baseline → **TIDAK memimpin transisi regime** (kontemporer/konfirmasi, bukan leading).

## Kesimpulan
1. **Funding = diskriminator regime bull/bear yang era-stable** — crypto-native, berhasil memisahkan state perilaku (berlawanan dengan macro-liquidity yang gagal).
2. **Funding = indikator stress valid** — negatif di krisis, stress tinggi di tercile low.
3. **Funding TIDAK memimpin** — perannya konfirmasi/pembacaan state, bukan prediktor forward.
4. Implikasi SFC: funding layak jadi **lapisan konfirmasi/context regime** (era-stable, valid), sesuai tujuan SFC membaca perilaku. JANGAN jadikan prediktor leading (kontradiksi dengan temuan no-lead).

Caveat: fwd30 (kontrarian) in-sample lemah, butuh OOS; BitMEX satu exchange — konfirmasi lintas-exchange (Binance funding 2019+) disarankan.
