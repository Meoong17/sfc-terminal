# Behavior/Regime-Reading Validity Test — Hasil

Menguji faktor-faktor yang sebelumnya ditolak sebagai PREDIKTOR, sekarang sebagai
**pembacaan state/perilaku BTC** (SFC = regime detection system conditioned on
liquidity & macro state; objective = baca/deteksi regime, BUKAN forecast harga).
Script: `analysis/behavior_reading_validity_test.py`, hasil `.behavior_reading_test.json`.

## Bar validasi (bukan predictive-OOS)
- A. Crisis-elevation (Pitfall 9): apakah pembacaan naik/turun tepat saat perilaku
     benar-benar stres (vs kontrol 180 hari sebelumnya), sesuai tanda konstruk.
- B. State-discrimination: apakah pembacaan tinggi/rendah bertepatan dengan perilaku
     BTC yang berbeda (realized vol, downside semidev, max drawdown, worst-day).
- C. Convergent: Spearman kontemporan pembacaan vs dimensi perilaku.

Dimensi perilaku = realized_vol(30d), downside_semidev(30d), maxdd(30d), worst_day(30d).

## Hasil per faktor
### GLF (global liquidity — input conditioning SFC)
- State-discrimination: **SIGNIFIKAN di 4/4 dimensi**. vol b-t=-0.156, downside -0.193,
  maxdd +2.912, worst_day +2.568 (semua SIG). Spearman: vol +0.264, downside +0.283,
  maxdd -0.144, worst_day -0.241 (semua p=0).
- Interpretasi: GLF **benar-benar memisahkan state perilaku BTC**. High-GLF (expansion)
  bertepatan dengan vol tinggi, risk-taking, drawdown dalam; low-GLF (contraction) sebaliknya.
  Ini bukan "gagal" (high-likuiditas ≠ selalu tenang) — ini **separasi regime yang valid**:
  GLF membelah pasar jadi regime yang perilakunya berbeda. GLF = pembaca regime yang sah.

### Term premium / SLR M91 (sovereign duration stress)
- Crisis-elevation: benar 3/4 (COVID +12.7, Luna +5.1, Carry +6.2; FTX -0.96 gagal).
- TAPI state-discrimination **BALIK**: tercile term-prem RENDAH yang vol-nya lebih tinggi
  & drawdown lebih dalam (b-t vol +0.094, maxdd -1.93, SIG). Spearman kontemporan vs vol ≈ 0 (-0.015).
- Interpretasi: **TIDAK konsisten sebagai regime-reader** — ia naik di sebagian krisis
  tapi tak terus-menerus melacak state perilaku; hubungan tercile berlawanan dengan temuan
  krisis. Tidak andal mendeteksi regime.

### ΔM2 liquidity impulse
- State-discrimination: lemah (vol/downside n.s.; maxdd -0.662 SIG, worst_day +0.457 SIG).
  Spearman kecil (downside -0.053, worst_day +0.05).
- Interpretasi: diskriminator perilaku **lemah**; pembacaan impulse hampir tidak memisahkan
  state.

### Order flow / ETF flow (price-derived)
- State-discrimination: sebagian besar n.s. (order_flow hanya worst_day SIG; etf_flow
  downside & maxdd SIG). Spearman kecil.
- Interpretasi: lemah; dan karena price-derived, bukan pembaca state eksternal yang bersih.

## Kesimpulan (objective = deteksi regime)
1. **GLF valid sebagai variabel conditioning regime** — ia memisahkan state perilaku BTC
   dengan kuat (signifikan 4/4, Spearman berarti). Ini mendukung desain SFC (regime
   detection conditioned on liquidity) dan keputusan repo mempertahankan GLF.
2. **Faktor macro-liquidity yang ditolak (term-prem/M91, ΔM2 impulse, flow) lemah juga
   sebagai regime-reader** — penolakan tetap berlaku, tapi kini dengan ALASAN YANG BENAR:
   mereka tidak andal memisahkan state perilaku, bukan (alasan keliru) "tidak prediktif."
3. Koreksi metodologis: bar predictive-OOS tidak tepat untuk komponen SFC; bar yang tepat
   adalah validitas pengukuran (crisis-elevation + state-discrimination + kontemporan).
   Hasil ini konsisten di kedua sisi: GLF = valid condition, sisanya lemah.

## Catatan / langkah lanjut (opsional)
Uji ini adalah state-separation kontemporan. Uji yang lebih langsung untuk "regime detection
system": apakah conditioning pada GLF/likuiditas meningkatkan klasifikasi regime (mis. pisah
bull/bear/stress lebih baik daripada tanpa conditioning). Bisa ditambahkan bila dikehendaki.

## Aset
- `analysis/behavior_reading_validity_test.py` — test suite (reusable)
- `.behavior_reading_test.json` — hasil (gitignored runtime cache)
