# SFC Feature / Weight Audit (2026-08-08)

Audit "fitur berbobot besar tapi tak berdampak": (1) ukur kontribusi efektif tiap
faktor di composite, (2) uji prediktivitas input faktor terhadap forward return
secara leakage-free (era-split + purged-CV), (3) rekomendasi hapus/kurangi.
Pure analysis — belum ada perubahan produksi.

## 1. Kontribusi efektif faktor (riwayat 11.733 snapshot data.json, sampel 257)
`_FACTOR_WT = {Lt:0.66, St:1.34, Rt:1.0, Ft:1.0, Sc:1.0}` (z_score ensemble).
Rata-rata |value| × weight sebagai % total:

| Faktor | weight | mean|val| | mean|val|×wt | %kontribusi |
|---|---|---|---|---|
| Rt (Sentiment: FNG+whale) | 1.00 | 1.915 | 2.001 | **43.1%** |
| Ft (Systemic: DVOL+buy) | 1.00 | 1.051 | 1.051 | 22.6% |
| Lt (Liquidity: GLO+btc_24h+value) | 0.66 | 0.716 | 0.663 | 14.3% |
| Sc (External: DXY gate) | 1.00 | 0.448 | 0.467 | 10.1% |
| **St (Structural: dom+pc_oi+ms)** | **1.34** | **0.160** | **0.462** | **9.9%** |

Temuan: **St punya weight TERBESAR (1.34) tapi kontribusi efektif TERKECIL (9.9%)**
karena inputnya (dominance ~55, put/call ~0.8) duduk di dekat pusat sigmoid →
faktor nyaris nol (mean|val| 0.160). Weight 1.34 pada dasarnya terbuang.
Sebaliknya Rt (FNG) mendominasi 43% — dan (lihat §2) FNG TIDAK punya edge OOS.

Ketersediaan input: semua terisi (dom/pc_oi/fng/dvol/dxy/m2 98-100%, q10_* 84-87%)
→ tidak ada dead-weight dari input None. Semua terpasang, pertanyaannya adalah
apakah bobotnya layak.

## 2. Uji prediktif input faktor vs forward return (purged-CV, embargo=h, kanonik 9yr)
Skrip `analysis/audit_factor_predictive.py`. pooledAUC = OOS AUC binary up/down
(López de Prado purged-CV); era1IC/era2IC = Spearman IC per era.

| Input (faktor) | h | pooledAUC | era1IC | era2IC | Verdict |
|---|---|---|---|---|---|
| FNG (Rt) | 30d | 0.412 | +0.06 | +0.08 | REVERSED (no edge) |
| FNG (Rt) | 90d | 0.457 | +0.14 | +0.18 | none (no edge) |
| GLO liquidity_stress (Lt) | 30d | 0.360 | −0.17 | +0.28 | REVERSED + era-FLIP |
| GLO liquidity_stress (Lt) | 90d | 0.286 | −0.21 | +0.52 | REVERSED + era-FLIP |
| realized_vol_30d (Ft) | 30d | 0.406 | +0.04 | +0.01 | REVERSED |
| realized_vol_30d (Ft) | 90d | 0.335 | −0.07 | −0.14 | REVERSED |
| momentum_30d (Lt) | 30d | 0.413 | +0.15 | +0.03 | REVERSED (h-inkonsisten) |
| btc_24h raw (Lt) | 90d | 0.325 | +0.05 | +0.03 | REVERSED |

Pola: IC era-split positif & konsisten (mis. FNG +0.14/+0.18) TAPI pooled AUC
purged-CV ≤ 0.5 → **edge hilang saat label-overlap dikoreksi** (sama persis dgn
temuan momentum: IC non-purged over-estimates). GLO liquidity malah terbalik DAN
era-flip → secara arah kontraproduktif.

## 3. Rekomendasi (belum dieksekusi — butuh konfirmasi, menyentuh produksi live)
1. **GLO liquidity_stress (input Lt)** — terbalik + era-flip: kandidat HAPUS dari
   perhitungan Lt (atau re-centering). Bukti paling kuat & paling jelas merugikan.
2. **Rt (FNG)** — kontribusi terbesar (43%) tapi tak ada edge OOS purged-CV: kandidat
   KURANGI weight, bukan naikkan. Bobot besar tak dibenarkan bukti.
3. **St (dominance+put/call)** — weight 1.34 terbesar tapi kontribusi efektif 9.9%
   (input di pusat sigmoid): kandidat NORMALISASI weight ke ~1.0 (atau re-center),
   karena bobot ekstra tidak menghasilkan dampak nyata.
4. Ft (realized_vol) negatif di horizon panjang = tanda risk-premium yang "benar"
   utk input risiko; bukan kandidat hapus — hanya pastikan tanda & horizon jelas.
5. Sc (DXY) kontributor kecil (10%) + gate; tak diuji (butuh sumber DXY) — tidak
   mendesak.

CATATAN interpretasi: model adalah skor STRESS (probabilitas krisis), bukan
prediktor return langsung. "Tak ada edge di forward return" ≠ model tak berguna utk
tali pengaman/drawdown. Namun bukti di sini menolak gagasan bahwa input berwawasan
besar memberi nilai prediktif OOS — jadi menaikkan bobotnya tak bisa dibenarkan.
Perubahan produksi apa pun harus dibahas & disetujui dulu (khususnya karena bobot
mengubah ukuran posisi live).

## 4. Perubahan yang DIAPLIKASIKAN (2026-08-08, setelah konfirmasi user)
Temuan call-site: score_factors_from_market dipanggil (line ~2287) TANPA glo_score →
`glo_score` selalu None di Lt saat runtime; cabang "GLO→Lt" adalah DEAD CODE. GLO
live masuk ensemble via method m33_glo (method_scores_dict['m33_glo']), BUKAN faktor
Lt. Jadi:
1. **Dihapus**: cabang dead code GLO di score_factors_from_market (Lt hanya btc_24h
   + onchain_value + ETF/fiscal adjustments). Tanpa perubahan perilaku (sudah None).
2. **St weight 1.34 → 1.0**: redistribusi ke total tetap 5 (Lt 0.72, St 1.0, Rt 1.09,
   Ft 1.09, Sc 1.10) supaya z_score scale tak bergeser. Dampak kecil (St dorman).
3. **TIDAK disentuh**: m33_glo method — butuh analisis kontribusi ensemble terpisah
   sebelum keputusan (dicatat sebagai follow-up hipotesis, lihat §5).

## 5. Follow-up: m33_glo method — KONFIRMASI INERT (2026-08-08)
Diperiksa tuntas cara m33_glo masuk ke skor akhir:
- sfc_pct = p1·m1m6_avg + p2·new_avg + p3·inst_avg (collect.py ~2645).
- Grup m1m6/m7m19/m20m31 memakai daftar nama HARDCODED m1-m31 (~2611-2613) yang
  TIDAK memuat m33_glo; tidak ada pembacaan downstream atas filtered_scores[
  "m33_glo"].
- m33_glo_score hanya: masuk method_scores_dict (2555), +1 ke total_active_methods
  (count tampilan, 2947), dan dilaporkan ke data.json (4231). NOL dampak ke
  sfc_pct / composite_confidence / sfc_effective.

Kesimpulan: **m33_glo DIHITUNG & DILAPORKAN tapi TIDAK memengaruhi skor** (metode
yatim — tampaknya sempat dimaksudkan masuk ensemble via causal_filter, tapi daftar
grup tak pernah diperluas ke m32/m33). Dan karena sinyal GLO sendiri terbalik +
era-flip (lihat §2), ia TIDAK boleh diwire ke ensemble (akan menambah sinyal
merugikan).

Perubahan yang DITERAPKAN:
1. Hapus kredit `+1` m33_glo dari total_active_methods (2947) — sebelumnya
   menyesatkan (seolah aktif memberi kontribusi). Hanya memengaruhi count tampilan,
   skor tidak berubah.
2. Tambah komentar di method_scores_dict["m33_glo"] (2555) mendokumentasikan status
   inert + larangan wiring.
3. Compute m33_glo TETAP dijalankan & dilaporkan (dashboard/transparansi), tapi
   tidak dianggap memberi kontribusi skor.

Verifikasi: py_compile OK; AST menegaskan total_active_methods tak lagi memakai
m33_glo_score; sfc/composite tak berubah (perubahan hanya display count).
