# News Stress Sinkronasi Audit

## Skala Data

### `news_stress` (dari collect.py)
- **Range**: -30.0 sampai +30.0 (di-capped)
- **Negatif**: bearish word count < bullish word count → berita positif → SFC turun
- **Positif**: bearish > bullish → berita negatif → SFC naik
- **Kontribusi ke SFC**: `effective_sfc = sfc_pct + news_stress + liq_mod`
  - Contoh: `sfc_base=13.15 + news_stress=-0.2 + liq_mod=1.8 = effective_sfc=14.75`
  - news_stress = -0.2 artinya berita positif mengurangi SFC 0.2 poin persen

### Field yang Perlu DIVERIFIKASI SKALA

## 1. Display Locations

### A. News Δ header (line 1598)
```
<span>News Δ: ${fmtNum(d.news_stress,1)}</span>
```
**Skala**: `fmtNum(-0.2, 1)` → `-0.2` ✓ (correct, nilai asli)
**Masalah sblmnya**: hardcoded `+` prefix → `+-0.2` → ✅ SUDAH FIX

### B. Security & Events card (line 2064)
```
style="color:${
  (d.news_stress||0)<0     ? 'var(--green)'
  : (d.news_stress||0)>5   ? 'var(--red)'
  : (d.news_stress||0)>1   ? 'var(--amber)'
  : 'var(--text-1)'
}"
```
```
${fmtNum(d.news_stress,1)}pp
```
**Sebelum**: `*100` → `-20%` ❌ SALAH
**Sekarang**: `fmtNum(-0.2,1)` → `-0.2pp` ✅
**Warna**: negatif = hijau ✅ (berita positif = calming)

**Threshold sudah sesuai skala asli** ✅
- `<0` → hijau (calming)
- `0-1` → default (minor)
- `1-5` → amber
- `>5` → merah

## 2. Field Lain — Verifikasi Skala

### cascade_risk (line 1757)
```
${fmtNum((d.cascade_risk||0)*100,0)}%
```
**Data**: `0.1` (0-1.0 scale)
**Display**: `0.1*100 = 10%` ✅ (0-100% scale, benar)

### transition_risk (line 1820)
```
${fmtNum((d.transition_risk||0)*100,0)}%
```
**Data**: `0.2` (0-1.0 scale)
**Display**: `20%` ✅

### readiness_score (line 1821)
```
${((d.readiness_score||0)*100).toFixed(0)}%
```
**Data**: `0.3` (0-1.0 scale)
**Display**: `30%` ✅

### adv_crisis_prob (line 2060)
```
${fmtNum(((d.adv_crisis_prob||0)*100).toFixed(1),1)}%
```
**Data**: `0.05` (0-1.0 scale)
**Display**: `5.0%` ✅

### adv_uncertainty (line 2062)
```
${fmtNum((d.adv_uncertainty||0)*100,1)}%
```
**Data**: `0.15` (0-1.0 scale)
**Display**: `15.0%` ✅

### liq_density (line 1755)
```
${fmtNum((d.liq_density||0)*100,1)}%
```
**Data**: `0.3` (0-1.0 scale)
**Display**: `30.0%` ✅

### kelly_p_win (line 1850)
```
${((d.kelly_p_win||0.5)*100).toFixed(0)}%
```
**Data**: `0.3` (0-1.0 scale)
**Display**: `30%` ✅

### kelly_fraction (line 1855)
```
${((d.kelly_fraction||0)*100).toFixed(1)}%
```
**Data**: `0.1` (0-1.0 scale)
**Display**: `10.0%` ✅

### composite_confidence (line 1518, 1787)
```
${(conf*100).toFixed(0)}%
```
**conf** = `Math.min((d.composite_confidence||0), 0.95)` → `0.3`
**Display**: `30%` ✅

### confidence_components (line 1791)
**Data dari collect.py line 1478-1485**:
  - method_agree: 0.3 (0-1)
  - rsi: -0.07 (-0.10 to +0.03)
  - sopr: -0.05 or 0
  - dvol: 0.0
  - cascade_penalty: -0.10 / -0.05 / 0
  - fear_penalty: -0.06 / 0

```
${(v*100).toFixed(1)}%
```
method_agree = 30.0% ✅ (adalah kontribusi poin persen ke confidence)
rsi = -7.0% ✅ (adalah penalty/boost dalam persen)
semua OK ✅

## 3. fetchBtcLive (line 2177)
```javascript
const floor = currentData
  ? price * (1 - Math.min((currentData.sfc_effective || currentData.sfc || 0), 80) / 100 * 0.6)
  : 0;
```
- `sfc_effective` dari data.json = 14.75 (udah dalam %)
- `14.75 / 100 * 0.6 = 0.0885`
- `1 - 0.0885 = 0.9115`
- `floor = price * 0.9115`
✅ SFC dimasukkan sebagai persen → dibagi 100 → benar

## 4. Cascade Risk di Security Card (line 2055)
```
(d.cascade_risk||0) > 0.5 ? `CASCADE RISK ${((d.cascade_risk||0)*100).toFixed(0)}%`
```
**Data cascade_risk**: 0.1
`0.1*100 = 10%` — tapi threshold CASCADE hanya muncul `>0.5`, jadi untuk 0.1 nggak muncul. Threshold 0.5 = cascade risk 50% — cukup masuk akal. ✅

## 5. Signal Pill & Signal Type

### pillarReason logic (cek di render function)
Perlu lihat bagaimana pillReason dibuat.

Lihat baris-baris signal logic di render()

## Kesimpulan
- **news_stress**: ✅ sudah diperbaiki — display `pp`, warna hijau untuk negatif
- **Semua field lain**: ✅ skala `*100` sudah benar karena field-field dari collect.py memang dalam 0-1 scale
- **Konsistensi**: news_stress satu-satunya field yang skala aslinya ±0..±30 (bukan 0-1)
