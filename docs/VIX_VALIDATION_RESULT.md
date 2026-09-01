# VIX / Risk-Appetite Regime Validation — Hasil (negatif, era-flip)

Upgrade paling menjanjikan (berdasarkan studi inflation-transmission yang menandai VIX
sebagai channel paling robust) diuji sebagai conditioning regime — dan GAGAL era-stability.
Script: `analysis/vix_regime_validation.py`, hasil `.vix_regime_validation.json`.

## Hasil full-sample (menyesatkan!)
- TREND (bull/bear): menambah term_prem_spread (US30Y-US2Y) ke baseline ret20+vol
  **+0.145 AUC** (0.487→0.632); VIX **+0.057** (0.487→0.545). Ini yang paling tinggi
  dari semua conditioning yang pernah diuji — terlihat bagus.
- STRESS (high-drawdown): baseline (realized-vol) sudah 0.759; menambah VIX/macro
  justru MERUGI (−0.065, redundan dengan vol).
- Forward predictive: VIX → next-month OOS R²=−0.07, DM p=0.42 → TIDAK ada edge OOS.

## Era-stability (menentukan)
| variabel | era1 2017-21 | era2 2021-26 | eraA 17-19 | eraB 20-22 | eraC 23-26 |
|---|---|---|---|---|---|
| term_prem_spread | 0.79 | **0.41** | 0.86 | 0.84 | **0.26** |
| VIX | 0.55 | **0.22** | 0.31 | 0.44 | 0.32 |

- term_prem_spread: kuat 2017-2022 (0.79-0.86) → **kolaps/terbalik 2023-2026 (0.26-0.41)**. ERA-FLIP.
- VIX: lemah atau terbalik di SEMUA era (0.22-0.55, mayoritas <0.5). Bukan separator yang stabil.

## Verdict
**VIX dan term-premium-spread DITOLAK sebagai conditioning regime.** "Keunggulan"
full-sample (+0.145/+0.057) adalah artefak era spesifik yang tidak bertahan di era
terbaru. Ini menutup kandidat macro-conditioning terakhir yang masuk akal. Konsisten
dengan seluruh rangkaian: tidak ada conditioning likuiditas/macro/risk-appetite yang
memberikan separasi regime yang era-stable untuk BTC.

## Koreksi klaim sebelumnya
Studi inflation-transmission menandai VIX "1-month-lag predictive." Walk-forward OOS
pada data terkini (2017-2026, n=109) menunjukkan VIX **tidak** prediktif forward
(OOS R²<0). Edge VIX bersifat KONTEMPORAN/dalam-model, bukan walk-forward OOS, dan
era-flip sebagai regime-conditioner. Jangan jadikan VIX input regime/prediksi.

## Implikasi upgrade (objektif)
Upgrade yang "masuk akal secara naratif" (tambah VIX, term-premium, likuiditas, flow,
China CCI) semuanya sekarang tertutup secara empiris. Satu-satunya signal yang era-stable
adalah core stress-gauge (berbasis perilaku). Upgrade SFC yang didukung = pertajam signal
perilaku/stress yang validated + perbaiki kejujuran display (era3 surfacing), BUKAN
menambah conditioning macro.

## Aset
- `analysis/vix_regime_validation.py` — test suite (reusable)
- `.vix_regime_validation.json` — hasil (gitignored runtime cache)
