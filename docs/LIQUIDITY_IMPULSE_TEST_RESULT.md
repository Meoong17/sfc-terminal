# Liquidity Impulse Test — Hasil (1.docx priority #1: ΔM2 / ΔΔM2)

Menguji klaim terkuat dari `C/1.docx`: ΔM2 (liquidity impulse) adalah komponen
paling kuat untuk BTC, berdasarkan Model D' β=+3.44 p=0.002 (IN-SAMPLE). Ini
menguji apakah formulasi impulse/acceleration yang justru diusulkan dokumen
(`L=ΔM2`, `LI=Z(ΔM2)`, `LA=ΔLI`) punya edge OUT-OF-SAMPLE. Script:
`analysis/liquidity_impulse_test.py`, hasil `.liquidity_impulse_test.json`.

## Metode
- M2 (M2SL, FRED), ΔM2 level change; LI = z(ΔM2) trailing 12m; LA = ΔLI.
- Predictor LAGGED 1 bulan (menghormati ~1 bulan lag rilis M2) — tidak ada lookahead.
- Sample: BTC monthly ret 2015-02..2026-08 (n=139).
- Tes: walk-forward OOS expanding (AR vs AR+LI vs AR+LI+LA), Diebold-Mariano,
  purged-CV/embargo OOS AUC (P(BTC_{t+1}>0)), Spearman IC.

## Hasil
| komponen | walk-forward OOS R2 (vs AR) | purged-CV AUC | Spearman IC |
|---|---|---|---|
| LI (z ΔM2) | **-0.011** (DM p=0.77) | 0.468 | 0.079 (p=0.36) |
| LI+LA | **-0.15 .. -0.17** (DM p=0.26) | 0.514 (n.s.) | -0.077 (p=0.37) |
| M2yoy | (via GLF) | 0.458 | 0.093 (p=0.27) |

- Walk-forward OOS R² **NEGATIF** untuk LI dan LI+LA → menambah impulse justru
  memperburuk prediksi dibanding AR-only. LA (acceleration) malah jauh lebih buruk.
- Purged-CV AUC ≤ 0.5 (0.468 / 0.514 / 0.458) → tidak ada skill memprediksi arah.
- Spearman IC semua ≈ 0 dan n.s. → tidak ada hubungan monoton.

## Verdict
**DITOLAK.** ΔM2/ΔΔM2 liquidity impulse tidak punya edge OOS sama sekali.
Signifikansi in-sample (p=0.002) TIDAK bertahan out-of-sample — persis pelajaran
Pitfall 32 (semua model macro monthly OOS R²≤0) dan Pitfall 26. Formulasi impulse/
acceleration dari 1.docx tidak menyelamatkannya. Ini konsisten dengan keputusan repo
2026-08 yang sudah MENGHAPUS m2_yoy dari sfc_effective.

## Status komponen 1.docx/Ya.docx (semua yang bisa diuji SUDAH DITOLAK)
1. ΔM2 Liquidity Impulse  -> DITOLAK (OOS R²<0, AUC≤0.5) — test ini
2. China Credit Impulse  -> DITOLAK (China M2 era-flip, -7pp era terbaru)
3. Real Yield + Term Prem -> DITOLAK (= SLR M91, n.s. & wrong-sign)
4. Behavior layer        -> DITOLAK (momentum + flow ditolak purged-CV)
5. HMM regime            -> ada di repo (behavior_state); conditioning tak menyelamatkan flow
6. Transmission (Imp×Beh) -> DITOLAK (SLR interaction = artefak momentum)
7. Liquidity→Flow→Price   -> DITOLAK (SLR + flow test)
8. XGBoost turun bobot    -> sudah sejalan repo (XGB DITOLAK)

## Implikasi
1.docx/Ya.docx mengusulkan re-arsitektur berbasis transmisi macro-liquidity yang
bukti empiris repo secara konsisten menolak. Satu-satunya yang bertahan adalah sinyal
yang sudah validated repo: core stress-gauge (era-stable 30d gap era2 -4.59/era3 -5.15)
dan Sc-DXY (additive). Sesuai Pitfall 18, berhenti membangun SFC v3 di atas cetak biru ini.

## Aset
- `analysis/liquidity_impulse_test.py` — test suite (reusable, lagged predictor)
- `.liquidity_impulse_test.json` — hasil (gitignored runtime cache)
