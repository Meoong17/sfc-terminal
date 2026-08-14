# SFC Terminal — Project Status

_Last updated: 2026-08-14_

## Model focus

Global liquidity analysis as a predictive factor for BTC price behavior.
Non-liquidity ML overlays (CNN pattern recognition, GNN cross-asset risk,
DRL trading signals) exist but are explicitly out of the primary scope —
see `docs/ARCHITECTURE.md` §9 for their status.

## Audit summary

A full audit (42+ findings) was completed covering every module in this
repo. Categories, with counts:

| Category | Count | Status |
|---|---|---|
| Circular labeling (model trained to predict its own input) | 5 | All fixed — see below |
| Unit-scaling bugs (threshold mismatched to actual data range) | 2 | Fixed |
| Safety layer computed but not wired to the value it should protect | 3 | Fixed |
| Security (session forgery, exposed infrastructure) | 2 | Fixed |
| Formula/calibration errors (ES multiplier, quantile clamping, etc.) | 6 | Fixed |
| Code smell / minor inconsistency | ~15 | Mostly left as-is (documented, non-breaking) |

### Circular labeling — the most significant category

Five independent modules were found to be trained (or evaluated) using a
label/target derived from their own input features, rather than from an
independent ground truth. This is the single most important thing to
understand about this codebase's history:

1. **`ml_ensemble.py`** — label computed from `sfc_pct`, which is itself
   built from the same method scores used as training features. Verified
   empirically: a naive threshold rule with zero training reached ~77%
   agreement with this label.
2. **`ensemble_meta.py`** (XGBoost) — target was `sfc_effective`, built
   from a subset of the same features (M1-M6). A linear regression on
   simulated data reached R²=1.000 exactly this way.
3. **`train_mamba.py`** — same pattern, target = `sfc_effective`, which
   is also one of the 39 input features.
4. **`hmm_regime.py`** — lighter version: `sfc_effective` was one of 5
   clustering features, meaning CRISIS regime detection partly "knew its
   own answer." Not a trained label (HMM is unsupervised), so lower
   severity than #1-3.
5. **`qlstm_model.py`** — target was `features[:, :6] @ weights`, a fixed
   linear combination of M1-M6, which are also part of the LSTM's input
   sequence.

**Fix applied to all five**: replaced the self-referential target with
labels derived from realized BTC price outcome (a lookahead window,
independent of any method score), sourced from `ml_ensemble.py`'s
`resolve_pending_labels()` — one shared source of truth rather than five
separate reimplementations.

**What this means practically**: any historical accuracy/MAE/Sharpe
figures produced before this fix are not meaningful and should not be
compared against post-fix numbers. All affected models need to be
retrained from data collected after the fix (`data_collection.json`'s
`labels` field, populated going forward).

## Why the repo isn't fully reorganized into subfolders

A dependency review found that many modules use `__file__`-relative paths
to locate their own cache/model/data files. An independent re-verification
(grepping every root `.py` file for `__file__` usage, then classifying
each match as a genuine resource-path lookup vs. incidental use) found
**43 of 55 root files** have this dependency — higher than the initial
review's estimate of 24/47, because that review undercounted files using
`SFC_DIR = os.path.dirname(os.path.abspath(__file__))`-style module-level
constants. Two additional files (`build_etf_cache.py`, `check_cards.py`)
were found to use **hardcoded absolute paths** (`/home/ubuntu/sfc/...`)
instead of `__file__`-relative ones — these are arguably more dangerous
to move, since they'd silently keep reading from the OLD location without
raising any error at all, rather than failing to find the file.

**10 files were confirmed to have neither dependency** and were physically
moved, with `collect.py`'s imports updated accordingly:

| Moved to | Files |
|---|---|
| `ml/` | `dynamic_feature_selector.py`, `dynamic_feature_weighting.py`, `feature_engineering.py`, `sfc_advanced.py` |
| `analysis/` | `multi_timeframe.py`, `liquidity_zscore_calibration.py`, `post_process.py` |
| `data_sources/` | `news_processor.py`, `news_sources.py` |

`inject_data.py` was evaluated but kept at root — `sfc-pipeline.sh` calls
it directly by filename (`$PYTHON inject_data.py data.json index.html`),
so moving it would require a shell script change too, for no benefit
since it isn't imported by any other Python module anyway.

**Verification performed**: `py_compile` on every touched file, plus a
runtime `import` test from `collect.py`'s own directory context (with
external libraries the sandbox lacked — `ta`, `vaderSentiment` — mocked
out to isolate whether the *path* resolution specifically worked, since
those import failures are pre-existing environment gaps unrelated to the
move). All 7 relocated-and-imported modules resolved correctly.

The remaining 45 files stay at root. Moving them would additionally
require rewriting each one's path constant (not just `collect.py`'s
import line) — a much larger, higher-risk change deferred for now.
`docs/ARCHITECTURE.md` provides the same organizational clarity via
documentation instead.

## Shell scripts — evaluated for relocation, kept at root

All 5 `.sh` files (`sfc-pipeline.sh`, `mamba-weekly-train.sh`,
`weekly-model-train.sh`, `ws-watchdog.sh`, `update-tunnel-url.sh`) use
`REPO_DIR="$HOME/sfc"`-style absolute paths and `cd` into that directory
before doing anything — so unlike the Python modules, their own physical
location on disk doesn't actually matter to their internal logic.

They were NOT moved into `scripts/` anyway, because of a more important
finding: `sfc-pipeline.sh` contains the comment
`# Cron runs ~/.hermes/scripts/sfc-pipeline.sh — sync both when editing`.
This means the version cron actually executes is a **separate copy** at
`~/.hermes/scripts/`, not the one in this git repo. Moving the repo copy
into a subfolder would do nothing to what cron runs, but would make it
easier to forget that a second copy exists and needs manual syncing —
increasing the risk of the two versions silently diverging, which is a
worse failure mode than a cosmetic root-level file list. Left as-is
intentionally; if this project moves to sourcing cron directly from the
git repo path instead of a separate `~/.hermes/` copy, revisit this.

## Dead code removed

- `xai_explainer_q5.py` — confirmed via `grep` that its output fields
  (`m70_shap_*`, `m71_lime_*`) appear only inside `index.html`/`app.js`'s
  embedded data blob (`var __EMBEDDED_DATA = {...}`, i.e. raw JSON
  `collect.py` produces), never in actual rendering code (no
  `d.m70_shap...` template reference exists, unlike `d.xai_top_features`
  which does appear twice in both files). Removed.
- `xai_shap_runner.py`, `xai_explorer_fix.py` — referenced in earlier
  audit notes as also unused, but were not present in the repo snapshot
  this cleanup was performed against (may have been removed in an earlier
  session, or never committed).

## Known limitations / open items

- `qlstm_daemon.py`, `qlstm_model.py`, `qlstm_enhanced.py`,
  `hybrid_correction.py`, `proadapt.py` live in a separate `sfc2/`
  directory not included in this repo's zip exports — audited via
  individually uploaded files, not the full extracted archive.
- China M2 liquidity component (`global_liquidity_engine.py`) previously used
  FRED series ID `MYAGM2CNM189N`, which resolved but froze at 2019-08-01
  (never returned None, so it silently fed a stale reading into GLF). Replaced
  2026-08 with a chinadata.live CSV fetch (PBC-sourced, monthly, no API key,
  includes a 3-month staleness guard). China YoY z-score calibration was
  re-derived from the 2015–2026 history (mean 9.53%, std 1.73%).
- China M2 calibration status (2026-08): **ESTIMATED, NOT walk-forward
  validated.** `analysis/walk_forward_china_m2.py` (point-in-time expanding
  z-score, no look-ahead, vs BTC forward returns, 101 usable points
  2018-01..2026-05) shows the standalone edge is ERA-UNSTABLE: full-sample gaps
  are significant at 30d (+5.34pp) and 180d (+59.58pp) but NOT at 90d, and the
  era split reverses sign (2015-2020 +54.03pp SIG vs 2021-2026 −7.08pp n.s.).
  The confound check confirms HIGH-z months cluster in 2020-2023 (a bull run),
  so the "expansion→high returns" gap is era-driven, not causal. Do NOT raise
  the 0.04 weight or present this cutoff as validated. The source fix (live data
  vs frozen 2019) stands on its own as a data-integrity improvement.
- `correlation_analysis.py` requires live git commit history to run
  (`extract_snapshots()` calls `git log`) — needs to be run on the VPS,
  not from an extracted zip without `.git`.

## Validation verdicts 2026-08 — candidate factors REJECTED (do not add to model)

Three macro/credit factors were each tested via walk-forward validation
(point-in-time expanding z-score, no look-ahead, bootstrap CI, era-split).
**All three failed to show a robust, era-stable predictive edge. Do NOT blend
any of them into sfc_effective or raise their weights.**

| Factor | Test (script) | Full-sample | Era-split | Verdict |
|---|---|---|---|---|
| China M2 YoY (GLF comp, 0.04 wt) | `walk_forward_china_m2.py` (101 pts 2018-26) | 30d +5.34 SIG, 90d n.s., 180d +59.6 SIG | 2015-20 +54pp SIG vs 2021-26 −7pp n.s. (sign flip) | ESTIMATED, era-unstable |
| JPY carry spread (US10Y−JP10Y) | `walk_forward_carry_jpy.py` (3976 pts 2015-26) | 30d −3.75, 90d −12.99, 180d −11.22 all SIG | 2015-20 −43pp SIG vs 2021-26 +18pp SIG (TOTAL sign flip) | REJECTED, era-dependent |
| HY credit spread (BAMLH0A0HYM2) | `walk_forward_hy_spread.py` (845 pts 2024-26 only) | 30d n.s., 90d +8.19 SIG (wrong dir), 180d −5.95 SIG | n/a (data too short) | NOT validated, contradictory horizons |

Common causes: sign/direction flips across eras or horizons, bucket confounds
(single bull-run dominates), and too-short history (HY only 2024+; no
2015-2020 era split possible). All three remain documented HYPOTHESES, not
confirmed signals. Source fixes stand on their own as data-integrity
improvements (China M2 live data vs frozen 2019 FRED).
- No cron/scheduler for `update_etf_cache.py` exists in this repo — ETF
  flow data (M81/M82) is manually maintained via Farside CSV entry.

## Inflation-transmission adjudication (2026-08-14) — inflation is NOT a direct BTC driver

Adjudication of the "BTC = inflation hedge" hypothesis (see ~/B/1.docx, Model A-D)
via `analysis/inflation_transmission_models.py` (monthly panel 2017-09..2026-06,
n=105, Binance Vision BTC + FRED CPI/PPI/PCE/DFII10/DTWEXBGS/M2SL/VIXCLS, HC1 robust SE).

**Surprise definition (honest limitation):** analyst-consensus series are NOT
obtainable on this host — ForexFactory & Investing.com both return HTTP 403
(Cloudflare), Bloomberg is paid. Surprise is therefore a MODEL-BASED proxy
(MoM vs trailing-12m mean, standardized), labeled explicitly as not-analyst-consensus.
Monthly frequency; the intraday event-study layer (release timestamps) is a
documented follow-up, not run here.

| Model | Regressors | AdjR² | AIC | BIC | Sig vars (p<0.05) |
|---|---|---|---|---|---|
| A (inflation-only) | CPI_s, PPI_s, PCE_s | −0.017 | 938 | 949 | none |
| B (+monetary) | + RealYield, DXY | −0.020 | 940 | 956 | none |
| C (+liquidity) | + M2 YoY | −0.010 | 940 | 959 | none |
| D (FULL) | + VIX | +0.036 | 936 | 958 | VIX (−0.94, p=0.004) |

- **Key test (1.docx §10):** inflation surprise is NOT significant even in Model A
  (CPI p=0.46, PPI p=0.54, PCE p=0.48) and stays insignificant after controls in D
  → inflation is an upstream shock, not a direct driver. Consistent with the
  transmission framing (surprise → real rates/liquidity/risk → BTC).
- **Channels that add explanatory power:** incremental AdjR² A→B (monetary) −0.003,
  B→C (liquidity) +0.010, C→D (risk) +0.046. Risk appetite (VIX, robustly negative)
  and liquidity (M2, positive) are the operative channels; real yield & DXY are weak
  at monthly frequency.
- **Predictive check (1-month lag, no lookahead):** inflation surprise has zero
  predictive power; VIX (−) and M2 (+) remain significant (D-lag AdjR² 0.049).
- **Era-split:** era3 (2022-26) shows CPI_surp (+) vs PCE_surp (−) significant with
  OPPOSITE signs in the same model — incoherent, not a robust inflation driver
  (small n, no multiplicity correction). Do not over-read.
- **Inflation LEVEL (YoY)** also insignificant (AdjR² 0.028) — level is not a useful
  direct BTC indicator either.

**Verdict:** BTC = direct inflation hedge → NOT ESTABLISHED. Do NOT blend inflation
as a BTC driver. Risk appetite (VIX) and liquidity (M2) are the stronger channels and
already represented in the model (VIX via risk factors, M2 via liquidity). Dashboard
CPI YoY display was de-emphasized (removed the ">4% → red" direct-bearish heuristic)
to match this.

## Inflation — uji statistik lengkap (2026-08-14) — KONFIRMASI: bukan driver
`analysis/inflation_assumptions_tests.py` (new) — tiga lapis pengujian pada Model A-D
(surprise proxy model-based MoM-vs-12bln, n=105, 2017-09..2026-06). Menambah uji
instrumen (ADF+Granger), uji asumsi klasik, dan uji hipotesis pada script adjudikasi existing.

**UJI INSTRUMEN:** BTC, CPI/PPI/PCE surprise, VIX = STATIONARY (ADF p<0.001).
RealYield/DXY/M2yoy I(1) level (caveat spurious mild; surprise & VIX stationary).
Granger (monthly) surprise→BTC: PPI l1 p=0.033 & PCE l1 p=0.026 NOMINAL, tapi keduanya
**ARTIFACT** — gagal BH-FDR (0 dari 21 sel q<0.10), gagal sub-period (half2 p=0.20/0.42),
gagal drop-outlier (p=0.99/0.76). VIX→BTC Granger p=0.68 (VIX = barometer coincident,
bukan lead lagged). Tidak ada hubungan lagged surprise yang sahih.

**UJI ASUMSI KLASIK (Model A vs D):** normalitas residual A TOLAK (Shapiro p=0.017) tapi
D lolos (p=0.42) — model penuh lebih baik-spesifikasi. Homoskedastisitas OK (BP/White p>0.1),
HC1 tepat. Autokorelasi: DW≈1.62, Ljung-Box lag1 marginal (p=0.055-0.072) → autocorelasi
positif ringan lag-1 pada return bulanan BTC (Newey-West lebih tepat daripada HC1; caveat).
Multikolinearitas: CPI-PCE berkorelasi (VIF 6.8-7.6) tapi <10, tak parah. Spesifikasi RESET
p=0.24/0.47 → linieritas cukup.

**UJI HIPOTESIS:** F-joint blok inflation surprise = 0 TIDAK signifikan di SEMUA model
(A p=0.78 ... D p=0.56). TES KUNCI: surprise TIDAK pernah signifikan bahkan di Model A
sendiri (CPI p=0.46/PPI 0.54/PCE 0.48) — bukan "hilang setelah kontrol", memang sejak awal
tak ada. Partial-F nested: satu-satunya kanal yang signifikan menaikkan model adalah
**C→D (+VIX risk appetite) p=0.0195**; monetary (+RealYield,DXY) p=0.43 & liquidity (+M2)
p=0.16 TIDAK signifikan. Koefisien VIX di D b=-0.94 p=0.004***; M2 b=+0.98 p=0.07 (marginal).

**VERDICT:** Bitcoin TIDAK merespons inflation surprise secara sahih (kontemporer & lagged,
joint & individual, semua null atau artifact). Respons DOMINAN ke RISK APPETITE (VIX):
negatif, signifikan, satu-satunya penambah model yang signifikan, bertahan di era2. Likuiditas
(M2) lemah/positif-marginal; real yield & DXY tak signifikan pada frekuensi bulanan. Konfirmasi
tesis 1.docx: inflasi = shock hulu yang ditransmisikan lewat kondisi finansial (risk/liquidity),
BUKAN driver langsung. Era3 tetap incoherent (CPI_surp(+) vs PCE_surp(−) lawan tanda). JANGAN
blend inflasi sebagai driver BTC.

## WFV Stress-Gap era-stability (2026-08-14) — edge IS era-stable (unlike L8 subset)

`analysis/walk_forward_validation.py` now writes per-era (era1/era2/era3) calm-vs-stress
forward-return gaps + an `era_stable` flag (era2 AND era3 both significantly negative),
surfaced to the dashboard cards (Weekly 7d / Monthly 30d Stress Gap). Regenerated
`.walk_forward_summary.json` (n=4238, 2026-08-14):

| Horizon | Full-sample | era1 (2014-17) | era2 (2018-21) | era3 (2022-26) | era_stable |
|---|---|---|---|---|---|
| 7d | −1.55pp sig | +4.4pp n.s. | −1.34pp sig | −0.84pp sig | TRUE |
| 30d | −7.46pp sig | +10.8pp n.s. | −4.59pp sig | −5.15pp sig | TRUE |

Unlike the L8 Tail-Risk subset (which era-flips in era3), the CORE stress->return edge
HOLDS across era2 and era3 — both recent eras significantly negative, so the "green /
Significant" dashboard label is era-defensible. Caveat: era1 (partial, small stress
sample) is positive but not significant; the era2/era3 comparison is the regime-relevant
one. Display now shows "Era2 / Era3 / era-stable ✓ or era-flip ⚠" on each card.


## XGBoost meta-ensemble audit (2026-08-07) — blend DISABLED

The XGBoost meta-ensemble (`models/ensemble_meta.py`) was blended into
`effective_sfc` with a weight derived from a circular heuristic confidence.
Audit (all verified empirically from live data) found:

1. **Target is a rare 6h price-drop probability, NOT a stress index.** The
   target `y` = probability BTC falls ≥3% within 6h (base rate 2.8%, positives
   1.8% on 18,287 training samples). Yet it was blended 1:1 into the 0-100
   `effective_sfc` stress scale — a scale/semantic mismatch.
2. **Confidence was circular.** `confidence = 0.5 + 0.5*(1 - |pred-50|/50)`
   was derived from the prediction itself; the tree-std branch was dead code.
3. **Validation was a single 85/15 chronological split** (not walk-forward),
   unlike `analysis/walk_forward_validation.py` (11 years, bootstrap CI).

Actions taken:
- **Blend DISABLED** (`collect.py`): `xgb_blend_weight` pinned to 0.0,
  `effective_sfc` returns to the unblended ensemble (~11-12). XGBoost remains
  display-only, labeled honestly as "P(6h drop)".
- **Circuit breaker synced** (`analysis/circuit_breaker.py`): removed the blend
  term from `mid` reconstruction; scenario tests 8/9 updated to the no-blend
  case. All CB tests pass.
- **Confidence now tree-ensemble derived** (`ensemble_meta.py`): per-tree
  marginal contributions via `Booster.predict(iteration_range=(0,k))`
  differences; confidence = f(CI width / target std). Non-circular.
- **Walk-forward validation added** (`analysis/walk_forward_xgboost.py`). Result
  (`.walk_forward_xgboost.json`, 2026-08-07, 16,191 OOS samples, 4 expanding folds):
  **verdict = STAY_DISABLED.** Pooled AUC 0.514 (≈ random), Brier 0.0089 vs naive
  0.0086 (no better), calibration over-predicts & non-monotonic. Per-fold AUC:
  0.726 → 0.617 → 0.437 → 0.422 — discrimination collapses and inverts in the two
  most recent folds. Not era-stable, not calibrated. Blend stays off.

Re-enable policy: XGBoost may be re-blended only after walk-forward validation
shows calibrated, era-stable discrimination AND the output is scale-calibrated
onto the stress index (percentile mapping, not 1:1).

## Academic methods — staged validation (2026-08-08)

Academic method research (mathematics/economics/coding) → `docs/ACADEMIC_METHODS.md`; implementation sketches → `analysis/sfc_methods_academic.py`. Every method is empirically tested before integration. Results:

| Stage | Method | Empirical test | Verdict |
|-------|--------|-------------|---------|
| 1 | EVT-POT VaR/ES | `analysis/sfc_evt_validate.py` — walk-forward backtest, 1531 OOS days | **NOT validated** vs the incumbent. The incumbent `m11_var` (empirical percentile) is better at 95% (err 0.12 vs EVT 0.20, Kupiec p=0.253 vs 0.059); tied at 99%. EVT is not better → do not replace m11_var. Note: the initial test vs a normal (strawman) baseline was misleading; corrected to compare against the incumbent. |
| 2 | CISS composite | `analysis/sfc_ciss_validate.py` — daily panel of 17 points (git data.json) | **NOT validated.** Sample too small and regime too calm (components nearly constant); composite→forward-30d correlation positive but insignificant (rho +0.22, p=0.39); near-identical to CISS (corr 0.994). Integration deferred until a longer panel with stress episodes is available. |
| 3 | Purged/embargo CV XGBoost | `analysis/walk_forward_xgboost_purged.py` — embargo 6, bootstrap CI | **STAY_DISABLED reinforced.** Pooled AUC 0.486 < 0.5 (CI [0.464,0.563]); calibration gate failed (resolution 0). Embargo REVEALS look-ahead leakage in single-path WFV (early-fold AUC 0.726/0.617 → 0.274/0.383 after purging). XGBoost remains display-only. |

Key lessons: (a) compare new methods against the INCUMBENT (empirical
percentile), not a strawman; (b) models whose labels depend on the future
MUST be validated with purged-CV + embargo (López de Prado), not
single-path WFV — single-path over-estimates skill; (c) purged-CV is now
the recommended validation standard for assessing new factors before they
enter the effective signal.

## 9-year canonical series re-validation (2026-08-08) — docs/canonical_revalidation.md
Walk-forward re-run on the CANONICAL Binance BTCUSDT prices (2017-2026) + era-split:
- **L8 subset (GLF liq + L6 expect): ERA-UNSTABLE.** Strongly significant over the full
  window (30d +0.072***, 90d +0.264***, 365d +0.51***) BUT the era-split flips sign:
  2017-21 +0.20/+0.61 vs 2022-26 −0.11/−0.37. Significance is driven by 2017-21; the effect
  INVERTS in the most recent regime. → do not rely on it for live / do not raise the cutoff.
- **SFC pct (trend continuation): ERA-CONSISTENT at 30d/90d** (era1 +0.047/+0.335,
  era2 +0.056/+0.139, both positive), although 365d inverts (−0.48). More robust
  than L8 at short-to-medium horizons.
Script `analysis/revalidate_canonical.py` (purely analytical, reads cache + Binance, does not
fetch FRED). The monthly Binance cron also re-runs this validation.

## Purged-CV momentum (2026-08-08) — MOMENTUM REJECTED (docs/feature_validation.md)
A non-purged IC screen had flagged momentum (mom_30/90) as the "most robust" feature
(era-stable, passed BH). Leakage-free purged-CV validation (embargo=h, `analysis/
purged_cv_momentum.py`) contradicts it: pooled mom_30 AUC only 0.520 at 7d (marginal),
0.413 at 30d & 0.371 at 90d (BELOW chance); mom_90 <0.5. **Momentum has no reliable
OOS edge** — the apparent edge is an overlapping-label artifact that inflates effective
sample size n. → Do NOT add momentum to the signal/pipeline. Reinforces the
purged-CV/embargo standard (non-purged IC over-estimates skill).

## External-model gate (2026-08-14) — KRONOS REJECTED for BTC (docs: n/a, skill ref)
Evaluated the financial foundation model **Kronos** (shiyu-coder/Kronos, AAAI 2026,
MIT, ~37k stars) on the canonical Binance daily BTC (2017-2026) via walk-forward
OOS. Kronos-mini (4.1M params, ctx 2048): lookback 256 → forecast 30d, stride 30,
100 windows.
- RMSE 5,533 vs naive-persistence 3,778 (Kronos ~146% worse); R² mean −10.0; only
  4% of windows R²>0.
- Sign-accuracy 46% vs the trivial "always-up" 51% baseline (BELOW base rate);
  systemic bear bias (predicts up only 39% vs actual 51%).
- Era-split: 2018-21 sign-acc 53.3% vs 48.9%; 2022-26 40.0% vs 52.7% → the only
  edge was an early-bull artifact, gone in the latest era.
- **Verdict: REJECT.** Foundation model ≠ edge on BTC; pre-trained on equities/
  A-share does not transfer. Do NOT blend, do NOT fine-tune "to rescue" it.
  Fine-tuning cannot manufacture edge absent from the data regime.

## Funding / premium purged-CV (2026-08-14) — REJECTED as predictive drivers
`analysis/purged_cv_funding.py` (new) — the definitive gate for the funding/leverage dim
(1 of 5 L8 dims), which is currently DEAD in live (`m13_funding=None`,
`funding_imbalance=0.0`). Uses the 6+ year Binance Vision funding + premium (canonical,
2020+ full + 2017+ premium via `data_sources/binance_features.py`), López de Prado
purged-CV + embargo=h, 5 folds, logistic on the single feature, pooled OOS AUC + era-split
(era1 2017-21 / era2 2022-26). n=3278 daily, 2017-08-17..2026-08-07.

| Feature | Horiz | Pooled AUC | era1 AUC | era2 AUC | Verdict |
|---|---|---|---|---|---|
| funding | 7d | 0.442 | 0.462 | 0.466 | below chance |
| funding | 30d | 0.392 | 0.373 | 0.450 | below chance |
| funding | 90d | 0.345 | 0.393 | 0.325 | below chance |
| premium | 7d | 0.443 | 0.478 | 0.465 | below chance |
| premium | 30d | 0.396 | 0.424 | 0.444 | below chance |
| premium | 90d | 0.334 | 0.469 | 0.302 | below chance |

**Every pooled AUC is BELOW 0.5, in both eras** — WORSE than chance, i.e. noise (inverted).
The IC-screen "candidate" flag at funding 7d/30d (`validate_features_purged.py`) was an
overlapping-label artifact; purged-CV collapses it exactly as it did momentum (0.413@30d).
**Verdict: REJECT.** Funding/premium carry no genuine OOS predictive information for BTC
up/down. Do NOT wire funding as a scoring driver and do NOT populate the dead
`funding_imbalance` into `sfc_effective`. Keep the live funding dim out of scoring
(`m13_funding=None` is the honest state, not a gap to fill). Summary: `.purged_cv_funding.json`.

## Trend-continuation probabilities — era-split (2026-08-14) — headline overstates today
`analysis/era_split_trend_continuation.py` (new) era-splits the LIVE-displayed
continuation probabilities (cont_prob_30d/90d/180d = P(forward return > 0) per signal
bucket) by reusing the cached `.walk_forward_trend_continuation.json` series (4227 daily,
2014-12→2026-08; era1=1093/era2=1460/era3=1674) — NO FRED re-fetch. Writes
`.trend_continuation_era.json`; `data_sources/trend_continuation.py` now loads it and
surfaces `era3_probability` + `era_stable` per horizon (additive, display-only, no scoring).

Key finding — the full-sample bucket probabilities are INFLATED BY era1's one-way bull run
(2014-17), the classic time-period confound. P(cont) drops sharply after era1:

| Horizon | Full-sample (displayed) | era3 CALM (today) | era_stable |
|---|---|---|---|
| 30d | 0.587 | **0.518** (≈ coin flip) | **FALSE** (era3 dips below baseline) |
| 90d | 0.629 | 0.560 | TRUE (marginal) |
| 180d | 0.694 | 0.669 | FALSE |

Unconditional P(cont) by era confirms the confound: 30d era1 0.701 → era2 0.522 → era3 0.533;
90d 0.79 → 0.531 → 0.521; 180d 0.91 → 0.515 → 0.614. era1 was a $300→$20k bull; era2/era3
are ~coin-flip.

**Verdict: the headline cont_prob_30d=0.587 (and 0.629/0.694) OVERSTATE today's trend
continuation.** Honest today values (era3 CALM) are 0.518/0.560/0.669, and 30d/180d are
era-UNSTABLE. Do NOT present the full-sample number as today's probability without the
era3 value + era_stable flag. Display recommendation: show "Era3 0.518 · era-stable ⚠"
alongside any headline P(cont), mirroring the stress-gap card discipline. No scoring change
(these were already display-only research estimates; this just makes the display honest).

## Early-era data (Kaggle Bitstamp) — baseline + causal + intraday (2026-08-14)
Downloaded `mczielinski/bitcoin-historical-data` (Kaggle, CC BY-SA 4.0, 1-min
Bitstamp OHLCV, 2012-01→present). Cross-checked vs canonical Binance daily
(3,278 overlap days): mean rel. diff 0.190%, 95.8% days within 1% — consistent.
Its only added value is the 2012-2016 window missing from Binance Vision (2017+).
New scripts: `analysis/causal_glf_btc_early_era.py`, `analysis/causal_glf_btc_early_robust.py`,
`analysis/intraday_2012_2017.py`. Results (`.causal_glf_btc_early_era.json`,
`.intraday_2012_2017.json`):

| Question | Result | Verdict |
|---|---|---|
| Trend/momentum baseline 2012-2017 | B&H 2,776x / CAGR 274.8% / maxDD −84.9% beats ALL timing strategies; timing only trims DD to −70% at big return cost | No timing edge in this one-way bull |
| GLF→BTC causal, era 2012-2017 (71 mo) | Granger min_p 0.537; weekly OLS n.s.; OOS GLF HURTS (DM p≈0.01, wrong sign); posterior P(H1)=0.004 | NOT confirmed; matches full-sample GLF null |
| VAR-Granger "significant" p=0.020 @ lag6 | Fails BH-FDR (gate 0.0083), first-half 0.079 n.s., second-half 0.805 n.s., drop-outlier p=0.68 | ARTIFACT — do not report as finding |
| Intraday hourly profile (UTC) | Vol/volume peak ~13-15h, trough ~4-5h; amplitude only ~109 vs 86 index (mild) | Weak seasonality |
| Realized vol clustering | Daily RV autocorr lag1 0.663 (strong); GARCH(1,1) α=0.222 β=0.771, persistence 0.993, half-life ~104d | STRONG volatility clustering |
| Volume-price | corr(daily volume, \|cc return\|) = 0.503 | Modest positive |
| Funding-proxy | NOT feasible from spot OHLCV (no perp funding, no tick direction, no futures basis) | N/A — need Binance Vision funding instead |

Key lesson (echoes existing): a nominally-significant lag in a multi-lag causal
search is USUALLY a multiple-comparison artifact — run the robustness battery
(BH-FDR, sub-period, drop-outlier) before reporting it. The era-2012-2017
GLF→BTC "p=0.020" is the same class as the earlier full-sample GLF result.


