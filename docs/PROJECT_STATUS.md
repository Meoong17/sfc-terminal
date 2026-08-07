# SFC Terminal — Project Status

_Last updated: 2026-08-07_

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
