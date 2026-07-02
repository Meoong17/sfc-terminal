# SFC Terminal — Project Status

_Last updated: 2026-07-02_

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
- China M2 liquidity component (`global_liquidity_engine.py`) uses FRED
  series ID `MYAGM2CNM189N`, added without live network verification —
  confirm this resolves on FRED before relying on it; the code fails
  safe (component simply excluded) if the ID is wrong.
- `correlation_analysis.py` requires live git commit history to run
  (`extract_snapshots()` calls `git log`) — needs to be run on the VPS,
  not from an extracted zip without `.git`.
- No cron/scheduler for `update_etf_cache.py` exists in this repo — ETF
  flow data (M81/M82) is manually maintained via Farside CSV entry.
