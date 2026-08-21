# SFC Terminal — Architecture Map

> **Update**: 10 modules confirmed to have zero `__file__`-relative path
> dependencies were physically moved into `ml/`, `analysis/`, and
> `data_sources/` subfolders (see below). All corresponding imports in
> `collect.py` were updated and verified. Every other module still lives
> at repo root — each was checked individually and found to depend on
> `os.path.dirname(os.path.abspath(__file__))` (or a hardcoded absolute
> path) to locate its own cache/model/data file, so moving those without
> also rewriting each path reference would silently break them. See
> `docs/PROJECT_STATUS.md` for the full verification methodology.

## How to read this map

Each module is listed under the layer it conceptually belongs to. The
`→ factors` column shows which of the five core stress factors
(`Lt`, `St`, `Rt`, `Ft`, `Sc`) it feeds into inside `collect.py`, or
`display` if it only produces dashboard output without affecting
`sfc_pct`/`effective_sfc`.

---

## 1. Entry point & orchestration

| File | Role |
|---|---|
| `collect.py` | Single orchestrator. Runs every pipeline cycle, calls all modules below, computes `sfc_pct` via `calculate_sfc_ensemble()`, writes `data.json`. |
| `sfc-pipeline.sh` | Shell wrapper: retry logic, `flock` locking, git commit/push, `index.html` restore via `git update-index --skip-worktree`. |
| `inject_data.py` | Embeds `data.json` into `index.html` for the static dashboard. Stays at root — invoked directly by `sfc-pipeline.sh` with a hardcoded path. |

## 2. Liquidity layer (core focus of this model)

| File | → factors | Role |
|---|---|---|
| `global_liquidity_engine.py` | Lt | Fed/ECB/BOJ/China/M2/TGA/RRP/DXY weighted composite (GLF). |
| `stablecoin_intelligence.py` | St | Enhanced Stablecoin Liquidity Index (SLI) — mint/burn, reserve ratio, USDT/USDC divergence. Built on top of `stablecoin_liquidity.py`. |
| `stablecoin_liquidity.py` | St | Base M76-M80 stablecoin metrics (supply growth, SSR, exchange flow, velocity, dominance). |
| `fiscal_liquidity.py` | Lt | M83-M84: US Treasury General Account (TGA), Reverse Repo (RRP). |
| `etf_flow.py` | Lt, Rt | M81-M82: spot BTC ETF net flows (Farside-sourced, manually updated — see `update_etf_cache.py`). |
| `liquidity_momentum.py` | — (display) | Rate-of-change of GLF over 30 days. |
| `market_positioning_index.py` | Ft | Funding rate, open interest, liquidation clustering (MPI). |
| `repo_market_stress.py` | Ft | M86: SOFR-EFFR spread — repo/funding market stress. |
| `market_data_fetcher.py` | (feeds M69) | ETH (Binance)/Gold (GoldAPI)/SPX (Twelve Data) for cross-asset analysis. |
| `update_etf_cache.py` / `build_etf_cache.py` | — | Manual Farside data entry + `total_btc` derivation. **Note**: `build_etf_cache.py` uses a hardcoded absolute path (`/home/ubuntu/sfc/...`), not `__file__`-relative — do not move without fixing that first. |

## 3. ML / prediction layer

| File | Feeds into | Role |
|---|---|---|
| `ml_ensemble.py` | display (`ml_ensemble_score`) | RandomForest classifier, price-outcome labeled (fixed from circular `sfc_pct`-derived labels). |
| `ensemble_meta.py` | `effective_sfc` (blended) | XGBoost meta-ensemble over M1-M31, price-outcome labeled. |
| `train_mamba.py` / `mamba_encoder.py` | `sfc_pct` (small nudge) | Mamba state-space model, price-outcome labeled. |
| `hmm_regime.py` | regime override (CRISIS/BEAR) | Gaussian HMM regime detector, features now exclude `sfc_effective` (uses `m2_yoy` instead). |
| `causal_inference.py` | causal blend weights | Granger causality filter — note: cannot detect contemporaneous (same-day) relationships by design. |
| `ml/dynamic_feature_weighting.py` | logging only | **Moved.** Regime-conditional weight tables; `apply_dynamic_weights()` output is NOT wired into `effective_sfc` (only `get_sfc_effective_with_dynamic_weights()` is). |
| `ml/dynamic_feature_selector.py` | filters Mamba input | **Moved.** Per-regime feature filtering for Mamba; `sfc_effective`/`sfc_base` are hardcoded `__core__` (always kept). |
| `ml/feature_engineering.py` | technical indicators | **Moved.** ~14 consolidated technical features (reduced from 25+ redundant oscillators). |
| `ml/sfc_advanced.py` | various | **Moved.** Advanced SFC computation helpers. |
| `online_learning.py` | `effective_sfc` (EWMA correction) | Adaptive baseline correction via EWMA + Kalman filter. |
| `correlation_analysis.py` | standalone tool | Not called from `collect.py`. Run manually (needs live git history) to audit Lt/St weight allocation against real BTC returns. |

## 4. Quantum-hybrid layer

| File | Role |
|---|---|
| `qlstm_model.py` | `QLSTMVolatilityPredictor` — genuine hybrid quantum (PennyLane `AngleEmbedding`+`StronglyEntanglingLayers`) + classical LSTM. Target fixed from circular formula to price-outcome labels. |
| `qlstm_enhanced.py` | Orchestrates model load + GARCH correction + ProAdapt online weighting. `_compute_ensemble_target()` circularity fixed → `_load_latest_resolved_label()`. |
| `qlstm_daemon.py` | Background process, runs inference every 30 min, writes `.qlstm_cache.json`. Decouples expensive quantum inference from `collect.py`'s 5-min cycle — not redundant, see `docs/PROJECT_STATUS.md`. |
| `qlstm_watchdog.py` | Restarts `qlstm_daemon.py` if the PID is dead — mirrors `ws-watchdog.sh` for `binance_ws.py`. |
| `hybrid_correction.py` | GARCH residual correction layer for QLSTM output. |
| `proadapt.py` | Online adaptive weighting between QLSTM/GARCH/hybrid predictions. |
| `xai_explainer.py` | Permutation importance on the QLSTM model — this is the *real* source of `xai_top_features` shown on the dashboard (not `xai_explainer_q5.py`, which is unused). |

## 5. Safety / production-readiness layer

| File | Role |
|---|---|
| `circuit_breaker.py` | Range validation + trip logic. Fixed: clamped values no longer count as failures; `prob_quantiles` now validated. |
| `drift_detection.py` | Feature drift detection. Fixed: broken KS-test replaced with z-test; Benjamini-Hochberg correction added. |
| `confidence_calibration.py` | ECE calibration against real price outcomes. |
| `data_quality.py` | IsolationForest outlier detection + Kalman imputation. Its cleaned output feeds monitoring fields only — `_apply_factor_outlier_guard()` in `collect.py` is the guard that actually protects `sfc_pct`. |
| `market_impact.py` | Almgren-Chriss slippage model for `paper_trader.py`. |

## 6. Data sources / daemons

| File | Role |
|---|---|
| `binance_ws.py` | WebSocket BTC price daemon, writes `btc_ws.json`. `collect.py`'s `get_btc()` checks staleness (5 min threshold) before trusting this file. |
| `data_sources/news_processor.py` | **Moved.** CryptoPanic news sentiment, source-weighted, cross-source confirmation (resets each `collect.py` process). |
| `data_sources/news_sources.py` | **Moved.** News aggregation helpers (black swan detection, stress v2). |
| `liquidation_client.py` | CoinGlass liquidation data. |
| `methods_institutional.py` | Institutional-grade method scores (M72+ range). |
| `onchain_fetch.py` | Whale/exchange flow aggregation. |
| `check_cards.py` | Diagnostic script — reads `data.json`/`btc_ws.json` via hardcoded absolute path. |

## 6.5 Display-only research outputs (institutional, NOT blended into scoring)

These are cautious-rollout, display-only reads — they re-combine signals that already
feed scoring (or read research caches) and are deliberately NOT folded back into
`sfc_effective` / signal / `kelly_fraction`, to avoid double-counting the same signal.

| File | data.json field(s) | Role |
|---|---|---|
| `data_sources/trend_strength.py` | `trend_strength_score`/`_label`/`_domains` | 0-100 Trend Strength Score = weighted blend of momentum (RSI/MACD/OBV), multi-TF alignment, and HMM structure. Default weights 0.40/0.35/0.25, missing domains redistributed. |
| `data_sources/trend_continuation.py` | `cont_prob_*`, `cont_bucket`, `cont_era3_*`, `cont_era_stable_*` | Walk-forward-calibrated P(forward return>0) per signal bucket (CALM/ELEVATED/STRESS), with honest latest-era (era3) values + era-stability flags (avoids era1-inflated full-sample overclaim). |
| `data_sources/momentum_overlay.py` (P4) | `mo_score`/`_bias`/`_regime_action`/`_bucket`/`_confidence` | **Regime-Gated Momentum Overlay.** Raw momentum has no reliable OOS edge (purged-CV rejected it, docs/feature_validation.md), but it is regime-CONDITIONAL. This re-combines `trend_strength.momentum` + `trend_continuation` buckets: it FOLLOWS momentum in regimes where era3 continuation holds (ELEVATED) and FADES it (contrarian) where era3 continuation fails/reverses (STRESS; CALM eroded at 30/90d). Action = margin-weighted vote of `(era3 p_cont - baseline)` across 30/90/180d, era-stability down-weights uncertain horizons. |

## 7. Trading

| File | Role |
|---|---|
| `paper_trader.py` | Simulated LONG/SHORT execution with delay-based slippage (fixed: no longer uses a fixed RNG seed). Sharpe now computed from daily snapshots with √365 annualization. |
| `trading/drl_agent.py` | DRL trading signal (M68) — receives real market state but output is display-only, not blended into `sfc_pct`. |

## 8. Web / dashboard

| File | Role |
|---|---|
| `worker/index.js` | Cloudflare Worker: proxy, multi-user KV state, session auth. Fixed: HMAC-signed session tokens, hardcoded VPS IP moved to secret. |
| `wrangler.toml` | Cloudflare config. Secrets (`BACKUP_URL`, `SESSION_SECRET`) set via `wrangler secret put`, not committed. |
| `sse_server.py` | Server-sent events for live dashboard updates, exposes `/health` with `btc_age`. |
| `index.html` / `app.js` | Dashboard frontend. |

## 9. Not currently wired to `sfc_pct` (display-only or inactive)

| File | Status |
|---|---|
| `models/cnn_attention_module.py` (M65) | Uses real historical method-score windows instead of a constant array. Display-only. |
| `risk/gnn_module.py` (M69) | Wired to real ETH/Gold/SPX data where available; falls back per-asset. `m69_is_simulated` flag reports which case applies. Conceptually risk-contagion, not liquidity. |
| `optimization/genetic_algorithm.py` (M66) | **Deactivated** by user decision — was a literal placeholder. |
| `data_augmentation/timegan_module.py` (M67) | **Deactivated** by user decision — augmented data was never fed back into training. |

## 10. Analysis tools (standalone, not part of the automated pipeline)

| File | Run when |
|---|---|
| `liquidity_lag_analysis.py` | Periodically, to check whether the empirical GLF→BTC lag has shifted. |
| `analysis/liquidity_zscore_calibration.py` | **Moved.** Periodically, to verify hardcoded Fed/ECB/BOJ/M2 mean/std constants against real FRED history. |
| `analysis/multi_timeframe.py` | **Moved.** Ad hoc multi-timeframe alignment check. |
| `analysis/post_process.py` | **Moved.** Post-processing helpers, not called from `collect.py`. |
| `m2_analysis.py` | Ad hoc M2-specific analysis. |
| `backtest_script.py` | Standalone backtest runner. |

---

## Files confirmed unused by the live dashboard (candidates for removal)

None of these output fields (`m70_shap_*`, `m71_lime_*`) are referenced
anywhere in `index.html`/`app.js`:

- `xai_explainer_q5.py` — SHAP input was `torch.randn()` (random noise), LIME was trained on a synthetic hardcoded formula.

