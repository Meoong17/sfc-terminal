# SFC Terminal — Project Status

> Last updated: 2026-07-02

## Health

| Component | Status | Notes |
|-----------|--------|-------|
| Pipeline (collect.py) | 🟢 Running | Auto-commit every ~6 min |
| QLSTM inference | 🟡 Early stage | Trained on 36 labels, 0% stress rate — needs more stress data |
| QLSTM daemon | 🟢 Ready | Watchdog monitors + auto-restarts |
| XAI (M70-M71) | 🟢 Fixed | `xai_explainer_q5.py` restored from BackupSFC (was broken) |
| Dashboard | 🟢 Live | GitHub Pages, tab-based layout |
| Paper trader | 🟢 Running | Simulated trades in paper_history.json |

## Current File Inventory (56 Python modules)

### Root — Core Pipeline (56 flat files)
```
collect.py                  # entry point (3,742 lines)
post_process.py             # final score adjustments
feature_engineering.py      # feature transforms
market_impact.py            # market impact estimation
sfc_advanced.py             # RegimeDetector, UncertaintyQuantifier
probabilistic_output.py     # probability distributions
```

### Root — Liquidity & Market Data (15)
```
global_liquidity_engine.py  # global liquidity factor
stablecoin_intelligence.py  # stablecoin metrics aggregation
stablecoin_liquidity.py     # stablecoin liquidity index
fiscal_liquidity.py         # fiscal liquidity metrics
etf_flow.py                 # ETF flow analysis
liquidity_momentum.py       # liquidity momentum metrics
market_positioning_index.py # market positioning
market_data_fetcher.py      # cross-asset data (FRED, CMC, CoinGecko)
repo_market_stress.py       # repo market stress
onchain_fetch.py            # on-chain metrics (22)
liquidity_lag_analysis.py   # lag correlation analysis
liquidity_zscore_calibration.py # z-score calibration
m2_analysis.py              # M2 money supply analysis
multi_timeframe.py          # multi-timeframe signals
correlation_analysis.py     # correlation analysis
```

### Root — ML & Models (9)
```
ml_ensemble.py              # ensemble model
ensemble_meta.py            # meta-ensemble (XGBoost)
train_mamba.py              # Mamba SSM training
mamba_encoder.py            # Mamba inference encoder
hmm_regime.py               # Hidden Markov Model regime detection
causal_inference.py         # causal inference filter
dynamic_feature_weighting.py # adaptive feature weights
dynamic_feature_selector.py # feature selection
online_learning.py          # EWMA online learning
```

### Root — QLSTM Subsystem (7)
```
qlstm_enhanced.py           # inference module (label-based ProAdapt)
qlstm_model.py              # model definition + training (v3, Pennylane)
qlstm_daemon.py             # background daemon (30-min cycle)
qlstm_watchdog.py           # health monitor + auto-restart
hybrid_correction.py        # QLSTM + GARCH hybrid
proadapt.py                 # online learning weight adaptation
xai_explainer.py            # QLSTM feature importance
```

### Root — Safety & Quality (4)
```
circuit_breaker.py          # trading circuit breaker
drift_detection.py          # feature/concept drift detection
confidence_calibration.py   # prediction confidence calibration
data_quality.py             # data integrity validation
```

### Root — Data Sources & Fetching (6)
```
news_processor.py           # news impact scoring
news_sources.py             # 23-source RSS aggregator
liquidation_client.py       # exchange liquidation data
methods_institutional.py    # institutional flow metrics
update_etf_cache.py         # ETF cache updater
build_etf_cache.py          # ETF cache builder
```

### Root — Trading & Web (4)
```
paper_trader.py             # simulated trading engine
inject_data.py              # data injection (legacy — page fetches data.json live)
sse_server.py               # Server-Sent Events
fetch_historical_btc.py     # historical BTC data fetcher
```

### Root — XAI & Explainability (2)
```
xai_explainer_q5.py         # SHAP + LIME (M70-M71) — restored 2026-07-02
merge_env.py                # environment variable merger
```

### Subdirectories (existing)
```
models/
  cnn_attention_module.py   # CNN + Attention stress scorer
risk/
  gnn_module.py             # GNN systemic risk
optimization/
  genetic_algorithm.py      # genetic algorithm optimizer
trading/
  drl_agent.py              # DRL trading signals (M68)
data_augmentation/
  timegan_module.py         # TimeGAN synthetic data
```

## Known Issues

### 1. QLSTM training data imbalance
- 36 resolved labels, **0% stress rate** (all calm)
- Model predicts ~0 constantly — not useful until stress events accumulate
- Labels resolve ~6 hours after observation via `ml_ensemble.py` `resolve_pending_labels()`
- **Mitigation:** schedule weekly retraining; distribution will improve with more data

### 2. Flat directory structure
- 56 Python files in root — hard to navigate
- Cannot simply `mv` to subdirs: 24 files use `__file__` for resource paths (cache files, model paths, data paths)
- `collect.py` has 94 `try/except` blocks — broken imports fail silently
- **Status:** restructuring deferred (see ARCHITECTURE.md for details)

### 3. collect.py monolithic
- 3,742 lines, single file
- All methods scored inline, no modular decomposition
- Hard to test individual components in isolation

### 4. Silent failure patterns
- `collect.py` pattern: `try: from X import Y ... except ImportError: Y = fallback`
- If a file is moved/deleted, pipeline doesn't crash — silently uses neutral fallback
- No monitoring for which modules are actually running vs falling back

## Recent Changes

| Date | Change | Commit |
|------|--------|--------|
| 2026-07-02 | QLSTM v3 integration (label-based target, watchdog, local deps) | `6f41e10` |
| 2026-07-02 | Fix: restore xai_explainer_q5.py (M70 SHAP + M71 LIME) | `b94c332` |
| 2026-07-02 | Retrained qlstm_model.pt with real labels | `6f41e10` |
| 2026-07-02 | Added pennylane + pennylane-lightning to .venv | `6f41e10` |

## Dependencies Added

```
pennylane==0.45.1
pennylane-lightning==0.45.0
```

## Next Steps (Deferred)

- [ ] Restructure into subdirectories (requires 24+ path fixes + 30+ import updates)
- [ ] Add monitoring for module availability (which modules are in fallback mode)
- [ ] Schedule QLSTM weekly retraining via cron
- [ ] Wait for more stress labels to accumulate before relying on QLSTM signal
