# SFC Terminal — Architecture

## Overview

SFC (Stress Forecasting Composite) is a real-time BTC market stress indicator running as a GitHub Pages dashboard. The pipeline collects data every ~6 minutes, scores market stress through 30+ methods, and pushes results to a static dashboard.

**Entry point:** `collect.py` (3,742 lines) — single monolithic collector called by `sfc-pipeline.sh`.

## Pipeline Flow

```
cron (sfc-pipeline.sh, every ~6 min)
  │
  ▼
collect.py
  ├── 1. Data collection (on-chain, cross-asset, news, liquidations)
  ├── 2. Method scoring (30+ methods → individual stress signals)
  ├── 3. Ensemble aggregation (ml_ensemble + ensemble_meta)
  ├── 4. Circuit breaker + drift detection + calibration
  ├── 5. QLSTM inference (via daemon cache or fallback)
  ├── 6. Paper trading (paper_trader.py)
  └── 7. Output → data.json, paper_trades.json, paper_history.json
       │
       ▼
  git commit + push → GitHub Pages (index.html + app.js)
```

## Subsystems

### QLSTM (Quantum LSTM) — `6f41e106`
Hybrid quantum-classical model for volatility prediction.

```
qlstm_watchdog.py  ──cron 10min──▶  monitors daemon health
qlstm_daemon.py    ──30min cycle──▶  writes .qlstm_cache.json
qlstm_enhanced.py  ──fallback────▶  direct inference if cache stale

Companion modules:
  qlstm_model.py        — model definition + training (v3, Pennylane)
  hybrid_correction.py  — QLSTM + GARCH residual correction
  proadapt.py           — online learning weight adaptation
  xai_explainer.py      — QLSTM feature importance (permutation)

Training target: Real BTC price-outcome labels (not circular M1-M6 formula)
```

### Core Methods (30+)
- M1-M6: Statistical (KLR, Logit, Bayes, EWC, QReg, Regime)
- M7-M19: Advanced statistical (Fisher, Monte Carlo, VaR, CVaR, etc.)
- M20-M31: Market microstructure (OBI, TradeFlow, Spread, CAPE, etc.)
- M32: QLSTM (Hybrid Quantum LSTM)
- M65-M69: ML ensemble features (CNN Attention, GA, TimeGAN, DRL, SystemicRisk)
- M70-M71: XAI explainability (SHAP + LIME)
- M72-M80: Macro liquidity + stablecoin metrics

### Safety Layer
- `circuit_breaker.py` — halts trading on abnormal conditions
- `drift_detection.py` — detects feature/concept drift
- `confidence_calibration.py` — calibrates prediction confidence
- `data_quality.py` — validates input data integrity

### Post-Processing
- `post_process.py` — final score adjustments
- `feature_engineering.py` — feature transformations
- `market_impact.py` — market impact estimation
- `sfc_advanced.py` — RegimeDetector, UncertaintyQuantifier
- `probabilistic_output.py` — probability distributions instead of point estimates

### Data Sources
- `market_data_fetcher.py` — cross-asset market data (FRED, CMC, CoinGecko)
- `onchain_fetch.py` — on-chain metrics (22 metrics)
- `news_processor.py` + `news_sources.py` — 23-source news aggregator
- `liquidation_client.py` — exchange liquidation data
- `methods_institutional.py` — institutional flow metrics
- `update_etf_cache.py` + `build_etf_cache.py` — ETF flow data

### Trading
- `paper_trader.py` — simulated trading engine
- `trading/drl_agent.py` — DRL-based trading signals (M68)

### Dashboard
- `index.html` + `app.js` — GitHub Pages SPA dashboard
- `sse_server.py` — Server-Sent Events for live updates

## Key Architectural Decisions

1. **Bare imports, not packages** — `collect.py` uses `from module import ...` for all 30+ modules. No Python packages (`__init__.py`). This is why restructuring into subdirectories is high-risk (see PROJECT_STATUS.md).

2. **Silent fallbacks** — 94 `try/except` blocks in collect.py. Most `except ImportError` silently fall back to neutral values. A broken import path causes NO crash — just silently degraded output.

3. **File-based caching** — Most modules write JSON cache files alongside themselves (`.qlstm_cache.json`, `.global_liquidity_cache.json`, etc.). These use `os.path.dirname(__file__)` — relocating a file breaks its cache path.

4. **Dual QLSTM path** — Daemon (primary, fast) + fallback direct import (slow, loads torch). Watchdog ensures daemon stays alive.

## Dependencies

- Python: numpy, scipy, pandas, scikit-learn, torch, pennylane, pennylane-lightning
- Quantum: Pennylane with lightning.qubit backend (C++)
- XAI: xai_venv (isolated venv for SHAP/LIME to avoid numpy conflicts)
- External APIs: FRED, CoinMarketCap, CoinGecko, Binance, OKX, news RSS feeds

## Directory Structure (current)

```
sfc/
├── collect.py              # entry point (3,742 lines)
├── sfc-pipeline.sh         # cron wrapper
├── *.py                    # 50+ flat modules (see PROJECT_STATUS.md)
├── models/                 # CNN attention, XGBoost meta
├── risk/                   # GNN systemic risk
├── optimization/           # Genetic algorithm
├── trading/                # DRL agent
├── data_augmentation/      # TimeGAN
├── worker/                 # Cloudflare Worker
├── docs/                   # Documentation (this file)
├── index.html + app.js     # GitHub Pages dashboard
├── data.json               # live pipeline output
├── data_collection.json    # time-series feature store
├── .qlstm_cache.json       # QLSTM daemon output
└── .venv/                  # virtual environment
```
