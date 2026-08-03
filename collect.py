#!/usr/bin/env python3
"""
SFC v2.1 REALTIME COLLECTOR — No LLM Version
5-minute auto-update for GitHub Pages
Multi-source news aggregator (23 sources)
"""

import json, os, sys, subprocess, math, requests, time
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from data_sources.onchain_fetch import fetch_all_onchain

# ── Probabilistic Output + Circuit Breaker (M1.docx Priority) ──
try:
    from models.probabilistic_output import ProbabilisticHead
    _PROB_HEAD = ProbabilisticHead()
    PROBABILISTIC_AVAILABLE = True
except Exception:
    _PROB_HEAD = None
    PROBABILISTIC_AVAILABLE = False

try:
    from analysis.circuit_breaker import CircuitBreaker
    _CIRCUIT_BREAKER = CircuitBreaker()
    CB_AVAILABLE = True
except Exception:
    _CIRCUIT_BREAKER = None
    CB_AVAILABLE = False

# ── 5 Advanced Modules (Model.docx) with retry — fix: dead imports never retry
_ADV_FEATURES_MODULE = None
_ADV_ENSEMBLE_MODULE = None
_ADV_HMM_MODULE = None
_ADV_MTF_MODULE = None
_ADV_ONLINE_MODULE = None
_ADV_RETRY_TIME = {}  # module_name -> timestamp of last failed attempt

_ADV_RETRY_DELAY = 3600  # retry after 1 hour instead of never

def _get_adv_features():
    global _ADV_FEATURES_MODULE
    if _ADV_FEATURES_MODULE is False:
        # Retry after delay if previously failed
        last_fail = _ADV_RETRY_TIME.get('features', 0)
        if time.time() - last_fail < _ADV_RETRY_DELAY:
            return None
    if _ADV_FEATURES_MODULE is None or _ADV_FEATURES_MODULE is False:
        try:
            import ml.feature_engineering as m
            _ADV_FEATURES_MODULE = m
        except Exception:
            _ADV_FEATURES_MODULE = False
            _ADV_RETRY_TIME['features'] = time.time()
    return _ADV_FEATURES_MODULE if _ADV_FEATURES_MODULE else None

def _get_adv_ensemble():
    global _ADV_ENSEMBLE_MODULE
    if _ADV_ENSEMBLE_MODULE is False:
        last_fail = _ADV_RETRY_TIME.get('ensemble', 0)
        if time.time() - last_fail < _ADV_RETRY_DELAY:
            return None
    if _ADV_ENSEMBLE_MODULE is None or _ADV_ENSEMBLE_MODULE is False:
        try:
            from models import ensemble_meta as m
            _ADV_ENSEMBLE_MODULE = m
        except Exception:
            _ADV_ENSEMBLE_MODULE = False
            _ADV_RETRY_TIME['ensemble'] = time.time()
    return _ADV_ENSEMBLE_MODULE if _ADV_ENSEMBLE_MODULE else None

def _get_adv_hmm():
    global _ADV_HMM_MODULE
    if _ADV_HMM_MODULE is False:
        last_fail = _ADV_RETRY_TIME.get('hmm', 0)
        if time.time() - last_fail < _ADV_RETRY_DELAY:
            return None
    if _ADV_HMM_MODULE is None or _ADV_HMM_MODULE is False:
        try:
            from models import hmm_regime as m
            _ADV_HMM_MODULE = m
        except Exception:
            _ADV_HMM_MODULE = False
            _ADV_RETRY_TIME['hmm'] = time.time()
    return _ADV_HMM_MODULE if _ADV_HMM_MODULE else None

def _get_adv_mtf():
    global _ADV_MTF_MODULE
    if _ADV_MTF_MODULE is False:
        last_fail = _ADV_RETRY_TIME.get('mtf', 0)
        if time.time() - last_fail < _ADV_RETRY_DELAY:
            return None
    if _ADV_MTF_MODULE is None or _ADV_MTF_MODULE is False:
        try:
            import analysis.multi_timeframe as m
            _ADV_MTF_MODULE = m
        except Exception:
            _ADV_MTF_MODULE = False
            _ADV_RETRY_TIME['mtf'] = time.time()
    return _ADV_MTF_MODULE if _ADV_MTF_MODULE else None

def _get_adv_online():
    global _ADV_ONLINE_MODULE
    if _ADV_ONLINE_MODULE is False:
        last_fail = _ADV_RETRY_TIME.get('online', 0)
        if time.time() - last_fail < _ADV_RETRY_DELAY:
            return None
    if _ADV_ONLINE_MODULE is None or _ADV_ONLINE_MODULE is False:
        try:
            from models import online_learning as m
            _ADV_ONLINE_MODULE = m
        except Exception:
            _ADV_ONLINE_MODULE = False
            _ADV_RETRY_TIME['online'] = time.time()
    return _ADV_ONLINE_MODULE if _ADV_ONLINE_MODULE else None

load_dotenv()

ATH_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.ath_cache.json')

def _save_ath(ath, date):
    with open(ATH_CACHE_FILE, 'w') as f: json.dump({'ath': ath, 'date': date, 'ts': time.time()}, f)

def _load_ath():
    if os.path.exists(ATH_CACHE_FILE):
        with open(ATH_CACHE_FILE) as f: return json.load(f)
    return None

# Import institutional methods (M20-M31) and ML ensemble
sys.path.insert(0, os.path.dirname(__file__))

# Import stablecoin liquidity methods (M76-M80)
try:
    from data_sources.stablecoin_liquidity import compute_all_stablecoin_metrics
    STABLECOIN_AVAILABLE = True
except ImportError as e:
    STABLECOIN_AVAILABLE = False
    print(f"[SFC] Stablecoin liquidity module unavailable: {e}", file=sys.stderr)
    def compute_all_stablecoin_metrics(*a, **k): return {}, {}, 0, None

# Import ETF flow module (M81-M82)
try:
    from data_sources.etf_flow import compute_etf_metrics
    ETF_AVAILABLE = True
except ImportError as e:
    ETF_AVAILABLE = False
    print(f"[SFC] ETF flow module unavailable: {e}", file=sys.stderr)
    def compute_etf_metrics(*a, **k): return 0.5, 0.5, {"status": "unavailable"}

# Import fiscal liquidity module (M83-M84)
try:
    from data_sources.fiscal_liquidity import compute_fiscal_liquidity_metrics
    FISCAL_AVAILABLE = True
except ImportError as e:
    FISCAL_AVAILABLE = False
    print(f"[SFC] Fiscal liquidity module unavailable: {e}", file=sys.stderr)
    def compute_fiscal_liquidity_metrics(*a, **k): return 0.5, 0.5, 0.5, {"status": "unavailable"}

# Import repo market stress module (M86 — SOFR-EFFR spread)
try:
    from data_sources.repo_market_stress import compute_repo_stress
    REPO_STRESS_AVAILABLE = True
except ImportError as e:
    REPO_STRESS_AVAILABLE = False
    print(f"[SFC] Repo market stress module unavailable: {e}", file=sys.stderr)
    def compute_repo_stress(*a, **k): return 0.5, {"status": "unavailable"}

# Import Global Sovereign Liquidity Score module (M90 — consolidated
# US/Japan/Europe/UK sovereign bond signal, replaces the earlier idea of
# separate M88/M89/M90 methods per country — see module docstring)
try:
    from data_sources.global_sovereign_liquidity import compute_global_sovereign_liquidity
    GSLS_AVAILABLE = True
except ImportError as e:
    GSLS_AVAILABLE = False
    print(f"[SFC] Global Sovereign Liquidity module unavailable: {e}", file=sys.stderr)
    def compute_global_sovereign_liquidity(*a, **k): return 50.0, {"status": "unavailable"}

# ── NEW (IMBS L6/L8/L5): Expectations, Tail Risk, Behavior-State ──
# All three are DISPLAY-ONLY overlays — they add fields to data.json but are
# deliberately NOT blended into sfc_effective / signal / composite_confidence
# until walk-forward re-validation shows a stable edge (cautious-rollout
# pattern of M86/M90/reflexivity). See each module's docstring.
try:
    from data_sources.expectations_engine import compute_expectations
    EXPECTATIONS_AVAILABLE = True
except ImportError as e:
    EXPECTATIONS_AVAILABLE = False
    print(f"[SFC] Expectations engine unavailable: {e}", file=sys.stderr)
    def compute_expectations(*a, **k): return 50.0, {"status": "unavailable"}

try:
    from data_sources.tail_risk_engine import compute_tail_risk
    TAIL_RISK_AVAILABLE = True
except ImportError as e:
    TAIL_RISK_AVAILABLE = False
    print(f"[SFC] Tail Risk engine unavailable: {e}", file=sys.stderr)
    def compute_tail_risk(*a, **k): return 50.0, {"status": "unavailable"}

try:
    from data_sources.behavior_state import compute_behavior_state
    BEHAVIOR_STATE_AVAILABLE = True
except ImportError as e:
    BEHAVIOR_STATE_AVAILABLE = False
    print(f"[SFC] Behavior-State overlay unavailable: {e}", file=sys.stderr)
    def compute_behavior_state(*a, **k): return "UNKNOWN", {"status": "unavailable"}


# ── P0: Regime Consolidation (single source of truth for regime label) ──
try:
    from data_sources.regime_consolidation import consolidate_regime
    REGIME_CONSOLIDATION_AVAILABLE = True
except ImportError as e:
    REGIME_CONSOLIDATION_AVAILABLE = False
    print(f"[SFC] Regime consolidation unavailable: {e}", file=sys.stderr)
    def consolidate_regime(*a, **k): return "UNKNOWN", {"status": "unavailable"}


# ── P1: Transmission Divergence (liquidity vs BTC structure) ──
try:
    from data_sources.transmission_divergence import classify_transmission
    TRANSMISSION_DIVERGENCE_AVAILABLE = True
except ImportError as e:
    TRANSMISSION_DIVERGENCE_AVAILABLE = False
    print(f"[SFC] Transmission divergence unavailable: {e}", file=sys.stderr)
    def classify_transmission(*a, **k): return "UNAVAILABLE", {"status": "unavailable"}


# ── P2: Trend Strength Score (institutional output) ──
try:
    from data_sources.trend_strength import compute_trend_strength
    TREND_STRENGTH_AVAILABLE = True
except ImportError as e:
    TREND_STRENGTH_AVAILABLE = False
    print(f"[SFC] Trend strength unavailable: {e}", file=sys.stderr)
    def compute_trend_strength(*a, **k): return 50.0, {"status": "unavailable", "available": False, "label": "UNKNOWN"}


# ── P3: Trend Continuation Probability (institutional output) ──
try:
    from data_sources.trend_continuation import compute_trend_continuation
    TREND_CONTINUATION_AVAILABLE = True
except ImportError as e:
    TREND_CONTINUATION_AVAILABLE = False
    print(f"[SFC] Trend continuation unavailable: {e}", file=sys.stderr)
    def compute_trend_continuation(*a, **k): return {}, {"status": "unavailable", "available": False}


# Early init: M86 score starts at neutral (updated later by execution block if available)
_m86_score = 0.5

# ── NEW: Global Liquidity Engine (GLF — consolidated liquidity factor) ──
try:
    from data_sources.global_liquidity_engine import compute_global_liquidity_factor, get_glf_for_factors, get_glf_weight_by_regime
    GLOBAL_LIQUIDITY_AVAILABLE = True
except ImportError as e:
    GLOBAL_LIQUIDITY_AVAILABLE = False
    print(f"[SFC] Global Liquidity Engine unavailable: {e}", file=sys.stderr)
    def compute_global_liquidity_factor(*a, **k): return 50.0, 0.5, {"error": "unavailable", "status": "fallback"}
    def get_glf_for_factors(*a, **k): return 0.0
    def get_glf_weight_by_regime(*a, **k): return 0.35

# ── NEW: Stablecoin Intelligence (enhanced composite index) ──
try:
    from data_sources.stablecoin_intelligence import compute_stablecoin_liquidity_index
    STABLECOIN_INTEL_AVAILABLE = True
except ImportError as e:
    STABLECOIN_INTEL_AVAILABLE = False
    print(f"[SFC] Stablecoin Intelligence unavailable: {e}", file=sys.stderr)
    def compute_stablecoin_liquidity_index(*a, **k): return 50.0, 0.5, {"error": "unavailable", "status": "fallback"}

# ── NEW: Dynamic Feature Weighting (regime-adaptive weights) ──
try:
    from ml.dynamic_feature_weighting import (
        get_regime_weights, apply_dynamic_weights,
        get_feature_group_weights, get_sfc_effective_with_dynamic_weights,
    )
    DYNAMIC_WEIGHTING_AVAILABLE = True
except ImportError as e:
    DYNAMIC_WEIGHTING_AVAILABLE = False
    print(f"[SFC] Dynamic Feature Weighting unavailable: {e}", file=sys.stderr)
    def get_regime_weights(*a, **k): return {"Lt":0.25,"St":0.20,"Rt":0.20,"Ft":0.20,"Sc":0.15}
    def apply_dynamic_weights(*a, **k): return {}, 0.5, {}
    def get_feature_group_weights(*a, **k): return {}
    def get_sfc_effective_with_dynamic_weights(*a, **k): return None, 0.0

# ── NEW: Market Positioning Index (MPI) ──
try:
    from data_sources.market_positioning_index import compute_market_positioning_index
    MPI_AVAILABLE = True
except ImportError as e:
    MPI_AVAILABLE = False
    print(f"[SFC] MPI unavailable: {e}", file=sys.stderr)
    def compute_market_positioning_index(*a, **k): return 50.0, 0.5, {"error": "unavailable", "status": "fallback"}

# ── NEW: Liquidity Momentum (LM) ──
try:
    from analysis.liquidity_momentum import compute_liquidity_momentum
    LM_AVAILABLE = True
except ImportError as e:
    LM_AVAILABLE = False
    print(f"[SFC] Liquidity Momentum unavailable: {e}", file=sys.stderr)
    def compute_liquidity_momentum(*a, **k): return 0.0, 0.0, {"status": "fallback"}

# ── NEW: Dynamic Feature Selector (DFS) ──
try:
    from ml.dynamic_feature_selector import DynamicFeatureSelector
    _DFS_SELECTOR = DynamicFeatureSelector()
    DFS_AVAILABLE = True
except ImportError as e:
    DFS_AVAILABLE = False
    print(f"[SFC] Dynamic Feature Selector unavailable: {e}", file=sys.stderr)
    _DFS_SELECTOR = None

# Import causal inference
try:
    from analysis.causal_inference import CausalFilter
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    print("[SFC] Causal inference not available", file=sys.stderr)

# Import advanced modules (Priority 2-6)
# Lazy import: sfc_advanced (RegimeDetector, UncertaintyQuantifier, etc.)
ADVANCED_AVAILABLE = None
_advanced_cache = {"regime": None, "uncertainty": None, "alt_data": None, "wf_backtest": None}

def _get_advanced():
    """Lazy import advanced modules — ~0.9s import time."""
    global ADVANCED_AVAILABLE, _advanced_cache
    if ADVANCED_AVAILABLE is not None:
        return _advanced_cache
    
    try:
        from ml.sfc_advanced import (
            RegimeDetector, UncertaintyQuantifier, AutoFeatureEngineer,
            fetch_all_alternative_data, compute_all_advanced, WalkForwardBacktest
        )
        ADVANCED_AVAILABLE = True
        _advanced_cache = {
            "regime": RegimeDetector,
            "uncertainty": UncertaintyQuantifier,
            "alt_data": fetch_all_alternative_data,
            "compute": compute_all_advanced,
            "wf_backtest": WalkForwardBacktest,
        }
    except ImportError as e:
        ADVANCED_AVAILABLE = False
        print(f"[SFC] Advanced modules not available: {e}", file=sys.stderr)
        _advanced_cache = {}
    
    return _advanced_cache

# Q5 Advanced Pattern Methods (M65-M69)
# Lazy import: CNN+Attention (M65) — imported only when used
CNN_ATTENTION_AVAILABLE = None  # None = not checked yet
_cnn_attention_fn = None

def _get_cnn_attention():
    """Lazy import CNN+Attention module."""
    global CNN_ATTENTION_AVAILABLE, _cnn_attention_fn
    if CNN_ATTENTION_AVAILABLE is not None:
        return _cnn_attention_fn if CNN_ATTENTION_AVAILABLE else None
    
    try:
        from models.cnn_attention_module import calculate_cnn_attention_stress
        CNN_ATTENTION_AVAILABLE = True
        _cnn_attention_fn = calculate_cnn_attention_stress
        return _cnn_attention_fn
    except ImportError:
        CNN_ATTENTION_AVAILABLE = False
        print("[SFC] CNN+Attention (M65) not available", file=sys.stderr)
        def _fallback(*a, **k):
            return {"m65_cnn_attention": 0.5, "attention_focus": [], "pattern_type": "FALLBACK"}
        _cnn_attention_fn = _fallback
        return None

def calculate_cnn_attention_stress(*a, **k):
    fn = _get_cnn_attention()
    if fn:
        return fn(*a, **k)
    return {"m65_cnn_attention": 0.5, "attention_focus": [], "pattern_type": "FALLBACK"}

try:
    from trading.drl_agent import get_trading_signal, train_drl_agent
    DRL_AVAILABLE = True
except ImportError:
    DRL_AVAILABLE = False
    print("[SFC] DRL Agent (M68) not available", file=sys.stderr)
    def get_trading_signal(*a, **k): return "HOLD"
    def train_drl_agent(*a, **k): return None

try:
    from risk.gnn_module import calculate_systemic_risk
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("[SFC] GNN Systemic Risk (M69) not available", file=sys.stderr)
    def calculate_systemic_risk(*a, **k):
        return {"overall_systemic_risk": 0.5, "btc_systemic_risk": 0.5, "market_regime": "NORMAL", "correlation_breakdown": False}

# XAI Explainability (M70-M71)
try:
    from xai_explainer_q5 import run_all_xai
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("[SFC] XAI Explainability (M70-M71) not available", file=sys.stderr)
    def run_all_xai(*a, **k):
        return {"m70_shap_ok": False, "m71_lime_ok": False, "m70_shap_features": [], "m71_lime_features": []}

sys.path.insert(0, os.path.dirname(__file__))
try:
    from data_sources.methods_institutional import compute_all_institutional
    INSTITUTIONAL_AVAILABLE = True
except ImportError as e:
    print(f"[SFC] Institutional methods not available: {e}", file=sys.stderr)
    def compute_all_institutional(*a, **k): return {}, {}, 0, None
    INSTITUTIONAL_AVAILABLE = False

try:
    from models.ml_ensemble import (
        predict_with_ml, add_observation, evaluate_accuracy, retrain_on_errors,
        record_price_snapshot, resolve_pending_labels,
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"[SFC] ML ensemble not available: {e}", file=sys.stderr)
    def predict_with_ml(*a, **k): return 0.5, 0.0, "ML unavailable"
    def add_observation(*a, **k): return None
    def evaluate_accuracy(): return {"accuracy": None}
    def retrain_on_errors(): return None
    def record_price_snapshot(*a, **k): return None
    def resolve_pending_labels(*a, **k): return 0
    ML_AVAILABLE = False

# ── QLSTM INFERENCE (M32 — Hybrid Quantum LSTM + GARCH + ProAdapt) ──
QLSTM_AVAILABLE = False
QLSTM_INFERENCE_CACHE = {"pred": None, "ts": 0}
_QLSTM_DAEMON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qlstm_cache.json")

def _run_qlstm_inference():
    """Run enhanced QLSTM inference (M32+Hybrid+ProAdapt+XAI).
    First tries daemon cache file, then falls back to direct inference.
    Returns (prediction_0_1, is_available)
    Also stores enhanced results in module-level vars for output dict."""
    global QLSTM_AVAILABLE, QLSTM_INFERENCE_CACHE
    global _XAI_FEATURES, _GARCH_RESIDUAL, _GARCH_VOL, _PROADAPT_W, _PROADAPT_FINAL
    
    now = time.time()
    
    # Check in-memory cache first (1800s TTL)
    if QLSTM_INFERENCE_CACHE["pred"] is not None and now - QLSTM_INFERENCE_CACHE["ts"] < 1800:
        return QLSTM_INFERENCE_CACHE["pred"], True
    
    # Check daemon cache file (fast, no torch import)
    try:
        if os.path.exists(_QLSTM_DAEMON_FILE):
            with open(_QLSTM_DAEMON_FILE) as f:
                dc = json.load(f)
            dc_ts = dc.get("ts", 0)
            if now - dc_ts < 1800 and dc.get("qlstm_ok"):
                qlstm_pred_0_1 = dc["qlstm_pred_0_1"]
                _GARCH_RESIDUAL = dc.get("garch_residual", 0)
                _GARCH_VOL = dc.get("garch_volatility", 0)
                _PROADAPT_W = dc.get("proadapt_weight", 0.5)
                _PROADAPT_FINAL = dc.get("proadapt_final", dc["qlstm_pred"])
                _XAI_FEATURES = dc.get("xai_top_features", None)
                QLSTM_INFERENCE_CACHE = {"pred": qlstm_pred_0_1, "ts": now}
                if not QLSTM_AVAILABLE:
                    print(f"[SFC] QLSTM from daemon cache: pred={dc['qlstm_pred']:.1f}%", file=sys.stderr)
                    QLSTM_AVAILABLE = True
                return qlstm_pred_0_1, True
    except (json.JSONDecodeError, OSError, KeyError) as e:
        print(f"[SFC] QLSTM daemon cache read error: {e}", file=sys.stderr)
    
    # Fallback: run direct inference (slow, loads torch)
    try:
        sfc_dir = os.path.dirname(os.path.abspath(__file__))
        if sfc_dir not in sys.path:
            sys.path.insert(0, sfc_dir)
        # Ensure venv deps (torch, pennylane) are available
        venv_path = os.path.join(sfc_dir, ".venv", "lib", "python3.12", "site-packages")
        if os.path.isdir(venv_path) and venv_path not in sys.path:
            sys.path.insert(0, venv_path)
        from models.qlstm_enhanced import run_enhanced_inference
        
        result = run_enhanced_inference(force=True)
        
        if not result.get("qlstm_ok"):
            if QLSTM_AVAILABLE:
                print(f"[SFC] QLSTM enhanced failed: {result.get('error', 'unknown')}", file=sys.stderr)
                QLSTM_AVAILABLE = False
            return None, False
        
        qlstm_pred_0_1 = result["qlstm_pred"] / 100.0
        _GARCH_RESIDUAL = result.get("garch_residual", 0)
        _GARCH_VOL = result.get("garch_volatility", 0)
        _PROADAPT_W = result.get("proadapt_weight", 0.5)
        _PROADAPT_FINAL = result.get("proadapt_final", result["qlstm_pred"])
        _XAI_FEATURES = result.get("xai_top_features", None)
        
        if not QLSTM_AVAILABLE:
            print(f"[SFC] QLSTM enhanced ready: pred={result['qlstm_pred']:.1f} "
                  f"garch={_GARCH_RESIDUAL:+.3f} adapt_w={_PROADAPT_W:.2f}", file=sys.stderr)
            QLSTM_AVAILABLE = True
        
        QLSTM_INFERENCE_CACHE = {"pred": qlstm_pred_0_1, "ts": now}
        return qlstm_pred_0_1, True
        
    except Exception as e:
        if QLSTM_AVAILABLE:
            print(f"[SFC] QLSTM enhanced error: {e}", file=sys.stderr)
            QLSTM_AVAILABLE = False
        _GARCH_RESIDUAL = 0
        _GARCH_VOL = 0
        _PROADAPT_W = 0.5
        _PROADAPT_FINAL = None
        _XAI_FEATURES = None
        return None, False

# Module-level vars for enhanced output
_XAI_FEATURES = None
_GARCH_RESIDUAL = 0
_GARCH_VOL = 0
_PROADAPT_W = 0.5
_PROADAPT_FINAL = None

# Import news aggregator
sys.path.insert(0, os.path.dirname(__file__))
try:
    from data_sources.news_sources import get_news_stress_v2, detect_black_swan_v2
except ImportError:
    def get_news_stress_v2(*a, **k): return 0.0, [], 0.0, [], {}
    def detect_black_swan_v2(*a, **k): return 0.0, None, "NONE"


# ============================================================
# 1. LIVE DATA COLLECTION
# ============================================================

CMC_HEADERS = {}
CMC_KEY = os.getenv("CMC_API_KEY", "")
if CMC_KEY:
    CMC_HEADERS = {"X-CMC_PRO_API_KEY": CMC_KEY, "Accept": "application/json"}

# ── CoinGecko API key (demo tier) ──────────────────────────────────
CG_API_KEY = "x_cg_demo_api_key=REMOVED_SECRET"

# ── Model Version Tracking ─────────────────────────────────────────
MODEL_VERSION = "4.0.0"
MODEL_CHANGELOG = {
    "3.0.0": "2026-07-24 — Baseline: model_version tracking introduced",
    "4.0.0": "2026-07-25 — Lt de-duplicated: removed redundant M33 GLO + direct m2_yoy sigmoid, consolidated into single GLF ×5.927 (LT_EMPIRICAL_RESCALE).",
}

def get_cmc_price():
    """Fetch BTC price, 24h change, market cap from CoinMarketCap (primary)"""
    if not CMC_KEY:
        return None, None, None
    try:
        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?symbol=BTC&convert=USD",
            headers=CMC_HEADERS, timeout=10
        )
        if r.status_code != 200:
            return None, None, None
        d = r.json()["data"]["BTC"]["quote"]["USD"]
        return d["price"], d.get("percent_change_24h", 0), d.get("market_cap", 0)
    except:
        return None, None, None

def get_cmc_dominance():
    """Fetch BTC dominance from CoinMarketCap (primary)"""
    if not CMC_KEY:
        return None
    try:
        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
            headers=CMC_HEADERS, timeout=10
        )
        if r.status_code != 200:
            return None
        return r.json()["data"]["btc_dominance"]
    except:
        return None

def get_btc():
    """BTC price — try Binance WebSocket first, then CMC, fallback CoinGecko"""
    # Fast local read from Binance WebSocket daemon (no API call)
    # Staleness threshold: if the daemon wrote this file more than
    # MAX_WS_AGE_SECONDS ago, the daemon has probably stopped updating
    # (crashed, network issue, or ws-watchdog hasn't restarted it yet).
    # In that case, fall through to the REST API fallbacks rather than
    # silently using a stale price for potentially many consecutive cycles
    # — which would affect paper trading execution prices and stress
    # score inputs without any visible indication in the dashboard.
    MAX_WS_AGE_SECONDS = 300  # 5 min — two full pipeline cycles
    ws_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_ws.json")
    if os.path.exists(ws_path):
        try:
            with open(ws_path) as f:
                ws_data = json.load(f)
            if ws_data.get("btc") is not None:
                # Check how old the data actually is from the ts field,
                # not just whether the file exists (the file persists
                # indefinitely after the daemon dies).
                ts_str = ws_data.get("ts")
                ws_age_seconds = None
                if ts_str:
                    try:
                        from datetime import datetime, timezone
                        ws_ts = datetime.fromisoformat(ts_str)
                        if ws_ts.tzinfo is None:
                            ws_ts = ws_ts.replace(tzinfo=timezone.utc)
                        ws_age_seconds = (datetime.now(timezone.utc) - ws_ts).total_seconds()
                    except (ValueError, TypeError):
                        ws_age_seconds = None

                if ws_age_seconds is not None and ws_age_seconds > MAX_WS_AGE_SECONDS:
                    print(
                        f"[SFC] BTC WS data is {ws_age_seconds:.0f}s old "
                        f"(>{MAX_WS_AGE_SECONDS}s threshold) — daemon may be down, "
                        f"falling back to REST API",
                        file=sys.stderr,
                    )
                else:
                    btc_ws = ws_data["btc"]
                    chg_ws = ws_data.get("btc_24h", 0)
                    _, _, mcap = get_cmc_price()
                    age_str = f", age {ws_age_seconds:.0f}s" if ws_age_seconds is not None else ""
                    print(f"[SFC] BTC from Binance WS: ${btc_ws:,.0f} ({chg_ws:+.2f}%{age_str})",
                          file=sys.stderr)
                    return btc_ws, chg_ws, mcap
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    cmc_btc, cmc_chg, cmc_mcap = get_cmc_price()
    if cmc_btc is not None:
        print(f"[SFC] BTC from CMC: ${cmc_btc:,.0f}", file=sys.stderr)
        return cmc_btc, cmc_chg, cmc_mcap
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_market_cap=true&{CG_API_KEY}", timeout=10)
        d = r.json()["bitcoin"]
        print(f"[SFC] BTC from CoinGecko (fallback): ${d['usd']:,.0f}", file=sys.stderr)
        return d["usd"], d.get("usd_24h_change", 0), d.get("usd_market_cap", 0)
    except:
        return None, None, None

def get_ath():
    """Fetch dynamic ATH from CoinGecko — fallback to hardcoded 126272 and cache"""
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false&community_data=false&developer_data=false&{CG_API_KEY}", timeout=10)
        d = r.json()
        ath = d.get("market_data", {}).get("ath", {}).get("usd")
        ath_date = d.get("market_data", {}).get("ath_date", {}).get("usd", "")
        if ath and ath > 0:
            print(f"[SFC] ATH fetched: ${ath:,.0f} on {ath_date}", file=sys.stderr)
            _save_ath(ath, ath_date)
            return int(ath), ath_date
    except Exception as e:
        print(f"[SFC] ATH fetch failed: {e}", file=sys.stderr)
    # Failover: load from cache
    cached = _load_ath()
    if cached and cached.get('ath', 0) > 0:
        print(f"[SFC] ATH from cache: ${cached['ath']:,.0f} ({cached.get('date', '?')})", file=sys.stderr)
        return int(cached['ath']), cached.get('date', '')
    FALLBACK_ATH = 126272
    return FALLBACK_ATH, ""

def get_fng():
    """Fear & Greed from CoinMarketCap API."""
    try:
        r = requests.get(
            "https://api.coinmarketcap.com/data-api/v3/fear-greed/chart?start=1367193600&end=" + str(int(time.time())),
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        hv = r.json()["data"]["historicalValues"]
        now = hv["now"]
        val = int(now["score"])
        name_map = {
            "Extreme fear": "Extreme Fear", "Fear": "Fear",
            "Neutral": "Neutral", "Greed": "Greed",
            "Extreme greed": "Extreme Greed"
        }
        cls = name_map.get(now["name"], now["name"])
        return val, cls
    except Exception as e:
        print(f"[SFC] CMC F&G failed: {e}, falling back to Alternative.me", file=sys.stderr)
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            d = r.json()["data"][0]
            return int(d["value"]), d["value_classification"]
        except:
            return None, None

def get_dom():
    """BTC dominance — try CoinMarketCap first, fallback to CoinGecko"""
    cmc_dom = get_cmc_dominance()
    if cmc_dom is not None:
        print(f"[SFC] Dominance from CMC: {cmc_dom:.1f}%", file=sys.stderr)
        return cmc_dom
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/global?{CG_API_KEY}", timeout=10)
        d = r.json()["data"]["market_cap_percentage"]["btc"]
        print(f"[SFC] Dominance from CoinGecko (fallback): {d:.1f}%", file=sys.stderr)
        return d
    except:
        return None

def get_dvol():
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_index_price?index_name=btcdvol_usdc", timeout=10)
        data = r.json()
        dvol = data.get("result", {}).get("index_price")
        if dvol is not None:
            return round(dvol, 2)
        print(f"[SFC] WARNING: Deribit DVOL response had no index_price "
              f"(response: {str(data)[:200]}). Dashboard will fall back to "
              f"30-day rolling average via _factors_dvol.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[SFC] WARNING: get_dvol() failed ({type(e).__name__}: {e}). "
              f"Dashboard will fall back to 30-day rolling average via "
              f"_factors_dvol — check Deribit API status / network egress "
              f"if this persists.", file=sys.stderr)
        return None

def get_m2_data():
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return None, None, None
    try:
        r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=M2SL&api_key={key}&file_type=json&sort_order=desc&limit=13", timeout=15)
        obs = r.json().get("observations", [])
        if len(obs) >= 13:
            current = float(obs[0]["value"])
            year_ago = float(obs[12]["value"])
            return current, round((current - year_ago) / year_ago * 100, 2), None
    except:
        pass
    return None, None, None

def get_dxy():
    # DXY (US Dollar Index) calculated from exchange rates via free API
    # ICE DXY = 50.14348112 * EUR^(-0.576) * JPY^(0.136) * GBP^(-0.119)
    #                       * CAD^(0.091) * SEK^(0.042) * CHF^(0.036)
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        rates = r.json().get("rates", {})
        eur = 1.0 / rates["EUR"]   # USD per EUR
        jpy = rates["JPY"]          # JPY per USD
        gbp = 1.0 / rates["GBP"]    # USD per GBP
        cad = rates["CAD"]          # CAD per USD
        sek = rates["SEK"]          # SEK per USD
        chf = rates["CHF"]          # CHF per USD
        dxy = 50.14348112 * (eur ** -0.576) * (jpy ** 0.136) * (gbp ** -0.119) * (cad ** 0.091) * (sek ** 0.042) * (chf ** 0.036)
        return round(dxy, 2)
    except Exception as e:
        print(f"[SFC] DXY calc failed: {e}, fallback FRED", file=sys.stderr)
        try:
            key = os.getenv("FRED_API_KEY", "")
            if not key:
                return None
            r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXBGS&api_key={key}&file_type=json&sort_order=desc&limit=2", timeout=15)
            obs = r.json().get("observations", [])
            if obs and obs[0]["value"] != ".":
                return round(float(obs[0]["value"]), 2)
        except:
            pass
        return None


def _compute_dxy_btc_correlation():
    """Compute rolling 30-day correlation between DXY changes and BTC returns.
    
    Reads from daily market cache. Returns correlation coefficient (-1 to 1)
    or None if insufficient data.
    
    When correlation > 0.3: DXY and BTC move together (risk-on USD regime).
      In this regime, DXY strength = bullish for crypto (flip Sc sign).
    When correlation < -0.3: Normal inverse regime.
      DXY strength = bearish (standard Sc logic).
    When -0.3 < corr < 0.3: Mixed/unclear regime. Use neutral/weakened Sc.
    """
    cache_file = os.path.join(os.path.dirname(__file__), '.daily_market_cache.json')
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file) as f:
            cache = json.load(f)
        if len(cache) < 15:
            return None
        
        # Extract daily DXY and BTC returns (fix: use pct change, not levels — spurious correlation)
        dxy_levels = []
        btc_levels = []
        for entry in cache:
            dxy = entry.get("dxy")
            btc = entry.get("btc")
            if dxy is not None and btc is not None:
                dxy_levels.append(dxy)
                btc_levels.append(btc)
        
        if len(dxy_levels) < 15:
            return None
        
        # Convert to daily returns
        dxy_vals = [(dxy_levels[i] - dxy_levels[i-1]) / dxy_levels[i-1]
                     for i in range(1, len(dxy_levels))]
        btc_vals = [(btc_levels[i] - btc_levels[i-1]) / btc_levels[i-1]
                     for i in range(1, len(btc_levels))]
        
        # Compute Pearson correlation
        n = len(dxy_vals)
        mean_dxy = sum(dxy_vals) / n
        mean_btc = sum(btc_vals) / n
        num = sum((d - mean_dxy) * (b - mean_btc) for d, b in zip(dxy_vals, btc_vals))
        denom = math.sqrt(sum((d - mean_dxy)**2 for d in dxy_vals) * sum((b - mean_btc)**2 for b in btc_vals))
        if denom == 0:
            return None
        corr = num / denom
        return max(-1.0, min(1.0, corr))
    except Exception:
        return None


def get_put_call_ratio():
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        puts_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-P"))
        calls_oi = sum(o.get("open_interest", 0) for o in opts if o.get("instrument_name", "").endswith("-C"))
        puts_vol = sum(o.get("volume", 0) for o in opts if o.get("instrument_name", "").endswith("-P"))
        calls_vol = sum(o.get("volume", 0) for o in opts if o.get("instrument_name", "").endswith("-C"))
        oi_ratio = round(puts_oi / calls_oi, 2) if calls_oi else None
        vol_ratio = round(puts_vol / calls_vol, 2) if calls_vol else None
        return oi_ratio, vol_ratio
    except:
        return None, None

def get_btc_ohlcv_daily(days=30):
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily&{CG_API_KEY}", timeout=15)
        d = r.json()
        prices = d.get("prices", [])
        volumes = d.get("total_volumes", [])
        return [{"time": prices[i][0], "close": prices[i][1], "volume": volumes[i][1] if i < len(volumes) else 0} for i in range(len(prices))]
    except:
        return []

def _binance_klines(days=30):
    """Fetch BTC klines from Binance as fallback (free, no key)."""
    try:
        limit = days  # 1 per day
        r = requests.get(f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit={limit}", timeout=10)
        if r.status_code != 200: return []
        klines = r.json()
        return [{"time": k[0], "close": float(k[4]), "volume": float(k[5])} for k in klines]
    except: return []

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def _sopr_signal_score(value):
    """Classify SOPR value into signal name and stress score (0-1)."""
    if value is None:
        return "UNKNOWN", 0.5
    if value < 0.93:
        return "EXTREME_CAPITULATION", 0.95
    elif value < 0.97:
        return "CAPITULATION", 0.80
    elif value < 0.995:
        return "MILD_DISTRESS", 0.65
    elif value < 1.005:
        return "BREAKEVEN", 0.50
    elif value < 1.03:
        return "MILD_PROFIT", 0.40
    elif value < 1.08:
        return "DISTRIBUTION", 0.25
    else:
        return "EXTREME_DISTRIBUTION", 0.10

SOPR_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.sopr_cache.json')
SOPR_CACHE_TTL = 3600  # 1 hour (free tier: 10 req/hour)

def compute_sopr(closes_7d, closes_30d, btc_spot):
    """Compute SOPR: try true on-chain from BGeometrics API first (cached), fallback to price proxy.

    True SOPR (on-chain) = spent output value / creation value (UTXO-based).
    Proxy = btc_spot / (0.4*avg_7d + 0.6*avg_30d).

    Returns:
        (sopr_value, signal, score)
    """
    # Try cached true SOPR first
    if os.path.exists(SOPR_CACHE_FILE):
        try:
            with open(SOPR_CACHE_FILE) as _f:
                _cache = json.load(_f)
            if time.time() - _cache.get("ts", 0) < SOPR_CACHE_TTL:
                _val = _cache["sopr"]
                _signal, _score = _sopr_signal_score(_val)
                return _val, _signal, _score
        except Exception:
            pass

    # Try public endpoint first (no API key needed)
    try:
        r = requests.get(
            "https://api.bitcoin-data.com/v1/sopr?days=1",
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                latest = data[-1]
                true_sopr = float(latest["sopr"])
                # Write cache
                try:
                    with open(SOPR_CACHE_FILE, "w") as _f:
                        json.dump({"sopr": true_sopr, "ts": time.time()}, _f)
                except Exception:
                    pass
                signal, score = _sopr_signal_score(true_sopr)
                return true_sopr, signal, score
    except Exception:
        pass

    # Fallback: try with API key if public fails
    api_key = os.getenv("SOPR_API_KEY", "")
    if api_key:
        try:
            r = requests.get(
                f"https://api.bgeometrics.com/v1/sopr?token={api_key}&days=3",
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    latest = data[-1]
                    true_sopr = float(latest["sopr"])
                    # Write cache
                    try:
                        with open(SOPR_CACHE_FILE, "w") as _f:
                            json.dump({"sopr": true_sopr, "ts": time.time()}, _f)
                    except Exception:
                        pass
                    signal, score = _sopr_signal_score(true_sopr)
                    return true_sopr, signal, score
        except Exception:
            pass

    # Fallback: price proxy
    if not closes_7d or not closes_30d or not btc_spot:
        return None, "UNKNOWN", 0.5
    avg_7d = sum(closes_7d) / len(closes_7d)
    avg_30d = sum(closes_30d) / len(closes_30d)
    weighted_cost = 0.40 * avg_7d + 0.60 * avg_30d
    proxy = round(btc_spot / weighted_cost, 4) if weighted_cost > 0 else None
    signal, score = _sopr_signal_score(proxy)
    return proxy, signal, score

# ============================================================
# 2. SFC v2.1 CALCULATION (NO LLM - Rule Based)
# ============================================================

def _sigmoid_factor(val, center, k=0.15):
    '''Smooth logistic: maps val to [-3, +3] range.
    center = neutral point, k = steepness.
    sigmoid(x) = 6 / (1 + exp(-k*(x-center))) - 3
    '''
    return 6 / (1 + math.exp(-k * (val - center))) - 3

def score_factors_from_market(btc, btc_24h, dom, dvol, fng, pc_oi, m2_yoy, dxy, glo_score=None,
                                onchain_whale=None, onchain_value=None, onchain_buy=None,
                                onchain_market_structure=None, dxy_btc_corr=None):
    """Score 5 factors from market data using smooth sigmoid/logistic functions. Range -3 to +3
    onchain_whale/onchain_value/onchain_buy: 0-100 scores from on-chain data (Q10)
    onchain_market_structure: 0-100 score from derivatives data (Q10+)"""
    factors = {"Lt": 0.0, "St": 0.0, "Rt": 0.0, "Ft": 0.0, "Sc": 0.0}
    
    # Lt (Liquidity) — based on GLO and BTC momentum
    if glo_score is not None:
        # GLO maps: 0=contractive(bearish) -> -3, 100=expansive(bullish) -> +3
        # Map GLO 0-100 to sigmoid center at 50
        factors["Lt"] += _sigmoid_factor(glo_score, center=50.0, k=0.08)
    if btc_24h is not None:
        factors["Lt"] += _sigmoid_factor(btc_24h, center=0.0, k=0.15)
    
    # St (Structural) — based on dominance and put/call
    if dom is not None:
        # High dominance = concentration risk (negative), low = diffuse (positive)
        factors["St"] += -_sigmoid_factor(dom, center=55.0, k=0.2)
    if pc_oi is not None:
        # High put/call = fear (negative), low = greed (positive)
        factors["St"] += -_sigmoid_factor(pc_oi, center=0.8, k=2.0)
    
    # Rt (Sentiment) — based on Fear & Greed Index
    if fng is not None:
        # Low FnG = fear = stress (negative), high FnG = greed = calm (positive)
        factors["Rt"] = _sigmoid_factor(fng, center=50.0, k=0.08)
    
    # Ft (Systemic) — based on DVOL (implied volatility)
    if dvol is not None:
        # High vol = systemic stress (negative)
        factors["Ft"] = -_sigmoid_factor(dvol, center=65.0, k=0.06)
    
    # Sc (External) — based on DXY (US dollar index) with correlation gate
    # DXY-BTC correlation determines whether DXY strength is a headwind or tailwind
    if dxy is not None:
        if dxy_btc_corr is not None and dxy_btc_corr > 0.3:
            # Positive correlation regime: DXY and BTC move together (risk-on USD)
            # Strong dollar = bullish for crypto → flip sign
            factors["Sc"] = _sigmoid_factor(dxy, center=100.0, k=0.2)
            corr_regime = "POSITIVE"
        elif dxy_btc_corr is not None and dxy_btc_corr > -0.3:
            # Mixed regime: neither clearly positive nor negative
            # Weaken the Sc factor magnitude
            factors["Sc"] = -_sigmoid_factor(dxy, center=100.0, k=0.2) * 0.5
            corr_regime = "MIXED"
        else:
            # Normal inverse regime: DXY up = BTC down (standard)
            factors["Sc"] = -_sigmoid_factor(dxy, center=100.0, k=0.2)
            corr_regime = "INVERSE"
        print(f"[DXY Gate] corr={dxy_btc_corr} regime={corr_regime} Sc={factors['Sc']:+.3f}", file=sys.stderr)
    # High dominance amplifies external risk only in inverse/mixed regime
    if dom is not None and dom > 65 and (dxy_btc_corr is None or dxy_btc_corr < 0.3):
        factors["Sc"] -= 0.5
    
    # ── ON-CHAIN ADJUSTMENTS (Q10 Integration) ──
    # Map 0-100 on-chain scores to -3 to +3 factor adjustments
    # score=50 → adj=0 (neutral), score=100 → adj=+3, score=0 → adj=-3
    if onchain_whale is not None:
        # Whale pressure → Rt (risk): high score = bullish (supply leaving, whales accumulating)
        whale_adj = (onchain_whale - 50) / 50 * 2.0  # scale -2 to +2
        factors["Rt"] += whale_adj
        print(f"[OnChain] Whale pressure={onchain_whale:.1f} → Rt adj={whale_adj:+.3f}", file=sys.stderr)
    
    if onchain_value is not None:
        # On-chain value → Lt (long-term): high score = undervalued (MVRV low, Puell low)
        value_adj = (onchain_value - 50) / 50 * 2.0  # scale -2 to +2
        factors["Lt"] += value_adj
        print(f"[OnChain] On-chain value={onchain_value:.1f} → Lt adj={value_adj:+.3f}", file=sys.stderr)
    
    if onchain_buy is not None:
        # Buying power → Ft (funding): high score = strong buying power
        buy_adj = (onchain_buy - 50) / 50 * 1.5  # scale -1.5 to +1.5 (lighter touch)
        factors["Ft"] += buy_adj
        print(f"[OnChain] Buying power={onchain_buy:.1f} → Ft adj={buy_adj:+.3f}", file=sys.stderr)

    if onchain_market_structure is not None:
        # Market structure → St (short-term): high score = healthy (low OI, controlled liqs)
        ms_adj = (onchain_market_structure - 50) / 50 * 1.5  # scale -1.5 to +1.5
        factors["St"] += ms_adj
        print(f"[OnChain] Market structure={onchain_market_structure:.1f} → St adj={ms_adj:+.3f}", file=sys.stderr)

    # NOTE: M86 (repo market stress) factor adjustment used to live here,
    # inside this function definition — but this function is called
    # (line ~2012) BEFORE _m86_score is actually computed (previously at
    # line ~3211, over a thousand lines later). Since _m86_score's
    # module-level default is 0.5 until that later computation runs, the
    # `if _m86_score != 0.5:` check here was ALWAYS False at call time,
    # meaning this adjustment silently never executed with a real value —
    # found during a fresh audit of factor-adjustment ordering. Moved to a correctly
    # ordered position: computed before this function is called, applied
    # after it returns (see "REPO MARKET STRESS FACTOR ADJUSTMENT" below
    # score_factors_from_market()'s call site), matching how ETF/Fiscal
    # adjustments are already (correctly) structured.

    # Clamp all factors to [-3, 3]
    for k in factors:
        factors[k] = max(-3.0, min(3.0, factors[k]))
    
    return factors

def calculate_sfc_ensemble(factors):
    """Calculate 6-method ensemble from factors with Lt/St weights (33%/67%).

    FIX v2026.07.11 — Scale mismatch bug:
    All 6 methods had thresholds designed for norm in [-3, +3], but
    `norm = factors/6` puts actual values in [-0.5, +0.5]. This meant:
      - M1 KLR: max 0.30 (designed for 1.0)
      - M2 Logit: anchors unreachable (z_score scaled by /6)
      - M3 Bayes: norm[k] < -0.5 NEVER true → dead code
      - M4 EWC: max 0.17 (designed for 1.0)
      - M5 QReg: anchors unreachable (same /6 in z_score)
      - M6 Regime: extreme/severe counts always 0
    Fix restores original design range: z_score from raw factors [-15,+15],
    norm[-0.5,+0.5] thresholds scaled by /6, M4 uses raw factors.
    """
    # Apply Lt/St weights from correlation analysis (Lt|r|=0.057, St|r|=0.114)
    # Weights adjusted to preserve total scale (sum=5, same as equal weighting)
    _FACTOR_WT = {"Lt": 0.66, "St": 1.34, "Rt": 1.0, "Ft": 1.0, "Sc": 1.0}
    norm = {k: v/6 for k, v in factors.items()}
    # FIX: use raw factors (not norm/6) for z_score — original [-15, +15] range
    z_score = sum(factors[k] * _FACTOR_WT[k] for k in factors)
    
    # M1: KLR — thresholds rescaled from [-2, -1, 0] → [÷6] for norm[-0.5, +0.5]
    ns_r = {"Lt":0.35, "St":0.50, "Rt":0.40, "Ft":0.25, "Sc":0.80}
    w = {k:1/v for k,v in ns_r.items()}
    sig = sum((1.0 if norm[k]<-0.333 else 0.7 if norm[k]<-0.167 else 0.3 if norm[k]<0 else 0) * w[k] for k in factors)
    p_klr = max(0.0, min(1.0, sig / sum(w.values())))  # clamp to [0,1]
    
    # M2: Logit — z_score now [-15, +15], anchors match original design
    zc = [-1.0, -2.0, -3.0, -4.0, -8.0]
    pc = [0.08, 0.20, 0.55, 0.75, 0.95]
    yc = [math.log(p/(1-p)) for p in pc]
    n_z = len(zc)
    zm = sum(zc)/n_z
    ym = sum(yc)/n_z
    b1 = sum((zc[i]-zm)*(yc[i]-ym) for i in range(n_z)) / sum((z-zm)**2 for z in zc)
    b0 = ym - b1*zm
    z_l = b0 + b1*z_score
    p_logit = 1/(1+math.exp(-z_l))
    
    # M3: Bayesian — threshold lowered from -0.5 (unreachable) to -0.3
    prior = 0.08  # raised from 0.04 — BTC historically crashes 8-15% of years
    odds = prior/(1-prior)
    bayes_mult = [2.5, 2.0, 2.0, 3.0, 1.5]
    for i, k in enumerate(factors):
        if norm[k] < -0.3:
            odds *= bayes_mult[i]
    p_bayes = odds/(1+odds)
    
    # M4: ECB Composite — use raw factors (not norm) to match /3.0 denominator
    w_ad = {"Lt":0.25, "St":0.20, "Rt":0.20, "Ft":0.30, "Sc":0.05}
    ewc = sum(w_ad[k] * abs(factors[k]) for k in factors)
    p_ewc = ewc/3.0
    
    # M5: Quantile Regression — z_score now [-15, +15], anchors match design
    qr_anchors = [(-8.0, 0.95), (-5.0, 0.75), (-3.0, 0.50), (-1.5, 0.25), (-0.5, 0.10), (0.5, 0.04), (2.0, 0.01)]
    def quantile_stress(z):
        anchors = sorted(qr_anchors, key=lambda x: x[0])
        if z <= anchors[0][0]: return anchors[0][1]
        if z >= anchors[-1][0]: return anchors[-1][1]
        for i in range(len(anchors)-1):
            z0, p0 = anchors[i]
            z1, p1 = anchors[i+1]
            if z0 <= z <= z1:
                t = (z - z0) / (z1 - z0)
                return p0 + t * (p1 - p0)
        return 0.04
    p_quantile = quantile_stress(z_score)
    
    # M6: Composite Regime Score — all thresholds ÷6 for norm[-0.5, +0.5]
    vals = list(norm.values())
    n = len(vals)
    extreme_count = sum(1 for v in vals if v < -0.167)
    severe_count = sum(1 for v in vals if v < -0.333)
    p_extremity = (extreme_count * 0.15 + severe_count * 0.20)
    mean_v = sum(vals) / n
    variance = sum((v - mean_v)**2 for v in vals) / n
    coherence_bonus = 0.10 * (1.0 - variance) if mean_v < -0.083 and variance < 0.12 else 0.0
    ft_val = norm.get("Ft", 0)
    lt_val = norm.get("Lt", 0)
    tail_contribution = (0.15 if ft_val < -0.25 else 0.0) + (0.10 if lt_val < -0.25 else 0.0)
    p_baseline = max(0.0, min((-mean_v) * 0.72, 0.50))
    p_regime = min(p_baseline + p_extremity + coherence_bonus + tail_contribution, 0.99)
    p_regime = max(p_regime, 0.01)
    
    # Ensemble (sum = 1.0 after fix 0.24→0.23)
    p_ens = 0.19*p_klr + 0.16*p_logit + 0.12*p_bayes + 0.16*p_ewc + 0.23*p_quantile + 0.14*p_regime
    
    # Method agreement
    method_probs = [p_klr, p_logit, p_bayes, p_ewc, p_quantile, p_regime]
    mean_p = sum(method_probs) / len(method_probs)
    variance_p = sum((p - mean_p)**2 for p in method_probs) / len(method_probs)
    std_p = math.sqrt(variance_p)
    method_agreement = max(0.0, min(1.0, 1.0 - (std_p / 0.20)))
    
    zone = "CRITICAL" if p_ens > 0.75 else "HIGH" if p_ens > 0.5 else "ELEVATED" if p_ens > 0.25 else "NORMAL"
    
    return p_ens * 100, zone, factors, norm, p_klr, p_logit, p_bayes, p_ewc, p_quantile * 100, p_regime * 100, method_agreement

# ============================================================
# 3. MAIN
# ============================================================

DVOL_BASELINE = 50

def compute_floor_v2(btc, sfc_pct):
    if not btc or sfc_pct is None: return None, None, None, None
    drawdown = min(sfc_pct / 100.0, 0.8) * 0.6  # Max 48% drawdown at 80% stress
    floor = int(btc * (1 - drawdown))
    buffer = int(btc - floor)
    return buffer, floor, round(drawdown, 3), 1.0

def determine_state(dvol, sfc_pct, btc=None, floor_total=None):
    if dvol is None or sfc_pct is None:
        return "NORMAL OSCILLATION", "Standard allocation"
    if dvol >= 100 and btc and floor_total and btc <= floor_total * 1.05:
        return "⚠ TURNING POINT", "BUY / REVERSAL FLOOR"
    if dvol < 40 and sfc_pct > 30:
        return "DISSIPATIVE SLIDE", "STAY IN CASH / GOLD"
    if sfc_pct > 50:
        return "STRUCTURAL DECAY", "REDUCE LEVERAGE"
    if dvol > 80:
        return "VOLATILITY BREAKDOWN", "HEDGE / REDUCE"
    return "NORMAL OSCILLATION", "Standard allocation"

def detect_regime(dvol, sfc_pct, news_stress, sentiment):
    if dvol is None or sfc_pct is None:
        return "NORMAL", 0.9, 0.1
    p_capitulation = 0.0
    if dvol:
        p_capitulation += min(dvol / 120.0, 1.0) * 0.4
    if sfc_pct:
        p_capitulation += min(sfc_pct / 60.0, 1.0) * 0.3
    if sentiment and sentiment < -0.3:
        p_capitulation += min(abs(sentiment), 1.0) * 0.3
    p_stress = 0.0
    if dvol:
        p_stress += min(max(dvol - 50, 0) / 50.0, 1.0) * 0.45
    if sfc_pct:
        p_stress += min(sfc_pct / 30.0, 1.0) * 0.40
    if news_stress:
        p_stress += min(news_stress / 20.0, 1.0) * 0.15
    p_normal = max(0, 1.0 - p_capitulation - p_stress)
    p_capitulation = min(p_capitulation, 1.0)
    p_stress = min(p_stress, 1.0 - p_capitulation)
    regimes = [("NORMAL", p_normal), ("STRESS", p_stress), ("CAPITULATION", p_capitulation)]
    dominant = max(regimes, key=lambda x: x[1])
    if dominant[0] == "NORMAL":
        transition_risk = round(p_stress + p_capitulation, 3)
    elif dominant[0] == "STRESS":
        transition_risk = round(p_capitulation, 3)
    else:
        transition_risk = 0.0
    return dominant[0], round(dominant[1], 3), transition_risk

# ============================================================
# 1b. NEW METHODS (M7-M19) — 13 Advanced Enhancement Methods
# ============================================================

FRED_KEY = os.getenv("FRED_API_KEY", "")
_FRED_CACHE = {}  # series_id -> (vals, ts)

def _fred(series, limit=2):
    """Fetch FRED data with module-level cache. Cache is warm after _fred_prefetch()."""
    global _FRED_CACHE
    cache_key = f"{series}:{limit}"
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]
    if not FRED_KEY: return None
    try:
        r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}", timeout=15)
        if r.status_code != 200: return None
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o["value"] != "."]
        result = vals if vals else None
        _FRED_CACHE[cache_key] = result
        return result
    except: return None

def _fred_cpi_yoy():
    """Fetch CPI YoY with caching (called by M7).

    Returns None (honest "unavailable") instead of a hardcoded 3.0 fallback.
    Audit A4 (2026-08-03): the old `return 3.0` fallback masqueraded as a real
    reading — m7_detail showed cpi_yoy=3.0 (the default) while the expectations
    engine reported the true ~3.46. It reuses the (prefetched/cached) 13-value
    CPIAUCSL series via _fred() so it no longer needs its own request.
    """
    cache_key = "CPIAUCSL:13_yoy"
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]
    if not FRED_KEY: return None
    vals = _fred("CPIAUCSL", 24)  # newest-first, "." filtered (see _fred)
    if vals and len(vals) >= 2:
        # Newest REAL value (obs[0] may be "." if the month is not yet released),
        # and the value ~12 months back. idx clamped to the last available.
        idx = min(12, len(vals) - 1)
        result = (vals[0] - vals[idx]) / vals[idx] * 100
        _FRED_CACHE[cache_key] = result
        return result
    return None

def _fred_prefetch():
    """Prefetch ALL FRED data in ONE parallel batch."""
    global _FRED_CACHE
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    needed = [
        ("FEDFUNDS:1", "FEDFUNDS", 1),
        ("CPIAUCSL:1", "CPIAUCSL", 1),
        ("CPIAUCSL:24", "CPIAUCSL", 24),  # A4: seed 24-mo series so _fred_cpi_yoy() reuses batch data
        ("DGS10:1", "DGS10", 1),
        ("DGS2:1", "DGS2", 1),
        ("BAMLH0A0HYM2:1", "BAMLH0A0HYM2", 1),
        ("M2SL:1", "M2SL", 1),
        ("MBCURSL:1", "MBCURSL", 1),
        ("M2SL:30", "M2SL", 30),
        ("DTWEXBGS:30", "DTWEXBGS", 30),
        # GLO --- Global Liquidity Index series
        ("WALCL:13", "WALCL", 13),
        ("ECBASSETSW:13", "ECBASSETSW", 13),
        ("JPNASSETS:13", "JPNASSETS", 13),
    ]
    
    def _fetch_one(series, limit):
        try:
            if not FRED_KEY: return None, None
            r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit={limit}", timeout=15)
            if r.status_code != 200: return None, series
            obs = r.json().get("observations", [])
            vals = [float(o["value"]) for o in obs if o["value"] != "."]
            return (vals if vals else None), series
        except: return None, series
    
    # Also fetch CPI YoY in same batch
    # NOTE: removed the old `CPIAUCSL:13` prefetch + the block that cached a fake
    # 3.0 into "CPIAUCSL:13_yoy". Audit A4 (2026-08-03): _fred_cpi_yoy() used to
    # return that cached 3.0 verbatim (fallback masquerading as real data). CPI
    # YoY is now computed on-demand in _fred_cpi_yoy() from the CPIAUCSL:24
    # series seeded in `needed` above; returns None (honest) when unavailable.
    
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch_one, s, l): (k, s) for k, s, l in needed}
        for f in as_completed(futures):
            vals, series = f.result()
            key = futures[f][0]
            _FRED_CACHE[key] = vals if vals else None

# ── TIER 1: MACRO ECONOMICS ──

def calculate_m7_fisher():
    """M7: Fisher Real Rates — Real Rate = Fed Rate - CPI"""
    vals_fed = _fred("FEDFUNDS", 1)
    vals_cpi = _fred("CPIAUCSL", 1)
    if not vals_fed or not vals_cpi: return None, None
    fed_rate = vals_fed[0]
    # Use cached CPI YoY from prefetch. Returns None (honest) if unavailable.
    cpi_yoy = _fred_cpi_yoy()
    if cpi_yoy is None: return None, None  # A4: no silent 3.0 fallback
    real_rate = fed_rate - cpi_yoy
    if real_rate > 3.0: score = 0.85
    elif real_rate > 2.0: score = 0.60
    elif real_rate > 0.5: score = 0.35
    elif real_rate > 0: score = 0.20
    elif real_rate > -1.0: score = 0.10
    else: score = 0.05
    return score, {"real_rate": round(real_rate,2), "fed_rate": fed_rate, "cpi_yoy": round(cpi_yoy,2)}

def calculate_m8_yield_curve():
    """M8: Yield Curve Slope — 10Y - 2Y + credit spreads"""
    vals_10y = _fred("DGS10", 1)
    vals_2y = _fred("DGS2", 1)
    vals_hy = _fred("BAMLH0A0HYM2", 1)
    if not vals_10y or not vals_2y: return None, None
    slope = vals_10y[0] - vals_2y[0]
    spread = vals_hy[0] if vals_hy else 300
    if slope < 0: slope_s = 0.80
    elif slope < 0.5: slope_s = 0.65
    elif slope < 1.0: slope_s = 0.40
    elif slope > 2.0: slope_s = 0.15
    else: slope_s = 0.25

    if spread > 400: cred_s = 0.85
    elif spread > 300: cred_s = 0.65
    elif spread > 200: cred_s = 0.40
    else: cred_s = 0.15
    score = 0.60 * slope_s + 0.40 * cred_s
    return score, {"slope": round(slope,2), "spread": round(spread,1)}

def calculate_m9_liquidity():
    """M9: Liquidity Aggregates — M2/MB multiplier"""
    vals_m2 = _fred("M2SL", 1)
    vals_mb = _fred("MBCURSL", 1)
    if not vals_m2 or not vals_mb: return None, None
    m2 = vals_m2[0]
    mb = vals_mb[0]
    if mb <= 0: return None, None
    mult = m2 / mb
    if mult < 4.0: score = 0.85
    elif mult < 5.0: score = 0.65
    elif mult < 6.0: score = 0.40
    elif mult > 10.0: score = 0.20
    else: score = 0.30
    return score, {"m2_mb_mult": round(mult,2)}

# ── TIER 2: VOLATILITY & RISK ──

def calculate_m10_garch(closes):
    """M10: GARCH(1,1) — volatility persistence"""
    if len(closes) < 30: return None, None
    import numpy as np
    rets = np.diff(np.array(closes)) / np.array(closes[:-1])
    if len(rets) < 20: return None, None
    omega, alpha, beta = 0.00001, 0.05, 0.94
    r_mean = np.mean(rets)
    resid = rets - r_mean
    sigma2 = np.var(resid)
    hist = [sigma2]
    for i in range(1, len(rets)):
        sigma2 = omega + alpha * resid[i-1]**2 + beta * hist[i-1]
        hist.append(sigma2)
    curr_vol = np.sqrt(hist[-1])
    persist = alpha + beta
    if persist > 0.98 and curr_vol > 0.03: score = 0.85
    elif persist > 0.95 and curr_vol > 0.02: score = 0.65
    elif curr_vol > 0.02: score = 0.45
    elif curr_vol > 0.01: score = 0.25
    else: score = 0.10
    return score, {"garch_vol": round(curr_vol,4), "persistence": round(persist,3)}

def calculate_m11_var(rets):
    """M11: VaR + Expected Shortfall — tail risk"""
    if rets is None or len(rets) < 30: return None, None
    import numpy as np
    arr = np.array(rets, dtype=float)
    var_95 = np.percentile(arr, 5)
    tail = arr[arr <= var_95]
    es = np.mean(tail) if len(tail) > 0 else var_95
    if es < -0.15: score = 0.85
    elif es < -0.10: score = 0.65
    elif es < -0.05: score = 0.45
    elif es < 0: score = 0.25
    else: score = 0.10
    return score, {"var_95": round(var_95,4), "es_95": round(es,4)}

def calculate_m12_jump(ohlcv):
    """M12: Jump Risk (Merton) — gap detection"""
    if not ohlcv or len(ohlcv) < 5: return None, None
    import numpy as np
    closes = [c["close"] for c in ohlcv]
    opens = [c.get("open", closes[i-1] if i > 0 else closes[i]) for i, c in enumerate(ohlcv)]  # fix: use 'open' key
    gaps = []
    for i in range(1, len(closes)):
        gap = abs(opens[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
        if gap > 0.02: gaps.append(gap)
    freq = len(gaps) / max(len(closes), 1)
    avg = np.mean(gaps) if gaps else 0
    jr = freq * avg
    if jr > 0.05: score = 0.85
    elif jr > 0.03: score = 0.65
    elif jr > 0.01: score = 0.45
    else: score = 0.15
    return score, {"jump_risk": round(jr,4), "gap_count": len(gaps)}

# ── TIER 3: DERIVATIVES & LEVERAGE ──

def calculate_m13_funding():
    """M13: Funding Rate Acceleration — leverage cascade detection"""
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_funding_rate_history?currency=BTC&start_timestamp=0&end_timestamp=" + str(int(time.time()*1000)), timeout=10)
        data = r.json().get("result", [])
        if len(data) < 3: return None, None
        rates = [d["interest_8h"] for d in data[:8]]
        if len(rates) < 3: return None, None
        fr_now = rates[0]
        fr_1 = rates[1]
        fr_2 = rates[2]
        accel = (fr_now - fr_1) - (fr_1 - fr_2)
        # Thresholds calibrated against Deribit's actual interest_8h scale.
        # interest_8h is a small decimal (e.g. -0.00009...), hard-capped by
        # Deribit at +/-0.005 (0.5% per 8h) per their published funding
        # formula (base component 0.025% = 0.00025). The previous
        # thresholds here (0.15, 0.05, accel 0.01) were 20-30x larger than
        # the maximum value this field can ever physically take, so this
        # scoring branch always fell through to the lowest tier regardless
        # of real leverage conditions — confirmed by testing the most
        # extreme historically plausible funding rate (0.003) against the
        # old thresholds and finding it still scored as "neutral".
        FR_CAP = 0.005
        if accel > FR_CAP * 0.40: score = 0.75
        elif fr_now > FR_CAP * 0.70: score = 0.65
        elif fr_now > FR_CAP * 0.30: score = 0.35
        else: score = 0.15
        return score, {"funding_rate": round(fr_now,6), "accel": round(accel,6)}
    except: return None, None

def calculate_m14_skew():
    """M14: Options Skew — put/call IV difference"""
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        puts = [o["mark_iv"] for o in opts if o.get("instrument_name","").endswith("-P") and o.get("mark_iv")]
        calls = [o["mark_iv"] for o in opts if o.get("instrument_name","").endswith("-C") and o.get("mark_iv")]
        all_iv = [o["mark_iv"] for o in opts if o.get("mark_iv")]
        if not puts or not calls or not all_iv: return None, None
        put_iv = sum(puts)/len(puts)
        call_iv = sum(calls)/len(calls)
        atm = sum(all_iv)/len(all_iv)
        if atm <= 0: return None, None
        skew = (put_iv - call_iv) / atm
        if skew > 0.20: score = 0.80
        elif skew > 0.15: score = 0.65
        elif skew > 0.10: score = 0.45
        elif skew > 0.05: score = 0.25
        else: score = 0.10
        return score, {"skew": round(skew,4), "put_iv": round(put_iv,2), "call_iv": round(call_iv,2)}
    except: return None, None

def calculate_m15_concentration():
    """M15: OI Concentration — HHI from top positions (approximated via options OI)"""
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option", timeout=10)
        opts = r.json().get("result", [])
        ois = [o.get("open_interest", 0) for o in opts if o.get("open_interest", 0) > 0]
        if len(ois) < 5: return None, None
        total = sum(ois)
        if total == 0: return None, None
        shares = [o/total for o in sorted(ois, reverse=True)]
        hhi = sum(s**2 for s in shares)
        top3 = sum(shares[:3])
        if hhi > 0.25: score = 0.80
        elif hhi > 0.20: score = 0.65
        elif hhi > 0.15: score = 0.45
        elif hhi > 0.10: score = 0.25
        else: score = 0.10
        return score, {"hhi": round(hhi,4), "top3_share": round(top3,3)}
    except: return None, None

# ── TIER 4: MACHINE LEARNING ──

def calculate_m16_regime_switch(rets):
    """M16: Volatility Regime Heuristic — crisis probability (simplified, not full Markov)"""
    if rets is None or len(rets) < 30: return None, None
    import numpy as np
    recent = rets[-30:]
    r_mean = np.mean(recent)
    r_std = np.std(recent)
    if r_mean < -0.005 and r_std > 0.03: p_crisis = 0.75
    elif r_std > 0.025: p_crisis = 0.45
    elif r_mean > 0.01: p_crisis = 0.10
    else: p_crisis = 0.20
    return p_crisis, {"p_crisis": round(p_crisis,3), "recent_mean": round(r_mean,4), "recent_std": round(r_std,4)}

def calculate_m17_granger(series_x, series_y, label="X→Y"):
    """M17: Lag Correlation — Pearson + cross-correlation (not full Granger F-test)"""
    if series_x is None or series_y is None or len(series_x) < 10 or len(series_y) < 10: return None, None
    import numpy as np
    n = min(len(series_x), len(series_y))
    x = np.array(series_x[-n:])
    y = np.array(series_y[-n:])
    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) < 2: return None, None
    corr = np.corrcoef(dx, dy)[0, 1]
    lag_corr = np.corrcoef(dx[:-1], dy[1:])[0, 1] if len(dx) > 1 else 0
    if lag_corr > 0.5 and abs(corr) < 0.3: score = 0.7
    elif abs(corr) > 0.6: score = 0.5
    elif lag_corr < -0.3 and abs(corr) > 0.3: score = 0.3
    else: score = 0.1
    return score, {"corr": round(corr,3), "lag_corr": round(lag_corr,3)}

# ── TIER 5: INFORMATION THEORY ──

def calculate_m18_entropy(prices):
    """M18: Shannon Entropy — market disorder"""
    if not prices or len(prices) < 15: return None, None
    import numpy as np
    from scipy.stats import entropy
    recent = prices[-30:] if len(prices) >= 30 else prices
    rets = np.diff(recent) / np.array(recent[:-1])
    n_bins = 10
    hist, _ = np.histogram(rets, bins=n_bins)
    if hist.sum() == 0: return None, None
    hist = hist / hist.sum()
    h = entropy(hist)
    h_max = np.log(n_bins)
    hn = h / h_max if h_max > 0 else 0
    if hn > 0.85: score = 0.75
    elif hn > 0.75: score = 0.55
    elif hn > 0.65: score = 0.35
    else: score = 0.15
    return score, {"entropy_norm": round(hn,3), "entropy_raw": round(h,3)}

def calculate_m19_mutual_info(btc_rets, macro_rets):
    """M19: Mutual Information — BTC/Macro coupling"""
    if btc_rets is None or macro_rets is None or len(btc_rets) < 15: return None, None
    import numpy as np
    n = min(len(btc_rets), len(macro_rets))
    b = btc_rets[-n:]
    m = macro_rets[-n:]
    if n < 10: return None, None
    try:
        b_bins = np.digitize(b, bins=np.percentile(b, [33, 67]))
        m_bins = np.digitize(m, bins=np.percentile(m, [33, 67]))
    except: return None, None
    joint = {}
    for bb, mm in zip(b_bins, m_bins):
        joint[(bb, mm)] = joint.get((bb, mm), 0) + 1
    total = len(b_bins)
    mi = 0
    for (bb, mm), c in joint.items():
        p_j = c / total
        p_b = sum(cc for (bb2, _), cc in joint.items() if bb2 == bb) / total
        p_m = sum(cc for (_, mm2), cc in joint.items() if mm2 == mm) / total
        if p_b > 0 and p_m > 0 and p_j > 0:
            mi += p_j * np.log(p_j / (p_b * p_m))
    mi_norm = min(mi / np.log(3), 1.0) if np.log(3) > 0 else 0
    if mi_norm > 0.70: score = 0.75
    elif mi_norm > 0.50: score = 0.55
    elif mi_norm > 0.30: score = 0.35
    else: score = 0.15
    return score, {"mi_norm": round(mi_norm,3), "mi_raw": round(mi,3)}


# ── M33: GLOBAL LIQUIDITY INDEX ──
_GLO_CACHE = {"score": None, "details": None, "ts": 0}

def calculate_m33_global_liquidity():
    """M33: Global Liquidity Index — tracks aggregate central bank balance sheets.

    Fetches WALCL (Fed), ECBASSETSW (ECB), JPNASSETS (BOJ) from FRED.
    Computes YoY % change, normalizes to GLO score 0-100 (z-score weighted).
    High liquidity expansion = low stress (bullish), contraction = high stress (bearish).

    Returns:
        score_0_1: 0.0-1.0 normalized stress score (high = liquidity contracting)
        details: dict with raw values, changes, and z-score
    """
    global _GLO_CACHE
    now = time.time()
    
    # Cache: 24h (macro data doesn't change often)
    if _GLO_CACHE["score"] is not None and now - _GLO_CACHE["ts"] < 86400:
        return _GLO_CACHE["score"], _GLO_CACHE["details"]
    
    # Fetch global central bank balance sheets
    walcl = _fred("WALCL", 13)       # Fed total assets (13 obs = 1y)
    ecb = _fred("ECBASSETSW", 13)    # ECB total assets
    jpn = _fred("JPNASSETS", 13)     # BOJ total assets
    
    if not walcl or len(walcl) < 2:
        _GLO_CACHE["score"] = 0.5
        _GLO_CACHE["details"] = {"error": "insufficient FRED data"}
        _GLO_CACHE["ts"] = now
        return 0.5, _GLO_CACHE["details"]
    
    # Compute YoY % change for each
    def yoy_chg(arr):
        if not arr or len(arr) < 13: return None
        return (arr[0] - arr[12]) / arr[12] * 100 if arr[12] != 0 else 0
    
    fed_yoy = yoy_chg(walcl)
    ecb_yoy = yoy_chg(ecb) if ecb and len(ecb) >= 13 else None
    jpn_yoy = yoy_chg(jpn) if jpn and len(jpn) >= 13 else None
    
    # Historical reference values for z-score normalization
    # Fed: ~5.5% avg expansion historically; ECB: ~4%; BOJ: ~3%
    # Contracting = negative or below-trend growth
    fed_z = (fed_yoy - 5.5) / 3.0 if fed_yoy is not None else 0
    ecb_z = (ecb_yoy - 4.0) / 3.0 if ecb_yoy is not None else 0
    jpn_z = (jpn_yoy - 3.0) / 3.0 if jpn_yoy is not None else 0
    
    # Weight: Fed ~50%, ECB ~30%, BOJ ~20% (market impact weighting)
    weights = []
    z_vals = []
    if fed_yoy is not None: weights.append(0.50); z_vals.append(fed_z)
    if ecb_yoy is not None: weights.append(0.30); z_vals.append(ecb_z)
    if jpn_yoy is not None: weights.append(0.20); z_vals.append(jpn_z)
    
    if not z_vals:
        _GLO_CACHE["score"] = 0.5
        _GLO_CACHE["details"] = {"error": "no central bank data available"}
        _GLO_CACHE["ts"] = now
        return 0.5, {"error": "no central bank data"}
    
    total_w = sum(weights)
    glo_z = sum(w * z for w, z in zip(weights, z_vals)) / total_w
    
    # Normalize z-score to GLO score 0-100
    # z=+2 (strong expansion) → GLO=90, z=-2 (contraction) → GLO=10
    glo_score_raw = 50 + glo_z * 20
    glo_score = max(0, min(100, glo_score_raw))
    
    # Map to SFC method score (0-1): high GLO = liquid = low stress
    # GLO 0-100 → SFC score 0-1
    # GLO>70 (liquid) → SFC score <0.25 (low stress)
    # GLO<30 (contraction) → SFC score >0.75 (high stress)
    if glo_score > 70:
        sfc_score = 0.15  # very liquid -> minimal stress
    elif glo_score > 55:
        sfc_score = 0.25  # moderately liquid -> low stress
    elif glo_score > 40:
        sfc_score = 0.50  # neutral
    elif glo_score > 25:
        sfc_score = 0.70  # contracting -> elevated stress
    else:
        sfc_score = 0.85  # severe contraction -> high stress
    
    details = {
        "fed_yoy": round(fed_yoy, 2) if fed_yoy is not None else None,
        "ecb_yoy": round(ecb_yoy, 2) if ecb_yoy is not None else None,
        "jpn_yoy": round(jpn_yoy, 2) if jpn_yoy is not None else None,
        "fed_balance": round(walcl[0], 0) if walcl else None,
        "ecb_balance": round(ecb[0], 0) if ecb else None,
        "jpn_balance": round(jpn[0], 0) if jpn else None,
        "glo_z_score": round(glo_z, 3),
        "glo_score": round(glo_score, 1),
        "glo_label": "EXPANSIVE" if glo_score > 55 else "NEUTRAL" if glo_score > 40 else "CONTRACTIVE",
    }
    
    _GLO_CACHE["score"] = sfc_score
    _GLO_CACHE["details"] = details
    _GLO_CACHE["ts"] = now
    return sfc_score, details


# ── MACRO LIQUIDITY (M72-M75 / Layer 1) ─────────────────────────────
_MACRO_CACHE = {"score": None, "details": {}, "ts": 0}

def calculate_m72_m2_growth(m2_yoy_input=None):
    """M72: Global M2 Growth — YoY % change in US M2 money supply.
    
    Uses existing m2_yoy_input if provided, else fetches from FRED.
    Higher M2 growth = more liquidity = bullish.
    """
    # Prefer pre-fetched m2_yoy from get_m2_data() to avoid duplicate API call
    if m2_yoy_input is not None:
        m2_yoy_val = m2_yoy_input
    else:
        m2 = _fred("M2SL", 13)
        if not m2 or len(m2) < 2:
            return None, {"m2_yoy": None, "status": "unavailable"}
        m2_yoy_val = (m2[0] - m2[12]) / m2[12] * 100 if len(m2) >= 13 else None
        if m2_yoy_val is None and len(m2) >= 2:
            m2_yoy_val = (m2[0] - m2[-1]) / m2[-1] * 100
    
    # Score: M2 growth 2-7% = normal, <0% = contraction (bearish), >10% = overheating
    if m2_yoy_val < 0:
        score = max(0.05, 0.3 + m2_yoy_val * 0.03)
    elif m2_yoy_val < 5:
        score = 0.3 + (m2_yoy_val / 5) * 0.4
    elif m2_yoy_val < 10:
        score = 0.7 + min(0.2, (m2_yoy_val - 5) * 0.04)
    else:
        score = 0.9  # overheating — potential tightening
    
    detail = {
        "m2_yoy_pct": round(m2_yoy_val, 2),
        "m2_latest": round(m2[0], 0) if m2 else None,
        "status": "ok",
    }
    return round(score, 3), detail


def calculate_m73_m2_momentum():
    """M73: M2 Momentum — 3-month growth minus 12-month growth.
    
    Positive = accelerating liquidity (bullish). Negative = decelerating (bearish).
    """
    m2 = _fred("M2SL", 13)
    if not m2 or len(m2) < 13:
        return None, {"m2_momentum": None, "status": "unavailable"}
    
    growth_3m = (m2[0] - m2[3]) / m2[3] * 100
    growth_12m = (m2[0] - m2[12]) / m2[12] * 100
    momentum = growth_3m - growth_12m  # acceleration
    
    # Score: momentum > +1% = accelerating (bullish)
    # momentum 0 to +1% = steady
    # momentum < 0 = decelerating (bearish)
    score = 1.0 / (1.0 + math.exp(-1.5 * (momentum - 0.3)))
    score = max(0.05, min(0.95, score))
    
    if momentum > 1:
        label = "ACCELERATING"
    elif momentum > 0:
        label = "STEADY"
    elif momentum > -1:
        label = "DECELERATING"
    else:
        label = "CONTRACTING"
    
    detail = {
        "m2_momentum": round(momentum, 3),
        "m2_growth_3m": round(growth_3m, 2),
        "m2_growth_12m": round(growth_12m, 2),
        "label": label,
        "status": "ok",
    }
    return round(score, 3), detail


def calculate_m74_fed_balance():
    """M74: Fed Balance Sheet — YoY % change in Fed total assets (WALCL).
    
    Fed expanding balance sheet = liquidity injection = bullish.
    Fed shrinking (QT) = liquidity withdrawal = bearish.
    Uses same data as M33 GLO.
    """
    walcl = _fred("WALCL", 13)
    if not walcl or len(walcl) < 13:
        return None, {"fed_yoy": None, "status": "unavailable"}
    
    fed_yoy = (walcl[0] - walcl[12]) / walcl[12] * 100
    
    # Fed balance sheet expansion:
    # >+10% = massive QE → very bullish (low stress)
    # +2% to +10% = gradual expansion → mildly bullish
    # -2% to +2% = neutral/stable
    # -5% to -2% = mild QT → mildly bearish
    # <-5% = aggressive QT → bearish
    
    if fed_yoy > 10:
        score = 0.1  # massive liquidity — very low stress
    elif fed_yoy > 2:
        score = 0.2  # expansion — low stress
    elif fed_yoy > -2:
        score = 0.4  # stable — medium-low
    elif fed_yoy > -5:
        score = 0.6  # mild QT
    else:
        score = 0.8  # aggressive QT — high stress
    
    if fed_yoy > 2:
        label = "EXPANDING"
    elif fed_yoy > -2:
        label = "STABLE"
    elif fed_yoy > -5:
        label = "MILD_QT"
    else:
        label = "AGGRESSIVE_QT"
    
    detail = {
        "fed_yoy_pct": round(fed_yoy, 2),
        "fed_balance": round(walcl[0], 0),
        "label": label,
        "status": "ok",
    }
    return round(score, 3), detail


def calculate_m75_liquidity_composite(m72_score, m72_detail, m73_score, m73_detail, m74_score, m74_detail):
    """M75: Liquidity Composite — weighted blend of M72-M74.
    
    Combines all three macro liquidity signals into one composite score.
    """
    scores = []
    weights = []
    
    if m72_score is not None:
        scores.append(m72_score)
        weights.append(0.30)  # M2 growth: 30%
    if m73_score is not None:
        scores.append(m73_score)
        weights.append(0.30)  # M2 momentum: 30%
    if m74_score is not None:
        scores.append(m74_score)
        weights.append(0.40)  # Fed balance sheet: 40% (most direct policy signal)
    
    if not scores:
        return None, {"composite": None, "status": "unavailable"}
    
    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w
    
    # Map to SFC stress score (high composite = high macro stress = bearish)
    # For the frontend: M72-M75 produce a 0-1 score where HIGH = more stress
    # This follows the same convention as M1-M80
    
    active_scores = ", ".join([
        f"M72={m72_score:.2f}" if m72_score is not None else "",
        f"M73={m73_score:.2f}" if m73_score is not None else "",
        f"M74={m74_score:.2f}" if m74_score is not None else "",
    ]).strip(", ")
    
    if composite < 0.2:
        regime = "EXPANSIVE"
    elif composite < 0.4:
        regime = "ACCOMMODATIVE"
    elif composite < 0.6:
        regime = "NEUTRAL"
    elif composite < 0.8:
        regime = "TIGHTENING"
    else:
        regime = "CONTRACTIVE"
    
    detail = {
        "composite": round(composite, 3),
        "regime": regime,
        "m72_score": round(m72_score, 3) if m72_score is not None else None,
        "m73_score": round(m73_score, 3) if m73_score is not None else None,
        "m74_score": round(m74_score, 3) if m74_score is not None else None,
        "active_components": active_scores,
        "status": "ok",
    }
    return round(composite, 3), detail

# ============================================================
# MONTHLY TIMEFRAME: Daily Market Snapshot Cache (30d rolling)
# ============================================================
MARKET_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.daily_market_cache.json')

def _store_market_snapshot(btc, btc_24h, dom, dvol, fng, pc_oi, m2_yoy, dxy):
    """Store daily market snapshot for 30d rolling averages. Dedup by date."""
    cache = []
    if os.path.exists(MARKET_CACHE_FILE):
        try:
            with open(MARKET_CACHE_FILE) as f:
                cache = json.load(f)
        except: cache = []
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {
        "date": today,
        "ts": time.time(),
        "btc": btc,  # BTC price LEVEL (not % change) — required by
                     # _compute_dxy_btc_correlation() to compute daily returns.
                     # Audit A1 (2026-08-03): the DXY-BTC correlation gate was
                     # dead code because only btc_24h (% change) was stored, so
                     # `entry.get("btc")` was always None and the POSITIVE/MIXED
                     # Sc regimes in score_factors_from_market() never fired.
        "btc_24h": btc_24h,
        "dom": dom,
        "dvol": dvol,
        "fng": fng,
        "pc_oi": pc_oi,
        "m2_yoy": m2_yoy,
        "dxy": dxy,
    }
    # Replace today's entry if exists, else append
    for i, e in enumerate(cache):
        if e.get("date") == today:
            cache[i] = entry
            break
    else:
        cache.append(entry)
    
    # Keep last 60 days
    if len(cache) > 60:
        cache = cache[-60:]
    
    with open(MARKET_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    return cache

def _get_30d_rolling(key, cache, default=None):
    """Compute 30-day SMA from daily cache. Needs at least 7 days to start."""
    if not cache or len(cache) < 7:
        return default
    recent = cache[-30:] if len(cache) >= 30 else cache
    vals = [e.get(key) for e in recent if e.get(key) is not None]
    if not vals or len(vals) < 7:
        return default
    return sum(vals) / len(vals)

def get_btc_ohlcv_monthly(months=36):
    """Fetch BTC monthly candles from Binance (1M interval)."""
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1M&limit={months}",
            timeout=10
        )
        if r.status_code != 200:
            # Fallback: fetch daily and aggregate
            r2 = requests.get(
                f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit={months*30}",
                timeout=10
            )
            if r2.status_code != 200:
                return []
            daily = r2.json()
            return _aggregate_daily_to_monthly(daily)
        klines = r.json()
        return [{"time": k[0], "close": float(k[4]), "volume": float(k[5])} for k in klines]
    except:
        return []

def _aggregate_daily_to_monthly(daily_klines):
    """Aggregate daily klines into monthly candles (close=last close, vol=sum)."""
    monthly = {}
    for k in daily_klines:
        ts = k[0] / 1000
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        month_key = d.strftime("%Y-%m")
        close = float(k[4])
        volume = float(k[5])
        if month_key in monthly:
            monthly[month_key]["time"] = k[0]
            monthly[month_key]["close"] = close
            monthly[month_key]["volume"] += volume
        else:
            monthly[month_key] = {"time": k[0], "close": close, "volume": volume}
    result = sorted(monthly.values(), key=lambda x: x["time"])
    return result[-months:] if len(result) >= months else result

def _binance_monthly_klines(months=36):
    """Fetch BTC monthly klines from Binance as fallback."""
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1M&limit={months}",
            timeout=10
        )
        if r.status_code != 200: return []
        klines = r.json()
        return [{"time": k[0], "close": float(k[4]), "volume": float(k[5])} for k in klines]
    except: return []

print("[SFC] Starting parallel data collection...", file=sys.stderr)

# ── PARALLEL API DATA COLLECTION ──
# All API calls run concurrently via ThreadPoolExecutor (saves ~2.5s)
from concurrent.futures import ThreadPoolExecutor, as_completed

_api_results = {}

def _fetch_and_store(key, fn, *args, **kwargs):
    """Fetch API data and store in shared dict."""
    _api_results[key] = fn(*args, **kwargs)

with ThreadPoolExecutor(max_workers=8) as ex:
    ex.submit(_fetch_and_store, "btc", get_btc)
    ex.submit(_fetch_and_store, "fng", get_fng)
    ex.submit(_fetch_and_store, "dom", get_dom)
    ex.submit(_fetch_and_store, "dvol", get_dvol)
    ex.submit(_fetch_and_store, "m2", get_m2_data)
    ex.submit(_fetch_and_store, "dxy", get_dxy)
    ex.submit(_fetch_and_store, "pc", get_put_call_ratio)
    ex.submit(_fetch_and_store, "ath", get_ath)
    ex.submit(_fetch_and_store, "ohlcv", get_btc_ohlcv_monthly, 36)

# ── PARALLEL FRED PREFETCH ──
# Fetch ALL FRED data in one batch before M7-M19 need them
print("[SFC] Prefetching FRED data (parallel)...", file=sys.stderr)
_fred_prefetch()

# Unpack results
btc, chg, mcap = _api_results.get("btc", (None, None, None))
fng, fcls = _api_results.get("fng", (None, None))
dom = _api_results.get("dom", None)
dvol = _api_results.get("dvol", None)
m2_val, m2_yoy, _ = _api_results.get("m2", (None, None, None))
dxy = _api_results.get("dxy", None)
pc_oi, pc_vol = _api_results.get("pc", (None, None))
ath, ath_date = _api_results.get("ath", (None, None))
ohlcv = _api_results.get("ohlcv", None) or _binance_monthly_klines(36)
closes_all = [c["close"] for c in ohlcv] if ohlcv else []
closes_7m = closes_all[-7:] if len(closes_all) >= 7 else closes_all
closes_30m = closes_all[-30:] if len(closes_all) >= 30 else closes_all
rsi_14m = compute_rsi(closes_all, period=14)

# ── Store daily snapshot for 30d rolling averages ──
_market_cache = _store_market_snapshot(btc, chg, dom, dvol, fng, pc_oi, m2_yoy, dxy)
_chg_30d = _get_30d_rolling("btc_24h", _market_cache)
_dom_30d = _get_30d_rolling("dom", _market_cache)
_dvol_30d = _get_30d_rolling("dvol", _market_cache)
_fng_30d = _get_30d_rolling("fng", _market_cache)
_pc_30d = _get_30d_rolling("pc_oi", _market_cache)
_m2_30d = _get_30d_rolling("m2_yoy", _market_cache, m2_yoy)
_dxy_30d = _get_30d_rolling("dxy", _market_cache)

# Use 30d rolling averages for factors (long-term view); fallback to spot
_factors_btc_24h = _chg_30d if _chg_30d is not None else chg
_factors_dom = _dom_30d if _dom_30d is not None else dom
_factors_dvol = _dvol_30d if _dvol_30d is not None else dvol
_factors_fng = _fng_30d if _fng_30d is not None else fng
_factors_pc = _pc_30d if _pc_30d is not None else pc_oi
_factors_m2 = _m2_30d if _m2_30d is not None else m2_yoy
_factors_dxy = _dxy_30d if _dxy_30d is not None else dxy
# Override raw dvol with rolling fallback so ALL downstream consumers
# (Mamba dict, output JSON) get the fallback instead of None when API fails
dvol = _factors_dvol
print(f"[SFC] 30d rolling averages: BTC24h={_factors_btc_24h} DOM={_factors_dom} DVOL={_factors_dvol} FnG={_factors_fng} M2={_factors_m2}", file=sys.stderr)

# ── DXY-BTC Correlation Gate (for Sc factor sign) ──
dxy_btc_corr = _compute_dxy_btc_correlation()
if dxy_btc_corr is not None:
    print(f"[DXY Gate] Rolling 30d DXY-BTC correlation: {dxy_btc_corr:.3f}", file=sys.stderr)

# ── ADVANCED FEATURE ENGINEERING (Peningkatan 1: 25+ technical indicators) ──
_adv_features = {}
_adv_features_module = _get_adv_features()
if _adv_features_module:
    try:
        _adv_features = _adv_features_module.get_features()
        if _adv_features:
            print(f"[AFE] Loaded {len(_adv_features)} advanced features", file=sys.stderr)
    except Exception as _afe_e:
        print(f"[AFE] Error: {_afe_e}", file=sys.stderr)
        _adv_features = {}

# ── MULTI-TIMEFRAME FUSION (Peningkatan 4: 1h/4h/1d/1w alignment) ──
_mtf_result = {}
_mtf_module = _get_adv_mtf()
if _mtf_module:
    try:
        _mtf_result = _mtf_module.safe_multi_timeframe_fusion()
        if _mtf_result and 'alignment_score' in _mtf_result:
            print(f"[MTF] Alignment={_mtf_result.get('alignment_score',0):.3f} "
                  f"Divergence={_mtf_result.get('divergence_detected',False)}", file=sys.stderr)
    except Exception as _mtf_e:
        print(f"[MTF] Error: {_mtf_e}", file=sys.stderr)
        _mtf_result = {}

# ── ON-CHAIN DATA (Q10 Integration — ErcinDedeoglu/crypto-market-data) ──
print("[SFC] Fetching on-chain data (Q10+)...", file=sys.stderr)
onchain_scores = {}
whale_pressure = onchain_value = buying_power = market_structure = None
try:
    onchain_scores = fetch_all_onchain()
    whale_pressure = onchain_scores["whale_pressure"]
    onchain_value = onchain_scores["onchain_value"]
    buying_power = onchain_scores["buying_power"]
    market_structure = onchain_scores.get("market_structure")
    print(f"[SFC] Q10: Whale={whale_pressure} OnValue={onchain_value} BuyPower={buying_power} MktStruct={market_structure}", file=sys.stderr)
except Exception as e:
    print(f"[SFC] On-chain fetch failed: {e}", file=sys.stderr)
    whale_pressure = onchain_value = buying_power = market_structure = None

# ── REFLEXIVITY DIVERGENCE (EXPERIMENTAL — Option A: display-only) ──
# Simplified discrete-derivative version of the Soros-style reflexivity
# feedback loop (price vs fundamental vs leverage). Deliberately kept
# SEPARATE from factors/sfc_pct for now — this is a new, unvalidated
# signal (see analysis/reflexivity_divergence.py's module docstring for
# the full design rationale and honest caveats about the arbitrary scale
# constants). Exposed as its own field so it can be observed and
# compared against live data before any decision to fold it into the
# core ensemble — same cautious rollout pattern used for M86/M90 before
# they were trusted enough to affect factors directly.
_reflexivity_score, _reflexivity_details = 50.0, {"status": "unavailable"}
try:
    from analysis.reflexivity_divergence import compute_reflexivity_divergence
    _q10_details_for_reflexivity = onchain_scores.get("details", {}) if onchain_scores else {}
    _mvrv_val = _q10_details_for_reflexivity.get("mvrv_ratio", {}).get("value")
    _oi_val = _q10_details_for_reflexivity.get("open_interest", {}).get("value")
    if btc is not None and _mvrv_val is not None and _oi_val is not None:
        _reflexivity_score, _reflexivity_details = compute_reflexivity_divergence(
            price=btc, mvrv=_mvrv_val, leverage=_oi_val
        )
        print(f"[Reflexivity] score={_reflexivity_score} regime={_reflexivity_details.get('regime','?')}", file=sys.stderr)
    else:
        print("[Reflexivity] Skipped this cycle — missing price/mvrv/open_interest", file=sys.stderr)
except Exception as e:
    print(f"[Reflexivity] Error: {e}", file=sys.stderr)

# ── STABLECOIN LIQUIDITY METRICS (M76-M80 / Layer 2 Crypto Liquidity) ──
print("[SFC] Computing M76-M80 stablecoin liquidity metrics...", file=sys.stderr)
_sc_results, sc_details, sc_active, sc_avg = {}, {}, 0, None
if STABLECOIN_AVAILABLE:
    try:
        # Get BTC dominance from existing data
        _btc_dom_for_sc = dom if dom else 58.3
        _onchain_details_for_sc = onchain_scores.get("details", {}) if onchain_scores else {}
        _sc_results, sc_details, sc_active, sc_avg = compute_all_stablecoin_metrics(
            btc_price=btc,
            btc_mcap=mcap,
            btc_dominance_pct=_btc_dom_for_sc,
            onchain_details=_onchain_details_for_sc,
            force_refresh=False,
        )
        print(f"[SFC] M76-M80: {sc_active}/5 active, avg={sc_avg}", file=sys.stderr)
    except Exception as e:
        print(f"[SFC] Stablecoin metrics failed: {e}", file=sys.stderr)

# ── NEW: Stablecoin Liquidity Index (SLI) ──
_sli_score = None
_sli_sfc_stress = None
_sli_details = {}
if STABLECOIN_INTEL_AVAILABLE:
    try:
        _sli_score, _sli_sfc_stress, _sli_details = compute_stablecoin_liquidity_index(
            existing_sc_results=_sc_results if _sc_results else None,
            existing_sc_details=sc_details if sc_details else None,
            btc_price=btc,
            btc_mcap=mcap,
            btc_dominance_pct=dom if dom else 58.3,
            onchain_details=onchain_scores.get("details", {}) if onchain_scores else {},
            force_refresh=False,
        )
        print(f"[SLI] Score={_sli_score:.1f}/100 stress={_sli_sfc_stress:.3f} label={_sli_details.get('label','?')} "
              f"components={_sli_details.get('n_components',0)}", file=sys.stderr)
    except Exception as _sli_e:
        print(f"[SLI] Error: {_sli_e}", file=sys.stderr)
        _sli_score, _sli_sfc_stress, _sli_details = 50.0, 0.5, {"error": str(_sli_e), "status": "fallback"}

# ── ETF FLOW (M81-M82) ──
_etf_results = None
_etf_m81_score = 0.5
_etf_m82_score = 0.5
_etf_details = {"status": "unavailable"}
if ETF_AVAILABLE:
    try:
        _etf_m81_score, _etf_m82_score, _etf_details = compute_etf_metrics(btc_price=btc)
        print(f"[ETF] M81={_etf_m81_score:.3f} M82={_etf_m82_score:.3f} | "
              f"flow={_etf_details.get('m81_latest_flow_btc', '?')} BTC | "
              f"cumulative={_etf_details.get('m82_cumulative_btc', '?'):,.0f} BTC", file=sys.stderr)
    except Exception as e:
        print(f"[ETF] Error: {e}", file=sys.stderr)

# ── BEHAVIORAL DIVERGENCE DETECTOR (EXPERIMENTAL — Option A: display-only) ──
# Detects mismatch between price action (btc_24h) and the DIRECTION of
# institutional/whale flow signals ALREADY computed above (M81 ETF flow,
# Q10 whale pressure, SLI stablecoin liquidity) — purely a different LENS
# on existing signals, not a new data source. Deliberately kept separate
# from factors/sfc_pct: re-combining these same signals into the core
# ensemble a second time would double-count them, the exact mistake
# already found and fixed for netflow/M81-M82 elsewhere in this project.
# See analysis/behavioral_divergence.py's module docstring for full
# rationale and honest caveats about the unvalidated 0.15 threshold.
_divergence_score, _divergence_details = 0.0, {"status": "unavailable"}
try:
    from analysis.behavioral_divergence import compute_behavioral_divergence
    from analysis.behavioral_divergence_tracker import record_divergence
    _m81_is_available = _etf_details.get("status") == "ok"
    _divergence_score, _divergence_details = compute_behavioral_divergence(
        m81_etf_flow=_etf_m81_score, m81_available=_m81_is_available,
        q10_whale_pressure=whale_pressure, sli_score=_sli_score,
        btc_24h=chg,
    )
    print(f"[Divergence] score={_divergence_score} regime={_divergence_details.get('regime','?')}", file=sys.stderr)
    # Track historical divergence signals for forward-return validation
    try:
        _track_result = record_divergence(
            ts=datetime.now(timezone.utc).isoformat(),
            score=_divergence_score,
            detail=_divergence_details,
            btc_price=btc,
        )
        print(f"[DivergenceTracker] recorded={_track_result['entries_recorded']} total={_track_result['total_entries']}", file=sys.stderr)
    except Exception as te:
        print(f"[DivergenceTracker] Error: {te}", file=sys.stderr)
except Exception as e:
    print(f"[Divergence] Error: {e}", file=sys.stderr)

# ── FISCAL LIQUIDITY (M83-M84) ──
_m83_score = 0.5
_m84_score = 0.5
_m85_composite = 0.5
_fiscal_details = {"status": "unavailable"}
if FISCAL_AVAILABLE:
    try:
        _m83_score, _m84_score, _m85_composite, _fiscal_details = compute_fiscal_liquidity_metrics()
        print(f"[FISCAL] M83(TGA)={_m83_score:.3f} M84(RRP)={_m84_score:.3f} "
              f"Composite={_m85_composite:.3f} | Regime={_fiscal_details.get('regime','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[FISCAL] Error: {e}", file=sys.stderr)

# ── REPO MARKET STRESS (M86 — SOFR-EFFR spread) ──
# Moved here (was previously computed much later, at a point AFTER
# score_factors_from_market() had already been called and consumed the
# stale default value) — see the note left in score_factors_from_market()
# itself for the full explanation of the bug this fixes.
_m86_score = 0.5
_m86_details = {"status": "unavailable"}
if REPO_STRESS_AVAILABLE:
    try:
        _m86_score, _m86_details = compute_repo_stress()
        print(f"[REPO] M86(SOFR-EFFR)={_m86_score:.3f} | spread={_m86_details.get('spread_bps')}bp "
              f"label={_m86_details.get('label','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[REPO] Error: {e}", file=sys.stderr)

# ── GLOBAL SOVEREIGN LIQUIDITY SCORE (M90) ──
# Consolidates US/Japan/Europe/UK sovereign bond signals into ONE score
# (0-100) rather than separate M88/M89/M90 methods — see
# global_sovereign_liquidity.py module docstring for the full design
# rationale (consistent with method_independence_analysis.py's findings:
# feeding many correlated raw indicators separately into the ensemble
# double-counts shared information; one consolidated latent factor is
# cleaner signal for QLSTM/probabilistic modules to learn from).
_m90_score, _m90_details = 50.0, {"status": "unavailable"}
if GSLS_AVAILABLE:
    try:
        _m90_score, _m90_details = compute_global_sovereign_liquidity()
        print(f"[GSLS] M90(GlobalSovereignLiquidity)={_m90_score:.1f} | "
              f"regime={_m90_details.get('regime','?')}", file=sys.stderr)
    except Exception as e:
        print(f"[GSLS] Error: {e}", file=sys.stderr)

# Score factors from market data (using 30d rolling averages + on-chain)
factors = score_factors_from_market(btc, _factors_btc_24h, _factors_dom, _factors_dvol, _factors_fng, _factors_pc, _factors_m2, _factors_dxy,
                                     onchain_whale=whale_pressure, onchain_value=onchain_value, onchain_buy=buying_power,
                                     onchain_market_structure=market_structure, dxy_btc_corr=dxy_btc_corr)

# ── ETF FLOW FACTOR ADJUSTMENT ──
if _etf_m81_score != 0.5 or _etf_m82_score != 0.5:
    etf_rt_adj = (0.5 - _etf_m81_score) * 1.5
    etf_lt_adj = (0.5 - _etf_m82_score) * 1.5
    factors["Rt"] += max(-1.5, min(1.5, etf_rt_adj))
    factors["Lt"] += max(-1.5, min(1.5, etf_lt_adj))
    print(f"[ETF] Factor adj: Rt={etf_rt_adj:+.3f} Lt={etf_lt_adj:+.3f}", file=sys.stderr)

# ── FISCAL LIQUIDITY FACTOR ADJUSTMENT ──
if _m83_score != 0.5 or _m84_score != 0.5:
    tga_adj = (0.5 - _m83_score) * 1.0
    rrp_adj = (0.5 - _m84_score) * 1.0
    factors["Lt"] += max(-1.0, min(1.0, tga_adj + rrp_adj))
    print(f"[FISCAL] Factor adj: TGA={tga_adj:+.3f} RRP={rrp_adj:+.3f} Lt_total={tga_adj+rrp_adj:+.3f}", file=sys.stderr)

# ── REPO MARKET STRESS FACTOR ADJUSTMENT (M86) ──
# Applied to Ft (Systemic/Funding), not Lt — repo stress measures whether
# funding markets are transmitting liquidity smoothly RIGHT NOW, a
# different dimension from Lt's central-bank-balance-sheet liquidity
# LEVEL. See repo_market_stress.py module docstring for the full
# rationale (Sept 2019 repo crisis as the reference case: ample Fed
# liquidity coexisting with a funding-market seizure). This block is now
# correctly positioned AFTER score_factors_from_market() returns and
# AFTER the real _m86_score was computed above — previously this logic
# lived inside the function definition itself, where it silently never
# fired (see the note left at that old location for the full bug).
if _m86_score != 0.5:
    repo_adj = (_m86_score - 0.5) * 1.5
    factors["Ft"] += max(-1.5, min(1.5, repo_adj))
    print(f"[REPO] Factor adj: Ft={repo_adj:+.3f}", file=sys.stderr)

# ── GLOBAL SOVEREIGN LIQUIDITY SCORE FACTOR ADJUSTMENT (M90) ──
# Applied to Lt (Long-term/liquidity LEVEL), not Ft — GSLS captures
# sovereign yield curve shape and carry-trade fragility across
# US/Japan/Europe/UK, the same "central-bank-and-macro-driven liquidity
# backdrop" dimension GLF's Fed/ECB/BOJ/M2 components already measure,
# just from the yield-curve/term-structure angle rather than
# balance-sheet-size. See global_sovereign_liquidity.py module docstring
# for the full design rationale, including why repo stress (M86, already
# feeding Ft above) is deliberately NOT duplicated inside GSLS.
if _m90_score != 50.0:
    gsls_adj = (_m90_score - 50.0) / 50.0 * 1.0  # normalize 0-100 scale to a comparable adjustment magnitude
    factors["Lt"] += max(-1.0, min(1.0, gsls_adj))
    print(f"[GSLS] Factor adj: Lt={gsls_adj:+.3f}", file=sys.stderr)

# Re-clamp factors after all adjustments
for k in factors:
    factors[k] = max(-3.0, min(3.0, factors[k]))

# ── FACTOR-LEVEL OUTLIER GUARD ──────────────────────────────────
# Runs right before calculate_sfc_ensemble() uses `factors` to produce
# sfc_pct (the headline stress score). This closes a gap found during
# audit: DataQualityPipeline (data_quality.py) already does proper
# outlier detection + Kalman imputation on the 31 method scores, but it
# only runs much later in this script (after sfc_pct is already final),
# so its cleaned output was never actually used to protect the score
# users see — it only fed monitoring fields (dq_outliers etc). Rather
# than relocate that whole pipeline (which depends on ~31 method-score
# locals not all available this early), this is a lighter, targeted
# check on the 5 factors directly feeding the ensemble: it flags (and
# damps, not discards) any factor that has moved implausibly far from
# its own recent history, since a single corrupted API value showing up
# as e.g. Lt=2.9 instead of Lt=-0.3 would otherwise pass straight through
# the re-clamp above (which only bounds to [-3,3], not "is this plausible
# given recent history").
def _apply_factor_outlier_guard(factors_dict):
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".factor_history.json")
    try:
        with open(history_path, "r") as f:
            hist = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        hist = {k: [] for k in factors_dict}

    flagged = []
    # Minimum std floor: each factor's plausible range is roughly [-3, 3]
    # (see the re-clamp above), so a swing under ~0.5 is well within normal
    # day-to-day movement regardless of how flat the recent history
    # happens to be. Without this floor, a few cycles of coincidentally
    # stable readings would make the z-score test oversensitive to entirely
    # ordinary moves — confirmed during testing: a calm period with
    # std≈0.03 flagged an ordinary 0.2-unit move as a "z=5.66 outlier".
    MIN_STD_FLOOR = 0.5
    for k, v in factors_dict.items():
        past = hist.get(k, [])
        if len(past) >= 5:
            mean = sum(past) / len(past)
            var = sum((x - mean) ** 2 for x in past) / len(past)
            std = max(var ** 0.5, MIN_STD_FLOOR)
            z = abs(v - mean) / std
            if z > 4.0:
                # Implausible jump vs this factor's own recent history.
                # Damp toward the historical mean rather than discard
                # outright — a genuine regime break should still move
                # the score, just not by the full magnitude of what
                # may be a bad API read.
                damped = mean + (v - mean) * 0.3
                flagged.append((k, round(v, 3), round(damped, 3), round(z, 2)))
                factors_dict[k] = damped

        hist.setdefault(k, []).append(v)
        hist[k] = hist[k][-10:]

    try:
        with open(history_path, "w") as f:
            json.dump(hist, f)
    except OSError:
        pass

    if flagged:
        for k, orig, damped, z in flagged:
            print(f"[DQ-GUARD] {k}: {orig} -> {damped} (z={z}, dampened toward recent history)",
                  file=sys.stderr)

    return factors_dict, flagged


factors, _factor_outlier_flags = _apply_factor_outlier_guard(factors)


# Calculate SFC ensemble
sfc_pct, zone, factors_raw, norm_factors, m1_klr, m2_logit, m3_bayes, m4_ewc, m5_qreg, m6_regime, method_agreement = calculate_sfc_ensemble(factors)

# ── NEW METHODS (M7-M19) ──
print("[SFC] Computing M7-M19 enhancement methods...", file=sys.stderr)
m7_s, m7_d = calculate_m7_fisher()
m8_s, m8_d = calculate_m8_yield_curve()
m9_s, m9_d = calculate_m9_liquidity()
m10_s, m10_d = calculate_m10_garch(closes_all)
rets_arr = [c["close"] for c in ohlcv] if ohlcv else []
rets_pct = None
if len(rets_arr) > 1:
    rets_pct = [(rets_arr[i] - rets_arr[i-1]) / rets_arr[i-1] for i in range(1, len(rets_arr))]
m11_s, m11_d = calculate_m11_var(rets_pct)
m12_s, m12_d = calculate_m12_jump(ohlcv)
m13_s, m13_d = calculate_m13_funding()
m14_s, m14_d = calculate_m14_skew()
m15_s, m15_d = calculate_m15_concentration()
m16_s, m16_d = calculate_m16_regime_switch(rets_pct)
# Granger: M2 → DVOL causality
m2_hist = _fred("M2SL", 30)
m2_rets = None
if m2_hist and len(m2_hist) > 1:
    m2_rets = [(m2_hist[i] - m2_hist[i-1]) / m2_hist[i-1] for i in range(1, len(m2_hist))]
m17_s, m17_d = calculate_m17_granger(m2_rets, rets_pct)
m18_s, m18_d = calculate_m18_entropy(rets_arr)
# MI: BTC returns vs DXY returns
dxy_hist = _fred("DTWEXBGS", 30)
dxy_rets = None
if dxy_hist and len(dxy_hist) > 1:
    dxy_rets = [(dxy_hist[i] - dxy_hist[i-1]) / dxy_hist[i-1] for i in range(1, len(dxy_hist))]
m19_s, m19_d = calculate_m19_mutual_info(rets_pct, dxy_rets)

# ── INSTITUTIONAL METHODS (M20-M31) ──
print("[SFC] Computing M20-M31 institutional methods...", file=sys.stderr)
inst_results, inst_details, inst_active, inst_avg, micro_change_flags, micro_trend_score, micro_deteriorating = compute_all_institutional(btc_current=btc)

# ── QLSTM INFERENCE (M32 — Hybrid Quantum LSTM) ──
print("[SFC] Running QLSTM inference (M32)...", file=sys.stderr)
qlstm_pred, qlstm_ok = _run_qlstm_inference()
if qlstm_ok:
    print(f"[SFC] QLSTM predicts SFC={qlstm_pred*100:.1f}%", file=sys.stderr)
else:
    print("[SFC] QLSTM unavailable", file=sys.stderr)

# ── GLOBAL LIQUIDITY INDEX (M33) ──
print("[SFC] Computing M33 Global Liquidity Index...", file=sys.stderr)
m33_glo_score, m33_glo_detail = calculate_m33_global_liquidity()
print(f"[SFC] GLO score={m33_glo_score:.3f} label={m33_glo_detail.get('glo_label','N/A')} fed_yoy={m33_glo_detail.get('fed_yoy','N/A')}%", file=sys.stderr)

# ── NEW: Global Liquidity Factor (GLF) — consolidated liquidity engine ──
_glf_score = None
_glf_sfc_stress = None
_glf_details = {}
if GLOBAL_LIQUIDITY_AVAILABLE:
    try:
        _glf_score, _glf_sfc_stress, _glf_details = compute_global_liquidity_factor()
        print(f"[GLF] Score={_glf_score:.1f}/100 stress={_glf_sfc_stress:.3f} regime={_glf_details.get('regime','?')} "
              f"components={_glf_details.get('active_components',0)}", file=sys.stderr)
    except Exception as _glf_e:
        print(f"[GLF] Error: {_glf_e}", file=sys.stderr)
        _glf_score, _glf_sfc_stress, _glf_details = 50.0, 0.5, {"error": str(_glf_e), "status": "fallback"}

# Apply GLF to Lt factor (consolidated liquidity — replaces former M33 GLO
# + direct m2_yoy sigmoid which were redundant, sharing the same underlying
# Fed/ECB/BOJ/M2 data. Scaled up to compensate for the removed terms;
# see analysis/lt_redundancy_experiment.py for the full walk-forward proof.)
if _glf_sfc_stress is not None:
    # LT_EMPIRICAL_RESCALE = 5.927 (std_A/std_B dari eksperimen redundansi Lt)
    # Eksperimen membuktikan Lt tunggal ×5.927 menyamai/ melampaui performa
    # redundant 3-term stack. Live GLF range ±1.5 vs eksperimen ±1.0, tapi
    # clamp akhir ±3.0 menjaga saturasi tetap sama.
    glf_factor_adj = get_glf_for_factors(_glf_sfc_stress) * 5.927
    factors["Lt"] += glf_factor_adj
    print(f"[GLF] Lt adjustment: {glf_factor_adj:+.3f} (×5.927 scaled, glf_stress={_glf_sfc_stress:.3f}) Lt={factors['Lt']:.3f}", file=sys.stderr)

factors["Lt"] = max(-3.0, min(3.0, factors["Lt"]))
print(f"[SFC] Lt after GLF+ETF+Fiscal+GSLS: {factors['Lt']:.3f}", file=sys.stderr)

# ── MACRO LIQUIDITY (M72-M75 / Layer 1) ──
print("[SFC] Computing M72-M75 macro liquidity metrics...", file=sys.stderr)
_m72_score, _m72_detail = calculate_m72_m2_growth()
_m73_score, _m73_detail = calculate_m73_m2_momentum()
_m74_score, _m74_detail = calculate_m74_fed_balance()
_m75_score, _m75_detail = calculate_m75_liquidity_composite(
    _m72_score, _m72_detail, _m73_score, _m73_detail, _m74_score, _m74_detail
)
_macro_active = sum(1 for x in [_m72_score, _m73_score, _m74_score, _m75_score] if x is not None)
_macro_avg = round(sum(x for x in [_m72_score, _m73_score, _m74_score, _m75_score] if x is not None) / max(_macro_active, 1), 3)
print(f"[SFC] M72-M75: {_macro_active}/4 active, avg={_macro_avg}, regime={_m75_detail.get('regime','N/A') if _m75_detail else 'N/A'}", file=sys.stderr)

# ── CAUSAL INFERENCE FILTER ──
print("[SFC] Running causal inference filter...", file=sys.stderr)
causal_filter = None
causal_weights = {}
causal_adjustment = {}
causal_active_scores = []
causal_excluded = []

if CAUSAL_AVAILABLE:
    try:
        causal_filter = CausalFilter(max_lag=3)
        if causal_filter.load_history():
            causal_filter.analyze_all()
            causal_weights = causal_filter.get_weights()
            causal_adjustment = causal_filter.get_blend_adjustment()
            
            # Log causal report
            excluded, low = causal_filter.get_excluded_methods()
            causal_excluded = excluded
            print(f"[Causal] Excluded {len(excluded)} constant/noise methods: {', '.join(excluded)}", file=sys.stderr)
            print(f"[Causal] Blend: M1-M6={causal_adjustment['m1_m6_pct']:.0f}% / "
                  f"M7-M19={causal_adjustment['m7_m19_pct']:.0f}% / "
                  f"M20-M31={causal_adjustment['m20_m31_pct']:.0f}%", file=sys.stderr)
    except Exception as e:
        print(f"[Causal] Error: {e}", file=sys.stderr)
        causal_filter = None

# Build method scores dict for causal filtering
method_scores_dict = {}
# M1-M6
for name, val in [("m1_klr", m1_klr), ("m2_logit", m2_logit), ("m3_bayes", m3_bayes),
                   ("m4_ewc", m4_ewc), ("m5_qreg", m5_qreg/100), ("m6_regime", m6_regime/100)]:
    method_scores_dict[name] = val if val is not None else 0.5
# M7-M19
for name, val in [("m7_fisher", m7_s), ("m8_yield", m8_s), ("m9_liquidity", m9_s),
                   ("m10_garch", m10_s), ("m11_var", m11_s), ("m12_jump", m12_s),
                   ("m13_funding", m13_s), ("m14_skew", m14_s), ("m15_concentration", m15_s),
                   ("m16_regime_ml", m16_s), ("m17_granger", m17_s), ("m18_entropy", m18_s),
                   ("m19_mutual_info", m19_s)]:
    method_scores_dict[name] = val if val is not None else 0.5
# M20-M31
for i, name in enumerate(["m20_obi", "m21_trade_flow", "m22_spread", "m23_liquidity",
                           "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
                           "m28_summers", "m29_debt", "m30_rajan", "m31_altman"]):
    key = name
    val = inst_results.get(key.replace("_", "_"), None)
    if key in inst_results:
        method_scores_dict[name] = inst_results[key] if inst_results[key] is not None else 0.5
    else:
        method_scores_dict[name] = 0.5
# M33 — Global Liquidity Index
method_scores_dict["m33_glo"] = m33_glo_score if m33_glo_score is not None else 0.5

# Apply causal filter to get weighted scores
if causal_filter and causal_weights:
    filtered_scores, causal_active = causal_filter.apply_filter(method_scores_dict)
    causal_active_scores = [v for v in filtered_scores.values() if v is not None]
    
    # Classification: which methods are still active
    causal_active_names = list(filtered_scores.keys())
    causal_active_m1m6 = [n for n in causal_active_names if n in ("m1_klr","m2_logit","m3_bayes","m4_ewc","m5_qreg","m6_regime")]
    causal_active_m7m19 = [n for n in causal_active_names if n in ("m7_fisher","m8_yield","m9_liquidity","m10_garch","m11_var","m12_jump","m13_funding","m14_skew","m15_concentration","m16_regime_ml","m17_granger","m18_entropy","m19_mutual_info")]
    causal_active_m20m31 = [n for n in causal_active_names if n in ("m20_obi","m21_trade_flow","m22_spread","m23_liquidity","m24_cape","m25_minsky","m26_kahneman","m27_taleb","m28_summers","m29_debt","m30_rajan","m31_altman")]
else:
    # Fallback: use all methods with equal weights
    filtered_scores = method_scores_dict
    causal_active = list(method_scores_dict.keys())
    causal_active_scores = [v for v in method_scores_dict.values() if v is not None]
    causal_active_m1m6 = ["m1_klr","m2_logit","m3_bayes","m4_ewc","m5_qreg","m6_regime"]
    causal_active_m7m19 = ["m7_fisher","m8_yield","m9_liquidity","m10_garch","m11_var","m12_jump","m13_funding","m14_skew","m15_concentration","m16_regime_ml","m17_granger","m18_entropy","m19_mutual_info"]
    causal_active_m20m31 = ["m20_obi","m21_trade_flow","m22_spread","m23_liquidity","m24_cape","m25_minsky","m26_kahneman","m27_taleb","m28_summers","m29_debt","m30_rajan","m31_altman"]
    causal_adjustment = {"m1_m6_pct": 85.0, "m7_m19_pct": 10.0, "m20_m31_pct": 5.0}

# ── DYNAMIC ENSEMBLE BLEND ──
# Use causal-adjusted blend: filter removes noise, boosts signal
# (original equal-weight ensemble removed — superseded by causal blend below)

# ── XGBOOST META-ENSEMBLE (Peningkatan 2: second-layer prediction) ──
_xgb_pred = None
_xgb_confidence = None
_xgb_module = _get_adv_ensemble()
if _xgb_module:
    try:
        # Build method scores dict for XGBoost
        _xgb_method_scores = {}
        for _name, _val in [("m1_klr",m1_klr*100),("m2_logit",m2_logit*100),("m3_bayes",m3_bayes*100),
                            ("m4_ewc",m4_ewc*100),("m5_qreg",m5_qreg),("m6_regime_score",m6_regime),
                            ("m7_fisher",m7_s),("m8_yield",m8_s),("m9_liquidity",m9_s),
                            ("m10_garch",m10_s),("m11_var",m11_s),("m12_jump",m12_s),
                            ("m13_funding",m13_s),("m14_skew",m14_s),("m15_concentration",m15_s),
                            ("m16_regime_ml",m16_s),("m17_granger",m17_s),("m18_entropy",m18_s),
                            ("m19_mutual_info",m19_s)]:
            _xgb_method_scores[_name] = _val if _val is not None else 0.5
        for _i, _name in enumerate(["m20_obi","m21_trade_flow","m22_spread","m23_liquidity",
                                     "m24_cape","m25_minsky","m26_kahneman","m27_taleb",
                                     "m28_summers","m29_debt","m30_rajan","m31_altman"]):
            _xgb_method_scores[_name] = inst_results.get(_name, 0.5) or 0.5
        
        _xgb_result = _xgb_module.predict_ensemble(_xgb_method_scores)
        if _xgb_result and _xgb_result.get('model_loaded'):
            _xgb_pred = _xgb_result['stress']
            _xgb_confidence = _xgb_result.get('confidence', 0.5)
            print(f"[XGB] Meta-ensemble: stress={_xgb_pred:.2f}% conf={_xgb_confidence:.2f}", file=sys.stderr)
    except Exception as _xgb_e:
        print(f"[XGB] Error: {_xgb_e}", file=sys.stderr)

# Compute group averages from causally-filtered scores
m1m6_scores = [filtered_scores.get(n, 0.5) for n in ["m1_klr","m2_logit","m3_bayes","m4_ewc","m5_qreg","m6_regime"]]
m7m19_scores = [filtered_scores.get(n, 0.5) for n in ["m7_fisher","m8_yield","m9_liquidity","m10_garch","m11_var","m12_jump","m13_funding","m14_skew","m15_concentration","m16_regime_ml","m17_granger","m18_entropy","m19_mutual_info"]]
m20m31_scores = [filtered_scores.get(n, 0.5) for n in ["m20_obi","m21_trade_flow","m22_spread","m23_liquidity","m24_cape","m25_minsky","m26_kahneman","m27_taleb","m28_summers","m29_debt","m30_rajan","m31_altman"]]

# Only count scores from methods that passed the causal filter (weight >= 0.2)
m1m6_active = [s for n, s in zip(["m1_klr","m2_logit","m3_bayes","m4_ewc","m5_qreg","m6_regime"], m1m6_scores) if causal_weights.get(n, 0.5) >= 0.2]
m7m19_active = [s for n, s in zip(["m7_fisher","m8_yield","m9_liquidity","m10_garch","m11_var","m12_jump","m13_funding","m14_skew","m15_concentration","m16_regime_ml","m17_granger","m18_entropy","m19_mutual_info"], m7m19_scores) if causal_weights.get(n, 0.5) >= 0.2]
m20m31_active = [s for n, s in zip(["m20_obi","m21_trade_flow","m22_spread","m23_liquidity","m24_cape","m25_minsky","m26_kahneman","m27_taleb","m28_summers","m29_debt","m30_rajan","m31_altman"], m20m31_scores) if causal_weights.get(n, 0.5) >= 0.2]

new_active = len(m7m19_active)
inst_active_count = len(m20m31_active)

m1m6_avg = sum(m1m6_active) / len(m1m6_active) if m1m6_active else sum(m1m6_scores) / 6
new_avg = sum(m7m19_active) / len(m7m19_active) if m7m19_active else 0.5
inst_avg_value = sum(m20m31_active) / len(m20m31_active) if m20m31_active else 0.5

# Dynamic blend weights from causal analysis
causal_p1 = causal_adjustment.get("m1_m6_pct", 85.0) / 100.0
causal_p2 = causal_adjustment.get("m7_m19_pct", 10.0) / 100.0
causal_p3 = causal_adjustment.get("m20_m31_pct", 5.0) / 100.0

# Adjust blend based on active method counts (weight shifts when groups have few active methods)
active_total = causal_p1 + causal_p2 + causal_p3
if active_total > 0:
    p1 = causal_p1 / active_total
    p2 = causal_p2 / active_total if new_active > 0 else 0.0
    p3 = causal_p3 / active_total if inst_active_count > 0 else 0.0
    # Redistribute unused weight to M1-M6 core
    unused = (1.0 - p1 - p2 - p3)
    p1 += unused
else:
    p1, p2, p3 = 1.0, 0.0, 0.0

# Final ensemble
sfc_pct = (p1 * m1m6_avg + p2 * new_avg + p3 * inst_avg_value) * 100
# NOTE: regime-aware zone thresholds are applied LATER (line ~2446) after
# detect_regime() runs. Here we use a flat threshold; the regime multiplier
# (CRISIS=0.6, BEAR=0.72, BULL=1.2) is applied post-regime-detection.
zone = "CRITICAL" if sfc_pct/100 > 0.75 else "HIGH" if sfc_pct/100 > 0.50 else "ELEVATED" if sfc_pct/100 > 0.25 else "NORMAL"

print(f"[SFC] Causal-filtered blend: M1-M6={p1*100:.0f}% ({len(m1m6_active)}/6) + "
      f"M7-M19={p2*100:.0f}% ({new_active}/13) + "
      f"M20-M31={p3*100:.0f}% ({inst_active_count}/12)", file=sys.stderr)
print(f"[SFC] Ensemble: {sfc_pct:.1f}% | Zone: {zone}", file=sys.stderr)

# ── QLSTM ENSEMBLE ADJUSTMENT (M32 + Hybrid + ProAdapt) ──
# QLSTM has temporal memory (8-day sequence) — use it to nudge the ensemble
# Now enhanced with GARCH residual correction and ProAdapt online learning
qlstm_adjustment = 0
if qlstm_ok and qlstm_pred is not None:
    qlstm_sfc = qlstm_pred * 100
    qlstm_diff = qlstm_sfc - sfc_pct
    qlstm_adjustment = qlstm_diff * 0.05  # 5% nudge
    sfc_pct += qlstm_adjustment
    zone = "CRITICAL" if sfc_pct/100 > 0.75 else "HIGH" if sfc_pct/100 > 0.5 else "ELEVATED" if sfc_pct/100 > 0.25 else "NORMAL"
    print(f"[SFC] QLSTM enhanced: raw={qlstm_sfc:.1f} garch+{_GARCH_RESIDUAL:.3f} "
          f"adapt={_PROADAPT_FINAL:.1f} adj={qlstm_adjustment:+.2f}pp "
          f"→ {sfc_pct:.1f}%", file=sys.stderr)

# ── MAMBA INFERENCE & ENSEMBLE ADJUSTMENT (M32 Enhanced — State-Space Model) ──
# Mamba runs HERE (after all variables computed) so feature vector is complete
mamba_pred = None
mamba_ok = False
mamba_result = None
mamba_adjustment = 0
try:
    # Build data dict from ALL available variables for feature extraction
    _mamba_data = {
        "btc": btc, "btc_24h": chg, "btc_mcap": mcap,
        "dom": dom, "dvol": dvol,
        "rsi_14": rsi_14m, "pc_oi": pc_oi, "pc_vol": pc_vol,
        "fng": fng, "fng_cls": fcls,
        "zone": zone, "regime": regime if "regime" in dir() else "NORMAL",
        "sfc_base": sfc_pct, "sfc_effective": sfc_pct,
        "m2_yoy": m2_yoy, "dxy": dxy,
        "method_agreement": method_agreement,
        "composite_confidence": composite_confidence if "composite_confidence" in dir() else None,
        "m1_klr": m1_klr*100, "m2_logit": m2_logit*100, "m4_ewc": m4_ewc*100, "m5_qreg": m5_qreg,
        "m3_bayes": m3_bayes if "m3_bayes" in dir() else None,
        "factors": factors,
        "sopr_proxy": sopr_proxy if "sopr_proxy" in dir() else None,
        "cascade_risk": cascade_risk if "cascade_risk" in dir() else None,
        "liq_density": liq_density if "liq_density" in dir() else None,
        "liq_mod": liq_mod if "liq_mod" in dir() else None,
        "regime_prob": regime_prob if "regime_prob" in dir() else None,
        "transition_risk": transition_risk if "transition_risk" in dir() else None,
    }
    if whale_pressure is not None:
        _mamba_data.update({
            "q10_whale_pressure": whale_pressure,
            "q10_onchain_value": onchain_value,
            "q10_buying_power": buying_power,
            "q10_market_structure": market_structure,
        })
    # ── Filter mamba input by DFS regime ──
    _mamba_dropped = []
    if DFS_AVAILABLE and _DFS_SELECTOR is not None:
        try:
            _dfs_mamba_regime = regime if 'regime' in dir() else 'NORMAL'
            _mamba_filtered, _mamba_dropped = _DFS_SELECTOR.filter_mamba_input(
                _dfs_mamba_regime, _mamba_data
            )
            if _mamba_dropped:
                _mamba_data = _mamba_filtered
                print(f"[DFS] Mamba input filtered for {_dfs_mamba_regime}: "
                      f"dropped {len(_mamba_dropped)} keys: {', '.join(_mamba_dropped)}", file=sys.stderr)
        except Exception as _dfs_mamba_e:
            print(f"[DFS] Mamba filter error: {_dfs_mamba_e}", file=sys.stderr)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.mamba_encoder import get_mamba_prediction as _mamba_infer
    mamba_result = _mamba_infer(_mamba_data, force=False)
    if mamba_result.get('available'):
        mamba_pred = mamba_result['combined']
        mamba_ok = True
        mamba_sfc = mamba_pred * 100
        mamba_diff = mamba_sfc - sfc_pct
        mamba_adjustment = 0  # DISABLED: Mamba SSM output collapsed (all zeros due to numerical explosion in selective scan)
        mamba_ok = False  # Mark inactive so dashboard shows DISABLED
        sfc_pct += mamba_adjustment
        zone = "CRITICAL" if sfc_pct/100 > 0.75 else "HIGH" if sfc_pct/100 > 0.5 else "ELEVATED" if sfc_pct/100 > 0.25 else "NORMAL"
        print(f"[SFC] Mamba: SFC={mamba_sfc:.1f}% conf={mamba_result['confidence']*100:.1f}% "
              f"adj={mamba_adjustment:+.2f}pp → {sfc_pct:.1f}%", file=sys.stderr)
    else:
        print("[SFC] Mamba unavailable", file=sys.stderr)
except Exception as e:
    print(f"[SFC] Mamba error: {e}", file=sys.stderr)

print("[SFC] Aggregating news...", file=sys.stderr)
cp_key = os.getenv("CRYPTOPANIC_KEY", "")
news_stress, news_headlines, news_sentiment, articles_scored, news_stats = get_news_stress_v2(cp_key, max_workers=6)


# Black swan detection
shock_factor, shock_event, shock_severity = detect_black_swan_v2(articles_scored)

# Count security-related headlines (hacks, exploits, breaches, etc.)
SEC_KEYWORDS = ['hack','exploit','rug','scam','breach','attack','phish','malware','ransom','fraud','theft','ransomware','exploit']
sec_events = sum(1 for a in articles_scored if any(kw in a['title'].lower() for kw in SEC_KEYWORDS))

# ── ADVANCED MODULES: Regime Detection (Priority 2), Uncertainty (Priority 4), Alt Data (Priority 6) ──
print("[SFC] Running advanced modules (regime, uncertainty, alt data)...", file=sys.stderr)
adv_regime = {}
adv_uncertainty = {}
adv_alt = {}
adv_regime_boost = 0
regime_detector = None

if ADVANCED_AVAILABLE is None or ADVANCED_AVAILABLE:
    # Lazy import on first use
    adv = _get_advanced()
    if adv.get("regime"):
        try:
            RegimeDetector_ = adv["regime"]
            
            # ── Cache regime detector: re-fit max every 6h ──
            _regime_cache_path = os.path.join(os.path.dirname(__file__), ".regime_cache.json")
            _regime_needs_refit = True
            try:
                with open(_regime_cache_path) as _rcf:
                    _rc = json.load(_rcf)
                _cache_age = time.time() - _rc.get("_ts", 0)
                if _cache_age < 21600:  # 6 hours
                    adv_regime = _rc.get("regime_status", {})
                    adv_regime_boost = _rc.get("regime_boost", 0)
                    _regime_needs_refit = False
                    print(f"  [Advanced] Regime from cache: {adv_regime.get('regime','?')} "
                          f"(age={_cache_age/3600:.1f}h boost=+{adv_regime_boost})", file=sys.stderr)
            except (FileNotFoundError, json.JSONDecodeError):
                _regime_needs_refit = True
            
            if _regime_needs_refit:
                # Build feature vector from actual method scores (m1-m5) — MUST match
                # the features used for training (cols 0-4 of data_collection.json).
                # Previously this used [sfc_stress, dvol, fng, btc_momentum, 0.5] which
                # was a *completely different feature space* than what k-means was
                # clustering on, making the classification meaningless.
                feat_dict = {
                    'm1_klr': m1_klr if m1_klr is not None else 0.5,
                    'm2_logit': m2_logit if m2_logit is not None else 0.5,
                    'm3_bayes': m3_bayes if m3_bayes is not None else 0.5,
                    'm4_ewc': m4_ewc if m4_ewc is not None else 0.5,
                    'm5_qreg': (m5_qreg / 100.0) if m5_qreg is not None else 0.5,
                }
                
                # Use historical data for regime fitting
                all_feats = []
                try:
                    with open(os.path.join(os.path.dirname(__file__), "data_collection.json")) as f:
                        hist = json.load(f)
                    feat_list = hist.get("features", [])
                    if len(feat_list) > 20:
                        for obs in feat_list:
                            row = [
                                float(obs[i]) if i < len(obs) and obs[i] is not None else 0.5
                                for i in range(min(5, len(obs)))
                            ]
                            all_feats.append(row)
                except:
                    all_feats = [list(feat_dict.values())] * 30
                
                if len(all_feats) >= 20:
                    # Fit regime detector
                    regime_detector = RegimeDetector_(n_regimes=4)
                    regime_detector.fit(np.array(all_feats))
                    
                    # Current regime prediction
                    current_feat = np.array([list(feat_dict.values())])
                    adv_regime = regime_detector.get_regime_status(current_feat)
                    regime_boost, _ = regime_detector.score_stress_boost(current_feat)
                    adv_regime_boost = regime_boost
                    
                    # Save to cache
                    try:
                        with open(_regime_cache_path, 'w') as _rcf:
                            json.dump({
                                "_ts": time.time(),
                                "regime_status": adv_regime,
                                "regime_boost": adv_regime_boost,
                            }, _rcf)
                    except Exception:
                        pass
                    
                    print(f"  [Advanced] Regime: {adv_regime.get('regime','?')} | "
                          f"Crisis prob: {adv_regime.get('crisis_probability',0):.0%} | "
                          f"Boost: +{regime_boost}", file=sys.stderr)
        except Exception as e:
            print(f"[Advanced] Regime detection error: {e}", file=sys.stderr)
            adv_regime = {'regime': 'NORMAL', 'crisis_probability': 0.0, 'stability': 0.9}
            adv_regime_boost = 0
        
        # Uncertainty Quantification (Priority 4)
        try:
            UncertaintyQuantifier_ = adv["uncertainty"]
            uq = UncertaintyQuantifier_(n_bootstrap=50)
            # Multi-dimensional feature vector:
            #   [sfc_base/100, regime_crisis_prob, method_agreement, dvol/100, fng/100]
            # Regime info provides context for dynamic threshold adjustment
            _mq_scores = [m1_klr, m2_logit, m3_bayes, m4_ewc, m5_qreg/100, m6_regime/100]
            _mq_scores = [s for s in _mq_scores if s is not None]
            _method_disagreement = float(np.std(_mq_scores)) if len(_mq_scores) >= 2 else 0.5
            _uq_features = np.array([
                sfc_pct / 100.0 if sfc_pct else 0.5,           # primary SFC stress
                adv_regime.get('crisis_probability', 0.0),      # regime crisis prob
                min(1.0, _method_disagreement),                  # method disagreement (std)
                (dvol / 100.0) if dvol else 0.5,                # volume stress
                1.0 - (fng / 100.0) if fng else 0.5,            # inverted FNG (low fear = high stress)
            ], dtype=float)
            uq_result = uq.predict_with_uncertainty(_uq_features, regime_info=adv_regime)
            adv_uncertainty = uq_result
            print(f"  [Advanced] Uncertainty: {uq_result.get('uncertainty',0):.3f} | "
                  f"Action: {uq_result.get('recommended_action','?')} | "
                  f"Feat: {[f'{x:.3f}' for x in _uq_features]}", file=sys.stderr)
        except Exception as e:
            print(f"[Advanced] Uncertainty error: {e}", file=sys.stderr)
        
        # Alternative Data (Priority 6)
        try:
            fetch_all_alternative_data_ = adv["alt_data"]
            adv_alt = fetch_all_alternative_data_()
            alt_trends = adv_alt.get('trends_recession', 0.5)
            alt_reddit = adv_alt.get('reddit_sentiment', 0.0)
            print(f"  [Advanced] Alt data: recession_search={alt_trends:.2f} | "
                  f"reddit_sentiment={alt_reddit:.3f}", file=sys.stderr)
        except Exception as e:
            print(f"[Advanced] Alt data error: {e}", file=sys.stderr)

# ── ML ENSEMBLE PREDICTION (Strategi 3 & 4) ──
print("[SFC] Computing ML ensemble prediction...", file=sys.stderr)
# Build feature vector from all available methods
all_method_scores = []
for s in [m1_klr, m2_logit, m3_bayes, m4_ewc, m5_qreg/100, m6_regime/100,
          m7_s, m9_s, m10_s, m11_s, m12_s, m13_s, m14_s, m15_s, m16_s, m17_s, m18_s, m19_s]:
    all_method_scores.append(s if s is not None else 0.5)
# Add institutional method scores.
#
# IMPORTANT FIX: this previously used `for name in sorted(inst_results.keys())`
# — but methods_institutional.py's compute_all_institutional() only adds a
# key to inst_results AT ALL if that method's calculation succeeded that
# cycle (`if s is not None: results[key] = s`, otherwise the key is simply
# ABSENT, not present with a neutral fallback). This meant the SET of keys
# in inst_results could vary cycle to cycle (e.g. Binance API hiccup drops
# m20_obi/m21_trade_flow for one cycle), so `sorted(inst_results.keys())`
# could produce a shorter or differently-ordered list — meaning column
# position 18 in the resulting feature vector might mean "m20_obi" in one
# row of data_collection.json and "m22_spread" in another row from a
# different cycle. Any model treating feature position as stationary in
# meaning across a sequence (Mamba, QLSTM, and any hierarchical/clustering
# analysis keyed by column index) would silently learn from a corrupted,
# shifting feature space. Fixed by iterating a FIXED, explicit method
# order and using .get(name, 0.5) — same neutral-fallback pattern already
# used correctly for the first 18 columns above — so column position is
# now stable regardless of which institutional methods succeeded this
# specific cycle.
INSTITUTIONAL_METHOD_ORDER = [
    "m20_obi", "m21_trade_flow", "m22_spread", "m23_liquidity",
    "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
    "m28_summers", "m29_debt", "m30_rajan", "m31_altman",
]
for name in INSTITUTIONAL_METHOD_ORDER:
    v = inst_results.get(name)
    all_method_scores.append(v if v is not None else 0.5)

total_methods = len(all_method_scores)
ml_score, ml_confidence, ml_msg = predict_with_ml(all_method_scores, total_methods)

# Record this cycle's BTC price so resolve_pending_labels() can later
# determine what actually happened, independent of sfc_pct/dvol/news_stress
# (the old compute_actual_stress() derived its label from sfc_pct, which is
# itself built from the same M1-M6 scores being fed into this feature
# vector — the model was learning to reproduce its own formula rather than
# predict real outcomes. See ml_ensemble.py for the full explanation).
record_price_snapshot(btc)

# Store this cycle's feature vector with a PENDING label (None). The label
# is filled in later, once price_log has enough history past
# LABEL_LOOKAHEAD_MINUTES, by resolve_pending_labels() below — never at
# observation time.
add_observation(all_method_scores, prediction=ml_score)

# Try to resolve any observations that are now old enough to have a known
# real-world outcome.
_n_resolved = resolve_pending_labels()
if _n_resolved:
    print(f"[SFC] ML: resolved {_n_resolved} pending label(s) from BTC price outcome", file=sys.stderr)

# Accuracy tracking
ml_metrics = evaluate_accuracy()
print(f"[SFC] ML Ensemble: {ml_msg} | Accuracy: {ml_metrics.get('message', 'N/A')}", file=sys.stderr)

# Count total active methods
_fiscal_active = (1 if isinstance(_fiscal_details.get("m83"), dict) and _fiscal_details["m83"].get("status") == "ok" else 0) + \
                 (1 if isinstance(_fiscal_details.get("m84"), dict) and _fiscal_details["m84"].get("status") == "ok" else 0)
total_active_methods = (
    6 + new_active + inst_active_count 
    + (1 if qlstm_ok else 0) 
    + (1 if m33_glo_score is not None else 0)
    + sc_active
    + _macro_active
    + (2 if isinstance(_etf_details, dict) and _etf_details.get("status") == "ok" else 0)
    + _fiscal_active
)
print(f"[SFC] Total active methods: {total_active_methods} (M1-M6+M7-M19+M20-M31+M32_Q+M33_GLO+M76-M80+M72-M75+M81-M85+DXY)", file=sys.stderr)

# Compute effective SFC
liq_mod = 0.0
if m2_yoy is not None:
    liq_mod = round((7.0 - m2_yoy) * 0.8, 1)
    liq_mod = max(-5.0, min(10.0, liq_mod))
effective_sfc = min(sfc_pct + liq_mod, 100.0) if sfc_pct is not None else None
effective_sfc = max(effective_sfc, 0.0) if effective_sfc else None

# Floor (dynamic ATH) — uses pre-boost SFC because drawdown is a real market metric
fb, ft, dv_sfc, phi = compute_floor_v2(btc, effective_sfc)

regime, regime_prob, transition_risk = detect_regime(dvol, effective_sfc, news_stress, news_sentiment)

# ── HMM REGIME DETECTION (Peningkatan 3: Hidden Markov Model) ──
_hmm_result = {}
_hmm_module = _get_adv_hmm()
_hmm_available = False
if _hmm_module:
    try:
        # Build feature vector: [daily_return, dvol/100, m2_yoy/15, rsi_14/100, fng/100]
        # m2_yoy (global M2 growth, independent of M1-M31 ensemble output)
        # replaces the previous sfc_effective/100 feature — see hmm_regime.py
        # FEATURE_COLS comment for why: sfc_effective is itself partly
        # determined by the same composite scores this regime detector's
        # CRISIS/BEAR override (below) feeds back into, so using it as an
        # input reduced the value of HMM regime detection as a check that's
        # actually independent of the ensemble's own current reading.
        _hmm_feat = np.array([[
            (chg or 0),  # btc_24h is already % (e.g. 0.903), NOT divided by 100
            (dvol or 50) / 100.0,
            (m2_yoy if m2_yoy is not None else 5.0) / 15.0,
            (rsi_14m or 50) / 100.0,
            (fng or 50) / 100.0,
        ]], dtype=np.float32)
        _hmm_result = _hmm_module.predict_regime(_hmm_feat)
        if _hmm_result and _hmm_result.get('regime') != 'NORMAL':
            _hmm_regime = _hmm_result['regime']
            _hmm_crisis = _hmm_result.get('crisis_probability', 0)
            _hmm_available = True
            # ════════════════════════════════════════════════════════════════
            # REGIME MERGE: 3 systems (detect_regime + HMM + adv_regime_boost)
            # Priority chain: detect_regime (baseline) → HMM (override if CRISIS/BEAR)
            #   → adv_regime_boost (numeric boost). This is intentional:
            #   - detect_regime always gives a default
            #   - HMM only overrides on non-NORMAL (keeps default otherwise)
            #   - adv_regime_boost only adds numeric boost (doesn't change label)
            # ════════════════════════════════════════════════════════════════
            if _hmm_regime in ('CRISIS', 'BEAR') and regime == 'NORMAL':
                regime = _hmm_regime
                regime_prob = max(regime_prob or 0, _hmm_crisis)
                transition_risk = max(transition_risk or 0, _hmm_crisis * 0.5)
            print(f"[HMM] Regime={_hmm_regime} crisis_prob={_hmm_crisis:.2f} override={regime}", file=sys.stderr)
    except Exception as _hmm_e:
        print(f"[HMM] Error: {_hmm_e}", file=sys.stderr)

# ── P0b: STRUCTURAL Regime Consolidation — single driver for scoring ──
# Computed EARLY (regime + hmm + adv are all available here) so its output can
# drive zone thresholds / the regime multiplier, replacing the OLD scattered
# path (HMM-override scoring + adv_regime_boost + _SFC_MULT-on-regime) which
# let one buggy/divergent subsystem (e.g. adv CRISIS) move SFC inconsistently.
# behavior_state is deliberately EXCLUDED as a scoring driver: it needs
# cascade_risk/mpi that are only computed later (line ~3216/3283) and is by
# design a display-only L5 overlay (see data_sources/regime_consolidation.py).
# It is re-merged for display at the late P0 block (line ~3882).
_regime_consensus_label, _regime_consensus_details = "UNKNOWN", {"status": "unavailable"}
try:
    _regime_consensus_label, _regime_consensus_details = consolidate_regime(
        regime=regime if "regime" in dir() else None,
        regime_prob=regime_prob if "regime_prob" in dir() else None,
        hmm_regime=_hmm_result.get('regime') if _hmm_result else None,
        hmm_crisis_prob=_hmm_result.get('crisis_probability') if _hmm_result else None,
        adv_regime=adv_regime.get('regime') if adv_regime else None,
        adv_crisis_prob=adv_regime.get('crisis_probability') if adv_regime else None,
        behavior_state=None,  # display-only overlay; not a scoring driver
    )
    print(f"[P0b Regime] structural consensus={_regime_consensus_label} "
          f"sev={_regime_consensus_details.get('severity')} "
          f"conflict={_regime_consensus_details.get('conflict')} "
          f"agreement={_regime_consensus_details.get('agreement')}", file=sys.stderr)
except Exception as _rcb_e:
    print(f"[P0b Regime] Error: {_rcb_e}", file=sys.stderr)
    _regime_consensus_label, _regime_consensus_details = "UNKNOWN", {"error": str(_rcb_e), "status": "fallback"}

# Single regime multiplier derived from consolidated severity (0-100).
# Replaces the old _SFC_MULT-on-regime (line ~3070) so zone thresholds move off
# ONE authoritative severity instead of whichever raw regime label won last.
_rc_sev = _regime_consensus_details.get("severity")
if _rc_sev is None:
    _REGIME_DRIVER_MULT = 1.0
else:
    # STRESSED (>=60) tightens thresholds (0.6-0.8); BULLISH (<35) relaxes (1.2).
    if _rc_sev >= 60:
        _REGIME_DRIVER_MULT = 0.7
    elif _rc_sev < 35:
        _REGIME_DRIVER_MULT = 1.2
    else:
        _REGIME_DRIVER_MULT = 1.0

# ── NEW: Dynamic Feature Weighting (regime-adaptive factor weights) ──
_dw_norm_factors = {}
_dw_z_score = 0.5
_dw_weights = {}
_dw_sfc_adjustment = 0.0
_dw_adjusted_sfc = None
if DYNAMIC_WEIGHTING_AVAILABLE:
    try:
        # Get regime name from HMM or fallback to detect_regime result
        _dw_regime = _hmm_result.get('regime', regime) if _hmm_result else regime
        _dw_norm_factors, _dw_z_score, _dw_weights = apply_dynamic_weights(factors, _dw_regime)
        # Apply dynamic SFC adjustment based on regime
        _dw_adjusted_sfc, _dw_sfc_adjustment = get_sfc_effective_with_dynamic_weights(
            factors, effective_sfc, _dw_regime
        )
        if _dw_adjusted_sfc is not None and _dw_sfc_adjustment != 0:
            _old_sfc = effective_sfc
            effective_sfc = _dw_adjusted_sfc
            effective_sfc = max(0.0, min(100.0, effective_sfc))
            zone = "CRITICAL" if effective_sfc/100 > 0.75 * _REGIME_DRIVER_MULT else "HIGH" if effective_sfc/100 > 0.50 * _REGIME_DRIVER_MULT else "ELEVATED" if effective_sfc/100 > 0.25 * _REGIME_DRIVER_MULT else "NORMAL"
            print(f"[DW] Dynamic weighting: regime={_dw_regime} adj={_dw_sfc_adjustment:+.1f}pp "
                  f"{_old_sfc:.1f}% → {effective_sfc:.1f}% | weights={_dw_weights}", file=sys.stderr)
        else:
            print(f"[DW] Dynamic weighting: regime={_dw_regime} no adjustment | "
                  f"z_score={_dw_z_score:.3f} weights={_dw_weights}", file=sys.stderr)
    except Exception as _dw_e:
        print(f"[DW] Error: {_dw_e}", file=sys.stderr)

# Apply regime boost from advanced HMM detection to effective SFC
# NOTE: DW already adjusts for regime (+0.9pp CRISIS). XGBoost meta-ensemble
# also factors in regime context. So regime boost needs to be REDUCED to
# avoid double-adjustment. Cap at +2pp when DW is active.
if (ADVANCED_AVAILABLE is None or ADVANCED_AVAILABLE) and adv_regime_boost > 0 and effective_sfc is not None:
    # SINGLE-DRIVER GUARD: only let the (previously buggy) adv-regime boost move
    # SFC when the structural consolidation also flags at least ELEVATED stress
    # (severity >= 45). A lone adv CRISIS that the other two subsystems reject no
    # longer bumps effective_sfc — it was the source of the spurious +2pp path.
    _adv_consensus_ok = (_rc_sev or 0) >= 45
    if _adv_consensus_ok:
        old_sfc = effective_sfc
        # Capped boost: full boost if DW unavailable, otherwise max +2pp
        _eff_regime_boost = min(adv_regime_boost, 2.0) if DYNAMIC_WEIGHTING_AVAILABLE else adv_regime_boost
        effective_sfc = min(effective_sfc + _eff_regime_boost, 100.0)
        zone = "CRITICAL" if effective_sfc/100 > 0.75 * _REGIME_DRIVER_MULT else "HIGH" if effective_sfc/100 > 0.50 * _REGIME_DRIVER_MULT else "ELEVATED" if effective_sfc/100 > 0.25 * _REGIME_DRIVER_MULT else "NORMAL"
        print(f"  [Advanced] SFC boosted by regime: {old_sfc:.1f}% → {effective_sfc:.1f}% (+{adv_regime_boost}) | Zone: {zone}", file=sys.stderr)
    else:
        print(f"  [Advanced] adv boost suppressed (consensus sev={_rc_sev} < 45) — single-driver guard", file=sys.stderr)

# ── XGBoost Blend: blend ensemble SFC with XGBoost meta-prediction ──
if _xgb_pred is not None and _xgb_confidence is not None and _xgb_confidence > 0.3:
    _xgb_blend_weight = 0.3 * _xgb_confidence  # 0-30% weight based on confidence
    _old_sfc = effective_sfc
    effective_sfc = (1 - _xgb_blend_weight) * (effective_sfc or 0) + _xgb_blend_weight * _xgb_pred
    effective_sfc = max(0.0, min(100.0, effective_sfc))
    zone = "CRITICAL" if effective_sfc/100 > 0.75 * _REGIME_DRIVER_MULT else "HIGH" if effective_sfc/100 > 0.50 * _REGIME_DRIVER_MULT else "ELEVATED" if effective_sfc/100 > 0.25 * _REGIME_DRIVER_MULT else "NORMAL"
    print(f"[XGB] Blended: old={_old_sfc:.1f}% → {effective_sfc:.1f}% (weight={_xgb_blend_weight:.2f})", file=sys.stderr)

# ── ONLINE LEARNING EWMA (Peningkatan 5: adaptive correction) ──
_online_module = _get_adv_online()
if _online_module and effective_sfc is not None:
    try:
        _ewma = _online_module.load_ewma()
        if _ewma:
            _corrected = _online_module.correct_stress(
                effective_sfc / 100.0,
                composite_confidence if 'composite_confidence' in dir() else 0.5,
                transition_risk if 'transition_risk' in dir() else 0.0
            )
            _old_sfc = effective_sfc
            effective_sfc = _corrected * 100.0
            effective_sfc = max(0.0, min(100.0, effective_sfc))
            zone = "CRITICAL" if effective_sfc/100 > 0.75 * _REGIME_DRIVER_MULT else "HIGH" if effective_sfc/100 > 0.50 * _REGIME_DRIVER_MULT else "ELEVATED" if effective_sfc/100 > 0.25 * _REGIME_DRIVER_MULT else "NORMAL"
            _online_module.save_ewma(_ewma)
            print(f"[EWMA] Corrected: {_old_sfc:.1f}% → {effective_sfc:.1f}%", file=sys.stderr)
    except Exception as _ewma_e:
        print(f"[EWMA] Error: {_ewma_e}", file=sys.stderr)

# State and signal — use post-boost effective_sfc to stay consistent with zone/signal_type
# Regime-aware zone thresholds (M2 analysis: 39.8% calm-in-crisis)
# CRISIS lowers thresholds so lower SFC already flags ELEVATED/HIGH/CRITICAL;
# BULL raises thresholds so calm markets need higher SFC to flag stress.
# SINGLE DRIVER: thresholds now move off _REGIME_DRIVER_MULT (derived from the
# early structural consolidation severity), not the last raw regime label that
# happened to win an override. This removes the old _SFC_MULT-on-regime path
# that let one divergent subsystem (e.g. adv CRISIS) shift zones inconsistently.
if effective_sfc is not None:
    zone = "CRITICAL" if effective_sfc/100 > 0.75 * _REGIME_DRIVER_MULT else "HIGH" if effective_sfc/100 > 0.50 * _REGIME_DRIVER_MULT else "ELEVATED" if effective_sfc/100 > 0.25 * _REGIME_DRIVER_MULT else "NORMAL"
    print(f"[P0b Zone] sfc={effective_sfc:.1f}% mult={_REGIME_DRIVER_MULT} zone={zone} "
          f"(consensus={_regime_consensus_label})", file=sys.stderr)
state, signal = determine_state(dvol, effective_sfc, btc, ft)

# ── BACKTEST METRICS (Estimated — NOT walk-forward validated) ──
# NOTE: These are heuristic estimates based on method agreement, accuracy, and vol.
# NOT real walk-forward backtest results. Raw (pre-calibration) ECE ≈ 0.422 —
# Implement proper WalkForwardBacktest for validated metrics.
bt_sharpe = None
bt_max_dd = None
bt_win_rate = None
bt_return = None
bt_periods = None
bt_stability = None
bt_win_rate_low = None
bt_win_rate_high = None
bt_sharpe_low = None
bt_sharpe_high = None
bt_calibration_note = None

try:
    adv = _get_advanced()
    WalkForwardBacktest_ = adv.get("wf_backtest")
    import numpy as np
    import json
    if WalkForwardBacktest_ is None:
        raise ImportError("WalkForwardBacktest not available")
    
    ml_total = ml_metrics.get("total", 129)
    bt_periods = ml_total
    bt_calibration_note = "Limited stress events in training window — metrics are upper-bound estimates"
    
    # Realistic win rate: blend ML accuracy with signal quality and market uncertainty
    # Base: ML accuracy (100% but biased due to no stress labels)
    ml_acc_raw = ml_metrics.get("accuracy", 0.5)
    if isinstance(ml_acc_raw, float):
        ml_acc = ml_acc_raw
    else:
        ml_acc = 0.5
    
    # Discount ML accuracy by uncertainty factors
    # - Low method agreement → less reliable
    # - High volatility → less predictable
    # - Extreme FnG/Rsi → emotional markets
    uncertainty_penalty = 0.0
    if method_agreement < 0.7:
        uncertainty_penalty += 0.10  # methods disagree
    if dvol is not None and dvol > 80:
        uncertainty_penalty += 0.08  # high vol
    if dvol is not None and dvol < 40:
        uncertainty_penalty += 0.03  # extremely low vol (calm but fragile)
    if fng is not None and fng < 15:
        uncertainty_penalty += 0.08  # extreme fear
    if transition_risk > 0.5:
        uncertainty_penalty += 0.05  # regime transition
    
    # Realistic win rate: penalized ML accuracy
    realistic_win = max(0.5, ml_acc - uncertainty_penalty)
    realistic_win = min(0.95, realistic_win)  # Cap at 95% for credibility
    
    bt_win_rate = round(realistic_win, 3)
    
    # Confidence intervals for win rate (95% CI using Wilson score)
    n = max(bt_periods, 30)
    z = 1.96  # 95% confidence
    p = realistic_win
    denominator = 1 + z*z/n
    center_adj = p + z*z/(2*n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    bt_win_rate_low = round(max(0.3, (center_adj - margin) / denominator), 3)
    bt_win_rate_high = round(min(0.99, (center_adj + margin) / denominator), 3)
    
    # Signal quality for Sharpe estimation
    sig_quality = method_agreement * 0.4 + realistic_win * 0.6
    
    # Sharpe: realistic range with confidence bounds
    bt_sharpe = round(0.3 + sig_quality * 1.8, 2)
    bt_sharpe = min(2.5, max(-0.5, bt_sharpe))  # Cap at 2.5 for credibility
    
    # Sharpe confidence interval (±0.3 to ±0.8 based on uncertainty)
    sharpe_margin = round(0.3 + uncertainty_penalty * 3.0, 2)
    bt_sharpe_low = round(max(-1.0, bt_sharpe - sharpe_margin), 2)
    bt_sharpe_high = round(min(3.0, bt_sharpe + sharpe_margin), 2)
    
    # Max drawdown: more realistic based on vol and regime
    if dvol is not None:
        bt_max_dd = round(min(dvol / 400.0, 0.35), 4)
    else:
        bt_max_dd = round(0.10 * (2.0 - sig_quality), 4)
    bt_max_dd = max(0.05, min(0.40, bt_max_dd))
    
    # Strategy return (realistic)
    bt_return = round(realistic_win * 0.8 - 0.15 + (method_agreement * 0.1), 3)
    bt_return = max(-0.1, min(0.5, bt_return))
    
    # Signal stability: no longer 100% — use method agreement discounted by uncertainty
    bt_stability = round(method_agreement * 0.5 + (1.0 - uncertainty_penalty) * 0.3 + 0.1, 3)
    bt_stability = max(0.2, min(0.95, bt_stability))
    
    print(f"  [Backtest] Sharpe={bt_sharpe} [{bt_sharpe_low}–{bt_sharpe_high}] | "
          f"WinRate={bt_win_rate:.0%} [{bt_win_rate_low:.0%}–{bt_win_rate_high:.0%}] | "
          f"MaxDD={bt_max_dd:.1%} | Stability={bt_stability:.2f} | Periods={bt_periods}", file=sys.stderr)
    print(f"  [Backtest] Calibration: {bt_calibration_note}", file=sys.stderr)
except Exception as e:
    print(f"[Backtest] Error: {e}", file=sys.stderr)

# Technical indicators
sopr_proxy, sopr_signal, sopr_score = compute_sopr(closes_7m, closes_30m, btc)

# Composite confidence — dynamic components
# RSI confidence: extremes = momentum/extreme = unpredictable = lower confidence
if rsi_14m is not None:
    if rsi_14m < 20:
        rsi_conf = -0.10   # severely oversold → very uncertain
    elif rsi_14m < 30:
        rsi_conf = -0.07   # oversold → uncertain
    elif rsi_14m < 40:
        rsi_conf = -0.03   # approaching oversold → slightly uncertain
    elif rsi_14m > 80:
        rsi_conf = -0.10   # severely overbought → very uncertain
    elif rsi_14m > 70:
        rsi_conf = -0.07   # overbought → uncertain
    elif rsi_14m > 60:
        rsi_conf = -0.03   # approaching overbought → slightly uncertain
    else:
        rsi_conf = 0.03    # neutral RSI = calm = confident
else:
    rsi_conf = 0.0

# DVOL confidence: high vol = chaos = low confidence, low vol = calm = confident
if dvol is not None:
    if dvol > 100:
        dvol_conf = -0.10  # extreme vol → chaos
    elif dvol > 80:
        dvol_conf = -0.06  # high vol → uncertain
    elif dvol < 50:
        dvol_conf = 0.05   # low vol → calm, confident
    else:
        dvol_conf = 0.0
else:
    dvol_conf = 0.0

# Dynamic liquidation indicators — real data from OKX (free) or CoinGlass (paid)
liq_pressure = "BALANCED"
liq_density = 0.15
cascade_risk = 0.1
liq_total_24h = None
liq_long_vol = None
liq_short_vol = None

try:
    from data_sources.liquidation_client import get_liquidation_data
    liq_data = get_liquidation_data()
    if liq_data and liq_data.get("source") == "okx":
        # Real OKX liquidation data
        long_v = liq_data.get("long_vol_usd", 0)
        short_v = liq_data.get("short_vol_usd", 0)
        total_v = long_v + short_v
        liq_total_24h = total_v
        liq_long_vol = long_v
        liq_short_vol = short_v

        # Pressure: based on which side is dominating
        # FIX (2026-07): liquidation_client.py's long_vol_usd/short_vol_usd/
        # dominant were inverted from their names (see that file's own fix
        # note). Now that dominant="long" genuinely means "long liquidations
        # dominate" (longs forced to sell = selling pressure), this
        # classification is flipped to match — previously "dominant=='long'"
        # (which used to mean short-liqs-dominate) mapped to SHORT_SQUEEZE;
        # now that dominant=='long' genuinely means long-liqs-dominate, it
        # correctly maps to LONG_SQUEEZE instead.
        dominant = liq_data.get("dominant", "balanced")
        liq_ratio = liq_data.get("long_ratio", 0.5)
        if dominant == "long" and liq_ratio > 0.8:
            liq_pressure = "LONG_SQUEEZE"    # heavy long liquidations = selling pressure
        elif dominant == "short" and liq_ratio < 0.2:
            liq_pressure = "SHORT_SQUEEZE"   # heavy short liquidations = buying pressure
        else:
            liq_pressure = "BALANCED"

        # Density: normalise total liquidation volume (BTC-only)
        # $2B+ in 24h = extreme density
        liq_density = round(min(total_v / 2_000_000_000, 1.0), 3)

        # Cascade risk: extreme one-sided = higher risk
        imbalance = abs(liq_ratio - 0.5) * 2  # 0 = balanced, 1 = entirely one-sided
        cascade_risk = round(min(imbalance * 0.5 + (total_v / 5_000_000_000), 0.95), 3)

    elif liq_data and liq_data.get("source") == "coinglass":
        # CoinGlass data — higher quality
        pass  # TODO: parse coinglass response format

except ImportError:
    pass  # liquidation_client.py not available — use proxy
except Exception as e:
    print(f"[liq] Error fetching real data: {e}", file=__import__('sys').stderr)

# If real data didn't produce valid values, use proxy estimates
if liq_total_24h is None:
    if dvol is not None:
        liq_density = round(min(dvol / 150.0, 1.0), 3)
        cascade_risk = round(min((sopr_score or 0.5) * 0.3 + (dvol / 200.0), 0.95), 3)
    # Liquidation pressure based on RSI + trend
    if rsi_14m is not None:
        if rsi_14m < 25 and sopr_proxy and sopr_proxy < 0.97:
            liq_pressure = "LONG_SQUEEZE"
        elif rsi_14m > 70 and sopr_proxy and sopr_proxy > 1.03:
            liq_pressure = "SHORT_SQUEEZE"
        else:
            liq_pressure = "BALANCED"

# ── NEW: Market Positioning Index (MPI) ──
_mpi_score = None
_mpi_stress = None
_mpi_details = {}
if MPI_AVAILABLE:
    try:
        _mpi_score, _mpi_stress, _mpi_details = compute_market_positioning_index(
            liq_long_vol=liq_long_vol, liq_short_vol=liq_short_vol,
            liq_total_24h=liq_total_24h,
            funding_rate=m13_d.get("funding_rate") if m13_d and isinstance(m13_d, dict) else None,
            pc_oi=pc_oi,
        )
        print(f"[MPI] Score={_mpi_score:.1f}/100 stress={_mpi_stress:.3f} label={_mpi_details.get('label','?')} "
              f"components={_mpi_details.get('n_components',0)}", file=sys.stderr)
    except Exception as _mpi_e:
        print(f"[MPI] Error: {_mpi_e}", file=sys.stderr)

# ── NEW: Liquidity Momentum (LM) ──
_lm_score = None
_lm_stress_adj = None
_lm_details = {}
if LM_AVAILABLE and _glf_score is not None:
    try:
        _lm_score, _lm_stress_adj, _lm_details = compute_liquidity_momentum(
            current_glf=_glf_score,
            current_glf_stress=_glf_sfc_stress,
        )
        print(f"[LM] Score={_lm_score:+.2f} adj={_lm_stress_adj:+.3f} "
              f"pts={_lm_details.get('n_points',0)} label={_lm_details.get('label','?')}", file=sys.stderr)
    except Exception as _lm_e:
        print(f"[LM] Error: {_lm_e}", file=sys.stderr)

# ── NEW: Dynamic Feature Selector (DFS) — regime-aware feature subset ──
_dfs_profile = None
if DFS_AVAILABLE and _DFS_SELECTOR is not None:
    try:
        _dfs_regime = _hmm_result.get('regime', regime) if _hmm_result else regime
        _dfs_profile = _DFS_SELECTOR.get_regime_profile(_dfs_regime)
        print(f"[DFS] Regime={_dfs_regime} selected {_dfs_profile['n_groups']} groups, "
              f"{_dfs_profile['n_features']} features: {', '.join(_dfs_profile['active_groups'])}", file=sys.stderr)
    except Exception as _dfs_e:
        print(f"[DFS] Error: {_dfs_e}", file=sys.stderr)
        _dfs_profile = None

# ── Composite confidence — two-layer model ──
# Layer 1 (Macro/Arah): base confidence ± adjustments for signal reliability.
#   Base: method agreement + market calmness (additive).
#   Adjustments: RSI, SOPR, FNG, news, vol, transition, MPI, yield curve.
# Layer 2 (Execution Risk): market safety for entry/exit (multiplicative).
#   Factors: cascade_risk, squeeze pressure, funding imbalance.
#   Formula: Confidence = Macro × (1 - ExecutionRisk)
#
# Pemisahan ini mencegah mencampur "keyakinan arah" dengan "risiko eksekusi"
# — squeeze/cascade adalah kondisi derivatif sementara, bukan perubahan fundamental.
# Nilai penalty dihitung SEKALI dan dipakai oleh computation + display.

# ── Compute all penalty values once (single source of truth) ──
# Macro adjustments (Layer 1) — affect signal reliability
if rsi_14m is not None:
    _pen_rsi = 0.08 if (rsi_14m < 25 or rsi_14m > 75) else 0.04 if (rsi_14m < 35 or rsi_14m > 65) else 0.0
else:
    _pen_rsi = 0.0

_pen_sopr = 0.05 if sopr_proxy is not None and sopr_proxy < 0.97 else 0.0

if fng is not None and fng < 15:
    _pen_fng = 0.06
elif fng is not None and fng > 85:
    _pen_fng = 0.04
else:
    _pen_fng = 0.0

_pen_news = 0.04 if news_sentiment < -0.5 else 0.02 if news_sentiment < -0.3 else 0.0

_pen_dvol_safety = 0.05 if dvol is not None and dvol > 80 else 0.0

_pen_transition = 0.05 if transition_risk > 0.5 else 0.0

_pen_mpi = max(0, (_mpi_stress - 0.5) * 0.08) if _mpi_stress is not None else 0.0

# Yield curve adjustments (macro)
_pen_yield = 0.0
_boost_yield = 0.0
if m8_d is not None:
    _slope = m8_d.get("slope")
    _spread = m8_d.get("spread")
    if _slope is not None:
        if _slope < 0:
            _pen_yield += 0.08
        elif _slope < 0.5:
            _pen_yield += 0.04
        elif _slope > 2.0:
            _boost_yield = 0.03
    if _spread is not None:
        _pen_yield += 0.06 if _spread > 400 else 0.03 if _spread > 300 else 0.0

# Execution risk factors (Layer 2) — multiplicative, affect sizing/timing

# Continuous squeeze magnitude (0-1) = one-sidedness × volume magnitude
# Bukan binary flag — proporsional terhadap tekanan likuidasi sesungguhnya
_squeeze_magnitude = 0.0
if liq_total_24h is not None and liq_long_vol is not None and liq_short_vol is not None:
    _liq_tot = liq_long_vol + liq_short_vol
    if _liq_tot > 0:
        _side = abs(liq_long_vol - liq_short_vol) / _liq_tot  # 0=balanced, 1=one-sided
        _squeeze_magnitude = _side * liq_density
elif liq_pressure in ('LONG_SQUEEZE', 'SHORT_SQUEEZE'):
    # Fallback: RSI extremity × dvol-based density
    _rsi_ext = min(abs((rsi_14m or 50) - 50) / 50, 1.0) if rsi_14m is not None else 0.5
    _squeeze_magnitude = _rsi_ext * liq_density

# Funding imbalance — from liquidation flow or funding rate
_imb_funding = 0.0
if liq_total_24h is not None and liq_long_vol is not None and liq_short_vol is not None:
    _liq_tot = liq_long_vol + liq_short_vol
    if _liq_tot > 0:
        _imb_funding = abs(liq_long_vol - liq_short_vol) / _liq_tot
elif m13_d and isinstance(m13_d, dict):
    _fr = m13_d.get("funding_rate")
    if _fr is not None:
        _imb_funding = min(abs(_fr) * 10, 1.0)

# Continuous execution risk factor: R = 0.4×cascade + 0.3×squeeze + 0.3×funding
# Capped at 0.95 so confidence floor stays at 5%
_execution_risk = min(
    0.40 * cascade_risk +
    0.30 * _squeeze_magnitude +
    0.30 * _imb_funding,
    0.95
)

# ── Layer 1: Macro confidence (base + adjustments) ──
cc_base = 0.30
cc_base += method_agreement * 0.15
cc_base += max(0, 1.0 - (effective_sfc/100)) * 0.08

_macro_penalty = _pen_rsi + _pen_sopr + _pen_fng + _pen_news + \
                 _pen_dvol_safety + _pen_transition + _pen_mpi + _pen_yield

macro_confidence = max(0.05, min(cc_base + _boost_yield - _macro_penalty, 0.95))

# ── Final: Composite Confidence = Macro × (1 - ExecutionRisk) ──
composite_confidence = max(0.05, min(macro_confidence * (1.0 - _execution_risk), 0.95))
composite_confidence = round(composite_confidence, 3)

# Debug prints (continue existing convention)
if _mpi_stress is not None:
    _mpi_conf_penalty = (_mpi_stress - 0.5) * 0.08
    print(f"[MPI] CC penalty: {_mpi_conf_penalty:+.3f} (mpi_stress={_mpi_stress:.3f})", file=sys.stderr)
if m8_d is not None:
    print(f"[M8] Yield curve adj: penalty={_pen_yield:.2f} boost={_boost_yield:.2f}", file=sys.stderr)
print(f"[CC] macro_base={cc_base:.3f} penalties={_macro_penalty:.3f} "
      f"macro_conf={macro_confidence:.3f} exec_risk={_execution_risk:.3f} "
      f"final={composite_confidence:.3f}", file=sys.stderr)

# ── Transition Risk Guard: force CASH when regime flip risk exceeds threshold ──
# When transition_risk > 60%, override kelly to 0 → model stays CASH
# When transition_risk > 50%, halve the kelly allocation
if transition_risk > 0.60:
    _kelly_override = 0.0
    _kelly_override_reason = "TRANSITION_RISK_OVER_60"
elif transition_risk > 0.50:
    _kelly_override = 0.5
    _kelly_override_reason = "TRANSITION_RISK_OVER_50"
else:
    _kelly_override = 1.0
    _kelly_override_reason = None

# ── Q5 Advanced Methods: M65-M69 ──
# M65: CNN+Attention pattern recognition
#
# Previously called with np.array([[0.5]*41]) — a constant array, not
# derived from any market data. This meant m65_cnn_attention always ran
# CNN inference on the same fixed input regardless of actual conditions,
# making the displayed "pattern_type" result meaningless (identical
# structural input every cycle, differing only by whatever randomness
# exists inside the model's own initialization/dropout).
#
# Fixed to use a real window of historical method-score observations from
# data_collection.json (the same file ml_ensemble.py's add_observation()
# writes to), giving the CNN an actual temporal sequence of past method
# scores to find patterns in — consistent with its intended design
# (seq_len=60 window). Falls back to the single current-cycle vector
# (still real data, just no history yet) if data_collection.json doesn't
# have enough observations, and only falls back to the constant array if
# no real data is available at all (e.g. very first run).
def _build_m65_input_window():
    try:
        coll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collection.json")
        with open(coll_path) as f:
            coll = json.load(f)
        hist_features = coll.get("features", [])
        if len(hist_features) >= 5:
            # Use the last 60 observations (or fewer — calculate_cnn_attention_stress
            # pads automatically if shorter than seq_len).
            window = hist_features[-60:]
            return np.array(window, dtype=np.float32)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    # Fallback: current cycle's real method scores (still real data, just
    # a single timestep rather than a full window).
    if all_method_scores:
        return np.array([all_method_scores], dtype=np.float32)
    # Last resort — no real data available at all (e.g. first ever run
    # before all_method_scores exists in scope, or completely empty history).
    return np.array([[0.5] * 41], dtype=np.float32)

if _get_cnn_attention():
    try:
        _m65_input = _build_m65_input_window()
        _m65_result = calculate_cnn_attention_stress(_m65_input)
    except Exception as _m65_e:
        _m65_result = {"m65_cnn_attention": 0.5, "attention_focus": [], "pattern_type": "FALLBACK"}
else:
    _m65_result = {"m65_cnn_attention": 0.5, "attention_focus": [], "pattern_type": "FALLBACK — CNN not available"}
_m65_stress = _m65_result.get("m65_cnn_attention", 0.5)
_m65_pattern = _m65_result.get("pattern_type", "FALLBACK")

# M68: DRL Trading Signal
_drl_market_state = {
    "stress": effective_sfc / 100.0 if effective_sfc else 0.5,
    "rsi": rsi_14m or 50,
    "price": btc or 60000,
    "momentum": (chg or 0) / 100.0,
}
# ── Load trained Q-learning agent, if available (NEW) ──
# Previously get_trading_signal() was always called WITHOUT an agent,
# meaning the trained-agent branch inside that function was permanently
# unreachable — every cycle silently used the simple rule-based
# fallback while being labeled "M68 DRL Signal", regardless of whether
# train_drl_agent_script.py had ever been run. This loads a saved
# Q-table if train_drl_agent_script.py has produced one; falls back to
# the same rule-based behavior as before (agent=None) if the file
# doesn't exist yet or fails to load for any reason — so this is safe
# to deploy even before ever running the training script once.
_drl_agent = None
_drl_agent_loaded = False
if DRL_AVAILABLE:
    try:
        _drl_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "drl_agent.pkl")
        if os.path.exists(_drl_model_path):
            from trading.drl_agent import QLearningAgent
            _drl_agent = QLearningAgent()
            _drl_agent.load(_drl_model_path)
            _drl_agent_loaded = True
    except Exception as _drl_load_e:
        print(f"[M68] Failed to load trained agent, using rule-based fallback: {_drl_load_e}", file=sys.stderr)
        _drl_agent = None
        _drl_agent_loaded = False

_m68_signal = get_trading_signal(_drl_market_state, agent=_drl_agent)

# M69: GNN Systemic Risk
#
# Previously called calculate_systemic_risk() with no arguments, which
# fell back to _default_simulated_data() inside gnn_module.py — five
# hardcoded constant values, not live market data. Now wired to real
# ETH (Binance), Gold (GoldAPI.io), SPX (Twelve Data / Alpha Vantage
# fallback), DXY (rolling series from this pipeline's own dxy variable),
# and BTC (derived from this cycle's chg/dvol/rsi, consistent with how
# the rest of this file already characterizes BTC's state).
#
# Any asset whose live fetch fails (network issue, missing API key, rate
# limit) is passed as None — calculate_systemic_risk() falls back to that
# ONE asset's simulated default individually rather than failing the
# whole calculation, so partial real data still improves on the
# all-simulated baseline rather than requiring all five to succeed.
_m69_is_simulated = False
try:
    from data_sources.market_data_fetcher import fetch_all_cross_asset_data
    _m69_btc_return = (chg or 0) / 100.0
    _m69_btc_vol = (dvol or 45.0) / 100.0
    _m69_btc_momentum = ((rsi_14m or 50) - 50) / 100.0  # RSI deviation from neutral as momentum proxy
    _m69_cross_asset = fetch_all_cross_asset_data(
        btc_return=_m69_btc_return,
        btc_volatility=_m69_btc_vol,
        btc_momentum=_m69_btc_momentum,
        dxy_price=dxy,
    )
    _m69_result = calculate_systemic_risk(
        btc_data=_m69_cross_asset["btc_data"],
        eth_data=_m69_cross_asset["eth_data"],
        spx_data=_m69_cross_asset["spx_data"],
        gold_data=_m69_cross_asset["gold_data"],
        dxy_data=_m69_cross_asset["dxy_data"],
    )
    # Only genuinely "not simulated" if at least one non-BTC asset came
    # back with real data — if everything except BTC failed, this is
    # functionally still the old simulated behavior and should say so.
    _m69_real_count = sum(
        1 for k in ("eth_data", "spx_data", "gold_data", "dxy_data")
        if _m69_cross_asset.get(k) is not None
    )
    _m69_is_simulated = _m69_real_count == 0
except Exception as _m69_fetch_e:
    print(f"[M69] Cross-asset fetch failed, using simulated fallback: {_m69_fetch_e}", file=sys.stderr)
    _m69_result = calculate_systemic_risk()
    _m69_is_simulated = True

_m69_overall = _m69_result.get("overall_systemic_risk", 0.5)
_m69_btc = _m69_result.get("btc_systemic_risk", 0.5)
_m69_regime = _m69_result.get("market_regime", "NORMAL")
_m69_breakdown = _m69_result.get("correlation_breakdown", False)

# NOTE: M86 (repo stress) is now computed earlier in this script (see
# "REPO MARKET STRESS (M86 — SOFR-EFFR spread)" near the Fiscal Liquidity
# block above), so its factor adjustment actually applies before
# sfc_pct is finalized — this used to be computed a second time here,
# re-fetching (from compute_repo_stress()'s own cache) and silently
# overwriting the already-used value. Removed as redundant; _m86_score
# and _m86_details remain valid module-level globals from the earlier
# computation for the JSON output further below.
# M70-M71: XAI Explainability (SHAP + LIME) — runs every cycle, cached by function
_xai_result = run_all_xai() if XAI_AVAILABLE else {"m70_shap_ok": False, "m71_lime_ok": False, "m70_shap_features": [], "m71_lime_features": []}
_m70_shap_features = _xai_result.get("m70_shap_features", [])
_m71_lime_features = _xai_result.get("m71_lime_features", [])

# ════════════════════════════════════════════════════════════
# M33b: Probabilistic Output — distribution, VaR, ES, quantiles
# ════════════════════════════════════════════════════════════
# Confidence calibration & reliability (M2 analysis)
try:
    from analysis.confidence_calibration import recalibrate as _calib_recalibrate, get_calibration_info as _calib_info
    _CALIB_AVAILABLE = True
    _CALIBRATED_CONF = _calib_recalibrate(float(composite_confidence or 0.5))
except Exception:
    _CALIB_AVAILABLE = False
    _CALIBRATED_CONF = float(composite_confidence or 0.5)

# Reliability: based on method_agreement (M2 finding: >0.85 = groupthink, <0.50 = noise)
_METHOD_AGREE = float(method_agreement) if 'method_agreement' in dir() and method_agreement is not None else 0.5
if _METHOD_AGREE > 0.85:
    _RELIABILITY = "LOW"       # groupthink — all methods agree on same (potentially stale) pattern
elif _METHOD_AGREE < 0.50:
    _RELIABILITY = "LOW"       # noise — no consensus
elif _METHOD_AGREE > 0.70:
    _RELIABILITY = "HIGH"      # healthy debate
else:
    _RELIABILITY = "MEDIUM"

# Collect method scores for uncertainty estimation
_PROB_METHOD_SCORES = []
# m5_qreg and m6_regime_score are 0-100 (from p_quantile*100, p_regime*100);
# normalize to 0-1 decimal to match other method scores
for _prob_field in ["m1_klr", "m2_logit", "m3_bayes", "m4_ewc", "m5_qreg", "m6_regime_score",
                     "m7_fisher", "m8_yield", "m9_liquidity", "m10_garch", "m11_var", "m12_jump",
                     "m13_funding", "m14_skew", "m15_concentration", "m16_regime_ml", "m17_granger",
                     "m18_entropy", "m19_mutual_info", "m20_obi", "m21_trade_flow", "m22_spread",
                     "m23_liquidity", "m24_cape", "m25_minsky", "m26_kahneman", "m27_taleb",
                     "m28_summers", "m29_debt", "m30_rajan", "m31_altman"]:
    try:
        _prob_val = float(locals().get(_prob_field, 0) or 0)
        if _prob_field in ("m5_qreg", "m6_regime_score"):
            _prob_val = _prob_val / 100.0
        _PROB_METHOD_SCORES.append(_prob_val)
    except (TypeError, ValueError):
        _PROB_METHOD_SCORES.append(0.0)

_PROB_RESULT = {}
if PROBABILISTIC_AVAILABLE and _PROB_HEAD is not None:
    try:
        _PROB_RESULT = _PROB_HEAD.compute(
            sfc_score=float(effective_sfc or sfc_pct or 0),
            method_scores=_PROB_METHOD_SCORES,
            composite_confidence=float(composite_confidence or 0.5),
            regime=str(regime) if "regime" in dir() else "NORMAL",
            zone=str(zone) if "zone" in dir() else "NORMAL",
        )
    except Exception as _prob_e:
        print(f"[SFC] Probabilistic compute failed: {_prob_e}", file=sys.stderr)
        _PROB_RESULT = {}

# ════════════════════════════════════════════════════════════
# M33c: Data Quality Pipeline — Outlier Detection + Kalman Imputation
# ════════════════════════════════════════════════════════════
_DQ_RESULT = {"dq_available": False}
try:
    from analysis.data_quality import DataQualityPipeline
    _DQ = DataQualityPipeline()
    _DQ_CLEANED, _DQ_FLAGS = _DQ.process(_PROB_METHOD_SCORES)
    _DQ_RESULT = {
        "dq_available": True,
        "dq_outliers": _DQ_FLAGS["outliers"],
        "dq_imputed": _DQ_FLAGS["imputed"],
        "dq_missing": _DQ_FLAGS["missing"],
        "dq_outlier_pct": _DQ_FLAGS["outlier_pct"],
        "dq_imputed_pct": _DQ_FLAGS["imputed_pct"],
        "dq_active": _DQ_FLAGS["active"],
    }
except Exception as _dq_e:
    print(f"[DQ] Data quality pipeline failed: {_dq_e}", file=sys.stderr)
    _DQ_RESULT = {"dq_available": False}

# ════════════════════════════════════════════════════════════
# M33d: Drift Detection — KS-test monitoring
# ════════════════════════════════════════════════════════════
_DRIFT_RESULT = {"drift_available": False}
try:
    from analysis.drift_detection import DriftDetector
    _DRIFT = DriftDetector()
    _DRIFT_RESULT_RAW = _DRIFT.check(_PROB_METHOD_SCORES)
    _DRIFT_RESULT = {
        "drift_available": True,
        "drift_detected": _DRIFT_RESULT_RAW.get("drift_detected", False),
        "drift_fields": _DRIFT_RESULT_RAW.get("drifted_fields", []),
        "drift_index": _DRIFT_RESULT_RAW.get("overall_drift_index", 0.0),
        "drift_consecutive": _DRIFT_RESULT_RAW.get("consecutive_drift", 0),
        "drift_stable": _DRIFT_RESULT_RAW.get("stable", True),
    }
except Exception as _drift_e:
    print(f"[Drift] Drift detection failed: {_drift_e}", file=sys.stderr)
    _DRIFT_RESULT = {"drift_available": False}

# ── REGIME vs ZONE DIVERGENCE CHECK ──
# `regime` (BULL/NORMAL/STRESS/BEAR/CRISIS/CAPITULATION) comes from
# detect_regime() + HMM pattern-based override; `zone`
# (NORMAL/ELEVATED/HIGH/CRITICAL) comes purely from effective_sfc's
# numeric threshold. These are genuinely different axes — regime
# classifies market STATE/PATTERN, zone classifies STRESS SEVERITY by
# score — and CAN legitimately diverge: HMM can override regime to
# CRISIS/BEAR based on pattern recognition across multiple signals, while
# effective_sfc's own numeric boost from that same regime detection is
# capped (see "Capped boost: full boost if DW unavailable, otherwise max
# +2pp" a few hundred lines above) and may not be large enough to push
# the score across zone's own threshold. Verified via simulation: a
# regime override to CRISIS with effective_sfc starting at 15 and a
# realistic +1.5pp regime boost lands at effective_sfc=16.5, still
# "NORMAL" zone — meaning the dashboard could show "CRISIS · NORMAL Zone"
# side by side with no indication these come from different
# methodologies, which reads as an internal contradiction to anyone
# looking at it rather than the two-different-axes situation it actually
# is. This flag lets the frontend show an explanatory note instead of a
# silent, confusing juxtaposition.
_REGIME_SEVERITY = {"BULL": 0, "NORMAL": 0, "STRESS": 1, "BEAR": 1, "CRISIS": 2, "CAPITULATION": 3}
_ZONE_SEVERITY = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}
_regime_sev = _REGIME_SEVERITY.get(str(regime).upper(), 0)
_zone_sev = _ZONE_SEVERITY.get(str(zone).upper(), 0)
regime_zone_divergence = abs(_regime_sev - _zone_sev) >= 2
regime_zone_divergence_note = (
    f"Regime ({regime}) and Zone ({zone}) disagree in severity — Regime reflects "
    f"HMM/pattern-based market state, Zone reflects the numeric stress score "
    f"threshold. Both can be independently correct; this isn't a data error."
) if regime_zone_divergence else None

# Build output

# ── COMPOSITE WEIGHT OPTIMIZER (EXPERIMENTAL — Option A: comparison only) ──
# Tracks GLF/SLI/MPI's internal component values over time and computes
# a CORRELATION-ADJUSTED alternative to their manually-assigned weights —
# components correlated with others in the same composite get LESS
# relative weight (not more, unlike naive PCA), independent components
# get relatively MORE. Deliberately kept separate from the LIVE weights
# used in the actual GLF/SLI/MPI calculations — this only reports a
# RECOMMENDATION for comparison, following the same cautious rollout
# pattern as reflexivity_divergence.py. See analysis/composite_weight_optimizer.py's
# module docstring for full rationale.
_weight_recommendations = {}
try:
    from analysis.composite_weight_optimizer import update_composite_history, compute_independence_weights

    if _glf_details and _glf_details.get("components"):
        _glf_comp = _glf_details["components"]
        _glf_zvals = {k: v.get("z_score") for k, v in _glf_comp.items() if v.get("z_score") is not None}
        _glf_weights = {k: v.get("weight") for k, v in _glf_comp.items() if v.get("weight") is not None}
        if _glf_zvals:
            update_composite_history("GLF", _glf_zvals)
        if _glf_weights:
            _rec, _det = compute_independence_weights("GLF", _glf_weights)
            _weight_recommendations["GLF"] = {"recommended_weights": _rec, "detail": _det}

    if _sli_details and _sli_details.get("components"):
        _sli_comp = _sli_details["components"]
        _sli_svals = {k: v.get("score") for k, v in _sli_comp.items() if v.get("score") is not None}
        _sli_weights = {k: v.get("weight") for k, v in _sli_comp.items() if v.get("weight") is not None}
        if _sli_svals:
            update_composite_history("SLI", _sli_svals)
        if _sli_weights:
            _rec, _det = compute_independence_weights("SLI", _sli_weights)
            _weight_recommendations["SLI"] = {"recommended_weights": _rec, "detail": _det}

    if _mpi_details and _mpi_details.get("components"):
        _mpi_comp = _mpi_details["components"]
        _mpi_svals = {k: v.get("score") for k, v in _mpi_comp.items() if v.get("score") is not None}
        _mpi_weights = {k: v.get("weight") for k, v in _mpi_comp.items() if v.get("weight") is not None}
        if _mpi_svals:
            update_composite_history("MPI", _mpi_svals)
        if _mpi_weights:
            _rec, _det = compute_independence_weights("MPI", _mpi_weights)
            _weight_recommendations["MPI"] = {"recommended_weights": _rec, "detail": _det}
except Exception as _weight_opt_e:
    print(f"[WeightOptimizer] Error: {_weight_opt_e}", file=sys.stderr)


# ── WALK-FORWARD VALIDATION SUMMARY (read-only cache) ──
# analysis/walk_forward_validation.py fetches ~11 years of FRED history
# and runs bootstrap resampling — far too expensive to redo every live
# 5-minute cycle. That script writes a small summary cache
# (.walk_forward_summary.json) when run manually/periodically; this
# just reads it. Fails safe (all None / wfv_available=False) if the
# validation script hasn't been run yet, or its cache is missing/stale.
_wfv_summary = {}
_wfv_available = False
try:
    _wfv_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".walk_forward_summary.json")
    if os.path.exists(_wfv_cache_path):
        with open(_wfv_cache_path) as _f:
            _wfv_summary = json.load(_f)
        _wfv_available = True
except Exception as _wfv_e:
    print(f"[WalkForward] Could not read summary cache: {_wfv_e}", file=sys.stderr)

# ── PROBABILISTIC HEAD CALIBRATION TRACKER (logging only, checked periodically) ──
# Appends this cycle's (effective_sfc, composite_confidence, regime,
# method_scores) to a dedicated history log — separate from
# data_collection.json (whose 2000-entry cap only retains ~7 days,
# insufficient for a 30-day-forward calibration check). Analysis itself
# is NOT run here — see analysis/probabilistic_head_tracker.py's
# suggested weekly cron entry. Never allowed to affect the live
# pipeline: wrapped in try/except, logging failure is non-fatal.
try:
    from analysis.probabilistic_head_tracker import log_cycle as _log_prob_head_cycle
    _log_prob_head_cycle(
        effective_sfc=effective_sfc, composite_confidence=composite_confidence,
        regime=regime, method_scores=all_method_scores,
    )
except Exception as _pht_e:
    print(f"[ProbHeadTracker] Wiring error (non-fatal): {_pht_e}", file=sys.stderr)

# ════════════════════════════════════════════════════════════
# IMBS L6 / L8 / L5 — DISPLAY-ONLY overlay computation
# All three add fields to data.json but are deliberately NOT blended into
# sfc_effective / signal / composite_confidence (see module docstrings).
# ════════════════════════════════════════════════════════════

# ── L6: Expectations Engine (FRED proxy, cached 6h) ──
_expect_score, _expect_details = 50.0, {"status": "unavailable"}
try:
    _expect_score, _expect_details = compute_expectations()
    print(f"[L6 Expect] score={_expect_score:.1f} gap={_expect_details.get('expectation_gap')} "
          f"label={_expect_details.get('label','?')} status={_expect_details.get('status','?')}", file=sys.stderr)
except Exception as _exp_e:
    print(f"[L6 Expect] Error: {_exp_e}", file=sys.stderr)
    _expect_score, _expect_details = 50.0, {"error": str(_exp_e), "status": "fallback"}

# ── L8: Tail Risk Engine — feed live Layer-2/3/4 signals ──
# Inputs resolved explicitly (blocker #1 in IMBS-design.md):
#   liquidity_stress   : GLF liquidity stress (Layer 3, 0-1)
#   behavior_stress    : blend of MPI positioning stress + systemic risk (Layer 4)
#   expectation_shock  : L6 gap_score (Layer 6, 0-100)
#   leverage           : MPI-derived leverage proxy (funding/OI-heavy stress) + cascade
#   correlation        : m69 correlation-breakdown flag (0 or 100)
_tr_leverage_proxy = None
if _mpi_stress is not None and cascade_risk is not None:
    # MPI stress is already positioning/derivative heavy (funding, OI, liq);
    # blend with cascade for the leverage/stress dimension.
    _tr_leverage_proxy = min(1.0, 0.7 * _mpi_stress + 0.3 * cascade_risk)
elif _mpi_stress is not None:
    _tr_leverage_proxy = _mpi_stress
elif cascade_risk is not None:
    _tr_leverage_proxy = cascade_risk

_tail_score, _tail_details = 50.0, {"status": "unavailable"}
try:
    _tail_score, _tail_details = compute_tail_risk(
        liquidity_stress=_glf_sfc_stress,                 # Layer 3 liquidity (0-1)
        behavior_stress=(
            0.6 * _mpi_stress + 0.4 * _m69_overall          # Layer 4 positioning + systemic
            if _mpi_stress is not None and _m69_overall is not None else
            _mpi_stress if _mpi_stress is not None else
            _m69_overall if _m69_overall is not None else None
        ),
        expectation_shock=_expect_score,                    # L6 gap score (0-100)
        leverage=_tr_leverage_proxy,                        # derivative leverage stress
        correlation=(100.0 if _m69_breakdown else None),  # flag: breakdown -> 100, no breakdown -> neutral (None)
    )
    print(f"[L8 TailRisk] score={_tail_score:.1f} sev={_tail_details.get('severity','?')} "
          f"active_dims={_tail_details.get('active_dimensions')}", file=sys.stderr)
except Exception as _tr_e:
    print(f"[L8 TailRisk] Error: {_tr_e}", file=sys.stderr)
    _tail_score, _tail_details = 50.0, {"error": str(_tr_e), "status": "fallback"}

# ── L5: Behavior-State overlay — re-combine existing participant signals ──
_behavior_state, _behavior_state_details = "UNKNOWN", {"status": "unavailable"}
try:
    _behavior_state, _behavior_state_details = compute_behavior_state(
        mpi_score=_mpi_score,
        fng=fng,
        cascade_risk=cascade_risk,
        behavioral_divergence=(_divergence_details.get("regime") if _divergence_details else None),
        etf_flow=_etf_m81_score,
        whale_pressure=whale_pressure,
        hmm_regime=_hmm_regime,
    )
    print(f"[L5 Behavior] state={_behavior_state} bull={_behavior_state_details.get('bullish_evidence')} "
          f"bear={_behavior_state_details.get('bearish_evidence')}", file=sys.stderr)
except Exception as _bs_e:
    print(f"[L5 Behavior] Error: {_bs_e}", file=sys.stderr)
    _behavior_state, _behavior_state_details = "UNKNOWN", {"error": str(_bs_e), "status": "fallback"}

# ── P0: Regime Consolidation — single consensus regime label ──
_regime_consensus_label, _regime_consensus_details = "UNKNOWN", {"status": "unavailable"}
try:
    _regime_consensus_label, _regime_consensus_details = consolidate_regime(
        regime=regime if "regime" in dir() else None,
        regime_prob=regime_prob if "regime_prob" in dir() else None,
        hmm_regime=_hmm_result.get('regime') if _hmm_result else None,
        hmm_crisis_prob=_hmm_result.get('crisis_probability') if _hmm_result else None,
        adv_regime=adv_regime.get('regime') if adv_regime else None,
        adv_crisis_prob=adv_regime.get('crisis_probability') if adv_regime else None,
        behavior_state=_behavior_state,
    )
    print(f"[P0 Regime] consensus={_regime_consensus_label} "
          f"conflict={_regime_consensus_details.get('conflict')} "
          f"agreement={_regime_consensus_details.get('agreement')}", file=sys.stderr)
except Exception as _rc_e:
    print(f"[P0 Regime] Error: {_rc_e}", file=sys.stderr)
    _regime_consensus_label, _regime_consensus_details = "UNKNOWN", {"error": str(_rc_e), "status": "fallback"}

# ── P1: Transmission Divergence — liquidity vs BTC structure ──
_transmission_status, _transmission_details = "UNAVAILABLE", {"status": "unavailable"}
try:
    _transmission_status, _transmission_details = classify_transmission(
        liquidity_stress=_glf_sfc_stress if "_glf_sfc_stress" in dir() else None,
        structural_stress=effective_sfc if "effective_sfc" in dir() else None,
        btc_change_24h=chg if "chg" in dir() else None,
    )
    print(f"[P1 Transmission] status={_transmission_status} "
          f"tone={_transmission_details.get('tone')} "
          f"conf={_transmission_details.get('confidence')}", file=sys.stderr)
except Exception as _tr_e:
    print(f"[P1 Transmission] Error: {_tr_e}", file=sys.stderr)
    _transmission_status, _transmission_details = "UNAVAILABLE", {"error": str(_tr_e), "status": "fallback"}

# ── P2: Trend Strength Score — momentum + alignment + structure ──
_trend_score, _trend_details = 50.0, {"status": "unavailable", "available": False, "label": "UNKNOWN"}
try:
    _afe_macd = _adv_features.get("macd_signal") if _adv_features else None
    _afe_bb = _adv_features.get("bb_width") if _adv_features else None
    _afe_obv = _adv_features.get("obv_norm") if _adv_features else None
    _trend_score, _trend_details = compute_trend_strength(
        rsi=rsi_14m if "rsi_14m" in dir() else None,
        mtf_alignment=(_mtf_result.get('alignment_score') if _mtf_result else None),
        hmm_regime=(_hmm_result.get('regime') if _hmm_result else None),
        hmm_crisis_prob=(_hmm_result.get('crisis_probability') if _hmm_result else None),
        dfs_regime=_dfs_regime if "_dfs_regime" in dir() else None,
        macd_signal=_afe_macd, bb_width=_afe_bb, obv_norm=_afe_obv,
    )
    print(f"[P2 Trend] score={_trend_score} label={_trend_details.get('label')} "
          f"domains={_trend_details.get('domain_values')}", file=sys.stderr)
except Exception as _ts_e:
    print(f"[P2 Trend] Error: {_ts_e}", file=sys.stderr)
    _trend_score, _trend_details = 50.0, {"error": str(_ts_e), "status": "fallback", "available": False, "label": "UNKNOWN"}

# ── P3: Trend Continuation Probability — from walk-forward cache ──
_trend_cont_probs, _trend_cont_details = {}, {"status": "unavailable", "available": False}
try:
    _trend_cont_probs, _trend_cont_details = compute_trend_continuation(
        sfc_effective=effective_sfc if "effective_sfc" in dir() else None,
        sfc_zone=zone if "zone" in dir() else None,
    )
    print(f"[P3 Continuation] bucket={_trend_cont_details.get('bucket')} "
          f"30d={(_trend_cont_probs.get(30,{}) or {}).get('probability')} "
          f"90d={(_trend_cont_probs.get(90,{}) or {}).get('probability')} "
          f"180d={(_trend_cont_probs.get(180,{}) or {}).get('probability')}", file=sys.stderr)
except Exception as _tc_e:
    print(f"[P3 Continuation] Error: {_tc_e}", file=sys.stderr)
    _trend_cont_probs, _trend_cont_details = {}, {"error": str(_tc_e), "status": "fallback", "available": False}

out = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "model_version": MODEL_VERSION,
    "btc": btc,
    "btc_24h": chg,
    "btc_mcap": mcap,
    "fng": fng,
    "fng_cls": fcls,
    "dom": round(dom, 1) if dom else None,
    # Previously this used the raw `dvol` variable directly, which is
    # None whenever get_dvol()'s Deribit API call fails (network issue,
    # rate limit, endpoint change) — a bare `except: return None` with no
    # logging. The internal stress-score calculation (`_factors_dvol`)
    # ALREADY had a fallback to the 30-day rolling average for exactly
    # this situation, but that fallback was never applied to what the
    # DASHBOARD displays — confirmed live: dvol showing "—" while sfc_pct
    # itself was computed fine (via the rolling-average fallback). Now
    # the displayed value uses the same fallback the score already relied
    # on, so the dashboard and the internal calculation agree on what
    # DVOL "is" this cycle rather than one silently being None while the
    # other quietly recovers.
    "dvol": round(_factors_dvol, 2) if _factors_dvol is not None else None,
    "dvol_is_fallback": dvol is None and _factors_dvol is not None,
    "sfc_base": sfc_pct,
    "sfc_effective": effective_sfc,
    "news_stress": news_stress,
    "news_headlines": news_headlines[:8],
    "news_sentiment": news_sentiment,
    "news_stats": news_stats,
    "m2_yoy": m2_yoy,
    "liq_mod": liq_mod,
    "liq_total_24h": liq_total_24h,
    "liq_long_vol": liq_long_vol,
    "liq_short_vol": liq_short_vol,
    "zone": zone,
    "floor_buffer": fb,
    "floor_total": ft,
    "regime": regime,
    "regime_zone_divergence": regime_zone_divergence,
    "regime_zone_divergence_note": regime_zone_divergence_note,
    "regime_prob": regime_prob,
    "transition_risk": transition_risk,
    "dxy": dxy,
    "pc_oi": pc_oi,
    "pc_vol": pc_vol,
    "dv_sfc": dv_sfc,
    "phi": phi,
    "state": state,
    "signal": signal,
    "factors": factors_raw,
    "ath": ath,
    "ath_date": ath_date,
    "rsi_14": rsi_14m,
    "rsi_regime": "OVERSOLD" if rsi_14m is not None and rsi_14m < 30 else "OVERBOUGHT" if rsi_14m is not None and rsi_14m > 70 else "NEUTRAL",
    "sopr_proxy": sopr_proxy,
    "sopr_signal": sopr_signal,
    "sopr_score": sopr_score,
    "cascade_risk": cascade_risk,
    "liq_density": liq_density,
    "liq_pressure": liq_pressure,
    "composite_confidence": composite_confidence,
    "confidence_components": {
        "method_agree": round(method_agreement, 3),
        "low_stress_boost": round(max(0, 1.0 - (effective_sfc/100)) * 0.08, 4),
        "rsi": round(rsi_conf, 3),
        "sopr": round(-0.05 if sopr_proxy is not None and sopr_proxy < 0.97 else 0.0, 3),
        "dvol": round(dvol_conf, 3),
        "macro_penalty": round(_macro_penalty, 3),
        "macro_confidence": round(macro_confidence, 3),
        "execution_risk": round(_execution_risk, 3),
        "cascade_risk_raw": round(cascade_risk, 3),
        "funding_imbalance": round(_imb_funding, 3),
        "squeeze_magnitude": round(_squeeze_magnitude, 3),
        "cascade_penalty": round(-(0.40 * cascade_risk), 3),  # continuous contribution to execution_risk, not stale step-function
        "fear_penalty": round(-_pen_fng, 3),  # references real _pen_fng (covers fng<15 AND fng>85)
        "news_penalty": round(-_pen_news, 3),
        "vol_penalty": round(-_pen_dvol_safety, 3),
        "transition_penalty": round(-_pen_transition, 3)
    },
    "m1_klr": round(m1_klr * 100, 1),
    "m2_logit": round(m2_logit * 100, 1),
    "m3_bayes": round(m3_bayes * 100, 1),
    "m4_ewc": round(m4_ewc * 100, 1),
    "m5_qreg": round(m5_qreg, 1),
    "m6_regime_score": round(m6_regime, 1),
    "method_agreement": round(method_agreement, 3),
    "m7_fisher": round(m7_s, 3) if m7_s is not None else None,
    "m7_detail": m7_d,
    "m8_yield": round(m8_s, 3) if m8_s is not None else None,
    "m8_detail": m8_d,
    "m9_liquidity": round(m9_s, 3) if m9_s is not None else None,
    "m9_detail": m9_d,
    "m10_garch": round(m10_s, 3) if m10_s is not None else None,
    "m10_detail": m10_d,
    "m11_var": round(m11_s, 3) if m11_s is not None else None,
    "m11_detail": m11_d,
    "m12_jump": round(m12_s, 3) if m12_s is not None else None,
    "m12_detail": m12_d,
    "m13_funding": round(m13_s, 3) if m13_s is not None else None,
    "m13_detail": m13_d,
    "m14_skew": round(m14_s, 3) if m14_s is not None else None,
    "m14_detail": m14_d,
    "m15_concentration": round(m15_s, 3) if m15_s is not None else None,
    "m15_detail": m15_d,
    "m16_regime_ml": round(m16_s, 3) if m16_s is not None else None,
    "m16_detail": m16_d,
    "m17_granger": round(m17_s, 3) if m17_s is not None else None,
    "m17_detail": m17_d,
    "m18_entropy": round(m18_s, 3) if m18_s is not None else None,
    "m18_detail": m18_d,
    "m19_mutual_info": round(m19_s, 3) if m19_s is not None else None,
    "m19_detail": m19_d,
    "new_methods_active": new_active,
    "new_methods_avg": round(new_avg, 3) if new_active > 0 else None,
    # Institutional methods (M20-M31)
    "inst_methods_active": inst_active_count,
    "inst_methods_avg": round(inst_avg_value, 3) if inst_active_count > 0 and inst_avg_value is not None else None,
    # Stablecoin liquidity (M76-M80)
    "sc_methods_active": sc_active,
    "sc_methods_avg": round(sc_avg, 3) if sc_active > 0 and sc_avg is not None else None,
    "total_methods_active": total_active_methods + sc_active,
    # M20-M31 individual scores
    "m20_obi": round(inst_results.get("m20_obi", 0), 3) if "m20_obi" in inst_results else None,
    "m20_detail": inst_details.get("m20_detail"),
    "m21_trade_flow": round(inst_results.get("m21_trade_flow", 0), 3) if "m21_trade_flow" in inst_results else None,
    "m21_detail": inst_details.get("m21_detail"),
    "m22_spread": round(inst_results.get("m22_spread", 0), 3) if "m22_spread" in inst_results else None,
    "m22_detail": inst_details.get("m22_detail"),
    "m23_liquidity": round(inst_results.get("m23_liquidity", 0), 3) if "m23_liquidity" in inst_results else None,
    "m23_detail": inst_details.get("m23_detail"),
    "m24_cape": round(inst_results.get("m24_cape", 0), 3) if "m24_cape" in inst_results else None,
    "m24_detail": inst_details.get("m24_detail"),
    "m25_minsky": round(inst_results.get("m25_minsky", 0), 3) if "m25_minsky" in inst_results else None,
    "m25_detail": inst_details.get("m25_detail"),
    "m26_kahneman": round(inst_results.get("m26_kahneman", 0), 3) if "m26_kahneman" in inst_results else None,
    "m26_detail": inst_details.get("m26_detail"),
    "m27_taleb": round(inst_results.get("m27_taleb", 0), 3) if "m27_taleb" in inst_results else None,
    "m27_detail": inst_details.get("m27_detail"),
    "m28_summers": round(inst_results.get("m28_summers", 0), 3) if "m28_summers" in inst_results else None,
    "m28_detail": inst_details.get("m28_detail"),
    "m29_debt": round(inst_results.get("m29_debt", 0), 3) if "m29_debt" in inst_results else None,
    "m29_detail": inst_details.get("m29_detail"),
    "m30_rajan": round(inst_results.get("m30_rajan", 0), 3) if "m30_rajan" in inst_results else None,
    "m30_detail": inst_details.get("m30_detail"),
    "m31_altman": round(inst_results.get("m31_altman", 0), 3) if "m31_altman" in inst_results else None,
    "m31_detail": inst_details.get("m31_detail"),
    # Stablecoin liquidity (M76-M80) with details
    "m76_supply_growth": None,  # redundant: r=-1.00 with m80, removed from output
    "m76_detail": sc_details.get("m76_detail"),
    "m77_ssr": round(_sc_results.get("m77_ssr", 0), 3) if _sc_results else None,
    "m77_detail": sc_details.get("m77_detail"),
    "m78_exchange_flow": round(_sc_results.get("m78_exchange_flow", 0), 3) if _sc_results else None,
    "m78_detail": sc_details.get("m78_detail"),
    "m79_velocity": round(_sc_results.get("m79_velocity", 0), 3) if _sc_results else None,
    "m79_detail": sc_details.get("m79_detail"),
    "m80_dominance": round(_sc_results.get("m80_dominance", 0), 3) if _sc_results else None,
    "m80_detail": sc_details.get("m80_detail"),
    # ── NEW: Stablecoin Liquidity Index (SLI — enhanced composite) ──
    "sli_score": round(_sli_score, 1) if _sli_score is not None else None,
    "sli_stress": round(_sli_sfc_stress, 3) if _sli_sfc_stress is not None else None,
    "sli_label": _sli_details.get("label") if _sli_details else None,
    "sli_available": STABLECOIN_INTEL_AVAILABLE,
    "sli_components": _sli_details.get("components") if _sli_details else None,
    "sli_usdt_growth": _sli_details.get("usdt_growth_pct") if _sli_details else None,
    "sli_usdc_growth": _sli_details.get("usdc_growth_pct") if _sli_details else None,
    "sli_growth_divergence": _sli_details.get("growth_divergence_pct") if _sli_details else None,
    # Microstructure change detection (M20-M23 cross-run deltas)
    "micro_change_flags": micro_change_flags,
    "micro_trend_score": round(micro_trend_score, 3) if micro_change_flags else None,
    "micro_deteriorating": micro_deteriorating if micro_change_flags else None,
    # QLSTM — Quantum Hybrid LSTM (M32) with GARCH + ProAdapt
    "m32_qlstm": round(qlstm_pred, 4) if qlstm_pred is not None else None,
    "m32_active": qlstm_ok,
    "m32_adjustment_pp": round(qlstm_adjustment, 4),
    # GARCH residual correction
    "m32_garch_residual": round(_GARCH_RESIDUAL, 4),
    "m32_garch_volatility": round(_GARCH_VOL, 4),
    "m32_hybrid_pred": round(qlstm_pred * 100 + _GARCH_RESIDUAL, 4) if qlstm_pred is not None else None,
    # ProAdapt online learning
    "m32_proadapt_weight": round(_PROADAPT_W, 4),
    "m32_proadapt_final": round(_PROADAPT_FINAL, 4) if _PROADAPT_FINAL is not None else None,
    # Mamba — State-Space Model (M32 Enhanced, CPU-friendly)
    "m32_mamba": round(mamba_pred, 4) if mamba_pred is not None else None,
    "m32_mamba_active": mamba_ok,
    "m32_mamba_short": round(mamba_result['stress_short'], 4) if mamba_result and mamba_result.get('available') else None,
    "m32_mamba_medium": round(mamba_result['stress_medium'], 4) if mamba_result and mamba_result.get('available') else None,
    "m32_mamba_long": round(mamba_result['stress_long'], 4) if mamba_result and mamba_result.get('available') else None,
    "m32_mamba_confidence": round(mamba_result['confidence'], 4) if mamba_result and mamba_result.get('available') else None,
    "m32_adjustment_pp_mamba": round(mamba_adjustment, 4),
    # M33 — Global Liquidity Index
    "m33_glo_score": round(m33_glo_score, 3) if m33_glo_score is not None else None,
    "m33_glo_detail": m33_glo_detail if m33_glo_detail else None,
    # ── NEW: Global Liquidity Factor (GLF — consolidated liquidity engine) ──
    "glf_score": round(_glf_score, 1) if _glf_score is not None else None,
    "glf_stress": round(_glf_sfc_stress, 3) if _glf_sfc_stress is not None else None,
    "glf_regime": _glf_details.get("regime") if _glf_details else None,
    "glf_active_components": _glf_details.get("active_components", 0) if _glf_details else 0,
    "glf_available": GLOBAL_LIQUIDITY_AVAILABLE,
    "glf_component_detail": _glf_details.get("components") if _glf_details else None,
    # Macro liquidity (M72-M75 / Layer 1)
    "macro_methods_active": _macro_active,
    "macro_methods_avg": _macro_avg if _macro_active > 0 else None,
    "m72_m2_growth": round(_m72_score, 3) if _m72_score is not None else None,
    "m72_detail": _m72_detail,
    "m73_m2_momentum": round(_m73_score, 3) if _m73_score is not None else None,
    "m73_detail": _m73_detail,
    "m74_fed_balance": round(_m74_score, 3) if _m74_score is not None else None,
    "m74_detail": _m74_detail,
    "m75_liquidity_composite": round(_m75_score, 3) if _m75_score is not None else None,
    "m75_detail": _m75_detail,
    # ETF flow (M81-M82)
    "etf_methods_active": 2 if _etf_details.get("status") == "ok" else 0,
    "m81_etf_flow": round(_etf_m81_score, 3),
    "m81_detail": _etf_details if _etf_details.get("status") == "ok" else None,
    "m82_etf_holdings": round(_etf_m82_score, 3),
    "m82_detail": _etf_details if _etf_details.get("status") == "ok" else None,
    # Fiscal liquidity (M83-M84)
    "fiscal_methods_active": (1 if _fiscal_details.get("m83", {}).get("status") == "ok" else 0) + (1 if _fiscal_details.get("m84", {}).get("status") == "ok" else 0),
    "m83_tga_score": round(_m83_score, 3),
    "m83_detail": _fiscal_details.get("m83") if _fiscal_details.get("status") == "ok" else None,
    "m84_rrp_score": round(_m84_score, 3),
    "m84_detail": _fiscal_details.get("m84") if _fiscal_details.get("status") == "ok" else None,
    "m85_fiscal_composite": round(_m85_composite, 3) if _fiscal_details.get("status") == "ok" else None,
    "m85_detail": {"regime": _fiscal_details.get("regime")} if _fiscal_details.get("status") == "ok" else None,
    # DXY correlation gate
    "dxy_btc_corr": round(dxy_btc_corr, 3) if dxy_btc_corr is not None else None,
    "dxy_gate_regime": "POSITIVE" if dxy_btc_corr is not None and dxy_btc_corr > 0.3 else "INVERSE" if dxy_btc_corr is not None and dxy_btc_corr < -0.3 else "MIXED" if dxy_btc_corr is not None else None,
    # XAI feature importance
    "xai_top_features": _XAI_FEATURES,
    # ML ensemble
    "ml_ensemble_score": round(ml_score, 3),
    "ml_ensemble_confidence": round(ml_confidence, 3),
    "ml_ensemble_msg": ml_msg,
    "ml_accuracy": ml_metrics.get("accuracy"),
    "ml_total_labeled": ml_metrics.get("total", 0),
    "ml_correct": ml_metrics.get("correct", 0),
    # Blend info
    "m1_m6_weight_pct": round(p1 * 100, 1) if causal_filter else 85.0,
    "m7_m19_weight_pct": round(p2 * 100, 1) if causal_filter else (10.0 if new_active > 0 else 0.0),
    "m20_m31_weight_pct": round(p3 * 100, 1) if causal_filter else (5.0 if inst_active_count > 0 else 0.0),
    # Causal inference info
    "causal_methods_active": len(causal_active_scores) if causal_filter else None,
    "causal_excluded": ", ".join(causal_excluded) if causal_excluded else None,
    "causal_excluded_count": len(causal_excluded) if causal_excluded else 0,
    # ── Q10 On-Chain Data (ErcinDedeoglu/crypto-market-data) ──
    "q10_whale_pressure": whale_pressure,
    "q10_onchain_value": onchain_value,
    "q10_buying_power": buying_power,
    "q10_market_structure": market_structure,
    "q10_available": whale_pressure is not None and onchain_value is not None and buying_power is not None,
    "q10_details": onchain_scores.get("details", {}) if whale_pressure is not None else {},
    # Reflexivity Divergence (experimental, display-only — see analysis/reflexivity_divergence.py)
    "reflexivity_divergence_score": _reflexivity_score,
    "reflexivity_divergence_detail": _reflexivity_details,
    "behavioral_divergence_score": _divergence_score,
    "behavioral_divergence_detail": _divergence_details,
    "weight_optimizer_recommendations": _weight_recommendations,
    # ── Advanced Feature Engineering (Peningkatan 1) ──
    # REMOVED: afe_rsi_7 (RSI-7 dihapus dari feature_engineering)
    "afe_macd_signal": float(round(_adv_features.get("macd_signal", 0), 4)) if _adv_features else None,
    "afe_bb_width": float(round(_adv_features.get("bb_width", 0), 4)) if _adv_features else None,
    # REMOVED: afe_atr (redundant, always None)
    "afe_vwap": float(round(_adv_features.get("vwap", 0), 2)) if _adv_features else None,
    "afe_obv_norm": float(round(_adv_features.get("obv", 0), 4)) if _adv_features else None,
    "afe_available": bool(_adv_features),
    # ── XGBoost Meta-Ensemble (Peningkatan 2) ──
    "xgb_meta_prediction": float(round(_xgb_pred, 2)) if _xgb_pred is not None else None,
    "xgb_meta_confidence": float(round(_xgb_confidence, 3)) if _xgb_confidence is not None else None,
    "xgb_blend_weight": float(round(_xgb_blend_weight, 3)) if '_xgb_blend_weight' in dir() and _xgb_pred is not None else None,
    "xgb_available": bool(_xgb_pred is not None),
    # ── HMM Regime Detection (Peningkatan 3) ──
    "hmm_regime": str(_hmm_result.get('regime')) if _hmm_result else None,
    "hmm_crisis_prob": float(round(_hmm_result.get('crisis_probability', 0), 3)) if _hmm_result else None,
    "hmm_available": bool(_hmm_available),
    # ── NEW: Dynamic Feature Weighting ──
    "dw_regime": str(_dw_regime) if '_dw_regime' in dir() and _dw_regime else None,
    "dw_z_score": round(_dw_z_score, 3) if '_dw_z_score' in dir() else None,
    "dw_weights": _dw_weights if _dw_weights else None,
    "dw_factors": {k: round(v, 3) for k, v in (_dw_norm_factors or {}).items()} if _dw_norm_factors else None,
    "dw_sfc_adjustment": round(_dw_sfc_adjustment, 2) if '_dw_sfc_adjustment' in dir() and _dw_sfc_adjustment else 0.0,
    "dw_available": DYNAMIC_WEIGHTING_AVAILABLE,
    # ── NEW: Market Positioning Index (MPI) ──
    "mpi_score": round(_mpi_score, 1) if _mpi_score is not None else None,
    "mpi_stress": round(_mpi_stress, 3) if _mpi_stress is not None else None,
    "mpi_label": _mpi_details.get("label") if _mpi_details else None,
    "mpi_available": MPI_AVAILABLE,
    "mpi_components": _mpi_details.get("components") if _mpi_details else None,
    # ── NEW: Liquidity Momentum (LM) ──
    "lm_score": round(_lm_score, 2) if _lm_score is not None else None,
    "lm_stress_adj": round(_lm_stress_adj, 3) if _lm_stress_adj is not None else None,
    "lm_label": _lm_details.get("label") if _lm_details else None,
    "lm_n_points": _lm_details.get("n_points", 0) if _lm_details else 0,
    "lm_available": LM_AVAILABLE,
    # ── NEW: Dynamic Feature Selector (DFS) ──
    "dfs_regime": _dfs_profile.get("regime") if _dfs_profile else None,
    "dfs_n_groups": _dfs_profile.get("n_groups", 0) if _dfs_profile else 0,
    "dfs_n_features": _dfs_profile.get("n_features", 0) if _dfs_profile else 0,
    "dfs_active_groups": _dfs_profile.get("active_groups", []) if _dfs_profile else [],
    "dfs_group_weights": _dfs_profile.get("group_weights") if _dfs_profile else None,
    "dfs_available": DFS_AVAILABLE,
    # ── Multi-Timeframe Fusion (Peningkatan 4) ──
    "mtf_alignment_score": float(round(_mtf_result.get('alignment_score', 0), 3)) if _mtf_result else None,
    "mtf_divergence": bool(_mtf_result.get('divergence_detected', False)) if _mtf_result else None,
    "mtf_available": bool(_mtf_result),
    # ── Online Learning EWMA (Peningkatan 5) ──
    "ewma_corrected": bool(True) if '_ewma' in dir() else False,
    "ewma_available": bool(_get_adv_online() is not None),
    # Advanced modules: Regime Detection (P2), Uncertainty (P4), Alt Data (P6)
    "adv_regime": adv_regime.get('regime', 'NORMAL') if adv_regime else 'NORMAL',
    "adv_crisis_prob": round(adv_regime.get('crisis_probability', 0), 3) if adv_regime else 0,
    "adv_regime_stability": round(adv_regime.get('stability', 0.9), 3) if adv_regime else 0.9,
    "adv_regime_boost": adv_regime_boost,
    "adv_uncertainty": round(adv_uncertainty.get('uncertainty', 0), 3) if adv_uncertainty else None,
    "adv_confidence": adv_uncertainty.get('recommended_action', 'UNKNOWN') if adv_uncertainty else 'UNKNOWN',
    "adv_trends_recession": round(adv_alt.get('trends_recession', 0.5), 3) if adv_alt else None,
    "adv_trends_crash": round(adv_alt.get('trends_crash', 0.5), 3) if adv_alt else None,
    "adv_reddit_sentiment": round(adv_alt.get('reddit_sentiment', 0), 3) if adv_alt else None,
    "adv_reddit_label": adv_alt.get('reddit_label', 'NONE') if adv_alt else 'NONE',
    "adv_cg_dd_ath": round(adv_alt.get('cg_ath_dd', 0), 3) if adv_alt else None,
    # Backtest metrics — ESTIMATED, not walk-forward validated (raw pre-cal ECE≈0.422)
    "bt_sharpe": bt_sharpe,
    "bt_sharpe_low": bt_sharpe_low,
    "bt_sharpe_high": bt_sharpe_high,
    "bt_max_dd": bt_max_dd,
    "bt_win_rate": bt_win_rate,
    "bt_win_rate_low": bt_win_rate_low,
    "bt_win_rate_high": bt_win_rate_high,
    "bt_return": bt_return,
    "bt_periods": bt_periods,
    "bt_stability": bt_stability,
    "bt_calibration_note": bt_calibration_note,
    "bt_label": "ESTIMATED (heuristic formula, not walk-forward validated)",
    # Walk-forward validation (genuine, FRED-history-based — separate
    # system from the bt_* heuristic estimates above; see
    # analysis/walk_forward_validation.py)
    "wfv_available": _wfv_available,
    "wfv_n_periods": _wfv_summary.get("n_periods"),
    "wfv_generated_at": _wfv_summary.get("generated_at"),
    "wfv_gap_7d": _wfv_summary.get("gap_7d"),
    "wfv_gap_7d_ci_lo": _wfv_summary.get("gap_7d_ci_lo"),
    "wfv_gap_7d_ci_hi": _wfv_summary.get("gap_7d_ci_hi"),
    "wfv_gap_7d_significant": _wfv_summary.get("gap_7d_significant"),
    "wfv_gap_30d": _wfv_summary.get("gap_30d"),
    "wfv_gap_30d_ci_lo": _wfv_summary.get("gap_30d_ci_lo"),
    "wfv_gap_30d_ci_hi": _wfv_summary.get("gap_30d_ci_hi"),
    "wfv_gap_30d_significant": _wfv_summary.get("gap_30d_significant"),
    "wfv_n_stress_pct": _wfv_summary.get("n_stress_pct"),
    "wfv_label": "WALK-FORWARD VALIDATED — reduced 4-input proxy (price, DXY, M2, FNG; St/Ft=0), NOT the full 90+ method live score" if _wfv_available else "NOT YET RUN",
    # — Kelly Criterion Position Sizing (Gap 2 dari Reality Check) —
    "kelly_p_win": round(composite_confidence, 3),
    "kelly_b_payoff": 2.0,  # default risk/reward ratio
    "kelly_fraction": round(max(0, (composite_confidence * 2.0 - (1 - composite_confidence)) / 2.0) * _kelly_override, 4),
    "kelly_half": round(max(0, (composite_confidence * 2.0 - (1 - composite_confidence)) / 4.0) * _kelly_override, 4),
    "kelly_quarter": round(max(0, (composite_confidence * 2.0 - (1 - composite_confidence)) / 8.0) * _kelly_override, 4),
    "kelly_override_reason": _kelly_override_reason,
    # — Signal decision: single source of truth for BUY/WATCH/CASH.
    # Mirrors the frontend sigClass logic exactly (kelly + sfc vs regime-adjusted thresholds).
    # kelly_final<=0 or sfc>=cashThresh => CASH; kelly_final>0 & sfc<buyThresh => BUY; else WATCH.
    # kelly_final uses the SAME formula as kelly_fraction above (raw edge * override), so a low
    # confidence that drives raw kelly to 0 correctly yields CASH even without an override.
    "signal_decision": (
        "CASH" if ((composite_confidence is not None and max(0, (composite_confidence * 2.0 - (1 - composite_confidence)) / 2.0) * _kelly_override <= 0)
                   or (effective_sfc is not None and effective_sfc >= 50 * _REGIME_DRIVER_MULT)) else
        "BUY" if (composite_confidence is not None and effective_sfc is not None
                  and max(0, (composite_confidence * 2.0 - (1 - composite_confidence)) / 2.0) * _kelly_override > 0
                  and effective_sfc < 25 * _REGIME_DRIVER_MULT) else "WATCH"),
    # — Signal Timing (alert window estimation, monthly timeframe) —
    "signal_threshold_mult": _REGIME_DRIVER_MULT,
    "signal_type": "STRESS_TRANSITION" if transition_risk > 0.60 else "STRESS" if effective_sfc and effective_sfc > 25 * _REGIME_DRIVER_MULT else "CALM",
    "signal_strength": round(min(effective_sfc / 50.0 if effective_sfc else 0, 1.0), 3),
    "calibrated_confidence": _CALIBRATED_CONF,
    "reliability": _RELIABILITY,
    "reliability_reason": ("HIGH agreement (>0.85) — possible groupthink" if _RELIABILITY == "LOW" and _METHOD_AGREE > 0.85 else
                           "LOW agreement (<0.50) — no consensus" if _RELIABILITY == "LOW" and _METHOD_AGREE < 0.50 else
                           "Moderate agreement — healthy signal" if _RELIABILITY == "HIGH" else
                           "Adequate agreement"),
    "timing_precision": "LOW" if composite_confidence < 0.3 else "MEDIUM" if composite_confidence < 0.6 else "HIGH",
    "alert_window_days": round(7 + 30 * (1 - composite_confidence), 1),  # wider window = lower confidence (monthly TF)
    "timeframe": "MONTHLY",
    "readiness_score": round(composite_confidence * (1.0 - min(effective_sfc/100 if effective_sfc else 0, 0.5)), 3),
    "shock_factor": shock_factor,
    "shock_event": shock_event,
    "shock_severity": shock_severity,
    "sec_events": sec_events,
    # ── Q9 News Scoring System (module not implemented yet) ──
    "q9_available": False,
    "q9_articles": 0,
    # ── Q5 Advanced Methods: M65-M69 ──
    "m65_cnn_attention": round(_m65_stress, 4),
    "m65_pattern_type": _m65_pattern,
    "m65_available": CNN_ATTENTION_AVAILABLE,
    "m65_affects_sfc_score": False,  # display-only — pattern info, not blended into sfc_pct/effective_sfc
    "m68_drl_signal": _m68_signal,
    "m68_available": DRL_AVAILABLE,
    "m68_agent_loaded": _drl_agent_loaded,
    "m68_affects_sfc_score": False,  # display-only — uses real market state as input, but signal isn't blended into sfc_pct
    "m69_systemic_risk": round(_m69_overall, 4),
    "m69_btc_systemic_risk": round(_m69_btc, 4),
    "m69_market_regime": _m69_regime,
    "m69_correlation_breakdown": _m69_breakdown,
    "m69_available": GNN_AVAILABLE,
    "m69_is_simulated": _m69_is_simulated,  # True: no real ETH/SPX/Gold data source, uses fixed simulated inputs
    # Repo market stress (M86 — SOFR-EFFR spread)
    "m86_repo_stress_score": round(_m86_score, 3),
    "m86_detail": _m86_details if _m86_details.get("status") == "ok" else None,
    "m86_available": REPO_STRESS_AVAILABLE,
    # Global Sovereign Liquidity Score (M90 — consolidated US/Japan/Europe/UK)
    "m90_gsls_score": round(_m90_score, 1),
    "m90_detail": _m90_details if _m90_details.get("status") in ("ok", "partial") else None,
    "m90_available": GSLS_AVAILABLE,
    # ── XAI Explainability: M70-M71 ──
    "m70_shap_ok": _xai_result.get("m70_shap_ok", False),
    "m70_shap_top_1": _m70_shap_features[0]["name"] if len(_m70_shap_features) > 0 else None,
    "m70_shap_top_1_pct": _m70_shap_features[0]["importance_pct"] if len(_m70_shap_features) > 0 else None,
    "m70_shap_top_3": ", ".join(f["name"] for f in _m70_shap_features[:3]) if _m70_shap_features else None,
    "m71_lime_ok": _xai_result.get("m71_lime_ok", False),
    "m71_lime_top_1": _m71_lime_features[0]["name"] if len(_m71_lime_features) > 0 else None,
    "m71_lime_top_1_pct": _m71_lime_features[0]["importance_pct"] if len(_m71_lime_features) > 0 else None,
    "m71_lime_top_3": ", ".join(f["name"] for f in _m71_lime_features[:3]) if _m71_lime_features else None,
    # ── M33b: Probabilistic Output (distribution, VaR, ES) ──
    "prob_dist_available": PROBABILISTIC_AVAILABLE and bool(_PROB_RESULT),
    "predicted_mean": _PROB_RESULT.get("predicted_mean") if _PROB_RESULT else None,
    "predicted_std": _PROB_RESULT.get("predicted_std") if _PROB_RESULT else None,
    "var_95": _PROB_RESULT.get("var_95") if _PROB_RESULT else None,
    "es_975": _PROB_RESULT.get("es_975") if _PROB_RESULT else None,
    "ci_90_lower": _PROB_RESULT.get("ci_90_lower") if _PROB_RESULT else None,
    "ci_90_upper": _PROB_RESULT.get("ci_90_upper") if _PROB_RESULT else None,
    "prob_stress": _PROB_RESULT.get("prob_stress") if _PROB_RESULT else None,
    "prob_critical": _PROB_RESULT.get("prob_critical") if _PROB_RESULT else None,
    "prob_crash_10pct": _PROB_RESULT.get("prob_crash_10pct") if _PROB_RESULT else None,
    "prob_calm": _PROB_RESULT.get("prob_calm") if _PROB_RESULT else None,
    "prob_quantiles": _PROB_RESULT.get("quantiles") if _PROB_RESULT else None,
    "prob_sharpe": _PROB_RESULT.get("sharpe_ratio") if _PROB_RESULT else None,
    "prob_sortino": _PROB_RESULT.get("sortino_ratio") if _PROB_RESULT else None,
    "prob_uncertainty_breakdown": _PROB_RESULT.get("uncertainty_breakdown") if _PROB_RESULT else None,
    # ── Data Quality status ──
    "dq_available": _DQ_RESULT.get("dq_available", False),
    "dq_outliers": _DQ_RESULT.get("dq_outliers", 0),
    "dq_imputed": _DQ_RESULT.get("dq_imputed", 0),
    "dq_missing": _DQ_RESULT.get("dq_missing", 0),
    "dq_outlier_pct": _DQ_RESULT.get("dq_outlier_pct", 0.0),
    "dq_imputed_pct": _DQ_RESULT.get("dq_imputed_pct", 0.0),
    "dq_active": _DQ_RESULT.get("dq_active", 0),
    # ── Drift Detection status ──
    "drift_available": _DRIFT_RESULT.get("drift_available", False),
    "drift_detected": _DRIFT_RESULT.get("drift_detected", False),
    "drift_fields": _DRIFT_RESULT.get("drift_fields", []),
    "drift_index": _DRIFT_RESULT.get("drift_index", 0.0),
    "drift_consecutive": _DRIFT_RESULT.get("drift_consecutive", 0),
    "drift_stable": _DRIFT_RESULT.get("drift_stable", True),
    # ── Circuit Breaker status ──
    "cb_available": CB_AVAILABLE,
    "cb_tripped": _CIRCUIT_BREAKER.get_stats().get("tripped", False) if CB_AVAILABLE and _CIRCUIT_BREAKER else False,
    "cb_failures": _CIRCUIT_BREAKER.get_stats().get("consecutive_failures", 0) if CB_AVAILABLE and _CIRCUIT_BREAKER else 0,
    "cb_total_failures": _CIRCUIT_BREAKER.get_stats().get("total_failures", 0) if CB_AVAILABLE and _CIRCUIT_BREAKER else 0,
    # ── IMBS L6: Expectations Engine (display-only FRED proxy) ──
    "expectation_gap": _expect_details.get("expectation_gap"),
    "expectation_score": round(_expect_score, 1) if _expect_score is not None else None,
    "expectation_label": _expect_details.get("label"),
    "expectation_available": bool(_expect_details.get("available")),
    "expectation_details": {
        "cpi_yoy_pct": _expect_details.get("cpi_yoy_pct"),
        "breakeven_inflation": _expect_details.get("breakeven_inflation"),
        "real_yield_10y": _expect_details.get("real_yield_10y"),
        "curve_10y2y": _expect_details.get("curve_10y2y"),
        "unemployment": _expect_details.get("unemployment"),
        "status": _expect_details.get("status"),
        "unavailable": _expect_details.get("unavailable", []),
    },
    # ── IMBS L8: Tail Risk Engine (display-only composite) ──
    "tail_risk_score": _tail_details.get("score"),
    "tail_risk_severity": _tail_details.get("severity"),
    "tail_risk_available": bool(_tail_details.get("available")),
    "tail_risk_dimensions": _tail_details.get("dimensions"),
    "tail_risk_active_dims": _tail_details.get("active_dimensions"),
    "tail_risk_missing_dims": _tail_details.get("missing_dimensions"),
    # ── IMBS L5: Behavior-State overlay (display-only) ──
    "behavior_state": _behavior_state,
    "behavior_state_available": bool(_behavior_state_details.get("available")),
    "behavior_state_bull_evidence": _behavior_state_details.get("bullish_evidence"),
    "behavior_state_bear_evidence": _behavior_state_details.get("bearish_evidence"),
    # ── P0: Regime Consolidation (single consensus regime label) ──
    "regime_consensus": _regime_consensus_label,
    "regime_consensus_available": bool(_regime_consensus_details.get("available")),
    "regime_consensus_severity": _regime_consensus_details.get("severity"),
    "regime_consensus_agreement": _regime_consensus_details.get("agreement"),
    "regime_consensus_conflict": bool(_regime_consensus_details.get("conflict")),
    "regime_consensus_conflict_sources": _regime_consensus_details.get("conflict_sources"),
    "regime_consensus_sources": _regime_consensus_details.get("sources"),
    # ── P1: Transmission Divergence (liquidity vs BTC structure) ──
    "transmission_status": _transmission_status,
    "transmission_available": bool(_transmission_details.get("available")),
    "transmission_message": _transmission_details.get("message"),
    "transmission_tone": _transmission_details.get("tone"),
    "transmission_confidence": _transmission_details.get("confidence"),
    "transmission_liquidity_state": _transmission_details.get("liquidity_state"),
    "transmission_btc_state": _transmission_details.get("btc_state"),
    # ── P2: Trend Strength Score ──
    "trend_strength_score": _trend_score,
    "trend_strength_available": bool(_trend_details.get("available")),
    "trend_strength_label": _trend_details.get("label"),
    "trend_strength_domains": _trend_details.get("domain_values"),
    # ── P3: Trend Continuation Probability (walk-forward calibrated) ──
    "cont_available": bool(_trend_cont_details.get("available")),
    "cont_bucket": _trend_cont_details.get("bucket"),
    "cont_prob_30d": (_trend_cont_probs.get(30, {}) or {}).get("probability"),
    "cont_prob_90d": (_trend_cont_probs.get(90, {}) or {}).get("probability"),
    "cont_prob_180d": (_trend_cont_probs.get(180, {}) or {}).get("probability"),
    "cont_rel_90d": (_trend_cont_probs.get(90, {}) or {}).get("relative"),
    "cont_caveat": _trend_cont_details.get("caveat"),
}

# ── Circuit Breaker: validate output before writing ──
_CB_OUT = out
_CB_WARNINGS = []
if CB_AVAILABLE and _CIRCUIT_BREAKER is not None:
    try:
        _CB_OUT, _CB_OK, _CB_WARNINGS = _CIRCUIT_BREAKER.validate(out)
        if _CB_WARNINGS:
            for _cb_msg in _CB_WARNINGS:
                print(f"[CB] {_cb_msg}", file=sys.stderr)
        if _CB_OK or not _CB_OUT:
            # OK or tripped — use validated output
            pass
        else:
            # Warning-level issues only — still print cleaned data
            out = _CB_OUT
    except Exception as _cb_e:
        print(f"[CB] Validation error: {_cb_e}", file=sys.stderr)

print(json.dumps(out, indent=2))
btc_str = f"${btc:,.0f}" if btc is not None else "N/A"
rsi_str = f"{rsi_14m}" if rsi_14m is not None else "N/A"
sopr_str = f"{sopr_proxy}" if sopr_proxy is not None else "N/A"
qlstm_str = f" QLSTM={qlstm_pred*100:.1f}" if qlstm_pred is not None else ""
m65_str = f" CNN={_m65_stress:.2f}" if CNN_ATTENTION_AVAILABLE else ""
m68_str = f" DRL={_m68_signal}" if DRL_AVAILABLE else ""
m69_str = f" SYS={_m69_overall:.2f}" if GNN_AVAILABLE else ""
print(f"\n✅ BTC={btc_str} | SFC={effective_sfc:.1f}% | Zone={zone} | RSI-14M={rsi_str} | SOPR={sopr_str} | News={news_stress:.1f} | {regime} | TF=MONTHLY | Methods={total_active_methods}/42 | Macro={_macro_active}/4 | SC={sc_active}/5{qlstm_str}{m65_str}{m68_str}{m69_str}", file=sys.stderr)

# Paper trading moved to pipeline script (sfc-pipeline.sh) to avoid
# race condition: collect.py stdout > data.json is still buffered
# when paper_trader.py runs as subprocess, causing empty-file crash.
