#!/usr/bin/env python3
"""
QLSTM Enhanced Inference — Integration Module
==============================================
Combines: QLSTM inference + GARCH residual correction + 
          ProAdapt online learning + XAI feature importance.

Called from collect.py during each data collection cycle.
Singleton cache to avoid reloading model every 30s call.
"""

import contextlib, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

CACHE = {
    "qlstm_model": None,
    "model_input_dim": None,
    "model_seq_len": None,
    "model_hidden_dim": None,
    "last_ts": 0,
    "cached_result": None,
}

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFC_DIR = os.path.dirname(SCRIPT_DIR)  # /home/ubuntu

# Paths
QLSTM_MODEL_PATH = os.path.join(SCRIPT_DIR, "qlstm_model.pt")
QLSTM_MODULE_PATH = os.path.join(os.path.dirname(__file__), "qlstm_model.py")
DATA_PATH = os.path.join(SFC_DIR, "sfc", "data_collection.json")

# Error tracking
LAST_ERROR = None


def _load_qlstm_model():
    """Load or return cached QLSTM model."""
    if CACHE["qlstm_model"] is not None:
        return CACHE["qlstm_model"], CACHE["model_input_dim"], CACHE["model_seq_len"], CACHE["model_hidden_dim"]
    
    if not os.path.exists(QLSTM_MODEL_PATH) or not os.path.exists(QLSTM_MODULE_PATH):
        return None, None, None, None
    
    import importlib.util
    import torch
    
    checkpoint = torch.load(QLSTM_MODEL_PATH, map_location="cpu", weights_only=False)
    input_dim = checkpoint["input_dim"]
    seq_len = checkpoint["seq_len"]
    hidden_dim = checkpoint.get("hidden_dim", 16)
    
    spec = importlib.util.spec_from_file_location("qlstm_model", QLSTM_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    QLSTMVolatilityPredictor = mod.QLSTMVolatilityPredictor
    
    model = QLSTMVolatilityPredictor(input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    CACHE["qlstm_model"] = model
    CACHE["model_input_dim"] = input_dim
    CACHE["model_seq_len"] = seq_len
    CACHE["model_hidden_dim"] = hidden_dim
    
    return model, input_dim, seq_len, hidden_dim


def _load_and_normalize():
    """Load features, pad to max_len, normalize."""
    if not os.path.exists(DATA_PATH):
        return None, None, None
    
    with open(DATA_PATH) as f:
        data = json.load(f)
    feats = data.get("features", [])
    if not feats:
        return None, None, None
    
    max_len = max(len(f) for f in feats)
    feats_padded = []
    for f in feats:
        f_arr = np.array(f, dtype=np.float32)
        if len(f_arr) < max_len:
            f_arr = np.concatenate([f_arr, np.full(max_len - len(f_arr), 0.5, dtype=np.float32)])
        feats_padded.append(f_arr)
    features = np.array(feats_padded)
    
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    features_norm = (features - mean) / std
    
    return features, features_norm, max_len


def _load_latest_resolved_label(data):
    """
    Return the most recent RESOLVED (non-None) label from data_collection.json,
    as a 0-100 scale value, or None if no resolved label is available yet.

    Replaces the previous _compute_ensemble_target(), which computed
    "ground truth" as (features[:, :6] @ weights[:6]) * 100.0 — a fixed
    linear combination of M1-M6, the SAME features QLSTM receives as
    input. ProAdapt was therefore being told "the actual outcome" was a
    formula built from QLSTM's own inputs, so add_observation() below was
    training the online-adaptation weight (w_qlstm vs w_garch) to agree
    with that formula rather than with what actually happened to BTC price.

    This reads "labels" instead — populated by ml_ensemble.py's
    resolve_pending_labels() from REALIZED BTC price outcome (see that
    file for the exact price-outcome computation), independent of any
    method score. If the most recent observations are all still pending
    (label=None, not enough time elapsed since they were recorded — labels
    resolve ~6 hours later), this returns None and the caller skips
    recording that cycle's ProAdapt observation rather than guessing.
    """
    labels = data.get("labels", [])
    for label in reversed(labels):
        if label is not None:
            return float(label) * 100.0
    return None


def run_enhanced_inference(force=False):
    """
    Main entry point for collect.py.
    Returns dict with:
      - qlstm_pred: float (0-100)
      - qlstm_ok: bool
      - garch_residual: float
      - garch_volatility: float
      - hybrid_pred: float (QLSTM + GARCH)
      - proadapt_weight: float
      - proadapt_final: float
      - xai_top_features: list or None
      - xai_ok: bool
    """
    global LAST_ERROR
    
    now = time.time()
    # Cache for 120s unless forced
    if not force and CACHE["cached_result"] and now - CACHE["last_ts"] < 120:
        return CACHE["cached_result"]
    
    try:
        with contextlib.redirect_stdout(sys.stderr):
            # 1. Load model
            model, input_dim, seq_len, hidden_dim = _load_qlstm_model()
            if model is None:
                return {"qlstm_ok": False, "error": "model_not_found"}
            
            # 2. Load data
            features, features_norm, max_len = _load_and_normalize()
            if features_norm is None or len(features_norm) < seq_len + 2:
                return {"qlstm_ok": False, "error": "insufficient_data"}
            
            # 3. QLSTM prediction on latest window
            latest_seq = features_norm[-seq_len:]
            import torch
            inp = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qlstm_pred = model(inp).item()  # 0-100 SFC
            qlstm_normalized = qlstm_pred / 100.0  # 0-1
            
            # 4. GARCH residual correction
            try:
                # FIX (2026-07, per SFC_Path_Inventory.docx diagnosis): was
                # inserting SCRIPT_DIR (the grandparent /home/ubuntu/sfc)
                # into sys.path, but hybrid_correction.py actually lives in
                # THIS file's own directory (models/) — the import always
                # failed silently, causing garch_residual/garch_volatility
                # to fall back to 0.0 on every single cycle. Insert this
                # file's own directory instead.
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from hybrid_correction import run_hybrid_correction
                result = run_hybrid_correction()
                garch_residual = result["garch_residual"]
                garch_volatility = result["garch_volatility"]
                hybrid_pred = result["final_pred"]
            except Exception as e:
                garch_residual = 0.0
                garch_volatility = 0.0
                hybrid_pred = qlstm_pred
            
            # 5. ProAdapt online learning
            try:
                from proadapt import add_observation, get_adaptive_prediction, load_state
                
                with open(DATA_PATH) as _f:
                    _data_for_labels = json.load(_f)
                pro_state = load_state()
                # Ground truth for ProAdapt's online weight update — see
                # _load_latest_resolved_label() docstring for why this
                # replaced the previous circular-formula approach.
                latest_actual = _load_latest_resolved_label(_data_for_labels)
                
                # Get adaptive prediction
                adapt = get_adaptive_prediction(
                    qlstm_pred / 100.0,  # 0-1 scale
                    garch_residual / 100.0,
                    hybrid_pred / 100.0
                )
                
                proadapt_weight = adapt.get("w_qlstm", 0.5)
                proadapt_final = adapt.get("final_pred", 0.5) * 100.0
                fine_tune = adapt.get("fine_tune_needed", False)
                
                # Record current observation only if we have a REAL resolved
                # outcome to compare against — if the latest observations
                # are all still pending, skip rather than record a
                # meaningless comparison.
                if latest_actual is not None:
                    add_observation(
                        qlstm_pred / 100.0,
                        garch_residual / 100.0,
                        hybrid_pred / 100.0,
                        latest_actual / 100.0
                    )
            except Exception as e:
                proadapt_weight = 0.5
                proadapt_final = hybrid_pred
                fine_tune = False
            
            # 6. XAI feature importance
            xai_features = None
            try:
                from xai_explainer import get_top_features
                xai_features = get_top_features(DATA_PATH, QLSTM_MODEL_PATH, QLSTM_MODULE_PATH, top_n=5)
            except Exception:
                pass
            
            result = {
                "qlstm_pred": round(qlstm_pred, 4),
                "qlstm_ok": True,
                "garch_residual": round(garch_residual, 4),
                "garch_volatility": round(garch_volatility, 4),
                "hybrid_pred": round(hybrid_pred, 4),
                "proadapt_weight": round(proadapt_weight, 4),
                "proadapt_final": round(proadapt_final, 4),
                "fine_tune_needed": fine_tune,
                "xai_top_features": xai_features,
                "xai_ok": xai_features is not None,
            }
            
            CACHE["cached_result"] = result
            CACHE["last_ts"] = now
            LAST_ERROR = None
            return result
        
    except Exception as e:
        LAST_ERROR = str(e)
        return {
            "qlstm_ok": False,
            "error": str(e),
            "qlstm_pred": None,
            "garch_residual": 0.0,
            "garch_volatility": 0.0,
            "hybrid_pred": None,
            "proadapt_weight": 0.5,
            "proadapt_final": None,
            "fine_tune_needed": False,
            "xai_top_features": None,
            "xai_ok": False,
        }


if __name__ == "__main__":
    result = run_enhanced_inference(force=True)
    print(json.dumps({k: v for k, v in result.items() if k != "xai_top_features"}, indent=2))
    if result.get("xai_top_features"):
        print("\nXAI Top Features:")
        for f in result["xai_top_features"]:
            print(f"  {f['name']:<18} {f['importance_pct']:>5.1f}%  {f['direction']}")
    print(f"\nError: {result.get('error', 'None')}")
