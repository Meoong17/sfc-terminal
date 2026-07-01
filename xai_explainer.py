#!/usr/bin/env python3
"""
XAI Explainer — Feature Importance for SFC QLSTM
=================================================
Lightweight permutation importance that runs during each collect cycle.
Output: top 10 features with importance scores and direction.
"""

import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

import torch

FEATURE_NAMES = [
    "M1_KLR", "M2_Logit", "M3_Bayes", "M4_EWC", "M5_QReg", "M6_Regime",
    "M7_Fisher", "M8_MonteCarlo", "M9_LiquidityGap", "M10_VaR",
    "M11_CVaR", "M12_MaxDrawdown", "M13_FundingRate", "M14_Skew",
    "M15_Kurtosis", "M16_Sharpe", "M17_Granger", "M18_Entropy",
    "M19_MutualInfo", "M20_OBI", "M21_TradeFlow", "M22_Spread",
    "M23_LiquidityDepth", "M24_CAPE", "M25_Minsky", "M26_Kahneman",
    "M27_Taleb", "M28_Summers", "M29_Debt", "M30_Rajan", "M31_Altman"
]

def get_top_features(data_path, model_path, model_module_path, top_n=8):
    """
    Load QLSTM model, run permutation importance on latest features.
    Returns list of {name, importance, direction} dicts.
    """
    import importlib.util
    
    # Load model
    if not os.path.exists(model_path) or not os.path.exists(model_module_path) or not os.path.exists(data_path):
        return None
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_dim = checkpoint["input_dim"]
    seq_len = checkpoint["seq_len"]
    hidden_dim = checkpoint.get("hidden_dim", 16)
    
    spec = importlib.util.spec_from_file_location("qlstm_model", model_module_path)
    qlstm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qlstm_mod)
    QLSTMVolatilityPredictor = qlstm_mod.QLSTMVolatilityPredictor
    
    model = QLSTMVolatilityPredictor(input_dim, hidden_dim=hidden_dim, seq_len=seq_len)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load & normalize data
    with open(data_path) as f:
        data = json.load(f)
    feats = data.get("features", [])
    if len(feats) < seq_len + 2:
        return None
    
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
    
    # Use latest 500 samples for permutation (or all if less)
    n_samples = min(500, len(features_norm) - seq_len)
    start_idx = len(features_norm) - seq_len - n_samples
    
    # Build all windows
    all_windows = []
    for i in range(start_idx, len(features_norm) - seq_len):
        all_windows.append(features_norm[i:i+seq_len])
    all_windows = np.array(all_windows)  # (n_windows, seq_len, n_features)
    win_tensor = torch.tensor(all_windows, dtype=torch.float32)
    
    # Baseline predictions
    with torch.no_grad():
        baseline = model(win_tensor).numpy()  # (n_windows,)
    
    actual_input_dim = min(input_dim, features_norm.shape[1])
    
    # Permutation importance per feature
    results = []
    for feat_idx in range(min(actual_input_dim, len(FEATURE_NAMES))):
        win_perm = all_windows.copy()
        # Shuffle across time and samples
        perm_idx = np.random.permutation(win_perm.shape[0])
        win_perm[:, :, feat_idx] = win_perm[perm_idx, :, feat_idx]
        
        perm_tensor = torch.tensor(win_perm, dtype=torch.float32)
        with torch.no_grad():
            perm_preds = model(perm_tensor).numpy()
        
        # Importance = increase in MSE when feature is shuffled
        mse_baseline = np.mean((baseline - baseline.mean())**2)
        mse_permuted = np.mean((perm_preds - perm_preds.mean())**2)
        importance = mse_permuted - mse_baseline
        
        # Direction: correlation between feature value and prediction delta
        feat_values = all_windows[:, -1, feat_idx]  # last timestep
        direction = np.corrcoef(feat_values, perm_preds - baseline)[0, 1] if len(feat_values) > 1 else 0
        
        results.append({
            "idx": feat_idx,
            "name": FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"F{feat_idx+1}",
            "importance": float(importance),
            "direction": "positive" if direction > 0.05 else "negative" if direction < -0.05 else "neutral",
            "corr": float(direction)
        })
    
    # Sort by absolute importance, take top_n
    results.sort(key=lambda r: abs(r["importance"]), reverse=True)
    top = results[:top_n]
    
    # Normalize importance to percentages
    total_imp = sum(abs(r["importance"]) for r in top)
    if total_imp > 0:
        for r in top:
            r["importance_pct"] = round(abs(r["importance"]) / total_imp * 100, 1)
    else:
        for r in top:
            r["importance_pct"] = 0
    
    return top


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sfc_dir = os.path.dirname(script_dir)  # /home/ubuntu
    data_path = os.path.join(sfc_dir, "sfc", "data_collection.json")
    model_path = os.path.join(script_dir, "qlstm_model.pt")
    model_module = os.path.join(script_dir, "qlstm_model.py")
    
    features = get_top_features(data_path, model_path, model_module)
    if features:
        print(f"\n  Top {len(features)} Features by Permutation Importance:")
        print(f"  {'#':<3} {'Feature':<18} {'Importance':<12} {'Direction':<12}")
        print(f"  {'─'*45}")
        for i, f in enumerate(features, 1):
            print(f"  {i:<3} {f['name']:<18} {f['importance_pct']:<10.1f}% {f['direction']:<12}")
        
        out_path = os.path.join(script_dir, "xai_features.json")
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"\n  Saved: {out_path}")
    else:
        print("  XAI: No data or model available")
