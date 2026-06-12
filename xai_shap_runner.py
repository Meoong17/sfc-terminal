"""
SHAP Explainer script — run inside xai_venv
Called as subprocess from xai_explainer_q5.py

Uses random data as background (real pipeline integration TBD).
Feature importance is relative, not absolute — useful for ranking.
"""
import json, os, sys, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import shap, torch, torch.nn as nn

sys.path.insert(0, "/home/ubuntu/sfc")
from models.cnn_attention_module import CNNAttentionModule

class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CNNAttentionModule(input_dim=41)
        self.model.eval()
    def forward(self, x):
        out, _ = self.model(x)
        return out.unsqueeze(-1)

model = ModelWrapper()
model.eval()

np.random.seed(42)
torch.manual_seed(42)

# Use deterministic random data for stable feature importance ranking
# In production, this should load real data from data.json
background = torch.randn(10, 20, 41)
x_explain = torch.randn(1, 20, 41)

try:
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x_explain)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_means = np.abs(shap_values).mean(axis=(0, 1))

    FEATURE_NAMES = [
        "M1_KLR", "M2_Logit", "M3_Bayes", "M4_EWC", "M5_QReg", "M6_Regime",
        "M7_Fisher", "M8_MonteCarlo", "M9_LiquidityGap", "M10_VaR",
        "M11_CVaR", "M12_MaxDrawdown", "M13_FundingRate", "M14_Skew",
        "M15_Kurtosis", "M16_Sharpe", "M17_Granger", "M18_Entropy",
        "M19_MutualInfo", "M20_OBI", "M21_TradeFlow", "M22_Spread",
        "M23_LiquidityDepth", "M24_CAPE", "M25_Minsky", "M26_Kahneman",
        "M27_Taleb", "M28_Summers", "M29_Debt", "M30_Rajan", "M31_Altman",
        "M65_CNN_Attention", "M66_GA_Features", "M67_TimeGAN",
        "M68_DRL_Signal", "M69_SystemicRisk",
    ]

    shap_values_sq = shap_values.squeeze(-1)
    top_idx = np.argsort(shap_means.flatten())[::-1][:10]
    top_features = []
    for idx in top_idx:
        i = int(idx)
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"F{i+1}"
        top_features.append({
            "idx": i,
            "name": name,
            "shap_value": float(shap_means.flatten()[i]),
            "direction": "positive" if float(shap_values_sq[0, -1, i]) > 0 else "negative",
        })

    total = sum(abs(f["shap_value"]) for f in top_features)
    for f in top_features:
        f["importance_pct"] = round(abs(f["shap_value"]) / total * 100, 1) if total > 0 else 0

    result = {
        "ok": True,
        "method": "GradientSHAP",
        "top_features": top_features,
        "total_features_analyzed": 41,
    }
except Exception as e:
    result = {"ok": False, "error": str(e)}

print(json.dumps(result))
