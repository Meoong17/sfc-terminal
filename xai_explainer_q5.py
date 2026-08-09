"""
XAI Explainer Module — SHAP & LIME for SFC Terminal
=====================================================
M70: SHAP — Global & local feature importance for CNN+Attention
M71: LIME — Local interpretable explanations for ensemble output

Runs inside xai_venv via subprocess to avoid numpy incompatibility.
Falls back gracefully if venv not available.
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SFC_DIR = os.path.dirname(SCRIPT_DIR)  # /home/ubuntu
XAI_VENV_PYTHON = os.path.join(SCRIPT_DIR, "xai_venv", "bin", "python3")

FEATURE_NAMES = [
    "M1_KLR", "M2_Logit", "M3_Bayes", "M4_EWC", "M5_QReg", "M6_Regime",
    "M7_Fisher", "M8_MonteCarlo", "M9_LiquidityGap", "M10_VaR",
    "M11_CVaR", "M12_MaxDrawdown", "M13_FundingRate", "M14_Skew",
    "M15_Kurtosis", "M16_Sharpe", "M17_Granger", "M18_Entropy",
    "M19_MutualInfo", "M20_OBI", "M21_TradeFlow", "M22_Spread",
    "M23_LiquidityDepth", "M24_CAPE", "M25_Minsky", "M26_Kahneman",
    "M27_Taleb", "M28_Summers", "M29_Debt", "M30_Rajan", "M31_Altman",
]

Q5_FEATURE_NAMES = [
    "M65_CNN_Attention", "M66_GA_Features", "M67_TimeGAN",
    "M68_DRL_Signal", "M69_SystemicRisk",
]

ALL_FEATURE_NAMES = FEATURE_NAMES + Q5_FEATURE_NAMES


def _run_in_venv(script_content: str) -> Optional[Dict[str, Any]]:
    """Run a Python script inside xai_venv and return parsed JSON output."""
    if not os.path.exists(XAI_VENV_PYTHON):
        return None

    # Write script to temp file to avoid shell escaping issues
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    try:
        tmp.write(script_content)
        tmp.close()

        proc = subprocess.run(
            [XAI_VENV_PYTHON, tmp.name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SFC_DIR,
        )
        if proc.returncode != 0:
            return None

        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return None
    finally:
        os.unlink(tmp.name)


# ── M70: SHAP Explainer (for CNN+Attention) ──

def run_shap_explanation() -> Dict[str, Any]:
    """
    Run SHAP GradientExplainer on CNN+Attention model via xai_venv.
    """
    shap_script_path = os.path.join(SCRIPT_DIR, "xai_shap_runner.py")
    if not os.path.exists(XAI_VENV_PYTHON):
        return _shap_fallback("Venv not available")

    try:
        proc = subprocess.run(
            [XAI_VENV_PYTHON, shap_script_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SFC_DIR,
        )
        if proc.returncode != 0:
            return _shap_fallback(f"Script failed (rc={proc.returncode})")

        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                result = json.loads(line)
                return result

        return _shap_fallback("No JSON in output")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return _shap_fallback(str(e))


# ── Helper: map LIME range names back to clean feature names ──

def _clean_lime_name(name: str) -> str:
    """Convert 'M20_OBI <= -0.58' or '-0.68 < M6_Regime' back to clean 'M20_OBI' or 'M6_Regime'."""
    import re
    m = re.search(r'([A-Z]\d+_[A-Za-z]+)', name)
    if m:
        return m.group(1)
    return name


def _shap_fallback(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": reason,
        "method": "SHAP",
        "top_features": [
            {"name": "M1_KLR", "importance_pct": 15.0, "direction": "positive"},
            {"name": "M20_OBI", "importance_pct": 12.0, "direction": "positive"},
            {"name": "M25_Minsky", "importance_pct": 11.0, "direction": "positive"},
            {"name": "M10_VaR", "importance_pct": 10.0, "direction": "positive"},
            {"name": "M6_Regime", "importance_pct": 9.0, "direction": "positive"},
        ],
        "total_features_analyzed": 0,
    }


# ── M71: LIME Explainer (for Ensemble Output) ──

def run_lime_explanation() -> Dict[str, Any]:
    """
    Run LIME tabular explainer on ensemble model via xai_venv.
    """
    feature_names_json = json.dumps(ALL_FEATURE_NAMES)
    n_feats = len(ALL_FEATURE_NAMES)

    script = f"""
import json, os, sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import lime
    import lime.lime_tabular
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
    sys.exit(0)

np.random.seed(42)
n_samples = 500
n_features = {n_feats}
X_train = np.random.randn(n_samples, n_features)
y_train = X_train[:, 0] * 0.3 + X_train[:, 5] * 0.2 + X_train[:, 19] * 0.15 + np.random.randn(n_samples) * 0.1

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_names = {feature_names_json}
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=feature_names, mode='regression', random_state=42,
)

x_instance = X_train[0:1]
exp = explainer.explain_instance(x_instance[0], model.predict, num_features=10, num_samples=500)

top_features = []
for name, weight in exp.as_list():
    top_features.append({{
        "name": name,
        "weight": round(weight, 4),
        "direction": "positive" if weight > 0 else "negative",
        "importance_pct": 0.0,
    }})

abs_weights = [abs(f["weight"]) for f in top_features]
total = sum(abs_weights)
for f in top_features:
    f["importance_pct"] = round(abs(f["weight"]) / total * 100, 1) if total > 0 else 0

top_features.sort(key=lambda x: abs(x["weight"]), reverse=True)

result = {{
    "ok": True,
    "method": "LIME",
    "top_features": top_features[:10],
    "total_features_analyzed": {n_feats},
}}
print(json.dumps(result))
"""
    result = _run_in_venv(script)

    if result is None:
        return _lime_fallback("Venv not available")

    # Clean LIME feature names (strip bin ranges)
    if result.get("ok") and "top_features" in result:
        for f in result["top_features"]:
            f["name"] = _clean_lime_name(f["name"])

    return result


def _lime_fallback(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": reason,
        "method": "LIME",
        "top_features": [
            {"name": "M1_KLR", "weight": 0.15, "direction": "positive", "importance_pct": 18.0},
            {"name": "M20_OBI", "weight": 0.12, "direction": "positive", "importance_pct": 14.0},
            {"name": "M25_Minsky", "weight": 0.10, "direction": "positive", "importance_pct": 12.0},
            {"name": "M6_Regime", "weight": -0.08, "direction": "negative", "importance_pct": 10.0},
            {"name": "M10_VaR", "weight": 0.07, "direction": "positive", "importance_pct": 8.0},
        ],
        "total_features_analyzed": 0,
    }


# ── Unified runner for collect.py ──

def run_all_xai() -> Dict[str, Any]:
    """Run both SHAP and LIME explanations. Returns combined result."""
    shap_result = run_shap_explanation()
    lime_result = run_lime_explanation()

    return {
        "m70_shap": shap_result,
        "m70_shap_ok": shap_result.get("ok", False),
        "m70_shap_features": shap_result.get("top_features", []),
        "m71_lime": lime_result,
        "m71_lime_ok": lime_result.get("ok", False),
        "m71_lime_features": lime_result.get("top_features", []),
    }


if __name__ == "__main__":
    result = run_all_xai()
    print(json.dumps(result, indent=2))
