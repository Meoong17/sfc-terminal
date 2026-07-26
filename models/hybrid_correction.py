#!/usr/bin/env python3
"""
Hybrid Correction: QLSTM + GARCH(1,1) residual correction
==========================================================
Loads trained QLSTM model, computes residuals against ensemble SFC,
fits GARCH(1,1) on residuals, and produces corrected forecast.
"""

import json
import os
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
# FIX (2026-07, per SFC_Path_Inventory.docx diagnosis): was hardcoded to
# /home/ubuntu/sfc2/qlstm_model.pt — a STALE, different-architecture model
# file (20,040 bytes, Jun 11) in an inactive old project directory, not
# the one collect.py actually trains/uses (/home/ubuntu/sfc/qlstm_model.pt,
# 19,784 bytes, Jul 16). This line was never actually being executed
# before the sys.path fix in qlstm_enhanced.py (the import failed first),
# so this bug was previously masked — fixing that import without also
# fixing this would have introduced a NEW problem (loading the wrong
# model). Both fixed together.
MODEL_PATH = "/home/ubuntu/sfc/qlstm_model.pt"
DATA_PATH = "/home/ubuntu/sfc/data_collection.json"

# QIGWO weights (same as in training)
QIGWO_WEIGHTS = np.array([0.19, 0.16, 0.12, 0.16, 0.24, 0.14])

# Model hyperparams — NOTE: input_dim, seq_len, hidden_dim are OVERRIDDEN
# at load time from the checkpoint (the source of truth). The defaults below
# are fallbacks only; the actual values come from the saved .pt file.
DEFAULT_HIDDEN_DIM = 16
DEFAULT_N_QUBITS = 4
DEFAULT_N_QLAYERS = 2


def load_model():
    """Load the QLSTM model architecture with trained weights.
    Reads input_dim / seq_len / hidden_dim from the checkpoint itself
    so the architecture always matches (fix: was hardcoded to INPUT_DIM=31
    when checkpoint was saved with input_dim=30 → state_dict mismatch)."""
    # Import here so module can be imported without triggering pennylane init
    from qlstm_model import QLSTMVolatilityPredictor

    device = torch.device("cpu")

    # Load checkpoint first — it IS the source of truth for dimensions
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = QLSTMVolatilityPredictor(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt.get("hidden_dim", DEFAULT_HIDDEN_DIM),
        seq_len=ckpt["seq_len"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"  [hybrid] Loaded QLSTM model | input_dim={ckpt['input_dim']}"
          f" seq_len={ckpt['seq_len']}"
          f" hidden_dim={ckpt.get('hidden_dim', DEFAULT_HIDDEN_DIM)}"
          f" | val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model, device, ckpt["input_dim"], ckpt["seq_len"]


def load_features():
    """Load feature array from data_collection.json."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    features_raw = data["features"]
    # Pad / unify feature lengths to max length
    max_len = max(len(f) for f in features_raw)
    features = []
    for f in features_raw:
        arr = np.array(f, dtype=np.float32)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), 0.5, dtype=np.float32)])
        features.append(arr)
    features = np.array(features)
    print(f"  [hybrid] Loaded features: {features.shape[0]} samples, {features.shape[1]} features")
    return features


def normalize_features(features):
    """Normalize features using mean/std (same as training)."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std, mean, std


def compute_ensemble_target(features_raw, idx):
    """Compute the QIGWO-weighted ensemble SFC for sample at index idx.
    Uses the ORIGINAL (non-normalized) features.
    """
    return float(features_raw[idx, :6] @ QIGWO_WEIGHTS * 100.0)


def run_hybrid_correction():
    """
    Main pipeline:
      1. Load model + data
      2. Normalize features
      3. Predict QLSTM for all historical windows (seq_len .. N)
      4. Compute residuals (target - QLSTM_pred)
      5. Fit GARCH(1,1) on residual series
      6. Predict next QLSTM + next GARCH residual
      7. Return results dict
    """
    print("\n[hybrid] === QLSTM + GARCH Hybrid Correction ===\n")

    # ── Load model ──
    model, device, input_dim, seq_len = load_model()

    # ── Load features ──
    features_raw = load_features()  # original, non-normalized
    N = features_raw.shape[0]

    if N < seq_len + 1:
        raise ValueError(f"Not enough samples ({N}) for seq_len={seq_len}")

    features_norm, mean, std = normalize_features(features_raw)

    # ── Historical QLSTM predictions ──
    qlstm_preds = []
    targets = []

    print(f"  [hybrid] Computing {N - seq_len} historical predictions...")
    with torch.no_grad():
        for i in range(seq_len, N):
            # Window of normalized features
            window = features_norm[i - seq_len : i]  # (seq_len, input_dim)
            inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)
            inp = inp.to(device)

            pred = model(inp).item()  # float (0-100)
            tgt = compute_ensemble_target(features_raw, i)

            qlstm_preds.append(pred)
            targets.append(tgt)

    qlstm_preds = np.array(qlstm_preds)  # shape: (N - seq_len,)
    targets = np.array(targets)

    # ── Residuals: actual - QLSTM_pred ──
    residuals = targets - qlstm_preds

    # ── GARCH(1,1) on residuals ──
    from arch import arch_model

    print(f"  [hybrid] Fitting GARCH(1,1) on {len(residuals)} residuals...")
    am = arch_model(residuals, vol="Garch", p=1, q=1, dist="normal")
    garch_res = am.fit(disp="off", update_freq=0)
    print(f"  [hybrid] GARCH fitted | AIC={garch_res.aic:.2f} | BIC={garch_res.bic:.2f}")

    # Forecast next residual (1 step ahead)
    forecast = garch_res.forecast(horizon=1)
    next_garch_residual = float(forecast.mean.values[-1, 0])
    next_garch_volatility = float(np.sqrt(forecast.variance.values[-1, 0]))

    # ── Final prediction for next step ──
    # The "next" QLSTM prediction uses the LAST seq_len window
    # (indices N-seq_len .. N-1), which predicts target at index N
    last_window = features_norm[N - seq_len : N]  # (seq_len, input_dim)
    inp_last = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        next_qlstm_pred = model(inp_last).item()

    final_pred = next_qlstm_pred + next_garch_residual

    # ── Build result ──
    result = {
        "qlstm_pred": round(next_qlstm_pred, 4),
        "garch_residual": round(next_garch_residual, 4),
        "final_pred": round(final_pred, 4),
        "garch_volatility": round(next_garch_volatility, 4),
        "residual_history": [round(float(r), 4) for r in residuals],
    }

    print(f"\n  [hybrid] Next QLSTM pred:  {result['qlstm_pred']:.4f}")
    print(f"  [hybrid] GARCH residual:   {result['garch_residual']:.4f}")

    # Clamp to [0, 100]
    result["final_pred_raw"] = result["final_pred"]
    result["final_pred"] = round(max(0.0, min(100.0, result["final_pred"])), 4)
    print(f"  [hybrid] Final prediction: {result['final_pred']:.4f} "
          f"(clamped: {result['final_pred_raw'] != result['final_pred']})")
    print(f"  [hybrid] GARCH volatility: {result['garch_volatility']:.4f}")
    print(f"  [hybrid] Residual history: {len(result['residual_history'])} points"
          f"  [{min(residuals):.2f}, {max(residuals):.2f}]")
    print()

    return result


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    result = run_hybrid_correction()
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hybrid_correction_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [hybrid] Result saved to {out_path}")
