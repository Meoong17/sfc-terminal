"""
CNN + Attention module for pattern recognition (SFC signal enhancement).

Implements Teknik 1 from Q5:
  Conv1D feature extraction -> MultiheadAttention -> scalar output (stress / 0-1)

Classes:
    CNNAttentionModule : 1D CNN + MultiheadAttention encoder
    SFCEnhancedModel    : Wrapper combining CNN+Attention with QLSTM placeholder
    calculate_cnn_attention_stress : Standalone inference helper
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Torch is optional — fallback gracefully if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ---------------------------------------------------------------------------
# CNNAttentionModule (requires torch)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class CNNAttentionModule(nn.Module):
        """
        1D CNN + MultiheadAttention pattern-recognition module.

        Architecture
        ------------
            input (B, T, 41)  ---conv1--->  (B, 128, T)  ---conv2--->  (B, 64, T)
                ---permute--->  (B, T, 64)  ---MHSA--->  (B, T, 64)
                ---last-timestep--->  (B, 64)  ---linear--->  (B, 1)

        Returns
        -------
            output          : (B,)  - sigmoid-activated scalar per sample
            attention_weights : (B, H, T, T) or None if no MHSA output
        """

        def __init__(
            self,
            input_dim: int = 41,
            conv1_channels: int = 128,
            conv2_channels: int = 64,
            kernel_size: int = 3,
            dropout_cnn: float = 0.3,
            embed_dim: int = 64,
            num_heads: int = 4,
            dropout_attn: float = 0.2,
        ) -> None:
            super().__init__()

            # --- CNN towers (1D) ------------------------------------------------
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=conv1_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_cnn),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=conv1_channels,
                    out_channels=conv2_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(),
                nn.Dropout(dropout_cnn),
            )

            # --- Multihead Self-Attention ---------------------------------------
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout_attn,
            )

            # --- Output projection ----------------------------------------------
            self.fc_out = nn.Linear(embed_dim, 1)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # x: (B, T, C) -> (B, C, T) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.conv1(x)  # (B, conv1_channels, T)
            x = self.conv2(x)  # (B, conv2_channels, T)
            x = x.permute(0, 2, 1)  # (B, T, embed_dim)

            attn_output, attn_weights = self.attn(x, x, x)
            last_step = attn_output[:, -1, :]  # (B, embed_dim)
            raw = self.fc_out(last_step)  # (B, 1)
            output = torch.sigmoid(raw).squeeze(-1)  # (B,)
            return output, attn_weights

    # ---------------------------------------------------------------------------
    # SFCEnhancedModel
    # ---------------------------------------------------------------------------

    class SFCEnhancedModel(nn.Module):
        """
        Higher-level SFC model combining CNN+Attention with a placeholder for
        QLSTM (quantum LSTM).  Currently only the CNN+Attention branch is wired.
        """

        def __init__(self, input_dim: int = 41, cnn_kwargs: Optional[Dict[str, Any]] = None) -> None:
            super().__init__()
            cnn_kwargs = cnn_kwargs or {}
            self.cnn_attn = CNNAttentionModule(input_dim=input_dim, **cnn_kwargs)
            self.qlstm_placeholder: nn.Module = nn.Identity()
            self.combine = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cnn_out, _ = self.cnn_attn(x)  # (B,)
            qlstm_out = x[:, -1, :].mean(dim=-1)  # (B,)
            combined = torch.stack([cnn_out, qlstm_out], dim=-1)  # (B, 2)
            stress = torch.sigmoid(self.combine(combined)).squeeze(-1)  # (B,)
            return stress


# ---------------------------------------------------------------------------
# Fallback when torch is not available
# ---------------------------------------------------------------------------

if not TORCH_AVAILABLE:
    class CNNAttentionModule:
        def __init__(self, *a, **k):
            raise RuntimeError("PyTorch is required for CNNAttentionModule")

    class SFCEnhancedModel:
        def __init__(self, *a, **k):
            raise RuntimeError("PyTorch is required for SFCEnhancedModel")


# ---------------------------------------------------------------------------
# calculate_cnn_attention_stress  -  stand-alone inference helper
# ---------------------------------------------------------------------------

def calculate_cnn_attention_stress(
    data_window: Union[List[Dict[str, Any]], np.ndarray],
    model_path: str = "",
    device=None,
    input_dim: int = 41,
    seq_len: int = 60,
) -> Dict[str, Any]:
    """
    Run inference with CNNAttentionModule on a window of market data.
    Falls back gracefully if PyTorch is not available.
    """
    if not TORCH_AVAILABLE:
        return _fallback_result("FALLBACK - PyTorch not installed")

    if device is None:
        device = torch.device("cpu")

    # --- Convert input to numpy array ---------------------------------------
    if isinstance(data_window, list):
        try:
            arr = np.array([list(d.values()) for d in data_window], dtype=np.float32)
        except (AttributeError, ValueError, TypeError):
            arr = np.array(data_window, dtype=np.float32)
    elif isinstance(data_window, np.ndarray):
        arr = data_window.astype(np.float32)
    else:
        return _fallback_result("FALLBACK - invalid input type")

    # --- Shape validation ---------------------------------------------------
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    while arr.ndim > 3:
        arr = arr.squeeze(-2) if arr.shape[-2] == 1 else arr

    B, T, C = arr.shape

    # Pad or truncate time dimension
    if T < seq_len:
        pad = np.zeros((B, seq_len - T, C), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif T > seq_len:
        arr = arr[:, :seq_len, :]

    # Pad or truncate feature dimension
    if C > input_dim:
        arr = arr[:, :, :input_dim]
    elif C < input_dim:
        pad = np.zeros((B, seq_len, input_dim - C), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)

    # --- Try loading model --------------------------------------------------
    try:
        model = CNNAttentionModule(input_dim=input_dim).to(device)
        model.eval()

        if model_path and model_path.strip():
            try:
                state = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state, strict=False)
            except (FileNotFoundError, RuntimeError, ValueError):
                pass

        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(device)
            output, attn_weights = model(tensor)
            stress_val = float(output[0].cpu().item())

        if attn_weights is not None:
            attn_focus = attn_weights[0].mean(dim=0).cpu().numpy()
            focus_flat = attn_focus[-1, :].tolist()
        else:
            focus_flat = []

        if stress_val > 0.7:
            pattern = "HIGH_STRESS - strong divergence / trend exhaustion"
        elif stress_val > 0.5:
            pattern = "MODERATE_STRESS - developing pattern"
        elif stress_val > 0.3:
            pattern = "LOW_STRESS - normal market noise"
        else:
            pattern = "NO_STRESS - stable conditions"

        return {
            "m65_cnn_attention": round(stress_val, 6),
            "attention_focus": focus_flat,
            "pattern_type": pattern,
        }

    except Exception as exc:
        return _fallback_result(f"FALLBACK - inference error: {exc}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fallback_result(reason: str = "FALLBACK") -> Dict[str, Any]:
    """Return a safe fallback dict when the model cannot be loaded or run."""
    return {
        "m65_cnn_attention": 0.5,
        "attention_focus": [],
        "pattern_type": reason,
    }
