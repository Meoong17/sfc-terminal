# mamba_encoder.py
# Mamba State-Space Model Encoder untuk SFC Terminal
# Pure PyTorch implementation (CPU-friendly, no CUDA/Triton needed)
# Replaces/enriches QLSTM (M32) with efficient state-space modeling

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import sys
import numpy as np
from pathlib import Path
from einops import rearrange, repeat
from collections import OrderedDict
from datetime import datetime, timezone

# ================================================================
# 1. SELECTIVE SCAN KERNEL (Pure PyTorch, No CUDA needed)
# Based on mamba-minimal by johnma2006
# ================================================================
def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """
    Selective scan implementation for inference (CPU-friendly).
    
    Args:
        u: (batch, dim, seq_len) — input sequence
        delta: (batch, dim, seq_len) — step size
        A: (dim, dstate) — state transition
        B: (batch, dim, seq_len, dstate) — input projection
        C: (batch, dim, seq_len, dstate) — output projection
        D: (dim,) — skip connection
    Returns:
        y: (batch, dim, seq_len) — output
    """
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Initialize state
    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=u.dtype)
    ys = []
    
    delta = delta.unsqueeze(-1)  # (batch, dim, seq_len, 1)
    
    # A discretization
    A = A.unsqueeze(0).expand(batch, -1, -1)  # (batch, dim, dstate)
    
    # Loop over sequence length (time steps)
    for i in range(u.shape[2]):
        # Discretize A: dA = exp(delta * A)
        dA = torch.exp(delta[:, :, i, :] * A)  # (batch, dim, dstate)
        
        # Discretize B: dB = delta * B
        dB = delta[:, :, i, :] * B[:, :, i, :]  # (batch, dim, dstate)
        
        # Update state: x = x * dA + dB * u
        x = x * dA + dB * u[:, :, i].unsqueeze(-1)
        
        # Compute output: y = (x * C).sum(dim=-1)
        y = (x * C[:, :, i, :]).sum(dim=-1)
        ys.append(y)
    
    y = torch.stack(ys, dim=2)  # (batch, dim, seq_len)
    
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    
    return y


# ================================================================
# 2. MAMBA BLOCK
# ================================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        """
        Mamba state-space block.
        
        Args:
            d_model: Input dimension (e.g., number of features)
            d_state: SSM state dimension (default 16)
            d_conv: Convolution kernel size
            expand_factor: Expansion factor for inner dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand_factor * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM projection (for A, B, C, delta)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Delta projection
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize A as negative-log-space (1, 2, ..., d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Initialize dt bias
        self.dt_bias = nn.Parameter(torch.ones(self.d_inner))
        
        # Initialize conv bias to zero
        self.conv1d.bias.data.zero_()
        
        # Initialize dt_proj weight with small values
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # 1. Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution with SiLU
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]  # remove padding
        x = rearrange(x, 'b d l -> b l d')
        
        # 3. SiLU activation
        x = F.silu(x)
        z = F.silu(z)
        
        # 4. SSM parameters
        x_dbl = self.x_proj(x)  # (batch, seq_len, d_state*2 + 1)
        delta_raw, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        # delta_raw: (batch, seq_len, 1) — we need to expand to d_inner
        delta_raw = delta_raw.squeeze(-1)  # (batch, seq_len)
        # Expand delta to d_inner using dt_proj
        # dt_proj maps from d_state-dim space to d_inner
        # We'll use a simpler approach: repeat and scale
        delta = delta_raw.unsqueeze(-1).expand(-1, -1, self.d_inner)  # (batch, seq_len, d_inner)
        delta = delta + self.dt_bias.unsqueeze(0).unsqueeze(0)
        delta = F.softplus(delta)
        delta = delta.permute(0, 2, 1)  # (batch, d_inner, seq_len)
        
        # B, C: (batch, seq_len, d_state) -> (batch, d_inner, seq_len, d_state)
        B = B.unsqueeze(1).expand(-1, self.d_inner, -1, -1)
        C = C.unsqueeze(1).expand(-1, self.d_inner, -1, -1)
        
        # A: (d_inner, d_state) — negative exponential of A_log
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # x: (batch, d_inner, seq_len)
        x = x.permute(0, 2, 1)  # (batch, d_inner, seq_len)
        
        # 5. Selective scan
        y = selective_scan(
            u=x,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            delta_bias=None,  # already applied above
            delta_softplus=False  # already applied above
        )  # (batch, d_inner, seq_len)
        
        # 6. Output gate (multiplication with z)
        y = rearrange(y, 'b d l -> b l d')
        y = y * z
        
        # 7. Output projection
        out = self.out_proj(y)  # (batch, seq_len, d_model)
        
        return out


# ================================================================
# 3. MAMBA ENCODER — Generates multi-horizon stress prediction
# ================================================================
class MambaEncoder(nn.Module):
    """
    Mamba encoder for SFC Terminal.
    Processes sequential features and outputs multi-horizon predictions.
    
    Input: (batch, seq_len, input_dim) — feature sequence
    Output: dict with:
        - stress_short: near-term stress (0-1)
        - stress_medium: medium-term stress (0-1)
        - stress_long: long-term stress (0-1)
        - combined: weighted combined stress (0-1)
    """
    
    def __init__(self, input_dim=64, d_model=128, d_state=16, d_conv=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Multi-head output: short, medium, long-term stress
        self.stress_head = nn.Linear(d_model, 3)  # [short, medium, long]
        
        # Confidence head
        self.confidence_head = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'conv' not in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — feature sequence
        Returns:
            dict with stress predictions
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Pooling: take mean over sequence length
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        
        # Multi-horizon stress predictions
        stress_logits = self.stress_head(x_pooled)  # (batch, 3)
        stress = torch.sigmoid(stress_logits)  # (batch, 3) — 0 to 1
        
        # Confidence
        confidence = torch.sigmoid(self.confidence_head(x_pooled))  # (batch, 1)
        
        # Combined: weighted average (short=0.5, medium=0.3, long=0.2)
        weights = torch.tensor([0.5, 0.3, 0.2], device=x.device)
        combined = (stress * weights.unsqueeze(0)).sum(dim=1, keepdim=True)  # (batch, 1)
        
        return {
            'stress_short': stress[:, 0:1],
            'stress_medium': stress[:, 1:2],
            'stress_long': stress[:, 2:3],
            'combined': combined,
            'confidence': confidence,
        }


# ================================================================
# 4. FEATURE ENGINEERING — Build feature vectors from SFC data
# ================================================================
def build_feature_vector(data):
    """
    Build a feature vector from a single data snapshot.
    Returns numpy array of shape (n_features,).
    
    Extracts ~30 key features from SFC data.json:
    - Price action: btc_price, btc_24h_change
    - Technical: rsi_14, dvol, pc_oi, pc_vol
    - On-chain: sopr, cascade_risk, liq_density, liq_pressure
    - Macro: m2_yoy, dxy, fng
    - SFC state: sfc_base, sfc_effective, zone_numeric
    - Factors: Lt, St, Rt, Ft, Sc
    - Methods: m1-m6 scores
    - Institutional: signal strength, timing precision
    """
    features = []
    
    # Helper: safe float extraction
    def get_val(d, *keys, default=0.0):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] is not None:
                try:
                    return float(d[k])
                except (ValueError, TypeError):
                    pass
        return default
    
    # 1. Price & Market (5 features)
    features.append(get_val(data, 'btc') / 100000.0)  # normalize BTC ~0.6
    features.append(get_val(data, 'btc_24h') / 20.0)  # normalize to [-1, 1]
    features.append(get_val(data, 'btc_mcap', 'market_cap') / 2e12)  # normalize
    features.append(get_val(data, 'dom') / 100.0)  # dominance 0-1
    features.append(get_val(data, 'dvol') / 100.0)  # DVOL 0-1
    
    # 2. Technical Indicators (5 features)
    features.append(get_val(data, 'rsi_14') / 100.0)  # RSI 0-1
    features.append(get_val(data, 'pc_oi'))
    features.append(get_val(data, 'pc_vol'))
    features.append(get_val(data, 'fng') / 100.0)  # Fear & Greed 0-1
    features.append(get_val(data, 'sopr_proxy'))
    
    # 3. On-chain & Risk (5 features)
    features.append(get_val(data, 'cascade_risk'))
    features.append(get_val(data, 'liq_density'))
    features.append(get_val(data, 'liq_mod') / 5.0)  # normalize
    features.append(get_val(data, 'sopr_score'))
    features.append(get_val(data, 'regime_prob'))
    
    # 4. Macro (4 features)
    features.append(get_val(data, 'm2_yoy') / 20.0)  # normalize
    features.append(get_val(data, 'dxy') / 120.0)  # normalize
    features.append(get_val(data, 'transition_risk'))
    features.append(get_val(data, 'dv_sfc'))
    
    # 5. SFC State (5 features)
    features.append(get_val(data, 'sfc_base') / 100.0)
    features.append(get_val(data, 'sfc_effective') / 100.0)
    
    # Zone encoding: NORMAL=0, ELEVATED=1, HIGH=2, CRITICAL=3
    zone_map = {'NORMAL': 0.0, 'ELEVATED': 0.33, 'HIGH': 0.66, 'CRITICAL': 1.0}
    features.append(zone_map.get(str(get_val(data, 'zone', default='NORMAL')), 0.0))
    
    # Regime encoding
    regime_map = {'BULL': 0.0, 'NORMAL': 0.25, 'STRESS': 0.5, 'CAPITULATION': 0.75, 'CRISIS': 1.0}
    features.append(regime_map.get(str(get_val(data, 'regime', default='NORMAL')), 0.25))
    
    features.append(get_val(data, 'phi'))
    
    # 6. SFC Factors (5 features)
    factors = data.get('factors', {})
    features.append(max(-3.0, min(3.0, get_val(factors, 'Lt'))) / 3.0)  # Lt
    features.append(max(-3.0, min(3.0, get_val(factors, 'St'))) / 3.0)  # St
    features.append(max(-3.0, min(3.0, get_val(factors, 'Rt'))) / 3.0)  # Rt
    features.append(max(-3.0, min(3.0, get_val(factors, 'Ft'))) / 3.0)  # Ft
    features.append(max(-3.0, min(3.0, get_val(factors, 'Sc'))) / 3.0)  # Sc
    
    # 7. Method Ensemble (6 features)
    features.append(get_val(data, 'method_agreement'))
    features.append(get_val(data, 'composite_confidence'))
    features.append(get_val(data, 'm1_klr') / 10.0)
    features.append(get_val(data, 'm2_logit') / 10.0)
    features.append(get_val(data, 'm4_ewc') / 10.0)
    features.append(get_val(data, 'm5_qreg') / 10.0)
    
    # 8. Q10 On-Chain (4 features) — available after Q10 integration
    features.append(get_val(data, 'q10_whale_pressure') / 100.0)
    features.append(get_val(data, 'q10_onchain_value') / 100.0)
    features.append(get_val(data, 'q10_buying_power') / 100.0)
    features.append(get_val(data, 'q10_market_structure') / 100.0)
    
    return np.array(features, dtype=np.float32)


def load_daily_cache_features(cache_path='/home/ubuntu/sfc/.daily_market_cache.json'):
    """
    Load historical daily cache entries as feature sequence.
    Returns numpy array of shape (n_days, n_features).
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    
    if not isinstance(cache, list) or len(cache) == 0:
        return None
    
    features_list = []
    for entry in cache:
        vec = build_feature_vector(entry)
        features_list.append(vec)
    
    return np.array(features_list, dtype=np.float32)


def build_feature_sequence(data, seq_len=30):
    """
    Build a complete feature sequence for Mamba inference.
    
    Strategy:
    1. Use daily cache entries for historical data (up to seq_len)
    2. Pad with replicated current snapshot if not enough history
    
    Returns:
        numpy array of shape (seq_len, n_features)
    """
    # Build current feature vector
    current_vec = build_feature_vector(data)
    n_features = len(current_vec)
    
    # Try loading historical cache
    historical = load_daily_cache_features()
    
    if historical is not None and len(historical) > 0:
        # Combine: historical + current
        combined = list(historical)
        combined.append(current_vec)
        
        if len(combined) >= seq_len:
            # Take last seq_len entries
            return np.array(combined[-seq_len:], dtype=np.float32)
        else:
            # Pad front with replicated first entry
            n_pad = seq_len - len(combined)
            pad_vec = combined[0].copy()
            padded = [pad_vec] * n_pad + combined
            return np.array(padded, dtype=np.float32)
    else:
        # No history — replicate current vector
        return np.tile(current_vec, (seq_len, 1))


# ================================================================
# 5. MODEL MANAGEMENT
# ================================================================
def create_default_model(input_dim=64):
    """
    Create Mamba model with default configuration.
    Suitable for SFC Terminal stress prediction.
    """
    model = MambaEncoder(
        input_dim=input_dim,
        d_model=128,
        d_state=16,
        d_conv=4,
        n_layers=2
    )
    model.eval()
    return model


def load_model(model_path, input_dim=64):
    """
    Load trained Mamba model weights.
    Falls back to random initialized if file not found.
    """
    model = create_default_model(input_dim=input_dim)
    
    if model_path and Path(model_path).exists():
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[Mamba] Loaded trained model from {model_path}", file=sys.stderr)
        except Exception as e:
            print(f"[Mamba] Failed to load model: {e}, using random weights", file=sys.stderr)
    else:
        print("[Mamba] No trained model found. Using random initialized weights.", file=sys.stderr)
    
    return model


def save_model(model, model_path):
    """Save model weights."""
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[Mamba] Model saved to {model_path}", file=sys.stderr)


# ================================================================
# 6. INFERENCE WRAPPER
# ================================================================
MAMBA_MODEL = None
MAMBA_INPUT_DIM = 64
MAMBA_CACHE = {"result": None, "ts": 0}
MAMBA_CACHE_TTL = 600  # 10 minutes
MAMBA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mamba_weights.pth")


def get_mamba_model():
    """Get or initialize Mamba model (singleton)."""
    global MAMBA_MODEL
    if MAMBA_MODEL is None:
        MAMBA_MODEL = load_model(MAMBA_MODEL_PATH, input_dim=MAMBA_INPUT_DIM)
    return MAMBA_MODEL


def get_mamba_prediction(data, force=False):
    """
    Run Mamba inference on SFC data.
    
    Args:
        data: dict — single SFC data snapshot (e.g., from data.json)
        force: bool — bypass cache
        
    Returns:
        dict with:
            - stress_short: float (0-1)
            - stress_medium: float (0-1)
            - stress_long: float (0-1)
            - combined: float (0-1)
            - confidence: float (0-1)
            - available: bool
    """
    global MAMBA_CACHE, MAMBA_MODEL, MAMBA_INPUT_DIM
    
    now = __import__('time').time()
    
    # Check cache
    if not force and MAMBA_CACHE["result"] is not None and now - MAMBA_CACHE["ts"] < MAMBA_CACHE_TTL:
        return MAMBA_CACHE["result"]
    
    try:
        # Build feature sequence first to determine actual dimension
        feature_seq = build_feature_sequence(data, seq_len=30)
        actual_dim = feature_seq.shape[1]
        
        # If model exists but wrong dimension, rebuild
        if MAMBA_MODEL is not None and MAMBA_INPUT_DIM != actual_dim:
            MAMBA_MODEL = None
        
        # Re-init model if needed with correct dim
        if MAMBA_MODEL is None:
            MAMBA_INPUT_DIM = actual_dim
            # Try dimension-specific path first, fall back to generic
            dim_path = MAMBA_MODEL_PATH.replace('mamba_weights.pth', f'mamba_weights_{actual_dim}.pth')
            if not os.path.exists(dim_path):
                # Copy generic weights to dim-specific path for future loads
                if os.path.exists(MAMBA_MODEL_PATH):
                    import shutil
                    os.makedirs(os.path.dirname(dim_path), exist_ok=True)
                    shutil.copy2(MAMBA_MODEL_PATH, dim_path)
                    print(f"[Mamba] Copied weights to {dim_path}", file=sys.stderr)
            MAMBA_MODEL = load_model(dim_path, input_dim=actual_dim)
        
        # Convert to tensor
        x = torch.tensor(feature_seq, dtype=torch.float32).unsqueeze(0)  # (1, 30, n_features)
        
        # Run inference
        with torch.no_grad():
            result = MAMBA_MODEL(x)
        
        # Extract values
        out = {
            'stress_short': float(result['stress_short'].squeeze().cpu().numpy()),
            'stress_medium': float(result['stress_medium'].squeeze().cpu().numpy()),
            'stress_long': float(result['stress_long'].squeeze().cpu().numpy()),
            'combined': float(result['combined'].squeeze().cpu().numpy()),
            'confidence': float(result['confidence'].squeeze().cpu().numpy()),
            'available': True,
        }
        
        # Update cache
        MAMBA_CACHE = {"result": out, "ts": now}
        return out
        
    except Exception as e:
        print(f"[Mamba] Inference error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            'stress_short': 0.5,
            'stress_medium': 0.5,
            'stress_long': 0.5,
            'combined': 0.5,
            'confidence': 0.0,
            'available': False,
            'error': str(e),
        }


# ================================================================
# 7. DIRECT TESTING ENTRY POINT
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MAMBA ENCODER — Direct Test")
    print("=" * 60)
    
    # Load test data
    test_path = '/home/ubuntu/sfc/data.json'
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)
        print(f"Loaded test data: {len(test_data)} keys")
    else:
        print("No test data found, using synthetic data")
        test_data = {
            'btc': 65660.0,
            'btc_24h': -1.4,
            'btc_mcap': 1.31e12,
            'dom': 58.4,
            'dvol': 37.6,
            'rsi_14': 45.2,
            'fng': 24,
            'zone': 'NORMAL',
            'regime': 'NORMAL',
            'sfc_base': 10.8,
            'sfc_effective': 17.6,
        }
    
    # Build feature vector
    vec = build_feature_vector(test_data)
    print(f"Feature vector: {len(vec)} dimensions")
    print(f"Feature values: min={vec.min():.4f}, max={vec.max():.4f}, mean={vec.mean():.4f}")
    
    # Build sequence
    seq = build_feature_sequence(test_data, seq_len=30)
    print(f"Feature sequence: {seq.shape}")
    
    # Run inference
    result = get_mamba_prediction(test_data, force=True)
    
    print()
    print("─" * 40)
    print("MAMBA INFERENCE RESULTS")
    print("─" * 40)
    print(f"  Available:    {result.get('available', False)}")
    print(f"  Short-term:   {result.get('stress_short', 0)*100:.2f}%")
    print(f"  Medium-term:  {result.get('stress_medium', 0)*100:.2f}%")
    print(f"  Long-term:    {result.get('stress_long', 0)*100:.2f}%")
    print(f"  Combined:     {result.get('combined', 0)*100:.2f}%")
    print(f"  Confidence:   {result.get('confidence', 0)*100:.2f}%")
    print("─" * 40)
    
    if result.get('available'):
        print("✅ Mamba encoder berfungsi normal!")
    else:
        print("❌ Mamba encoder gagal:", result.get('error'))
