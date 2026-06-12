"""
gnn_module.py — Systemic Risk Calculator (GNN-inspired, numpy-based).

Implements a SystemicRiskGNN class using pure PyTorch (no torch_geometric)
and a SystemicRiskCalculator using pure numpy for environments where
PyTorch is not available.

All modules are designed to be importable without error even if optional
dependencies (torch) are missing.
"""

import warnings
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Stub so the file can still be imported without torch
    class nn:
        class Module:
            pass


# ---------------------------------------------------------------------------
# SystemicRiskGNN — PyTorch-based GNN aggregator
# ---------------------------------------------------------------------------
class SystemicRiskGNN(nn.Module if TORCH_AVAILABLE else object):
    """Simple GNN-style risk aggregator using pure PyTorch.

    Performs message passing by averaging neighbour features, then passes
    through a small MLP to produce a scalar systemic-risk score per asset.

    Parameters
    ----------
    num_assets : int
        Number of assets in the graph (default 5).
    hidden_dim : int
        Hidden dimension of the internal MLP (default 16).
    input_features : int
        Number of features per node (default 3 = return, volatility, momentum).
    """

    def __init__(
        self,
        num_assets: int = 5,
        hidden_dim: int = 16,
        input_features: int = 3,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to instantiate SystemicRiskGNN. "
                "Install it via `pip install torch`."
            )
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.input_features = input_features

        # Linear1: (num_assets * input_features) -> hidden_dim
        self.linear1 = nn.Linear(num_assets * input_features, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        # Linear2: hidden_dim -> 1 (per-asset risk score)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: "torch.Tensor",
        edge_index: "Optional[torch.Tensor]" = None,
        edge_weight: "Optional[torch.Tensor]" = None,
    ) -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        node_features : Tensor of shape (num_assets, input_features)
            Per-asset features: [return, volatility, momentum].
        edge_index : Tensor of shape (2, num_edges) or None
            Edge connectivity (source -> target). If None, message passing
            is skipped and only the MLP is applied.
        edge_weight : Tensor of shape (num_edges,) or None
            Optional edge weights (ignored in current implementation).

        Returns
        -------
        Tensor of shape (num_assets, 1)
            Systemic risk score per asset (0-1 range after sigmoid).
        """
        # ---- Message passing step ----
        if edge_index is not None and edge_index.size(1) > 0:
            # Simple mean aggregation of neighbour features
            src, tgt = edge_index  # (num_edges,), (num_edges,)
            aggregated = torch.zeros_like(node_features)
            # Count neighbours per target node for mean
            neighbour_count = torch.zeros(
                node_features.size(0), device=node_features.device
            )

            for s, t in zip(src, tgt):
                aggregated[t] += node_features[s]
                neighbour_count[t] += 1

            # Avoid division by zero
            neighbour_count = neighbour_count.clamp(min=1).unsqueeze(1)
            aggregated = aggregated / neighbour_count

            # Combine original features + aggregated neighbour info
            combined = torch.cat([node_features, aggregated], dim=1)
        else:
            # No edges: pad with zeros for same dimensions
            combined = torch.cat(
                [node_features, torch.zeros_like(node_features)], dim=1
            )

        # ---- MLP ----
        # Flatten: (num_assets, 2 * input_features) -> (1, num_assets * 2 * input_features)
        batch = combined.view(1, -1)

        h = self.linear1(batch)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.linear2(h)  # (1, 1)

        # Expand back to per-asset scores by using the same scalar for all
        # (a simple GNN readout). For a more advanced version, use node-level.
        risk = out.view(1)  # scalar
        return torch.full(
            (node_features.size(0), 1),
            torch.sigmoid(risk).item(),
            device=node_features.device,
        )


# ---------------------------------------------------------------------------
# SystemicRiskCalculator — Pure numpy variant (no PyTorch required)
# ---------------------------------------------------------------------------
class SystemicRiskCalculator:
    """Numpy-based systemic risk calculator.

    Computes a correlation-weighted systemic risk score from asset-level
    return, volatility, and momentum data.
    """

    ASSET_NAMES = ["btc", "eth", "spx", "gold", "dxy"]

    @staticmethod
    def _to_array(
        data: dict, keys: list[str]
    ) -> np.ndarray:
        """Build a (5,) array from a dict of per-asset data."""
        return np.array([data.get(k, 0.0) for k in keys], dtype=np.float64)

    @staticmethod
    def _pseudo_corr(
        returns: np.ndarray, volatilities: np.ndarray, momenta: np.ndarray
    ) -> np.ndarray:
        """Compute a simple 5x5 pseudo-correlation matrix.

        Uses cosine similarity between (return, vol, momentum) vectors
        as a proxy for correlation when we only have a single time-point.
        """
        n = len(returns)
        features = np.column_stack([returns, volatilities, momenta])
        # Normalise each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        features_normed = features / norms
        corr = features_normed @ features_normed.T  # (n, n)
        # Clamp to valid correlation range
        corr = np.clip(corr, -1.0, 1.0)
        # Set diagonal to 1.0
        np.fill_diagonal(corr, 1.0)
        return corr

    def calculate_systemic_risk(
        self,
        btc_data: dict,
        eth_data: dict,
        spx_data: dict,
        gold_data: dict,
        dxy_data: dict,
    ) -> dict:
        """Compute systemic risk scores for a 5-asset universe.

        Parameters
        ----------
        btc_data, eth_data, spx_data, gold_data, dxy_data : dict
            Each dict should have keys: 'return', 'volatility', 'momentum'.
            All values should be floats.

        Returns
        -------
        dict with keys:
            - btc_systemic_risk : float (0-1)
            - eth_systemic_risk : float (0-1)
            - spx_systemic_risk : float (0-1)
            - overall_systemic_risk : float (0-1)
            - correlation_breakdown : bool
            - market_regime : str ('NORMAL', 'STRESS', 'CRISIS')
        """
        all_data = [btc_data, eth_data, spx_data, gold_data, dxy_data]

        # Extract feature vectors (5, 3)
        returns = np.array([d.get("return", 0.0) for d in all_data], dtype=np.float64)
        vols = np.array(
            [d.get("volatility", 0.0) for d in all_data], dtype=np.float64
        )
        momenta = np.array(
            [d.get("momentum", 0.0) for d in all_data], dtype=np.float64
        )

        # ---- Correlation matrix ----
        corr = self._pseudo_corr(returns, vols, momenta)

        # Average correlation (excluding diagonal)
        n = corr.shape[0]
        off_diag_count = n * (n - 1)
        avg_corr = (np.sum(corr) - n) / off_diag_count if off_diag_count > 0 else 0.0

        # ---- Volatility weights ----
        vol_sum = np.sum(vols) + 1e-12
        vol_weights = vols / vol_sum

        # ---- Per-asset weighted risk ----
        # For each asset, risk = mean(correlation with others) * volatility_weight
        per_asset_risk = np.zeros(n)
        for i in range(n):
            # Mean correlation of asset i with all *other* assets
            others = [j for j in range(n) if j != i]
            if others:
                mean_corr_i = np.mean([corr[i, j] for j in others])
            else:
                mean_corr_i = 0.0
            # Rescale from [-1, 1] to [0, 1]; negate so high positive
            # correlation => higher risk (contagion)
            risk_score = max(0.0, (mean_corr_i + 1.0) / 2.0)
            per_asset_risk[i] = risk_score * vol_weights[i]

        # Normalise per-asset risks to [0, 1] across assets
        max_risk = np.max(per_asset_risk) if np.max(per_asset_risk) > 0 else 1.0
        per_asset_risk_norm = per_asset_risk / max_risk

        # ---- Overall systemic risk ----
        overall = float(np.clip(np.mean(per_asset_risk_norm), 0.0, 1.0))

        # ---- Correlation breakdown detection ----
        correlation_breakdown = avg_corr < -0.5

        # ---- Market regime ----
        if overall > 0.7:
            regime = "CRISIS"
        elif overall > 0.4:
            regime = "STRESS"
        else:
            regime = "NORMAL"

        return {
            "btc_systemic_risk": float(per_asset_risk_norm[0]),
            "eth_systemic_risk": float(per_asset_risk_norm[1]),
            "spx_systemic_risk": float(per_asset_risk_norm[2]),
            "gold_systemic_risk": float(per_asset_risk_norm[3]),
            "dxy_systemic_risk": float(per_asset_risk_norm[4]),
            "overall_systemic_risk": overall,
            "correlation_breakdown": bool(correlation_breakdown),
            "market_regime": regime,
        }


# ---------------------------------------------------------------------------
# Default simulated data  (used when real data is unavailable)
# ---------------------------------------------------------------------------
def _default_simulated_data() -> list[dict]:
    """Return simulated per-asset data for a normal market regime.

    Returns
    -------
    list of dict: [btc_data, eth_data, spx_data, gold_data, dxy_data]
    """
    return [
        {"return": 0.02, "volatility": 0.35, "momentum": 0.01},
        {"return": 0.015, "volatility": 0.30, "momentum": 0.02},
        {"return": 0.005, "volatility": 0.15, "momentum": 0.005},
        {"return": 0.003, "volatility": 0.10, "momentum": -0.002},
        {"return": -0.001, "volatility": 0.08, "momentum": 0.001},
    ]


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------
def calculate_systemic_risk(
    btc_data: Optional[dict] = None,
    eth_data: Optional[dict] = None,
    spx_data: Optional[dict] = None,
    gold_data: Optional[dict] = None,
    dxy_data: Optional[dict] = None,
) -> dict:
    """Convenience function that computes systemic risk.

    If any data argument is None, a default simulated value is used so
    the function never fails.

    Parameters
    ----------
    btc_data, eth_data, spx_data, gold_data, dxy_data : dict or None
        Each dict should have keys: 'return', 'volatility', 'momentum'.
        Pass None to use a default simulated value.

    Returns
    -------
    dict with risk scores and metadata.
    """
    default = _default_simulated_data()
    data_args = [btc_data, eth_data, spx_data, gold_data, dxy_data]
    resolved = [
        d if d is not None else default[i] for i, d in enumerate(data_args)
    ]

    calculator = SystemicRiskCalculator()
    return calculator.calculate_systemic_risk(*resolved)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = calculate_systemic_risk()
        print("=== Systemic Risk Report (default simulated data) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Test with stress-like data
        stress_result = calculate_systemic_risk(
            btc_data={"return": -0.05, "volatility": 0.60, "momentum": -0.10},
            eth_data={"return": -0.04, "volatility": 0.55, "momentum": -0.08},
            spx_data={"return": -0.03, "volatility": 0.40, "momentum": -0.06},
            gold_data={"return": 0.01, "volatility": 0.20, "momentum": 0.02},
            dxy_data={"return": 0.02, "volatility": 0.12, "momentum": 0.03},
        )
        print("\n=== Systemic Risk Report (stress scenario) ===")
        for k, v in stress_result.items():
            print(f"  {k}: {v}")
