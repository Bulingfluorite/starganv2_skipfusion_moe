"""ZtoFiLM: shared MLP that maps (z_src, z_tgt) → per-layer (γ, β) for FiLM conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ZtoFiLM(nn.Module):
    """Shared MLP that maps (z_src, z_tgt) to per-layer FiLM parameters (γ, β).

    FiLM formula (same as AdaIN): (1 + γ) · x + β
    Zero-init on last layer ensures identity start (γ=0, β=0).

    Args:
        z_dim: Dimension of each z vector (default 2 for complex z_src, z_tgt).
        hidden: Hidden layer width.
        ch_list: List of channel dimensions per FiLM conditioning point.
    """

    def __init__(self, z_dim: int = 2, hidden: int = 64, ch_list: List[int] = None):
        super().__init__()
        if ch_list is None:
            raise ValueError("ch_list is required (list of channel dims per FiLM point)")
        self.ch_list = ch_list
        total_out = 2 * sum(ch_list)  # gamma + beta for each layer

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total_out),
        )

        # Zero-init last linear layer → identity start (γ=0, β=0)
        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Map (z_src, z_tgt) to per-layer FiLM parameters.

        Args:
            z_src: Source latent code, shape (B,) or (B, 1) or (B, z_dim).
            z_tgt: Target latent code, shape (B,) or (B, 1) or (B, z_dim).

        Returns:
            (gammas_list, betas_list) where each list has len(ch_list) tensors
            of shape (B, ch_list[i]).
        """
        # Ensure at least 2D
        if z_src.dim() == 1:
            z_src = z_src.unsqueeze(-1)
        if z_tgt.dim() == 1:
            z_tgt = z_tgt.unsqueeze(-1)

        # Stack into (B, 2) or (B, z_dim*2) — we need exactly z_dim total input
        z = torch.cat([z_src, z_tgt], dim=-1)  # (B, 2*z_src_dim)

        h = self.mlp(z)  # (B, total_out)

        # Split into gammas and betas per channel list entry
        gammas = []
        betas = []
        offset = 0
        for ch in self.ch_list:
            gammas.append(h[:, offset:offset + ch])
            offset += ch
        for ch in self.ch_list:
            betas.append(h[:, offset:offset + ch])
            offset += ch

        return gammas, betas


class DynamicZtoFiLM(nn.Module):
    """Dynamic FiLM: predicts gamma/beta from encoder bottleneck features.

    Unlike ZtoFiLM (static modality-onehot conditioning), this class takes
    GAP(encoder_bottleneck) as input, making modulation content-adaptive.

    Input:
        encoder_features: (B, 512, 32, 32) — encoder bottleneck output
        z_src: (B, 1) — source field strength scalar

    Architecture:
        GAP(encoder_features) → (B, 512)
        concat with z_src → (B, 513)
        MLP(513 → hidden → hidden → total_out) where total_out = 2 * sum(ch_list)
        split into gammas[0..N], betas[0..N]

    Zero-init on last Linear layer for identity start.
    """
    def __init__(self, z_dim: int = 513, hidden: int = 64, ch_list: List[int] = None):
        super().__init__()
        if ch_list is None:
            raise ValueError("ch_list is required")
        self.ch_list = ch_list
        total_out = 2 * sum(ch_list)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total_out),
        )
        last_linear = self.mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, encoder_features: torch.Tensor, z_src: torch.Tensor):
        gap = F.adaptive_avg_pool2d(encoder_features, (1, 1))
        gap = gap.squeeze(-1).squeeze(-1)
        if z_src.dim() == 1:
            z_src = z_src.unsqueeze(-1)
        z = torch.cat([gap, z_src], dim=-1)
        h = self.mlp(z)
        gammas, betas = [], []
        offset = 0
        for ch in self.ch_list:
            gammas.append(h[:, offset:offset + ch])
            offset += ch
        for ch in self.ch_list:
            betas.append(h[:, offset:offset + ch])
            offset += ch
        return gammas, betas


class MoEFiLMRouter(nn.Module):
    """Sample-level router: s_use(64) -> raw logits (B, num_experts)."""
    def __init__(self, style_dim=64, num_experts=2):
        super().__init__()
        self.linear = nn.Linear(style_dim, num_experts)

    def forward(self, s_use):
        return self.linear(s_use)


class MoEFiLMExpert(nn.Module):
    """FiLM Expert: s_use(64) -> MLP -> (gamma_list, beta_list). Zero-init on last Linear."""
    def __init__(self, style_dim=64, hidden=64, ch_list=None):
        super().__init__()
        if ch_list is None:
            raise ValueError("ch_list is required")
        self.ch_list = ch_list
        total_out = 2 * sum(ch_list)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total_out),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s_use):
        h = self.mlp(s_use)
        gammas, betas = [], []
        offset = 0
        for ch in self.ch_list:
            gammas.append(h[:, offset:offset + ch])
            offset += ch
        for ch in self.ch_list:
            betas.append(h[:, offset:offset + ch])
            offset += ch
        return gammas, betas


class MoEFiLM(nn.Module):  # DEPRECATED: replaced by FourExpertFiLM
    """Router + 2 Experts + weighted blending. Returns (gamma, beta, weights, entropy)."""
    def __init__(self, style_dim=64, num_experts=2, hidden=64, ch_list=None, layer_split=8):
        super().__init__()
        self.num_experts = num_experts
        self.router = MoEFiLMRouter(style_dim, num_experts)
        self.experts = nn.ModuleList([
            MoEFiLMExpert(style_dim, hidden, ch_list) for _ in range(num_experts)
        ])
        self.layer_split = layer_split
        n_total = len(ch_list)
        self.layer_bias = nn.Parameter(torch.zeros(n_total, num_experts))
        with torch.no_grad():
            self.layer_bias[:layer_split, 0] = 1.0
            self.layer_bias[:layer_split, 1] = -1.0
            self.layer_bias[layer_split:, 0] = -1.0
            self.layer_bias[layer_split:, 1] = 1.0

    def forward(self, s_use):
        router_logits = self.router(s_use)  # (B, num_experts)
        router_logits = router_logits.unsqueeze(1) + self.layer_bias  # (B, n_total, num_experts)
        weights = F.softmax(router_logits, dim=-1)  # (B, n_total, num_experts)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)  # (B, n_total)
        expert_outputs = [expert(s_use) for expert in self.experts]
        gammas, betas = [], []
        for layer_idx in range(len(expert_outputs[0][0])):
            w = weights[:, layer_idx, :]  # (B, num_experts)
            g_blend = sum(w[:, k:k+1] * expert_outputs[k][0][layer_idx] for k in range(self.num_experts))
            b_blend = sum(w[:, k:k+1] * expert_outputs[k][1][layer_idx] for k in range(self.num_experts))
            gammas.append(g_blend)
            betas.append(b_blend)
        return gammas, betas, weights.mean(dim=1), entropy.mean(dim=1)


class FourExpertFiLM(nn.Module):
    """Four independent expert MLPs, each handling non-overlapping decoder layers.

    No Router, no softmax blending, no layer_bias, no load balancing.
    Each expert receives different input features relevant to its assigned decoder layers:

        Expert0 (layers 0-3):  s_use(64) + bottleneck_gap(512) + field_scalar(1) → input_dim=577
        Expert1 (layers 4-7):  s_use(64) + feat3_GAP(512) + field_scalar(1)     → input_dim=577
        Expert2 (layers 8-11): s_use(64) + feat2_GAP(256)                       → input_dim=320
        Expert3 (layers 12-13): s_use(64) + feat0_GAP(64)                      → input_dim=128

    Each expert MLP: Linear(input_dim→hidden) → ReLU → Linear(hidden→2*sum(ch_slice)).
    Zero-init on last Linear for identity start (γ=0, β=0).

    Args:
        style_dim: Dimension of s_use style code.
        hidden: Hidden layer width for each expert MLP.
        ch_list: List of channel dimensions per FiLM conditioning point (14 entries).
        layer_ranges: List of (start_film_idx, end_film_idx) per expert.
    """

    def __init__(self, style_dim=64, hidden=64, ch_list=None, layer_ranges=None):
        super().__init__()
        if ch_list is None:
            raise ValueError("ch_list is required")
        if layer_ranges is None:
            layer_ranges = [(0, 4), (4, 8), (8, 12), (12, 14)]

        self.ch_list = ch_list
        self.layer_ranges = layer_ranges

        # Per-expert input dimensions: (style_dim + auxiliary feature dim)
        # Expert0: s_use + bottleneck_gap + field_scalar  (577 with style_dim=64)
        # Expert1: s_use + feat3_gap + field_scalar        (577 with style_dim=64)
        expert_input_dims = [style_dim + 512 + 1, style_dim + 512 + 1, style_dim + 256, style_dim + 64]

        self.expert_ch_slices = []
        self.expert_mlps = nn.ModuleList()

        for i, (start, end) in enumerate(layer_ranges):
            ch_slice = ch_list[start:end]
            self.expert_ch_slices.append(ch_slice)
            total_out = 2 * sum(ch_slice)  # gamma + beta

            mlp = nn.Sequential(
                nn.Linear(expert_input_dims[i], hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, total_out),
            )
            # Zero-init last linear layer → identity start
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            self.expert_mlps.append(mlp)

    def forward(self, s_use, field_scalar, feat3_gap, feat2_gap, feat0_gap, bottleneck_gap=None):
        """Forward pass: run each expert on its assigned input, concatenate results.

        Args:
            s_use: Style code, shape (B, style_dim).
            field_scalar: Field strength, shape (B, 1) or (B,).
            feat3_gap: Encoder bottleneck GAP, shape (B, 512).
            feat2_gap: Encoder mid-level GAP, shape (B, 256).
            feat0_gap: Encoder early-level GAP, shape (B, 64).
            bottleneck_gap: Optional decoder bottleneck GAP, shape (B, 512).
                If None, Expert0 uses zeros (backward compat with old checkpoints).

        Returns:
            (gammas, betas): Two lists of 14 tensors each, shape (B, ch_list[i]).
        """
        # Normalise field_scalar to 2D
        if field_scalar.dim() == 1:
            field_scalar = field_scalar.unsqueeze(-1)

        # Backward compat: when bottleneck_gap is not provided, use zeros
        if bottleneck_gap is None:
            bottleneck_gap = torch.zeros(s_use.shape[0], 512, device=s_use.device, dtype=s_use.dtype)

        # Pack per-expert inputs
        inputs = [
            torch.cat([s_use, bottleneck_gap, field_scalar], dim=-1),
            torch.cat([s_use, feat3_gap, field_scalar], dim=-1),
            torch.cat([s_use, feat2_gap], dim=-1),
            torch.cat([s_use, feat0_gap], dim=-1),
        ]

        all_gammas = []
        all_betas = []

        for i, mlp in enumerate(self.expert_mlps):
            h = mlp(inputs[i])                      # (B, 2 * sum(ch_slice))
            ch_slice = self.expert_ch_slices[i]
            half = sum(ch_slice)

            # Split into per-layer gammas (first half) and betas (second half)
            offset_g = 0
            for ch in ch_slice:
                all_gammas.append(h[:, offset_g:offset_g + ch])
                offset_g += ch
            offset_b = half
            for ch in ch_slice:
                all_betas.append(h[:, offset_b:offset_b + ch])
                offset_b += ch

        # Store gammas for monitoring (accessed by train.py logging)
        self.last_gammas = all_gammas

        return all_gammas, all_betas


class FieldMLP(nn.Module):
    """Field-strength modulation: s_use(64)+field(1) -> MLP -> gamma_fld, beta_fld. Zero-init last Linear."""
    def __init__(self, style_dim=64, hidden=64, ch_list=None):
        super().__init__()
        if ch_list is None:
            raise ValueError("ch_list is required")
        self.ch_list = ch_list
        total_out = 2 * sum(ch_list)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total_out),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s_use, field):
        if field.dim() == 1:
            field = field.unsqueeze(-1)
        x = torch.cat([s_use, field], dim=-1)
        h = self.mlp(x)
        gammas, betas = [], []
        offset = 0
        for ch in self.ch_list:
            gammas.append(h[:, offset:offset + ch])
            offset += ch
        for ch in self.ch_list:
            betas.append(h[:, offset:offset + ch])
            offset += ch
        return gammas, betas
