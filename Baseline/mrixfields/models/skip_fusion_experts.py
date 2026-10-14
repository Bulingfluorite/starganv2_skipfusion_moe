import torch
import torch.nn as nn


class SkipFusionExpert(nn.Module):
    """Lightweight FiLM expert that modulates SkipFusion outputs for texture control.

    Gamma-only modulation (no beta): scales fused skip features by (1 + gamma),
    where gamma is predicted from the style code and a global feature-gap descriptor.
    Field-independent by design — no field_scalar input.

    Zero-init on the last Linear layer ensures identity start (gamma=0),
    preserving pretrained SkipFusion behavior until the expert learns to adjust.

    Args:
        channels: Number of channels in the SkipFusion output (= in_channels).
        style_dim: Dimension of the style code s_use (default 64).
        hidden: Hidden layer width (default 64).
    """

    def __init__(self, channels: int, style_dim: int = 64, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(style_dim + channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        s_use: torch.Tensor,
        feat_gap: torch.Tensor,
    ) -> torch.Tensor:
        """Apply texture modulation to fused skip features.

        Args:
            x: Fused feature map from SkipFusion, shape (B, C, H, W).
            s_use: Style code, shape (B, style_dim).
            feat_gap: Global feature descriptor (GAP of channels), shape (B, C).

        Returns:
            Modulated feature map, same shape as x: (B, C, H, W).
        """
        gamma = self.mlp(torch.cat([s_use, feat_gap], dim=-1))  # (B, C)
        self.last_gamma = gamma.detach()
        return (1 + gamma.view(gamma.shape[0], -1, 1, 1)) * x
