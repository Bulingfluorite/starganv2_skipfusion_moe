"""Laplacian Attention Module for StyleEncoder enhancement.

Source: LASTGAN (Nature Scientific Reports, 2025)
Effect: NRMSE↓30%, LPIPS↓30% on brain MRI multi-sequence synthesis.

Insert point: StyleEncoder shared CNN → LaplacianAttention → unshared FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianAttention(nn.Module):
    """Multi-scale Laplacian edge attention for StyleEncoder.

    Extracts Laplacian edges at 3 scales, applies channel+spatial attention,
    and adds residual to preserve original features.

    Args:
        in_channels: Number of input feature channels.
        reduction: Channel reduction ratio for attention.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Laplacian kernel (3×3)
        self.register_buffer('laplacian_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0))

        self.scales = [1, 2, 4]  # Multi-scale: 1x, 0.5x, 0.25x

        # Channel attention (SENet style)
        total_channels = in_channels * len(self.scales)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(total_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def _laplacian_pyramid(self, x):
        """Extract multi-scale Laplacian edge features."""
        features = []
        for scale in self.scales:
            if scale > 1:
                h, w = x.shape[2], x.shape[3]
                dh, dw = max(1, h // scale), max(1, w // scale)
                x_scaled = F.interpolate(
                    x, size=(dh, dw),
                    mode='bilinear', align_corners=False
                )
            else:
                x_scaled = x

            # Apply Laplacian per channel
            kernel = self.laplacian_kernel.repeat(x_scaled.size(1), 1, 1, 1)
            lap = F.conv2d(x_scaled, kernel, padding=1, groups=x_scaled.size(1))
            lap = F.interpolate(
                lap, size=x.shape[2:],
                mode='bilinear', align_corners=False
            )
            features.append(lap)

        return torch.cat(features, dim=1)

    def forward(self, x):
        """Apply Laplacian attention.

        Args:
            x: StyleEncoder intermediate features (B, C, H, W)

        Returns:
            Attention-enhanced features (B, C, H, W)
        """
        lap_features = self._laplacian_pyramid(x)
        ca = self.channel_attn(lap_features)  # Channel attention
        sa = self.spatial_attn(lap_features)  # Spatial attention
        attended = x * ca * sa
        return attended + x  # Residual connection
