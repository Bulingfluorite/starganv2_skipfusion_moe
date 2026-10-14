import torch
import torch.nn as nn


class SkipFusion(nn.Module):
    """1×1 Conv fusion for U-Net skip connections.
    
    Concatenates encoder and decoder features then projects back to
    original channel count via a learnable 1×1 convolution.
    
    Args:
        in_channels: Number of channels in each of the encoder and decoder
            features.  The concatenated tensor has 2*in_channels channels;
            the 1×1 Conv projects it back to in_channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        # Identity-init: skip starts as decoder passthrough, gradually learns
        # conv(cat([enc, dec])) ≈ dec initially, so pretrained decoder
        # path is preserved while encoder features are blended in over time.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            # Select decoder half: weight[:, in_channels:, 0, 0] = I
            self.conv.weight[:, in_channels:, 0, 0] = torch.eye(in_channels)

    def forward(
        self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor
    ) -> torch.Tensor:
        """Fuse encoder and decoder feature maps.

        Args:
            encoder_feat: (B, in_channels, H, W)
            decoder_feat: (B, in_channels, H, W)

        Returns:
            (B, in_channels, H, W) fused feature map.
        """
        x = torch.cat([encoder_feat, decoder_feat], dim=1)
        return self.conv(x)
