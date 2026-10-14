"""U-Net StarGAN v2 Generator with 4-layer skip connections.

Extends the StarGANv2Generator from MRxfield's stargan_v2.py by adding
U-Net style skip connections at four resolutions (256→128→64→32),
fused via learnable 1x1 convolutions from SkipFusion.

Building blocks (ResBlk, AdaIN, AdainResBlk) and base StarGANv2Generator
are copied inline from MRxfield/Baseline/mrixfields/models/stargan_v2.py
to keep this module standalone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mrixfields.models.skip_fusion import SkipFusion
from mrixfields.models.film import ZtoFiLM


_VALID_IMG_SIZES = {32, 64, 128, 256, 512}


def _validate_img_size(img_size: int) -> None:
    """Validate that img_size is a supported power of 2."""
    if img_size not in _VALID_IMG_SIZES:
        raise ValueError(
            f"img_size must be one of {sorted(_VALID_IMG_SIZES)}, got {img_size}"
        )


# --------------------------------------------------------------------------- #
#  Building blocks (copied from MRxfield stargan_v2.py)
# --------------------------------------------------------------------------- #


class _ResBlk(nn.Module):
    """Residual block with optional downsampling and normalization.

    Based on StarGAN v2 official.
    """

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class _AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Modulates features using style code: (1 + gamma) * norm(x) + beta
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s, film_gamma=None, film_beta=None):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        if film_gamma is not None and film_beta is not None:
            merged_gamma = gamma + film_gamma.view(gamma.shape)
            merged_beta = beta + film_beta.view(beta.shape)
            # Clamp only when FiLM contributes non-zero values (preserve zero-init identity)
            if film_gamma.abs().max() > 0:
                merged_gamma = torch.clamp(merged_gamma, min=-0.999)
            return (1 + merged_gamma) * self.norm(x) + merged_beta
        return (1 + gamma) * self.norm(x) + beta


class _AdainResBlk(nn.Module):
    """Residual block with AdaIN for style injection.

    Based on StarGAN v2 official.
    """

    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = _AdaIN(style_dim, dim_in)
        self.norm2 = _AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, film_gamma=None, film_beta=None):
        fg1 = film_gamma[0] if film_gamma is not None else None
        fb1 = film_beta[0] if film_beta is not None else None
        fg2 = film_gamma[1] if film_gamma is not None else None
        fb2 = film_beta[1] if film_beta is not None else None
        x = self.norm1(x, s, fg1, fb1)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s, fg2, fb2)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s, film_gamma=None, film_beta=None):
        out = self._residual(x, s, film_gamma, film_beta)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# --------------------------------------------------------------------------- #
#  Base Generator (copied from MRxfield stargan_v2.py)
# --------------------------------------------------------------------------- #


class StarGANv2Generator(nn.Module):
    """StarGAN v2 generator with AdaIN style injection.

    Encoder-decoder architecture where decoder blocks use AdaIN
    to inject target style codes.

    Adapted: 3ch RGB -> 1ch grayscale MRI, removed HighPass/FAN.

    Parameters:
        img_size (int) -- input image size (default 128 for MRI slices)
        style_dim (int) -- style code dimension
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, input_nc=1):
        super().__init__()
        _validate_img_size(img_size)
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(input_nc, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, input_nc, 1, 1, 0),
            nn.Tanh(),
        )

        # Down/up-sampling blocks
        repeat_num = int(math.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                _ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, _AdainResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # Bottleneck blocks
        for _ in range(2):
            self.encode.append(
                _ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, _AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s):
        """Forward pass.

        Args:
            x: Input image (B, 1, H, W).
            s: Style code (B, style_dim).

        Returns:
            Generated image (B, 1, H, W).
        """
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


# --------------------------------------------------------------------------- #
#  U-Net Generator with Skip Connections
# --------------------------------------------------------------------------- #


class UnetStarGANv2Generator(StarGANv2Generator):
    """StarGAN v2 Generator with 4-layer U-Net skip connections.

    Extends StarGANv2Generator by adding skip connections at four
    resolutions fused via learnable 1x1 convolutions (SkipFusion).

    Skip architecture (for img_size=512):
        encode[0] (256×256, 64ch)   --skip-->  decode[5] (256×256, 64ch)
        encode[1] (128×128, 128ch)  --skip-->  decode[4] (128×128, 128ch)
        encode[2] (64×64, 256ch)    --skip-->  decode[3] (64×64, 256ch)
        encode[3] (32×32, 512ch)    --skip-->  decode[2] (32×32, 512ch)

    Parameters:
        img_size (int) -- input image size (must be 512)
        style_dim (int) -- style code dimension
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI, 3 for RGB)
    """

    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, input_nc=1):
        super().__init__(img_size, style_dim, max_conv_dim, input_nc)
        if img_size != 512:
            raise ValueError(
                f"UnetStarGANv2Generator requires img_size=512. "
                f"Got img_size={img_size}. The skip connection indices "
                f"(encode[0..3]→decode[5..2]) are hardcoded for 512×512 inputs."
            )
        # Channel counts are resolution-dependent; these values assume img_size=512.
        # encode[0] output: 256×256, 64ch   → decode[5] output: 256×256, 64ch
        # encode[1] output: 128×128, 128ch  → decode[4] output: 128×128, 128ch
        # encode[2] output:  64×64, 256ch   → decode[3] output:  64×64, 256ch
        # encode[3] output:  32×32, 512ch   → decode[2] output:  32×32, 512ch
        self.skip_fusion_32 = SkipFusion(512)   # encode[3] → decode[2] at 32×32
        self.skip_fusion_64 = SkipFusion(256)   # encode[2] → decode[3] at 64×64
        self.skip_fusion_128 = SkipFusion(128)  # encode[1] → decode[4] at 128×128
        self.skip_fusion_256 = SkipFusion(64)   # encode[0] → decode[5] at 256×256

    def forward(self, x, s):
        """Forward pass with U-Net skip connections.

        Args:
            x: Input image (B, input_nc, H, W).
            s: Style code (B, style_dim).

        Returns:
            Generated image (B, input_nc, H, W).
        """
        x = self.from_rgb(x)

        # Encode: capture features at skip resolutions
        x = self.encode[0](x)
        feat0 = x  # 256×256, 64ch

        x = self.encode[1](x)
        feat1 = x  # 128×128, 128ch

        x = self.encode[2](x)
        feat2 = x  # 64×64, 256ch

        x = self.encode[3](x)
        feat3 = x  # 32×32, 512ch

        # Remaining encode (bottleneck + deepest downsample)
        for i in range(4, len(self.encode)):
            x = self.encode[i](x)

        # Bottleneck blocks (decode[0..1])
        for i in range(len(self.decode) - 5):
            x = self.decode[i](x, s)

        # decode[2]: 16²→32², 512ch → skip_fusion_32 with feat3
        x = self.decode[2](x, s)
        x = self.skip_fusion_32(feat3, x)

        # decode[3]: 32²→64², 256ch → skip_fusion_64 with feat2
        x = self.decode[3](x, s)
        x = self.skip_fusion_64(feat2, x)

        # decode[4]: 64²→128², 128ch → skip_fusion_128 with feat1
        x = self.decode[4](x, s)
        x = self.skip_fusion_128(feat1, x)

        # decode[5]: 128²→256², 64ch → skip_fusion_256 with feat0
        x = self.decode[5](x, s)
        x = self.skip_fusion_256(feat0, x)

        # decode[6]: 256²→512², 32ch → to_rgb
        x = self.decode[6](x, s)

        return self.to_rgb(x)


# --------------------------------------------------------------------------- #
#  FiLM-enabled U-Net Generator
# --------------------------------------------------------------------------- #


class FiLMUnetStarGANv2Generator(UnetStarGANv2Generator):
    """FiLM-conditioned UnetStarGANv2Generator with continuous field-strength conditioning.

    Injects FiLM (Feature-wise Linear Modulation) into every InstanceNorm/AdaIN
    layer via a shared ZtoFiLM MLP that maps (z_src, z_tgt) → per-layer (γ, β).

    Encoder: FiLM applied after each ResBlk output (2 FiLM points per block).
    Decoder: FiLM merged additively into AdaIN γ/β before modulation.

    With zero-init ZtoFiLM weights: film_enabled=True output == baseline output.
    With film_enabled=False: falls through to super().forward() with zero overhead.

    Parameters:
        img_size (int) -- input image size (must be 512)
        style_dim (int) -- style code dimension
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels
        film_hidden (int) -- hidden layer width for ZtoFiLM MLP
        film_enabled (bool) -- whether to enable FiLM conditioning
    """

    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, input_nc=1,
                 film_hidden=64, film_enabled=True, use_dynamic_film=False,
                 use_moe_film=False, use_skipfusion_experts=False):
        super().__init__(img_size, style_dim, max_conv_dim, input_nc)
        self.film_enabled = film_enabled
        self.use_dynamic_film = use_dynamic_film
        self.use_moe_film = use_moe_film
        self.use_skipfusion_experts = use_skipfusion_experts
        if use_skipfusion_experts:
            if not use_moe_film:
                raise ValueError(
                    "use_skipfusion_experts=True requires use_moe_film=True. "
                    "SkipFusionExperts rely on s_use from the MoE routing path."
                )
            from mrixfields.models.skip_fusion_experts import SkipFusionExpert
            self.sfe_32 = SkipFusionExpert(512, style_dim=style_dim, hidden=film_hidden)
            self.sfe_64 = SkipFusionExpert(256, style_dim=style_dim, hidden=film_hidden)
            self.sfe_128 = SkipFusionExpert(128, style_dim=style_dim, hidden=film_hidden)
            self.sfe_256 = SkipFusionExpert(64, style_dim=style_dim, hidden=film_hidden)
        if film_enabled:
            # Build ch_list from DECODER ONLY (same for all FiLM paths)
            ch_list = []
            for block in self.decode:
                ch_list.append(block.norm1.norm.num_features)
                ch_list.append(block.norm2.norm.num_features)
            if use_dynamic_film:
                from mrixfields.models.film import DynamicZtoFiLM
                self.dynamic_z_to_film = DynamicZtoFiLM(z_dim=513, hidden=film_hidden, ch_list=ch_list)
            elif use_moe_film:
                from mrixfields.models.film import FourExpertFiLM, FieldMLP
                self.moe_film = FourExpertFiLM(style_dim=style_dim, hidden=film_hidden, ch_list=ch_list, layer_ranges=[(0,4),(4,8),(8,12),(12,14)])
                self.field_mlp = FieldMLP(style_dim=style_dim, hidden=film_hidden, ch_list=ch_list)
            else:
                self.z_to_film = ZtoFiLM(z_dim=5, hidden=film_hidden, ch_list=ch_list)

    def forward(self, x, s, z_src=None, z_tgt=None, s_use=None):
        """Forward pass with optional FiLM conditioning.

        Args:
            x: Input image (B, input_nc, H, W).
            s: Style code (B, style_dim).
            z_src: Source field strength (B,) or (B, 1) or (B, 2).
            z_tgt: Target field strength (B,) or (B, 1) or (B, 2).
            s_use: Style segment for MoE FiLM path (B, style_dim).

        Returns:
            Generated image (B, input_nc, H, W).
        """
        if not self.film_enabled or z_src is None:
            return super().forward(x, s)

        # Dynamic FiLM path: use encoder bottleneck features
        if self.use_dynamic_film:
            x = self.from_rgb(x)
            x = self.encode[0](x); feat0 = x
            x = self.encode[1](x); feat1 = x
            x = self.encode[2](x); feat2 = x
            x = self.encode[3](x); feat3 = x
            for i in range(4, len(self.encode)):
                x = self.encode[i](x)

            gammas, betas = self.dynamic_z_to_film(x, z_src)

            def _film_pair(idx):
                return gammas[idx:idx + 2], betas[idx:idx + 2]

            film_idx = 0
            for i in range(len(self.decode) - 5):
                fg, fb = _film_pair(film_idx); film_idx += 2
                x = self.decode[i](x, s, film_gamma=fg, film_beta=fb)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[2](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_32(feat3, x)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[3](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_64(feat2, x)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[4](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_128(feat1, x)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[5](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_256(feat0, x)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[6](x, s, film_gamma=fg, film_beta=fb)
            return self.to_rgb(x)

        # MoE FiLM path: run encoder first, then compute GAP features for experts
        if self.use_moe_film:
            # Step 1: Run encoder FIRST
            x = self.from_rgb(x)
            x = self.encode[0](x); feat0 = x
            x = self.encode[1](x); feat1 = x
            x = self.encode[2](x); feat2 = x
            x = self.encode[3](x); feat3 = x
            for i in range(4, len(self.encode)):
                x = self.encode[i](x)

            # Step 1.5: Compute bottleneck GAP from final encoder output
            bottleneck_gap = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B, 512)

            # Step 2: Compute GAP from encoder features
            feat0_gap = F.adaptive_avg_pool2d(feat0, 1).squeeze(-1).squeeze(-1)  # (B,64)
            feat1_gap = F.adaptive_avg_pool2d(feat1, 1).squeeze(-1).squeeze(-1)  # (B,128)
            feat2_gap = F.adaptive_avg_pool2d(feat2, 1).squeeze(-1).squeeze(-1)  # (B,256)
            feat3_gap = F.adaptive_avg_pool2d(feat3, 1).squeeze(-1).squeeze(-1)  # (B,512)

            # Step 3: Call FourExpertFiLM with GAP features
            gammas_exp, betas_exp = self.moe_film(
                s_use, z_src, feat3_gap, feat2_gap, feat0_gap,
                bottleneck_gap=bottleneck_gap,
            )

            # Step 4: FieldMLP (unchanged, additive on all layers)
            gammas_fld, betas_fld = self.field_mlp(s_use, z_src)

            # Step 5: Additive merge
            gammas = [g_e + g_f for g_e, g_f in zip(gammas_exp, gammas_fld)]
            betas  = [b_e + b_f for b_e, b_f in zip(betas_exp, betas_fld)]

            # Step 6: Decoder with FiLM (same as before)
            def _film_pair(idx):
                return gammas[idx:idx + 2], betas[idx:idx + 2]

            film_idx = 0
            for i in range(len(self.decode) - 5):
                fg, fb = _film_pair(film_idx); film_idx += 2
                x = self.decode[i](x, s, film_gamma=fg, film_beta=fb)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[2](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_32(feat3, x)
            if self.use_skipfusion_experts:
                x = self.sfe_32(x, s_use, feat3_gap)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[3](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_64(feat2, x)
            if self.use_skipfusion_experts:
                x = self.sfe_64(x, s_use, feat2_gap)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[4](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_128(feat1, x)
            if self.use_skipfusion_experts:
                x = self.sfe_128(x, s_use, feat1_gap)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[5](x, s, film_gamma=fg, film_beta=fb)
            x = self.skip_fusion_256(feat0, x)
            if self.use_skipfusion_experts:
                x = self.sfe_256(x, s_use, feat0_gap)
            fg, fb = _film_pair(film_idx); film_idx += 2
            x = self.decode[6](x, s, film_gamma=fg, film_beta=fb)
            return self.to_rgb(x)

        # Static FiLM path
        gammas, betas = self.z_to_film(z_src, z_tgt)

        x = self.from_rgb(x)

        def _apply_film(feat, idx):
            g = gammas[idx].view(feat.size(0), -1, 1, 1)
            b = betas[idx].view(feat.size(0), -1, 1, 1)
            return (1 + g) * feat + b

        x = self.encode[0](x); feat0 = x
        x = _apply_film(_apply_film(x, 0), 1)

        x = self.encode[1](x); feat1 = x
        x = _apply_film(_apply_film(x, 2), 3)

        x = self.encode[2](x); feat2 = x
        x = _apply_film(_apply_film(x, 4), 5)

        x = self.encode[3](x); feat3 = x
        x = _apply_film(_apply_film(x, 6), 7)

        film_idx = 8
        for i in range(4, len(self.encode)):
            x = self.encode[i](x)
            x = _apply_film(_apply_film(x, film_idx), film_idx + 1)
            film_idx += 2

        def _film_pair(idx):
            return gammas[idx:idx + 2], betas[idx:idx + 2]

        for i in range(len(self.decode) - 5):
            fg, fb = _film_pair(film_idx)
            x = self.decode[i](x, s, film_gamma=fg, film_beta=fb)
            film_idx += 2

        fg, fb = _film_pair(film_idx); film_idx += 2
        x = self.decode[2](x, s, film_gamma=fg, film_beta=fb)
        x = self.skip_fusion_32(feat3, x)

        fg, fb = _film_pair(film_idx); film_idx += 2
        x = self.decode[3](x, s, film_gamma=fg, film_beta=fb)
        x = self.skip_fusion_64(feat2, x)

        fg, fb = _film_pair(film_idx); film_idx += 2
        x = self.decode[4](x, s, film_gamma=fg, film_beta=fb)
        x = self.skip_fusion_128(feat1, x)

        fg, fb = _film_pair(film_idx); film_idx += 2
        x = self.decode[5](x, s, film_gamma=fg, film_beta=fb)
        x = self.skip_fusion_256(feat0, x)

        fg, fb = _film_pair(film_idx); film_idx += 2
        x = self.decode[6](x, s, film_gamma=fg, film_beta=fb)

        return self.to_rgb(x)
