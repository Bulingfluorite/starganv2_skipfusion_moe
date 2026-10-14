"""StarGAN v2 for multi-domain image translation (Task 3: Any→Any).

Based on the official StarGAN v2 implementation:
    clovaai/stargan-v2/core/model.py

Reference: Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple
Domains", CVPR 2020.

Adaptations from official code:
    - 3-channel RGB → 1-channel grayscale MRI
    - num_domains: 2 → 5 (0.1T, 1.5T, 3T, 5T, 7T)
    - Removed HighPass filter and FAN face detection (not relevant for MRI)
    - Removed nn.DataParallel wrapping (handled externally)

Usage for Task 3: Single model trained on all 5 field strengths simultaneously.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import for MoE losses (plain module, not nn.Module — must not go into nets dict)
_moe_losses_module = None

def _get_moe_losses():
    global _moe_losses_module
    if _moe_losses_module is None:
        from mrixfields.models import moe_losses as _moe_losses_module
    return _moe_losses_module


_VALID_IMG_SIZES = {32, 64, 128, 256, 512}


def _validate_img_size(img_size: int) -> None:
    """Validate that img_size is a supported power of 2."""
    if img_size not in _VALID_IMG_SIZES:
        raise ValueError(
            f"img_size must be one of {sorted(_VALID_IMG_SIZES)}, got {img_size}"
        )


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #

class ResBlk(nn.Module):
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


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Modulates features using style code: (1 + gamma) * norm(x) + beta
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
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
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# --------------------------------------------------------------------------- #
#  Generator
# --------------------------------------------------------------------------- #

class StarGANv2Generator(nn.Module):
    """StarGAN v2 generator with AdaIN style injection.

    Encoder-decoder architecture where decoder blocks use AdaIN
    to inject target style codes.

    Adapted: 3ch RGB → 1ch grayscale MRI, removed HighPass/FAN.

    Parameters:
        img_size (int) -- input image size (default 128 for MRI slices)
        style_dim (int) -- style code dimension
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, style_dim=64, max_conv_dim=512, input_nc=3):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
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
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # Bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

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
#  Mapping Network
# --------------------------------------------------------------------------- #

class MappingNetwork(nn.Module):
    """Maps random latent codes to domain-specific style codes.

    Shared MLP layers followed by per-domain branches.

    Parameters:
        latent_dim (int) -- input latent dimension
        style_dim (int) -- output style dimension
        num_domains (int) -- number of domains (5 for MRIxFields)
    """

    def __init__(self, latent_dim=16, style_dim=64, num_domains=5):
        super().__init__()
        layers = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, style_dim),
            ))

    def forward(self, z, y):
        """
        Args:
            z: Latent code (B, latent_dim).
            y: Target domain label (B,) as LongTensor.

        Returns:
            Style code (B, style_dim).
        """
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # (B, num_domains, style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # (B, style_dim)
        return s


# --------------------------------------------------------------------------- #
#  Style Encoder
# --------------------------------------------------------------------------- #

class StyleEncoder(nn.Module):
    """Encodes a reference image into a domain-specific style code.

    Parameters:
        img_size (int) -- input image size
        style_dim (int) -- output style dimension
        num_domains (int) -- number of domains
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, style_dim=64, num_domains=5,
                 max_conv_dim=512, input_nc=3, use_lap_attn=False):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2 ** 14 // img_size
        blocks = [nn.Conv2d(input_nc, dim_in, 3, 1, 1)]

        repeat_num = int(math.log2(img_size)) - 2
        dim_out = dim_in  # fallback when repeat_num == 0
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        self.shared_dim_out = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        self.shared = nn.Sequential(*blocks)

        self.use_lap_attn = use_lap_attn
        if use_lap_attn:
            from mrixfields.models.laplacian_attention import LaplacianAttention
            self.lap_attn = LaplacianAttention(dim_out)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        """
        Args:
            x: Reference image (B, 1, H, W).
            y: Domain label (B,) as LongTensor.

        Returns:
            Style code (B, style_dim).
        """
        h = self.shared(x)
        if self.use_lap_attn:
            h = self.lap_attn(h)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out = torch.stack(out, dim=1)  # (B, num_domains, style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        s = out[idx, y]  # (B, style_dim)
        return s


# --------------------------------------------------------------------------- #
#  Discriminator
# --------------------------------------------------------------------------- #

class StarGANv2Discriminator(nn.Module):
    """Multi-domain discriminator for StarGAN v2.

    Outputs one real/fake score per domain.

    Parameters:
        img_size (int) -- input image size
        num_domains (int) -- number of domains
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
    """

    def __init__(self, img_size=128, num_domains=5, max_conv_dim=512, input_nc=1):
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2 ** 14 // img_size
        blocks = [nn.Conv2d(input_nc, dim_in, 3, 1, 1)]

        repeat_num = int(math.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, dim_out, 4, 1, 0))
        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(nn.Conv2d(dim_out, num_domains, 1, 1, 0))
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        """
        Args:
            x: Input image (B, 1, H, W).
            y: Domain label (B,) as LongTensor.

        Returns:
            Discriminator output (B,) — score for the specified domain.
        """
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (B, num_domains)
        idx = torch.arange(y.size(0), device=y.device)
        out = out[idx, y]  # (B,)
        return out


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator: independent discriminators at downsampled scales.

    Uses avg-pooling to create input pyramids (original, 1/2, 1/4 scale),
    then applies an independent StarGANv2Discriminator per scale.
    Losses are averaged across scales.

    Parameters:
        img_size (int) -- input image size (original scale)
        num_domains (int) -- number of domains
        max_conv_dim (int) -- maximum conv channels
        input_nc (int) -- input channels (1 for grayscale MRI)
        num_scales (int) -- number of scales (3 for 1x, 0.5x, 0.25x)
    """

    def __init__(self, img_size=128, num_domains=5, max_conv_dim=512, input_nc=1,
                 num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            StarGANv2Discriminator(img_size // (2 ** i), num_domains,
                                   max_conv_dim, input_nc)
            for i in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1,
                                       count_include_pad=False)

    def forward(self, x, y):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x, y))
        return outputs


# --------------------------------------------------------------------------- #
#  Loss functions (from StarGAN v2 solver.py)
# --------------------------------------------------------------------------- #

def adv_loss(logits, target):
    """Binary cross-entropy adversarial loss."""
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    """R1 gradient penalty for real images."""
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def freq_domain_loss(x_fake, x_real):
    """FFT-based frequency domain consistency loss.

    Computes L1 distance between magnitude spectra of generated and real images.
    Penalizes frequency-domain discrepancies that spatial losses may miss.

    Args:
        x_fake: Generated images (B, C, H, W).
        x_real: Target real images (B, C, H, W).

    Returns:
        Scalar loss.
    """
    fake_fft = torch.fft.fft2(x_fake.float())
    real_fft = torch.fft.fft2(x_real.float())
    fake_mag = torch.abs(fake_fft)
    real_mag = torch.abs(real_fft)
    return F.l1_loss(fake_mag, real_mag)


def compute_d_loss(nets, x_real, y_org, y_trg, z_trg=None, x_ref=None,
                   lambda_reg=1.0):
    """Compute discriminator loss (from StarGAN v2 official solver.py).

    Supports both single-scale and multi-scale discriminators.
    For multi-scale, losses are averaged across scales.

    Args:
        nets: Dict with generator, discriminator, mapping_network, style_encoder.
        x_real: Real images.
        y_org: Original domain labels.
        y_trg: Target domain labels.
        z_trg: Latent code (for latent-guided).
        x_ref: Reference image (for reference-guided).
        lambda_reg: R1 regularization weight.

    Returns:
        (loss, loss_dict)
    """
    assert (z_trg is None) != (x_ref is None)

    disc = nets["discriminator"]
    is_multiscale = isinstance(disc, MultiScaleDiscriminator)

    if is_multiscale:
        discriminators = disc.discriminators
        num_scales = len(discriminators)
        x_real_center = x_real[:, 1:2, :, :] if x_real.shape[1] > 1 else x_real
        x_real_pyramid = [x_real_center]
        for _ in range(1, num_scales):
            x_real_pyramid.append(
                F.avg_pool2d(x_real_pyramid[-1], 3, stride=2, padding=1))

        # Generate fake images once (shared across scales)
        with torch.no_grad():
            if z_trg is not None:
                s_trg = nets["mapping_network"](z_trg, y_trg)
            else:
                s_trg = nets["style_encoder"](x_ref, y_trg)
            x_fake = nets["generator"](x_real, s_trg)
            # Extract center channel for discriminator (generator outputs 3ch, disc expects 1ch)
            x_fake_center = x_fake[:, 1:2, :, :] if x_fake.shape[1] > 1 else x_fake

        total_real = 0.0
        total_fake = 0.0
        total_reg = 0.0

        for i, d in enumerate(discriminators):
            x_s = x_real_pyramid[i].detach().requires_grad_().float()
            x_s.requires_grad_(True)
            out_real = d(x_s, y_org)
            total_real += adv_loss(out_real, 1)
            total_reg += r1_reg(out_real, x_s)

            x_fake_s = x_fake_center
            for _ in range(i):
                x_fake_s = F.avg_pool2d(x_fake_s, 3, stride=2, padding=1)
            out_fake = d(x_fake_s, y_trg)
            total_fake += adv_loss(out_fake, 0)

        loss_real = total_real / num_scales
        loss_fake = total_fake / num_scales
        loss_reg = total_reg / num_scales
        loss = loss_real + loss_fake + lambda_reg * loss_reg
        return loss, dict(real=loss_real.item(), fake=loss_fake.item(),
                         reg=loss_reg.item())

    # Original single-scale path
    x_real.requires_grad_()
    x_real_center = (x_real[:, 1:2, :, :] if x_real.shape[1] > 1 else x_real).float()

    # Compute fake images early so APA can reuse x_fake_center
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets["mapping_network"](z_trg, y_trg)
        else:
            s_trg = nets["style_encoder"](x_ref, y_trg)
        x_fake = nets["generator"](x_real, s_trg)
    x_fake_center = x_fake[:, 1:2, :, :] if x_fake.shape[1] > 1 else x_fake

    x_real_center.requires_grad_(True)
    out = nets["discriminator"](x_real_center, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real_center)

    out = nets["discriminator"](x_fake_center, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, dict(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())


def compute_g_loss(nets, x_real, y_org, y_trg, z_trgs=None, x_refs=None,
                   z_src=None, z_tgt=None, s_use=None, s_use_sc=None,
                   lambda_sty=1.0, lambda_ds=1.0, lambda_cyc=1.0,
                   lambda_id_l1=0.0, lambda_id_lpips=0.0, lambda_id_ssim=0.0,
                   lambda_freq=0.0, lambda_mssim=0.0,
                   lambda_cyc_lpips=0.0, lambda_cyc_ssim=0.0,
                   lambda_lpips_direct=0.0,
                   lambda_lb=0.0, lambda_seg=0.0,
                   s_full=None, target_mod_idx=None,
                   perceptual_loss=None, ssim_loss=None, msssim_loss=None):
    """Compute generator loss (from StarGAN v2 official solver.py).

    Supports both single-scale and multi-scale discriminators.
    For multi-scale, adversarial loss is averaged across scales.

    Args:
        nets: Dict with generator, discriminator, mapping_network, style_encoder.
        x_real: Real images.
        y_org: Original domain labels.
        y_trg: Target domain labels.
        z_trgs: Pair of latent codes [z1, z2] (for latent-guided).
        x_refs: Pair of reference images [ref1, ref2] (for reference-guided).
        lambda_sty: Style reconstruction loss weight.
        lambda_ds: Diversity sensitive loss weight.
        lambda_cyc: Cycle consistency loss weight.
        lambda_id_l1: Identity L1 loss weight (default 0, disabled).
        lambda_id_lpips: Identity LPIPS loss weight (default 0, disabled).
        lambda_id_ssim: Identity SSIM loss weight (default 0, disabled).
        lambda_freq: Frequency domain loss weight (default 0, disabled).
        perceptual_loss: Callable (pred, target)->scalar for LPIPS (optional).
        ssim_loss: Callable (pred, target)->scalar returning SSIM [0,1] (optional).

    Returns:
        (loss, loss_dict)
    """
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    disc = nets["discriminator"]
    is_multiscale = isinstance(disc, MultiScaleDiscriminator)

    gen = nets["generator"]
    gen_kwargs = {}
    if hasattr(gen, 'film_enabled') and gen.film_enabled and z_src is not None:
        gen_kwargs = {'z_src': z_src, 'z_tgt': z_tgt}
        if s_use is not None:
            gen_kwargs['s_use'] = s_use

    # Adversarial loss
    if z_trgs is not None:
        s_trg = nets["mapping_network"](z_trg, y_trg)
    else:
        s_trg = nets["style_encoder"](x_ref, y_trg)

    x_fake = nets["generator"](x_real, s_trg, **gen_kwargs)
    x_fake_center = x_fake[:, 1:2, :, :] if x_fake.shape[1] > 1 else x_fake

    if is_multiscale:
        discriminators = disc.discriminators
        num_scales = len(discriminators)
        total_adv = 0.0
        x_fake_scaled = x_fake_center
        for i, d in enumerate(discriminators):
            if i > 0:
                x_fake_scaled = F.avg_pool2d(x_fake_scaled, 3, stride=2, padding=1)
            total_adv += adv_loss(d(x_fake_scaled, y_trg), 1)
        loss_adv = total_adv / num_scales
    else:
        out = nets["discriminator"](x_fake_center, y_trg)
        loss_adv = adv_loss(out, 1)

    # Style reconstruction loss
    s_pred = nets["style_encoder"](x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # Diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets["mapping_network"](z_trg2, y_trg)
    else:
        s_trg2 = nets["style_encoder"](x_ref2, y_trg)
    x_fake2 = nets["generator"](x_real, s_trg2, **gen_kwargs)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # Cycle consistency loss
    s_org = nets["style_encoder"](x_real, y_org)
    rec_kwargs = dict(gen_kwargs)
    if s_use_sc is not None:
        rec_kwargs['s_use'] = s_use_sc
    x_rec = nets["generator"](x_fake, s_org, **rec_kwargs)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # Cycle LPIPS loss (use center channel for multi-modal stacked input)
    loss_cyc_lpips = torch.tensor(0.0, device=x_real.device)
    if lambda_cyc_lpips > 0 and perceptual_loss is not None:
        _cyc_rec = x_rec[:, 1:2, :, :] if x_rec.shape[1] > 1 else x_rec
        _cyc_real = x_real[:, 1:2, :, :] if x_real.shape[1] > 1 else x_real
        loss_cyc_lpips = perceptual_loss(_cyc_rec, _cyc_real)

    # Cycle SSIM loss (use center channel — SSIM kernel expects 1ch)
    loss_cyc_ssim = torch.tensor(0.0, device=x_real.device)
    if lambda_cyc_ssim > 0 and ssim_loss is not None:
        _cyc_rec = x_rec[:, 1:2, :, :] if x_rec.shape[1] > 1 else x_rec
        _cyc_real = x_real[:, 1:2, :, :] if x_real.shape[1] > 1 else x_real
        loss_cyc_ssim = 1.0 - ssim_loss(_cyc_rec, _cyc_real)

    # Direct LPIPS loss (fake vs real, use center channel)
    loss_lpips_direct = torch.tensor(0.0, device=x_real.device)
    if lambda_lpips_direct > 0 and perceptual_loss is not None:
        _dir_fake = x_fake[:, 1:2, :, :] if x_fake.shape[1] > 1 else x_fake
        _dir_real = x_real[:, 1:2, :, :] if x_real.shape[1] > 1 else x_real
        loss_lpips_direct = perceptual_loss(_dir_fake, _dir_real)

    # Frequency domain loss
    loss_freq = torch.tensor(0.0, device=x_real.device)
    if lambda_freq > 0:
        loss_freq = freq_domain_loss(x_fake, x_real)

    # MS-SSIM loss
    loss_msssim = torch.tensor(0.0, device=x_real.device)
    if lambda_mssim > 0 and msssim_loss is not None:
        loss_msssim = msssim_loss(x_fake, x_real)

    # Identity loss: generator should be identity when given its own style
    loss_id_l1 = torch.tensor(0.0, device=x_real.device)
    loss_id_lpips = torch.tensor(0.0, device=x_real.device)
    loss_id_ssim = torch.tensor(0.0, device=x_real.device)

    if lambda_id_l1 > 0 or lambda_id_lpips > 0 or lambda_id_ssim > 0:
        id_kwargs = dict(gen_kwargs)
        if s_use_sc is not None:
            id_kwargs['s_use'] = s_use_sc
        x_id = nets["generator"](x_real, s_org, **id_kwargs)
        if lambda_id_l1 > 0:
            loss_id_l1 = F.l1_loss(x_id, x_real)
        if lambda_id_lpips > 0 and perceptual_loss is not None:
            loss_id_lpips = perceptual_loss(x_id, x_real)
        if lambda_id_ssim > 0 and ssim_loss is not None:
            loss_id_ssim = 1.0 - ssim_loss(x_id, x_real)

    # Segmentation loss (MoE seg only)
    moe_seg = torch.tensor(0.0, device=x_real.device)

    if hasattr(gen, 'use_moe_film') and gen.use_moe_film and s_use is not None:
        if s_full is not None and target_mod_idx is not None and lambda_seg > 0:
            moe_seg = _get_moe_losses().compute_seg_loss(s_full, target_mod_idx) * lambda_seg

    loss = (loss_adv + lambda_sty * loss_sty
            - lambda_ds * loss_ds + lambda_cyc * loss_cyc
            + lambda_freq * loss_freq
            + lambda_mssim * loss_msssim
            + lambda_id_l1 * loss_id_l1
            + lambda_id_lpips * loss_id_lpips
            + lambda_id_ssim * loss_id_ssim
            + lambda_cyc_lpips * loss_cyc_lpips
            + lambda_cyc_ssim * loss_cyc_ssim
            + lambda_lpips_direct * loss_lpips_direct
            + moe_seg)

    loss_dict = dict(adv=loss_adv.item(), sty=loss_sty.item(),
                     ds=loss_ds.item(), cyc=loss_cyc.item(),
                     s_trg_mean=s_trg.abs().mean().item(),
                     s_pred_mean=s_pred.abs().mean().item())
    if lambda_freq > 0:
        loss_dict['freq'] = loss_freq.item()
    if lambda_mssim > 0:
        loss_dict['msssim'] = loss_msssim.item()
    if lambda_id_l1 > 0:
        loss_dict['id_l1'] = loss_id_l1.item()
    if lambda_id_lpips > 0:
        loss_dict['id_lpips'] = loss_id_lpips.item()
    if lambda_id_ssim > 0:
        loss_dict['id_ssim'] = loss_id_ssim.item()
    if lambda_cyc_lpips > 0:
        loss_dict['cyc_lpips'] = loss_cyc_lpips.item()
    if lambda_cyc_ssim > 0:
        loss_dict['cyc_ssim'] = loss_cyc_ssim.item()
    if lambda_lpips_direct > 0:
        loss_dict['lpips_direct'] = loss_lpips_direct.item()
    if lambda_seg > 0:
        loss_dict['seg'] = moe_seg.item() if isinstance(moe_seg, torch.Tensor) else moe_seg
    return loss, loss_dict


def moving_average(model, model_test, beta=0.999):
    """Exponential moving average of model parameters."""
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


# --------------------------------------------------------------------------- #
#  Build helper
# --------------------------------------------------------------------------- #

def build_stargan_v2(img_size=128, style_dim=64, latent_dim=16, num_domains=5,
                     max_conv_dim=512, input_nc=3, multiscale_disc=True,
                     num_disc_scales=3, use_unet=False, use_lap_attn=False,
                     use_film=False, film_hidden=64, use_attadain=False,
                     use_dynamic_film=False, use_moe_film=False,
                     use_skipfusion_experts=False):
    """Build StarGAN v2 networks.

    Args:
        multiscale_disc: If True, use MultiScaleDiscriminator.
        num_disc_scales: Number of scales for multi-scale discriminator.
        use_unet: If True, use UnetStarGANv2Generator (with skip connections)
            instead of StarGANv2Generator.

    Returns:
        nets: Dict of training networks.
        nets_ema: Dict of EMA networks (for inference).
    """
    if use_unet:
        from mrixfields.models.unet_generator import UnetStarGANv2Generator
        generator = UnetStarGANv2Generator(img_size, style_dim, max_conv_dim, input_nc)
    else:
        generator = StarGANv2Generator(img_size, style_dim, max_conv_dim, input_nc)
    if use_film:
        from mrixfields.models.unet_generator import FiLMUnetStarGANv2Generator
        generator = FiLMUnetStarGANv2Generator(
            img_size, style_dim, max_conv_dim, input_nc,
            film_hidden=film_hidden, film_enabled=True,
            use_dynamic_film=use_dynamic_film,
            use_moe_film=use_moe_film,
            use_skipfusion_experts=use_skipfusion_experts,
        )
    mapping_network = MappingNetwork(latent_dim, style_dim, num_domains)
    style_encoder = StyleEncoder(img_size, style_dim, num_domains, max_conv_dim, input_nc,
                                  use_lap_attn=use_lap_attn)
    if multiscale_disc:
        discriminator = MultiScaleDiscriminator(img_size, num_domains, max_conv_dim,
                                                1, num_scales=num_disc_scales)
    else:
        discriminator = StarGANv2Discriminator(img_size, num_domains, max_conv_dim, 1)

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = dict(
        generator=generator,
        mapping_network=mapping_network,
        style_encoder=style_encoder,
        discriminator=discriminator,
    )
    nets_ema = dict(
        generator=generator_ema,
        mapping_network=mapping_network_ema,
        style_encoder=style_encoder_ema,
    )

    if use_moe_film:
        from mrixfields.models.moe_mapping import SharedMappingNetwork
        nets["moe_mapping"] = SharedMappingNetwork(
            latent_dim=latent_dim, hidden=512,
            style_dim=style_dim, num_modalities=3,
        )
        nets_ema["moe_mapping"] = copy.deepcopy(nets["moe_mapping"])

    return nets, nets_ema
