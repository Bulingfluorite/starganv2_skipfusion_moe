"""Unified training script for MRIxFields2026 Task 3 (StarGAN v2).

Supports:
    - StarGAN v2 (multi-domain training)
    - StarGAN v2 fine-tuning (paired, supervised)

Usage:
    # Single GPU
    python scripts/train.py --config configs/task3/stargan/609_v1.yaml --mode retro_scratch

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=2 scripts/train.py --config configs/task3/stargan/609_v1.yaml --mode retro_scratch --dist
"""

import argparse
import contextlib
import itertools
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.finetune_logger import FinetuneLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

from mrixfields.models.stargan_v2 import (
    build_stargan_v2, compute_d_loss, compute_g_loss, moving_average,
)
from mrixfields.data.dataset import PairedMRIDataset, MultiDomainMRIDataset, _stargan_aug_transform
from mrixfields.data.cached_dataset import CachedPairedDataset, CachedMultiDomainDataset
from mrixfields.losses.perceptual import PerceptualLoss
from mrixfields.losses.structure import SSIMLoss, StructureLoss
from mrixfields.data.utils import ABBR_TO_SPLIT as SPLIT_MAP
from mrixfields.env import load_env, get_data_dir, get_preprocessed_dir, get_output_dir, get_device
from mrixfields.utils_dist import init_dist, get_dist_info, is_main_process, cleanup_dist
# SPLIT_MAP: yaml config "split" value → actual data directory name.
# split is the data subset (independent from --mode which controls pipeline stage).


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_monitor_config(cfg: dict) -> tuple:
    """Parse monitor and auto_save_best config keys.

    Returns: (monitor_enabled, best_enabled, best_alpha, best_min_step, id_lpips_save_enabled)
    """
    monitor_cfg = cfg.get('monitor', {})
    monitor_enabled = monitor_cfg.get('enabled', False)
    best_cfg = cfg.get('auto_save_best', {})
    best_enabled = best_cfg.get('enabled', False)
    best_alpha = best_cfg.get('alpha', 2.0)
    best_min_step = best_cfg.get('min_step', 10000)
    # Legacy: check if id_lpips_best_save exists in pretrain section
    train_cfg = cfg.get('pretrain', cfg.get('training', {}))
    id_lpips_cfg = train_cfg.get('id_lpips_best_save', {})
    id_lpips_save_enabled = id_lpips_cfg.get('enabled', False)
    return monitor_enabled, best_enabled, best_alpha, best_min_step, id_lpips_save_enabled


# --------------------------------------------------------------------------- #
#  StarGAN v2 Training
# --------------------------------------------------------------------------- #

def train_stargan(cfg: dict, device: torch.device, use_dist: bool = False,
                  resume_ckpt: str = None):
    """Train StarGAN v2 for multi-domain translation (Task 3)."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    domains = cfg["domains"]
    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    nets, nets_ema = build_stargan_v2(
        img_size=model_cfg.get("img_size", 128),
        style_dim=model_cfg.get("style_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_domains=model_cfg.get("num_domains", 5),
        max_conv_dim=model_cfg.get("max_conv_dim", 512),
        input_nc=model_cfg.get("input_nc", 1),
        multiscale_disc=train_cfg.get("multiscale_disc", True),
        num_disc_scales=train_cfg.get("num_disc_scales", 3),
        use_unet=model_cfg.get("use_unet", False),
        use_lap_attn=model_cfg.get("use_lap_attn", False),
        use_film=model_cfg.get("use_film", False),
        film_hidden=model_cfg.get("film_hidden", 64),
        use_dynamic_film=model_cfg.get("use_dynamic_film", False),
        use_moe_film=model_cfg.get("use_moe_film", False),
        use_skipfusion_experts=model_cfg.get("use_skipfusion_experts", False),
    )
    for key in nets:
        nets[key] = nets[key].to(device)
    for key in nets_ema:
        nets_ema[key] = nets_ema[key].to(device)

    # ── Module activation audit ──
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  MODULE ACTIVATION AUDIT")
        print(f"{'='*60}")

        gen = nets["generator"]
        gen_cls = gen.__class__.__name__
        print(f"\n[Generator] {gen_cls}")
        print(f"  input_nc={model_cfg.get('input_nc',1)}  num_domains={model_cfg.get('num_domains',5)}  img_size={model_cfg.get('img_size',512)}")
        print(f"  use_unet={'✓' if model_cfg.get('use_unet') else '✗'}  use_lap_attn={'✓' if model_cfg.get('use_lap_attn') else '✗'}")
        print(f"  use_film={'✓' if model_cfg.get('use_film') else '✗'}  use_moe_film={'✓' if model_cfg.get('use_moe_film') else '✗'}  use_dynamic_film={'✓' if model_cfg.get('use_dynamic_film') else '✗'}")
        n_gen = sum(p.numel() for p in gen.parameters())

        n_skips = 0
        for attr in ['skip_fusion_32','skip_fusion_64','skip_fusion_128','skip_fusion_256']:
            if hasattr(gen, attr):
                n_skips += sum(p.numel() for p in getattr(gen, attr).parameters())
        print(f"  SkipFusion params: {n_skips:,}")

        if hasattr(gen, 'moe_film'):
            moe = gen.moe_film
            print(f"\n  ▶ FourExpertFiLM (4 experts by decoder depth)")
            for i, (s, e) in enumerate(moe.layer_ranges):
                mlp = moe.expert_mlps[i]
                n_expert = sum(p.numel() for p in mlp.parameters())
                print(f"    Expert{i} layers[{s}-{e}]: in={mlp[0].in_features} hid={mlp[0].out_features} params={n_expert:,}")
        if hasattr(gen, 'field_mlp'):
            n_fmlp = sum(p.numel() for p in gen.field_mlp.parameters())
            print(f"  ▶ FieldMLP params: {n_fmlp:,}")
        if hasattr(gen, 'film_enabled'):
            print(f"  ▶ film_enabled: {gen.film_enabled}")

        print(f"  Total Generator params: {n_gen:,}")

        mn = nets["mapping_network"]
        n_mn = sum(p.numel() for p in mn.parameters())
        print(f"\n[MappingNetwork] num_domains={model_cfg.get('num_domains',5)}")
        print(f"  shared + {model_cfg.get('num_domains',5)} unshared branches  Total: {n_mn:,}")

        se = nets["style_encoder"]
        n_se = sum(p.numel() for p in se.parameters())
        lap_attn = hasattr(se, 'lap_attn')
        print(f"\n[StyleEncoder]")
        print(f"  LaplacianAttention: {'✓ ACTIVE' if lap_attn else '✗ none'}")
        print(f"  Total params: {n_se:,}")

        disc = nets["discriminator"]
        n_disc = sum(p.numel() for p in disc.parameters())
        is_ms = hasattr(disc, 'discriminators')
        print(f"\n[Discriminator] MultiScale: {'✓' if is_ms else '✗'}  Total: {n_disc:,}")

        total = n_gen + n_mn + n_se + n_disc
        if "moe_mapping" in nets:
            mm = nets["moe_mapping"]
            n_mm = sum(p.numel() for p in mm.parameters())
            total += n_mm
            print(f"\n[SharedMappingNetwork] {mm.__class__.__name__}  Total: {n_mm:,}")

        print(f"\n{'='*60}")
        print(f"  TOTAL MODEL: {total:,} params  ({total/1e6:.1f}M)")
        print(f"{'='*60}\n")

    # DDP wrap training networks
    if use_dist:
        for key in nets:
            nets[key] = DDP(nets[key], device_ids=[device])

    num_domains = model_cfg.get("num_domains", 5)
    modalities = data_cfg.get("modalities") or []
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split", "retro_train")
    input_nc_model = model_cfg.get("input_nc", 1)

    if preprocessed_dir:
        dataset = CachedMultiDomainDataset(
            preprocessed_dir=preprocessed_dir, split=split,
            modalities=modalities, field_strengths=domains, crop_size=crop_size,
            transform=_stargan_aug_transform(crop_size),
            input_nc=input_nc_model,
        )
    else:
        split_full = SPLIT_MAP.get(split, split)
        dataset = MultiDomainMRIDataset(
            data_root=data_cfg["data_dir"], split=split_full,
            modalities=modalities, field_strengths=domains, crop_size=crop_size,
            transform=_stargan_aug_transform(crop_size),
        )

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    # Optimizers (use .module params if DDP-wrapped)
    def _params(key):
        return nets[key].module.parameters() if use_dist else nets[key].parameters()

    lr = train_cfg.get("lr", 0.0001)
    f_lr = train_cfg.get("f_lr", 1e-6)
    beta1 = train_cfg.get("beta1", 0.0)
    beta2 = train_cfg.get("beta2", 0.99)
    wd = train_cfg.get("weight_decay", 1e-4)
    use_amp = train_cfg.get("use_amp", False)

    optim_d = torch.optim.Adam(_params("discriminator"),
                                lr=lr, betas=(beta1, beta2), weight_decay=wd)
    g_param_groups = [
        {"params": _params("generator"), "lr": lr},
        {"params": _params("mapping_network"), "lr": f_lr},
        {"params": _params("style_encoder"), "lr": lr},
    ]
    if "moe_mapping" in nets:
        g_param_groups.append({"params": _params("moe_mapping"), "lr": f_lr})
    optim_g = torch.optim.Adam(g_param_groups, lr=lr, betas=(beta1, beta2), weight_decay=wd)
    optims = {
        "generator": optim_g,
        "discriminator": optim_d,
    }

    scaler = GradScaler("cuda", enabled=use_amp)

    latent_dim = model_cfg.get("latent_dim", 16)
    total_iters = train_cfg.get("total_iters", 200000)
    lambda_ds = train_cfg.get("lambda_ds", 1.0)
    ds_iter = train_cfg.get("ds_iter", 100000)
    ema_beta = train_cfg.get("ema_beta", 0.999)
    lambda_id_l1 = train_cfg.get("lambda_id_l1", 0.0)
    lambda_id_lpips = train_cfg.get("lambda_id_lpips", 0.0)
    lambda_id_ssim = train_cfg.get("lambda_id_ssim", 0.0)
    lambda_freq = train_cfg.get("lambda_freq", 0.0)
    lambda_mssim = train_cfg.get("lambda_mssim", 0.0)
    lambda_sty = train_cfg.get("lambda_sty", 1.0)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

    # Auto-save best id_lpips checkpoint
    id_lpips_save_cfg = train_cfg.get("id_lpips_best_save", {})
    id_lpips_save_enabled = id_lpips_save_cfg.get("enabled", False)
    id_lpips_save_threshold = id_lpips_save_cfg.get("threshold", 0.07)
    id_lpips_save_min_steps = id_lpips_save_cfg.get("min_steps", ds_iter)

    # Monitor / auto_save_best config (top-level keys)
    monitor_cfg = cfg.get('monitor', {})
    monitor_enabled = monitor_cfg.get('enabled', False)
    best_cfg = cfg.get('auto_save_best', {})
    best_enabled = best_cfg.get('enabled', False)
    best_alpha = best_cfg.get('alpha', 2.0)
    best_min_step = best_cfg.get('min_step', 10000)

    # Initialize identity loss modules (only when needed)
    perceptual_loss = None
    ssim_loss_fn = None
    if lambda_id_lpips > 0:
        from mrixfields.losses.perceptual import PerceptualLoss
        perceptual_loss = PerceptualLoss().to(device)
    if lambda_id_ssim > 0 or monitor_enabled:
        from mrixfields.losses.structure import SSIMLoss
        ssim_loss_fn = SSIMLoss().to(device)

    msssim_loss_fn = None
    if lambda_mssim > 0:
        from mrixfields.losses.structure import MS_SSIMLoss
        msssim_loss_fn = MS_SSIMLoss().to(device)

    # For compute_d_loss / compute_g_loss, pass unwrapped nets
    def _unwrap(d):
        return {k: (v.module if use_dist else v) for k, v in d.items()}
    # Resume from checkpoint
    step = 0
    optim_resumed = False
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        nets_unwrap = _unwrap(nets)
        ckpt_nets = ckpt.get("nets_ema", ckpt.get("nets", {}))
        for k in nets_unwrap:
            if k in ckpt_nets:
                nets_unwrap[k].load_state_dict(ckpt_nets[k], strict=False)
        for k in nets_ema:
            if k in ckpt_nets:
                nets_ema[k].load_state_dict(ckpt_nets[k], strict=False)
        if is_main_process():
            print(f"Loaded weights from {resume_ckpt} (strict=False, source={'nets_ema' if 'nets_ema' in ckpt else 'nets'})")
            print("Skipping optimizer/scaler load — starting with fresh LR schedule")


    # LR scheduler: decay in the second half of training
    lr_decay_start = train_cfg.get("lr_decay_start", total_iters // 2)
    lr_schedule_type = train_cfg.get("lr_schedule", "linear")
    def _lr_lambda(current_step):
        if current_step < lr_decay_start:
            return 1.0
        if lr_schedule_type == "cosine":
            progress = (current_step - lr_decay_start) / (total_iters - lr_decay_start)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        # original linear path
        return 1.0 - (current_step - lr_decay_start) / (total_iters - lr_decay_start + 1)

    scheduler_last_epoch = (step - 1) if (step > 0 and optim_resumed) else -1
    schedulers = {
        k: torch.optim.lr_scheduler.LambdaLR(v, _lr_lambda, last_epoch=scheduler_last_epoch)
        for k, v in optims.items()
    }
    # When optimizers start fresh but we're resuming at step > 0,
    # manually advance schedulers to the correct epoch.
    if step > 0 and not optim_resumed:
        for _ in range(step):
            for s in schedulers.values():
                s.step()

    if is_main_process():
        print(f"Training StarGAN v2: {num_domains} domains, {total_iters} iters"
              + (f" (DDP: {world_size} GPUs)" if use_dist else "")
              + (f" (resuming from step {step})" if step > 0 else "")
              + f" (LR {lr_schedule_type} decay from step {lr_decay_start})")
    epoch_count = 0
    best_score = -float('inf')
    best_ckpt_step = 0
    best_checkpoints = []  # [(score, path), ...] sorted descending
    top_k = train_cfg.get("top_k_checkpoints", 0)  # 0 = disabled (unlimited)
    accum_step = 0
    while step < total_iters:
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch_count)
        epoch_count += 1
        for batch in loader:
            if gradient_accumulation_steps > 1 and accum_step == 0:
                _reset_grad(optims)
            if step >= total_iters:
                break

            x_real = batch["image"].to(device)
            y_org = batch["domain"].to(device).long()
            x_ref = batch["ref_image"].to(device)
            y_trg = batch["ref_domain"].to(device).long()

            z_trg = torch.randn(x_real.size(0), latent_dim, device=device)
            z_trg2 = torch.randn(x_real.size(0), latent_dim, device=device)

            current_lambda_ds = lambda_ds * min(1.0, step / ds_iter)
            current_lambda_sty = lambda_sty * min(1.0, step / ds_iter)

            # Alternate between latent-guided and reference-guided
            use_latent = (step % 2 == 0)

            # D step
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_latent:
                    d_loss, d_losses = compute_d_loss(
                        _unwrap(nets), x_real, y_org, y_trg, z_trg=z_trg,
                        lambda_reg=train_cfg.get("lambda_reg", 1.0),
                    )
                else:
                    d_loss, d_losses = compute_d_loss(
                        _unwrap(nets), x_real, y_org, y_trg, x_ref=x_ref,
                        lambda_reg=train_cfg.get("lambda_reg", 1.0),
                    )
            is_accum_boundary = (accum_step + 1 == gradient_accumulation_steps)
            loss_scale = 1.0 / gradient_accumulation_steps

            d_sync = nets["discriminator"].no_sync() if (use_dist and not is_accum_boundary) else contextlib.nullcontext()
            with d_sync:
                scaler.scale(d_loss * loss_scale).backward()

            # G step
            # Build z_src (field strength) for FiLM conditioning
            # y_org is domain label 0-14: field = y_org % 5, normalized to [0, 1]
            z_src = (y_org % 5).float().unsqueeze(-1) / 4.0  # (B, 1)
            # MoE: build s_use from SharedMappingNetwork
            s_use = None
            s_full = None
            target_mod_idx = None
            if model_cfg.get("use_moe_film", False):
                target_mod_idx = y_trg // 5  # 0=T1W, 1=T2W, 2=T2FLAIR
                y_mod_onehot = F.one_hot(target_mod_idx, num_classes=3).float().to(device)
                z_mapping = torch.randn(y_trg.size(0), model_cfg.get("latent_dim", 16), device=device)
                field = (y_trg % 5).float().unsqueeze(-1) / 4.0
                nets_unwrap_moe = _unwrap(nets)
                s_full = nets_unwrap_moe["moe_mapping"](z_mapping, y_mod_onehot, field)  # (B, 192)
                # Slice target modality segment
                s_use = torch.stack([
                    s_full[i, idx*64:(idx+1)*64] 
                    for i, idx in enumerate(target_mod_idx)
                ])  # (B, 64)
                # Source modality slice for cycle consistency / identity paths
                source_mod_idx = y_org // 5  # 0=T1W, 1=T2W, 2=T2FLAIR
                s_use_sc = torch.stack([
                    s_full[i, idx*64:(idx+1)*64]
                    for i, idx in enumerate(source_mod_idx)
                ])
            else:
                s_use_sc = None
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_latent:
                    g_loss, g_losses = compute_g_loss(
                        _unwrap(nets), x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2],
                        lambda_sty=current_lambda_sty,
                        lambda_ds=current_lambda_ds,
                        lambda_cyc=train_cfg.get("lambda_cyc", 1.0),
                        lambda_id_l1=lambda_id_l1, lambda_id_lpips=lambda_id_lpips,
                        lambda_id_ssim=lambda_id_ssim, lambda_freq=lambda_freq,
                        lambda_mssim=lambda_mssim,
                        perceptual_loss=perceptual_loss, ssim_loss=ssim_loss_fn,
                        msssim_loss=msssim_loss_fn,
                        lambda_cyc_lpips=train_cfg.get("lambda_cyc_lpips", 0.0),
                        lambda_cyc_ssim=train_cfg.get("lambda_cyc_ssim", 0.0),
                        lambda_lpips_direct=train_cfg.get("lambda_lpips_direct", 0.0),
                        lambda_lb=model_cfg.get("lambda_lb", 0.0),
                        lambda_seg=model_cfg.get("lambda_seg", 0.0),
                        s_full=s_full, target_mod_idx=target_mod_idx,
                        z_src=z_src,
                        s_use=s_use, s_use_sc=s_use_sc,
                    )
                else:
                    # For reference-guided, use x_ref and a shuffled version as x_ref2
                    perm = torch.randperm(x_ref.size(0), device=device)
                    x_ref2 = x_ref[perm]
                    g_loss, g_losses = compute_g_loss(
                        _unwrap(nets), x_real, y_org, y_trg, x_refs=[x_ref, x_ref2],
                        lambda_sty=current_lambda_sty,
                        lambda_ds=current_lambda_ds,
                        lambda_cyc=train_cfg.get("lambda_cyc", 1.0),
                        lambda_id_l1=lambda_id_l1, lambda_id_lpips=lambda_id_lpips,
                        lambda_id_ssim=lambda_id_ssim, lambda_freq=lambda_freq,
                        lambda_mssim=lambda_mssim,
                        perceptual_loss=perceptual_loss, ssim_loss=ssim_loss_fn,
                        msssim_loss=msssim_loss_fn,
                        lambda_cyc_lpips=train_cfg.get("lambda_cyc_lpips", 0.0),
                        lambda_cyc_ssim=train_cfg.get("lambda_cyc_ssim", 0.0),
                        lambda_lpips_direct=train_cfg.get("lambda_lpips_direct", 0.0),
                        lambda_lb=model_cfg.get("lambda_lb", 0.0),
                        lambda_seg=model_cfg.get("lambda_seg", 0.0),
                        s_full=s_full, target_mod_idx=target_mod_idx,
                        z_src=z_src,
                        s_use=s_use, s_use_sc=s_use_sc,
                    )
            g_keys = ["generator", "mapping_network", "style_encoder"]
            if "moe_mapping" in nets:
                g_keys.append("moe_mapping")
            if use_dist and not is_accum_boundary:
                g_sync_ctxs = [nets[k].no_sync() for k in g_keys]
                with contextlib.ExitStack() as stack:
                    for ctx in g_sync_ctxs:
                        stack.enter_context(ctx)
                    scaler.scale(g_loss * loss_scale).backward()
            else:
                scaler.scale(g_loss * loss_scale).backward()

            if is_accum_boundary:
                scaler.step(optims["discriminator"])
                scaler.step(optims["generator"])
                scaler.update()

                # EMA update (on unwrapped parameters)
                nets_unwrap = _unwrap(nets)
                moving_average(nets_unwrap["generator"], nets_ema["generator"], ema_beta)
                moving_average(nets_unwrap["mapping_network"], nets_ema["mapping_network"], ema_beta)
                moving_average(nets_unwrap["style_encoder"], nets_ema["style_encoder"], ema_beta)
                if "moe_mapping" in nets_unwrap:
                    moving_average(nets_unwrap["moe_mapping"], nets_ema["moe_mapping"], ema_beta)

                # LR scheduler step (only if scaler didn't skip)
                if scaler.get_scale() > 0:
                    for sched in schedulers.values():
                        sched.step()

                _reset_grad(optims)
                accum_step = 0
                step += 1
                loader.dataset.current_step = step  # P3
            else:
                accum_step += 1
            if is_main_process() and step % train_cfg.get("print_every", 1000) == 0:
                current_lr = schedulers["generator"].get_last_lr()[0]
                d_detail = ", ".join(f"{k}: {v:.4f}" for k, v in d_losses.items())
                g_detail = ", ".join(f"{k}: {v:.4f}" for k, v in g_losses.items())
                current_mode = 'L' if use_latent else 'R'
                print(f"[{step}/{total_iters}] D: {d_loss.item():.4f} ({d_detail})")
                print(f"  G: {g_loss.item():.4f} ({g_detail}) "
                      f"sty_ds={current_lambda_sty:.2f}/{current_lambda_ds:.2f} mode={current_mode} "
                      f"lr: {current_lr:.2e}")

                # ── Module activity indicators ──
                gen_unwrapped = _unwrap(nets)["generator"]
                module_bits = []

                if g_losses.get("s_trg_mean", 0) > 0:
                    module_bits.append(f"│s_trg│={g_losses['s_trg_mean']:.3f}")

                if g_losses.get("s_pred_mean", 0) > 0:
                    module_bits.append(f"│s_pred│={g_losses['s_pred_mean']:.3f}")

                if hasattr(gen_unwrapped, 'moe_film'):
                    moe = gen_unwrapped.moe_film
                    expert_bits = []
                    for i, (start, end) in enumerate(moe.layer_ranges):
                        n_layers = end - start
                        expert_bits.append(f"E{i}({n_layers}L)")
                    module_bits.append(f"MoE: [{', '.join(expert_bits)}]")

                if s_use is not None:
                    module_bits.append(f"│s_use│={s_use.norm(dim=-1).mean().item():.3f}")

                se_unwrapped = _unwrap(nets)["style_encoder"]
                if hasattr(se_unwrapped, 'lap_attn'):
                    module_bits.append("LapAttn:✓")

                if module_bits:
                    print(f"  modules: {' | '.join(module_bits)}")

                # ── Expert gamma norm monitoring ──
                if hasattr(gen_unwrapped, 'moe_film') and hasattr(gen_unwrapped.moe_film, 'last_gammas'):
                    try:
                        last_gammas = gen_unwrapped.moe_film.last_gammas
                        layer_ranges = gen_unwrapped.moe_film.layer_ranges
                        gamma_parts = []
                        for i, (start, end) in enumerate(layer_ranges):
                            expert_gammas = last_gammas[start:end]
                            norm = sum(g.norm() for g in expert_gammas) / len(expert_gammas)
                            gamma_parts.append(f"E{i}:γ={norm:.4f}")
                            if norm < 0.01 and step >= 5000:
                                print(f"  [EXPERT WARNING] Expert {i} gamma norm < 0.01 (near-zero output) at step {step}")
                        print(f"  gamma_norms: {' '.join(gamma_parts)}")
                    except Exception:
                        pass

                if getattr(gen_unwrapped, 'use_skipfusion_experts', False):
                    try:
                        sfe_parts = []
                        for res in [32, 64, 128, 256]:
                            sfe = getattr(gen_unwrapped, f'sfe_{res}', None)
                            if sfe is not None and hasattr(sfe, 'last_gamma'):
                                norm = sfe.last_gamma.norm()
                                sfe_parts.append(f"SFE_{res}:γ={norm:.4f}")
                        if sfe_parts:
                            print(f"  sfe_gamma_norms: {' '.join(sfe_parts)}")
                    except Exception:
                        pass

                if monitor_enabled:
                    try:
                        with torch.no_grad():
                            if use_latent:
                                s_trg = nets_ema["mapping_network"](z_trg, y_trg)
                            else:
                                s_trg = nets_ema["style_encoder"](x_ref, y_trg)
                            x_fake = nets_ema["generator"](x_real, s_trg)
                            # 2.5D: generator outputs 3 channels (prev/current/next).
                            # Monitor SSIM/nRMSE on center channel only (current slice),
                            # matching what the discriminator does in compute_d_loss.
                            if x_fake.shape[1] > 1:
                                x_fake_mon = x_fake[:, 1:2, :, :]
                                x_real_mon = x_real[:, 1:2, :, :]
                            else:
                                x_fake_mon = x_fake
                                x_real_mon = x_real
                            ssim_val = 1.0 - ssim_loss_fn(x_fake_mon, x_real_mon)
                            nrmse_val = torch.norm(x_fake_mon - x_real_mon) / torch.norm(x_real_mon)
                        print(f"  SSIM: {ssim_val:.4f}, nRMSE: {nrmse_val:.4f}")
                    except Exception as e:
                        print(f"  [WARN] Monitor failed: {e}")
                    if best_enabled and step >= best_min_step:
                        try:
                            score = ssim_val - best_alpha * nrmse_val
                            if not torch.isnan(score):
                                ckpt_path = None
                                if top_k > 0:
                                    # Top-K mode: maintain sorted list
                                    ckpt_path = weights_dir / f"checkpoint_best_{step}.pth"
                                    save_dict = {
                                        "step": step,
                                        "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                                        "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
                                        "optims": {k: v.state_dict() for k, v in optims.items()},
                                        "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                                        "best_score": score,
                                    }
                                    if use_amp:
                                        save_dict["amp_scaler"] = scaler.state_dict()
                                    torch.save(save_dict, ckpt_path)
                                    best_checkpoints.append((score, ckpt_path))
                                    best_checkpoints.sort(key=lambda x: x[0], reverse=True)
                                    while len(best_checkpoints) > top_k:
                                        _, old_path = best_checkpoints.pop()
                                        if old_path.exists():
                                            old_path.unlink()
                                            print(f"  --- Removed old checkpoint: {old_path.name}")
                                    best_score = best_checkpoints[0][0]
                                    best_ckpt_step = step
                                    print(f"  >>> Best checkpoint saved: step={step}, score={score:.4f} (SSIM={ssim_val:.4f}, nRMSE={nrmse_val:.4f})")
                                else:
                                    # Legacy: unlimited best saves
                                    if score > best_score:
                                        best_score = score
                                        best_ckpt_step = step
                                        ckpt_path = weights_dir / f"checkpoint_best_{step}.pth"
                                        save_dict = {
                                            "step": step,
                                            "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                                            "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
                                            "optims": {k: v.state_dict() for k, v in optims.items()},
                                            "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                                            "best_score": best_score,
                                        }
                                        if use_amp:
                                            save_dict["amp_scaler"] = scaler.state_dict()
                                        torch.save(save_dict, ckpt_path)
                                        print(f"  >>> Best checkpoint saved: step={step}, score={score:.4f} (SSIM={ssim_val:.4f}, nRMSE={nrmse_val:.4f})")
                        except Exception as e:
                            print(f"  [WARN] Auto-save failed: {e}")

            if is_main_process() and step % train_cfg.get("save_every", 10000) == 0:
                ckpt = weights_dir / f"checkpoint_{step}.pth"
                save_dict = {
                    "step": step,
                    "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                    "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
                    "optims": {k: v.state_dict() for k, v in optims.items()},
                    "schedulers": {k: v.state_dict() for k, v in schedulers.items()},
                }
                if use_amp:
                    save_dict["amp_scaler"] = scaler.state_dict()
                torch.save(save_dict, ckpt)
                print(f"Saved: {ckpt}")

            # Auto-save best id_lpips checkpoint
            if id_lpips_save_enabled and is_main_process():
                current_id_lpips = g_losses.get("id_lpips", None)
                if (current_id_lpips is not None
                        and current_id_lpips < id_lpips_save_threshold
                        and step >= id_lpips_save_min_steps):
                    ckpt = weights_dir / f"checkpoint_best_idlpips_{step}.pth"
                    save_dict = {
                        "step": step,
                        "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                        "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
                        "optims": {k: v.state_dict() for k, v in optims.items()},
                    }
                    if use_amp:
                        save_dict["amp_scaler"] = scaler.state_dict()
                    torch.save(save_dict, ckpt)
                    print(f"*** Best id_lpips save: {current_id_lpips:.4f} at step {step} ***")
                    print(f"Saved: {ckpt}")

    if is_main_process():
        print(f"StarGAN v2 training complete. Saved to: {output_dir}")


def _adapt_checkpoint_state_dict(ckpt_sd: dict, model_sd: dict, model_key: str) -> dict:
    """Adapt checkpoint state dict to match model architecture (Phase 1→Phase 2).

    Handles: 1ch→3ch conv expansion (replicate), 15→5 domain truncation,
             new layer skipping (unet/lap_attn).
    """
    adapted = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        target_shape = model_sd[k].shape
        if v.shape == target_shape:
            adapted[k] = v
            continue
        # 4D tensor: Conv2d weight [out_ch, in_ch, h, w]
        if v.dim() == 4:
            # Case 1: 1ch→Nch input expansion (from_rgb, first style_enc/disc conv)
            if v.shape[1] != target_shape[1] and v.shape[1] == 1 and target_shape[1] % v.shape[1] == 0:
                ratio = target_shape[1] // v.shape[1]
                adapted[k] = v.repeat(1, ratio, 1, 1)
                print(f"  [adapt] {k}: 1ch→{target_shape[1]}ch input (replicated {ratio}x)")
                continue
            # Case 2: 1→N output expansion (to_rgb)
            if v.shape[0] != target_shape[0] and v.shape[0] == 1 and target_shape[0] % v.shape[0] == 0:
                ratio = target_shape[0] // v.shape[0]
                adapted[k] = v.repeat(ratio, 1, 1, 1)
                print(f"  [adapt] {k}: 1ch→{target_shape[0]}ch output (replicated {ratio}x)")
                continue
            # Case 3: domain count truncation (15→5 in discriminator output)
            if v.shape[0] > target_shape[0] and v.shape[1:] == target_shape[1:]:
                adapted[k] = v[:target_shape[0]]
                print(f"  [adapt] {k}: {v.shape[0]}→{target_shape[0]} domains (truncated)")
                continue
        # 1D tensor: bias or 1D param — truncate if bigger
        if v.dim() == 1 and v.shape[0] > target_shape[0]:
            adapted[k] = v[:target_shape[0]]
            print(f"  [adapt] {k}: {v.shape[0]}→{target_shape[0]} (truncated)")
            continue
        # Can't adapt — skip with strict=False
        skipped.append(f"{k}:{list(v.shape)}!={list(target_shape)}")
    if skipped:
        print(f"  [skip:{model_key}] {len(skipped)} params: {skipped[:3]}...")
    return adapted


def train_stargan_finetune(cfg: dict, device: torch.device, use_dist: bool = False,
                           resume_ckpt: str = None):
    """Fine-tune pretrained StarGAN v2 with paired data (L1 + LPIPS).

    Uses prospective paired data to supervise the multi-domain generator.
    For each batch, samples paired (source, target) images across all domain
    combinations, translates source→target, and computes reconstruction loss.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    domains = cfg["domains"]
    finetune_cfg = cfg.get("finetune", {})
    csv_every = train_cfg.get("csv_every", None)
    print_every = train_cfg.get("print_every", 100)
    freeze_cfg = finetune_cfg.get("freeze_encoder", {})
    freeze_enabled = freeze_cfg.get("enabled", False)
    freeze_epochs = freeze_cfg.get("freeze_epochs", 30)
    curriculum_cfg = finetune_cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    curriculum_phases = curriculum_cfg.get("phases", [])
    weight_7t = finetune_cfg.get("weight_7t", 1.0)

    def _curriculum_filter(wrapped_list, field_pairs):
        """Filter wrapped datasets to only include specified field_strength pairs."""
        allowed = set()
        for pair in field_pairs:
            s = str(pair[0])
            t = str(pair[1])
            if not s.endswith("T"):
                s += "T"
            if not t.endswith("T"):
                t += "T"
            allowed.add((s, t))
        return [w for i, w in enumerate(wrapped_list)
                if (all_pairs[i][1], all_pairs[i][2]) in allowed]

    rank, world_size = get_dist_info()
    output_dir = Path(train_cfg["output_dir"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    if is_main_process():
        weights_dir.mkdir(parents=True, exist_ok=True)

    logger = FinetuneLogger(output_dir, print_every=print_every, csv_every=csv_every) \
        if is_main_process() else None

    # Build StarGAN v2 networks
    nets, nets_ema = build_stargan_v2(
        img_size=model_cfg.get("img_size", 128),
        style_dim=model_cfg.get("style_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_domains=model_cfg.get("num_domains", 5),
        max_conv_dim=model_cfg.get("max_conv_dim", 512),
        input_nc=model_cfg.get("input_nc", 1),
        multiscale_disc=train_cfg.get("multiscale_disc", True),
        num_disc_scales=train_cfg.get("num_disc_scales", 3),
        use_unet=model_cfg.get("use_unet", False),
        use_lap_attn=model_cfg.get("use_lap_attn", False),
        use_film=model_cfg.get("use_film", False),
        film_hidden=model_cfg.get("film_hidden", 64),
        use_dynamic_film=model_cfg.get("use_dynamic_film", False),
        use_moe_film=model_cfg.get("use_moe_film", False),
        use_skipfusion_experts=model_cfg.get("use_skipfusion_experts", False),
    )

    # Load pretrained weights (skipped when resuming — resume_ckpt will overwrite below)
    pretrain_ckpt = cfg.get("pretrain", {}).get("checkpoint")
    if resume_ckpt:
        if is_main_process():
            print(f"Resume requested ({resume_ckpt}); skipping pretrain checkpoint load")
    elif pretrain_ckpt:
        state_dict = torch.load(pretrain_ckpt, map_location=device)
        source_key = "nets_ema" if "nets_ema" in state_dict else "nets"
        if is_main_process():
            print(f"Loading pretrained StarGAN v2 from: {pretrain_ckpt} (source: {source_key})")
            print(f"  Adapting pretrain checkpoint to current model architecture")
        for k in nets_ema:
            ckpt_sd = state_dict.get(source_key, {}).get(k, {})
            if not ckpt_sd:
                if is_main_process():
                    print(f"  [warn] No {source_key}[{k}] in checkpoint, skipping")
                continue
            adapted = _adapt_checkpoint_state_dict(ckpt_sd, nets_ema[k].state_dict(), k)
            nets_ema[k].load_state_dict(adapted, strict=False)
            # Initialize training nets from EMA (better starting point)
            if k in nets:
                nets[k].load_state_dict(adapted, strict=False)
        if is_main_process():
            print(f"  Pretrained weights loaded (strict=False, mismatched layers skipped).")
    else:
        if is_main_process():
            print("No pretrained checkpoint, training StarGAN v2 finetune from scratch")

    # EWC snapshot (anti-forgetting): save reference params for EWC penalty
    ewc_enabled = finetune_cfg.get("lambda_ewc", 0.0) > 0
    ewc_opt_params = {}
    ewc_targets = []
    if ewc_enabled:
        from mrixfields.models.moe_losses import snapshot_model_params
        ewc_targets = finetune_cfg.get("ewc_on", [])
        for target in ewc_targets:
            if target in nets:
                parts = target.split(".")
                obj = nets[parts[0]]
                for p in parts[1:]:
                    obj = getattr(obj, p)
                ewc_opt_params.update(snapshot_model_params(obj, prefix=target + "."))

    for key in nets:
        nets[key] = nets[key].to(device)
    for key in nets_ema:
        nets_ema[key] = nets_ema[key].to(device)

    freeze_cfg = finetune_cfg.get("freeze_encoder", {})
    freeze_encoder_only = isinstance(freeze_cfg, dict) and freeze_cfg.get("enabled", False)
    if use_dist:
        for key in nets:
            nets[key] = DDP(nets[key], device_ids=[device],
                             find_unused_parameters=freeze_encoder_only)

    if freeze_enabled or freeze_encoder_only:
        gen = nets["generator"].module if use_dist else nets["generator"]
        frozen_count = 0
        for name, param in gen.named_parameters():
            if name.startswith("encode."):
                param.requires_grad = False
                frozen_count += 1
        if is_main_process():
            print(f"[Freeze] Froze {frozen_count} encoder params "
                  f"(generator.encode.*), unfreeze at epoch {freeze_epochs}")

    # Build paired datasets for all (modality, src_field, tgt_field) combinations.
    # Only same-modality pairs are used (cross-modality pairing is not defined).
    from mrixfields.data.utils import FIELD_TO_DOMAIN, get_joint_domain
    modalities = data_cfg.get("modalities") or []
    use_joint = len(modalities) > 1
    crop_size_raw = data_cfg.get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    preprocessed_dir = data_cfg.get("preprocessed_dir")
    split = train_cfg.get("split") or finetune_cfg.get("split", "pro_train")

    def _domain_label(modality: str, fs: str) -> int:
        return get_joint_domain(modality, fs) if use_joint else FIELD_TO_DOMAIN[fs]

    all_pairs = []
    for modality in modalities:
        for src_fs in domains:
            for tgt_fs in domains:
                if src_fs == tgt_fs:
                    continue
                try:
                    if preprocessed_dir:
                        ds = CachedPairedDataset(
                            preprocessed_dir=preprocessed_dir, split=split,
                            modality=modality, source_field=src_fs,
                            target_field=tgt_fs, crop_size=crop_size,
                            input_nc=model_cfg.get("input_nc", 1),
                        )
                    else:
                        split_full = SPLIT_MAP.get(split, split)
                        ds = PairedMRIDataset(
                            data_root=data_cfg["data_dir"], split=split_full,
                            modality=modality, source_field=src_fs,
                            target_field=tgt_fs, crop_size=crop_size,
                        )
                    all_pairs.append((modality, src_fs, tgt_fs, ds))
                except (FileNotFoundError, ValueError):
                    continue

    if not all_pairs:
        raise FileNotFoundError("No paired data found for any (modality, src, tgt) combination")

    # Combine all pairs into one dataset with domain labels
    from torch.utils.data import ConcatDataset

    class _PairedWithDomains(torch.utils.data.Dataset):
        """Wraps a PairedMRIDataset to add domain indices."""
        def __init__(self, dataset, src_domain, tgt_domain, tgt_fs=""):
            self.dataset = dataset
            self.src_domain = src_domain
            self.tgt_domain = tgt_domain
            self.tgt_fs = tgt_fs
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            batch = self.dataset[idx]
            batch["src_domain"] = self.src_domain
            batch["tgt_domain"] = self.tgt_domain
            return batch

    wrapped = []
    for modality, src_fs, tgt_fs, ds in all_pairs:
        src_d = _domain_label(modality, src_fs)
        tgt_d = _domain_label(modality, tgt_fs)
        wrapped.append(_PairedWithDomains(ds, src_d, tgt_d, tgt_fs=tgt_fs))
        if is_main_process():
            print(f"  Paired data: {modality} {src_fs}→{tgt_fs} "
                  f"(domain {src_d}→{tgt_d}): {len(ds)} samples")
    _full_wrapped = wrapped

    if curriculum_enabled and curriculum_phases:
        phase0 = None
        for phase in curriculum_phases:
            if phase["start_epoch"] <= 0 < phase["end_epoch"]:
                phase0 = phase
                break
        if phase0 is not None:
            wrapped = _curriculum_filter(_full_wrapped, phase0.get("field_pairs", []))
        if is_main_process() and phase0 is not None:
            print(f"[Curriculum] Phase init: epochs {phase0['start_epoch']}-{phase0['end_epoch']}, "
                  f"pairs: {phase0.get('field_pairs', [])}, datasets: {len(wrapped)}")

    combined_dataset = ConcatDataset(wrapped)

    batch_size = train_cfg.get("batch_size", 4)
    if use_dist:
        batch_size = batch_size // world_size

    # 7T-weighted sampling
    if weight_7t > 1.0:
        weights = []
        n_7t = 0
        for w in wrapped:
            if w.tgt_fs == "7T":
                weights.extend([weight_7t] * len(w))
                n_7t += len(w)
            else:
                weights.extend([1.0] * len(w))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(combined_dataset), replacement=True)
        if is_main_process():
            print(f"7T samples weighted ×{weight_7t:.1f} "
                  f"(N_7T={n_7t}, total={len(combined_dataset)})")
    elif use_dist:
        sampler = DistributedSampler(combined_dataset, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )

    # Retrospective replay loader (anti-forgetting)
    replay_ratio = finetune_cfg.get("replay_ratio", 0.0)
    retro_loader = None
    if replay_ratio > 0:
        if preprocessed_dir:
            from mrixfields.data.cached_dataset import CachedMultiDomainDataset
            retro_dataset = CachedMultiDomainDataset(
                preprocessed_dir=preprocessed_dir,
                split=finetune_cfg.get("retro_split", "retro_train"),
                modalities=modalities,
                field_strengths=domains,
                crop_size=crop_size,
                input_nc=model_cfg.get("input_nc", 1),
            )
        else:
            from mrixfields.data.dataset import MultiDomainMRIDataset
            retro_dataset = MultiDomainMRIDataset(
                data_root=data_cfg.get("data_dir", data_cfg["data_dir"]),
                modalities=modalities,
                field_strengths=domains,
                crop_size=crop_size,
                split=finetune_cfg.get("retro_split", "retro_train"),
            )
        retro_sampler = DistributedSampler(retro_dataset, shuffle=True) if use_dist else None
        retro_bs = max(1, int(batch_size * replay_ratio))
        retro_loader = DataLoader(retro_dataset, batch_size=retro_bs,
            shuffle=(retro_sampler is None), sampler=retro_sampler,
            num_workers=2, drop_last=False, pin_memory=True)
    # Optimizer: only fine-tune generator and mapping network
    def _params(key):
        return nets[key].module.parameters() if use_dist else nets[key].parameters()

    lr = train_cfg.get("lr", 5e-5)
    use_amp = train_cfg.get("use_amp", False)
    opt_params = list(itertools.chain(_params("generator"), _params("mapping_network")))
    if "moe_mapping" in nets:
        opt_params.extend(_params("moe_mapping"))
    optimizer = torch.optim.Adam(
        opt_params,
        lr=lr,
        betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
        weight_decay=train_cfg.get("weight_decay", finetune_cfg.get("weight_decay", 0.0)),
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_lpips = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)

    lambda_l1 = finetune_cfg.get("loss_l1", 1.0)
    lambda_lpips = finetune_cfg.get("loss_lpips", 0.1)
    lambda_ssim = finetune_cfg.get("loss_ssim", 0.0)
    lambda_edge = finetune_cfg.get("loss_edge", 0.0)
    if lambda_edge > 0:
        criterion_edge = StructureLoss(ssim_weight=0.0, edge_weight=1.0).to(device)

    epochs = train_cfg.get("epochs", 50)
    latent_dim = model_cfg.get("latent_dim", 16)
    ema_beta = train_cfg.get("ema_beta", finetune_cfg.get("ema_beta", 0.999))

    def _unwrap(d):
        return {k: (v.module if use_dist else v) for k, v in d.items()}

    # Resume from finetune checkpoint (overrides pretrain init)
    start_epoch = 0
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        nets_unwrap = _unwrap(nets)
        if "nets" in ckpt:
            for k in nets_unwrap:
                if k in ckpt["nets"]:
                    nets_unwrap[k].load_state_dict(ckpt["nets"][k])
        if "nets_ema" in ckpt:
            for k in nets_ema:
                if k in ckpt["nets_ema"]:
                    nets_ema[k].load_state_dict(ckpt["nets_ema"][k])
        start_epoch = ckpt.get("epoch", 0)
        if use_amp and "amp_scaler" in ckpt:
            scaler.load_state_dict(ckpt["amp_scaler"])
            if is_main_process():
                print(f"[AMP] Loaded scaler state from finetune checkpoint")
        elif use_amp:
            if is_main_process():
                print("[AMP] No scaler state in finetune checkpoint, initializing new GradScaler")
        if is_main_process():
            print(f"Resumed StarGAN v2 finetune from {resume_ckpt} at epoch {start_epoch}")

    if is_main_process():
        print(f"StarGAN v2 fine-tuning: {len(all_pairs)} pairs, {len(combined_dataset)} samples, "
              f"epochs {start_epoch+1}-{epochs}" + (f" (DDP: {world_size} GPUs)" if use_dist else ""))

    _curriculum_phase = -1

    for epoch in range(start_epoch, epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        if freeze_enabled and epoch == freeze_epochs:
            gen = nets["generator"].module if use_dist else nets["generator"]
            unfrozen_count = 0
            for name, param in gen.named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
            opt_params = list(itertools.chain(_params("generator"), _params("mapping_network")))
            if "moe_mapping" in nets:
                opt_params.extend(_params("moe_mapping"))
            optimizer = torch.optim.Adam(
                opt_params,
                lr=lr,
                betas=(train_cfg.get("beta1", 0.5), train_cfg.get("beta2", 0.999)),
                weight_decay=train_cfg.get("weight_decay", finetune_cfg.get("weight_decay", 0.0)),
            )
            if is_main_process():
                print(f"[Unfreeze] Unfroze all {unfrozen_count} generator params, "
                      f"re-created optimizer at epoch {epoch+1}")

        if curriculum_enabled and curriculum_phases:
            new_phase = -1
            for pi, ph in enumerate(curriculum_phases):
                if ph["start_epoch"] <= epoch < ph["end_epoch"]:
                    new_phase = pi
                    cur_phase = ph
                    break
            if new_phase >= 0 and new_phase != _curriculum_phase:
                _curriculum_phase = new_phase
                field_pairs = cur_phase.get("field_pairs", [])
                filtered_wrapped = _curriculum_filter(_full_wrapped, field_pairs)
                combined_dataset = ConcatDataset(filtered_wrapped)
                if weight_7t > 1.0:
                    weights = []
                    n_7t = 0
                    for w in filtered_wrapped:
                        if w.tgt_fs == "7T":
                            weights.extend([weight_7t] * len(w))
                            n_7t += len(w)
                        else:
                            weights.extend([1.0] * len(w))
                    sampler = WeightedRandomSampler(
                        weights, num_samples=len(combined_dataset), replacement=True)
                elif use_dist:
                    sampler = DistributedSampler(combined_dataset, shuffle=True)
                else:
                    sampler = None
                loader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    num_workers=train_cfg.get("num_workers", 4),
                    drop_last=True,
                    pin_memory=True,
                )
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
                if is_main_process():
                    print(f"[Curriculum] Phase {new_phase}: epochs {cur_phase['start_epoch']}-{cur_phase['end_epoch']}, "
                          f"pairs: {field_pairs}, datasets: {len(filtered_wrapped)}, samples: {len(combined_dataset)}")

        nets["generator"].train()
        nets["mapping_network"].train()
        batch_idx = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", mininterval=300.0) if is_main_process() else loader

        for batch in pbar:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            tgt_domain = batch["tgt_domain"].to(device).long()

            # Retrospective replay batch (anti-forgetting)
            retro_batch = None
            if retro_loader is not None:
                if 'retro_iter' not in locals() or batch_idx % len(retro_loader) == 0:
                    retro_iter = iter(retro_loader)
                try:
                    retro_batch = next(retro_iter)
                except StopIteration:
                    retro_iter = iter(retro_loader)
                    retro_batch = next(retro_iter)

            # MoE: build s_use from SharedMappingNetwork
            nets_unwrap = _unwrap(nets)
            s_use = None
            s_full = None
            target_mod_idx = None
            src_domain = batch["src_domain"].to(device).long()
            z_src = (src_domain % 5).float().unsqueeze(-1) / 4.0
            if model_cfg.get("use_moe_film", False):
                target_mod_idx = tgt_domain // 5
                y_mod_onehot = F.one_hot(target_mod_idx, 3).float().to(device)
                z_map = torch.randn(tgt_domain.size(0), model_cfg.get("latent_dim", 16), device=device)
                field_t = (tgt_domain % 5).float().unsqueeze(-1) / 4.0
                s_full = nets_unwrap["moe_mapping"](z_map, y_mod_onehot, field_t)
                s_use = torch.stack([s_full[i, idx*64:(idx+1)*64] for i, idx in enumerate(target_mod_idx)])

            # Generate style code for target domain
            z = torch.randn(source.size(0), latent_dim, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                s_trg = nets_unwrap["mapping_network"](z, tgt_domain)
                gen_kwargs = {}
                if s_use is not None:
                    gen_kwargs["s_use"] = s_use
                    gen_kwargs["z_src"] = z_src
                pred = nets_unwrap["generator"](source, s_trg, **gen_kwargs)

                loss_l1_val = criterion_l1(pred, target)
                loss_lpips_val = criterion_lpips(pred, target)

                # Retrospective replay loss (identity preservation on retro data)
                loss_retro = torch.tensor(0.0, device=device)
                lambda_retro = finetune_cfg.get("lambda_retro", 0.5)
                if retro_batch is not None and lambda_retro > 0:
                    r_img = retro_batch["image"].to(device)
                    r_org = retro_batch["domain"].to(device)
                    r_trg = retro_batch["ref_domain"].to(device)
                    s_org_r = nets_unwrap["style_encoder"](r_img, r_org)
                    r_kwargs = {}
                    if model_cfg.get("use_moe_film", False):
                        tmod_r = r_org // 5
                        yoh_r = F.one_hot(tmod_r, 3).float().to(device)
                        zr = torch.randn(r_trg.size(0), 16, device=device)
                        fr = (r_trg % 5).float().unsqueeze(-1) / 4.0
                        sr = nets_unwrap["moe_mapping"](zr, yoh_r, fr)
                        su_r = torch.stack([sr[i, idx*64:(idx+1)*64] for i, idx in enumerate(tmod_r)])
                        r_kwargs["s_use"] = su_r
                        r_kwargs["z_src"] = (r_org % 5).float().unsqueeze(-1) / 4.0
                    x_id_r = nets_unwrap["generator"](r_img, s_org_r, **r_kwargs)
                    loss_retro = F.l1_loss(x_id_r, r_img)

                # EWC loss (elastic weight consolidation)
                loss_ewc = torch.tensor(0.0, device=device)
                lambda_ewc = finetune_cfg.get("lambda_ewc", 0.0)
                if ewc_enabled and lambda_ewc > 0:
                    from mrixfields.models.moe_losses import compute_ewc_loss
                    for ewc_key in ewc_targets:
                        parts = ewc_key.split(".")
                        obj = nets_unwrap[parts[0]]
                        for p in parts[1:]:
                            obj = getattr(obj, p)
                        loss_ewc += compute_ewc_loss(obj, ewc_opt_params, {}, lambda_ewc=1.0)
                    loss_ewc = loss_ewc * lambda_ewc

                loss = lambda_l1 * loss_l1_val + lambda_lpips * loss_lpips_val \
                     + lambda_retro * loss_retro + loss_ewc
                if lambda_ssim > 0:
                    loss = loss + lambda_ssim * criterion_ssim(pred.float(), target.float())
                if lambda_edge > 0:
                    loss = loss + lambda_edge * criterion_edge(pred.float(), target.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            moving_average(nets_unwrap["generator"], nets_ema["generator"], ema_beta)
            moving_average(nets_unwrap["mapping_network"], nets_ema["mapping_network"], ema_beta)
            if "moe_mapping" in nets_unwrap:
                moving_average(nets_unwrap["moe_mapping"], nets_ema["moe_mapping"], ema_beta)

            if logger is not None:
                # Divide by current scale to get original (unscaled) loss values
                current_scale = scaler.get_scale() if use_amp else 1.0
                logger.log_batch(epoch + 1, batch_idx,
                                loss.item() / current_scale,
                                loss_l1_val.item(),
                                loss_lpips_val.item())
            batch_idx += 1

        if logger is not None:
            logger.log_epoch(epoch + 1)

        if is_main_process() and (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            ckpt = weights_dir / f"checkpoint_epoch{epoch+1}.pth"
            save_dict = {
                "epoch": epoch + 1,
                "nets": {k: v.state_dict() for k, v in _unwrap(nets).items()},
                "nets_ema": {k: v.state_dict() for k, v in nets_ema.items()},
            }
            if use_amp:
                save_dict["amp_scaler"] = scaler.state_dict()
            torch.save(save_dict, ckpt)

    if logger is not None:
        logger.close()
    if is_main_process():
        print(f"StarGAN v2 fine-tuning complete. Saved to: {output_dir}")


def _reset_grad(optims):
    for optim in optims.values():
        optim.zero_grad()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def _save_actual_config(cfg: dict, output_dir: Path):
    """Save the merged runtime config (yaml + CLI overrides) for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_actual.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _build_pretrain_cfg(cfg: dict, output_dir: str) -> dict:
    """Build a config dict for the pretrain step by merging top-level and pretrain sections."""
    pretrain_cfg = cfg["pretrain"]
    return {
        **cfg,
        "training": {
            **pretrain_cfg,
            "output_dir": output_dir,
        },
    }


def _build_stargan_finetune_cfg(cfg: dict, output_dir: str, pretrain_ckpt: str) -> dict:
    """Build a config dict for StarGAN v2 fine-tuning (keeps multi-domain architecture)."""
    ft_cfg = cfg["finetune"]
    return {
        **cfg,
        "pretrain": {"method": "stargan_v2", "checkpoint": pretrain_ckpt},
        "training": {
            **ft_cfg,
            "output_dir": output_dir,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="retro_scratch",
                        choices=["retro_scratch", "pro_scratch", "pro_pretrained"],
                        help="Training mode: retro_scratch (pretrain only), pro_scratch (supervised from scratch), pro_pretrained (finetune pretrained model)")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pretrained generator checkpoint (skip pretrain step)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training (StarGAN v2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dist", action="store_true",
                        help="Enable distributed training (use with torchrun)")
    args = parser.parse_args()

    # Initialize distributed training
    use_dist = args.dist
    if use_dist:
        local_rank = init_dist()
        device = torch.device(f"cuda:{local_rank}")
    else:
        cfg_tmp = load_config(args.config)  # peek for device
        load_env()
        env_device = get_device()
        device = torch.device(args.device if args.device != "cuda" else env_device)
        if not torch.cuda.is_available():
            device = torch.device("cpu")

    cfg = load_config(args.config)

    # Inject paths from .env into config
    load_env()
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = get_data_dir()
    cfg["data"]["preprocessed_dir"] = get_preprocessed_dir()

    method = cfg["method"]
    task_name = cfg.get("task_name", f"{method}_unknown")
    rank, world_size = get_dist_info()

    # Record runtime info
    cfg["_runtime"] = {
        "mode": args.mode,
        "config_path": str(Path(args.config).resolve()),
        "pretrain_ckpt": args.pretrain_ckpt,
        "device": str(device),
        "dist": use_dist,
        "world_size": world_size,
    }

    # Output dirs with mode suffix
    output_dir = get_output_dir()
    output_base = Path(output_dir) / task_name / method / args.mode

    if is_main_process():
        print(f"Device: {device}")
        print(f"Task:   {task_name}")
        print(f"Mode:   {args.mode}")
        if use_dist:
            print(f"DDP:    {world_size} GPUs")
        print(f"Output: {output_base}")
        _save_actual_config(cfg, output_base)

    # Auto-detect resume checkpoint if not explicitly specified.
    # Matches both epoch-based (checkpoint_epoch{N}.pth) and step-based
    # (checkpoint_{step}.pth, used by StarGAN v2 pretrain) naming.
    if args.resume is None:
        weights_dir = output_base / "weights"
        if weights_dir.exists():
            def _ckpt_num(p):
                suffix = p.stem[len("checkpoint_"):]
                if suffix.startswith("epoch"):
                    suffix = suffix[len("epoch"):]
                try:
                    return int(suffix)
                except ValueError:
                    return None

            numbered = []
            for p in weights_dir.glob("checkpoint_*.pth"):
                n = _ckpt_num(p)
                if n is not None:
                    numbered.append((n, p))
            numbered.sort()
            if numbered:
                args.resume = str(numbered[-1][1])
                if is_main_process():
                    print(f"Auto-resume detected: {args.resume}")

    # ------------------------------------------------------------------ #
    #  Mode: retro_scratch — pretrain only (unpaired, retrospective data)
    # ------------------------------------------------------------------ #
    if args.mode == "retro_scratch":
        output_dir = str(output_base)
        pretrain_run_cfg = _build_pretrain_cfg(cfg, output_dir)

        if method == "stargan_v2":
            train_stargan(pretrain_run_cfg, device, use_dist, resume_ckpt=args.resume)
        else:
            raise ValueError(f"Unknown method for pretrain: {method} (only stargan_v2 is supported)")

    # ------------------------------------------------------------------ #
    #  Mode: pro_pretrained — finetune a pretrained model (paired, prospective)
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_pretrained":
        pretrain_ckpt = args.pretrain_ckpt
        if pretrain_ckpt is None:
            pretrain_step = cfg.get("finetune", {}).get("pretrain_step", 500000)
            default_ckpt = output_base.parent / "retro_scratch" / "weights" / f"checkpoint_{pretrain_step}.pth"
            if not default_ckpt.exists():
                raise FileNotFoundError(
                    f"No pretrain checkpoint found at {default_ckpt}. "
                    f"Run --mode retro_scratch first, or use --pretrain_ckpt to specify."
                )
            pretrain_ckpt = str(default_ckpt)
        if is_main_process():
            print(f"Loading pretrain checkpoint: {pretrain_ckpt}")

        if method == "stargan_v2":
            ft_run_cfg = _build_stargan_finetune_cfg(cfg, str(output_base), pretrain_ckpt)
            train_stargan_finetune(ft_run_cfg, device, use_dist, resume_ckpt=args.resume)
        else:
            raise ValueError(f"Unknown method for pro_pretrained: {method} (only stargan_v2 is supported)")

    # ------------------------------------------------------------------ #
    #  Mode: pro_scratch — supervised training from scratch (paired, prospective)
    # ------------------------------------------------------------------ #
    elif args.mode == "pro_scratch":
        if is_main_process():
            print("Mode=pro_scratch: training from random initialization (no pretrain checkpoint)")
        if method == "stargan_v2":
            ft_run_cfg = _build_stargan_finetune_cfg(cfg, str(output_base), None)
            train_stargan_finetune(ft_run_cfg, device, use_dist, resume_ckpt=args.resume)
        else:
            raise ValueError(f"Unknown method for pro_scratch: {method} (only stargan_v2 is supported)")

    if is_main_process():
        print(f"\nTraining complete ({args.mode}). Output: {output_base}")

    if use_dist:
        cleanup_dist()


if __name__ == "__main__":
    main()
