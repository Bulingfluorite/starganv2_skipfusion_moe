#!/usr/bin/env python3
"""Post-training MoE diagnosis: router weight distribution, expert ablation, entropy analysis."""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from mrixfields.models.stargan_v2 import build_stargan_v2
from mrixfields.data.dataset import MultiDomainMRIDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="MoE diagnosis for mx5-moe-film checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("pretrain", {})
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build networks
    nets, _ = build_stargan_v2(
        img_size=model_cfg.get("img_size", 512),
        style_dim=model_cfg.get("style_dim", 64),
        latent_dim=model_cfg.get("latent_dim", 16),
        num_domains=model_cfg.get("num_domains", 15),
        max_conv_dim=model_cfg.get("max_conv_dim", 512),
        input_nc=model_cfg.get("input_nc", 1),
        multiscale_disc=train_cfg.get("multiscale_disc", True),
        use_unet=model_cfg.get("use_unet", True),
        use_lap_attn=model_cfg.get("use_lap_attn", True),
        use_film=model_cfg.get("use_film", True),
        film_hidden=model_cfg.get("film_hidden", 64),
        use_moe_film=model_cfg.get("use_moe_film", True),
    )
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    for key in nets:
        if key in ckpt.get("nets", ckpt):
            nets[key].load_state_dict(ckpt["nets"][key] if "nets" in ckpt else ckpt[key])
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    for key in nets:
        nets[key] = nets[key].to(device)
        nets[key].eval()
    
    gen = nets["generator"]
    moe_mapping = nets.get("moe_mapping")
    if moe_mapping is None:
        print("ERROR: moe_mapping not found in checkpoint. Is this an MoE model?")
        return
    if not hasattr(gen, 'use_moe_film') or not gen.use_moe_film:
        print("ERROR: Generator was not built with use_moe_film=True")
        return
    
    # Load dataset
    modalities = cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"])
    dataset = MultiDomainMRIDataset(
        data_dir=args.data_dir,
        modalities=modalities,
        domains=cfg.get("domains", ["0.1T", "1.5T", "3T", "5T", "7T"]),
        crop_size=model_cfg.get("img_size", 512),
        split="val",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Collect router weights per modality
    per_modality_weights = defaultdict(list)
    per_modality_entropy = defaultdict(list)
    num_samples = 0
    
    print(f"\n{'='*60}")
    print(f"Running inference on up to {args.num_samples} samples...")
    
    with torch.no_grad():
        for batch in loader:
            if num_samples >= args.num_samples:
                break
            
            x_real = batch["img"].to(device)
            y_org = batch["y_org"].to(device)
            y_trg = batch["y_trg"].to(device)
            
            B = x_real.size(0)
            for i in range(B):
                if num_samples >= args.num_samples:
                    break
                
                # Build s_use for a single sample
                target_mod_idx = y_trg[i].item() // 5  # 0=T1W, 1=T2W, 2=T2FLAIR
                y_mod_onehot = F.one_hot(torch.tensor([target_mod_idx]), 3).float().to(device)
                z = torch.randn(1, 16, device=device)
                field = ((y_trg[i].item() % 5) / 4.0).unsqueeze(0).unsqueeze(0).to(device)
                s_full = moe_mapping(z, y_mod_onehot, field)
                s_use = s_full[:, target_mod_idx*64:(target_mod_idx+1)*64]
                
                # Forward through generator
                s_style = torch.randn(1, 64, device=device)  # dummy style
                z_src = ((y_org[i].item() % 5) / 4.0).unsqueeze(0).unsqueeze(0).to(device)
                
                _ = gen(x_real[i:i+1], s_style, s_use=s_use, z_src=z_src)
                
                if hasattr(gen, 'router_weights'):
                    w = gen.router_weights.cpu()
                    ent = gen.router_entropy.cpu()
                    mod_name = modalities[target_mod_idx]
                    per_modality_weights[mod_name].append(w[0].tolist())
                    per_modality_entropy[mod_name].append(ent[0].item())
                
                num_samples += 1
            
            if num_samples % 20 == 0:
                print(f"  Processed {num_samples}/{args.num_samples} samples")
    
    # Print results
    print(f"\n{'='*60}")
    print("ROUTER WEIGHT DISTRIBUTION BY MODALITY")
    print(f"{'='*60}")
    print(f"{'Modality':<10} {'Samples':<8} {'w1_mean':<10} {'w1_std':<10} {'w2_mean':<10} {'w2_std':<10} {'Entropy':<10}")
    print("-" * 60)
    
    for mod in modalities:
        weights = per_modality_weights.get(mod, [])
        if weights:
            w_tensor = torch.tensor(weights)  # (N, 2)
            w1_mean, w1_std = w_tensor[:, 0].mean().item(), w_tensor[:, 0].std().item()
            w2_mean, w2_std = w_tensor[:, 1].mean().item(), w_tensor[:, 1].std().item()
            ent_mean = sum(per_modality_entropy[mod]) / len(per_modality_entropy[mod])
            print(f"{mod:<10} {len(weights):<8} {w1_mean:<10.4f} {w1_std:<10.4f} {w2_mean:<10.4f} {w2_std:<10.4f} {ent_mean:<10.4f}")
        else:
            print(f"{mod:<10} {'0':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
    
    print(f"\nDiagnosis complete. {num_samples} samples analyzed.")


if __name__ == "__main__":
    main()
