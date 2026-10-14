"""Chain inference script: multi-step StarGAN v2 field strength conversion.

Instead of direct source→target conversion, this script steps through
adjacent field strengths to accumulate gradual transformations.

Example (0.1T → 7T):
    0.1T → 1.5T → 3T → 5T → 7T  (4 steps, intermediate results in memory)

Usage:
    python scripts/chain_infer.py \
        --config configs/task3/stargan/any_to_any_all_modalities.yaml \
        --checkpoint $OUTPUT_DIR/task3_any_to_any_multimodal/stargan_v2/retro_scratch/weights/model_final.pth \
        --input_dir $DATA_DIR/Training_prospective/T1W/0.1T/ \
        --output_dir $INFERENCE_DIR/chain/ \
        --source_field 0.1T \
        --target_field 7T
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.models.stargan_v2 import build_stargan_v2
from mrixfields.data.utils import load_nifti, save_nifti, FIELD_STRENGTHS, FIELD_TO_DOMAIN
from mrixfields.data.transforms import CenterCropOrPad
from mrixfields.env import load_env, get_inference_dir, get_device


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_generator_state(state_dict: dict) -> dict:
    if "model" in state_dict:
        for prefix in ("netG.", "netG_AB."):
            gen_state = {k[len(prefix):]: v for k, v in state_dict["model"].items()
                         if k.startswith(prefix)}
            if gen_state:
                return gen_state
        raise ValueError(
            f"Checkpoint has 'model' key but no generator prefix found. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict['model'])}"
        )
    if "generator" in state_dict:
        return state_dict["generator"]
    if all(not k.startswith("net") for k in state_dict):
        return state_dict
    raise ValueError(
        f"Unrecognized checkpoint format. Top-level keys: {list(state_dict.keys())}"
    )


def load_generator(cfg: dict, checkpoint_path: str, device: torch.device):
    method = cfg["method"]
    model_cfg = cfg["model"]

    if method == "stargan_v2":
        nets, nets_ema = build_stargan_v2(
            img_size=model_cfg.get("img_size", 128),
            style_dim=model_cfg.get("style_dim", 64),
            latent_dim=model_cfg.get("latent_dim", 16),
            num_domains=model_cfg.get("num_domains", 5),
            max_conv_dim=model_cfg.get("max_conv_dim", 512),
            input_nc=model_cfg.get("input_nc", 1),
            use_unet=model_cfg.get("use_unet", False),
            use_lap_attn=model_cfg.get("use_lap_attn", False),
            use_film=model_cfg.get("use_film", False),
            film_hidden=model_cfg.get("film_hidden", 64),
            use_moe_film=model_cfg.get("use_moe_film", False),
            use_dynamic_film=model_cfg.get("use_dynamic_film", False),
            use_skipfusion_experts=model_cfg.get("use_skipfusion_experts", False),
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "nets_ema" in state_dict:
            for k in nets_ema:
                nets_ema[k].load_state_dict(state_dict["nets_ema"][k])
        for k in nets_ema:
            nets_ema[k].to(device).eval()
        return nets_ema, "stargan_v2"
    else:
        raise ValueError(f"Chain inference only supports stargan_v2, got: {method}")


def predict_volume_slice(
    model,
    volume: np.ndarray,
    crop_size: tuple = None,
    slice_axis: int = 2,
    device: torch.device = torch.device("cpu"),
    target_domain: int = None,
    latent_dim: int = 16,
    style_dim: int = 64,
) -> np.ndarray:
    """Run inference on a full 3D volume, slice by slice.

    Input volume in [0,1]; output in [0,1].
    """
    use_crop = crop_size is not None
    if use_crop:
        crop = CenterCropOrPad(crop_size)

    original_shape = volume.shape
    n_slices = volume.shape[slice_axis]
    vol_norm = volume.astype(np.float32)
    output = np.zeros_like(volume)

    with torch.no_grad():
        if target_domain is None:
            raise ValueError("target_domain is required")
        z = torch.randn(1, latent_dim, device=device)
        y = torch.tensor([target_domain], device=device).long()
        s_trg = model["mapping_network"](z, y)

        for i in range(n_slices):
            slicing = [slice(None)] * volume.ndim
            slicing[slice_axis] = i
            s = tuple(slicing)
            slc = vol_norm[s]

            slc_input = crop(slc) if use_crop else slc
            slc_scaled = slc_input * 2.0 - 1.0
            tensor = torch.from_numpy(slc_scaled).float().unsqueeze(0).unsqueeze(0).to(device)

            pred = model["generator"](tensor, s_trg)
            pred_np = pred.squeeze().cpu().numpy()

            if use_crop:
                if slice_axis == 2:
                    orig_slice_shape = (original_shape[0], original_shape[1])
                elif slice_axis == 1:
                    orig_slice_shape = (original_shape[0], original_shape[2])
                else:
                    orig_slice_shape = (original_shape[1], original_shape[2])
                uncrop = CenterCropOrPad(orig_slice_shape)
                pred_np = uncrop(pred_np)

            pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5
            output[s] = pred_np

    return output


def build_chain(source_field: str, target_field: str) -> list:
    """Build the chain of adjacent field strengths from source to target.

    Returns list of (from_field, to_field) tuples.
    """
    src_idx = FIELD_STRENGTHS.index(source_field)
    tgt_idx = FIELD_STRENGTHS.index(target_field)

    if src_idx == tgt_idx:
        return []

    step = 1 if tgt_idx > src_idx else -1
    chain = []
    for i in range(src_idx, tgt_idx, step):
        chain.append((FIELD_STRENGTHS[i], FIELD_STRENGTHS[i + step]))
    return chain


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Chain Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    load_env()
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: INFERENCE_DIR from .env)")
    parser.add_argument("--source_field", type=str, required=True,
                        help="Source field strength, e.g. '0.1T'")
    parser.add_argument("--target_field", type=str, required=True,
                        help="Target field strength, e.g. '7T'")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = get_inference_dir()

    cfg = load_config(args.config)
    device_str = args.device or get_device()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model, model_type = load_generator(cfg, args.checkpoint, device)
    print(f"Loaded model ({model_type}) from: {args.checkpoint}")

    chain = build_chain(args.source_field, args.target_field)
    if not chain:
        print(f"Source and target are the same: {args.source_field}")
        return

    chain_str = " → ".join([args.source_field] + [t for _, t in chain])
    print(f"Chain: {chain_str} ({len(chain)} steps)")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    crop_size_raw = cfg["data"].get("crop_size")
    crop_size = tuple(crop_size_raw) if crop_size_raw else None
    slice_axis = cfg["data"].get("slice_axis", 2)
    model_cfg = cfg["model"]
    latent_dim = model_cfg.get("latent_dim", 16)
    style_dim = model_cfg.get("style_dim", 64)

    for nifti_path in tqdm(nifti_files, desc="Chain Inference"):
        original_img = nib.load(str(nifti_path))
        original_affine = original_img.affine
        original_ornt = nib.io_orientation(original_affine)

        data, canonical_affine = load_nifti(nifti_path)
        canonical_ornt = nib.io_orientation(canonical_affine)

        current_data = data
        for step_idx, (src_f, tgt_f) in enumerate(chain):
            tgt_domain = FIELD_TO_DOMAIN[tgt_f]
            current_data = predict_volume_slice(
                model, current_data,
                crop_size=crop_size,
                slice_axis=slice_axis,
                device=device,
                target_domain=tgt_domain,
                latent_dim=latent_dim,
                style_dim=style_dim,
            )
            print(f"  {nifti_path.name}: Step {step_idx + 1}/{len(chain)} {src_f} → {tgt_f} done")

        transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
        current_data = nib.orientations.apply_orientation(current_data, transform)

        rel_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        save_nifti(current_data, original_affine, out_path)

    print(f"Chain inference complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
