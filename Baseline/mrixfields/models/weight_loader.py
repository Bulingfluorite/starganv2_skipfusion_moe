"""Utility to warm-start networks from a pretrained StarGAN v2 checkpoint.

Loads matching parameters from checkpoint_90000.pth for any network
(generator, discriminator, mapping_network, style_encoder).  New parameters
(e.g. skip fusion layers in UnetStarGANv2Generator) stay at their default
initialization.
"""

import torch
from typing import Optional


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    key: Optional[str] = None,
) -> torch.nn.Module:
    """Load pretrained weights from a checkpoint, keeping new params at init.

    Args:
        model: Network instance (e.g. UnetStarGANv2Generator).
        checkpoint_path: Path to the training checkpoint .pth file.
        key: Key in ckpt["nets"] (e.g. "generator", "discriminator").
            If None, uses the class name to guess (discriminator / generator).

    Returns:
        Model with loaded weights (new params unchanged).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    nets_ckpt = ckpt["nets"]

    # Auto-detect key from model class name when not provided
    if key is None:
        class_name = model.__class__.__name__.lower()
        if "discriminator" in class_name:
            key = "discriminator"
        elif "generator" in class_name:
            key = "generator"
        elif "mapping" in class_name:
            key = "mapping_network"
        elif "style" in class_name:
            key = "style_encoder"
        else:
            raise ValueError(
                f"Cannot auto-detect key for {model.__class__.__name__}. "
                f"Please provide key= explicitly."
            )

    state_dict = nets_ckpt[key]

    total_params = len(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    loaded = total_params - len(missing)
    print(
        f"  [{key}] Loaded {loaded}/{total_params} params, "
        f"{len(missing)} new params (random init)"
    )
    if missing:
        labels = [k.split(".")[0] for k in missing]
        unique = sorted(set(labels))
        print(f"    New modules: {unique}")
    if unexpected:
        print(f"    Unexpected keys in checkpoint: {unexpected}")

    return model
