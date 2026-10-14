# Third-party code and licensing notes

The code in this repository has several origins. Our own contributions are released under the MIT
terms in `LICENSE`; the files listed below keep the terms of their upstream projects.

## MRIxFields2026 challenge baseline — MIT

`Baseline/`, `Evaluation/` and `Submission/` follow the official challenge code base at
<https://github.com/MRIxFields/MRIxFields2026> (MIT), which is itself the common starting point for
challenge entries.

Our additions on top of it:

- `Baseline/mrixfields/models/film.py` — FiLM / MoE-FiLM conditioning heads (`FourExpertFiLM`,
  `FieldMLP`, `ZtoFiLM`, `DynamicZtoFiLM`)
- `Baseline/mrixfields/models/moe_mapping.py` — shared modality-segment mapping network
- `Baseline/mrixfields/models/moe_losses.py` — modality segmentation / load-balancing / EWC helpers
- `Baseline/mrixfields/models/skip_fusion.py`, `skip_fusion_experts.py` — identity-initialised skip fusion
- `Baseline/mrixfields/models/laplacian_attention.py` — multi-scale Laplacian edge attention
- `Baseline/mrixfields/models/unet_generator.py` — the U-Net generator and its FiLM injection points
- `Baseline/configs/task3/stargan/*` and the two-phase training recipe in `Baseline/scripts/train.py`

## StarGAN v2 — CC BY-NC 4.0 (non-commercial)

`Baseline/mrixfields/models/stargan_v2.py` and the StarGAN v2 building blocks reproduced in
`Baseline/mrixfields/models/unet_generator.py` are adapted from
[clovaai/stargan-v2](https://github.com/clovaai/stargan-v2),
Copyright (c) 2020-present NAVER Corp., licensed under
**Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

**Consequence:** those portions may not be used commercially. The repository as a whole is therefore
research / academic use only. If you need a fully permissive licence, replace the StarGAN v2 derived
files with your own implementation.

## CycleGAN / CUT — BSD / MIT

`Baseline/mrixfields/models/networks.py` reproduces `ResnetGenerator` and `NLayerDiscriminator` from
[junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and the
`encode_only` feature / `PatchSampleF` from
[taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation).

## Other dependencies and references

- LPIPS (`losses/perceptual.py`) — [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) (BSD-2-Clause)
- `pytorch_msssim` — multi-scale SSIM loss
- Multi-scale Laplacian edge attention as used in `laplacian_attention.py` follows the LASTGAN
  formulation (*Scientific Reports*, 2025)
- MoFME — the router + FiLM-expert conditioning pattern that inspired the expert modules
- EWC — Kirkpatrick et al., *PNAS* 2017; van Garderen et al., MIDL 2019
