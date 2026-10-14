"""MoE-specific losses: Modality Segmentation and Load Balancing."""

import torch


def compute_seg_loss(s, target_mod_idx, num_modalities=3, style_dim=64):
    """Penalize L1 norm of non-target modality segments in s.
    Args:
        s: (B, num_modalities * style_dim) — full mapping output
        target_mod_idx: (B,) — 0/1/2
        num_modalities: 3
        style_dim: 64
    Returns: scalar loss
    """
    B = s.size(0)
    total_dim = num_modalities * style_dim
    mask = torch.ones(B, total_dim, device=s.device)
    for i in range(B):
        start = target_mod_idx[i].item() * style_dim
        end = start + style_dim
        mask[i, start:end] = 0.0
    loss = (s.abs() * mask).sum() / B
    return loss


def compute_load_balancing_loss(router_weights, eps=1e-5):
    """Load balancing: Var(p) / E[p]^2. For K=2, minimized at p=[0.5,0.5].
    Args:
        router_weights: (B, K) — softmax weights
        eps: numerical stability
    Returns: scalar loss
    """
    p = router_weights.mean(dim=0)
    var = torch.var(p, unbiased=False)
    mean_sq = p.mean() ** 2
    loss = var / (mean_sq + eps)
    return loss


def compute_moe_losses(s, target_mod_idx, router_weights, lambda_seg=0.1, lambda_lb=0.001):
    """Aggregate MoE losses. Returns dict with 'seg', 'lb' keys."""
    seg = compute_seg_loss(s, target_mod_idx) * lambda_seg if lambda_seg > 0 else torch.tensor(0.0)
    lb = compute_load_balancing_loss(router_weights) * lambda_lb if lambda_lb > 0 else torch.tensor(0.0)
    return {'seg': seg, 'lb': lb}


# ---- Elastic Weight Consolidation (EWC) for Anti-Catastrophic Forgetting ----

def compute_fisher_information(model, dataloader, num_samples=100, device='cuda'):
    """Compute diagonal Fisher information matrix for EWC.
    
    Fisher = E[(∂log p(y|x,θ)/∂θ)^2] approximated via squared gradients
    of the negative log-likelihood on retrospective samples.
    
    Args:
        model: nn.Module to compute Fisher for
        dataloader: DataLoader of retrospective data
        num_samples: max samples to use (default 100 for efficiency)
        device: compute device
    
    Returns:
        dict: {param_name: fisher_diagonal_tensor} for all trainable params
    """
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        # For StarGAN v2: we approximate Fisher from generator's output distribution
        # Using squared gradients of a simple reconstruction-like objective
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        model.zero_grad()
        # Simple proxy: use L2 norm of output as pseudo-log-likelihood
        # In practice, Fisher is computed ONCE at end of pretraining
        with torch.enable_grad():
            output = model(x)
            if isinstance(output, torch.Tensor):
                loss = output.pow(2).sum()
            else:
                loss = sum(o.pow(2).sum() for o in output if isinstance(o, torch.Tensor))
            loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)
        
        count += x.size(0) if x.dim() >= 2 else 1
    
    # Average over samples
    for name in fisher:
        fisher[name] /= max(count, 1)
    
    model.zero_grad()
    model.train()
    return fisher


def compute_ewc_loss(model, opt_params, fisher_info, lambda_ewc=10.0):
    """EWC penalty: L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ*_i)^2
    
    Penalizes deviation from pretrained weights (opt_params), weighted by
    Fisher information (parameter importance). Higher F_i → more penalty for changing θ_i.
    
    Args:
        model: current model (nn.Module)
        opt_params: dict {name: pretrained_param_tensor} — the optimal weights
        fisher_info: dict {name: fisher_diagonal_tensor} — from compute_fisher_information
        lambda_ewc: EWC strength (default 10.0, from van Garderen et al. MIDL 2019)
    
    Returns:
        scalar EWC loss
    """
    ewc_loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher_info and name in opt_params:
            fisher = fisher_info[name]
            opt = opt_params[name]
            # F_i * (θ_i - θ*_i)^2
            ewc_loss += (fisher * (param - opt).pow(2)).sum()
    
    return (lambda_ewc / 2.0) * ewc_loss


def snapshot_model_params(model, prefix=''):
    """Capture current model weights for EWC optimal parameters.
    
    Args:
        model: nn.Module
        prefix: optional prefix for keys
    
    Returns:
        dict: {prefix+name: param.data.clone()} for all trainable params
    """
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[prefix + name] = param.data.clone()
    return snapshot
