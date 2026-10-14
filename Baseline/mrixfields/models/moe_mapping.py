"""Shared Mapping Network for MoE+FiLM: maps (z, modality_onehot, field) → s(192) split into 3×64 modality segments."""

import torch
import torch.nn as nn

class SharedMappingNetwork(nn.Module):
    """Maps (z, y_mod_onehot, field) → s(192) partitioned as [s_T1|s_T2|s_FLAIR] each 64D.
    
    Input:
        z: (B, 16) latent noise
        y_mod_onehot: (B, 3) one-hot modality label
        field: (B, 1) normalized field strength [0,1]
    
    Output:
        s: (B, 192) full style vector
    """
    
    def __init__(self, latent_dim=16, hidden=512, style_dim=64, num_modalities=3):
        super().__init__()
        self.style_dim = style_dim
        self.num_modalities = num_modalities
        input_dim = latent_dim + num_modalities + 1  # 20
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, style_dim * num_modalities),
        )
        
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, z, y_mod_onehot, field):
        x = torch.cat([z, y_mod_onehot, field], dim=-1)
        s = self.mlp(x)
        return s
