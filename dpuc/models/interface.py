
from __future__ import annotations
import torch
from torch import nn
from dpuc.models.common import MLP, TinyTransformer

class InterfaceModel(nn.Module):
    def __init__(self, action_dim: int = 5, slot_dim: int = 9, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 4, residual_components: int = 2):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.slot_proj = nn.Linear(slot_dim, hidden_dim)
        self.encoder = TinyTransformer(hidden_dim, num_heads, num_layers, 0.1)
        self.answer_head = MLP(hidden_dim, hidden_dim, 3, depth=2)
        self.residual_mu = MLP(hidden_dim, hidden_dim, residual_components * 2, depth=2)
        self.residual_logvar = MLP(hidden_dim, hidden_dim, residual_components * 2, depth=2)
        self.value_head = MLP(hidden_dim, hidden_dim, 1, depth=2)
        self.calibration_head = MLP(hidden_dim, hidden_dim, 1, depth=2)

    def forward(self, action_feat: torch.Tensor, slot_feat: torch.Tensor):
        x = self.action_proj(action_feat) + self.slot_proj(slot_feat)
        x = self.encoder(x.unsqueeze(1)).squeeze(1)
        return {
            'answer_logits': self.answer_head(x),
            'residual_mu': self.residual_mu(x),
            'residual_logvar': self.residual_logvar(x),
            'value': self.value_head(x).squeeze(-1),
            'calibration': self.calibration_head(x).squeeze(-1),
        }
