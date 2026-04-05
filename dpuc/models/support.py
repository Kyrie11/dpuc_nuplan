
from __future__ import annotations
import torch
from torch import nn
from dpuc.models.common import MLP

class SupportUtilityModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(in_dim, hidden_dim, out_dim, depth=3)
    def forward(self, feat: torch.Tensor):
        return torch.sigmoid(self.net(feat))
