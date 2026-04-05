
from __future__ import annotations
from torch import nn
from dpuc.models.common import MLP

class DBIModel(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(in_dim, hidden_dim, 1, depth=3)
    def forward(self, feat):
        return self.net(feat).squeeze(-1)
