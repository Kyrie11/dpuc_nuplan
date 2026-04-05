
from __future__ import annotations
import math
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class TinyTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
    def forward(self, x):
        return self.encoder(x)
