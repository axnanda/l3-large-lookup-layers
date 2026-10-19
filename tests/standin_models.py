"""Agent B's minimal stand-in for ``l3lab.models.MLP3`` (used until Agent A's models.py lands).

Mirrors the spec: fc1, fc2, fc3 with ReLU and optional dropout; ``ww_layers()`` returns
``[(name, module, role)]`` with roles 'mlp' (fc1, fc2) and 'head' (fc3).
The sweep scripts can be pointed at this file with ``--model_module tests/standin_models.py``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    def __init__(self, h: int = 256, d_in: int = 784, n_out: int = 10, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

    def ww_layers(self):
        return [("fc1", self.fc1, "mlp"), ("fc2", self.fc2, "mlp"), ("fc3", self.fc3, "head")]


__all__ = ["MLP3"]
