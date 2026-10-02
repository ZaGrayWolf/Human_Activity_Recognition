import math
import torch
import torch.nn as nn


class MLP(nn.Module):                      # Ayush Bansal
    def __init__(self, in_dim=36, n_cls=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)
