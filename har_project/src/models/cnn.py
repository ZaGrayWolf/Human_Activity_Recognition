import math
import torch
import torch.nn as nn


class CNN1D(nn.Module):                    # Sree Sai Thottempudi
    def __init__(self, in_ch=9, n_cls=6):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):                  # x: (B,128,9)
        return self.head(self.features(x.transpose(1, 2)))
