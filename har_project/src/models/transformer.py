import math
import torch
import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerNet(nn.Module):           # Kunwar Abhuday Singh
    """Patch-embedding Transformer encoder: 128 steps -> 16 tokens of 8 steps."""
    def __init__(self, in_ch=9, patch=8, d=64, heads=4, layers=2, ff=128, n_cls=6):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(in_ch * patch, d)
        self.pos = PosEnc(d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, x):                  # (B,128,9)
        B, T, C = x.shape
        x = x.reshape(B, T // self.patch, self.patch * C)
        x = self.encoder(self.pos(self.embed(x)))
        return self.head(x.mean(1))
