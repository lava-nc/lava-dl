import torch
from torch import nn

class ReLU1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0, max=1)

