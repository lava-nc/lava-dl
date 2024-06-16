import torch
from lava.lib.dl import slayer
import torch.nn.functional as F


def quantize_8bit(x, scale=(1 << 6), descale=False):
    if descale is False:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-256 / scale, 254 / scale)
    else:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-256 / scale, 254 / scale) * scale


def quantize_5bit(x, scale=(1 << 6), descale=False):
    if descale is False:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-32 / scale, 30 / scale)
    else:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-32 / scale, 30 / scale) * scale


def quantize_Nbit(x, scale=(1 << 6), descale=False, N=8):
    base = 2**N
    if descale is False:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-base / scale, base-2 / scale)
    else:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-base / scale, base-2 / scale) * scale


def event_rate(x):
    if x.shape[-1] == 1:
        return torch.mean(torch.abs((x) > 0).to(x.dtype)).item()
    else:
        return torch.mean(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item()


class SparsityMonitor:
    def __init__(self, max_rate=0.01, lam=1):
        self.max_rate = max_rate
        self.lam = lam
        self.loss_list = []

    def clear(self):
        self.loss_list = []

    @property
    def loss(self):
        return self.lam * sum(self.loss_list)

    def append(self, x):
        mean_event_rate = torch.mean(torch.abs(x))
        self.loss_list.append(F.mse_loss(F.relu(mean_event_rate - self.max_rate),
                                         torch.zeros_like(mean_event_rate)))
