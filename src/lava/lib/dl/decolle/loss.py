# Copyright : (c) UC Regents, Emre Neftci, 2022 Intel Corporation
# Licence : GPLv3

from torch import Tensor
from torch.nn.modules.loss import _Loss

"""This module provides a custom loss function to train models with DECOLLE."""


def _regularization(reg: float, voltage: Tensor) -> Tensor:
    return reg * torch.mean(torch.relu(voltage + .01)) \
        + reg * 3e-3 * torch.mean(torch.relu(0.1 - torch.sigmoid(voltage)))


class DECOLLELoss(_Loss):
    """
    Computes the DECOLLE Loss for the model, comprising the sum of the per-layer
    local pseudo-losses, and a regularization term. Mathematically,

    .. math::
        L = \\sum_t \\sum_{l} \\ell_{t}^{l} +
         \\text{reg} * \\left( \\langle \\text{reLu}(V_{i, t}^{l} + 0.01)
          \\rangle_{i} + 0.003 * \\langle \\text{reLu}(0.1 - V_{i, t}^{l})
           \\rangle_{i} \\right)

    Parameters
    ----------
    loss_fn: object
        local Pytorch loss function used for each layer
    reg: float
        strength of regulation
    reduction: str
        "mean" or "sum"
    """

    def __init__(self, loss_fn: _Loss, reg: float = 0, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super(DECOLLELoss, self).__init__(size_average, reduce, reduction)
        self.loss_fn = loss_fn(size_average, reduce, reduction)
        self.reg = reg

    def forward(self, readouts, voltages, target):
        loss = 0
        for r, v in zip(readouts, voltages):
            for t in range(r.shape[-1]):
                loss_t = self.loss_fn(r[..., t], target)
                if self.reg > 0.:
                    loss_t += _regularization(reg, v)
                if self.reduction == 'mean':
                    loss_t /= r.shape[-1]
                loss += loss_t
        return loss
