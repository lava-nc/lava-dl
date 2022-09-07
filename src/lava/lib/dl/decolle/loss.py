# Copyright : (c) UC Regents, Emre Neftci, 2022 Intel Corporation
# Licence : GPLv3

import torch

"""This module provides a custom loss function to train models with DECOLLE."""


class DECOLLELoss(object):
    """
    Adapted from https://github.com/nmi-lab/decolle-public
    Computes the DECOLLE Loss for the model, comprising the sum of the per-layer
    local pseudo-losses, and a regularization term. Mathematically,

    .. math::
        L = \\sum_t \\sum_{l} \\ell_{t}^{l} +
         reg * \\left( \\langle \\text{reLu}(V_{i, t}^{l} + 0.01) \\rangle_{i}
          + 0.003 * \\langle \\text{reLu}(0.1 - V_{i, t}^{l}) \\rangle_{i}
           \\right)

    Parameters
    ----------
    loss_fn: object
        local Pytorch loss function used for each layer
    net: object
        model to optimize
    reg: float
        strength of regulation
    reduction: str
        "mean" or "sum"
    """

    def __init__(self, loss_fn, net, reg=0, reduction='mean'):
        self.nlayers = len(net.readout_layers)
        self.loss_fn = loss_fn
        self.reg = reg
        self.reduction = reduction

    def __len__(self):
        return self.nlayers

    def __call__(self, readouts, voltages, target):
        loss = 0

        for r, v in zip(readouts, voltages):
            for t in range(r.shape[-1]):
                loss_t = self.loss_fn(r[..., t], target)
                if self.reg > 0.:
                    vflat = v.reshape(v.shape[0], -1)

                    loss_t += self.reg * torch.mean(torch.relu(vflat + .01))
                    loss_t += self.reg * 3e-3 \
                        * torch.mean(torch.relu(0.1 - torch.sigmoid(vflat)))
                if self.reduction == 'mean':
                    loss_t /= r.shape[-1]
                loss += loss_t
        return loss
