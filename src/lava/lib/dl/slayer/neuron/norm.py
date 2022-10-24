# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Neuron normalization methods."""

import torch


class MeanOnlyBatchNorm(torch.nn.Module):
    """Implements mean only batch norm with optional user defined quantization
    using pre-hook-function. The mean of batchnorm translates to negative bias
    of the neuron.

    Parameters
    ----------
    num_features : int
        number of features. It is automatically initialized on first run if the
        value is None. Default is None.
    momentum : float
        momentum of mean calculation. Defaults to 0.1.
    pre_hook_fx : function pointer or lambda
        pre-hook-function that is applied to the normalization output.
        User can provide a quantization method as needed.
        Defaults to None.

    Attributes
    ----------
    num_features
    momentum
    pre_hook_fx
    running_mean : torch tensor
        running mean estimate.
    update : bool
        enable mean estimte update.
    """
    def __init__(self, num_features=None, momentum=0.1, pre_hook_fx=None):
        """ """
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        if pre_hook_fx is None:
            self.pre_hook_fx = lambda x: x
        else:
            self.pre_hook_fx = pre_hook_fx
        self.register_buffer(
            'running_mean',
            torch.zeros(1 if num_features is None else num_features)
        )
        self.reset_parameters()

        self.update = True

    def reset_parameters(self):
        """Reset states."""
        self.running_mean.zero_()

    @property
    def bias(self):
        """Equivalent bias shift."""
        return -self.pre_hook_fx(self.running_mean, descale=True)

    def forward(self, inp):
        """
        """
        size = inp.shape
        if self.num_features is None:
            self.num_features = inp.shape[1]

        if self.training and self.update is True:
            mean = torch.mean(
                inp.view(size[0], self.num_features, -1),
                dim=[0, 2]
            )
            # n = inp.numel() / inp.shape[1]

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean \
                    + self.momentum * mean
        else:
            mean = self.running_mean

        if len(size) == 2:
            out = (inp - self.pre_hook_fx(mean.view(1, -1)))
        elif len(size) == 3:
            out = (inp - self.pre_hook_fx(mean.view(1, -1, 1)))
        elif len(size) == 4:
            out = (inp - self.pre_hook_fx(mean.view(1, -1, 1, 1)))
        elif len(size) == 5:
            out = (inp - self.pre_hook_fx(mean.view(1, -1, 1, 1, 1)))
        else:
            print(f'Found unexpected number of dims {len(size)} in input.')

        return out


class WgtScaleBatchNorm(torch.nn.Module):
    """Implements batch norm with variance scale in powers of 2. This allows
    eventual normalizaton to be implemented with bit-shift in a hardware
    friendly manner. Optional user defined quantization can be enabled using a
    pre-hook-function. The mean of batchnorm translates to negative bias of the
    neuron.

    Parameters
    ----------
    num_features : int
        number of features. It is automatically initialized on first run if the
        value is None. Default is None.
    momentum : float
        momentum of mean calculation. Defaults to 0.1.
    weight_exp_bits : int
        number of allowable bits for weight exponentation. Defaults to 3.
    eps : float
        infitesimal value. Defaults to 1e-5.
    pre_hook_fx : function pointer or lambda
        pre-hook-function that is applied to the normalization output.
        User can provide a quantization method as needed.
        Defaults to None.

    Attributes
    ----------
    num_features
    momentum
    weight_exp_bits
    eps
    pre_hook_fx
    running_mean : torch tensor
        running mean estimate.
    running_var : torch tensor
        running variance estimate.
    update : bool
        enable mean estimte update.
    """
    def __init__(
        self,
        num_features=None, momentum=0.1,
        weight_exp_bits=3, eps=1e-5,
        pre_hook_fx=None
    ):
        """ """
        super(WgtScaleBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight_exp_bits = weight_exp_bits
        self.eps = eps
        if pre_hook_fx is None:
            self.pre_hook_fx = lambda x: x
        else:
            self.pre_hook_fx = pre_hook_fx
        self.register_buffer(
            'running_mean',
            torch.zeros(1 if num_features is None else num_features)
        )
        self.register_buffer(
            'running_var',
            torch.zeros(1)
        )
        self.reset_parameters()

        self.update = True

    def reset_parameters(self):
        """Reset states."""
        self.running_mean.zero_()
        self.running_var.zero_()

    def std(self, var):
        """
        """
        std = torch.sqrt(var + self.eps)
        return torch.pow(2., torch.ceil(torch.log2(std)).clamp(
            -self.weight_exp_bits, self.weight_exp_bits
        ))

    @property
    def bias(self):
        """Equivalent bias shift."""
        return -self.pre_hook_fx(self.running_mean, descale=True)

    @property
    def weight_exp(self):
        """Equivalent weight exponent value."""
        return torch.ceil(torch.log2(torch.sqrt(self.running_var + self.eps)))

    def forward(self, inp):
        """
        """
        size = inp.shape

        if self.num_features is None:
            self.num_features = inp.shape[1]

        if self.training and self.update is True:
            mean = torch.mean(
                inp.view(size[0], self.num_features, -1),
                dim=[0, 2]
            )
            var = torch.var(inp, unbiased=False)
            n = inp.numel() / inp.shape[1]

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean \
                    + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var  \
                    + self.momentum * var * n / (n + 1)
        else:
            mean = self.running_mean
            var = self.running_var

        std = self.std(var)

        if len(size) == 2:
            out = (
                inp - self.pre_hook_fx(mean.view(1, -1))
            ) / std.view(1, -1)
        elif len(size) == 3:
            out = (
                inp - self.pre_hook_fx(mean.view(1, -1, 1))
            ) / std.view(1, -1, 1)
        elif len(size) == 4:
            out = (
                inp - self.pre_hook_fx(mean.view(1, -1, 1, 1))
            ) / std.view(1, -1, 1, 1)
        elif len(size) == 5:
            out = (
                inp - self.pre_hook_fx(mean.view(1, -1, 1, 1, 1))
            ) / std.view(1, -1, 1, 1, 1)
        else:
            print(f'Found unexpected number of dims {len(size)}.')

        return out
