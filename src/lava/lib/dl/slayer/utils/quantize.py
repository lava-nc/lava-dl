# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Quantization utility."""

import torch


class _quantize(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, step=1):
        """
        """
        # return input
        # print('input quantized with step', step)
        return torch.round(input / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        """
        """
        return gradOutput, None


def quantize(input, step=1):
    """Implements quantization of parameters. Rounds the parameter before
    quantization.

    Parameters
    ----------
    input : torch tensor
        input tensor
    step : float
        quantization step. Default is 1.

    Returns
    -------
    torch tensor
        quantized tensor

    Examples
    --------

    >>> # Quantize in step of 0.5
    >>> x_quantized = quantize(x, step=0.5)
    """
    return _quantize.apply(input, step)
